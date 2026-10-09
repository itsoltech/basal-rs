//! Bounded, cancellation-safe input framing. A record survives a cancelled `next` future.

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

use super::{Failure, InputFormat, RETAINED_BUFFER};

/// Each `fill_buf` on stdin or a file is a round trip to Tokio's blocking pool; 64 KiB keeps large records to few trips.
const READ_BUFFER: usize = 64 * 1024;

pub(super) struct Input {
    reader: BufReader<Box<dyn AsyncRead + Unpin + Send>>,
    format: InputFormat,
    limit: usize,
    /// A record which spans several reads; empty while a record fits in the reader buffer.
    buffer: Vec<u8>,
    oversized: bool,
    finished: bool,
}

impl Input {
    pub async fn open(source: &str, format: InputFormat, limit: usize) -> Result<Self> {
        let reader: Box<dyn AsyncRead + Unpin + Send> = if source == "-" {
            Box::new(tokio::io::stdin())
        } else {
            Box::new(tokio::fs::File::open(source).await.with_context(|| format!("opening {source}"))?)
        };
        Ok(Self {
            reader: BufReader::with_capacity(READ_BUFFER, reader),
            format,
            limit,
            buffer: Vec::new(),
            oversized: false,
            finished: false,
        })
    }

    /// Only `fill_buf` awaits; the state is updated synchronously after it, so cancelling this future loses no data.
    pub async fn next(&mut self) -> Result<Option<Result<Value, Failure>>> {
        while !self.finished {
            let available = self.reader.fill_buf().await.context("reading client input")?;
            let eof = available.is_empty();
            let newline = self.format.stream().then(|| available.iter().position(|b| *b == b'\n')).flatten();
            let count = newline.map_or(available.len(), |at| at + 1);
            let payload = newline.unwrap_or(count);
            let complete = eof || newline.is_some();
            let record = if self.oversized {
                None
            } else if payload > self.limit.saturating_sub(self.buffer.len()) {
                self.oversized = true;
                // The rest of the record is discarded up to its end; do not hold the partial copy meanwhile.
                self.buffer = Vec::new();
                None
            } else if complete && self.buffer.is_empty() {
                // The whole record is in the reader buffer: decode it in place, without an intermediate copy.
                decode(self.format, &available[..payload])
            } else {
                self.buffer.extend_from_slice(&available[..payload]);
                complete.then(|| decode(self.format, &self.buffer)).flatten()
            };
            self.reader.consume(count);
            self.finished = eof;
            if !complete {
                continue;
            }
            self.buffer.clear();
            if self.buffer.capacity() > RETAINED_BUFFER {
                // One huge record should not pin its peak allocation for the rest of the stream.
                self.buffer = Vec::new();
            }
            if std::mem::take(&mut self.oversized) {
                return Ok(Some(Err(Failure::message("input", format!("record exceeds {} bytes", self.limit)))));
            }
            if record.is_some() {
                return Ok(record);
            }
        }
        Ok(None)
    }
}

/// `None` is a blank stream line, which is skipped.
fn decode(format: InputFormat, bytes: &[u8]) -> Option<Result<Value, Failure>> {
    if format.stream() && bytes.iter().all(u8::is_ascii_whitespace) {
        return None;
    }
    Some(match format {
        InputFormat::Text | InputFormat::Lines => {
            let bytes = if format == InputFormat::Lines { bytes.strip_suffix(b"\r").unwrap_or(bytes) } else { bytes };
            std::str::from_utf8(bytes)
                .map(|s| Value::String(s.to_owned()))
                .map_err(|e| Failure::message("input", e.to_string()))
        }
        InputFormat::Json | InputFormat::Jsonl => {
            serde_json::from_slice(bytes).map_err(|e| Failure::message("input", e.to_string()))
        }
    })
}
