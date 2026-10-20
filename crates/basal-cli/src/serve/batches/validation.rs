//! Bounded JSONL framing and envelope validation before native request planning.

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufRead, AsyncBufReadExt};

/// Limits are checked before extending a line or constructing a JSON value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InputLimits {
    pub max_line_bytes: usize,
    pub max_records: u64,
    pub max_json_depth: usize,
    pub max_json_nodes: usize,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum InputError {
    #[error("reading batch input failed")]
    Io(#[source] std::io::Error),
    #[error("batch input line exceeds its byte limit")]
    LineTooLarge,
    #[error("batch input exceeds its record limit")]
    TooManyRecords,
    #[error("empty batch input line")]
    EmptyLine,
    #[error("batch input position overflow")]
    PositionOverflow,
    #[error("batch input allocation failed")]
    Allocation,
    #[error("batch input JSON exceeds its nesting limit")]
    JsonDepth,
    #[error("batch input JSON exceeds its node limit")]
    JsonNodes,
    #[error("batch input JSON memory bound overflow")]
    JsonMemory,
    #[error("invalid batch input JSON envelope")]
    Json(#[source] serde_json::Error),
    #[error("custom_id must contain 1..256 UTF-8 bytes")]
    CustomId,
    #[error("batch input method must be POST")]
    Method,
    #[error("batch input url must be /v1/basal")]
    Endpoint,
    #[error("batch input body must be an object")]
    Body,
    #[error("streaming batch requests are unsupported")]
    Streaming,
}

/// Byte range excludes LF, or CRLF, and indexes the original immutable file.
pub(super) struct InputLine {
    pub number: u64,
    pub offset: u64,
    pub bytes: Vec<u8>,
}

impl InputLine {
    pub fn sha256(&self) -> [u8; 32] {
        Sha256::digest(&self.bytes).into()
    }
}

pub(super) struct JsonlReader<R> {
    reader: R,
    limits: InputLimits,
    offset: u64,
    records: u64,
}

impl<R: AsyncBufRead + Unpin> JsonlReader<R> {
    pub fn new(reader: R, limits: InputLimits) -> Self {
        Self { reader, limits, offset: 0, records: 0 }
    }

    /// The caller reserves the line's memory before calling this method. We never
    /// use `read_until`, whose allocation could exceed the limit before returning.
    pub async fn next(&mut self) -> Result<Option<InputLine>, InputError> {
        let start = self.offset;
        let mut bytes = Vec::new();
        let ended_with_lf = loop {
            let available = self.reader.fill_buf().await.map_err(InputError::Io)?;
            if available.is_empty() {
                if bytes.is_empty() {
                    return Ok(None);
                }
                break false;
            }
            let lf = available.iter().position(|&b| b == b'\n');
            let content = lf.unwrap_or(available.len());
            let next_len = bytes.len().checked_add(content).ok_or(InputError::LineTooLarge)?;
            // Permit a possible final CR in addition to the content limit.
            if next_len > self.limits.max_line_bytes.saturating_add(1) {
                return Err(InputError::LineTooLarge);
            }
            bytes.try_reserve_exact(content).map_err(|_| InputError::Allocation)?;
            bytes.extend_from_slice(&available[..content]);
            let consumed = content + usize::from(lf.is_some());
            self.offset = self.offset.checked_add(consumed as u64).ok_or(InputError::PositionOverflow)?;
            self.reader.consume(consumed);
            if lf.is_some() {
                break true;
            }
        };
        if ended_with_lf && bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        if bytes.len() > self.limits.max_line_bytes {
            return Err(InputError::LineTooLarge);
        }
        if bytes.is_empty() {
            return Err(InputError::EmptyLine);
        }
        let number = self.records.checked_add(1).ok_or(InputError::TooManyRecords)?;
        if number > self.limits.max_records {
            return Err(InputError::TooManyRecords);
        }
        self.records = number;
        Ok(Some(InputLine { number, offset: start, bytes }))
    }
}

/// Conservative upper reservation for the raw buffer, decoded strings and JSON
/// containers. This is lexical preflight, not a substitute for JSON parsing.
/// The caller must acquire the returned budget before `parse_record`.
pub(super) fn json_memory_bound(bytes: &[u8], limits: InputLimits) -> Result<usize, InputError> {
    let mut depth = 0usize;
    let mut nodes = 1usize;
    let mut quoted = false;
    let mut escaped = false;
    for &b in bytes {
        if quoted {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                quoted = false;
            }
            continue;
        }
        match b {
            b'"' => {
                quoted = true;
                nodes = nodes.checked_add(1).ok_or(InputError::JsonNodes)?;
            }
            b'{' | b'[' => {
                depth = depth.checked_add(1).ok_or(InputError::JsonDepth)?;
                if depth > limits.max_json_depth {
                    return Err(InputError::JsonDepth);
                }
                nodes = nodes.checked_add(1).ok_or(InputError::JsonNodes)?;
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            b',' | b':' => nodes = nodes.checked_add(1).ok_or(InputError::JsonNodes)?,
            _ => {}
        }
        if nodes > limits.max_json_nodes {
            return Err(InputError::JsonNodes);
        }
    }
    // Include container spare capacity and owned strings/numbers. Native planning
    // and media scratch require their own PlanBudget reservations after parsing.
    bytes
        .len()
        .checked_mul(4)
        .and_then(|n| nodes.checked_mul(256).and_then(|m| n.checked_add(m)))
        .ok_or(InputError::JsonMemory)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRecord {
    custom_id: String,
    method: String,
    url: String,
    body: Value,
}

pub(super) struct InputRecord {
    pub custom_id: String,
    pub body: Value,
}

/// Call only after successful lexical preflight and budget reservation. Persist
/// `InputLine.bytes`/its range, never a reserialized version of this body.
pub(super) fn parse_record(bytes: &[u8]) -> Result<InputRecord, InputError> {
    let WireRecord { custom_id, method, url, body } = serde_json::from_slice(bytes).map_err(InputError::Json)?;
    if custom_id.is_empty() || custom_id.len() > 256 {
        return Err(InputError::CustomId);
    }
    if method != "POST" {
        return Err(InputError::Method);
    }
    if url != "/v1/basal" {
        return Err(InputError::Endpoint);
    }
    if !body.is_object() {
        return Err(InputError::Body);
    }
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(InputError::Streaming);
    }
    Ok(InputRecord { custom_id, body })
}
