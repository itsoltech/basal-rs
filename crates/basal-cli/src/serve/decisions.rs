//! Opt-in propensity log at final online dispatch, after all model/task rounds. No background batch records.
//! Hashing owns the incoming body on a bounded blocking job; the dedicated writer owns each final response until
//! the line is flushed and acknowledged. Neither operation runs on Tokio or GPU workers.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use basal_core::pyjson;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};

use super::LargeBodyPermit;

const ANSWER_FIELDS: &[&str] =
    &["type", "choice", "action", "selected", "tool", "tools", "probabilities", "confidence"];
const QUEUE_RECORDS: usize = 32;

/// Share a terminal writer failure across acknowledgements without flattening its source chain into text.
#[derive(Debug, Clone)]
struct WriterFailure(Arc<anyhow::Error>);

impl std::fmt::Display for WriterFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "decision log writer failed: {}", self.0)
    }
}

impl std::error::Error for WriterFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref().as_ref())
    }
}

struct Record {
    request_sha256: String,
    response: Value,
    reply: oneshot::Sender<Result<Value>>,
}

/// The small final-dispatch seam shared by online dialects. Future transports must call it once per final result.
#[derive(Clone)]
pub(super) struct DecisionLog {
    tx: mpsc::Sender<Record>,
}

pub(super) struct HashedRequest {
    pub(super) body: Value,
    pub(super) hash: String,
    pub(super) job: tokio::sync::OwnedSemaphorePermit,
    pub(super) large: Option<LargeBodyPermit>,
}

impl DecisionLog {
    /// Move the original body and its job/large-body permits through the blocking pool, without cloning the body.
    /// These permits bound concurrent hashing even if the HTTP future is cancelled.
    pub(super) async fn hash_request(
        body: Value,
        job: tokio::sync::OwnedSemaphorePermit,
        large: Option<LargeBodyPermit>,
    ) -> Result<HashedRequest> {
        tokio::task::spawn_blocking(move || {
            let mut hash = Sha256::new();
            hash.update(b"[");
            hash.update(pyjson::dumps_sorted(body.get("state").unwrap_or(&Value::Null)).as_bytes());
            hash.update(b", ");
            hash.update(pyjson::dumps_sorted(body.get("questions").unwrap_or(&Value::Null)).as_bytes());
            hash.update(b"]");
            HashedRequest { body, hash: super::batches::config::hex(&hash.finalize()), job, large }
        })
        .await
        .context("decision log request hashing worker stopped")
    }

    /// A full queue or failed line is an explicit internal request error. Return the response only after the write;
    /// moving it through the worker avoids a large clone/serialization on the executor.
    pub(super) async fn record(&self, request_sha256: String, response: Value) -> Result<Value> {
        let (reply, ack) = oneshot::channel();
        self.tx.try_send(Record { request_sha256, response, reply }).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => anyhow!("decision log queue is full"),
            mpsc::error::TrySendError::Closed(_) => anyhow!("decision log writer stopped"),
        })?;
        ack.await.context("decision log writer stopped before acknowledging the line")?
    }
}

/// The synchronous process owner. Open before readiness; drain and join after the HTTP runtime and its senders
/// are dropped. Joining and filesystem operations never run on an async executor.
pub(super) struct DecisionWriter {
    log: DecisionLog,
    thread: JoinHandle<Result<()>>,
}

impl DecisionWriter {
    pub(super) fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path).context("opening decision log")?;
        let (tx, mut rx) = mpsc::channel::<Record>(QUEUE_RECORDS);
        let thread = std::thread::Builder::new()
            .name("decision-log".into())
            .spawn(move || {
                let mut writer = BufWriter::new(file);
                let mut failure: Option<WriterFailure> = None;
                while let Some(record) = rx.blocking_recv() {
                    if failure.is_none() {
                        if let Err(error) = write_record(&mut writer, &record).context("writing decision log") {
                            failure = Some(WriterFailure(Arc::new(error)));
                        }
                    }
                    let result = match &failure {
                        Some(error) => Err(error.clone().into()),
                        None => Ok(record.response),
                    };
                    let _ = record.reply.send(result);
                }
                writer.flush().context("flushing decision log at shutdown")?;
                match failure {
                    Some(error) => Err(error.into()),
                    None => Ok(()),
                }
            })
            .context("starting decision log writer")?;
        Ok(Self { log: DecisionLog { tx }, thread })
    }

    pub(super) fn log(&self) -> DecisionLog {
        self.log.clone()
    }

    pub(super) fn finish(self) -> Result<()> {
        drop(self.log);
        self.thread.join().map_err(|_| anyhow!("decision log writer panicked"))?
    }
}

fn write_record(writer: &mut impl Write, record: &Record) -> Result<()> {
    let answers: serde_json::Map<String, Value> = record.response["answers"]
        .as_object()
        .context("decision response has no answers object")?
        .iter()
        .map(|(name, answer)| {
            let mut fields: serde_json::Map<String, Value> = answer
                .as_object()
                .into_iter()
                .flat_map(|a| a.iter())
                .filter(|(key, _)| ANSWER_FIELDS.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            // TypeSafe noul serves only P(true), while the propensity whitelist excludes the `noul` field.
            // Record its implied binary distribution on this thread without extending the HTTP response.
            if !fields.contains_key("probabilities") && answer["type"].as_str() == Some("noul") {
                if let Some(p) = answer["noul"].as_f64() {
                    fields.insert("probabilities".into(), json!({"true": p, "false": 1.0 - p}));
                    fields.insert("confidence".into(), json!(p.max(1.0 - p)));
                }
            }
            (name.clone(), fields.into())
        })
        .collect();
    let line = json!({
        "ts": SystemTime::now().duration_since(UNIX_EPOCH).context("system clock is before Unix epoch")?.as_secs_f64(),
        "model": record.response["model"], "request_sha256": record.request_sha256,
        "reviewed": Value::Null, "answers": answers,
    });
    // Build the whole line before writing. A failed/partial filesystem write is never acknowledged as success.
    let mut bytes = serde_json::to_vec(&line).context("serializing decision log record")?;
    bytes.push(b'\n');
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}
