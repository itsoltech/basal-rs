//! `basal client`: pipe-friendly System One requests over HTTP or a single local model session.

mod http;
mod input;
mod request;

use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use basal_core::{DecideError, ModelManifest};
use clap::{ArgGroup, Args, ValueEnum};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::task::JoinSet;

use input::Input;
use request::Requests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum InputFormat {
    /// Entire input is one UTF-8 text state
    Text,
    /// Entire input is one JSON state (or one full request with --request)
    Json,
    /// Each nonblank line is a text state
    Lines,
    /// Each nonblank line is a JSON state (or a full request with --request)
    Jsonl,
}

impl InputFormat {
    fn stream(self) -> bool {
        matches!(self, Self::Lines | Self::Jsonl)
    }
}

#[derive(Args)]
#[command(group(ArgGroup::new("questions").required(true).args(["ask", "template", "request"])))]
#[command(group(ArgGroup::new("answer_type").args(["choice", "level", "label"])))]
#[command(
    after_help = "Examples:\n  cat text.txt | basal client --ask 'Is this urgent?'\n  cat text.txt | basal client --ask 'Which department?' --choice sales --choice support --value\n  jq -c '.[]' data.json | basal client --template questions.json --input jsonl --jobs 8\n  cat text.txt | basal client --local --model mini --ask 'Is this urgent?'\n\nJSON goes to stdout; diagnostics go to stderr. See docs/CLIENT.md for Bash recipes."
)]
pub(super) struct ClientArgs {
    /// Question about stdin; noul by default, or choice/score/multi with options below
    #[arg(long)]
    ask: Option<String>,
    /// Choice option KEY or KEY=DESCRIPTION (repeat to define alternatives)
    #[arg(long, requires = "ask", action = clap::ArgAction::Append)]
    choice: Vec<String>,
    /// Score level description, ordered from 0 upwards (repeat)
    #[arg(long, requires = "ask", action = clap::ArgAction::Append)]
    level: Vec<String>,
    /// Multi label KEY or KEY=DESCRIPTION (repeat)
    #[arg(long, requires = "ask", action = clap::ArgAction::Append)]
    label: Vec<String>,
    /// Multi selection probability threshold, 0..1
    #[arg(long, default_value_t = 0.5, requires = "label")]
    threshold: f64,
    /// Request JSON with questions; input replaces its state (supports all question types)
    #[arg(long)]
    template: Option<PathBuf>,
    /// Full request JSON file, or - for stdin; --input jsonl reads many full requests
    #[arg(long, conflicts_with_all = ["state", "state_pointer"])]
    request: Option<String>,
    /// State file, or - for stdin
    #[arg(long)]
    state: Option<String>,
    /// Input framing; default text for --ask/--template, json for --request
    #[arg(long, value_enum)]
    input: Option<InputFormat>,
    /// Select the state inside each JSON input, e.g. /text; output retains the entire input
    #[arg(long)]
    state_pointer: Option<String>,
    /// Print just the single answer: choice/action as text, noul/score as number, multi as JSON array
    #[arg(long, conflicts_with = "keep_going")]
    value: bool,
    /// Continue after failed stream records; emit error records and exit 1 after EOF
    #[arg(long)]
    keep_going: bool,
    /// Maximum concurrent HTTP requests; ordered output and bounded buffering
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    /// Server base URL (BASAL_URL, else http://127.0.0.1:8000)
    #[arg(long, conflicts_with = "local")]
    url: Option<String>,
    /// Model name (BASAL_MODEL, else template/request model, else automatic selection)
    #[arg(long)]
    model: Option<String>,
    /// Environment variable containing the bearer token; no token is required for a local server
    #[arg(long, default_value = "BASAL_API_KEY")]
    key_env: String,
    /// Timeout in seconds per HTTP attempt, including response body
    #[arg(long, default_value_t = 120)]
    timeout: u64,
    /// Extra attempts for transient HTTP/connection errors; backoff and Retry-After, no retries by default
    #[arg(long, default_value_t = 0)]
    retries: u32,
    /// Maximum bytes in one input record or template (default 2 MiB)
    #[arg(long, default_value_t = 2 * 1024 * 1024)]
    max_input_bytes: usize,
    /// Load a local model once, process stdin until EOF, then exit; no HTTP server is started
    #[arg(long)]
    local: bool,
    /// Local weight/activation precision
    #[arg(long, default_value = "f16", requires = "local")]
    dtype: String,
    /// Local CUDA GEMM table (default: same automatic table selection as serve)
    #[arg(long, requires = "local")]
    gemm_table: Option<PathBuf>,
}

#[derive(Serialize)]
struct Failure {
    kind: &'static str,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<Value>,
}

impl Failure {
    fn message(kind: &'static str, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), status: None, detail: None }
    }

    fn decision(error: DecideError) -> Self {
        let kind = match &error {
            DecideError::Validation(_) | DecideError::Usage(_) => "request",
            DecideError::Unsupported(_) => "unsupported",
            DecideError::Internal(_) => "internal",
        };
        Self { kind, message: error.to_string(), status: None, detail: Some(error.to_json()) }
    }
}

fn environment(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok((!value.is_empty()).then_some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!("{name} must contain UTF-8 text"),
    }
}

/// Input and output buffers above this capacity are released after their record instead of being reused.
const RETAINED_BUFFER: usize = 256 * 1024;

struct Record {
    index: u64,
    /// Kept only when the output envelope echoes it.
    input: Option<Value>,
    result: Result<Value, Failure>,
}

/// Stream record; serialized from references, in the same field order as before, without copying input or response.
#[derive(Serialize)]
struct Envelope<'a> {
    index: u64,
    ok: bool,
    input: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a Failure>,
}

struct Output {
    writer: tokio::io::Stdout,
    buffer: Vec<u8>,
    stream: bool,
    value: bool,
}

impl Output {
    /// Queues one line; Tokio writes it in the background. Call [`Output::flush`] before waiting for more input.
    async fn write(&mut self, record: Record) -> Result<bool> {
        self.buffer.clear();
        match &record.result {
            Ok(response) if self.value => match answer_value(response) {
                Ok(Value::String(value)) => self.buffer.extend_from_slice(value.as_bytes()),
                Ok(value) => serde_json::to_writer(&mut self.buffer, value)?,
                Err(error) => {
                    crate::warn!("record {}: {}", record.index, error.message);
                    return Ok(true);
                }
            },
            Ok(response) if !self.stream => serde_json::to_writer(&mut self.buffer, response)?,
            Err(error) if !self.stream => {
                crate::warn!("{}", serde_json::to_string(error)?);
                return Ok(true);
            }
            result => {
                let envelope = Envelope {
                    index: record.index,
                    ok: result.is_ok(),
                    input: record.input.as_ref(),
                    response: result.as_ref().ok(),
                    error: result.as_ref().err(),
                };
                serde_json::to_writer(&mut self.buffer, &envelope)?;
            }
        }
        self.buffer.push(b'\n');
        self.writer.write_all(&self.buffer).await.context("writing client output")?;
        if self.buffer.capacity() > RETAINED_BUFFER {
            self.buffer = Vec::new();
        }
        Ok(record.result.is_err())
    }

    async fn flush(&mut self) -> Result<()> {
        self.writer.flush().await.context("flushing client output")
    }
}

/// Borrows the answer field from the response; a multi `selected` array is serialized in place, not copied.
fn answer_value(response: &Value) -> Result<&Value, Failure> {
    let answers = response
        .get("answers")
        .and_then(Value::as_object)
        .filter(|answers| answers.len() == 1)
        .ok_or_else(|| Failure::message("output", "--value requires exactly one answer"))?;
    let answer = answers.values().next().ok_or_else(|| Failure::message("output", "missing answer"))?;
    let field = match answer.get("type").and_then(Value::as_str) {
        Some("noul") => "noul",
        Some("choice") => "choice",
        Some("score") => "score",
        Some("multi") => "selected",
        Some("act") => "action",
        _ => return Err(Failure::message("output", "unknown answer type")),
    };
    answer.get(field).ok_or_else(|| Failure::message("output", format!("missing answer field {field}")))
}

pub(super) fn run(args: ClientArgs) -> Result<u8> {
    let format = args.input.unwrap_or(if args.request.is_some() { InputFormat::Json } else { InputFormat::Text });
    ensure!((1..=256).contains(&args.jobs), "--jobs must be between 1 and 256");
    ensure!(args.timeout > 0 && args.timeout <= 86400, "--timeout must be between 1 and 86400 seconds");
    ensure!(args.retries <= 10, "--retries must be at most 10");
    ensure!(
        (1..=64 * 1024 * 1024).contains(&args.max_input_bytes),
        "--max-input-bytes must be between 1 byte and 64 MiB"
    );
    ensure!(args.threshold.is_finite() && (0.0..=1.0).contains(&args.threshold), "--threshold must be between 0 and 1");
    ensure!(
        args.request.is_none() || matches!(format, InputFormat::Json | InputFormat::Jsonl),
        "--request requires --input json or jsonl"
    );
    ensure!(!args.keep_going || format.stream(), "--keep-going requires --input lines or jsonl");
    ensure!(!args.local || args.jobs == 1, "--local processes records sequentially; --jobs must be 1");
    if args.local {
        basal_gpu::Precision::parse(&args.dtype)?;
    }
    if let Some(pointer) = &args.state_pointer {
        ensure!(
            matches!(format, InputFormat::Json | InputFormat::Jsonl),
            "--state-pointer requires --input json or jsonl"
        );
        Requests::validate_pointer(pointer)?;
    }
    let requests = Requests::new(&args)?;
    // The Linux launcher delegates only local execution. HTTP works without CUDA libraries or a GPU.
    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    if args.local {
        use std::os::unix::process::CommandExt;
        if let Some(bin) = crate::paths::cuda_binary() {
            let error = crate::cuda_command(&bin).args(std::env::args_os().skip(1)).exec();
            return Err(error).context("starting local CUDA client");
        }
    }
    // Records wait on I/O; client CPU work (validation, JSON) runs in parallel at most `--jobs` ways. With one job or
    // local inference a current-thread runtime is enough and starts no worker threads. Stdin/stdout/file I/O uses
    // Tokio's blocking pool in either case.
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(args.jobs);
    let mut builder = if args.local || workers == 1 {
        tokio::runtime::Builder::new_current_thread()
    } else {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder.worker_threads(workers);
        builder
    };
    let runtime = builder.enable_all().build().context("creating client runtime")?;
    let result = (|| {
        let source = args.request.as_deref().or(args.state.as_deref()).unwrap_or("-");
        let input = runtime.block_on(Input::open(source, format, args.max_input_bytes))?;
        let output = Output {
            writer: tokio::io::stdout(),
            buffer: Vec::new(),
            stream: format.stream() && !args.value,
            value: args.value,
        };
        if args.local {
            run_local(&runtime, &args, &requests, input, output)
        } else {
            let remote = Arc::new(http::Remote::new(&args)?);
            runtime.block_on(async {
                tokio::select! {
                    result = run_remote(&args, Arc::new(requests), remote, input, output) => result,
                    signal = tokio::signal::ctrl_c() => {
                        signal.context("listening for Ctrl-C")?;
                        Ok(130)
                    }
                }
            })
        }
    })();
    // Tokio's stdin is backed by a blocking read which cannot be cancelled. Do not wait for it after Ctrl-C.
    runtime.shutdown_background();
    match result {
        Err(error)
            if error.chain().any(|cause| {
                cause.downcast_ref::<io::Error>().is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
            }) =>
        {
            Ok(0)
        }
        result => result,
    }
}

async fn run_remote(
    args: &ClientArgs,
    requests: Arc<Requests>,
    remote: Arc<http::Remote>,
    mut input: Input,
    mut output: Output,
) -> Result<u8> {
    let mut tasks = JoinSet::new();
    let mut ready = BTreeMap::new();
    let mut index = 0u64;
    let mut next_output = 0u64;
    let mut eof = false;
    let mut failed = false;
    let echo = output.stream;
    loop {
        let written = next_output;
        while let Some(record) = ready.remove(&next_output) {
            let error = output.write(record).await?;
            failed |= error;
            next_output += 1;
            if error && !args.keep_going {
                output.flush().await?;
                // Dropping JoinSet aborts the other HTTP tasks; already admitted server work may still finish.
                return Ok(1);
            }
        }
        // One flush per batch of consecutive ready records, before waiting for input or responses.
        if next_output != written {
            output.flush().await?;
        }
        if eof && tasks.is_empty() {
            return Ok(u8::from(failed));
        }
        tokio::select! {
            record = input.next(), if !eof && tasks.len() + ready.len() < args.jobs => {
                match record? {
                    None => eof = true,
                    Some(Err(error)) => {
                        ready.insert(index, Record { index, input: None, result: Err(error) });
                        index += 1;
                    }
                    Some(Ok(value)) => {
                        let requests = Arc::clone(&requests);
                        let remote = Arc::clone(&remote);
                        let record_index = index;
                        index += 1;
                        tasks.spawn(async move {
                            // Without an envelope the input is consumed by the request instead of being copied.
                            let (input, body) = requests.build(value, echo);
                            let result = match body {
                                Ok(body) => remote.infer(body).await,
                                Err(error) => Err(error),
                            };
                            Record { index: record_index, input, result }
                        });
                    }
                }
            }
            record = tasks.join_next(), if !tasks.is_empty() => {
                if let Some(record) = record {
                    let record = record.context("HTTP worker stopped")?;
                    ready.insert(record.index, record);
                }
            }
        }
    }
}

/// GPU loading and inference run on the caller, outside the Tokio executor. Only input/output use the runtime.
fn run_local(
    runtime: &tokio::runtime::Runtime,
    args: &ClientArgs,
    requests: &Requests,
    mut input: Input,
    mut output: Output,
) -> Result<u8> {
    let mut engine = None;
    let mut failed = false;
    let mut index = 0u64;
    while let Some(record) = runtime.block_on(input.next())? {
        let (original, result) = match record {
            Err(error) => (None, Err(error)),
            Ok(value) => {
                // Only the envelope keeps the input; otherwise its state moves into the request before model load.
                let (original, body) = requests.build(value, output.stream);
                let result = body.and_then(|mut body| {
                    if engine.is_none() {
                        let name = requests
                            .configured_model()
                            .or_else(|| body.get("model").and_then(Value::as_str))
                            .unwrap_or(crate::config::DEFAULT_MODEL.short)
                            .to_string();
                        // Validate the request before downloading weights or preparing a CUDA table.
                        if body.get("model").is_none() {
                            body["model"] = json!(request::model_name(&name));
                        }
                        Requests::validate(&body)?;
                        let loaded =
                            load_local(args, &name).map_err(|e| Failure::message("local", format!("{e:#}")))?;
                        body["model"] = json!(loaded.manifest.name);
                        engine = Some(loaded);
                    }
                    let engine = engine.as_mut().ok_or_else(|| Failure::message("local", "model was not loaded"))?;
                    if body.get("model").is_none() || requests.configured_model().is_some() {
                        body["model"] = json!(engine.manifest.name);
                    }
                    // `decide` starts with the same `parse_request` validation and error; do not parse twice.
                    engine.decide(&body).map_err(Failure::decision)
                });
                (original, result)
            }
        };
        // A failed model load cannot be repaired by skipping input records. Do not repeatedly download/load it.
        let model_failed = result.as_ref().err().is_some_and(|error| error.kind == "local");
        let error = runtime.block_on(async {
            let error = output.write(Record { index, input: original, result }).await?;
            output.flush().await?;
            anyhow::Ok(error)
        })?;
        failed |= error;
        if error && (!args.keep_going || model_failed) {
            return Ok(1);
        }
        index += 1;
    }
    Ok(u8::from(failed))
}

fn load_local(args: &ClientArgs, name: &str) -> Result<basal_core::Engine<basal_gpu::GpuBackend>> {
    let mut model = crate::config::model_ref(name)?;
    model.dtype = args.dtype.clone();
    if let Some(table) = &args.gemm_table {
        model.gemm_table = table.display().to_string();
    }
    let dir = crate::config::model_dir(&model)?;
    let manifest = ModelManifest::load(&dir)?;
    let table = crate::config::gemm_table(&model, &manifest, &crate::config::default_gemm_cache())?;
    let gpu = crate::GpuArgs {
        dtype: args.dtype.clone(),
        readout: "f32".into(),
        kernels: "fused".into(),
        no_prefix_cache: false,
        state_cache_mb: 0,
        tree_max_tokens: basal_core::engine::TREE_MAX_TOKENS,
        gemm_table: table.clone(),
    };
    let engine = crate::gpu_engine_at(&dir, &gpu)?;
    if let Some(table) = table {
        crate::config::check_gemm_coverage(&engine.manifest.name, &engine.backend.gemm_table_missing(), &table)?;
    }
    Ok(engine)
}
