//! Official MCP SDK tools, optional embedded HTTP transport and remote stdio bridge.

use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ClientNotification, ClientRequest, ContentBlock,
    GetExtensions, Implementation, JsonRpcMessage, ListToolsResult, PaginatedRequestParams, RequestId,
    ServerCapabilities, ServerConfig, ServerResult, Tool, ToolAnnotations,
};
use rmcp::service::{NotificationContext, RequestContext, RxJsonRpcMessage, TxJsonRpcMessage};
use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService, Transport};
use rmcp::{ErrorData, RoleServer, ServerHandler, Service, ServiceExt};
use serde_json::{json, Map, Value};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

/// Matches the ordinary native HTTP body limit. Checked by the SDK before JSON parsing.
const MAX_BODY: usize = 2 * 1024 * 1024;
const STDIO_INFLIGHT: usize = 4;

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Switch {
    On,
    #[default]
    Off,
}

#[derive(clap::Args)]
pub(crate) struct Args {
    /// Running basal server (the bridge always posts native requests to /v1/basal)
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,
    /// Finite positive seconds per HTTP decision request
    #[arg(long, default_value = "60", value_parser = positive_timeout)]
    timeout: Duration,
}

fn positive_timeout(value: &str) -> std::result::Result<Duration, String> {
    let seconds = value.parse::<f64>().map_err(|e| e.to_string())?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("timeout must be finite and positive".into());
    }
    let duration = Duration::try_from_secs_f64(seconds).map_err(|e| e.to_string())?;
    if duration.is_zero() {
        return Err("timeout must be at least one nanosecond".into());
    }
    Ok(duration)
}

#[derive(Clone)]
enum Backend {
    Embedded(crate::serve::McpDispatch),
    Remote { client: reqwest::Client, url: reqwest::Url },
}

#[derive(Clone)]
pub(crate) struct Tools {
    backend: Backend,
    model_name: String,
}

impl ServerHandler for Tools {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("basal", env!("CARGO_PKG_VERSION")))
            .with_instructions(format!(
                "basal ({}) answers typed questions about a state without generating text, returning calibrated \
                 probabilities. Use it for routing, yes/no checks, scores and label sets. Put every fact and rule \
                 needed in state: the model decides only from what it reads. Ask one condition per yes/no question; \
                 use multi for several labels. Describe what each option means; offer a cannot-be-determined \
                 option when information may be missing. Ask in the state's language (Polish or English). Compute \
                 numbers and dates yourself, then ask a decision about them. Use decide_many for questions over \
                 the same state. Confidence is the calibrated probability of the returned answer; below your \
                 task's required confidence, check the case or ask the user. 2–255 options per question (basal-1.x: \
                 2–10). The configured default model handles execution; this model name is informative only.",
                self.model_name
            ))
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> std::result::Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult { tools: vec![tool(false), tool(true)], ..Default::default() })
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        match name {
            "decide" => Some(tool(false)),
            "decide_many" => Some(tool(true)),
            _ => None,
        }
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> std::result::Result<CallToolResponse, ErrorData> {
        if request.name != "decide" && request.name != "decide_many" {
            return Err(ErrorData::invalid_params(format!("unknown tool: {}", request.name), None));
        }
        let result = async {
            let body = native_body(&request.name, request.arguments.unwrap_or_default())?;
            let response = match &self.backend {
                Backend::Embedded(dispatch) => dispatch.decide(body, &context).await?,
                Backend::Remote { client, url } => remote_decide(client, url, body).await?,
            };
            let mut response = match response {
                Value::Object(response) => response,
                _ => anyhow::bail!("basal response is not an object"),
            };
            if request.name == "decide_many" {
                return Ok(Value::Object(response));
            }
            let model = response.remove("model").context("basal response has no model")?;
            let mut answers = response.remove("answers").context("basal response has no answers")?;
            let answer = answers.get_mut("answer").context("basal response has no answer")?.take();
            let mut answer = match answer {
                Value::Object(answer) => answer,
                _ => anyhow::bail!("basal answer is not an object"),
            };
            let mut result = Map::new();
            result.insert("model".into(), model);
            result.append(&mut answer);
            Ok::<_, anyhow::Error>(Value::Object(result))
        };
        // SDK cancellation drops the native receiver / reqwest future and all attached permits.
        let result = tokio::select! {
            result = result => result,
            _ = context.ct.cancelled() => Err(anyhow::anyhow!("MCP request cancelled")),
        };
        Ok(match result {
            Ok(value) => CallToolResult::structured(value),
            Err(error) => CallToolResult::error(vec![ContentBlock::text(format!("{error:#}"))]),
        }
        .into())
    }
}

fn tool(many: bool) -> Tool {
    let mut properties = Map::new();
    properties.insert(
        "state".into(),
        json!({
            "type": ["string", "object"],
            "description": "The text or JSON object the decision is based on, with every fact and rule it needs."
        }),
    );
    let (name, description, required) = if many {
        properties.insert("questions".into(), json!({
            "type": "object", "additionalProperties": {"type":"object"},
            "description": "name -> native HTTP question: type choice|noul|score|multi|act|tool, instructions, criteria, \
             plus type fields (multi: threshold, min, max, objective; act: costs, max_error; tool: tools, extra, rank, \
             max_error; any: option_keys, evidence). Native tasks classify, verify, compress and extract are also \
             accepted with their native HTTP fields."
        }));
        properties.insert(
            "facts".into(),
            json!({
                "enum": ["off", "auto", null], "default": null,
                "description": "auto: append computed calendar and money facts to a Polish state."
            }),
        );
        (
            "decide_many",
            "Answer several typed questions about one state, including all native HTTP question types \
         and tasks, cost-aware actions with deferral, and next-tool selection. Returns the full native response: \
         answers, model and usage.",
            vec!["state", "questions"],
        )
    } else {
        properties.insert(
            "question".into(),
            json!({
                "type": "string", "description": "The question: one condition, phrased for the state."
            }),
        );
        properties.insert("type".into(), json!({
            "enum": ["choice", "noul", "score", "multi"], "default": "choice",
            "description": "choice: one option; noul: probability of yes; score: ordered level; multi: every applicable option."
        }));
        properties.insert(
            "options".into(),
            json!({
                "anyOf": [
                    {"type": "object", "additionalProperties": {"type": ["string", "null"]}},
                    {"type": "array", "items": {"type": "string"}}, {"type": "null"}
                ], "default": null,
                "description": "choice/multi: key -> meaning, or list of keys; score: levels, lowest first; \
                 noul: optional {true: meaning, false: meaning}. Option insertion order is preserved."
            }),
        );
        (
            "decide",
            "Answer one typed question about a state with calibrated probabilities. Returns choice, noul, \
         score or selected, probabilities and confidence, plus the actual model name.",
            vec!["state", "question"],
        )
    };
    properties.insert(
        "lang".into(),
        json!({
            "enum": ["pl", "en", null], "default": null,
            "description": "Prompt language; default Polish if state/question has Polish diacritics, otherwise English."
        }),
    );
    let schema = json!({"type": "object", "properties": properties, "required": required});
    Tool::new(name, description, schema.as_object().expect("object schema").clone())
        .with_annotations(ToolAnnotations::new().read_only(true).destructive(false).idempotent(true).open_world(false))
}

/// Only native fields cross the inference boundary; JSON-RPC metadata and tool arguments never become API fields.
/// Values/Map retain options, question and criteria order through both transports.
fn native_body(name: &str, mut args: Map<String, Value>) -> Result<Value> {
    let state = args.remove("state").context("state is required")?;
    anyhow::ensure!(state.is_string() || state.is_object(), "state must be a string or object");
    let mut body = Map::new();
    body.insert("state".into(), state);
    if let Some(lang) = args.remove("lang").filter(|v| !v.is_null()) {
        anyhow::ensure!(matches!(lang.as_str(), Some("pl" | "en")), "lang must be pl, en or null");
        body.insert("lang".into(), lang);
    }
    if name == "decide_many" {
        let questions = args.remove("questions").context("questions is required")?;
        anyhow::ensure!(
            questions.as_object().is_some_and(|q| q.values().all(Value::is_object)),
            "questions must map names to native question objects"
        );
        body.insert("questions".into(), questions);
        if let Some(facts) = args.remove("facts").filter(|v| !v.is_null()) {
            anyhow::ensure!(matches!(facts.as_str(), Some("off" | "auto")), "facts must be off, auto or null");
            body.insert("facts".into(), facts);
        }
    } else {
        let question = args.remove("question").context("question is required")?;
        anyhow::ensure!(question.is_string(), "question must be a string");
        let kind = args.remove("type").unwrap_or_else(|| json!("choice"));
        anyhow::ensure!(
            matches!(kind.as_str(), Some("choice" | "noul" | "score" | "multi")),
            "type must be choice, noul, score or multi"
        );
        let mut answer = Map::new();
        answer.insert("type".into(), kind);
        answer.insert("instructions".into(), question);
        if let Some(options) = args.remove("options").filter(|v| !v.is_null()) {
            anyhow::ensure!(
                match &options {
                    Value::Object(map) => map.values().all(|v| v.is_string() || v.is_null()),
                    Value::Array(list) => list.iter().all(Value::is_string),
                    _ => false,
                },
                "options must be a string-to-string/null object, a string list or null"
            );
            answer.insert("criteria".into(), options);
        }
        let mut questions = Map::new();
        questions.insert("answer".into(), Value::Object(answer));
        body.insert("questions".into(), Value::Object(questions));
    }
    Ok(Value::Object(body))
}

pub(crate) fn http_service(
    dispatch: crate::serve::McpDispatch,
    addr: SocketAddr,
    model_name: String,
    stop: CancellationToken,
) -> StreamableHttpService<Tools, LocalSessionManager> {
    // Explicit allowlists keep Origin validation enabled and include the actual listening port, IPv6 brackets,
    // and a concrete non-loopback bind. Wildcard binds do not turn DNS rebinding protection off.
    let mut hosts = vec![
        format!("localhost:{}", addr.port()),
        format!("127.0.0.1:{}", addr.port()),
        format!("[::1]:{}", addr.port()),
    ];
    if !addr.ip().is_unspecified() {
        hosts.push(addr.to_string());
    }
    let origins: Vec<_> = hosts.iter().map(|host| format!("http://{host}")).collect();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_stateless_protocol_metadata_required(false)
        .with_max_request_body_bytes(MAX_BODY)
        .with_cancellation_token(stop)
        .with_allowed_hosts(hosts)
        .with_allowed_origins(origins)
        .enforce_origin_validation();
    let tools = Tools { backend: Backend::Embedded(dispatch), model_name };
    StreamableHttpService::new(move || Ok(tools.clone()), Arc::new(LocalSessionManager::default()), config)
}

async fn response_bytes(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        response.content_length().is_none_or(|n| n <= limit as u64),
        "basal response exceeds {limit} bytes"
    );
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.context("reading basal response")? {
        anyhow::ensure!(chunk.len() <= limit.saturating_sub(bytes.len()), "basal response exceeds {limit} bytes");
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn remote_decide(client: &reqwest::Client, url: &reqwest::Url, body: Value) -> Result<Value> {
    let response = client
        .post(url.join("/v1/basal")?)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(serde_json::to_vec(&body)?)
        .send()
        .await
        .context("POST /v1/basal")?;
    let status = response.status();
    let retry = response.headers().get(reqwest::header::RETRY_AFTER).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let bytes = response_bytes(response, MAX_BODY).await?;
    if status != reqwest::StatusCode::OK {
        let message = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|v| v.get("error")?.as_str().map(str::to_owned))
            .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
        anyhow::bail!(
            "basal server {status}: {message}{}",
            retry.map(|v| format!("; Retry-After: {v}")).unwrap_or_default()
        );
    }
    serde_json::from_slice(&bytes).context("decoding basal response")
}

async fn model_name(client: &reqwest::Client, url: &reqwest::Url) -> Option<String> {
    let response = client.get(url.join("/v1/models").ok()?).timeout(Duration::from_secs(5)).send().await.ok()?;
    let bytes = response_bytes(response.error_for_status().ok()?, 64 * 1024).await.ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    value["models"][0]["name"].as_str().map(str::to_owned)
}

/// Bound SDK stdio parsing and outstanding requests, including responses blocked on stdout. This adapter uses
/// SDK messages/codecs unchanged; the small ID map transfers admission permits to terminal write futures.
struct BoundedStdio {
    inner: AsyncRwTransport<RoleServer, rmcp::transport::StdioReader, rmcp::transport::StdioWriter>,
    slots: Arc<Semaphore>,
    pending: PendingStdio,
    decoding: Arc<Semaphore>,
    /// receive() is polled inside the SDK select loop. Keep a partially written rejection across dropped polls.
    rejection: Option<StdioRejection>,
    stop: CancellationToken,
}

struct StdioRejection {
    write: Pin<Box<dyn Future<Output = std::io::Result<()>> + Send>>,
    _decoding: Arc<OwnedSemaphorePermit>,
}

struct PendingRequest {
    _permit: Arc<OwnedSemaphorePermit>,
    cancellation: Option<CancellationToken>,
}

type PendingStdio = Arc<Mutex<HashMap<RequestId, PendingRequest>>>;

/// SDK service cancellation intentionally suppresses a response. Release its pending output permit there too,
/// rather than waiting for a send that will never happen. Handler extensions retain their allocation permit.
struct StdioTools {
    tools: Tools,
    pending: PendingStdio,
}

impl Service<RoleServer> for StdioTools {
    async fn handle_request(
        &self,
        request: ClientRequest,
        context: RequestContext<RoleServer>,
    ) -> std::result::Result<ServerResult, ErrorData> {
        let ct = context.ct.clone();
        let id = context.id.clone();
        if let Some(pending) = self.pending.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&id) {
            pending.cancellation = Some(ct.clone());
        }
        let result = self.tools.handle_request(request, context).await;
        if ct.is_cancelled() {
            self.pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
        }
        result
    }

    async fn handle_notification(
        &self,
        notification: ClientNotification,
        context: NotificationContext<RoleServer>,
    ) -> std::result::Result<(), ErrorData> {
        let cancelled = match &notification {
            ClientNotification::CancelledNotification(notification) => notification.params.request_id.clone(),
            _ => None,
        };
        let result = self.tools.handle_notification(notification, context).await;
        // Cancellation can arrive after the request handler returned but before the SDK picked its response.
        // The SDK then suppresses that response. Its notification path cancels the context before calling us.
        if let Some(id) = cancelled {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            if pending
                .get(&id)
                .is_some_and(|request| request.cancellation.as_ref().is_some_and(CancellationToken::is_cancelled))
            {
                pending.remove(&id);
            }
        }
        result
    }

    fn get_info(&self) -> ServerConfig {
        ServerHandler::get_info(&self.tools)
    }
}

impl Transport<RoleServer> for BoundedStdio {
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleServer>,
    ) -> impl Future<Output = std::io::Result<()>> + Send + 'static {
        let id = match &item {
            JsonRpcMessage::Response(response) => Some(response.id.clone()),
            JsonRpcMessage::Error(error) => error.id.clone(),
            _ => None,
        };
        // Transfer the output permit into the write future before spawning it. Cancellation notifications can
        // release a handler awaiting SDK dispatch, but cannot release a response already blocked on stdout.
        let permit = id.as_ref().and_then(|id| self.pending.lock().unwrap_or_else(|e| e.into_inner()).remove(id));
        let suppressed = id.is_some() && permit.is_none();
        let stop = self.stop.clone();
        let send = (!suppressed).then(|| self.inner.send(item));
        async move {
            let result = match send {
                Some(send) => tokio::select! {
                    result = send => result,
                    _ = stop.cancelled() => Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted, "MCP stdio shutting down",
                    )),
                },
                None => Ok(()), // a cancelled request whose response the SDK had already picked up
            };
            drop(permit);
            if result.is_err() {
                stop.cancel();
            }
            result
        }
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleServer>> {
        loop {
            if let Some(rejection) = &mut self.rejection {
                let result = rejection.write.as_mut().await;
                self.rejection = None;
                if result.is_err() {
                    self.stop.cancel();
                    return None;
                }
            }
            // Exactly one bounded SDK parser, plus at most four outstanding request/response allocations.
            // Notification handlers retain this decode slot, so they cannot accumulate detached tasks either.
            let decoding = Arc::new(self.decoding.clone().acquire_owned().await.ok()?);
            let Some(mut message) = self.inner.receive().await else {
                self.stop.cancel();
                return None;
            };
            match &mut message {
                JsonRpcMessage::Request(request) => {
                    let slot = match self.slots.clone().try_acquire_owned() {
                        Ok(slot) => Arc::new(slot),
                        Err(_) => {
                            // Retain exactly one rejection across SDK select cancellations; do not spawn a queue.
                            let reply = if matches!(request.request, ClientRequest::CallToolRequest(_)) {
                                JsonRpcMessage::response(ServerResult::CallToolResult(CallToolResult::error(vec![
                                    ContentBlock::text("MCP stdio bridge overloaded: at most four outstanding requests; retry after a short delay"),
                                ])), request.id.clone())
                            } else {
                                JsonRpcMessage::error(
                                    ErrorData::internal_error("MCP stdio bridge overloaded", None),
                                    Some(request.id.clone()),
                                )
                            };
                            self.rejection =
                                Some(StdioRejection { write: Box::pin(self.inner.send(reply)), _decoding: decoding });
                            continue;
                        }
                    };
                    let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
                    if pending.contains_key(&request.id) {
                        crate::warn!("MCP stdio: duplicate in-flight request ID; closing transport");
                        self.stop.cancel();
                        return None;
                    }
                    request.request.extensions_mut().insert(slot.clone());
                    pending.insert(request.id.clone(), PendingRequest { _permit: slot, cancellation: None });
                }
                JsonRpcMessage::Notification(notification) => {
                    notification.notification.extensions_mut().insert(decoding);
                }
                _ => {}
            }
            return Some(message);
        }
    }

    async fn close(&mut self) -> std::io::Result<()> {
        self.stop.cancel();
        // Drop a paused rejection write before SDK close tries to take the writer mutex.
        self.rejection = None;
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.inner.close().await
    }
}

pub(crate) fn run(args: Args) -> Result<()> {
    let url = reqwest::Url::parse(&args.url).context("--url")?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
        "--url requires an HTTP(S) host"
    );
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let result = runtime.block_on(async move {
        let stop = CancellationToken::new();
        let _stop = stop.clone().drop_guard();
        let client = reqwest::Client::builder().timeout(args.timeout).build().context("basal HTTP client")?;
        let name = tokio::select! {
            name = model_name(&client, &url) => name.unwrap_or_else(|| "basal".into()),
            _ = crate::serve::shutdown_signal() => return Ok(()),
        };
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let tools = StdioTools {
            tools: Tools { backend: Backend::Remote { client, url }, model_name: name },
            pending: pending.clone(),
        };
        let (read, write) = rmcp::transport::stdio_pipes();
        let transport = BoundedStdio {
            inner: AsyncRwTransport::new(read, write).with_max_line_length(MAX_BODY),
            slots: Arc::new(Semaphore::new(STDIO_INFLIGHT)),
            pending,
            decoding: Arc::new(Semaphore::new(1)),
            rejection: None,
            stop: stop.clone(),
        };
        let service = tokio::select! {
            service = tools.serve_with_ct(transport, stop.clone()) => service.context("MCP stdio initialization")?,
            _ = crate::serve::shutdown_signal() => return Ok(()),
        };
        let waiting = service.waiting();
        tokio::pin!(waiting);
        tokio::select! {
            result = &mut waiting => { result.context("MCP stdio transport")?; }
            _ = crate::serve::shutdown_signal() => {
                stop.cancel();
                waiting.await.context("MCP stdio shutdown")?;
            }
        }
        Ok(())
    });
    // Tokio's fallback stdio reader may block on a terminal. MCP hosts use pipes; do not wait for a terminal read
    // after a signal. Transport/reqwest tasks are cancelled before the runtime is released.
    runtime.shutdown_timeout(Duration::from_secs(2));
    result
}
