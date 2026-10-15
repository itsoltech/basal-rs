//! Async HTTP client for the native Basal endpoint (`POST /v1/basal`).
//!
//! Reuse a [`Client`] to share its connection pool. The server validates requests and computes decisions;
//! this crate preserves native probabilities and confidence. No GPU runtime or automatic retries are needed.
//! See `examples/decide.rs` for usage. Dropping an in-flight future cancels the client-side request.

mod questions;
pub use questions::*;

use std::{fmt, net::IpAddr, time::Duration};

use bytes::Bytes;
use http::{Method, Request, Uri};
use http_body_util::{BodyExt, Full};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::{
    client::legacy::connect::HttpConnector,
    rt::{TokioExecutor, TokioTimer},
};

use http::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER, USER_AGENT};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{Map, Value};

/// A native question type.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    /// Choose one of 2–10 options.
    Choice,
    /// Probability of "yes".
    Noul,
    /// Weighted mean of 2–10 ordered levels, indexed from zero.
    Score,
    /// Select labels independently (basal-1.5).
    Multi,
    /// Select the action with minimum expected cost (basal-1.5).
    Act,
}

/// Native question. Criteria and cost matrices are JSON objects or arrays; preserve their order.
/// Additional native fields (including future extensions) go in `extra`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Question {
    /// Question type, serialized as `type`.
    #[serde(rename = "type")]
    pub kind: QuestionType,
    /// Instruction text, or another JSON value accepted by the native API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    /// Native criteria object or ordered list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Value>,
    /// Additional question fields, such as `costs`, `threshold`, `evidence`, or `option_keys`.
    /// Do not repeat `type`, `instructions`, or `criteria` here.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Question {
    /// Start a question with instruction text; add criteria and extension fields as needed.
    pub fn new(kind: QuestionType, instructions: impl Into<String>) -> Self {
        Self { kind, instructions: Some(Value::String(instructions.into())), criteria: None, extra: Map::new() }
    }
}

/// Native inference request. A missing model uses the server's default; basal-rs accepts a model name.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionRequest {
    /// Any non-null JSON state; its rendering belongs to the server.
    pub state: Value,
    /// Question IDs mapped to native questions, in insertion order.
    pub questions: Map<String, Value>,
    /// Exact model name advertised by `/v1/models`, or omitted for the server default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional native fields such as `facts: "auto"`. Do not repeat the fields above.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl DecisionRequest {
    /// Create a request for the server's default model. Insert questions with [`Self::insert_question`].
    pub fn new(state: impl Into<Value>) -> Self {
        Self { state: state.into(), questions: Map::new(), model: None, extra: Map::new() }
    }

    /// Insert a typed question, preserving insertion order. Replaces an existing question of the same ID.
    pub fn insert_question(&mut self, id: impl Into<String>, question: &Question) -> Result<(), Error> {
        self.questions.insert(id.into(), serde_json::to_value(question).map_err(Error::Encode)?);
        Ok(())
    }
}

/// Native answer. Fields used by other question types are absent. Unknown fields are retained in `extra`.
#[derive(Clone, Debug, Deserialize)]
pub struct Answer {
    /// Question type.
    #[serde(rename = "type")]
    pub kind: QuestionType,
    /// Native probabilities (for multi: independent P(yes) values).
    pub probabilities: indexmap::IndexMap<String, f64>,
    /// Native max probability for choice, noul, score and act. Multi uses `set_confidence`.
    pub confidence: Option<f64>,
    /// Chosen key for choice.
    pub choice: Option<String>,
    /// P(yes) for noul.
    pub noul: Option<f64>,
    /// Weighted zero-based score.
    pub score: Option<f64>,
    /// Displayed option texts for score, indexed by key.
    pub legend: Option<indexmap::IndexMap<String, String>>,
    /// Selected keys for multi.
    pub selected: Option<Vec<String>>,
    /// Multi selection threshold.
    pub threshold: Option<f64>,
    /// Probability of precisely the selected set for multi.
    pub set_confidence: Option<f64>,
    /// Chosen action for act.
    pub action: Option<String>,
    /// Most probable outcome for act.
    pub answer: Option<String>,
    /// Expected costs by action for act.
    pub expected_costs: Option<indexmap::IndexMap<String, f64>>,
    /// Calibration status for act.
    pub calibration: Option<String>,
    /// Refused automatic action for act.
    pub refused: Option<String>,
    /// Message when the requested certified threshold is unavailable.
    pub max_error: Option<String>,
    /// Evidence offsets count Unicode characters, not bytes or UTF-16 code units.
    pub evidence: Option<Vec<EvidenceSpan>>,
    /// Future native fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Supporting span in the original state.
#[derive(Clone, Debug, Deserialize)]
pub struct EvidenceSpan {
    /// Verbatim span text.
    pub text: String,
    /// Inclusive start in Unicode characters.
    pub start: usize,
    /// Exclusive end in Unicode characters.
    pub end: usize,
    /// Span probability.
    pub probability: f64,
}

/// Server-reported usage, with native extension fields retained.
#[derive(Clone, Debug, Deserialize)]
pub struct Usage {
    /// Tokens in the option-order prompts.
    pub input_tokens: u64,
    /// Zero: the model does not generate text.
    pub output_tokens: u64,
    /// Optional question count.
    pub questions: Option<u64>,
    /// Optional branch count.
    pub branches: Option<u64>,
    /// Server-side latency, in milliseconds.
    pub latency_ms: Option<f64>,
    /// Future usage fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Native inference result keyed by question ID.
#[derive(Clone, Debug, Deserialize)]
pub struct DecisionResponse {
    /// Model actually used.
    pub model: String,
    /// Answers in server order.
    pub answers: indexmap::IndexMap<String, Answer>,
    /// Server-reported usage.
    pub usage: Usage,
    /// Future envelope fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Model metadata. The exact `name` is accepted by basal-rs as a request's model.
#[derive(Clone, Debug, Deserialize)]
pub struct ModelMetadata {
    /// Model name.
    pub name: String,
    /// Native metadata fields, retained without interpretation.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Model discovery result.
#[derive(Clone, Debug, Deserialize)]
pub struct ModelsResponse {
    /// Models served by this process.
    pub models: Vec<ModelMetadata>,
    /// Future discovery fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Typed SDK failures. Display and Debug omit response bodies and transport details.
#[non_exhaustive]
pub enum Error {
    /// Invalid URL, timeout, response limit or bearer token.
    InvalidConfig(&'static str),
    /// Invalid typed question or ergonomic request.
    InvalidRequest(&'static str),
    /// A typed response is missing required fields or contradicts its question schema.
    InvalidResponse(&'static str),
    /// Connection or response-body failure. Original details are available explicitly through Error::source.
    Transport(Box<dyn std::error::Error + Send + Sync>),
    /// Request JSON serialization failure.
    Encode(serde_json::Error),
    /// Successful response did not match the native JSON contract.
    Protocol(serde_json::Error),
    /// Entire request (including body read) exceeded its deadline.
    Timeout,
    /// Actual streamed response exceeds the configured limit.
    ResponseTooLarge,
    /// Non-success response, including native validation 422 and overload 529.
    Http {
        /// HTTP status code.
        status: u16,
        /// JSON error body, or a JSON string holding a non-JSON response. Explicit access only.
        body: Value,
        /// Unmodified Retry-After; no automatic retry.
        retry_after: Option<String>,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid Basal client configuration: {message}"),
            Self::InvalidRequest(message) => write!(f, "invalid Basal request: {message}"),
            Self::InvalidResponse(message) => write!(f, "invalid Basal answer: {message}"),
            Self::Transport(_) => f.write_str("Basal HTTP transport failed"),
            Self::Encode(_) => f.write_str("could not encode Basal request"),
            Self::Protocol(_) => f.write_str("invalid Basal JSON response"),
            Self::Timeout => f.write_str("Basal HTTP request timed out"),
            Self::ResponseTooLarge => f.write_str("Basal response exceeded the configured byte limit"),
            Self::Http { status, .. } => write!(f, "Basal HTTP {status}"),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Error bodies and nested transport diagnostics can contain reflected documents or credentials.
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(source) => Some(source.as_ref()),
            Self::Encode(source) | Self::Protocol(source) => Some(source),
            _ => None,
        }
    }
}

/// Connection configuration. The API key is intentionally not printable through `Debug`.
pub struct ClientOptions<'a> {
    /// Server root, optionally with a proxy prefix, without `/v1`.
    pub base_url: &'a str,
    /// Optional bearer token for a reverse proxy; basal-rs itself does not authenticate.
    pub api_key: Option<&'a str>,
    /// Whole HTTP request timeout, including response-body reading.
    pub timeout: Duration,
    /// Maximum response bytes (default 8 MiB), including error bodies.
    pub max_response_bytes: usize,
}

impl Default for ClientOptions<'_> {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000",
            api_key: None,
            timeout: Duration::from_secs(120),
            max_response_bytes: 8 * 1024 * 1024,
        }
    }
}

/// Cloneable, pooled HTTP/1 client with verified TLS. No redirects, automatic retries or environment proxies.
/// DNS hostnames must be ASCII (use punycode for IDN). A Tokio runtime is required for requests.
#[derive(Clone)]
pub struct Client {
    http: hyper_util::client::legacy::Client<HttpsConnector<HttpConnector>, Full<Bytes>>,
    base_url: String,
    headers: HeaderMap,
    timeout: Duration,
    max_response_bytes: usize,
}

impl Client {
    /// Connect to a server root with the default timeout and no authentication.
    pub fn new(base_url: &str) -> Result<Self, Error> {
        Self::with_options(ClientOptions { base_url, ..ClientOptions::default() })
    }

    /// Validate configuration and build a connection pool. TLS uses an explicit per-client ring provider
    /// and the platform's certificate verifier; no process-global crypto provider is installed.
    pub fn with_options(options: ClientOptions<'_>) -> Result<Self, Error> {
        let url: Uri = options.base_url.parse().map_err(|_| Error::InvalidConfig("invalid URL"))?;
        let scheme = url.scheme_str().unwrap_or_default();
        let host = url.host().unwrap_or_default();
        if !matches!(scheme, "http" | "https")
            || host.is_empty()
            || options.base_url.contains(['?', '#'])
            || url.authority().is_some_and(|a| a.as_str().contains('@'))
        {
            return Err(Error::InvalidConfig("expected an HTTP(S) URL without credentials, query or fragment"));
        }
        let loopback = host.eq_ignore_ascii_case("localhost")
            || host.trim_matches(['[', ']']).parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback());
        if scheme != "https" && !loopback {
            return Err(Error::InvalidConfig("base_url requires HTTPS or loopback HTTP"));
        }
        if options.timeout.is_zero() || options.max_response_bytes == 0 {
            return Err(Error::InvalidConfig("timeout and max_response_bytes must be positive"));
        }
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(USER_AGENT, HeaderValue::from_static(concat!("basal-client-rust/", env!("CARGO_PKG_VERSION"))));
        if let Some(key) = options.api_key {
            if key.is_empty() || key.contains(['\r', '\n']) {
                return Err(Error::InvalidConfig("api_key must be a nonempty, single-line token"));
            }
            let mut value = HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|_| Error::InvalidConfig("invalid bearer token"))?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }
        let connector = HttpsConnectorBuilder::new()
            .with_provider_and_platform_verifier(rustls::crypto::ring::default_provider())
            .map_err(|source| Error::Transport(Box::new(source)))?
            .https_or_http()
            .enable_http1()
            .build();
        let http = hyper_util::client::legacy::Client::builder(TokioExecutor::new())
            .retry_canceled_requests(false)
            .pool_max_idle_per_host(8)
            .pool_timer(TokioTimer::new())
            .pool_idle_timeout(Duration::from_secs(90))
            .build(connector);
        Ok(Self {
            http,
            base_url: format!("{}/", options.base_url.trim_end_matches('/')),
            headers,
            timeout: options.timeout,
            max_response_bytes: options.max_response_bytes,
        })
    }

    async fn send<T: DeserializeOwned>(&self, method: Method, path: &str, body: Option<Vec<u8>>) -> Result<T, Error> {
        let work = async {
            // Endpoint paths are fixed SDK constants; question data cannot choose another origin.
            let mut request = Request::builder().method(method).uri(format!("{}{path}", self.base_url));
            if let Some(headers) = request.headers_mut() {
                *headers = self.headers.clone();
            }
            if body.is_some() {
                request = request.header(CONTENT_TYPE, "application/json");
            }
            let request = request
                .body(Full::new(Bytes::from(body.unwrap_or_default())))
                .map_err(|source| Error::Transport(Box::new(source)))?;
            let response = self.http.request(request).await.map_err(|source| Error::Transport(Box::new(source)))?;
            let status = response.status();
            let retry_after = response.headers().get(RETRY_AFTER).and_then(|v| v.to_str().ok()).map(String::from);
            let mut body = response.into_body();
            let mut bytes = Vec::new();
            while let Some(frame) = body.frame().await {
                let frame = frame.map_err(|source| Error::Transport(Box::new(source)))?;
                if let Ok(data) = frame.into_data() {
                    if data.len() > self.max_response_bytes.saturating_sub(bytes.len()) {
                        return Err(Error::ResponseTooLarge);
                    }
                    bytes.extend_from_slice(&data);
                }
            }
            if !status.is_success() {
                let body = serde_json::from_slice(&bytes)
                    .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
                return Err(Error::Http { status: status.as_u16(), body, retry_after });
            }
            serde_json::from_slice(&bytes).map_err(Error::Protocol)
        };
        tokio::time::timeout(self.timeout, work).await.map_err(|_| Error::Timeout)?
    }

    /// Decide from a JSON state and typed question builder.
    pub async fn decide(&self, state: impl Into<Value>, questions: &Questions) -> Result<TypedDecisionResponse, Error> {
        self.decide_with_options(state, questions, DecisionOptions::default()).await
    }

    /// Typed decision with explicit model/facts options.
    pub async fn decide_with_options(
        &self,
        state: impl Into<Value>,
        questions: &Questions,
        options: DecisionOptions<'_>,
    ) -> Result<TypedDecisionResponse, Error> {
        let request = questions.request(state.into(), options)?;
        let response = self.decide_request(&request).await?;
        questions.clone().parse(response)
    }

    /// POST `/v1/basal`. Request fields and capability limits are validated by the server.
    pub async fn decide_request(&self, request: &DecisionRequest) -> Result<DecisionResponse, Error> {
        self.send(Method::POST, "v1/basal", Some(serde_json::to_vec(request).map_err(Error::Encode)?)).await
    }

    /// GET `/v1/models`.
    pub async fn models(&self) -> Result<ModelsResponse, Error> {
        self.send(Method::GET, "v1/models", None).await
    }
}
