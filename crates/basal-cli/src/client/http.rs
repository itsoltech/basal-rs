//! HTTP transport: one connection pool, bounded bodies, timeouts and opt-in retries.

use std::time::Duration;

use anyhow::{ensure, Context, Result};
use reqwest::{header, Method, Url};
use serde_json::{json, Value};
use tokio::sync::OnceCell;

use super::{ClientArgs, Failure};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// reqwest's own text names only the stage ("error sending request"); its causes say what failed (refused connection,
/// DNS, TLS, timeout). The URL is left out.
fn describe(error: reqwest::Error) -> String {
    format!("{:#}", anyhow::Error::new(error.without_url()))
}

pub(super) struct Remote {
    client: reqwest::Client,
    base: Url,
    retries: u32,
    model: OnceCell<String>,
}

impl Remote {
    pub fn new(args: &ClientArgs) -> Result<Self> {
        let url = match &args.url {
            Some(url) => url.clone(),
            None => super::environment("BASAL_URL")?.unwrap_or_else(|| "http://127.0.0.1:8000".into()),
        };
        let mut base = Url::parse(&url).context("invalid server base URL")?;
        ensure!(matches!(base.scheme(), "http" | "https"), "server URL must use http or https");
        ensure!(
            base.username().is_empty()
                && base.password().is_none()
                && base.query().is_none()
                && base.fragment().is_none(),
            "server base URL must not contain credentials, a query or a fragment"
        );
        base.set_path(&format!("{}/", base.path().trim_end_matches('/')));
        let mut headers = header::HeaderMap::new();
        if let Some(key) = super::environment(&args.key_env)? {
            let mut authorization = header::HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|_| anyhow::anyhow!("{} contains an invalid bearer token", args.key_env))?;
            authorization.set_sensitive(true);
            headers.insert(header::AUTHORIZATION, authorization);
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent(concat!("basal-client/", env!("CARGO_PKG_VERSION")))
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(args.timeout.min(10)))
            .timeout(Duration::from_secs(args.timeout))
            .build()
            .context("creating HTTP client")?;
        Ok(Self { client, base, retries: args.retries, model: OnceCell::new() })
    }

    pub async fn infer(&self, mut body: Value) -> Result<Value, Failure> {
        if body.get("model").is_none() {
            // Reject malformed records before making a discovery request.
            body["model"] = json!("client-discovery");
            super::request::Requests::validate(&body)?;
            let name = self.model.get_or_try_init(|| self.discover_model()).await?;
            body["model"] = json!(name);
        } else {
            super::request::Requests::validate(&body)?;
        }
        // Only the encoded bytes wait for the server's answer; the request tree is released before sending.
        let encoded = serde_json::to_vec(&body).map_err(|e| Failure::message("input", e.to_string()))?;
        drop(body);
        let response = self.request(Method::POST, "v1/systemone", Some(encoded)).await?;
        if !response.get("answers").is_some_and(Value::is_object) {
            return Err(Failure::message("protocol", "server response has no answers object"));
        }
        Ok(response)
    }

    async fn discover_model(&self) -> Result<String, Failure> {
        let response = self.request(Method::GET, "v1/models", None).await?;
        let models = response
            .get("models")
            .and_then(Value::as_array)
            .ok_or_else(|| Failure::message("protocol", "server response has no models array"))?;
        if models.len() != 1 {
            return Err(Failure::message(
                "configuration",
                "server must expose exactly one model for automatic selection; set --model or BASAL_MODEL",
            ));
        }
        models[0]
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .ok_or_else(|| Failure::message("protocol", "server model has no name"))
    }

    async fn request(&self, method: Method, path: &str, body: Option<Vec<u8>>) -> Result<Value, Failure> {
        let url = self.base.join(path).map_err(|e| Failure::message("configuration", e.to_string()))?;
        let mut request = self.client.request(method, url);
        if let Some(body) = body {
            request = request.header(header::CONTENT_TYPE, "application/json").body(body);
        }
        let request = request.build().map_err(|e| Failure::message("http", describe(e)))?;
        let mut attempt = 0;
        loop {
            // A buffered body is reference counted: an attempt shares the encoded bytes instead of copying them.
            let Some(copy) = request.try_clone() else {
                return Err(Failure::message("http", "request body cannot be repeated"));
            };
            let (result, retry_after) = self.receive(copy).await;
            let error = match result {
                Ok(value) => return Ok(value),
                Err(error) => error,
            };
            let transient =
                error.kind == "transport" || matches!(error.status, Some(408 | 429 | 500 | 502 | 503 | 504 | 529));
            if attempt == self.retries || !transient {
                return Err(error);
            }
            // An unrecognised or excessive Retry-After is not shortened: return the original error.
            let delay = match retry_after {
                Some(value) => match value.parse::<u64>() {
                    Ok(seconds) if seconds <= 30 => seconds,
                    _ => return Err(error),
                },
                None => 1u64 << attempt.min(4),
            };
            attempt += 1;
            tokio::time::sleep(Duration::from_secs(delay)).await;
        }
    }

    async fn receive(&self, request: reqwest::Request) -> (Result<Value, Failure>, Option<String>) {
        let mut response = match self.client.execute(request).await {
            Ok(response) => response,
            Err(error) => {
                // A connection closed or reset while sending is as transient as a refused connection.
                let kind = if error.is_builder() { "http" } else { "transport" };
                return (Err(Failure::message(kind, describe(error))), None);
            }
        };
        let status = response.status();
        let retry_after = response.headers().get(header::RETRY_AFTER).and_then(|v| v.to_str().ok()).map(str::to_string);
        let mut bytes = Vec::new();
        match response.content_length().map(usize::try_from) {
            Some(Ok(length)) if length <= MAX_RESPONSE_BYTES => bytes.reserve_exact(length),
            Some(_) => return (Err(Failure::message("protocol", "server response exceeds 16 MiB")), None),
            None => {}
        }
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) if chunk.len() <= MAX_RESPONSE_BYTES.saturating_sub(bytes.len()) => {
                    bytes.extend_from_slice(&chunk)
                }
                Ok(Some(_)) => return (Err(Failure::message("protocol", "server response exceeds 16 MiB")), None),
                Ok(None) => break,
                Err(error) => return (Err(Failure::message("transport", describe(error))), retry_after),
            }
        }
        let parsed = serde_json::from_slice(&bytes);
        let result = if status.is_success() {
            parsed.map_err(|e| Failure::message("protocol", format!("server returned invalid JSON: {e}")))
        } else {
            Err(Failure {
                kind: "http",
                message: format!("HTTP {}", status.as_u16()),
                status: Some(status.as_u16()),
                detail: parsed.ok(),
            })
        };
        (result, retry_after)
    }
}
