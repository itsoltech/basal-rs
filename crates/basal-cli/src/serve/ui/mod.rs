//! Browser UI of the server (`ui:`, `basal serve --ui`; off by default): static pages under `/ui/` and the account
//! API under `/ui/api/` (`api`). Description: `docs/UI.md`.
//!
//! ```yaml
//! ui:
//!   enabled: true         # default false: no /ui route exists
//!   session_ttl: 12h      # lifetime of a login session (5m to 30d)
//!   cookie_secure: true   # false only for plain-HTTP access other than localhost
//!   capture_requests: false  # keep the bodies of the last 200 inference requests for admins; needs auth
//! ```
//!
//! The pages (`crates/basal-cli/ui/`) are compiled into the binary: plain HTML, CSS and JavaScript modules, without a
//! build step or third-party code. Every `/ui` answer carries a strict Content-Security-Policy (only scripts, styles and
//! requests of this origin; no inline scripts or styles; no framing) and `Cache-Control: no-store` for the API.
//!
//! With `auth` the pages log in with a user id and password (`session`); without it there are no accounts: the
//! pages offer the Playground, health and the Batch API of `batches.owner` to anyone who reaches the server, like
//! the API itself.

pub(crate) mod api;
#[cfg(test)]
mod http_tests;
pub(crate) mod session;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, Path};
use axum::http::header::{CACHE_CONTROL, CONTENT_SECURITY_POLICY, CONTENT_TYPE, LOCATION};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, delete, get, post};
use axum::Router;
use serde::Deserialize;

use super::batches::dto::ApiError;
use super::batches::store::Store;

/// Largest body of an account request.
const MAX_BODY_BYTES: usize = 16 * 1024;
const MIN_SESSION_TTL: Duration = Duration::from_secs(5 * 60);
const MAX_SESSION_TTL: Duration = Duration::from_secs(30 * 86_400);
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; \
                   connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

/// The `ui:` section.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UiConfig {
    #[serde(default)]
    pub(crate) enabled: bool,
    #[serde(default = "default_session_ttl")]
    session_ttl: String,
    #[serde(default = "default_cookie_secure")]
    cookie_secure: bool,
    /// Also `basal serve --ui-capture-requests`.
    #[serde(default)]
    pub(crate) capture_requests: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            session_ttl: default_session_ttl(),
            cookie_secure: default_cookie_secure(),
            capture_requests: false,
        }
    }
}

fn default_session_ttl() -> String {
    "12h".into()
}

fn default_cookie_secure() -> bool {
    true
}

/// Checked `ui:` settings.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UiSettings {
    pub(crate) session_ttl: Duration,
    pub(crate) cookie_secure: bool,
    /// Keep the bodies of recent inference requests for the server view (only with accounts).
    pub(crate) capture_requests: bool,
}

impl UiConfig {
    /// Checked settings; `auth`: accounts are enabled.
    pub(crate) fn validate(&self, auth: bool) -> anyhow::Result<UiSettings> {
        let session_ttl =
            crate::keys::parse_duration(&self.session_ttl).map_err(|e| anyhow::anyhow!("ui.session_ttl: {e}"))?;
        anyhow::ensure!(
            (MIN_SESSION_TTL..=MAX_SESSION_TTL).contains(&session_ttl),
            "ui.session_ttl must be between 5m and 30d"
        );
        anyhow::ensure!(
            !self.capture_requests || auth,
            "ui.capture_requests needs auth (auth.enabled or --auth): request contents are shown to administrators only"
        );
        Ok(UiSettings { session_ttl, cookie_secure: self.cookie_secure, capture_requests: self.capture_requests })
    }
}

/// Features of this server the pages adapt to (`GET /ui/api/config`).
pub(crate) struct UiInfo {
    pub(crate) batches: bool,
    pub(crate) metrics: bool,
    pub(crate) mcp: bool,
}

/// One embedded page file: name under `/ui/`, content type, content.
struct Asset {
    name: &'static str,
    content_type: &'static str,
    body: &'static str,
}

macro_rules! assets {
    ($(($name:literal, $type:expr)),+ $(,)?) => {
        &[$(Asset { name: $name, content_type: $type, body: include_str!(concat!("../../../ui/", $name)) }),+]
    };
}

const JS: &str = "text/javascript; charset=utf-8";

const ASSETS: &[Asset] = assets![
    ("index.html", "text/html; charset=utf-8"),
    ("style.css", "text/css; charset=utf-8"),
    ("favicon.svg", "image/svg+xml"),
    ("app.js", JS),
    ("api.js", JS),
    ("ui.js", JS),
    ("playground.js", JS),
    ("templates.js", JS),
    ("admin.js", JS),
    ("account.js", JS),
    ("batches.js", JS),
    ("server.js", JS),
    ("charts.js", JS),
];

/// Routes of the UI. `store` is the key store with `auth` (accounts), else `None` (only the pages and `config`);
/// `stats` the statistics of the admin server view (with accounts).
pub(super) fn router(
    settings: UiSettings,
    info: &UiInfo,
    store: Option<Store>,
    stats: Option<Arc<crate::serve::stats::Stats>>,
) -> Router<()> {
    let accounts = store.is_some();
    let config = serde_json::json!({
        "auth": accounts,
        "batches": info.batches,
        "metrics": info.metrics,
        "mcp": info.mcp,
        "stats": stats.is_some(),
        "version": crate::version(),
    });
    if accounts {
        tokio::task::spawn_blocking(session::warm_up);
    }
    let state = Arc::new(api::UiState::new(settings, config, store, stats));
    let routes = Router::new().route("/ui/api/config", get(api::config));
    let routes = if accounts {
        routes
            .route("/ui/api/login", post(api::login))
            .route("/ui/api/logout", post(api::logout))
            .route("/ui/api/me", get(api::me))
            .route("/ui/api/me/password", post(api::change_password))
            .route("/ui/api/me/keys", get(api::my_keys).post(api::create_my_key))
            .route("/ui/api/me/keys/{id}/revoke", post(api::revoke_my_key))
            .route("/ui/api/admin/users", get(api::users).post(api::create_user))
            .route("/ui/api/admin/users/{id}", delete(api::delete_user))
            .route("/ui/api/admin/users/{id}/disable", post(api::disable_user))
            .route("/ui/api/admin/users/{id}/enable", post(api::enable_user))
            .route("/ui/api/admin/users/{id}/password", post(api::set_password))
            .route("/ui/api/admin/keys", get(api::keys).post(api::create_user_key))
            .route("/ui/api/admin/keys/{id}/revoke", post(api::revoke_key))
            .route("/ui/api/admin/stats", get(api::stats_snapshot))
            .route("/ui/api/admin/stats/reset", post(api::reset_stats))
            .route("/ui/api/admin/requests", get(api::recent_requests))
            .route("/ui/api/admin/requests/{id}", get(api::recent_request))
    } else {
        routes
    };
    routes
        .route("/ui/api/{*rest}", any(api_not_found))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
        .route("/ui", get(|| async { (StatusCode::PERMANENT_REDIRECT, [(LOCATION, "/ui/")]) }))
        .route("/ui/", get(|| async { asset("index.html") }))
        .route("/ui/{file}", get(|Path(file): Path<String>| async move { asset(&file) }))
        .layer(axum::middleware::map_response(security_headers))
}

async fn api_not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "invalid_request_error", "Unknown UI route.").with_code("not_found")
}

fn asset(name: &str) -> Response {
    match ASSETS.iter().find(|a| a.name == name) {
        Some(a) => ([(CONTENT_TYPE, a.content_type), (CACHE_CONTROL, "no-cache")], a.body).into_response(),
        None => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

async fn security_headers(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(CONTENT_SECURITY_POLICY, HeaderValue::from_static(CSP));
    headers.insert(HeaderName::from_static("x-content-type-options"), HeaderValue::from_static("nosniff"));
    headers.insert(HeaderName::from_static("x-frame-options"), HeaderValue::from_static("DENY"));
    headers.insert(HeaderName::from_static("referrer-policy"), HeaderValue::from_static("no-referrer"));
    if !headers.contains_key(CACHE_CONTROL) {
        headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_limits() {
        let settings = UiConfig::default().validate(false).unwrap();
        assert_eq!(settings.session_ttl, Duration::from_secs(12 * 3600));
        assert!(settings.cookie_secure);
        let config = |ttl: &str| UiConfig { session_ttl: ttl.into(), ..UiConfig::default() };
        assert!(config("1m").validate(false).is_err());
        assert!(config("31d").validate(false).is_err());
        assert!(config("30d").validate(false).is_ok());
    }

    #[test]
    fn assets_are_embedded_with_types() {
        assert!(ASSETS.iter().all(|a| !a.body.is_empty()));
        assert_eq!(asset("index.html").headers()[CONTENT_TYPE], "text/html; charset=utf-8");
        assert_eq!(asset("../Cargo.toml").status(), StatusCode::NOT_FOUND);
        let index = ASSETS.iter().find(|a| a.name == "index.html").unwrap().body;
        assert!(!index.contains("<script>") && !index.contains("style="), "inline code breaks the CSP");
    }
}
