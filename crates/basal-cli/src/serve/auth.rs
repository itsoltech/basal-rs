//! API keys of the server (`auth:`): bearer tokens stored as SHA-256 digests in the configured database.
//!
//! ```yaml
//! auth:
//!   enabled: true          # default false: no key is checked, every route is open
//!   protect_metrics: false # true: GET /metrics needs a key with scope `metrics`
//! ```
//!
//! With `enabled`, every route except `/health`, `/health/batches` and (by default) `/metrics` needs
//! `Authorization: Bearer bsk_<key_id>_<secret>`, also on a loopback listener; the listener address, forwarding
//! headers and request fields never authenticate. A key carries scopes (route groups, [`ApiScopes`]) and belongs to
//! a user: the Batch/Files routes use the user's id as their storage namespace instead of `batches.owner`, so all keys
//! of one user share its files and batches. A missing, malformed, unknown, revoked or expired key, or a key of a
//! disabled user, is the same 401; a valid key without the route's scope is a 403.
//!
//! Users and keys are managed with `basal users` and `basal keys` (`crate::keys`); the server only reads them.
//! The token is shown once at creation and never stored: the database keeps the SHA-256 of the whole bearer value,
//! compared in constant time. The secret has 256 random bits, so a plain digest needs no password hashing.
//!
//! Each key and its user are read from the database at most once per [`CACHE_TTL`] per process, so a revocation,
//! expiry or disabled user takes effect within that time, and when the database is unreachable after it requests are refused (503), never let
//! through. Uses are recorded in `last_used_at` at most once per [`TOUCH_INTERVAL`] per key, in the background.
//!
//! With the browser UI (`serve::ui`) a request without an `Authorization` header may instead carry the session
//! cookie of a logged-in user, together with `X-Basal-UI: 1` (otherwise 403). Sessions are read from the database on
//! every request (no cache: logout, a disabled user and a password change act at once) and give the scopes of
//! `ui::session::scopes`. The account routes `/ui/api/...` accept only sessions; the pages, `/ui/api/config`,
//! `/ui/api/login` and `/ui/api/logout` are public (the last two check the UI header themselves).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, WWW_AUTHENTICATE};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::{mpsc, Semaphore, SemaphorePermit};

use super::batches::dto::ApiError;
use super::batches::store::{
    ApiKeyId, ApiKeyRecord, ApiScopes, ApiSessionLookup, OwnerId, Sha256Digest, Store, UnixMillis,
};
use super::ui::api::{ui_header_required, UiSession};
use super::ui::session;
use crate::config_support::SecretString;

/// Prefix of every token; lets secret scanners recognise leaked keys.
const TOKEN_PREFIX: &str = "bsk_";
/// Random bytes of a secret (base64url without padding: 43 characters).
const SECRET_BYTES: usize = 32;
const SECRET_CHARS: usize = 43;
/// How long a key read from the database is trusted.
pub(crate) const CACHE_TTL: Duration = Duration::from_secs(30);
/// Least time between two recorded uses of one key.
pub(crate) const TOUCH_INTERVAL: Duration = Duration::from_secs(60);
/// Most cached key ids. A full cache first drops unknown ids and stale entries, so a flood of random ids cannot
/// evict the fresh entries of existing keys; only a cache full of fresh existing keys is emptied.
const MAX_CACHED: usize = 10_000;
/// Most concurrent key reads from the database; further requests wait at most [`LOOKUP_WAIT`], then get a 503.
const MAX_LOOKUPS: usize = 16;
const LOOKUP_WAIT: Duration = Duration::from_secs(5);
/// Pending use records; more are dropped (the next use after the interval records again).
const TOUCH_QUEUE: usize = 1024;

/// The `auth:` section.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AuthConfig {
    #[serde(default)]
    pub(crate) enabled: bool,
    #[serde(default)]
    pub(crate) protect_metrics: bool,
}

/// A new token: shown once to the operator, stored only as its digest.
pub(crate) struct NewToken {
    pub(crate) id: ApiKeyId,
    pub(crate) token: SecretString,
    pub(crate) sha256: Sha256Digest,
}

/// A token with a fresh random key id and secret.
pub(crate) fn generate_token() -> Result<NewToken, getrandom::Error> {
    let mut bytes = [0u8; 8 + SECRET_BYTES];
    getrandom::fill(&mut bytes)?;
    let (id_bytes, secret) = bytes.split_at(8);
    let id = ApiKeyId::parse(&super::batches::store::hex(id_bytes)).expect("16 lowercase hex characters");
    let token = SecretString::new(format!(
        "{TOKEN_PREFIX}{}_{}",
        id.as_str(),
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(secret)
    ));
    bytes.fill(0);
    let sha256 = Sha256Digest(Sha256::digest(token.expose().as_bytes()).into());
    Ok(NewToken { id, token, sha256 })
}

/// The key id of a well-formed token `bsk_<16 hex>_<43 base64url>`.
fn token_key_id(token: &str) -> Option<ApiKeyId> {
    let rest = token.strip_prefix(TOKEN_PREFIX)?;
    let (id, secret) = rest.split_at_checked(ApiKeyId::LEN)?;
    let secret = secret.strip_prefix('_')?;
    if secret.len() != SECRET_CHARS || !secret.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
        return None;
    }
    ApiKeyId::parse(id).ok()
}

/// What a request needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Need {
    Nothing,
    Scope(ApiScopes),
    /// Any valid key or session (unknown routes: no route is revealed without one).
    AnyKey,
    /// A UI session, not a key (account routes of the UI).
    Session,
}

/// `ui`: the browser UI is served (its routes exist only then).
fn need(path: &str, protect_metrics: bool, ui: bool) -> Need {
    let under = |prefix: &str| path == prefix || path.strip_prefix(prefix).is_some_and(|r| r.starts_with('/'));
    match path {
        "/health" | "/health/batches" => Need::Nothing,
        "/metrics" if !protect_metrics => Need::Nothing,
        "/metrics" => Need::Scope(ApiScopes::METRICS),
        "/mcp" => Need::Scope(ApiScopes::MCP),
        "/v1/basal" | "/v1/systemone" | "/v1/decisions" | "/v1/models" => Need::Scope(ApiScopes::INFERENCE),
        _ if under("/v1/files") || under("/v1/batches") => Need::Scope(ApiScopes::BATCHES),
        // Logout also clears the cookie of an expired session; it checks the UI header itself.
        "/ui/api/config" | "/ui/api/login" | "/ui/api/logout" if ui => Need::Nothing,
        _ if ui && under("/ui/api") => Need::Session,
        _ if ui && under("/ui") => Need::Nothing,
        _ => Need::AnyKey,
    }
}

struct Cached {
    /// `None`: no key with this id, or its user is disabled.
    record: Option<ApiKeyRecord>,
    /// Database time of the read and the local instant it corresponds to.
    db_now: UnixMillis,
    fetched: Instant,
    /// Last use sent for recording.
    touched: Option<Instant>,
}

impl Cached {
    fn now(&self) -> UnixMillis {
        let elapsed = i64::try_from(self.fetched.elapsed().as_millis()).unwrap_or(i64::MAX);
        UnixMillis(self.db_now.0.saturating_add(elapsed))
    }
}

/// Verifies API keys against the database with a short-lived cache.
pub(crate) struct Authenticator {
    store: Store,
    protect_metrics: bool,
    /// The browser UI is served: its routes and session cookies are accepted.
    ui: bool,
    /// Never held across an await.
    cache: Mutex<HashMap<ApiKeyId, Cached>>,
    lookups: Semaphore,
    touches: mpsc::Sender<ApiKeyId>,
}

impl std::fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Authenticator").field("protect_metrics", &self.protect_metrics).finish_non_exhaustive()
    }
}

/// Why a request is refused.
enum Refusal {
    Unauthenticated,
    Forbidden,
    Unavailable,
    /// A session cookie without `X-Basal-UI: 1`.
    UiHeaderMissing,
}

impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        match self {
            Self::Unauthenticated => {
                let mut response =
                    ApiError::new(StatusCode::UNAUTHORIZED, "authentication_error", "Missing or invalid API key.")
                        .with_code("invalid_api_key")
                        .into_response();
                response.headers_mut().insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
                response
            }
            Self::Forbidden => {
                ApiError::new(StatusCode::FORBIDDEN, "permission_error", "The API key has no scope for this route.")
                    .with_code("insufficient_scope")
                    .into_response()
            }
            Self::Unavailable => ApiError::unavailable(5).into_response(),
            Self::UiHeaderMissing => ui_header_required().into_response(),
        }
    }
}

impl Authenticator {
    /// Start the authenticator over a connected store; spawns the task that records uses (it ends with the
    /// authenticator). Call inside the runtime.
    pub(crate) fn start(store: Store, config: &AuthConfig, ui: bool) -> Arc<Self> {
        let (touches, mut queue) = mpsc::channel::<ApiKeyId>(TOUCH_QUEUE);
        let recorder = store.clone();
        tokio::spawn(async move {
            while let Some(id) = queue.recv().await {
                if let Err(e) = recorder.touch_api_key(&id, TOUCH_INTERVAL).await {
                    crate::warn!("auth: recording the use of key {}: {e}", id.as_str());
                }
            }
        });
        Arc::new(Self {
            store,
            protect_metrics: config.protect_metrics,
            ui,
            cache: Mutex::new(HashMap::new()),
            lookups: Semaphore::new(MAX_LOOKUPS),
            touches,
        })
    }

    /// The key of `token` when it is active at the database time, with an unchanged digest.
    async fn verify(&self, token: &str) -> Result<ApiKeyRecord, Refusal> {
        let id = token_key_id(token).ok_or(Refusal::Unauthenticated)?;
        let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        let fresh = |c: &Cached| c.fetched.elapsed() < CACHE_TTL;
        let cached = self.with_cache(|cache| cache.get(&id).filter(|c| fresh(c)).map(|c| (c.record.clone(), c.now())));
        let (record, now) = match cached {
            Some(hit) => hit,
            None => {
                let _permit = self.lookup_permit().await?;
                // Stamped before the read: the entry never outlives CACHE_TTL from the data it holds.
                let fetched = Instant::now();
                let lookup = self.store.lookup_api_key(&id).await.map_err(|e| {
                    crate::warn!("auth: reading key {}: {e}", id.as_str());
                    Refusal::Unavailable
                })?;
                let (record, db_now) = match lookup {
                    Some(l) => ((!l.user_disabled).then_some(l.record), l.now),
                    None => (None, UnixMillis(0)),
                };
                let entry = Cached { record, db_now, fetched, touched: None };
                let hit = (entry.record.clone(), entry.now());
                self.with_cache(|cache| {
                    if cache.len() >= MAX_CACHED {
                        cache.retain(|_, c| c.record.is_some() && fresh(c));
                        if cache.len() >= MAX_CACHED {
                            cache.clear();
                        }
                    }
                    let touched = cache.get(&id).and_then(|c| c.touched);
                    cache.insert(id.clone(), Cached { touched, ..entry });
                });
                hit
            }
        };
        match record {
            Some(record) if bool::from(record.token_sha256.0.ct_eq(&digest)) && record.active_at(now) => {
                self.record_use(&id);
                Ok(record)
            }
            _ => Err(Refusal::Unauthenticated),
        }
    }

    /// The active session `id` with its user, read from the database (the lookup limit applies).
    async fn verify_session(&self, id: &Sha256Digest) -> Result<ApiSessionLookup, Refusal> {
        let _permit = self.lookup_permit().await?;
        match self.store.lookup_api_session(id).await {
            Ok(Some(lookup)) if lookup.active() => Ok(lookup),
            Ok(_) => Err(Refusal::Unauthenticated),
            Err(e) => {
                crate::warn!("auth: reading a UI session: {e}");
                Err(Refusal::Unavailable)
            }
        }
    }

    /// A permit of the lookup limit, waited for at most [`LOOKUP_WAIT`].
    async fn lookup_permit(&self) -> Result<SemaphorePermit<'_>, Refusal> {
        tokio::time::timeout(LOOKUP_WAIT, self.lookups.acquire())
            .await
            .map_err(|_| Refusal::Unavailable)?
            .map_err(|_| Refusal::Unavailable)
    }

    /// Queue a use record unless one was queued within [`TOUCH_INTERVAL`].
    fn record_use(&self, id: &ApiKeyId) {
        let due = self.with_cache(|cache| match cache.get_mut(id) {
            Some(c) if c.touched.is_none_or(|t| t.elapsed() >= TOUCH_INTERVAL) => {
                c.touched = Some(Instant::now());
                true
            }
            _ => false,
        });
        if due {
            let _ = self.touches.try_send(id.clone());
        }
    }

    fn with_cache<R>(&self, f: impl FnOnce(&mut HashMap<ApiKeyId, Cached>) -> R) -> R {
        f(&mut self.cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }
}

/// The token of the single `Authorization: Bearer <token>` header (scheme case-insensitive).
fn bearer(request: &Request) -> Option<&str> {
    let mut values = request.headers().get_all(AUTHORIZATION).iter();
    let (Some(value), None) = (values.next(), values.next()) else {
        return None;
    };
    let (scheme, token) = value.to_str().ok()?.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

/// Middleware of the whole router: authenticate a key (or, with the UI, a session), check the route's scope, then
/// pass the request on with the caller's [`OwnerId`] (and for a session its [`UiSession`]) as extensions (clients
/// cannot set extensions). Use with `axum::middleware::from_fn_with_state(authenticator, require_key)`.
pub(crate) async fn require_key(State(auth): State<Arc<Authenticator>>, request: Request, next: Next) -> Response {
    let need = need(request.uri().path(), auth.protect_metrics, auth.ui);
    if need == Need::Nothing {
        return next.run(request).await;
    }
    match authenticate(&auth, need, request).await {
        Ok(request) => next.run(request).await,
        Err(refusal) => refusal.into_response(),
    }
}

async fn authenticate(auth: &Authenticator, need: Need, mut request: Request) -> Result<Request, Refusal> {
    let session = match session::cookie_session(request.headers()) {
        Some(id) if auth.ui && !request.headers().contains_key(AUTHORIZATION) => Some(id),
        _ => None,
    };
    let (owner, scopes) = match session {
        Some(id) => {
            if !session::has_ui_header(request.headers()) {
                return Err(Refusal::UiHeaderMissing);
            }
            let lookup = auth.verify_session(&id).await?;
            let scopes = session::scopes(lookup.user.role);
            let owner = lookup.user.id.clone();
            request.extensions_mut().insert(UiSession { id, user: lookup.user });
            (owner, scopes)
        }
        None if need == Need::Session => return Err(Refusal::Unauthenticated),
        None => {
            let record = auth.verify(bearer(&request).ok_or(Refusal::Unauthenticated)?).await?;
            (record.user, record.scopes)
        }
    };
    if let Need::Scope(scope) = need {
        if !scopes.contains(scope) {
            return Err(Refusal::Forbidden);
        }
    }
    request.extensions_mut().insert::<OwnerId>(owner);
    Ok(request)
}

/// Connect the key store with the runtime account after the database's automatic migration (when configured).
/// Refuses a missing or outdated schema: run `basal database migrate` first.
pub(crate) async fn connect(database: &crate::database::Database) -> anyhow::Result<Store> {
    use anyhow::Context as _;
    database.prepare().await.context("preparing the database of the API keys")?;
    let options = database.options(crate::database::Account::Runtime).await?;
    super::batches::connect_store(&options, super::batches::store::StoreSettings::default())
        .await
        .context("connecting the database of the API keys")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_need_scopes() {
        let scope = |path| need(path, false, false);
        assert_eq!(scope("/health"), Need::Nothing);
        assert_eq!(scope("/metrics"), Need::Nothing);
        assert_eq!(need("/metrics", true, false), Need::Scope(ApiScopes::METRICS));
        assert_eq!(scope("/v1/decisions"), Need::Scope(ApiScopes::INFERENCE));
        assert_eq!(scope("/v1/files/file_1/content"), Need::Scope(ApiScopes::BATCHES));
        assert_eq!(scope("/v1/filesx"), Need::AnyKey, "a prefix is not a path segment");
        assert_eq!(scope("/mcp"), Need::Scope(ApiScopes::MCP));
    }

    #[test]
    fn ui_routes_exist_only_with_the_ui() {
        assert_eq!(need("/ui/", false, false), Need::AnyKey, "no UI: unknown route");
        assert_eq!(need("/ui/api/login", false, false), Need::AnyKey);
        assert_eq!(need("/ui", false, true), Need::Nothing);
        assert_eq!(need("/ui/app.js", false, true), Need::Nothing);
        assert_eq!(need("/ui/api/config", false, true), Need::Nothing);
        assert_eq!(need("/ui/api/login", false, true), Need::Nothing);
        assert_eq!(need("/ui/api/logout", false, true), Need::Nothing);
        assert_eq!(need("/ui/api/me", false, true), Need::Session);
        assert_eq!(need("/ui/api/admin/users", false, true), Need::Session);
        assert_eq!(need("/ui/apix", false, true), Need::Nothing, "a page name, not the API");
        assert_eq!(need("/uix", false, true), Need::AnyKey);
    }

    #[test]
    fn token_format() {
        let token = generate_token().unwrap();
        assert_eq!(token_key_id(token.token.expose()), Some(token.id.clone()));
        assert_eq!(token_key_id(&token.token.expose()[1..]), None);
        assert_eq!(token_key_id(&format!("{}x", token.token.expose())), None);
    }
}
