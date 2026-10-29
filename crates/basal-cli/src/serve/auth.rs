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
use tokio::sync::{mpsc, Semaphore};

use super::batches::dto::ApiError;
use super::batches::store::{ApiKeyId, ApiKeyRecord, ApiScopes, OwnerId, Sha256Digest, Store, UnixMillis};
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
    /// Any valid key (unknown routes: no route is revealed without one).
    AnyKey,
}

fn need(path: &str, protect_metrics: bool) -> Need {
    let under = |prefix: &str| path == prefix || path.strip_prefix(prefix).is_some_and(|r| r.starts_with('/'));
    match path {
        "/health" | "/health/batches" => Need::Nothing,
        "/metrics" if !protect_metrics => Need::Nothing,
        "/metrics" => Need::Scope(ApiScopes::METRICS),
        "/mcp" => Need::Scope(ApiScopes::MCP),
        "/v1/basal" | "/v1/systemone" | "/v1/models" => Need::Scope(ApiScopes::INFERENCE),
        _ if under("/v1/files") || under("/v1/batches") => Need::Scope(ApiScopes::BATCHES),
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
        }
    }
}

impl Authenticator {
    /// Start the authenticator over a connected store; spawns the task that records uses (it ends with the
    /// authenticator). Call inside the runtime.
    pub(crate) fn start(store: Store, config: &AuthConfig) -> Arc<Self> {
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
                let _permit = tokio::time::timeout(LOOKUP_WAIT, self.lookups.acquire())
                    .await
                    .map_err(|_| Refusal::Unavailable)?
                    .map_err(|_| Refusal::Unavailable)?;
                let lookup = self.store.lookup_api_key(&id).await.map_err(|e| {
                    crate::warn!("auth: reading key {}: {e}", id.as_str());
                    Refusal::Unavailable
                })?;
                let (record, db_now) = match lookup {
                    Some(l) => ((!l.user_disabled).then_some(l.record), l.now),
                    None => (None, UnixMillis(0)),
                };
                let entry = Cached { record, db_now, fetched: Instant::now(), touched: None };
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

/// Middleware of the whole router: authenticate, check the route's scope, then pass the request on with the key's
/// [`OwnerId`] as an extension (clients cannot set extensions). Use with
/// `axum::middleware::from_fn_with_state(authenticator, require_key)`.
pub(crate) async fn require_key(State(auth): State<Arc<Authenticator>>, mut request: Request, next: Next) -> Response {
    let need = need(request.uri().path(), auth.protect_metrics);
    if need == Need::Nothing {
        return next.run(request).await;
    }
    let verified = match bearer(&request) {
        Some(token) => auth.verify(token).await,
        None => Err(Refusal::Unauthenticated),
    };
    let record = match verified {
        Ok(record) => record,
        Err(refusal) => return refusal.into_response(),
    };
    if let Need::Scope(scope) = need {
        if !record.scopes.contains(scope) {
            return Refusal::Forbidden.into_response();
        }
    }
    request.extensions_mut().insert::<OwnerId>(record.user);
    next.run(request).await
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
