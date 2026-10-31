//! Account API of the browser UI (`/ui/api/...`): login, logout, the caller's account and keys, and the admin's
//! users and keys. Contract: `docs/UI.md`.
//!
//! `serve::auth::require_key` authenticates these routes: everything except `config`, `login` and `logout` needs a session
//! cookie with `X-Basal-UI: 1` (API keys are refused) and gets the session's [`UiSession`]. Admin routes check the
//! role read with the session on this request. Bodies are small JSON objects parsed strictly (unknown fields: 400).
//! Answers are `Cache-Control: no-store` (set with the security headers of the UI router).

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::header::SET_COOKIE;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use super::session::{self, LoginThrottle};
use super::UiSettings;
use crate::serve::auth::{generate_token, CACHE_TTL};
use crate::serve::batches::dto::{percent_decode, ApiError};
use crate::serve::batches::store::{
    check_api_name, ApiKeyId, ApiKeyRecord, ApiRole, ApiScopes, ApiUserRecord, NewApiKey, NewApiSession, OwnerId,
    PasswordHash, QuotaUsage, Sha256Digest, Store, StoreError, UnixMillis,
};
use crate::serve::stats::Stats;

/// Most users and keys a list answers.
const MAX_LIST: u32 = 1000;
/// Concurrent password hashes (19 MiB and tens of milliseconds of CPU each); more wait at most [`HASH_WAIT`].
const MAX_HASHING: usize = 2;
const HASH_WAIT: Duration = Duration::from_secs(10);
const RETRY_AFTER_S: u32 = 5;

/// The session of a cookie-authenticated request, set by `serve::auth::require_key` (clients cannot set extensions).
#[derive(Clone, Debug)]
pub(crate) struct UiSession {
    pub(crate) id: Sha256Digest,
    /// Read from the database with the session on this request.
    pub(crate) user: ApiUserRecord,
}

/// Shared state of the account routes.
pub(super) struct UiState {
    pub(super) settings: UiSettings,
    /// Answer of `GET /ui/api/config`.
    pub(super) info: Value,
    /// The key store; `None` without `auth` (no accounts: only `config` is routed).
    pub(super) store: Option<Store>,
    /// Statistics of the admin server view; set with accounts.
    stats: Option<Arc<Stats>>,
    throttle: LoginThrottle,
    hashing: Semaphore,
}

impl UiState {
    pub(super) fn new(settings: UiSettings, info: Value, store: Option<Store>, stats: Option<Arc<Stats>>) -> Self {
        Self { settings, info, store, stats, throttle: LoginThrottle::default(), hashing: Semaphore::new(MAX_HASHING) }
    }

    fn store(&self) -> &Store {
        self.store.as_ref().expect("account routes are routed only with a store")
    }

    /// Run blocking password work under the hashing limit.
    async fn hashing<R: Send + 'static>(&self, work: impl FnOnce() -> R + Send + 'static) -> Result<R, ApiError> {
        let _permit =
            tokio::time::timeout(HASH_WAIT, self.hashing.acquire()).await.map_err(|_| busy())?.map_err(|_| busy())?;
        tokio::task::spawn_blocking(work).await.map_err(|e| {
            crate::warn!("ui: password hashing task: {e}");
            ApiError::internal()
        })
    }

    async fn hash(&self, password: String) -> Result<PasswordHash, ApiError> {
        self.hashing(move || session::hash_password(&password)).await?.map_err(|e| {
            crate::warn!("ui: {e:#}");
            ApiError::internal()
        })
    }

    /// Check and hash a new password, if one is given.
    async fn hash_new(&self, password: Option<String>) -> Result<Option<PasswordHash>, ApiError> {
        let Some(password) = password else {
            return Ok(None);
        };
        check_new_password(&password)?;
        self.hash(password).await.map(Some)
    }
}

type ApiResult<T> = Result<T, ApiError>;

fn busy() -> ApiError {
    ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", "Too many logins at once; try again.")
        .with_code("busy")
        .with_retry_after(RETRY_AFTER_S)
}

/// HTTP answer of a store error that the caller has not mapped.
fn store_error(error: StoreError) -> ApiError {
    match error {
        StoreError::Invalid(value) => ApiError::invalid_request(value.to_string()),
        e => {
            crate::warn!("ui: account database: {e}");
            ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", "The account database is unavailable.")
                .with_code("database_unavailable")
                .with_retry_after(RETRY_AFTER_S)
        }
    }
}

fn no_randomness(error: getrandom::Error) -> ApiError {
    crate::warn!("ui: reading system randomness: {error}");
    ApiError::internal()
}

fn not_found(what: &'static str) -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "invalid_request_error", format!("No such {what}.")).with_code("not_found")
}

fn conflict(message: &'static str, code: &'static str) -> ApiError {
    ApiError::new(StatusCode::CONFLICT, "invalid_request_error", message).with_code(code)
}

/// A strict JSON object body.
fn body<T: DeserializeOwned>(bytes: &[u8]) -> ApiResult<T> {
    serde_json::from_slice(bytes).map_err(|e| ApiError::invalid_request(format!("Invalid request body: {e}")))
}

/// `?all=true` and, with `user`, `?user=<id>`; other parameters are a 400.
fn list_query(uri: &Uri, with_user: bool) -> ApiResult<(bool, Option<OwnerId>)> {
    let (mut all, mut user) = (false, None);
    for pair in uri.query().unwrap_or("").split('&').filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        let value = percent_decode(value).ok_or_else(|| ApiError::invalid_request("Malformed query string."))?;
        match key {
            "all" => all = value == "true",
            "user" if with_user => user = Some(parse_user(&value)?),
            _ => return Err(ApiError::invalid_request("Unknown query parameter.").with_param(key.to_owned())),
        }
    }
    Ok((all, user))
}

fn parse_user(id: &str) -> ApiResult<OwnerId> {
    OwnerId::parse(id).map_err(|e| ApiError::invalid_request(e.to_string()).with_param("user"))
}

fn parse_key_id(id: &str) -> ApiResult<ApiKeyId> {
    ApiKeyId::parse(id).map_err(|_| not_found("key"))
}

fn check_new_password(password: &str) -> ApiResult<()> {
    session::check_password(password)
        .map_err(|message| ApiError::invalid_request(message).with_param("password").with_code("weak_password"))
}

fn secs(t: UnixMillis) -> i64 {
    t.as_unix_seconds()
}

fn user_json(user: &ApiUserRecord) -> Value {
    json!({
        "id": user.id.as_str(),
        "name": user.name,
        "role": user.role.as_str(),
        "created_at": secs(user.created_at),
        "disabled_at": user.disabled_at.map(secs),
        "has_password": user.password.is_some(),
    })
}

fn key_json(key: &ApiKeyRecord) -> Value {
    json!({
        "id": key.id.as_str(),
        "user": key.user.as_str(),
        "name": key.name,
        "scopes": key.scopes.names(),
        "created_at": secs(key.created_at),
        "expires_at": key.expires_at.map(secs),
        "revoked_at": key.revoked_at.map(secs),
        "last_used_at": key.last_used_at.map(secs),
        "status": crate::keys::status(key),
    })
}

fn usage_json(usage: &QuotaUsage) -> Value {
    json!({
        "used_bytes": usage.used_bytes,
        "reserved_bytes": usage.reserved_bytes,
        "active_batches": usage.active_batches,
        "retained_batches": usage.retained_batches,
        "files": usage.files,
        "open_sessions": usage.open_sessions,
    })
}

fn me_json(user: &ApiUserRecord) -> Value {
    json!({ "user": user_json(user), "scopes": session::scopes(user.role).names() })
}

fn admin(session: &UiSession) -> ApiResult<()> {
    match session.user.role {
        ApiRole::Admin => Ok(()),
        ApiRole::User => Err(ApiError::new(StatusCode::FORBIDDEN, "permission_error", "This needs an admin account.")
            .with_code("admin_required")),
    }
}

/// Refuse an admin action on the admin's own account (it would lock the admin out).
fn not_self(session: &UiSession, user: &OwnerId) -> ApiResult<()> {
    if session.user.id == *user {
        return Err(conflict("An admin cannot do this to their own account.", "self"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Public

pub(super) async fn config(State(state): State<Arc<UiState>>) -> Json<Value> {
    Json(state.info.clone())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Login {
    user: String,
    password: String,
}

/// Every failure is the same 401 (unknown user, no password, wrong password, disabled user), after the same hashing
/// work; ten failures of one user id within 15 minutes refuse it (429) until the window ends.
pub(super) async fn login(State(state): State<Arc<UiState>>, headers: HeaderMap, bytes: Bytes) -> ApiResult<Response> {
    if !session::has_ui_header(&headers) {
        return Err(ui_header_required());
    }
    let Login { user, password } = body(&bytes)?;
    let invalid = || {
        ApiError::new(StatusCode::UNAUTHORIZED, "authentication_error", "Invalid user or password.")
            .with_code("invalid_credentials")
    };
    // A malformed id names no account and reveals none: refused without hashing or throttle state.
    let Ok(id) = OwnerId::parse(&user) else {
        return Err(invalid());
    };
    if let Err(wait) = state.throttle.check(&id) {
        return Err(ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "Too many failed logins for this user; try again later.",
        )
        .with_code("too_many_attempts")
        .with_retry_after(u32::try_from(wait.as_secs()).unwrap_or(u32::MAX)));
    }
    let record = state.store().api_user(&id).await.map_err(store_error)?;
    let hash = record.as_ref().filter(|u| u.disabled_at.is_none()).and_then(|u| u.password.clone());
    let verified_hash = hash.clone();
    let verified = state.hashing(move || session::verify_password(&password, hash.as_ref())).await?;
    let Some(record) = record.filter(|_| verified) else {
        state.throttle.failed(&id);
        return Err(invalid());
    };
    state.throttle.succeeded(&id);
    let new = session::generate_session().map_err(no_randomness)?;
    let session =
        NewApiSession { id: new.id, user: record.id, password: verified_hash, ttl: state.settings.session_ttl };
    let created = state.store().create_api_session(&session).await.map_err(|e| match e {
        StoreError::NotFound | StoreError::UserDisabled => invalid(),
        e => store_error(e),
    })?;
    let cookie = session::set_cookie(&new.token, state.settings.session_ttl, state.settings.cookie_secure);
    Ok(([(SET_COOKIE, cookie)], Json(me_json(&created.user))).into_response())
}

/// 403 for a cookie request without `X-Basal-UI: 1` (also answered by `serve::auth`).
pub(crate) fn ui_header_required() -> ApiError {
    ApiError::new(StatusCode::FORBIDDEN, "permission_error", "Requests of the UI need the header X-Basal-UI: 1.")
        .with_code("ui_header_required")
}

// ---------------------------------------------------------------------------------------------------------------
// The caller's account

/// Delete the session of the cookie, if any, and clear the cookie: also for an expired session, which the auth
/// middleware would refuse (so this route is public and checks the UI header itself).
pub(super) async fn logout(State(state): State<Arc<UiState>>, headers: HeaderMap) -> ApiResult<Response> {
    if !session::has_ui_header(&headers) {
        return Err(ui_header_required());
    }
    if let Some(id) = session::cookie_session(&headers) {
        state.store().delete_api_session(&id).await.map_err(store_error)?;
    }
    Ok((StatusCode::NO_CONTENT, [(SET_COOKIE, session::clear_cookie(state.settings.cookie_secure))]).into_response())
}

pub(super) async fn me(Extension(session): Extension<UiSession>) -> Json<Value> {
    Json(me_json(&session.user))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChangePassword {
    current_password: String,
    new_password: String,
}

/// Change the caller's password; the caller's other sessions end. A wrong current password counts as a failed login.
pub(super) async fn change_password(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    bytes: Bytes,
) -> ApiResult<StatusCode> {
    let ChangePassword { current_password, new_password } = body(&bytes)?;
    check_new_password(&new_password)?;
    let user = &session.user.id;
    if state.throttle.check(user).is_err() {
        return Err(ApiError::new(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", "Too many failed attempts.")
            .with_code("too_many_attempts"));
    }
    let hash = session.user.password.clone();
    if !state.hashing(move || session::verify_password(&current_password, hash.as_ref())).await? {
        state.throttle.failed(user);
        return Err(ApiError::invalid_request("The current password is wrong.")
            .with_param("current_password")
            .with_code("wrong_password"));
    }
    let hash = state.hash(new_password).await?;
    state.store().set_api_user_password(&session.user.id, Some(&hash), Some(&session.id)).await.map_err(store_error)?;
    Ok(StatusCode::NO_CONTENT)
}

pub(super) async fn my_keys(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    uri: Uri,
) -> ApiResult<Json<Value>> {
    let (all, _) = list_query(&uri, false)?;
    let keys = state.store().list_api_keys(Some(&session.user.id), all, MAX_LIST).await.map_err(store_error)?;
    Ok(Json(json!({ "data": keys.iter().map(key_json).collect::<Vec<_>>() })))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NewKey {
    /// Required on the admin route, refused on the caller's own.
    #[serde(default)]
    user: Option<String>,
    name: String,
    scopes: Vec<String>,
    #[serde(default)]
    expires_in: Option<String>,
}

/// Validated parts of a new key.
fn new_key_parts(key: &NewKey) -> ApiResult<(ApiScopes, Option<Duration>)> {
    check_api_name(&key.name).map_err(|e| ApiError::invalid_request(e.to_string()).with_param("name"))?;
    let scopes = ApiScopes::parse(&key.scopes.join(","))
        .map_err(|e| ApiError::invalid_request(e.to_string()).with_param("scopes"))?;
    let expires_in = key
        .expires_in
        .as_deref()
        .map(crate::keys::parse_duration)
        .transpose()
        .map_err(|e| ApiError::invalid_request(e).with_param("expires_in"))?;
    Ok((scopes, expires_in))
}

async fn create_key(
    state: &UiState,
    user: OwnerId,
    key: NewKey,
    scopes: ApiScopes,
    expires_in: Option<Duration>,
) -> ApiResult<Response> {
    let token = generate_token().map_err(no_randomness)?;
    let new = NewApiKey { id: token.id, user, name: key.name, token_sha256: token.sha256, scopes, expires_in };
    let record = state.store().create_api_key(&new).await.map_err(|e| match e {
        StoreError::NotFound => not_found("user"),
        StoreError::UserDisabled => conflict("The user is disabled.", "user_disabled"),
        e => store_error(e),
    })?;
    let body = json!({ "key": key_json(&record), "token": token.token.expose() });
    Ok((StatusCode::CREATED, Json(body)).into_response())
}

/// A key of the caller: scopes `inference`, `batches`, `mcp`, and `metrics` only for admins.
pub(super) async fn create_my_key(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    bytes: Bytes,
) -> ApiResult<Response> {
    let key: NewKey = body(&bytes)?;
    if key.user.is_some() {
        return Err(ApiError::invalid_request("A key of your own account has no user field.").with_param("user"));
    }
    let (scopes, expires_in) = new_key_parts(&key)?;
    if scopes.contains(ApiScopes::METRICS) && session.user.role != ApiRole::Admin {
        return Err(ApiError::new(StatusCode::FORBIDDEN, "permission_error", "Only admins can create metrics keys.")
            .with_param("scopes")
            .with_code("admin_required"));
    }
    create_key(&state, session.user.id, key, scopes, expires_in).await
}

/// Revoke a key of the caller; another user's key is a 404.
pub(super) async fn revoke_my_key(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let id = parse_key_id(&id)?;
    let key = state.store().lookup_api_key(&id).await.map_err(store_error)?;
    if key.is_none_or(|k| k.record.user != session.user.id) {
        return Err(not_found("key"));
    }
    revoke(&state, &id).await
}

async fn revoke(state: &UiState, id: &ApiKeyId) -> ApiResult<Json<Value>> {
    let key = state.store().revoke_api_key(id).await.map_err(|e| match e {
        StoreError::NotFound => not_found("key"),
        e => store_error(e),
    })?;
    Ok(Json(key_json(&key)))
}

// ---------------------------------------------------------------------------------------------------------------
// Admin

/// Users with their active key count and Batch usage (`null` when it cannot be read).
pub(super) async fn users(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    uri: Uri,
) -> ApiResult<Json<Value>> {
    admin(&session)?;
    let (all, _) = list_query(&uri, false)?;
    let store = state.store();
    let users = store.list_api_users(all, MAX_LIST).await.map_err(store_error)?;
    let keys = store.list_api_keys(None, false, 100_000).await.map_err(store_error)?;
    let mut data = Vec::with_capacity(users.len());
    for user in &users {
        let mut row = user_json(user);
        row["active_keys"] =
            json!(keys.iter().filter(|k| k.user == user.id && crate::keys::status(k) == "active").count());
        row["usage"] = store.quota_usage(&user.id).await.map_or(Value::Null, |u| usage_json(&u));
        data.push(row);
    }
    Ok(Json(json!({ "data": data })))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NewUser {
    id: String,
    name: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    password: Option<String>,
}

pub(super) async fn create_user(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    bytes: Bytes,
) -> ApiResult<Response> {
    admin(&session)?;
    let NewUser { id, name, role, password } = body(&bytes)?;
    let id = OwnerId::parse(&id).map_err(|e| ApiError::invalid_request(e.to_string()).with_param("id"))?;
    check_api_name(&name).map_err(|e| ApiError::invalid_request(e.to_string()).with_param("name"))?;
    let role = role
        .as_deref()
        .map_or(Ok(ApiRole::User), ApiRole::parse)
        .map_err(|e| ApiError::invalid_request(e.to_string()).with_param("role"))?;
    let hash = state.hash_new(password).await?;
    let user = state
        .store()
        .create_api_user(&id, &name, role, hash.as_ref())
        .await
        .map_err(store_error)?
        .ok_or_else(|| conflict("A user with this id exists.", "user_exists"))?;
    Ok((StatusCode::CREATED, Json(user_json(&user))).into_response())
}

fn user_not_found(error: StoreError) -> ApiError {
    match error {
        StoreError::NotFound => not_found("user"),
        e => store_error(e),
    }
}

/// Disable a user (its keys stop within 30 s, its sessions at once) or enable it again.
async fn set_disabled(state: &UiState, session: &UiSession, id: &str, disabled: bool) -> ApiResult<Json<Value>> {
    admin(session)?;
    let id = parse_user(id)?;
    not_self(session, &id)?;
    let user = state.store().set_api_user_disabled(&id, disabled).await.map_err(user_not_found)?;
    Ok(Json(user_json(&user)))
}

pub(super) async fn disable_user(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    set_disabled(&state, &session, &id, true).await
}

pub(super) async fn enable_user(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    set_disabled(&state, &session, &id, false).await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetPassword {
    password: Option<String>,
}

/// Set (or with `null` clear) a user's password; its sessions end.
pub(super) async fn set_password(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    Path(id): Path<String>,
    bytes: Bytes,
) -> ApiResult<Json<Value>> {
    admin(&session)?;
    let id = parse_user(&id)?;
    // The admin's own password changes through `change_password`, which asks for the current one.
    not_self(&session, &id)?;
    let SetPassword { password } = body(&bytes)?;
    let hash = state.hash_new(password).await?;
    let user = state.store().set_api_user_password(&id, hash.as_ref(), None).await.map_err(user_not_found)?;
    Ok(Json(user_json(&user)))
}

/// Delete a user disabled at least [`CACHE_TTL`] ago without Batch data, with its keys and sessions.
pub(super) async fn delete_user(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    admin(&session)?;
    let id = parse_user(&id)?;
    not_self(&session, &id)?;
    state.store().delete_api_user(&id, CACHE_TTL).await.map_err(|e| match e {
        StoreError::UserNotDisabled => conflict(
            "Disable the user and wait 30 seconds (servers may still accept its cached keys) before deleting it.",
            "user_not_disabled",
        ),
        StoreError::UserHasData => conflict(
            "The user still has Batch files, batches or upload sessions; delete them or let them expire first.",
            "user_has_data",
        ),
        e => user_not_found(e),
    })?;
    Ok(Json(json!({ "id": id.as_str(), "deleted": true })))
}

pub(super) async fn keys(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    uri: Uri,
) -> ApiResult<Json<Value>> {
    admin(&session)?;
    let (all, user) = list_query(&uri, true)?;
    let keys = state.store().list_api_keys(user.as_ref(), all, MAX_LIST).await.map_err(store_error)?;
    Ok(Json(json!({ "data": keys.iter().map(key_json).collect::<Vec<_>>() })))
}

pub(super) async fn create_user_key(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    bytes: Bytes,
) -> ApiResult<Response> {
    admin(&session)?;
    let key: NewKey = body(&bytes)?;
    let user = parse_user(
        key.user.as_deref().ok_or_else(|| ApiError::invalid_request("user is required.").with_param("user"))?,
    )?;
    let (scopes, expires_in) = new_key_parts(&key)?;
    create_key(&state, user, key, scopes, expires_in).await
}

pub(super) async fn revoke_key(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    admin(&session)?;
    revoke(&state, &parse_key_id(&id)?).await
}

/// The statistics of an admin's request (404 on a server without them).
fn stats<'a>(state: &'a UiState, session: &UiSession) -> ApiResult<&'a Stats> {
    admin(session)?;
    state.stats.as_deref().ok_or_else(|| not_found("statistics"))
}

/// `GET /ui/api/admin/stats[?since=SEQ]`: the server view's snapshot, with the history after point `SEQ` (0: all
/// of it and the configuration).
pub(super) async fn stats_snapshot(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    uri: Uri,
) -> ApiResult<Json<Value>> {
    let stats = stats(&state, &session)?;
    let mut since = 0;
    for pair in uri.query().unwrap_or("").split('&').filter(|p| !p.is_empty()) {
        match pair.split_once('=') {
            Some(("since", value)) => {
                since = value.parse().map_err(|_| ApiError::invalid_request("Invalid since.").with_param("since"))?;
            }
            _ => return Err(ApiError::invalid_request("Unknown query parameter.")),
        }
    }
    Ok(Json(stats.snapshot(since)))
}

/// `POST /ui/api/admin/stats/reset`: zero the counters, the history and the recent requests of this process.
pub(super) async fn reset_stats(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
) -> ApiResult<StatusCode> {
    stats(&state, &session)?.reset();
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /ui/api/admin/requests`: the most recent inference requests, newest first.
pub(super) async fn recent_requests(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
) -> ApiResult<Json<Value>> {
    Ok(Json(stats(&state, &session)?.recent()))
}

/// `GET /ui/api/admin/requests/{id}`: one recent request, with its bodies when they are captured.
pub(super) async fn recent_request(
    State(state): State<Arc<UiState>>,
    Extension(session): Extension<UiSession>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let stats = stats(&state, &session)?;
    id.parse().ok().and_then(|id| stats.request(id)).map(Json).ok_or_else(|| not_found("request"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_query_parses_known_parameters_only() {
        let uri: Uri = "/ui/api/admin/keys?all=true&user=acme%40corp".parse().unwrap();
        let (all, user) = list_query(&uri, true).unwrap();
        assert!(all);
        assert_eq!(user.unwrap().as_str(), "acme@corp");
        let uri: Uri = "/ui/api/me/keys?user=acme".parse().unwrap();
        assert!(list_query(&uri, false).is_err(), "user is not a parameter of the caller's keys");
        let uri: Uri = "/ui/api/me/keys?limit=3".parse().unwrap();
        assert!(list_query(&uri, true).is_err());
    }

    #[test]
    fn new_key_validation() {
        let key = |scopes: &[&str], expires_in: Option<&str>| NewKey {
            user: None,
            name: "crm".into(),
            scopes: scopes.iter().map(|s| (*s).to_owned()).collect(),
            expires_in: expires_in.map(str::to_owned),
        };
        let (scopes, expires) = new_key_parts(&key(&["batches", "inference"], Some("90d"))).unwrap();
        assert_eq!(scopes.to_text(), "inference,batches");
        assert_eq!(expires, Some(Duration::from_secs(90 * 86_400)));
        assert!(new_key_parts(&key(&[], None)).is_err(), "no scope");
        assert!(new_key_parts(&key(&["admin"], None)).is_err(), "unknown scope");
        assert!(new_key_parts(&key(&["inference"], Some("soon"))).is_err(), "bad duration");
    }

    #[test]
    fn strict_bodies() {
        assert!(body::<Login>(br#"{"user":"a","password":"b"}"#).is_ok());
        assert!(body::<Login>(br#"{"user":"a","password":"b","remember":true}"#).is_err());
        assert!(body::<SetPassword>(br#"{"password":null}"#).unwrap().password.is_none());
    }
}
