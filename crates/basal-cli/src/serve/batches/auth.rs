//! Owner identity of the Batch and Files routes.
//!
//! Every request needs `Authorization: Bearer <key_id>.<secret>`, also on a loopback listener and behind a local
//! proxy; the listener address, `Forwarded`/`X-Forwarded-*` and any client header or body field never identify the
//! owner. [`require_owner`] resolves the owner from the token and inserts it as a `store::OwnerId` request extension
//! (clients cannot set extensions); a missing or wrong token is a 401. The authentication of the inference routes
//! (`/v1/basal`, `/v1/systemone`) is not touched.
//!
//! Tokens are compared by SHA-256 in constant time; the server keeps only digests and never logs a token.
//!
//! - Local profile: owner `local`, key id `local`. On its first start the server writes a token of 32 random bytes to
//!   a private file (directory 0700, file 0600, created exclusively without following a symlink); a later start
//!   keeps the file after checking its owner and permissions. A local client reads the token from that file.
//! - Production: a JSON file maps key ids to owners and token digests (`sha256` of the whole bearer value
//!   `<key_id>.<secret>`, lowercase hex), for example `{"version":1,"keys":[{"key_id":"acme-2026-10","owner":"acme",
//!   "sha256":"…"}]}`. Rotation: restart with the new key next to the old one of the same owner, then without the
//!   old one. Secrets have at least 256 random bits ([`generate_token`]).

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use super::config::{ensure_private_dir, hex, AuthSettings, QuotaPolicy, SecretString};
use super::dto::ApiError;
use super::store;

/// Owner of the local profile.
pub(crate) const LOCAL_OWNER: &str = "local";
/// Key id of the local token.
const LOCAL_KEY_ID: &str = "local";
/// Random bytes of a generated token secret.
const SECRET_BYTES: usize = 32;
/// base64url length of 32 bytes without padding.
const MIN_SECRET_CHARS: usize = 43;
const MAX_SECRET_CHARS: usize = 256;
const MAX_KEY_ID: usize = 64;
const MAX_TOKENS_FILE_BYTES: u64 = 1024 * 1024;
const MAX_TOKEN_FILE_BYTES: u64 = 1024;
const MAX_KEYS: usize = 10_000;

/// The owner of a Batch/Files request, set only by [`require_owner`] from a verified token (or by the server for its
/// own work). 1–64 of `a-z`, `0-9`, `_`, `-`, starting with a letter or digit.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct OwnerId(Arc<str>);

impl OwnerId {
    pub(crate) fn parse(s: &str) -> Option<Self> {
        let valid = !s.is_empty()
            && s.len() <= 64
            && s.as_bytes()[0].is_ascii_alphanumeric()
            && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
        valid.then(|| Self(Arc::from(s)))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for OwnerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OwnerId({})", self.0)
    }
}

/// Starting the authenticator failed; no message contains a token.
#[derive(Debug, thiserror::Error)]
pub(crate) enum AuthError {
    #[error("the local batch token is only accepted on a loopback listener, not {0}")]
    LoopbackRequired(SocketAddr),
    #[error("batch token file {path}: {op}")]
    Io {
        path: PathBuf,
        op: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("batch token file {path}: {reason}")]
    Insecure { path: PathBuf, reason: &'static str },
    #[error("batch tokens file {path}: {reason}")]
    Invalid { path: PathBuf, reason: String },
    #[error("batch token owner `{0}` has no entry in the quota policy")]
    UnknownOwner(String),
    #[error("invalid key id (1-64 of A-Z, a-z, 0-9, `_`, `-`)")]
    InvalidKeyId,
    #[error("no system randomness for the batch token")]
    Random(#[source] getrandom::Error),
}

struct KeyEntry {
    owner: store::OwnerId,
    digest: [u8; 32],
}

/// Verifies bearer tokens of the Batch/Files routes. Holds key ids, owners and token digests only.
pub(crate) struct Authenticator {
    keys: HashMap<Box<str>, KeyEntry>,
    /// Compared against when the key id is unknown, so that path takes as long as a wrong secret.
    decoy: [u8; 32],
}

impl std::fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Authenticator").field("keys", &self.keys.len()).finish_non_exhaustive()
    }
}

/// What [`Authenticator::load`] found, for the startup log line (a location, never the secret).
#[derive(Debug)]
pub(crate) enum AuthStartup {
    /// Local token file; `created` on the first start.
    LocalToken {
        path: PathBuf,
        created: bool,
    },
    Tokens {
        keys: usize,
        owners: usize,
    },
}

impl Authenticator {
    /// Load the configured tokens: create or check the local token file, or read the production tokens file. Every
    /// owner must have limits in the quota policy. Blocking (file I/O): call before the runtime starts or from a
    /// blocking task.
    pub(crate) fn load(
        settings: &AuthSettings,
        listen: SocketAddr,
        quota: &QuotaPolicy,
    ) -> Result<(Self, AuthStartup), AuthError> {
        let mut decoy = [0u8; 32];
        getrandom::fill(&mut decoy).map_err(AuthError::Random)?;
        match settings {
            AuthSettings::LocalToken { token_file } => {
                if !listen.ip().is_loopback() {
                    return Err(AuthError::LoopbackRequired(listen));
                }
                let owner =
                    store::OwnerId::parse(LOCAL_OWNER).map_err(|_| AuthError::UnknownOwner(LOCAL_OWNER.into()))?;
                if quota.owner(owner.as_str()).is_none() {
                    return Err(AuthError::UnknownOwner(LOCAL_OWNER.into()));
                }
                let (token, created) = ensure_local_token(token_file)?;
                let digest: [u8; 32] = Sha256::digest(token.expose().as_bytes()).into();
                let keys = HashMap::from([(Box::from(LOCAL_KEY_ID), KeyEntry { owner, digest })]);
                Ok((Self { keys, decoy }, AuthStartup::LocalToken { path: token_file.clone(), created }))
            }
            AuthSettings::Tokens { tokens_file } => {
                let keys = read_tokens_file(tokens_file)?;
                let mut owners = HashSet::new();
                for entry in keys.values() {
                    if quota.owner(entry.owner.as_str()).is_none() {
                        return Err(AuthError::UnknownOwner(entry.owner.as_str().into()));
                    }
                    owners.insert(entry.owner.clone());
                }
                let startup = AuthStartup::Tokens { keys: keys.len(), owners: owners.len() };
                Ok((Self { keys, decoy }, startup))
            }
        }
    }

    /// Owner of a request from its single `Authorization: Bearer` header; 401 otherwise.
    pub(crate) fn authenticate(&self, headers: &HeaderMap) -> Result<store::OwnerId, ApiError> {
        let mut values = headers.get_all(AUTHORIZATION).iter();
        let (Some(value), None) = (values.next(), values.next()) else {
            return Err(ApiError::unauthorized());
        };
        let token = value.to_str().ok().and_then(bearer_token).ok_or_else(ApiError::unauthorized)?;
        let (key_id, _) = token.split_once('.').ok_or_else(ApiError::unauthorized)?;
        let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        match self.keys.get(key_id) {
            Some(entry) if bool::from(entry.digest.ct_eq(&digest)) => Ok(entry.owner.clone()),
            Some(_) => Err(ApiError::unauthorized()),
            None => {
                let _ = std::hint::black_box(self.decoy.ct_eq(&digest));
                Err(ApiError::unauthorized())
            }
        }
    }
}

/// Middleware of the Batch/Files routes: authenticate, then pass the request on with its `store::OwnerId` extension.
/// Use with `axum::middleware::from_fn_with_state(authenticator, require_owner)`.
pub(crate) async fn require_owner(
    State(auth): State<Arc<Authenticator>>,
    mut request: Request,
    next: Next,
) -> Response {
    match auth.authenticate(request.headers()) {
        Ok(owner) => {
            request.extensions_mut().insert(owner);
            next.run(request).await
        }
        Err(e) => e.into_response(),
    }
}

/// The token of `Bearer <token>` (scheme case-insensitive) when it has the shape `<key_id>.<secret>`.
fn bearer_token(header: &str) -> Option<&str> {
    let (scheme, token) = header.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let (key_id, secret) = token.split_once('.')?;
    (valid_key_id(key_id) && valid_secret(secret)).then_some(token)
}

fn valid_key_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= MAX_KEY_ID && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn valid_secret(s: &str) -> bool {
    (MIN_SECRET_CHARS..=MAX_SECRET_CHARS).contains(&s.len())
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// A new bearer token and its digest for the tokens file.
pub(crate) struct GeneratedToken {
    /// `<key_id>.<secret>`: shown once to the operator, never stored by the server.
    pub(crate) token: SecretString,
    /// `sha256` of the tokens file entry (lowercase hex).
    pub(crate) sha256: String,
}

/// A token for `key_id` with a secret of 32 random bytes (base64url without padding).
pub(crate) fn generate_token(key_id: &str) -> Result<GeneratedToken, AuthError> {
    if !valid_key_id(key_id) {
        return Err(AuthError::InvalidKeyId);
    }
    let mut secret = [0u8; SECRET_BYTES];
    getrandom::fill(&mut secret).map_err(AuthError::Random)?;
    let token =
        SecretString::new(format!("{key_id}.{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(secret)));
    secret.fill(0);
    let sha256 = hex(&Sha256::digest(token.expose().as_bytes()));
    Ok(GeneratedToken { token, sha256 })
}

/// Read the local token, creating it on the first start. The file is written to a temporary name and linked into
/// place (never replacing an existing file), so a concurrent start reads either no file or the whole token.
fn ensure_local_token(path: &Path) -> Result<(SecretString, bool), AuthError> {
    use std::os::unix::fs::OpenOptionsExt;
    let io = |op: &'static str| move |source: std::io::Error| AuthError::Io { path: path.to_path_buf(), op, source };
    let dir = path.parent().ok_or_else(|| AuthError::Insecure { path: path.into(), reason: "has no directory" })?;
    ensure_private_dir(dir).map_err(io("checking its directory"))?;
    if let Some(token) = read_local_token(path)? {
        return Ok((token, false));
    }
    let generated = generate_token(LOCAL_KEY_ID)?;
    let tmp = dir.join(format!(".local-token.{}.tmp", uuid::Uuid::new_v4().simple()));
    let written = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&tmp)?;
        file.write_all(generated.token.expose().as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()
    })();
    let linked = written.and_then(|()| std::fs::hard_link(&tmp, path));
    let _ = std::fs::remove_file(&tmp);
    let created = match linked {
        Ok(()) => true,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => false,
        Err(e) => return Err(io("creating")(e)),
    };
    std::fs::File::open(dir).and_then(|d| d.sync_all()).map_err(io("syncing its directory"))?;
    if created {
        return Ok((generated.token, true));
    }
    // Another start created it first: use that one.
    let token = read_local_token(path)?.ok_or_else(|| AuthError::Insecure { path: path.into(), reason: "vanished" })?;
    Ok((token, false))
}

/// The existing local token (`None` when there is no file): a regular file of this user, mode 0600 or stricter, not
/// a symlink, holding `local.<secret>`.
fn read_local_token(path: &Path) -> Result<Option<SecretString>, AuthError> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
    let insecure = |reason| AuthError::Insecure { path: path.into(), reason };
    let file = match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => return Err(insecure("is a symbolic link")),
        Err(source) => return Err(AuthError::Io { path: path.into(), op: "opening", source }),
    };
    let meta = file.metadata().map_err(|source| AuthError::Io { path: path.into(), op: "reading", source })?;
    if !meta.is_file() {
        return Err(insecure("is not a regular file"));
    }
    if meta.uid() != super::config::effective_uid() {
        return Err(insecure("is owned by another user"));
    }
    if meta.permissions().mode() & 0o077 != 0 {
        return Err(insecure("is accessible to group or others (expected mode 0600)"));
    }
    if meta.len() > MAX_TOKEN_FILE_BYTES {
        return Err(insecure("is too large"));
    }
    let mut text = String::new();
    file.take(MAX_TOKEN_FILE_BYTES).read_to_string(&mut text).map_err(|source| AuthError::Io {
        path: path.into(),
        op: "reading",
        source,
    })?;
    let token = SecretString::new(text);
    let value = token.expose().strip_suffix('\n').unwrap_or(token.expose());
    let valid = value.split_once('.').is_some_and(|(id, secret)| id == LOCAL_KEY_ID && valid_secret(secret));
    if !valid {
        return Err(insecure("does not hold a local batch token"));
    }
    Ok(Some(SecretString::new(value.to_string())))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokensFile {
    version: u32,
    keys: Vec<TokenEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenEntry {
    key_id: String,
    owner: String,
    sha256: String,
}

fn read_tokens_file(path: &Path) -> Result<HashMap<Box<str>, KeyEntry>, AuthError> {
    use std::os::unix::fs::PermissionsExt;
    let invalid = |reason: String| AuthError::Invalid { path: path.into(), reason };
    let io = |op| move |source| AuthError::Io { path: path.into(), op, source };
    let file = std::fs::File::open(path).map_err(io("opening"))?;
    let meta = file.metadata().map_err(io("reading"))?;
    if !meta.is_file() {
        return Err(AuthError::Insecure { path: path.into(), reason: "is not a regular file" });
    }
    if meta.permissions().mode() & 0o022 != 0 {
        return Err(AuthError::Insecure { path: path.into(), reason: "is writable by group or others" });
    }
    if meta.len() > MAX_TOKENS_FILE_BYTES {
        return Err(invalid("larger than 1 MiB".into()));
    }
    let mut bytes = Vec::new();
    file.take(MAX_TOKENS_FILE_BYTES + 1).read_to_end(&mut bytes).map_err(io("reading"))?;
    let parsed: TokensFile = serde_json::from_slice(&bytes).map_err(|e| invalid(format!("not valid: {e}")))?;
    if parsed.version != 1 {
        return Err(invalid(format!("version {} is not supported (1)", parsed.version)));
    }
    if parsed.keys.is_empty() || parsed.keys.len() > MAX_KEYS {
        return Err(invalid(format!("needs 1..={MAX_KEYS} keys")));
    }
    let mut keys = HashMap::with_capacity(parsed.keys.len());
    let mut digests = HashSet::with_capacity(parsed.keys.len());
    for (i, entry) in parsed.keys.into_iter().enumerate() {
        if !valid_key_id(&entry.key_id) {
            return Err(invalid(format!("keys[{i}].key_id: 1-64 of A-Z, a-z, 0-9, `_`, `-`")));
        }
        let owner = OwnerId::parse(&entry.owner)
            .and_then(|o| store::OwnerId::parse(o.as_str()).ok())
            .ok_or_else(|| invalid(format!("keys[{i}].owner: 1-64 of a-z, 0-9, `_`, `-`")))?;
        let digest =
            parse_digest(&entry.sha256).ok_or_else(|| invalid(format!("keys[{i}].sha256: 64 lowercase hex digits")))?;
        if !digests.insert(digest) {
            return Err(invalid(format!("keys[{i}].sha256: the same token is listed twice")));
        }
        if keys.insert(entry.key_id.into_boxed_str(), KeyEntry { owner, digest }).is_some() {
            return Err(invalid(format!("keys[{i}].key_id: listed twice")));
        }
    }
    Ok(keys)
}

/// 64 lowercase hex digits.
pub(crate) fn parse_digest(s: &str) -> Option<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = nibble(b[2 * i])? << 4 | nibble(b[2 * i + 1])?;
    }
    Some(out)
}
