//! Accounts of the browser UI: argon2id passwords, session tokens and their cookie, and the login throttle.
//!
//! A session token `bss_<43 base64url>` (256 random bits) lives only in the `basal_session` cookie (HttpOnly,
//! SameSite=Strict, Secure unless `ui.cookie_secure: false`); the database keeps its SHA-256 as the session id.
//! Every request authenticated by the cookie must also carry `X-Basal-UI: 1`: browsers send that header cross-origin
//! only after a CORS preflight, which this server never approves, so another site cannot make requests with it.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use argon2::{Argon2, PasswordHasher as _, PasswordVerifier as _};
use axum::http::header::COOKIE;
use axum::http::{HeaderMap, HeaderValue};
use base64::Engine as _;
use sha2::{Digest, Sha256};

use crate::config_support::SecretString;
use crate::serve::batches::store::{ApiRole, ApiScopes, OwnerId, PasswordHash, Sha256Digest};

/// Name of the session cookie.
const COOKIE_NAME: &str = "basal_session";
/// Header every cookie-authenticated request carries (value `1`).
pub(crate) const UI_HEADER: &str = "x-basal-ui";
const TOKEN_PREFIX: &str = "bss_";
const SECRET_BYTES: usize = 32;
const SECRET_CHARS: usize = 43;
const SALT_BYTES: usize = 16;
/// Password length limits: at least 12 characters; the upper bound only caps the hashing input.
const MIN_PASSWORD_CHARS: usize = 12;
const MAX_PASSWORD_BYTES: usize = 1024;
/// Failed logins of one user id within [`THROTTLE_WINDOW`] before further attempts are refused until it ends.
const MAX_FAILURES: u32 = 10;
const THROTTLE_WINDOW: Duration = Duration::from_secs(15 * 60);
/// Most user ids the throttle tracks; when full, expired entries go first, then the oldest.
const MAX_TRACKED: usize = 10_000;

/// Scopes of a session: inference and the user's files and batches; admins also read `/metrics`.
pub(crate) fn scopes(role: ApiRole) -> ApiScopes {
    let base = ApiScopes::INFERENCE.union(ApiScopes::BATCHES);
    match role {
        ApiRole::User => base,
        ApiRole::Admin => base.union(ApiScopes::METRICS),
    }
}

/// Refuse a password shorter than [`MIN_PASSWORD_CHARS`] characters or longer than 1024 bytes.
pub(crate) fn check_password(password: &str) -> Result<(), &'static str> {
    if password.chars().count() < MIN_PASSWORD_CHARS {
        return Err("The password must have at least 12 characters.");
    }
    if password.len() > MAX_PASSWORD_BYTES {
        return Err("The password must have at most 1024 bytes.");
    }
    Ok(())
}

/// Argon2id PHC hash of a password with a fresh random salt (default parameters: 19 MiB, 2 passes). Blocking: about
/// as much CPU time as one verification.
pub(crate) fn hash_password(password: &str) -> anyhow::Result<PasswordHash> {
    let mut salt = [0u8; SALT_BYTES];
    getrandom::fill(&mut salt).map_err(|e| anyhow::anyhow!("reading system randomness: {e}"))?;
    let hash = Argon2::default()
        .hash_password_with_salt(password.as_bytes(), &salt)
        .map_err(|e| anyhow::anyhow!("hashing the password: {e}"))?;
    PasswordHash::parse(&hash.to_string()).map_err(anyhow::Error::new)
}

static DUMMY: OnceLock<Option<PasswordHash>> = OnceLock::new();

fn dummy_hash() -> Option<&'static PasswordHash> {
    DUMMY.get_or_init(|| hash_password("dummy password of no account").ok()).as_ref()
}

/// Compute the dummy hash of [`verify_password`] now, so the first login of an unknown user is not slower. Blocking.
pub(crate) fn warm_up() {
    dummy_hash();
}

/// Whether `password` matches `hash`. Without a hash a fixed dummy hash is verified instead, so an unknown user or
/// one without a password takes as long as a wrong password. Blocking.
pub(crate) fn verify_password(password: &str, hash: Option<&PasswordHash>) -> bool {
    let Some(hash) = hash else {
        if let Some(dummy) = dummy_hash() {
            let _ = Argon2::default().verify_password(password.as_bytes(), dummy.as_str());
        }
        return false;
    };
    Argon2::default().verify_password(password.as_bytes(), hash.as_str()).is_ok()
}

/// A new session token (for the cookie) and its id (SHA-256, for the database).
pub(crate) struct NewSession {
    pub(crate) token: SecretString,
    pub(crate) id: Sha256Digest,
}

pub(crate) fn generate_session() -> Result<NewSession, getrandom::Error> {
    let mut secret = [0u8; SECRET_BYTES];
    getrandom::fill(&mut secret)?;
    let token =
        SecretString::new(format!("{TOKEN_PREFIX}{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(secret)));
    secret.fill(0);
    let id = session_digest(token.expose());
    Ok(NewSession { token, id })
}

fn session_digest(token: &str) -> Sha256Digest {
    Sha256Digest(Sha256::digest(token.as_bytes()).into())
}

/// The session id of the well-formed `basal_session` cookie, if the request has exactly one.
pub(crate) fn cookie_session(headers: &HeaderMap) -> Option<Sha256Digest> {
    let mut tokens = headers
        .get_all(COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|pair| pair.trim().strip_prefix(COOKIE_NAME)?.strip_prefix('='));
    let (Some(token), None) = (tokens.next(), tokens.next()) else {
        return None;
    };
    let secret = token.strip_prefix(TOKEN_PREFIX)?;
    let well_formed =
        secret.len() == SECRET_CHARS && secret.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    well_formed.then(|| session_digest(token))
}

/// Whether the request carries `X-Basal-UI: 1`.
pub(crate) fn has_ui_header(headers: &HeaderMap) -> bool {
    headers.get(UI_HEADER).is_some_and(|v| v.as_bytes() == b"1")
}

/// `Set-Cookie` of a new session.
pub(crate) fn set_cookie(token: &SecretString, ttl: Duration, secure: bool) -> HeaderValue {
    cookie(token.expose(), ttl.as_secs(), secure)
}

/// `Set-Cookie` that removes the session cookie.
pub(crate) fn clear_cookie(secure: bool) -> HeaderValue {
    cookie("", 0, secure)
}

fn cookie(value: &str, max_age: u64, secure: bool) -> HeaderValue {
    let secure = if secure { "; Secure" } else { "" };
    let text = format!("{COOKIE_NAME}={value}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Strict{secure}");
    HeaderValue::from_str(&text).expect("token and attributes are visible ASCII")
}

/// Failed logins per user id in this process; refuses a user id after [`MAX_FAILURES`] within the window.
#[derive(Default)]
pub(crate) struct LoginThrottle {
    /// Failures and the start of their window, per valid user id (at most 128 bytes). Never held across an await.
    failures: Mutex<HashMap<OwnerId, (u32, Instant)>>,
}

impl LoginThrottle {
    /// `Err(retry after)` while `user` is refused.
    pub(crate) fn check(&self, user: &OwnerId) -> Result<(), Duration> {
        let failures = self.lock();
        match failures.get(user) {
            Some(&(count, start)) if count >= MAX_FAILURES && start.elapsed() < THROTTLE_WINDOW => {
                Err(THROTTLE_WINDOW.saturating_sub(start.elapsed()))
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn failed(&self, user: &OwnerId) {
        let mut failures = self.lock();
        if failures.len() >= MAX_TRACKED && !failures.contains_key(user) {
            failures.retain(|_, (_, start)| start.elapsed() < THROTTLE_WINDOW);
            // Still full: forget the oldest window, never all of them (that would lift every lockout).
            if failures.len() >= MAX_TRACKED {
                if let Some(oldest) = failures.iter().min_by_key(|(_, (_, start))| *start).map(|(k, _)| k.clone()) {
                    failures.remove(&oldest);
                }
            }
        }
        let entry = failures.entry(user.clone()).or_insert((0, Instant::now()));
        if entry.1.elapsed() >= THROTTLE_WINDOW {
            *entry = (0, Instant::now());
        }
        entry.0 += 1;
    }

    pub(crate) fn succeeded(&self, user: &OwnerId) {
        self.lock().remove(user);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<OwnerId, (u32, Instant)>> {
        self.failures.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_round_trip() {
        let hash = hash_password("correct horse battery").unwrap();
        assert!(hash.as_str().starts_with("$argon2id$"));
        assert!(verify_password("correct horse battery", Some(&hash)));
        assert!(!verify_password("correct horse battery!", Some(&hash)));
        assert!(!verify_password("correct horse battery", None));
    }

    #[test]
    fn password_policy() {
        assert!(check_password("short").is_err());
        assert!(check_password("twelve chars").is_ok());
        assert!(check_password("ąęółżźćńśąęó").is_ok(), "counts characters, not bytes");
        assert!(check_password(&"x".repeat(1025)).is_err());
    }

    #[test]
    fn session_cookie_round_trip() {
        let session = generate_session().unwrap();
        let set = set_cookie(&session.token, Duration::from_secs(60), true);
        let set = set.to_str().unwrap();
        assert!(set.contains("HttpOnly") && set.contains("SameSite=Strict") && set.contains("Secure"));
        let pair = set.split(';').next().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(COOKIE, HeaderValue::from_str(&format!("theme=dark; {pair}")).unwrap());
        assert_eq!(cookie_session(&headers), Some(session.id));
    }

    #[test]
    fn session_cookie_refuses_malformed_or_repeated() {
        let session = generate_session().unwrap();
        let token = session.token.expose();
        let mut headers = HeaderMap::new();
        headers
            .insert(COOKIE, HeaderValue::from_str(&format!("basal_session={token}; basal_session={token}")).unwrap());
        assert_eq!(cookie_session(&headers), None, "two cookies");
        headers.insert(COOKIE, HeaderValue::from_static("basal_session=bss_short"));
        assert_eq!(cookie_session(&headers), None, "short secret");
        headers.insert(COOKIE, HeaderValue::from_str(&format!("basal_session={}", &token[4..])).unwrap());
        assert_eq!(cookie_session(&headers), None, "no prefix");
        assert!(!clear_cookie(false).to_str().unwrap().contains("Secure"));
    }

    #[test]
    fn ui_header_must_be_one() {
        let mut headers = HeaderMap::new();
        assert!(!has_ui_header(&headers));
        headers.insert(UI_HEADER, HeaderValue::from_static("true"));
        assert!(!has_ui_header(&headers));
        headers.insert(UI_HEADER, HeaderValue::from_static("1"));
        assert!(has_ui_header(&headers));
    }

    #[test]
    fn throttle_refuses_after_max_failures() {
        let throttle = LoginThrottle::default();
        let (acme, beta) = (OwnerId::parse("acme").unwrap(), OwnerId::parse("beta").unwrap());
        for _ in 0..MAX_FAILURES {
            assert!(throttle.check(&acme).is_ok());
            throttle.failed(&acme);
        }
        assert!(throttle.check(&acme).is_err());
        assert!(throttle.check(&beta).is_ok(), "per user id");
        throttle.succeeded(&acme);
        assert!(throttle.check(&acme).is_ok());
    }

    #[test]
    fn session_scopes_by_role() {
        assert_eq!(scopes(ApiRole::User).to_text(), "inference,batches");
        assert_eq!(scopes(ApiRole::Admin).to_text(), "inference,batches,metrics");
    }
}
