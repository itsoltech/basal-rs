//! API users, their keys and UI sessions: insertion, listing, enabling/disabling, passwords and deletion of users,
//! key and session lookup for authentication, revocation, expiry and throttled last use.
//!
//! These rows are independent of every batch table, so the operations take no batch lock (deletion only reads the
//! user's quota row). A key or session created while its user is being disabled is harmless: authentication checks
//! the user on every lookup. Times come from the database clock.

use std::time::Duration;

use super::super::error::StoreError;
use super::super::sql::{args, db_now, SqlRow, SqlTx};
use super::super::types::{
    ApiKeyId, ApiKeyLookup, ApiKeyRecord, ApiRole, ApiScopes, ApiSessionLookup, ApiUserRecord, NewApiKey,
    NewApiSession, OwnerId, PasswordHash, Sha256Digest, UnixMillis,
};
use super::common::{corrupt, digest_from_db, lock_quota, millis, owner_from_db};

const COLUMNS: &str = "id, user_id, name, token_sha256, scopes, created_at, expires_at, revoked_at, last_used_at";

fn record<R: SqlRow>(row: &R) -> Result<ApiKeyRecord, StoreError> {
    Ok(ApiKeyRecord {
        id: ApiKeyId::parse(&row.text_at(0)?).map_err(|_| corrupt("api key id"))?,
        user: owner_from_db(row.text_at(1)?)?,
        name: row.text_at(2)?,
        token_sha256: digest_from_db(row.blob_at(3)?)?,
        scopes: ApiScopes::parse(&row.text_at(4)?).map_err(|_| corrupt("api key scopes"))?,
        created_at: UnixMillis(row.i64_at(5)?),
        expires_at: row.opt_i64_at(6)?.map(UnixMillis),
        revoked_at: row.opt_i64_at(7)?.map(UnixMillis),
        last_used_at: row.opt_i64_at(8)?.map(UnixMillis),
    })
}

async fn get<T: SqlTx>(tx: &mut T, id: &ApiKeyId) -> Result<Option<ApiKeyRecord>, StoreError> {
    let sql = format!("SELECT {COLUMNS} FROM api_keys WHERE id = $1");
    tx.fetch_optional(&sql, args![id.as_str()]).await?.as_ref().map(record).transpose()
}

const USER_COLUMNS: &str = "id, name, created_at, disabled_at, role, password_hash";

fn user_record<R: SqlRow>(row: &R) -> Result<ApiUserRecord, StoreError> {
    Ok(ApiUserRecord {
        id: owner_from_db(row.text_at(0)?)?,
        name: row.text_at(1)?,
        created_at: UnixMillis(row.i64_at(2)?),
        disabled_at: row.opt_i64_at(3)?.map(UnixMillis),
        role: ApiRole::parse(&row.text_at(4)?).map_err(|_| corrupt("api user role"))?,
        password: row
            .opt_text_at(5)?
            .map(|hash| PasswordHash::parse(&hash).map_err(|_| corrupt("api user password hash")))
            .transpose()?,
    })
}

/// The user `id`, if any.
pub(in super::super) async fn get_user<T: SqlTx>(
    tx: &mut T,
    id: &OwnerId,
) -> Result<Option<ApiUserRecord>, StoreError> {
    let sql = format!("SELECT {USER_COLUMNS} FROM api_users WHERE id = $1");
    tx.fetch_optional(&sql, args![id.as_str()]).await?.as_ref().map(user_record).transpose()
}

/// Insert a user; `None` when the id is taken.
pub(in super::super) async fn create_user<T: SqlTx>(
    tx: &mut T,
    id: &OwnerId,
    name: &str,
    role: ApiRole,
    password: Option<&PasswordHash>,
) -> Result<Option<ApiUserRecord>, StoreError> {
    let now = db_now(tx).await?;
    let sql = tx.dialect().insert_if_absent(
        "api_users",
        &["id", "name", "created_at", "role", "password_hash"],
        &["$1", "$2", "$3", "$4", "$5"],
        1,
    );
    let hash = password.map(PasswordHash::as_str);
    if tx.execute(&sql, args![id.as_str(), name, now, role.as_str(), hash]).await? == 0 {
        return Ok(None);
    }
    get_user(tx, id).await?.ok_or(corrupt("inserted api user vanished")).map(Some)
}

/// Users ordered by id, disabled ones only with `include_disabled`; at most `limit`.
pub(in super::super) async fn list_users<T: SqlTx>(
    tx: &mut T,
    include_disabled: bool,
    limit: u32,
) -> Result<Vec<ApiUserRecord>, StoreError> {
    let d = tx.dialect();
    let filter = if include_disabled { "1 = 1" } else { "disabled_at IS NULL" };
    let sql = format!(
        "SELECT {}{USER_COLUMNS} FROM api_users WHERE {filter} ORDER BY id{}",
        d.top(&limit.to_string()),
        d.limit(&limit.to_string())
    );
    tx.fetch_all(&sql, args![]).await?.iter().map(user_record).collect()
}

/// Disable or enable a user; disabling again keeps the first time and ends the user's sessions. `NotFound` for an
/// unknown id.
pub(in super::super) async fn set_user_disabled<T: SqlTx>(
    tx: &mut T,
    id: &OwnerId,
    disabled: bool,
) -> Result<ApiUserRecord, StoreError> {
    if disabled {
        let now = db_now(tx).await?;
        tx.execute(
            "UPDATE api_users SET disabled_at = $1 WHERE id = $2 AND disabled_at IS NULL",
            args![now, id.as_str()],
        )
        .await?;
        tx.execute("DELETE FROM api_sessions WHERE user_id = $1", args![id.as_str()]).await?;
    } else {
        tx.execute("UPDATE api_users SET disabled_at = NULL WHERE id = $1", args![id.as_str()]).await?;
    }
    get_user(tx, id).await?.ok_or(StoreError::NotFound)
}

/// Set or clear a user's password and end its sessions except `keep` (the session that changed its own password).
/// `NotFound` for an unknown id.
pub(in super::super) async fn set_user_password<T: SqlTx>(
    tx: &mut T,
    id: &OwnerId,
    password: Option<&PasswordHash>,
    keep: Option<&Sha256Digest>,
) -> Result<ApiUserRecord, StoreError> {
    let hash = password.map(PasswordHash::as_str);
    if tx.execute("UPDATE api_users SET password_hash = $1 WHERE id = $2", args![hash, id.as_str()]).await? == 0 {
        return Err(StoreError::NotFound);
    }
    let keep = keep.map_or_else(String::new, |k| k.to_hex());
    tx.execute("DELETE FROM api_sessions WHERE user_id = $1 AND id <> $2", args![id.as_str(), keep]).await?;
    get_user(tx, id).await?.ok_or(StoreError::NotFound)
}

/// Delete a user with its keys and sessions. Only a user disabled at least `disabled_for` ago (no server still
/// accepts its cached keys) without Batch data (quota row all zero) is deleted: `UserNotDisabled`, `UserHasData`,
/// `NotFound`. Batch metadata already counted out of the quota stays until the Batch retention purges it.
pub(in super::super) async fn delete_user<T: SqlTx>(
    tx: &mut T,
    id: &OwnerId,
    disabled_for: Duration,
) -> Result<(), StoreError> {
    let user = get_user(tx, id).await?.ok_or(StoreError::NotFound)?;
    let now = db_now(tx).await?;
    if user.disabled_at.is_none_or(|at| now.saturating_sub(at.0) < millis(disabled_for)) {
        return Err(StoreError::UserNotDisabled);
    }
    // The quota lock is the first lock of every Batch operation that adds data, so none can commit after this check.
    if !lock_quota(tx, id).await?.is_empty() {
        return Err(StoreError::UserHasData);
    }
    tx.execute("DELETE FROM api_sessions WHERE user_id = $1", args![id.as_str()]).await?;
    tx.execute("DELETE FROM api_keys WHERE user_id = $1", args![id.as_str()]).await?;
    tx.execute("DELETE FROM api_users WHERE id = $1", args![id.as_str()]).await?;
    Ok(())
}

/// Insert a session of an existing, enabled user whose password is still the verified one (`NotFound`; disabled or
/// changed meanwhile: `UserDisabled`) and delete the user's expired sessions.
pub(in super::super) async fn create_session<T: SqlTx>(
    tx: &mut T,
    session: &NewApiSession,
) -> Result<ApiSessionLookup, StoreError> {
    let user = get_user(tx, &session.user).await?.ok_or(StoreError::NotFound)?;
    if user.disabled_at.is_some() || user.password != session.password {
        return Err(StoreError::UserDisabled);
    }
    let now = db_now(tx).await?;
    tx.execute("DELETE FROM api_sessions WHERE user_id = $1 AND expires_at <= $2", args![user.id.as_str(), now])
        .await?;
    let expires_at = now.saturating_add(millis(session.ttl));
    tx.execute(
        "INSERT INTO api_sessions (id, user_id, created_at, expires_at) VALUES ($1, $2, $3, $4)",
        args![session.id.to_hex(), user.id.as_str(), now, expires_at],
    )
    .await?;
    Ok(ApiSessionLookup { user, expires_at: UnixMillis(expires_at), now: UnixMillis(now) })
}

/// The session `id` (also expired) with its user and the database time of the read.
pub(in super::super) async fn lookup_session<T: SqlTx>(
    tx: &mut T,
    id: &Sha256Digest,
) -> Result<Option<ApiSessionLookup>, StoreError> {
    let row =
        tx.fetch_optional("SELECT user_id, expires_at FROM api_sessions WHERE id = $1", args![id.to_hex()]).await?;
    let Some(row) = row else {
        return Ok(None);
    };
    let user_id = owner_from_db(row.text_at(0)?)?;
    let user = get_user(tx, &user_id).await?.ok_or(corrupt("api session without its user"))?;
    Ok(Some(ApiSessionLookup { user, expires_at: UnixMillis(row.i64_at(1)?), now: UnixMillis(db_now(tx).await?) }))
}

/// Delete a session (logout); deleting a missing one is not an error.
pub(in super::super) async fn delete_session<T: SqlTx>(tx: &mut T, id: &Sha256Digest) -> Result<(), StoreError> {
    tx.execute("DELETE FROM api_sessions WHERE id = $1", args![id.to_hex()]).await?;
    Ok(())
}

/// Insert a key of an existing user: `NotFound` for an unknown user, `UserDisabled` for a disabled one.
pub(in super::super) async fn create<T: SqlTx>(tx: &mut T, key: &NewApiKey) -> Result<ApiKeyRecord, StoreError> {
    if get_user(tx, &key.user).await?.ok_or(StoreError::NotFound)?.disabled_at.is_some() {
        return Err(StoreError::UserDisabled);
    }
    let now = db_now(tx).await?;
    let expires_at = key.expires_in.map(|d| now.saturating_add(millis(d)));
    tx.execute(
        "INSERT INTO api_keys (id, user_id, name, token_sha256, scopes, created_at, expires_at) \
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
        args![
            key.id.as_str(),
            key.user.as_str(),
            key.name.as_str(),
            key.token_sha256.0.as_slice(),
            key.scopes.to_text(),
            now,
            expires_at
        ],
    )
    .await?;
    get(tx, &key.id).await?.ok_or(corrupt("inserted api key vanished"))
}

/// Keys ordered by creation, optionally of one user and without revoked ones; at most `limit`.
pub(in super::super) async fn list<T: SqlTx>(
    tx: &mut T,
    user: Option<&OwnerId>,
    include_revoked: bool,
    limit: u32,
) -> Result<Vec<ApiKeyRecord>, StoreError> {
    let d = tx.dialect();
    let revoked = if include_revoked { "" } else { " AND revoked_at IS NULL" };
    let (filter, args) = match user {
        Some(user) => ("user_id = $1", args![user.as_str()]),
        None => ("1 = 1", args![]),
    };
    let sql = format!(
        "SELECT {}{COLUMNS} FROM api_keys WHERE {filter}{revoked} ORDER BY created_at, id{}",
        d.top(&limit.to_string()),
        d.limit(&limit.to_string())
    );
    tx.fetch_all(&sql, args).await?.iter().map(record).collect()
}

pub(in super::super) async fn lookup<T: SqlTx>(tx: &mut T, id: &ApiKeyId) -> Result<Option<ApiKeyLookup>, StoreError> {
    let Some(record) = get(tx, id).await? else {
        return Ok(None);
    };
    let user = get_user(tx, &record.user).await?.ok_or(corrupt("api key without its user"))?;
    Ok(Some(ApiKeyLookup { record, user_disabled: user.disabled_at.is_some(), now: UnixMillis(db_now(tx).await?) }))
}

/// Revoke a key; an already revoked key keeps its first revocation time.
pub(in super::super) async fn revoke<T: SqlTx>(tx: &mut T, id: &ApiKeyId) -> Result<ApiKeyRecord, StoreError> {
    let now = db_now(tx).await?;
    tx.execute("UPDATE api_keys SET revoked_at = $1 WHERE id = $2 AND revoked_at IS NULL", args![now, id.as_str()])
        .await?;
    get(tx, id).await?.ok_or(StoreError::NotFound)
}

/// Let a key expire `within` from now unless it already expires earlier.
pub(in super::super) async fn expire_within<T: SqlTx>(
    tx: &mut T,
    id: &ApiKeyId,
    within: Duration,
) -> Result<ApiKeyRecord, StoreError> {
    let at = db_now(tx).await?.saturating_add(millis(within));
    tx.execute(
        "UPDATE api_keys SET expires_at = $1 WHERE id = $2 AND (expires_at IS NULL OR expires_at > $1)",
        args![at, id.as_str()],
    )
    .await?;
    get(tx, id).await?.ok_or(StoreError::NotFound)
}

/// Record a use now unless the last recorded use is younger than `interval`.
pub(in super::super) async fn touch<T: SqlTx>(tx: &mut T, id: &ApiKeyId, interval: Duration) -> Result<(), StoreError> {
    let now = db_now(tx).await?;
    tx.execute(
        "UPDATE api_keys SET last_used_at = $1 WHERE id = $2 AND (last_used_at IS NULL OR last_used_at <= $3)",
        args![now, id.as_str(), now.saturating_sub(millis(interval))],
    )
    .await?;
    Ok(())
}
