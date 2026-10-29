//! API users and keys: insertion, listing, enabling/disabling users, key lookup for authentication, revocation,
//! expiry and throttled last use.
//!
//! These rows are independent of every batch table, so the operations take no batch lock. A key created while its
//! user is being disabled is harmless: authentication checks the user on every lookup. Times come from the database
//! clock.

use std::time::Duration;

use super::super::error::StoreError;
use super::super::sql::{args, db_now, SqlRow, SqlTx};
use super::super::types::{
    ApiKeyId, ApiKeyLookup, ApiKeyRecord, ApiScopes, ApiUserRecord, NewApiKey, OwnerId, UnixMillis,
};
use super::common::{corrupt, digest_from_db, millis, owner_from_db};

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

const USER_COLUMNS: &str = "id, name, created_at, disabled_at";

fn user_record<R: SqlRow>(row: &R) -> Result<ApiUserRecord, StoreError> {
    Ok(ApiUserRecord {
        id: owner_from_db(row.text_at(0)?)?,
        name: row.text_at(1)?,
        created_at: UnixMillis(row.i64_at(2)?),
        disabled_at: row.opt_i64_at(3)?.map(UnixMillis),
    })
}

async fn get_user<T: SqlTx>(tx: &mut T, id: &OwnerId) -> Result<Option<ApiUserRecord>, StoreError> {
    let sql = format!("SELECT {USER_COLUMNS} FROM api_users WHERE id = $1");
    tx.fetch_optional(&sql, args![id.as_str()]).await?.as_ref().map(user_record).transpose()
}

/// Insert a user; `None` when the id is taken.
pub(in super::super) async fn create_user<T: SqlTx>(
    tx: &mut T,
    id: &OwnerId,
    name: &str,
) -> Result<Option<ApiUserRecord>, StoreError> {
    let now = db_now(tx).await?;
    let sql = tx.dialect().insert_if_absent("api_users", &["id", "name", "created_at"], &["$1", "$2", "$3"], 1);
    if tx.execute(&sql, args![id.as_str(), name, now]).await? == 0 {
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

/// Disable or enable a user; disabling again keeps the first time. `NotFound` for an unknown id.
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
    } else {
        tx.execute("UPDATE api_users SET disabled_at = NULL WHERE id = $1", args![id.as_str()]).await?;
    }
    get_user(tx, id).await?.ok_or(StoreError::NotFound)
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
