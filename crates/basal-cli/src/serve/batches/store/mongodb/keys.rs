//! API keys (MongoDB): the same operations as `ops::keys`, one small document per key with `_id` = key id.

use std::time::Duration;

use mongodb::bson::{doc, Document};

use super::super::error::StoreError;
use super::super::ops::common::millis;
use super::super::types::{ApiKeyId, ApiKeyLookup, ApiKeyRecord, ApiScopes, NewApiKey, OwnerId, UnixMillis};
use super::convert::{digest, owner};
use super::driver::{bin, bytes, corrupt, int, mtx, opt_i64, opt_int, text, Core, Tx, API_KEYS};

fn record(doc: &Document) -> Result<ApiKeyRecord, StoreError> {
    Ok(ApiKeyRecord {
        id: ApiKeyId::parse(&text(doc, "_id")?).map_err(|_| corrupt("api key id"))?,
        owner: owner(text(doc, "owner_id")?)?,
        name: text(doc, "name")?,
        token_sha256: digest(bin(doc, "token_sha256")?)?,
        scopes: ApiScopes::parse(&text(doc, "scopes")?).map_err(|_| corrupt("api key scopes"))?,
        created_at: UnixMillis(int(doc, "created_at")?),
        expires_at: opt_int(doc, "expires_at")?.map(UnixMillis),
        revoked_at: opt_int(doc, "revoked_at")?.map(UnixMillis),
        last_used_at: opt_int(doc, "last_used_at")?.map(UnixMillis),
    })
}

async fn get_tx(tx: &mut Tx<'_>, id: &ApiKeyId) -> Result<ApiKeyRecord, StoreError> {
    tx.one(API_KEYS, doc! { "_id": id.as_str() }).await?.as_ref().map(record).transpose()?.ok_or(StoreError::NotFound)
}

pub(super) async fn create(core: &Core, key: &NewApiKey) -> Result<ApiKeyRecord, StoreError> {
    let now = core.server_now().await?;
    let row = doc! {
        "_id": key.id.as_str(),
        "owner_id": key.owner.as_str(),
        "name": key.name.as_str(),
        "token_sha256": bytes(&key.token_sha256.0),
        "scopes": key.scopes.to_text(),
        "created_at": now,
        "expires_at": opt_i64(key.expires_in.map(|d| now.saturating_add(millis(d)))),
        "revoked_at": opt_i64(None),
        "last_used_at": opt_i64(None),
    };
    mtx!(core, tx => async {
        tx.insert(API_KEYS, row.clone()).await?;
        get_tx(tx, &key.id).await
    })
}

pub(super) async fn list(
    core: &Core,
    owner_id: Option<&OwnerId>,
    include_revoked: bool,
    limit: u32,
) -> Result<Vec<ApiKeyRecord>, StoreError> {
    let mut filter = Document::new();
    if let Some(owner_id) = owner_id {
        filter.insert("owner_id", owner_id.as_str());
    }
    if !include_revoked {
        filter.insert("revoked_at", mongodb::bson::Bson::Null);
    }
    let rows = core.read_many(API_KEYS, filter, doc! { "created_at": 1_i32, "_id": 1_i32 }, i64::from(limit)).await?;
    rows.iter().map(record).collect()
}

pub(super) async fn lookup(core: &Core, id: &ApiKeyId) -> Result<Option<ApiKeyLookup>, StoreError> {
    let Some(row) = core.read_one(API_KEYS, doc! { "_id": id.as_str() }).await? else {
        return Ok(None);
    };
    Ok(Some(ApiKeyLookup { record: record(&row)?, now: UnixMillis(core.server_now().await?) }))
}

pub(super) async fn revoke(core: &Core, id: &ApiKeyId) -> Result<ApiKeyRecord, StoreError> {
    let now = core.server_now().await?;
    mtx!(core, tx => async {
        tx.update(API_KEYS, doc! { "_id": id.as_str(), "revoked_at": null }, doc! { "$set": { "revoked_at": now } })
            .await?;
        get_tx(tx, id).await
    })
}

pub(super) async fn expire_within(core: &Core, id: &ApiKeyId, within: Duration) -> Result<ApiKeyRecord, StoreError> {
    let at = core.server_now().await?.saturating_add(millis(within));
    mtx!(core, tx => async {
        let filter = doc! { "_id": id.as_str(), "$or": [{ "expires_at": null }, { "expires_at": { "$gt": at } }] };
        tx.update(API_KEYS, filter, doc! { "$set": { "expires_at": at } }).await?;
        get_tx(tx, id).await
    })
}

pub(super) async fn touch(core: &Core, id: &ApiKeyId, interval: Duration) -> Result<(), StoreError> {
    let now = core.server_now().await?;
    let before = now.saturating_sub(millis(interval));
    mtx!(core, tx => async {
        let filter =
            doc! { "_id": id.as_str(), "$or": [{ "last_used_at": null }, { "last_used_at": { "$lte": before } }] };
        tx.update(API_KEYS, filter, doc! { "$set": { "last_used_at": now } }).await.map(drop)
    })
}
