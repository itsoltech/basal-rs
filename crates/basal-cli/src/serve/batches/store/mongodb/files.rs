//! Upload sessions, file reads, logical delete, expiry and physical GC decisions (MongoDB).

use mongodb::bson::{doc, Bson, Document};

use super::super::error::{InvalidValue, StoreError};
use super::super::ops::common::{admit, millis, unsigned, QuotaDelta};
use super::super::types::{
    FileGcCandidate, FileId, FilePurpose, FileRecord, FileState, NewUpload, OwnerId, Page, PageRequest,
    SessionGcCandidate, SessionId, SessionKind, StoredObject, UnixMillis, UploadSession,
};
use super::common::{active_policy, apply_quota, lock_quota, owner_limits};
use super::convert::{artifact, file_id, file_row, owner};
use super::driver::{bytes, corrupt, int, mtx, opt_int, opt_text, text, Core, Tx, FILES, SESSIONS};

const MAX_FILENAME_BYTES: usize = 255;

fn check_filename(filename: &str) -> Result<(), StoreError> {
    if filename.is_empty() || filename.len() > MAX_FILENAME_BYTES || filename.chars().any(char::is_control) {
        return Err(StoreError::Invalid(InvalidValue::new("filename", "must have 1..255 bytes")));
    }
    Ok(())
}

async fn reserve_upload_tx(tx: &mut Tx<'_>, upload: &NewUpload) -> Result<UploadSession, StoreError> {
    check_filename(&upload.filename)?;
    let (usage, now) = lock_quota(tx, &upload.owner).await?;
    let policy = active_policy(tx).await?;
    let limits = owner_limits(tx, policy.revision, &upload.owner).await?;
    let reserve = policy.max_upload.saturating_add(policy.object_charge);
    let delta = QuotaDelta { reserved: reserve, files: 1, sessions: 1, ..QuotaDelta::default() };
    admit(&usage, &limits, &delta)?;
    let lease_until = now.saturating_add(millis(tx.settings().upload_lease));
    let session = SessionId::new_random();
    let file = FileId::new_random();
    let expires_after = upload.expires_after.map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX));
    tx.insert(
        SESSIONS,
        doc! {
            "_id": session.as_str(),
            "owner_id": upload.owner.as_str(),
            "kind": "upload",
            "artifact_ref": upload.artifact.as_str(),
            "secondary_ref": Bson::Null,
            "generation": 0_i64,
            "phase": "open",
            "lease_until": lease_until,
            "reserved_bytes": reserve,
            "max_bytes": policy.max_upload,
            "file_id": file.as_str(),
            "batch_id": Bson::Null,
            "claim_id": Bson::Null,
            "line_no": Bson::Null,
            "filename": upload.filename.as_str(),
            "expires_after_secs": super::driver::opt_i64(expires_after),
            "created_at": now,
        },
    )
    .await?;
    apply_quota(tx, &upload.owner, delta).await?;
    Ok(UploadSession { id: session, max_bytes: unsigned(policy.max_upload)? })
}

pub(super) async fn reserve_upload(core: &Core, upload: &NewUpload) -> Result<UploadSession, StoreError> {
    mtx!(core, tx => reserve_upload_tx(tx, upload))
}

fn upload_filter(owner: &OwnerId, session: &SessionId) -> Document {
    doc! { "_id": session.as_str(), "owner_id": owner.as_str(), "kind": "upload", "phase": "open" }
}

async fn complete_upload_tx(
    tx: &mut Tx<'_>,
    owner_id: &OwnerId,
    session: &SessionId,
    object: StoredObject,
) -> Result<FileRecord, StoreError> {
    lock_quota(tx, owner_id).await?;
    let (doc, now) = tx.lock(SESSIONS, upload_filter(owner_id, session)).await?.ok_or(StoreError::SessionClosed)?;
    if int(&doc, "lease_until")? < now {
        // GC removes the object of an expired session; its reservation stays until then.
        return Err(StoreError::SessionClosed);
    }
    let file = file_id(text(&doc, "file_id")?)?;
    let object_ref = artifact(text(&doc, "artifact_ref")?)?;
    let reserved = int(&doc, "reserved_bytes")?;
    let max_bytes = int(&doc, "max_bytes")?;
    let filename = opt_text(&doc, "filename")?.ok_or(corrupt("upload filename"))?;
    let expires_after = opt_int(&doc, "expires_after_secs")?;
    let size = super::super::ops::common::signed(object.bytes)?;
    if size > max_bytes {
        return Err(StoreError::InputUnusable("upload exceeds the size limit"));
    }
    let policy = active_policy(tx).await?;
    let charged = size.saturating_add(policy.object_charge);
    if charged > reserved {
        return Err(corrupt("upload charge exceeds its reservation"));
    }
    let retention = expires_after.map_or(millis(tx.settings().file_retention), |s| s.saturating_mul(1000));
    let expires_at = now.saturating_add(retention);
    tx.insert(
        FILES,
        doc! {
            "_id": file.as_str(),
            "owner_id": owner_id.as_str(),
            "purpose": "batch",
            "filename": filename.as_str(),
            "artifact_ref": object_ref.as_str(),
            "bytes": size,
            "charged_bytes": charged,
            "sha256": bytes(&object.sha256.0),
            "created_at": now,
            "expires_at": expires_at,
            "state": "active",
            "pin_count": 0_i64,
            "deleted_at": Bson::Null,
            "gc_after": Bson::Null,
            "batch_id": Bson::Null,
        },
    )
    .await?;
    tx.delete_one(SESSIONS, doc! { "_id": session.as_str() }).await?;
    // The file slot reserved with the session stays taken by the file.
    apply_quota(tx, owner_id, QuotaDelta { used: charged, reserved: -reserved, sessions: -1, ..QuotaDelta::default() })
        .await?;
    Ok(FileRecord {
        owner: owner_id.clone(),
        id: file,
        purpose: FilePurpose::Batch,
        filename,
        bytes: object.bytes,
        sha256: object.sha256,
        created_at: UnixMillis(now),
        expires_at: Some(UnixMillis(expires_at)),
        state: FileState::Active,
        artifact: object_ref,
    })
}

pub(super) async fn complete_upload(
    core: &Core,
    owner: &OwnerId,
    session: &SessionId,
    object: StoredObject,
) -> Result<FileRecord, StoreError> {
    mtx!(core, tx => complete_upload_tx(tx, owner, session, object))
}

async fn abort_upload_tx(tx: &mut Tx<'_>, owner: &OwnerId, session: &SessionId) -> Result<(), StoreError> {
    if let Some((_, now)) = tx.lock(SESSIONS, upload_filter(owner, session)).await? {
        tx.update(
            SESSIONS,
            doc! { "_id": session.as_str() },
            doc! { "$set": { "phase": "abandoned", "lease_until": now } },
        )
        .await?;
    }
    Ok(())
}

pub(super) async fn abort_upload(core: &Core, owner: &OwnerId, session: &SessionId) -> Result<(), StoreError> {
    mtx!(core, tx => abort_upload_tx(tx, owner, session))
}

/// Active, unexpired file of an owner; everything else is `NotFound`.
pub(super) async fn get_file(core: &Core, owner: &OwnerId, id: &FileId) -> Result<FileRecord, StoreError> {
    let now = core.server_now().await?;
    let doc = core
        .read_one(FILES, doc! { "_id": id.as_str(), "owner_id": owner.as_str(), "state": "active" })
        .await?
        .ok_or(StoreError::NotFound)?;
    let file = file_row(&doc)?.record;
    if file.expires_at.is_some_and(|at| at.0 <= now) {
        return Err(StoreError::NotFound);
    }
    Ok(file)
}

pub(super) async fn list_files(
    core: &Core,
    owner: &OwnerId,
    purpose: Option<FilePurpose>,
    page: &PageRequest<FileId>,
) -> Result<Page<FileRecord>, StoreError> {
    let limit = i64::from(page.limit.clamp(1, 100));
    let now = core.server_now().await?;
    let mut filter = doc! {
        "owner_id": owner.as_str(),
        "state": "active",
        "$and": [{ "$or": [{ "expires_at": Bson::Null }, { "expires_at": { "$gt": now } }] }],
    };
    if let Some(purpose) = purpose {
        filter.insert("purpose", purpose.as_str());
    }
    if let Some(after) = &page.after {
        let cursor = core
            .read_one(FILES, doc! { "_id": after.as_str(), "owner_id": owner.as_str() })
            .await?
            .ok_or(StoreError::NotFound)?;
        let at = int(&cursor, "created_at")?;
        filter.get_array_mut("$and").map_err(|_| corrupt("filter"))?.push(Bson::Document(doc! {
            "$or": [{ "created_at": { "$lt": at } }, { "created_at": at, "_id": { "$lt": after.as_str() } }]
        }));
    }
    let docs = core.read_many(FILES, filter, doc! { "created_at": -1_i32, "_id": -1_i32 }, limit + 1).await?;
    let has_more = docs.len() as i64 > limit;
    let items = docs.iter().take(limit as usize).map(|d| file_row(d).map(|f| f.record)).collect::<Result<_, _>>()?;
    Ok(Page { items, has_more })
}

async fn delete_file_tx(tx: &mut Tx<'_>, owner: &OwnerId, id: &FileId) -> Result<FileRecord, StoreError> {
    let (doc, now) = tx
        .lock(FILES, doc! { "_id": id.as_str(), "owner_id": owner.as_str(), "state": "active" })
        .await?
        .ok_or(StoreError::NotFound)?;
    let file = file_row(&doc)?;
    if file.record.expires_at.is_some_and(|at| at.0 <= now) {
        return Err(StoreError::NotFound);
    }
    if file.pin_count > 0 {
        return Err(StoreError::FileInUse);
    }
    let gc_after = now.saturating_add(millis(tx.settings().download_grace));
    tx.update(
        FILES,
        doc! { "_id": id.as_str() },
        doc! { "$set": { "state": "deleted", "deleted_at": now, "gc_after": gc_after } },
    )
    .await?;
    Ok(FileRecord { state: FileState::Deleted, ..file.record })
}

pub(super) async fn delete_file(core: &Core, owner: &OwnerId, id: &FileId) -> Result<FileRecord, StoreError> {
    mtx!(core, tx => delete_file_tx(tx, owner, id))
}

async fn expire_files_tx(tx: &mut Tx<'_>, now: i64, limit: u32) -> Result<u64, StoreError> {
    let due = tx
        .many(
            FILES,
            doc! { "state": "active", "expires_at": { "$ne": Bson::Null, "$lte": now } },
            doc! { "state": 1_i32, "expires_at": 1_i32 },
            i64::from(limit),
        )
        .await?;
    let ids: Vec<Bson> = due.iter().map(|d| d.get("_id").cloned().unwrap_or(Bson::Null)).collect();
    if ids.is_empty() {
        return Ok(0);
    }
    let gc_after = now.saturating_add(millis(tx.settings().download_grace));
    tx.update_many(
        FILES,
        doc! { "_id": { "$in": ids }, "state": "active" },
        doc! { "$set": { "state": "deleted", "deleted_at": now, "gc_after": gc_after } },
    )
    .await
}

/// Mark up to `limit` expired files deleted (also pinned inputs: the pin only defers physical GC).
pub(super) async fn expire_files(core: &Core, limit: u32) -> Result<u64, StoreError> {
    let now = core.server_now().await?;
    mtx!(core, tx => expire_files_tx(tx, now, limit))
}

pub(super) async fn file_gc_candidates(core: &Core, limit: u32) -> Result<Vec<FileGcCandidate>, StoreError> {
    let now = core.server_now().await?;
    let docs = core
        .read_many(
            FILES,
            doc! { "state": "deleted", "gc_after": { "$lte": now }, "pin_count": 0_i64 },
            doc! { "state": 1_i32, "gc_after": 1_i32 },
            i64::from(limit),
        )
        .await?;
    docs.iter()
        .map(|d| {
            Ok(FileGcCandidate {
                owner: owner(text(d, "owner_id")?)?,
                file_id: file_id(text(d, "_id")?)?,
                artifact: artifact(text(d, "artifact_ref")?)?,
            })
        })
        .collect()
}

async fn purge_file_tx(tx: &mut Tx<'_>, owner_id: &OwnerId, id: &FileId) -> Result<bool, StoreError> {
    lock_quota(tx, owner_id).await?;
    let Some((doc, now)) = tx
        .lock(FILES, doc! { "_id": id.as_str(), "owner_id": owner_id.as_str(), "state": "deleted", "pin_count": 0_i64 })
        .await?
    else {
        return Ok(false);
    };
    if opt_int(&doc, "gc_after")?.is_none_or(|at| at > now) {
        return Ok(false);
    }
    let charged = int(&doc, "charged_bytes")?;
    tx.delete_one(FILES, doc! { "_id": id.as_str() }).await?;
    apply_quota(tx, owner_id, QuotaDelta { used: -charged, files: -1, ..QuotaDelta::default() }).await?;
    Ok(true)
}

/// Remove a physically deleted file and release its bytes and slot. Idempotent.
pub(super) async fn purge_file(core: &Core, owner: &OwnerId, id: &FileId) -> Result<bool, StoreError> {
    mtx!(core, tx => purge_file_tx(tx, owner, id))
}

/// Upload and result sessions whose writer is finished or cut off (`lease_until + io_guard` passed).
pub(super) async fn session_gc_candidates(core: &Core, limit: u32) -> Result<Vec<SessionGcCandidate>, StoreError> {
    let now = core.server_now().await?;
    let horizon = now.saturating_sub(millis(core.settings.io_guard));
    let docs = core
        .read_many(
            SESSIONS,
            doc! { "kind": { "$in": ["upload", "result"] }, "lease_until": { "$lt": horizon } },
            doc! { "lease_until": 1_i32 },
            i64::from(limit),
        )
        .await?;
    docs.iter()
        .map(|d| {
            let mut artifacts = vec![artifact(text(d, "artifact_ref")?)?];
            if let Some(secondary) = opt_text(d, "secondary_ref")? {
                artifacts.push(artifact(secondary)?);
            }
            Ok(SessionGcCandidate {
                session: SessionId::parse(&text(d, "_id")?).map_err(|_| corrupt("session id"))?,
                owner: owner(text(d, "owner_id")?)?,
                artifacts,
            })
        })
        .collect()
}

async fn release_session_tx(tx: &mut Tx<'_>, owner_id: &OwnerId, session: &SessionId) -> Result<bool, StoreError> {
    lock_quota(tx, owner_id).await?;
    let Some((doc, now)) = tx
        .lock(
            SESSIONS,
            doc! { "_id": session.as_str(), "owner_id": owner_id.as_str(), "kind": { "$in": ["upload", "result"] } },
        )
        .await?
    else {
        return Ok(false);
    };
    let horizon = now.saturating_sub(millis(tx.settings().io_guard));
    if int(&doc, "lease_until")? >= horizon {
        return Ok(false);
    }
    let kind = SessionKind::from_db(&text(&doc, "kind")?).ok_or(corrupt("session kind"))?;
    let reserved = int(&doc, "reserved_bytes")?;
    tx.delete_one(SESSIONS, doc! { "_id": session.as_str() }).await?;
    let delta = match kind {
        SessionKind::Upload => QuotaDelta { reserved: -reserved, files: -1, sessions: -1, ..QuotaDelta::default() },
        _ => QuotaDelta { reserved: -reserved, ..QuotaDelta::default() },
    };
    apply_quota(tx, owner_id, delta).await?;
    Ok(true)
}

/// Release a GC'd session after its objects were deleted. Idempotent.
pub(super) async fn release_session(core: &Core, owner: &OwnerId, session: &SessionId) -> Result<bool, StoreError> {
    mtx!(core, tx => release_session_tx(tx, owner, session))
}
