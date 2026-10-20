//! Upload sessions, file reads, logical delete, expiry and physical GC decisions.

use super::super::error::StoreError;
use super::super::sql::{args, SqlRow, SqlTx};
use super::super::types::{
    FileGcCandidate, FileId, FilePurpose, FileRecord, FileState, NewUpload, OwnerId, Page, PageRequest,
    SessionGcCandidate, SessionId, SessionKind, StoreSettings, StoredObject, UnixMillis, UploadSession,
};
use super::common::{
    active_policy, admit, apply_quota, artifact_from_db, corrupt, file_row, lock_quota, millis, owner_from_db,
    owner_limits, signed, QuotaDelta, FILE_COLUMNS,
};
use crate::serve::batches::store::sql::db_now;

const MAX_FILENAME_BYTES: usize = 255;

fn check_filename(filename: &str) -> Result<(), StoreError> {
    if filename.is_empty() || filename.len() > MAX_FILENAME_BYTES || filename.chars().any(char::is_control) {
        return Err(StoreError::Invalid(super::super::error::InvalidValue::new("filename", "must have 1..255 bytes")));
    }
    Ok(())
}

/// Reserve a file slot, an upload session slot and `max_upload_bytes + object_charge` before the first byte.
pub(in super::super) async fn reserve_upload<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    upload: &NewUpload,
) -> Result<UploadSession, StoreError> {
    check_filename(&upload.filename)?;
    let usage = lock_quota(tx, &upload.owner).await?;
    let policy = active_policy(tx).await?;
    let limits = owner_limits(tx, policy.revision, &upload.owner).await?;
    let reserve = policy.max_upload.saturating_add(policy.object_charge);
    let delta = QuotaDelta { reserved: reserve, files: 1, sessions: 1, ..QuotaDelta::default() };
    admit(&usage, &limits, &delta)?;
    let now = db_now(tx).await?;
    let lease_until = now.saturating_add(millis(settings.upload_lease));
    let session = SessionId::new_random();
    let file_id = FileId::new_random();
    let expires_after = upload.expires_after.map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX));
    tx.execute(
        "INSERT INTO artifact_sessions (id, owner_id, kind, artifact_ref, phase, lease_until, reserved_bytes, \
         max_bytes, file_id, filename, expires_after_secs, created_at) \
         VALUES ($1, $2, 'upload', $3, 'open', $4, $5, $6, $7, $8, $9, $10)",
        args![
            session.as_str(),
            upload.owner.as_str(),
            upload.artifact.as_str(),
            lease_until,
            reserve,
            policy.max_upload,
            file_id.as_str(),
            upload.filename.as_str(),
            expires_after,
            now
        ],
    )
    .await?;
    apply_quota(tx, &upload.owner, delta).await?;
    Ok(UploadSession { id: session, max_bytes: super::common::unsigned(policy.max_upload)? })
}

/// Turn a finished upload into a file: the session reservation becomes the file's actual charge.
pub(in super::super) async fn complete_upload<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    owner: &OwnerId,
    session: &SessionId,
    object: StoredObject,
) -> Result<FileRecord, StoreError> {
    lock_quota(tx, owner).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT file_id, artifact_ref, reserved_bytes, max_bytes, filename, expires_after_secs, lease_until \
         FROM artifact_sessions{} WHERE id = $1 AND owner_id = $2 AND kind = 'upload' AND phase = 'open'{}",
        d.lock_hint, d.for_update
    );
    let row =
        tx.fetch_optional(&sql, args![session.as_str(), owner.as_str()]).await?.ok_or(StoreError::SessionClosed)?;
    let file_id = super::common::file_id_from_db(row.text_at(0)?)?;
    let artifact = artifact_from_db(row.text_at(1)?)?;
    let reserved = row.i64_at(2)?;
    let max_bytes = row.i64_at(3)?;
    let filename = row.opt_text_at(4)?.ok_or(corrupt("upload filename"))?;
    let expires_after = row.opt_i64_at(5)?;
    let lease_until = row.i64_at(6)?;
    let now = db_now(tx).await?;
    if lease_until < now {
        // GC removes the object of an expired session; its reservation stays until then.
        return Err(StoreError::SessionClosed);
    }
    let bytes = signed(object.bytes)?;
    if bytes > max_bytes {
        return Err(StoreError::InputUnusable("upload exceeds the size limit"));
    }
    let policy = active_policy(tx).await?;
    let charged = bytes.saturating_add(policy.object_charge);
    if charged > reserved {
        return Err(corrupt("upload charge exceeds its reservation"));
    }
    let retention = expires_after.map_or(millis(settings.file_retention), |s| s.saturating_mul(1000));
    let expires_at = now.saturating_add(retention);
    tx.execute(
        "INSERT INTO batch_files (owner_id, id, purpose, filename, artifact_ref, bytes, charged_bytes, sha256, \
         created_at, expires_at, state, pin_count) VALUES ($1, $2, 'batch', $3, $4, $5, $6, $7, $8, $9, 'active', 0)",
        args![
            owner.as_str(),
            file_id.as_str(),
            filename.as_str(),
            artifact.as_str(),
            bytes,
            charged,
            object.sha256.0.to_vec(),
            now,
            expires_at
        ],
    )
    .await?;
    tx.execute("DELETE FROM artifact_sessions WHERE id = $1", args![session.as_str()]).await?;
    // The file slot reserved with the session stays taken by the file.
    apply_quota(tx, owner, QuotaDelta { used: charged, reserved: -reserved, sessions: -1, ..QuotaDelta::default() })
        .await?;
    Ok(FileRecord {
        owner: owner.clone(),
        id: file_id,
        purpose: FilePurpose::Batch,
        filename,
        bytes: object.bytes,
        sha256: object.sha256,
        created_at: UnixMillis(now),
        expires_at: Some(UnixMillis(expires_at)),
        state: FileState::Active,
        artifact,
    })
}

/// Abandon an upload; GC deletes the object after the I/O guard and then releases the session.
pub(in super::super) async fn abort_upload<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    session: &SessionId,
) -> Result<(), StoreError> {
    let now = db_now(tx).await?;
    tx.execute(
        "UPDATE artifact_sessions SET phase = 'abandoned', lease_until = $3 \
         WHERE id = $1 AND owner_id = $2 AND kind = 'upload' AND phase = 'open'",
        args![session.as_str(), owner.as_str(), now],
    )
    .await?;
    Ok(())
}

/// Active, unexpired file of an owner; everything else is `NotFound`.
pub(in super::super) async fn get_file<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    id: &FileId,
) -> Result<FileRecord, StoreError> {
    let now = db_now(tx).await?;
    let sql = format!(
        "SELECT {FILE_COLUMNS} FROM batch_files WHERE owner_id = $1 AND id = $2 AND state = 'active' \
         AND (expires_at IS NULL OR expires_at > $3)"
    );
    let row = tx.fetch_optional(&sql, args![owner.as_str(), id.as_str(), now]).await?.ok_or(StoreError::NotFound)?;
    Ok(file_row(&row)?.record)
}

pub(in super::super) async fn list_files<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    purpose: Option<FilePurpose>,
    page: &PageRequest<FileId>,
) -> Result<Page<FileRecord>, StoreError> {
    let limit = i64::from(page.limit.clamp(1, 100));
    let now = db_now(tx).await?;
    let (cursor_at, cursor_id) = match &page.after {
        Some(after) => {
            let row = tx
                .fetch_optional(
                    "SELECT created_at FROM batch_files WHERE owner_id = $1 AND id = $2",
                    args![owner.as_str(), after.as_str()],
                )
                .await?
                .ok_or(StoreError::NotFound)?;
            (Some(row.i64_at(0)?), Some(after.as_str().to_owned()))
        }
        None => (None, None),
    };
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}{FILE_COLUMNS} FROM batch_files WHERE owner_id = $1 AND state = 'active' \
         AND (expires_at IS NULL OR expires_at > $2) AND ($3 IS NULL OR purpose = $3) \
         AND ($4 IS NULL OR created_at < $4 OR (created_at = $4 AND id < $5)) \
         ORDER BY created_at DESC, id DESC{}",
        d.top("$6"),
        d.limit("$6")
    );
    let rows = tx
        .fetch_all(&sql, args![owner.as_str(), now, purpose.map(FilePurpose::as_str), cursor_at, cursor_id, limit + 1])
        .await?;
    let has_more = rows.len() as i64 > limit;
    let items = rows.iter().take(limit as usize).map(|r| file_row(r).map(|f| f.record)).collect::<Result<_, _>>()?;
    Ok(Page { items, has_more })
}

/// Logical delete: reads answer 404 at once; the object is removed after `download_grace`.
pub(in super::super) async fn delete_file<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    owner: &OwnerId,
    id: &FileId,
) -> Result<FileRecord, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {FILE_COLUMNS} FROM batch_files{} WHERE owner_id = $1 AND id = $2 AND state = 'active'{}",
        d.lock_hint, d.for_update
    );
    let row = tx.fetch_optional(&sql, args![owner.as_str(), id.as_str()]).await?.ok_or(StoreError::NotFound)?;
    let file = file_row(&row)?;
    let now = db_now(tx).await?;
    if file.record.expires_at.is_some_and(|at| at.0 <= now) {
        return Err(StoreError::NotFound);
    }
    if file.pin_count > 0 {
        return Err(StoreError::FileInUse);
    }
    let gc_after = now.saturating_add(millis(settings.download_grace));
    tx.execute(
        "UPDATE batch_files SET state = 'deleted', deleted_at = $3, gc_after = $4 WHERE owner_id = $1 AND id = $2",
        args![owner.as_str(), id.as_str(), now, gc_after],
    )
    .await?;
    Ok(FileRecord { state: FileState::Deleted, ..file.record })
}

/// Mark up to `limit` expired files deleted (also pinned inputs: the pin only defers physical GC).
pub(in super::super) async fn expire_files<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    limit: u32,
) -> Result<u64, StoreError> {
    let now = db_now(tx).await?;
    let gc_after = now.saturating_add(millis(settings.download_grace));
    let sql = tx.dialect().bounded_update(
        "batch_files",
        "owner_id, id",
        "state = 'deleted', deleted_at = $1, gc_after = $2",
        "state = 'active' AND expires_at IS NOT NULL AND expires_at <= $1",
        "expires_at",
        "$3",
    );
    tx.execute(&sql, args![now, gc_after, i64::from(limit)]).await
}

/// Deleted, unpinned files past `gc_after`: delete the object, then call `purge_file`.
pub(in super::super) async fn file_gc_candidates<T: SqlTx>(
    tx: &mut T,
    limit: u32,
) -> Result<Vec<FileGcCandidate>, StoreError> {
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}owner_id, id, artifact_ref FROM batch_files WHERE state = 'deleted' AND gc_after <= $1 \
         AND pin_count = 0 ORDER BY gc_after{}",
        d.top("$2"),
        d.limit("$2")
    );
    let rows = tx.fetch_all(&sql, args![now, i64::from(limit)]).await?;
    rows.iter()
        .map(|row| {
            Ok(FileGcCandidate {
                owner: owner_from_db(row.text_at(0)?)?,
                file_id: super::common::file_id_from_db(row.text_at(1)?)?,
                artifact: artifact_from_db(row.text_at(2)?)?,
            })
        })
        .collect()
}

/// Remove the row of a physically deleted file and release its bytes and slot. Idempotent.
pub(in super::super) async fn purge_file<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    id: &FileId,
) -> Result<bool, StoreError> {
    lock_quota(tx, owner).await?;
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT charged_bytes FROM batch_files{} WHERE owner_id = $1 AND id = $2 AND state = 'deleted' \
         AND gc_after <= $3 AND pin_count = 0{}",
        d.lock_hint, d.for_update
    );
    let Some(row) = tx.fetch_optional(&sql, args![owner.as_str(), id.as_str(), now]).await? else {
        return Ok(false);
    };
    let charged = row.i64_at(0)?;
    tx.execute("DELETE FROM batch_files WHERE owner_id = $1 AND id = $2", args![owner.as_str(), id.as_str()]).await?;
    apply_quota(tx, owner, QuotaDelta { used: -charged, files: -1, ..QuotaDelta::default() }).await?;
    Ok(true)
}

/// Sessions whose writer is finished or cut off (`lease_until + io_guard` passed): abandoned or expired uploads and
/// orphaned/expired result objects. Export sessions are released only by finalization (adopt or publish).
pub(in super::super) async fn session_gc_candidates<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    limit: u32,
) -> Result<Vec<SessionGcCandidate>, StoreError> {
    let now = db_now(tx).await?;
    let horizon = now.saturating_sub(millis(settings.io_guard));
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}id, owner_id, kind, artifact_ref, secondary_ref FROM artifact_sessions \
         WHERE kind IN ('upload', 'result') AND lease_until < $1 ORDER BY lease_until{}",
        d.top("$2"),
        d.limit("$2")
    );
    let rows = tx.fetch_all(&sql, args![horizon, i64::from(limit)]).await?;
    rows.iter()
        .map(|row| {
            let mut artifacts = vec![artifact_from_db(row.text_at(3)?)?];
            if let Some(secondary) = row.opt_text_at(4)? {
                artifacts.push(artifact_from_db(secondary)?);
            }
            Ok(SessionGcCandidate {
                session: SessionId::parse(&row.text_at(0)?).map_err(|_| corrupt("session id"))?,
                owner: owner_from_db(row.text_at(1)?)?,
                artifacts,
            })
        })
        .collect()
}

/// Release a GC'd session after its objects were deleted. Idempotent; re-checks eligibility under the quota lock.
pub(in super::super) async fn release_session<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    owner: &OwnerId,
    session: &SessionId,
) -> Result<bool, StoreError> {
    lock_quota(tx, owner).await?;
    let now = db_now(tx).await?;
    let horizon = now.saturating_sub(millis(settings.io_guard));
    let d = tx.dialect();
    let sql = format!(
        "SELECT kind, reserved_bytes FROM artifact_sessions{} WHERE id = $1 AND owner_id = $2 \
         AND kind IN ('upload', 'result') AND lease_until < $3{}",
        d.lock_hint, d.for_update
    );
    let Some(row) = tx.fetch_optional(&sql, args![session.as_str(), owner.as_str(), horizon]).await? else {
        return Ok(false);
    };
    let kind = SessionKind::from_db(&row.text_at(0)?).ok_or(corrupt("session kind"))?;
    let reserved = row.i64_at(1)?;
    tx.execute("DELETE FROM artifact_sessions WHERE id = $1", args![session.as_str()]).await?;
    let delta = match kind {
        SessionKind::Upload => QuotaDelta { reserved: -reserved, files: -1, sessions: -1, ..QuotaDelta::default() },
        _ => QuotaDelta { reserved: -reserved, ..QuotaDelta::default() },
    };
    apply_quota(tx, owner, delta).await?;
    Ok(true)
}
