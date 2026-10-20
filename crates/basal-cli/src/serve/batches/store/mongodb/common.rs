//! Shared MongoDB domain helpers: quota and batch control locks, policy reads, claim rows.
//!
//! Lock order is the one of the SQL adapters: quota (owner) → schedule (shard) → batch control → items, claims,
//! settlements → files and sessions. A lock is a write to the control document (see `driver`), so the order also
//! bounds write-conflict cycles; every item, claim and settlement change happens after its batch control lock.

use mongodb::bson::{doc, Bson, Document};

use super::super::error::StoreError;
use super::super::ops::common::{BatchRow, PolicyRow, QuotaDelta};
use super::super::types::{BatchRef, LeaseKind, OwnerId, QuotaLimits, QuotaUsage};
use super::convert::{batch_row, quota_usage};
use super::driver::{corrupt, int, uint, Tx, BATCHES, CLAIMS, LIMITS, POLICY, QUOTA};

/// Lock (creating when absent) the quota document of an owner and return its usage and the lock time.
pub(super) async fn lock_quota(tx: &mut Tx<'_>, owner: &OwnerId) -> Result<(QuotaUsage, i64), StoreError> {
    let defaults = doc! {
        "used_bytes": 0_i64,
        "reserved_bytes": 0_i64,
        "active_batches": 0_i64,
        "retained_batches": 0_i64,
        "files": 0_i64,
        "open_sessions": 0_i64,
        "version": 0_i64,
    };
    let (doc, now) = tx.lock_or_create(QUOTA, owner.as_str(), defaults).await?;
    Ok((quota_usage(&doc)?, now))
}

/// Apply a quota delta in the transaction that changes the owner documents of those bytes. A result below zero is
/// refused (the filter guards every decreasing field) instead of hiding drift.
pub(super) async fn apply_quota(tx: &mut Tx<'_>, owner: &OwnerId, delta: QuotaDelta) -> Result<(), StoreError> {
    let fields = [
        ("used_bytes", delta.used),
        ("reserved_bytes", delta.reserved),
        ("active_batches", delta.active),
        ("retained_batches", delta.retained),
        ("files", delta.files),
        ("open_sessions", delta.sessions),
    ];
    if fields.iter().all(|(_, v)| *v == 0) {
        return Ok(());
    }
    let mut filter = doc! { "_id": owner.as_str() };
    let mut inc = doc! { "version": 1_i64 };
    for (field, value) in fields {
        if value < 0 {
            filter.insert(field, doc! { "$gte": value.saturating_neg() });
        }
        if value != 0 {
            inc.insert(field, value);
        }
    }
    if tx.update(QUOTA, filter, doc! { "$inc": inc }).await? != 1 {
        return Err(corrupt("quota would become negative"));
    }
    Ok(())
}

pub(super) async fn active_policy(tx: &mut Tx<'_>) -> Result<PolicyRow, StoreError> {
    let doc = tx.one_sorted(POLICY, doc! {}, doc! { "_id": -1_i32 }).await?.ok_or(StoreError::PolicyMissing)?;
    Ok(PolicyRow {
        revision: int(&doc, "_id")?,
        object_charge: int(&doc, "object_charge_bytes")?,
        item_overhead: int(&doc, "item_overhead_bytes")?,
        max_upload: int(&doc, "max_upload_bytes")?,
        max_records: int(&doc, "max_records_per_batch")?,
    })
}

pub(super) fn limits_from(doc: &Document, prefix: &str) -> Result<QuotaLimits, StoreError> {
    Ok(QuotaLimits {
        max_bytes: uint(doc, &format!("{prefix}max_bytes"))?,
        max_active_batches: uint(doc, &format!("{prefix}max_active_batches"))?,
        max_retained_batches: uint(doc, &format!("{prefix}max_retained_batches"))?,
        max_files: uint(doc, &format!("{prefix}max_files"))?,
        max_open_upload_sessions: uint(doc, &format!("{prefix}max_open_upload_sessions"))?,
    })
}

pub(super) async fn owner_limits(tx: &mut Tx<'_>, revision: i64, owner: &OwnerId) -> Result<QuotaLimits, StoreError> {
    let doc = tx
        .one(LIMITS, doc! { "revision": revision, "owner_id": owner.as_str() })
        .await?
        .ok_or(StoreError::OwnerNotProvisioned)?;
    limits_from(&doc, "")
}

pub(super) fn batch_filter(batch: &BatchRef) -> Document {
    doc! { "_id": batch.id.as_str(), "owner_id": batch.owner.as_str() }
}

/// Lock the batch control document (owner-scoped) and return it with the lock time.
pub(super) async fn lock_batch(tx: &mut Tx<'_>, batch: &BatchRef) -> Result<(BatchRow, i64), StoreError> {
    let (doc, now) = tx.lock(BATCHES, batch_filter(batch)).await?.ok_or(StoreError::NotFound)?;
    Ok((batch_row(&doc)?, now))
}

pub(super) async fn try_lock_batch(tx: &mut Tx<'_>, batch: &BatchRef) -> Result<Option<(BatchRow, i64)>, StoreError> {
    match tx.lock(BATCHES, batch_filter(batch)).await? {
        Some((doc, now)) => Ok(Some((batch_row(&doc)?, now))),
        None => Ok(None),
    }
}

/// Read a batch without locking it.
pub(super) async fn read_batch(tx: &mut Tx<'_>, batch: &BatchRef) -> Result<Option<BatchRow>, StoreError> {
    tx.one(BATCHES, batch_filter(batch)).await?.map(|doc| batch_row(&doc)).transpose()
}

/// Update fields of a locked batch.
pub(super) async fn set_batch(tx: &mut Tx<'_>, batch: &BatchRef, update: Document) -> Result<(), StoreError> {
    if tx.update(BATCHES, batch_filter(batch), update).await? != 1 {
        return Err(corrupt("locked batch disappeared"));
    }
    Ok(())
}

#[expect(clippy::too_many_arguments, reason = "one claim document with its fields")]
pub(super) async fn insert_claim(
    tx: &mut Tx<'_>,
    claim_id: &str,
    kind: LeaseKind,
    worker: &str,
    generation: i64,
    batch: &BatchRef,
    control_epoch: i64,
    lease_until: i64,
    now: i64,
) -> Result<(), StoreError> {
    tx.insert(
        CLAIMS,
        doc! {
            "_id": claim_id,
            "kind": kind.as_db(),
            "worker_id": worker,
            "generation": generation,
            "owner_id": batch.owner.as_str(),
            "batch_id": batch.id.as_str(),
            "control_epoch": control_epoch,
            "lease_until": lease_until,
            "state": "active",
            "release_reason": Bson::Null,
            "created_at": now,
        },
    )
    .await
}

/// Finish the claim of a phase lease and clear the batch phase slot (batch control must be locked).
pub(super) async fn end_phase(tx: &mut Tx<'_>, batch: &BatchRow) -> Result<(), StoreError> {
    if let Some(claim) = &batch.phase_claim_id {
        tx.update(CLAIMS, doc! { "_id": claim.as_str(), "state": "active" }, doc! { "$set": { "state": "finished" } })
            .await?;
    }
    set_batch(
        tx,
        &batch.batch_ref(),
        doc! { "$set": { "phase_kind": Bson::Null, "phase_claim_id": Bson::Null, "phase_lease_until": Bson::Null } },
    )
    .await
}

/// Number of item documents of a batch.
pub(super) async fn count_batch_items(tx: &mut Tx<'_>, batch: &BatchRef) -> Result<i64, StoreError> {
    tx.count(super::driver::ITEMS, doc! { "owner_id": batch.owner.as_str(), "batch_id": batch.id.as_str() }).await
}
