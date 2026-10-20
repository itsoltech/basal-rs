//! Catalog and quota policy installation, quota drift recomputation and bounded counters (MongoDB).

use mongodb::bson::{doc, Bson, Document};

use super::super::error::StoreError;
use super::super::ops::admin::Counters;
use super::super::ops::common::signed;
use super::super::policy::{catalog_digest, policy_digest, validate_catalog, validate_policy};
use super::super::types::{
    BacklogEntry, BatchStatus, Catalog, CatalogEntry, CatalogSpec, Fingerprint, InstallOutcome, ModelName, OwnerId,
    PlannerVersion, QuotaDrift, QuotaPolicy, QuotaPolicySpec, QuotaUsage, Revision,
};
use super::common::limits_from;
use super::convert::{digest, owner, quota_usage};
use super::driver::{
    bin, bytes, corrupt, int, mtx, opt_text, text, uint, Core, Tx, BATCHES, CATALOG, CATALOG_REVISIONS, CLAIMS, FILES,
    ITEMS, LIMITS, POLICY, QUOTA, SESSIONS, SETTLEMENTS,
};

/// Largest number of owners of one policy revision read back at once.
const MAX_POLICY_OWNERS: i64 = 100_000;

pub(super) async fn latest_catalog(core: &Core) -> Result<Option<Catalog>, StoreError> {
    let Some(head) = core.read_many(CATALOG_REVISIONS, doc! {}, doc! { "_id": -1_i32 }, 1).await?.into_iter().next()
    else {
        return Ok(None);
    };
    catalog_revision(core, Revision(int(&head, "_id")?)).await
}

/// Read one immutable revision retained for accepted batches.
pub(super) async fn catalog_revision(core: &Core, revision: Revision) -> Result<Option<Catalog>, StoreError> {
    let Some(head) = core.read_one(CATALOG_REVISIONS, doc! { "_id": revision.0 }).await? else {
        return Ok(None);
    };
    let revision = revision.0;
    let rows = core.read_many(CATALOG, doc! { "revision": revision }, doc! { "model_name": 1_i32 }, 1024).await?;
    let entries = rows
        .iter()
        .map(|row| {
            Ok(CatalogEntry {
                model_name: ModelName::parse(&text(row, "model_name")?).map_err(|_| corrupt("catalog model"))?,
                fingerprint: Fingerprint::parse(&text(row, "fingerprint")?)
                    .map_err(|_| corrupt("catalog fingerprint"))?,
                planner_version: PlannerVersion::parse(&text(row, "planner_version")?)
                    .map_err(|_| corrupt("catalog planner"))?,
            })
        })
        .collect::<Result<_, StoreError>>()?;
    let catalog = Catalog {
        revision: Revision(revision),
        spec_sha256: digest(bin(&head, "spec_sha256")?)?,
        spec: CatalogSpec {
            default_model: ModelName::parse(&text(&head, "default_model")?).map_err(|_| corrupt("catalog default"))?,
            entries,
        },
    };
    validate_catalog(&catalog.spec).map_err(|_| corrupt("invalid immutable catalog"))?;
    if catalog_digest(&catalog.spec) != catalog.spec_sha256 {
        return Err(corrupt("immutable catalog digest mismatch"));
    }
    Ok(Some(catalog))
}

async fn install_catalog_tx(tx: &mut Tx<'_>, spec: &CatalogSpec) -> Result<InstallOutcome, StoreError> {
    let digest_value = catalog_digest(spec);
    let latest = tx.one_sorted(CATALOG_REVISIONS, doc! {}, doc! { "_id": -1_i32 }).await?;
    let next = match latest {
        Some(row) if bin(&row, "spec_sha256")? == digest_value.0 => {
            return Ok(InstallOutcome::Unchanged(Revision(int(&row, "_id")?)));
        }
        Some(row) => int(&row, "_id")? + 1,
        None => 1,
    };
    // A concurrent install of the same revision number fails on the unique `_id` and is retried by the operator.
    let now = tx.server_time().await?;
    tx.insert(
        CATALOG_REVISIONS,
        doc! { "_id": next, "spec_sha256": bytes(&digest_value.0), "default_model": spec.default_model.as_str(), "installed_at": now },
    )
    .await?;
    for entry in &spec.entries {
        tx.insert(
            CATALOG,
            doc! {
                "revision": next,
                "model_name": entry.model_name.as_str(),
                "fingerprint": entry.fingerprint.as_str(),
                "planner_version": entry.planner_version.as_str(),
            },
        )
        .await?;
    }
    Ok(InstallOutcome::Installed(Revision(next)))
}

/// Install a new immutable catalog revision, or report the identical latest one.
pub(super) async fn install_catalog(core: &Core, spec: &CatalogSpec) -> Result<InstallOutcome, StoreError> {
    validate_catalog(spec)?;
    mtx!(core, tx => install_catalog_tx(tx, spec))
}

pub(super) async fn active_quota_policy(core: &Core) -> Result<Option<QuotaPolicy>, StoreError> {
    let Some(head) = core.read_many(POLICY, doc! {}, doc! { "_id": -1_i32 }, 1).await?.into_iter().next() else {
        return Ok(None);
    };
    let revision = int(&head, "_id")?;
    let rows =
        core.read_many(LIMITS, doc! { "revision": revision }, doc! { "owner_id": 1_i32 }, MAX_POLICY_OWNERS).await?;
    let owners = rows
        .iter()
        .map(|row| Ok((owner(text(row, "owner_id")?)?, limits_from(row, "")?)))
        .collect::<Result<_, StoreError>>()?;
    Ok(Some(QuotaPolicy {
        revision: Revision(revision),
        spec_sha256: digest(bin(&head, "spec_sha256")?)?,
        spec: QuotaPolicySpec {
            object_charge_bytes: uint(&head, "object_charge_bytes")?,
            item_overhead_bytes: uint(&head, "item_overhead_bytes")?,
            max_upload_bytes: uint(&head, "max_upload_bytes")?,
            max_records_per_batch: uint(&head, "max_records_per_batch")?,
            installation: limits_from(&head, "inst_")?,
            owners,
        },
    }))
}

fn limits_doc(limits: &super::super::types::QuotaLimits, prefix: &str) -> Result<Document, StoreError> {
    Ok(doc! {
        format!("{prefix}max_bytes"): signed(limits.max_bytes)?,
        format!("{prefix}max_active_batches"): signed(limits.max_active_batches)?,
        format!("{prefix}max_retained_batches"): signed(limits.max_retained_batches)?,
        format!("{prefix}max_files"): signed(limits.max_files)?,
        format!("{prefix}max_open_upload_sessions"): signed(limits.max_open_upload_sessions)?,
    })
}

fn within(usage: &QuotaUsage, limits: &super::super::types::QuotaLimits) -> bool {
    usage.used_bytes.saturating_add(usage.reserved_bytes) <= limits.max_bytes
        && usage.active_batches <= limits.max_active_batches
        && usage.retained_batches <= limits.max_retained_batches
        && usage.files <= limits.max_files
        && usage.open_sessions <= limits.max_open_upload_sessions
}

async fn install_quota_policy_tx(tx: &mut Tx<'_>, spec: &QuotaPolicySpec) -> Result<InstallOutcome, StoreError> {
    let digest_value = policy_digest(spec);
    // Write every quota document: any concurrent quota change write-conflicts with this administrative transaction.
    tx.update_many(QUOTA, doc! {}, doc! { "$inc": { "lock_seq": 1_i64 } }).await?;
    let rows = tx.many(QUOTA, doc! {}, doc! { "_id": 1_i32 }, MAX_POLICY_OWNERS).await?;
    let latest = tx.one_sorted(POLICY, doc! {}, doc! { "_id": -1_i32 }).await?;
    let next = match latest {
        Some(row) if bin(&row, "spec_sha256")? == digest_value.0 => {
            return Ok(InstallOutcome::Unchanged(Revision(int(&row, "_id")?)));
        }
        Some(row) => int(&row, "_id")? + 1,
        None => 1,
    };
    for row in &rows {
        let owner_id = owner(text(row, "_id")?)?;
        let usage = quota_usage(row)?;
        let committed = usage != QuotaUsage { version: usage.version, ..QuotaUsage::default() };
        match spec.owners.iter().find(|(o, _)| o == &owner_id) {
            Some((_, limits)) if !within(&usage, limits) => return Err(StoreError::PolicyBelowCommitted(owner_id)),
            None if committed => return Err(StoreError::PolicyBelowCommitted(owner_id)),
            _ => {}
        }
    }
    let now = tx.server_time().await?;
    let mut head = doc! {
        "_id": next,
        "spec_sha256": bytes(&digest_value.0),
        "object_charge_bytes": signed(spec.object_charge_bytes)?,
        "item_overhead_bytes": signed(spec.item_overhead_bytes)?,
        "max_upload_bytes": signed(spec.max_upload_bytes)?,
        "max_records_per_batch": signed(spec.max_records_per_batch)?,
        "installed_at": now,
    };
    head.extend(limits_doc(&spec.installation, "inst_")?);
    tx.insert(POLICY, head).await?;
    for (owner_id, limits) in &spec.owners {
        let mut row = doc! { "revision": next, "owner_id": owner_id.as_str() };
        row.extend(limits_doc(limits, "")?);
        tx.insert(LIMITS, row).await?;
    }
    Ok(InstallOutcome::Installed(Revision(next)))
}

/// Install a quota policy revision (administrative; run with batch admission paused).
pub(super) async fn install_quota_policy(core: &Core, spec: &QuotaPolicySpec) -> Result<InstallOutcome, StoreError> {
    validate_policy(spec)?;
    mtx!(core, tx => install_quota_policy_tx(tx, spec))
}

pub(super) async fn quota_usage_of(core: &Core, owner_id: &OwnerId) -> Result<QuotaUsage, StoreError> {
    match core.read_one(QUOTA, doc! { "_id": owner_id.as_str() }).await? {
        Some(doc) => quota_usage(&doc),
        None => Ok(QuotaUsage::default()),
    }
}

async fn derived_usage_tx(tx: &mut Tx<'_>, owner_id: &str) -> Result<QuotaUsage, StoreError> {
    let by_owner = doc! { "owner_id": owner_id };
    let files_count = tx.count(FILES, by_owner.clone()).await?;
    let files_bytes = tx.sum(FILES, by_owner.clone(), "charged_bytes").await?;
    let items = tx.sum(ITEMS, by_owner.clone(), "quota_used").await?;
    let batches = tx.count(BATCHES, by_owner.clone()).await?;
    let mut unreleased = by_owner.clone();
    unreleased.insert("released", false);
    let active = tx.count(BATCHES, unreleased).await?;
    let errors = tx.sum(BATCHES, by_owner.clone(), "errors_bytes").await?;
    let completion = tx.sum(BATCHES, by_owner.clone(), "completion_reserved").await?;
    let export = tx.sum(BATCHES, by_owner.clone(), "export_reserved").await?;
    let mut open_settlements = by_owner.clone();
    open_settlements.insert("accounted", false);
    let settlements = tx.sum(SETTLEMENTS, open_settlements, "reserved_bytes").await?;
    let sessions = tx.sum(SESSIONS, by_owner.clone(), "reserved_bytes").await?;
    let mut uploads = by_owner;
    uploads.insert("kind", "upload");
    let uploads = tx.count(SESSIONS, uploads).await?;
    let unsigned = |v: i64| u64::try_from(v).map_err(|_| corrupt("negative aggregate"));
    Ok(QuotaUsage {
        used_bytes: unsigned(files_bytes + items + errors)?,
        reserved_bytes: unsigned(completion + export + settlements + sessions)?,
        active_batches: unsigned(active)?,
        retained_batches: unsigned(batches)?,
        files: unsigned(files_count + uploads + 2 * active)?,
        open_sessions: unsigned(uploads)?,
        version: 0,
    })
}

/// Owners whose `batch_quota` differs from the sum of their owner documents (bounded doctor check). Each owner is
/// compared inside one snapshot transaction so the totals and the owner documents are consistent.
pub(super) async fn quota_drift(core: &Core, owner_limit: u32) -> Result<Vec<QuotaDrift>, StoreError> {
    let rows = core.read_many(QUOTA, doc! {}, doc! { "_id": 1_i32 }, i64::from(owner_limit)).await?;
    let mut drift = Vec::new();
    for row in &rows {
        let owner_id = text(row, "_id")?;
        let (recorded, derived) = mtx!(core, tx => async {
            let current = tx.one(QUOTA, doc! { "_id": owner_id.as_str() }).await?.ok_or(corrupt("quota document"))?;
            let recorded = QuotaUsage { version: 0, ..quota_usage(&current)? };
            let derived = derived_usage_tx(tx, &owner_id).await?;
            Ok::<_, StoreError>((recorded, derived))
        })?;
        if recorded != derived {
            drift.push(QuotaDrift { owner: owner(owner_id)?, recorded, derived });
        }
    }
    Ok(drift)
}

pub(super) async fn counters(core: &Core) -> Result<Counters, StoreError> {
    let now = core.server_now().await?;
    let rows = core
        .read_aggregate(
            BATCHES,
            vec![doc! { "$group": { "_id": "$status", "n": { "$sum": 1_i64 } } }, doc! { "$limit": 16_i64 }],
        )
        .await?;
    let batches_by_status = rows
        .iter()
        .map(|row| Ok((BatchStatus::from_db(&text(row, "_id")?).ok_or(corrupt("batch status"))?, uint(row, "n")?)))
        .collect::<Result<_, StoreError>>()?;
    let count = |coll: &'static str, filter: Document| async move {
        let n = core
            .coll(coll)
            .count_documents(filter)
            .max_time(core.op_timeout)
            .await
            .map_err(|e| super::driver::redact(&e))?;
        Ok::<u64, StoreError>(n)
    };
    Ok(Counters {
        batches_by_status,
        expired_claims: count(CLAIMS, doc! { "state": "active", "lease_until": { "$lt": now } }).await?,
        unaccounted_settlements: count(SETTLEMENTS, doc! { "accounted": false, "kind": { "$ne": Bson::Null } }).await?,
        pending_cleanup: count(BATCHES, doc! { "cleanup_state": "pending" }).await?,
        sessions_open: count(SESSIONS, doc! {}).await?,
    })
}

pub(super) async fn backlog(core: &Core, limit: u32) -> Result<Vec<BacklogEntry>, StoreError> {
    let rows = core
        .read_aggregate(
            BATCHES,
            vec![
                doc! { "$match": { "status": { "$in": ["validating", "in_progress"] } } },
                doc! { "$group": {
                    "_id": { "fingerprint": "$fingerprint", "status": "$status" },
                    "batches": { "$sum": 1_i64 },
                    "queued": { "$sum": "$queued" },
                } },
                doc! { "$sort": { "_id.fingerprint": 1_i32, "_id.status": 1_i32 } },
                doc! { "$limit": i64::from(limit) },
            ],
        )
        .await?;
    rows.iter()
        .map(|row| {
            let key = row.get_document("_id").map_err(|_| corrupt("backlog key"))?;
            Ok(BacklogEntry {
                fingerprint: opt_text(key, "fingerprint")?
                    .map(|v| Fingerprint::parse(&v))
                    .transpose()
                    .map_err(|_| corrupt("fingerprint"))?,
                status: BatchStatus::from_db(&text(key, "status")?).ok_or(corrupt("batch status"))?,
                batches: uint(row, "batches")?,
                queued: uint(row, "queued")?,
            })
        })
        .collect()
}
