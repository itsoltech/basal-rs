//! Administrative operations (catalog and quota policy installation, versioned migrations) and diagnostics.

use super::super::error::StoreError;
use super::super::policy::{catalog_digest, policy_digest, validate_catalog, validate_policy};
use super::super::sql::{args, db_now, SqlRow, SqlTx};
use super::super::types::{
    hex, BacklogEntry, BatchStatus, Catalog, CatalogEntry, CatalogSpec, Fingerprint, InstallOutcome, ModelName,
    OwnerId, PlannerVersion, QuotaDrift, QuotaLimits, QuotaPolicy, QuotaPolicySpec, QuotaUsage, Revision, SchemaState,
};
use super::common::{corrupt, digest_from_db, owner_from_db, quota_usage, scalar, signed, QUOTA_COLUMNS};

/// One embedded, versioned migration script.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct Migration {
    pub version: i64,
    pub id: &'static str,
    pub sql: &'static str,
}

impl Migration {
    fn checksum(&self) -> String {
        use sha2::{Digest, Sha256};
        hex(&Sha256::digest(self.sql.as_bytes()))
    }
}

/// Read-only schema check used by runtime start and doctor; never creates objects.
pub(in super::super) async fn schema_state<T: SqlTx>(
    tx: &mut T,
    migrations: &[Migration],
) -> Result<SchemaState, StoreError> {
    let expected = migrations.last().map_or(0, |m| m.version);
    let exists_sql = tx.dialect().table_exists;
    if scalar(tx, exists_sql, args!["batch_schema"]).await? == 0 {
        return Ok(SchemaState::Empty);
    }
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}version, checksum, state FROM batch_schema ORDER BY version{}",
        d.top("1000"),
        d.limit("1000")
    );
    let rows = tx.fetch_all(&sql, args![]).await?;
    let mut version = 0;
    for row in &rows {
        version = row.i64_at(0)?;
        let checksum = row.text_at(1)?;
        let state = row.text_at(2)?;
        if version > expected {
            return Ok(SchemaState::TooNew { version, expected });
        }
        let known = migrations.iter().find(|m| m.version == version);
        if state != "applied" || known.is_none_or(|m| m.checksum() != checksum) {
            return Ok(SchemaState::Damaged { version });
        }
    }
    Ok(match version {
        0 => SchemaState::Empty,
        v if v == expected => SchemaState::Current { version },
        v => SchemaState::Outdated { version: v, expected },
    })
}

/// Apply pending migrations in one transaction. Scripts only create objects; nothing is dropped or rewritten.
/// A newer or damaged schema is refused instead of being changed.
pub(in super::super) async fn apply_migrations<T: SqlTx>(
    tx: &mut T,
    migrations: &[Migration],
) -> Result<SchemaState, StoreError> {
    let state = schema_state(tx, migrations).await?;
    let applied = match state {
        SchemaState::Empty => 0,
        SchemaState::Current { .. } => return Ok(state),
        SchemaState::Outdated { version, .. } => version,
        SchemaState::TooNew { .. } | SchemaState::Damaged { .. } => return Err(StoreError::Schema(state)),
    };
    let schema_table = tx.dialect().schema_table;
    tx.execute(schema_table, args![]).await?;
    for migration in migrations.iter().filter(|m| m.version > applied) {
        let now = db_now(tx).await?;
        tx.execute(
            "INSERT INTO batch_schema (version, migration_id, checksum, state, applied_at) \
             VALUES ($1, $2, $3, 'applying', $4)",
            args![migration.version, migration.id, migration.checksum(), now],
        )
        .await?;
        tx.execute_script(migration.sql).await?;
        tx.execute("UPDATE batch_schema SET state = 'applied' WHERE version = $1", args![migration.version]).await?;
    }
    schema_state(tx, migrations).await
}

// ---------------------------------------------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------------------------------------------

pub(in super::super) async fn latest_catalog<T: SqlTx>(tx: &mut T) -> Result<Option<Catalog>, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}revision, spec_sha256, default_model FROM batch_catalog_revisions ORDER BY revision DESC{}",
        d.top("1"),
        d.limit("1")
    );
    let Some(head) = tx.fetch_optional(&sql, args![]).await? else {
        return Ok(None);
    };
    catalog_revision(tx, Revision(head.i64_at(0)?)).await
}

/// Read one immutable revision retained for accepted batches.
pub(in super::super) async fn catalog_revision<T: SqlTx>(
    tx: &mut T,
    revision: Revision,
) -> Result<Option<Catalog>, StoreError> {
    let Some(head) = tx
        .fetch_optional(
            "SELECT revision, spec_sha256, default_model FROM batch_catalog_revisions WHERE revision = $1",
            args![revision.0],
        )
        .await?
    else {
        return Ok(None);
    };
    let revision = revision.0;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}model_name, fingerprint, planner_version FROM batch_catalog WHERE revision = $1 \
         ORDER BY model_name{}",
        d.top("1024"),
        d.limit("1024")
    );
    let rows = tx.fetch_all(&sql, args![revision]).await?;
    let entries = rows
        .iter()
        .map(|row| {
            Ok(CatalogEntry {
                model_name: ModelName::parse(&row.text_at(0)?).map_err(|_| corrupt("catalog model"))?,
                fingerprint: Fingerprint::parse(&row.text_at(1)?).map_err(|_| corrupt("catalog fingerprint"))?,
                planner_version: PlannerVersion::parse(&row.text_at(2)?).map_err(|_| corrupt("catalog planner"))?,
            })
        })
        .collect::<Result<_, StoreError>>()?;
    let catalog = Catalog {
        revision: Revision(revision),
        spec_sha256: digest_from_db(head.blob_at(1)?)?,
        spec: CatalogSpec {
            default_model: ModelName::parse(&head.text_at(2)?).map_err(|_| corrupt("catalog default"))?,
            entries,
        },
    };
    validate_catalog(&catalog.spec).map_err(|_| corrupt("invalid immutable catalog"))?;
    if catalog_digest(&catalog.spec) != catalog.spec_sha256 {
        return Err(corrupt("immutable catalog digest mismatch"));
    }
    Ok(Some(catalog))
}

/// Install a new immutable catalog revision, or report the identical latest one. Accepted batches keep theirs.
pub(in super::super) async fn install_catalog<T: SqlTx>(
    tx: &mut T,
    spec: &CatalogSpec,
) -> Result<InstallOutcome, StoreError> {
    validate_catalog(spec)?;
    let digest = catalog_digest(spec);
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}revision, spec_sha256 FROM batch_catalog_revisions{} ORDER BY revision DESC{}{}",
        d.top("1"),
        d.lock_hint,
        d.limit("1"),
        d.for_update
    );
    let latest = tx.fetch_optional(&sql, args![]).await?;
    let next = match latest {
        Some(row) if row.blob_at(1)? == digest.0 => return Ok(InstallOutcome::Unchanged(Revision(row.i64_at(0)?))),
        Some(row) => row.i64_at(0)? + 1,
        None => 1,
    };
    let now = db_now(tx).await?;
    tx.execute(
        "INSERT INTO batch_catalog_revisions (revision, spec_sha256, default_model, installed_at) VALUES ($1, $2, $3, $4)",
        args![next, digest.0.to_vec(), spec.default_model.as_str(), now],
    )
    .await?;
    for entry in &spec.entries {
        tx.execute(
            "INSERT INTO batch_catalog (revision, model_name, fingerprint, planner_version) VALUES ($1, $2, $3, $4)",
            args![next, entry.model_name.as_str(), entry.fingerprint.as_str(), entry.planner_version.as_str()],
        )
        .await?;
    }
    Ok(InstallOutcome::Installed(Revision(next)))
}

// ---------------------------------------------------------------------------------------------------------------
// Quota policy
// ---------------------------------------------------------------------------------------------------------------

fn limits_from<R: SqlRow>(row: &R, first: usize) -> Result<QuotaLimits, StoreError> {
    Ok(QuotaLimits {
        max_bytes: row.u64_at(first)?,
        max_active_batches: row.u64_at(first + 1)?,
        max_retained_batches: row.u64_at(first + 2)?,
        max_files: row.u64_at(first + 3)?,
        max_open_upload_sessions: row.u64_at(first + 4)?,
    })
}

pub(in super::super) async fn active_quota_policy<T: SqlTx>(tx: &mut T) -> Result<Option<QuotaPolicy>, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}revision, spec_sha256, object_charge_bytes, item_overhead_bytes, max_upload_bytes, \
         max_records_per_batch, inst_max_bytes, inst_max_active_batches, inst_max_retained_batches, \
         inst_max_files, inst_max_open_upload_sessions FROM batch_quota_policy ORDER BY revision DESC{}",
        d.top("1"),
        d.limit("1")
    );
    let Some(head) = tx.fetch_optional(&sql, args![]).await? else {
        return Ok(None);
    };
    let revision = head.i64_at(0)?;
    let sql = format!(
        "SELECT {}owner_id, max_bytes, max_active_batches, max_retained_batches, max_files, max_open_upload_sessions \
         FROM batch_quota_limits WHERE revision = $1 ORDER BY owner_id{}",
        d.top("100000"),
        d.limit("100000")
    );
    let rows = tx.fetch_all(&sql, args![revision]).await?;
    let owners = rows
        .iter()
        .map(|row| Ok((owner_from_db(row.text_at(0)?)?, limits_from(row, 1)?)))
        .collect::<Result<_, StoreError>>()?;
    Ok(Some(QuotaPolicy {
        revision: Revision(revision),
        spec_sha256: digest_from_db(head.blob_at(1)?)?,
        spec: QuotaPolicySpec {
            object_charge_bytes: head.u64_at(2)?,
            item_overhead_bytes: head.u64_at(3)?,
            max_upload_bytes: head.u64_at(4)?,
            max_records_per_batch: head.u64_at(5)?,
            installation: limits_from(&head, 6)?,
            owners,
        },
    }))
}

fn within(usage: &QuotaUsage, limits: &QuotaLimits) -> bool {
    usage.used_bytes.saturating_add(usage.reserved_bytes) <= limits.max_bytes
        && usage.active_batches <= limits.max_active_batches
        && usage.retained_batches <= limits.max_retained_batches
        && usage.files <= limits.max_files
        && usage.open_sessions <= limits.max_open_upload_sessions
}

/// Install a new policy revision. Locks every quota row (owner order), refuses limits below current commitments and
/// owners with commitments that the new policy drops. Run with batch admission paused.
pub(in super::super) async fn install_quota_policy<T: SqlTx>(
    tx: &mut T,
    spec: &QuotaPolicySpec,
) -> Result<InstallOutcome, StoreError> {
    validate_policy(spec)?;
    let digest = policy_digest(spec);
    let d = tx.dialect();
    let sql =
        format!("SELECT owner_id, {QUOTA_COLUMNS} FROM batch_quota{} ORDER BY owner_id{}", d.lock_hint, d.for_update);
    let rows = tx.fetch_all(&sql, args![]).await?;
    let sql = format!(
        "SELECT {}revision, spec_sha256 FROM batch_quota_policy ORDER BY revision DESC{}",
        d.top("1"),
        d.limit("1")
    );
    let latest = tx.fetch_optional(&sql, args![]).await?;
    let next = match latest {
        Some(row) if row.blob_at(1)? == digest.0 => return Ok(InstallOutcome::Unchanged(Revision(row.i64_at(0)?))),
        Some(row) => row.i64_at(0)? + 1,
        None => 1,
    };
    for row in &rows {
        let owner = owner_from_db(row.text_at(0)?)?;
        let usage = QuotaUsage {
            used_bytes: row.u64_at(1)?,
            reserved_bytes: row.u64_at(2)?,
            active_batches: row.u64_at(3)?,
            retained_batches: row.u64_at(4)?,
            files: row.u64_at(5)?,
            open_sessions: row.u64_at(6)?,
            version: row.i64_at(7)?,
        };
        let committed = usage != QuotaUsage { version: usage.version, ..QuotaUsage::default() };
        match spec.owners.iter().find(|(o, _)| o == &owner) {
            Some((_, limits)) if !within(&usage, limits) => return Err(StoreError::PolicyBelowCommitted(owner)),
            None if committed => return Err(StoreError::PolicyBelowCommitted(owner)),
            _ => {}
        }
    }
    let now = db_now(tx).await?;
    let i = &spec.installation;
    tx.execute(
        "INSERT INTO batch_quota_policy (revision, spec_sha256, object_charge_bytes, item_overhead_bytes, \
         max_upload_bytes, max_records_per_batch, inst_max_bytes, inst_max_active_batches, inst_max_retained_batches, \
         inst_max_files, inst_max_open_upload_sessions, installed_at) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
        args![
            next,
            digest.0.to_vec(),
            signed(spec.object_charge_bytes)?,
            signed(spec.item_overhead_bytes)?,
            signed(spec.max_upload_bytes)?,
            signed(spec.max_records_per_batch)?,
            signed(i.max_bytes)?,
            signed(i.max_active_batches)?,
            signed(i.max_retained_batches)?,
            signed(i.max_files)?,
            signed(i.max_open_upload_sessions)?,
            now
        ],
    )
    .await?;
    for (owner, limits) in &spec.owners {
        tx.execute(
            "INSERT INTO batch_quota_limits (revision, owner_id, max_bytes, max_active_batches, max_retained_batches, \
             max_files, max_open_upload_sessions) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            args![
                next,
                owner.as_str(),
                signed(limits.max_bytes)?,
                signed(limits.max_active_batches)?,
                signed(limits.max_retained_batches)?,
                signed(limits.max_files)?,
                signed(limits.max_open_upload_sessions)?
            ],
        )
        .await?;
    }
    Ok(InstallOutcome::Installed(Revision(next)))
}

pub(in super::super) async fn quota_usage_of<T: SqlTx>(tx: &mut T, owner: &OwnerId) -> Result<QuotaUsage, StoreError> {
    let sql = format!("SELECT {QUOTA_COLUMNS} FROM batch_quota WHERE owner_id = $1");
    match tx.fetch_optional(&sql, args![owner.as_str()]).await? {
        Some(row) => quota_usage(&row),
        None => Ok(QuotaUsage::default()),
    }
}

/// Recompute an owner's usage from its owner rows (files, items, batches, open settlements, sessions).
async fn derived_usage<T: SqlTx>(tx: &mut T, owner: &str) -> Result<QuotaUsage, StoreError> {
    let count = tx.dialect().count_all;
    let files = tx
        .fetch_optional(
            &format!(
                "SELECT {count}, CAST(COALESCE(SUM(charged_bytes), 0) AS BIGINT) FROM batch_files WHERE owner_id = $1"
            ),
            args![owner],
        )
        .await?
        .ok_or(corrupt("aggregate returned no row"))?;
    let items = scalar(
        tx,
        "SELECT CAST(COALESCE(SUM(quota_used), 0) AS BIGINT) FROM batch_items WHERE owner_id = $1",
        args![owner],
    )
    .await?;
    let batches = tx
        .fetch_optional(
            &format!(
                "SELECT {count}, CAST(COALESCE(SUM(CASE WHEN released = 0 THEN 1 ELSE 0 END), 0) AS BIGINT), \
                 CAST(COALESCE(SUM(errors_bytes), 0) AS BIGINT), \
                 CAST(COALESCE(SUM(completion_reserved + export_reserved), 0) AS BIGINT) FROM batches \
                 WHERE owner_id = $1"
            ),
            args![owner],
        )
        .await?
        .ok_or(corrupt("aggregate returned no row"))?;
    let settlements = scalar(
        tx,
        "SELECT CAST(COALESCE(SUM(reserved_bytes), 0) AS BIGINT) FROM batch_settlements \
         WHERE owner_id = $1 AND accounted = 0",
        args![owner],
    )
    .await?;
    let sessions = tx
        .fetch_optional(
            "SELECT CAST(COALESCE(SUM(reserved_bytes), 0) AS BIGINT), \
             CAST(COALESCE(SUM(CASE WHEN kind = 'upload' THEN 1 ELSE 0 END), 0) AS BIGINT) \
             FROM artifact_sessions WHERE owner_id = $1",
            args![owner],
        )
        .await?
        .ok_or(corrupt("aggregate returned no row"))?;
    let unsigned = |v: i64| u64::try_from(v).map_err(|_| corrupt("negative aggregate"));
    let active = unsigned(batches.i64_at(1)?)?;
    let uploads = unsigned(sessions.i64_at(1)?)?;
    Ok(QuotaUsage {
        used_bytes: unsigned(files.i64_at(1)? + items + batches.i64_at(2)?)?,
        reserved_bytes: unsigned(batches.i64_at(3)? + settlements + sessions.i64_at(0)?)?,
        active_batches: active,
        retained_batches: unsigned(batches.i64_at(0)?)?,
        files: unsigned(files.i64_at(0)?)? + uploads + 2 * active,
        open_sessions: uploads,
        version: 0,
    })
}

/// Owners whose `batch_quota` differs from the sum of their owner rows (bounded doctor check).
pub(in super::super) async fn quota_drift<T: SqlTx>(tx: &mut T, limit: u32) -> Result<Vec<QuotaDrift>, StoreError> {
    let d = tx.dialect();
    let sql =
        format!("SELECT {}owner_id, {QUOTA_COLUMNS} FROM batch_quota ORDER BY owner_id{}", d.top("$1"), d.limit("$1"));
    let rows = tx.fetch_all(&sql, args![i64::from(limit)]).await?;
    let mut drift = Vec::new();
    for row in &rows {
        let owner = row.text_at(0)?;
        let recorded = QuotaUsage {
            used_bytes: row.u64_at(1)?,
            reserved_bytes: row.u64_at(2)?,
            active_batches: row.u64_at(3)?,
            retained_batches: row.u64_at(4)?,
            files: row.u64_at(5)?,
            open_sessions: row.u64_at(6)?,
            version: 0,
        };
        let derived = derived_usage(tx, &owner).await?;
        if derived != recorded {
            drift.push(QuotaDrift { owner: owner_from_db(owner)?, recorded, derived });
        }
    }
    Ok(drift)
}

/// Bounded counters for doctor and `/health/batches` (no identifiers of clients or batches).
pub(in super::super) struct Counters {
    pub batches_by_status: Vec<(BatchStatus, u64)>,
    pub expired_claims: u64,
    pub unaccounted_settlements: u64,
    pub pending_cleanup: u64,
    pub sessions_open: u64,
}

pub(in super::super) async fn counters<T: SqlTx>(tx: &mut T) -> Result<Counters, StoreError> {
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let count_all = d.count_all;
    let sql = format!("SELECT {}status, {count_all} FROM batches GROUP BY status{}", d.top("16"), d.limit("16"));
    let rows = tx.fetch_all(&sql, args![]).await?;
    let batches_by_status = rows
        .iter()
        .map(|row| Ok((BatchStatus::from_db(&row.text_at(0)?).ok_or(corrupt("batch status"))?, row.u64_at(1)?)))
        .collect::<Result<_, StoreError>>()?;
    let count = |v: i64| u64::try_from(v).map_err(|_| corrupt("negative count"));
    Ok(Counters {
        batches_by_status,
        expired_claims: count(
            scalar(
                tx,
                &format!("SELECT {count_all} FROM batch_claims WHERE state = 'active' AND lease_until < $1"),
                args![now],
            )
            .await?,
        )?,
        unaccounted_settlements: count(
            scalar(
                tx,
                &format!("SELECT {count_all} FROM batch_settlements WHERE accounted = 0 AND kind IS NOT NULL"),
                args![],
            )
            .await?,
        )?,
        pending_cleanup: count(
            scalar(tx, &format!("SELECT {count_all} FROM batches WHERE cleanup_state = 'pending'"), args![]).await?,
        )?,
        sessions_open: count(scalar(tx, &format!("SELECT {count_all} FROM artifact_sessions"), args![]).await?)?,
    })
}

pub(in super::super) async fn backlog<T: SqlTx>(tx: &mut T, limit: u32) -> Result<Vec<BacklogEntry>, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}fingerprint, status, {}, CAST(COALESCE(SUM(queued), 0) AS BIGINT) FROM batches \
         WHERE status IN ('validating', 'in_progress') GROUP BY fingerprint, status \
         ORDER BY fingerprint, status{}",
        d.top("$1"),
        d.count_all,
        d.limit("$1")
    );
    let rows = tx.fetch_all(&sql, args![i64::from(limit)]).await?;
    rows.iter()
        .map(|row| {
            Ok(BacklogEntry {
                fingerprint: row
                    .opt_text_at(0)?
                    .map(|v| Fingerprint::parse(&v))
                    .transpose()
                    .map_err(|_| corrupt("fingerprint"))?,
                status: BatchStatus::from_db(&row.text_at(1)?).ok_or(corrupt("batch status"))?,
                batches: row.u64_at(2)?,
                queued: row.u64_at(3)?,
            })
        })
        .collect()
}
