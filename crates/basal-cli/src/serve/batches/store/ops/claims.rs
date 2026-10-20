//! Cost-DRR execution claims, lease renewal and release, solo marking, result sessions and lease recovery.

use std::collections::BTreeMap;
use std::time::Duration;

use super::super::error::StoreError;
use super::super::policy::abandonment;
use super::super::policy::{drr_pick, effective_credit, spend, DrrCandidate, DrrParams};
use super::super::sql::{args, db_now, placeholders, SqlRow, SqlTx, Val};
use super::super::types::ReleaseReport;
use super::super::types::{
    ArtifactRef, BatchRef, BatchStatus, ClaimId, ClaimOutcome, ClaimRequest, ClaimToken, ClaimedItem, CustomId,
    LeaseKind, LeaseTiming, LeaseUpdate, OwnerId, ReleaseReason, Renewal, SessionId, StoreSettings, UnixMillis,
    WorkClaim,
};
use super::common::{
    active_policy, apply_quota, artifact_from_db, batch_id_from_db, corrupt, digest_from_db, end_phase, free_bytes,
    insert_claim, lock_batch, lock_quota, millis, owner_from_db, owner_limits, phase_lease_valid, signed, unsigned,
    BatchRow, QuotaDelta,
};
use super::lifecycle::decide_open_settlement_with_cost;

/// Bounded owner and batch windows; both rotate before truncation.
const MAX_CANDIDATE_OWNERS: usize = 32;
const MAX_BATCHES_PER_OWNER: usize = 8;

/// Upper bound of items in one execution claim.
pub(in super::super) const MAX_CLAIM_ITEMS: u32 = 1024;

/// One attempt of `claim_items`. A changed DRR choice or a quota-blocked owner ends the attempt; the caller starts a
/// new short transaction instead of taking another owner's quota lock under the shard lock.
pub(in super::super) enum ClaimAttempt {
    Done(ClaimOutcome),
    /// Roll back and retry (the advisory choice changed under the locks).
    Retry,
    /// Commit (the owner was marked quota-blocked) and retry without it.
    Blocked(OwnerId),
}

#[derive(Clone, Debug)]
struct BatchCandidate {
    id: String,
    credit: i64,
    last_turn: i64,
    head_cost: u64,
}

#[derive(Clone, Debug)]
struct OwnerCandidate {
    owner: String,
    credit: i64,
    last_turn: i64,
    batch_turn: i64,
    batch_cursor: Option<String>,
    batches: Vec<BatchCandidate>,
}

#[derive(Clone, Debug)]
struct Pick {
    owner: String,
    batch: String,
    turn: i64,
    batch_turn: i64,
    owner_credit: i64,
    batch_credit: i64,
}

struct ShardView {
    turn: i64,
    cursor_owner: Option<String>,
    owners: Vec<OwnerCandidate>,
    /// Earliest end of a quota block among skipped owners.
    blocked_until: Option<i64>,
}

fn rotate_start<T>(items: &[T], key: impl Fn(&T) -> &str, cursor: Option<&str>) -> usize {
    match cursor {
        // Start after the cursor: the first key greater than it, wrapping to 0.
        Some(cursor) => items.iter().position(|item| key(item) > cursor).unwrap_or(0),
        None => 0,
    }
}

/// Owner-first bounded view. Cursor windows rotate before truncation; quota-blocked owners never occupy slots.
/// Deficits are joined by exact owner, so a large historical deficit table cannot erase debt.
async fn shard_view<T: SqlTx>(
    tx: &mut T,
    shard: &str,
    excluded: &[OwnerId],
    now: i64,
) -> Result<Option<ShardView>, StoreError> {
    let Some(schedule) =
        tx.fetch_optional("SELECT turn, cursor_owner FROM batch_schedule WHERE shard = $1", args![shard]).await?
    else {
        return Ok(None);
    };
    let turn = schedule.i64_at(0)?;
    let cursor_owner = schedule.opt_text_at(1)?;
    let d = tx.dialect();
    let mut owner_names = Vec::new();
    // At most two indexed ranges: after the cursor, then wrap. Exclusions are bound values, never SQL text.
    for wrap in [false, true] {
        if owner_names.len() >= MAX_CANDIDATE_OWNERS || (wrap && cursor_owner.is_none()) {
            break;
        }
        let range = if wrap { "b.owner_id <= $3" } else { "($3 IS NULL OR b.owner_id > $3)" };
        let exclusion = if excluded.is_empty() {
            String::new()
        } else {
            format!(" AND b.owner_id NOT IN ({})", placeholders(5, excluded.len()))
        };
        let sql = format!(
            "SELECT DISTINCT {}b.owner_id FROM batches b \
             LEFT JOIN batch_deficits d ON d.shard = $1 AND d.owner_id = b.owner_id \
             LEFT JOIN batch_quota q ON q.owner_id = b.owner_id \
             WHERE b.fingerprint = $1 AND b.status = 'in_progress' AND b.queued > 0 AND b.expires_at > $2 \
             AND {range}{exclusion} \
             AND (d.quota_blocked_version IS NULL OR d.blocked_until IS NULL OR d.blocked_until <= $2 \
                  OR q.version IS NULL OR q.version <> d.quota_blocked_version) \
             AND EXISTS (SELECT 1 FROM batch_items i WHERE i.owner_id = b.owner_id AND i.batch_id = b.id \
                         AND i.state = 'pending' AND i.eligible_at <= $2) \
             ORDER BY b.owner_id{}",
            d.top("$4"),
            d.limit("$4")
        );
        let mut values = args![shard, now, cursor_owner.as_deref(), (MAX_CANDIDATE_OWNERS - owner_names.len()) as i64];
        values.extend(excluded.iter().map(|owner| Val::from(owner.as_str())));
        for row in tx.fetch_all(&sql, values).await? {
            owner_names.push(row.text_at(0)?);
        }
    }
    if owner_names.is_empty() {
        return Ok(Some(ShardView { turn, cursor_owner, owners: Vec::new(), blocked_until: None }));
    }
    let names = placeholders(2, owner_names.len());
    let sql = format!(
        "SELECT owner_id, credit, last_turn, batch_turn, batch_cursor \
                       FROM batch_deficits WHERE shard = $1 AND owner_id IN ({names})"
    );
    let mut values = args![shard];
    values.extend(owner_names.iter().map(Val::from));
    let deficits = tx.fetch_all(&sql, values).await?;
    let mut owners: BTreeMap<String, OwnerCandidate> = owner_names
        .iter()
        .map(|owner| {
            (
                owner.clone(),
                OwnerCandidate {
                    owner: owner.clone(),
                    credit: 0,
                    last_turn: turn,
                    batch_turn: 0,
                    batch_cursor: None,
                    batches: Vec::new(),
                },
            )
        })
        .collect();
    for row in deficits {
        if let Some(owner) = owners.get_mut(&row.text_at(0)?) {
            owner.credit = row.i64_at(1)?;
            owner.last_turn = row.i64_at(2)?;
            owner.batch_turn = row.i64_at(3)?;
            owner.batch_cursor = row.opt_text_at(4)?;
        }
    }
    let names = placeholders(3, owner_names.len());
    // Window ranking is server-side and only for the bounded owner set. The head lookup is evaluated for at
    // most MAX_CANDIDATE_OWNERS * MAX_BATCHES_PER_OWNER returned batches, in this single network request.
    let sql = format!(
        "WITH candidates AS (SELECT b.owner_id, b.id, b.sched_credit, b.sched_turn, \
         ROW_NUMBER() OVER (PARTITION BY b.owner_id ORDER BY \
             CASE WHEN b.id > COALESCE(d.batch_cursor, '') THEN 0 ELSE 1 END, b.id) AS candidate_no \
         FROM batches b LEFT JOIN batch_deficits d ON d.shard = $1 AND d.owner_id = b.owner_id \
         WHERE b.fingerprint = $1 AND b.status = 'in_progress' AND b.queued > 0 AND b.expires_at > $2 \
         AND b.owner_id IN ({names}) \
         AND EXISTS (SELECT 1 FROM batch_items i WHERE i.owner_id = b.owner_id AND i.batch_id = b.id \
                     AND i.state = 'pending' AND i.eligible_at <= $2)) \
         SELECT b.owner_id, b.id, b.sched_credit, b.sched_turn, \
             (SELECT {}i.cost FROM batch_items i WHERE i.owner_id = b.owner_id AND i.batch_id = b.id \
              AND i.state = 'pending' AND i.eligible_at <= $2 ORDER BY i.line_no{}) \
         FROM candidates b WHERE candidate_no <= {MAX_BATCHES_PER_OWNER} ORDER BY b.owner_id, b.id",
        d.top("1"),
        d.limit("1")
    );
    let mut values = args![shard, now];
    values.extend(owner_names.iter().map(Val::from));
    for row in tx.fetch_all(&sql, values).await? {
        let Some(head_cost) = row.opt_i64_at(4)? else { continue };
        if let Some(owner) = owners.get_mut(&row.text_at(0)?) {
            owner.batches.push(BatchCandidate {
                id: row.text_at(1)?,
                credit: row.i64_at(2)?,
                last_turn: row.i64_at(3)?,
                head_cost: unsigned(head_cost)?,
            });
        }
    }
    let owners = owners.into_values().filter(|owner| !owner.batches.is_empty()).collect();
    Ok(Some(ShardView { turn, cursor_owner, owners, blocked_until: None }))
}

/// Two-level DRR: the owner by cost deficit in the shard, then its batch by cost deficit among its batches.
fn pick(view: &ShardView, params: DrrParams) -> Option<Pick> {
    let mut heads = Vec::with_capacity(view.owners.len());
    for owner in &view.owners {
        let batches: Vec<DrrCandidate> = owner
            .batches
            .iter()
            .map(|b| DrrCandidate { credit: b.credit, last_turn: b.last_turn, head_cost: b.head_cost })
            .collect();
        let start = rotate_start(&owner.batches, |b| &b.id, owner.batch_cursor.as_deref());
        let (index, batch_turn) = drr_pick(&batches, start, owner.batch_turn, params)?;
        heads.push((index, batch_turn, batches[index]));
    }
    let owners: Vec<DrrCandidate> = view
        .owners
        .iter()
        .zip(&heads)
        .map(|(o, (_, _, head))| DrrCandidate { credit: o.credit, last_turn: o.last_turn, head_cost: head.head_cost })
        .collect();
    let start = rotate_start(&view.owners, |o| &o.owner, view.cursor_owner.as_deref());
    let (owner_index, turn) = drr_pick(&owners, start, view.turn, params)?;
    let owner = &view.owners[owner_index];
    let (batch_index, batch_turn, batch_head) = heads[owner_index];
    Some(Pick {
        owner: owner.owner.clone(),
        batch: owner.batches[batch_index].id.clone(),
        turn,
        batch_turn,
        owner_credit: effective_credit(&owners[owner_index], turn, params),
        batch_credit: effective_credit(&batch_head, batch_turn, params),
    })
}

struct HeadItem {
    item: ClaimedItem,
    length: u64,
}

async fn pending_items<T: SqlTx>(
    tx: &mut T,
    batch: &BatchRow,
    limit: u32,
    now: i64,
) -> Result<Vec<HeadItem>, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}line_no, custom_id, input_offset, input_length, input_sha256, requires_solo, failure_attempts, \
         cost, result_bound, host_bound FROM batch_items WHERE owner_id = $1 AND batch_id = $2 \
         AND state = 'pending' AND eligible_at <= $4 ORDER BY line_no{}",
        d.top("$3"),
        d.limit("$3")
    );
    let rows = tx
        .fetch_all(&sql, args![batch.record.owner.as_str(), batch.record.id.as_str(), i64::from(limit.max(1)), now])
        .await?;
    rows.iter()
        .map(|row| {
            let length = row.u64_at(3)?;
            Ok(HeadItem {
                item: ClaimedItem {
                    line_no: row.u64_at(0)?,
                    custom_id: CustomId::from_bytes(row.blob_at(1)?).map_err(|_| corrupt("custom_id"))?,
                    offset: row.u64_at(2)?,
                    length,
                    input_sha256: digest_from_db(row.blob_at(4)?)?,
                    requires_solo: row.bool_at(5)?,
                    failure_attempts: u32::try_from(row.i64_at(6)?).map_err(|_| corrupt("failure attempts"))?,
                    cost: row.u64_at(7)?,
                    result_bound: row.u64_at(8)?,
                    host_bound: row.u64_at(9)?,
                },
                length,
            })
        })
        .collect()
}

/// One claim transaction: advisory DRR choice, then quota(owner) → shard → batch → items, re-checking the choice.
pub(in super::super) async fn claim_attempt<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    request: &ClaimRequest,
    excluded: &[OwnerId],
) -> Result<ClaimAttempt, StoreError> {
    let shard = request.target.fingerprint.as_str();
    let params = DrrParams::new(request.shard.quantum, request.shard.max_record_cost);
    let now = db_now(tx).await?;
    let Some(view) = shard_view(tx, shard, excluded, now).await? else {
        return Ok(ClaimAttempt::Done(ClaimOutcome::Idle));
    };
    let Some(advisory) = pick(&view, params) else {
        return Ok(ClaimAttempt::Done(match view.blocked_until {
            Some(until) => ClaimOutcome::QuotaBlocked {
                retry_after: Duration::from_millis(unsigned(until.saturating_sub(now).max(1))?),
            },
            None => ClaimOutcome::Idle,
        }));
    };
    let owner = OwnerId::parse(&advisory.owner).map_err(|_| corrupt("owner id"))?;
    let usage = lock_quota(tx, &owner).await?;
    let d = tx.dialect();
    let sql = format!("SELECT turn FROM batch_schedule{} WHERE shard = $1{}", d.lock_hint, d.for_update);
    tx.fetch_optional(&sql, args![shard]).await?.ok_or(corrupt("shard row missing"))?;
    let now = db_now(tx).await?;
    let Some(view) = shard_view(tx, shard, excluded, now).await? else { return Ok(ClaimAttempt::Retry) };
    let Some(choice) = pick(&view, params) else { return Ok(ClaimAttempt::Retry) };
    if choice.owner != advisory.owner {
        return Ok(ClaimAttempt::Retry);
    }
    let batch_ref = BatchRef { owner: owner.clone(), id: batch_id_from_db(choice.batch.clone())? };
    let batch = lock_batch(tx, &batch_ref).await?;
    let now = db_now(tx).await?;
    if batch.record.status != BatchStatus::InProgress
        || now >= batch.record.expires_at.0
        || batch.record.fingerprint.as_ref() != Some(&request.target.fingerprint)
    {
        return Ok(ClaimAttempt::Retry);
    }
    let policy = active_policy(tx).await?;
    let limits = owner_limits(tx, policy.revision, &owner).await?;
    let mut free = free_bytes(&usage, &limits);
    let max_items = request.max_items.clamp(1, MAX_CLAIM_ITEMS);
    let heads = pending_items(tx, &batch, max_items, now).await?;
    let Some(first) = heads.first() else { return Ok(ClaimAttempt::Retry) };
    let first_reserve = first.item.result_bound.saturating_mul(2);
    if first_reserve > free {
        // Skip this owner without granting credit; it returns after its quota version changes or the backoff.
        let until = now.saturating_add(millis(settings.quota_blocked_backoff));
        let insert = d.insert_if_absent("batch_deficits", &["shard", "owner_id", "last_turn"], &["$1", "$2", "$3"], 2);
        tx.execute(&insert, args![shard, owner.as_str(), view.turn]).await?;
        tx.execute(
            "UPDATE batch_deficits SET quota_blocked_version = $3, blocked_until = $4 WHERE shard = $1 AND owner_id = $2",
            args![shard, owner.as_str(), usage.version, until],
        )
        .await?;
        tx.execute(
            "UPDATE batch_schedule SET cursor_owner = $2, version = version + 1 WHERE shard = $1",
            args![shard, owner.as_str()],
        )
        .await?;
        return Ok(ClaimAttempt::Blocked(owner));
    }
    if first.item.host_bound > request.host_budget_free {
        return Ok(ClaimAttempt::Done(ClaimOutcome::NeedsHostBudget { required: first.item.host_bound }));
    }
    let mut chosen: Vec<&HeadItem> = Vec::new();
    let (mut cost, mut input_bytes, mut host) = (0u64, 0u64, 0u64);
    let credit = u64::try_from(choice.owner_credit.max(0)).unwrap_or(0);
    for head in &heads {
        let reserve = head.item.result_bound.saturating_mul(2);
        if !chosen.is_empty() {
            let solo = head.item.requires_solo || chosen[0].item.requires_solo;
            if solo
                || chosen.len() >= max_items as usize
                || cost.saturating_add(head.item.cost) > credit
                || input_bytes.saturating_add(head.length) > request.max_input_bytes
                || host.saturating_add(head.item.host_bound) > request.host_budget_free
                || reserve > free
            {
                break;
            }
        }
        cost = cost.saturating_add(head.item.cost);
        input_bytes = input_bytes.saturating_add(head.length);
        host = host.saturating_add(head.item.host_bound);
        free = free.saturating_sub(reserve);
        chosen.push(head);
    }
    let claim_id = ClaimId::new_random();
    let lease_until = now.saturating_add(millis(settings.execution_lease));
    insert_claim(
        tx,
        claim_id.as_str(),
        LeaseKind::Execution,
        request.worker.as_str(),
        1,
        &batch_ref,
        batch.record.control_epoch,
        lease_until,
        now,
    )
    .await?;
    let mut reserved = 0i64;
    for head in &chosen {
        let line_no = signed(head.item.line_no)?;
        let reserve = signed(head.item.result_bound.saturating_mul(2))?;
        reserved = reserved.saturating_add(reserve);
        tx.execute(
            "UPDATE batch_items SET state = 'claimed', claim_id = $4, generation = generation + 1 \
             WHERE owner_id = $1 AND batch_id = $2 AND line_no = $3 AND state = 'pending'",
            args![owner.as_str(), batch_ref.id.as_str(), line_no, claim_id.as_str()],
        )
        .await?;
        tx.execute(
            "INSERT INTO batch_settlements (claim_id, line_no, owner_id, batch_id, shard, reserved_bytes, \
             estimated_cost, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            args![
                claim_id.as_str(),
                line_no,
                owner.as_str(),
                batch_ref.id.as_str(),
                shard,
                reserve,
                signed(head.item.cost)?,
                now
            ],
        )
        .await?;
    }
    let count = chosen.len() as i64;
    tx.execute(
        "UPDATE batches SET queued = queued - $3, sched_credit = $4, sched_turn = $5 WHERE owner_id = $1 AND id = $2",
        args![
            owner.as_str(),
            batch_ref.id.as_str(),
            count,
            spend(choice.batch_credit, cost, settings.max_debt),
            choice.batch_turn
        ],
    )
    .await?;
    let sql = d.insert_if_absent("batch_deficits", &["shard", "owner_id", "last_turn"], &["$1", "$2", "$3"], 2);
    tx.execute(&sql, args![shard, owner.as_str(), choice.turn]).await?;
    tx.execute(
        "UPDATE batch_deficits SET credit = $3, last_turn = $4, reserved_cost = reserved_cost + $5, batch_turn = $6, \
         batch_cursor = $7, quota_blocked_version = NULL, blocked_until = NULL WHERE shard = $1 AND owner_id = $2",
        args![
            shard,
            owner.as_str(),
            spend(choice.owner_credit, cost, settings.max_debt),
            choice.turn,
            signed(cost)?,
            choice.batch_turn,
            batch_ref.id.as_str()
        ],
    )
    .await?;
    tx.execute(
        "UPDATE batch_schedule SET turn = $2, cursor_owner = $3, quantum = $4, credit_cap = $5, version = version + 1 \
         WHERE shard = $1",
        args![shard, choice.turn, owner.as_str(), params.quantum, params.cap],
    )
    .await?;
    apply_quota(tx, &owner, QuotaDelta { reserved, ..QuotaDelta::default() }).await?;
    let input = {
        let row = tx
            .fetch_optional(
                "SELECT artifact_ref, bytes, sha256 FROM batch_files WHERE owner_id = $1 AND id = $2",
                args![owner.as_str(), batch.record.input_file_id.as_str()],
            )
            .await?
            .ok_or(corrupt("pinned input file missing"))?;
        super::super::types::InputFile {
            artifact: artifact_from_db(row.text_at(0)?)?,
            bytes: row.u64_at(1)?,
            sha256: digest_from_db(row.blob_at(2)?)?,
        }
    };
    Ok(ClaimAttempt::Done(ClaimOutcome::Claimed(Box::new(WorkClaim {
        token: ClaimToken {
            claim_id,
            worker: request.worker.clone(),
            kind: LeaseKind::Execution,
            batch: batch_ref,
            generation: 1,
            control_epoch: batch.record.control_epoch,
        },
        timing: LeaseTiming {
            server_now: UnixMillis(now),
            lease_until: UnixMillis(lease_until),
            batch_expires_at: batch.record.expires_at,
        },
        input,
        items: chosen.into_iter().map(|h| h.item.clone()).collect(),
    }))))
}

pub(in super::super) struct ClaimRow {
    pub worker: String,
    pub generation: i64,
    pub lease_until: i64,
    pub state: String,
}

pub(in super::super) async fn load_claim<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
) -> Result<Option<ClaimRow>, StoreError> {
    let Some(row) = tx
        .fetch_optional(
            "SELECT worker_id, generation, lease_until, state FROM batch_claims WHERE id = $1 AND owner_id = $2 \
             AND batch_id = $3 AND kind = $4",
            args![token.claim_id.as_str(), token.batch.owner.as_str(), token.batch.id.as_str(), token.kind.as_db()],
        )
        .await?
    else {
        return Ok(None);
    };
    Ok(Some(ClaimRow {
        worker: row.text_at(0)?,
        generation: row.i64_at(1)?,
        lease_until: row.i64_at(2)?,
        state: row.text_at(3)?,
    }))
}

/// Execution claim is active, owned by the token's worker and generation; `now` checks the lease too.
pub(in super::super) fn execution_claim_valid(claim: &Option<ClaimRow>, token: &ClaimToken, now: Option<i64>) -> bool {
    claim.as_ref().is_some_and(|c| {
        c.state == "active"
            && c.worker == token.worker.as_str()
            && c.generation == token.generation
            && now.is_none_or(|now| c.lease_until >= now)
    })
}

fn phase_status_matches(batch: &BatchRow, kind: LeaseKind) -> bool {
    match kind {
        LeaseKind::Discovery | LeaseKind::Validation => batch.record.status == BatchStatus::Validating,
        LeaseKind::Finalization => batch.record.status == BatchStatus::Finalizing,
        LeaseKind::Execution => false,
    }
}

/// Renew any lease or claim. Reports the database time, the new lease end and whether results are still accepted.
pub(in super::super) async fn renew_claim<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    token: &ClaimToken,
) -> Result<LeaseUpdate<Renewal>, StoreError> {
    let batch = lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
    let timing = |lease_until: i64| LeaseTiming {
        server_now: UnixMillis(now),
        lease_until: UnixMillis(lease_until),
        batch_expires_at: batch.record.expires_at,
    };
    if token.kind == LeaseKind::Execution {
        let claim = load_claim(tx, token).await?;
        if !execution_claim_valid(&claim, token, Some(now)) {
            return Ok(LeaseUpdate::Stale);
        }
        let accepting = batch.record.status == BatchStatus::InProgress
            && batch.record.control_epoch == token.control_epoch
            && now < batch.record.expires_at.0;
        let mut lease_until = claim.map_or(now, |c| c.lease_until);
        if accepting {
            lease_until = now.saturating_add(millis(settings.execution_lease));
            tx.execute(
                "UPDATE batch_claims SET lease_until = $2 WHERE id = $1",
                args![token.claim_id.as_str(), lease_until],
            )
            .await?;
            tx.execute(
                "UPDATE artifact_sessions SET lease_until = $2 WHERE claim_id = $1 AND kind = 'result' AND phase = 'open'",
                args![token.claim_id.as_str(), lease_until],
            )
            .await?;
        }
        return Ok(LeaseUpdate::Applied(Renewal {
            timing: timing(lease_until),
            accepting,
            control_epoch: batch.record.control_epoch,
        }));
    }
    if !phase_status_matches(&batch, token.kind) || !phase_lease_valid(&batch, token, token.kind, now) {
        return Ok(LeaseUpdate::Stale);
    }
    let duration = match token.kind {
        LeaseKind::Discovery => settings.discovery_lease,
        LeaseKind::Validation => settings.validation_lease,
        _ => settings.finalization_lease,
    };
    let lease_until = now.saturating_add(millis(duration));
    tx.execute(
        "UPDATE batches SET phase_lease_until = $3 WHERE owner_id = $1 AND id = $2",
        args![token.batch.owner.as_str(), token.batch.id.as_str(), lease_until],
    )
    .await?;
    tx.execute("UPDATE batch_claims SET lease_until = $2 WHERE id = $1", args![token.claim_id.as_str(), lease_until])
        .await?;
    if token.kind == LeaseKind::Finalization {
        tx.execute(
            "UPDATE artifact_sessions SET lease_until = $3 WHERE owner_id = $1 AND batch_id = $2 AND kind = 'export' \
             AND generation = $4",
            args![token.batch.owner.as_str(), token.batch.id.as_str(), lease_until, token.generation],
        )
        .await?;
    }
    Ok(LeaseUpdate::Applied(Renewal {
        timing: timing(lease_until),
        accepting: true,
        control_epoch: batch.record.control_epoch,
    }))
}

/// Return the unfinished items of a claim to the queue (requeue does not count as a failure attempt).
pub(in super::super) async fn release_claim<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
    reason: ReleaseReason,
    costs: &BTreeMap<i64, i64>,
) -> Result<LeaseUpdate<ReleaseReport>, StoreError> {
    let batch = lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
    if token.kind != LeaseKind::Execution {
        if !phase_lease_valid(&batch, token, token.kind, now) {
            return Ok(LeaseUpdate::Stale);
        }
        end_phase(tx, &batch).await?;
        return Ok(LeaseUpdate::Applied(ReleaseReport::default()));
    }
    let claim = load_claim(tx, token).await?;
    if !execution_claim_valid(&claim, token, None) {
        return Ok(LeaseUpdate::Stale);
    }
    let report = return_items(
        tx,
        &batch,
        token.claim_id.as_str(),
        "released",
        now,
        costs,
        !matches!(reason, ReleaseReason::Shutdown | ReleaseReason::HostBudget),
    )
    .await?;
    tx.execute(
        "UPDATE batch_claims SET state = 'released', release_reason = $2 WHERE id = $1",
        args![token.claim_id.as_str(), reason.as_db()],
    )
    .await?;
    Ok(LeaseUpdate::Applied(report))
}

/// Requeue the items still held by a claim and decide their settlement rows with `kind`.
async fn return_items<T: SqlTx>(
    tx: &mut T,
    batch: &BatchRow,
    claim_id: &str,
    kind: &str,
    now: i64,
    costs: &BTreeMap<i64, i64>,
    defer: bool,
) -> Result<ReleaseReport, StoreError> {
    let owner = batch.record.owner.as_str();
    let id = batch.record.id.as_str();
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}line_no, abandon_attempts FROM batch_items WHERE claim_id = $1 AND owner_id = $2 AND batch_id = $3 \
         AND state = 'claimed' ORDER BY line_no{}",
        d.top("4096"),
        d.limit("4096")
    );
    let lines = tx.fetch_all(&sql, args![claim_id, owner, id]).await?;
    let mut report = ReleaseReport::default();
    for line in &lines {
        let line_no = line.i64_at(0)?;
        decide_open_settlement_with_cost(tx, claim_id, line_no, kind, batch, now, costs.get(&line_no).copied()).await?;
        let (attempts, eligible) = abandonment(line.i64_at(1)?, defer, now);
        report.requeued += 1;
        report.estimated += u32::from(!costs.contains_key(&line_no));
        if defer {
            report.next_eligible_at =
                Some(UnixMillis(report.next_eligible_at.map_or(eligible, |at| at.0.min(eligible))));
        }
        tx.execute(
            "UPDATE batch_items SET state = 'pending', claim_id = NULL, abandon_attempts = $4, eligible_at = $5 WHERE owner_id = $1 AND batch_id = $2 \
             AND line_no = $3",
            args![owner, id, line_no, attempts, eligible],
        )
        .await?;
    }
    let count = lines.len() as i64;
    tx.execute("UPDATE batches SET queued = queued + $3 WHERE owner_id = $1 AND id = $2", args![owner, id, count])
        .await?;
    Ok(report)
}

/// Persist `requires_solo` for items of a failed group so they never join a group again, also after restart.
pub(in super::super) async fn mark_solo<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
    lines: &[u64],
) -> Result<LeaseUpdate<()>, StoreError> {
    if lines.is_empty() || lines.len() > 4096 {
        return Err(StoreError::Invalid(super::super::error::InvalidValue::new("lines", "must list 1..4096 lines")));
    }
    lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
    let claim = load_claim(tx, token).await?;
    if token.kind != LeaseKind::Execution || !execution_claim_valid(&claim, token, Some(now)) {
        return Ok(LeaseUpdate::Stale);
    }
    let sql = format!(
        "UPDATE batch_items SET requires_solo = 1 WHERE owner_id = $1 AND batch_id = $2 AND claim_id = $3 \
         AND line_no IN ({})",
        placeholders(4, lines.len())
    );
    let mut values: Vec<Val> = args![token.batch.owner.as_str(), token.batch.id.as_str(), token.claim_id.as_str()];
    for line in lines {
        values.push(Val::from(signed(*line)?));
    }
    tx.execute(&sql, values).await?;
    Ok(LeaseUpdate::Applied(()))
}

/// Open a result session before writing a large result object, so GC protects it while the claim lives.
/// The session carries no own reservation: the claim's settlement row covers the bytes.
pub(in super::super) async fn begin_result_upload<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
    line_no: u64,
    artifact: &ArtifactRef,
) -> Result<SessionId, StoreError> {
    let batch = lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
    let claim = load_claim(tx, token).await?;
    if token.kind != LeaseKind::Execution
        || !execution_claim_valid(&claim, token, Some(now))
        || batch.record.control_epoch != token.control_epoch
        || batch.record.status != BatchStatus::InProgress
    {
        return Err(StoreError::LeaseStale);
    }
    let line = signed(line_no)?;
    let held = tx
        .fetch_optional(
            "SELECT 1 FROM batch_items WHERE owner_id = $1 AND batch_id = $2 AND line_no = $3 AND claim_id = $4 \
             AND state = 'claimed'",
            args![token.batch.owner.as_str(), token.batch.id.as_str(), line, token.claim_id.as_str()],
        )
        .await?;
    if held.is_none() {
        return Err(StoreError::LeaseStale);
    }
    let session = SessionId::new_random();
    let lease_until = claim.map_or(now, |c| c.lease_until);
    tx.execute(
        "INSERT INTO artifact_sessions (id, owner_id, kind, artifact_ref, phase, lease_until, batch_id, claim_id, \
         line_no, created_at) VALUES ($1, $2, 'result', $3, 'open', $4, $5, $6, $7, $8)",
        args![
            session.as_str(),
            token.batch.owner.as_str(),
            artifact.as_str(),
            lease_until,
            token.batch.id.as_str(),
            token.claim_id.as_str(),
            line,
            now
        ],
    )
    .await?;
    Ok(session)
}

/// Execution claims whose lease ended (bounded).
pub(in super::super) async fn expired_claims<T: SqlTx>(
    tx: &mut T,
    limit: u32,
) -> Result<Vec<(String, BatchRef)>, StoreError> {
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}id, owner_id, batch_id FROM batch_claims WHERE kind = 'execution' AND state = 'active' \
         AND lease_until < $1 ORDER BY lease_until{}",
        d.top("$2"),
        d.limit("$2")
    );
    let rows = tx.fetch_all(&sql, args![now, i64::from(limit)]).await?;
    rows.iter()
        .map(|row| {
            Ok((
                row.text_at(0)?,
                BatchRef { owner: owner_from_db(row.text_at(1)?)?, id: batch_id_from_db(row.text_at(2)?)? },
            ))
        })
        .collect()
}

/// Requeue the items of one expired claim (re-checked under batch control). Healthy claims are never touched.
pub(in super::super) async fn recover_claim<T: SqlTx>(
    tx: &mut T,
    claim_id: &str,
    batch: &BatchRef,
) -> Result<bool, StoreError> {
    let row = lock_batch(tx, batch).await?;
    let now = db_now(tx).await?;
    let changed = tx
        .execute(
            "UPDATE batch_claims SET state = 'expired' WHERE id = $1 AND state = 'active' AND lease_until < $2",
            args![claim_id, now],
        )
        .await?;
    if changed == 0 {
        return Ok(false);
    }
    return_items(tx, &row, claim_id, "expired", now, &BTreeMap::new(), true).await?;
    Ok(true)
}
