//! Cost-DRR execution claims, renewal, release, solo marking, result sessions and lease recovery (MongoDB).
//!
//! The DRR state is the same as in the SQL adapters: `batch_schedule` (turn, cursor, quantum, cap) per shard,
//! `batch_deficits` per (shard, owner) and `sched_credit/sched_turn` on each batch. The decision uses the shared
//! `policy::drr_pick`; the shard document is write-locked after the owner's quota document.

use std::collections::BTreeMap;
use std::time::Duration;

use mongodb::bson::{doc, Bson, Document};

use super::super::error::{InvalidValue, StoreError};
use super::super::ops::claims::{execution_claim_valid, ClaimAttempt, ClaimRow, MAX_CLAIM_ITEMS};
use super::super::ops::common::{free_bytes, millis, phase_lease_valid, signed, unsigned, BatchRow, QuotaDelta};
use super::super::policy::abandonment;
use super::super::policy::{drr_pick, effective_credit, spend, DrrCandidate, DrrParams};
use super::super::types::ReleaseReport;
use super::super::types::{
    ArtifactRef, BatchRef, BatchStatus, ClaimId, ClaimOutcome, ClaimRequest, ClaimToken, ClaimedItem, CustomId,
    LeaseKind, LeaseTiming, LeaseUpdate, OwnerId, ReleaseReason, Renewal, SessionId, UnixMillis, WorkClaim,
};
use super::common::{
    active_policy, apply_quota, end_phase, insert_claim, lock_batch, lock_quota, owner_limits, set_batch,
};
use super::convert::{batch_id, digest, owner};
use super::driver::{
    bin, corrupt, flag, int, mtx, opt_int, opt_text, text, uint, Core, Tx, BATCHES, CLAIMS, DEFICITS, ITEMS, QUOTA,
    SCHEDULE, SESSIONS, SETTLEMENTS,
};
use super::ingest::input_file;
use super::lifecycle::{decide_open_settlement_with_cost, items_of};

/// Owner and batch windows rotate before truncation.
const MAX_CANDIDATE_OWNERS: usize = 32;
const MAX_BATCHES_PER_OWNER: usize = 8;
/// Attempts of one `claim_items` call.
const MAX_CLAIM_ATTEMPTS: usize = 8;

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
    blocked_until: Option<i64>,
}

fn rotate_start<T>(items: &[T], key: impl Fn(&T) -> &str, cursor: Option<&str>) -> usize {
    match cursor {
        Some(cursor) => items.iter().position(|item| key(item) > cursor).unwrap_or(0),
        None => 0,
    }
}

/// Owner-first cursor windows and server-side bounded batch/head lookups. There are at most two aggregation
/// requests per view, rather than one network request for each batch head.
async fn shard_view(
    tx: &mut Tx<'_>,
    shard: &str,
    excluded: &[OwnerId],
    now: i64,
) -> Result<Option<ShardView>, StoreError> {
    let Some(schedule) = tx.one(SCHEDULE, doc! { "_id": shard }).await? else { return Ok(None) };
    let turn = int(&schedule, "turn")?;
    let cursor_owner = opt_text(&schedule, "cursor_owner")?;
    let mut owners = BTreeMap::new();
    for wrap in [false, true] {
        if owners.len() >= MAX_CANDIDATE_OWNERS || (wrap && cursor_owner.is_none()) {
            break;
        }
        let mut owner_filter =
            doc! { "$nin": excluded.iter().map(|owner| Bson::String(owner.as_str().to_owned())).collect::<Vec<_>>() };
        if let Some(cursor) = &cursor_owner {
            owner_filter.insert(if wrap { "$lte" } else { "$gt" }, cursor.as_str());
        }
        let eligible = doc! { "$or": [ { "eligible_at": { "$lte": now } }, { "eligible_at": { "$exists": false } } ] };
        let pipeline = vec![
            doc! { "$match": { "fingerprint": shard, "status": "in_progress", "queued": { "$gt": 0_i64 },
            "expires_at": { "$gt": now }, "owner_id": owner_filter } },
            doc! { "$group": { "_id": "$owner_id" } },
            doc! { "$sort": { "_id": 1_i32 } },
            doc! { "$lookup": { "from": DEFICITS, "let": { "owner": "$_id" }, "pipeline": [
                { "$match": { "$expr": { "$and": [ { "$eq": ["$owner_id", "$$owner"] },
                    { "$eq": ["$shard", { "$literal": shard }] } ] } } }, { "$limit": 1_i64 }
            ], "as": "deficit" } },
            doc! { "$unwind": { "path": "$deficit", "preserveNullAndEmptyArrays": true } },
            doc! { "$lookup": { "from": QUOTA, "localField": "_id", "foreignField": "_id", "as": "quota" } },
            doc! { "$match": { "$expr": { "$not": [ { "$and": [
                { "$gt": [ { "$ifNull": ["$deficit.blocked_until", 0_i64] }, now ] },
                { "$ne": [ { "$ifNull": ["$deficit.quota_blocked_version", -1_i64] }, -1_i64 ] },
                { "$eq": ["$deficit.quota_blocked_version", { "$arrayElemAt": ["$quota.version", 0_i32] }] }
            ] } ] } } },
            doc! { "$lookup": { "from": BATCHES, "let": { "owner": "$_id",
                "cursor": { "$ifNull": ["$deficit.batch_cursor", ""] } }, "pipeline": [
                { "$match": { "fingerprint": shard, "status": "in_progress", "queued": { "$gt": 0_i64 },
                    "expires_at": { "$gt": now }, "$expr": { "$eq": ["$owner_id", "$$owner"] } } },
                { "$project": { "_id": 1_i32, "owner_id": 1_i32, "sched_credit": 1_i32, "sched_turn": 1_i32 } },
                { "$lookup": { "from": ITEMS, "let": { "batch": "$_id", "owner": "$owner_id" }, "pipeline": [
                    { "$match": { "state": "pending", "$or": eligible.get_array("$or").map_err(|_| corrupt("eligible filter"))?.clone(),
                        "$expr": { "$and": [ { "$eq": ["$batch_id", "$$batch"] }, { "$eq": ["$owner_id", "$$owner"] } ] } } },
                    { "$sort": { "line_no": 1_i32 } }, { "$limit": 1_i64 }, { "$project": { "cost": 1_i32 } }
                ], "as": "head" } },
                { "$unwind": "$head" },
                { "$addFields": { "cursor_rank": { "$cond": [ { "$gt": ["$_id", "$$cursor"] }, 0_i32, 1_i32 ] } } },
                { "$sort": { "cursor_rank": 1_i32, "_id": 1_i32 } },
                { "$limit": MAX_BATCHES_PER_OWNER as i64 }
            ], "as": "batches" } },
            doc! { "$match": { "batches.0": { "$exists": true } } },
            doc! { "$project": { "_id": 1_i32, "deficit": 1_i32, "batches": 1_i32 } },
            doc! { "$limit": (MAX_CANDIDATE_OWNERS - owners.len()) as i64 },
        ];
        for row in tx.aggregate(BATCHES, pipeline).await? {
            let owner = text(&row, "_id")?;
            let deficit = row.get_document("deficit").ok();
            let mut candidate = OwnerCandidate {
                owner: owner.clone(),
                credit: deficit.map(|d| int(d, "credit")).transpose()?.unwrap_or(0),
                last_turn: deficit.map(|d| int(d, "last_turn")).transpose()?.unwrap_or(turn),
                batch_turn: deficit.map(|d| int(d, "batch_turn")).transpose()?.unwrap_or(0),
                batch_cursor: deficit.map(|d| opt_text(d, "batch_cursor")).transpose()?.flatten(),
                batches: Vec::new(),
            };
            for batch in row.get_array("batches").map_err(|_| corrupt("candidate batches"))? {
                let batch = batch.as_document().ok_or(corrupt("candidate batch"))?;
                let head = batch.get_document("head").map_err(|_| corrupt("candidate head"))?;
                candidate.batches.push(BatchCandidate {
                    id: text(batch, "_id")?,
                    credit: int(batch, "sched_credit")?,
                    last_turn: int(batch, "sched_turn")?,
                    head_cost: uint(head, "cost")?,
                });
            }
            candidate.batches.sort_by(|a, b| a.id.cmp(&b.id));
            owners.insert(owner, candidate);
        }
    }
    Ok(Some(ShardView { turn, cursor_owner, owners: owners.into_values().collect(), blocked_until: None }))
}

/// Two-level DRR: owner by cost deficit in the shard, then its batch by cost deficit (same rule as SQL).
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

fn claimed_item(doc: &Document) -> Result<ClaimedItem, StoreError> {
    Ok(ClaimedItem {
        line_no: uint(doc, "line_no")?,
        custom_id: CustomId::from_bytes(bin(doc, "custom_id")?).map_err(|_| corrupt("custom_id"))?,
        offset: uint(doc, "input_offset")?,
        length: uint(doc, "input_length")?,
        input_sha256: digest(bin(doc, "input_sha256")?)?,
        requires_solo: flag(doc, "requires_solo")?,
        failure_attempts: u32::try_from(int(doc, "failure_attempts")?).map_err(|_| corrupt("failure attempts"))?,
        cost: uint(doc, "cost")?,
        result_bound: uint(doc, "result_bound")?,
        host_bound: uint(doc, "host_bound")?,
    })
}

async fn claim_attempt_tx(
    tx: &mut Tx<'_>,
    request: &ClaimRequest,
    excluded: &[OwnerId],
    now: i64,
) -> Result<ClaimAttempt, StoreError> {
    let settings = *tx.settings();
    let shard = request.target.fingerprint.as_str();
    let params = DrrParams::new(request.shard.quantum, request.shard.max_record_cost);
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
    let claim_owner = OwnerId::parse(&advisory.owner).map_err(|_| corrupt("owner id"))?;
    let (usage, _) = lock_quota(tx, &claim_owner).await?;
    tx.lock(SCHEDULE, doc! { "_id": shard }).await?.ok_or(corrupt("shard document missing"))?;
    // The snapshot is unchanged under the locks (a concurrent change write-conflicts), so the choice is re-checked.
    let Some(view) = shard_view(tx, shard, excluded, now).await? else { return Ok(ClaimAttempt::Retry) };
    let Some(choice) = pick(&view, params) else { return Ok(ClaimAttempt::Retry) };
    if choice.owner != advisory.owner {
        return Ok(ClaimAttempt::Retry);
    }
    let key = BatchRef { owner: claim_owner.clone(), id: batch_id(choice.batch.clone())? };
    let (batch, now) = lock_batch(tx, &key).await?;
    if batch.record.status != BatchStatus::InProgress
        || now >= batch.record.expires_at.0
        || batch.record.fingerprint.as_ref() != Some(&request.target.fingerprint)
    {
        return Ok(ClaimAttempt::Retry);
    }
    let policy = active_policy(tx).await?;
    let limits = owner_limits(tx, policy.revision, &claim_owner).await?;
    let mut free = free_bytes(&usage, &limits);
    let max_items = request.max_items.clamp(1, MAX_CLAIM_ITEMS);
    let mut pending = items_of(&key);
    pending.insert("state", "pending");
    pending.insert("$or", vec![doc! { "eligible_at": { "$lte": now } }, doc! { "eligible_at": { "$exists": false } }]);
    let heads = tx
        .many(ITEMS, pending, doc! { "line_no": 1_i32 }, i64::from(max_items))
        .await?
        .iter()
        .map(claimed_item)
        .collect::<Result<Vec<_>, _>>()?;
    let Some(first) = heads.first() else { return Ok(ClaimAttempt::Retry) };
    if first.result_bound.saturating_mul(2) > free {
        let until = now.saturating_add(millis(settings.quota_blocked_backoff));
        let key = doc! { "shard": shard, "owner_id": claim_owner.as_str() };
        if tx
            .update(
                DEFICITS,
                key.clone(),
                doc! { "$set": { "quota_blocked_version": usage.version, "blocked_until": until } },
            )
            .await?
            == 0
        {
            let mut fresh = key;
            fresh.extend(doc! { "credit": 0_i64, "last_turn": view.turn, "reserved_cost": 0_i64,
            "batch_turn": 0_i64, "batch_cursor": Bson::Null,
            "quota_blocked_version": usage.version, "blocked_until": until });
            tx.insert(DEFICITS, fresh).await?;
        }
        tx.update(
            SCHEDULE,
            doc! { "_id": shard },
            doc! { "$set": { "cursor_owner": claim_owner.as_str() }, "$inc": { "version": 1_i64 } },
        )
        .await?;
        return Ok(ClaimAttempt::Blocked(claim_owner));
    }
    if first.host_bound > request.host_budget_free {
        return Ok(ClaimAttempt::Done(ClaimOutcome::NeedsHostBudget { required: first.host_bound }));
    }
    let mut chosen: Vec<&ClaimedItem> = Vec::new();
    let (mut cost, mut input_bytes, mut host) = (0u64, 0u64, 0u64);
    let credit = u64::try_from(choice.owner_credit.max(0)).unwrap_or(0);
    for item in &heads {
        let reserve = item.result_bound.saturating_mul(2);
        if !chosen.is_empty() {
            let solo = item.requires_solo || chosen[0].requires_solo;
            if solo
                || chosen.len() >= max_items as usize
                || cost.saturating_add(item.cost) > credit
                || input_bytes.saturating_add(item.length) > request.max_input_bytes
                || host.saturating_add(item.host_bound) > request.host_budget_free
                || reserve > free
            {
                break;
            }
        }
        cost = cost.saturating_add(item.cost);
        input_bytes = input_bytes.saturating_add(item.length);
        host = host.saturating_add(item.host_bound);
        free = free.saturating_sub(reserve);
        chosen.push(item);
    }
    let claim_id = ClaimId::new_random();
    let lease_until = now.saturating_add(millis(settings.execution_lease));
    insert_claim(
        tx,
        claim_id.as_str(),
        LeaseKind::Execution,
        request.worker.as_str(),
        1,
        &key,
        batch.record.control_epoch,
        lease_until,
        now,
    )
    .await?;
    let mut reserved = 0i64;
    for item in &chosen {
        let line_no = signed(item.line_no)?;
        let reserve = signed(item.result_bound.saturating_mul(2))?;
        reserved = reserved.saturating_add(reserve);
        let mut item_key = items_of(&key);
        item_key.insert("line_no", line_no);
        item_key.insert("state", "pending");
        tx.update(
            ITEMS,
            item_key,
            doc! { "$set": { "state": "claimed", "claim_id": claim_id.as_str() }, "$inc": { "generation": 1_i64 } },
        )
        .await?;
        tx.insert(
            SETTLEMENTS,
            doc! {
                "claim_id": claim_id.as_str(),
                "line_no": line_no,
                "owner_id": claim_owner.as_str(),
                "batch_id": key.id.as_str(),
                "shard": shard,
                "reserved_bytes": reserve,
                "estimated_cost": signed(item.cost)?,
                "kind": Bson::Null,
                "operation_id": Bson::Null,
                "accepted_bytes": 0_i64,
                "orphan_session_id": Bson::Null,
                "actual_cost": 0_i64,
                "accounted": false,
                "decided_at": Bson::Null,
                "created_at": now,
            },
        )
        .await?;
    }
    set_batch(
        tx,
        &key,
        doc! {
            "$set": { "sched_credit": spend(choice.batch_credit, cost, settings.max_debt), "sched_turn": choice.batch_turn },
            "$inc": { "queued": -(chosen.len() as i64) },
        },
    )
    .await?;
    let deficit = doc! {
        "credit": spend(choice.owner_credit, cost, settings.max_debt),
        "last_turn": choice.turn,
        "batch_turn": choice.batch_turn,
        "batch_cursor": key.id.as_str(),
        "quota_blocked_version": Bson::Null,
        "blocked_until": Bson::Null,
    };
    let deficit_key = doc! { "shard": shard, "owner_id": claim_owner.as_str() };
    if tx
        .update(
            DEFICITS,
            deficit_key.clone(),
            doc! { "$set": deficit.clone(), "$inc": { "reserved_cost": signed(cost)? } },
        )
        .await?
        == 0
    {
        let mut fresh = deficit_key;
        fresh.extend(deficit);
        fresh.insert("reserved_cost", signed(cost)?);
        tx.insert(DEFICITS, fresh).await?;
    }
    tx.update(
        SCHEDULE,
        doc! { "_id": shard },
        doc! {
            "$set": { "turn": choice.turn, "cursor_owner": claim_owner.as_str(), "quantum": params.quantum, "credit_cap": params.cap },
            "$inc": { "version": 1_i64 },
        },
    )
    .await?;
    apply_quota(tx, &claim_owner, QuotaDelta { reserved, ..QuotaDelta::default() }).await?;
    let input = input_file(tx, &batch).await?;
    Ok(ClaimAttempt::Done(ClaimOutcome::Claimed(Box::new(WorkClaim {
        token: ClaimToken {
            claim_id,
            worker: request.worker.clone(),
            kind: LeaseKind::Execution,
            batch: key,
            generation: 1,
            control_epoch: batch.record.control_epoch,
        },
        timing: LeaseTiming {
            server_now: UnixMillis(now),
            lease_until: UnixMillis(lease_until),
            batch_expires_at: batch.record.expires_at,
        },
        input,
        items: chosen.into_iter().cloned().collect(),
    }))))
}

/// Claim the next microseries. Each attempt is one short transaction; quota-blocked owners are recorded and skipped
/// in a new transaction.
pub(super) async fn claim_items(core: &Core, request: &ClaimRequest) -> Result<ClaimOutcome, StoreError> {
    let mut excluded: Vec<OwnerId> = Vec::new();
    for _ in 0..MAX_CLAIM_ATTEMPTS {
        let now = core.server_now().await?;
        let outcome = mtx!(core, tx => claim_attempt_tx(tx, request, &excluded, now));
        match outcome {
            Ok(ClaimAttempt::Done(ClaimOutcome::Idle)) if !excluded.is_empty() => {
                return Ok(ClaimOutcome::QuotaBlocked { retry_after: core.settings.quota_blocked_backoff });
            }
            Ok(ClaimAttempt::Done(result)) => return Ok(result),
            Ok(ClaimAttempt::Blocked(blocked)) => excluded.push(blocked),
            Ok(ClaimAttempt::Retry) => {}
            Err(error) if error.is_retryable() => {}
            Err(error) => return Err(error),
        }
    }
    Ok(ClaimOutcome::QuotaBlocked { retry_after: core.settings.quota_blocked_backoff })
}

pub(super) async fn load_claim(tx: &mut Tx<'_>, token: &ClaimToken) -> Result<Option<ClaimRow>, StoreError> {
    let Some(doc) = tx
        .one(
            CLAIMS,
            doc! {
                "_id": token.claim_id.as_str(),
                "owner_id": token.batch.owner.as_str(),
                "batch_id": token.batch.id.as_str(),
                "kind": token.kind.as_db(),
            },
        )
        .await?
    else {
        return Ok(None);
    };
    Ok(Some(ClaimRow {
        worker: text(&doc, "worker_id")?,
        generation: int(&doc, "generation")?,
        lease_until: int(&doc, "lease_until")?,
        state: text(&doc, "state")?,
    }))
}

fn phase_status_matches(batch: &BatchRow, kind: LeaseKind) -> bool {
    match kind {
        LeaseKind::Discovery | LeaseKind::Validation => batch.record.status == BatchStatus::Validating,
        LeaseKind::Finalization => batch.record.status == BatchStatus::Finalizing,
        LeaseKind::Execution => false,
    }
}

async fn renew_claim_tx(tx: &mut Tx<'_>, token: &ClaimToken) -> Result<LeaseUpdate<Renewal>, StoreError> {
    let settings = *tx.settings();
    let (batch, now) = lock_batch(tx, &token.batch).await?;
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
            tx.update(CLAIMS, doc! { "_id": token.claim_id.as_str() }, doc! { "$set": { "lease_until": lease_until } })
                .await?;
            tx.update_many(
                SESSIONS,
                doc! { "claim_id": token.claim_id.as_str(), "kind": "result", "phase": "open" },
                doc! { "$set": { "lease_until": lease_until } },
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
    set_batch(tx, &token.batch, doc! { "$set": { "phase_lease_until": lease_until } }).await?;
    tx.update(CLAIMS, doc! { "_id": token.claim_id.as_str() }, doc! { "$set": { "lease_until": lease_until } }).await?;
    if token.kind == LeaseKind::Finalization {
        tx.update_many(
            SESSIONS,
            doc! {
                "owner_id": token.batch.owner.as_str(),
                "batch_id": token.batch.id.as_str(),
                "kind": "export",
                "generation": token.generation,
            },
            doc! { "$set": { "lease_until": lease_until } },
        )
        .await?;
    }
    Ok(LeaseUpdate::Applied(Renewal {
        timing: timing(lease_until),
        accepting: true,
        control_epoch: batch.record.control_epoch,
    }))
}

pub(super) async fn renew_claim(core: &Core, token: &ClaimToken) -> Result<LeaseUpdate<Renewal>, StoreError> {
    mtx!(core, tx => renew_claim_tx(tx, token))
}

/// Requeue the items still held by a claim and decide their settlements with `kind`.
async fn return_items(
    tx: &mut Tx<'_>,
    batch: &BatchRow,
    claim_id: &str,
    kind: &str,
    now: i64,
    costs: &BTreeMap<i64, i64>,
    defer: bool,
) -> Result<ReleaseReport, StoreError> {
    let key = batch.batch_ref();
    let mut held = items_of(&key);
    held.insert("claim_id", claim_id);
    held.insert("state", "claimed");
    let lines = tx.many(ITEMS, held, doc! { "line_no": 1_i32 }, i64::from(MAX_CLAIM_ITEMS)).await?;
    let mut report = ReleaseReport::default();
    for line in &lines {
        let line_no = int(line, "line_no")?;
        decide_open_settlement_with_cost(tx, claim_id, line_no, kind, batch, now, costs.get(&line_no).copied()).await?;
        let (attempts, eligible) = abandonment(opt_int(line, "abandon_attempts")?.unwrap_or(0), defer, now);
        report.requeued += 1;
        report.estimated += u32::from(!costs.contains_key(&line_no));
        if defer {
            report.next_eligible_at =
                Some(UnixMillis(report.next_eligible_at.map_or(eligible, |at| at.0.min(eligible))));
        }
        let mut item_key = items_of(&key);
        item_key.insert("line_no", line_no);
        tx.update(ITEMS, item_key, doc! { "$set": { "state": "pending", "claim_id": Bson::Null, "abandon_attempts": attempts, "eligible_at": eligible } }).await?;
    }
    let count = lines.len() as i64;
    set_batch(tx, &key, doc! { "$inc": { "queued": count } }).await?;
    Ok(report)
}

async fn release_claim_tx(
    tx: &mut Tx<'_>,
    token: &ClaimToken,
    reason: ReleaseReason,
    costs: &BTreeMap<i64, i64>,
) -> Result<LeaseUpdate<ReleaseReport>, StoreError> {
    let (batch, now) = lock_batch(tx, &token.batch).await?;
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
    tx.update(
        CLAIMS,
        doc! { "_id": token.claim_id.as_str() },
        doc! { "$set": { "state": "released", "release_reason": reason.as_db() } },
    )
    .await?;
    Ok(LeaseUpdate::Applied(report))
}

pub(super) async fn release_claim(
    core: &Core,
    token: &ClaimToken,
    reason: ReleaseReason,
    costs: &BTreeMap<i64, i64>,
) -> Result<LeaseUpdate<ReleaseReport>, StoreError> {
    mtx!(core, tx => release_claim_tx(tx, token, reason, costs))
}

async fn mark_solo_tx(tx: &mut Tx<'_>, token: &ClaimToken, lines: &[u64]) -> Result<LeaseUpdate<()>, StoreError> {
    let (_, now) = lock_batch(tx, &token.batch).await?;
    let claim = load_claim(tx, token).await?;
    if token.kind != LeaseKind::Execution || !execution_claim_valid(&claim, token, Some(now)) {
        return Ok(LeaseUpdate::Stale);
    }
    let lines: Vec<i64> = lines.iter().map(|l| signed(*l)).collect::<Result<_, _>>()?;
    let mut filter = items_of(&token.batch);
    filter.insert("claim_id", token.claim_id.as_str());
    filter.insert("line_no", doc! { "$in": lines });
    tx.update_many(ITEMS, filter, doc! { "$set": { "requires_solo": true } }).await?;
    Ok(LeaseUpdate::Applied(()))
}

pub(super) async fn mark_solo(core: &Core, token: &ClaimToken, lines: &[u64]) -> Result<LeaseUpdate<()>, StoreError> {
    if lines.is_empty() || lines.len() > 4096 {
        return Err(StoreError::Invalid(InvalidValue::new("lines", "must list 1..4096 lines")));
    }
    mtx!(core, tx => mark_solo_tx(tx, token, lines))
}

async fn begin_result_upload_tx(
    tx: &mut Tx<'_>,
    token: &ClaimToken,
    line_no: u64,
    artifact: &ArtifactRef,
) -> Result<SessionId, StoreError> {
    let (batch, now) = lock_batch(tx, &token.batch).await?;
    let claim = load_claim(tx, token).await?;
    if token.kind != LeaseKind::Execution
        || !execution_claim_valid(&claim, token, Some(now))
        || batch.record.control_epoch != token.control_epoch
        || batch.record.status != BatchStatus::InProgress
    {
        return Err(StoreError::LeaseStale);
    }
    let line = signed(line_no)?;
    let mut held = items_of(&token.batch);
    held.insert("line_no", line);
    held.insert("claim_id", token.claim_id.as_str());
    held.insert("state", "claimed");
    if tx.one(ITEMS, held).await?.is_none() {
        return Err(StoreError::LeaseStale);
    }
    let session = SessionId::new_random();
    tx.insert(
        SESSIONS,
        doc! {
            "_id": session.as_str(),
            "owner_id": token.batch.owner.as_str(),
            "kind": "result",
            "artifact_ref": artifact.as_str(),
            "secondary_ref": Bson::Null,
            "generation": 0_i64,
            "phase": "open",
            "lease_until": claim.map_or(now, |c| c.lease_until),
            "reserved_bytes": 0_i64,
            "max_bytes": 0_i64,
            "file_id": Bson::Null,
            "batch_id": token.batch.id.as_str(),
            "claim_id": token.claim_id.as_str(),
            "line_no": line,
            "filename": Bson::Null,
            "expires_after_secs": Bson::Null,
            "created_at": now,
        },
    )
    .await?;
    Ok(session)
}

pub(super) async fn begin_result_upload(
    core: &Core,
    token: &ClaimToken,
    line_no: u64,
    artifact: &ArtifactRef,
) -> Result<SessionId, StoreError> {
    mtx!(core, tx => begin_result_upload_tx(tx, token, line_no, artifact))
}

async fn recover_claim_tx(tx: &mut Tx<'_>, claim_id: &str, batch: &BatchRef) -> Result<bool, StoreError> {
    let (row, now) = lock_batch(tx, batch).await?;
    let changed = tx
        .update(
            CLAIMS,
            doc! { "_id": claim_id, "state": "active", "lease_until": { "$lt": now } },
            doc! { "$set": { "state": "expired" } },
        )
        .await?;
    if changed == 0 {
        return Ok(false);
    }
    return_items(tx, &row, claim_id, "expired", now, &BTreeMap::new(), true).await?;
    Ok(true)
}

/// Requeue items of at most `limit` expired execution claims, one transaction per claim.
pub(super) async fn recover_expired_claims(core: &Core, limit: u32) -> Result<u64, StoreError> {
    let now = core.server_now().await?;
    let expired = core
        .read_many(
            CLAIMS,
            doc! { "kind": "execution", "state": "active", "lease_until": { "$lt": now } },
            doc! { "lease_until": 1_i32 },
            i64::from(limit),
        )
        .await?;
    let mut recovered = 0;
    for claim in &expired {
        let claim_id = text(claim, "_id")?;
        let key = BatchRef { owner: owner(text(claim, "owner_id")?)?, id: batch_id(text(claim, "batch_id")?)? };
        if mtx!(core, tx => recover_claim_tx(tx, &claim_id, &key))? {
            recovered += 1;
        }
    }
    Ok(recovered)
}
