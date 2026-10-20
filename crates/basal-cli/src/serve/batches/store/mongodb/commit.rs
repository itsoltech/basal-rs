//! Result commit with the operation ledger and exactly-once settlement (MongoDB).

use std::collections::BTreeMap;

use sha2::Digest;

use mongodb::bson::{doc, Bson};

use super::super::error::StoreError;
use super::super::ops::claims::execution_claim_valid;
use super::super::ops::common::{signed, unsigned, QuotaDelta};
use super::super::policy::settle_credit;
use super::super::types::{
    AccountingProgress, BatchRef, BatchStatus, ClaimToken, CommitReport, ItemCommitOutcome, ItemResult, LeaseKind,
    OperationId, OwnerId, RejectReason, ResultDisposition, ResultPayload, Sha256Digest,
};
use super::claims::load_claim;
use super::common::{apply_quota, lock_batch, lock_quota, read_batch, set_batch};
use super::convert::{batch_id, owner};
use super::driver::{
    bin, bytes, corrupt, flag, int, mtx, opt_text, text, Core, Tx, CLAIMS, DEFICITS, ITEMS, OPERATIONS, SCHEDULE,
    SESSIONS, SETTLEMENTS,
};
use super::lifecycle::items_of;

fn encode_outcomes(items: &[(u64, ItemCommitOutcome)]) -> String {
    items
        .iter()
        .map(|(line, outcome)| match outcome {
            ItemCommitOutcome::Accepted => format!("{line}:A"),
            ItemCommitOutcome::Requeued => format!("{line}:Q"),
            ItemCommitOutcome::Rejected(reason) => format!("{line}:R:{}", reason.as_db()),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_outcomes(text: &str) -> Result<Vec<(u64, ItemCommitOutcome)>, StoreError> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    text.split(',')
        .map(|entry| {
            let mut parts = entry.splitn(3, ':');
            let line = parts.next().and_then(|v| v.parse::<u64>().ok()).ok_or(corrupt("commit outcome"))?;
            let outcome = match (parts.next(), parts.next()) {
                (Some("A"), None) => ItemCommitOutcome::Accepted,
                (Some("Q"), None) => ItemCommitOutcome::Requeued,
                (Some("R"), Some(reason)) => {
                    ItemCommitOutcome::Rejected(RejectReason::from_db(reason).ok_or(corrupt("commit outcome"))?)
                }
                _ => return Err(corrupt("commit outcome")),
            };
            Ok((line, outcome))
        })
        .collect()
}

fn payload_len(payload: &ResultPayload) -> u64 {
    match payload {
        ResultPayload::Inline(bytes) => bytes.len() as u64,
        ResultPayload::Stored { object, .. } => object.bytes,
    }
}

#[expect(clippy::too_many_arguments, reason = "one settlement decision with its fields")]
async fn decide(
    tx: &mut Tx<'_>,
    token: &ClaimToken,
    line_no: i64,
    kind: &str,
    operation: &OperationId,
    accepted_bytes: i64,
    orphan: Option<String>,
    actual_cost: u64,
    now: i64,
) -> Result<(), StoreError> {
    let changed = tx
        .update(
            SETTLEMENTS,
            doc! { "claim_id": token.claim_id.as_str(), "line_no": line_no, "kind": Bson::Null },
            doc! { "$set": {
                "kind": kind,
                "operation_id": operation.as_str(),
                "accepted_bytes": accepted_bytes,
                "orphan_session_id": super::driver::opt_str(orphan.as_deref()),
                "actual_cost": signed(actual_cost)?,
                "decided_at": now,
            } },
        )
        .await?;
    if changed != 1 {
        return Err(corrupt("claimed item without an open settlement"));
    }
    set_batch(tx, &token.batch, doc! { "$inc": { "accounting_pending": 1_i64 } }).await
}

async fn commit_results_tx(
    tx: &mut Tx<'_>,
    operation: &OperationId,
    token: &ClaimToken,
    results: &[ItemResult],
    digest: Sha256Digest,
) -> Result<CommitReport, StoreError> {
    let settings = *tx.settings();
    let owner_id = token.batch.owner.as_str();
    if let Some(recorded) =
        tx.one(OPERATIONS, doc! { "owner_id": owner_id, "kind": "commit", "operation_id": operation.as_str() }).await?
    {
        if bin(&recorded, "request_sha256")? != digest.0 {
            return Err(StoreError::IdempotencyConflict);
        }
        let items = decode_outcomes(&text(&recorded, "outcome")?)?;
        let status = read_batch(tx, &token.batch).await?.map_or(BatchStatus::Completed, |b| b.record.status);
        return Ok(CommitReport { items, replayed: true, batch_status: status });
    }
    let (batch, now) = lock_batch(tx, &token.batch).await?;
    let claim = load_claim(tx, token).await?;
    let claim_ok = token.kind == LeaseKind::Execution && execution_claim_valid(&claim, token, Some(now));
    let control_ok =
        batch.record.status == BatchStatus::InProgress && batch.record.control_epoch == token.control_epoch;
    let deadline_ok = now < batch.record.expires_at.0;
    let mut sorted: Vec<&ItemResult> = results.iter().collect();
    sorted.sort_by_key(|r| r.line_no);
    let (mut completed, mut failed, mut requeued) = (0i64, 0i64, 0i64);
    let mut outcomes = Vec::with_capacity(sorted.len());
    for result in sorted {
        let line_no = signed(result.line_no)?;
        let mut key = items_of(&token.batch);
        key.insert("line_no", line_no);
        let item = tx.one(ITEMS, key.clone()).await?;
        let held = item
            .as_ref()
            .map(|d| -> Result<_, StoreError> {
                Ok((
                    text(d, "state")?,
                    opt_text(d, "claim_id")?,
                    super::driver::uint(d, "result_bound")?,
                    int(d, "failure_attempts")?,
                ))
            })
            .transpose()?;
        let reject = match &held {
            None => Some(RejectReason::NotClaimed),
            Some((state, _, _, _)) if matches!(state.as_str(), "succeeded" | "failed" | "cancelled" | "expired") => {
                Some(RejectReason::AlreadyTerminal)
            }
            Some((_, claim_id, _, _)) if claim_id.as_deref() != Some(token.claim_id.as_str()) => {
                let ours =
                    tx.one(SETTLEMENTS, doc! { "claim_id": token.claim_id.as_str(), "line_no": line_no }).await?;
                Some(if ours.is_some() { RejectReason::Stale } else { RejectReason::NotClaimed })
            }
            Some(_) if !claim_ok => Some(RejectReason::Stale),
            Some(_) if !control_ok => Some(RejectReason::ControlClosed),
            Some(_) if !deadline_ok => Some(RejectReason::DeadlinePassed),
            Some((_, _, bound, _)) if payload_len(&result.payload) > *bound => Some(RejectReason::ExceedsResultBound),
            Some(_) => None,
        };
        if let Some(reason) = reject {
            if let ResultPayload::Stored { session, .. } = &result.payload {
                // The uploaded object loses the race; GC deletes it after the I/O guard.
                tx.update(
                    SESSIONS,
                    doc! { "_id": session.as_str(), "claim_id": token.claim_id.as_str(), "phase": "open" },
                    doc! { "$set": { "phase": "orphaned" } },
                )
                .await?;
            }
            outcomes.push((result.line_no, ItemCommitOutcome::Rejected(reason)));
            continue;
        }
        let attempts = held.as_ref().map_or(0, |h| h.3);
        let retry_left = attempts + 1 < i64::from(settings.max_failure_attempts);
        if result.disposition == ResultDisposition::Retry && retry_left {
            let orphan = match &result.payload {
                ResultPayload::Stored { session, .. } => {
                    tx.update(
                        SESSIONS,
                        doc! { "_id": session.as_str(), "claim_id": token.claim_id.as_str() },
                        doc! { "$set": { "phase": "orphaned" } },
                    )
                    .await?;
                    Some(session.as_str().to_owned())
                }
                ResultPayload::Inline(_) => None,
            };
            tx.update(
                ITEMS,
                key,
                doc! { "$set": { "state": "pending", "claim_id": Bson::Null }, "$inc": { "failure_attempts": 1_i64 } },
            )
            .await?;
            decide(tx, token, line_no, "requeued", operation, 0, orphan, result.actual_cost, now).await?;
            requeued += 1;
            outcomes.push((result.line_no, ItemCommitOutcome::Requeued));
            continue;
        }
        let (state, kind) = match result.disposition {
            ResultDisposition::Output => ("succeeded", "output"),
            _ => ("failed", "error"),
        };
        let attempts_delta = i64::from(result.disposition == ResultDisposition::Retry);
        let size = signed(payload_len(&result.payload))?;
        let mut set = doc! { "state": state, "claim_id": Bson::Null, "result_kind": kind, "result_bytes": size, "accepted_at": now };
        match &result.payload {
            ResultPayload::Inline(line) => {
                set.insert("result_inline", bytes(line));
                set.insert("result_sha256", bytes(&sha2::Sha256::digest(line)));
            }
            ResultPayload::Stored { session, artifact, object } => {
                let removed = tx
                    .delete_one(
                        SESSIONS,
                        doc! {
                            "_id": session.as_str(),
                            "kind": "result",
                            "claim_id": token.claim_id.as_str(),
                            "line_no": line_no,
                            "artifact_ref": artifact.as_str(),
                            "phase": "open",
                        },
                    )
                    .await?;
                if removed != 1 {
                    return Err(StoreError::ResultInvalid("stored result has no open session of this claim"));
                }
                set.insert("result_ref", artifact.as_str());
                set.insert("result_sha256", bytes(&object.sha256.0));
            }
        }
        tx.update(ITEMS, key, doc! { "$set": set, "$inc": { "failure_attempts": attempts_delta } }).await?;
        decide(tx, token, line_no, "accepted", operation, size, None, result.actual_cost, now).await?;
        if state == "succeeded" {
            completed += 1;
        } else {
            failed += 1;
        }
        outcomes.push((result.line_no, ItemCommitOutcome::Accepted));
    }
    set_batch(
        tx,
        &token.batch,
        doc! { "$inc": { "completed": completed, "failed": failed, "queued": requeued, "mutation_seq": 1_i64 } },
    )
    .await?;
    let finished = batch.finished() as i64 + completed + failed;
    let mut status = batch.record.status;
    let batch_claims = doc! {
        "owner_id": token.batch.owner.as_str(),
        "batch_id": token.batch.id.as_str(),
        "kind": "execution",
        "state": "active",
    };
    if status == BatchStatus::InProgress && finished == signed(batch.record.counts.total)? {
        set_batch(
            tx,
            &token.batch,
            doc! { "$set": { "status": "finalizing", "terminal_target": "completed", "finalizing_at": now } },
        )
        .await?;
        tx.update_many(CLAIMS, batch_claims, doc! { "$set": { "state": "finished" } }).await?;
        status = BatchStatus::Finalizing;
    } else {
        let mut still_held = items_of(&token.batch);
        still_held.insert("claim_id", token.claim_id.as_str());
        still_held.insert("state", "claimed");
        if tx.count(ITEMS, still_held).await? == 0 {
            tx.update(
                CLAIMS,
                doc! { "_id": token.claim_id.as_str(), "state": "active" },
                doc! { "$set": { "state": "finished" } },
            )
            .await?;
        }
    }
    tx.insert(
        OPERATIONS,
        doc! {
            "owner_id": owner_id,
            "kind": "commit",
            "operation_id": operation.as_str(),
            "request_sha256": bytes(&digest.0),
            "batch_id": token.batch.id.as_str(),
            "outcome": encode_outcomes(&outcomes),
            "created_at": now,
        },
    )
    .await?;
    Ok(CommitReport { items: outcomes, replayed: false, batch_status: status })
}

/// Record results of one claim (see the SQL contract): ledger first, then acceptance under batch control.
pub(super) async fn commit_results(
    core: &Core,
    operation: &OperationId,
    token: &ClaimToken,
    results: &[ItemResult],
    digest: Sha256Digest,
) -> Result<CommitReport, StoreError> {
    mtx!(core, tx => commit_results_tx(tx, operation, token, results, digest))
}

pub(super) async fn owners_with_pending_settlements(core: &Core, limit: u32) -> Result<Vec<OwnerId>, StoreError> {
    let rows = core
        .read_aggregate(
            SETTLEMENTS,
            vec![
                doc! { "$match": { "accounted": false, "kind": { "$ne": Bson::Null } } },
                doc! { "$group": { "_id": "$owner_id" } },
                doc! { "$sort": { "_id": 1_i32 } },
                doc! { "$limit": i64::from(limit) },
            ],
        )
        .await?;
    rows.iter().map(|row| owner(text(row, "_id")?)).collect()
}

async fn settle_pending_tx(tx: &mut Tx<'_>, owner_id: &OwnerId, limit: u32) -> Result<AccountingProgress, StoreError> {
    let settings = *tx.settings();
    lock_quota(tx, owner_id).await?;
    let pending = doc! { "owner_id": owner_id.as_str(), "accounted": false, "kind": { "$ne": Bson::Null } };
    let Some(first) = tx.one(SETTLEMENTS, pending.clone()).await? else {
        return Ok(AccountingProgress::default());
    };
    let shard = text(&first, "shard")?;
    let (schedule, _) =
        tx.lock(SCHEDULE, doc! { "_id": shard.as_str() }).await?.ok_or(corrupt("shard document missing"))?;
    let cap = int(&schedule, "credit_cap")?;
    let mut filter = pending;
    filter.insert("shard", shard.as_str());
    let rows = tx
        .many(
            SETTLEMENTS,
            filter,
            doc! { "batch_id": 1_i32, "claim_id": 1_i32, "line_no": 1_i32 },
            i64::from(limit.max(1)),
        )
        .await?;
    let mut batches: Vec<String> = rows.iter().map(|r| text(r, "batch_id")).collect::<Result<_, _>>()?;
    batches.sort();
    batches.dedup();
    let mut batch_credit = BTreeMap::new();
    for id in &batches {
        let (batch, _) = lock_batch(tx, &BatchRef { owner: owner_id.clone(), id: batch_id(id.clone())? }).await?;
        batch_credit.insert(id.clone(), batch.sched_credit);
    }
    let deficit_key = doc! { "shard": shard.as_str(), "owner_id": owner_id.as_str() };
    let (mut credit, mut reserved_cost) = match tx.one(DEFICITS, deficit_key.clone()).await? {
        Some(d) => (int(&d, "credit")?, int(&d, "reserved_cost")?),
        None => (0, 0),
    };
    let mut delta = QuotaDelta::default();
    let mut settled = 0u64;
    for row in &rows {
        let claim_id = text(row, "claim_id")?;
        let line_no = int(row, "line_no")?;
        let batch = text(row, "batch_id")?;
        let reserved = int(row, "reserved_bytes")?;
        let estimated = unsigned(int(row, "estimated_cost")?)?;
        let kind = opt_text(row, "kind")?.ok_or(corrupt("settlement kind"))?;
        let accepted = int(row, "accepted_bytes")?;
        let orphan = opt_text(row, "orphan_session_id")?;
        let actual = unsigned(int(row, "actual_cost")?)?;
        if flag(row, "accounted")? {
            continue;
        }
        let changed = tx
            .update(
                SETTLEMENTS,
                doc! { "claim_id": claim_id.as_str(), "line_no": line_no, "accounted": false },
                doc! { "$set": { "accounted": true } },
            )
            .await?;
        if changed != 1 {
            continue;
        }
        delta.reserved -= reserved;
        let batch_key = BatchRef { owner: owner_id.clone(), id: batch_id(batch.clone())? };
        if kind == "accepted" && accepted > 0 {
            let mut item_key = items_of(&batch_key);
            item_key.insert("line_no", line_no);
            tx.update(ITEMS, item_key, doc! { "$inc": { "quota_used": accepted } }).await?;
            set_batch(tx, &batch_key, doc! { "$inc": { "export_reserved": accepted } }).await?;
            delta.used += accepted;
            delta.reserved += accepted;
        } else if let Some(session) = orphan {
            // The source half (one result_bound) keeps covering the uploaded object until GC deletes it.
            let half = reserved / 2;
            if tx
                .update(SESSIONS, doc! { "_id": session.as_str() }, doc! { "$inc": { "reserved_bytes": half } })
                .await?
                == 1
            {
                delta.reserved += half;
            }
        }
        credit = settle_credit(credit, estimated, actual, cap, settings.max_debt);
        reserved_cost = reserved_cost.saturating_sub(signed(estimated)?).max(0);
        if let Some(value) = batch_credit.get_mut(&batch) {
            *value = settle_credit(*value, estimated, actual, cap, settings.max_debt);
        }
        set_batch(tx, &batch_key, doc! { "$inc": { "accounting_pending": -1_i64 } }).await?;
        settled += 1;
    }
    for (id, value) in &batch_credit {
        let key = BatchRef { owner: owner_id.clone(), id: batch_id(id.clone())? };
        set_batch(tx, &key, doc! { "$set": { "sched_credit": *value } }).await?;
    }
    tx.update(DEFICITS, deficit_key, doc! { "$set": { "credit": credit, "reserved_cost": reserved_cost } }).await?;
    apply_quota(tx, owner_id, delta).await?;
    Ok(AccountingProgress { settled, remaining: rows.len() as u64 >= u64::from(limit.max(1)) })
}

/// Account decided settlements of one owner exactly once (quota → shard → batches → settlements).
pub(super) async fn settle_pending(core: &Core, owner: &OwnerId, limit: u32) -> Result<AccountingProgress, StoreError> {
    mtx!(core, tx => settle_pending_tx(tx, owner, limit))
}
