//! Result commit with the operation ledger, and exactly-once settlement of claim reservations.

use super::super::error::StoreError;
use super::super::policy::settle_credit;
use super::super::sql::{args, db_now, SqlRow, SqlTx};
use super::super::types::{
    AccountingProgress, BatchRef, BatchStatus, ClaimToken, CommitReport, ItemCommitOutcome, ItemResult, LeaseKind,
    OperationId, OwnerId, RejectReason, ResultDisposition, ResultPayload, Sha256Digest, StoreSettings,
};
use super::claims::{execution_claim_valid, load_claim};
use super::common::{
    apply_quota, batch_id_from_db, corrupt, lock_batch, lock_quota, owner_from_db, signed, unsigned, QuotaDelta,
};

/// Largest number of records in one commit.
pub(in super::super) const MAX_COMMIT_ITEMS: usize = 32;
/// Largest sum of inline result bytes in one commit.
pub(in super::super) const MAX_COMMIT_INLINE: usize = 1024 * 1024;

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

/// Shape checks done before any transaction (bounded commit, inline limits, payload sizes).
pub(in super::super) fn check_results(results: &[ItemResult]) -> Result<(), StoreError> {
    use super::super::types::{INLINE_RESULT_MAX, RESULT_MAX};
    if results.is_empty() || results.len() > MAX_COMMIT_ITEMS {
        return Err(StoreError::ResultInvalid("a commit carries 1..32 results"));
    }
    let mut inline = 0usize;
    for (index, result) in results.iter().enumerate() {
        if results[..index].iter().any(|other| other.line_no == result.line_no) {
            return Err(StoreError::ResultInvalid("duplicate line in commit"));
        }
        match &result.payload {
            ResultPayload::Inline(bytes) => {
                if bytes.is_empty() || bytes.len() > INLINE_RESULT_MAX || bytes.last() != Some(&b'\n') {
                    return Err(StoreError::ResultInvalid("inline result must be one JSONL line up to 64 KiB"));
                }
                inline += bytes.len();
            }
            ResultPayload::Stored { object, .. } => {
                if object.bytes == 0 || object.bytes > RESULT_MAX {
                    return Err(StoreError::ResultInvalid("stored result size out of range"));
                }
            }
        }
    }
    if inline > MAX_COMMIT_INLINE {
        return Err(StoreError::ResultInvalid("inline results exceed 1 MiB per commit"));
    }
    Ok(())
}

fn payload_len(payload: &ResultPayload) -> u64 {
    match payload {
        ResultPayload::Inline(bytes) => bytes.len() as u64,
        ResultPayload::Stored { object, .. } => object.bytes,
    }
}

/// Record results of one claim. A known operation id returns its stored outcome even after the lease ended; a
/// different digest is a conflict. Acceptance requires, after the batch lock: active unexpired claim, unchanged
/// control epoch, `in_progress` and the deadline not passed.
pub(in super::super) async fn commit_results<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    operation: &OperationId,
    token: &ClaimToken,
    results: &[ItemResult],
    digest: Sha256Digest,
) -> Result<CommitReport, StoreError> {
    let owner = token.batch.owner.as_str();
    let id = token.batch.id.as_str();
    if let Some(row) = tx
        .fetch_optional(
            &format!(
                "SELECT request_sha256, outcome FROM batch_operations WHERE owner_id = $1 AND kind = 'commit' \
                 AND {}",
                tx.dialect().exact_eq("operation_id", "$2")
            ),
            args![owner, operation.as_str()],
        )
        .await?
    {
        if row.blob_at(0)? != digest.0 {
            return Err(StoreError::IdempotencyConflict);
        }
        let items = decode_outcomes(&row.text_at(1)?)?;
        let status = super::common::load_batch(tx, &token.batch.owner, &token.batch.id, false)
            .await?
            .map_or(BatchStatus::Completed, |b| b.record.status);
        return Ok(CommitReport { items, replayed: true, batch_status: status });
    }
    let batch = lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
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
        let item = tx
            .fetch_optional(
                "SELECT state, claim_id, result_bound, failure_attempts FROM batch_items \
                 WHERE owner_id = $1 AND batch_id = $2 AND line_no = $3",
                args![owner, id, line_no],
            )
            .await?;
        let held = item.as_ref().map(|row| -> Result<_, StoreError> {
            Ok((row.text_at(0)?, row.opt_text_at(1)?, row.u64_at(2)?, row.i64_at(3)?))
        });
        let held = held.transpose()?;
        let reject = match &held {
            None => Some(RejectReason::NotClaimed),
            Some((state, _, _, _)) if matches!(state.as_str(), "succeeded" | "failed" | "cancelled" | "expired") => {
                Some(RejectReason::AlreadyTerminal)
            }
            Some((_, claim_id, _, _)) if claim_id.as_deref() != Some(token.claim_id.as_str()) => {
                let ours = tx
                    .fetch_optional(
                        "SELECT 1 FROM batch_settlements WHERE claim_id = $1 AND line_no = $2",
                        args![token.claim_id.as_str(), line_no],
                    )
                    .await?;
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
                tx.execute(
                    "UPDATE artifact_sessions SET phase = 'orphaned' WHERE id = $1 AND claim_id = $2 AND phase = 'open'",
                    args![session.as_str(), token.claim_id.as_str()],
                )
                .await?;
            }
            outcomes.push((result.line_no, ItemCommitOutcome::Rejected(reason)));
            continue;
        }
        let attempts = held.as_ref().map_or(0, |h| h.3);
        let retry_left = attempts + 1 < i64::from(settings.max_failure_attempts);
        let outcome = match result.disposition {
            ResultDisposition::Retry if retry_left => ItemCommitOutcome::Requeued,
            _ => ItemCommitOutcome::Accepted,
        };
        if outcome == ItemCommitOutcome::Requeued {
            let orphan = match &result.payload {
                ResultPayload::Stored { session, .. } => {
                    tx.execute(
                        "UPDATE artifact_sessions SET phase = 'orphaned' WHERE id = $1 AND claim_id = $2",
                        args![session.as_str(), token.claim_id.as_str()],
                    )
                    .await?;
                    Some(session.as_str().to_owned())
                }
                ResultPayload::Inline(_) => None,
            };
            tx.execute(
                "UPDATE batch_items SET state = 'pending', claim_id = NULL, failure_attempts = failure_attempts + 1 \
                 WHERE owner_id = $1 AND batch_id = $2 AND line_no = $3",
                args![owner, id, line_no],
            )
            .await?;
            decide(tx, token, line_no, "requeued", operation, 0, orphan, result.actual_cost, now).await?;
            requeued += 1;
            outcomes.push((result.line_no, outcome));
            continue;
        }
        let (state, kind) = match result.disposition {
            ResultDisposition::Output => ("succeeded", "output"),
            _ => ("failed", "error"),
        };
        let attempts_delta = i64::from(result.disposition == ResultDisposition::Retry);
        match &result.payload {
            ResultPayload::Inline(bytes) => {
                tx.execute(
                    "UPDATE batch_items SET state = $4, claim_id = NULL, result_kind = $5, result_inline = $6, \
                     result_bytes = $7, result_sha256 = $8, accepted_at = $9, \
                     failure_attempts = failure_attempts + $10 WHERE owner_id = $1 AND batch_id = $2 AND line_no = $3",
                    args![
                        owner,
                        id,
                        line_no,
                        state,
                        kind,
                        bytes.clone(),
                        bytes.len() as i64,
                        sha2_digest(bytes),
                        now,
                        attempts_delta
                    ],
                )
                .await?;
            }
            ResultPayload::Stored { session, artifact, object } => {
                let removed = tx
                    .execute(
                        "DELETE FROM artifact_sessions WHERE id = $1 AND kind = 'result' AND claim_id = $2 \
                         AND line_no = $3 AND artifact_ref = $4 AND phase = 'open'",
                        args![session.as_str(), token.claim_id.as_str(), line_no, artifact.as_str()],
                    )
                    .await?;
                if removed != 1 {
                    return Err(StoreError::ResultInvalid("stored result has no open session of this claim"));
                }
                tx.execute(
                    "UPDATE batch_items SET state = $4, claim_id = NULL, result_kind = $5, result_ref = $6, \
                     result_bytes = $7, result_sha256 = $8, accepted_at = $9, \
                     failure_attempts = failure_attempts + $10 WHERE owner_id = $1 AND batch_id = $2 AND line_no = $3",
                    args![
                        owner,
                        id,
                        line_no,
                        state,
                        kind,
                        artifact.as_str(),
                        signed(object.bytes)?,
                        object.sha256.0.to_vec(),
                        now,
                        attempts_delta
                    ],
                )
                .await?;
            }
        }
        let bytes = signed(payload_len(&result.payload))?;
        decide(tx, token, line_no, "accepted", operation, bytes, None, result.actual_cost, now).await?;
        if state == "succeeded" {
            completed += 1;
        } else {
            failed += 1;
        }
        outcomes.push((result.line_no, outcome));
    }
    tx.execute(
        "UPDATE batches SET completed = completed + $3, failed = failed + $4, queued = queued + $5, \
         mutation_seq = mutation_seq + 1 WHERE owner_id = $1 AND id = $2",
        args![owner, id, completed, failed, requeued],
    )
    .await?;
    let finished = batch.finished() as i64 + completed + failed;
    let mut status = batch.record.status;
    if status == BatchStatus::InProgress && finished == signed(batch.record.counts.total)? {
        tx.execute(
            "UPDATE batches SET status = 'finalizing', terminal_target = 'completed', finalizing_at = $3 \
             WHERE owner_id = $1 AND id = $2",
            args![owner, id, now],
        )
        .await?;
        tx.execute(
            "UPDATE batch_claims SET state = 'finished' WHERE owner_id = $1 AND batch_id = $2 AND kind = 'execution' \
             AND state = 'active'",
            args![owner, id],
        )
        .await?;
        status = BatchStatus::Finalizing;
    } else {
        tx.execute(
            "UPDATE batch_claims SET state = 'finished' WHERE id = $1 AND state = 'active' AND NOT EXISTS \
             (SELECT 1 FROM batch_items WHERE claim_id = $1 AND state = 'claimed')",
            args![token.claim_id.as_str()],
        )
        .await?;
    }
    tx.execute(
        "INSERT INTO batch_operations (owner_id, kind, operation_id, request_sha256, batch_id, outcome, created_at) \
         VALUES ($1, 'commit', $2, $3, $4, $5, $6)",
        args![owner, operation.as_str(), digest.0.to_vec(), id, encode_outcomes(&outcomes), now],
    )
    .await?;
    Ok(CommitReport { items: outcomes, replayed: false, batch_status: status })
}

fn sha2_digest(bytes: &[u8]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).to_vec()
}

#[expect(clippy::too_many_arguments, reason = "one settlement decision with its columns")]
async fn decide<T: SqlTx>(
    tx: &mut T,
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
        .execute(
            "UPDATE batch_settlements SET kind = $3, operation_id = $4, accepted_bytes = $5, orphan_session_id = $6, \
             actual_cost = $7, decided_at = $8 WHERE claim_id = $1 AND line_no = $2 AND kind IS NULL",
            args![
                token.claim_id.as_str(),
                line_no,
                kind,
                operation.as_str(),
                accepted_bytes,
                orphan,
                signed(actual_cost)?,
                now
            ],
        )
        .await?;
    if changed != 1 {
        return Err(corrupt("claimed item without an open settlement"));
    }
    tx.execute(
        "UPDATE batches SET accounting_pending = accounting_pending + 1 WHERE owner_id = $1 AND id = $2",
        args![token.batch.owner.as_str(), token.batch.id.as_str()],
    )
    .await?;
    Ok(())
}

/// Owners with decided, unaccounted settlements (bounded).
pub(in super::super) async fn owners_with_pending_settlements<T: SqlTx>(
    tx: &mut T,
    limit: u32,
) -> Result<Vec<OwnerId>, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT DISTINCT {}owner_id FROM batch_settlements WHERE accounted = 0 AND kind IS NOT NULL \
         ORDER BY owner_id{}",
        d.top("$1"),
        d.limit("$1")
    );
    let rows = tx.fetch_all(&sql, args![i64::from(limit)]).await?;
    rows.iter().map(|row| owner_from_db(row.text_at(0)?)).collect()
}

/// Account up to `limit` decided settlements of one owner and one shard, exactly once, in one short transaction:
/// quota → shard → batches (ascending) → settlement rows. Releases the execution reserve, moves accepted bytes to
/// the item (source copy) and the batch export reserve, moves the source half of an orphaned upload to its session,
/// and corrects DRR credit by the measured cost.
pub(in super::super) async fn settle_pending<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    owner: &OwnerId,
    limit: u32,
) -> Result<AccountingProgress, StoreError> {
    lock_quota(tx, owner).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}shard FROM batch_settlements WHERE owner_id = $1 AND accounted = 0 AND kind IS NOT NULL{}",
        d.top("1"),
        d.limit("1")
    );
    let Some(shard_row) = tx.fetch_optional(&sql, args![owner.as_str()]).await? else {
        return Ok(AccountingProgress::default());
    };
    let shard = shard_row.text_at(0)?;
    let sql = format!("SELECT credit_cap FROM batch_schedule{} WHERE shard = $1{}", d.lock_hint, d.for_update);
    let cap = tx.fetch_optional(&sql, args![shard.as_str()]).await?.ok_or(corrupt("shard row missing"))?.i64_at(0)?;
    let sql = format!(
        "SELECT {}claim_id, line_no, batch_id, reserved_bytes, estimated_cost, kind, accepted_bytes, \
         orphan_session_id, actual_cost FROM batch_settlements WHERE owner_id = $1 AND shard = $2 \
         AND accounted = 0 AND kind IS NOT NULL ORDER BY batch_id, claim_id, line_no{}",
        d.top("$3"),
        d.limit("$3")
    );
    let rows = tx.fetch_all(&sql, args![owner.as_str(), shard.as_str(), i64::from(limit.max(1))]).await?;
    let mut batches: Vec<String> = rows.iter().map(|r| r.text_at(2)).collect::<Result<_, _>>()?;
    batches.sort();
    batches.dedup();
    let mut batch_credit = std::collections::BTreeMap::new();
    for id in &batches {
        let batch = lock_batch(tx, &BatchRef { owner: owner.clone(), id: batch_id_from_db(id.clone())? }).await?;
        batch_credit.insert(id.clone(), batch.sched_credit);
    }
    let deficit = tx
        .fetch_optional(
            "SELECT credit, reserved_cost FROM batch_deficits WHERE shard = $1 AND owner_id = $2",
            args![shard.as_str(), owner.as_str()],
        )
        .await?;
    let (mut credit, mut reserved_cost) = match deficit {
        Some(row) => (row.i64_at(0)?, row.i64_at(1)?),
        None => (0, 0),
    };
    let mut delta = QuotaDelta::default();
    let mut settled = 0u64;
    for row in &rows {
        let claim_id = row.text_at(0)?;
        let line_no = row.i64_at(1)?;
        let batch_id = row.text_at(2)?;
        let reserved = row.i64_at(3)?;
        let estimated = unsigned(row.i64_at(4)?)?;
        let kind = row.opt_text_at(5)?.ok_or(corrupt("settlement kind"))?;
        let accepted = row.i64_at(6)?;
        let orphan = row.opt_text_at(7)?;
        let actual = unsigned(row.i64_at(8)?)?;
        let changed = tx
            .execute(
                "UPDATE batch_settlements SET accounted = 1 WHERE claim_id = $1 AND line_no = $2 AND accounted = 0",
                args![claim_id.as_str(), line_no],
            )
            .await?;
        if changed != 1 {
            continue;
        }
        delta.reserved -= reserved;
        if kind == "accepted" && accepted > 0 {
            tx.execute(
                "UPDATE batch_items SET quota_used = quota_used + $4 WHERE owner_id = $1 AND batch_id = $2 \
                 AND line_no = $3",
                args![owner.as_str(), batch_id.as_str(), line_no, accepted],
            )
            .await?;
            tx.execute(
                "UPDATE batches SET export_reserved = export_reserved + $3 WHERE owner_id = $1 AND id = $2",
                args![owner.as_str(), batch_id.as_str(), accepted],
            )
            .await?;
            delta.used += accepted;
            delta.reserved += accepted;
        } else if let Some(session) = orphan {
            // The source half (one result_bound) keeps covering the uploaded object until GC deletes it.
            let half = reserved / 2;
            let moved = tx
                .execute(
                    "UPDATE artifact_sessions SET reserved_bytes = reserved_bytes + $2 WHERE id = $1",
                    args![session.as_str(), half],
                )
                .await?;
            if moved == 1 {
                delta.reserved += half;
            }
        }
        credit = settle_credit(credit, estimated, actual, cap, settings.max_debt);
        reserved_cost = reserved_cost.saturating_sub(signed(estimated)?).max(0);
        if let Some(batch) = batch_credit.get_mut(&batch_id) {
            *batch = settle_credit(*batch, estimated, actual, cap, settings.max_debt);
        }
        tx.execute(
            "UPDATE batches SET accounting_pending = accounting_pending - 1 WHERE owner_id = $1 AND id = $2",
            args![owner.as_str(), batch_id.as_str()],
        )
        .await?;
        settled += 1;
    }
    for (id, credit) in &batch_credit {
        tx.execute(
            "UPDATE batches SET sched_credit = $3 WHERE owner_id = $1 AND id = $2",
            args![owner.as_str(), id.as_str(), *credit],
        )
        .await?;
    }
    tx.execute(
        "UPDATE batch_deficits SET credit = $3, reserved_cost = $4 WHERE shard = $1 AND owner_id = $2",
        args![shard.as_str(), owner.as_str(), credit, reserved_cost],
    )
    .await?;
    apply_quota(tx, owner, delta).await?;
    Ok(AccountingProgress { settled, remaining: rows.len() as u64 >= u64::from(limit.max(1)) })
}
