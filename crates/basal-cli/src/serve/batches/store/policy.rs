//! Pure rules shared by every adapter: reservation sizes, the synthetic expiry line, DRR selection, canonical
//! digests and policy validation. Keeping them here guarantees that all engines compute the same quota.

use sha2::{Digest, Sha256};

use super::error::{InvalidValue, StoreError};
use super::types::{
    BatchIssue, CatalogSpec, ClaimToken, CustomId, ItemResult, PublishRequest, QuotaLimits, QuotaPolicySpec,
    ResultDisposition, ResultPayload, Sha256Digest,
};

/// Error code and message of a record that was not executed before the completion window closed.
pub(crate) const EXPIRED_CODE: &str = "batch_expired";
pub(crate) const EXPIRED_MESSAGE: &str = "This request could not be executed before the completion window expired.";

/// Length of a `batch_req_<32 hex>` request identifier.
pub(crate) const REQUEST_ID_LEN: usize = 42;

/// Shortest valid record line without its separator:
/// `{"custom_id":"a","method":"POST","url":"/v1/basal","body":{}}` (61 bytes). Keys cannot be shorter and escapes
/// only lengthen strings, so `⌈file bytes / 61⌉` bounds the number of records of any accepted file.
pub(crate) const MIN_RECORD_LINE_BYTES: u64 = 61;

/// Bytes charged per stored validation issue (code, message, param, line and row overhead).
pub(crate) const ISSUE_CHARGE_BYTES: u64 = 512;

fn json_string_len(value: &str) -> u64 {
    // serde_json escaping is deterministic; the writer uses the same function.
    serde_json::to_string(value).map_or(0, |s| s.len() as u64)
}

const EXPIRED_PREFIX: &str = "{\"id\":\"";
const EXPIRED_MIDDLE: &str = "\",\"custom_id\":";

fn expired_suffix() -> String {
    format!(
        ",\"response\":null,\"error\":{{\"code\":\"{EXPIRED_CODE}\",\"message\":{}}}}}\n",
        serde_json::to_string(EXPIRED_MESSAGE).unwrap_or_default()
    )
}

/// Exact length of the synthetic expiry line of a record, LF included.
pub(crate) fn expired_line_len(custom_id: &CustomId) -> u64 {
    (EXPIRED_PREFIX.len() + REQUEST_ID_LEN + EXPIRED_MIDDLE.len() + expired_suffix().len()) as u64
        + json_string_len(custom_id.as_str())
}

/// Append the synthetic `batch_expired` line (exactly [`expired_line_len`] bytes) to `out`.
pub(crate) fn write_expired_line(
    out: &mut Vec<u8>,
    request_id: &str,
    custom_id: &CustomId,
) -> Result<(), InvalidValue> {
    if request_id.len() != REQUEST_ID_LEN || !request_id.starts_with("batch_req_") {
        return Err(InvalidValue::new("request_id", "must be batch_req_ followed by 32 hex digits"));
    }
    let custom =
        serde_json::to_string(custom_id.as_str()).map_err(|_| InvalidValue::new("custom_id", "cannot be encoded"))?;
    out.extend_from_slice(EXPIRED_PREFIX.as_bytes());
    out.extend_from_slice(request_id.as_bytes());
    out.extend_from_slice(EXPIRED_MIDDLE.as_bytes());
    out.extend_from_slice(custom.as_bytes());
    out.extend_from_slice(expired_suffix().as_bytes());
    Ok(())
}

/// Reserve of one validated record: its expiry line plus the fixed item row charge.
pub(crate) fn record_reserve(custom_id: &CustomId, item_overhead: u64) -> u64 {
    expired_line_len(custom_id) + item_overhead
}

/// Upper bound of [`record_reserve`] over every admissible custom_id: 256 bytes that each escape to `\u00XX`.
pub(crate) fn max_record_reserve(item_overhead: u64) -> u64 {
    let worst_custom_id = (CustomId::MAX_BYTES as u64) * 6 + 2;
    (EXPIRED_PREFIX.len() + REQUEST_ID_LEN + EXPIRED_MIDDLE.len() + expired_suffix().len()) as u64
        + worst_custom_id
        + item_overhead
}

/// Parts of the completion reserve taken at create.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompletionReserve {
    /// `object_charge` of the output and error file slots.
    pub files: u64,
    /// Stored validation issues (`errors.data`).
    pub issues: u64,
    /// `min(max_records, ⌈bytes / min_line⌉) × max_record_reserve`.
    pub records: u64,
    pub max_records: u64,
}

impl CompletionReserve {
    pub(crate) fn total(&self) -> u64 {
        self.files.saturating_add(self.issues).saturating_add(self.records)
    }
}

pub(crate) fn completion_reserve(input_bytes: u64, policy: &QuotaPolicySpec) -> CompletionReserve {
    let by_size = input_bytes.div_ceil(MIN_RECORD_LINE_BYTES).max(1);
    let max_records = by_size.min(policy.max_records_per_batch);
    CompletionReserve {
        files: policy.object_charge_bytes.saturating_mul(2),
        issues: ISSUE_CHARGE_BYTES * BatchIssue::MAX_PER_BATCH as u64,
        records: max_records.saturating_mul(max_record_reserve(policy.item_overhead_bytes)),
        max_records,
    }
}

/// Truncate to at most `max` bytes on a character boundary.
pub(crate) fn truncate_utf8(value: &str, max: usize) -> String {
    if value.len() <= max {
        return value.to_owned();
    }
    let mut end = max;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

/// Bound the issue fields to the contract limits.
pub(crate) fn bounded_issue(issue: &BatchIssue) -> BatchIssue {
    BatchIssue {
        code: truncate_utf8(&issue.code, BatchIssue::MAX_CODE_BYTES),
        message: truncate_utf8(&issue.message, BatchIssue::MAX_MESSAGE_BYTES),
        param: issue.param.as_deref().map(|p| truncate_utf8(p, BatchIssue::MAX_CODE_BYTES)),
        line: issue.line,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// DRR
// ---------------------------------------------------------------------------------------------------------------

/// Candidate of one DRR level (owner in a shard, or batch of an owner). Credit accrues lazily: an entry last
/// settled at `last_turn` has `min(cap, credit + (turn − last_turn) × quantum)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DrrCandidate {
    pub credit: i64,
    pub last_turn: i64,
    /// Cost of the record at the head of the candidate's queue.
    pub head_cost: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DrrParams {
    pub quantum: i64,
    pub cap: i64,
}

impl DrrParams {
    /// `credit_cap ≥ max_record_cost + quantum`, so the largest admissible record gathers enough credit in at most
    /// `⌈max_record_cost / quantum⌉` turns.
    pub(crate) fn new(quantum: u64, max_record_cost: u64) -> Self {
        let quantum = i64::try_from(quantum.max(1)).unwrap_or(i64::MAX);
        let cost = i64::try_from(max_record_cost).unwrap_or(i64::MAX);
        Self { quantum, cap: cost.saturating_add(quantum) }
    }
}

pub(crate) fn effective_credit(candidate: &DrrCandidate, turn: i64, params: DrrParams) -> i64 {
    let turns = turn.saturating_sub(candidate.last_turn).max(0);
    // Keep the debt until after accrual: saturating the product first could leave a negative starting credit
    // permanently below the cap. Two signed 64-bit factors and one credit fit in i128.
    let credit = i128::from(candidate.credit) + i128::from(turns) * i128::from(params.quantum);
    credit.clamp(i128::from(i64::MIN), i128::from(params.cap)) as i64
}

fn needed(candidate: &DrrCandidate, params: DrrParams) -> i64 {
    // A record above the configured maximum still progresses once its owner reaches the cap.
    i64::try_from(candidate.head_cost).unwrap_or(i64::MAX).min(params.cap)
}

/// Pick the first candidate from `start` (rotating) whose credit covers its head record. When none does, the turn
/// jumps by the smallest number of quanta after which one does ("turn skip"), so idle GPU time is never spent on
/// empty turns. Returns the index and the possibly advanced turn.
pub(crate) fn drr_pick(
    candidates: &[DrrCandidate],
    start: usize,
    turn: i64,
    params: DrrParams,
) -> Option<(usize, i64)> {
    if candidates.is_empty() {
        return None;
    }
    let n = candidates.len();
    let order = (0..n).map(|k| (start + k) % n);
    if let Some(index) =
        order.clone().find(|&i| effective_credit(&candidates[i], turn, params) >= needed(&candidates[i], params))
    {
        return Some((index, turn));
    }
    let skip = candidates
        .iter()
        .map(|c| {
            let missing = (i128::from(needed(c, params)) - i128::from(effective_credit(c, turn, params))).max(1);
            // Cost minus debt may exceed i64::MAX. Divide before narrowing; quantum is positive by construction.
            let quantum = i128::from(params.quantum);
            let turns = missing / quantum + i128::from(missing % quantum != 0);
            turns.min(i128::from(i64::MAX)) as i64
        })
        .min()
        .unwrap_or(1)
        .max(1);
    let turn = turn.saturating_add(skip);
    order
        .clone()
        .find(|&i| effective_credit(&candidates[i], turn, params) >= needed(&candidates[i], params))
        .map(|index| (index, turn))
}

/// Credit after spending `cost`, bounded below by the maximum debt.
pub(crate) fn spend(credit: i64, cost: u64, max_debt: u64) -> i64 {
    let cost = i64::try_from(cost).unwrap_or(i64::MAX);
    let floor = -i64::try_from(max_debt).unwrap_or(i64::MAX);
    credit.saturating_sub(cost).max(floor)
}

/// Settlement correction: refund the estimate, charge the measured cost, keep the credit within `[−debt, cap]`.
pub(crate) fn settle_credit(credit: i64, estimated: u64, actual: u64, cap: i64, max_debt: u64) -> i64 {
    let floor = -i64::try_from(max_debt).unwrap_or(i64::MAX);
    // Refund and actual charge can cancel. Clamping an intermediate sum would lose valid credit.
    let settled = i128::from(credit) + i128::from(estimated) - i128::from(actual);
    settled.clamp(i128::from(floor), i128::from(cap)) as i64
}

// ---------------------------------------------------------------------------------------------------------------
// Digests
// ---------------------------------------------------------------------------------------------------------------

struct Canonical(Sha256);

impl Canonical {
    fn new(domain: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain.as_bytes());
        hasher.update([0]);
        Self(hasher)
    }

    fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.0.update((value.len() as u64).to_be_bytes());
        self.0.update(value);
        self
    }

    fn str(&mut self, value: &str) -> &mut Self {
        self.bytes(value.as_bytes())
    }

    fn u64(&mut self, value: u64) -> &mut Self {
        self.0.update(value.to_be_bytes());
        self
    }

    fn i64(&mut self, value: i64) -> &mut Self {
        self.0.update(value.to_be_bytes());
        self
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest(self.0.finalize().into())
    }
}

/// Digest of a commit request; a retry with the same operation id must reproduce it exactly.
pub(crate) fn commit_digest(token: &ClaimToken, results: &[ItemResult]) -> Sha256Digest {
    let mut c = Canonical::new("basal.batch.commit.v1");
    c.str(token.claim_id.as_str()).str(token.worker.as_str()).i64(token.generation).u64(results.len() as u64);
    for result in results {
        c.u64(result.line_no).u64(result.actual_cost).str(match result.disposition {
            ResultDisposition::Output => "output",
            ResultDisposition::Error => "error",
            ResultDisposition::Retry => "retry",
        });
        match &result.payload {
            ResultPayload::Inline(bytes) => {
                c.str("inline").bytes(&Sha256::digest(bytes));
            }
            ResultPayload::Stored { session, artifact, object } => {
                c.str("stored").str(session.as_str()).str(artifact.as_str()).u64(object.bytes).bytes(&object.sha256.0);
            }
        }
    }
    c.finish()
}

pub(crate) fn publish_digest(token: &ClaimToken, request: &PublishRequest) -> Sha256Digest {
    let mut c = Canonical::new("basal.batch.publish.v1");
    c.str(token.claim_id.as_str()).i64(token.generation);
    for file in [&request.output, &request.error] {
        match file {
            Some(file) => {
                c.str("some").u64(file.object.bytes).bytes(&file.object.sha256.0).str(&file.filename);
            }
            None => {
                c.str("none");
            }
        }
    }
    c.finish()
}

fn limits_digest(c: &mut Canonical, limits: &QuotaLimits) {
    c.u64(limits.max_bytes)
        .u64(limits.max_active_batches)
        .u64(limits.max_retained_batches)
        .u64(limits.max_files)
        .u64(limits.max_open_upload_sessions);
}

/// Digest of a policy, independent of owner order; hosts compare it with their configuration.
pub(crate) fn policy_digest(spec: &QuotaPolicySpec) -> Sha256Digest {
    let mut c = Canonical::new("basal.batch.quota-policy.v1");
    c.u64(spec.object_charge_bytes)
        .u64(spec.item_overhead_bytes)
        .u64(spec.max_upload_bytes)
        .u64(spec.max_records_per_batch);
    limits_digest(&mut c, &spec.installation);
    let mut owners: Vec<_> = spec.owners.iter().collect();
    owners.sort_by(|a, b| a.0.cmp(&b.0));
    c.u64(owners.len() as u64);
    for (owner, limits) in owners {
        c.str(owner.as_str());
        limits_digest(&mut c, limits);
    }
    c.finish()
}

pub(crate) fn catalog_digest(spec: &CatalogSpec) -> Sha256Digest {
    let mut c = Canonical::new("basal.batch.catalog.v1");
    c.str(spec.default_model.as_str());
    let mut entries: Vec<_> = spec.entries.iter().collect();
    entries.sort_by(|a, b| a.model_name.cmp(&b.model_name));
    c.u64(entries.len() as u64);
    for entry in entries {
        c.str(entry.model_name.as_str()).str(entry.fingerprint.as_str()).str(entry.planner_version.as_str());
    }
    c.finish()
}

/// Finite limits, unique owners and Σ owner limits ≤ installation limits in every dimension.
pub(crate) fn validate_policy(spec: &QuotaPolicySpec) -> Result<(), StoreError> {
    const MAX: u64 = i64::MAX as u64 / 4;
    let scalars =
        [spec.object_charge_bytes, spec.item_overhead_bytes, spec.max_upload_bytes, spec.max_records_per_batch];
    if scalars.iter().any(|&v| v == 0 || v > MAX) {
        return Err(StoreError::PolicyInvalid("charges and per-batch limits must be positive and finite"));
    }
    if spec.owners.is_empty() {
        return Err(StoreError::PolicyInvalid("policy has no owners"));
    }
    let mut sum = QuotaLimits {
        max_bytes: 0,
        max_active_batches: 0,
        max_retained_batches: 0,
        max_files: 0,
        max_open_upload_sessions: 0,
    };
    for (index, (owner, limits)) in spec.owners.iter().enumerate() {
        if spec.owners[..index].iter().any(|(other, _)| other == owner) {
            return Err(StoreError::PolicyInvalid("duplicate owner"));
        }
        let values = [
            limits.max_bytes,
            limits.max_active_batches,
            limits.max_retained_batches,
            limits.max_files,
            limits.max_open_upload_sessions,
        ];
        if values.iter().any(|&v| v > MAX) {
            return Err(StoreError::PolicyInvalid("owner limit is not finite"));
        }
        if limits.max_retained_batches < limits.max_active_batches {
            return Err(StoreError::PolicyInvalid("retained batches must cover active batches"));
        }
        sum.max_bytes = sum.max_bytes.saturating_add(limits.max_bytes);
        sum.max_active_batches = sum.max_active_batches.saturating_add(limits.max_active_batches);
        sum.max_retained_batches = sum.max_retained_batches.saturating_add(limits.max_retained_batches);
        sum.max_files = sum.max_files.saturating_add(limits.max_files);
        sum.max_open_upload_sessions = sum.max_open_upload_sessions.saturating_add(limits.max_open_upload_sessions);
    }
    let i = &spec.installation;
    if sum.max_bytes > i.max_bytes
        || sum.max_active_batches > i.max_active_batches
        || sum.max_retained_batches > i.max_retained_batches
        || sum.max_files > i.max_files
        || sum.max_open_upload_sessions > i.max_open_upload_sessions
    {
        return Err(StoreError::PolicyInvalid("sum of owner limits exceeds the installation limits"));
    }
    Ok(())
}

pub(crate) fn validate_catalog(spec: &CatalogSpec) -> Result<(), StoreError> {
    if spec.entries.is_empty() || spec.entries.len() > 1024 {
        return Err(StoreError::PolicyInvalid("catalog needs 1..1024 entries"));
    }
    for (index, entry) in spec.entries.iter().enumerate() {
        if spec.entries[..index].iter().any(|other| other.model_name == entry.model_name) {
            return Err(StoreError::PolicyInvalid("catalog maps a model name twice"));
        }
    }
    if !spec.entries.iter().any(|e| e.model_name == spec.default_model) {
        return Err(StoreError::PolicyInvalid("default model is not in the catalog"));
    }
    Ok(())
}

/// Persistent abandonment deferral, separate from genuine execution failure attempts.
/// Shutdown and host admission failures do not imply a failed record.
pub(crate) fn abandonment(previous: i64, defer: bool, now: i64) -> (i64, i64) {
    if !defer {
        return (previous, now);
    }
    let attempt = previous.saturating_add(1).clamp(1, 16);
    let delay = (1_i64 << (attempt - 1)).min(30) * 1000;
    (attempt, now.saturating_add(delay))
}

pub(crate) fn abandoned_costs(
    costs: &[super::types::AbandonedCost],
) -> Result<std::collections::BTreeMap<i64, i64>, StoreError> {
    if costs.len() > 4096 {
        return Err(InvalidValue::new("abandoned costs", "more than 4096 records").into());
    }
    let mut values = std::collections::BTreeMap::new();
    for cost in costs {
        let line = i64::try_from(cost.line_no).map_err(|_| InvalidValue::new("line_no", "outside storage range"))?;
        let ns =
            i64::try_from(cost.actual_cost).map_err(|_| InvalidValue::new("actual_cost", "outside storage range"))?;
        if line < 1 || values.insert(line, ns).is_some() {
            return Err(InvalidValue::new("abandoned costs", "invalid or duplicate line").into());
        }
    }
    Ok(values)
}
