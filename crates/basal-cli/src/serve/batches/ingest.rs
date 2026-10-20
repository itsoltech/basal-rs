//! Discovery and full validation of input files, before any inference.
//!
//! Discovery: any host may take the short discovery lease of a new batch. It reads the first line (bounded),
//! applies the envelope preflight and takes the `model` field, then resolves it against the batch's pinned catalog
//! revision with `Store::resolve_target`. A model outside the catalog, or an empty or malformed first line, fails
//! the batch; a model only missing on this host is not an error (other hosts declare its fingerprint).
//!
//! Validation: only a host that declares the batch's fingerprint and planner version takes the validation lease
//! (renewed during I/O and CPU work). It streams the whole immutable input through the bounded JSONL reader, which
//! checks the file's recorded length and SHA-256 at the end. Every line is planned on the planning threads like an
//! execution (slot `R`, native planner, the same budgets, numbers and semantics as online) and appended in bounded
//! chunks: byte range, line hash, cost, `result_bound` and `host_bound`. The seal follows only after every record.
//! Data errors (up to the issue limit, with line numbers) fail the batch; a lost lease or an unavailable store
//! stops the attempt without failing it, so another lease continues. Records are never fetched from URLs.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::BufReader;

use super::convert::{self, issue};
use super::dto::codes;
use super::lease::LeaseKeeper;
use super::planning::{input_issue, PlanFailure, PlanJob, PlannedRecord};
use super::service::Shared;
use super::store::{
    AppendOutcome, BatchIssue, CustomId, DiscoveryLease, ModelName, SealRequest, Sha256Digest, Store, StoreError,
    TargetResolution, ValidatedRecord, ValidationLease,
};
use super::validation::{InputError, JsonlReader};

/// Records appended per store call (the store accepts at most 1024).
const APPEND_CHUNK: usize = 512;

/// Discovery and validation loop of this host until shutdown.
pub(super) async fn run(shared: Arc<Shared>) {
    let min = Duration::from_millis(shared.settings.execution.poll_idle_min_ms);
    let max = Duration::from_millis(shared.settings.execution.poll_idle_max_ms);
    let mut idle = min;
    while !shared.stop.is_cancelled() {
        let worked = match shared.store() {
            Some(store) if shared.processing_open() => step(&shared, &store).await,
            _ => false,
        };
        idle = if worked { min } else { (idle * 2).min(max) };
        if !worked {
            tokio::select! {
                () = shared.stop.cancelled() => break,
                () = tokio::time::sleep(idle) => {}
            }
        }
    }
}

/// One discovery and one validation attempt; true when a lease was taken.
async fn step(shared: &Arc<Shared>, store: &Store) -> bool {
    let mut worked = false;
    let sent = Instant::now();
    match store.claim_discovery(&shared.worker).await {
        Ok(Some(lease)) => {
            worked = true;
            if let Err(e) = discover(shared, store, lease, sent).await {
                crate::warn!("batch discovery: {e}");
            }
        }
        Ok(None) => {}
        Err(e) => crate::warn!("batch discovery claim: {e}"),
    }
    let targets = shared.targets();
    if targets.is_empty() || shared.stop.is_cancelled() {
        return worked;
    }
    let sent = Instant::now();
    match store.claim_validation(&shared.worker, &targets).await {
        Ok(Some(lease)) => {
            worked = true;
            if let Err(e) = validate(shared, store, lease, sent).await {
                crate::warn!("batch validation: {e}");
            }
        }
        Ok(None) => {}
        Err(e) => crate::warn!("batch validation claim: {e}"),
    }
    worked
}

/// Why an attempt stopped without a decision about the batch.
#[derive(Debug, thiserror::Error)]
enum Abort {
    #[error("the lease was lost")]
    LeaseLost,
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("reading the input: {0}")]
    Input(String),
    #[error("planning: {0}")]
    Planning(String),
}

async fn fail(
    store: &Store,
    lease: &super::store::ClaimToken,
    issues: &[BatchIssue],
    truncated: bool,
) -> Result<(), Abort> {
    match store.fail_validation(lease, issues, truncated).await {
        Ok(_) => Ok(()),
        Err(StoreError::LeaseStale) => Err(Abort::LeaseLost),
        Err(e) => Err(e.into()),
    }
}

/// The first record of an input file, read through a bounded range.
enum FirstLine {
    Empty,
    Issue(&'static str, String),
    /// The record's `model` field (None: absent or not a string, i.e. the catalog's default model).
    Model(Option<String>),
}

async fn first_line(shared: &Shared, input: &super::store::InputFile, deadline: Instant) -> Result<FirstLine, Abort> {
    let limits = shared.planning.limits().input;
    let key = convert::artifact_key(&input.artifact).map_err(|e| Abort::Input(e.to_string()))?;
    let window = (limits.max_line_bytes as u64).saturating_add(2).min(input.bytes);
    let reader = shared
        .artifacts
        .read_range(&key, input.bytes, 0..window, None, Some(deadline))
        .await
        .map_err(|e| Abort::Input(e.to_string()))?;
    let mut lines =
        JsonlReader::new(BufReader::new(reader), super::validation::InputLimits { max_records: 1, ..limits });
    let first = match lines.next().await {
        Ok(Some(line)) => line,
        Ok(None) => return Ok(FirstLine::Empty),
        Err(e) => {
            return match input_issue(&e) {
                Some((code, message)) => Ok(FirstLine::Issue(code, message)),
                None => Err(Abort::Input(e.to_string())),
            }
        }
    };
    match shared.planning.model_of(first.bytes).await {
        Ok(model) => Ok(FirstLine::Model(model)),
        Err(PlanFailure::Data { code, message }) => Ok(FirstLine::Issue(code, message)),
        Err(PlanFailure::Interrupted(_)) => Err(Abort::LeaseLost),
        Err(PlanFailure::Unavailable(reason)) => Err(Abort::Planning(reason)),
    }
}

async fn discover(shared: &Arc<Shared>, store: &Store, lease: DiscoveryLease, sent: Instant) -> Result<(), Abort> {
    let token = lease.token;
    // The short discovery lease is not renewed: the first-line read ends before its local deadline.
    let safety = Duration::from_millis(shared.settings.leases.safety_margin_ms);
    let deadline = super::lease::local_deadline(sent, &lease.timing, safety);
    let model = match first_line(shared, &lease.input, deadline).await? {
        FirstLine::Empty => {
            return fail(store, &token, &[issue(codes::EMPTY_FILE, "the input file has no records", None)], false).await
        }
        FirstLine::Issue(code, message) => return fail(store, &token, &[issue(code, &message, Some(1))], false).await,
        FirstLine::Model(model) => model,
    };
    let name = match model.as_deref().map(ModelName::parse).transpose() {
        Ok(name) => name,
        Err(_) => {
            let message = "the model name is not a valid catalog name";
            return fail(store, &token, &[issue(codes::MODEL_NOT_FOUND, message, Some(1))], false).await;
        }
    };
    match store.resolve_target(&token, name.as_ref()).await {
        Ok(TargetResolution::Resolved(_)) => Ok(()),
        Ok(TargetResolution::ModelNotInCatalog) => {
            let message = "the model of the first record is not in the deployment catalog";
            fail(store, &token, &[issue(codes::MODEL_NOT_FOUND, message, Some(1))], false).await
        }
        Err(StoreError::LeaseStale) => Err(Abort::LeaseLost),
        Err(e) => Err(e.into()),
    }
}

/// Issues of a validation pass: bounded, with a truncation mark when more were found.
struct Issues {
    list: Vec<BatchIssue>,
    max: usize,
    truncated: bool,
}

impl Issues {
    fn push(&mut self, issue: BatchIssue) {
        if self.list.len() < self.max {
            self.list.push(issue);
        } else {
            self.truncated = true;
        }
    }

    fn full(&self) -> bool {
        self.truncated
    }
}

async fn validate(shared: &Arc<Shared>, store: &Store, lease: ValidationLease, sent: Instant) -> Result<(), Abort> {
    let Some(profile) = shared.profile_for(&lease.target.fingerprint) else {
        // The store only offers declared fingerprints; a profile removed meanwhile lets the lease lapse.
        return Err(Abort::Planning("the batch's profile is not served here".into()));
    };
    let leases = &shared.settings.leases;
    let keeper = LeaseKeeper::start(
        store.clone(),
        lease.token.clone(),
        &lease.timing,
        sent,
        Duration::from_millis(leases.validation_renew_ms),
        Duration::from_millis(leases.safety_margin_ms),
    );
    let default_model = default_model(shared, store, &lease, keeper.deadline()).await?;
    let object = convert::object(
        &lease.input.artifact,
        &super::store::StoredObject { bytes: lease.input.bytes, sha256: lease.input.sha256 },
    )
    .map_err(|e| Abort::Input(e.to_string()))?;
    let reader = shared.artifacts.read(&object, None).await.map_err(|e| Abort::Input(e.to_string()))?;
    let buffer = usize::try_from(shared.settings.host.upload_buffer_bytes).unwrap_or(1 << 20);
    let mut lines = JsonlReader::new(BufReader::with_capacity(buffer, reader), shared.planning.limits().input);
    let mut issues =
        Issues { list: Vec::new(), max: shared.settings.limits.max_validation_errors as usize, truncated: false };
    let mut pending: Vec<ValidatedRecord> = Vec::new();
    let in_flight_max = shared.settings.host.planning_workers as usize + 1;
    let mut in_flight: VecDeque<InFlight> = VecDeque::new();
    let mut total = 0u64;
    let mut eof = false;
    while !(eof && in_flight.is_empty()) {
        if keeper.lost() || shared.stop.is_cancelled() {
            abort_all(in_flight);
            return Err(Abort::LeaseLost);
        }
        if issues.full() {
            // The issue list is complete: no more lines are read, the jobs in flight finish.
            eof = true;
        }
        // Keep the planning threads busy while lines are read in order.
        if !eof && in_flight.len() < in_flight_max {
            match lines.next().await {
                Ok(Some(line)) => {
                    total = line.number;
                    let sha = line.sha256();
                    if line.number <= lease.ingested && issues.list.is_empty() {
                        // Appended by an earlier lease: its record is stored; only the file digest is still checked.
                        continue;
                    }
                    let job = PlanJob {
                        profile: Arc::clone(&profile),
                        keep_plan: false,
                        stop: Some(keeper.stop_token()),
                        default_model: default_model.clone(),
                        line: Vec::new(),
                    };
                    let (offset, length) = (line.offset, line.bytes.len() as u64);
                    let planning = Arc::clone(shared);
                    let job = PlanJob { line: line.bytes, ..job };
                    let handle = tokio::spawn(async move { planning.planning.plan(job).await });
                    in_flight.push_back((line.number, offset, length, sha, handle));
                    continue;
                }
                Ok(None) => eof = true,
                Err(InputError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidData => {
                    abort_all(in_flight);
                    let message = "the stored input file does not match its recorded SHA-256";
                    issues.push(issue(codes::INPUT_CHANGED, message, None));
                    return fail(store, keeper.token(), &issues.list, issues.truncated).await;
                }
                Err(e) => match input_issue(&e) {
                    Some((code, message)) => {
                        // Framing is lost after this line: report it and stop reading.
                        issues.push(issue(code, &message, Some(total + 1)));
                        eof = true;
                    }
                    None => {
                        abort_all(in_flight);
                        return Err(Abort::Input(e.to_string()));
                    }
                },
            }
        }
        let Some((line_no, offset, length, sha, handle)) = in_flight.pop_front() else { continue };
        let planned = match handle.await {
            Ok(result) => result,
            Err(_) => Err(PlanFailure::Unavailable("planning task stopped".into())),
        };
        match planned {
            Ok(record) if issues.list.is_empty() => {
                let custom_id =
                    CustomId::from_bytes(record.custom_id.into_bytes()).map_err(|e| Abort::Store(e.into()))?;
                pending.push(ValidatedRecord {
                    line_no,
                    custom_id,
                    offset,
                    length,
                    input_sha256: Sha256Digest(sha),
                    cost: record.cost,
                    result_bound: record.result_bound,
                    host_bound: record.host_bound,
                });
                if pending.len() >= APPEND_CHUNK {
                    append(store, &keeper, &mut pending, &mut issues).await?;
                }
            }
            Ok(_) => {}
            Err(PlanFailure::Data { code, message }) => issues.push(issue(code, &message, Some(line_no))),
            Err(PlanFailure::Interrupted(_)) => {
                abort_all(in_flight);
                return Err(Abort::LeaseLost);
            }
            Err(PlanFailure::Unavailable(reason)) => {
                abort_all(in_flight);
                return Err(Abort::Planning(reason));
            }
        }
    }
    if issues.list.is_empty() {
        append(store, &keeper, &mut pending, &mut issues).await?;
    }
    if total == 0 && issues.list.is_empty() {
        issues.push(issue(codes::EMPTY_FILE, "the input file has no records", None));
    }
    if !issues.list.is_empty() {
        return fail(store, keeper.token(), &issues.list, issues.truncated).await;
    }
    if keeper.lost() {
        return Err(Abort::LeaseLost);
    }
    let seal = SealRequest { total, input_sha256: lease.input.sha256, fingerprint: lease.target.fingerprint.clone() };
    match store.seal_input(keeper.token(), &seal).await {
        Ok(_) => Ok(()),
        Err(StoreError::LeaseStale) => Err(Abort::LeaseLost),
        Err(e) => Err(e.into()),
    }
}

/// A line being planned: line number, byte offset, length, line SHA-256 and the planning task.
type InFlight = (u64, u64, u64, [u8; 32], tokio::task::JoinHandle<Result<PlannedRecord, PlanFailure>>);

fn abort_all(in_flight: VecDeque<InFlight>) {
    // Planning jobs observe the lease's stop token; their results are not needed any more.
    for (.., handle) in in_flight {
        handle.abort();
    }
}

/// Append the pending records; a binary duplicate custom_id becomes an issue (and stops further appends).
async fn append(
    store: &Store,
    keeper: &LeaseKeeper,
    pending: &mut Vec<ValidatedRecord>,
    issues: &mut Issues,
) -> Result<(), Abort> {
    for chunk in pending.chunks(APPEND_CHUNK) {
        match store.append_validated(keeper.token(), chunk).await {
            Ok(AppendOutcome::Appended { .. }) => {}
            Ok(AppendOutcome::DuplicateCustomId { line_no, first_line_no }) => {
                let message = format!("custom_id repeats the custom_id of line {first_line_no}");
                issues.push(issue(codes::DUPLICATE_CUSTOM_ID, &message, Some(line_no)));
                break;
            }
            Err(StoreError::LeaseStale) => return Err(Abort::LeaseLost),
            Err(e) => return Err(e.into()),
        }
    }
    pending.clear();
    Ok(())
}

/// The default model of the batch's pinned catalog revision. Discovery resolved the target from the first record:
/// when that record names no model the target is the pinned default; otherwise read that immutable revision.
async fn default_model(
    shared: &Shared,
    store: &Store,
    lease: &ValidationLease,
    deadline: Instant,
) -> Result<Option<String>, Abort> {
    if let FirstLine::Model(None) = first_line(shared, &lease.input, deadline).await? {
        return Ok(Some(lease.target.model_name.as_str().to_string()));
    }
    let batch = match store.get_batch(&lease.token.batch.owner, &lease.token.batch.id).await {
        Ok(b) => b,
        Err(StoreError::NotFound) => return Err(Abort::LeaseLost),
        Err(e) => return Err(e.into()),
    };
    let catalog = store
        .catalog_revision(batch.catalog_revision)
        .await?
        .ok_or(StoreError::Corrupt("pinned catalog revision missing"))?;
    Ok(Some(catalog.spec.default_model.as_str().to_string()))
}
