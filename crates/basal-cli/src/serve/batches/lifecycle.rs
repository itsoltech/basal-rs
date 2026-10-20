//! Lifecycle supervisor: bounded steps over expiry, drains, settlements, expired claims, finalization, file and
//! session GC, post-publication cleanup and metadata retention, repeated until shutdown.
//!
//! Every step is one short store operation (or one bounded page); shutdown is observed between operations, so no
//! store future is dropped during cooperative shutdown. Object deletion always precedes the transaction that releases its bytes, and
//! happens only for candidates the store returned (their grace or I/O guard has passed).
//!
//! One separate, service-owned finalizer streams the accepted, immutable result lines in line order into fresh export objects. Expiry lines
//! are written with the store's own serializer (`write_expired_line`); cancelled records get no line. The writers
//! follow the renewed lease deadline, and the files are published only after both objects are durable. A retry of
//! an earlier generation first waits until the store reports its writer stopped (lease end plus I/O guard). It
//! then deletes that generation's objects and adopts its reservation (`adopt_export`). The completion reserve
//! taken at create covers every byte, so no new quota is needed.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::artifacts::{ArtifactError, ArtifactKey, ArtifactKind, ArtifactWriter};
use super::convert;
use super::dto;
use super::lease::LeaseKeeper;
use super::service::Shared;
use super::store::{
    self, ExportItem, ExportKeys, ExportPayload, FinalizationClaim, FinalizationLease, OperationId, PublishRequest,
    PublishedFile, Store, StoreError,
};

/// Rows or candidates handled per store call.
const PAGE: u32 = 64;
/// Export items read per store call.
const EXPORT_PAGE: u32 = 256;
/// Retries of a publication whose commit outcome is unknown (same operation id).
const PUBLISH_RETRIES: u32 = 5;
/// Interval of the policy and catalog re-check.
const POLICY_CHECK: Duration = Duration::from_secs(60);

pub(super) async fn run(shared: Arc<Shared>) {
    let idle = Duration::from_millis(shared.settings.execution.poll_idle_max_ms);
    let mut last_policy = Instant::now();
    while !shared.stop.is_cancelled() {
        let mut worked = false;
        if let Some(store) = shared.store() {
            if last_policy.elapsed() >= POLICY_CHECK {
                shared.check_policy(&store).await;
                shared.set_artifacts(shared.artifacts.probe().await);
                last_policy = Instant::now();
            }
            match step(&shared, &store).await {
                Ok(w) => worked = w,
                Err(e) => crate::warn!("batch lifecycle: {e}"),
            }
        }
        if !worked {
            tokio::select! {
                () = shared.stop.cancelled() => break,
                () = tokio::time::sleep(idle) => {}
            }
        }
    }
}

/// One persistent export worker: at most one export on this process, independent of all control duties.
/// The service owns and joins this task alongside `run`, including during graceful shutdown.
pub(super) async fn run_finalizer(shared: Arc<Shared>) {
    let idle = Duration::from_millis(shared.settings.execution.poll_idle_max_ms);
    while !shared.stop.is_cancelled() {
        let mut worked = false;
        if let Some(store) = shared.store() {
            match finalize(&shared, &store).await {
                Ok(w) => worked = w,
                Err(e) => crate::warn!("batch finalizer: {e}"),
            }
        }
        if !worked {
            tokio::select! {
                () = shared.stop.cancelled() => break,
                () = tokio::time::sleep(idle) => {}
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum StepError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("artifacts: {0}")]
    Artifacts(#[from] ArtifactError),
}

/// One round of every duty; true when anything was done.
async fn step(shared: &Arc<Shared>, store: &Store) -> Result<bool, StepError> {
    let mut worked = false;
    for batch in store.due_expirations(PAGE).await? {
        store.expire_batch(&batch).await?;
        worked = true;
    }
    for batch in store.draining_batches(PAGE).await? {
        let progress = store.advance_drain(&batch, EXPORT_PAGE).await?;
        worked |= progress.materialized > 0;
    }
    for owner in store.owners_with_pending_settlements(PAGE).await? {
        worked |= store.settle_pending(&owner, EXPORT_PAGE).await?.settled > 0;
    }
    worked |= store.recover_expired_claims(PAGE).await? > 0;
    if shared.stop.is_cancelled() {
        return Ok(worked);
    }
    worked |= store.expire_files(PAGE).await? > 0;
    for candidate in store.file_gc_candidates(PAGE).await? {
        delete(shared, &candidate.artifact).await?;
        worked |= store.purge_file(&candidate.owner, &candidate.file_id).await?;
    }
    for candidate in store.session_gc_candidates(PAGE).await? {
        for artifact in &candidate.artifacts {
            delete(shared, artifact).await?;
        }
        worked |= store.release_session(&candidate.owner, &candidate.session).await?;
    }
    for batch in store.cleanup_batches(PAGE).await? {
        let page = store.cleanup_page(&batch, EXPORT_PAGE).await?;
        for artifact in &page.artifacts {
            delete(shared, artifact).await?;
        }
        store.cleanup_commit(&batch, page.through_line_no).await?;
        worked = true;
    }
    worked |= store.purge_metadata(PAGE).await? > 0;
    Ok(worked)
}

/// Delete an object named by the store. A reference this server cannot have written is reported, not deleted.
async fn delete(shared: &Shared, artifact: &store::ArtifactRef) -> Result<(), StepError> {
    match convert::artifact_key(artifact) {
        Ok(key) => Ok(shared.artifacts.delete(&key).await?),
        Err(e) => Err(StepError::Artifacts(e)),
    }
}

fn export_keys() -> Result<(ArtifactKey, ArtifactKey, ExportKeys), StoreError> {
    let output = ArtifactKey::generate(ArtifactKind::Export);
    let error = ArtifactKey::generate(ArtifactKind::Export);
    let keys = ExportKeys { output: convert::artifact_ref(&output)?, error: convert::artifact_ref(&error)? };
    Ok((output, error, keys))
}

/// Take one finalization lease, if any, and export the batch.
async fn finalize(shared: &Arc<Shared>, store: &Store) -> Result<bool, StepError> {
    let (output, error, keys) = export_keys()?;
    let sent = Instant::now();
    let lease = match store.claim_finalization(&shared.worker, &keys).await? {
        FinalizationClaim::Idle => return Ok(false),
        FinalizationClaim::Ready(lease) => lease,
        FinalizationClaim::NeedsFencing { token, timing, previous } => {
            if timing.server_now < previous.writer_stopped_after {
                // The earlier writer may still run; its reservation stays put until the I/O guard has passed.
                return Ok(true);
            }
            delete(shared, &previous.keys.output).await?;
            delete(shared, &previous.keys.error).await?;
            match store.adopt_export(&token, &previous.session, &keys).await {
                Ok(lease) => lease,
                Err(StoreError::FencingPending | StoreError::LeaseStale) => return Ok(true),
                Err(e) => return Err(e.into()),
            }
        }
    };
    match export(shared, store, lease, (output, error), sent).await {
        Ok(()) | Err(ExportStop::LeaseLost) => Ok(true),
        Err(ExportStop::Failed(e)) => Err(e),
    }
}

enum ExportStop {
    LeaseLost,
    Failed(StepError),
}

impl From<StoreError> for ExportStop {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::LeaseStale => Self::LeaseLost,
            other => Self::Failed(StepError::Store(other)),
        }
    }
}

impl From<ArtifactError> for ExportStop {
    fn from(e: ArtifactError) -> Self {
        Self::Failed(StepError::Artifacts(e))
    }
}

/// A lazily created export object (a class without lines publishes no file).
struct ExportFile {
    key: ArtifactKey,
    writer: Option<ArtifactWriter>,
}

impl ExportFile {
    async fn write(&mut self, shared: &Shared, keeper: &LeaseKeeper, bytes: &[u8]) -> Result<(), ExportStop> {
        if keeper.lost() || shared.stop.is_cancelled() {
            return Err(ExportStop::LeaseLost);
        }
        if self.writer.is_none() {
            let writer = shared.artifacts.create(self.key, u64::MAX / 4, keeper.deadline(), None).await?;
            self.writer = Some(writer);
        }
        let Some(writer) = self.writer.as_mut() else { return Err(ExportStop::LeaseLost) };
        writer.set_deadline(keeper.deadline());
        writer.write(bytes).await?;
        Ok(())
    }

    async fn finish(self, keeper: &LeaseKeeper, filename: String) -> Result<Option<PublishedFile>, ExportStop> {
        let Some(mut writer) = self.writer else { return Ok(None) };
        writer.set_deadline(keeper.deadline());
        let object = writer.finish().await?;
        Ok(Some(PublishedFile { object: convert::stored(&object), filename }))
    }
}

async fn export(
    shared: &Arc<Shared>,
    store: &Store,
    lease: FinalizationLease,
    (output, error): (ArtifactKey, ArtifactKey),
    sent: Instant,
) -> Result<(), ExportStop> {
    let leases = &shared.settings.leases;
    let keeper = LeaseKeeper::start(
        store.clone(),
        lease.token.clone(),
        &lease.timing,
        sent,
        Duration::from_millis(leases.finalization_renew_ms),
        Duration::from_millis(leases.safety_margin_ms),
    );
    if lease.keys.output != convert::artifact_ref(&output)? || lease.keys.error != convert::artifact_ref(&error)? {
        return Err(ExportStop::Failed(StepError::Store(StoreError::Corrupt("export lease names other keys"))));
    }
    let mut out = ExportFile { key: output, writer: None };
    let mut err = ExportFile { key: error, writer: None };
    // Lines written per class; completed and expired exports cover every record, cancelled ones may not.
    let mut lines = 0u64;
    let chunk = usize::try_from(shared.settings.host.upload_buffer_bytes).unwrap_or(1 << 20);
    let mut after = 0u64;
    loop {
        if keeper.lost() || shared.stop.is_cancelled() {
            return Err(ExportStop::LeaseLost);
        }
        let items = store.read_export_items(keeper.token(), after, EXPORT_PAGE).await?;
        let Some(last) = items.last() else { break };
        after = line_of(last);
        for item in items {
            match item {
                ExportItem::Output { payload, .. } => {
                    copy(shared, &keeper, &mut out, payload, chunk).await?;
                    lines += 1;
                }
                ExportItem::Error { payload, .. } => {
                    copy(shared, &keeper, &mut err, payload, chunk).await?;
                    lines += 1;
                }
                ExportItem::Expired { custom_id, .. } => {
                    let mut line = Vec::new();
                    store::write_expired_line(&mut line, &dto::new_request_id(), &custom_id)
                        .map_err(|e| ExportStop::Failed(StepError::Store(e.into())))?;
                    err.write(shared, &keeper, &line).await?;
                    lines += 1;
                }
                ExportItem::Cancelled { .. } => {}
            }
        }
    }
    if keeper.lost() || shared.stop.is_cancelled() {
        return Err(ExportStop::LeaseLost);
    }
    let complete = matches!(lease.target, store::TerminalStatus::Completed | store::TerminalStatus::Expired);
    if complete && lines != lease.total {
        // Never publish a partial export of a batch whose every record has a line.
        return Err(ExportStop::Failed(StepError::Store(StoreError::Corrupt(
            "export lines differ from the batch total",
        ))));
    }
    let batch = lease.token.batch.id.as_str().to_string();
    let request = PublishRequest {
        output: out.finish(&keeper, dto::result_filename(&batch, true)).await?,
        error: err.finish(&keeper, dto::result_filename(&batch, false)).await?,
    };
    if keeper.lost() || shared.stop.is_cancelled() {
        return Err(ExportStop::LeaseLost);
    }
    // One operation id for every retry of this publication: an unknown commit is reconciled, never repeated.
    let operation = OperationId::new_internal();
    let mut attempt = 0;
    loop {
        match store.publish_files(&operation, keeper.token(), &request).await {
            Ok(_) => return Ok(()),
            Err(StoreError::CommitUnknown(_) | StoreError::Busy(_) | StoreError::Unavailable(_))
                if attempt < PUBLISH_RETRIES && !keeper.lost() =>
            {
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(200 * u64::from(attempt))).await;
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn line_of(item: &ExportItem) -> u64 {
    match item {
        ExportItem::Output { line_no, .. }
        | ExportItem::Error { line_no, .. }
        | ExportItem::Expired { line_no, .. }
        | ExportItem::Cancelled { line_no } => *line_no,
    }
}

/// Copy one accepted result line (inline, or a stored object verified by length and SHA-256) into an export file.
async fn copy(
    shared: &Shared,
    keeper: &LeaseKeeper,
    file: &mut ExportFile,
    payload: ExportPayload,
    chunk: usize,
) -> Result<(), ExportStop> {
    match payload {
        ExportPayload::Inline(bytes) => file.write(shared, keeper, &bytes).await,
        ExportPayload::Stored { artifact, object } => {
            use tokio::io::AsyncReadExt;
            let object = convert::object(&artifact, &object)?;
            let mut reader = shared.artifacts.read(&object, Some(keeper.deadline())).await?;
            let mut buf = vec![0u8; chunk.clamp(4096, 1 << 22)];
            loop {
                let n = reader.read(&mut buf).await.map_err(|e| {
                    ExportStop::Failed(StepError::Artifacts(ArtifactError::from_read("read result", e)))
                })?;
                if n == 0 {
                    return Ok(());
                }
                file.write(shared, keeper, &buf[..n]).await?;
            }
        }
    }
}
