-- Batch API store, schema version 1 (PostgreSQL).
-- Every byte reserved or used in batch_quota has exactly one owner row: batches (completion and export reserves,
-- stored issues), batch_items (source results and their row charge via batches), batch_settlements (execution
-- reserves of open claims), artifact_sessions (uploads, orphaned results, exports) or batch_files.
-- Integers are milliseconds since the epoch for *_at columns; booleans are 0/1.

CREATE TABLE batch_quota_policy (
    revision BIGINT PRIMARY KEY,
    spec_sha256 BYTEA NOT NULL,
    object_charge_bytes BIGINT NOT NULL CHECK (object_charge_bytes > 0),
    item_overhead_bytes BIGINT NOT NULL CHECK (item_overhead_bytes > 0),
    max_upload_bytes BIGINT NOT NULL CHECK (max_upload_bytes > 0),
    max_records_per_batch BIGINT NOT NULL CHECK (max_records_per_batch > 0),
    inst_max_bytes BIGINT NOT NULL,
    inst_max_active_batches BIGINT NOT NULL,
    inst_max_retained_batches BIGINT NOT NULL,
    inst_max_files BIGINT NOT NULL,
    inst_max_open_upload_sessions BIGINT NOT NULL,
    installed_at BIGINT NOT NULL
);

CREATE TABLE batch_quota_limits (
    revision BIGINT NOT NULL REFERENCES batch_quota_policy (revision),
    owner_id TEXT NOT NULL,
    max_bytes BIGINT NOT NULL,
    max_active_batches BIGINT NOT NULL,
    max_retained_batches BIGINT NOT NULL,
    max_files BIGINT NOT NULL,
    max_open_upload_sessions BIGINT NOT NULL,
    PRIMARY KEY (revision, owner_id)
);

CREATE TABLE batch_catalog_revisions (
    revision BIGINT PRIMARY KEY,
    spec_sha256 BYTEA NOT NULL,
    default_model TEXT NOT NULL,
    installed_at BIGINT NOT NULL
);

CREATE TABLE batch_catalog (
    revision BIGINT NOT NULL REFERENCES batch_catalog_revisions (revision),
    model_name TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    planner_version TEXT NOT NULL,
    PRIMARY KEY (revision, model_name)
);

CREATE TABLE batch_quota (
    owner_id TEXT PRIMARY KEY,
    used_bytes BIGINT NOT NULL DEFAULT 0 CHECK (used_bytes >= 0),
    reserved_bytes BIGINT NOT NULL DEFAULT 0 CHECK (reserved_bytes >= 0),
    active_batches BIGINT NOT NULL DEFAULT 0 CHECK (active_batches >= 0),
    retained_batches BIGINT NOT NULL DEFAULT 0 CHECK (retained_batches >= 0),
    files BIGINT NOT NULL DEFAULT 0 CHECK (files >= 0),
    open_sessions BIGINT NOT NULL DEFAULT 0 CHECK (open_sessions >= 0),
    version BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE batch_files (
    owner_id TEXT NOT NULL,
    id TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose IN ('batch', 'batch_output', 'batch_error')),
    filename TEXT NOT NULL,
    artifact_ref TEXT NOT NULL,
    bytes BIGINT NOT NULL CHECK (bytes >= 0),
    charged_bytes BIGINT NOT NULL CHECK (charged_bytes >= 0),
    sha256 BYTEA NOT NULL,
    created_at BIGINT NOT NULL,
    expires_at BIGINT,
    state TEXT NOT NULL CHECK (state IN ('active', 'deleted')),
    pin_count BIGINT NOT NULL DEFAULT 0 CHECK (pin_count >= 0),
    deleted_at BIGINT,
    gc_after BIGINT,
    batch_id TEXT,
    PRIMARY KEY (owner_id, id)
);

CREATE INDEX batch_files_list ON batch_files (owner_id, created_at, id);
CREATE INDEX batch_files_expiry ON batch_files (state, expires_at);
CREATE INDEX batch_files_gc ON batch_files (state, gc_after);

CREATE TABLE batches (
    owner_id TEXT NOT NULL,
    id TEXT NOT NULL,
    input_file_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    catalog_revision BIGINT NOT NULL,
    model_name TEXT,
    fingerprint TEXT,
    planner_version TEXT,
    status TEXT NOT NULL CHECK (status IN ('validating', 'in_progress', 'draining', 'finalizing',
                                           'completed', 'failed', 'expired', 'cancelled')),
    terminal_target TEXT CHECK (terminal_target IN ('completed', 'failed', 'expired', 'cancelled')),
    created_at BIGINT NOT NULL,
    expires_at BIGINT NOT NULL,
    in_progress_at BIGINT,
    finalizing_at BIGINT,
    cancelling_at BIGINT,
    terminal_at BIGINT,
    control_epoch BIGINT NOT NULL DEFAULT 0,
    mutation_seq BIGINT NOT NULL DEFAULT 0,
    sealed BIGINT NOT NULL DEFAULT 0,
    total BIGINT NOT NULL DEFAULT 0,
    completed BIGINT NOT NULL DEFAULT 0,
    failed BIGINT NOT NULL DEFAULT 0,
    cancelled BIGINT NOT NULL DEFAULT 0,
    expired BIGINT NOT NULL DEFAULT 0,
    queued BIGINT NOT NULL DEFAULT 0 CHECK (queued >= 0),
    accounting_pending BIGINT NOT NULL DEFAULT 0 CHECK (accounting_pending >= 0),
    object_charge BIGINT NOT NULL,
    item_overhead BIGINT NOT NULL,
    max_records BIGINT NOT NULL,
    reserve_records BIGINT NOT NULL,
    staged_reserve BIGINT NOT NULL DEFAULT 0,
    completion_reserved BIGINT NOT NULL CHECK (completion_reserved >= 0),
    export_reserved BIGINT NOT NULL DEFAULT 0 CHECK (export_reserved >= 0),
    errors_bytes BIGINT NOT NULL DEFAULT 0,
    error_count BIGINT NOT NULL DEFAULT 0,
    errors_truncated BIGINT NOT NULL DEFAULT 0,
    output_file_id TEXT,
    error_file_id TEXT,
    released BIGINT NOT NULL DEFAULT 0,
    cleanup_state TEXT NOT NULL DEFAULT 'none' CHECK (cleanup_state IN ('none', 'pending', 'done')),
    metadata_expires_at BIGINT,
    metadata TEXT NOT NULL,
    output_expires_after_secs BIGINT,
    phase_kind TEXT,
    phase_claim_id TEXT,
    phase_generation BIGINT NOT NULL DEFAULT 0,
    phase_lease_until BIGINT,
    sched_credit BIGINT NOT NULL DEFAULT 0,
    sched_turn BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (owner_id, id)
);

CREATE INDEX batches_list ON batches (owner_id, created_at, id);
CREATE INDEX batches_active ON batches (fingerprint, status, owner_id, id);
CREATE INDEX batches_phase ON batches (status, phase_lease_until);
CREATE INDEX batches_deadline ON batches (status, expires_at);
CREATE INDEX batches_cleanup ON batches (cleanup_state, terminal_at);
CREATE INDEX batches_metadata_expiry ON batches (metadata_expires_at);

CREATE TABLE batch_errors (
    owner_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    seq BIGINT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    param TEXT,
    line_no BIGINT,
    PRIMARY KEY (owner_id, batch_id, seq)
);

CREATE TABLE batch_items (
    owner_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    line_no BIGINT NOT NULL CHECK (line_no >= 1),
    custom_id BYTEA NOT NULL,
    input_offset BIGINT NOT NULL,
    input_length BIGINT NOT NULL,
    input_sha256 BYTEA NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'claimed', 'succeeded', 'failed', 'cancelled', 'expired')),
    generation BIGINT NOT NULL DEFAULT 0,
    claim_id TEXT,
    failure_attempts BIGINT NOT NULL DEFAULT 0,
    requires_solo BIGINT NOT NULL DEFAULT 0,
    cost BIGINT NOT NULL,
    record_reserve BIGINT NOT NULL,
    result_bound BIGINT NOT NULL,
    host_bound BIGINT NOT NULL,
    result_kind TEXT CHECK (result_kind IN ('output', 'error')),
    result_inline BYTEA,
    result_ref TEXT,
    result_bytes BIGINT,
    result_sha256 BYTEA,
    accepted_at BIGINT,
    quota_used BIGINT NOT NULL DEFAULT 0 CHECK (quota_used >= 0),
    PRIMARY KEY (owner_id, batch_id, line_no)
);

CREATE UNIQUE INDEX batch_items_custom_id ON batch_items (owner_id, batch_id, custom_id);
CREATE INDEX batch_items_ready ON batch_items (owner_id, batch_id, state, line_no);
CREATE INDEX batch_items_claim ON batch_items (claim_id) WHERE claim_id IS NOT NULL;

CREATE TABLE batch_claims (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('discovery', 'validation', 'execution', 'finalization')),
    worker_id TEXT NOT NULL,
    generation BIGINT NOT NULL,
    owner_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    control_epoch BIGINT NOT NULL,
    lease_until BIGINT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active', 'released', 'expired', 'finished')),
    release_reason TEXT,
    created_at BIGINT NOT NULL
);

CREATE INDEX batch_claims_expiry ON batch_claims (kind, state, lease_until);
CREATE INDEX batch_claims_batch ON batch_claims (owner_id, batch_id);

-- One row per claimed item: holds the execution reserve (2 × result_bound) and the reserved DRR cost until it is
-- accounted. kind is NULL while the claim is open and written exactly once; accounted flips 0 → 1 exactly once.
CREATE TABLE batch_settlements (
    claim_id TEXT NOT NULL,
    line_no BIGINT NOT NULL,
    owner_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    shard TEXT NOT NULL,
    reserved_bytes BIGINT NOT NULL CHECK (reserved_bytes >= 0),
    estimated_cost BIGINT NOT NULL,
    kind TEXT CHECK (kind IN ('accepted', 'requeued', 'released', 'revoked', 'expired')),
    operation_id TEXT,
    accepted_bytes BIGINT NOT NULL DEFAULT 0,
    orphan_session_id TEXT,
    actual_cost BIGINT NOT NULL DEFAULT 0,
    accounted BIGINT NOT NULL DEFAULT 0,
    decided_at BIGINT,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (claim_id, line_no)
);

CREATE INDEX batch_settlements_pending ON batch_settlements (owner_id, accounted, shard);
CREATE INDEX batch_settlements_batch ON batch_settlements (owner_id, batch_id);

CREATE TABLE batch_operations (
    owner_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('create', 'commit', 'publish')),
    operation_id TEXT NOT NULL,
    request_sha256 BYTEA NOT NULL,
    batch_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (owner_id, kind, operation_id)
);

CREATE INDEX batch_operations_batch ON batch_operations (owner_id, batch_id);

CREATE TABLE batch_schedule (
    shard TEXT PRIMARY KEY,
    turn BIGINT NOT NULL DEFAULT 0,
    cursor_owner TEXT,
    quantum BIGINT NOT NULL,
    credit_cap BIGINT NOT NULL,
    version BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE batch_deficits (
    shard TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    credit BIGINT NOT NULL DEFAULT 0,
    last_turn BIGINT NOT NULL DEFAULT 0,
    reserved_cost BIGINT NOT NULL DEFAULT 0,
    batch_turn BIGINT NOT NULL DEFAULT 0,
    batch_cursor TEXT,
    quota_blocked_version BIGINT,
    blocked_until BIGINT,
    PRIMARY KEY (shard, owner_id)
);

CREATE TABLE artifact_sessions (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('upload', 'result', 'export')),
    artifact_ref TEXT NOT NULL,
    secondary_ref TEXT,
    generation BIGINT NOT NULL DEFAULT 0,
    phase TEXT NOT NULL CHECK (phase IN ('open', 'abandoned', 'orphaned')),
    lease_until BIGINT NOT NULL,
    reserved_bytes BIGINT NOT NULL DEFAULT 0 CHECK (reserved_bytes >= 0),
    max_bytes BIGINT NOT NULL DEFAULT 0,
    file_id TEXT,
    batch_id TEXT,
    claim_id TEXT,
    line_no BIGINT,
    filename TEXT,
    expires_after_secs BIGINT,
    created_at BIGINT NOT NULL
);

CREATE INDEX artifact_sessions_lease ON artifact_sessions (kind, lease_until);
CREATE INDEX artifact_sessions_phase ON artifact_sessions (phase, lease_until);
CREATE INDEX artifact_sessions_batch ON artifact_sessions (owner_id, batch_id);
CREATE INDEX artifact_sessions_claim ON artifact_sessions (claim_id, line_no) WHERE claim_id IS NOT NULL;
