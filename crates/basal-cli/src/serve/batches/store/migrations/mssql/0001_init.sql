-- Batch API store, schema version 1 (SQL Server 2019 or later).
-- Same tables, keys and owner rows as the PostgreSQL schema. Every byte reserved or used in batch_quota has exactly
-- one owner row: batches (completion and export reserves, stored issues), batch_items (source results and their
-- row charge via batches), batch_settlements (execution reserves of open claims), artifact_sessions (uploads,
-- orphaned results, exports) or batch_files.
-- Integers are milliseconds since the epoch for *_at columns; booleans are 0/1.
-- Identifiers and enum columns use the binary collation Latin1_General_100_BIN2: comparisons are case and accent
-- sensitive and order by code point, as in Rust. SQL Server ignores trailing spaces (strings) and trailing zero
-- bytes (binaries) in `=`, so keys over values that may end in them also include the stored length
-- (*_len persisted computed columns) and lookups compare DATALENGTH too (Dialect::exact_eq).
-- custom_id is VARBINARY(256): the exact UTF-8 bytes, no text collation.
-- Index keys stay below 900 bytes (clustered) and 1700 bytes (nonclustered).
-- Filtered indexes and indexed computed columns require the SET options that every adapter session sets
-- (ANSI_NULLS, ANSI_PADDING, ANSI_WARNINGS, ARITHABORT, CONCAT_NULL_YIELDS_NULL, QUOTED_IDENTIFIER ON,
-- NUMERIC_ROUNDABORT OFF).

CREATE TABLE batch_quota_policy (
    revision BIGINT NOT NULL PRIMARY KEY,
    spec_sha256 VARBINARY(32) NOT NULL,
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
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    max_bytes BIGINT NOT NULL,
    max_active_batches BIGINT NOT NULL,
    max_retained_batches BIGINT NOT NULL,
    max_files BIGINT NOT NULL,
    max_open_upload_sessions BIGINT NOT NULL,
    PRIMARY KEY (revision, owner_id)
);

CREATE TABLE batch_catalog_revisions (
    revision BIGINT NOT NULL PRIMARY KEY,
    spec_sha256 VARBINARY(32) NOT NULL,
    default_model NVARCHAR(256) COLLATE Latin1_General_100_BIN2 NOT NULL,
    installed_at BIGINT NOT NULL
);

CREATE TABLE batch_catalog (
    revision BIGINT NOT NULL REFERENCES batch_catalog_revisions (revision),
    model_name NVARCHAR(256) COLLATE Latin1_General_100_BIN2 NOT NULL,
    model_name_len AS CAST(DATALENGTH(model_name) AS INT) PERSISTED NOT NULL,
    fingerprint NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    planner_version NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    PRIMARY KEY (revision, model_name, model_name_len)
);

CREATE TABLE batch_quota (
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL PRIMARY KEY,
    used_bytes BIGINT NOT NULL DEFAULT 0 CHECK (used_bytes >= 0),
    reserved_bytes BIGINT NOT NULL DEFAULT 0 CHECK (reserved_bytes >= 0),
    active_batches BIGINT NOT NULL DEFAULT 0 CHECK (active_batches >= 0),
    retained_batches BIGINT NOT NULL DEFAULT 0 CHECK (retained_batches >= 0),
    files BIGINT NOT NULL DEFAULT 0 CHECK (files >= 0),
    open_sessions BIGINT NOT NULL DEFAULT 0 CHECK (open_sessions >= 0),
    version BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE batch_files (
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    purpose NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL
        CHECK (purpose IN ('batch', 'batch_output', 'batch_error')),
    filename NVARCHAR(255) NOT NULL,
    artifact_ref NVARCHAR(512) COLLATE Latin1_General_100_BIN2 NOT NULL,
    bytes BIGINT NOT NULL CHECK (bytes >= 0),
    charged_bytes BIGINT NOT NULL CHECK (charged_bytes >= 0),
    sha256 VARBINARY(32) NOT NULL,
    created_at BIGINT NOT NULL,
    expires_at BIGINT,
    state NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL CHECK (state IN ('active', 'deleted')),
    pin_count BIGINT NOT NULL DEFAULT 0 CHECK (pin_count >= 0),
    deleted_at BIGINT,
    gc_after BIGINT,
    batch_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    PRIMARY KEY (owner_id, id)
);

CREATE INDEX batch_files_list ON batch_files (owner_id, created_at, id);
CREATE INDEX batch_files_expiry ON batch_files (state, expires_at);
CREATE INDEX batch_files_gc ON batch_files (state, gc_after);

CREATE TABLE batches (
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    input_file_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    endpoint NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    catalog_revision BIGINT NOT NULL,
    model_name NVARCHAR(256) COLLATE Latin1_General_100_BIN2,
    fingerprint NVARCHAR(128) COLLATE Latin1_General_100_BIN2,
    planner_version NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    status NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL
        CHECK (status IN ('validating', 'in_progress', 'draining', 'finalizing',
                          'completed', 'failed', 'expired', 'cancelled')),
    terminal_target NVARCHAR(16) COLLATE Latin1_General_100_BIN2
        CHECK (terminal_target IN ('completed', 'failed', 'expired', 'cancelled')),
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
    output_file_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    error_file_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    released BIGINT NOT NULL DEFAULT 0,
    cleanup_state NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL DEFAULT 'none'
        CHECK (cleanup_state IN ('none', 'pending', 'done')),
    metadata_expires_at BIGINT,
    metadata NVARCHAR(MAX) NOT NULL,
    output_expires_after_secs BIGINT,
    phase_kind NVARCHAR(16) COLLATE Latin1_General_100_BIN2,
    phase_claim_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
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
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    batch_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    seq BIGINT NOT NULL,
    code NVARCHAR(64) NOT NULL,
    message NVARCHAR(256) NOT NULL,
    param NVARCHAR(64),
    line_no BIGINT,
    PRIMARY KEY (owner_id, batch_id, seq)
);

CREATE TABLE batch_items (
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    batch_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    line_no BIGINT NOT NULL CHECK (line_no >= 1),
    custom_id VARBINARY(256) NOT NULL,
    custom_id_len AS CAST(DATALENGTH(custom_id) AS INT) PERSISTED NOT NULL,
    input_offset BIGINT NOT NULL,
    input_length BIGINT NOT NULL,
    input_sha256 VARBINARY(32) NOT NULL,
    state NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL
        CHECK (state IN ('pending', 'claimed', 'succeeded', 'failed', 'cancelled', 'expired')),
    generation BIGINT NOT NULL DEFAULT 0,
    claim_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    failure_attempts BIGINT NOT NULL DEFAULT 0,
    requires_solo BIGINT NOT NULL DEFAULT 0,
    cost BIGINT NOT NULL,
    record_reserve BIGINT NOT NULL,
    result_bound BIGINT NOT NULL,
    host_bound BIGINT NOT NULL,
    result_kind NVARCHAR(16) COLLATE Latin1_General_100_BIN2 CHECK (result_kind IN ('output', 'error')),
    result_inline VARBINARY(MAX),
    result_ref NVARCHAR(512) COLLATE Latin1_General_100_BIN2,
    result_bytes BIGINT,
    result_sha256 VARBINARY(32),
    accepted_at BIGINT,
    quota_used BIGINT NOT NULL DEFAULT 0 CHECK (quota_used >= 0),
    PRIMARY KEY (owner_id, batch_id, line_no)
);

CREATE UNIQUE INDEX batch_items_custom_id ON batch_items (owner_id, batch_id, custom_id, custom_id_len);
CREATE INDEX batch_items_ready ON batch_items (owner_id, batch_id, state, line_no);
CREATE INDEX batch_items_claim ON batch_items (claim_id) WHERE claim_id IS NOT NULL;

CREATE TABLE batch_claims (
    id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL PRIMARY KEY,
    kind NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL
        CHECK (kind IN ('discovery', 'validation', 'execution', 'finalization')),
    worker_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    generation BIGINT NOT NULL,
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    batch_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    control_epoch BIGINT NOT NULL,
    lease_until BIGINT NOT NULL,
    state NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL
        CHECK (state IN ('active', 'released', 'expired', 'finished')),
    release_reason NVARCHAR(32) COLLATE Latin1_General_100_BIN2,
    created_at BIGINT NOT NULL
);

CREATE INDEX batch_claims_expiry ON batch_claims (kind, state, lease_until);
CREATE INDEX batch_claims_batch ON batch_claims (owner_id, batch_id);

-- One row per claimed item: holds the execution reserve (2 × result_bound) and the reserved DRR cost until it is
-- accounted. kind is NULL while the claim is open and written exactly once; accounted flips 0 → 1 exactly once.
CREATE TABLE batch_settlements (
    claim_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    line_no BIGINT NOT NULL,
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    batch_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    shard NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    reserved_bytes BIGINT NOT NULL CHECK (reserved_bytes >= 0),
    estimated_cost BIGINT NOT NULL,
    kind NVARCHAR(16) COLLATE Latin1_General_100_BIN2
        CHECK (kind IN ('accepted', 'requeued', 'released', 'revoked', 'expired')),
    operation_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2,
    accepted_bytes BIGINT NOT NULL DEFAULT 0,
    orphan_session_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    actual_cost BIGINT NOT NULL DEFAULT 0,
    accounted BIGINT NOT NULL DEFAULT 0,
    decided_at BIGINT,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (claim_id, line_no)
);

CREATE INDEX batch_settlements_pending ON batch_settlements (owner_id, accounted, shard);
CREATE INDEX batch_settlements_batch ON batch_settlements (owner_id, batch_id);

CREATE TABLE batch_operations (
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    kind NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL CHECK (kind IN ('create', 'commit', 'publish')),
    operation_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    operation_id_len AS CAST(DATALENGTH(operation_id) AS INT) PERSISTED NOT NULL,
    request_sha256 VARBINARY(32) NOT NULL,
    batch_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    outcome NVARCHAR(4000) COLLATE Latin1_General_100_BIN2 NOT NULL,
    created_at BIGINT NOT NULL,
    PRIMARY KEY (owner_id, kind, operation_id, operation_id_len)
);

CREATE INDEX batch_operations_batch ON batch_operations (owner_id, batch_id);

CREATE TABLE batch_schedule (
    shard NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL PRIMARY KEY,
    turn BIGINT NOT NULL DEFAULT 0,
    cursor_owner NVARCHAR(128) COLLATE Latin1_General_100_BIN2,
    quantum BIGINT NOT NULL,
    credit_cap BIGINT NOT NULL,
    version BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE batch_deficits (
    shard NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    credit BIGINT NOT NULL DEFAULT 0,
    last_turn BIGINT NOT NULL DEFAULT 0,
    reserved_cost BIGINT NOT NULL DEFAULT 0,
    batch_turn BIGINT NOT NULL DEFAULT 0,
    batch_cursor NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    quota_blocked_version BIGINT,
    blocked_until BIGINT,
    PRIMARY KEY (shard, owner_id)
);

CREATE TABLE artifact_sessions (
    id NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL PRIMARY KEY,
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    kind NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL CHECK (kind IN ('upload', 'result', 'export')),
    artifact_ref NVARCHAR(512) COLLATE Latin1_General_100_BIN2 NOT NULL,
    secondary_ref NVARCHAR(512) COLLATE Latin1_General_100_BIN2,
    generation BIGINT NOT NULL DEFAULT 0,
    phase NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL CHECK (phase IN ('open', 'abandoned', 'orphaned')),
    lease_until BIGINT NOT NULL,
    reserved_bytes BIGINT NOT NULL DEFAULT 0 CHECK (reserved_bytes >= 0),
    max_bytes BIGINT NOT NULL DEFAULT 0,
    file_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    batch_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    claim_id NVARCHAR(64) COLLATE Latin1_General_100_BIN2,
    line_no BIGINT,
    filename NVARCHAR(255),
    expires_after_secs BIGINT,
    created_at BIGINT NOT NULL
);

CREATE INDEX artifact_sessions_lease ON artifact_sessions (kind, lease_until);
CREATE INDEX artifact_sessions_phase ON artifact_sessions (phase, lease_until);
CREATE INDEX artifact_sessions_batch ON artifact_sessions (owner_id, batch_id);
CREATE INDEX artifact_sessions_claim ON artifact_sessions (claim_id, line_no) WHERE claim_id IS NOT NULL;
