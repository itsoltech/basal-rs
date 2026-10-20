ALTER TABLE batch_items ADD abandon_attempts INTEGER NOT NULL DEFAULT 0 CHECK (abandon_attempts >= 0 AND abandon_attempts <= 16);
ALTER TABLE batch_items ADD eligible_at INTEGER NOT NULL DEFAULT 0;
CREATE INDEX batch_items_eligible ON batch_items (owner_id, batch_id, state, eligible_at, line_no);
