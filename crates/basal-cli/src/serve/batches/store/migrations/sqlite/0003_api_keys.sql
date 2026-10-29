-- API keys of the server (`auth:`). The token is never stored, only the SHA-256 of the whole bearer value.
-- owner_id is the Batch storage namespace of the key's requests; scopes is a comma list of route groups.
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    token_sha256 BLOB NOT NULL,
    scopes TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    revoked_at INTEGER,
    last_used_at INTEGER
);
CREATE INDEX api_keys_owner ON api_keys (owner_id, created_at, id);
