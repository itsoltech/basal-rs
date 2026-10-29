-- API users and their keys (`auth:`). A user's id is the Batch storage namespace (owner_id) of the requests of all
-- its keys. The token is never stored, only the SHA-256 of the whole bearer value; scopes is a comma list of route
-- groups. Users are disabled, never deleted, so their data keeps its namespace.
CREATE TABLE api_users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    disabled_at BIGINT
);
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES api_users (id),
    name TEXT NOT NULL,
    token_sha256 BYTEA NOT NULL,
    scopes TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    expires_at BIGINT,
    revoked_at BIGINT,
    last_used_at BIGINT
);
CREATE INDEX api_keys_user ON api_keys (user_id, created_at, id);
