-- API users and their keys (`auth:`). A user's id is the Batch storage namespace (owner_id) of the requests of all
-- its keys. The token is never stored, only the SHA-256 of the whole bearer value; scopes is a comma list of route
-- groups. Users are disabled, never deleted, so their data keeps its namespace.
-- Ids are owner ids / lowercase hex without spaces, so plain `=` under the binary collation is exact.
CREATE TABLE api_users (
    id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL PRIMARY KEY,
    name NVARCHAR(128) NOT NULL,
    created_at BIGINT NOT NULL,
    disabled_at BIGINT
);
CREATE TABLE api_keys (
    id NVARCHAR(32) COLLATE Latin1_General_100_BIN2 NOT NULL PRIMARY KEY,
    user_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL REFERENCES api_users (id),
    name NVARCHAR(128) NOT NULL,
    token_sha256 VARBINARY(32) NOT NULL,
    scopes NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    created_at BIGINT NOT NULL,
    expires_at BIGINT,
    revoked_at BIGINT,
    last_used_at BIGINT
);
CREATE INDEX api_keys_user ON api_keys (user_id, created_at, id);
