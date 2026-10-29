-- API keys of the server (`auth:`). The token is never stored, only the SHA-256 of the whole bearer value.
-- owner_id is the Batch storage namespace of the key's requests; scopes is a comma list of route groups.
-- id and scopes are lowercase hex / ASCII without trailing spaces, so plain `=` is exact.
CREATE TABLE api_keys (
    id NVARCHAR(32) COLLATE Latin1_General_100_BIN2 NOT NULL PRIMARY KEY,
    owner_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
    name NVARCHAR(128) NOT NULL,
    token_sha256 VARBINARY(32) NOT NULL,
    scopes NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL,
    created_at BIGINT NOT NULL,
    expires_at BIGINT,
    revoked_at BIGINT,
    last_used_at BIGINT
);
CREATE INDEX api_keys_owner ON api_keys (owner_id, created_at, id);
