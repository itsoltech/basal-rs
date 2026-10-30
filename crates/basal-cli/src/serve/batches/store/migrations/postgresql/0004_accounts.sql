-- Accounts of the browser UI (`ui:`): a role and an optional password per API user, and login sessions.
-- password_hash is a PHC string (argon2id); NULL: the user cannot log in to the UI (its keys still work). A session
-- id is the lowercase hex SHA-256 of the session token, never the token. Deleting a user deletes its keys and
-- sessions first.
ALTER TABLE api_users
    ADD COLUMN role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    ADD COLUMN password_hash TEXT;
CREATE TABLE api_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES api_users (id),
    created_at BIGINT NOT NULL,
    expires_at BIGINT NOT NULL
);
CREATE INDEX api_sessions_user ON api_sessions (user_id, expires_at);
