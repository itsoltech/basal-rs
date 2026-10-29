//! Domain operations written once against [`super::sql::SqlTx`]; each function runs inside one transaction opened
//! by [`super::Store`].

pub(super) mod admin;
pub(super) mod claims;
pub(super) mod commit;
pub(super) mod common;
pub(super) mod files;
pub(super) mod finalize;
pub(super) mod ingest;
pub(super) mod keys;
pub(super) mod lifecycle;
