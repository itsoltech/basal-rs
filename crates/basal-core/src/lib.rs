//! Backend-independent core of basal-rs: TypeSafe System One contract, the basal-1.0 prompt and letter-readout
//! protocol, shared-prefix packing and decision math.

pub mod decision;
pub mod engine;
pub mod evidence;
pub mod facts;
pub mod gate;
pub mod large_choice;
pub mod manifest;
pub mod pack;
pub mod prompt;
pub mod pyjson;
pub mod request;
pub mod tokenize;

pub use engine::{Backend, Batching, Engine, ItemResult, Prepared};
pub use manifest::ModelManifest;
pub use request::{DecideError, Item, QType};
