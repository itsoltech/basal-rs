//! Conditional trusted-local calibration membership, captured before semantic reads.

use std::collections::TryReserveError;
use std::fmt;
use std::fs::{self, File, Metadata};
use std::io::{self, Read};
use std::mem::size_of;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::raw_source::RawCharge;
use super::semantics::{CalibrationProbeRole, CalibrationReadRole, GgufSemanticError};
use super::{GgufRawSourceBudget, GgufRawSourceError, Sha256Digest};

const MAX_SCANNED: u64 = 4096;
const MAX_MATCHES: u64 = 256;
const MAX_NAME: u64 = 255;
const MAX_NAMES: u64 = 64 * 1024;
const MAX_FILE: u64 = 64 * 1024 * 1024;
const MAX_CONTENT: u64 = 128 * 1024 * 1024;
const MAX_METADATA: u64 = 256 * 1024;

/// Acknowledgment of an external arrangement; this token acquires no lock.
///
/// Before entry, the caller must actually exclude membership changes (creation,
/// removal, rename, exchange/replacement), root/ancestor resolution changes and
/// calibration content/metadata writes, including other descriptors, processes,
/// hardlinks, aliases and writable mappings. An explicit override's resolution
/// and target are covered separately, even outside the directory. Keep the real
/// arrangement through success or error return, and companions trusted under
/// their existing contract. If unavailable, do not use this route.
///
/// An explicit override must already resolve to a trusted regular file; its
/// inherited open-before-type-check can block on a violating FIFO. This is no
/// hostile-filesystem, wall-clock, snapshot or authenticity guarantee. Read atime
/// changes are excluded from stamps. False acknowledgment invalidates consistency
/// claims but does not make safe I/O unsafe. The calibration guard can end after
/// return; the separate GGUF inode obligation continues through its final reads,
/// rehash and publication.
pub struct CallerCalibrationDirectoryLease {
    _acknowledged: (),
}

impl CallerCalibrationDirectoryLease {
    /// Acknowledge the already held arrangement described on this type.
    pub fn acknowledge_trusted_local_contract() -> Self {
        Self { _acknowledged: () }
    }
}

/// One exact matching basename and its uninterpreted retained content.
pub struct GgufCalibrationVariant {
    name: String,
    bytes: Vec<u8>,
    sha256: Sha256Digest,
    // None only during private enumeration; every successful record has a stamp.
    _stamp: Option<SourceStamp>,
}

impl GgufCalibrationVariant {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> Sha256Digest {
        self.sha256
    }
}

/// Complete sorted admitted namespace, conditional on the caller's exclusion.
///
/// Constructed only together with its semantic owner. Empty entries mean a
/// completed empty capture; an absent owner inventory means never captured.
pub struct GgufCalibrationInventory {
    entries: Vec<GgufCalibrationVariant>,
    read_indices: [Option<usize>; 2],
    directory_stamp: SourceStamp,
    _lease: CallerCalibrationDirectoryLease,
    name_bytes: u64,
    metadata_capacity: u64,
    content_bytes: u64,
    content_capacity: u64,
    // Entries and all their content backing drop before this aggregate charge.
    raw_charge: Option<RawCharge>,
}

impl GgufCalibrationInventory {
    pub fn entries(&self) -> &[GgufCalibrationVariant] {
        &self.entries
    }

    /// Borrow the independently bound automatic read role. Explicit always
    /// returns None; its actual bytes belong to `calibration().files()`.
    /// An absent role alone does not prove a name was absent or even probed.
    pub fn entry_for_read(&self, role: CalibrationReadRole) -> Option<&GgufCalibrationVariant> {
        let slot = match role {
            CalibrationReadRole::Candidate => 0,
            CalibrationReadRole::OrdersVariant => 1,
            CalibrationReadRole::Explicit => return None,
        };
        self.read_indices[slot].and_then(|index| self.entries.get(index))
    }

    pub(super) fn find(&self, name: &str) -> Option<usize> {
        self.entries.binary_search_by(|entry| entry.name.as_bytes().cmp(name.as_bytes())).ok()
    }

    pub(super) fn bind_read(&mut self, role: CalibrationReadRole, index: usize) {
        match role {
            CalibrationReadRole::Candidate => self.read_indices[0] = Some(index),
            CalibrationReadRole::OrdersVariant => self.read_indices[1] = Some(index),
            CalibrationReadRole::Explicit => {}
        }
    }

    pub(super) fn capture(
        directory: &Path,
        lease: CallerCalibrationDirectoryLease,
        budget: Option<&GgufRawSourceBudget>,
    ) -> Result<(Self, File), GgufCalibrationInventoryError> {
        if !cfg!(all(
            target_pointer_width = "64",
            target_endian = "little",
            any(target_os = "linux", target_os = "macos")
        )) {
            return Err(GgufCalibrationInventoryError::UnsupportedHost);
        }
        let handle = File::open(directory).map_err(|source| io_error(InventoryOperation::OpenDirectory, source))?;
        let metadata = handle.metadata().map_err(|source| io_error(InventoryOperation::DirectoryMetadata, source))?;
        if !metadata.is_dir() {
            return Err(GgufCalibrationInventoryError::NonDirectory);
        }
        let directory_stamp = SourceStamp::from_metadata(&metadata)?;
        // Precede entries so unwind also destroys raw backing before release.
        let raw_charge = budget.map(GgufRawSourceBudget::charge);
        let mut entries = Vec::new();
        let count = to_usize(MAX_MATCHES, InventoryResource::Matches)?;
        let record = to_u64(size_of::<GgufCalibrationVariant>(), InventoryResource::MetadataCapacity)?;
        let requested = record.checked_mul(MAX_MATCHES).ok_or(arithmetic(InventoryResource::MetadataCapacity))?;
        limit(InventoryResource::MetadataCapacity, requested, MAX_METADATA)?;
        entries.try_reserve_exact(count).map_err(allocation)?;
        let metadata_capacity = to_u64(entries.capacity(), InventoryResource::MetadataCapacity)?
            .checked_mul(record)
            .ok_or(arithmetic(InventoryResource::MetadataCapacity))?;
        limit(InventoryResource::MetadataCapacity, metadata_capacity, MAX_METADATA)?;
        let mut inventory = Self {
            entries,
            read_indices: [None; 2],
            directory_stamp,
            _lease: lease,
            name_bytes: 0,
            metadata_capacity,
            content_bytes: 0,
            content_capacity: 0,
            raw_charge,
        };
        inventory.enumerate(directory)?;
        inventory.entries.sort_unstable_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
        if inventory.entries.windows(2).any(|pair| pair[0].name == pair[1].name) {
            return Err(GgufCalibrationInventoryError::DuplicateName);
        }
        for index in 0..inventory.entries.len() {
            inventory.read_entry(directory, index)?;
        }
        inventory.check_directory(&handle, InventoryPhase::AfterCapture)?;
        Ok((inventory, handle))
    }

    fn enumerate(&mut self, directory: &Path) -> Result<(), GgufCalibrationInventoryError> {
        let mut scanned = 0_u64;
        let mut matches = 0_u64;
        let mut longest = 0_u64;
        let mut invalid_utf8 = false;
        let mut violated = false;
        let traversal = fs::read_dir(directory).map_err(|source| io_error(InventoryOperation::Enumerate, source))?;
        for entry in traversal {
            let entry = entry.map_err(|source| io_error(InventoryOperation::Enumerate, source))?;
            scanned = add(scanned, 1, InventoryResource::Scanned)?;
            limit(InventoryResource::Scanned, scanned, MAX_SCANNED)?;
            let name = entry.file_name();
            drop(entry);
            let Some(name) = name.to_str() else {
                invalid_utf8 = true;
                violated = true;
                continue;
            };
            if !name.starts_with("CALIBRATION") || !name.ends_with(".json") {
                continue;
            }
            matches = add(matches, 1, InventoryResource::Matches)?;
            let length = to_u64(name.len(), InventoryResource::NameBytes)?;
            longest = longest.max(length);
            self.name_bytes = add(self.name_bytes, length, InventoryResource::TotalNameBytes)?;
            violated |= longest > MAX_NAME || matches > MAX_MATCHES || self.name_bytes > MAX_NAMES;
            if violated {
                continue;
            }
            let requested = add(self.metadata_capacity, length, InventoryResource::MetadataCapacity)?;
            limit(InventoryResource::MetadataCapacity, requested, MAX_METADATA)?;
            let mut owned = String::new();
            owned.try_reserve_exact(name.len()).map_err(allocation)?;
            let capacity = to_u64(owned.capacity(), InventoryResource::MetadataCapacity)?;
            self.metadata_capacity = add(self.metadata_capacity, capacity, InventoryResource::MetadataCapacity)?;
            limit(InventoryResource::MetadataCapacity, self.metadata_capacity, MAX_METADATA)?;
            owned.push_str(name);
            // The complete fixed record reservation precedes enumeration.
            self.entries.push(GgufCalibrationVariant {
                name: owned,
                bytes: Vec::new(),
                sha256: Sha256Digest([0; 32]),
                _stamp: None,
            });
        }
        // Deterministic semantic refusal precedence after complete bounded EOF.
        if invalid_utf8 {
            return Err(GgufCalibrationInventoryError::InvalidName { reason: InventoryNameReason::InvalidUtf8 });
        }
        limit(InventoryResource::NameBytes, longest, MAX_NAME)?;
        limit(InventoryResource::Matches, matches, MAX_MATCHES)?;
        limit(InventoryResource::TotalNameBytes, self.name_bytes, MAX_NAMES)
    }

    fn read_entry(&mut self, directory: &Path, index: usize) -> Result<(), GgufCalibrationInventoryError> {
        let entry = &mut self.entries[index];
        let path = directory.join(&entry.name);
        let metadata =
            fs::symlink_metadata(&path).map_err(|source| io_error(InventoryOperation::EntryMetadata, source))?;
        if !metadata.is_file() {
            return Err(GgufCalibrationInventoryError::NonRegular { index });
        }
        let stamp = SourceStamp::from_metadata(&metadata)?;
        limit(InventoryResource::FileBytes, stamp.len, MAX_FILE)?;
        self.content_bytes = add(self.content_bytes, stamp.len, InventoryResource::ContentBytes)?;
        limit(InventoryResource::ContentBytes, self.content_bytes, MAX_CONTENT)?;
        let mut file = File::open(&path).map_err(|source| io_error(InventoryOperation::OpenEntry, source))?;
        stamp.compare(&SourceStamp::from_file(&file)?, InventoryPhase::EntryOpen)?;
        let length = to_usize(stamp.len, InventoryResource::FileBytes)?;
        let requested = add(self.content_capacity, stamp.len, InventoryResource::ContentCapacity)?;
        limit(InventoryResource::ContentCapacity, requested, MAX_CONTENT)?;
        let capacity = match self.raw_charge.as_mut() {
            Some(charge) => {
                entry.bytes = charge
                    .allocate(length)
                    .map_err(|source| GgufCalibrationInventoryError::RawSource { index, source })?;
                // Exact requested layout, known before allocation, not a capacity
                // observation granting retroactive authority. Each entry is read once.
                stamp.len
            }
            None => {
                entry.bytes.try_reserve_exact(length).map_err(allocation)?;
                to_u64(entry.bytes.capacity(), InventoryResource::ContentCapacity)?
            }
        };
        limit(InventoryResource::FileCapacity, capacity, MAX_FILE)?;
        self.content_capacity = add(self.content_capacity, capacity, InventoryResource::ContentCapacity)?;
        limit(InventoryResource::ContentCapacity, self.content_capacity, MAX_CONTENT)?;
        let mut scratch = [0_u8; 8192];
        let mut hash = Sha256::new();
        let mut remaining = stamp.len;
        while remaining != 0 {
            let amount = to_usize(remaining.min(8192), InventoryResource::FileBytes)?;
            let read = read_retry(&mut file, &mut scratch[..amount])?;
            if read == 0 {
                return Err(GgufCalibrationInventoryError::SourceChanged {
                    phase: InventoryPhase::EntryRead,
                    field: InventoryStampField::EarlyEof,
                });
            }
            hash.update(&scratch[..read]);
            // read <= remaining, initially length: the exact backing cannot grow.
            entry.bytes.extend_from_slice(&scratch[..read]);
            remaining = remaining
                .checked_sub(to_u64(read, InventoryResource::FileBytes)?)
                .ok_or(arithmetic(InventoryResource::FileBytes))?;
        }
        let mut eof = [0_u8; 1];
        if read_retry(&mut file, &mut eof)? != 0 {
            return Err(GgufCalibrationInventoryError::SourceChanged {
                phase: InventoryPhase::EntryRead,
                field: InventoryStampField::Growth,
            });
        }
        stamp.compare(&SourceStamp::from_file(&file)?, InventoryPhase::EntryEnd)?;
        entry.sha256 = Sha256Digest(hash.finalize().into());
        entry._stamp = Some(stamp);
        Ok(())
    }

    pub(super) fn check_directory(
        &self,
        handle: &File,
        phase: InventoryPhase,
    ) -> Result<(), GgufCalibrationInventoryError> {
        let metadata = handle.metadata().map_err(|source| io_error(InventoryOperation::DirectoryMetadata, source))?;
        self.directory_stamp.compare(&SourceStamp::from_metadata(&metadata)?, phase)
    }
}

struct SourceStamp {
    dev: u64,
    ino: u64,
    file_type: u32,
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

impl SourceStamp {
    fn from_metadata(metadata: &Metadata) -> Result<Self, GgufCalibrationInventoryError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Ok(Self {
                dev: metadata.dev(),
                ino: metadata.ino(),
                file_type: metadata.mode() & 0o170000,
                len: metadata.len(),
                mtime: (metadata.mtime(), metadata.mtime_nsec()),
                ctime: (metadata.ctime(), metadata.ctime_nsec()),
            })
        }
        #[cfg(not(unix))]
        {
            let _ = metadata;
            Err(GgufCalibrationInventoryError::UnsupportedHost)
        }
    }

    fn from_file(file: &File) -> Result<Self, GgufCalibrationInventoryError> {
        let metadata = file.metadata().map_err(|source| io_error(InventoryOperation::EntryMetadata, source))?;
        Self::from_metadata(&metadata)
    }

    fn compare(&self, after: &Self, phase: InventoryPhase) -> Result<(), GgufCalibrationInventoryError> {
        for (changed, field) in [
            (self.dev != after.dev, InventoryStampField::Device),
            (self.ino != after.ino, InventoryStampField::Inode),
            (self.file_type != after.file_type, InventoryStampField::Type),
            (self.len != after.len, InventoryStampField::Length),
            (self.mtime != after.mtime, InventoryStampField::Mtime),
            (self.ctime != after.ctime, InventoryStampField::Ctime),
        ] {
            if changed {
                return Err(GgufCalibrationInventoryError::SourceChanged { phase, field });
            }
        }
        Ok(())
    }
}

fn read_retry(file: &mut File, bytes: &mut [u8]) -> Result<usize, GgufCalibrationInventoryError> {
    loop {
        match file.read(bytes) {
            Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
            Err(source) => return Err(io_error(InventoryOperation::ReadEntry, source)),
            Ok(read) => return Ok(read),
        }
    }
}

fn to_u64(value: usize, resource: InventoryResource) -> Result<u64, GgufCalibrationInventoryError> {
    u64::try_from(value).map_err(|_| arithmetic(resource))
}

fn to_usize(value: u64, resource: InventoryResource) -> Result<usize, GgufCalibrationInventoryError> {
    usize::try_from(value).map_err(|_| arithmetic(resource))
}

fn add(left: u64, right: u64, resource: InventoryResource) -> Result<u64, GgufCalibrationInventoryError> {
    left.checked_add(right).ok_or(arithmetic(resource))
}

fn limit(resource: InventoryResource, actual: u64, limit: u64) -> Result<(), GgufCalibrationInventoryError> {
    if actual > limit {
        Err(GgufCalibrationInventoryError::Limit { resource, actual, limit })
    } else {
        Ok(())
    }
}

fn arithmetic(resource: InventoryResource) -> GgufCalibrationInventoryError {
    GgufCalibrationInventoryError::Arithmetic { resource }
}

fn allocation(source: TryReserveError) -> GgufCalibrationInventoryError {
    GgufCalibrationInventoryError::Allocation { source }
}

fn io_error(operation: InventoryOperation, source: io::Error) -> GgufCalibrationInventoryError {
    GgufCalibrationInventoryError::Io { operation, source }
}

/// Fixed local resource categories, not whole-process admission.
#[derive(Clone, Copy, Debug)]
pub enum InventoryResource {
    Scanned,
    Matches,
    NameBytes,
    TotalNameBytes,
    MetadataCapacity,
    FileBytes,
    ContentBytes,
    FileCapacity,
    ContentCapacity,
    BindingNameCapacity,
}

#[derive(Clone, Copy, Debug)]
pub enum InventoryOperation {
    OpenDirectory,
    DirectoryMetadata,
    Enumerate,
    EntryMetadata,
    OpenEntry,
    ReadEntry,
}

#[derive(Clone, Copy, Debug)]
pub enum InventoryNameReason {
    InvalidUtf8,
}

#[derive(Clone, Copy, Debug)]
pub enum InventoryPhase {
    EntryOpen,
    EntryRead,
    EntryEnd,
    AfterCapture,
    BeforeInstall,
}

#[derive(Clone, Copy, Debug)]
pub enum InventoryStampField {
    Device,
    Inode,
    Type,
    Length,
    Mtime,
    Ctime,
    EarlyEof,
    Growth,
}

#[derive(Clone, Copy, Debug)]
pub enum InventoryReadMismatch {
    Structure,
    Mode,
    Content,
    Decision,
    Name,
}

/// Typed refusals. Display/Debug never include input text or nested causes,
/// even for externally constructed variants. Source chains remain untrusted.
#[derive(thiserror::Error)]
pub enum GgufCalibrationInventoryError {
    #[error("calibration inventory raw source admission failed")]
    RawSource {
        index: usize,
        #[source]
        source: GgufRawSourceError,
    },
    #[error("unsupported calibration inventory host")]
    UnsupportedHost,
    #[error("calibration inventory I/O failed")]
    Io {
        operation: InventoryOperation,
        #[source]
        source: io::Error,
    },
    #[error("calibration inventory root is not a directory")]
    NonDirectory,
    #[error("invalid calibration inventory name")]
    InvalidName { reason: InventoryNameReason },
    #[error("calibration inventory resource limit exceeded")]
    Limit { resource: InventoryResource, actual: u64, limit: u64 },
    #[error("duplicate calibration inventory name")]
    DuplicateName,
    #[error("calibration inventory entry is not a direct regular file")]
    NonRegular { index: usize },
    #[error("calibration inventory source changed")]
    SourceChanged { phase: InventoryPhase, field: InventoryStampField },
    #[error("calibration inventory allocation failed")]
    Allocation {
        #[source]
        source: TryReserveError,
    },
    #[error("calibration inventory arithmetic failed")]
    Arithmetic { resource: InventoryResource },
    #[error("calibration inventory actual read mismatch")]
    ReadSetMismatch { role: CalibrationReadRole, reason: InventoryReadMismatch },
    #[error("calibration inventory reached probe mismatch")]
    ProbeMismatch { role: CalibrationProbeRole },
    #[error("calibration inventory selected provenance mismatch")]
    SelectedMismatch,
}

impl fmt::Debug for GgufCalibrationInventoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

/// Distinguishes capture/cross-binding refusal from the original semantic cause.
#[derive(thiserror::Error)]
pub enum GgufInventoryReadError {
    #[error("GGUF calibration inventory failed")]
    Inventory {
        #[source]
        source: GgufCalibrationInventoryError,
    },
    #[error("GGUF inventory semantic read failed")]
    Semantic {
        #[source]
        source: GgufSemanticError,
    },
}

impl fmt::Debug for GgufInventoryReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl From<GgufCalibrationInventoryError> for GgufInventoryReadError {
    fn from(source: GgufCalibrationInventoryError) -> Self {
        Self::Inventory { source }
    }
}
