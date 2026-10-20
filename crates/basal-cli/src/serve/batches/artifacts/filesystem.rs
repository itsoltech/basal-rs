//! Artifacts in a private local directory.
//!
//! ```text
//! <root>/                      0700, owned by the server user, not a symlink
//!   objects/<kind>/<xx>/<hex>  finished objects (xx: first two hex digits), 0600
//!   staging/<kind>-<hex>.part  data being written, same filesystem as objects/
//! ```
//!
//! A write goes to the staging file (created exclusively, not following symlinks), is flushed and `fsync`ed, then
//! hard-linked to its object path — `link` never replaces an existing file, so a key cannot be overwritten — and
//! the staging name is removed; both directories are `fsync`ed before `finish` returns. Paths are built only from
//! [`ArtifactKey`]s; files are opened with `O_NOFOLLOW`. Deleting a key removes its object and staging file and
//! `fsync`s their directories.

use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{self, Seek, SeekFrom};
use std::ops::Range;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Instant;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt, BufWriter};

use super::{blocking, ArtifactError, ArtifactKey, ArtifactKind};
use crate::serve::batches::config::{effective_uid, ensure_private_dir};

/// A private artifact directory.
#[derive(Debug)]
pub(crate) struct FilesystemStore {
    root: PathBuf,
    buffer_bytes: usize,
}

impl FilesystemStore {
    /// Create or check the root and its `objects/<kind>` and `staging` directories (0700, this user, no symlinks);
    /// a directory with wider permissions is refused, not changed.
    pub(crate) async fn open(root: PathBuf, buffer_bytes: usize) -> Result<Self, ArtifactError> {
        let root = blocking(move || {
            private_dir(&root)?;
            let root = root.canonicalize().map_err(ArtifactError::io("resolve root"))?;
            private_dir(&root.join("staging"))?;
            private_dir(&root.join("objects"))?;
            for kind in ArtifactKind::ALL {
                private_dir(&root.join("objects").join(kind.as_str()))?;
            }
            Ok(root)
        })
        .await?;
        Ok(Self { root, buffer_bytes: buffer_bytes.max(4096) })
    }

    fn object_path(&self, key: &ArtifactKey) -> PathBuf {
        let hex = key.hex();
        self.root.join("objects").join(key.kind().as_str()).join(&hex[..2]).join(hex)
    }

    fn staging_path(&self, key: &ArtifactKey) -> PathBuf {
        self.root.join("staging").join(format!("{}-{}.part", key.kind().as_str(), key.hex()))
    }

    pub(crate) async fn create(&self, key: ArtifactKey) -> Result<FsWriter, ArtifactError> {
        let staging = self.staging_path(&key);
        let object = self.object_path(&key);
        let (file, staging, object) = blocking(move || {
            if exists_nofollow(&object)? {
                return Err(ArtifactError::AlreadyExists);
            }
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&staging)
                .map_err(|e| match e.kind() {
                    io::ErrorKind::AlreadyExists => ArtifactError::AlreadyExists,
                    _ => ArtifactError::Io { op: "create", source: e },
                })?;
            Ok((file, staging, object))
        })
        .await?;
        Ok(FsWriter {
            file: Some(BufWriter::with_capacity(self.buffer_bytes, tokio::fs::File::from_std(file))),
            staging,
            object,
        })
    }

    pub(crate) async fn open_range(
        &self,
        key: &ArtifactKey,
        object_len: u64,
        range: Range<u64>,
    ) -> Result<Pin<Box<dyn AsyncRead + Send>>, ArtifactError> {
        let path = self.object_path(key);
        let start = range.start;
        let file = blocking(move || {
            let mut file = open_object(&path)?.ok_or(ArtifactError::NotFound)?;
            let meta = file.metadata().map_err(ArtifactError::io("stat"))?;
            if !meta.is_file() || meta.len() != object_len {
                return Err(ArtifactError::Integrity);
            }
            file.seek(SeekFrom::Start(start)).map_err(ArtifactError::io("seek"))?;
            Ok(file)
        })
        .await?;
        let len = range.end - range.start;
        Ok(Box::pin(tokio::fs::File::from_std(file).take(len)))
    }

    pub(crate) async fn delete(&self, key: &ArtifactKey) -> Result<(), ArtifactError> {
        let object = self.object_path(key);
        let staging = self.staging_path(key);
        blocking(move || {
            remove_if_present(&staging)?;
            sync_dir(parent(&staging)?)?;
            if remove_if_present(&object)? {
                sync_dir(parent(&object)?)?;
            }
            Ok(())
        })
        .await
    }

    pub(crate) async fn probe(&self) -> Result<(), ArtifactError> {
        let root = self.root.clone();
        let probe = self.root.join("staging").join(format!("probe-{}.part", uuid::Uuid::new_v4().simple()));
        blocking(move || {
            for dir in [root.clone(), root.join("staging"), root.join("objects")] {
                private_dir(&dir)?;
            }
            let result = (|| {
                let mut f = OpenOptions::new()
                    .write(true)
                    .read(true)
                    .create_new(true)
                    .mode(0o600)
                    .custom_flags(libc::O_NOFOLLOW)
                    .open(&probe)?;
                io::Write::write_all(&mut f, b"basal-probe")?;
                f.sync_all()?;
                f.seek(SeekFrom::Start(0))?;
                let mut back = Vec::new();
                io::Read::read_to_end(&mut f, &mut back)?;
                if back != b"basal-probe" {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "probe read back different data"));
                }
                Ok(())
            })();
            let removed = remove_if_present(&probe);
            result.map_err(ArtifactError::io("probe"))?;
            removed?;
            sync_dir(&root.join("staging"))
        })
        .await
    }
}

/// Writer of one staging file.
pub(crate) struct FsWriter {
    /// `None` once finished or discarded.
    file: Option<BufWriter<tokio::fs::File>>,
    staging: PathBuf,
    object: PathBuf,
}

impl FsWriter {
    pub(crate) async fn write(&mut self, data: &[u8]) -> Result<(), ArtifactError> {
        let file = self.file.as_mut().ok_or(ArtifactError::Closed)?;
        file.write_all(data).await.map_err(ArtifactError::io("write"))
    }

    /// Flush and `fsync` the staging file, link it to the object path without replacing anything, remove the
    /// staging name and `fsync` both directories. On error the staging file is removed (an object that was already
    /// linked stays until GC deletes the key).
    pub(crate) async fn finish(mut self, deadline: Instant) -> Result<(), ArtifactError> {
        let result = self.finish_inner(deadline).await;
        if result.is_err() {
            self.discard().await;
        }
        result
    }

    async fn finish_inner(&mut self, deadline: Instant) -> Result<(), ArtifactError> {
        let mut file = self.file.take().ok_or(ArtifactError::Closed)?;
        file.flush().await.map_err(ArtifactError::io("flush"))?;
        let file = file.into_inner();
        file.sync_all().await.map_err(ArtifactError::io("sync"))?;
        drop(file);
        if Instant::now() >= deadline {
            return Err(ArtifactError::DeadlineExceeded);
        }
        let (staging, object) = (self.staging.clone(), self.object.clone());
        blocking(move || {
            let dir = parent(&object)?;
            match DirBuilder::new().mode(0o700).create(dir) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(ArtifactError::Io { op: "create directory", source: e }),
            }
            let meta = std::fs::symlink_metadata(dir).map_err(ArtifactError::io("stat directory"))?;
            if !meta.is_dir() || meta.file_type().is_symlink() || meta.uid() != effective_uid() {
                return Err(ArtifactError::Insecure("an object directory is not a private directory"));
            }
            std::fs::hard_link(&staging, &object).map_err(|e| match e.kind() {
                io::ErrorKind::AlreadyExists => ArtifactError::AlreadyExists,
                _ => ArtifactError::Io { op: "link", source: e },
            })?;
            sync_dir(dir)?;
            std::fs::remove_file(&staging).map_err(ArtifactError::io("remove staging"))?;
            sync_dir(parent(&staging)?)
        })
        .await
    }

    /// Remove the staging file (idempotent).
    pub(crate) async fn discard(mut self) {
        drop(self.file.take());
        let staging = self.staging.clone();
        let _ = blocking(move || remove_if_present(&staging).map(drop)).await;
    }

    pub(crate) fn discard_blocking(mut self) {
        drop(self.file.take());
        let _ = remove_if_present(&self.staging);
    }
}

fn private_dir(dir: &Path) -> Result<(), ArtifactError> {
    ensure_private_dir(dir).map_err(|e| match e.kind() {
        io::ErrorKind::PermissionDenied => {
            ArtifactError::Insecure("a store directory is not private (mode 0700, this user)")
        }
        _ => ArtifactError::Io { op: "prepare directory", source: e },
    })
}

fn parent(path: &Path) -> Result<&Path, ArtifactError> {
    path.parent().ok_or(ArtifactError::InvalidKey)
}

/// Open an object without following a symlink; `None` when it does not exist.
fn open_object(path: &Path) -> Result<Option<File>, ArtifactError> {
    match OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path) {
        Ok(f) => Ok(Some(f)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => Err(ArtifactError::Insecure("an object is a symlink")),
        Err(e) => Err(ArtifactError::Io { op: "open", source: e }),
    }
}

fn exists_nofollow(path: &Path) -> Result<bool, ArtifactError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(ArtifactError::Io { op: "stat", source: e }),
    }
}

/// Remove a file (never a directory); `false` when it was not there.
fn remove_if_present(path: &Path) -> Result<bool, ArtifactError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(ArtifactError::Io { op: "remove", source: e }),
    }
}

fn sync_dir(dir: &Path) -> Result<(), ArtifactError> {
    File::open(dir).and_then(|d| d.sync_all()).map_err(ArtifactError::io("sync directory"))
}
