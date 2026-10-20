//! Checks shared by every configuration section that the server validates before it loads a model: typed errors
//! that name the field (never a secret value), absolute paths, bounded numbers, secrets read from one environment
//! variable or one file, and private local directories. Used by `database:` ([`crate::database`]) and `batches:`
//! (`serve::batches::config`, which re-exports the public items).

use std::io::Read;
use std::path::{Path, PathBuf};

/// Largest secret read from an environment variable or file.
const MAX_SECRET_BYTES: u64 = 64 * 1024;

/// A configuration that cannot be used. Messages name the field and the reason, never a secret value.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ConfigError {
    #[error("{field}: {reason}")]
    Invalid { field: String, reason: String },
    #[error("{field}: reading the secret from {source_desc}: {error}")]
    SecretUnavailable { field: &'static str, source_desc: String, error: String },
    #[error("{field}: {path}: {error}")]
    File {
        field: &'static str,
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
}

/// A [`ConfigError::Invalid`] of `field`.
pub(crate) fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ConfigError {
    ConfigError::Invalid { field: field.into(), reason: reason.into() }
}

/// `path` as given when it is absolute and has no `..` component.
pub(crate) fn absolute(field: &'static str, path: &Path) -> Result<PathBuf, ConfigError> {
    if !path.is_absolute() {
        return Err(invalid(field, "must be an absolute path"));
    }
    if path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err(invalid(field, "must not contain `..`"));
    }
    Ok(path.to_path_buf())
}

// ---------------------------------------------------------------------------------------------------------------
// Secrets

/// A secret value (connection string, access key). `Debug` prints no content; the buffer is overwritten on drop
/// (best effort: copies made by drivers are outside its control).
pub(crate) struct SecretString(String);

impl SecretString {
    pub(crate) fn new(value: String) -> Self {
        Self(value)
    }

    /// The secret itself, for the one call that needs it (a driver option, an HTTP header).
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretString([redacted])")
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        let mut bytes = std::mem::take(&mut self.0).into_bytes();
        bytes.fill(0);
        std::hint::black_box(&bytes);
    }
}

/// Where one secret comes from: exactly one environment variable or one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SecretSource {
    Env(String),
    File(PathBuf),
}

impl SecretSource {
    /// The source of `<field>_env` / `<field>_file`: at most one of them; `None` when both are absent.
    pub(crate) fn from_pair(
        field: &'static str,
        env: &Option<String>,
        file: &Option<PathBuf>,
    ) -> Result<Option<Self>, ConfigError> {
        match (env, file) {
            (Some(_), Some(_)) => Err(invalid(field, "set either the `_env` or the `_file` source, not both")),
            (Some(name), None) => {
                let valid = !name.is_empty()
                    && name.len() <= 128
                    && name.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                    && !name.as_bytes()[0].is_ascii_digit();
                if !valid {
                    return Err(invalid(field, "environment variable names use A-Z, 0-9 and _ (not first digit)"));
                }
                Ok(Some(Self::Env(name.clone())))
            }
            (None, Some(path)) => Ok(Some(Self::File(absolute(field, path)?))),
            (None, None) => Ok(None),
        }
    }

    /// Where the secret is read from, for messages (never its value).
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Env(name) => format!("environment variable {name}"),
            Self::File(path) => format!("file {}", path.display()),
        }
    }

    /// Read the secret: a non-empty UTF-8 value; one final line break of a file is dropped. A file must be a regular
    /// file of at most 64 KiB that is not writable by group or others. Blocking (file I/O).
    pub(crate) fn read(&self, field: &'static str) -> Result<SecretString, ConfigError> {
        let fail = |error: &str| ConfigError::SecretUnavailable {
            field,
            source_desc: self.describe(),
            error: error.to_string(),
        };
        let value = match self {
            Self::Env(name) => match std::env::var(name) {
                Ok(v) => v,
                Err(std::env::VarError::NotPresent) => return Err(fail("not set")),
                Err(std::env::VarError::NotUnicode(_)) => return Err(fail("not valid UTF-8")),
            },
            Self::File(path) => {
                let file = std::fs::File::open(path).map_err(|e| fail(&e.kind().to_string()))?;
                let meta = file.metadata().map_err(|e| fail(&e.kind().to_string()))?;
                if !meta.is_file() {
                    return Err(fail("not a regular file"));
                }
                if meta.len() > MAX_SECRET_BYTES {
                    return Err(fail("larger than 64 KiB"));
                }
                {
                    use std::os::unix::fs::PermissionsExt;
                    if meta.permissions().mode() & 0o022 != 0 {
                        return Err(fail("writable by group or others"));
                    }
                }
                let mut bytes = Vec::with_capacity(meta.len() as usize);
                file.take(MAX_SECRET_BYTES + 1).read_to_end(&mut bytes).map_err(|e| fail(&e.kind().to_string()))?;
                if bytes.len() as u64 > MAX_SECRET_BYTES {
                    return Err(fail("larger than 64 KiB"));
                }
                let mut text = String::from_utf8(bytes).map_err(|e| {
                    let mut b = e.into_bytes();
                    b.fill(0);
                    fail("not valid UTF-8")
                })?;
                if text.ends_with('\n') {
                    text.pop();
                    if text.ends_with('\r') {
                        text.pop();
                    }
                }
                text
            }
        };
        let secret = SecretString::new(value);
        if secret.expose().is_empty() {
            return Err(fail("empty"));
        }
        Ok(secret)
    }
}

/// An absolute path of an existing regular file (blocking: reads its metadata).
pub(crate) fn existing_file(field: &'static str, path: &Path) -> Result<PathBuf, ConfigError> {
    let path = absolute(field, path)?;
    let meta = std::fs::metadata(&path).map_err(|error| ConfigError::File { field, path: path.clone(), error })?;
    if !meta.is_file() {
        return Err(invalid(field, "not a regular file"));
    }
    Ok(path)
}

/// `value` within `min..=max`.
pub(crate) fn within<T: PartialOrd + std::fmt::Display + Copy>(
    field: &str,
    value: T,
    min: T,
    max: T,
) -> Result<T, ConfigError> {
    if value < min || value > max {
        return Err(invalid(field, format!("{value} is outside {min}..={max}")));
    }
    Ok(value)
}

// ---------------------------------------------------------------------------------------------------------------
// Private local directories

/// Create `dir` (and missing parents, mode 0700) or check an existing one: a real directory, not a symlink, owned by
/// this user and not accessible to group or others. An existing directory with wider permissions is refused, not
/// changed. Blocking.
pub(crate) fn ensure_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
    match std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    let meta = std::fs::symlink_metadata(dir)?;
    let reason = if meta.file_type().is_symlink() {
        Some("is a symbolic link")
    } else if !meta.is_dir() {
        Some("is not a directory")
    } else if meta.uid() != effective_uid() {
        Some("is owned by another user")
    } else if meta.permissions().mode() & 0o077 != 0 {
        Some("is accessible to group or others (expected mode 0700)")
    } else {
        None
    };
    match reason {
        Some(reason) => Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, reason)),
        None => Ok(()),
    }
}

/// Effective user id of this process.
pub(crate) fn effective_uid() -> u32 {
    // SAFETY: geteuid has no preconditions, touches no memory of this process and cannot fail (POSIX).
    unsafe { libc::geteuid() }
}
