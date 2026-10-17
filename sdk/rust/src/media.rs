//! Inline media and request language settings. No helper reads files or fetches URLs.
use crate::Error;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};

/// Supported inline input kinds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    /// Encoded image.
    Image,
    /// Encoded audio.
    Audio,
}
/// One inline media item, in request order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaInput {
    /// Image or audio.
    #[serde(rename = "type")]
    pub kind: MediaKind,
    /// Base64 or a data URL. HTTP URLs are not fetched.
    pub data: String,
}
impl MediaInput {
    /// Encode image bytes. The server validates the codec and limits.
    pub fn image(data: &[u8]) -> Self {
        Self { kind: MediaKind::Image, data: STANDARD.encode(data) }
    }
    /// Encode audio bytes. The server validates the codec and limits.
    pub fn audio(data: &[u8]) -> Self {
        Self { kind: MediaKind::Audio, data: STANDARD.encode(data) }
    }
    /// Use existing base64 or a data URL, without fetching remote content.
    pub fn from_base64(kind: MediaKind, data: impl Into<String>) -> Result<Self, Error> {
        let input = Self { kind, data: data.into() };
        input.validate()?;
        Ok(input)
    }
    pub(crate) fn validate(&self) -> Result<(), Error> {
        let lower = self.data.get(..8).unwrap_or(&self.data).to_ascii_lowercase();
        if self.data.trim().is_empty() || lower.starts_with("http://") || lower.starts_with("https://") {
            return Err(Error::InvalidRequest("media requires inline base64 or a data URL"));
        }
        Ok(())
    }
}
/// Explicit prompt language.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// Polish.
    Pl,
    /// English.
    En,
}
/// Image resolution budget.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaDetail {
    /// Default image budget.
    Default,
    /// High image budget.
    High,
}
/// Media accounting reported by the server.
#[derive(Debug, Clone, Deserialize)]
pub struct MediaUsage {
    /// Number of images.
    pub images: u64,
    /// Number of audio inputs.
    pub audio: u64,
    /// Total soft tokens over executed prompts.
    pub media_tokens: u64,
    /// Requested image detail.
    pub detail: MediaDetail,
}
