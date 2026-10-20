//! Bounded native media decoding and the processor contracts of the pinned Gemma 4 models.
//! No URL fetching, external decoder processes or persistent storage of request media.

use std::io::Cursor;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use base64::Engine;
use serde_json::Value;

use crate::execution::StopToken;
use crate::planner::{PlanBudget, PlanStage, Reservation, UnboundedFeature};

pub const MAX_ITEMS: usize = 8;
pub const MAX_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_PIXELS: u64 = 50_000_000;
pub const SAMPLE_RATE: u32 = 16000;
/// `image::Limits::max_alloc` of an image decoder: the transient bound reserved for every image without a proven
/// smaller format bound.
pub const DECODER_ALLOC: u64 = 400 * 1024 * 1024;
/// Ring buffer of symphonia's media source stream (its default `buffer_len`).
const AUDIO_STREAM_BUFFER: u64 = 64 * 1024;
/// Allowance for codec state (tables, synthesis/LPC buffers) of the bounded codecs; an estimator constant, not a
/// measurement.
const AUDIO_CODEC_STATE: u64 = 4 * 1024 * 1024;
/// Allowance for the 512-point FFT plan and scratch of the mel analysis; an estimator constant, not a measurement.
const FFT_SCRATCH: u64 = 1024 * 1024;

/// Reserve the bytes computed by `bytes` for `stage` when planning under a bounded budget; the unbounded path
/// computes no bound.
fn reserve(
    budget: Option<&PlanBudget>,
    stage: PlanStage,
    bytes: impl FnOnce() -> u64,
) -> Result<Option<Reservation<'_>>> {
    Ok(match budget {
        Some(b) => Some(b.reserve(stage, bytes())?),
        None => None,
    })
}

fn retain(r: Option<Reservation<'_>>) {
    if let Some(r) = r {
        r.retain();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Image,
    Audio,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }
}

#[derive(Clone)]
pub struct ImageInput {
    pub patches: Vec<f32>,
    pub positions: Vec<(i64, i64)>,
    pub patch_dim: usize,
    pub soft_tokens: usize,
    pub patch_width: usize,
    pub patch_height: usize,
}

#[derive(Clone)]
pub struct AudioInput {
    pub features: Vec<f32>,
    pub valid: Vec<bool>,
    pub feature_dim: usize,
    pub soft_tokens: usize,
}

#[derive(Clone)]
pub enum Input {
    Image(ImageInput),
    Audio(AudioInput),
}

impl Input {
    pub fn kind(&self) -> Kind {
        match self {
            Self::Image(_) => Kind::Image,
            Self::Audio(_) => Kind::Audio,
        }
    }
    pub fn soft_tokens(&self) -> usize {
        match self {
            Self::Image(x) => x.soft_tokens,
            Self::Audio(x) => x.soft_tokens,
        }
    }
}

#[derive(Clone)]
pub struct ProcessedMedia {
    pub items: Vec<Input>,
    pub high_detail: bool,
    /// Internal, collision-free markers for positioned System One assets. Native media leave this absent.
    pub(crate) asset_markers: Option<Vec<String>>,
}

impl std::fmt::Debug for ProcessedMedia {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessedMedia")
            .field("items", &self.items.len())
            .field("soft_tokens", &self.items.iter().map(Input::soft_tokens).sum::<usize>())
            .field("high_detail", &self.high_detail)
            .finish()
    }
}

impl ProcessedMedia {
    pub fn image_count(&self) -> usize {
        self.items.iter().filter(|x| x.kind() == Kind::Image).count()
    }
    pub fn audio_count(&self) -> usize {
        self.items.len() - self.image_count()
    }
    /// Exact processor placeholder expansion; strings are model special tokens, not data from the media payload.
    pub fn placeholders(&self) -> Vec<String> {
        self.items
            .iter()
            .map(|x| match x {
                Input::Image(x) => format!("<|image>{}<image|>", "<|image|>".repeat(x.soft_tokens)),
                Input::Audio(x) => format!("<|audio>{}<audio|>", "<|audio|>".repeat(x.soft_tokens)),
            })
            .collect()
    }
}

pub struct Processor {
    unified: bool,
    max_audio_seconds: f64,
    audio_enabled: bool,
}

impl Processor {
    pub fn load(dir: &Path, unified: bool) -> Result<Self> {
        Self::load_with_audio(dir, unified, true)
    }

    /// The processor metadata may advertise audio even when the model has no audio tower.
    pub fn load_with_audio(dir: &Path, unified: bool, audio_enabled: bool) -> Result<Self> {
        let v: Value = serde_json::from_slice(&std::fs::read(dir.join("processor_config.json"))?)?;
        let expected = if unified { "Gemma4UnifiedProcessor" } else { "Gemma4Processor" };
        ensure!(v.get("processor_class").and_then(Value::as_str) == Some(expected), "unsupported Gemma processor");
        let image = v.get("image_processor").context("image processor missing")?;
        ensure!(
            image["patch_size"] == 16
                && image["pooling_kernel_size"] == 3
                && image["do_resize"] == true
                && image["do_rescale"] == true
                && image["do_normalize"] == false
                && image["resample"] == 3
                && image["rescale_factor"].as_f64() == Some(1.0 / 255.0),
            "unsupported image processor settings"
        );
        let audio = v.get("feature_extractor").context("audio processor missing")?;
        ensure!(audio["sampling_rate"] == SAMPLE_RATE, "unsupported audio sample rate");
        if unified {
            ensure!(
                audio["audio_samples_per_token"] == 640 && audio["feature_size"] == 640,
                "unsupported waveform frame processor"
            );
        } else {
            ensure!(
                audio["feature_size"] == 128
                    && audio["frame_length"] == 320
                    && audio["hop_length"] == 160
                    && audio["fft_length"] == 512
                    && audio["dither"] == 0.0
                    && audio["preemphasis"] == 0.0
                    && audio["per_bin_mean"].is_null()
                    && audio["per_bin_stddev"].is_null()
                    && audio["mel_floor"] == 0.001
                    && audio["min_frequency"] == 0.0
                    && audio["max_frequency"] == 8000.0
                    && audio["input_scale_factor"] == 1.0,
                "unsupported mel processor settings"
            );
        }
        let window = v["audio_seq_length"]
            .as_f64()
            .zip(v["audio_ms_per_token"].as_f64())
            .map(|(n, ms)| n * ms / 1000.0)
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 30.0)
            .context("invalid audio window")?;
        Ok(Self { unified, max_audio_seconds: window, audio_enabled })
    }

    pub fn process(&self, entries: &Value, detail: &str) -> Result<ProcessedMedia> {
        self.process_with(entries, detail, None)
    }

    /// [`Processor::process`], optionally under a bounded planning budget: every allocation with a size known in
    /// advance is reserved first (decoded payload, decoder transients up to [`DECODER_ALLOC`], image buffers,
    /// resampling scratch, audio PCM and analysis), padded features stay reserved until the end of the plan, and the
    /// stop token is checked between media. Audio codecs without a verified bound are refused with
    /// [`UnboundedFeature`]. The decoded values are those of the unbounded path.
    pub fn process_with(
        &self,
        entries: &Value,
        detail: &str,
        bounded: Option<(&PlanBudget, Option<&StopToken>)>,
    ) -> Result<ProcessedMedia> {
        let budget = bounded.map(|(b, _)| b);
        let stop = bounded.and_then(|(_, s)| s);
        let max_soft = match detail {
            "default" => 280,
            "high" => 1120,
            _ => anyhow::bail!("detail must be default or high"),
        };
        let entries = entries
            .as_array()
            .filter(|v| !v.is_empty() && v.len() <= MAX_ITEMS)
            .context("media must contain 1..8 image/audio objects")?;
        let mut items = Vec::with_capacity(entries.len());
        let mut processed_bytes = 0usize;
        for (i, entry) in entries.iter().enumerate() {
            if let Some(stop) = stop {
                stop.check()?;
            }
            ensure!(
                self.audio_enabled || entry.get("type").and_then(Value::as_str) != Some("audio"),
                "media {i}: this model does not support audio"
            );
            let data = entry
                .get("data")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .with_context(|| format!("media {i}: data must be non-empty base64"))?;
            let data =
                if data.starts_with("data:") { data.split_once(',').context("invalid data URL")?.1 } else { data };
            ensure!(data.len().saturating_mul(3) / 4 <= MAX_BYTES, "media {i}: more than 20 MiB");
            let _raw = reserve(budget, PlanStage::Base64, || (data.len() as u64).saturating_mul(3) / 4 + 3)?;
            let raw = base64::engine::general_purpose::STANDARD
                .decode(data)
                .with_context(|| format!("media {i}: invalid base64"))?;
            ensure!(raw.len() <= MAX_BYTES, "media {i}: more than 20 MiB");
            let item = match entry.get("type").and_then(Value::as_str) {
                Some("image") => Input::Image(
                    image(&raw, max_soft, self.unified, budget).with_context(|| format!("media {i}: image"))?,
                ),
                Some("audio") => Input::Audio(
                    audio(&raw, self.max_audio_seconds, self.unified, budget)
                        .with_context(|| format!("media {i}: audio"))?,
                ),
                _ => anyhow::bail!("media {i}: type must be image or audio"),
            };
            let floats = match &item {
                Input::Image(x) => x.patches.len(),
                Input::Audio(x) => x.features.len(),
            };
            processed_bytes = processed_bytes
                .checked_add(floats.checked_mul(4).context("media feature size overflow")?)
                .context("media size overflow")?;
            ensure!(processed_bytes <= 256 * 1024 * 1024, "media processed features exceed 256 MiB");
            items.push(item);
        }
        Ok(ProcessedMedia { items, high_detail: detail == "high", asset_markers: None })
    }
}

fn image(raw: &[u8], max_soft: usize, unified: bool, budget: Option<&PlanBudget>) -> Result<ImageInput> {
    use image::ImageDecoder;
    // padded features depend on `detail` only: reserved before decoding and kept with the plan
    let features = reserve(budget, PlanStage::ImageFeatures, || {
        let (size, padded) = if unified { (48u64, max_soft as u64) } else { (16, max_soft as u64 * 9) };
        padded * size * size * 3 * 4 + padded * 16
    })?;
    let mut reader = image::ImageReader::new(Cursor::new(raw)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    // the decoder's own allocations are limited to exactly the reserved transient
    let transient = reserve(budget, PlanStage::ImageDecoder, || DECODER_ALLOC)?;
    limits.max_alloc = Some(DECODER_ALLOC);
    reader.limits(limits);
    let decoder = reader.into_decoder()?;
    let (w, h) = decoder.dimensions();
    ensure!(w > 0 && h > 0 && u64::from(w) * u64::from(h) <= MAX_PIXELS, "image exceeds 50 million pixels");
    // `from_decoder` allocates the decoded buffer outside the decoder limits; `into_rgb8` may convert it
    let decoded_buffer = reserve(budget, PlanStage::ImageBuffer, || decoder.total_bytes())?;
    let decoded = image::DynamicImage::from_decoder(decoder)?;
    drop(transient);
    let _rgb_buffer = reserve(budget, PlanStage::ImageBuffer, || u64::from(w) * u64::from(h) * 3)?;
    let rgb = decoded.into_rgb8();
    drop(decoded_buffer);
    let factor = ((max_soft * 9 * 16 * 16) as f64 / (f64::from(w) * f64::from(h))).sqrt();
    let mut height = ((factor * f64::from(h) / 48.0).floor() as usize) * 48;
    let mut width = ((factor * f64::from(w) / 48.0).floor() as usize) * 48;
    ensure!(height > 0 || width > 0, "invalid image aspect ratio");
    if height == 0 {
        height = 48;
        width = ((w / h) as usize * 48).min(max_soft * 48);
    } else if width == 0 {
        width = 48;
        height = ((h / w) as usize * 48).min(max_soft * 48);
    }
    ensure!(height.checked_mul(width).is_some_and(|n| n <= max_soft * 9 * 16 * 16), "image patch budget exceeded");
    let _resized = reserve(budget, PlanStage::Bicubic, || bicubic_bound(rgb.width(), rgb.height(), width, height))?;
    let rgb = bicubic_rgb(&rgb, width, height)?;
    let size = if unified { 48 } else { 16 };
    let (pw, ph) = (width / size, height / size);
    let real = pw * ph;
    let soft = if unified { real } else { real / 9 };
    let padded = if unified { max_soft } else { max_soft * 9 };
    let dim = size * size * 3;
    let mut patches = vec![0.0f32; padded * dim];
    let mut positions = vec![(-1, -1); padded];
    for y in 0..ph {
        for x in 0..pw {
            let i = y * pw + x;
            positions[i] = (x as i64, y as i64);
            for py in 0..size {
                for px in 0..size {
                    let pixel = rgb.get_pixel((x * size + px) as u32, (y * size + py) as u32);
                    for c in 0..3 {
                        patches[i * dim + (py * size + px) * 3 + c] = f32::from(pixel[c]) * (1.0 / 255.0);
                    }
                }
            }
        }
    }
    retain(features);
    Ok(ImageInput { patches, positions, patch_dim: dim, soft_tokens: soft, patch_width: pw, patch_height: ph })
}

/// Host bytes of [`bicubic_rgb`]: both weight tables (f64 then fixed-point), the cached horizontal rows when they
/// are cached, the accumulator and the output image (a clone when the size does not change).
fn bicubic_bound(in_w: u32, in_h: u32, width: usize, height: usize) -> u64 {
    let out = (width as u64).saturating_mul(height as u64).saturating_mul(3);
    if (width as u32, height as u32) == (in_w, in_h) {
        return out;
    }
    let weights = |input: u64, output: u64| {
        let scale = input as f64 / output.max(1) as f64;
        let max_size = (2.0 * scale.max(1.0)).ceil() as u64 * 2 + 1;
        output.saturating_mul(32 + 2 * 8 * max_size)
    };
    let rows = (in_h as u64).saturating_mul(width as u64).saturating_mul(3);
    let rows = if rows <= 32 << 20 { rows } else { 0 };
    out.saturating_add(weights(in_w as u64, width as u64))
        .saturating_add(weights(in_h as u64, height as u64))
        .saturating_add(rows)
        .saturating_add((width as u64).saturating_mul(24))
}

/// Largest decoded packet (frames) of a codec with a verified bound for bounded planning: PCM and FLAC from the
/// stream parameters, MPEG audio layers from their fixed frame size. Other codecs (Vorbis setup tables, unknown
/// packet sizes) are refused.
fn bounded_packet_frames(params: &symphonia::core::codecs::audio::AudioCodecParameters) -> Result<u64> {
    use symphonia::core::codecs::audio::well_known as id;
    let codec = params.codec;
    let frames = if codec >= id::CODEC_ID_PCM_S32LE && codec <= id::CODEC_ID_PCM_MULAW {
        params.max_frames_per_packet
    } else if codec == id::CODEC_ID_FLAC {
        Some(params.max_frames_per_packet.unwrap_or(65535).min(65535))
    } else if codec == id::CODEC_ID_MP3 || codec == id::CODEC_ID_MP2 || codec == id::CODEC_ID_MP1 {
        Some(1152)
    } else {
        None
    };
    frames.filter(|f| *f > 0 && *f <= 1 << 20).ok_or_else(|| {
        UnboundedFeature(format!("audio codec {codec} has no verified host memory bound for background planning"))
            .into()
    })
}

/// Torchvision's uint8 antialiased bicubic: Keys (-0.5), normalized boundary weights, horizontal then
/// vertical passes with signed fixed-point weights and a rounded/clamped uint8 intermediate.
/// Computing one output row at a time also bounds scratch space for extreme input aspect ratios.
fn bicubic_rgb(input: &image::RgbImage, width: usize, height: usize) -> Result<image::RgbImage> {
    if (width as u32, height as u32) == input.dimensions() {
        return Ok(input.clone());
    }
    let (horizontal, hp) = resize_weights(input.width() as usize, width);
    let (vertical, vp) = resize_weights(input.height() as usize, height);
    let mut out = image::RgbImage::new(width as u32, height as u32);
    // Pillow's separable resize rounds horizontal rows to u8 before the vertical pass.
    // Cache those exact rows once; the previous loop recomputed them for each vertical tap.
    // Bound temporary storage for extreme aspect ratios and keep the original path otherwise.
    let horizontal_pixels = (input.height() as usize)
        .checked_mul(width)
        .and_then(|n| n.checked_mul(3))
        .filter(|bytes| *bytes <= 32 << 20)
        .map(|bytes| {
            let mut rows = vec![0u8; bytes];
            for y in 0..input.height() as usize {
                for (x, (start_x, wx)) in horizontal.iter().enumerate() {
                    let mut pixel = [1i64 << (hp - 1); 3];
                    for (dx, &wx) in wx.iter().enumerate() {
                        let rgb = input.get_pixel((start_x + dx) as u32, y as u32);
                        for c in 0..3 {
                            pixel[c] += i64::from(rgb[c]) * wx;
                        }
                    }
                    for c in 0..3 {
                        rows[(y * width + x) * 3 + c] = (pixel[c] >> hp).clamp(0, 255) as u8;
                    }
                }
            }
            rows
        });
    let mut accum = vec![0i64; width.checked_mul(3).context("resize dimensions overflow")?];
    for (y, (start_y, wy)) in vertical.iter().enumerate() {
        accum.fill(1i64 << (vp - 1));
        for (dy, &wy) in wy.iter().enumerate() {
            if let Some(rows) = &horizontal_pixels {
                let row = &rows[(start_y + dy) * width * 3..(start_y + dy + 1) * width * 3];
                for (sum, pixel) in accum.iter_mut().zip(row) {
                    *sum += i64::from(*pixel) * wy;
                }
            } else {
                for (x, (start_x, wx)) in horizontal.iter().enumerate() {
                    let mut pixel = [1i64 << (hp - 1); 3];
                    for (dx, &wx) in wx.iter().enumerate() {
                        let rgb = input.get_pixel((start_x + dx) as u32, (start_y + dy) as u32);
                        for c in 0..3 {
                            pixel[c] += i64::from(rgb[c]) * wx;
                        }
                    }
                    for c in 0..3 {
                        accum[x * 3 + c] += (pixel[c] >> hp).clamp(0, 255) * wy;
                    }
                }
            }
        }
        for x in 0..width {
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgb(std::array::from_fn(|c| (accum[x * 3 + c] >> vp).clamp(0, 255) as u8)),
            );
        }
    }
    Ok(out)
}

fn resize_weights(input: usize, output: usize) -> (Vec<(usize, Vec<i64>)>, u32) {
    let scale = input as f64 / output as f64;
    let support = 2.0 * scale.max(1.0);
    let inverse = 1.0 / scale.max(1.0);
    let max_size = support.ceil() as usize * 2 + 1;
    let mut largest = 0.0f64;
    let weights: Vec<_> = (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let size = (((center + support + 0.5) as usize).min(input) - start).min(max_size);
            let mut weights: Vec<f64> = (0..size)
                .map(|j| {
                    let x = ((j + start) as f64 - center + 0.5).abs() * inverse;
                    if x < 1.0 {
                        ((1.5 * x - 2.5) * x) * x + 1.0
                    } else if x < 2.0 {
                        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let total = weights.iter().sum::<f64>();
            for w in &mut weights {
                *w /= total;
                largest = largest.max(*w);
            }
            (start, weights)
        })
        .collect();
    let precision = (0..22).find(|p| (0.5 + largest * (1u64 << (p + 1)) as f64) as u64 >= 32768).unwrap_or(22);
    (
        weights
            .into_iter()
            .map(|(start, w)| (start, w.into_iter().map(|w| (w * (1u64 << precision) as f64).round() as i64).collect()))
            .collect(),
        precision,
    )
}

fn audio(raw: &[u8], max_seconds: f64, unified: bool, budget: Option<&PlanBudget>) -> Result<AudioInput> {
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    // the payload copy, the stream ring buffer, packets and metadata (each at most the payload)
    let _container = reserve(budget, PlanStage::AudioDecoder, || {
        (raw.len() as u64).saturating_mul(3).saturating_add(AUDIO_STREAM_BUFFER)
    })?;
    let stream = MediaSourceStream::new(Box::new(Cursor::new(raw.to_vec())), Default::default());
    let mut format =
        symphonia::default::get_probe().probe(&Hint::new(), stream, FormatOptions::default(), Default::default())?;
    let track = format.default_track(TrackType::Audio).context("no audio track")?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).context("missing audio codec")?.clone();
    let track_id = track.id;
    let rate = params.sample_rate.filter(|r| *r > 0 && *r <= 384000).context("invalid audio sample rate")?;
    ensure!(params.channels.as_ref().is_none_or(|c| c.count() <= 64), "audio has more than 64 channels");
    // decoder buffer (up to 8 B per sample) and the interleaved f32 copy of the largest packet
    let packet_samples = match budget {
        Some(_) => {
            let channels = params.channels.as_ref().map_or(64, |c| c.count()) as u64;
            Some(bounded_packet_frames(&params)?.saturating_mul(channels))
        }
        None => None,
    };
    let _codec = reserve(budget, PlanStage::AudioDecoder, || {
        packet_samples.unwrap_or(0).saturating_mul(12).saturating_add(AUDIO_CODEC_STATE)
    })?;
    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(&params, &Default::default())?;
    let limit = (max_seconds * f64::from(rate)).floor() as usize;
    // mono samples, at most twice the window after growth
    let _mono = reserve(budget, PlanStage::AudioPcm, || (limit as u64).saturating_mul(8))?;
    let mut mono = Vec::new();
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        ensure!(
            packet_samples.is_none_or(|n| decoded.samples_interleaved() as u64 <= n),
            "audio packet larger than the codec's declared maximum"
        );
        ensure!(decoded.spec().rate() == rate, "audio sample rate changes within the stream");
        let channels = decoded.spec().channels().count();
        ensure!(channels > 0 && channels <= 64, "invalid audio channel count");
        ensure!(
            mono.len().checked_add(decoded.frames()).is_some_and(|n| n <= limit),
            "audio exceeds the model's window"
        );
        let mut samples = vec![0.0f32; decoded.samples_interleaved()];
        decoded.copy_to_slice_interleaved(&mut samples);
        for frame in samples.chunks_exact(channels) {
            let mean = frame.iter().sum::<f32>() / channels as f32;
            ensure!(mean.is_finite(), "non-finite audio samples");
            mono.push(mean);
        }
    }
    ensure!(!mono.is_empty(), "empty audio");
    let _resampled = reserve(budget, PlanStage::AudioPcm, || resample_bound(mono.len(), rate))?;
    let mono = resample(&mono, rate);
    if unified {
        let frames = mono.len().div_ceil(640);
        // the resize may move the samples into a new buffer of the padded length
        let features_reservation =
            reserve(budget, PlanStage::AudioFeatures, || (frames as u64) * 640 * 4 + frames as u64)?;
        let mut features = mono;
        features.resize(frames * 640, 0.0);
        retain(features_reservation);
        Ok(AudioInput { features, valid: vec![true; frames], feature_dim: 640, soft_tokens: frames })
    } else {
        let features_reservation = reserve(budget, PlanStage::AudioFeatures, || {
            // frames of `mel`
            let frames = (mono.len().div_ceil(128) * 128 + 160).saturating_sub(321) / 160 + 1;
            (frames as u64) * (128 * 4 + 1)
        })?;
        // filters, bands, window, FFT buffers and plan
        let _analysis =
            reserve(budget, PlanStage::AudioAnalysis, || 257 * 128 * 8 + 257 * 128 * 16 + 512 * 16 + FFT_SCRATCH)?;
        let input = mel(&mono)?;
        retain(features_reservation);
        Ok(input)
    }
}

/// Host bytes of [`resample`] of `len` samples at `rate`: the output and both tap tables.
fn resample_bound(len: usize, rate: u32) -> u64 {
    if rate == SAMPLE_RATE {
        return len as u64 * 4;
    }
    let (mut a, mut b) = (rate, SAMPLE_RATE);
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    let (up, down) = (u64::from(SAMPLE_RATE / a), u64::from(rate / a));
    let taps = 2 * 10 * up.max(down) + 1;
    (len as u64).saturating_mul(up).div_ceil(down).saturating_mul(4).saturating_add(taps * 12)
}

fn bessel(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..=64 {
        term *= x * x / (4.0 * (k * k) as f64);
        sum += term;
        if term < sum * f64::EPSILON {
            break;
        }
    }
    sum
}

/// scipy.signal.resample_poly's default zero-padded Kaiser FIR, with the same centered output convention.
fn resample(x: &[f32], rate: u32) -> Vec<f32> {
    if rate == SAMPLE_RATE {
        return x.to_vec();
    }
    let (mut a, mut b) = (rate, SAMPLE_RATE);
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    let (up, down) = ((SAMPLE_RATE / a) as usize, (rate / a) as usize);
    let max = up.max(down);
    let half = 10 * max;
    let len = 2 * half + 1;
    let cutoff = 1.0 / max as f64;
    let mut taps = Vec::with_capacity(len);
    for i in 0..len {
        let pos = i as f64 - half as f64;
        let sinc = if pos == 0.0 {
            cutoff
        } else {
            (std::f64::consts::PI * cutoff * pos).sin() / (std::f64::consts::PI * pos)
        };
        let window = bessel(5.0 * (1.0 - (pos / half as f64).powi(2)).max(0.0).sqrt()) / bessel(5.0);
        taps.push(sinc * window);
    }
    let sum = taps.iter().sum::<f64>();
    let taps: Vec<f32> = taps.into_iter().map(|v| (v / sum) as f32 * up as f32).collect();
    let n = (x.len() * up).div_ceil(down);
    (0..n)
        .map(|j| {
            let center = j * down + half;
            let first = center.saturating_sub(len - 1).div_ceil(up);
            let last = (center / up).min(x.len() - 1);
            let mut value = 0.0f32;
            for i in first..=last {
                value += x[i] * taps[center - i * up];
            }
            value
        })
        .collect()
}

fn mel(x: &[f32]) -> Result<AudioInput> {
    use rustfft::num_complex::Complex;
    let padded = x.len().div_ceil(128) * 128;
    let frames = (padded + 160).saturating_sub(321) / 160 + 1;
    ensure!(padded + 160 >= 321, "audio is too short for the model's analysis window");
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(512);
    let window: Vec<f32> =
        (0..320).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / 320.0).cos()) as f32).collect();
    let melmax = 2595.0 * (1.0 + 8000.0f64 / 700.0).log10();
    let edges: Vec<f64> = (0..130).map(|i| 700.0 * (10.0f64.powf(melmax * i as f64 / 129.0 / 2595.0) - 1.0)).collect();
    let mut filters = vec![0.0; 257 * 128];
    for bin in 0..257 {
        let hz = bin as f64 * 16000.0 / 512.0;
        for k in 0..128 {
            filters[bin * 128 + k] = ((hz - edges[k]) / (edges[k + 1] - edges[k]))
                .min((edges[k + 2] - hz) / (edges[k + 2] - edges[k + 1]))
                .max(0.0);
        }
    }
    // Zero filter coefficients contribute only signed zero to a nonnegative finite
    // energy. Preserve the bin order of every nonzero product and the same f64 sum.
    let bands: [Vec<(usize, f64)>; 128] = std::array::from_fn(|k| {
        (0..257)
            .filter_map(|bin| {
                let weight = filters[bin * 128 + k];
                (weight != 0.0).then_some((bin, weight))
            })
            .collect()
    });
    let mut features = vec![0.0f32; frames * 128];
    let mut valid = vec![false; frames];
    for i in 0..frames {
        let end = i * 160 + 320;
        valid[i] = end >= 160 && end - 160 < x.len();
        let mut data = vec![Complex::new(0.0, 0.0); 512];
        for n in 0..320 {
            let pos = i * 160 + n;
            let sample = if pos >= 160 { x.get(pos - 160).copied().unwrap_or(0.0) } else { 0.0 };
            data[n].re = f64::from(sample * window[n]);
        }
        fft.process(&mut data);
        // Each FFT magnitude is shared by all 128 mel filters. Keep the same f64
        // magnitude operation; non-finite inputs retain the original zero-product behavior.
        let magnitudes: [f64; 257] = std::array::from_fn(|bin| data[bin].norm());
        let finite = magnitudes.iter().all(|value| value.is_finite());
        for k in 0..128 {
            let energy: f64 = if finite {
                bands[k].iter().map(|&(bin, weight)| magnitudes[bin] * weight).sum()
            } else {
                (0..257).map(|bin| magnitudes[bin] * filters[bin * 128 + k]).sum()
            };
            features[i * 128 + k] = (energy + 0.001).ln() as f32 * if valid[i] { 1.0 } else { 0.0 };
        }
    }
    let soft_tokens = valid.iter().step_by(4).filter(|v| **v).count();
    ensure!(soft_tokens > 0, "audio has no valid model frames");
    Ok(AudioInput { features, valid, feature_dim: 128, soft_tokens })
}
