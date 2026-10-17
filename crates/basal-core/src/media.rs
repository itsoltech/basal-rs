//! Bounded native media decoding and the processor contracts of the pinned Gemma 4 models.
//! No URL fetching, external decoder processes or persistent storage of request media.

use std::io::Cursor;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use base64::Engine;
use serde_json::Value;

pub const MAX_ITEMS: usize = 8;
pub const MAX_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_PIXELS: u64 = 50_000_000;
pub const SAMPLE_RATE: u32 = 16000;

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
            let raw = base64::engine::general_purpose::STANDARD
                .decode(data)
                .with_context(|| format!("media {i}: invalid base64"))?;
            ensure!(raw.len() <= MAX_BYTES, "media {i}: more than 20 MiB");
            let item = match entry.get("type").and_then(Value::as_str) {
                Some("image") => {
                    Input::Image(image(&raw, max_soft, self.unified).with_context(|| format!("media {i}: image"))?)
                }
                Some("audio") => Input::Audio(
                    audio(&raw, self.max_audio_seconds, self.unified).with_context(|| format!("media {i}: audio"))?,
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
        Ok(ProcessedMedia { items, high_detail: detail == "high" })
    }
}

fn image(raw: &[u8], max_soft: usize, unified: bool) -> Result<ImageInput> {
    use image::ImageDecoder;
    let mut reader = image::ImageReader::new(Cursor::new(raw)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(400 * 1024 * 1024);
    reader.limits(limits);
    let decoder = reader.into_decoder()?;
    let (w, h) = decoder.dimensions();
    ensure!(w > 0 && h > 0 && u64::from(w) * u64::from(h) <= MAX_PIXELS, "image exceeds 50 million pixels");
    let rgb = image::DynamicImage::from_decoder(decoder)?.into_rgb8();
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
    Ok(ImageInput { patches, positions, patch_dim: dim, soft_tokens: soft, patch_width: pw, patch_height: ph })
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

fn audio(raw: &[u8], max_seconds: f64, unified: bool) -> Result<AudioInput> {
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    let stream = MediaSourceStream::new(Box::new(Cursor::new(raw.to_vec())), Default::default());
    let mut format =
        symphonia::default::get_probe().probe(&Hint::new(), stream, FormatOptions::default(), Default::default())?;
    let track = format.default_track(TrackType::Audio).context("no audio track")?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).context("missing audio codec")?.clone();
    let track_id = track.id;
    let rate = params.sample_rate.filter(|r| *r > 0 && *r <= 384000).context("invalid audio sample rate")?;
    ensure!(params.channels.as_ref().is_none_or(|c| c.count() <= 64), "audio has more than 64 channels");
    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(&params, &Default::default())?;
    let limit = (max_seconds * f64::from(rate)).floor() as usize;
    let mut mono = Vec::new();
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
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
    let mono = resample(&mono, rate);
    if unified {
        let frames = mono.len().div_ceil(640);
        let mut features = mono;
        features.resize(frames * 640, 0.0);
        Ok(AudioInput { features, valid: vec![true; frames], feature_dim: 640, soft_tokens: frames })
    } else {
        mel(&mono)
    }
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
