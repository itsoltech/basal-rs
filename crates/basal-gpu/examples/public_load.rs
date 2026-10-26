//! Manual probe of the public constructors `GpuBackend::load` and `GpuBackend::load_split`, which take no execution
//! plan (not a test or CI gate; it runs the GPU). It loads a Llama model, warms the static template prefixes, runs
//! one forward over a rendered prompt and prints JSON lines: the load options and backend description, the warmed
//! prefix lengths and the letter logits of the forward (values and f32 bits), each with its time.
//!
//! ```text
//! cargo run -p basal-gpu --features cuda --release --example public_load -- --model DIR \
//!     [--dtype f16|bf16|f32] [--readout f32|bf16] [--kernels fused|candle] [--quantization none|q8_0|q4_k] \
//!     [--device N | --split SPEC]
//! ```
use std::time::Instant;

use anyhow::{bail, Context, Result};
use basal_core::pack::pack;
use basal_core::prompt::{render_model, Lang};
use basal_core::{Backend, Engine, ModelManifest};
use basal_gpu::{GpuBackend, Kernels, LayerSplit, Precision, Quantization, Readout};
use serde_json::json;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |name: &str| -> Result<Option<&str>> {
        match args.iter().position(|a| a == name) {
            Some(i) => args.get(i + 1).map(|v| Some(v.as_str())).with_context(|| format!("{name} needs a value")),
            None => Ok(None),
        }
    };
    let model = opt("--model")?.context("--model DIR required")?;
    let dtype = opt("--dtype")?.unwrap_or("f16");
    let readout_name = opt("--readout")?.unwrap_or("f32");
    let kernels_name = opt("--kernels")?.unwrap_or("fused");
    let quantization_name = opt("--quantization")?.unwrap_or("none");
    let precision = Precision::parse(dtype)?;
    let readout = match readout_name {
        "f32" => Readout::F32,
        "bf16" => Readout::Bf16Rounded,
        other => bail!("unknown readout {other:?} (f32, bf16)"),
    };
    let kernels = Kernels::parse(kernels_name)?;
    let quantization: Quantization = quantization_name.parse()?;
    let split: Option<LayerSplit> = opt("--split")?.map(str::parse).transpose()?;
    let device: usize = opt("--device")?.map(str::parse).transpose()?.unwrap_or(0);

    let manifest = ModelManifest::load(std::path::Path::new(model))?;
    let cfg = manifest.config.llama()?.clone();
    let t = Instant::now();
    let backend = match &split {
        Some(split) => GpuBackend::load_split(&manifest.dir, &cfg, precision, readout, kernels, quantization, split)?,
        None => GpuBackend::load(&manifest.dir, &cfg, precision, readout, kernels, quantization, device)?,
    };
    let load_s = t.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({
            "event": "load",
            "constructor": if split.is_some() { "GpuBackend::load_split" } else { "GpuBackend::load" },
            "model": manifest.name,
            "dtype": dtype,
            "readout": readout_name,
            "kernels": kernels_name,
            "quantization": quantization_name,
            "placement": split.as_ref().map_or_else(|| format!("device {device}"), |s| format!("split {s}")),
            "load_s": load_s,
            "describe": backend.describe(),
        })
    );

    let mut engine = Engine::new(manifest, backend)?;
    let t = Instant::now();
    let prefixes = engine.warm_static_prefixes()?;
    println!("{}", json!({"event": "warm_static_prefixes", "lengths": prefixes, "s": t.elapsed().as_secs_f64()}));

    let m = &engine.manifest;
    let text = render_model(
        m.chat_style,
        m.prompt_format,
        &m.bos_token,
        "The probe loaded the model through its public constructor.",
        "Did the forward run?",
        &["yes", "no"],
        Lang::En,
    )?;
    let ids = engine.tok.encode(&text)?;
    let &[.., a, b] = ids.as_slice() else { bail!("the rendered prompt has fewer than two tokens") };
    // The last two prompt tokens stand in for letters: any token ids are valid readout columns for this probe.
    let letters = vec![vec![a, b]];
    let row = pack(std::slice::from_ref(&ids));
    let t = Instant::now();
    let logits = engine.backend.letter_logits(&[&row], &[&letters])?;
    let s = t.elapsed().as_secs_f64();
    let values: Vec<f32> = logits.iter().flatten().flatten().copied().collect();
    println!(
        "{}",
        json!({
            "event": "forward",
            "tokens": ids.len(),
            "letters": [a, b],
            "logits": values,
            "logits_f32_bits": values.iter().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>(),
            "finite": values.iter().all(|v| v.is_finite()),
            "s": s,
            "attention_kernel": engine.backend.describe().get("attention_kernel").cloned(),
        })
    );
    Ok(())
}
