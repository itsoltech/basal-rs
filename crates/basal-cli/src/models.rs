//! The catalog of published basal models and their versioned execution profiles: the only place where a model name
//! selects runtime policy.
//!
//! A catalog entry names a Hugging Face repository, the revision this runtime pins and the profile of the model. The
//! profile is matched by the name in the model's `basal.json` ([`ModelManifest::name`]); a model of another name
//! gets the generic profile of its architecture, without measured model/GPU rules. Metadata, tokenizer, prompt
//! contract and calibration always come from the model files (`ModelManifest`), never from this catalog.
//!
//! A profile holds data only: its kernel set, the versioned dispatch rule with the GPUs on which measured
//! policies apply ([`basal_gpu::DispatchRule`]), the ordered list of bundled GEMM artifacts it may use (pinned by
//! the digest of their normalized content, `gemm_artifact`) and the scheduling policy of `basal serve`. Profiles are
//! versioned (`.v1`): a change of measured policy is a new profile version, not an edit of an existing one; a
//! new GEMM table in `gemm-tables/` is used only by the profiles that list it.

use std::time::Duration;

use basal_core::architecture::Architecture;
use basal_core::ModelManifest;
use basal_gpu::{AttentionProducts, DispatchRule, KernelSet, ResolvedExecutionPlan};

/// Owner of a repository given without one (`--model basal-1.5-max` is `Remek/basal-1.5-max`).
pub const DEFAULT_OWNER: &str = "Remek";

/// The published models of the catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelId {
    OmniPico,
    Basal15Mini,
    Basal15_4_5B,
    Basal15Max,
    Basal20Mini,
    Basal20_4_5B,
    Basal20Max,
    Omni10Mini,
    Omni10Max,
    Omni11Max,
    Omni10Ultra,
}

/// Model of the built-in configuration: the pinned basal-2.0-4.5B checkpoint.
pub const DEFAULT_MODEL_ID: ModelId = ModelId::Basal20_4_5B;

/// A published basal model pinned to a metadata revision. Runtime verification is recorded separately in reports.
pub struct CatalogEntry {
    pub id: ModelId,
    /// short name for `--model`
    pub short: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    /// checkpoint size (download), GB
    pub size_gb: f64,
    pub profile: &'static Profile,
}

impl CatalogEntry {
    /// The model name of its `basal.json` (the repository name).
    pub fn name(&self) -> &'static str {
        self.repo.rsplit('/').next().unwrap_or(self.repo)
    }

    /// `dtype` of `--model NAME`: Gemma and Qwen3.5 resolve their architecture/device default (`auto`), Llama f16.
    pub fn cli_dtype(&self) -> &'static str {
        match self.profile.dispatch {
            DispatchRule::GemmaV1 { .. }
            | DispatchRule::Qwen35V1
            | DispatchRule::Qwen35V2
            | DispatchRule::Qwen35V3
            | DispatchRule::Qwen35V4 => "auto",
            _ => "f16",
        }
    }
}

/// A bundled GEMM table a profile may use: its file in `gemm-tables/` and the SHA-256 of its normalized content.
#[derive(Clone, Copy, Debug)]
pub struct GemmPin {
    pub file: &'static str,
    pub sha256: &'static str,
}

/// What a plan must provide for a scheduling feature to apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanFeature {
    /// The first stage computes Gemma attention products in FP16.
    F16AttentionProducts,
}

/// How the main lane of `basal serve` collects requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Start every batch as soon as the lane is free.
    Immediate,
    /// After a batch of several requests, when the queue is empty and the model was busy less than `idle` ago,
    /// collect returning clients for at most `window` before planning the next batch. Applies only when the plan
    /// has `requires`.
    CoalesceReturning { window: Duration, idle: Duration, requires: PlanFeature },
}

impl Profile {
    /// The bundled GEMM tables of the profile, in order of preference.
    pub fn gemm_pins(&self) -> impl Iterator<Item = &'static GemmPin> {
        self.gemm.iter().flat_map(|set| set.iter())
    }
}

impl SchedulingPolicy {
    /// The policy for an instance with `plan`: features the plan lacks fall back to [`Self::Immediate`].
    pub fn resolve(self, plan: &ResolvedExecutionPlan) -> Self {
        match self {
            Self::CoalesceReturning { requires: PlanFeature::F16AttentionProducts, .. }
                if plan.first_stage_attention_products() == Some(AttentionProducts::F16) =>
            {
                self
            }
            Self::CoalesceReturning { .. } => Self::Immediate,
            Self::Immediate => self,
        }
    }
}

/// The versioned execution profile of a model.
pub struct Profile {
    pub id: &'static str,
    pub kernel_set: KernelSet,
    pub dispatch: DispatchRule,
    /// Bundled GEMM tables in order of preference (sets of tables, each set in its order); the first valid for the
    /// GPU, cuBLASLt version, dtype and weight shapes is used.
    pub gemm: &'static [&'static [GemmPin]],
    pub scheduling: SchedulingPolicy,
    /// `gemm_table: auto` may import a table of the GEMM cache of earlier versions (`<model>-<dtype>.json`, made
    /// before profiles existed). Only the v1 compatibility profiles do; a later profile version starts from its
    /// own pins and bindings.
    pub legacy_cache_import: bool,
}

const ADA: &str = "NVIDIA RTX 6000 Ada Generation";
const H100_PCIE: &str = "NVIDIA H100 PCIe";

const LLAMA_V1: DispatchRule = DispatchRule::LlamaV1 { balanced_auto_gpus: &[] };
/// Basal 2.0: `auto` selects the balanced QK1 attention on the measured GPU
/// (reports/precision-performance-ada-2026-10-16).
const BASAL_2_0_V1: DispatchRule = DispatchRule::LlamaV1 { balanced_auto_gpus: &[ADA] };
/// Omni 1.0 mini/max: FP16 decoder and FP16 attention products on the measured GPU.
const OMNI_1_0_V1: DispatchRule =
    DispatchRule::GemmaV1 { f16_decoder_auto_gpus: &[ADA], f16_attention_auto_gpus: &[ADA], fused_ops_auto_gpus: &[] };
/// Omni 1.1 max: FP16 decoder, but FP32 attention products on the same GPU (only the decoder was evaluated there).
const OMNI_1_1_MAX_V1: DispatchRule =
    DispatchRule::GemmaV1 { f16_decoder_auto_gpus: &[ADA], f16_attention_auto_gpus: &[], fused_ops_auto_gpus: &[] };
/// Omni Ultra: FP16 decoder with the fused FP16 operations and FP32 attention products on H100 PCIe
/// (reports/ultra-attention-2026-10-17).
const OMNI_ULTRA_V1: DispatchRule = DispatchRule::GemmaV1 {
    f16_decoder_auto_gpus: &[H100_PCIE],
    f16_attention_auto_gpus: &[],
    fused_ops_auto_gpus: &[H100_PCIE],
};
const GEMMA_V1: DispatchRule =
    DispatchRule::GemmaV1 { f16_decoder_auto_gpus: &[], f16_attention_auto_gpus: &[], fused_ops_auto_gpus: &[] };

/// Llama models of hidden size 1536 (basal-1.5 mini and Basal 2.0 mini share the weight shapes and these tables).
const GEMM_LLAMA_MINI: &[GemmPin] = &[
    pin(
        "nvidia-a100-sxm4-80gb--basal-1.5-mini--f16--cublaslt120901.json",
        "f98dcb47010a7070491c93ce1ac03ffd887fbd3876f8e7d1fe8d0bc856ec9df5",
    ),
    pin(
        "nvidia-a10--basal-1.5-mini--f16--cublaslt120901.json",
        "57a70b88de87101479dcad57ea16ff227b0ad829b5a131fd736e9658eb63e44f",
    ),
    pin(
        "nvidia-geforce-rtx-4060--basal-2.0-mini--f16--cublaslt120901.json",
        "2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96",
    ),
    pin(
        "nvidia-h100-pcie--basal-1.5-mini--f16--cublaslt120901.json",
        "36c9cc820ca937545871ac02b1d068f123fb7b33f54c4e8a3e08cf4e8279774f",
    ),
    pin(
        "nvidia-l40--basal-1.5-mini--f16--cublaslt120901.json",
        "cc16731f5a3c86a7ce37463cad667392397f83fbbe81c272f0827e06bdc8b107",
    ),
    pin(
        "nvidia-l40s--basal-1.5-mini--f16--cublaslt120901.json",
        "c292c9b000fb1e27c16e3559a34e9711921cb7ebb00d37b4a8bb5d311696f15f",
    ),
    pin(
        "nvidia-l4--basal-1.5-mini--f16--cublaslt120901.json",
        "77c1e248b2f3f5a3024f6599c1fc01dd6480e5a8a4c9b1f9a251e1089537f1ba",
    ),
    pin(
        "nvidia-rtx-6000-ada-generation--basal-1.5-mini--f16--cublaslt120901.json",
        "cf5ba1a8a57955a2c97e10cfe0ecdb814ac3413a86a9d59e44aa290e1b776575",
    ),
    pin(
        "nvidia-rtx-a5000--basal-1.5-mini--f16--cublaslt120901.json",
        "153cd69293884eb3a1d707d2e2733e041127da43c04f68fff68941033aa7952f",
    ),
    pin(
        "nvidia-rtx-a6000--basal-1.5-mini--f16--cublaslt120901.json",
        "3d6dd11b6fa29ba668dc421cfee25f804b1140842371b73c00a4de72f76f8488",
    ),
    pin(
        "nvidia-rtx-pro-6000-blackwell-server-edition--basal-1.5-mini--f16--cublaslt120901.json",
        "97ee8354f70a62dd81564dc8dac926e1cc6a23b53bf50e2f7b252be55e8a8945",
    ),
];

/// Llama models of hidden size 2048 (basal-1.5 4.5B and Basal 2.0 4.5B).
const GEMM_LLAMA_4_5B: &[GemmPin] = &[
    pin(
        "nvidia-a100-sxm4-80gb--basal-1.5-4.5B--f16--cublaslt120901.json",
        "55e992b4fc119c77668eb2bd178a436be24576245c6af3b70de972d1759cf2d8",
    ),
    pin(
        "nvidia-a10--basal-1.5-4.5B--f16--cublaslt120901.json",
        "7df0c9317d8f234a37fc1dcd415d94ab8b88327f56844dd9a6a45b01f9681f7e",
    ),
    pin(
        "nvidia-h100-pcie--basal-1.5-4.5B--f16--cublaslt120901.json",
        "30304d2d0ac1216ad3f0912116f6bff5895f0ce727223f5e9bae7bd8e31d506a",
    ),
    pin(
        "nvidia-l40--basal-1.5-4.5B--f16--cublaslt120901.json",
        "17be700da3794da36cdaef6b21bcb9388e291b75d76f5bd616b2e0251b5faa20",
    ),
    pin(
        "nvidia-l40s--basal-1.5-4.5B--f16--cublaslt120901.json",
        "a7caf2e6c293460d35d38eb2e66e4f57e79de1e6e1c6c719f7791ab91740badd",
    ),
    pin(
        "nvidia-l4--basal-1.5-4.5B--f16--cublaslt120901.json",
        "c10fb77a7cd05d34c4e323c8219d1995c37acabed0eda7af938fb306929aaf7f",
    ),
    pin(
        "nvidia-rtx-6000-ada-generation--basal-1.5-4.5B--f16--cublaslt120901.json",
        "6d90b8cbe6a87fc40ea45f36a6660c834b9e7823b79553ab1ab75e1231b7d6c2",
    ),
    pin(
        "nvidia-rtx-a5000--basal-1.5-4.5B--f16--cublaslt120901.json",
        "e4b7b1cfdb274177276283ad0ca305534a56c119a58b85aa047ffe74f1c6c71c",
    ),
    pin(
        "nvidia-rtx-a6000--basal-1.5-4.5B--f16--cublaslt120901.json",
        "05373e78220aa63e9a5b01c2d956c224dba0c8f96cc8cf456cf958a867f92fd6",
    ),
    pin(
        "nvidia-rtx-pro-6000-blackwell-server-edition--basal-1.5-4.5B--f16--cublaslt120901.json",
        "ff8b32201c8ac632eb7db97a23cb59e5f0158c5e35a7d14270ec8cdcb6a78cdb",
    ),
];

/// Llama models of hidden size 4096 (basal-1.5 max and Basal 2.0 max).
const GEMM_LLAMA_MAX: &[GemmPin] = &[
    pin(
        "nvidia-a100-sxm4-80gb--basal-1.5-max--f16--cublaslt120901.json",
        "d3349e60a3d1dc5ca1d1fa2ae439b17a134bc7633b3ed619fbf6525e293e189b",
    ),
    pin(
        "nvidia-h100-pcie--basal-1.5-max--f16--cublaslt120901.json",
        "1f7dd10e53306bed6340586c828224ee3bb392ab1d53544a6ca2c900b6c6ec05",
    ),
    pin(
        "nvidia-l40--basal-1.5-max--f16--cublaslt120901.json",
        "58a5dab437c9157d707fb89bb2eb6d4b14e88dbe346519db183ef3abb81c1deb",
    ),
    pin(
        "nvidia-l40s--basal-1.5-max--f16--cublaslt120901.json",
        "af1feed5f5a62768b42d9eeec8d735d855c4945d4a9d4af370b530c738e76f8f",
    ),
    pin(
        "nvidia-rtx-6000-ada-generation--basal-1.5-max--f16--cublaslt120901.json",
        "e69f67b91ee28ed891f2522c4804238fe698927eb8573ff3221321f2e28090fc",
    ),
    pin(
        "nvidia-rtx-a6000--basal-1.5-max--f16--cublaslt120901.json",
        "abdb19a12f5e3e5611548c43d52f036cd1e9501572f89f031d8a1febb5af970e",
    ),
    pin(
        "nvidia-rtx-pro-6000-blackwell-server-edition--basal-1.5-max--f16--cublaslt120901.json",
        "caa4d7d3c6b7e767a4d738f8c1f90db15622125fa5b7f46a868368cd4519ed35",
    ),
];

const GEMM_OMNI_MINI: &[GemmPin] = &[pin(
    "nvidia-rtx-6000-ada-generation--basal-omni-1.0-mini--f16--cublaslt120901.json",
    "3575e793ad4a7ecb738b6248f1913c151a3179b770fe6db8a5e4ebd2a46fb8fe",
)];

/// Omni 1.0 max and 1.1 max (the table applies only where it covers the model's weight shapes).
const GEMM_OMNI_MAX: &[GemmPin] = &[pin(
    "nvidia-rtx-6000-ada-generation--basal-omni-1.0-max--f16--cublaslt120901.json",
    "d9cdfb83903b38d165af616c21539d87fb0d527c3a2113cd320defd068f74ad0",
)];

const GEMM_OMNI_ULTRA: &[GemmPin] = &[pin(
    "nvidia-h100-pcie--basal-omni-1.0-ultra--f16--cublaslt120901.json",
    "452d8dff7074b34ce3a26a19e2a637a74907fb111a44c5b1a614e79e61142d70",
)];

/// Models outside the catalog: every table bundled when the profiles were introduced, so that such a model keeps
/// the table it used before (these sets hold at most one table per GPU and weight shapes); later tables are not
/// added.
const GEMM_ANY_V1: &[&[GemmPin]] =
    &[GEMM_LLAMA_MINI, GEMM_LLAMA_4_5B, GEMM_LLAMA_MAX, GEMM_OMNI_MINI, GEMM_OMNI_MAX, GEMM_OMNI_ULTRA];

const fn pin(file: &'static str, sha256: &'static str) -> GemmPin {
    GemmPin { file, sha256 }
}

/// A v1 profile: kernel set V1, the measured policies as before profiles existed, and import of the earlier GEMM
/// cache.
const fn v1(
    id: &'static str,
    dispatch: DispatchRule,
    gemm: &'static [&'static [GemmPin]],
    scheduling: SchedulingPolicy,
) -> Profile {
    Profile { id, kernel_set: KernelSet::V1, dispatch, gemm, scheduling, legacy_cache_import: true }
}

const fn llama(id: &'static str, dispatch: DispatchRule, gemm: &'static [&'static [GemmPin]]) -> Profile {
    v1(id, dispatch, gemm, SchedulingPolicy::Immediate)
}

const fn gemma(
    id: &'static str,
    dispatch: DispatchRule,
    gemm: &'static [&'static [GemmPin]],
    scheduling: SchedulingPolicy,
) -> Profile {
    v1(id, dispatch, gemm, scheduling)
}

static BASAL_15_MINI: Profile = llama("basal-1.5-mini.v1", LLAMA_V1, &[GEMM_LLAMA_MINI]);
static BASAL_15_4_5B: Profile = llama("basal-1.5-4.5B.v1", LLAMA_V1, &[GEMM_LLAMA_4_5B]);
static BASAL_15_MAX: Profile = llama("basal-1.5-max.v1", LLAMA_V1, &[GEMM_LLAMA_MAX]);
static BASAL_20_MINI: Profile = llama("basal-2.0-mini.v1", BASAL_2_0_V1, &[GEMM_LLAMA_MINI]);
static BASAL_20_4_5B: Profile = llama("basal-2.0-4.5B.v1", BASAL_2_0_V1, &[GEMM_LLAMA_4_5B]);
static BASAL_20_MAX: Profile = llama("basal-2.0-max.v1", BASAL_2_0_V1, &[GEMM_LLAMA_MAX]);
static OMNI_10_MINI: Profile =
    gemma("basal-omni-1.0-mini.v1", OMNI_1_0_V1, &[GEMM_OMNI_MINI], SchedulingPolicy::Immediate);
/// Omni 1.0 max with FP16 attention products: collect returning clients after a concurrent batch, avoiding
/// repeated 1+(N-1) forwards.
static OMNI_10_MAX: Profile = gemma(
    "basal-omni-1.0-max.v1",
    OMNI_1_0_V1,
    &[GEMM_OMNI_MAX],
    SchedulingPolicy::CoalesceReturning {
        window: Duration::from_millis(10),
        idle: Duration::from_millis(100),
        requires: PlanFeature::F16AttentionProducts,
    },
);
static OMNI_11_MAX: Profile =
    gemma("basal-omni-1.1-max.v1", OMNI_1_1_MAX_V1, &[GEMM_OMNI_MAX], SchedulingPolicy::Immediate);
static OMNI_10_ULTRA: Profile =
    gemma("basal-omni-1.0-ultra.v1", OMNI_ULTRA_V1, &[GEMM_OMNI_ULTRA], SchedulingPolicy::Immediate);

/// Profile of a Llama model outside the catalog: no measured model/GPU rules.
static LLAMA_GENERIC: Profile = llama("llama-generic.v1", LLAMA_V1, GEMM_ANY_V1);
/// Profile of a Gemma (Omni) model outside the catalog: BF16 `auto`, FP32 attention products.
static GEMMA_GENERIC: Profile = gemma("gemma-generic.v1", GEMMA_V1, GEMM_ANY_V1, SchedulingPolicy::Immediate);

static OMNI_PICO: Profile = Profile {
    id: "basal-omni-pico.v4",
    kernel_set: KernelSet::Qwen35V4,
    dispatch: DispatchRule::Qwen35V4,
    gemm: &[&[pin(
        "nvidia-geforce-rtx-4060--basal-omni-pico--bf16--cublaslt120901.json",
        "72fb35fcb899c71c6ed7e52ddb64e20686848f2f7ed8304401147bdac1bdee2e",
    )]],
    scheduling: SchedulingPolicy::Immediate,
    legacy_cache_import: false,
};
static QWEN35_GENERIC: Profile = Profile {
    id: "qwen35-generic.v1",
    kernel_set: KernelSet::Qwen35V1,
    dispatch: DispatchRule::Qwen35V1,
    gemm: &[],
    ..OMNI_PICO
};

pub static CATALOG: [CatalogEntry; 11] = [
    entry(
        ModelId::OmniPico,
        "pico",
        "Remek/basal-omni-pico",
        "a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9",
        1.51,
        &OMNI_PICO,
    ),
    entry(
        ModelId::Basal15Mini,
        "mini",
        "Remek/basal-1.5-mini",
        "1978d0705ce09cd2d7d8d3e87b304468121e7255",
        3.1,
        &BASAL_15_MINI,
    ),
    entry(
        ModelId::Basal15_4_5B,
        "4.5B",
        "Remek/basal-1.5-4.5B",
        "784a683bfadcc8865238fc0fc74a83b4000269bc",
        9.1,
        &BASAL_15_4_5B,
    ),
    entry(
        ModelId::Basal15Max,
        "max",
        "Remek/basal-1.5-max",
        "be1b5ee7e7a9755a931262fa7fab4f59be0fd03c",
        22.3,
        &BASAL_15_MAX,
    ),
    entry(
        ModelId::Basal20Mini,
        "2.0-mini",
        "Remek/basal-2.0-mini",
        "cb81b32ce89b0b87e62fcdad28f0435e25c953f3",
        3.1,
        &BASAL_20_MINI,
    ),
    entry(
        ModelId::Basal20_4_5B,
        "2.0-4.5B",
        "Remek/basal-2.0-4.5B",
        "e5ba9149760d99b73b1963d8e0ccf948b58afacb",
        9.1,
        &BASAL_20_4_5B,
    ),
    entry(
        ModelId::Basal20Max,
        "2.0-max",
        "Remek/basal-2.0-max",
        "2fffc5cc41e5cf7a2f752f38f17e7b4d9a4a8252",
        22.3,
        &BASAL_20_MAX,
    ),
    entry(
        ModelId::Omni10Mini,
        "omni-mini",
        "Remek/basal-omni-1.0-mini",
        "73c993dc3d9ab3cb8887ac0b1ee1cf1c92783240",
        15.9,
        &OMNI_10_MINI,
    ),
    entry(
        ModelId::Omni10Max,
        "omni-max",
        "Remek/basal-omni-1.0-max",
        "e538228c4f3ffd96dde298a073ed10247da0d637",
        23.9,
        &OMNI_10_MAX,
    ),
    entry(
        ModelId::Omni11Max,
        "omni-1.1-max",
        "Remek/basal-omni-1.1-max",
        "c8ab50cd973c116b3ca6bd9e4d574f713bab1f68",
        23.92,
        &OMNI_11_MAX,
    ),
    entry(
        ModelId::Omni10Ultra,
        "omni-ultra",
        "Remek/basal-omni-1.0-ultra",
        "b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4",
        62.55,
        &OMNI_10_ULTRA,
    ),
];

const fn entry(
    id: ModelId,
    short: &'static str,
    repo: &'static str,
    revision: &'static str,
    size_gb: f64,
    profile: &'static Profile,
) -> CatalogEntry {
    CatalogEntry { id, short, repo, revision, size_gb, profile }
}

/// The catalog entry of `id`.
pub fn get(id: ModelId) -> &'static CatalogEntry {
    CATALOG.iter().find(|e| e.id == id).unwrap_or_else(|| unreachable!("every ModelId has one catalog entry"))
}

/// The model of the built-in configuration.
pub fn default_model() -> &'static CatalogEntry {
    get(DEFAULT_MODEL_ID)
}

/// A catalog model by its short name, `owner/name` repository or repository name (case-insensitive).
pub fn find(name: &str) -> Option<&'static CatalogEntry> {
    if name.eq_ignore_ascii_case("omni-pico") {
        return Some(get(ModelId::OmniPico));
    }
    CATALOG.iter().find(|m| {
        m.short.eq_ignore_ascii_case(name) || m.repo.eq_ignore_ascii_case(name) || m.name().eq_ignore_ascii_case(name)
    })
}

/// A catalog model by its repository `owner/name`.
pub fn by_repo(repo: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|m| m.repo == repo)
}

/// The generic Llama profile, without measured model/GPU rules or pins: the profile of a custom thinking checkpoint,
/// whatever its `basal.json` name.
pub fn generic_llama_profile() -> &'static Profile {
    &LLAMA_GENERIC
}

/// The profile of a loaded model: the catalog's for its `basal.json` name, else the generic one of its
/// architecture.
pub fn profile_for(manifest: &ModelManifest) -> &'static Profile {
    if let Some(entry) = CATALOG.iter().find(|m| m.name() == manifest.name) {
        return entry.profile;
    }
    match manifest.config {
        Architecture::Llama(_) => &LLAMA_GENERIC,
        Architecture::Gemma4(_) => &GEMMA_GENERIC,
        Architecture::Qwen35(_) => &QWEN35_GENERIC,
    }
}
