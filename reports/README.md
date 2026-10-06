# Raporty

Każdy katalog zawiera opis (`README.md`) i dane, z których powstały liczby.
Metodyka: [docs/BENCHMARKS.md](../docs/BENCHMARKS.md).

## basal-1.5-max

| Katalog | Zawartość |
|---|---|
| [rust-cuda-1.5-max](rust-cuda-1.5-max/README.md) | zgodność z FP32 upstream, pojedyncza decyzja, długie stany, serwer HTTP, współbieżność, tabele GEMM |
| [rust-cuda-1.5-max/mixed-load](rust-cuda-1.5-max/mixed-load/README.md) | ruch mieszany, Rust a upstream, warianty harmonogramu |
| [rust-cuda-1.5-max/profile-floor](rust-cuda-1.5-max/profile-floor/README.md) | rozkład czasu forwardu: GEMM, attention, reszta |
| [rust-cuda-1.5-max/attn-precision](rust-cuda-1.5-max/attn-precision/README.md) | attention na tensor cores: warianty precyzji, organizacja kernela |
| [rust-cuda-1.5-max/gemm-retune](rust-cuda-1.5-max/gemm-retune/README.md) | tabela GEMM niezależna od partii dobrana na M 128–16384 |
| [rust-cuda-1.5-max/gemm-equiv](rust-cuda-1.5-max/gemm-equiv/README.md) | algorytm GEMM na klasę M z grupy bitowo identycznych algorytmów |
| [rust-cuda-1.5-max/attn-kernel](rust-cuda-1.5-max/attn-kernel/README.md) | iteracje kernela attention |
| [rust-cuda-1.5-max/power](rust-cuda-1.5-max/power/README.md) | konfiguracja domyślna przy 250 W i 300 W, ruch mieszany |
| [rust-cuda-1.5-max/multi-model](rust-cuda-1.5-max/multi-model/README.md) | basal-1.5-max, 1.5-4.5B i 1.5-mini w jednym procesie |
| [baseline-cuda-1.5-max](baseline-cuda-1.5-max/README.md) | upstream v1.5.0 na tej samej karcie |
| `reference-1.5-max-fp32`, `-cases`, `-cases2` | referencja upstream FP32: 44 przykłady basal-bench, żądania System One z `multi`, `act`, `facts`, `evidence` |

## basal-1.0-4.5B

| Katalog | Zawartość |
|---|---|
| [rust-cuda-v1](rust-cuda-v1/README.md) | pierwsza wersja CUDA: zgodność, pojedyncza decyzja, żądania, profil |
| [rust-cuda-v2](rust-cuda-v2/README.md) | niezależność od partii, serwer HTTP |
| [baseline-cuda](baseline-cuda/README.md) | upstream na RTX 6000 Ada, telemetria GPU |
| [metal-m1-pro](metal-m1-pro/README.md) | backend Metal na Apple M1 Pro |
| [large-choice](large-choice/README.md) | Choice 11–255: ocena strategii grupowej |
| `reference-cuda-fp32`, `reference-cuda-bf16`, `reference-mlx-bf16` | referencje upstream: PyTorch FP32 i BF16 na CUDA, MLX BF16 |

Pliki `reference-*.log` to logi eksportu referencji.
