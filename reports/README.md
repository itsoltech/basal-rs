# Raporty

Nowe raporty zapisują tylko: co sprawdzono, wynik, zmierzoną poprawę lub
regresję oraz zakres weryfikacji. Warunki i ograniczenia podajemy krótko,
gdy są potrzebne do interpretacji liczb. Hipotezy i plany nie trafiają do
raportów; surowe dane przechowujemy poza Gitem. Starsze katalogi mogą
zawierać dane pomiarowe.
Metodyka: [docs/BENCHMARKS.md](../docs/BENCHMARKS.md).

## Kampanie optymalizacji

| Katalog | Zawartość |
|---|---|
| [inference-campaign-2026-10-09](inference-campaign-2026-10-09/README.md) | H100 i RTX 6000 Ada: SiLU, RMSNorm, cache stanów, budżet partii i harmonogram; zgodność wyników, latency i throughput |
| [inference-profile-h100-2026-10-09](inference-profile-h100-2026-10-09/README.md) | H100: Nsight Systems/Compute, SW128 w WGMMA, bitowa zgodność, latency/throughput oraz HTTP A/B/B/A |
| [multigpu-placement-2026-10-11](multigpu-placement-2026-10-11/README.md) | 4.5B i mini na dwóch RTX 6000 Ada: bitowa zgodność, przypisanie modeli do GPU, ruch równoległy i widoczność CUDA |
| [multigpu-isolation-2026-10-11](multigpu-isolation-2026-10-11/README.md) | RTX 6000 Ada: mini bez/z obciążonym 4.5B, stare/nowe single-GPU, identyczny ruch A/B/B/A i pełne odpowiedzi |
| [multigpu-replicas-2026-10-11](multigpu-replicas-2026-10-11/README.md) | RTX 6000 Ada: jedna i dwie repliki mini, zgodność i routing, single-GPU A/B z obserwowanym spadkiem mediany przepustowości |
| [multigpu-replicas-control-2026-10-11](multigpu-replicas-control-2026-10-11/README.md) | RTX 6000 Ada: dodatkowa kontrola single-GPU A/B, żądany zegar 1200 MHz, wspólna maska CPU i wszystkie 16 przebiegów |
| [multigpu-layer-revalidation-2026-10-14](multigpu-layer-revalidation-2026-10-14/README.md) | podział warstw, bieżące źródła (najnowszy pomiar): zestawienie L1–L3 na dwóch RTX 6000 Ada, utrzymane sygnały i ich rozstrzygnięcia; [erratum `load_s`](multigpu-layer-revalidation-2026-10-14/LOAD_S.md) |
| [multigpu-layer-l1-revalidation-2026-10-13](multigpu-layer-l1-revalidation-2026-10-13/README.md) | podział warstw L1: bitowa zgodność, odmowy, HTTP, pojemność pod sztucznym limitem, profile transferów; [poprawka poleceń](multigpu-layer-l1-revalidation-2026-10-13/COMMANDS.md) |
| [multigpu-layer-l2-revalidation-2026-10-14](multigpu-layer-l2-revalidation-2026-10-14/README.md) | podział warstw L2: regresja `none` na jednym GPU, kompromis podziału; [decyzja użytkownika](multigpu-layer-l2-revalidation-2026-10-14/DECISION.md) |
| [multigpu-layer-l3-revalidation-2026-10-14](multigpu-layer-l3-revalidation-2026-10-14/README.md) | podział warstw L3: regresja `q8_0` i `q4_k` na jednym GPU, podział w tych trybach; [rozstrzygnięcie koordynatora](multigpu-layer-l3-revalidation-2026-10-14/DECISION.md) |
| [multigpu-layer-2026-10-13](multigpu-layer-2026-10-13/README.md) | podział warstw, poprzedni kandydat (historia): zestawienie L1–L3 z [poprawką zakresu](multigpu-layer-2026-10-13/SCOPE.md) |

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

## Obrazy kontenerów

| Katalog | Zawartość |
|---|---|
| [docker-images](docker-images/README.md) | obrazy z GHCR na RTX 6000 Ada: start od zera, `-sm80` a `latest`, `-sm90` na starszej karcie |

## basal-1.5-4.5B i basal-1.5-mini

| Katalog | Zawartość |
|---|---|
| [compat-1.5-small](compat-1.5-small/README.md) | zgodność z upstream FP32: basal-bench, `/v1/basal`, `multi`, `act`, `facts`, `evidence` |
| [perf-1.5-small](perf-1.5-small/README.md) | wydajność wobec upstream v1.5.0: pojedyncza decyzja, HTTP, długie stany, ruch mieszany |
| `reference-basal-1.5-{4.5B,mini}-{fp32,bf16}` | referencje upstream FP32 i ścieżka serwowana BF16 |

## Instalacja i buildy

| Katalog | Zawartość |
|---|---|
| [install-packages](install-packages/README.md) | paczki macOS i Linux zainstalowane `install.sh` bez publikacji: `doctor`, `setup`, `serve`, `update` |
| [release-builds](release-builds/README.md) | profil `dist` (LTO): wyniki bitowo równe, szybkość bez zmian; czasy buildów w GitHub Actions z cache |
| [cuda-multi-ptx](cuda-multi-ptx/README.md) | kernele CUDA dla 8.0, 8.9 i 9.0 w jednej binarce: wyniki bitowo równe |

## Apple Silicon (Metal)

| Katalog | Zawartość |
|---|---|
| [metal-m1-pro-1.5](metal-m1-pro-1.5/README.md) | M1 Pro, basal-1.5-max, 4.5B i mini: zgodność z FP32 upstream, pojedyncza decyzja wobec upstream MLX bf16, f16 i 8-bit |
| [metal-m2-max-1.5](metal-m2-max-1.5/README.md) | M2 Max, basal-1.5-max, 4.5B i mini: zgodność z FP32 upstream, pojedyncza decyzja wobec upstream MLX bf16, f16 i 8-bit |
| [metal-m2-max-tree](metal-m2-max-tree/README.md) | M2 Max: attention po jednostkach drzewa (wynik niezależny od pakowania) wobec SDPA, wczytywanie basal-1.5-max; CUDA bez zmian |
| `reference-basal-1.5-{max,4.5B,mini}-{mlx,mlx-q8,mlx-f16}` | eksporty upstream v1.5.0 MLX na M1 Pro |

## Intel Arc (Vulkan)

| Katalog | Zawartość |
|---|---|
| [intel-arc-meteor-lake-2026-10-10](intel-arc-meteor-lake-2026-10-10/README.md) | mini, 4.5B i max: zgodność z FP32, HTTP, dobór GEMM, poprawki ładowania i kolejki oraz ograniczenia pamięci |
| [intel-arc-meteor-lake-refactor-2026-10-11](intel-arc-meteor-lake-refactor-2026-10-11/README.md) | refaktor bez mapowania checkpointu: bitowa zgodność mini, 4.5B i max (w tym Tree), czas ładowania, odrzucony wariant GEMM |
| [intel-arc-meteor-lake-fixes-2026-10-11](intel-arc-meteor-lake-fixes-2026-10-11/README.md) | wybór dGPU, `/health` 503 po utracie urządzenia, loader bez kopii tensora, jedno urządzenie w `doctor`: bitowa zgodność mini, 4.5B i max |
| [intel-arc-meteor-lake-gemm-2026-10-11](intel-arc-meteor-lake-gemm-2026-10-11/README.md) | optymalizacja GEMM: pomiary HTTP przed/po dla mini i 4.5B, projekcje wszystkich modeli i ograniczenie benchmarku max |
| [intel-arc-meteor-lake-memory-2026-10-11](intel-arc-meteor-lake-memory-2026-10-11/README.md) | szacunek pamięci cgroup i wag Intel, dynamiczny budżet iGPU, pomiary HTTP oraz odmowy po ładowaniu max z powodu swapu |

## API

| Katalog | Zawartość |
|---|---|
| [gemm-invariant-v2](gemm-invariant-v2/README.md) | druga wersja doboru algorytmów tabeli GEMM: H100 i RTX 6000 Ada, czas, niezależność od partii, zgodność z FP32 |
| [context-h100](context-h100/README.md) | H100: stany 512–16k tokenów, 1 i 5 pytań, basal-rs wobec upstream; tabela GEMM wobec heurystyki cuBLASLt; sekcje forwardu |
| [perf-gpus](perf-gpus/README.md) | H100, A100, L40S, RTX 6000 Ada, RTX A6000: pojedyncza decyzja, HTTP 1/8/32 klientów, ruch mieszany, energia; basal-rs i upstream na trzech modelach |
| [decision-sets](decision-sets/README.md) | 900 pytań z 9 zbiorów (PL, EN; choice, noul, score) wobec upstream FP32: basal-rs f16 i f32, upstream BF16; basal-1.5-max, 4.5B i mini, RTX 6000 Ada i A100 |
| [choice-sets](choice-sets/README.md) | Choice 59–150 opcji na zbiorach intencji (Banking77, CLINC150, MASSIVE pl): strategie grupowe na trzech modelach wobec TypeSafe jev, czekanie na GPU |
| [large-choice-1.5](large-choice-1.5/README.md) | Choice 11–255 (strategia grupowa) na basal-1.5 mini, 4.5B i max, Metal M2 Max |
| [typesafe-api-edge-cases](typesafe-api-edge-cases/README.md) | /v1/systemone wobec API TypeSafe: 38 przypadków brzegowych, statusy i błędy, SDK |

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
