# basal-1.0-4.5B na Apple M1 Pro (Metal)

Data: 2026-10-04. Apple M1 Pro (GPU 14 rdzeni, 32 GB), model
`Remek/basal-1.0-4.5B` w rewizji `b95288041cb1975bb930c6f0410819273fd2f54f`,
upstream `3fa2eeab` z backendem MLX.

## Zgodność

Punkt odniesienia: forward FP32 runtime'u (wagi bf16 rozszerzone do f32,
attention i odczyt w f32); jego zgodność z PyTorch FP32 upstream sprawdzono
później na CUDA (różnica logitu 7·10⁻⁵, [rust-cuda-v1](../rust-cuda-v1/README.md)).
Zestaw: 44 przykłady basal-bench i 31 pytań System One o identycznych
tokenach. Różnice prawdopodobieństw po uśrednieniu porządków.

| Wariant względem FP32 | Decyzje 44 | Pytania System One | Maks. różnica logitu | Maks. różnica p_avg | Plik |
|---|---:|---:|---:|---:|---|
| Rust FP16, jedno pytanie na forward | 44/44 | 31/31 | 0,051 | 0,0039 | `compare-rust-f32-vs-f16-single.json` |
| Rust FP16, wiersze-drzewa | 44/44 | 31/31 | 0,051 | 0,0030 | `compare-rust-f32-vs-f16-tree.json` |
| Rust BF16 | 43/44 | 31/31 | 0,306 | 0,0261 | `compare-rust-f32-vs-bf16-single.json` |
| Upstream MLX BF16 | 44/44 | 29/29 | 0,405 | 0,0507 | `compare-rust-f32-vs-mlx-bf16.json` |

Rozbieżna decyzja BF16 dotyczy przykładu z marginesem 0,003 między opcjami.
FP16 jest bliżej FP32 niż BF16, dlatego jest domyślną precyzją.

## Wydajność

Pomiar w parach, kolejno w jednej sesji (`paired/`), metodyka basal-bench:

| Wariant | Mediana | p95 | Jeden porządek | Decyzje/s |
|---|---:|---:|---:|---:|
| Upstream MLX BF16 | 978 ms | 1256 ms | 760 ms | 0,98 |
| Upstream MLX FP16 | 846 ms | 1119 ms | 653 ms | 1,08 |
| Rust FP16 | 657 ms | 951 ms | 497 ms | 1,59 |
| Rust BF16 | 625 ms | 892 ms | 486 ms | 1,48 |

Rust FP16 ma medianę niższą o 22% niż MLX FP16 i o 33% niż MLX BF16, a
przepustowość wyższą o 48% i 62%. W osobnej sesji przy mniejszym obciążeniu
karty Rust FP16 osiągnął medianę 495 ms i 1,87 decyzji/s
(`bench-rust-f16.json`), upstream MLX BF16 847 ms i 1,19 decyzji/s
(`bench-upstream-mlx-bf16.json`).

Żądania z kilkoma pytaniami o jeden stan (jeden klient, mediany;
`requests-rust-tree.json`, `requests-upstream-mlx-f16.json`):

| Żądanie | Pytania | Upstream MLX FP16 | Rust FP16 |
|---|---:|---:|---:|
| cx-01 | 3 | 2232 ms | 1280 ms |
| cx-03 | 3 | 3197 ms | 1805 ms |
| 14 pytań o jeden stan | 14 | 14973 ms | 7518 ms |
| jedno pytanie | 1 | 836 ms | 678 ms |

Cache stanu między żądaniami (14 żądań po jednym pytaniu o 5 stanów): mediana
647 ms bez cache, 489 ms z cache (`same-state-cold.json`,
`same-state-warm.json`).

GEMM zajmuje 85–90% czasu forwardu i osiąga 3,2–4,2 TFLOPS FP16 przy
szacowanym szczycie M1 Pro ~4,6 TFLOPS, więc pojedynczej decyzji nie da się
tu znacząco skrócić bez zmniejszenia liczby operacji.

Pliki `compare-*.json` wskazują eksporty, z których powstały; same eksporty
nie są przechowywane. Referencja upstream MLX BF16:
[../reference-mlx-bf16](../reference-mlx-bf16/).
