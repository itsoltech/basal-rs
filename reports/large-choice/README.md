# Choice 11–255: ocena strategii grupowej

Data: 2026-10-04. basal-1.0-4.5B, Rust FP16 na RTX 6000 Ada (tabela GEMM).
Strategia opisana w [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md#choice-11255).
Polecenie: `basal eval-large-choice`, dane w [eval-v3-topics-cuda](eval-v3-topics-cuda/)
(`items.jsonl`, `summary.json`), log `eval-v3-cuda.log`.

Każdy z 44 przykładów basal-bench dostaje dystraktory do n opcji, w dwóch
ziarnach (88 pytań na rozmiar). Dystraktory to opcje przykładów o innym
temacie (mapa `TOPICS` w `crates/basal-cli/src/large_eval.rs`), żeby nie
trafiały do puli synonimy lub tłumaczenia poprawnej odpowiedzi. Bezpośrednie
pytanie z samymi oryginalnymi opcjami: 70/88 trafnych.

| n | Trafność | Argmax oryginalnych opcji = pytanie bezpośrednie | Mediana TV wobec pytania bezpośredniego | Mediana masy na oryginalnych opcjach | Ten sam zwycięzca w obu ziarnach | Mediana czasu | Mediana liczby promptów |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 66/88 | 85/88 | 0,047 | 0,92 | 42/44 | 115 ms | 10 |
| 20 | 61/88 | 83/88 | 0,035 | 0,83 | 38/44 | 156 ms | 10 |
| 40 | 56/88 | 83/88 | 0,051 | 0,70 | 35/44 | 340 ms | 18 |
| 100 | 36/60 | 56/60 | 0,095 | 0,52 | 25/30 | 1000 ms | 42 |

Przy n=100 ocenionych jest 60 pytań: dla przykładów z dużych grup
tematycznych (pytania tak/nie) pula nie ma 98 dystraktorów z innych tematów.

Kolejność oryginalnych opcji między sobą zgadza się z pytaniem bezpośrednim w
93–97% przypadków. Spadek trafności wynika z dystraktorów, które przejmują
coraz więcej masy (przy n=100 mediana 0,48). Upstream nie obsługuje więcej niż
10 opcji, więc nie ma punktu odniesienia, który pozwoliłby oddzielić błąd
strategii od trudności wyboru spośród 100 opcji. Mapa tematów jest zgrubna;
część dystraktorów może być sensowną odpowiedzią.
