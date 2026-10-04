# basal-1.0-4.5B na RTX 6000 Ada: niezależność od partii i serwer HTTP

Data: 2026-10-04/05, warunki jak w [rust-cuda-v1](../rust-cuda-v1/README.md)
(limit mocy 300 W, w części pomiarów 250 W).

## Zmiany i ich pomiar

- Pomijanie kafelków kluczy w całości zamaskowanych w attention: wynik
  bitowo identyczny (`compare-f16-table-vs-skip.json`,
  `compare-f32-attn2-vs-skip.json`: różnica 0,0); A/B w parach
  (`ab-attn-skip/`, 3 rundy ABBA, 39 pytań): szybciej na 39/39, mediana
  ilorazu ~0,97.
- Przeszukanie GEMM na wagach rotowanych powyżej L2 (`gemm/`,
  `ab-gemm-cold/`): inny wybór w 176/208 klasach, w forwardzie mediana +1%,
  p95 −7%, przepustowość −3%; różnice algorytmów z mikrobenchmarku prawie nie
  przenoszą się na forward.
- Tryb niezależny od partii: jedna konfiguracja cuBLASLt bez split-K na
  kształt wag (`gemm-search --invariant`), odczyt liter własnym kernelem,
  przechwytywanie prefiksu tym samym kernelem attention, attention po węzłach
  drzewa. Pojedynczo, partiami budżetowymi i bez cache prefiksu różnica 0,0
  (`compare-inv2-single-vs-*.json`).
- Serwer HTTP `basal serve` i `tools/bench/loadtest.py`: upstream `fast`
  18–26 żądań/s, Rust 44–45 (`loadtest-*.json`); p50 sekwencyjnie 40,3 vs
  24,7 ms.
- Endpoint `/v1/basal` na 1.0: `compare-upstream-endpoint-1.0.json`.

Późniejsze zmiany (attention na tensor cores, pakowanie trie, harmonogram)
mierzono na basal-1.5-max: [rust-cuda-1.5-max](../rust-cuda-1.5-max/README.md).
