# H100 PCIe: brakująca tabela mini

2026-10-10, NVIDIA H100 PCIe, sterownik 580.126.09, limit mocy 350 W.
Basal 0.1.7 (`85ca8751`), cuBLASLt 12.9.1, f16, wyszukiwanie invariant v2.
Bez blokady zegara; zegarów tej sesji nie zapisano. Czasy nie są pomiarem
przyspieszenia względem poprzedniego buildu ani porównaniem z inną kartą.

Tabela mini powstała w 259 s. Tabele max i 4.5B były już w repozytorium
i nie były w tej sesji generowane ponownie.

Weryfikacja z wygenerowaną tabelą:

- 44 przykłady oraz przypadki System One: eksporty single/tree/budget mają
  bajtowo identyczne `bench.jsonl` i `systemone.jsonl`; porównania
  [single/tree](mini/compare-single-vs-tree.json) i
  [single/budget](mini/compare-single-vs-budget.json).
- [44 przykłady wobec FP32](mini/compare-fp32-44.json): 44/44 decyzji;
  max |Δ logit| 0,02204, max |Δ skalibrowane prawdopodobieństwo| 0,00464.
- [900 pytań wobec FP32](mini/compare-fp32-900.json): 899/900 decyzji;
  max |Δ logit| 0,05047, max |Δ skalibrowane prawdopodobieństwo| 0,00641.
  Rozbieżność `bench/137`: FP32 wybiera opcję 2 (0,33564 wobec 0,33485
  dla opcji 3), f16 opcję 3 (0,33533 wobec 0,33495 dla opcji 2).
- [Pomiar offline](mini/bench.json): mediana lat2 5,05 ms, p95 5,29 ms,
  throughput Budget 453,0 decyzji/s. [Drabina kontekstu](mini/requests.json)
  obejmuje około 512, 1792 i 4096 tokenów, 1/5 pytań, po 3 powtórzenia.

To nie jest bitowa zgodność f16 z FP32. Różnice formatu eksportu pytań
`multi` w referencji System One pozostają widoczne w porównaniu; nie są
przemilczane jako zgodność tokenów.

Runner: [run-gemm-tables-cloud.sh](../../../tools/perf/run-gemm-tables-cloud.sh).
Surowe eksporty są w ignorowanym cache. Maszynę usunięto po pobraniu wyników.
