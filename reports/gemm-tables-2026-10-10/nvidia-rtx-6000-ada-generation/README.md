# RTX 6000 Ada: brakujące tabele max i mini

2026-10-10, NVIDIA RTX 6000 Ada Generation, sterownik 580.126.09,
limit mocy 300 W. Basal 0.1.7 (`85ca8751`), cuBLASLt 12.9.1, f16,
wyszukiwanie invariant v2. Zegar nie był blokowany; zarejestrowany zakres
SM 210–2730 MHz obejmuje również przerwy. Log zaczyna się w trakcie
weryfikacji max, po jego wyszukiwaniu; pełny log pozostaje w cache.

Generowanie: max 704 s, mini 208 s. Tabela 4.5B była już w binarce
i nie była w tej sesji generowana ponownie.

Oba modele przeszły bajtowe porównanie `bench.jsonl` i `systemone.jsonl`
między eksportami single/tree/budget. Każda tabela ma 25 klas M, 4 kształty
wag i 100 wpisów. Wyniki FP32 i pomiary są w katalogach
[max](max/) i [mini](mini/).

Max:

- [44 przykłady wobec FP32](max/compare-fp32-44.json): 44/44 decyzji,
  max |Δ logit| 0,27117, max |Δ skalibrowane prawdopodobieństwo| 0,00077.
- [900 pytań wobec FP32](max/compare-fp32-900.json): 900/900 decyzji,
  max |Δ logit| 0,11403, max |Δ skalibrowane prawdopodobieństwo| 0,00678.
- [Pomiar offline](max/bench.json): mediana lat2 44,41 ms, p95 52,77 ms,
  throughput Budget 30,4 decyzji/s.

Mini: [900 pytań wobec FP32](mini/compare-fp32-900.json) daje 899/900 decyzji,
max |Δ logit| 0,05518, max |Δ skalibrowane prawdopodobieństwo| 0,00583.
Szczegóły rozbieżności pozostają w `flips_cal`, a wszystkie porównania
prawdopodobieństw i pełnych odpowiedzi System One w plikach porównania.
[44 przykłady](mini/compare-fp32-44.json): 44/44 decyzji, max |Δ logit|
0,02452, max |Δ skalibrowane prawdopodobieństwo| 0,00602.
[Pomiar offline](mini/bench.json): mediana lat2 7,19 ms, p95 7,96 ms,
throughput Budget 198,9 decyzji/s.

To zgodność batch-invariance w zbadanych eksportach, nie bitowa zgodność
f16 z FP32. Drabina kontekstu w `requests.json` obejmuje około 512, 1792
i 4096 tokenów, 1/5 pytań, 3 powtórzenia. Czasy nie stanowią pomiaru
przyspieszenia względem poprzedniej tabeli ani porównania między kartami.
Maszynę usunięto po pobraniu wyników; surowe eksporty są w ignorowanym cache.
