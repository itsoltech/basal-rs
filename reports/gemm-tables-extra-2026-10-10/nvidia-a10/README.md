# A10: tabele mini i 4.5B

2026-10-10, NVIDIA A10, compute capability 8.6, sterownik 570.148.08,
limit mocy 150 W. Basal 0.1.7 (`85ca8751`), cuBLASLt 12.9.1, f16,
invariant search v2. Użyto bibliotek forward compatibility CUDA 12.9.
Zegar nie był blokowany; przez cały przebieg runnera rejestrowano zegary
SM, moc i temperaturę. Max pominięto z powodu zapasu pamięci na GPU 24 GB.

Generowanie trwało: mini 324 s, 4.5B 544 s.
Każda tabela ma 25 klas M, 4 kształty wag i 100 wpisów.
Obie przeszły bajtowe porównanie `bench.jsonl` i `systemone.jsonl`
między eksportami single/tree/budget oraz pełny przebieg runnera.

Wobec FP32 na 900 pytaniach:

- [Mini](mini/compare-fp32-900.json): 899/900 decyzji, max |Δ logit|
  0,05518, max |Δ skalibrowane prawdopodobieństwo| 0,00583.
- [4.5B](4.5B/compare-fp32-900.json): 899/900 decyzji, max |Δ logit|
  0,05521, max |Δ skalibrowane prawdopodobieństwo| 0,00569.

Zestaw 44 przykładów ma 44/44 zgodnych decyzji dla obu modeli.
W katalogach są także porównania odpowiedzi System One, szczegóły
`flips_cal`, `bench.json` i drabina `requests.json` (około 512/1792/4096
tokenów, 1/5 pytań, 3 powtórzenia).
Nie deklarujemy bitowej zgodności f16 z FP32 ani przyspieszenia.
Surowe eksporty i logi są w ignorowanym cache kampanii.
Maszynę usunięto po pobraniu wyników.
