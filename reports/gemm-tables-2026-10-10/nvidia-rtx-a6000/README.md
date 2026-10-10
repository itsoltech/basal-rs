# RTX A6000: tabele mini, 4.5B i max

2026-10-10, NVIDIA RTX A6000, sterownik 580.126.09, limit mocy 300 W.
Basal 0.1.7 (`85ca8751`), cuBLASLt 12.9.1, f16, invariant v2.
Zegar nie był blokowany. Log w cache obejmuje końcówkę weryfikacji mini
oraz generowanie i weryfikację 4.5B/max; SM 210–1950 MHz, także w przerwach.
Nie obejmuje generowania mini ani jego pierwszych eksportów.

Generowanie trwało: mini 335 s, 4.5B 460 s, max 953 s.
Każda tabela ma 25 klas M, 4 kształty wag i 100 wpisów.
Wszystkie trzy przeszły bajtowe porównanie `bench.jsonl` i `systemone.jsonl`
między eksportami single/tree/budget.

Wobec FP32 na 900 pytaniach:

- [Mini](mini/compare-fp32-900.json): 899/900 decyzji, max |Δ logit|
  0,05518, max |Δ skalibrowane prawdopodobieństwo| 0,00583.
- [4.5B](4.5B/compare-fp32-900.json): 899/900 decyzji, max |Δ logit|
  0,05521, max |Δ skalibrowane prawdopodobieństwo| 0,00569.
- [Max](max/compare-fp32-900.json): 900/900 decyzji, max |Δ logit|
  0,11403, max |Δ skalibrowane prawdopodobieństwo| 0,00678.

W każdym katalogu są też porównania 44 przykładów i odpowiedzi System One,
szczegóły `flips_cal`, [benchmark mini](mini/bench.json),
[benchmark 4.5B](4.5B/bench.json), [benchmark max](max/bench.json)
oraz drabina kontekstu `requests.json` (około 512, 1792 i 4096 tokenów,
1/5 pytań, 3 powtórzenia).

Nie deklarujemy bitowej zgodności f16 z FP32 ani przyspieszenia względem
poprzedniej tabeli. Pomiary czasu nie są porównaniem między kartami.
Surowe eksporty są w ignorowanym cache; maszynę usunięto po pobraniu wyników.
