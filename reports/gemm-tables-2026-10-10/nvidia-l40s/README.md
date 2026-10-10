# L40S: tabele mini, 4.5B i max

2026-10-10, NVIDIA L40S, sterownik 580.126.09, limit mocy 350 W.
Basal 0.1.7 (`85ca8751`), cuBLASLt 12.9.1, f16, invariant v2.
Zegar nie był blokowany. Log w cache obejmuje część weryfikacji 4.5B
oraz generowanie i weryfikację max; SM 210–2520 MHz, również w przerwach.
Zegarów pomiaru mini nie zapisano.

Generowanie trwało: mini 214 s, 4.5B 294 s, max 561 s.
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
