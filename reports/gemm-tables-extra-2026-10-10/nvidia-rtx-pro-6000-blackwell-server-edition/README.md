# RTX PRO 6000 Blackwell Server Edition: mini, 4.5B i max

2026-10-10, NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability
12.0, sterownik 580.126.09, limit mocy 600 W. Basal 0.1.7 (`85ca8751`),
cuBLASLt 12.9.1, f16, invariant search v2. Zegar nie był blokowany;
rejestrowano zegary SM, moc i temperaturę przez cały przebieg runnera.

Generowanie trwało: mini 54 s, 4.5B 79 s, max 156 s.
Każda tabela ma 25 klas M, 4 kształty wag i 100 wpisów.
Wszystkie trzy przeszły bajtowe porównanie `bench.jsonl` i `systemone.jsonl`
między eksportami single/tree/budget oraz pełny przebieg runnera.

Wobec FP32 na 900 pytaniach:

- [Mini](mini/compare-fp32-900.json): 899/900 decyzji, max |Δ logit|
  0,05047, max |Δ skalibrowane prawdopodobieństwo| 0,00641.
- [4.5B](4.5B/compare-fp32-900.json): 899/900 decyzji, max |Δ logit|
  0,05219, max |Δ skalibrowane prawdopodobieństwo| 0,00408.
- [Max](max/compare-fp32-900.json): 900/900 decyzji, max |Δ logit|
  0,13016, max |Δ skalibrowane prawdopodobieństwo| 0,00468.

W każdym katalogu są porównania 44 przykładów i odpowiedzi System One,
szczegóły `flips_cal`, benchmark `bench.json` i drabina `requests.json`
(około 512/1792/4096 tokenów, 1/5 pytań, 3 powtórzenia).
Zestaw 44 przykładów ma 44/44 zgodnych decyzji dla każdego modelu.

To weryfikacja tego wydania i tych zestawów na tej konkretnej nazwie GPU;
nie potwierdza wszystkich kart Blackwell ani wariantów Workstation/Max-Q.
Nie deklarujemy bitowej zgodności f16 z FP32 ani przyspieszenia.
Surowe eksporty i logi są w ignorowanym cache kampanii.
Maszynę usunięto po pobraniu wyników.
