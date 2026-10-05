# Ruch mieszany: różne stany, długości, liczba i typy pytań

Data: 2026-10-05, basal-1.5-max, RTX 6000 Ada z limitem mocy 250 W, warunki
jak w [../README.md](../README.md). Klient: `tools/bench/loadtest.py`.
Ruch: `tools/bench/mixed.jsonl` z `tools/bench/make_mixed.py`, 400 żądań w
stałej losowej kolejności, średnio 2,16 pytania na żądanie.

| Klasa | Udział | Opis |
|---|---:|---|
| short-1q | 40% | jedno pytanie choice o krótki stan (44 przykłady basal-bench, PL i EN) |
| short-multi | 20% | 2–3 pytania o krótki stan, co dziesiąte 14 pytań |
| features | 10% | multi, act, evidence, facts "auto" (`systemone_cases_1.5.jsonl`) |
| doc-1k-2k | 20% | 1–5 pytań o dokument ~1k lub ~2k tokenów (6 dokumentów) |
| doc-4k-8k | 8% | 1–5 pytań o dokument ~4k lub ~8k tokenów (4 dokumenty) |
| doc-16k | 2% | 1–3 pytania o dokument ~16k tokenów (2 dokumenty) |

Typy pytań: 418 choice, 233 noul, 121 score, 78 multi, 14 act. Żądania o ten
sam dokument mają wspólny stan.

## Rust a upstream (`run.sh`)

Rust w wersji z jednym torem i kolejnością przyjścia (przed zmianą
harmonogramu) oraz upstream v1.5.0 `basal-serve --mode fast`, ten sam ruch:

| Faza | Rust żądania/s | Rust p50 / p95 | Upstream żądania/s | Upstream p50 / p95 |
|---|---:|---:|---:|---:|
| sekwencyjnie | 1,74 (104/min) | 141 ms / 3,3 s | 0,29 (17/min) | 202 ms / 21,0 s |
| 8 klientów | 1,75 | 2,8 s / 13,7 s | 0,28 | 9,5 s / 118 s |
| 32 klientów | 1,77 | 18,0 s / 31,8 s | – | – |

Czas sekwencyjny (mediana) według klasy:

| Klasa | Rust | Upstream |
|---|---:|---:|
| short-1q | 70 ms | 115 ms |
| short-multi | 151 ms | 206 ms |
| features | 140 ms | 177 ms |
| doc-1k-2k | 623 ms | 782 ms |
| doc-4k-8k | 3,2 s | 14,8 s |
| doc-16k | 8,3 s | 83,7 s |

Większość różnicy przepustowości to długie dokumenty: upstream dzieli wiersze
po 3072 tokeny, więc stan długiego dokumentu liczy dla każdego pytania i każdej
gałęzi multi osobno. Pomiar upstream przerwano po fazie 8 klientów (pozostałe
fazy trwałyby ~2,5 h; przy obciążeniach otwartych z fazy Rust jego kolejka
rosłaby bez ograniczenia), więc jest tylko `upstream.log` bez `upstream.json`.

Przepustowość Rust nie rośnie z liczbą klientów: GPU przez cały czas pracuje na
limicie mocy (246 W), ~65 J na decyzję. Przy jednym torze i kolejności
przyjścia krótkie żądania czekają za długimi: przy napływie otwartym 0,89
żądania/s (połowa przepustowości) short-1q ma p95 9,4 s, bo forward dokumentu
16k zajmuje GPU przez ~8 s.

## Harmonogram (`run-sched.sh`)

Jedna binarka, trzy warianty: `--schedule fifo --long-tokens 0` (kolejność
przyjścia, jeden tor), `--schedule hrrn --long-tokens 0` (najwyższy stosunek
odpowiedzi, jeden tor) i domyślny `hrrn` z drugim torem dla żądań powyżej 4096
tokenów, który po każdej warstwie oddaje GPU, gdy główny tor ma partię (po
co najmniej 100 ms pracy). Napływ otwarty (Poisson, ten sam ciąg) 0,9, 1,3 i
1,6 żądania/s, 400 żądań w fazie. p50 / p95:

| Napływ | Wariant | Przepustowość | short-1q | short-multi | features | doc-1k-2k | doc-4k-8k | doc-16k |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0,9/s | fifo | 0,93 | 83 ms / 8,6 s | 391 ms / 5,6 s | 228 ms / 5,2 s | 0,8 / 8,8 s | 3,8 / 9,2 s | 8,2 / 12,3 s |
| 0,9/s | hrrn | 0,93 | 85 ms / 6,0 s | 377 ms / 4,7 s | 238 ms / 4,1 s | 0,8 / 8,6 s | 3,4 / 10,1 s | 8,2 / 16,3 s |
| 0,9/s | 2 tory | 0,93 | 82 ms / 0,84 s | 183 ms / 0,88 s | 147 ms / 0,60 s | 0,7 / 1,2 s | 4,8 / 10,2 s | 10,5 / 19,0 s |
| 1,3/s | fifo | 1,34 | 2,1 / 10,7 s | 1,8 / 9,1 s | 2,3 / 10,9 s | 2,1 / 12,0 s | 6,3 / 15,3 s | 9,4 / 13,7 s |
| 1,3/s | hrrn | 1,34 | 1,5 / 7,4 s | 1,7 / 7,2 s | 1,2 / 4,9 s | 1,7 / 8,6 s | 6,5 / 16,8 s | 9,8 / 18,7 s |
| 1,3/s | 2 tory | 1,34 | 124 ms / 1,17 s | 263 ms / 1,32 s | 170 ms / 1,02 s | 0,8 / 1,5 s | 7,7 / 16,0 s | 12,3 / 21,3 s |
| 1,6/s | fifo | 1,61 | 5,9 / 18,2 s | 5,4 / 21,1 s | 5,0 / 20,9 s | 5,3 / 20,8 s | 10,9 / 25,3 s | 14,1 / 20,7 s |
| 1,6/s | hrrn | 1,59 | 2,4 / 8,4 s | 2,6 / 9,4 s | 2,5 / 7,3 s | 3,0 / 9,3 s | 10,0 / 25,1 s | 15,5 / 45,2 s |
| 1,6/s | 2 tory | 1,55 | 161 ms / 1,49 s | 286 ms / 1,57 s | 328 ms / 1,54 s | 0,9 / 1,7 s | 18,9 / 34,0 s | 25,2 / 58,2 s |

Przy dwóch torach klasy short, features i doc-1k-2k (90% ruchu) mają p95 poniżej 1,8 s
do 1,6 żądania/s. Kosztem jest czas długich dokumentów: oddają GPU krótkim
partiom, więc przy 1,6/s doc-4k-8k ma p50 18,9 s zamiast 10,9 s, a
przepustowość (1,55/s) jest tuż pod napływem (kolejka długiego toru rośnie).
W zamkniętej pętli 32 klientów (stałe przeciążenie) przepustowość wariantów
wynosi 1,85–1,98/s; HRRN i dwa tory dają wtedy krótkim żądaniom p95 5,5–6,1 s
(fifo 28,6 s), a dokumenty 16k czekają ~150 s.

Odpowiedzi są identyczne we wszystkich fazach i wariantach (różnica
prawdopodobieństw względem pierwszej odpowiedzi 0,0). Próba
`smoke_lanes.py`: dokument 16k i równocześnie 20 kolejnych krótkich żądań; z
dwoma torami krótkie odpowiedzi 185–244 ms, a dokument 9,26 s; w jednym torze
pierwsze krótkie żądanie czekało 7,2 s, dokument 9,02 s. Odpowiedzi obu
wariantów identyczne.

## Ograniczenia

- Pojedyncze przebiegi każdego wariantu, kolejno w jednej sesji (bez ABBA).
  `fifo` z tej binarki dopuszcza do partii także późniejsze żądania, które
  mieszczą się w budżecie; w `rust.json` (wersja przed zmianą harmonogramu) partia kończyła
  się na pierwszym niemieszczącym się żądaniu.
- Ruch syntetyczny: proporcje klas i dokumenty są założeniem, nie pomiarem
  ruchu produkcyjnego. Przepustowość zależy głównie od udziału długich
  dokumentów.
- Limit mocy 250 W.
