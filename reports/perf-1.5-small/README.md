# basal-1.5-4.5B i basal-1.5-mini: wydajność

Data: 2026-10-06, RTX 6000 Ada, limit mocy 300 W (domyślny). basal-rs w
konfiguracji domyślnej (tabele GEMM z `gemm_table: auto`, HRRN, tor długich
żądań) wobec upstream v1.5.0 `fast` (BF16, torch.compile, CUDA graphs) na tej
samej karcie, ten sam klient. Skrypt: `run.sh`. Zgodność wyników:
[compat-1.5-small](../compat-1.5-small/README.md).

basal-1.5-mini ma limit 8192 pozycji, więc dla niego długie stany kończą się na
~4k tokenów, a ruch mieszany pomija dokumenty ~8k i ~16k (371 z 400 żądań,
średnio 2,14 pytania na żądanie).

## Pojedyncza decyzja (metodyka basal-bench, 44 przykłady)

| Model | Wariant | lat2 mediana | lat2 p95 | lat1 | Decyzje/s |
|---|---|---:|---:|---:|---:|
| 1.5-4.5B | upstream `fast` | 31,5 ms | 41,6 ms | 27,2 ms | 27,8 |
| 1.5-4.5B | basal-rs | 20,4 ms | 24,2 ms | 18,8 ms | 66,0 |
| 1.5-mini | upstream `fast` | 9,9 ms | 12,6 ms | 9,0 ms | 93,1 |
| 1.5-mini | basal-rs | 7,9 ms | 8,6 ms | 7,3 ms | 166,7 |

## HTTP, jedno pytanie na żądanie (44 przykłady w pętli)

| Model | Klienci | Upstream: żądania/s, p50 | basal-rs: żądania/s, p50 | Energia upstream / basal-rs |
|---|---:|---:|---:|---:|
| 1.5-4.5B | 1 | 24,8, 41 ms | 36,4, 27 ms | 11,2 / 7,5 J |
| 1.5-4.5B | 8 | 26,0, 300 ms | 47,0, 167 ms | 11,4 / 6,2 J |
| 1.5-4.5B | 32 | 21,8, 1501 ms | 54,5, 573 ms | 13,7 / 5,5 J |
| 1.5-mini | 1 | 55,7, 18 ms | 75,8, 13 ms | 3,4 / 2,2 J |
| 1.5-mini | 8 | 80,9, 101 ms | 132,2, 58 ms | 3,4 / 2,0 J |
| 1.5-mini | 32 | 60,2, 528 ms | 141,7, 214 ms | 4,9 / 2,0 J |

Energia: J na decyzję (moc GPU × czas / decyzje). Odpowiedzi upstream na to
samo żądanie różnią się zależnie od partii (do 0,031 prawdopodobieństwa),
basal-rs: 0,0.

## Długie stany (mediany z 2 powtórzeń)

| Stan (tokeny) | 4.5B upstream, 1 / 5 pytań | 4.5B basal-rs, 1 / 5 pytań | mini upstream, 1 / 5 pytań | mini basal-rs, 1 / 5 pytań |
|---:|---:|---:|---:|---:|
| 1 078 | 139 / 207 ms | 98 / 139 ms | 37 / 67 ms | 37 / 59 ms |
| 2 146 | 251 / 351 ms | 191 / 252 ms | 88 / 130 ms | 76 / 105 ms |
| 4 169 | 1 071 / 7 479 ms | 381 / 467 ms | 377 / 2 590 ms | 152 / 195 ms |
| 8 221 | 2 948 / 20 742 ms | 900 / 1 026 ms | | |
| 16 527 | 8 453 / 58 734 ms | 2 409 / 2 610 ms | | |

## Ruch mieszany (`tools/bench/mixed.jsonl`)

| Model | Faza | Upstream: żądania/s, p50 / p95 | basal-rs: żądania/s, p50 / p95 |
|---|---|---:|---:|
| 1.5-4.5B | sekwencyjnie | 0,72 (43/min), 74 ms / 8,7 s | 4,88 (293/min), 49 ms / 1,1 s |
| 1.5-4.5B | 32 klientów | 0,99 (59/min), 12,3 s / 102 s | 5,60 (336/min), 1,6 s / 24,1 s |
| 1.5-mini | sekwencyjnie | 15,0 (899/min), 24 ms / 132 ms | 26,9 (1612/min), 19 ms / 124 ms |
| 1.5-mini | 32 klientów | 13,6 (818/min), 2,0 s / 4,5 s | 41,7 (2501/min), 752 ms / 1,2 s |

basal-rs, napływ otwarty (Poisson) przy 50 / 75 / 90% przepustowości
zamkniętej pętli: 4.5B 2,9 / 4,3 / 4,7 żądania/s z p95 2,0 / 3,5 / 7,5 s;
mini 21,5 / 31,9 / 37,0 żądania/s z p95 0,42 / 0,80 / 1,18 s. Upstream w
napływie otwartym nie mierzony.

Bez błędów w żadnej fazie. Pojedyncze przebiegi, po kolei (najpierw basal-rs,
potem upstream dla każdego modelu).
