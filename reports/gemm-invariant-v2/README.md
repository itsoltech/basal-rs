# Tabela GEMM niezależna od partii: wersja 2 doboru algorytmów

Data: 2026-10-08. Zmiana w `crates/basal-gpu/src/cublaslt.rs`
(`INVARIANT_SEARCH_VERSION = 2`): do kandydatów tabeli niezależnej od partii
(`gemm_table: auto`) dochodzą konfiguracje z heurystyki cuBLASLt bez split-K.
Na Hopperze ustawiają one rozmiar klastra i kształt wewnętrzny kerneli,
których dotychczasowy przegląd (algorytm, kafel, etapy, swizzle) nie
obejmował. Każdy kandydat jest dodatkowo sprawdzany: pierwsze m wierszy jego
wyniku przy największej klasie M musi być bitowo równe wynikowi przy klasie
m; kandydat, którego wiersze zależą od M, odpada. Tabele z wcześniejszą wersją
doboru są przy starcie serwera generowane ponownie.

Pomiar ([tools/perf/run-gemm-check.sh](../../tools/perf/run-gemm-check.sh)):
wynajęte H100 PCIe (350 W, sterownik 580) i RTX 6000 Ada (300 W, sterownik
580), basal-rs 0.1.4 z tą zmianą. Na każdej karcie tabela starej wersji
(binarka wydania 0.1.4) i nowej, generowane przez `basal serve`; czasy żądań
(drabina kontekstu do 4k tokenów, `basal bench-requests`, mediany z 3) i
pojedynczej decyzji (`basal bench`) ze starą tabelą, nową i z heurystyką
cuBLASLt (bez tabeli).

## Czas

H100 PCIe:

| | stara tabela | nowa tabela | heurystyka |
|---|---:|---:|---:|
| 4.5B, pojedyncza decyzja (mediana) | 16,9 ms | 14,1 ms | 14,3 ms |
| 4.5B, decyzje/s | 105 | 137 | 145 |
| 4.5B, stan 512 tokenów, 1 pytanie | 33 ms | 24 ms | 24 ms |
| 4.5B, stan 1792, 1 pytanie | 103 ms | 84 ms | 79 ms |
| 4.5B, stan 4096, 5 pytań | 348 ms | 293 ms | 286 ms |
| max, pojedyncza decyzja | 26,0 ms | 21,2 ms | 21,3 ms |
| max, decyzje/s | 52 | 70 | 75 |
| max, stan 512, 1 pytanie | 57 ms | 42 ms | 43 ms |
| max, stan 1792, 1 pytanie | 200 ms | 154 ms | 148 ms |
| max, stan 4096, 5 pytań | 648 ms | 530 ms | 521 ms |

RTX 6000 Ada:

| | stara tabela | nowa tabela | heurystyka |
|---|---:|---:|---:|
| 4.5B, pojedyncza decyzja | 20,5 ms | 20,3 ms | 22,4 ms |
| 4.5B, stan 1792, 1 pytanie | 152 ms | 153 ms | 183 ms |
| 4.5B, stan 4096, 1 pytanie | 392 ms | 377 ms | 465 ms |

Na H100 nowa tabela skraca czasy o 17–27% (pojedyncza decyzja, drabina do
4k) i zbliża się do heurystyki cuBLASLt; na RTX 6000 Ada daje to samo co
stara albo do 4% mniej, a heurystyka jest tam wolniejsza o 10–20%.

Generowanie tabeli (jednorazowe, przy pierwszym starcie na danej karcie):
H100 4.5B 171 → 359 s, max 397 → 687 s; RTX 6000 Ada 4.5B 245 → 327 s.

## Wynik

| | H100 4.5B | H100 max | RTX 6000 Ada 4.5B |
|---|---|---|---|
| eksport pojedynczo wobec drzewa i partii (44 przykłady) | 0,0 | 0,0 | 0,0 |
| 44 przykłady wobec FP32 upstream | 44/44 | 44/44 | 44/44 |
| 900 pytań [decision-sets](../decision-sets/README.md) wobec FP32 | 899/900 | 900/900 | 899/900 |

Wynik nie zależy od pakowania partii z nową tabelą. Zgodność z FP32 jak
wcześniej: jedyna różnica 4.5B to remis w FP32 (`dyk-15` na RTX 6000 Ada,
jak w [decision-sets](../decision-sets/README.md); na H100 `polemo2-in-85`,
0,476 / 0,473, jak na A100). „8 mismatches” tokenów przy 44 przykładach to znane różnice
formatu eksportu pytań `multi` z [compat-1.5-small](../compat-1.5-small/README.md).

Koszt sesji: ~$4 (H100 Scaleway $3,30/h i RTX 6000 Ada $0,97/h, ok. 1 h).
