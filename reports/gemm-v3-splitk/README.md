# GEMM v3 (split-K w tabeli niezależnej od partii) i podział czasu krótkiej decyzji na H100

Data: 2026-10-09. Wynik negatywny: zmiana nie weszła do kodu.

## Hipoteza

Przy krótkim prompcie (~150–600 tokenów) projekcje QKV, O i down mają na H100 (132 SM) za mało bloków, a tabela GEMM
niezależna od partii (v2) dopuszcza tylko algorytmy bez podziału K. Wersja 3 wyszukiwania dopuszczała też podział K na
stałą, niezależną od M liczbę części z redukcją w f32 (`CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE`), pod tą samą
kontrolą bitowej niezależności wierszy co v2.

## Pomiar

- H100 PCIe (Hyperstack, 350 W, sterownik 570, CUDA 12.9.1 z `basal setup`), zegar SM 1404 MHz,
  [run-gemm-check.sh](../../tools/perf/run-gemm-check.sh): tabela v2 z wydania 0.1.5 (`old`), tabela v3 z tego
  builda (`new`) i sama heurystyka cuBLASLt bez ograniczeń (`heuristic`), 3 powtórzenia w rotacji, mediana bez
  pierwszego ([h100/tables-4.5B.md](h100/tables-4.5B.md), [h100/tables-max.md](h100/tables-max.md)).
- RTX 6000 Ada (serwer testowy, zegar 1710 MHz), tabela v2 z binarki wobec v3 wygenerowanej tym buildem,
  4 powtórzenia ([ada/tables-4.5B.md](ada/tables-4.5B.md), [ada/run.sh](ada/run.sh)).

| | H100 4.5B v3 / v2 | H100 max v3 / v2 | H100 4.5B heurystyka / v2 | Ada 4.5B v3 / v2 |
|---|---:|---:|---:|---:|
| pojedyncza decyzja | 0,998 | 0,995 | 1,012 | 1,110 |
| 512 tok., 1 / 5 pytań | 0,987 / 1,033 | 0,989 / 1,032 | 0,983 / 0,925 | 1,039 / 1,021 |
| 1792 tok., 1 / 5 pytań | 0,998 / 0,993 | 1,004 / 1,003 | 0,928 / 0,943 | 0,972 / 0,987 |
| 4096 tok., 1 / 5 pytań | 0,989 / 0,994 | 1,004 / 0,999 | 0,933 / 0,971 | 1,044 / 1,043 |
| 16384 tok., 1 / 5 pytań | 0,997 / 1,002 | 0,997 / 0,997 | 1,006 / 1,004 | 1,043 / 1,044 |

Wyniki modelu z tabelą v3 są bitowo te same co z v2: zestaw 900 pytań wobec FP32 ma identyczne liczby (4.5B
899/900, max |Δ logit| 0,0522 na H100; max 900/900, 0,1302; Ada 4.5B 899/900, 0,0552), niezależność od partii
0,0 (single wobec tree i budget). v3 wybrał więc dla każdego kształtu tę samą grupę algorytmów bez podziału K, a
różnice czasu na Adzie (+11% pojedyncza decyzja) wynikają z innego wyboru członków grupy dla klas M: wyszukiwanie
mierzy czasy z rozrzutem, a tabela v2 z binarki trafiła lepiej.

Diagnostyka wyszukiwania na H100 (`BASAL_GEMM_EQUIV=1`, [h100/search-diag.log](h100/search-diag.log)): dla QKV, O i
gate_up żaden kandydat z podziałem K nie jest wśród najszybszych przy M ≤ 256. Dla down (K = 11008) kandydaci z
podziałem na 3 i 4 części mają te same wiersze przy każdym M, ale przy dużych M cuBLASLt ich nie przyjmuje (bufor
roboczy redukcji w f32 większy niż 32 MiB), a przy M ≤ 256 są wolniejsi od najlepszego algorytmu bez podziału
(1,145–1,151 wobec 1,065 względem najlepszego w klasie). Nawet heurystyka cuBLASLt bez ograniczeń nie skraca
pojedynczej decyzji; przyspiesza tylko 512–4096 tokenów przy kilku pytaniach (do 7,5%, max do 11%), kosztem
niezależności od partii.

## Podział czasu krótkiej decyzji na H100

`BASAL_HOST_TIMING` (pojedyncza decyzja 4.5B, średnia z 40, [h100/host-timing.log](h100/host-timing.log)):
przygotowanie 0,01 ms, wrzucenie forwardu 2,90 ms, czekanie na GPU 8,97 ms. CPU wyprzedza GPU, więc CUDA graphs
nie skróciłyby decyzji w istotny sposób.

Nsight Systems 2025.1 (wersja zgodna ze sterownikiem 570), forward pojedynczej decyzji (545 kerneli, 12,1 ms,
[h100/bench-forward.md](h100/bench-forward.md); kernele `nvjet_*` to GEMM cuBLASLt na Hopperze): GEMM 63,7%,
attention 23,7%, bias+SiLU 5,2%, norma z residualem 4,8%, `qkv_rope` 2,2%, przerwy między kernelami 7,7%.
Attention zajmuje prawie ćwierć krótkiej decyzji: przy ~250 tokenach jedno uruchomienie ma kilkadziesiąt bloków
po 128 wierszy na 132 SM.

Generowanie tabel v3: H100 4.5B 380 s, max 695 s; Ada 4.5B 587 s (v2: 359 / 687 / 327 s).
