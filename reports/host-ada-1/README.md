# Cache po stronie hosta i podział czasu decyzji (RTX 6000 Ada)

Data: 2026-10-09. Serwer testowy z RTX 6000 Ada, basal-1.5-4.5B f16, tabela GEMM dla tej karty.

Zmiany (bez zmiany obliczeń):

- uchwyty funkcji kerneli wyszukiwane raz na urządzenie (`fused::cu::func`), a nie przy każdym wywołaniu
  (`get_or_load_custom_func` pytał sterownik ~300 razy na forward), atrybuty attention ustawiane raz;
- algorytm cuBLASLt wybrany i sprawdzony dla danego (M, N, K, dtype) zapamiętany (`Lt::chosen`): każdy z 240 GEMM-ów
  na forward nie kopiuje listy algorytmów i nie woła `cublasLtMatmulAlgoCheck`;
- `BASAL_HOST_TIMING`: każdy odczyt CUDA wypisuje czas przygotowania na CPU, wrzucania forwardu i czekania na GPU.

Eksport 44 przykładów basal-bench (drzewo) wobec builda bez zmian: różnica 0,0
([compare-H-C.json](compare-H-C.json)). Czas (zegar SM 1710 MHz, 5 powtórzeń w kolejności A B / B A, mediana bez
pierwszego, [tables.md](tables.md), [run.sh](run.sh)): pojedyncza decyzja 21,3 → 21,3 ms (0,999), stany 512–1792
tokenów 0,998–1,005.

Podział pojedynczej decyzji (`BASAL_HOST_TIMING`, domyślny zegar, średnia z 40): przygotowanie 0,01 ms, wrzucenie
forwardu 4,76 ms, czekanie na GPU 13,88 ms. CPU wyprzedza GPU o ponad 13 ms, więc na tej karcie praca hosta jest
ukryta; zmiany nie skracają decyzji, zmniejszają tylko pracę CPU wątku modelu. Na H100 nie mierzono.
