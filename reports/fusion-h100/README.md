# Fuzja residual+RMSNorm i wektoryzacja bias+SiLU na H100

Data: 2026-10-08. Wynajęta H100 PCIe (350 W, sterownik 580), skrypt
[tools/perf/run-ab.sh](../../tools/perf/run-ab.sh), tabele
[tools/perf/ab_table.py](../../tools/perf/ab_table.py) (`tables.md`).

- A: basal-rs z `main` po zmianie doboru GEMM
  ([gemm-invariant-v2](../gemm-invariant-v2/README.md)), commit `50902cf`.
- B: A oraz dwie zmiany w kernelach CUDA:
  - dodanie residualu i RMSNorm następnego bloku w jednym kernelu
    (`residual_rmsnorm`), z arytmetyką i kolejnością sumowania kernela
    `rmsnorm` z candle; zastępuje `bias_residual` i `rms_norm` (2 z 3
    normalizacji w każdej warstwie) i jedno czytanie strumienia residualnego;
  - `bias_silu_mul` po 8 elementów na wątek z dostępem 16-bajtowym, ta sama
    arytmetyka na element.
- B+pipe: B z `BASAL_ATT=tc-pipe` (attention z dwustopniowym ładowaniem K/V
  przez `cp.async`, ten sam porządek obliczeń co domyślny kernel).

Obie binarki z tą samą tabelą GEMM (wersja 2, ta karta), modele
basal-1.5-4.5B i max.

## Wyniki

Eksporty 44 przykładów basal-bench (drzewo): A wobec B i B wobec B+pipe:
różnica 0,0 (logity, prawdopodobieństwa) dla obu modeli; B wobec FP32 upstream
44/44 decyzji, jak A.

## Czas (B / A)

| | 4.5B | max |
|---|---:|---:|
| pojedyncza decyzja (mediana) | 14,2 → 13,3 ms (0,94) | 20,9 → 20,1 ms (0,96) |
| stany 512–4096 tokenów | 0,94–0,96 | 0,95–0,98 (512, 1 pytanie: 1,02) |
| stany 8192–16384 tokenów | 0,97–0,99 | 0,97–1,00 |

basal-1.5-4.5B, stan 1792 tokenów, jedno pytanie: 82 → 77 ms (upstream na
tej karcie: 84 ms, [context-h100](../context-h100/README.md)).

`tc-pipe` wobec B: 0,99 przy 16k tokenów, 1,01–1,08 (wolniej) przy krótszych
stanach; zostaje opcją, domyślny kernel bez zmian.

Pojedynczy przebieg każdego wariantu, po kolei (A, B, B+pipe) w jednym
procesie na model.
