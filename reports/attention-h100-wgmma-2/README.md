# Attention na wgmma z nakładaniem ładowania K/V (H100)

Data: 2026-10-09. Wynajęta H100 PCIe (Hyperstack, 350 W, sterownik 570 z biblioteką zgodności CUDA 12.9),
biblioteki CUDA 12.9.1 z `basal setup`, tabele GEMM z binarki. Build `bfe4e05` z wariantem `wgp` (drzewo robocze),
[run-wg-cloud.sh](../../tools/perf/run-wg-cloud.sh) z `WG_VARIANTS="tc wg wgp"`. Koszt sesji ~$4,7.

- `tc`: `attn_tree_tc` (mma.sync).
- `wg`: `attn_tree_wg` w wersji z `main` (`bfe4e05`, [attention-h100-wgmma](../attention-h100-wgmma/README.md)),
  pierwszy raz uruchomiona na H100 w tej postaci.
- `wgp`: `wg` z dwoma buforami K/V wypełnianymi przez `cp.async`, następny kafelek ładuje się w trakcie liczenia
  bieżącego (112 KiB shared memory na blok, nadal dwa bloki na SM, 128 rejestrów); tylko forward f16.

## Wyniki

Eksporty 44 przykładów basal-bench, 4.5B i max, w drzewie i pojedynczo: `wg` i `wgp` wobec `tc` różnica 0,0,
pojedynczo wobec drzewa 0,0 ([wg.log](wg.log), `compare-*.json`). Wobec FP32 upstream bez zmian: 4.5B 44/44,
max |Δ logit| 0,049; max 44/44, 0,398.

## Czas

Zegar SM zablokowany na 1404 MHz ([lock.txt](lock.txt), [clocks.csv](clocks.csv)), 4 powtórzenia w kolejności
`tc wg wgp`, potem odwrotnie; mediana powtórzeń 2–4 ([tables-4.5B.md](tables-4.5B.md),
[tables-max.md](tables-max.md)).

| 4.5B | `tc` | `wg` / `tc` | `wgp` | `wgp` / `tc` |
|---|---:|---:|---:|---:|
| pojedyncza decyzja | 14,1 ms | 0,967 | 13,4 ms | 0,953 |
| 512 tok., 1 / 5 pytań | 22,4 / 47,6 ms | 0,988 / 0,923 | 21,6 / 41,8 ms | 0,964 / 0,876 |
| 1792 tok., 1 / 5 pytań | 76,7 / 113,7 ms | 0,910 / 0,936 | 67,6 / 104,2 ms | 0,881 / 0,916 |
| 4096 tok., 1 / 5 pytań | 214,6 / 274,0 ms | 0,899 / 0,916 | 184,1 / 239,0 ms | 0,858 / 0,872 |
| 8192 tok., 1 / 5 pytań | 523,2 / 637,4 ms | 0,861 / 0,871 | 424,4 / 525,0 ms | 0,811 / 0,824 |
| 16384 tok., 1 / 5 pytań | 1560 / 1785 ms | 0,824 / 0,832 | 1191 / 1381 ms | 0,763 / 0,774 |

| max | `tc` | `wg` / `tc` | `wgp` | `wgp` / `tc` |
|---|---:|---:|---:|---:|
| pojedyncza decyzja | 20,3 ms | 0,977 | 19,7 ms | 0,968 |
| 512 tok., 1 / 5 pytań | 38,2 / 88,8 ms | 0,942 / 0,941 | 35,4 / 84,9 ms | 0,927 / 0,956 |
| 1792 tok., 1 / 5 pytań | 136,9 / 208,8 ms | 0,955 / 0,933 | 128,7 / 191,3 ms | 0,940 / 0,916 |
| 4096 tok., 1 / 5 pytań | 371,8 / 479,9 ms | 0,932 / 0,914 | 344,9 / 428,1 ms | 0,928 / 0,892 |
| 8192 tok., 1 / 5 pytań | 919,5 / 1117 ms | 0,884 / 0,874 | 794,2 / 941,4 ms | 0,864 / 0,843 |
| 16384 tok., 1 / 5 pytań | 2735 / 3120 ms | 0,832 / 0,827 | 2175 / 2470 ms | 0,795 / 0,792 |

`wgp` jest szybszy od `wg` poza dwoma punktami max (512 tok. z 5 pytaniami 0,956 wobec 0,941, 2048 tok. z 1
pytaniem 0,963 wobec 0,961) i zostaje domyślnym attention na compute capability 9.0 w forwardzie f16 (bf16:
`attn_tree_tc`, `BASAL_ATT=wg` / `wgp` wymusza Hoppera).

Profile Nsight Systems i Compute z tej sesji nie powstały: wersja 2026.3 z repozytorium CUDA ([nsight-versions.txt](
nsight-versions.txt)) nie współpracuje ze sterownikiem 570 maszyny (nsys zawiesił się na pierwszym przebiegu, ncu:
„Cuda driver is not compatible”).
