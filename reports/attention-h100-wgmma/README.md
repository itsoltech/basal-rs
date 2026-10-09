# Attention na wgmma (Hopper) i forward z mniejszą liczbą kerneli na H100

Data: 2026-10-09. Wynajęta H100 PCIe (Massed Compute, 350 W, sterownik 580), jedna sesja (~1,6 h, ~$4,3).
Modele basal-1.5-4.5B i max, f16, tabele GEMM z binarki dla tej karty.

## Forward z commitu `b892eee` wobec `4cf07da`

[run-forward-cloud.sh](../../tools/perf/run-forward-cloud.sh), zmiany opisane w
[forward-ada-1](../forward-ada-1/README.md). Eksporty 44 przykładów basal-bench (drzewo): różnica 0,0 dla obu
modeli ([compare-4.5B-A-B.json](forward/compare-4.5B-A-B.json), [compare-max-A-B.json](forward/compare-max-A-B.json)).
Czas: 4 powtórzenia w kolejności A B B A, mediana powtórzeń 2–4; zegar nie był zablokowany (skrypt bez sudo,
[clocks.csv](forward/clocks.csv)).

| | 4.5B: A → B | max: A → B |
|---|---:|---:|
| pojedyncza decyzja | 13,5 → 12,7 ms (0,942) | 20,0 → 19,5 ms (0,973) |
| 512 tok., 1 pytanie | 22,3 → 21,2 ms (0,948) | 40,0 → 37,6 ms (0,940) |
| 1792–8192 tok. | 0,961–0,985 | 0,968–0,980 |
| 16384 tok. | 0,980 / 0,983 | 0,985 / 0,991 |

Pełne tabele: [tables-4.5B.md](forward/tables-4.5B.md), [tables-max.md](forward/tables-max.md).

## Attention na wgmma

Kernel `attn_tree_wg` (sekcja `BASAL_WGMMA` w `crates/basal-gpu/src/kernels.cu`, osobny PTX `compute_90a`) liczy to
samo co `attn_tree_tc`: Q·K z trzech iloczynów f16 hi/lo, P·V z P hi/lo, softmax i akumulacja w f32, te same kafelki
32 kluczy w tej samej kolejności. Mnożenia wykonuje `wgmma.mma_async` (warpgroup, m64n32k16 i m64n128k16) zamiast
`mma.sync` m16n8k16, w kolejności składników `attn_tree_tc`.

Trzy wersje, każda w tej samej sesji:

1. `wg`: A (Q, P) z rejestrów, K i V w shared memory w układzie K-major (V transponowane przy ładowaniu po jednym
   elemencie). 185 rejestrów.
2. `wg2`: V w układzie MN-major, zapisywane wektorowo tak, jak leży w pamięci (wgmma z transpozycją B).
3. `wg3`: jak `wg2`, a Q hi/lo zostaje w shared memory przez cały blok (A iloczynu Q·K z shared memory): 127
   rejestrów, dwa bloki na SM (96 KiB shared memory każdy).

Wyniki wszystkich trzech są bitowo równe `attn_tree_tc` na tej karcie: eksporty 4.5B i max w drzewie oraz
pojedynczo, różnica 0,0 ([v1](wg/v1), [v2](wg/v2), [v3](wg/v3), pliki `compare-*.json`). Zgodność z FP32
upstream jest więc ta sama co dotąd (4.5B: 44/44, max |Δ logit| 0,049; max: 44/44, 0,398).

Czas (zegar SM zablokowany na 1404 MHz, [clocks.csv](wg/v1/clocks.csv); mediana powtórzeń bez pierwszego):

| 4.5B | `tc` | `wg3` | `wg3` / `tc` | `wg` / `tc` | `wg2` / `tc` |
|---|---:|---:|---:|---:|---:|
| pojedyncza decyzja | 13,7 ms | 13,2 ms | 0,969 | 1,022 | 1,012 |
| 1792 tok., 1 / 5 pytań | 76,3 / 115,9 ms | 69,2 / 107,1 ms | 0,907 / 0,925 | 1,030 / 1,024 | 1,004 / 1,003 |
| 4096 tok., 1 / 5 pytań | 218,3 / 281,6 ms | 196,0 / 259,3 ms | 0,898 / 0,921 | 1,026 / 1,031 | 0,997 / 1,003 |
| 16384 tok., 1 / 5 pytań | 1569,8 / 1818,2 ms | 1303,5 / 1530,8 ms | 0,830 / 0,842 | 1,067 / 1,063 | 1,010 / 1,011 |

`tc` i `wg3` z jednego przebiegu ([tables-4.5B.md](wg/v3/tables-4.5B.md)), `wg` i `wg2` względem `tc` z
wcześniejszego przebiegu z tymi trzema ([v2](wg/v2/tables-4.5B.md); pierwszy, sam `wg`: [v1](wg/v1/tables-4.5B.md)).

| max | `tc` | `wg3` | `wg3` / `tc` |
|---|---:|---:|---:|
| pojedyncza decyzja | 19,8 ms | 19,3 ms | 0,976 |
| 512 tok., 1 pytanie | 37,8 ms | 35,5 ms | 0,939 |
| 1792 tok., 1 / 5 pytań | 139,9 / 210,5 ms | 135,1 / 198,2 ms | 0,966 / 0,941 |
| 4096 tok., 1 / 5 pytań | 379,7 / 497,5 ms | 357,6 / 456,4 ms | 0,942 / 0,917 |
| 16384 tok., 1 / 5 pytań | 2745,6 / 3180,3 ms | 2297,4 / 2642,7 ms | 0,837 / 0,831 |

max: [tables-max.md](wg/v3/tables-max.md).

Nsight Compute, jedno wywołanie attention przy 16384 tokenach ([ncu](ncu)):

| | czas | rejestry | occupancy |
|---|---:|---:|---:|
| `tc` | 16,96 ms | 220 | 12,5% |
| `wg2` | 17,36 ms | 184 | 12,5% |
| `wg3` | 12,78 ms | 127 | 24,4% |

Sama zamiana `mma.sync` na `wgmma` nie przyspiesza (`wg2`): kernel czeka na dane w shared memory przy jednym bloku
na SM. Zysk daje drugi blok na SM, możliwy dopiero po zwolnieniu rejestrów fragmentów Q.

W kodzie zostaje tylko `wg3`, pod nazwą `attn_tree_wg` (`BASAL_ATT=wg`); wersja po usunięciu wariantów `wg` i
`wg2` nie była już uruchamiana na H100, sprawdzono ją kompilacją (127 rejestrów, bez spilli) i przeglądem, a na RTX
6000 Ada eksport 4.5B jest bitowo równy `b892eee` (tam nadal `attn_tree_tc`). Jest domyślnym attention na kartach o
compute capability 9.0 w forwardzie f16 (`BASAL_ATT=tc` wraca do
`attn_tree_tc`). Forward bf16 zostaje przy `tc`: ścieżki z płaszczyzną V lo nie mierzono (`BASAL_ATT=wg` ją
wymusza).

Profil Nsight Systems z tej sesji ([run-profile-cloud.sh](../../tools/perf/run-profile-cloud.sh)) nie wszedł do
raportu: pierwsze uruchomienie nie miało uprawnień do instalacji narzędzi, a ncu w drugim trafił w wywołania
przechwytywania prefiksów zamiast forwardu (stąd `--launch-skip 200` w [wg/v3/run.sh](wg/v3/run.sh)).
