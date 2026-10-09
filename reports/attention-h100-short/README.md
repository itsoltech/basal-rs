# Attention w blokach po 64 wiersze dla krótkich promptów (H100)

Data: 2026-10-09. Wynajęta H100 PCIe (Hyperstack, 350 W, sterownik 570, CUDA 12.9.1), zegar SM zablokowany na
1404 MHz (przy maksymalnej mocy karta schodzi chwilami do ~1245 MHz, [auto/clocks.csv](auto/clocks.csv)), tabele
GEMM v2 z binarki, [run-wg-cloud.sh](../../tools/perf/run-wg-cloud.sh), 3 powtórzenia w rotacji, mediana bez
pierwszego.

Profil pojedynczej decyzji 4.5B ([gemm-v3-splitk](../gemm-v3-splitk/README.md)) przypisał attention 24% czasu GPU:
przy ~250 tokenach `attn_tree_wgp` (dwa warpgroupy, 128 wierszy na blok) uruchamia kilkadziesiąt bloków na 132 SM.

`attn_tree_wgp64`: to samo ciało kernela z jednym warpgroupem, 64 wiersze na blok (80 KiB shared memory, 156
rejestrów); każdy wiersz liczy te same kafelki kluczy w tej samej kolejności, więc wynik jest bitowo ten sam.

## Wymuszony `wgp64` wobec `wgp`

Eksporty 44 przykładów basal-bench, 4.5B i max, w drzewie i pojedynczo: różnica 0,0 ([forced](forced)).

| | 4.5B `wgp64` / `wgp` | max `wgp64` / `wgp` |
|---|---:|---:|
| pojedyncza decyzja | 0,962 | 0,978 |
| 512 tok., 1 / 5 pytań | 1,039 / 1,093 | 1,023 / 1,009 |
| 1792 tok., 1 / 5 pytań | 1,116 / 1,073 | 1,048 / 1,055 |
| 4096 tok., 1 pytanie | 1,168 | 1,108 |
| 16384 tok., 1 / 5 pytań | 1,370 / 1,354 | 1,329 / 1,287 |

Mniejsze bloki pomagają tylko wtedy, gdy bloków jest mniej niż SM; przy dłuższych stanach dwa warpgroupy na blok
lepiej wykorzystują wspólne kafelki K/V.

## Wybór według wielkości pracy

Domyślnie na compute capability 9.0 (forward f16) attention bierze `wgp64`, gdy dwukrotność liczby bloków po 128
wierszy nie przekracza liczby SM (`2 × Σ⌈rep × q_len / 128⌉ × NKV ≤ SM`), inaczej `wgp`; `BASAL_ATT=wgp` lub `wgp64`
wymusza jeden z nich. Pojedyncza decyzja to 52 (4.5B) i 112 (max) bloków po 64 wiersze; stan 512 tokenów już
140 i 288, więc idzie przez `wgp`. Oba warianty dają te same bity, więc wybór zależny od partii nie narusza
niezależności wyniku od partii.

Eksporty z wyborem (`auto`) wobec `wgp`: 0,0 dla 4.5B i max, w drzewie i pojedynczo ([auto](auto), [wg.log](
auto/wg.log)).

| | 4.5B `auto` / `wgp` | max `auto` / `wgp` |
|---|---:|---:|
| pojedyncza decyzja | 13,3 → 12,8 ms (0,960) | 19,7 → 19,3 ms (0,980) |
| 512–16384 tok. | 0,994–1,013 | 0,992–1,011 |

Pełne tabele: [auto/tables-4.5B.md](auto/tables-4.5B.md), [auto/tables-max.md](auto/tables-max.md),
[forced/tables-4.5B.md](forced/tables-4.5B.md), [forced/tables-max.md](forced/tables-max.md). Koszt sesji (razem z
pomiarem GEMM v3): ~$4,75.
