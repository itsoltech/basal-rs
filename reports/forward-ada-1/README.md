# Forward CUDA: mniej kerneli i pracy CPU na RTX 6000 Ada

Data: 2026-10-08. Serwer testowy z RTX 6000 Ada (sterownik 580), basal-1.5-4.5B f16, tabela GEMM z binarki.
Skrypty: [run-nsys.sh](../../tools/perf/run-nsys.sh) (profil), [run-var.sh](../../tools/perf/run-var.sh) (A/B),
[nsys_forward.py](../../tools/perf/nsys_forward.py), [variant_table.py](../../tools/perf/variant_table.py).

## Profil przed zmianami

Nsight Systems, limit 200 W ([profile-base/nsys.md](profile-base/nsys.md)), udział w czasie GPU jednego forwardu:

| stan | GEMM | attention | normy+residual | bias+SiLU | merge_heads | qkv_rope | split_hilo | przerwy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 tok., 1 pytanie | 69,6% | 21,2% | 2,9% | 2,7% | 1,6% | 1,3% | 0,4% | 1,2% |
| 1792 tok. | 65,9% | 25,9% | 2,5% | 2,9% | 1,4% | 1,1% | 0,3% | 0,4% |
| 4096 tok. | 55,1% | 36,6% | 2,3% | 3,4% | 1,2% | 1,0% | 0,2% | 0,2% |
| 16384 tok. | 27,2% | 67,9% | 1,2% | 2,2% | 0,8% | 0,6% | 0,1% | 0,0% |

Przerwy między kernelami w trakcie forwardu są małe. Nie obejmują pracy CPU przed pierwszym kernelem (plan,
cos/sin RoPE, kopie wejść).

Attention (Nsight Compute, 4096 tokenów): 216 rejestrów na wątek, jeden blok 256 wątków na SM (occupancy 17%),
pipeline tensor core zajęty w 63% aktywnych cykli. Bloki po 64 wiersze i odwrócona kolejność bloków (najdłuższe
najpierw) dały wynik bitowo ten sam, ale nie były szybsze (BM 64: do 11% wolniej przy 16k), więc nie weszły.

## Zmiany

Wszystkie bez zmiany obliczeń:

- `qkv_rope` zapisuje K i V od razu jako płaszczyzny f16 hi/lo dla attention na tensor core (te same zaokrąglenia
  co `split_hilo`); znikają dwa kernele `split_hilo` na warstwę i zapis f32 K/V, którego attention nie czyta
  (zostaje dla przechwytywania prefiksów). Płaszczyzna V lo, zerowa w forwardzie f16, nie jest zapisywana.
- Attention zapisuje wynik od razu jako scalone głowy `[token, NH*HD]` w dtype forwardu; znika `merge_heads`.
- Końcowa norma liczy się w kernelu `residual_rmsnorm` ostatniej warstwy.
- cos/sin RoPE każdej pozycji liczone raz przy ładowaniu modelu (16 MB RAM dla 32k pozycji), zamiast sin/cos w
  f64 dla każdego tokenu przy każdym forwardzie.
- Wyłączone śledzenie zdarzeń cudarc: przy jednym streamie cudarc nie czekał na te zdarzenia, a tworzył i niszczył
  dwa przy każdej alokacji.
- Shared memory kernela `attn_tree_tc` ustawiona na faktyczne 34 KiB (było 68 KiB, bez wpływu na czas).

Liczba kerneli na forward 4.5B: 726 → 546 w B2 (Nsight Systems), w B3 o jeden mniej (końcowa norma).

## Wyniki

Eksporty 44 przykładów basal-bench (drzewo): A wobec B2 i B3 różnica 0,0 ([compare-A-B3.json](compare-A-B3.json)).
Ostateczny build (B3 bez wariantów attention) wobec A, różnica 0,0: 4.5B f16 i bf16, max f16 (każdy z tabelą GEMM
niezależną od partii dla tej karty), [check-final](check-final/check.log); Metal: eksport basal-1.5-mini na Macu
bitowo równy poprzedniemu buildowi ([compare-metal-mini.json](check-final/compare-metal-mini.json)).

Czas, A/B z rotacją kolejności (A, B2, B3, potem odwrotnie; 4 powtórzenia), limit 300 W, zegar SM zablokowany na
1710 MHz ([clocks.csv](clocks.csv)); mediana powtórzeń 2–4 (pierwsze, na zimnej karcie, odbiegało od reszty):
[tables.md](tables.md). A: commit `4cf07da`; B2: fuzje `qkv_rope`/hi-lo i scalonych głów; B3: B2 oraz końcowa norma,
tablica RoPE i bez zdarzeń cudarc.

| | A | B3 | B3 / A |
|---|---:|---:|---:|
| pojedyncza decyzja (mediana) | 22,3 ms | 21,4 ms | 0,956 |
| 512 tok., 1 pytanie | 45,7 ms | 43,9 ms | 0,961 |
| 1792 tok., 1 / 5 pytań | 157,4 / 240,9 ms | 155,4 / 236,5 ms | 0,987 / 0,982 |
| 4096 tok., 1 / 5 pytań | 386,6 / 486,2 ms | 382,0 / 475,0 ms | 0,988 / 0,977 |
| 8192 tok., 1 / 5 pytań | 907,8 / 1061,8 ms | 895,3 / 1047,2 ms | 0,986 / 0,986 |
| 16384 tok., 5 pytań | 2759 ms | 2723 ms | 0,987 |

Wyjątki: 2048 tok. z 5 pytaniami 1,014 (B2 tamże 1,044: rozrzut pomiaru), 16384 tok. z 1 pytaniem 1,000.

Bez zablokowanego zegara (pierwsze A/B) karta zmieniała zegar SM od 210 do 2760 MHz przy 46–82 °C, a różnice
między buildami ginęły w tym rozrzucie; czasy samych kerneli w Nsight Systems były wtedy równe (attention) lub
krótsze (forward 361,0 → 356,6 ms przy 4096 tokenach).

Na H100 nie mierzono. Zysk po stronie CPU (RoPE, zdarzenia) powinien być tam względnie większy, bo forward jest
krótszy; wymaga pomiaru.
