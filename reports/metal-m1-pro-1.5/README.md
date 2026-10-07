# basal-1.5 na Apple M1 Pro (Metal)

Data: 2026-10-06. Apple M1 Pro (GPU 14 rdzeni, 32 GB pamięci wspólnej,
limit roboczy Metal 26,8 GB), macOS 27.0.1. Modele `Remek/basal-1.5-max`
(rewizja `be1b5ee7`), `Remek/basal-1.5-4.5B` (`784a683b`) i
`Remek/basal-1.5-mini` (`1978d070`). Upstream v1.5.0 (`cd63c083`) z MLX
0.32.3 i mlx-lm 0.32.0. basal-rs z commitu `f093e59` (Metal, f16).
Skrypty: `run.sh` (zgodność, MLX bf16 i 8-bit), `run-f16.sh` (MLX f16),
tabele: `summary.py`.

Warianty upstream:

- `mlx`: `basal-serve --mode mlx` na wagach bf16, ścieżka serwowana przez
  upstream na Macu;
- `mlx-q8`: wagi bloków dekodera w 8 bitach (affine, grupa 64) robione przy
  ładowaniu z bf16, odpowiednik portów `-MLX-8bit`;
- `mlx-f16`: ten sam backend z modelem rzutowanym po załadowaniu na float16
  (`set_dtype`). Upstream nie ma takiego trybu; dodaliśmy go, bo GPU M1 nie
  ma sprzętowej arytmetyki bf16 i porównanie tylko z bf16 faworyzowałoby
  basal-rs.

Przed każdym przebiegiem 60 s przerwy i zapis `pmset -g therm`
(`therm.log`, 37 wpisów): bez ostrzeżeń termicznych i ograniczeń
wydajności (w pierwszych 7 wpisach skrypt zapisywał tylko ostatnią linię
tego polecenia).

## Zgodność z upstream FP32

Referencje: eksporty FP32 upstream z CUDA (`../reference-1.5-max-fp32`,
`../reference-basal-1.5-*-fp32`). 44 przykłady basal-bench; `/v1/basal`
basal-rs wobec `Server.decide` upstream FP32 (pytania `choice`, `noul`,
`score`, `multi`, `act`, `evidence`, `facts: "auto"`). Eksporty upstream MLX:
`../reference-basal-1.5-*-mlx{,-q8,-f16}`.

| Model | Wariant | Decyzje 44 | Maks. różnica p po kalibracji | Maks. różnica logitu | `/v1/basal`: pytania z tymi samymi polami | Maks. różnica p |
|---|---|---:|---:|---:|---:|---:|
| max | basal-rs f16, pojedynczo | 44/44 | 0,0039 | 0,207 | 33/33 | 0,0014 |
| max | basal-rs f16, drzewo | 44/44 | 0,0062 | 0,307 | 33/33 | 0,0014 |
| max | upstream `mlx` (bf16) | 43/44 | 0,0832 | 1,142 | | |
| max | upstream `mlx-f16` | 44/44 | 0,0016 | 0,581 | | |
| max | upstream `mlx-q8` | 44/44 | 0,0135 | 1,267 | | |
| 4.5B | basal-rs f32 | 44/44 | 0,0001 | 0,0002 | 55/55 | 0,0000 |
| 4.5B | basal-rs f16, pojedynczo | 44/44 | 0,0045 | 0,052 | 55/55 | 0,0029 |
| 4.5B | basal-rs f16, drzewo | 44/44 | 0,0036 | 0,039 | 55/55 | 0,0029 |
| 4.5B | upstream `mlx` (bf16) | 44/44 | 0,0192 | 0,316 | | |
| 4.5B | upstream `mlx-f16` | 44/44 | 0,0079 | 0,051 | | |
| 4.5B | upstream `mlx-q8` | 44/44 | 0,0380 | 0,392 | | |
| mini | basal-rs f32 | 44/44 | 0,0000 | 0,0001 | 55/55 | 0,0000 |
| mini | basal-rs f16, pojedynczo | 44/44 | 0,0073 | 0,029 | 55/55 | 0,0040 |
| mini | basal-rs f16, drzewo | 44/44 | 0,0040 | 0,028 | 55/55 | 0,0040 |
| mini | upstream `mlx` (bf16) | 44/44 | 0,0546 | 0,243 | | |
| mini | upstream `mlx-f16` | 44/44 | 0,0066 | 0,028 | | |
| mini | upstream `mlx-q8` | 44/44 | 0,0710 | 0,477 | | |

basal-rs f16 daje decyzje FP32 na wszystkich przykładach, a różnice
prawdopodobieństw są tego samego rzędu co na CUDA
([compat-1.5-small](../compat-1.5-small/README.md),
[rust-cuda-1.5-max](../rust-cuda-1.5-max/README.md)). Upstream `mlx-f16`
mieści się w tym samym zakresie: na max bliżej FP32 niż basal-rs, na 4.5B
dalej. Różnica logitu 0,58 przy różnicy prawdopodobieństw 0,0016
(`mlx-f16`, max) to przesunięcie logitów wszystkich liter o podobną
wartość, które nie zmienia rozkładu. Upstream `mlx` (bf16) na max zmienia
decyzję w `bench/7`, gdzie FP32 daje 0,531 / 0,469.

Prompty i token IDs (`check-prompts-*.json`): te same 8 różnic formatu
eksportu co na CUDA (pytania `multi` i `facts`), opisanych w
[compat-1.5-small](../compat-1.5-small/README.md).

Na Metal wynik zależy od sposobu pakowania. Eksport `f16-single` (pytanie
pojedynczo) wobec `f16-tree` (drzewo z innymi pytaniami,
`compare-*-f16-single-vs-tree.json`): te same decyzje 44/44, maks. różnica
logitu 0,174 / 0,034 / 0,021 (max / 4.5B / mini), prawdopodobieństwa po
kalibracji do 0,0024 / 0,0081 / 0,0053. Na CUDA oba eksporty są bitowo
równe. Odpowiedź serwera na Macu może więc w tym zakresie zależeć od innych
żądań w tej samej partii.

Eksporty upstream MLX liczą logity tymi samymi funkcjami co
`MLXBackend._group` (wspólny prefiks w KV cache, sufiksy w jednej partii);
różnica ich prawdopodobieństw do `run_shared` upstream wynosi do 1,3·10⁻⁷.

## Pojedyncza decyzja

Metodyka basal-bench: 44 przykłady, oba porządki opcji, batch 1, 39 pomiarów
po odrzuceniu 5; przepustowość w grupach po 16 pytań. Kolejność przebiegów:
basal-rs 1, upstream `mlx`+`mlx-q8` 1, upstream 2, basal-rs 2 (`run.sh`),
potem basal-rs 3, `mlx-f16` 1, `mlx-f16` 2, basal-rs 4 (`run-f16.sh`).
Zakresy z przebiegów.

| Model | Wariant | lat2 mediana | lat2 p95 | lat1 | Decyzje/s | Pamięć |
|---|---|---:|---:|---:|---:|---:|
| max | basal-rs f16, przebiegi 1–2 | 1247–1323 ms | 1777–1801 ms | 790–794 ms | 0,87–0,89 | |
| max | basal-rs f16, przebiegi 3–4 | 1197–1199 ms | 1713–1714 ms | 765–766 ms | 0,95 | |
| max | upstream `mlx` (bf16) | 2264–2265 ms | 2676–2678 ms | 1350–1356 ms | 0,44 | 21,0 GB |
| max | upstream `mlx-f16` | 1969 ms | 2324 ms | 1180–1181 ms | 0,50–0,51 | 21,0 GB |
| max | upstream `mlx-q8` | 2392–2396 ms | 2952–2954 ms | 1760–1764 ms | 0,40 | 12,2–12,3 GB |
| 4.5B | basal-rs f16 | 495–502 ms | 647–672 ms | 349–356 ms | 2,12–2,18 | |
| 4.5B | upstream `mlx` (bf16) | 955–956 ms | 1145–1150 ms | 618 ms | 1,06 | 9,1 GB |
| 4.5B | upstream `mlx-f16` | 823–824 ms | 986–996 ms | 531–533 ms | 1,22–1,24 | 9,1 GB |
| 4.5B | upstream `mlx-q8` | 1060 ms | 1301–1304 ms | 779 ms | 0,90–0,91 | 5,4 GB |
| mini | basal-rs f16 | 170–174 ms | 220–223 ms | 119–121 ms | 6,31–6,35 | |
| mini | upstream `mlx` (bf16) | 318–323 ms | 378–386 ms | 205–208 ms | 3,18–3,19 | 3,2 GB |
| mini | upstream `mlx-f16` | 274–276 ms | 325–331 ms | 178 ms | 3,73 | 3,2 GB |
| mini | upstream `mlx-q8` | 357–362 ms | 442–446 ms | 263–264 ms | 2,68–2,70 | 2,1 GB |

Pamięć upstream według `basal-bench` (`mem_gb`); dla basal-rs nie mierzona
(wagi f16: max ~21,6 GB, 4.5B ~9 GB, mini ~3 GB).

Wobec `mlx-f16` (ta sama precyzja, przebiegi sąsiednie) basal-rs ma
medianę lat2 krótszą o 38% (mini), 40% (4.5B) i 39% (max), a przepustowość
1,70×, 1,74× i 1,88× wyższą. Wobec ścieżki serwowanej `mlx` (bf16): lat2
krótsza o 46%, 48% i 42–45%, przepustowość 2,0× dla wszystkich trzech
modeli. `mlx-q8` jest wolniejszy od `mlx`: decyzja to prefill kilkuset
tokenów ograniczony obliczeniami, a wagi 8-bit trzeba rozpakować przed
mnożeniem w 16 bitach; 8-bit zmniejsza zużycie pamięci.

basal-rs na max: przebiegi 1–2 o 4–10% wolniejsze od 3–4 przy tych samych
warunkach termicznych; przyczyny nie ustalono (max zajmuje ~22 z 32 GB
pamięci; zużycia swapu w czasie tych przebiegów nie zapisywano).

## Ładowanie max

basal-rs wczytuje max w 91–126 s (konwersja bf16 → f16), przy czym system
na chwilę używa do ~6 GB swapu; upstream MLX ładuje go w ~10 s (wagi
mapowane leniwie). Pomiary decyzji zaczynają się po załadowaniu.

## Czego nie mierzono

Serwer HTTP, długie stany, ruch mieszany, energia. Wyniki dotyczą M1 Pro;
nowsze układy z serii M mają inną obsługę formatów 16-bitowych, więc
proporcje między wariantami mogą być inne.
