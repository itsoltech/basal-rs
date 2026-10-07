# basal-1.5 na Apple M2 Max (Metal)

Data: 2026-10-07. Apple M2 Max (GPU 30 rdzeni, 32 GB pamięci wspólnej),
macOS 27.0.1, zasilanie sieciowe. Modele `Remek/basal-1.5-max` (rewizja
`be1b5ee7`), `Remek/basal-1.5-4.5B` (`784a683b`), `Remek/basal-1.5-mini`
(`1978d070`). Upstream v1.5.0 (`cd63c083`) z MLX 0.32.3 i mlx-lm 0.32.0.
basal-rs z commitu `854c475` (Metal, f16, attention po jednostkach drzewa).
Te same skrypty co na M1 Pro ([metal-m1-pro-1.5](../metal-m1-pro-1.5/README.md)):
`run.sh` (zgodność, MLX bf16 i 8-bit), `run-f16.sh` (MLX f16), tabele:
`summary.py`; eksporty upstream w `upstream-*`.

Warianty upstream: `mlx` (`basal-serve --mode mlx`, wagi bf16, ścieżka
serwowana na Macu), `mlx-q8` (8 bitów, grupa 64, odpowiednik `-MLX-8bit`),
`mlx-f16` (ten sam backend z modelem rzutowanym na float16, ta sama
precyzja co basal-rs; upstream nie ma takiego trybu).

Przed każdym przebiegiem 30 s przerwy i zapis `pmset -g therm` (`therm.log`,
bez ostrzeżeń). `cpu.log`: procesy innych programów powyżej 30% CPU co 30 s;
testy innego projektu skończyły się przed pierwszym pomiarem szybkości,
później tylko krótkie skoki procesów systemowych. Przebiegi tego samego
wariantu różnią się o mniej niż 0,3%.

## Zgodność z upstream FP32

Referencje: eksporty FP32 upstream z CUDA. 44 przykłady basal-bench;
`/v1/basal` basal-rs wobec `Server.decide` upstream FP32.

| Model | Wariant | Decyzje 44 | Maks. różnica p po kalibracji | Maks. różnica logitu | `/v1/basal`: pytania z tymi samymi polami | Maks. różnica p |
|---|---|---:|---:|---:|---:|---:|
| max | basal-rs f16 (pojedynczo i drzewo) | 44/44 | 0,0009 | 0,306 | 33/33 | 0,0013 |
| max | upstream `mlx` (bf16) | 43/44 | 0,0832 | 1,142 | | |
| max | upstream `mlx-f16` | 44/44 | 0,0016 | 0,581 | | |
| max | upstream `mlx-q8` | 44/44 | 0,0135 | 1,267 | | |
| 4.5B | basal-rs f32 | 44/44 | 0,0001 | 0,0002 | 55/55 | 0,0000 |
| 4.5B | basal-rs f16 (pojedynczo i drzewo) | 44/44 | 0,0021 | 0,043 | 55/55 | 0,0027 |
| 4.5B | upstream `mlx` (bf16) | 44/44 | 0,0192 | 0,316 | | |
| 4.5B | upstream `mlx-f16` | 44/44 | 0,0079 | 0,051 | | |
| 4.5B | upstream `mlx-q8` | 44/44 | 0,0380 | 0,392 | | |
| mini | basal-rs f32 | 44/44 | 0,0000 | 0,0001 | 55/55 | 0,0000 |
| mini | basal-rs f16 (pojedynczo i drzewo) | 44/44 | 0,0040 | 0,028 | 55/55 | 0,0041 |
| mini | upstream `mlx` (bf16) | 44/44 | 0,0546 | 0,243 | | |
| mini | upstream `mlx-f16` | 44/44 | 0,0066 | 0,028 | | |
| mini | upstream `mlx-q8` | 44/44 | 0,0710 | 0,477 | | |

Eksporty upstream `mlx` i `mlx-f16` na M2 Max są bitowo równe tym z M1 Pro. basal-rs
pojedynczo i w drzewie daje bitowo te same logity. Upstream `mlx` (bf16)
na max zmienia decyzję w `bench/7`, gdzie FP32 daje 0,531 / 0,469.

## Pojedyncza decyzja

Metodyka basal-bench (44 przykłady, oba porządki opcji, batch 1, 39 pomiarów
po odrzuceniu 5; przepustowość w grupach po 16 pytań). Kolejność: basal-rs 1,
upstream `mlx`+`mlx-q8` 1, upstream 2, basal-rs 2 (`run.sh`), basal-rs 3,
`mlx-f16` 1, `mlx-f16` 2, basal-rs 4 (`run-f16.sh`). Zakresy z przebiegów.

| Model | Wariant | lat2 mediana | lat2 p95 | lat1 | Decyzje/s | Pamięć |
|---|---|---:|---:|---:|---:|---:|
| max | basal-rs f16 | 526–527 ms | 662–663 ms | 366–367 ms | 2,33–2,34 | |
| max | upstream `mlx` (bf16) | 992 ms | 1239 ms | 625 ms | 1,04 | 21,0 GB |
| max | upstream `mlx-f16` | 886 ms | 1136–1137 ms | 569–570 ms | 1,18 | 21,0 GB |
| max | upstream `mlx-q8` | 1063 ms | 1293–1294 ms | 763 ms | 0,91 | 12,3 GB |
| 4.5B | basal-rs f16 | 235 ms | 305 ms | 167 ms | 5,26 | |
| 4.5B | upstream `mlx` (bf16) | 427 ms | 514 ms | 278 ms | 2,38 | 9,1 GB |
| 4.5B | upstream `mlx-f16` | 371–372 ms | 447–448 ms | 236–237 ms | 2,72 | 9,1 GB |
| 4.5B | upstream `mlx-q8` | 484–485 ms | 593–594 ms | 353–354 ms | 2,00 | 5,3–5,4 GB |
| mini | basal-rs f16 | 80 ms | 104 ms | 62 ms | 15,76–15,79 | |
| mini | upstream `mlx` (bf16) | 151 ms | 178–179 ms | 93 ms | 6,72 | 3,2 GB |
| mini | upstream `mlx-f16` | 130 ms | 154 ms | 79 ms | 7,81 | 3,2 GB |
| mini | upstream `mlx-q8` | 164 ms | 199 ms | 118 ms | 5,92 | 2,1 GB |

Pamięć upstream według `basal-bench` (`mem_gb`); dla basal-rs nie mierzona
(wagi f16: max ~21,6 GB, 4.5B ~9 GB, mini ~3 GB).

| Model | basal-rs wobec `mlx-f16`: lat2 / przepustowość | wobec `mlx` (bf16) | wobec `mlx-q8` |
|---|---:|---:|---:|
| max | −41% / 1,98× | −47% / 2,25× | −50% / 2,57× |
| 4.5B | −37% / 1,93× | −45% / 2,21× | −52% / 2,63× |
| mini | −38% / 2,02× | −47% / 2,35× | −51% / 2,66× |

Na M1 Pro (wersja z SDPA) przewaga wobec `mlx-f16` wynosiła 38–40% w lat2
i 1,70–1,88× w przepustowości. Na M2 Max MLX bf16 jest o 12–16% wolniejszy
od MLX f16 (na M1 Pro 15–16%).

## Czego nie mierzono

Serwer HTTP, długie stany, ruch mieszany, energia.
