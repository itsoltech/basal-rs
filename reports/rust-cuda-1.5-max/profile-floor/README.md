# Rozkład czasu forwardu basal-1.5-max

Data: 2026-10-05, RTX 6000 Ada, limit 250 W, tabela GEMM niezależna od
partii. Skrypt: `run.sh`.

## Krótkie pytania (`basal profile`)

Synchronizacja po każdej sekcji, 43 przykłady basal-bench po jednym na
forward (`profile-v2-batch1.json`, bez diagnostyki `--residual-max`):

| Sekcja | Udział |
|---|---:|
| gate/up (GEMM) | 42,1% |
| down + residual (GEMM) | 19,8% |
| qkv (GEMM) | 10,6% |
| o_proj + residual (GEMM) | 8,3% |
| bias + SiLU·up | 8,2% |
| attention | 7,1% |
| RMSNorm | 2,7% |
| bias + RoPE + podział głów | 1,2% |

GEMM zajmują ~81% czasu krótkiego pytania.

Przy 16 pytaniach na forward (`profile-v2-batch16.json`) sekcja RMSNorm ma
34,5%, ale to artefakt trybu profilowania: kernel RMSNorm trwa ~10 µs na
wywołanie (nsys), a czas sekcji obejmuje zwalnianie dużych tensorów poprzedniej
warstwy przy synchronizacji. Pierwsze pliki `profile-batch*.json` mierzono
jeszcze z redukcją `max|x|` po każdej warstwie (`fast_max_f32`, 169 ms na
forward), którą `basal profile` od tej wersji włącza tylko z
`--residual-max`.

## Bez synchronizacji (nsys)

Bezczynność GPU między kernelami w zwykłym wykonaniu: 0,4–1,8% czasu
forwardu (eksport partiami 44 przykładów i żądanie ze stanem ~2k tokenów).
Narzut hosta nie jest więc istotnym kosztem.

| Żądanie | Czas GPU | GEMM | Attention | Pozostałe |
|---|---:|---:|---:|---:|
| stan ~2k tokenów, 1 pytanie | 563 ms | 70,1% | 21,9% | 8,0% |
| stan ~16k, 1 pytanie | 6593 ms | 33,8% | 61,4% | 4,8% |
| stan ~16k, 5 pytań | 6961 ms | 33,2% | 61,9% | 4,9% |

Pliki `kern-*.csv` (sumy kerneli); raporty nsys nie są przechowywane.
Wyniki obejmują cały proces `basal decide`, w tym przygotowanie prefiksów
szablonu (2 × 50 wywołań na prefiksach 53 i 66 tokenów, pomijalne wobec
forwardu długiego stanu).
