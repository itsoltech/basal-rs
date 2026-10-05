# Attention na tensor cores: precyzja i organizacja kernela

Data: 2026-10-05, basal-1.5-max, RTX 6000 Ada, limit 250 W. Przy stanie 16k
tokenów attention zajmuje ~61% czasu GPU ([profile-floor](../profile-floor/README.md)).

## Warianty precyzji (`run.sh`, `summary.py`)

`attn_tree_tc` liczy Q·K z operandami f16 rozbitymi na część wysoką i niską
(3 MMA) oraz P·V z P rozbitym tak samo (2 MMA). Warianty (`BASAL_ATT`):
`tc-pv1` (P zaokrąglone do f16), `tc-qk1` (Q·K z samych części wysokich),
`tc-f16` (oba). Softmax, akumulacja i wynik zawsze w f32. Zgodność z FP32
upstream na 44 przykładach i z FP32 runtime'u na 10 żądaniach z długimi
stanami (1k–16k tokenów, 30 pytań; upstream FP32 na tych długościach nie
istnieje):

| Wariant | Decyzje 44 | Maks. różnica logitu | Maks. różnica p (kal.) | Długie: decyzje | Długie: maks. różnica p | 16k, 1 pytanie | 16k, 5 pytań |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tc` | 44/44 | 0,165 | 0,0015 | 30/30 | 0,0017 | 7279 ms | 7993 ms |
| `tc-pv1` | 44/44 | 0,339 | 0,0012 | 30/30 | 0,0006 | 7101 ms | 7768 ms |
| `tc-qk1` | 44/44 | 0,226 | 0,0044 | 30/30 | 0,0010 | 6697 ms | 7183 ms |
| `tc-f16` | 44/44 | 0,315 | 0,0068 | 30/30 | 0,0022 | 6498 ms | 7058 ms |

Pojedyncze MMA skraca forward 16k tylko o ~11% przy około dwukrotnie większej
różnicy logitów, więc domyślny pozostaje `tc`. Liczba MMA nie jest głównym
ograniczeniem kernela.

## Organizacja kernela bez zmiany wyników

Nsight Compute, jedno wywołanie w forwardzie stanu ~4k tokenów (wersja z
32 tokenami zapytań na blok): potok tensor cores aktywny 57,6% cykli,
trafienia L2 95%, DRAM 1,9% przepustowości, 2 aktywne warpy na scheduler
(zajętość 16,7%, ograniczona rejestrami i shared memory), przestoje głównie
na potoku arytmetycznym; ~144 mln instrukcji FP32 (softmax, `expf`,
rozbicie P).

| Wersja | Zmiana | 16k, 1 pytanie | 16k, 5 pytań | 8k, 1 pytanie | 2k, 1 pytanie |
|---|---|---:|---:|---:|---:|
| przed | Q w shared memory, 16 tokenów zapytań na blok, 68 KiB na blok | 7279 ms | 7993 ms | 2786 ms | 580 ms |
| [attn-occupancy](../attn-occupancy/) | fragmenty Q w rejestrach, 34 KiB na blok | 7072 ms | 7838 ms | 2745 ms | 560 ms |
| [attn-bm128](../attn-bm128/) | 32 tokeny zapytań (8 warpów) na blok | 6785 ms | 7388 ms | 2650 ms | 565 ms |

Obie zmiany dają bitowo te same logity co poprzedni kernel
(`compare-previous-vs-new.json`: różnica 0,0). Pomiary długich stanów to
pojedyncze przebiegi po sobie (`bench-requests`, mediana z 2 powtórzeń), bez
kolejności ABBA; różnice rzędu 3% mieszczą się w rozrzucie, spadek 7% przy
16k powtarza się dla 1 i 5 pytań.
