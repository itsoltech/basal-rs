# Tabela GEMM niezależna od partii: wybór na drabince M 128–16384

Data: 2026-10-05, basal-1.5-max, RTX 6000 Ada, limit 250 W.

Tryb niezależny od partii używa jednego algorytmu cuBLASLt (bez split-K) na
kształt wag dla każdej liczby wierszy M. Poprzednia tabela
(`../gemm/gemm-algos-f16-invariant.json`) wybierała go według średniego
spowolnienia względem najlepszego algorytmu w klasach M = 16…3072 (32 z 52
klas to M ≤ 512); większych M nie mierzyła, choć forwardy długich stanów i
partie serwera mają 4k–16k wierszy.

`basal gemm-search --invariant --m-classes 128,256,512,1024,2048,4096,8192,16384`:

- `gemm-algos-f16-invariant-ladder-totaltime.json` (kryterium: łączny czas):
  klasy 8192 i 16384 dominują wybór, algorytmy są 2–3× wolniejsze od
  najlepszych przy M = 128–512 (np. projekcja down przy M = 256: 0,63 vs
  0,21 ms). Odrzucona bez pomiaru A/B.
- `gemm-algos-f16-invariant-ladder.json` (kryterium: średnia geometryczna
  spowolnienia, każda klasa z tą samą wagą): spowolnienie względem
  najlepszego algorytmu klasy 1,0–1,56× (`gemm-search.log`). Wszystkie cztery
  algorytmy różnią się od poprzedniej tabeli.

## A/B (`run-ab.sh`)

| Pomiar | Poprzednia tabela | Nowa tabela |
|---|---:|---:|
| Pojedyncza decyzja, basal-bench, mediana lat2 (ABBA, 2 rundy) | 60,1–61,5 ms | 64,1–64,2 ms |
| basal-bench, przepustowość w grupach po 16 | 19,8–20,2 dec/s | 21,0–21,1 dec/s |
| HTTP, 1 klient, p50 | 68–70 ms | 68–69 ms |
| HTTP, 32 klientów (ABBA) | 17,25–17,53 żądania/s, p50 1816–1845 ms | 18,46–18,49 żądania/s, p50 1721–1724 ms |
| Energia przy 32 klientach | 14,2–14,5 J/decyzję | 13,5 J/decyzję |
| Stan 16k, 1 / 5 pytań | 7051 / 7758 ms | 6786 / 7316 ms |
| Stan 2k, 1 / 5 pytań | 603 / 784 ms | 583 / 759 ms |

W `ab-bench` nowa tabela jest wolniejsza na wszystkich 44 pytaniach
pojedynczej decyzji (mediana ilorazu 1,049), a szybsza przy partiach, długich
stanach i pod obciążeniem HTTP. Przez HTTP przy jednym kliencie różnicy nie
widać. Niezależność od partii zachowana (`compare-new-single-vs-tree.json`:
0,0; odpowiedzi pod obciążeniem identyczne we wszystkich fazach); względem
FP32 upstream 44/44, maks. różnica logitu 0,171
(`compare-upstream-fp32-vs-new.json`).

Dla serwera z ruchem współbieżnym i długimi dokumentami nowa tabela daje
~6% więcej żądań na sekundę i ~6% mniej energii na decyzję; dla pojedynczych
żądań bez kolejki poprzednia może być szybsza o kilka ms. Pomiary pojedyncze
(poza ABBA), karta pod limitem mocy.
