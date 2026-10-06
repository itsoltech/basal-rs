# Konfiguracja domyślna przy 250 W i 300 W

Data: 2026-10-06, basal-1.5-max, RTX 6000 Ada (domyślny limit mocy 300 W).
Konfiguracja domyślna: tabela GEMM `gemm-equiv/gemm-algos-f16-invariant-groups.json`,
kernel attention z `attn-kernel`, HRRN i drugi tor dla żądań powyżej 4096
tokenów. Skrypt: `run.sh` (kolejno 250 W, potem 300 W, `nvidia-smi -pl`),
zestawienie: `summary.py 250w 300w`.

| Pomiar | 250 W | 300 W |
|---|---:|---:|
| Jedno pytanie przez HTTP, 1 klient | 13,9 żądania/s, p50 72 ms | 17,3 żądania/s, p50 56 ms |
| Jedno pytanie przez HTTP, 32 klientów | 21,8 żądania/s, p50 1450 ms | 26,3 żądania/s, p50 1209 ms |
| Energia przy 32 klientach | 11,5 J/decyzję | 11,4 J/decyzję |
| Ruch mieszany, sekwencyjnie | 1,98 żądania/s (119/min) | 2,39 żądania/s (143/min) |
| Ruch mieszany, 32 klientów | 2,24 żądania/s (134/min) | 2,69 żądania/s (161/min) |
| Stan 16k, 1 / 5 pytań | 5240 / 5891 ms | 4816 / 5314 ms |
| Stan 2k, 1 / 5 pytań | 439 / 577 ms | 380 / 502 ms |
| Zegar SM pod obciążeniem | 880–1070 MHz | 1130–1310 MHz |

Ruch mieszany (`tools/bench/mixed.jsonl`), napływ otwarty, p95:

| Napływ | Krótkie pytanie 250 W | 300 W | Dokument 1k–2k 250 W | 300 W | Dokument 4k–8k (p50) 250 W | 300 W |
|---:|---:|---:|---:|---:|---:|---:|
| 0,9/s | 567 ms | 398 ms | 892 ms | 717 ms | 3,4 s | 2,4 s |
| 1,3/s | 878 ms | 579 ms | 1,18 s | 0,85 s | 5,0 s | 3,3 s |
| 1,6/s | 1,05 s | 0,67 s | 1,28 s | 1,01 s | 7,0 s | 4,2 s |
| 2,0/s | 1,20 s | 0,88 s | 1,53 s | 1,09 s | 17,1 s | 6,0 s |
| 2,4/s | 1,84 s | 1,04 s | 2,00 s | 1,20 s | 25,6 s | 13,3 s |

Przy 250 W napływ 2,0/s i 2,4/s przekracza przepustowość (zrealizowane 1,86
i 1,92/s; kolejka długich dokumentów rośnie), przy 300 W granica leży około
2,2–2,4/s (zrealizowane 2,06 i 2,25/s). Krótkie pytania mają p95 około
1 s do napływu ~2,4/s przy 300 W.

Względem pomiaru ruchu mieszanego sprzed zmian GEMM i attention
([mixed-load](../mixed-load/README.md), 250 W): sekwencyjnie 1,74 → 1,98
żądania/s, 32 klientów 1,86 → 2,24 żądania/s. Odpowiedzi identyczne we
wszystkich fazach obu przebiegów (różnica 0,0).

Pojedyncze przebiegi każdego limitu, po sobie (najpierw 250 W). Karta
pozostaje przy domyślnym limicie 300 W.
