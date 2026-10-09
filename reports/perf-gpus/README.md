# Wydajność na pięciu kartach NVIDIA

Data: 2026-10-08. basal-rs 0.1.4 i upstream v1.5.0 `fast` (BF16,
torch.compile, CUDA graphs) na tej samej karcie, ten sam klient, modele
basal-1.5-mini (`1978d070`), 4.5B (`784a683b`) i max (`be1b5ee7`).
Maszyny wynajęte w Shadeform, każda z jedną kartą, bez innych procesów i z
fabrycznym limitem mocy; wszystkie pięć mierzone równolegle tym samym
skryptem ([tools/perf/run-cloud.sh](../../tools/perf/run-cloud.sh),
[tools/cloud](../../tools/cloud/basal-cloud.py)). Tabele z danych:
[tools/perf/table.py](../../tools/perf/table.py), wynik w `tables.md`.

| Karta | Architektura, compute capability | Pamięć | Limit mocy | Sterownik | CPU maszyny (vCPU) |
|---|---|---:|---:|---|---|
| H100 80GB HBM3 | Hopper, 9.0 | 80 GB | 700 W | 535 + pakiet zgodności CUDA 12.9 / 13.0 | Xeon Platinum 8458P (16) |
| A100-SXM4-80GB | Ampere, 8.0 | 80 GB | 400 W | 580 | EPYC 7742 (16) |
| L40S | Ada, 8.9 | 48 GB | 350 W | 580 | Xeon Platinum 8462Y+ (12) |
| RTX 6000 Ada | Ada, 8.9 | 48 GB | 300 W | 580 | Xeon Gold 6338 (12) |
| RTX A6000 | Ampere, 8.6 | 48 GB | 300 W | 580 | Xeon E5-2699 v4 (6) |

basal-rs w konfiguracji domyślnej serwera: tabele GEMM `gemm_table: auto`
(wygenerowane na danej karcie przy pierwszym starcie), HRRN, tor długich
żądań, f16. H100 była dostępna tylko ze sterownikiem 535 (CUDA 12.2); obie
implementacje działały na nim przez biblioteki zgodności NVIDIA (forward
compatibility), co może wpływać na wynik.

Pomiary:

- pojedyncza decyzja: metodyka basal-bench (44 przykłady, oba porządki,
  batch 1; mediana i p95 z 39 pomiarów), `basal bench` i upstream `basal-bench`;
- HTTP, jedno pytanie na żądanie (44 przykłady w pętli), 1, 8 i 32 klientów:
  [tools/bench/loadtest.py](../../tools/bench/loadtest.py), oba serwery;
- ruch mieszany (`tools/bench/mixed.jsonl`: stany 0,1–16k tokenów, 1–14
  pytań; dla mini stany do ~4k tokenów), sekwencyjnie i 32 klientów: tylko
  basal-rs (upstream na max obsługuje ~17 żądań na minutę, co wydłużyłoby
  sesję o ponad godzinę na kartę).

## basal-rs wobec upstream

| Karta | Model | Pojedyncza decyzja: upstream / basal-rs (mediana) | Decyzje/s: upstream / basal-rs | HTTP 32 klientów: upstream / basal-rs (żądania/s) | Energia na decyzję (32 klientów): upstream / basal-rs |
|---|---|---:|---:|---:|---:|
| H100 | mini | 4,8 / 6,2 ms | 357 / 438 | 250 / 237 | 2,5 / 1,9 J |
| H100 | 4.5B | 9,8 / 13,2 ms | 144 / 172 | 98 / 155 | 7,1 / 4,0 J |
| H100 | max | 14,5 / 17,5 ms | 73 / 98 | 56 / 85 | 12,4 / 8,0 J |
| A100 | mini | 9,1 / 8,3 ms | 158 / 240 | 112 / 204 | 3,1 / 1,8 J |
| A100 | 4.5B | 19,8 / 19,5 ms | 63 / 92 | 48 / 80 | 8,2 / 4,7 J |
| A100 | max | 39,9 / 30,3 ms | 30 / 46 | 24 / 41 | 16,8 / 9,6 J |
| L40S | mini | 8,3 / 8,4 ms | 150 / 232 | 94 / 200 | 3,6 / 1,6 J |
| L40S | 4.5B | 21,5 / 21,6 ms | 49 / 82 | 33 / 72 | 10,6 / 4,6 J |
| L40S | max | 43,7 / 42,3 ms | 23 / 40 | 18 / 36 | 19,7 / 9,6 J |
| RTX 6000 Ada | mini | 9,0 / 7,8 ms | 103 / 189 | 91 / 157 | 3,3 / 1,8 J |
| RTX 6000 Ada | 4.5B | 28,8 / 20,8 ms | 30 / 63 | 20 / 54 | 15,3 / 5,5 J |
| RTX 6000 Ada | max | 66,2 / 43,5 ms | 13 / 30 | 10 / 27 | 29,4 / 11,2 J |
| RTX A6000 | mini | 13,0 / 11,4 ms | 86 / 128 | 58 / 93 | 5,0 / 3,1 J |
| RTX A6000 | 4.5B | 38,3 / 29,7 ms | 30 / 47 | 23 / 42 | 12,8 / 7,0 J |
| RTX A6000 | max | 78,0 / 50,2 ms | 14 / 24 | 10 / 21 | 31,1 / 14,1 J |

- Przepustowość serwera przy 32 klientach: basal-rs 1,5–2,8 raza wyższa niż
  upstream na wszystkich kartach i modelach, z wyjątkiem mini na H100 (0,95).
- Energia na decyzję: 1,3–2,8 raza niższa we wszystkich 15 parach.
- Opóźnienie pojedynczej decyzji: krótsze na RTX 6000 Ada i RTX A6000 (1,1–1,6
  raza), równe na L40S i A100 dla mini i 4.5B (A100 max: 1,3 raza krótsze),
  dłuższe na H100 (o 20–35%: 13,2 wobec 9,8 ms dla 4.5B). Na H100 upstream
  jest też szybszy przy jednym kliencie HTTP (4.5B: 83 wobec 66 żądań/s).
  Na najszybszej karcie czas obliczeń jest krótki, więc większy udział ma
  narzut uruchamiania kerneli; upstream używa CUDA graphs, basal-rs nie.
- Odpowiedzi: basal-rs na to samo żądanie zwraca bitowo to samo we
  wszystkich fazach (1, 8, 32 klientów, ruch mieszany); upstream różni się
  zależnie od partii o 0,001–0,068 prawdopodobieństwa. Bez błędów HTTP.

## HTTP: opóźnienia

Jedno pytanie na żądanie, p50 / p99 (ms):

| Karta | Model | 1 klient: upstream / basal-rs | 8 klientów: upstream / basal-rs | 32 klientów: upstream / basal-rs |
|---|---|---:|---:|---:|
| H100 | mini | 7 / 7 – 7 / 9 | 31 / 39 – 32 / 43 | 104 / 422 – 87 / 546 |
| H100 | 4.5B | 13 / 13 – 15 / 18 | 62 / 73 – 66 / 73 | 331 / 390 – 199 / 241 |
| H100 | max | 18 / 21 – 19 / 26 | 119 / 143 – 112 / 119 | 583 / 635 – 366 / 404 |
| A100 | mini | 11 / 12 – 11 / 12 | 76 / 85 – 47 / 57 | 275 / 435 – 146 / 204 |
| A100 | 4.5B | 29 / 32 – 23 / 26 | 149 / 177 – 116 / 127 | 679 / 740 – 390 / 473 |
| A100 | max | 45 / 47 – 36 / 44 | 295 / 350 – 221 / 235 | 1409 / 1506 – 778 / 838 |
| L40S | mini | 10 / 12 – 10 / 11 | 63 / 77 – 49 / 59 | 337 / 402 – 151 / 262 |
| L40S | 4.5B | 27 / 29 – 24 / 27 | 181 / 208 – 131 / 143 | 997 / 1056 – 434 / 488 |
| L40S | max | 52 / 62 – 46 / 52 | 361 / 428 – 260 / 293 | 1865 / 1948 – 886 / 935 |
| RTX 6000 Ada | mini | 12 / 14 – 10 / 11 | 106 / 122 – 56 / 63 | 326 / 484 – 189 / 460 |
| RTX 6000 Ada | 4.5B | 38 / 40 – 25 / 29 | 284 / 311 – 153 / 175 | 1637 / 1738 – 580 / 616 |
| RTX 6000 Ada | max | 88 / 103 – 53 / 69 | 683 / 785 – 354 / 394 | 3234 / 3356 – 1172 / 1244 |
| RTX A6000 | mini | 20 / 21 – 15 / 17 | 121 / 135 – 78 / 95 | 547 / 833 – 268 / 1177 |
| RTX A6000 | 4.5B | 48 / 54 – 36 / 43 | 293 / 361 – 206 / 226 | 1409 / 1527 – 749 / 905 |
| RTX A6000 | max | 102 / 107 – 65 / 83 | 636 / 752 – 413 / 448 | 3458 / 3743 – 1492 / 1621 |

Przy 32 klientach i mini p99 basal-rs bywa wyższe niż upstream (H100, RTX
A6000) przy niższym p50: przy krótkich żądaniach małego modelu serwer
pracuje blisko granicy CPU maszyny (RTX A6000: 6 vCPU).

## Ruch mieszany (basal-rs)

Żądania na minutę; p50 / p95 (s):

| Karta | mini sekwencyjnie | mini 32 klientów | 4.5B sekwencyjnie | 4.5B 32 klientów | max sekwencyjnie | max 32 klientów |
|---|---:|---:|---:|---:|---:|---:|
| H100 | 3196; 0,01 / 0,06 | 5536; 0,31 / 0,62 | 643; 0,02 / 0,51 | 819; 0,88 / 5,62 | 381; 0,03 / 0,87 | 454; 1,35 / 12,1 |
| A100 | 1950; 0,02 / 0,10 | 3204; 0,53 / 0,88 | 356; 0,04 / 0,93 | 433; 1,30 / 12,6 | 197; 0,07 / 1,68 | 231; 2,05 / 22,8 |
| L40S | 2047; 0,01 / 0,09 | 3176; 0,55 / 0,92 | 352; 0,04 / 0,88 | 430; 1,38 / 12,5 | 184; 0,08 / 1,77 | 214; 2,23 / 26,2 |
| RTX 6000 Ada | 1807; 0,02 / 0,10 | 2711; 0,63 / 1,16 | 295; 0,05 / 1,09 | 346; 1,66 / 16,8 | 145; 0,12 / 2,22 | 166; 3,03 / 35,6 |
| RTX A6000 | 1190; 0,02 / 0,17 | 1889; 0,88 / 1,54 | 200; 0,06 / 1,64 | 236; 2,01 / 22,4 | 105; 0,13 / 3,18 | 121; 3,55 / 46,5 |

Wysokie p95 przy 32 klientach to dokumenty 8k–16k tokenów czekające w
kolejce za innymi żądaniami w zamkniętej pętli; krótkie żądania mają własny
tor (podział na klasy w `mixed-rust-*.json`).

## Ograniczenia

- Jeden przebieg na kartę; powtórzenia na RTX 6000 Ada w chmurze i na naszym
  serwerze ([perf-1.5-small](../perf-1.5-small/README.md), 300 W) różnią się o
  kilka procent (np. 4.5B basal-rs 20,4 i 20,8 ms).
- Maszyny różnią się CPU, co wpływa na HTTP przy małych modelach.
- H100 przez biblioteki zgodności na sterowniku 535, nie na 580.
- Upstream w ruchu mieszanym nie był mierzony na tych kartach.

Czas sesji: pięć maszyn, 35–80 minut każda.
