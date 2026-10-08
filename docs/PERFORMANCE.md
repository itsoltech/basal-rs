# Wydajność

Jak basal-rs radzi sobie na różnych kartach i komputerach oraz jak zmienia się
wynik w różnych konfiguracjach. Każda liczba pochodzi z raportu w
[reports/](../reports/README.md) z danymi i opisem warunków; metodyka:
[BENCHMARKS.md](BENCHMARKS.md). Porównanie z upstream zawsze na tej samej
karcie. W ścieżkach serwowanych upstream liczy w BF16, a basal-rs w f16; na
Apple Silicon porównanie obejmuje także upstream w f16.

## Karty NVIDIA

Maszyny wynajęte, fabryczne limity mocy, konfiguracja domyślna basal-rs,
upstream v1.5.0 `fast` (BF16, torch.compile, CUDA graphs)
([perf-gpus](../reports/perf-gpus/README.md)).

Pojedyncza decyzja (metodyka basal-bench, mediana), upstream / basal-rs:

| Karta | mini | 4.5B | max |
|---|---:|---:|---:|
| H100 80 GB | 4,8 / 6,2 ms | 9,8 / 13,2 ms | 14,5 / 17,5 ms |
| A100 80 GB | 9,1 / 8,3 ms | 19,8 / 19,5 ms | 39,9 / 30,3 ms |
| L40S | 8,3 / 8,4 ms | 21,5 / 21,6 ms | 43,7 / 42,3 ms |
| RTX 6000 Ada | 9,0 / 7,8 ms | 28,8 / 20,8 ms | 66,2 / 43,5 ms |
| RTX A6000 | 13,0 / 11,4 ms | 38,3 / 29,7 ms | 78,0 / 50,2 ms |

Serwer HTTP, jedno pytanie na żądanie, 32 klientów: żądania/s (p99),
upstream / basal-rs:

| Karta | mini | 4.5B | max |
|---|---:|---:|---:|
| H100 80 GB | 250 (422 ms) / 237 (546 ms) | 98 (390 ms) / 155 (241 ms) | 56 (635 ms) / 85 (404 ms) |
| A100 80 GB | 112 (435 ms) / 204 (204 ms) | 48 (740 ms) / 80 (473 ms) | 24 (1506 ms) / 41 (838 ms) |
| L40S | 94 (402 ms) / 200 (262 ms) | 33 (1056 ms) / 72 (488 ms) | 18 (1948 ms) / 36 (935 ms) |
| RTX 6000 Ada | 91 (484 ms) / 157 (460 ms) | 20 (1738 ms) / 54 (616 ms) | 10 (3356 ms) / 27 (1244 ms) |
| RTX A6000 | 58 (833 ms) / 93 (1177 ms) | 23 (1527 ms) / 42 (905 ms) | 10 (3743 ms) / 21 (1621 ms) |

- Przepustowość: 1,5–2,8 raza wyższa niż upstream, poza mini na H100 (0,95).
- Energia na decyzję: 1,3–2,8 raza niższa na wszystkich kartach.
- Opóźnienie pojedynczej decyzji: krótsze na kartach do 300 W, równe na L40S
  i A100, dłuższe o 20–35% na H100, gdzie liczy się narzut uruchamiania
  kerneli (upstream używa CUDA graphs).
- Odpowiedź na to samo żądanie nie zależy od obciążenia (bitowo równa);
  upstream różni się zależnie od partii o 0,001–0,068 prawdopodobieństwa.

Ruch mieszany (stany 0,1–16k tokenów, 1–14 pytań), basal-rs, 32 klientów:
basal-1.5-max obsługuje od 121 żądań/min (RTX A6000) do 454 (H100);
basal-1.5-4.5B od 236 do 819; basal-1.5-mini od 1889 do 5536.

Kernele basal-rs są kompilowane dla compute capability 8.0, 8.9 i 9.0;
sprawdzone na wszystkich trzech (A100, RTX A6000 przez 8.0; L40S i RTX 6000
Ada; H100). Wymagany sterownik 575 lub nowszy; na kartach do centrów danych
ze starszym sterownikiem działa pakiet zgodności NVIDIA (H100 na 535).

## Limit mocy (RTX 6000 Ada, basal-1.5-max)

[power](../reports/rust-cuda-1.5-max/power/README.md): 250 W wobec domyślnych
300 W.

| Pomiar | 250 W | 300 W |
|---|---:|---:|
| HTTP, 1 klient | 13,9 żądania/s, p50 72 ms | 17,3 żądania/s, p50 56 ms |
| HTTP, 32 klientów | 21,8 żądania/s | 26,3 żądania/s |
| Energia na decyzję, 32 klientów | 11,5 J | 11,4 J |
| Ruch mieszany, 32 klientów | 134 żądania/min | 161 żądań/min |
| Stan 16k tokenów, 5 pytań | 5,9 s | 5,3 s |

Niższy limit kosztuje ~20% przepustowości przy tej samej energii na decyzję.

## Długie dokumenty

Stan ~16,5k tokenów, 5 pytań, RTX 6000 Ada 300 W: upstream 112 s, basal-rs
5,3 s ([rust-cuda-1.5-max](../reports/rust-cuda-1.5-max/README.md)).
basal-rs liczy wspólny stan raz dla wszystkich pytań (drzewo prefiksów), a
żądania powyżej 4096 tokenów idą do osobnego toru, który oddaje GPU krótkim
żądaniom między warstwami.

## Długość kontekstu (H100)

H100 PCIe 350 W, całe żądanie, jedno pytanie, upstream / basal-rs
([context-h100](../reports/context-h100/README.md)):

| Stan | 4.5B | max |
|---:|---:|---:|
| 512 tokenów | 23 / 31 ms | 41 / 56 ms |
| 1 792 tokenów | 84 / 100 ms | 152 / 194 ms |
| 4 096 tokenów | 487 / 267 ms | 892 / 510 ms |
| 16 384 tokenów | 4 200 / 1 743 ms | |
| 16 384 tokenów, 5 pytań | 29,5 / 2,0 s | |

Do ~2k tokenów domyślna tabela GEMM niezależna od partii jest na H100 o
20–35% wolniejsza od heurystyki cuBLASLt (wybór algorytmów pomija kernele
Hoppera); z heurystyką basal-rs jest tam porównywalny z upstream. Od ~4k
tokenów basal-rs jest szybszy 1,75–2,4 raza, przy pięciu pytaniach 10–15 razy.

## Apple Silicon (Metal)

Pojedyncza decyzja (mediana), upstream z MLX w bf16 (ścieżka serwowana) i f16
wobec basal-rs f16 ([M2 Max](../reports/metal-m2-max-1.5/README.md),
[M1 Pro](../reports/metal-m1-pro-1.5/README.md)):

| Model | M2 Max: MLX bf16 / f16 / basal-rs | M1 Pro: MLX bf16 / f16 / basal-rs |
|---|---:|---:|
| basal-1.5-max | 992 / 886 / 526 ms | 2265 / 1969 / 1197–1323 ms |
| basal-1.5-4.5B | 427 / 372 / 235 ms | 955 / 823 / 495–502 ms |
| basal-1.5-mini | 151 / 130 / 80 ms | 318–323 / 274–276 / 170–174 ms |

Na M2 Max 1,9–2,0 raza więcej decyzji na sekundę niż MLX f16. Laptop mierzony
na zasilaczu; na baterii i po przegrzaniu czasy rosną.

## Konfiguracja

| Ustawienie | Wpływ | Pomiar |
|---|---|---|
| `dtype: f16` (domyślne) | decyzje FP32 na 899–900 z 900 pytań; różnice to remisy w FP32 | [decision-sets](../reports/decision-sets/README.md) |
| `dtype: f32` | 900/900, logity do 0,0003 od FP32; wolniejsze, do sprawdzania zgodności | [decision-sets](../reports/decision-sets/README.md) |
| `gemm_table: auto` (domyślne) | tabela GEMM niezależna od partii, generowana raz na kartę i model; wynik nie zależy od obciążenia, pojedyncza decyzja ~8% szybsza niż heurystyka cuBLASLt (max, RTX 6000 Ada: 63,1 wobec 68,5 ms) | [ab-table-vs-invariant](../reports/rust-cuda-1.5-max/ab-table-vs-invariant/) |
| `gemm_table: none` | algorytmy wybierane przy pierwszym użyciu; wynik może zależeć od partii | |
| choice z 11–255 opcjami | strategia `luce2-1o` (domyślna od 0.1.4): 14–32 prompty na pytanie z 59–150 opcjami, 1,6–2,0 raza szybciej niż `luce2` z 0.1.3 przy tej samej trafności | [choice-sets](../reports/choice-sets/README.md) |
| `BASAL_CUDA_SYNC=blocking` | zwalnia ~0,7 rdzenia CPU przy pracy GPU, +2,5% czasu pytania | [choice-sets](../reports/choice-sets/README.md#czekanie-na-gpu-cuda) |
| `max_batch_tokens`, `long_tokens`, `schedule` | rozmiar partii, próg toru długich żądań, kolejność (HRRN, FIFO) | [mixed-load](../reports/rust-cuda-1.5-max/mixed-load/README.md) |

## Dobór karty

basal-1.5-max potrzebuje ok. 22 GB na wagi f16; mierzony był na kartach z 48 i
80 GB pamięci, mniejsze nie były sprawdzane. Na kartach do 300 W
(RTX 6000 Ada, RTX A6000) basal-rs daje największą przewagę nad upstream
(2,2–2,8 raza więcej żądań przy 32 klientach dla max i 4.5B). H100 obsługuje
najwięcej żądań, ale przewaga nad upstream jest tam najmniejsza. W
przeliczeniu na koszt wynajęcia (ceny Shadeform z 2026-10-08) najwięcej żądań
max na dolara daje L40S (36 żądań/s za $0,88/h), potem RTX A6000 (21 za $0,57/h)
i A100 (41 za $1,38/h).
