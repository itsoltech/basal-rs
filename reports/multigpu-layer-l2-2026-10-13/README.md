# Etap 05, podział warstw między GPU — sesja L2, 2026-10-13

Stan: raport danych z jednej zakończonej sesji L2. Sesja mierzy regresję
toru domyślnego jednego GPU i opisuje kompromis podziału `none`. To nie jest
akceptacja etapu 05 ani końcowe review. Sesja L3 (tryby kwantyzowane) jest
osobna. Raport nie twierdzi niczego o `q8_0` ani `q4_k`.

## Wynik w skrócie

- **Runner:** `complete: true`, `stopped: null`, kod wyjścia 3 (według
  reguły runnera: brak FAIL i SIGNAL, co najmniej jeden GAP). 124 kontrole:
  PASS 123, FAIL 0, GAP 1, SIGNAL 0. Jednostka L2 wystartowała i się
  zakończyła (114 PASS). Wcześniej było 10 kontroli wstępnych.
- **Regresja toru domyślnego** (A = BASE na GPU 0, B = CAND na GPU 0,
  `none`): w rundzie głównej trzy pola miały B gorsze we wszystkich parach.
  W `basal bench` (ab.py) było to `dec_s`: B niższe w 4 z 4 par, B/A
  0,9977–0,9998. W `basal benchmark` były to `mixed-c16 queue_ms.p99` (B/A
  1,0237 i 1,0117) oraz `mixed-c16 successful_latency_ms.p99` (1,0324 i
  1,0028). Każde z tych poleceń dostało dokładnie jedną rundę potwierdzenia,
  a A i B dostały profile przy batch 1. Po obu rundach każde z trzech pól ma
  parę, w której B nie jest gorsze. `dec_s` ma B gorsze w 6 z 8 par, a oba
  pola `benchmark` w 3 z 4 par. Utrzymane sygnały: 0. Nierozstrzygnięte: 0.
  Brakujące wartości: 0.
- **Wniosek (a):** brak powtarzalnego spowolnienia przy zmierzonej
  rozdzielczości po rundach potwierdzających. To nie jest dowód braku
  narzutu ani ogólny brak regresji. Reguła nie ma progu.
- **Kompromis podziału** (B wobec C = CAND `0:30,1:30`, opisowo, bez
  reguły): przy `mixed-c16` C ma przepustowość C/B 1,1804 i 1,1823 oraz
  p95 latencji 0,7220 i 0,7391. Przy `sequential-c1` i `mixed-c1` C ma
  przepustowość 0,9717–0,9725 B i p50 latencji 1,0162–1,0401 B. W ab.py
  mediana C/B per pytanie wynosi 1,00451–1,00766, a `dec_s` C/B
  0,9426–0,9459. Pula alokatora po ładowaniu: C 4610,7 / 4618,7 MiB (GPU 0 /
  1), B 9197,4 MiB na GPU 0.
- **Warunki:** w przebiegach `benchmark` A i B GPU 0 osiągało 84–85 °C, a
  zegar SM spadał do 1005–1050 MHz przy żądanych 1200 MHz. W przebiegach C
  GPU 0 miało 73–74 °C, a GPU 1 71 °C. Warunki opisują przebiegi. Nie
  filtrują par i nie są przyczyną wyników.

## Zakres i pochodzenie

| Element | Wartość |
|---|---|
| Sprzęt | 2 × NVIDIA RTX 6000 Ada Generation, 49140 MiB każda, sterownik 580.126.09, limit mocy 300 W, tryb persistence |
| Zegar i CPU | żądany SM 1200 MHz (`nvidia-smi -pm 1`, `nvidia-smi -lgc 1200,1200`, oba kod 0), maska 16 z 26 logicznych CPU |
| Biblioteki | cuBLAS/cuBLASLt 12.9.1.4, cudart 12.9.79, tabele GEMM cuBLASLt 120901 |
| Środowisko procesów | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `HF_HUB_OFFLINE=1`, `BASAL_NO_UPDATE_CHECK=1`; katalog JIT CUDA pusty przed sesją (0 plików) |
| Okno kroków pomiaru (UTC) | 11:37:07–12:50:38 |
| BASE | `d8a6a968…`, przebudowane `d42a400`, źródła `94ef2131…` (65 plików) |
| CAND | `dfd13b7e…`, `d42a400` + zmiany etapu 05, źródła `77b629ed…` (66 plików) |
| Build | te same binarki co w L1: ten sam builder i wspólny przepis `42fa4a4e…` (25 kroków), manifest buildu `6ae67566…` (375 plików) |
| Runner / helpery | `tools/perf/run-layer-split-cloud.py` `81437a1c…`, `tools/bench/ab.py` `17b188dc…`; w L2 nie działał żaden proces helpera rezerwacji |
| Model i tabela | basal-1.5-4.5B `784a683b…` (60 warstw), f16, `none`, `fused`; tabela `nvidia-rtx-6000-ada-generation--basal-1.5-4.5B--f16--cublaslt120901.json` `f3712077…` |
| Wejścia | `basal bench`: `reports/reference-basal-1.5-4.5B-fp32` (44 pytania, pomiar latencji na pytaniach 5..43); `basal benchmark`: wbudowane obciążenie `a645db4c…` (900 żądań sekwencyjnych i 900 mieszanych) |
| Surowe dane | manifest SHA-256 `fb0dd50f5d9289d725b072fca87eaaddadb8ce82aa0391447a18699cf97d65ca`: 105 plików, 8 katalogów, 14 311 073 B; zebrane i zweryfikowane bez błędów |
| Audyt ROOT | `07cc9a37…` (wersja r2): 350 kontroli bez uwag |

Pełne sumy znajdują się w `provenance.json`. Etykieta źródła BASE jest
deklaracją z rekordu buildu. Runner jej nie weryfikuje, więc pozostaje GAP
(niżej). Raport nie twierdzi, że binarki są odtwarzalne bit w bit. Nie
twierdzi też, że wykonywalny plik z sesji jest tym samym plikiem co w innych
sesjach. Wiąże go tylko suma SHA-256 sprawdzona przez runner.

## Protokół

Role: A = BASE, GPU 0; B = CAND, GPU 0; C = CAND, `--layer-split 0:30,1:30`.
Wszystkie role używają `none`, f16 i tej samej tabeli GEMM.

- `basal benchmark --scenario all --concurrency 1,16`: sześć świeżych
  procesów w kolejności A B C C B A. Pary regresji są stałe: A0/B1 i B4/A5.
  Role A/B są znormalizowane, więc B/A to zawsze CAND przez BASE. Pary
  kompromisu B1/C2 i C3/B4 są opisowe.
- `tools/bench/ab.py --rounds 4` z `basal bench --dtype f16`: A/B oraz B/C,
  kolejność `abbaabba`. Para k to `a-k` i `b-k`.
- 21 pól `benchmark`: 3 fazy (`sequential-c1`, `mixed-c1`, `mixed-c16`) × 7
  metryk. Latencja p50/p95/p99 i `queue_ms` p50/p99 są gorsze, gdy rosną (↑).
  `successful_requests_per_second` i `successful_questions_per_second` są
  gorsze, gdy maleją (↓).
- 4 pola ab.py: `lat2_ms` i `lat2_p95_ms` (↑), `dec_s` (↓) oraz mediana B/A
  per pytanie z `lat2_ms` 39 pytań 5..43 (gorzej > 1).
- Reguła W8/W4 bez progu. Sygnał to pole z B gorszym we wszystkich
  zaplanowanych parach (benchmark 2, ab.py 4). Sygnał uruchamia dokładnie
  jedną rundę potwierdzenia tego polecenia: benchmark A B B A albo ab.py
  rounds 4. Uruchamia też `basal profile` A i B z `--n 40 --batch 1`.
  Utrzymany SIGNAL wymaga B gorszego we wszystkich parach obu rund. Brak albo
  nieskończona wartość jest problemem dowodu i daje brak wniosku.
- Wcześniejsza uwaga o normalności różnic 1–2% nie jest regułą ani
  wyjątkiem. Raport żadnej różnicy nie pomija.

## Regresja toru domyślnego (A/B)

### ab.py (`basal bench`)

W każdej z 8 par A i B mają tę samą dokładność (0,7955, 35 z 44) i te same
pytania. Każda para ma 39 unikalnych indeksów 5..43 ze skończonym `lat2_ms`.

| Runda | Para | `lat2_ms` A / B | `lat2_p95_ms` A / B | `dec_s` A / B | mediana B/A per pytanie | B szybsze (z 39) | Pola z B gorszym |
|---|---:|---|---|---|---:|---:|---|
| główna | 0 | 25,524 / 25,724 | 32,849 / 32,636 | 64,28 / 64,22 | 1,00398 | 4 | `lat2_ms`, `dec_s`, `per_question_median_ratio` |
| główna | 1 | 25,626 / 25,576 | 32,525 / 32,500 | 64,40 / 64,25 | 0,99989 | 21 | `dec_s` |
| główna | 2 | 25,551 / 25,639 | 32,502 / 32,681 | 64,38 / 64,36 | 1,00024 | 15 | `lat2_ms`, `lat2_p95_ms`, `dec_s`, `per_question_median_ratio` |
| główna | 3 | 25,578 / 25,609 | 32,514 / 32,578 | 64,28 / 64,27 | 1,00048 | 17 | `lat2_ms`, `lat2_p95_ms`, `dec_s`, `per_question_median_ratio` |
| potwierdzenie | 0 | 25,621 / 25,606 | 32,611 / 32,483 | 62,84 / 64,00 | 0,99940 | 32 | — |
| potwierdzenie | 1 | 25,600 / 25,660 | 32,566 / 32,544 | 64,33 / 64,32 | 1,00101 | 10 | `lat2_ms`, `dec_s`, `per_question_median_ratio` |
| potwierdzenie | 2 | 25,617 / 25,585 | 32,682 / 32,484 | 64,38 / 64,36 | 0,99998 | 20 | `dec_s` |
| potwierdzenie | 3 | 25,598 / 25,624 | 32,598 / 32,517 | 64,31 / 64,37 | 1,00056 | 14 | `lat2_ms`, `per_question_median_ratio` |

Agregat ab.py (mediana `lat2_ms` pytania z 4 rund, potem B/A) wynosi
1,000822 w rundzie głównej (min. 0,998878, maks. 1,006166, B szybsze w 8 z 39
pytań). W potwierdzeniu wynosi 1,000308 (0,993805–1,009967, B szybsze w 17
z 39). Wartości policzono ponownie z `per_item` i są równe `ab.json`.

### `basal benchmark`, wszystkie 21 pól

Komórka to B/A. **g** oznacza B gorsze w kierunku pola. Potwierdzenie ma
kolejność A B B A i pary (przebieg 0 A, 1 B) oraz (3 A, 2 B).

| Pole (gorzej) | A0/B1 | B4/A5 | potw. A0/B1 | potw. B2/A3 |
|---|---:|---:|---:|---:|
| `sequential-c1 successful_latency_ms.p50` (↑) | 0,9988 | 0,9915 | 1,0046 **g** | 1,0075 **g** |
| `sequential-c1 successful_latency_ms.p95` (↑) | 1,0016 **g** | 0,9971 | 0,9991 | 1,0021 **g** |
| `sequential-c1 successful_latency_ms.p99` (↑) | 0,9971 | 0,9933 | 1,0018 **g** | 0,9970 |
| `sequential-c1 queue_ms.p50` (↑) | 0,9538 | 0,9786 | 1,0349 **g** | 1,2649 **g** |
| `sequential-c1 queue_ms.p99` (↑) | 0,9737 | 1,1069 **g** | 0,9762 | 0,9696 |
| `sequential-c1 successful_requests_per_second` (↓) | 0,9996 **g** | 1,0064 | 0,9980 **g** | 0,9956 **g** |
| `sequential-c1 successful_questions_per_second` (↓) | 0,9996 **g** | 1,0064 | 0,9980 **g** | 0,9956 **g** |
| `mixed-c1 successful_latency_ms.p50` (↑) | 1,0051 **g** | 0,9924 | 1,0074 **g** | 0,9973 |
| `mixed-c1 successful_latency_ms.p95` (↑) | 1,0007 **g** | 0,9996 | 1,0010 **g** | 1,0006 **g** |
| `mixed-c1 successful_latency_ms.p99` (↑) | 0,9068 | 0,9994 | 0,9920 | 0,9990 |
| `mixed-c1 queue_ms.p50` (↑) | 0,8931 | 1,0138 **g** | 0,9937 | 0,9795 |
| `mixed-c1 queue_ms.p99` (↑) | 1,0776 **g** | 0,9604 | 0,9708 | 0,9263 |
| `mixed-c1 successful_requests_per_second` (↓) | 1,0319 | 1,0008 | 0,9988 **g** | 1,0009 |
| `mixed-c1 successful_questions_per_second` (↓) | 1,0319 | 1,0008 | 0,9988 **g** | 1,0009 |
| `mixed-c16 successful_latency_ms.p50` (↑) | 0,9620 | 0,9658 | 1,0127 **g** | 1,0605 **g** |
| `mixed-c16 successful_latency_ms.p95` (↑) | 1,0185 **g** | 0,9967 | 1,0031 **g** | 0,9998 |
| `mixed-c16 successful_latency_ms.p99` (↑) | 1,0324 **g** | 1,0028 **g** | 1,0120 **g** | 0,9989 |
| `mixed-c16 queue_ms.p50` (↑) | 0,7820 | 1,0129 **g** | 1,0126 **g** | 1,2030 **g** |
| `mixed-c16 queue_ms.p99` (↑) | 1,0237 **g** | 1,0117 **g** | 1,0171 **g** | 0,9798 |
| `mixed-c16 successful_requests_per_second` (↓) | 1,0014 | 0,9997 **g** | 0,9903 **g** | 1,0053 |
| `mixed-c16 successful_questions_per_second` (↓) | 1,0014 | 0,9997 **g** | 0,9903 **g** | 1,0053 |

Wszystkie 10 przebiegów `benchmark` ma w każdej fazie 0 błędów, `complete`,
900 zaplanowanych i ukończonych żądań oraz 0 zmian odpowiedzi. Każdy
przebieg używa tej samej wersji wbudowanego obciążenia. W potwierdzeniu
siedem pól ma B gorsze w obu parach, choć nie były sygnałami rundy głównej:
`sequential-c1` latencja p50, `queue_ms.p50` i obie przepustowości,
`mixed-c1` latencja p95 oraz `mixed-c16` latencja p50 i `queue_ms.p50`.
Każde z nich ma w rundzie głównej parę z B niegorszym. Reguła ocenia w
potwierdzeniu tylko sygnały rundy głównej i nie uruchamia kolejnej rundy.
Wartości tych pól są w tabeli i w `facts.json`.

### Sygnały przez obie rundy

| Polecenie | Pole | Runda | Para | A | B | B/A | B gorsze |
|---|---|---|---|---:|---:|---:|---|
| bench | `dec_s` | główna | 0 | 64,285 | 64,217 | 0,9989 | tak |
| bench | `dec_s` | główna | 1 | 64,401 | 64,252 | 0,9977 | tak |
| bench | `dec_s` | główna | 2 | 64,382 | 64,356 | 0,9996 | tak |
| bench | `dec_s` | główna | 3 | 64,281 | 64,267 | 0,9998 | tak |
| bench | `dec_s` | potwierdzenie | 0 | 62,844 | 64,005 | 1,0185 | nie |
| bench | `dec_s` | potwierdzenie | 1 | 64,334 | 64,323 | 0,9998 | tak |
| bench | `dec_s` | potwierdzenie | 2 | 64,376 | 64,362 | 0,9998 | tak |
| bench | `dec_s` | potwierdzenie | 3 | 64,314 | 64,371 | 1,0009 | nie |
| benchmark | `mixed-c16 queue_ms.p99` | główna | A0/B1 | 39918,26 | 40864,59 | 1,0237 | tak |
| benchmark | `mixed-c16 queue_ms.p99` | główna | B4/A5 | 39998,57 | 40466,17 | 1,0117 | tak |
| benchmark | `mixed-c16 queue_ms.p99` | potwierdzenie | A0/B1 | 39786,18 | 40464,82 | 1,0171 | tak |
| benchmark | `mixed-c16 queue_ms.p99` | potwierdzenie | B2/A3 | 40706,64 | 39884,13 | 0,9798 | nie |
| benchmark | `mixed-c16 successful_latency_ms.p99` | główna | A0/B1 | 44711,00 | 46159,93 | 1,0324 | tak |
| benchmark | `mixed-c16 successful_latency_ms.p99` | główna | B4/A5 | 45126,23 | 45254,31 | 1,0028 | tak |
| benchmark | `mixed-c16 successful_latency_ms.p99` | potwierdzenie | A0/B1 | 44841,09 | 45381,38 | 1,0120 | tak |
| benchmark | `mixed-c16 successful_latency_ms.p99` | potwierdzenie | B2/A3 | 45285,41 | 45234,95 | 0,9989 | nie |

Każdy sygnał ma w potwierdzeniu parę z B niegorszym, więc żaden nie jest
utrzymany. Większość par ma jednak B gorsze (6 z 8 oraz 3 z 4). Pomiar nie
wyklucza więc małego narzutu w tych polach. Rozdzielczość tej sesji go nie
pokazuje. W `dec_s` różnice głównej rundy to −0,02% do −0,23%. W pierwszej
parze potwierdzenia A miało 62,844 dec/s, a pozostałe przebiegi A i B
64,0–64,4. Raport nie ustala przyczyny tej wartości.

### Profile A/B, batch 1 (diagnostyka)

`basal profile --n 40 --batch 1`, średnio 239,8 tokenu na element. Profil
synchronizuje sekcje, więc te czasy nie są latencją żądania ani dowodem
szybkości. `residual_abs_max` wynosi 0,0 w obu profilach.

| Sekcja | A ms/element | B ms/element |
|---|---:|---:|
| `attention` | 6,643 | 6,628 |
| `bias+rope+heads` | 0,926 | 0,911 |
| `bias+silu*up` | 1,154 | 1,139 |
| `down_proj+bias+residual` | 10,069 | 10,054 |
| `embed+inputs` | 0,494 | 0,487 |
| `final_norm` | 0,054 | 0,052 |
| `gate_up_proj` | 10,004 | 9,999 |
| `o_proj+bias+residual+rms_norm` | 2,376 | 2,378 |
| `qkv_proj` | 2,332 | 2,328 |
| `rms_norm` | 0,907 | 0,919 |
| suma sekcji | 34,959 | 34,896 |
| `ms_per_item_profiled` | 35,041 | 34,979 |

## Kompromis podziału (B/C, opisowo)

C ma te same obliczenia na dwóch GPU, etap po etapie. Tabele podają
wartości, C−B jest w `facts.json`. Pary nie mają reguły ani werdyktu.

### `basal benchmark`, wszystkie 21 pól

| Pole | B1 | C2 | C/B | B4 | C3 | C/B |
|---|---:|---:|---:|---:|---:|---:|
| `sequential-c1 successful_latency_ms.p50` | 26,90 | 27,86 | 1,0358 | 26,79 | 27,86 | 1,0401 |
| `sequential-c1 successful_latency_ms.p95` | 40,48 | 41,44 | 1,0238 | 40,39 | 41,34 | 1,0235 |
| `sequential-c1 successful_latency_ms.p99` | 48,89 | 50,11 | 1,0249 | 48,70 | 49,80 | 1,0226 |
| `sequential-c1 queue_ms.p50` | 0,4125 | 0,4860 | 1,1782 | 0,5020 | 0,4970 | 0,9900 |
| `sequential-c1 queue_ms.p99` | 1,0740 | 1,2500 | 1,1639 | 1,2740 | 1,1330 | 0,8893 |
| `sequential-c1 successful_requests_per_second` | 31,99 | 31,08 | 0,9717 | 32,03 | 31,13 | 0,9720 |
| `sequential-c1 successful_questions_per_second` | 31,99 | 31,08 | 0,9717 | 32,03 | 31,13 | 0,9720 |
| `mixed-c1 successful_latency_ms.p50` | 46,71 | 47,46 | 1,0162 | 46,43 | 47,47 | 1,0226 |
| `mixed-c1 successful_latency_ms.p95` | 1106,10 | 1141,57 | 1,0321 | 1105,72 | 1141,86 | 1,0327 |
| `mixed-c1 successful_latency_ms.p99` | 3065,27 | 3158,29 | 1,0303 | 3056,88 | 3152,10 | 1,0311 |
| `mixed-c1 queue_ms.p50` | 0,8605 | 0,9710 | 1,1284 | 0,9925 | 0,9770 | 0,9844 |
| `mixed-c1 queue_ms.p99` | 100,55 | 114,99 | 1,1436 | 96,22 | 91,16 | 0,9474 |
| `mixed-c1 successful_requests_per_second` | 4,8744 | 4,7390 | 0,9722 | 4,8790 | 4,7447 | 0,9725 |
| `mixed-c1 successful_questions_per_second` | 11,20 | 10,89 | 0,9722 | 11,21 | 10,90 | 0,9725 |
| `mixed-c16 successful_latency_ms.p50` | 607,55 | 558,02 | 0,9185 | 622,74 | 528,75 | 0,8491 |
| `mixed-c16 successful_latency_ms.p95` | 17453,36 | 12601,70 | 0,7220 | 17291,30 | 12779,21 | 0,7391 |
| `mixed-c16 successful_latency_ms.p99` | 46159,93 | 39829,91 | 0,8629 | 45254,31 | 39790,65 | 0,8793 |
| `mixed-c16 queue_ms.p50` | 218,68 | 220,51 | 1,0083 | 272,23 | 196,25 | 0,7209 |
| `mixed-c16 queue_ms.p99` | 40864,59 | 36195,43 | 0,8857 | 40466,17 | 36368,72 | 0,8987 |
| `mixed-c16 successful_requests_per_second` | 5,6630 | 6,6844 | 1,1804 | 5,6527 | 6,6835 | 1,1823 |
| `mixed-c16 successful_questions_per_second` | 13,01 | 15,36 | 1,1804 | 12,99 | 15,36 | 1,1823 |

Przy jednym kliencie C jest wolniejsze: latencja p50 rośnie o 1,6–4,0%, a
przepustowość spada o 2,75–2,83%. Przy 16 klientach mieszanego ruchu C ma o
18,0–18,2% wyższą przepustowość i niższą latencję p50/p95/p99. Pomiar nie
ustala, który mechanizm daje tę różnicę. Ta sama sesja pokazuje też, że GPU 0
pracowało w C chłodniej i z wyższym minimalnym zegarem SM (niżej).

### ab.py (`basal bench`)

| Para | `lat2_ms` B / C | `lat2_p95_ms` B / C | `dec_s` B / C | mediana C/B per pytanie | C szybsze (z 39) |
|---:|---|---|---|---:|---:|
| 0 | 25,544 / 25,664 | 32,496 / 32,776 | 64,32 / 60,68 | 1,00766 | 1 |
| 1 | 25,632 / 25,704 | 32,567 / 32,711 | 64,17 / 60,70 | 1,00506 | 1 |
| 2 | 25,638 / 25,636 | 32,583 / 32,698 | 64,25 / 60,56 | 1,00451 | 4 |
| 3 | 25,588 / 25,737 | 32,593 / 32,810 | 64,26 / 60,62 | 1,00616 | 1 |

Agregat ab.py C/B wynosi 1,006060 (1,000065–1,008733), a C jest szybsze w 0
z 39 pytań. B i C mają tę samą dokładność i te same pytania. Sonda GEMM
(`gpu_probe_tflops`, przed i po przebiegu) dała w przebiegach jednego GPU
103,1–104,4 TFLOPS. W przebiegach C dała 121,0–121,1 TFLOPS na GPU 0 i
104,4–104,7 TFLOPS na GPU 1. To diagnostyka warunków, a nie pomiar szybkości
modelu. Raport nie wyjaśnia tej różnicy. `basal bench` nie raportuje pul
ani transferów.

### Pamięć i `load_s`

| Przebieg | Rola | `load_s` | pula: po ładowaniu GPU 0 / 1 | pula: szczyt GPU 0 / 1 | min. wolna (próbki) GPU 0 / 1 | NVML `memory.used` maks. GPU 0 / 1 |
|---|---|---:|---|---|---|---|
| L2 0 | A | 12,828 | 9197,4 / — | 11951,1 / — | 35862 / — | 12648 / 4 |
| L2 1 | B | 12,930 | 9197,4 / — | 11686,7 / — | 36150 / — | 12360 / 4 |
| L2 2 | C | 11,794 | 4610,7 / 4618,7 | 7161,1 / 7227,6 | 40758 / 40662 | 7752 / 7848 |
| L2 3 | C | 11,725 | 4610,7 / 4618,7 | 7019,4 / 7227,6 | 40886 / 40662 | 7592 / 7848 |
| L2 4 | B | 12,051 | 9197,4 / — | 11686,7 / — | 36086 / — | 12424 / 4 |
| L2 5 | A | 12,053 | 9197,4 / — | 11679,9 / — | 36086 / — | 12424 / 4 |
| potw. 0 | A | 11,788 | 9197,4 / — | 11951,1 / — | 35926 / — | 12584 / 4 |
| potw. 1 | B | 11,886 | 9197,4 / — | 11951,1 / — | 35862 / — | 12648 / 4 |
| potw. 2 | B | 11,964 | 9197,4 / — | 11951,1 / — | 35926 / — | 12584 / 4 |
| potw. 3 | A | 12,203 | 9197,4 / — | 11951,1 / — | 35798 / — | 12712 / 4 |

Wszystkie wartości pamięci są w MiB. Źródła mają różne zakresy i ich nie
mieszamy:

- **Pula** to pula alokatora strumieniowego CUDA tego procesu
  (`cuMemAllocAsync`, `used_bytes`). Obejmuje tensory, przestrzenie robocze
  cuBLASLt i prefiksy. Nie obejmuje kontekstu CUDA, modułów ani uchwytów
  bibliotek. Górne `memory_pool` raportu to pierwsze urządzenie (w C etap 0,
  równe `layer_split.devices[0]`). Pule obu etapów C pochodzą z
  `layer_split.devices`. Po zamknięciu silnika pula wynosi 0 we wszystkich
  10 przebiegach.
- **Min. wolna** to najmniejsza wolna pamięć urządzenia
  (`resources.min_available_gpu_bytes_sampled`, próbki co 200 ms). Obejmuje
  wszystkie procesy na karcie. W C pochodzi z `resources.per_device`.
- **NVML** to `memory.used` z telemetrii przebiegu (MiB, co 200 ms, poziom
  urządzenia). GPU 1 w przebiegach A i B ma 4 MiB, bo nie jest używane.
- Szacunek wag z arytmetyki formatu (nie pomiar): 9 514 629 496 B dla
  jednego GPU, 4 757 312 700 B i 4 765 724 653 B dla etapów C.

`load_s` ma w BASE i CAND te same granice pomiaru. Obejmuje otwarcie
urządzeń, mapowanie, wgranie wszystkich etapów, synchronizację i tablice
RoPE. Nie obejmuje preflightu, tabeli GEMM, głowicy evidence ani uchwytu
cuBLASLt. W 24 procesach ab.py `load_s` wynosiło 11,94–13,59 s. Raport
nie ustala stanu pamięci podręcznej plików systemu między procesami. Nie
wyciąga więc z `load_s` wniosku o szybkości ładowania.

### Transfery między etapami (przebiegi C `benchmark`)

| Przebieg | Kierunek | Kopie | Bajty | `transfer_ms` | `gate_wait_ms` |
|---|---|---:|---:|---:|---:|
| L2 2 C | etap 0 → 1 (GPU 0 → 1) | 2418 | 22 975 938 560 | 7067,987 | 15435,088 |
| L2 3 C | etap 0 → 1 (GPU 0 → 1) | 2413 | 22 915 358 720 | 7128,675 | 15288,876 |

`transfer_ms` to suma czasu hosta udanych transferów granicy. Obejmuje
alokację i inicjalizację celu, opróżnienie, kopię API urządzenie–urządzenie,
potwierdzenie na obu GPU, zwolnienie źródła i wiązanie kontekstu. Liczniki
obejmują wszystkie fazy obciążenia z żądaniami rozgrzewającymi, aż do
zamknięcia, bez ładowania modelu. Fizyczna trasa kopii (P2P albo przez
hosta) nie jest ustalona.

## Warunki pomiaru (opisowo)

Telemetria `nvidia-smi` co 200 ms, osobny plik każdego przebiegu
`benchmark`, obie karty.

| Przebieg | Rola | GPU 0: temp. maks. °C / SM min. MHz / moc maks. W | GPU 1: temp. maks. °C / SM min. MHz / moc maks. W |
|---|---|---|---|
| L2 0 | A | 85 / 1050 / 295,6 | 47 / 1200 / 41,6 |
| L2 1 | B | 85 / 1020 / 293,8 | 44 / 1200 / 41,3 |
| L2 2 | C | 74 / 1140 / 251,8 | 71 / 1155 / 252,4 |
| L2 3 | C | 73 / 1140 / 249,9 | 71 / 1125 / 251,4 |
| L2 4 | B | 84 / 1020 / 293,4 | 61 / 1200 / 118,2 |
| L2 5 | A | 85 / 1005 / 295,4 | 44 / 1200 / 41,5 |
| potw. 0 | A | 84 / 1005 / 293,4 | 44 / 1200 / 41,1 |
| potw. 1 | B | 85 / 1020 / 294,4 | 44 / 1200 / 41,1 |
| potw. 2 | B | 84 / 1005 / 293,4 | 44 / 1200 / 41,2 |
| potw. 3 | A | 84 / 1020 / 295,6 | 44 / 1200 / 41,4 |

- Zegar SM był ustawiony na 1200 MHz. W przebiegach A i B `benchmark`
  spadał do 1005–1050 MHz przy limicie mocy 300 W i 84–85 °C. A i B
  pracowały w podobnych warunkach.
- W rundach ab.py GPU 0 miało do 68 °C (główna A/B i B/C) i 78 °C
  (potwierdzenie). Minimalny zegar SM wynosił 1200 MHz w rundach głównych i
  1125 MHz w potwierdzeniu.
- W żadnym kroku nie zauważono obcego procesu obliczeniowego ani braku
  próbek. Warunki niczego nie filtrują, a raport nie przypisuje im przyczyny
  wyniku.

## GAP (1, bez zmiany nazwy)

| Kontrola | Treść runnera |
|---|---|
| „base binary: declared source label” | deklaracja z rekordu buildu (`d42a400`, źródła `94ef2131`, 65 plików); runner nie weryfikuje źródła BASE, robi to audyt buildu |

Ten GAP pozostaje GAP-em. Późniejszy wynik go nie usuwa.

## Związek z L1

L1 (`reports/multigpu-layer-l1-2026-10-13`) objęło te same binarki i
źródła, ale inne obciążenia i flagi: równość bitową, odmowy, HTTP, pojemność
i profile transferów. Wynik L1: PASS 599, FAIL 0, GAP 42, SIGNAL 0, C5 PASS,
surowy manifest `2fec9dc6…`. L2 nie mierzy ponownie zgodności, jakości ani
pojemności. Nie przenosi też wniosków L1 na wydajność.

## Ograniczenia

- Jedna sesja, jedna maszyna, dwie identyczne karty RTX 6000 Ada 48 GB,
  model 4.5B, `none`. Nie wnioskujemy o innych kartach, modelach, mieszanych
  parach, Metal ani Intel.
- Wniosek (a) dotyczy rozdzielczości tej sesji. Brak utrzymanego sygnału nie
  dowodzi braku narzutu toru jednego GPU.
- Pary B/C są opisowe. Nie są twierdzeniem o przyspieszeniu podziału w innym
  ruchu, innej liczbie klientów ani na innym sprzęcie.
- `basal benchmark` ma ruch zamknięty: klient czeka na odpowiedź. Klient i
  serwer działają w jednym procesie. Pamięć jest próbkowana, a nie mierzona
  jako prawdziwy szczyt alokacji.
- Profile są diagnostyką sekcji, a nie latencją żądania.
- Brak twierdzeń o `q8_0` i `q4_k` (L3) oraz o fizycznej trasie transferu.

## Polecenia (ilustracyjne)

Rzeczywiste argv, ścieżki i środowisko zostają w prywatnych surowych
danych. `<base>`, `<candidate>`, `<models>`, `<out>` i `<table>` to symbole
zastępcze.

```sh
# benchmark: A (BASE, GPU 0), B (CAND, GPU 0), C (CAND, podział)
<base> --color never benchmark --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --device 0 --scenario all --concurrency 1,16 --out <out>/0-A.json
<candidate> --color never benchmark --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --layer-split 0:30,1:30 --scenario all --concurrency 1,16 --out <out>/2-C.json

# ab.py: 4 rundy abbaabba (A/B; B/C analogicznie z --layer-split 0:30,1:30 po stronie --b)
python3 tools/bench/ab.py --rounds 4 --out <out>/l2-bench-ab \
  --a "<base> --color never bench --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16" \
  --b "<candidate> --color never bench --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16"

# profil (diagnostyka potwierdzenia)
<candidate> --color never profile --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --n 40 --batch 1
```

## Pliki

- `facts.json` — statusy i liczności, GAP, protokół, wszystkie pary A/B
  (ab.py: 4 pola i statystyki per pytanie; `benchmark`: 21 pól z A, B, B−A,
  B/A i oceną kierunku) dla rundy głównej i potwierdzenia, sygnały, wniosek,
  profile, pary B/C (25 pól z C−B i C/B), przebiegi `benchmark` (pule,
  wolna pamięć, NVML, `load_s`, transfery, telemetria per GPU), warunki
  runnera i kroki. Liczby mają pełne bity, a zaokrąglenia występują tylko w
  README.
- `provenance.json` — sumy binarek, źródeł, runnera i helperów, modelu,
  tabeli GEMM, wejść i obciążenia, środowisko (karta, sterownik, biblioteki,
  zegar), manifest surowych danych, audyt ROOT, sumy plików L1.
- `validation.json` — kontrole autora: fakty wobec surowych danych i wartości
  runnera, sumy surowych plików, bieżące źródła, narzędzia i wejścia, tabele
  README odtworzone z `facts.json`, brak danych identyfikujących maszynę i
  danych finansowych.
