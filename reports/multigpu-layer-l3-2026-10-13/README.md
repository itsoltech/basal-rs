# Etap 05, podział warstw między GPU — sesja L3, 2026-10-13

> **Dane JSON w archiwum.** Pliki `facts.json`, `provenance.json` i `validation.json` z tego katalogu są poza
> Gitem, w archiwum pomiarów `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod ścieżką
> `reports/multigpu-layer-l3-2026-10-13/<plik>`. Odwołania do nich w tekście dotyczą plików z archiwum. Plik
> `reports/w4a16-p0-2026-10-12/facts.json` także jest w tym archiwum. Pobieranie i sumy kontrolne:
> [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Archiwum zawiera też wersję tego pliku
> sprzed dopisania tej noty. Przeniesienie danych nie jest nowym pomiarem: liczby, statusy i rozstrzygnięcia
> raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

Stan: raport danych z jednej zakończonej sesji L3 (tryby kwantyzowane
`q8_0` i `q4_k`). Sesja mierzy regresję toru jednego GPU w tych trybach
parami ab.py i opisuje kompromis podziału. To nie jest akceptacja etapu 05
ani końcowe review. Raport nie mierzy jakości ani zgodności.

## Wynik w skrócie

- **Runner:** `complete: true`, `stopped: null`, kod wyjścia 3 (według
  reguły runnera: brak FAIL i SIGNAL, co najmniej jeden GAP). 156 kontroli:
  PASS 154, FAIL 0, GAP 2, SIGNAL 0. Jednostka L3 wystartowała i się
  zakończyła (145 PASS, 1 GAP). Wcześniej było 10 kontroli wstępnych.
- **`q8_0`, A/B** (A = BASE na GPU 0, B = CAND na GPU 0): żadne z 4 pól
  ab.py nie ma B gorszego we wszystkich 4 parach. Sygnały główne: 0, pola
  otwarte: 0, więc runda potwierdzenia nie była potrzebna. `dec_s` ma B
  gorsze w 3 z 4 par (B/A 0,9976–1,0004). Wniosek (a).
- **`q4_k`, A/B:** sygnał główny `dec_s`: B niższe w 4 z 4 par, B/A
  0,99837–0,99992 (−0,01% do −0,16%). Dostał dokładnie jedną rundę
  potwierdzenia ab.py (4 pary) oraz profile A i B przy batch 1. W
  potwierdzeniu B jest gorsze w 2 z 4 par. Przez obie rundy B jest gorsze w
  6 z 8 par, czyli w większości, ale nie we wszystkich. Utrzymane sygnały: 0.
  Nierozstrzygnięte: 0. Brakujące wartości: 0. Wniosek (a).
- **Wniosek (a)** dla obu trybów: brak powtarzalnego spowolnienia przy
  zmierzonej rozdzielczości. Dotyczy tylko par ab.py (`basal bench`). To
  nie jest dowód braku narzutu, ogólny brak regresji ani twierdzenie o HTTP.
  Reguła nie ma progu. HTTP CAND na jednym GPU w `q8_0`/`q4_k` nie był
  mierzony (GAP niżej).
- **Poza regułą:** w `q4_k` `load_s` B było dłuższe w 7 z 8 par, o
  1,60–2,82 s. Profile `q4_k` (diagnostyka) dały A 187,566 i B 145,013
  ms/element. Raport nie ustala przyczyny żadnej z tych różnic.
- **Kompromis podziału** (B = CAND na GPU 0, C = CAND `0:30,1:30`,
  opisowo, bez reguły): w ab.py C ma `lat2_ms` C/B 1,137–1,178 (`q8_0`) i
  1,189–1,208 (`q4_k`), a `dec_s` C/B 0,912–0,924 i 0,915–0,919. C jest
  szybsze w 0 z 39 pytań w obu trybach.
- **HTTP z podziałem** (po jednym `basal benchmark` na tryb, opisowo): każda
  faza ma 0 błędów, 900 zaplanowanych i ukończonych żądań i 0 zmian
  odpowiedzi. Pula alokatora po ładowaniu: `q8_0` 2543,6 / 2551,6 MiB,
  `q4_k` 1441,1 / 1449,1 MiB (GPU 0 / 1).
- **`q4_k`:** jeden obiekt kwantyzacji w 25 ładowaniach. Jego statystyki
  są równe faktom P0 tych samych wag.
- **Warunki:** w rundach ab.py GPU 0 miało 63–65 °C i zegar SM 1200 MHz. W
  przebiegach HTTP GPU 0 miało 69 °C, GPU 1 72–73 °C, a minimalny zegar SM
  1125–1170 MHz. Warunki opisują przebiegi. Nie filtrują par i nie są
  przyczyną wyników.

## Zakres i pochodzenie

| Element | Wartość |
|---|---|
| Sprzęt | 2 × NVIDIA RTX 6000 Ada Generation, 49140 MiB każda, sterownik 580.126.09, limit mocy 300 W, tryb persistence |
| Zegar i CPU | żądany SM 1200 MHz (`nvidia-smi -pm 1`, `nvidia-smi -lgc 1200,1200`, oba kod 0), maska 16 z 26 logicznych CPU |
| Biblioteki | cuBLAS/cuBLASLt 12.9.1.4, cudart 12.9.79, tabele GEMM cuBLASLt 120901 |
| Środowisko procesów | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `HF_HUB_OFFLINE=1`, `BASAL_NO_UPDATE_CHECK=1`; katalog JIT CUDA pusty przed sesją (0 plików) |
| Okno kroków pomiaru (UTC) | 13:09:33–13:47:41 |
| BASE | `d8a6a968…`, przebudowane `d42a400`, źródła `94ef2131…` (65 plików) |
| CAND | `dfd13b7e…`, `d42a400` + zmiany etapu 05, źródła `77b629ed…` (66 plików) |
| Build | te same binarki co w L1 i L2: ten sam builder i wspólny przepis `42fa4a4e…` (25 kroków), manifest buildu `6ae67566…` (375 plików) |
| Runner / helpery | `tools/perf/run-layer-split-cloud.py` `81437a1c…`, `tools/bench/ab.py` `17b188dc…`; w L3 nie działał żaden proces helpera rezerwacji |
| Model i tabela | basal-1.5-4.5B `784a683b…` (60 warstw), f16, `q8_0` i `q4_k`, `fused`; tabela `nvidia-rtx-6000-ada-generation--basal-1.5-4.5B--f16--cublaslt120901.json` `f3712077…` |
| Wejścia | `basal bench`: `reports/reference-basal-1.5-4.5B-fp32` (44 pytania, pomiar latencji na pytaniach 5..43); `basal benchmark`: wbudowane obciążenie `a645db4c…` (900 żądań sekwencyjnych i 900 mieszanych); fakty P0 `reports/w4a16-p0-2026-10-12/facts.json` `44005ecc…` |
| Surowe dane | manifest SHA-256 `138a94874cb8a044408ac39875c39320e6dd2a52171137cc97206e7afc9b6544`: 119 plików, 9 katalogów, 4 798 768 B; zebrane i zweryfikowane bez błędów |
| Audyt ROOT | `8d43c260…`: 259 kontroli bez uwag |

Pełne sumy znajdują się w `provenance.json`. Etykieta źródła BASE jest
deklaracją z rekordu buildu. Runner jej nie weryfikuje, więc pozostaje GAP
(niżej). Raport nie twierdzi, że binarki są odtwarzalne bit w bit. Nie
twierdzi też, że wykonywalny plik z sesji jest tym samym plikiem co w innych
sesjach. Wiąże go tylko suma SHA-256 sprawdzona przez runner.

## Protokół

Role: A = BASE, GPU 0; B = CAND, GPU 0; C = CAND, `--layer-split 0:30,1:30`.
Każda para porównuje ten sam tryb kwantyzacji. Wszystkie role używają f16,
`fused` i tej samej tabeli GEMM. Role sprawdzono w surowych argv kroków.

- `tools/bench/ab.py --rounds 4` z `basal bench --dtype f16 --quantization
  <tryb>`: A/B dla `q8_0` i `q4_k` (reguła), B/C dla obu trybów (opisowo),
  kolejność `abbaabba`. Para k to `a-k` i `b-k`.
- 4 pola ab.py: `lat2_ms` i `lat2_p95_ms` (gorsze, gdy rosną), `dec_s`
  (gorsze, gdy maleje) oraz mediana B/A per pytanie z `lat2_ms` 39 pytań
  5..43 (gorzej > 1).
- Reguła W8/W4 bez progu. Sygnał to pole z B gorszym we wszystkich 4 parach.
  Sygnał uruchamia dokładnie jedną rundę potwierdzenia ab.py rounds 4 oraz
  `basal profile` A i B z `--n 40 --batch 1`. Utrzymany SIGNAL wymaga B
  gorszego we wszystkich parach obu rund. Brak albo nieskończona wartość
  jest problemem dowodu i daje brak wniosku.
- Po jednym `basal benchmark --scenario all --concurrency 1,16` CAND z
  podziałem w `q8_0` i `q4_k`. To jedno wykonanie na tryb, bez pary i bez
  reguły. 21 pól: 3 fazy (`sequential-c1`, `mixed-c1`, `mixed-c16`) × 7
  metryk.
- Plan z góry wyłącza HTTP (`basal benchmark`) nowego kandydata na jednym
  GPU w `q8_0` i `q4_k`. S3 opisuje tylko `d42a400`, więc nie jest dowodem
  dla CAND. Dla tych trybów dowodem wydajności są tylko pary ab.py.
- Wcześniejsza uwaga o normalności różnic 1–2% (i dyskusja o 0,11%) nie
  jest regułą ani wyjątkiem. Raport żadnej różnicy nie pomija.

## Regresja toru jednego GPU (A/B, ab.py)

W każdej z 12 par A i B mają tę samą dokładność (0,7955, 35 z 44) i te same
pytania. To kontrola spójności par, a nie ocena jakości. Każda para ma 39
unikalnych indeksów 5..43 ze skończonym `lat2_ms`. `load_s` nie jest polem
reguły.

| Tryb | Runda | Para | `lat2_ms` A / B | `lat2_p95_ms` A / B | `dec_s` A / B | mediana B/A per pytanie | B szybsze (z 39) | `load_s` A / B | Pola z B gorszym |
|---|---|---:|---|---|---|---:|---:|---|---|
| `q8_0` | główna | 0 | 42,259 / 42,371 | 49,834 / 49,497 | 56,43 / 56,35 | 0,99922 | 25 | 20,32 / 13,05 | `lat2_ms`, `dec_s` |
| `q8_0` | główna | 1 | 42,564 / 42,297 | 49,914 / 49,537 | 56,39 / 56,41 | 0,99881 | 29 | 13,02 / 13,44 | — |
| `q8_0` | główna | 2 | 42,321 / 42,305 | 49,686 / 50,083 | 56,42 / 56,28 | 0,99956 | 21 | 12,78 / 13,14 | `lat2_p95_ms`, `dec_s` |
| `q8_0` | główna | 3 | 42,268 / 42,261 | 49,489 / 49,773 | 56,47 / 56,36 | 1,00025 | 17 | 13,22 / 12,76 | `lat2_p95_ms`, `dec_s`, `per_question_median_ratio` |
| `q4_k` | główna | 0 | 40,916 / 40,890 | 48,716 / 48,359 | 56,53 / 56,49 | 1,00037 | 17 | 26,01 / 27,97 | `dec_s`, `per_question_median_ratio` |
| `q4_k` | główna | 1 | 40,888 / 40,970 | 48,393 / 48,563 | 56,57 / 56,48 | 1,00100 | 11 | 25,41 / 27,65 | `lat2_ms`, `lat2_p95_ms`, `dec_s`, `per_question_median_ratio` |
| `q4_k` | główna | 2 | 40,981 / 40,980 | 48,564 / 48,393 | 56,52 / 56,52 | 1,00029 | 17 | 25,65 / 28,47 | `dec_s`, `per_question_median_ratio` |
| `q4_k` | główna | 3 | 41,071 / 40,900 | 48,500 / 48,463 | 56,58 / 56,49 | 0,99990 | 23 | 28,03 / 27,95 | `dec_s` |
| `q4_k` | potwierdzenie | 0 | 40,961 / 40,953 | 48,534 / 48,943 | 56,33 / 56,54 | 0,99849 | 32 | 25,95 / 27,54 | `lat2_p95_ms` |
| `q4_k` | potwierdzenie | 1 | 41,025 / 40,998 | 48,515 / 48,378 | 56,50 / 56,43 | 0,99891 | 29 | 26,36 / 28,33 | `dec_s` |
| `q4_k` | potwierdzenie | 2 | 40,968 / 41,030 | 48,505 / 48,498 | 56,49 / 56,53 | 0,99986 | 21 | 26,20 / 27,89 | `lat2_ms` |
| `q4_k` | potwierdzenie | 3 | 40,908 / 40,927 | 48,523 / 48,362 | 56,64 / 56,54 | 0,99948 | 23 | 25,76 / 27,81 | `lat2_ms`, `dec_s` |

Liczba par z B gorszym i agregat ab.py (mediana `lat2_ms` pytania z 4 rund,
potem B/A; policzony ponownie z `per_item` i równy `ab.json`):

| Tryb | Runda | `lat2_ms` | `lat2_p95_ms` | `dec_s` | mediana B/A per pytanie | agregat ab.py B/A (min.–maks.) | B szybsze (z 39) |
|---|---|---:|---:|---:|---:|---|---:|
| `q8_0` | główna | 1 z 4 | 2 z 4 | 3 z 4 | 1 z 4 | 0,999433 (0,996868–1,002825) | 24 |
| `q4_k` | główna | 1 z 4 | 1 z 4 | 4 z 4 | 3 z 4 | 1,000469 (0,997606–1,002766) | 12 |
| `q4_k` | potwierdzenie | 2 z 4 | 1 z 4 | 2 z 4 | 0 z 4 | 0,999436 (0,995215–1,002906) | 27 |

### `q4_k`: sygnał `dec_s` przez obie rundy

| Runda | Para | A | B | B−A | B/A | B gorsze |
|---|---:|---:|---:|---:|---:|---|
| główna | 0 | 56,535 | 56,485 | −0,049 | 0,99913 | tak |
| główna | 1 | 56,566 | 56,483 | −0,084 | 0,99852 | tak |
| główna | 2 | 56,523 | 56,519 | −0,005 | 0,99992 | tak |
| główna | 3 | 56,581 | 56,488 | −0,092 | 0,99837 | tak |
| potwierdzenie | 0 | 56,330 | 56,536 | 0,206 | 1,00365 | nie |
| potwierdzenie | 1 | 56,503 | 56,426 | −0,076 | 0,99865 | tak |
| potwierdzenie | 2 | 56,486 | 56,530 | 0,044 | 1,00077 | nie |
| potwierdzenie | 3 | 56,643 | 56,540 | −0,103 | 0,99818 | tak |

W rundzie głównej B ma niższe `dec_s` we wszystkich 4 parach, więc to
sygnał główny. W potwierdzeniu pary 0 i 2 mają B niegorsze, więc sygnał nie
jest utrzymany. Większość par (6 z 8) ma jednak B gorsze. Pomiar nie
wyklucza więc małego narzutu w `dec_s`. Rozdzielczość tej sesji go nie
pokazuje. Mediana B/A per pytanie ma w rundzie głównej B gorsze w 3 z 4
par, ale para 3 ma B niegorsze, więc to nie jest sygnał. W potwierdzeniu
ma B gorsze w 0 z 4 par. Reguła ocenia w potwierdzeniu tylko sygnały rundy
głównej.

### `q4_k`: profile A/B, batch 1 (diagnostyka)

`basal profile --n 40 --batch 1`, średnio 239,8 tokenu na element. Najpierw
A, potem B, każdy w jednym procesie. Profil synchronizuje sekcje, więc te
czasy nie są latencją żądania ani dowodem szybkości. `residual_abs_max`
wynosi 0,0 w obu profilach. `dequant_ms_per_item` mierzy dekwantyzację
`q4_k` → f16 i jest już zawarte w sekcjach projekcji.

| Sekcja | A ms/element | B ms/element |
|---|---:|---:|
| `attention` | 6,779 | 6,745 |
| `bias+rope+heads` | 1,108 | 1,073 |
| `bias+silu*up` | 1,677 | 1,489 |
| `down_proj+bias+residual` | 23,345 | 28,097 |
| `embed+inputs` | 0,581 | 0,711 |
| `final_norm` | 0,276 | 0,011 |
| `gate_up_proj` | 142,331 | 91,884 |
| `o_proj+bias+residual+rms_norm` | 4,004 | 4,490 |
| `qkv_proj` | 4,096 | 4,099 |
| `rms_norm` | 3,261 | 6,312 |
| suma sekcji | 187,458 | 144,912 |
| `ms_per_item_profiled` | 187,566 | 145,013 |
| `dequant_ms_per_item` (zawarte w sekcjach projekcji) | 48,711 | 46,884 |

Profile różnią się o 42,6 ms/element, głównie w `gate_up_proj`. Te same
binarki w tych samych trybach dały w ab.py `lat2_ms` B/A 0,9958–1,0020.
Raport nie ustala przyczyny różnicy profili i nie wyciąga z niej wniosku o
szybkości.

### `load_s` (poza regułą)

W `q4_k` B miało dłuższe `load_s` w 7 z 8 par, o 1,60–2,82 s (A
25,41–28,03 s, B 27,54–28,47 s). W jednej parze B było krótsze o 0,08 s.
W `q8_0` pierwszy proces sesji (A, para 0) ładował się 20,32 s. Pozostałe
przebiegi `q8_0` A/B trwały 12,76–13,44 s. `load_s` obejmuje otwarcie
urządzeń, mapowanie, kwantyzację i wgranie wag, rozgrzewkę, synchronizację i
tablice RoPE. Raport nie rozkłada go na części i nie ustala stanu pamięci
podręcznej plików systemu. Nie ustala też przyczyny różnicy w `q4_k`.

## Kompromis podziału (B/C, ab.py, opisowo)

B = CAND na GPU 0, C = CAND `--layer-split 0:30,1:30`, ten sam tryb. Pary
nie mają reguły ani werdyktu. C−B jest w `facts.json`.

| Tryb | Para | `lat2_ms` B / C | `lat2_p95_ms` B / C | `dec_s` B / C | mediana C/B per pytanie | C szybsze (z 39) | `load_s` B / C |
|---|---:|---|---|---|---:|---:|---|
| `q8_0` | 0 | 42,085 / 49,589 | 49,171 / 56,223 | 56,18 / 51,44 | 1,15098 | 0 | 14,25 / 13,00 |
| `q8_0` | 1 | 42,056 / 48,806 | 49,117 / 56,086 | 56,47 / 51,99 | 1,14316 | 0 | 12,99 / 12,99 |
| `q8_0` | 2 | 42,264 / 48,071 | 49,523 / 55,293 | 56,39 / 52,12 | 1,12123 | 0 | 12,86 / 12,95 |
| `q8_0` | 3 | 42,311 / 49,717 | 49,611 / 56,571 | 56,45 / 51,49 | 1,14048 | 0 | 13,20 / 13,24 |
| `q4_k` | 0 | 40,997 / 49,541 | 48,544 / 57,181 | 56,30 / 51,75 | 1,17943 | 0 | 28,24 / 27,89 |
| `q4_k` | 1 | 40,980 / 48,714 | 48,428 / 56,573 | 56,46 / 51,71 | 1,16655 | 0 | 28,01 / 27,63 |
| `q4_k` | 2 | 40,925 / 48,897 | 48,440 / 56,394 | 56,43 / 51,64 | 1,17415 | 0 | 28,36 / 28,37 |
| `q4_k` | 3 | 41,060 / 49,533 | 48,584 / 57,195 | 56,35 / 51,77 | 1,19857 | 0 | 28,14 / 27,88 |

Agregat ab.py C/B wynosi 1,135932 (1,057552–1,179092) w `q8_0` i 1,179770
(1,113207–1,225807) w `q4_k`. C jest szybsze w 0 z 39 pytań w obu trybach.
B i C mają tę samą dokładność i te same pytania. `basal bench` nie raportuje
pul ani transferów.

### Sonda GEMM (diagnostyka warunków)

`gpu_probe_tflops` mierzy krótki GEMM przed przebiegiem i po nim. Nie jest
pomiarem szybkości modelu.

| Runda | Strona | GPU | sonda przed, TFLOPS | sonda po, TFLOPS |
|---|---|---:|---|---|
| `q8_0` A/B główna | A (BASE) | 0 | 121,1–121,3 | 121,2–121,3 |
| `q8_0` A/B główna | B (CAND) | 0 | 121,1–121,2 | 104,8–121,2 |
| `q8_0` B/C | B (CAND, 1 GPU) | 0 | 121,2–121,3 | 104,8–121,4 |
| `q8_0` B/C | C (CAND, podział) | 0 | 121,2–121,2 | 118,5–121,3 |
| `q8_0` B/C | C (CAND, podział) | 1 | 121,2–121,3 | 121,1–121,2 |
| `q4_k` A/B główna | A (BASE) | 0 | 121,1–121,2 | 104,7–105,0 |
| `q4_k` A/B główna | B (CAND) | 0 | 121,2–121,3 | 120,9–121,3 |
| `q4_k` A/B potwierdzenie | A (BASE) | 0 | 121,1–121,2 | 104,6–104,8 |
| `q4_k` A/B potwierdzenie | B (CAND) | 0 | 121,2–121,3 | 121,3–121,3 |
| `q4_k` B/C | B (CAND, 1 GPU) | 0 | 121,2–121,3 | 121,2–121,3 |
| `q4_k` B/C | C (CAND, podział) | 0 | 121,2–121,3 | 121,2–121,3 |
| `q4_k` B/C | C (CAND, podział) | 1 | 104,0–104,4 | 120,7–121,1 |

Wartości sondy skupiają się wokół 104–105 i 121 TFLOPS; jedna wynosi
118,5 TFLOPS. W
`q4_k` A/B sonda po przebiegu A wynosiła 104,6–105,0 TFLOPS we wszystkich 8
parach, a po B 120,9–121,3 TFLOPS. Raport nie wyjaśnia tych grup i nie
łączy ich z wynikami par.

## HTTP CAND z podziałem (opisowo, po jednym przebiegu)

`basal benchmark --scenario all --concurrency 1,16`, CAND
`--layer-split 0:30,1:30`, osobno `q8_0` i `q4_k`. W każdej z 3 faz obu
przebiegów: 0 błędów, `complete`, 900 zaplanowanych i ukończonych żądań, 0
zmian odpowiedzi. Oba przebiegi używają tej samej wersji wbudowanego
obciążenia. Bez powtórzeń i bez pary z jednym GPU, więc tabela nie jest
porównaniem.

| Pole | C `q8_0` | C `q4_k` |
|---|---:|---:|
| `sequential-c1 successful_latency_ms.p50` | 51,00 | 51,92 |
| `sequential-c1 successful_latency_ms.p95` | 64,46 | 65,85 |
| `sequential-c1 successful_latency_ms.p99` | 72,62 | 74,74 |
| `sequential-c1 queue_ms.p50` | 0,4670 | 0,5065 |
| `sequential-c1 queue_ms.p99` | 1,2000 | 1,2890 |
| `sequential-c1 successful_requests_per_second` | 18,42 | 18,13 |
| `sequential-c1 successful_questions_per_second` | 18,42 | 18,13 |
| `mixed-c1 successful_latency_ms.p50` | 70,87 | 71,31 |
| `mixed-c1 successful_latency_ms.p95` | 1150,10 | 1149,09 |
| `mixed-c1 successful_latency_ms.p99` | 3546,16 | 3253,88 |
| `mixed-c1 queue_ms.p50` | 0,9765 | 0,9590 |
| `mixed-c1 queue_ms.p99` | 92,27 | 99,32 |
| `mixed-c1 successful_requests_per_second` | 4,1522 | 4,2497 |
| `mixed-c1 successful_questions_per_second` | 9,5408 | 9,7648 |
| `mixed-c16 successful_latency_ms.p50` | 586,03 | 599,50 |
| `mixed-c16 successful_latency_ms.p95` | 13578,11 | 13363,48 |
| `mixed-c16 successful_latency_ms.p99` | 41623,01 | 41303,24 |
| `mixed-c16 queue_ms.p50` | 233,20 | 224,92 |
| `mixed-c16 queue_ms.p99` | 38032,72 | 37917,83 |
| `mixed-c16 successful_requests_per_second` | 6,3013 | 6,3186 |
| `mixed-c16 successful_questions_per_second` | 14,48 | 14,52 |

### Pamięć per GPU

| Przebieg | GPU | pula po ładowaniu (used / reserved) | pula szczyt ładowania | pula szczyt pracy | pula po zamknięciu | min. wolna (próbki) | NVML `memory.used` maks. | szacunek wag etapu |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| C `q8_0` | 0 | 2543,6 / 2656,0 | 2630,5 | 5097,7 | 0,0 | 42582 | 5928 | 2469,8 |
| C `q8_0` | 1 | 2551,6 / 2656,0 | 2637,9 | 5160,5 | 0,0 | 42454 | 6056 | 2477,8 |
| C `q4_k` | 0 | 1441,1 / 1536,0 | 1528,0 | 4103,8 | 0,0 | 43638 | 4872 | 1367,3 |
| C `q4_k` | 1 | 1449,1 / 1504,0 | 1535,4 | 4058,0 | 0,0 | 43606 | 4904 | 1375,3 |

Wszystkie wartości są w MiB. Źródła mają różne zakresy i ich nie mieszamy:

- **Pula** to pula alokatora strumieniowego CUDA tego procesu
  (`cuMemAllocAsync`, `used_bytes`, a przy ładowaniu także
  `reserved_bytes`). Obejmuje tensory, rezydentne bloki `q8_0`/`q4_k`,
  tymczasowe wagi f16, przestrzenie robocze cuBLASLt i prefiksy. Nie obejmuje
  kontekstu CUDA, modułów ani uchwytów bibliotek. Wartości per GPU pochodzą z
  `layer_split.devices`. Górne `memory_pool` raportu to etap 0 i jest mu
  równe w obu przebiegach.
- **Min. wolna** to najmniejsza wolna pamięć urządzenia
  (`resources.per_device`, próbki co 200 ms). Obejmuje wszystkie procesy na
  karcie.
- **NVML** to `memory.used` z telemetrii przebiegu (co 200 ms, poziom
  urządzenia).
- **Szacunek wag** wynika z arytmetyki formatu i nie jest pomiarem.

### `load_s` i transfery między etapami

| Przebieg | `load_s` | Kierunek | Kopie | Bajty | `transfer_ms` | `gate_wait_ms` |
|---|---:|---|---:|---:|---:|---:|
| C `q8_0` | 12,926 | etap 0 → 1 (GPU 0 → 1) | 2 421 | 22 978 322 432 | 12849,155 | 14760,146 |
| C `q4_k` | 28,254 | etap 0 → 1 (GPU 0 → 1) | 2 411 | 22 939 860 992 | 14408,801 | 14012,780 |

`transfer_ms` to suma czasu hosta udanych transferów granicy. Obejmuje
alokację i inicjalizację celu, opróżnienie, kopię API urządzenie–urządzenie,
potwierdzenie na obu GPU, zwolnienie źródła i wiązanie kontekstu. Liczniki
obejmują wszystkie fazy obciążenia z żądaniami rozgrzewającymi, aż do
zamknięcia, bez ładowania modelu. Fizyczna trasa kopii (P2P albo przez
hosta) nie jest ustalona.

## Obiekt kwantyzacji `q4_k`

Runner zapisał kanoniczną sumę SHA-256 opisu `q4_k` z każdego ładowania:
24 procesy ab.py (A/B, potwierdzenie, B/C) i przebieg HTTP. Wszystkie 25 sum
jest równych, czyli jest to jeden obiekt. Ten raport policzył te sumy
ponownie z surowych plików. Statystyki modelu i 4 typów projekcji (wiersze,
wartości, `sum_e2`, `sum_w2`, `max_abs_error` i R jako bity f64 oraz liczniki
grup) są w każdym ładowaniu równe faktom P0 tych samych wag (R
0,005808291323302594, 9 spłaszczonych grup ze 144 506 880). Każde ładowanie
ma też bitowo równą kontrolę rozgrzewki macierzy `o` warstwy 0 (4 194 304
wartości). To własność wczytanych wag, a nie pomiar jakości.

## Jakość i zgodność (kontekst, bez nowego pomiaru)

L3 nie mierzy jakości, zbioru 900 pytań ani porównania z FP32 upstream.
Decyzje jakościowe pozostają bez zmian:
[Q8](../../.itsol/initiatives/basal-multigpu/phases/03-w8a16/quality-decision.md)
jest opcjonalnym trybem z jawnymi różnicami jakości,
[Q4](../../.itsol/initiatives/basal-multigpu/phases/04-w4a16/quality-decision.md)
opcjonalnym trybem eksperymentalnym z jawnymi różnicami, a domyślny tryb to
`none`. Zgodność CAND z BASE w tym samym trybie pokazało
[L1](../multigpu-layer-l1-2026-10-13/README.md): pary jednego GPU i podziału
w `q8_0` i `q4_k` są bajtowo równe BASE. L3 tej zgodności nie mierzy
ponownie.

## Warunki pomiaru (opisowo)

Telemetria `nvidia-smi` co 200 ms, osobny plik każdego kroku ab.py i
każdego przebiegu HTTP, obie karty.

| Krok | GPU 0: temp. maks. °C / SM min. MHz / moc maks. W | GPU 1: temp. maks. °C / SM min. MHz / moc maks. W |
|---|---|---|
| `l3-q8-bench-ab` | 65 / 1200 / 283,6 | 43 / 1200 / 33,2 |
| `l3-q8-bench-single-split` | 64 / 1200 / 281,0 | 57 / 1200 / 155,7 |
| `l3-q4-bench-ab` | 64 / 1200 / 279,5 | 44 / 1200 / 33,4 |
| `l3-q4-bench-ab-confirmation` | 63 / 1200 / 279,4 | 44 / 1200 / 33,5 |
| `l3-q4-bench-single-split` | 63 / 1200 / 277,3 | 56 / 1200 / 155,8 |
| `l3-split-benchmark/0-C-q8.json` | 69 / 1170 / 246,9 | 73 / 1170 / 247,7 |
| `l3-split-benchmark/1-C-q4.json` | 69 / 1125 / 252,9 | 72 / 1155 / 244,2 |

- Zegar SM był ustawiony na 1200 MHz. W krokach ab.py nie spadał poniżej
  1200 MHz. W przebiegach HTTP spadał do 1125–1170 MHz.
- GPU 1 w krokach A/B nie jest używane (moc maks. 33,2–33,5 W). W krokach
  B/C i HTTP pracuje jako etap 1.
- Runner zapisał okna GPU 0 każdego z 42 przebiegów. Żadne nie ma braku
  telemetrii. W żadnym kroku nie zauważono obcego procesu obliczeniowego ani
  nieudanej próbki procesów. Warunki niczego nie filtrują, a raport nie
  przypisuje im przyczyny wyniku.

## GAP (2, bez zmiany nazwy)

| Indeks | Kontrola | Treść |
|---:|---|---|
| 4 | „base binary: declared source label” | declared by the build record (base: d42a400 rebuilt, exact commit d42a400d9eb1c0e83957067868859afead584477, source digest 94ef2131, 65 files, phase-05 paired builder r2); the runner does not verify the base's source (the parent's build audit) |
| 154 | „perf L3: the new candidate's single-GPU q8_0/q4_k HTTP benchmark (c1/c16)” | not measured by the plan: S3 describes only d42a400 and is not evidence for this candidate; for these modes the ab.py pairs are the only evidence |

Oba GAP-y pozostają GAP-ami. Późniejszy wynik ich nie usuwa. Raport nie
twierdzi więc o braku regresji HTTP CAND na jednym GPU w `q8_0`/`q4_k`.

## Związek z L1 i L2

L1 (`reports/multigpu-layer-l1-2026-10-13`) objęło równość bitową, odmowy,
HTTP, pojemność i profile transferów. Wynik L1: PASS 599, FAIL 0, GAP 42,
SIGNAL 0, C5 PASS, surowy manifest `2fec9dc6…`. L2
(`reports/multigpu-layer-l2-2026-10-13`) mierzyło tor domyślny `none`. L3
używa tych samych binarek i źródeł. Nie przenosi wniosków L1 ani L2 na tryby
kwantyzowane i nie łączy przebiegów różnych sesji w pary.

## Ograniczenia

- Jedna sesja, jedna maszyna, dwie identyczne karty RTX 6000 Ada 48 GB,
  model 4.5B, `q8_0` i `q4_k`. Nie wnioskujemy o innych kartach, modelach,
  mieszanych parach, Metal ani Intel.
- Wniosek (a) dotyczy par ab.py i rozdzielczości tej sesji. Brak
  utrzymanego sygnału nie dowodzi braku narzutu toru jednego GPU.
- HTTP CAND na jednym GPU w `q8_0`/`q4_k` nie był mierzony (GAP). HTTP z
  podziałem ma po jednym przebiegu na tryb i nie jest porównaniem.
- Pary B/C są opisowe. Nie są twierdzeniem o szybkości podziału w innym
  ruchu, innej liczbie klientów ani na innym sprzęcie.
- Różnica profili `q4_k`, dłuższe `load_s` B w `q4_k` i dwie grupy wartości
  sondy GEMM nie mają ustalonej przyczyny.
- `basal benchmark` ma ruch zamknięty: klient czeka na odpowiedź. Klient i
  serwer działają w jednym procesie. Pamięć jest próbkowana, a nie mierzona
  jako prawdziwy szczyt alokacji.
- Brak twierdzeń o jakości, zgodności (poza odnośnikiem do L1) i o fizycznej
  trasie transferu.

## Polecenia (ilustracyjne)

Rzeczywiste argv, ścieżki i środowisko zostają w prywatnych surowych
danych. `<base>`, `<candidate>`, `<models>`, `<out>`, `<table>` i `<tryb>`
to symbole zastępcze (`<tryb>` = `q8_0` albo `q4_k`).

```sh
# ab.py: A/B (BASE i CAND na GPU 0), 4 rundy abbaabba
python3 tools/bench/ab.py --rounds 4 --out <out>/l3-<tryb>-bench-ab \
  --a "<base> --color never bench --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --quantization <tryb>" \
  --b "<candidate> --color never bench --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --quantization <tryb>"

# ab.py: B/C (CAND na GPU 0 i CAND z podziałem); --b zamiast --device 0 ma --layer-split 0:30,1:30

# profil (diagnostyka potwierdzenia q4_k), A i B
<candidate> --color never profile --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --n 40 --batch 1 --quantization q4_k

# HTTP: CAND z podziałem, po jednym przebiegu na tryb
<candidate> --color never benchmark --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --layer-split 0:30,1:30 --scenario all --concurrency 1,16 --out <out>/0-C-q8.json --quantization q8_0
```

## Pliki

- `facts.json` — statusy i liczności, GAP, protokół, wszystkie pary A/B
  `q8_0` i `q4_k` (4 pola z A, B, B−A, B/A i oceną kierunku, statystyki per
  pytanie, `load_s`, sonda GEMM) z rundą potwierdzenia `q4_k`, sygnały,
  wnioski, profile, pary B/C (C−B, C/B), przebiegi HTTP z podziałem (21 pól,
  pule per GPU, wolna pamięć, NVML, `load_s`, transfery, telemetria per GPU),
  obiekt kwantyzacji `q4_k` wobec P0, warunki i kroki. Liczby mają pełne
  bity, a zaokrąglenia występują tylko w README.
- `provenance.json` — sumy binarek, źródeł, runnera i helperów, modelu,
  tabeli GEMM, wejść, faktów P0 i obciążenia, środowisko (karta, sterownik,
  biblioteki, zegar), manifest surowych danych, wybrane pola audytu ROOT,
  sumy plików raportów L1 i L2 oraz plików kontekstu.
- `validation.json` — kontrole autora: fakty wobec surowych danych i wartości
  runnera, sumy surowych plików, bieżące źródła, narzędzia i wejścia, tabele
  README odtworzone z `facts.json`, brak danych identyfikujących maszynę i
  danych finansowych.
