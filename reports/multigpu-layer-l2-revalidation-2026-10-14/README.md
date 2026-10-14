# Etap 05, podział warstw — rewalidacja L2 bieżącego kandydata, 2026-10-14

Stan: raport danych z jednej zakończonej sesji L2 (pomiar 2026-10-13 UTC)
dla bieżącego kandydata etapu 05 (źródła `831e68e0…`, 66 plików). Sesja
mierzy regresję toru domyślnego jednego GPU i opisuje kompromis podziału
`none`. Runner zgłosił jeden **utrzymany SIGNAL**: `basal benchmark`
`mixed-c1 queue_ms.p99`. Raport zawiera analizę tego wyniku z zebranych
surowych danych i przegląd źródeł tylko do odczytu. Nie znaleziono
przyczyny w kodzie. Według planu o tym konkretnym wyniku **decyduje
użytkownik**. Ta decyzja jeszcze nie zapadła. Raport nie akceptuje etapu 05
i nie jest końcowym review. Sesja L3 dla tych źródeł nie została
wykonana. Raport nie twierdzi niczego o `q8_0` ani `q4_k`.

## Dlaczego nowy raport

[Wcześniejszy raport L2](../multigpu-layer-l2-2026-10-13/README.md) dotyczy
poprzedniego kandydata (źródła `77b629ed…`, binarka `dfd13b7e…`). Po tamtym
pomiarze zmieniły się źródła kandydata. Doszły poprawka sprawdzania stanu
urządzeń po końcowym review R1 i integracja `main` (`4f68eb2`). Według
macierzy klas zmian planu to K3: nowa binarka unieważnia L1 i L2.
[L1 dla tych źródeł](../multigpu-layer-l1-revalidation-2026-10-13/README.md)
wykonano ponownie. Ten raport opisuje nowe, pełne L2 na tej samej parze
binarek. Wcześniejszy raport L2 pozostaje bez zmian. Jego liczby nie są
dowodem tego raportu.

## Wynik w skrócie

- **Runner:** `complete: true`, `stopped: null`, kod wyjścia 4 (według
  reguły runnera: brak FAIL, co najmniej jeden SIGNAL). 109 kontroli: PASS
  107, FAIL 0, GAP 1, SIGNAL 1. Liczności policzono ponownie z listy kontroli
  i z `checks.txt`. Jednostka L2 wystartowała i się zakończyła (98 PASS, 1
  SIGNAL). Wcześniej było 10 kontroli wstępnych. Wszystkie 14 kroków
  pomiaru zakończyło się kodem 0. W żadnym nie wykryto obcego procesu
  obliczeniowego ani braku próbek.
- **Regresja toru domyślnego, runda główna** (A = BASE na GPU 0, B = CAND
  na GPU 0, `none`): w ab.py (`basal bench`) żadne z 4 pól nie ma B gorszego
  we wszystkich 4 parach. Nie ma sygnału ani pola otwartego, więc ab.py nie
  dostało rundy potwierdzenia. W `basal benchmark` 9 z 21 pól ma B gorsze w
  obu parach A0/B1 i B4/A5. Brakujących wartości: 0.
- **Potwierdzenie:** jedna runda `basal benchmark` A B B A oraz `basal
  profile` A i B przy batch 1. Osiem z dziewięciu pól ma w potwierdzeniu
  parę z B niegorszym, więc nie są utrzymane.
- **Utrzymany SIGNAL:** `mixed-c1 queue_ms.p99`, B gorsze we wszystkich 4
  parach obu rund. A → B: 84,928 → 86,165 ms (+1,237 ms, +1,4565%), 84,388 →
  87,094 ms (+2,706 ms, +3,2066%), w potwierdzeniu 88,604 → 99,934 ms
  (+11,330 ms, +12,7872%) i 93,289 → 97,185 ms (+3,896 ms, +4,1763%).
  Nierozstrzygnięte: 0. Raport niczego tu nie pomija i nie stosuje progu.
- **Co mierzy to pole:** przy n = 900 p99 to wartość na randze 891. We
  wszystkich 10 przebiegach `benchmark` (8 A i B, 2 C) jest to
  najwolniejsze z 18 żądań `doc-4k-8k` o największym wejściu. Powyżej leży
  tylko 9 żądań `doc-16k` o największym wejściu. Pole jest więc maksimum z
  18 podobnych pomiarów w przebiegu. Mediana tych 18 żądań różni
  się między A i B o −0,885 do +0,234 ms, z różnym znakiem w różnych
  parach. W parach żądań o tym samym id B ma wyższe `queue_ms` w 360, 128,
  687 i 435 z 900 żądań.
- **Wynik end-to-end `mixed-c1`** (te same pary): latencja p99 B/A 0,9065,
  0,9982, 1,0014 i 0,9997, przepustowość żądań B/A 1,0308, 1,0016, 0,9969 i
  0,9999. Przebieg A0 rundy głównej ma jedno żądanie trwające 6003,9 ms
  (niżej). W parze potwierdzenia A0/B1 wszystkie 7 pól `mixed-c1` ma B
  gorsze. W pozostałych parach znaki są mieszane. To opis, a nie wyjaśnienie
  sygnału.
- **Przegląd źródeł (tylko do odczytu):** kod wykonywany w oknie
  `queue_ms` jednej instancji na jednym GPU jest w CAND tekstowo równy
  `d42a400` w 21 z 22 sprawdzonych elementów i w 3 całych plikach. Różni się
  tylko `has_evidence`. Wywołuje się je wyłącznie dla pytań z `evidence:
  true`, a żadne z 10 żądań na randze p99 takich pytań nie ma. **Nie
  znaleziono możliwej do wskazania przyczyny w kodzie.** To nie jest
  stwierdzenie, że przyczyny nie ma (niżej: czego analiza nie wyklucza).
- **Do decyzji użytkownika:** utrzymany SIGNAL `mixed-c1 queue_ms.p99` dla
  CAND `169aedfc…`. Decyzja nie jest podjęta i ten raport jej nie
  podejmuje. Uwaga o normalności 1–2% nie jest tu regułą akceptacji.
- **Kompromis podziału** (B wobec C = CAND `0:30,1:30`, opisowo, bez
  reguły): przy `mixed-c16` C ma przepustowość C/B 1,1801 i 1,1740 oraz p95
  latencji C/B 0,7478 i 0,7398. Przy `sequential-c1` i `mixed-c1` C ma
  przepustowość 0,9626–0,9732 B. W ab.py mediana C/B per pytanie wynosi
  1,00384–1,00889, a `dec_s` C/B 0,9415–0,9445. Pula alokatora po
  ładowaniu: C 4610,7 / 4618,7 MiB (GPU 0 / 1), B 9197,4 MiB na GPU 0.

## Zakres i pochodzenie

| Element | Wartość |
|---|---|
| Sprzęt | 2 × NVIDIA RTX 6000 Ada Generation, 49140 MiB każda, sterownik 580.126.09, limit mocy 300 W, tryb persistence |
| Zegar i CPU | żądany SM 1200 MHz (`nvidia-smi -pm 1`, `nvidia-smi -lgc 1200,1200`, oba kod 0), maska 16 z 26 logicznych CPU |
| Biblioteki | cuBLAS/cuBLASLt 12.9.1.4, cudart 12.9.79, tabele GEMM cuBLASLt 120901 |
| Środowisko procesów | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `HF_HUB_OFFLINE=1`, `BASAL_NO_UPDATE_CHECK=1`; katalog JIT CUDA pusty przed sesją (0 plików) |
| Okno kroków pomiaru (UTC) | 2026-10-13 20:57:29–22:08:02 |
| BASE | `44c8e399…`, przebudowane `d42a400`, źródła `94ef2131…` (65 plików) |
| CAND | `169aedfc…`, `d42a400` + zmiany etapu 05 + poprawka po review R1 + oczekujące scalenie `origin/main` `4f68eb2`, źródła `831e68e0…` (66 plików) |
| Build | te same binarki co w bieżącym L1: ten sam builder i wspólny przepis `2e37cb45…` (25 kroków), manifest buildu `2efaeca1…` (390 plików), akceptacja pary przez ROOT `cf05ffda…` |
| Runner / helpery | `tools/perf/run-layer-split-cloud.py` `81437a1c…`, `tools/bench/ab.py` `17b188dc…`; w L2 nie działał żaden proces helpera rezerwacji |
| Model i tabela | basal-1.5-4.5B `784a683b…` (60 warstw), f16, `none`, `fused`; tabela `nvidia-rtx-6000-ada-generation--basal-1.5-4.5B--f16--cublaslt120901.json` |
| Wejścia | `basal bench`: `reports/reference-basal-1.5-4.5B-fp32` (44 pytania, latencja na pytaniach 5..43); `basal benchmark`: wbudowane obciążenie `a645db4c…` (900 żądań sekwencyjnych i 900 mieszanych) |
| Surowe dane | manifest SHA-256 `481d6820026f13607913223e10b515e4ed92cb0faa825bd3070aed358aeff0cb`: 86 plików, 7 katalogów, 14 104 606 B; zebrane i zweryfikowane bez błędów, ten sam rozruch maszyny, runner i wrapper zakończone |
| Bieżące L1 | PASS 599, FAIL 0, GAP 42, SIGNAL 0, C5 PASS; surowy manifest `285936c2…`; audyt ROOT `a08961af…` |

Pełne sumy są w `provenance.json`. Etykieta źródła BASE jest deklaracją z
rekordu buildu. Runner jej nie weryfikuje, więc pozostaje GAP (niżej).
Raport nie twierdzi, że binarki są odtwarzalne bit w bit. Bieżące drzewo
robocze ma ten sam odcisk źródeł co CAND (66 plików, `831e68e0…`, lista
równa surowej `source-files.sha256`). Audyt ROOT tej sesji L2 nie był
gotowy w chwili pisania. Raport go nie czytał i na nim nie polega.

## Protokół

Role: A = BASE, GPU 0; B = CAND, GPU 0; C = CAND, `--layer-split 0:30,1:30`.
Wszystkie role używają `none`, f16 i tej samej tabeli GEMM.

- `basal benchmark --scenario all --concurrency 1,16`: sześć świeżych
  procesów w kolejności A B C C B A. Pary regresji są stałe: A0/B1 i B4/A5.
  Role są znormalizowane, więc B/A to zawsze CAND przez BASE. Pary
  kompromisu B1/C2 i C3/B4 są opisowe.
- `tools/bench/ab.py --rounds 4` z `basal bench --dtype f16`: A/B oraz B/C,
  kolejność `abbaabba`. Para k to `a-k` i `b-k`.
- 21 pól `benchmark`: 3 fazy (`sequential-c1`, `mixed-c1`, `mixed-c16`) × 7
  metryk. Latencja p50/p95/p99 i `queue_ms` p50/p99 są gorsze, gdy rosną (↑).
  `successful_requests_per_second` i `successful_questions_per_second` są
  gorsze, gdy maleją (↓). p50 to mediana (średnia dwóch środkowych przy
  parzystym n). p95 i p99 to wartość na randze `ceil(n·p)` posortowanej
  próby (`benchmark.rs`, funkcja `latency`; w raporcie `p99_method:
  nearest_rank`).
- 4 pola ab.py: `lat2_ms` i `lat2_p95_ms` (↑), `dec_s` (↓) oraz mediana B/A
  per pytanie z `lat2_ms` 39 pytań 5..43 (gorzej > 1).
- Reguła W8/W4 bez progu. Sygnał to pole z B gorszym we wszystkich
  zaplanowanych parach (benchmark 2, ab.py 4). Sygnał uruchamia dokładnie
  jedną rundę potwierdzenia tego polecenia (benchmark A B B A albo ab.py
  rounds 4) oraz `basal profile` A i B z `--n 40 --batch 1`. Utrzymany
  SIGNAL wymaga B gorszego we wszystkich parach obu rund. Brak albo
  nieskończona wartość jest problemem dowodu i daje brak wniosku.
- Według planu utrzymany SIGNAL wymaga analizy: profilu A/B i przeglądu
  ścieżki gorącej toru jednego etapu. Realna przyczyna oznacza poprawkę,
  review i rewalidację według macierzy klas zmian. Brak przyczyny oznacza
  decyzję użytkownika o tym konkretnym wyniku.

Wszystkie 21 pól każdej z 30 faz (10 przebiegów × 3 fazy) policzono ponownie
z wierszy `per_request` według tych definicji. Wyniki są równe wartościom
raportu `benchmark` co do bitu. Pary, sygnały, wniosek i profile są równe
wartościom runnera (`validation.json`).

## Regresja toru domyślnego (A/B)

### ab.py (`basal bench`), runda główna

| Para | `lat2_ms` A / B | `lat2_p95_ms` A / B | `dec_s` A / B | mediana B/A per pytanie | B szybsze (z 39) | Pola z B gorszym |
|---:|---|---|---|---:|---:|---|
| 0 | 25,627 / 25,587 | 32,554 / 32,413 | 64,11 / 64,21 | 0,99709 | 32 | — |
| 1 | 25,559 / 25,655 | 32,507 / 32,531 | 64,39 / 64,27 | 1,00155 | 9 | `lat2_ms`, `lat2_p95_ms`, `dec_s`, `per_question_median_ratio` |
| 2 | 25,550 / 25,501 | 32,499 / 32,445 | 64,17 / 64,41 | 0,99809 | 32 | — |
| 3 | 25,599 / 25,548 | 32,473 / 32,454 | 64,44 / 64,45 | 0,99939 | 33 | — |

Para 1 ma B gorsze we wszystkich czterech polach, a pary 0, 2 i 3 w
żadnym. Żadne pole nie jest więc sygnałem ani polem otwartym. Agregat
ab.py (mediana `lat2_ms` każdego pytania z 4 rund, potem B/A): mediana
0,99915, zakres 0,99431–1,00311, B szybsze w 29 z 39 pytań. We wszystkich
parach A i B mają tę samą dokładność (0,79545) i te same pytania. Runda
potwierdzenia ab.py nie była zaplanowana, więc nie istnieje.

### `basal benchmark`, wszystkie 21 pól

B/A każdej pary. **g** oznacza B gorsze w kierunku pola (↑ wyżej gorzej, ↓
niżej gorzej). Kolumny potwierdzenia to runda A B B A: pary (0 A, 1 B) i
(3 A, 2 B).

| Pole (gorzej) | A0/B1 | B4/A5 | potw. A0/B1 | potw. B2/A3 |
|---|---:|---:|---:|---:|
| `sequential-c1 successful_latency_ms.p50` (↑) | 0,9964 | 0,9987 | 1,0017 **g** | 1,0015 **g** |
| `sequential-c1 successful_latency_ms.p95` (↑) | 1,0044 **g** | 0,9966 | 1,0055 **g** | 1,0025 **g** |
| `sequential-c1 successful_latency_ms.p99` (↑) | 1,0007 **g** | 1,0019 **g** | 1,0046 **g** | 0,9984 |
| `sequential-c1 queue_ms.p50` (↑) | 0,9944 | 0,8658 | 1,0446 **g** | 1,0219 **g** |
| `sequential-c1 queue_ms.p99` (↑) | 1,0805 **g** | 1,0727 **g** | 1,0458 **g** | 0,9635 |
| `sequential-c1 successful_requests_per_second` (↓) | 0,9987 **g** | 1,0066 | 0,9973 **g** | 0,9992 **g** |
| `sequential-c1 successful_questions_per_second` (↓) | 0,9987 **g** | 1,0066 | 0,9973 **g** | 0,9992 **g** |
| `mixed-c1 successful_latency_ms.p50` (↑) | 1,0047 **g** | 0,9979 | 1,0040 **g** | 0,9982 |
| `mixed-c1 successful_latency_ms.p95` (↑) | 0,9986 | 1,0003 **g** | 1,0014 **g** | 0,9998 |
| `mixed-c1 successful_latency_ms.p99` (↑) | 0,9065 | 0,9982 | 1,0014 **g** | 0,9997 |
| `mixed-c1 queue_ms.p50` (↑) | 0,9716 | 0,8796 | 1,1210 **g** | 1,0026 **g** |
| `mixed-c1 queue_ms.p99` (↑) | 1,0146 **g** | 1,0321 **g** | 1,1279 **g** | 1,0418 **g** |
| `mixed-c1 successful_requests_per_second` (↓) | 1,0308 | 1,0016 | 0,9969 **g** | 0,9999 **g** |
| `mixed-c1 successful_questions_per_second` (↓) | 1,0308 | 1,0016 | 0,9969 **g** | 0,9999 **g** |
| `mixed-c16 successful_latency_ms.p50` (↑) | 1,0007 **g** | 1,0683 **g** | 0,9921 | 1,0006 **g** |
| `mixed-c16 successful_latency_ms.p95` (↑) | 1,0006 **g** | 0,9929 | 0,9907 | 1,0003 **g** |
| `mixed-c16 successful_latency_ms.p99` (↑) | 1,0027 **g** | 1,0039 **g** | 1,0043 **g** | 0,9859 |
| `mixed-c16 queue_ms.p50` (↑) | 1,0307 **g** | 1,1044 **g** | 0,9710 | 0,9333 |
| `mixed-c16 queue_ms.p99` (↑) | 1,0028 **g** | 1,0038 **g** | 0,9918 | 1,0007 **g** |
| `mixed-c16 successful_requests_per_second` (↓) | 0,9983 **g** | 0,9972 **g** | 1,0025 | 0,9981 **g** |
| `mixed-c16 successful_questions_per_second` (↓) | 0,9983 **g** | 0,9972 **g** | 1,0025 | 0,9981 **g** |

### Sygnały przez obie rundy

Dziewięć pól z B gorszym w obu parach rundy głównej, w kolejności
alfabetycznej. Wartości w ms albo na sekundę, z pełnymi bitami w
`facts.json`.

| Pole | Runda | Para | A | B | B−A | B/A | B gorsze |
|---|---|---|---:|---:|---:|---:|---|
| `mixed-c1 queue_ms.p99` | główna | A0/B1 | 84,928 | 86,165 | +1,237 | 1,0146 | tak |
| `mixed-c1 queue_ms.p99` | główna | B4/A5 | 84,388 | 87,094 | +2,706 | 1,0321 | tak |
| `mixed-c1 queue_ms.p99` | potwierdzenie | A0/B1 | 88,604 | 99,934 | +11,330 | 1,1279 | tak |
| `mixed-c1 queue_ms.p99` | potwierdzenie | B2/A3 | 93,289 | 97,185 | +3,896 | 1,0418 | tak |
| `mixed-c16 queue_ms.p50` | główna | A0/B1 | 274,806 | 283,241 | +8,435 | 1,0307 | tak |
| `mixed-c16 queue_ms.p50` | główna | B4/A5 | 245,219 | 270,812 | +25,594 | 1,1044 | tak |
| `mixed-c16 queue_ms.p50` | potwierdzenie | A0/B1 | 278,230 | 270,173 | −8,056 | 0,9710 | nie |
| `mixed-c16 queue_ms.p50` | potwierdzenie | B2/A3 | 259,213 | 241,916 | −17,297 | 0,9333 | nie |
| `mixed-c16 queue_ms.p99` | główna | A0/B1 | 39894,034 | 40007,576 | +113,542 | 1,0028 | tak |
| `mixed-c16 queue_ms.p99` | główna | B4/A5 | 40080,357 | 40233,101 | +152,744 | 1,0038 | tak |
| `mixed-c16 queue_ms.p99` | potwierdzenie | A0/B1 | 39983,078 | 39655,825 | −327,253 | 0,9918 | nie |
| `mixed-c16 queue_ms.p99` | potwierdzenie | B2/A3 | 40050,832 | 40079,144 | +28,312 | 1,0007 | tak |
| `mixed-c16 successful_latency_ms.p50` | główna | A0/B1 | 628,739 | 629,176 | +0,437 | 1,0007 | tak |
| `mixed-c16 successful_latency_ms.p50` | główna | B4/A5 | 581,032 | 620,726 | +39,694 | 1,0683 | tak |
| `mixed-c16 successful_latency_ms.p50` | potwierdzenie | A0/B1 | 628,761 | 623,779 | −4,983 | 0,9921 | nie |
| `mixed-c16 successful_latency_ms.p50` | potwierdzenie | B2/A3 | 587,959 | 588,291 | +0,331 | 1,0006 | tak |
| `mixed-c16 successful_latency_ms.p99` | główna | A0/B1 | 44646,352 | 44765,503 | +119,151 | 1,0027 | tak |
| `mixed-c16 successful_latency_ms.p99` | główna | B4/A5 | 44934,744 | 45111,302 | +176,559 | 1,0039 | tak |
| `mixed-c16 successful_latency_ms.p99` | potwierdzenie | A0/B1 | 44733,353 | 44925,775 | +192,422 | 1,0043 | tak |
| `mixed-c16 successful_latency_ms.p99` | potwierdzenie | B2/A3 | 45578,274 | 44935,769 | −642,504 | 0,9859 | nie |
| `mixed-c16 successful_questions_per_second` | główna | A0/B1 | 13,055 | 13,033 | −0,022 | 0,9983 | tak |
| `mixed-c16 successful_questions_per_second` | główna | B4/A5 | 13,083 | 13,047 | −0,037 | 0,9972 | tak |
| `mixed-c16 successful_questions_per_second` | potwierdzenie | A0/B1 | 13,050 | 13,082 | +0,032 | 1,0025 | nie |
| `mixed-c16 successful_questions_per_second` | potwierdzenie | B2/A3 | 13,097 | 13,072 | −0,025 | 0,9981 | tak |
| `mixed-c16 successful_requests_per_second` | główna | A0/B1 | 5,6817 | 5,6720 | −0,0097 | 0,9983 | tak |
| `mixed-c16 successful_requests_per_second` | główna | B4/A5 | 5,6939 | 5,6780 | −0,0159 | 0,9972 | tak |
| `mixed-c16 successful_requests_per_second` | potwierdzenie | A0/B1 | 5,6793 | 5,6932 | +0,0139 | 1,0025 | nie |
| `mixed-c16 successful_requests_per_second` | potwierdzenie | B2/A3 | 5,6999 | 5,6888 | −0,0111 | 0,9981 | tak |
| `sequential-c1 queue_ms.p99` | główna | A0/B1 | 0,9560 | 1,0330 | +0,0770 | 1,0805 | tak |
| `sequential-c1 queue_ms.p99` | główna | B4/A5 | 0,9490 | 1,0180 | +0,0690 | 1,0727 | tak |
| `sequential-c1 queue_ms.p99` | potwierdzenie | A0/B1 | 0,9380 | 0,9810 | +0,0430 | 1,0458 | tak |
| `sequential-c1 queue_ms.p99` | potwierdzenie | B2/A3 | 0,9580 | 0,9230 | −0,0350 | 0,9635 | nie |
| `sequential-c1 successful_latency_ms.p99` | główna | A0/B1 | 48,443 | 48,476 | +0,034 | 1,0007 | tak |
| `sequential-c1 successful_latency_ms.p99` | główna | B4/A5 | 48,627 | 48,721 | +0,094 | 1,0019 | tak |
| `sequential-c1 successful_latency_ms.p99` | potwierdzenie | A0/B1 | 48,749 | 48,972 | +0,223 | 1,0046 | tak |
| `sequential-c1 successful_latency_ms.p99` | potwierdzenie | B2/A3 | 48,685 | 48,609 | −0,076 | 0,9984 | nie |

Osiem pól ma w potwierdzeniu co najmniej jedną parę z B niegorszym, więc
nie są utrzymane. W wielu z nich B jest jednak gorsze w 3 z 4 par. Pomiar
nie wyklucza więc małego narzutu w tych polach. Jedno pole,
`mixed-c1 queue_ms.p99`, ma B gorsze w 4 z 4 par i jest utrzymanym SIGNAL.

### Profile A/B, batch 1 (diagnostyka)

`basal profile --n 40 --batch 1`, średnio 239,8 tokenu na element.
`residual_abs_max` wynosi 0,0 w obu profilach. Profil synchronizuje sekcje
forwardu, więc te czasy nie są latencją żądania. Okno `queue_ms` nie
zawiera żadnego forwardu, więc profile nie wyjaśniają różnicy
`queue_ms` ani jej nie wykluczają.

| Sekcja | A ms/element | B ms/element |
|---|---:|---:|
| `attention` | 6,545 | 6,556 |
| `bias+rope+heads` | 0,873 | 0,876 |
| `bias+silu*up` | 1,099 | 1,102 |
| `down_proj+bias+residual` | 9,896 | 9,896 |
| `embed+inputs` | 0,485 | 0,480 |
| `final_norm` | 0,045 | 0,051 |
| `gate_up_proj` | 9,942 | 9,955 |
| `o_proj+bias+residual+rms_norm` | 2,317 | 2,327 |
| `qkv_proj` | 2,265 | 2,275 |
| `rms_norm` | 0,855 | 0,857 |
| suma sekcji | 34,324 | 34,376 |
| `ms_per_item_profiled` | 34,398 | 34,453 |

`ms_per_item_profiled` B/A wynosi 1,0016, a suma sekcji B/A 1,0015. To
jeden profil na rolę, bez powtórzeń.

## Utrzymany SIGNAL: `mixed-c1 queue_ms.p99` — analiza

Analiza używa tylko zebranych surowych danych i źródeł w trybie do
odczytu. Nie uruchamia GPU, modelu ani binarek. Nie dodaje pomiarów ani
symulacji.

### Definicja i ścieżka pomiaru

- Serwer zapisuje `x-basal-queue-ms` jako `start − arrived` w ms z
  dokładnością 0,001 ms. `arrived` to `Instant::now()` na początku handlera
  HTTP, przed parsowaniem JSON (`serve.rs`, `systemone`). `start` to
  `Instant::now()` w wątku toru (`scheduler` albo `long_lane`) po odbiorze i
  zaplanowaniu żądania. Planowanie (`Engine::plan_request_as`) obejmuje
  parsowanie, walidację, tokenizację i planowanie promptów. Żądania powyżej
  `long_tokens` (4096 tokenów) przechodzą jeszcze do wątku długiego toru.
  Przyjęcie do batcha i wszystkie forwardy są po `start` i należą do
  `batch_compute_ms`.
- Klient `basal benchmark` czyta nagłówek każdej odpowiedzi
  (`benchmark/http.rs`). Wartość fazy to p99 z wszystkich 900 wierszy
  według definicji wyżej, czyli wartość na randze 891.
- `mixed-c1` to ruch zamknięty z jednym klientem: następne żądanie wychodzi
  po odpowiedzi na poprzednie. Kolejność 900 żądań jest taka sama we
  wszystkich 10 przebiegach, a każde żądanie ma batch 1. W tym oknie
  nie ma pracy GPU tego żądania. Są w nim parsowanie, tokenizacja i
  przekazanie między wątkami na CPU hosta.
  `queue_ms` rośnie z rozmiarem wejścia. Mediana dla 18 żądań `doc-4k-8k` o
  największym wejściu wynosi około 81–90 ms, a dla 9 żądań `doc-16k` o
  największym wejściu około 176–194 ms.

### Ogon rozkładu w każdym przebiegu

| Przebieg | Rola | `queue_ms.p99` | żądanie na randze 891 | 18 największych `doc-4k-8k`: p50 / maks. | 9 największych `doc-16k`: p50 / maks. | `queue_ms.p50` | średnia `queue_ms` |
|---|---|---:|---|---|---|---:|---:|
| L2 0 | A | 84,928 | `doc-4k-8k/7` | 82,044 / 84,928 | 176,410 / 178,088 | 0,8110 | 7,171 |
| L2 1 | B | 86,165 | `doc-4k-8k/31` | 82,279 / 86,165 | 178,515 / 213,312 | 0,7880 | 7,218 |
| L2 2 | C | 92,718 | `doc-4k-8k/19` | 82,239 / 92,718 | 182,421 / 187,598 | 0,7870 | 7,211 |
| L2 3 | C | 101,946 | `doc-4k-8k/7` | 89,823 / 101,946 | 194,056 / 225,786 | 0,9915 | 8,095 |
| L2 4 | B | 87,094 | `doc-4k-8k/63` | 81,254 / 87,094 | 175,817 / 205,699 | 0,7745 | 7,119 |
| L2 5 | A | 84,388 | `doc-4k-8k/67` | 82,139 / 84,388 | 178,658 / 182,041 | 0,8805 | 7,275 |
| potw. 0 | A | 88,604 | `doc-4k-8k/43` | 81,903 / 88,604 | 178,125 / 180,518 | 0,7810 | 7,164 |
| potw. 1 | B | 99,934 | `doc-4k-8k/15` | 81,980 / 99,934 | 177,715 / 210,713 | 0,8755 | 7,613 |
| potw. 2 | B | 97,185 | `doc-4k-8k/35` | 81,499 / 97,185 | 175,585 / 184,779 | 0,7815 | 7,165 |
| potw. 3 | A | 93,289 | `doc-4k-8k/59` | 81,935 / 93,289 | 177,197 / 181,779 | 0,7795 | 7,153 |

We wszystkich 10 przebiegach wartość na randze 891 to maksimum 18 żądań
`doc-4k-8k` o największym wejściu (za każdym razem inne id). Powyżej leży
dokładnie 9 żądań `doc-16k` o największym wejściu. Pole `queue_ms.p99` tej
fazy jest więc najwolniejszym z 18 podobnych pomiarów tego okna w
przebiegu. Mediana tej grupy w A wynosi 81,903–82,139 ms, a w B
81,254–82,279 ms. Maksimum w B jest wyższe od maksimum w parze A we
wszystkich 4 parach. W przebiegu C3 cały rozkład jest przesunięty w górę
(p50 grupy 89,823 ms, `queue_ms.p50` 0,9915 ms). Raport nie ustala
przyczyny tego przesunięcia.

### Pary A/B: ogon, żądania sparowane i wynik end-to-end

| Runda | Para | `queue_ms.p99` B−A (%) | maks. 18 największych `doc-4k-8k` B−A | p50 tej grupy B−A | żądania z B > A (z 900) | mediana B−A per żądanie | latencja p50 / p95 / p99 B/A | żądania/s B/A | `wall_s` A / B |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| główna | A0/B1 | +1,237 (+1,46%) | +1,237 | +0,234 | 360 | −0,0155 | 1,0047 / 0,9986 / 0,9065 | 1,0308 | 189,273 / 183,619 |
| główna | B4/A5 | +2,706 (+3,21%) | +2,706 | −0,885 | 128 | −0,1250 | 0,9979 / 1,0003 / 0,9982 | 1,0016 | 183,721 / 183,432 |
| potwierdzenie | A0/B1 | +11,330 (+12,79%) | +11,330 | +0,077 | 687 | +0,0810 | 1,0040 / 1,0014 / 1,0014 | 0,9969 | 183,514 / 184,082 |
| potwierdzenie | B2/A3 | +3,896 (+4,18%) | +3,896 | −0,436 | 435 | −0,0010 | 0,9982 / 0,9998 / 0,9997 | 0,9999 | 183,524 / 183,539 |

- Różnica pola jest różnicą maksimów grupy. Mediany tej grupy różnią się
  między A i B o −0,885 do +0,234 ms, z różnym znakiem w różnych parach.
- Po sparowaniu tych samych id (900 żądań) B ma wyższe `queue_ms` w 360,
  128, 687 i 435 żądaniach. Mediana różnicy B−A wynosi −0,0155, −0,1250,
  +0,0810 i −0,0010 ms. Tylko w parze potwierdzenia A0/B1 większość żądań
  ma w B wyższe `queue_ms`. W tej parze także `queue_ms.p50` B/A wynosi
  1,1210, a wszystkie 7 pól `mixed-c1` ma B gorsze. W pozostałych parach
  `queue_ms.p50` B/A wynosi 0,9716, 0,8796 i 1,0026.
- Suma `queue_ms` w `mixed-c1` to 3,41–3,83% sumy latencji żądań (6,407–7,286
  s z 183,3–190,2 s).
- W przebiegu A0 rundy głównej żądanie `features/14` (piętnaste w fazie)
  miało latencję 6003,9 ms, w tym `batch_compute_ms` 6002,983 ms. To jedyne
  żądanie powyżej 4000 ms w 10 przebiegach. Latencja p99 tego przebiegu
  wynosi 3364,201 ms, a w pozostałych przebiegach A i B 3048,958–3055,868
  ms. `wall_s` fazy wynosi 189,273 s, a w pozostałych przebiegach A i B
  183,432–184,082 s. Raport nie ustala przyczyny i nie usuwa tego wiersza.

### Ścieżka gorąca toru jednego etapu: `d42a400` wobec CAND

Porównanie tekstu źródeł z obiektów git `d42a400` i bieżącego drzewa (to
samo co CAND, odcisk `831e68e0…`). Element zaczyna się od sygnatury, a
kończy na zamykającym nawiasie z tym samym wcięciem.

| Plik | Element | Gdzie względem okna `queue_ms` | d42a400 = CAND |
|---|---|---|---|
| `crates/basal-cli/src/serve.rs` | `struct Timing` | typ mierzonej wartości | tak |
| `crates/basal-cli/src/serve.rs` | `struct Job` | niesie `arrived` | tak |
| `crates/basal-cli/src/serve.rs` | `async fn systemone(` | okno kolejki: ustawia `arrived`, potem parsowanie JSON | tak |
| `crates/basal-cli/src/serve.rs` | `async fn basal(` | okno kolejki (endpoint upstream, benchmark go nie używa) | tak |
| `crates/basal-cli/src/serve.rs` | `async fn decide(` | okno kolejki: model, slot inflight, wysłanie do kolejki; po odpowiedzi nagłówek `x-basal-queue-ms` | tak |
| `crates/basal-cli/src/serve.rs` | `impl Drop for Inflight` | okno kolejki | tak |
| `crates/basal-cli/src/serve.rs` | `fn submit(&self, job: Job)` | okno kolejki: wysłanie kanałem | tak |
| `crates/basal-cli/src/serve.rs` | `fn send(&self, mut job: Job) -> bool` | okno kolejki: wybór instancji | tak |
| `crates/basal-cli/src/serve.rs` | `fn plan_prompts(` | okno kolejki: lista promptów do oszacowania rozmiaru | tak |
| `crates/basal-cli/src/serve.rs` | `struct Waiting` | okno kolejki | tak |
| `crates/basal-cli/src/serve.rs` | `struct Lane` | okno kolejki | tak |
| `crates/basal-cli/src/serve.rs` | `fn admit(&mut self, start: Instant)` | koniec okna (`start` pobrany przed wywołaniem) | tak |
| `crates/basal-cli/src/serve.rs` | `fn take(&mut self, start: Instant)` | po `start` (przyjęcie do batcha) | tak |
| `crates/basal-cli/src/serve.rs` | `fn run<B: Backend>(` | liczy `queue_ms = start − arrived`; forwardy po `start` (`compute_ms`) | tak |
| `crates/basal-cli/src/serve.rs` | `fn answer(self, reply: Reply)` | po obliczeniach: odpowiedź do klienta | tak |
| `crates/basal-cli/src/serve.rs` | `fn scheduler<B: Backend>(` | okno kolejki: odbiór w głównym torze, `plan_request_as`, przekazanie długich żądań, `start` | tak |
| `crates/basal-cli/src/serve.rs` | `fn long_lane<B: Backend>(` | okno kolejki długich żądań: odbiór, `start` | tak |
| `crates/basal-core/src/engine.rs` | `pub fn plan_request_as(` | okno kolejki: parsowanie żądania, walidacja, planowanie | tak |
| `crates/basal-core/src/engine.rs` | `fn check_context(` | okno kolejki | tak |
| `crates/basal-core/src/engine.rs` | `pub fn plan_items(` | okno kolejki: tokenizacja i planowanie promptów | tak |
| `crates/basal-cli/src/benchmark.rs` | `fn latency(values: &[f64]) -> Value` | definicja percentyli raportu | tak |
| `crates/basal-gpu/src/lib.rs` | `fn has_evidence(&self) -> bool` | okno kolejki tylko dla pytań z `evidence: true` | **nie** |
| `crates/basal-cli/src/benchmark/http.rs` | cały plik | klient benchmarku / parsowanie żądania / mediana | tak |
| `crates/basal-core/src/request.rs` | cały plik | klient benchmarku / parsowanie żądania / mediana | tak |
| `crates/basal-cli/src/stats.rs` | cały plik | klient benchmarku / parsowanie żądania / mediana | tak |

- **Jedyna różnica w oknie:** `has_evidence` w `d42a400` czyta
  `self.evidence.is_some()`. W CAND wywołuje `self.head().is_ok_and(|h|
  h.evidence.is_some())`, czyli pole ostatniego etapu bez blokady i bez
  synchronizacji GPU. `plan_request_as` wywołuje je tylko dla pytań z
  `evidence: true`. W obciążeniu mieszanym ma je 22 z 900 żądań, a żadne z
  10 żądań na randze p99 ich nie ma.
- **`basal-core`:** zmieniły się tylko `engine.rs` (nowa domyślna metoda
  `Backend::set_device_gate`, wołana przy uruchamianiu serwera) i
  `gate.rs`. Bramka jest brana w forwardzie, po `start`. Pliki
  `request.rs` i `stats.rs` są bez zmian.
- **`serve.rs`:** zmienione fragmenty dotyczą uruchamiania (bramki per GPU,
  etykiety, `/health`, komentarze). Handler, `decide`, kierowanie, kanały,
  `scheduler`, `long_lane` i `Lane` są równe `d42a400`.
- **Klient `benchmark`:** `benchmark/http.rs` jest bez zmian. Zmiany w
  `benchmark.rs` dotyczą przygotowania (podział, pule per urządzenie,
  telemetria transferów), a nie wysyłania żądań i statystyk.
- **Wątek próbkowania zasobów** (ten sam proces, co 200 ms) zmienił się w
  CAND. Zbiera listę urządzeń zamiast jednego i sprawdza każdy limit. Przy
  skończonym limicie cgroup czyta też `memory.stat`. W tej sesji
  `cgroups` przed ładowaniem jest puste we wszystkich przebiegach B i C, więc
  ta gałąź nie była wykonana (obserwacja z migawki przed ładowaniem, bez
  pomiaru narzutu wątku). Średnie użycie CPU procesu w `mixed-c1` wynosi
  1,0008–1,0027 rdzenia w A, B i C.
- **Zależności:** w `Cargo.lock` zmienił się tylko parser YAML
  (`serde_yaml` → `serde-saphyr` i jego zależności). Używa go wyłącznie
  wczytywanie konfiguracji (`config.rs`). Polecenie `benchmark` nie czyta
  YAML.
- **Kod GPU** (`basal-gpu/src/lib.rs`, ponad 2200 zmienionych linii) działa w
  oknie `batch_compute_ms`, nie w oknie `queue_ms`. Przy jednym kliencie
  następne żądanie powstaje dopiero po odpowiedzi, a wątek toru odpowiada po
  powrocie z `run_plans`.

### Wniosek analizy

Zebrane dane i przegląd źródeł nie wskazują przyczyny utrzymanego SIGNAL w
kodzie CAND. Kod wykonywany między `arrived` a `start` dla żądań z rangi p99
jest tekstowo taki sam jak w `d42a400`. Wartość pola to maksimum 18 pomiarów
okna przed obliczeniami, w którym kod nie wykonuje pracy GPU tego żądania. Mediana tej grupy i większość sparowanych żądań
nie przesuwają się w stałym kierunku. Raport **nie** proponuje poprawki, bo
nie ma wskazanej przyczyny, i **nie** uznaje różnicy za szum.

Czego ta analiza nie wyklucza:

- wpływu całej binarki CAND: innego układu i inliningu kodu, stanu
  alokatora po innym kodzie GPU, innych wywołań bibliotek CUDA w forwardzie;
- wpływu wątku próbkowania zasobów CAND lub szeregowania wątków przez
  system;
- przyczyny po stronie hosta lub maszyny, której telemetria nie obejmuje
  (zegar i obciążenie CPU, przerwania). Telemetria dotyczy tylko GPU;
- zachowania przy innej liczbie przebiegów, innym ruchu albo innym
  sprzęcie. Każda runda ma 2 pary, a pole jest skrajną wartością małej grupy.

Raport nie ustala też, które z 18 żądań przeszły przez długi tor. Dane
`per_request` tego nie zapisują.

### Decyzja użytkownika (oczekująca)

Według planu („brak przyczyny → decyzja użytkownika o tym konkretnym
wyniku”) potrzebna jest decyzja o utrzymanym SIGNAL `basal benchmark`
`mixed-c1 queue_ms.p99` dla CAND `169aedfc…` (źródła `831e68e0…`) wobec
BASE `44c8e399…`. Wielkości do decyzji: +1,237, +2,706, +11,330 i +3,896 ms
(+1,4565%, +3,2066%, +12,7872%, +4,1763%) przy wartościach 84,388–99,934 ms.
Możliwe rozstrzygnięcia, bez rekomendacji tego raportu:

1. przyjąć ten konkretny wynik dla tej binarki CAND jako zmierzony i
   opisany, z ograniczeniami wyżej;
2. zlecić dalszą diagnostykę; nowe pomiary wymagają zrecenzowanej rewizji
   planu i runnera oraz nowej sesji według zasad planu;
3. uznać wynik za blokujący i wstrzymać etap do wskazania przyczyny.

Uwaga o normalności różnic 1–2% nie rozstrzyga tego wyniku automatycznie.
Do czasu decyzji SIGNAL pozostaje otwarty.

## Kompromis podziału (B/C, opisowo)

### `basal benchmark`, wszystkie 21 pól

B = CAND na GPU 0, C = CAND z `0:30,1:30`. Pary B1/C2 i C3/B4 (role
znormalizowane: C/B to zawsze podział przez jeden GPU). Bez reguły.

| Pole | B1 | C2 | C/B | B4 | C3 | C/B |
|---|---:|---:|---:|---:|---:|---:|
| `sequential-c1 successful_latency_ms.p50` | 26,60 | 27,80 | 1,0451 | 26,78 | 28,17 | 1,0519 |
| `sequential-c1 successful_latency_ms.p95` | 40,23 | 41,21 | 1,0244 | 40,18 | 41,76 | 1,0391 |
| `sequential-c1 successful_latency_ms.p99` | 48,48 | 49,74 | 1,0261 | 48,72 | 50,53 | 1,0371 |
| `sequential-c1 queue_ms.p50` | 0,3520 | 0,3620 | 1,0284 | 0,3870 | 0,4550 | 1,1757 |
| `sequential-c1 queue_ms.p99` | 1,0330 | 0,9190 | 0,8896 | 1,0180 | 1,3790 | 1,3546 |
| `sequential-c1 successful_requests_per_second` | 32,21 | 31,35 | 0,9732 | 32,17 | 30,96 | 0,9626 |
| `sequential-c1 successful_questions_per_second` | 32,21 | 31,35 | 0,9732 | 32,17 | 30,96 | 0,9626 |
| `mixed-c1 successful_latency_ms.p50` | 46,10 | 47,13 | 1,0225 | 46,02 | 47,64 | 1,0352 |
| `mixed-c1 successful_latency_ms.p95` | 1103,36 | 1140,35 | 1,0335 | 1103,53 | 1142,30 | 1,0351 |
| `mixed-c1 successful_latency_ms.p99` | 3049,70 | 3144,66 | 1,0311 | 3048,96 | 3152,96 | 1,0341 |
| `mixed-c1 queue_ms.p50` | 0,7880 | 0,7870 | 0,9987 | 0,7745 | 0,9915 | 1,2802 |
| `mixed-c1 queue_ms.p99` | 86,17 | 92,72 | 1,0761 | 87,09 | 101,95 | 1,1705 |
| `mixed-c1 successful_requests_per_second` | 4,9015 | 4,7609 | 0,9713 | 4,9065 | 4,7288 | 0,9638 |
| `mixed-c1 successful_questions_per_second` | 11,26 | 10,94 | 0,9713 | 11,27 | 10,87 | 0,9638 |
| `mixed-c16 successful_latency_ms.p50` | 629,18 | 538,17 | 0,8554 | 620,73 | 554,21 | 0,8928 |
| `mixed-c16 successful_latency_ms.p95` | 17190,24 | 12854,95 | 0,7478 | 17200,90 | 12725,20 | 0,7398 |
| `mixed-c16 successful_latency_ms.p99` | 44765,50 | 39758,83 | 0,8882 | 45111,30 | 39779,94 | 0,8818 |
| `mixed-c16 queue_ms.p50` | 283,24 | 201,34 | 0,7108 | 270,81 | 218,63 | 0,8073 |
| `mixed-c16 queue_ms.p99` | 40007,58 | 36238,27 | 0,9058 | 40233,10 | 35971,19 | 0,8941 |
| `mixed-c16 successful_requests_per_second` | 5,6720 | 6,6934 | 1,1801 | 5,6780 | 6,6658 | 1,1740 |
| `mixed-c16 successful_questions_per_second` | 13,03 | 15,38 | 1,1801 | 13,05 | 15,32 | 1,1740 |

### ab.py (`basal bench`)

| Para | `lat2_ms` B / C | `lat2_p95_ms` B / C | `dec_s` B / C | mediana C/B per pytanie | C szybsze (z 39) |
|---:|---|---|---|---:|---:|
| 0 | 25,662 / 25,718 | 32,433 / 32,718 | 64,41 / 60,64 | 1,00780 | 2 |
| 1 | 25,512 / 25,663 | 32,471 / 32,702 | 64,33 / 60,76 | 1,00889 | 0 |
| 2 | 25,578 / 25,671 | 32,509 / 32,675 | 64,47 / 60,78 | 1,00619 | 2 |
| 3 | 25,643 / 25,626 | 32,556 / 32,829 | 64,36 / 60,78 | 1,00384 | 3 |

Agregat ab.py C/B: mediana 1,00656, zakres 1,00100–1,01105, C szybsze w 0
z 39 pytań.

### Pamięć i `load_s`

Pula to pula alokatora CUDA tego procesu (`cuMemAllocAsync`) na danym
urządzeniu, bez kontekstu CUDA, modułów i uchwytów bibliotek. „Min. wolna”
to najmniejsza wolna pamięć urządzenia z próbek co 200 ms (wszystkie
procesy). NVML `memory.used` pochodzi z `nvidia-smi` co 200 ms (MiB, poziom
urządzenia). Wartości w MiB.

| Przebieg | Rola | `load_s` | pula: po ładowaniu GPU 0 / 1 | pula: szczyt GPU 0 / 1 | min. wolna (próbki) GPU 0 / 1 | NVML `memory.used` maks. GPU 0 / 1 |
|---|---|---:|---|---|---|---|
| L2 0 | A | 12,169 | 9197,4 / — | 11686,7 / — | 36086 / — | 12424 / 4 |
| L2 1 | B | 12,153 | 9197,4 / — | 11686,7 / — | 36086 / — | 12424 / 4 |
| L2 2 | C | 12,429 | 4610,7 / 4618,7 | 7161,1 / 7227,6 | 40758 / 40662 | 7752 / 7848 |
| L2 3 | C | 11,967 | 4610,7 / 4618,7 | 7161,1 / 7227,6 | 40758 / 40662 | 7752 / 7848 |
| L2 4 | B | 12,108 | 9197,4 / — | 11686,7 / — | 36118 / — | 12392 / 4 |
| L2 5 | A | 11,506 | 9197,4 / — | 11686,7 / — | 36086 / — | 12424 / 4 |
| potw. 0 | A | 12,362 | 9197,4 / — | 11686,7 / — | 36150 / — | 12360 / 4 |
| potw. 1 | B | 12,149 | 9197,4 / — | 11951,1 / — | 35830 / — | 12680 / 4 |
| potw. 2 | B | 12,223 | 9197,4 / — | 11686,7 / — | 36118 / — | 12392 / 4 |
| potw. 3 | A | 11,895 | 9197,4 / — | 11686,7 / — | 36150 / — | 12360 / 4 |

### Transfery między etapami (przebiegi C `benchmark`)

`transfer_ms` to suma czasu hosta udanych transferów granicy. Obejmuje
alokację i inicjalizację celu, opróżnienie, kopię API urządzenie–urządzenie,
potwierdzenie na obu GPU, zwolnienie źródła i wiązanie kontekstu.
`gate_wait_ms` to branie bramki następnego GPU i wiązanie jego kontekstu.
Fizyczna trasa kopii (P2P albo przez hosta) nie jest ustalona.

| Przebieg | Kierunek | Kopie | Bajty | `transfer_ms` | `gate_wait_ms` |
|---|---|---:|---:|---:|---:|
| L2 2 C | etap 0 → 1 (GPU 0 → 1) | 2423 | 22 919 921 664 | 7072,955 | 16102,535 |
| L2 3 C | etap 0 → 1 (GPU 0 → 1) | 2418 | 22 976 765 952 | 7248,133 | 16149,837 |

## Warunki pomiaru (opisowo)

Telemetria `nvidia-smi` co 200 ms, osobny plik każdego przebiegu
`benchmark`, obie karty.

| Przebieg | Rola | GPU 0: temp. maks. °C / SM min. / śr. MHz / moc maks. W | GPU 1: temp. maks. °C / SM min. / śr. MHz / moc maks. W |
|---|---|---|---|
| L2 0 | A | 80 / 1035 / 1196,8 / 288,4 | 48 / 1200 / 1200,0 / 35,3 |
| L2 1 | B | 81 / 1095 / 1195,8 / 289,9 | 46 / 1200 / 1200,0 / 34,4 |
| L2 2 | C | 70 / 1125 / 1199,9 / 247,0 | 74 / 1095 / 1199,7 / 251,9 |
| L2 3 | C | 70 / 1170 / 1199,9 / 249,3 | 74 / 1140 / 1199,6 / 252,1 |
| L2 4 | B | 81 / 1050 / 1196,5 / 288,7 | 64 / 1200 / 1200,0 / 128,1 |
| L2 5 | A | 80 / 1035 / 1196,2 / 287,8 | 45 / 1200 / 1200,0 / 34,4 |
| potw. 0 | A | 80 / 1080 / 1195,9 / 289,1 | 45 / 1200 / 1200,0 / 33,9 |
| potw. 1 | B | 80 / 1035 / 1196,8 / 288,8 | 45 / 1200 / 1200,0 / 34,1 |
| potw. 2 | B | 80 / 1050 / 1196,9 / 289,9 | 45 / 1200 / 1200,0 / 34,5 |
| potw. 3 | A | 80 / 1080 / 1196,6 / 288,2 | 45 / 1200 / 1200,0 / 34,2 |

- Zegar SM był ustawiony na 1200 MHz. W przebiegach A i B `benchmark`
  spadał do 1035–1095 MHz przy 80–81 °C i do 289,9 W. Średni zegar SM
  wynosił 1195,8–1196,9 MHz. A i B pracowały w podobnych warunkach.
- W rundach ab.py GPU 0 miało do 65 °C, a zegar SM 1200 MHz bez spadków.
- Warunki niczego nie filtrują, a raport nie przypisuje im przyczyny
  wyniku. Telemetria nie obejmuje CPU hosta.

## GAP i SIGNAL (bez zmiany nazw)

| Status | Kontrola | Treść runnera |
|---|---|---|
| GAP | „base binary: declared source label” | deklaracja z rekordu buildu (`d42a400`, źródła `94ef2131`, 65 plików); runner nie weryfikuje źródła BASE, robi to audyt buildu |
| SIGNAL | „perf L2 none: persistent regression signal (B worse in every pair of both rounds)” | `benchmark: mixed-c1 queue_ms.p99`; analiza, a potem poprawka i rewalidacja albo decyzja użytkownika |

GAP pozostaje GAP-em, a SIGNAL pozostaje SIGNAL-em. Późniejszy wynik ich
nie usuwa.

## Związek z L1

Bieżące L1 (`reports/multigpu-layer-l1-revalidation-2026-10-13`) objęło te
same binarki i źródła: równość bitową, odmowy, HTTP, pojemność i profile
transferów. Wynik: PASS 599, FAIL 0, GAP 42, SIGNAL 0, C5 PASS. Runner L2
sprawdził przed pomiarem, że L1 użyło tego samego runnera i helperów. L2 nie
mierzy ponownie zgodności, jakości ani pojemności. Nie przenosi też
wniosków L1 na wydajność.

## Korekty i zastrzeżenia autora

- W zleceniu podano czasy profili ok. 34,3976 i 34,4528 ms jako „sections”.
  W surowych danych to `ms_per_item_profiled`. Suma sekcji wynosi 34,324 i
  34,376 ms. Raport podaje obie wielkości.
- Pozostałe liczby ze zlecenia (statusy, wartości i różnice
  `mixed-c1 queue_ms.p99`, 9 sygnałów rundy głównej, puste
  `confirmation_bench`) zgadzają się z przeliczeniem.

## Ograniczenia

- Jedna sesja, jedna maszyna, dwie identyczne karty RTX 6000 Ada 48 GB,
  model 4.5B, `none`. Nie wnioskujemy o innych kartach, modelach, Metal ani
  Intel.
- Utrzymany SIGNAL jest opisany, a nie wyjaśniony. Brak wskazanej przyczyny
  w kodzie nie dowodzi braku narzutu.
- Pola, które nie są utrzymane, nie dowodzą braku narzutu toru jednego GPU.
  Reguła dotyczy rozdzielczości tej sesji.
- Pary B/C są opisowe. Nie są twierdzeniem o przyspieszeniu podziału w innym
  ruchu, innej liczbie klientów ani na innym sprzęcie.
- `basal benchmark` ma ruch zamknięty. Klient i serwer działają w jednym
  procesie. Pamięć jest próbkowana, a nie mierzona jako prawdziwy szczyt
  alokacji.
- Profile są diagnostyką sekcji forwardu, a nie latencją żądania. Nie
  obejmują okna `queue_ms`.
- Przegląd źródeł porównuje tekst. Nie mierzy kodu maszynowego ani zachowania
  bibliotek w czasie działania.
- Brak twierdzeń o `q8_0` i `q4_k` (L3) oraz o fizycznej trasie transferu.

## Dalsze kroki (nie są częścią tego raportu)

- Decyzja użytkownika o utrzymanym SIGNAL (wyżej). Według kryteriów
  zatrzymania planu jest to punkt raportu do koordynatora.
- Audyt ROOT tej sesji L2.
- Sesja L3 dla tych samych binarek, według kolejności planu i decyzji
  koordynatora.
- Końcowe niezależne review dopiero po pomiarach.

## Polecenia (ilustracyjne)

Rzeczywiste argv, ścieżki i środowisko zostają w prywatnych surowych
danych. `<base>`, `<candidate>`, `<model>`, `<out>` i `<table>` to symbole
zastępcze.

```sh
# benchmark: A (BASE, GPU 0), B (CAND, GPU 0), C (CAND, podział)
<base> --color never benchmark --model <model> --gemm-table crates/basal-cli/gemm-tables/<table> \
  --device 0 --scenario all --concurrency 1,16 --out <out>/0-A.json
<candidate> --color never benchmark --model <model> --gemm-table crates/basal-cli/gemm-tables/<table> \
  --layer-split 0:30,1:30 --scenario all --concurrency 1,16 --out <out>/2-C.json

# ab.py: 4 rundy abbaabba (A/B; B/C analogicznie, po stronie --b z --layer-split 0:30,1:30 zamiast --device 0)
python3 tools/bench/ab.py --rounds 4 --out <out>/l2-bench-ab \
  --a "<base> --color never bench --model <model> --gemm-table crates/basal-cli/gemm-tables/<table> --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16" \
  --b "<candidate> --color never bench --model <model> --gemm-table crates/basal-cli/gemm-tables/<table> --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16"

# profil (diagnostyka potwierdzenia), A i B
<base> --color never profile --model <model> --gemm-table crates/basal-cli/gemm-tables/<table> \
  --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --n 40 --batch 1
```

## Pliki

- `facts.json` — statusy i liczności, GAP i SIGNAL, protokół z definicją
  percentyli, pary A/B (ab.py: 4 pola i statystyki per pytanie;
  `benchmark`: 21 pól z A, B, B−A, B/A, % i oceną kierunku) dla rundy
  głównej i potwierdzenia, sygnały, wniosek, profile, analiza utrzymanego
  SIGNAL (ogon `mixed-c1` każdego przebiegu, grupy żądań, pary żądań o tym
  samym id, wynik end-to-end, porównanie źródeł), pary B/C (25 pól z C−B i
  C/B), przebiegi `benchmark` (pule, wolna pamięć, NVML, `load_s`,
  transfery, CPU procesu, telemetria per GPU), warunki runnera i kroki.
  Liczby mają pełne bity, a zaokrąglenia są tylko w README.
- `provenance.json` — sumy binarek, źródeł, runnera i helperów, modelu,
  tabeli GEMM, wejść i obciążenia, środowisko (karta, sterownik,
  biblioteki), akceptacja pary binarek, audyt bieżącego L1, sumy plików
  bieżącego raportu L1 i wcześniejszego raportu L2 (historia, nie dowód).
- `validation.json` — kontrole przeliczenia, spójności, niezmienności
  chronionych plików i prywatności tego raportu.
