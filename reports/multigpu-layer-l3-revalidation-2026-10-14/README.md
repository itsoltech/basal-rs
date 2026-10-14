# Etap 05, podział warstw — rewalidacja L3 bieżącego kandydata, 2026-10-14

Stan: raport danych z jednej zakończonej sesji L3 (pomiar 2026-10-14 UTC)
dla bieżącego kandydata etapu 05 (źródła `831e68e0…`, 66 plików). Sesja
mierzy regresję toru jednego GPU w trybach `q8_0` i `q4_k` parami ab.py i
opisuje kompromis podziału. Runner zgłosił jeden **utrzymany SIGNAL**:
`q4_k` `bench: dec_s`. Raport zawiera analizę tego wyniku z zebranych
surowych danych i przegląd źródeł tylko do odczytu. Nie znaleziono
przyczyny w kodzie. Według planu o tym konkretnym wyniku **decyduje
użytkownik**. Ta decyzja jeszcze nie zapadła. Raport nie akceptuje etapu 05,
nie jest końcowym review i nie mierzy jakości ani zgodności.

## Dlaczego nowy raport

[Wcześniejszy raport L3](../multigpu-layer-l3-2026-10-13/README.md) dotyczy
poprzedniego kandydata (źródła `77b629ed…`, binarka `dfd13b7e…`). Po tamtym
pomiarze zmieniły się źródła kandydata: doszły poprawka sprawdzania stanu
urządzeń po końcowym review R1 i integracja `main` (`4f68eb2`). Według
macierzy klas zmian planu to K3: nowa binarka unieważnia L1, L2 i L3.
[L1](../multigpu-layer-l1-revalidation-2026-10-13/README.md) i
[L2](../multigpu-layer-l2-revalidation-2026-10-14/README.md) dla tych źródeł
wykonano ponownie. Ten raport opisuje nowe, pełne L3 na tej samej parze
binarek. Wcześniejszy raport L3 pozostaje bez zmian. Jego liczby nie są
dowodem tego raportu.

## Wynik w skrócie

- **Runner:** `complete: true`, `stopped: null`, kod wyjścia 4 (według
  reguły runnera: brak FAIL, co najmniej jeden SIGNAL). 175 kontroli: PASS
  172, FAIL 0, GAP 2, SIGNAL 1. Liczności policzono ponownie z listy kontroli
  i z `checks.txt`. Jednostka L3 wystartowała i się zakończyła (163 PASS,
  1 GAP, 1 SIGNAL). Wcześniej było 10 kontroli wstępnych. Wszystkie 12
  kroków pomiaru zakończyło się kodem 0. W żadnym nie wykryto obcego procesu
  obliczeniowego ani braku próbek.
- **`q8_0`, A/B** (A = BASE na GPU 0, B = CAND na GPU 0): sygnał główny
  `dec_s` (B niższe w 4 z 4 par). Potwierdzenie: B niższe w 3 z 4 par, więc
  sygnał nie jest utrzymany. Wniosek (a): brak powtarzalnego spowolnienia
  przy zmierzonej rozdzielczości. To nie jest dowód braku narzutu. Przez obie
  rundy B ma niższe `dec_s` w 7 z 8 par (B/A 0,99731–1,00028).
- **`q4_k`, A/B:** sygnały główne `dec_s` i `lat2_ms` (B gorsze w 4 z 4
  par). Potwierdzenie: `dec_s` B gorsze w 4 z 4 par, `lat2_ms` w 3 z 4.
  **Utrzymany SIGNAL: `bench: dec_s`** (B niższe w 8 z 8 par).
  `lat2_ms` nie jest utrzymany. Nierozstrzygnięte: 0. Brakujące wartości: 0.
- **Wielkość SIGNAL:** `dec_s` B/A 0,994438–0,999895, czyli od −0,0105% do
  −0,5562%. W czasie jednego przebiegu 44 pytań to od +0,082 do +4,366 ms
  (mediana par +0,878 ms) przy 778,6–785,0 ms na przebieg. Rozrzut trzech
  przebiegów w jednym procesie sięga 3,96 ms. W 5 z 8 par różnica A/B jest
  mniejsza niż rozrzut przebiegów w każdym z dwóch procesów pary. To opis
  rozdzielczości, a nie uznanie różnicy za szum.
- **Przegląd źródeł (tylko do odczytu):** na ścieżce, którą mierzy `dec_s`
  dla jednego GPU w `q4_k`, zmiany CAND to księgowanie po stronie hosta
  (sprawdzenie stanu, właściciele bramek, porównania kontekstu i
  urządzenia) oraz zwolnienie tensorów odczytu przed pętlą hosta zamiast po
  niej. CAND nie dodaje synchronizacji GPU, alokacji, kopii ani kernela.
  Dekwantyzacja `q4_k` i czas życia jej tymczasowych wag f16 są bez zmian.
  `bench.rs`, przygotowanie i planowanie (`basal-core`) są tekstowo równe
  `d42a400`. **Nie znaleziono możliwej do wskazania przyczyny w kodzie.** To
  nie jest stwierdzenie, że przyczyny nie ma (niżej: czego analiza nie
  wyklucza).
- **Do decyzji użytkownika:** utrzymany SIGNAL `q4_k` `bench: dec_s` dla
  CAND `169aedfc…`. Decyzja nie jest podjęta i ten raport jej nie podejmuje.
  Przyjęcie wyniku L2 i uwaga o normalności 1–2% nie są tu regułą
  akceptacji.
- **Kompromis podziału** (B = CAND na GPU 0, C = CAND `0:30,1:30`, opisowo):
  w ab.py `dec_s` C/B 0,9160–0,9229 (`q8_0`) i 0,9152–0,9173 (`q4_k`),
  mediana C/B per pytanie 1,12804–1,13927 i 1,18348–1,20554. C jest
  szybsze w 0 z 39 pytań. HTTP z podziałem: po jednym przebiegu na tryb,
  każda faza z 0 błędów i 900 ukończonymi żądaniami.
- **`q4_k`:** jeden obiekt kwantyzacji w 25 ładowaniach, ze statystykami
  równymi faktom P0 tych samych wag.

## Zakres i pochodzenie

| Element | Wartość |
|---|---|
| Sprzęt | 2 × NVIDIA RTX 6000 Ada Generation, 49140 MiB każda, sterownik 580.126.09, limit mocy 300 W, tryb persistence |
| Zegar i CPU | żądany SM 1200 MHz (`nvidia-smi -pm 1`, `nvidia-smi -lgc 1200,1200`, oba kod 0), maska 16 z 26 logicznych CPU |
| Biblioteki | cuBLAS/cuBLASLt 12.9.1.4, cudart 12.9.79, tabele GEMM cuBLASLt 120901 |
| Środowisko procesów | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `HF_HUB_OFFLINE=1`, `BASAL_NO_UPDATE_CHECK=1`; katalog JIT CUDA pusty przed sesją (0 plików) |
| Okno kroków pomiaru (UTC) | 2026-10-14 05:06:01–05:51:23 |
| BASE | `44c8e399…`, przebudowane `d42a400`, źródła `94ef2131…` (65 plików) |
| CAND | `169aedfc…`, `d42a400` + zmiany etapu 05 + poprawka po review R1 + oczekujące scalenie `origin/main` `4f68eb2`, źródła `831e68e0…` (66 plików) |
| Build | te same binarki co w bieżącym L1 i L2: ten sam builder i wspólny przepis `2e37cb45…` (25 kroków), manifest buildu `2efaeca1…` (390 plików), akceptacja pary przez ROOT `cf05ffda…` |
| Runner / helpery | `tools/perf/run-layer-split-cloud.py` `81437a1c…`, `tools/bench/ab.py` `17b188dc…`; w L3 nie działał żaden proces helpera rezerwacji |
| Model i tabela | basal-1.5-4.5B `784a683b…` (60 warstw), f16, `q8_0` i `q4_k`, `fused`; tabela `nvidia-rtx-6000-ada-generation--basal-1.5-4.5B--f16--cublaslt120901.json` `f3712077…` |
| Wejścia | `basal bench`: `reports/reference-basal-1.5-4.5B-fp32` (44 pytania, latencja na pytaniach 5..43); `basal benchmark`: wbudowane obciążenie `a645db4c…` (900 żądań sekwencyjnych i 900 mieszanych); fakty P0 `reports/w4a16-p0-2026-10-12/facts.json` `44005ecc…` |
| Surowe dane | manifest SHA-256 `6bc97ffc6735dde633b96839f5ea5b6c20104d2d1d42ac4e7ebc6fbf8d9221a0`: 142 pliki, 10 katalogów, 5 146 785 B; zebrane i zweryfikowane bez błędów, ten sam rozruch maszyny, runner i wrapper zakończone |
| Bieżące L1 | PASS 599, FAIL 0, GAP 42, SIGNAL 0, C5 PASS; surowy manifest `285936c2…`; audyt ROOT `a08961af…` |
| Audyt ROOT tej sesji | `fc09bf8a…`: 301 kontroli bez uwag, 142 pliki; wyniki rund przeliczone tak samo jak w tym raporcie |

Pełne sumy są w `provenance.json`. Etykieta źródła BASE jest deklaracją z
rekordu buildu. Runner jej nie weryfikuje, więc pozostaje GAP (niżej).
Raport nie twierdzi, że binarki są odtwarzalne bit w bit. Bieżące drzewo
robocze ma ten sam odcisk źródeł co CAND (66 plików, `831e68e0…`, każda
linia równa surowej `source-files.sha256`). Po pomiarze, a przed udanym
zebraniem danych, wystąpiła przejściowa przerwa w połączeniu sieciowym. To
nie jest awaria silnika. Żadna metryka nie została przez to odfiltrowana ani
zmieniona.

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
- Według planu utrzymany SIGNAL wymaga analizy: profilu A/B i przeglądu
  ścieżki gorącej toru jednego etapu. Realna przyczyna oznacza poprawkę,
  review i rewalidację według macierzy klas zmian. Brak przyczyny oznacza
  decyzję użytkownika o tym konkretnym wyniku.
- Po jednym `basal benchmark --scenario all --concurrency 1,16` CAND z
  podziałem w `q8_0` i `q4_k`. To jedno wykonanie na tryb, bez pary i bez
  reguły (21 pól: 3 fazy × 7 metryk). Razem z 4 polami ab.py B/C to 25 pól
  opisowych.
- Plan z góry wyłącza HTTP (`basal benchmark`) nowego kandydata na jednym
  GPU w `q8_0` i `q4_k`. Dla tych trybów dowodem wydajności są tylko pary
  ab.py.

Wszystkie pola par, liczności, sygnały, wnioski i profile policzono ponownie
z surowych plików przebiegów. Są równe wartościom runnera
(`validation.json`).

## Regresja toru jednego GPU (A/B, ab.py)

W każdej z 16 par A i B mają tę samą dokładność (0,7955, 35 z 44) i te same
pytania. To kontrola spójności par, a nie ocena jakości. Każda para ma 39
unikalnych indeksów 5..43 ze skończonym `lat2_ms`. `load_s` nie jest polem
reguły.

| Tryb | Runda | Para | `lat2_ms` A / B | `lat2_p95_ms` A / B | `dec_s` A / B | mediana B/A per pytanie | B szybsze (z 39) | `load_s` A / B | Pola z B gorszym |
|---|---|---:|---|---|---|---:|---:|---|---|
| `q8_0` | główna | 0 | 42,350 / 42,320 | 49,741 / 49,543 | 56,394 / 56,345 | 0,99912 | 27 | 21,59 / 15,37 | `dec_s` |
| `q8_0` | główna | 1 | 42,423 / 42,321 | 49,651 / 49,744 | 56,539 / 56,387 | 0,99884 | 26 | 15,29 / 14,18 | `lat2_p95_ms`, `dec_s` |
| `q8_0` | główna | 2 | 42,356 / 42,271 | 49,591 / 49,522 | 56,562 / 56,421 | 1,00024 | 19 | 15,26 / 14,89 | `dec_s`, `per_question_median_ratio` |
| `q8_0` | główna | 3 | 42,381 / 42,250 | 49,719 / 49,483 | 56,469 / 56,421 | 0,99793 | 31 | 15,26 / 14,46 | `dec_s` |
| `q8_0` | potwierdzenie | 0 | 42,315 / 42,406 | 49,663 / 49,670 | 56,488 / 56,414 | 0,99991 | 20 | 15,54 / 14,95 | `lat2_ms`, `lat2_p95_ms`, `dec_s` |
| `q8_0` | potwierdzenie | 1 | 42,355 / 42,369 | 49,605 / 49,544 | 56,414 / 56,430 | 0,99841 | 35 | 14,13 / 14,68 | `lat2_ms` |
| `q8_0` | potwierdzenie | 2 | 42,386 / 42,310 | 49,604 / 49,582 | 56,441 / 56,397 | 0,99897 | 30 | 14,62 / 15,64 | `dec_s` |
| `q8_0` | potwierdzenie | 3 | 42,294 / 42,332 | 49,642 / 49,567 | 56,477 / 56,369 | 0,99998 | 20 | 13,75 / 15,84 | `lat2_ms`, `dec_s` |
| `q4_k` | główna | 0 | 40,991 / 41,041 | 48,456 / 48,612 | 56,507 / 56,411 | 1,00108 | 10 | 32,96 / 32,34 | `lat2_ms`, `lat2_p95_ms`, `dec_s`, `per_question_median_ratio` |
| `q4_k` | główna | 1 | 40,979 / 41,004 | 48,559 / 48,509 | 56,427 / 56,421 | 0,99975 | 23 | 33,25 / 29,34 | `lat2_ms`, `dec_s` |
| `q4_k` | główna | 2 | 41,125 / 41,304 | 48,608 / 48,812 | 56,364 / 56,051 | 1,00481 | 0 | 31,55 / 34,34 | `lat2_ms`, `lat2_p95_ms`, `dec_s`, `per_question_median_ratio` |
| `q4_k` | główna | 3 | 41,089 / 41,095 | 48,671 / 48,642 | 56,376 / 56,288 | 1,00002 | 19 | 35,47 / 34,87 | `lat2_ms`, `dec_s`, `per_question_median_ratio` |
| `q4_k` | potwierdzenie | 0 | 40,962 / 41,034 | 48,586 / 48,419 | 56,509 / 56,470 | 0,99905 | 26 | 33,40 / 31,76 | `lat2_ms`, `dec_s` |
| `q4_k` | potwierdzenie | 1 | 40,961 / 41,167 | 48,526 / 48,506 | 56,483 / 56,378 | 1,00170 | 11 | 31,06 / 33,99 | `lat2_ms`, `dec_s`, `per_question_median_ratio` |
| `q4_k` | potwierdzenie | 2 | 40,950 / 41,042 | 48,594 / 48,488 | 56,504 / 56,497 | 1,00001 | 19 | 32,04 / 33,63 | `lat2_ms`, `dec_s`, `per_question_median_ratio` |
| `q4_k` | potwierdzenie | 3 | 41,102 / 40,960 | 48,586 / 48,433 | 56,462 / 56,429 | 0,99858 | 34 | 31,64 / 33,31 | `dec_s` |

Liczba par z B gorszym i agregat ab.py (mediana `lat2_ms` pytania z 4 rund,
potem B/A; policzony ponownie z `per_item` i równy `ab.json`):

| Tryb | Runda | `lat2_ms` | `lat2_p95_ms` | `dec_s` | mediana B/A per pytanie | agregat ab.py B/A (min.–maks.) | B szybsze (z 39) |
|---|---|---:|---:|---:|---:|---|---:|
| `q8_0` | główna | 0 z 4 | 1 z 4 | 4 z 4 | 1 z 4 | 0,998935 (0,996635–1,002508) | 28 |
| `q8_0` | potwierdzenie | 3 z 4 | 1 z 4 | 3 z 4 | 0 z 4 | 0,999478 (0,997060–1,001314) | 24 |
| `q4_k` | główna | 4 z 4 | 2 z 4 | 4 z 4 | 3 z 4 | 1,000724 (0,997767–1,006361) | 13 |
| `q4_k` | potwierdzenie | 3 z 4 | 0 z 4 | 4 z 4 | 2 z 4 | 0,999896 (0,997047–1,004411) | 22 |

### Sygnały przez obie rundy

| Tryb | Pole | B gorsze (z 8 par) | Utrzymany | B/A (zakres) |
|---|---|---:|---|---|
| `q8_0` | `dec_s` | 7 | nie | 0,99731–1,00028 |
| `q4_k` | `dec_s` | 8 | tak (SIGNAL) | 0,99444–0,99990 |
| `q4_k` | `lat2_ms` | 7 | nie | 0,99653–1,00503 |

`q8_0` `dec_s` ma w potwierdzeniu parę 1 z B niegorszym (56,414 → 56,430),
więc nie jest utrzymany. `q4_k` `lat2_ms` ma w potwierdzeniu parę 3 z B
niegorszym (41,102 → 40,960). Oba te pola mają jednak B gorsze w 7 z 8 par.
Pomiar nie wyklucza więc małego narzutu także tam. Jedno pole, `q4_k`
`dec_s`, ma B gorsze w 8 z 8 par i jest utrzymanym SIGNAL.

## Utrzymany SIGNAL `q4_k` `bench: dec_s` — analiza

Analiza używa tylko zebranych surowych danych i źródeł w trybie do
odczytu. Nie uruchamia GPU, modelu ani binarek. Nie dodaje pomiarów,
symulacji ani progów.

### Co mierzy `dec_s`

Definicja z `crates/basal-cli/src/bench.rs` (plik tekstowo równy w
`d42a400` i CAND):

- `dec_s` liczy pierwsze 16 · 8 pytań referencji, tu wszystkie 44
  (`dec_s_questions: 44`). Pola latencji (`lat2_ms`, `lat2_p95_ms`, mediana
  per pytanie) używają innego zbioru: 39 wierszy (pytania 5..43) przy batch 1.
- Najpierw jedno nieliczone rozgrzanie pierwszych 16 pytań, potem 3
  liczone przebiegi. Każdy przebieg przygotowuje i uruchamia 44 pytania w
  grupach 16, 16 i 12 z `Batching::Budget` (porcje według budżetu tokenów w
  grupie). `dec_s` to mediana trzech wartości 44 / czas przebiegu.
- Zegar obejmuje `Engine::prepare` każdego pytania (renderowanie,
  tokenizacja, pakowanie) i `Engine::run` (wejścia hosta, forwardy z
  dekwantyzacjami `q4_k`, odczyt, obliczenia hosta).
- Kolejność w procesie: `all_items_s` (44 pytania, 32 na wywołanie, zaraz
  po ładowaniu), potem wiersze latencji, potem `dec_s`.

Mediany trzech przebiegów policzone ponownie z `dec_s_runs` są równe `dec_s`
każdego procesu.

### Osiem par: wartości i wpływ na czas

„Przebieg” to czas jednego przebiegu 44 pytań, który daje medianę `dec_s`
(44 / `dec_s`). Rozrzut to różnica najdłuższego i najkrótszego z trzech
przebiegów w procesie.

| Runda | Para | `dec_s` A | `dec_s` B | B−A (dec/s) | B/A | zmiana | przebieg A, ms | przebieg B, ms | B−A, ms | rozrzut 3 przebiegów A / B, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| główna | 0 | 56,50732 | 56,41085 | −0,09647 | 0,998293 | −0,1707% | 778,660 | 779,992 | +1,332 | 0,63 / 1,75 |
| główna | 1 | 56,42699 | 56,42107 | −0,00592 | 0,999895 | −0,0105% | 779,769 | 779,851 | +0,082 | 0,45 / 1,62 |
| główna | 2 | 56,36410 | 56,05061 | −0,31349 | 0,994438 | −0,5562% | 780,639 | 785,005 | +4,366 | 0,96 / 1,18 |
| główna | 3 | 56,37587 | 56,28809 | −0,08778 | 0,998443 | −0,1557% | 780,476 | 781,693 | +1,217 | 3,43 / 3,96 |
| potwierdzenie | 0 | 56,50918 | 56,47005 | −0,03913 | 0,999307 | −0,0693% | 778,635 | 779,174 | +0,540 | 0,26 / 0,85 |
| potwierdzenie | 1 | 56,48337 | 56,37773 | −0,10564 | 0,998130 | −0,1870% | 778,990 | 780,450 | +1,460 | 1,97 / 2,62 |
| potwierdzenie | 2 | 56,50446 | 56,49680 | −0,00766 | 0,999864 | −0,0136% | 778,700 | 778,805 | +0,106 | 1,80 / 0,52 |
| potwierdzenie | 3 | 56,46153 | 56,42929 | −0,03224 | 0,999429 | −0,0571% | 779,292 | 779,737 | +0,445 | 1,12 / 1,15 |

Trzy liczone przebiegi każdego procesu:

| Runda | Para | A: 3 przebiegi, dec/s | B: 3 przebiegi, dec/s |
|---|---:|---|---|
| główna | 0 | 56,507 / 56,464 / 56,510 | 56,418 / 56,292 / 56,411 |
| główna | 1 | 56,433 / 56,427 / 56,401 | 56,421 / 56,435 / 56,318 |
| główna | 2 | 56,305 / 56,374 / 56,364 | 56,131 / 56,047 / 56,051 |
| główna | 3 | 56,432 / 56,185 / 56,376 | 56,288 / 56,315 / 56,030 |
| potwierdzenie | 0 | 56,493 / 56,509 / 56,512 | 56,470 / 56,432 / 56,493 |
| potwierdzenie | 1 | 56,491 / 56,348 / 56,483 | 56,256 / 56,445 / 56,378 |
| potwierdzenie | 2 | 56,504 / 56,606 / 56,476 | 56,464 / 56,501 / 56,497 |
| potwierdzenie | 3 | 56,462 / 56,492 / 56,411 | 56,456 / 56,372 / 56,429 |

- B ma niższe `dec_s` we wszystkich 8 parach. Różnica czasu przebiegu wynosi
  +0,082 do +4,366 ms, mediana par +0,878 ms. Wszystkie 8 procesów A mieści
  się w 56,364–56,509 dec/s, a 8 procesów B w 56,051–56,497 dec/s. Zakresy
  się nakładają.
- Największa różnica (+4,366 ms, −0,5562%) jest w parze 2 rundy głównej.
  Wszystkie trzy przebiegi B tej pary (56,047–56,131 dec/s) są niższe od
  wszystkich trzech przebiegów A (56,305–56,374 dec/s).
- W 5 z 8 par różnica A/B jest mniejsza niż rozrzut przebiegów w każdym z
  dwóch procesów pary.

### Te same procesy: forward, przygotowanie, `all_items_s`, `load_s`

`lat2` forward i `lat2` prepare to mediany `lat2_forward_ms` i
`lat2_prepare_ms` z 39 wierszy latencji (batch 1, osobny etap procesu, poza
oknem `dec_s`). `all_items_s` to czas jednego przejścia 44 pytań zaraz po
ładowaniu (dłużej = gorzej). `q8_0` jest kontekstem tej samej sesji.

| Tryb | Runda | Para | `lat2` forward, mediana A / B ms | B−A | `lat2` prepare, mediana A / B ms | B−A | `all_items_s` A / B | `load_s` A / B |
|---|---|---:|---|---:|---|---:|---|---|
| `q4_k` | główna | 0 | 40,615 / 40,657 | +0,041 | 0,2836 / 0,2867 | +0,0031 | 0,73738 / 0,73572 | 32,96 / 32,34 |
| `q4_k` | główna | 1 | 40,632 / 40,657 | +0,025 | 0,3028 / 0,3093 | +0,0065 | 0,73445 / 0,73711 | 33,25 / 29,34 |
| `q4_k` | główna | 2 | 40,690 / 40,733 | +0,043 | 0,3249 / 0,4884 | +0,1635 | 0,73643 / 0,73948 | 31,55 / 34,34 |
| `q4_k` | główna | 3 | 40,701 / 40,671 | −0,030 | 0,3260 / 0,3645 | +0,0385 | 0,73780 / 0,73772 | 35,47 / 34,87 |
| `q4_k` | potwierdzenie | 0 | 40,625 / 40,630 | +0,005 | 0,3212 / 0,2786 | −0,0426 | 0,73485 / 0,73541 | 33,40 / 31,76 |
| `q4_k` | potwierdzenie | 1 | 40,632 / 40,664 | +0,032 | 0,2927 / 0,3737 | +0,0810 | 0,73481 / 0,73735 | 31,06 / 33,99 |
| `q4_k` | potwierdzenie | 2 | 40,616 / 40,659 | +0,044 | 0,3077 / 0,2616 | −0,0461 | 0,73477 / 0,73619 | 32,04 / 33,63 |
| `q4_k` | potwierdzenie | 3 | 40,699 / 40,652 | −0,047 | 0,2929 / 0,2635 | −0,0294 | 0,73720 / 0,73504 | 31,64 / 33,31 |
| `q8_0` | główna | 0 | 42,000 / 41,988 | −0,012 | 0,3011 / 0,2843 | −0,0168 | 0,73699 / 0,73799 | 21,59 / 15,37 |
| `q8_0` | główna | 1 | 42,033 / 41,931 | −0,102 | 0,2819 / 0,2763 | −0,0055 | 0,73747 / 0,73851 | 15,29 / 14,18 |
| `q8_0` | główna | 2 | 42,036 / 41,964 | −0,072 | 0,2594 / 0,2562 | −0,0032 | 0,73639 / 0,73821 | 15,26 / 14,89 |
| `q8_0` | główna | 3 | 42,031 / 41,907 | −0,123 | 0,2908 / 0,2595 | −0,0312 | 0,73878 / 0,73645 | 15,26 / 14,46 |
| `q8_0` | potwierdzenie | 0 | 42,012 / 41,937 | −0,075 | 0,2708 / 0,3031 | +0,0323 | 0,73849 / 0,73539 | 15,54 / 14,95 |
| `q8_0` | potwierdzenie | 1 | 42,028 / 41,935 | −0,093 | 0,2803 / 0,2594 | −0,0209 | 0,73698 / 0,73726 | 14,13 / 14,68 |
| `q8_0` | potwierdzenie | 2 | 42,036 / 41,967 | −0,070 | 0,2898 / 0,2778 | −0,0120 | 0,73822 / 0,73563 | 14,62 / 15,64 |
| `q8_0` | potwierdzenie | 3 | 42,002 / 41,955 | −0,047 | 0,2805 / 0,2772 | −0,0033 | 0,73555 / 0,73588 | 13,75 / 15,84 |

- `q4_k`: forward `lat2` B jest dłuższy w 6 z 8 par (−0,047 do +0,044 ms
  przy ok. 40,6–40,7 ms). Przygotowanie `lat2` B jest dłuższe w 5 z 8 par.
  `all_items_s` B jest dłuższe w 5 z 8 par. `load_s` B jest dłuższe w 4 z 8
  par.
- W parze 2 rundy głównej (największa różnica `dec_s`) mediana
  przygotowania `lat2` B wynosi 0,4884 ms wobec 0,3249 ms w A. Kod
  przygotowania jest tekstowo równy w obu binarkach. Raport nie ustala
  przyczyny tej różnicy i nie łączy jej przyczynowo z `dec_s`.
- `q8_0`: forward `lat2` B jest krótszy we wszystkich 8 parach (−0,123 do
  −0,012 ms), a `dec_s` B niższe w 7 z 8 par. W `q8_0` forward przy batch 1
  i `dec_s` mają więc przeciwne kierunki.

### Kontekst z tej i innych sesji (bez łączenia w pary)

| Źródło | Tryb | Pary z B niższym `dec_s` | B/A `dec_s` (zakres) | Status |
|---|---|---:|---|---|
| ta sesja L3 | `q8_0` | 7 z 8 | 0,99731–1,00028 | wniosek (a), sygnał nieutrzymany |
| ta sesja L3 | `q4_k` | 8 z 8 | 0,99444–0,99990 | utrzymany SIGNAL |
| L2 bieżące (te same binarki, inna sesja) | f16 `none` | 1 z 4 | 0,99817–1,00366 | bez sygnału `dec_s` |
| L3 2026-10-13 (inny kandydat, historyczne) | `q8_0` | 3 z 4 | 0,99760–1,00037 | wniosek (a) |
| L3 2026-10-13 (inny kandydat, historyczne) | `q4_k` | 6 z 8 | 0,99818–1,00365 | wniosek (a) |

Wiersze L2 i wcześniejszego L3 pochodzą z innych sesji. L2 mierzyło te same
binarki w f16. Wcześniejsze L3 mierzyło innego kandydata i inną binarkę BASE.
To kontekst, a nie dowód tego raportu. Przebiegów różnych sesji nie łączymy
w pary.

### Profile A/B, batch 1 (diagnostyka)

`basal profile --n 40 --batch 1`, średnio 239,8 tokenu na element. Najpierw
A, potem B, każdy w jednym procesie. Profil synchronizuje każdą sekcję i
każdą dekwantyzację, więc te czasy zawierają synchronizacje i nie są
latencją żądania ani przepustowością. Czas dekwantyzacji jest zawarty w
sekcjach projekcji. `residual_abs_max` wynosi 0,0 we wszystkich profilach.
Wartości w ms na element.

| Sekcja | `q8_0` A | `q8_0` B | `q4_k` A | `q4_k` B |
|---|---:|---:|---:|---:|
| `attention` | 6,762 | 6,679 | 6,736 | 6,726 |
| `bias+rope+heads` | 1,118 | 1,056 | 1,083 | 1,058 |
| `bias+silu*up` | 1,647 | 1,343 | 1,667 | 1,482 |
| `down_proj+bias+residual` | 57,381 | 37,623 | 23,277 | 28,013 |
| `embed+inputs` | 0,742 | 0,683 | 0,576 | 0,680 |
| `final_norm` | 0,028 | 0,011 | 0,276 | 0,011 |
| `gate_up_proj` | 146,771 | 55,501 | 143,009 | 92,350 |
| `o_proj+bias+residual+rms_norm` | 6,614 | 3,888 | 3,946 | 4,398 |
| `qkv_proj` | 7,491 | 4,094 | 4,040 | 4,079 |
| `rms_norm` | 5,059 | 0,564 | 3,236 | 6,310 |
| suma sekcji | 233,612 | 111,442 | 187,847 | 145,108 |
| `ms_per_item_profiled` | 233,734 | 111,538 | 187,964 | 145,217 |
| dekwantyzacja (zawarta w sekcjach projekcji) | 95,616 | 64,600 | 49,167 | 47,042 |

Profile `q4_k` dają A 187,964 i B 145,217 ms/element, a `q8_0` A 233,734 i
B 111,538 ms/element. W ab.py te same binarki mają przy batch 1 `lat2_ms`
40,950–41,304 ms (`q4_k`) i 42,250–42,423 ms (`q8_0`). Profil B jest
krótszy, a `dec_s` B niższe. Raport nie ustala przyczyny różnicy profili
i nie wyciąga z niej wniosku o szybkości. Profile nie wyjaśniają SIGNAL ani
go nie wykluczają. Wcześniejsze L3 (inny kandydat)
dało w `q4_k` bardzo podobne profile: 187,566 i 145,013 ms/element.

### Sonda GEMM (diagnostyka warunków)

`gpu_probe_tflops` mierzy krótki GEMM f16 tuż przed `bench` i tuż po nim, w
tym samym procesie. Funkcja sondy jest tekstowo równa w obu binarkach. Nie
jest to pomiar szybkości modelu.

| Runda | Strona | GPU | sonda przed, TFLOPS | sonda po, TFLOPS | wartości < 110 (z 8) |
|---|---|---:|---|---|---:|
| `q8_0` A/B główna | A (BASE) | 0 | 121,0–121,2 | 121,1–121,2 | 0 |
| `q8_0` A/B główna | B (CAND) | 0 | 121,0–121,2 | 104,7–121,2 | 2 |
| `q8_0` A/B potwierdzenie | A (BASE) | 0 | 120,9–121,2 | 121,2–121,2 | 0 |
| `q8_0` A/B potwierdzenie | B (CAND) | 0 | 121,0–121,1 | 104,5–104,8 | 4 |
| `q8_0` B/C | B (CAND, 1 GPU) | 0 | 121,0–121,1 | 104,6–121,2 | 2 |
| `q8_0` B/C | C (CAND, podział) | 0 | 121,1–121,2 | 121,1–121,3 | 0 |
| `q8_0` B/C | C (CAND, podział) | 1 | 121,1–121,2 | 121,0–121,1 | 0 |
| `q4_k` A/B główna | A (BASE) | 0 | 120,9–121,1 | 104,6–104,9 | 4 |
| `q4_k` A/B główna | B (CAND) | 0 | 121,1–121,3 | 121,1–121,2 | 0 |
| `q4_k` A/B potwierdzenie | A (BASE) | 0 | 120,8–121,0 | 104,6–104,7 | 4 |
| `q4_k` A/B potwierdzenie | B (CAND) | 0 | 121,1–121,2 | 121,1–121,3 | 0 |
| `q4_k` B/C | B (CAND, 1 GPU) | 0 | 120,9–121,2 | 121,1–121,3 | 0 |
| `q4_k` B/C | C (CAND, podział) | 0 | 121,1–121,2 | 121,1–121,2 | 0 |
| `q4_k` B/C | C (CAND, podział) | 1 | 104,0–104,2 | 120,8–121,0 | 4 |

Wartości tworzą dwie grupy: 104,0–104,9 TFLOPS (20 ze 112 odczytów) i
120,8–121,3 TFLOPS. Niskie wartości występują w procesach BASE i CAND, przed
pomiarem i po nim, na GPU 0 i GPU 1. W rundach `q4_k` A/B sonda po
przebiegu jest niska we wszystkich 8 procesach A i w żadnym procesie B. W
potwierdzeniu `q8_0` jest odwrotnie (4 z 4 procesów B). Raport nie wyjaśnia
tych grup i nie łączy ich z wynikami par.

### Warunki par `q4_k` (dane runnera, GPU 0)

| Runda | Para | A: temp. maks. °C / SM min. MHz / moc śr. W / wykorz. % | B: temp. maks. °C / SM min. MHz / moc śr. W / wykorz. % |
|---|---:|---|---|
| główna | 0 | 61 / 1200 / 94,2 / 22,7 | 62 / 1200 / 98,8 / 26,1 |
| główna | 1 | 62 / 1200 / 97,5 / 24,2 | 62 / 1200 / 102,2 / 26,1 |
| główna | 2 | 62 / 1200 / 99,4 / 24,2 | 62 / 1200 / 96,4 / 23,9 |
| główna | 3 | 62 / 1200 / 95,6 / 23,1 | 62 / 1200 / 96,1 / 23,8 |
| potwierdzenie | 0 | 62 / 1200 / 94,6 / 23,2 | 62 / 1200 / 100,0 / 26,3 |
| potwierdzenie | 1 | 62 / 1200 / 99,3 / 24,4 | 62 / 1200 / 97,1 / 22,9 |
| potwierdzenie | 2 | 62 / 1200 / 99,0 / 26,7 | 62 / 1200 / 98,1 / 24,7 |
| potwierdzenie | 3 | 62 / 1200 / 99,4 / 25,6 | 62 / 1200 / 97,4 / 24,1 |

Zegar SM nie spadał poniżej 1200 MHz. Temperatura maks. wynosiła 61–62 °C.
GPU 0 było przez prawie cały czas rund w stanie P2 przy zegarze pamięci
9501 MHz. Warunki A i B są podobne. Telemetria dotyczy tylko GPU i nie
obejmuje CPU hosta. Warunki niczego nie filtrują.

### Ścieżka gorąca toru jednego etapu: `d42a400` wobec CAND

Porównanie tekstu źródeł z obiektów git `d42a400` i bieżącego drzewa (to
samo co CAND, odcisk `831e68e0…`). Element zaczyna się od sygnatury, a
kończy na zamykającym nawiasie z tym samym wcięciem. Liczby linii pochodzą z
`difflib`.

| Plik | Element | Gdzie względem okna `dec_s` | d42a400 = CAND |
|---|---|---|---|
| `crates/basal-core/src/engine.rs` | `pub fn prepare(&self, item: Item) -> Result<Prepared>` | dec_s window: host preparation of an item | tak |
| `crates/basal-core/src/engine.rs` | `pub fn prepare(tok: &BasalTokenizer, bos: &str, item: Item) -> Result<Prepared>` | dec_s window: rendering, tokenisation, packing | tak |
| `crates/basal-core/src/engine.rs` | `pub fn run(&mut self, prepared: &[Prepared], batching: Batching) -> Result<Vec<ItemResult>>` | dec_s window: budget chunks, one Backend::letter_logits per chunk | tak |
| `crates/basal-core/src/engine.rs` | `fn finish(&self, p: &Prepared, letter_logits: Vec<Vec<f32>>) -> ItemResult` | dec_s window: host math per item | tak |
| `crates/basal-gpu/src/lib.rs` | `fn letter_logits(&mut self, rows: &[&Packed], letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>>` | dec_s window: entry of every forward | **nie** (−139 / +5 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn forward_ragged(` | dec_s window: every forward | **nie** (−166 / +29 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn matmul_t(` | dec_s window: 4 projections per layer, 240 per forward | **nie** (−3 / +5 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn dequantize_blocks(&self, q: &candle_core::quantized::QTensor) -> Result<Tensor>` | dec_s window: q4_k -> f16 temporary of every projection (non-profiled branch first) | **nie** (−3 / +4 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn residual_norm(` | dec_s window: 2 per layer | **nie** (−2 / +12 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn w(&self, t: &Tensor) -> Result<Tensor>` | dec_s window | tak |
| `crates/basal-gpu/src/lib.rs` | `fn rope_row(&self, pos: f64, cos: &mut [f32], sin: &mut [f32])` | dec_s window: host RoPE rows | tak |
| `crates/basal-gpu/src/lib.rs` | `fn best_prefix(&self, r: &Packed) -> Option<usize>` | dec_s window: template prefix per row | tak |
| `crates/basal-gpu/src/lib.rs` | `fn cache_state(` | dec_s window: returns early (state cache 0 bytes in bench) | **nie** (−4 / +4 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn tc_attention(&self) -> bool` | dec_s window | **nie** (−1 / +1 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn tree_attention(&self) -> bool` | dec_s window | **nie** (−1 / +1 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn mark(` | dec_s window: no-op without profiling (profile only) | **nie** (−2 / +2 linii) |
| `crates/basal-gpu/src/lib.rs` | `fn row_units(` | dec_s window: tree-attention units per row | tak |
| `crates/basal-gpu/src/lib.rs` | `pub fn gpu_device(device: usize) -> Result<Device>` | load only (event tracking off in both) | tak (różni się tylko komentarz) |
| `crates/basal-gpu/src/lib.rs` | `pub fn gemm_microbench(` | outside dec_s: the GEMM probe before and after bench::bench | tak |
| `crates/basal-gpu/src/fused.rs` | `pub fn residual_rmsnorm(` | dec_s window: 2 per layer | **nie** (−1 / +1 linii) |
| `crates/basal-gpu/src/fused.rs` | `impl CustomOp3 for ResidualRmsNorm` | dec_s window: launch of the fused residual + RMSNorm | **nie** (−0 / +9 linii) |
| `crates/basal-gpu/src/fused.rs` | `pub fn bias_silu_mul(` | dec_s window | tak |
| `crates/basal-gpu/src/fused.rs` | `pub fn qkv_rope(` | dec_s window (non tensor-core path) | tak |
| `crates/basal-gpu/src/fused.rs` | `pub fn qkv_rope_hilo(` | dec_s window | tak |
| `crates/basal-gpu/src/fused.rs` | `pub fn attention_tree(` | dec_s window | tak |
| `crates/basal-gpu/src/fused.rs` | `pub fn letter_logits(hidden: &Tensor, lm_head: &Tensor, ids: &Tensor) -> Result<Tensor>` | dec_s window: readout | tak |
| `crates/basal-gpu/src/cublaslt.rs` | `pub fn matmul_nt(lt: &Lt, x: &Tensor, w: &Tensor) -> Result<Tensor>` | dec_s window: every projection | tak |
| `crates/basal-gpu/src/cublaslt.rs` | `fn matmul(` | dec_s window: every projection (host side of the cuBLASLt call) | **nie** (−0 / +8 linii) |
| `crates/basal-gpu/src/cublaslt.rs` | `fn dispatch(` | dec_s window: every projection | tak |
| `crates/basal-cli/src/main.rs` | `Cmd::Bench { m, g, reference, out, lat_n, batching } =>` | around bench::bench: GEMM probe before/after, JSON fields | **nie** (−5 / +15 linii) |
| `crates/basal-cli/src/main.rs` | `fn gpu_engine(m: &ModelArgs, g: &GpuArgs) -> Result<Engine<GpuBackend>>` | load only | **nie** (−1 / +2 linii) |
| `crates/basal-cli/src/main.rs` | `fn gpu_engine_at(dir: &Path, g: &GpuArgs) -> Result<Engine<GpuBackend>>` | load only | **nie** (−10 / +24 linii) |
| `crates/basal-cli/src/bench.rs` | cały plik | — | tak |
| `crates/basal-cli/src/stats.rs` | cały plik | — | tak |
| `crates/basal-cli/src/reference.rs` | cały plik | — | tak |
| `crates/basal-core/src/pack.rs` | cały plik | — | tak |
| `crates/basal-core/src/decision.rs` | cały plik | — | tak |
| `crates/basal-gpu/src/quant.rs` | cały plik | — | **nie** |
| `crates/basal-gpu/src/fused.rs` | cały plik | — | **nie** |
| `crates/basal-gpu/src/cublaslt.rs` | cały plik | — | **nie** |
| `crates/basal-core/src/gate.rs` | cały plik | — | **nie** |
| `Cargo.toml` | cały plik | — | tak |
| `crates/basal-gpu/Cargo.toml` | cały plik | — | tak |
| `crates/basal-core/Cargo.toml` | cały plik | — | tak |
| `crates/basal-gpu/build.rs` | cały plik | — | tak |

Co się zmieniło na tej ścieżce (z lektury kodu, bez pomiaru ich czasu):

- **`letter_logits`** jest teraz opakowaniem. Sprawdza stan modelu (jeden
  odczyt atomowy), kopiuje właścicieli bramek (dwa jednoelementowe wektory,
  kopie uchwytu urządzenia i stanu) i woła `letter_logits_gated`. Ta funkcja
  ma przeniesione ciało z tymi samymi krokami: prefiksy, wiersze RoPE,
  segmenty, forward, odczyt i pętla hosta. Doszły `switch_to(0)`, `head()` i
  `mark_drained()`. Bez bramki i przy jednym etapie `leave()` kończy bez
  synchronizacji.
- **`forward_ragged`** buduje te same wejścia hosta i woła `run_stage` raz
  na etap (przy jednym GPU raz). Pętla warstw przeszła do `stage_layers` z
  tymi samymi wywołaniami kerneli w tej samej kolejności. Na warstwę doszły:
  indeks globalny, `p.part(0)` dla prefiksu, sprawdzenie położenia aktywne
  tylko przy podziale i `gates.checkpoint()` (bez bramki: bez
  synchronizacji).
- **`matmul_t`** bierze uchwyt cuBLASLt z wag etapu. Tymczasowa waga f16 z
  dekwantyzacji pozostaje zmienną lokalną tego wywołania i jest zwalniana
  przy jego końcu, jak w `d42a400`.
- **`dequantize_blocks`**: zmieniła się tylko gałąź profilowania
  (synchronizuje `q.device()` zamiast `self.dev`, to samo urządzenie). Gałąź
  bez profilowania, `q.dequantize_f16(&q.device())`, jest bez zmian.
- **cuBLASLt `matmul`** (240 razy na forward) i **fused residual + RMSNorm**
  (120 razy na forward) mają nowe sprawdzenia hosta: równość kontekstu
  (porównanie trzech pól, bez wywołania sterownika) i porównanie położenia
  urządzenia.
- **Kolejność zwolnień:** tensory odczytu (końcowe stany ukryte, id liter,
  logity) są zwalniane na końcu zakresu odczytu, po synchronicznej kopii do
  hosta, a przed pętlą hosta po literach. W `d42a400` zwalniane były na końcu
  `letter_logits`. W obu wersjach to zwolnienia w kolejności strumienia po
  tej samej synchronizacji.
- **Nie doszło** na tej ścieżce: synchronizacja GPU, alokacja, memset,
  kopia, wiązanie kontekstu ani nowy kernel. `bench.rs`, `stats.rs`,
  `reference.rs`, `pack.rs`, `decision.rs` i funkcje przygotowania w
  `engine.rs` są równe `d42a400`. `Cargo.toml` całego workspace, `build.rs`
  i manifesty `basal-gpu`/`basal-core` są bez zmian. W `Cargo.lock` zmienił
  się tylko parser YAML i jego zależności, bez zmian candle.
- **Poza oknem `dec_s`:** sonda GEMM przechodzi po urządzeniach przebiegu
  (jedno przy jednym GPU). Ładowanie idzie przez `load_stages` z jednym
  etapem. `load_s` CAND kończy się teraz przed wczytaniem głowy evidence,
  utworzeniem uchwytu cuBLASLt i synchronizacją publikacji (w `d42a400` po
  nich). Pola `load_s` A i B nie mierzą więc dokładnie tego samego.

### Wniosek analizy

Zebrane dane i przegląd źródeł nie wskazują przyczyny utrzymanego SIGNAL w
kodzie CAND. Na ścieżce mierzonej przez `dec_s` CAND dodaje księgowanie hosta
na forward, warstwę i projekcję oraz zmienia chwilę zwolnienia tensorów
odczytu, ale nie dodaje pracy GPU ani synchronizacji. Nie zmienia też
dekwantyzacji `q4_k`. Różnica jest mała (do 4,366 ms na
przebieg 44 pytań), ma ten sam znak we wszystkich 8 parach i jest
porównywalna z rozrzutem przebiegów w procesie. Raport **nie** proponuje
poprawki, bo nie ma wskazanej przyczyny, i **nie** uznaje różnicy za szum.

Czego ta analiza nie wyklucza:

- wpływu całej binarki CAND: innego układu i inliningu kodu, innych ścieżek
  instrukcji hosta;
- czasu wymienionych dodatków hosta, którego sesja nie mierzyła;
- stanu alokatora i sterownika (pula strumieniowa, mapowanie tymczasowych
  wag) po innym kodzie; sesja nie zapisuje alokacji ani czasów hosta w oknie
  `dec_s`;
- stanu CPU hosta i szeregowania wątków; telemetria dotyczy tylko GPU;
- zachowania przy innych modelach, kartach, kształtach batcha, dłuższych
  przebiegach albo większej liczbie par.

### Decyzja użytkownika (oczekująca)

Według planu („brak przyczyny → decyzja użytkownika o tym konkretnym
wyniku”) potrzebna jest decyzja o utrzymanym SIGNAL `q4_k` `bench: dec_s`
dla CAND `169aedfc…` (źródła `831e68e0…`) wobec BASE `44c8e399…`. `q4_k`
jest opcjonalnym trybem eksperymentalnym. Wielkości do decyzji, A → B:
56,50732 → 56,41085, 56,42699 → 56,42107, 56,36410 → 56,05061, 56,37587 →
56,28809 (runda główna) oraz 56,50918 → 56,47005, 56,48337 → 56,37773,
56,50446 → 56,49680, 56,46153 → 56,42929 dec/s (potwierdzenie). To od
−0,0105% do −0,5562%, czyli +0,082 do +4,366 ms na przebieg 44 pytań.
Możliwe rozstrzygnięcia, bez rekomendacji tego raportu:

1. przyjąć ten konkretny wynik dla tej binarki CAND jako zmierzony i
   opisany, z ograniczeniami wyżej;
2. zlecić dalszą diagnostykę; nowe pomiary wymagają zrecenzowanej rewizji
   planu i runnera oraz nowej sesji na wynajętym GPU według planu i
   `docs/CLOUD.md`;
3. uznać wynik za blokujący i wstrzymać etap do wskazania przyczyny.

Decyzja użytkownika o L2 dotyczyła tylko tamtego wyniku i nie przenosi się
na ten. Uwaga o normalności różnic 1–2% nie rozstrzyga tego wyniku
automatycznie. Do czasu decyzji SIGNAL pozostaje otwarty.

## Kompromis podziału (B/C, opisowo)

### ab.py (`basal bench`)

B = CAND na GPU 0, C = CAND `--layer-split 0:30,1:30`, ten sam tryb. Pary
nie mają reguły ani werdyktu. C−B jest w `facts.json`.

| Tryb | Para | `lat2_ms` B / C | `lat2_p95_ms` B / C | `dec_s` B / C | mediana C/B per pytanie | C szybsze (z 39) | `load_s` B / C |
|---|---:|---|---|---|---:|---:|---|
| `q8_0` | 0 | 42,365 / 49,502 | 49,599 / 56,290 | 56,17 / 51,45 | 1,13808 | 0 | 15,25 / 15,52 |
| `q8_0` | 1 | 42,247 / 48,893 | 49,533 / 56,196 | 56,44 / 52,05 | 1,13927 | 0 | 14,96 / 15,90 |
| `q8_0` | 2 | 42,399 / 48,887 | 49,548 / 55,994 | 56,38 / 52,03 | 1,13375 | 0 | 13,30 / 14,46 |
| `q8_0` | 3 | 42,376 / 48,886 | 49,577 / 56,138 | 56,43 / 52,03 | 1,12804 | 0 | 13,91 / 15,82 |
| `q4_k` | 0 | 40,993 / 49,543 | 48,525 / 57,177 | 56,36 / 51,70 | 1,18348 | 0 | 35,63 / 32,68 |
| `q4_k` | 1 | 41,045 / 49,655 | 48,426 / 57,344 | 56,42 / 51,68 | 1,18462 | 0 | 32,69 / 31,42 |
| `q4_k` | 2 | 41,012 / 49,677 | 48,476 / 57,369 | 56,36 / 51,63 | 1,20162 | 0 | 33,28 / 32,07 |
| `q4_k` | 3 | 41,030 / 49,686 | 48,607 / 57,202 | 56,43 / 51,65 | 1,20554 | 0 | 32,72 / 31,27 |

B i C mają tę samą dokładność i te same pytania. `basal bench` nie raportuje
pul ani transferów.

### HTTP CAND z podziałem (po jednym przebiegu)

`basal benchmark --scenario all --concurrency 1,16`, CAND
`--layer-split 0:30,1:30`, osobno `q8_0` i `q4_k`. W każdej z 3 faz obu
przebiegów: 0 błędów, `complete`, 900 zaplanowanych i ukończonych żądań, 0
zmian odpowiedzi. Bez powtórzeń i bez pary z jednym GPU, więc tabela nie jest
porównaniem.

| Pole | C `q8_0` | C `q4_k` |
|---|---:|---:|
| `sequential-c1 successful_latency_ms.p50` | 51,71 | 51,62 |
| `sequential-c1 successful_latency_ms.p95` | 65,43 | 65,34 |
| `sequential-c1 successful_latency_ms.p99` | 73,77 | 73,47 |
| `sequential-c1 queue_ms.p50` | 0,3855 | 0,5520 |
| `sequential-c1 queue_ms.p99` | 1,0180 | 1,2740 |
| `sequential-c1 successful_requests_per_second` | 18,12 | 18,31 |
| `sequential-c1 successful_questions_per_second` | 18,12 | 18,31 |
| `mixed-c1 successful_latency_ms.p50` | 71,47 | 70,45 |
| `mixed-c1 successful_latency_ms.p95` | 1154,00 | 1149,26 |
| `mixed-c1 successful_latency_ms.p99` | 3543,67 | 3269,03 |
| `mixed-c1 queue_ms.p50` | 0,8910 | 0,9595 |
| `mixed-c1 queue_ms.p99` | 88,30 | 103,32 |
| `mixed-c1 successful_requests_per_second` | 4,1313 | 4,2583 |
| `mixed-c1 successful_questions_per_second` | 9,4928 | 9,7847 |
| `mixed-c16 successful_latency_ms.p50` | 592,99 | 584,50 |
| `mixed-c16 successful_latency_ms.p95` | 13706,03 | 13586,47 |
| `mixed-c16 successful_latency_ms.p99` | 41641,42 | 41286,19 |
| `mixed-c16 queue_ms.p50` | 235,98 | 241,88 |
| `mixed-c16 queue_ms.p99` | 38183,07 | 37623,65 |
| `mixed-c16 successful_requests_per_second` | 6,2920 | 6,3276 |
| `mixed-c16 successful_questions_per_second` | 14,46 | 14,54 |

### Pamięć per GPU

| Przebieg | GPU | pula po ładowaniu (used / reserved) | pula szczyt ładowania | pula szczyt pracy | pula po zamknięciu | min. wolna (próbki) | NVML `memory.used` maks. | szacunek wag etapu |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| C `q8_0` | 0 | 2543,6 / 2592,0 | 2630,5 | 5097,7 | 0,0 | 42678 | 5832 | 2469,8 |
| C `q8_0` | 1 | 2551,6 / 2656,0 | 2637,9 | 5160,5 | 0,0 | 42454 | 6056 | 2477,8 |
| C `q4_k` | 0 | 1441,1 / 1568,0 | 1528,0 | 3995,2 | 0,0 | 43766 | 4744 | 1367,3 |
| C `q4_k` | 1 | 1449,1 / 1504,0 | 1535,4 | 4058,0 | 0,0 | 43606 | 4904 | 1375,3 |

Wszystkie wartości są w MiB. Źródła mają różne zakresy i ich nie mieszamy:

- **Pula** to pula alokatora strumieniowego CUDA tego procesu
  (`cuMemAllocAsync`) na danym urządzeniu: tensory, rezydentne bloki
  `q8_0`/`q4_k`, tymczasowe wagi f16, przestrzenie robocze cuBLASLt i
  prefiksy. Nie obejmuje kontekstu CUDA, modułów, uchwytów bibliotek ani
  alokacji synchronicznych.
- **Min. wolna** to najmniejsza wolna pamięć urządzenia z próbek co 200 ms.
  Obejmuje wszystkie procesy na karcie.
- **NVML** to `memory.used` z telemetrii przebiegu (co 200 ms, poziom
  urządzenia).
- **Szacunek wag** wynika z arytmetyki formatu i nie jest pomiarem.

### `load_s` i transfery między etapami

| Przebieg | `load_s` | Kierunek | Kopie | Bajty | `transfer_ms` | `gate_wait_ms` |
|---|---:|---|---:|---:|---:|---:|
| C `q8_0` | 14,991 | etap 0 → 1 (GPU 0 → 1) | 2 420 | 22 976 454 656 | 12941,208 | 14505,163 |
| C `q4_k` | 30,359 | etap 0 → 1 (GPU 0 → 1) | 2 413 | 22 932 520 960 | 14483,341 | 14266,706 |

`transfer_ms` to suma czasu hosta udanych transferów granicy. Obejmuje
alokację i inicjalizację celu, opróżnienie, kopię API urządzenie–urządzenie,
potwierdzenie na obu GPU, zwolnienie źródła i wiązanie kontekstu.
`gate_wait_ms` to branie bramki następnego GPU i wiązanie jego kontekstu.
Fizyczna trasa kopii (P2P albo przez hosta) nie jest ustalona.

## Obiekt kwantyzacji `q4_k`

Runner zapisał kanoniczną sumę SHA-256 opisu `q4_k` z każdego ładowania: 24
procesy ab.py (A/B, potwierdzenie, B/C) i przebieg HTTP. Wszystkie 25 sum
jest równych, czyli jest to jeden obiekt. Ten raport policzył te sumy
ponownie z surowych plików. Statystyki modelu i 4 typów projekcji są w każdym
ładowaniu równe faktom P0 tych samych wag (R 0,005808291323302594, 9
spłaszczonych grup ze 144 506 880). Każde ładowanie ma też bitowo równą
kontrolę rozgrzewki macierzy `o` warstwy 0 (4 194 304 wartości). To
własność wczytanych wag, a nie pomiar jakości.

## Jakość i zgodność (kontekst, bez nowego pomiaru)

L3 nie mierzy jakości, zbioru 900 pytań ani porównania z FP32 upstream.
Decyzje jakościowe pozostają bez zmian:
[Q8](../../.itsol/initiatives/basal-multigpu/phases/03-w8a16/quality-decision.md)
jest opcjonalnym trybem z jawnymi różnicami jakości,
[Q4](../../.itsol/initiatives/basal-multigpu/phases/04-w4a16/quality-decision.md)
opcjonalnym trybem eksperymentalnym z jawnymi różnicami, a domyślny tryb to
`none`. Zgodność CAND z BASE w tym samym trybie pokazało bieżące
[L1](../multigpu-layer-l1-revalidation-2026-10-13/README.md). L3 tej
zgodności nie mierzy ponownie.

## Warunki pomiaru (opisowo)

Telemetria `nvidia-smi` co 200 ms, osobny plik każdego kroku ab.py i
każdego przebiegu HTTP, obie karty.

| Krok | GPU 0: temp. maks. °C / SM min. MHz / moc maks. W | GPU 1: temp. maks. °C / SM min. MHz / moc maks. W |
|---|---|---|
| `l3-q8-bench-ab` | 65 / 1200 / 283,7 | 43 / 1200 / 33,0 |
| `l3-q8-bench-ab-confirmation` | 66 / 1200 / 285,1 | 44 / 1200 / 33,3 |
| `l3-q8-bench-single-split` | 64 / 1200 / 281,4 | 56 / 1200 / 156,9 |
| `l3-q4-bench-ab` | 62 / 1200 / 278,1 | 44 / 1200 / 33,5 |
| `l3-q4-bench-ab-confirmation` | 62 / 1200 / 278,3 | 44 / 1200 / 33,4 |
| `l3-q4-bench-single-split` | 62 / 1200 / 277,8 | 56 / 1200 / 155,7 |
| `l3-split-benchmark/0-C-q8.json` | 69 / 1140 / 242,7 | 73 / 1170 / 249,9 |
| `l3-split-benchmark/1-C-q4.json` | 69 / 1170 / 243,4 | 73 / 1170 / 249,7 |

- Zegar SM był ustawiony na 1200 MHz. W krokach ab.py nie spadał poniżej
  1200 MHz. W przebiegach HTTP spadał do 1140–1170 MHz.
- GPU 1 w krokach A/B nie jest używane (moc maks. 33,0–33,5 W). W krokach
  B/C i HTTP pracuje jako etap 1.
- Runner zapisał okna GPU 0 każdego przebiegu. Żadne nie ma braku
  telemetrii. Warunki niczego nie filtrują, a raport nie przypisuje im
  przyczyny wyniku. Telemetria nie obejmuje CPU hosta.

## GAP i SIGNAL (bez zmiany nazw)

| Status | Indeks | Kontrola | Treść |
|---|---:|---|---|
| GAP | 4 | „base binary: declared source label” | declared by the build record (base: d42a400 rebuilt, exact commit d42a400d9eb1c0e83957067868859afead584477, source digest 94ef2131, 65 files, phase-05 layer-placement paired builder r3); the runner does not verify the base's source (the parent's build audit) |
| GAP | 173 | „perf L3: the new candidate's single-GPU q8_0/q4_k HTTP benchmark (c1/c16)” | not measured by the plan: S3 describes only d42a400 and is not evidence for this candidate; for these modes the ab.py pairs are the only evidence |
| SIGNAL | 113 | „perf L3 q4_k: persistent regression signal (B worse in every pair of both rounds)” | ['bench: dec_s']: analysis (pairs, per-question, conditions, profiles, the changed path); a real cause -> fix and revalidation by the change-class matrix; none -> the user's decision on this result |

GAP pozostaje GAP-em, a SIGNAL pozostaje SIGNAL-em. Późniejszy wynik ich
nie usuwa. Raport nie twierdzi więc o braku regresji HTTP CAND na jednym GPU
w `q8_0`/`q4_k`.

## Związek z L1 i L2

Bieżące L1 (`reports/multigpu-layer-l1-revalidation-2026-10-13`) objęło te
same binarki i źródła: równość bitową, odmowy, HTTP, pojemność i profile
transferów (PASS 599, FAIL 0, GAP 42, SIGNAL 0, C5 PASS). Runner L3
sprawdził przed pomiarem, że L1 użyło tego samego runnera i helperów.
Bieżące L2 (`reports/multigpu-layer-l2-revalidation-2026-10-14`) mierzyło
tor domyślny `none` i ma utrzymany SIGNAL `mixed-c1 queue_ms.p99`.
Użytkownik przyjął ten konkretny wynik L2 i dopuścił L3
([DECISION.md](../multigpu-layer-l2-revalidation-2026-10-14/DECISION.md)).
Ta decyzja nie ustanawia progu i nie dotyczy wyniku L3. L3 nie przenosi
wniosków L1 ani L2 na tryby kwantyzowane i nie łączy przebiegów różnych
sesji w pary.

## Ograniczenia

- Jedna sesja, jedna maszyna, dwie identyczne karty RTX 6000 Ada 48 GB,
  model 4.5B, `q8_0` i `q4_k`. Nie wnioskujemy o innych kartach, modelach,
  Metal ani Intel.
- Utrzymany SIGNAL jest opisany, a nie wyjaśniony. Brak wskazanej przyczyny
  w kodzie nie dowodzi braku narzutu.
- Pola bez utrzymanego sygnału nie dowodzą braku narzutu toru jednego GPU.
  Reguła dotyczy rozdzielczości tej sesji.
- HTTP CAND na jednym GPU w `q8_0`/`q4_k` nie był mierzony (GAP). HTTP z
  podziałem ma po jednym przebiegu na tryb i nie jest porównaniem.
- Pary B/C są opisowe. Nie są twierdzeniem o szybkości podziału w innym
  ruchu, innej liczbie klientów ani na innym sprzęcie.
- Różnica profili, dwie grupy wartości sondy GEMM i różnice `load_s` nie
  mają ustalonej przyczyny.
- Przegląd źródeł porównuje tekst. Nie mierzy kodu maszynowego, czasu
  operacji hosta ani zachowania bibliotek w czasie działania.
- `basal benchmark` ma ruch zamknięty. Klient i serwer działają w jednym
  procesie. Pamięć jest próbkowana, a nie mierzona jako prawdziwy szczyt
  alokacji.
- Brak twierdzeń o jakości, zgodności (poza odnośnikiem do L1) i o fizycznej
  trasie transferu.

## Dalsze kroki (nie są częścią tego raportu)

- Decyzja użytkownika o utrzymanym SIGNAL (wyżej). Według kryteriów
  zatrzymania planu jest to punkt raportu do koordynatora.
- Akceptacja etapu i końcowe niezależne review dopiero po tej decyzji i po
  pomiarach.

## Polecenia (ilustracyjne)

Rzeczywiste argv, ścieżki i środowisko zostają w prywatnych surowych
danych. `<base>`, `<candidate>`, `<models>`, `<out>`, `<table>` i `<tryb>`
to symbole zastępcze (`<tryb>` = `q8_0` albo `q4_k`).

```sh
# ab.py: A/B (BASE i CAND na GPU 0), 4 rundy abbaabba; runda potwierdzenia tak samo
python3 tools/bench/ab.py --rounds 4 --out <out>/l3-<tryb>-bench-ab \
  --a "<base> --color never bench --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --quantization <tryb>" \
  --b "<candidate> --color never bench --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --quantization <tryb>"

# ab.py: B/C (CAND na GPU 0 i CAND z podziałem); --b zamiast --device 0 ma --layer-split 0:30,1:30

# profil (diagnostyka potwierdzenia), A i B
<candidate> --color never profile --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --device 0 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --n 40 --batch 1 --quantization <tryb>

# HTTP: CAND z podziałem, po jednym przebiegu na tryb
<candidate> --color never benchmark --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --layer-split 0:30,1:30 --scenario all --concurrency 1,16 --out <out>/0-C-q8.json --quantization q8_0
```

## Pliki

- `facts.json` — statusy i liczności, GAP i SIGNAL, protokół, wszystkie pary
  A/B `q8_0` i `q4_k` obu rund (4 pola z A, B, B−A, B/A, % i oceną
  kierunku, statystyki per pytanie, `load_s`, sonda GEMM), sygnały, wnioski,
  profile, diagnostyka SIGNAL (definicja `dec_s`, przebiegi każdego procesu,
  czasy przebiegów, forward i przygotowanie `lat2`, `all_items_s`, warunki
  par, grupy sondy, kontekst innych sesji, porównanie źródeł z opisem zmian),
  pary B/C, przebiegi HTTP z podziałem (21 pól, pule per GPU, wolna pamięć,
  NVML, `load_s`, transfery, telemetria per GPU), obiekt kwantyzacji `q4_k`
  wobec P0, warunki i kroki. Liczby mają pełne bity, a zaokrąglenia są
  tylko w README.
- `provenance.json` — sumy binarek, źródeł, runnera i helperów, modelu,
  tabeli GEMM, wejść, faktów P0 i obciążenia, środowisko (karta, sterownik,
  biblioteki, zegar), manifest surowych danych, akceptacja pary binarek,
  wybrane pola audytu ROOT tej sesji i bieżącego L1, sumy plików bieżących
  raportów L1 i L2 oraz wcześniejszego raportu L3 (historia, nie dowód).
- `validation.json` — kontrole przeliczenia, spójności, niezmienności
  chronionych plików i prywatności tego raportu.
