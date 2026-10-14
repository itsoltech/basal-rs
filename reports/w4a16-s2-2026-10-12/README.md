# W4A16 S2: wydajność rodziny `none` (4.5B), 2026-10-12

> **Dane JSON w archiwum.** Pliki `facts.json`, `perf-none-family.json` i `validation.json` z tego katalogu są
> poza Gitem, w archiwum pomiarów `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod ścieżką
> `reports/w4a16-s2-2026-10-12/<plik>`. Odwołania do nich w tekście dotyczą plików z archiwum. Pobieranie i
> sumy kontrolne: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Archiwum zawiera też wersję
> tego pliku sprzed dopisania tej noty. Przeniesienie danych nie jest nowym pomiarem: liczby, statusy i
> rozstrzygnięcia raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

Stan: pakiet danych do niezależnego review. To nie jest akceptacja etapu 04
ani decyzja M2 i pakiet nie wprowadza nowego kryterium.

## Wynik w skrócie

- **Regresja `none`:** wniosek (a). Przy zmierzonej rozdzielczości nie
  wykazano powtarzalnego spowolnienia kandydata `none` wobec bazy. Nie jest
  to dowód braku regresji ani zerowego narzutu. W pierwszej rundzie 11
  metryk `basal benchmark` było gorszych dla B w obu parach. Każda z nich
  przestała być gorsza w co najmniej jednej parze wymaganej rundy
  potwierdzenia. Dla `basal bench` nie było sygnału.
- **Kompromis f16 / `q4_k`:** opis bez bramki szybkości i bez przypisania
  przyczyny. Przy `q4_k`:
  - mediana latencji sequential c1 rośnie o 56,5% i 56,8%;
  - żądania/s sequential c1 spadają o 33,2%, a mixed c16 o 7,4% i 7,2%;
  - pula pamięci po załadowaniu wynosi 2 996 968 960 B, wobec 9 644 216 320 B
    dla f16;
  - `load_s` benchmarku wynosi 25,4–25,5 s, wobec 11,5–12,8 s dla B.
- **Ładowania `q4_k`:** wszystkie 6 ładowań zapisało ten sam obiekt
  kwantyzacji co S1 dla 4.5B (`1f15b087…`). Jego statystyki są równe P0 w
  bitach f64. Kontrola warm-upu objęła jedną macierz na ładowanie.

## Zakres i pochodzenie

Sesja fizyczna: jedna maszyna, 2 × NVIDIA RTX 6000 Ada Generation 48 GB,
sterownik 580.126.09, cuBLASLt 120901. Runner pracował od 09:15:12 do
10:29:55 UTC. Wszystkie pomiary wykonano na GPU 0, na modelu basal-1.5-4.5B
(rewizja `784a683b…`, wagi `77a592cf…`, config `2ac96f2e…`).

Warunki żądane:

- zegar SM 1200 MHz. Zapisano jedno polecenie `nvidia-smi -pm 1` i jedno
  `nvidia-smi -lgc 1200,1200`, oba z kodem 0; to jedno `-lgc` objęło oba
  GPU. Pole `control_commands_exit` (`[0, 0]`) podaje kody w kolejności
  `-pm`, `-lgc`;
- limit mocy 300 W;
- maska 16 z 26 logicznych CPU;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- bundled tabela GEMM `f37120770…`. W 28 logach procesów GPU tabela jest
  wczytana raz i nie ma śladu wyszukiwania.

Role:

| Rola | Binarka | Kwantyzacja |
|---|---|---|
| A | baza, przebudowa `750c146`, SHA-256 `3bdea953…` | `none` |
| B | kandydat W4A16, SHA-256 `a3ed4ade…`, 65 źródeł `94ef2131…` | `none` |
| C | ten sam kandydat | `q4_k` |

Runner `97e9df05…`. Surowe S1 i S2 są równe we wszystkich 26 porównanych
polach: binarki i ich etykiety, pliki provenance (`717e2bd2…`, `0b7fe421…`),
źródła, zasoby wbudowane, model, tabela, wejścia, P0, biblioteki CUDA, typ
GPU, jądro i model CPU. Odcisk S2 obejmuje wyłącznie
model 4.5B, bo mini nie był tu mierzony. Cztery przygotowane pliki JSONL
wejścia 900 oraz `inputs-900.json` są bajtowo równe surowym plikom S1.
Kolektor porównał je z lokalnie przygotowanym zestawem i dla
`inputs-900.json` zapisał `prepared_900_equal: false`. Sparsowany JSON jest
równy, więc to różnica serializacji, jak w S1; flagę zostawiamy bez zmian.

Surowe dane pozostają poza Gitem. Manifest
`w4-s2-20261012-02-raw-sha256-2026-10-12T103000.json` (`ac04a4bd…`) ma 101
wpisów: 91 plików (83 w katalogu przebiegu i 8 rekordów obok niego),
10 katalogów, łącznie 15 281 217 B. Rozmiar i SHA każdego pliku są zgodne,
nie ma plików dodatkowych.

Status runnera pozostaje bez zmian: `complete: true`, `stopped: null`,
kod wyjścia 3. Liczniki: PASS 87, FAIL 0, GAP 1, SIGNAL 0. Wszystkie klasy
FAIL mają wartość 0. Kod 3 oznacza wyłącznie GAP, bez FAIL i SIGNAL.
Jedyny GAP dotyczy etykiety źródła bazy przyjętej w deklarowanej postaci.
Pochodzenie bazy sprawdzono poza runnerem (obserwacja parenta).

### Pochodzenie wartości

Wartości pakietu wyliczyły dwa skrypty Pythona (biblioteka standardowa),
które tylko odczytywały pliki. Poza nimi autor wykonywał wyłącznie doraźne
polecenia odczytu (Python, `sha256sum`, `grep`, `sed`, `ls`, `cat`,
`find`). Nie uruchamiano binarek, modelu, GPU, kwantyzatora ani runnera.
Lista kontroli jest w `validation.json`.

Z surowych plików autor przeliczył niezależnie:

- 126 wartości par `basal benchmark` (6 par × 21 metryk): A, B, B−A, B/A i
  kierunek;
- 210 metryk faz (10 przebiegów × 3 fazy × 7) z 900 rekordów `per_request`
  i `wall_s`. p50 to mediana, p95/p99 to nearest-rank, a RPS/QPS to liczba
  udanych żądań lub pytań przez `wall_s`;
- `basal bench`: 8 par × 3 metryki oraz mediany 39 ilorazów per pytanie,
  liczby szybszych pytań i unikalność indeksów 5..43; także agregaty `ab.py`;
- sygnały, rundę potwierdzenia, trwałość i wniosek;
- profile, obiekty `q4_k` wobec P0, telemetrię CSV i liczniki statusów.

Bitowo równe wartościom zapisanym w surowych raportach i rekordach runnera
są: metryki faz i 126 wartości par benchmark, pary bench z medianami per pytanie, agregaty `ab.py`, rekordy
sygnałów, potwierdzenia, trwałości i wniosku, liczniki statusów oraz
kanoniczny SHA obiektu `q4_k`. Pozostałe pozycje porównano w innym
zakresie:

- statystyki `q4_k` są równe zatwierdzonemu P0 w bitach f64 (porównanie
  autora z P0). Runner też zapisuje pełne wiersze liczbowe silnika i P0
  dla 30 zakresów (6 ładowań × 5), wszystkie z `equal: true`, a każde
  ładowanie ma status PASS; wiersze autora są równe tym zapisom;
- agregaty całych CSV autora i okna warunków runnera mają różne zakresy.
  We wszystkich 10 przebiegach benchmark równe są cztery pola: minimalny
  zegar SM, maksymalna moc, maksymalna temperatura i maksymalna próbka
  pamięci. Liczba próbek różni się w confirmation 0-A: 1937 w oknie
  runnera, 1938 w całym CSV. Maksymalna przerwa między próbkami (okno
  runnera / cały CSV) różni się o 0,001 s w trzech przebiegach:
  primary 0-A 0,236 / 0,237 s, primary 1-B 0,206 / 0,207 s i
  confirmation 3-A 0,222 / 0,221 s;
- ponowna suma sekcji profilu (`math.fsum`) różni się od zapisanej
  `sections_sum_ms` o 1 ULP: A 34,936385824999995 wobec
  34,93638582499999, B 34,87475732499997 wobec 34,87475732499996. Obie
  wartości są w `perf-none-family.json`.

Tylko odczytane z surowych plików (pomiary, których nie da się przeliczyć)
są:

- czasy żądań, `wall_s` i `load_s`;
- liczniki puli, RSS i próbki NVML;
- sekcje profilu i sonda TFLOPS;
- trafność bench.

Przeniesione od runnera są statusy, okna warunków poszczególnych przebiegów
i rekordy procesów. Od parenta przeniesiono brak maszyn, stan ledgera i
blokad oraz dowód źródła i budowy binarek. To nie są nowe dowody autora.

## Protokół

Kolejność `basal benchmark --scenario all --concurrency 1,16`: A B C C B A,
każdy przebieg w świeżym procesie. Pary zadeklarowano przed pomiarem:

- regresja: A0/B1 i B4/A5;
- kompromis: B1/C2 i C3/B4. Przebiegi B1 i B4 należą do obu rodzajów par.

`ab.py --rounds 4` (`basal bench --dtype f16`, 44 pozycje, 5 rozgrzewki,
39 mierzonych) uruchomiono raz dla A/B i raz dla B/C, w kolejności
a b b a a b b a.

Metryki i kierunki:

- bench: `lat2_ms`, `lat2_p95_ms` (wyżej = gorzej), `dec_s` (niżej = gorzej)
  i mediana ilorazu per pytanie (> 1 = gorzej);
- benchmark, na fazę sequential-c1, mixed-c1 i mixed-c16: latencja udanych
  żądań p50/p95/p99 i kolejka p50/p99 (wyżej = gorzej) oraz żądania/s
  i pytania/s (niżej = gorzej). Razem 21 metryk.

Reguła ustalona przed pomiarem:

- sygnał to ta sama metryka gorsza dla B w każdej parze rundy;
- po sygnale wykonuje się dokładnie jedną rundę potwierdzenia dotkniętego
  polecenia oraz po jednym `basal profile` A i B przy batch 1;
- sygnał jest utrzymany tylko wtedy, gdy B jest gorsze we wszystkich parach
  obu rund;
- nie stosujemy progów, p-value, wykluczeń, nowego parowania ani dalszych
  rund.

## Regresja `none`

### `basal bench` A/B

| Para | Kolejność | lat2 p50 [ms] A / B | lat2 p95 [ms] A / B | dec/s A / B | Mediana B/A per pytanie | B szybsze | Trafność A / B |
|---|---|---|---|---|---:|---:|---|
| 0 | AB | 25,607 / 25,564 | 32,535 / 32,528 | 64,35 / 64,13 | 0,99916 | 27/39 | 35/44 / 35/44 |
| 1 | BA | 25,633 / 25,585 | 32,505 / 32,513 | 64,21 / 64,36 | 0,99954 | 28/39 | 35/44 / 35/44 |
| 2 | AB | 25,640 / 25,606 | 32,522 / 32,551 | 64,39 / 64,39 | 0,99982 | 21/39 | 35/44 / 35/44 |
| 3 | BA | 25,608 / 25,733 | 32,494 / 32,621 | 64,30 / 64,19 | 1,00085 | 13/39 | 35/44 / 35/44 |

Żadna z czterech metryk nie była gorsza dla B we wszystkich czterech
parach, więc nie było potwierdzenia bench. Agregat `ab.py` z czterech
rund daje medianę B/A per pytanie 0,99974; B było szybsze w 23 z 39
pytań. Trafność bench wynosi 35/44 w każdym przebiegu A, B i C. Bench
zapisuje tylko liczbę trafień, bez odpowiedzi, więc nie porównujemy tu
odpowiedzi między binarkami. Bitowa równość eksportów `none` z bazą to
wynik S1.

### `basal benchmark`: kompletność

Dziesięć świeżych procesów (sześć podstawowych i cztery potwierdzające)
zmierzyło po 900 żądań w każdej z trzech faz. Razem 27 000 odpowiedzi 200,
bez błędów i anulowań, z 900 unikalnymi ID na fazę. Wszystkie żądania mixed
zostały wysłane; zasada pominięcia 54 żądań z memory-mini nie dotyczy S2.
`answer_changes_vs_first_success` wynosi 0 we wszystkich 30 fazach. To
spójność odpowiedzi wewnątrz przebiegu, a nie porównanie między przebiegami
ani z FP32. Ustawienia wszystkich przebiegów są równe poza kwantyzacją.

### Sygnały podstawowe i potwierdzenie

Tabela podaje B/A−1 dla wszystkich 11 sygnałów podstawowych. Para
potwierdzenia 0 to przebiegi 0-A/1-B, a para 1 to 3-A/2-B.

| Metryka | Gorzej, gdy | Podst. A0/B1 | Podst. B4/A5 | Potw. para 0 | Potw. para 1 | B gorsze |
|---|---|---:|---:|---:|---:|---:|
| mixed-c1 queue_ms.p50 | wyżej | +14,4802% | +2,6580% | +4,0939% | -10,1889% | 3/4 |
| mixed-c16 queue_ms.p99 | wyżej | +1,7518% | +1,5409% | -0,7458% | +0,0122% | 3/4 |
| mixed-c16 successful_latency_ms.p99 | wyżej | +1,6796% | +0,2675% | +0,4949% | -0,5640% | 3/4 |
| mixed-c16 successful_questions_per_second | niżej | -0,6576% | -0,2323% | +0,2513% | +0,0549% | 2/4 |
| mixed-c16 successful_requests_per_second | niżej | -0,6576% | -0,2323% | +0,2513% | +0,0549% | 2/4 |
| sequential-c1 queue_ms.p50 | wyżej | +30,2358% | +3,4141% | -0,2155% | -17,6801% | 2/4 |
| sequential-c1 successful_latency_ms.p50 | wyżej | +0,7767% | +0,4230% | +0,0138% | -2,2996% | 3/4 |
| sequential-c1 successful_latency_ms.p95 | wyżej | +0,4275% | +0,0782% | +0,0008% | -0,7475% | 3/4 |
| sequential-c1 successful_latency_ms.p99 | wyżej | +0,0829% | +0,3860% | +0,1313% | -0,6460% | 3/4 |
| sequential-c1 successful_questions_per_second | niżej | -0,8422% | -0,4899% | -0,1988% | +0,7887% | 3/4 |
| sequential-c1 successful_requests_per_second | niżej | -0,8422% | -0,4899% | -0,1988% | +0,7887% | 3/4 |

Żaden sygnał nie utrzymał się we wszystkich czterech parach. Osiem metryk
było gorszych dla B w 3 z 4 par, a trzy w 2 z 4. O wyniku decyduje
ustalona reguła, a nie te liczebności. Zmienności nie używamy do
rozgrzeszania; warunek utrzymania po prostu nie zaszedł. Nie ustalono
przyczyny pojedynczych różnic. Wszystkie 21 metryk dla wszystkich czterech
par regresji (wartości A i B, B−A, B/A i liczby kierunków) zawiera
`perf-none-family.json`, również gdy B jest lepsze.

### Profil A/B (batch 1, 40 pozycji, f16)

Wartości w ms na pozycję, po jednym procesie na rolę. To profil sekcji, a
nie latencja żądań ani odrębna bramka szybkości.

| Sekcja | A | B | B−A | B/A−1 |
|---|---:|---:|---:|---:|
| attention | 6,6405 | 6,6363 | -0,0042 | -0,06% |
| bias+rope+heads | 0,9128 | 0,9112 | -0,0015 | -0,17% |
| bias+silu*up | 1,1495 | 1,1420 | -0,0075 | -0,65% |
| down_proj+bias+residual | 10,1090 | 10,0574 | -0,0516 | -0,51% |
| embed+inputs | 0,4947 | 0,4985 | +0,0038 | +0,77% |
| final_norm | 0,0488 | 0,0466 | -0,0022 | -4,48% |
| gate_up_proj | 9,9968 | 10,0030 | +0,0061 | +0,06% |
| o_proj+bias+residual+rms_norm | 2,3656 | 2,3676 | +0,0019 | +0,08% |
| qkv_proj | 2,3220 | 2,3105 | -0,0115 | -0,49% |
| rms_norm | 0,8966 | 0,9016 | +0,0050 | +0,55% |
| razem (profilowane) | 35,0164 | 34,9505 | -0,0659 | -0,19% |
| suma sekcji | 34,9364 | 34,8748 | -0,0616 | -0,18% |

W obu profilach `residual_abs_max` wynosi 0 we wszystkich 60 warstwach.

## Kompromis f16 / `q4_k` (B wobec C, opisowo)

Brak reguły sygnału i kryterium szybkości. Różnice nie mają przypisanej
przyczyny. Jakość `q4_k` nie jest tu mierzona ponownie. W S1 ścisłe
kryterium FP32 dało FAIL, a decyzja M2 zachowała `q4_k` jako opcjonalny
tryb eksperymentalny, z domyślnym `none`.

### `basal bench` B/C

| Para | Kolejność | lat2 p50 [ms] B / C | lat2 p95 [ms] B / C | dec/s B / C | Mediana C/B per pytanie | C szybsze | Trafność B / C |
|---|---|---|---|---|---:|---:|---|
| 0 | BC | 25,781 / 40,982 | 32,588 / 48,556 | 63,96 / 56,45 | 1,59593 | 0/39 | 35/44 / 35/44 |
| 1 | CB | 25,669 / 41,059 | 32,575 / 48,821 | 64,28 / 56,42 | 1,60127 | 0/39 | 35/44 / 35/44 |
| 2 | BC | 25,584 / 41,103 | 32,515 / 48,581 | 64,39 / 56,48 | 1,60478 | 0/39 | 35/44 / 35/44 |
| 3 | CB | 25,587 / 41,120 | 32,636 / 48,731 | 64,13 / 56,26 | 1,60707 | 0/39 | 35/44 / 35/44 |

Agregat `ab.py` daje medianę C/B per pytanie 1,60152. Równa trafność 35/44
na 44 pozycjach nie świadczy o zachowaniu jakości.

### `basal benchmark` B/C

| Metryka | B1 | C2 | C2/B1−1 | B4 | C3 | C3/B4−1 |
|---|---:|---:|---:|---:|---:|---:|
| sequential-c1 successful_latency_ms.p50 | 26,94 | 42,17 | +56,52% | 27,00 | 42,35 | +56,85% |
| sequential-c1 successful_latency_ms.p95 | 40,36 | 56,38 | +39,69% | 40,43 | 56,59 | +39,99% |
| sequential-c1 successful_latency_ms.p99 | 48,77 | 64,71 | +32,66% | 48,91 | 64,93 | +32,75% |
| sequential-c1 queue_ms.p50 | 0,469 | 0,361 | -23,11% | 0,469 | 0,477 | +1,60% |
| sequential-c1 queue_ms.p99 | 1,019 | 1,006 | -1,28% | 1,204 | 1,336 | +10,96% |
| sequential-c1 successful_requests_per_second | 31,887 | 21,311 | -33,17% | 31,837 | 21,256 | -33,23% |
| sequential-c1 successful_questions_per_second | 31,887 | 21,311 | -33,17% | 31,837 | 21,256 | -33,23% |
| mixed-c1 successful_latency_ms.p50 | 46,26 | 61,69 | +33,34% | 46,08 | 62,20 | +34,99% |
| mixed-c1 successful_latency_ms.p95 | 1104,29 | 1121,07 | +1,52% | 1105,50 | 1122,26 | +1,52% |
| mixed-c1 successful_latency_ms.p99 | 3052,66 | 3197,23 | +4,74% | 3055,37 | 3177,42 | +3,99% |
| mixed-c1 queue_ms.p50 | 0,925 | 0,937 | +1,30% | 1,023 | 1,056 | +3,18% |
| mixed-c1 queue_ms.p99 | 90,192 | 92,217 | +2,25% | 108,092 | 95,879 | -11,30% |
| mixed-c1 successful_requests_per_second | 4,893 | 4,489 | -8,26% | 4,885 | 4,482 | -8,26% |
| mixed-c1 successful_questions_per_second | 11,243 | 10,315 | -8,26% | 11,225 | 10,298 | -8,26% |
| mixed-c16 successful_latency_ms.p50 | 580,16 | 675,04 | +16,35% | 577,21 | 674,79 | +16,90% |
| mixed-c16 successful_latency_ms.p95 | 17325,75 | 18703,48 | +7,95% | 17142,77 | 18693,04 | +9,04% |
| mixed-c16 successful_latency_ms.p99 | 44900,93 | 48183,60 | +7,31% | 45051,53 | 48127,14 | +6,83% |
| mixed-c16 queue_ms.p50 | 246,549 | 284,844 | +15,53% | 239,397 | 293,050 | +22,41% |
| mixed-c16 queue_ms.p99 | 40096,388 | 43495,371 | +8,48% | 40468,297 | 43289,036 | +6,97% |
| mixed-c16 successful_requests_per_second | 5,695 | 5,275 | -7,37% | 5,681 | 5,273 | -7,18% |
| mixed-c16 successful_questions_per_second | 13,085 | 12,121 | -7,37% | 13,054 | 12,116 | -7,18% |

Latencje i kolejki w ms, przepustowość w żądaniach lub pytaniach na
sekundę. Kolejka sequential c1 i kolejka p99 mixed c1 mają w obu parach
przeciwne kierunki.

## Pamięć i czas ładowania (4.5B)

| Rola | Przebiegi | load_s benchmark [s] | Pula used po załadowaniu [B] | Pula used, szczyt ładowania [B] | Pula used, szczyt pracy [B] | Pula reserved, szczyt pracy [B] | Maks. próbka NVML GPU 0 [MiB] |
|---|---:|---|---|---|---|---|---|
| A | 4 | 11,62–12,88 | 9 644 216 320 | 9 644 216 320 | 12 254 394 932–12 531 592 828 | 12 482 248 704–12 750 684 160 | 12 424–12 680 |
| B | 4 | 11,48–12,81 | 9 644 216 320 | 9 644 216 320 | 12 254 394 932–12 531 592 828 | 12 415 139 840–12 717 129 728 | 12 360–12 648 |
| C | 2 | 25,39–25,54 | 2 996 968 960 | 3 083 395 848 | 5 732 696 344 | 6 006 243 328 | 6 248 |

Wartości `reserved` puli:

- po załadowaniu: 9 663 676 416 B dla A i B oraz 3 087 007 744 B dla C;
- w szczycie ładowania: 9 663 676 416 B dla A i B oraz 3 120 562 176 B
  dla C;
- po zamknięciu silnika `used` i `reserved` wynoszą 0 we wszystkich 10
  przebiegach.

Pula obejmuje tylko alokator stream-ordered. Nie obejmuje kontekstu CUDA,
modułów, uchwytów bibliotek ani alokacji synchronicznych, więc nie jest
całym VRAM. Szczyt pracy obejmuje wszystkie fazy razem, bez podziału na
fazy. NVML to próbki urządzenia co 200 ms, a nie szczyty.

Szacunek wag z arytmetyki formatu (nie pomiar) wynosi 9 514 629 496 B dla
`none` i 2 867 382 136 B dla `q4_k`. Bloki `q4_k` 240 macierzy projekcji
zajmują 2 601 192 960 B z paddingiem. Gęste f16 tych samych projekcji
zajmowałoby 9 248 440 320 B; obie liczby dotyczą wyłącznie projekcji.

Zakres `load_s` różni się między poleceniami:

- `basal benchmark` mierzy ładowanie bez tworzenia kontekstu CUDA, bo
  kontekst tworzy wcześniej sonda puli;
- `basal bench` i `basal profile` mierzą je razem z tworzeniem kontekstu.

| Polecenie i rola | load_s [s] |
|---|---|
| bench A (A/B) | 12,47–14,78 |
| bench B (A/B) | 12,28–13,56 |
| bench B (B/C) | 13,35–14,34 |
| bench C (B/C) | 32,98–38,89 |
| profil A / B (linia logu) | 11,6 / 11,5 |

Czas `load_s` przy `q4_k` obejmuje kwantyzację na hoście, ale jego podziału
silnik nie mierzy. Czasy procesów (rekordy runnera):

- benchmark C trwał 444,8 i 445,5 s przy CPU 646,6 i 649,4 s, wobec
  387,6–393,3 s i 404,0–408,7 s CPU dla A i B;
- wywołanie `ab.py` B/C trwało 273,5 s przy CPU 1070,6 s, a A/B 177,7 s
  przy CPU 235,8 s.

Tych czasów nie rozkładamy na składniki.

RSS hosta, w bajtach:

| Rola | sequential c1 | mixed c1 | mixed c16 |
|---|---|---|---|
| A | 712 769 536–714 477 568 | 852 324 352–927 469 568 | 897 380 352–940 036 096 |
| B | 712 220 672–713 703 424 | 852 361 216–853 397 504 | 898 207 744–899 772 416 |
| C | 725 389 312–726 114 304 | 866 177 024–866 263 040 | 911 028 224–911 634 432 |

Te wartości to `lifetime_peak_rss_bytes` fazy: próbkowany co 200 ms szczyt
`ru_maxrss` od startu procesu. Proces łączy serwer HTTP i generator
obciążenia, więc wartość obejmuje również ładowanie. Nie jest to
izolowany szczyt ładowania, przyrost w fazie ani zużycie hosta.

## Ładowania `q4_k`

Sześć ładowań C: cztery bench i dwa benchmark. Każde zapisało pełny obiekt
`quantization`.

- Kanoniczny SHA-256 obiektu (przeliczony) jest identyczny we wszystkich
  sześciu ładowaniach i równy obiektowi 4.5B z S1 (`1f15b087…`).
- Statystyki rekonstrukcji modelu i czterech typów projekcji są równe
  zatwierdzonemu P0 (`44005ecc…`): liczby całkowite, bity f64 `sum_e2`,
  `sum_w2`, `max_abs_error` i R oraz liczniki grup z pierwszymi pozycjami.
  Daje to 30 porównań. Kwantyzatora ani P0 nie uruchamiano ponownie.
- Względne RMS modelu wynosi 0,07621214682255444. Grupy: 9 spłaszczonych
  i 0 zerowych na 144 506 880.
- Warm-up w każdym ładowaniu: macierz warstwy 0 `o` (4 194 304 wartości),
  zdekwantyzowana na GPU, jest bitowo równa rekonstrukcji hosta. To jedna
  macierz na ładowanie. Dla pozostałych macierzy pozostaje argument
  źródłowy z S1; nie zmierzono ich.
- 20 strukturalnych raportów `none` (12 bench i 8 przebiegów benchmark
  A/B) nie ma obiektu kwantyzacji. Dwa ładowania `none` w profilach A/B
  nie mają opisu backendu w JSON; widać je tylko w logu. Razem 28 logów
  procesów GPU: 22 `none` i 6 `q4_k`.

## Warunki

- GPU 0, wszystkie 12 CSV:
  - próbki zegara SM: 1050–1200 MHz przy żądanych 1200 MHz, więc zegar nie
    był stały;
  - temperatura do 83 °C;
  - moc do 290,02 W przy limicie 300 W;
  - maksymalna przerwa między próbkami 0,237 s.
- Okna runnera: pokrycie co najmniej 0,999 w każdym przebiegu benchmark.
  Minimalny zegar SM, maksymalna moc, maksymalna temperatura i maksymalna
  próbka pamięci w oknach są równe agregatom całych CSV autora. Liczba
  próbek i maksymalna przerwa mają opisane wyżej różnice zakresu.
- GPU 1: wykorzystanie maksymalnie 0% i pamięć maksymalnie 4 MiB we
  wszystkich 12 CSV.
- Obserwator procesów obliczeniowych runnera: 0 obcych procesów i 0
  nieudanych odczytów w 14 poleceniach. Profile nie mają CSV, tylko po
  6 próbek tego obserwatora.
- Sonda GEMM `basal bench` przed i po: 103,2–120,9 TFLOPS.

Warunki opisują pomiar. Nie wykluczają przebiegów i nie dowodzą zakłócenia.

## Ograniczenia

- Jeden model, jeden typ karty, jedna sesja i GPU 0; wyniki nie są
  uogólniane. Brak CPU, innych zegarów, modeli i GPU oraz nsys.
- Wniosek (a) nie dowodzi braku regresji ani zerowego narzutu.
- Kompromis `q4_k` jest opisowy: bez bramki szybkości, bez przyczynowości
  i bez nowej oceny jakości.
- Spójność odpowiedzi benchmarku dotyczy wnętrza przebiegu. Pakiet nie
  zawiera porównania odpowiedzi między binarkami ani z FP32.
- Pamięć: pula tylko stream-ordered, NVML jako próbki, RSS jako
  próbkowany szczyt od startu procesu. Pakiet nie podaje izolowanego
  szczytu ładowania.
- Czas kwantyzacji na hoście w `load_s` nie jest rozłożony.
- Dekwantyzację GPU zmierzono dla jednej macierzy na ładowanie.
- Pakiet nie zmienia historycznych raportów, kryteriów ani statusów.

## Pliki

- `facts.json`: tożsamości i odciski, statusy runnera, protokół, wyniki
  zbiorcze, pamięć według roli, pochodzenie wartości, obserwacje parenta i
  ograniczenia.
- `perf-none-family.json`: wszystkie pary bench (39 ilorazów per pytanie)
  i benchmark (21 metryk), 10 przebiegów (fazy, pule, zasoby, warunki,
  CSV), sygnały, profile, sześć ładowań `q4_k` z wierszami P0 i rekordy
  14 poleceń. Wartości są surowe i przechodzą round trip bitowo.
- `validation.json`: 43 kontrole autora z wynikami, SHA odczytanych plików,
  metoda, obserwacje runnera, kolektora i parenta oraz zakres nieprzeliczony.
