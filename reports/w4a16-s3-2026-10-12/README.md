# W4A16 S3: wydajność rodziny `q8_0` (4.5B), 2026-10-12

Stan: pakiet danych z ograniczoną analizą do niezależnego review. To nie
jest akceptacja etapu 04, nie jest commit ani nowe kryterium. Pakiet nie
zmienia statusów runnera ani wcześniejszych raportów.

## Wynik w skrócie

- **Regresja `q8_0`, wynik runnera:** `complete: true`, `stopped: null`,
  kod wyjścia 4, PASS 86, FAIL 0, GAP 1, SIGNAL 1. Surowy wniosek runnera
  brzmi: „persistent signal: (b) or (c) after the parent's analysis”.
  Utrzymany sygnał dotyczy dwóch metryk mixed-c16: żądań/s i pytań/s. B
  było gorsze we wszystkich czterech parach obu rund. Zmiana żądań/s
  (B/A−1) wyniosła -0,3732%, -0,2413%, -0,2015% i
  -0,1076%. Obie metryki liczy się z tego samego `wall_s` fazy
  przy stałej liczbie żądań i pytań, więc nie są niezależne. Odpowiada im
  jedna kontrola SIGNAL.
- **Trzeci sygnał podstawowy** (mixed-c1 `queue_ms.p99`) odwrócił się w obu
  parach rundy potwierdzenia. `basal bench` nie miał sygnału.
- **Ograniczona analiza (warunki, profile, zmieniony kod):** w przeczytanym
  zakresie nie wskazano konkretnej przyczyny w zmienionej ścieżce Q8.
  Warunki i profile też nie wyjaśniają różnicy. Wniosek ograniczony to (c):
  przyczyna pozostaje nierozstrzygnięta. Nie twierdzimy, że kod jest wolny
  od defektu, że wykazano szum ani że regresji nie ma.
- **Decyzja użytkownika:** użytkownik zaakceptował zaobserwowany spadek
  przepustowości mixed-c16 o 0,11–0,37% jako tolerowany kompromis produktowy.
  Spadki rzędu 1–2% uznał za mieszczące się w normie
  (`phases/04-w4a16/performance-decision.md`). Nie powstaje automatyczny
  próg 2% ani nowa tolerancja. Surowy sygnał i status pozostają bez zmian.
- **Kompromis `q8_0` / `q4_k`** (B wobec C, opisowo, bez bramki): przy
  `q4_k`:
  - żądania/s sequential c1 rosną o +1,37% i +1,70%;
  - żądania/s mixed c16 rosną o +0,43% i +0,66%;
  - mediana C/B per pytanie w `basal bench` wynosi 0,97660;
  - pula po załadowaniu to 2 996 968 960 B, wobec
    5 309 140 480 B dla `q8_0`;
  - `load_s` benchmarku wynosi 25,44–26,13 s, wobec
    12,47–12,92 s.

  Jakości tu nie mierzono ponownie. Decyzja M2 o jakości pozostaje bez
  zmian: `q4_k` jest opcjonalny i eksperymentalny, a domyślny i
  rekomendowany jest `none`.
- **Ładowania `q4_k`:** we wszystkich 6 ładowaniach obiekt kwantyzacji ma
  ten sam kanoniczny SHA-256 `1f15b087…` co obiekt 4.5B z S1. Pełne
  statystyki są równe P0: 6 ładowań × 5 zakresów = 30 wierszy. Kontrola
  warm-upu objęła jedną macierz na ładowanie.

## Zakres i pochodzenie

Sesja fizyczna: jedna maszyna, 2 × NVIDIA RTX 6000 Ada Generation 48 GB,
sterownik 580.126.09, cuBLASLt 120901. Runner pracował od
10:59:31 do 12:20:53 UTC. Wszystkie pomiary
wykonano na GPU 0, na modelu basal-1.5-4.5B (rewizja `784a683b…`, wagi
`77a592cf…`, config `2ac96f2e…`).

Warunki żądane:

- zegar SM 1200 MHz. Zapisano jedno polecenie `nvidia-smi -pm 1` i jedno
  `nvidia-smi -lgc 1200,1200`, w tej kolejności, oba z kodem 0. Każde z nich
  objęło oba GPU, więc `control_commands_exit` to `[0, 0]`. Zegara pamięci
  nie blokowano;
- limit mocy 300 W;
- maska 16 z 26 logicznych CPU;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- bundled tabela GEMM `f37120770…`. W 28 logach procesów GPU tabela jest
  wczytana raz i nie ma śladu wyszukiwania.

Role:

| Rola | Binarka | Kwantyzacja |
|---|---|---|
| A | baza, przebudowa `750c146`, SHA-256 `3bdea953…` | `q8_0` |
| B | kandydat W4A16, SHA-256 `a3ed4ade…`, 65 źródeł `94ef2131…` | `q8_0` |
| C | ten sam kandydat | `q4_k` |

Obie binarki wypisują wersję `750c1465`, bo wbudowana etykieta to skrócony
commit `HEAD`. Kandydat zawiera dodatkowo zmiany drzewa roboczego.
Tożsamość wyznaczają SHA binarek i odcisk 65 źródeł, a nie etykieta.
Obie binarki pochodzą z tego samego sparowanego buildera, co w S1 i S2
(Ubuntu 22.04, CUDA 12.9.1, rustc 1.95.0 według obu plików provenance). Pliki provenance
`717e2bd2…` i `0b7fe421…` są równe lokalnym rekordom buildera, a zapisany
w nich SHA binarki jest równy SHA z odcisku. Pakiet nie przypisuje bazy
historycznej binarce W8 `21bdb…` i nie wykazuje niezależnej przebudowy.

Runner `97e9df05…`. Odcisk S3 jest równy surowemu S1 we wszystkich polach
poza trzema. Pola `models` i `gemm_tables` zawierają tylko 4.5B i są równe
wpisom 4.5B z S1. Pole `deadline` sesji fizycznej różni się celowo.
Odcisk zawiera `helpers_sha256` i `sources` (65 plików, `94ef2131…`).
65 linii `source-files.sha256` jest równych plikom tego drzewa roboczego,
więc analizowany kod to kod kandydata. Pięć plików wejścia 900 jest
bajtowo równych surowym plikom S1. Kolektor zapisał
`prepared_900_equal: false` tylko dla `inputs-900.json` i tę flagę
zostawiamy bez zmian.

Surowe dane pozostają poza Gitem. Manifest
`w4-s3-20261012-01-raw-sha256-2026-10-12T122127.json` (`904b070a…`) ma 101
wpisów: 91 plików (83 w katalogu przebiegu i
8 rekordów obok niego), 10 katalogów, łącznie
15 505 530 B. Rozmiar i SHA każdego pliku są zgodne, nie ma plików
dodatkowych ani brakujących. Kolektor zapisał status `verified`, pustą
listę błędów i `stop_requested: false`.

Status runnera pozostaje bez zmian. Kod 4 oznacza SIGNAL bez FAIL.
Wszystkie klasy FAIL mają wartość 0. Jedyny GAP dotyczy etykiety źródła
bazy przyjętej w deklarowanej postaci. Pochodzenie bazy sprawdził parent
poza runnerem i ten GAP pozostaje.

### Pochodzenie wartości

Wartości pakietu wyliczyły dwa skrypty Pythona (biblioteka standardowa),
które tylko odczytywały pliki. Poza nimi autor wykonywał doraźne polecenia
odczytu (Python, `sha256sum`, `grep`, `sed`, `head`, `tail`, `cut`, `sort`,
`uniq`, `awk`, `tr`, `ls`, `cat`, `find`, `wc`) oraz tylko-odczytowe
`git diff` / `git rev-parse`. Zapisywał wyłącznie we własnych katalogach
`work/` i `packet/`. Nie uruchamiano binarek,
modelu, GPU, kwantyzatora, P0, runnera, helperów ani `ab.py`. Kod runnera i
`ab.py` był czytany wyłącznie jako tekst definicji.

Autor przeliczył niezależnie z surowych plików:

- 210 metryk faz (10 przebiegów × 3 fazy × 7) z 900 rekordów `per_request`
  i `wall_s`. Definicje pochodzą z `benchmark.rs`: p50 to mediana, p95/p99
  to nearest-rank, a RPS/QPS to liczba udanych żądań lub pytań przez
  `wall_s`. Wszystkie wartości są bitowo równe zapisanym. Definicje są
  takie same w obu binarkach: `bench.rs`, `stats.rs` i `benchmark/http.rs`
  nie mają zmian, a diff `benchmark.rs` zmienia tylko teksty;
- 126 wartości par `basal benchmark` (6 par × 21 metryk): A, B, B−A, B/A i
  kierunek. Wszystkie są bitowo równe rekordom par runnera;
- `basal bench`: 8 par × 3 metryki, mediany 39 ilorazów per pytanie, liczby
  szybszych pytań, unikalność indeksów 5..43 i agregaty `ab.py`, bitowo
  równe obu `ab.json`. W każdym z 16 raportów: mediana i p95 `lat2` z
  `lat2_ms_all`, `dec_s` jako mediana trzech `dec_s_runs` i
  `acc = correct / with_gold`, wszystkie bitowo równe;
- sygnały, rundę potwierdzenia i trwałość. Są równe zapisom runnera, a
  kierunki wszystkich 21 metryk × 4 par są równe liście parenta;
- obiekty `q4_k`: kanoniczny SHA i pełne porównanie z P0;
- agregaty 12 CSV telemetrii, różnice profili, `wall_s` par i liczniki
  statusów.

Porównania w innym zakresie:

- okna warunków runnera i agregaty całych CSV autora. W 10 przebiegach
  benchmark równe są: minimalny zegar SM, maksymalna moc, temperatura,
  próbka pamięci, maksymalna przerwa i stany P. Średnie są zaokrąglone do
  0,1, jak w runnerze. Liczba próbek różni się tylko w confirmation 0-A:
  2168 w oknie runnera, 2169 w całym CSV;
- suma sekcji profilu A przeliczona przez `math.fsum` wynosi
  `208.50132619999974`, a zapisana
  `208.50132619999977` (różnica 1 ULP). Dla B obie wartości są
  bitowo równe (`245.18051057500014`).

Tylko odczytane z surowych plików (pomiary, których nie da się przeliczyć)
są:

- czasy żądań, `wall_s` i `load_s`;
- liczniki puli, RSS i próbki NVML;
- sekcje profilu, czas dekwantyzacji i sonda TFLOPS;
- liczniki trafności bench;
- czasy procesów.

Przeniesione od runnera są statusy, okna warunków, rekordy procesów i
obserwator obcych procesów. Od parenta przeniesiono stan maszyn, ledgera i
blokad oraz dowód źródła i budowy binarek. To nie są nowe dowody autora.

## Protokół

Kolejność `basal benchmark --scenario all --concurrency 1,16`: A B C C B A,
każdy przebieg w świeżym procesie. Pary zadeklarowano przed pomiarem:

- regresja: A0/B1 i B4/A5 (indeksy 0,1 oraz 5,4);
- kompromis: B1/C2 i C3/B4 (1,2 oraz 4,3). Przebiegi B1 i B4 należą do obu
  rodzajów par.

`ab.py --rounds 4` (`basal bench --dtype f16 --quantization …`: 44 pozycje,
5 rozgrzewki, 39 mierzonych) uruchomiono raz dla A/B i raz dla B/C, w kolejności
a b b a a b b a. W B/C stroną a jest B (`q8_0`), a stroną b jest C (`q4_k`).

Metryki i kierunki:

- bench: `lat2_ms`, `lat2_p95_ms` (wyżej = gorzej), `dec_s` (niżej = gorzej)
  i mediana ilorazu per pytanie (> 1 = gorzej);
- benchmark, w fazach sequential-c1, mixed-c1 i mixed-c16: latencja udanych
  żądań p50/p95/p99 i kolejka p50/p99 (wyżej = gorzej) oraz żądania/s i
  pytania/s (niżej = gorzej). Razem 21 metryk.

Reguła ustalona przed pomiarem:

- sygnał to ta sama metryka gorsza dla B w każdej parze rundy;
- po sygnale wykonuje się dokładnie jedną rundę potwierdzenia dotkniętego
  polecenia oraz po jednym `basal profile` A i B przy batch 1. Tu dotknięty
  był tylko benchmark: cztery świeże przebiegi A B B A, pary (0 A, 1 B) i
  (3 A, 2 B). Profile mają `--n 40 --batch 1`;
- sygnał jest utrzymany, gdy B jest gorsze we wszystkich parach obu rund.
  Wtedy następuje analiza warunków, profilu i zmienionego kodu. Poprawka
  wchodzi w grę tylko dla przyczyny w kodzie. Bez niej wniosek to (c) i
  decyzja użytkownika;
- nie stosujemy progów, p-value, wykluczeń, nowego parowania ani dalszych
  rund.

## Regresja `q8_0`

### `basal bench` A/B

| Para | Kolejność | lat2 p50 [ms] A / B | lat2 p95 [ms] A / B | dec/s A / B | Mediana B/A per pytanie | B szybsze | Trafność A / B |
|---|---|---|---|---|---:|---:|---|
| 0 | AB | 42,286 / 42,214 | 49,631 / 49,536 | 56,37 / 56,41 | 0,99843 | 35/39 | 35/44 / 35/44 |
| 1 | BA | 42,334 / 42,274 | 49,801 / 49,652 | 56,33 / 56,38 | 0,99830 | 35/39 | 35/44 / 35/44 |
| 2 | AB | 42,277 / 42,375 | 49,578 / 49,913 | 56,44 / 56,25 | 1,00201 | 4/39 | 35/44 / 35/44 |
| 3 | BA | 42,281 / 42,279 | 49,584 / 49,659 | 56,53 / 56,31 | 1,00094 | 11/39 | 35/44 / 35/44 |

Żadna z czterech metryk bench nie była gorsza dla B we wszystkich czterech
parach, więc nie było sygnału ani potwierdzenia bench. Agregat `ab.py` z
czterech rund daje medianę B/A per pytanie 1,0001074.
B było szybsze w 16 z 39 pytań. Trafność bench wynosi 35/44
w każdym z 16 raportów (A, B i C). Bench zapisuje tylko liczbę trafień,
bez odpowiedzi, więc równa trafność nie oznacza równych odpowiedzi.
Bajtowa równość eksportów `q8_0` kandydata z bazą (10 par eksportów i para
900) oraz równość 93 odpowiedzi HTTP to wcześniejszy wynik S1, a nie nowy
dowód S3. S3 nie porównuje odpowiedzi między przebiegami.

### `basal benchmark`: kompletność

Dziesięć świeżych procesów (sześć podstawowych i cztery potwierdzające)
zmierzyło po 900 żądań w każdej z trzech faz. Razem 27 000 odpowiedzi 200,
bez błędów i anulowań, z 900 unikalnymi ID na fazę. Wszystkie żądania mixed
zostały wysłane (0 pominiętych). `answer_changes_vs_first_success` wynosi 0
we wszystkich 30 fazach. To spójność odpowiedzi wewnątrz przebiegu, a nie
porównanie między przebiegami. Ustawienia wszystkich przebiegów są równe
poza kwantyzacją, a źródło obciążenia jest jedno.

### Sygnały podstawowe i potwierdzenie

B/A−1 dla trzech sygnałów podstawowych:

| Metryka | Gorzej, gdy | Podst. A0/B1 | Podst. B4/A5 | Potw. para 0 | Potw. para 1 | B gorsze |
|---|---|---:|---:|---:|---:|---:|
| mixed-c1 queue_ms.p99 | wyżej | +12,7415% | +1,8956% | -6,3314% | -5,0024% | 2/4 |
| mixed-c16 successful_questions_per_second | niżej | -0,3732% | -0,2413% | -0,2015% | -0,1076% | 4/4 |
| mixed-c16 successful_requests_per_second | niżej | -0,3732% | -0,2413% | -0,2015% | -0,1076% | 4/4 |

Para potwierdzenia 0 to przebiegi 0-A/1-B, a para 1 to 3-A/2-B. Sygnał
mixed-c1 `queue_ms.p99` nie powtórzył się. Utrzymały się żądania/s i
pytania/s mixed-c16. Ich ilorazy B/A są bitowo równe w 2 parach, a w
pozostałych 2 różnią się o 1 ULP. Oba pochodzą z tych samych 900 żądań i
2068 pytań fazy oraz jednego `wall_s`. W `wall_s` mixed-c16 (≈171 s) B trwało dłużej o
0,641, 0,416, 0,346 i 0,184 s.

### Wszystkie 21 metryk w czterech parach regresji

B/A−1; wartości A i B, B−A i kierunki są w `perf-q8-family.json`.

| Metryka | Gorzej, gdy | A0/B1 | B4/A5 | Potw. 0 | Potw. 1 | B gorsze |
|---|---|---:|---:|---:|---:|---:|
| sequential-c1 successful_latency_ms.p50 | wyżej | -0,4097% | +0,1946% | +0,4120% | +0,1543% | 3/4 |
| sequential-c1 successful_latency_ms.p95 | wyżej | -0,0337% | -0,1497% | +0,2356% | +0,0380% | 2/4 |
| sequential-c1 successful_latency_ms.p99 | wyżej | -0,4625% | +0,2356% | +0,0218% | +0,2055% | 3/4 |
| sequential-c1 queue_ms.p50 | wyżej | -24,8241% | +15,9472% | -2,4490% | +16,6667% | 2/4 |
| sequential-c1 queue_ms.p99 | wyżej | -7,2779% | -0,7498% | +1,9417% | -1,9964% | 1/4 |
| sequential-c1 successful_requests_per_second | niżej | +0,2523% | -0,2128% | -0,2707% | -0,0567% | 3/4 |
| sequential-c1 successful_questions_per_second | niżej | +0,2523% | -0,2128% | -0,2707% | -0,0567% | 3/4 |
| mixed-c1 successful_latency_ms.p50 | wyżej | +0,1872% | -0,1180% | +0,1138% | -0,1385% | 2/4 |
| mixed-c1 successful_latency_ms.p95 | wyżej | -0,0226% | -0,0074% | -0,0005% | -0,0117% | 0/4 |
| mixed-c1 successful_latency_ms.p99 | wyżej | -8,1997% | -0,3140% | +0,1318% | +0,2240% | 2/4 |
| mixed-c1 queue_ms.p50 | wyżej | -0,6547% | +0,7073% | -2,7748% | +1,1945% | 2/4 |
| mixed-c1 queue_ms.p99 | wyżej | +12,7415% | +1,8956% | -6,3314% | -5,0024% | 2/4 |
| mixed-c1 successful_requests_per_second | niżej | +2,6621% | -0,0004% | -0,1091% | +0,0085% | 2/4 |
| mixed-c1 successful_questions_per_second | niżej | +2,6621% | -0,0004% | -0,1091% | +0,0085% | 2/4 |
| mixed-c16 successful_latency_ms.p50 | wyżej | +5,0117% | -4,5910% | +4,6756% | +2,8748% | 3/4 |
| mixed-c16 successful_latency_ms.p95 | wyżej | +0,2508% | -0,4961% | -2,2572% | -0,2614% | 1/4 |
| mixed-c16 successful_latency_ms.p99 | wyżej | +3,2938% | -0,0426% | +0,0762% | +0,2045% | 3/4 |
| mixed-c16 queue_ms.p50 | wyżej | +26,6716% | -4,8493% | +22,7723% | +21,3337% | 3/4 |
| mixed-c16 queue_ms.p99 | wyżej | -0,5973% | +1,7866% | -0,0940% | +0,3777% | 2/4 |
| mixed-c16 successful_requests_per_second | niżej | -0,3732% | -0,2413% | -0,2015% | -0,1076% | 4/4 |
| mixed-c16 successful_questions_per_second | niżej | -0,3732% | -0,2413% | -0,2015% | -0,1076% | 4/4 |

Poza dwiema utrzymanymi metrykami żadna nie była gorsza dla B we
wszystkich czterech parach. W 3 z 4 par B było gorsze dla 7 metryk:
`sequential-c1 successful_latency_ms.p50`, `sequential-c1 successful_latency_ms.p99`, `sequential-c1 successful_requests_per_second`, `sequential-c1 successful_questions_per_second`, `mixed-c16 successful_latency_ms.p50`, `mixed-c16 successful_latency_ms.p99`, `mixed-c16 queue_ms.p50`. Przykładowo mediana kolejki mixed-c16 wzrosła o
+26,7%, +22,8% i +21,3%, a w B4/A5 spadła o
-4,8%. O wyniku decyduje ustalona reguła, a nie te liczebności.
Nie ustalono przyczyny pojedynczych różnic.

### Profil A/B (batch 1, 40 pozycji, `q8_0`)

Wartości w ms na pozycję, po jednym procesie na rolę, A przed B. To
zsynchronizowany profil sekcji, a nie latencja żądań mixed-c16 ani bramka
szybkości.

| Sekcja | A | B | B−A | B/A−1 |
|---|---:|---:|---:|---:|
| attention | 6,6531 | 6,6493 | -0,0037 | -0,06% |
| bias+rope+heads | 1,0592 | 1,0655 | +0,0063 | +0,60% |
| bias+silu*up | 1,4109 | 1,4251 | +0,0142 | +1,01% |
| down_proj+bias+residual | 39,8793 | 71,5541 | +31,6748 | +79,43% |
| embed+inputs | 0,6194 | 0,7934 | +0,1741 | +28,10% |
| final_norm | 0,1129 | 0,0574 | -0,0555 | -49,13% |
| gate_up_proj | 148,0385 | 153,2142 | +5,1756 | +3,50% |
| o_proj+bias+residual+rms_norm | 3,9460 | 3,8596 | -0,0863 | -2,19% |
| qkv_proj | 4,1786 | 4,2128 | +0,0342 | +0,82% |
| rms_norm | 2,6035 | 2,3489 | -0,2546 | -9,78% |
| razem (profilowane) | 208,5944 | 245,2813 | +36,6869 | +17,59% |
| suma sekcji | 208,5013 | 245,1805 | +36,6792 | +17,59% |
| w tym dekwantyzacja q8_0 (już w sekcjach projekcji) | 69,4157 | 97,3589 | +27,9432 | +40,25% |

W obu profilach `residual_abs_max` wynosi 0 we wszystkich 60 warstwach.
Dekwantyzowano po 4 624 220 160 wartości na pozycję w A i B.

## Ograniczona analiza: warunki, profile i zmieniony kod

Analiza dotyczy tylko utrzymanego sygnału. Nie otwiera ponownie przeglądu
poprawności kodu Rust.

### Mapa zmienionego kodu

`git diff 750c146` (tylko odczyt) obejmuje 8 plików, +1119/−141 linii.
Źródła: `crates/basal-gpu/src/{lib,quant,memory}.rs`,
`crates/basal-cli/src/{main,benchmark,config,doctor}.rs`,
`crates/basal-gpu/src/intel/mod.rs`. Bez zmian pozostają `kernels.cu`,
`fused.rs`, `cublaslt.rs`, `gemm.rs`, `build.rs`, `serve.rs`, `basal-core`,
`Cargo.toml` i `Cargo.lock`, więc także wersja Candle (0.11.0, z opisu
backendu).

Ścieżka `q8_0` na forward:

- `matmul_t`: wariant `ProjWeight::Q8(q)` → `self.dequantize_q8(q)` stał
  się wariantem `ProjWeight::Quantized(q)` → `self.dequantize_blocks(q)`;
- poza profilowaniem `dequantize_blocks` zwraca, jak wcześniej,
  `q.dequantize_f16(&q.device())`. To to samo wywołanie Candle, bez
  dodatkowej synchronizacji ani alokacji. Potem następuje ten sam GEMM
  cuBLASLt f16;
- przy profilowaniu ma tę samą sekwencję: synchronizacja, pomiar,
  dekwantyzacja, synchronizacja. Zmieniła się tylko nazwa pola
  (`q8_dequant` → `dequant`).

Ładowanie `q8_0`:

- `load_q8` stał się generycznym `load_blocks` z
  `quant::quantize_q8_0`, `keep_blocks = false`;
- te same odczyty, to samo `load_quantized` i te same kontrole (typ i
  bajty bloków, to samo urządzenie);
- warm-up `q8_0` (najmniejsza macierz warstwy 0) pozostał bez zmian;
- `quantize_q8_0` i `quantize_rows` nie mają zmian w diffie.

Pozostałe zmiany:

- pole `q8_stats` (`Option<Q8Stats>`) zastąpiło `quant_stats`
  (`Option<QuantStats>`). Jest klonowane w `fork()`, czyli przy tworzeniu
  drugiego silnika dla długiej kolejki przy starcie, a nie na żądanie;
- nowe ramiona `q4_k` nie wykonują się przy `q8_0`;
- `memory.rs` zmienił tylko komentarz;
- `benchmark.rs`, `config.rs`, `doctor.rs`, `main.rs` i `intel/mod.rs`
  zmieniły teksty, parsowanie i obsługę `q4_k`. Profil zachowuje klucze
  `q8_dequant_*`.

Co pokazują dane o tej mapie:

- 20 strukturalnych opisów backendu `q8_0` (12 bench i 8 benchmark; 8 bazy
  i 12 kandydata) jest identycznych, łącznie z obiektem `quantization`:
  144 506 880 bloków, 4 913 364 480 B z
  paddingiem, 3495796 bloków ze skalą subnormalną;
- pula po załadowaniu (5 309 140 480 B used,
  5 368 709 120 B reserved) i szczyt
  ładowania (5 395 567 368 B) są równe w A i B;
- szacunek wag z arytmetyki formatu (5 179 553 656 B) też
  jest równy;
- jedyne różnice raportów A i B przewidziane przez diff to teksty zakresu
  puli i metody szacunku w `benchmark.rs`. Są widoczne w surowych JSON.

### Mechanizmy i co pokazują dane

1. **Inna praca GPU na forward w ścieżce `q8_0`.** Źródło nie pokazuje
   takiej różnicy: wywołanie dekwantyzacji i GEMM są te same, a bloki,
   statystyki i pula są równe. Nie zmierzono jądra ani kodu maszynowego
   (brak nsys), więc to argument źródłowy, nie pomiar.
2. **Inny kod hosta lub układ binarki.** Binarki mają różny rozmiar
   (provenance: 271 840 560 i
   272 393 368 B). Zmiany typów i
   generyków mogą zmienić kod maszynowy, inlining i układ. Wpływ na
   mixed-c16 jest możliwy, ale niezmierzony i niewykazany. Zakresy czasu
   CPU procesów benchmark A (486,551–494,679 s) i B
   (485,114–489,261 s) nakładają się.
3. **Warunki (zegar, moc, temperatura).** To agregaty całych CSV przebiegu,
   bez okien faz, bo surowe raporty nie mają znaczników czasu faz. Różnica
   B−A średniego zegaru SM w czterech parach wynosi
   -0,29, -0,74, +0,55, +0,14 MHz, a odsetka próbek poniżej
   1200 MHz -0,30, +1,47, +0,27, -0,23 p.p. Kierunki są mieszane.
   Średnia moc B−A: +4,03, -0,69, -0,21, -0,21 W. Szczegóły
   pokazuje tabela w sekcji „Warunki”. Warunki nie dają spójnego
   wyjaśnienia ani nie wykluczają przebiegów.
4. **Profil batch 1.** Profil B był wolniejszy o 36,69 ms na
   pozycję (+17,59%). Dekwantyzacja to
   +27,94 ms, a różnica skupia się w `down_proj`
   (+31,67 ms) i `gate_up` (+5,18 ms). To
   jeden proces na rolę. Ścieżka profilowania ma w obu binarkach tę samą
   sekwencję synchronizacji. Wcześniejsze rekordy z innych sesji fizycznych
   nie są parami, ale dają kontekst:

| Rekord | Binarka | Razem [ms] | Dekwant. [ms] | down_proj [ms] | gate_up [ms] |
|---|---|---:|---:|---:|---:|
| S3 A (ten pakiet) | baza `3bdea953…` | 208,59 | 69,42 | 39,88 | 148,04 |
| S3 B (ten pakiet) | kandydat `a3ed4ade…` | 245,28 | 97,36 | 71,55 | 153,21 |
| S1 (inna sesja fizyczna) | kandydat `a3ed4ade…` | 232,80 | 95,45 | 57,24 | 146,09 |
| W8, raport `w8a16-2026-10-11` (inna sesja) | historyczna binarka W8 | 245,49 | 97,11 | 71,54 | 153,08 |

   Czas dekwantyzacji B (97,36 ms) jest bliski rekordowi S1
   tego samego kandydata (95,45 ms) i rekordowi historycznej binarki W8
   (97,11 ms). A (69,42 ms) jest niższe od wszystkich trzech. Te rekordy
   nie są parami i nie rozstrzygają, czy różnica S3 wynika z binarki, czy z
   procesu lub sesji. Profil S3 ma po jednym procesie na rolę, więc nie
   pokazuje rozrzutu między procesami. Raport W8 zaznaczał, że
   synchronizacje profilu mogą zmieniać oddawanie pamięci puli. To możliwy
   mechanizm wspólny dla obu binarek, niezmierzony. Różnica 17,6% nie
   przenosi się na bench pojedynczych pozycji (mediana B/A per pytanie 1,00011) ani na sequential c1
   (żądania/s B/A−1 od -0,27% do +0,25%). Profil nie mierzy latencji
   żądań mixed-c16 i nie dowodzi przyczyny ich różnicy.

### Wniosek ograniczony

Nie wykazano konkretnego defektu przyczynowego w zmienionym kodzie ścieżki
`q8_0`, więc pakiet nie proponuje poprawki ani optymalizacji. Przyczyna
utrzymanego spadku mixed-c16 o 0,11–0,37% pozostaje nierozstrzygnięta
(wniosek (c)). Nie twierdzimy, że kod jest wolny od defektu, że wynik to
szum statystyczny ani że regresji nie ma. Pakiet nie wprowadza też nowej
tolerancji. Nie znaleziono sprzeczności źródła z danymi, którą trzeba
zgłosić. Surowy wniosek runnera pozostaje zapisany bez zmian.

## Decyzja użytkownika

Użytkownik odpowiedział 2026-10-12 (rekord prywatny, SHA-256 odpowiedzi
`83897d39…`). Akceptuje obecny, zaobserwowany spadek
przepustowości `q8_0` kandydata W4 wobec przebudowanej bazy W8 w mixed-c16
(0,11–0,37%, cztery pary) jako tolerowany kompromis produktowy. Spadki rzędu
1–2% uznał za normalne. Decyzja jest zapisana w
`phases/04-w4a16/performance-decision.md` (`af330ead…`).

Zakres decyzji:

- to akceptacja bieżącego pomiaru, a nie dowód braku ryzyka lub regresji;
- nie tworzy automatycznej bramki 2% ani nie zmienia reguły historycznej;
- nie usuwa sygnału ani statusów i nie wymaga kolejnej płatnej sesji;
- nie dotyczy większego kosztu czasu `q4_k` wobec f16 z S2 ani decyzji
  jakościowej M2 (`q4_k` opcjonalny i eksperymentalny, domyślnie i
  rekomendowane `none`, ścisłe FAIL FP32 z S1 zapisane).

## Kompromis `q8_0` / `q4_k` (B wobec C, opisowo)

Brak reguły sygnału i kryterium szybkości. Różnice nie mają przypisanej
przyczyny. Jakości `q4_k` tu nie mierzono ponownie.

### `basal bench` B/C

| Para | Kolejność | lat2 p50 [ms] B / C | lat2 p95 [ms] B / C | dec/s B / C | Mediana C/B per pytanie | C szybsze | Trafność B / C |
|---|---|---|---|---|---:|---:|---|
| 0 | BC | 42,242 / 40,812 | 49,622 / 48,450 | 56,41 / 56,54 | 0,97589 | 39/39 | 35/44 / 35/44 |
| 1 | CB | 42,231 / 41,078 | 49,603 / 48,566 | 56,54 / 56,42 | 0,97703 | 39/39 | 35/44 / 35/44 |
| 2 | BC | 42,336 / 40,843 | 49,688 / 48,416 | 56,36 / 56,55 | 0,97367 | 39/39 | 35/44 / 35/44 |
| 3 | CB | 42,256 / 41,191 | 49,659 / 48,549 | 56,46 / 56,22 | 0,97795 | 39/39 | 35/44 / 35/44 |

Agregat `ab.py` daje medianę C/B per pytanie 0,97660. C było
szybsze w 39 z 39 pytań. Równa trafność 35/44 na 44 pozycjach nie świadczy
o równych odpowiedziach ani o zachowaniu jakości.

### `basal benchmark` B/C

| Metryka | B1 | C2 | C2/B1−1 | B4 | C3 | C3/B4−1 |
|---|---:|---:|---:|---:|---:|---:|
| sequential-c1 successful_latency_ms.p50 | 42,81 | 42,07 | -1,74% | 43,00 | 41,96 | -2,43% |
| sequential-c1 successful_latency_ms.p95 | 56,60 | 56,24 | -0,63% | 56,77 | 56,26 | -0,90% |
| sequential-c1 successful_latency_ms.p99 | 64,69 | 64,64 | -0,08% | 65,15 | 64,66 | -0,74% |
| sequential-c1 queue_ms.p50 | 0,374 | 0,475 | +27,01% | 0,483 | 0,473 | -2,17% |
| sequential-c1 queue_ms.p99 | 0,981 | 1,067 | +8,77% | 1,059 | 0,967 | -8,69% |
| sequential-c1 successful_requests_per_second | 21,065 | 21,353 | +1,37% | 21,000 | 21,358 | +1,70% |
| sequential-c1 successful_questions_per_second | 21,065 | 21,353 | +1,37% | 21,000 | 21,358 | +1,70% |
| mixed-c1 successful_latency_ms.p50 | 62,11 | 61,91 | -0,32% | 62,06 | 61,99 | -0,11% |
| mixed-c1 successful_latency_ms.p95 | 1120,78 | 1119,96 | -0,07% | 1120,89 | 1120,46 | -0,04% |
| mixed-c1 successful_latency_ms.p99 | 3194,17 | 3189,44 | -0,15% | 3185,69 | 3181,73 | -0,12% |
| mixed-c1 queue_ms.p50 | 0,911 | 0,893 | -1,87% | 0,925 | 0,932 | +0,70% |
| mixed-c1 queue_ms.p99 | 104,508 | 86,717 | -17,02% | 90,361 | 89,660 | -0,78% |
| mixed-c1 successful_requests_per_second | 4,465 | 4,486 | +0,46% | 4,468 | 4,481 | +0,29% |
| mixed-c1 successful_questions_per_second | 10,260 | 10,307 | +0,46% | 10,267 | 10,296 | +0,29% |
| mixed-c16 successful_latency_ms.p50 | 681,78 | 668,27 | -1,98% | 651,07 | 669,39 | +2,81% |
| mixed-c16 successful_latency_ms.p95 | 18857,37 | 18812,62 | -0,24% | 18767,99 | 18771,45 | +0,02% |
| mixed-c16 successful_latency_ms.p99 | 48565,53 | 48411,57 | -0,32% | 48638,09 | 48406,51 | -0,48% |
| mixed-c16 queue_ms.p50 | 299,659 | 283,916 | -5,25% | 287,399 | 289,805 | +0,84% |
| mixed-c16 queue_ms.p99 | 43503,088 | 43414,023 | -0,20% | 44402,430 | 43632,652 | -1,73% |
| mixed-c16 successful_requests_per_second | 5,239 | 5,262 | +0,43% | 5,220 | 5,254 | +0,66% |
| mixed-c16 successful_questions_per_second | 12,039 | 12,091 | +0,43% | 11,994 | 12,073 | +0,66% |

Latencje i kolejki są w ms, przepustowość w żądaniach lub pytaniach na
sekundę. Przeciwne kierunki w dwóch parach mają: `sequential-c1 queue_ms.p50`, `sequential-c1 queue_ms.p99`, `mixed-c1 queue_ms.p50`, `mixed-c16 successful_latency_ms.p50`, `mixed-c16 successful_latency_ms.p95`, `mixed-c16 queue_ms.p50`.

## Pamięć i czas ładowania (4.5B)

| Rola | Przebiegi | load_s benchmark [s] | Pula used po załadowaniu [B] | Pula used, szczyt ładowania [B] | Pula used, szczyt pracy [B] | Pula reserved, szczyt pracy [B] | Maks. próbka NVML GPU 0 [MiB] |
|---|---:|---|---|---|---|---|---|
| A | 4 | 12,53–12,86 | 5 309 140 480 | 5 395 567 368 | 8 044 867 864–8 196 516 988 | 8 321 499 136–8 455 716 864 | 8456–8584 |
| B | 4 | 12,47–12,92 | 5 309 140 480 | 5 395 567 368 | 8 044 867 864–8 196 516 988 | 8 220 835 840–8 422 162 432 | 8360–8552 |
| C | 2 | 25,44–26,13 | 2 996 968 960 | 3 083 395 848 | 5 732 696 344 | 5 939 134 464–6 174 015 488 | 6184–6408 |

Wartości `reserved` puli:

- po załadowaniu: 5 368 709 120 B dla A i B oraz
  3 053 453 312–3 087 007 744 B dla C;
- w szczycie ładowania: 5 435 817 984 B dla A i B oraz
  3 120 562 176 B dla C;
- po zamknięciu silnika `used` i `reserved` wynoszą 0 we wszystkich 10
  przebiegach.

Pula obejmuje tylko alokator stream-ordered. Nie obejmuje kontekstu CUDA,
modułów, uchwytów bibliotek ani alokacji synchronicznych, więc nie jest
całym VRAM. Szczyt pracy obejmuje wszystkie fazy razem. NVML to próbki
urządzenia co 200 ms, a nie szczyty.

Szacunek wag z arytmetyki formatu (nie pomiar) wynosi
5 179 553 656 B dla `q8_0` i
2 867 382 136 B dla `q4_k`. Bloki 240 macierzy
projekcji zajmują z paddingiem 4 913 364 480 B (`q8_0`) i
2 601 192 960 B (`q4_k`). Gęste f16 tych samych projekcji
zajmowałoby 9 248 440 320 B. Największy tymczasowy tensor f16 ma
90 177 536 B.

Zakres `load_s` różni się między poleceniami:

- `basal benchmark` mierzy ładowanie bez tworzenia kontekstu CUDA, bo
  kontekst tworzy wcześniej sonda puli;
- `basal bench` i `basal profile` mierzą je razem z tworzeniem kontekstu.

| Polecenie i rola | load_s [s] |
|---|---|
| bench A (A/B) | 12,84–19,83 |
| bench B (A/B) | 12,84–13,53 |
| bench B (B/C) | 12,82–13,61 |
| bench C (B/C) | 26,18–29,40 |
| profil A / B (linia logu) | 13,1 / 12,9 |

Pierwszy bench A ma `load_s` 19,83 s, a pozostałe są w
zakresie 12,84–12,98 s. Pakiet nie przypisuje tej różnicy przyczyny.
`load_s` `q4_k` obejmuje kwantyzację na hoście, ale silnik nie mierzy jej
osobno. Czasy procesów (rekordy runnera):

- benchmark C: 445,389–446,567 s, CPU 651,535–657,202 s;
- benchmark B: 433,988–435,442 s, CPU 485,114–489,261 s;
- benchmark A: 433,833–439,933 s, CPU 486,551–494,679 s;
- `ab.py` B/C: 252,167 s przy CPU 1269,847 s;
- `ab.py` A/B: 209,060 s przy CPU 563,213 s.

RSS hosta, w bajtach:

| Rola | sequential c1 | mixed c1 | mixed c16 |
|---|---|---|---|
| A | 727 760 896–728 928 256 | 867 889 152–941 178 880 | 914 149 376–955 449 344 |
| B | 728 010 752–728 420 352 | 867 946 496–869 388 288 | 913 080 320–914 702 336 |
| C | 725 917 696–725 942 272 | 865 239 040–866 279 424 | 910 970 880–911 048 704 |

Te wartości to `lifetime_peak_rss_bytes` fazy: próbkowany co 200 ms szczyt
`ru_maxrss` od startu procesu. Proces łączy serwer HTTP i generator
obciążenia, więc wartość obejmuje też ładowanie. Nie jest to izolowany
szczyt ładowania, przyrost w fazie ani zużycie hosta.

## Ładowania `q4_k` i `q8_0`

Sześć ładowań C: cztery bench i dwa benchmark. Każde zapisało pełny obiekt
`quantization`.

- Kanoniczny SHA-256 obiektu (przeliczony: posortowane klucze,
  `ensure_ascii=False`) jest identyczny we wszystkich sześciu ładowaniach.
  Jest równy zapisowi runnera i jedynemu obiektowi 4.5B z S1 (`1f15b087…`,
  6 ładowań S1).
- Autor porównał statystyki modelu i czterech typów projekcji z
  zatwierdzonym P0 (`44005ecc…`): liczby całkowite, bity f64 `sum_e2`,
  `sum_w2`, `max_abs_error` i R, status R oraz liczniki grup z pierwszymi
  pozycjami (null tylko przy liczniku 0). To 6 × 5 = 30 wierszy, wszystkie
  równe. `results.q4_p0_equality` runnera zawiera pełne wiersze liczbowe
  silnika i P0 dla tych samych 30 zakresów. Każdy wiersz ma `equal: true`,
  a status PASS jest przypisany do ładowania, nie do wiersza. Wiersze autora
  są równe wierszom runnera. Kwantyzatora ani P0 nie uruchamiano ponownie.
- Względne RMS modelu wynosi `0.07621214682255444`. Grupy:
  9 spłaszczonych i 0 zerowych na 144 506 880. Linie logów zgadzają się z
  obiektami.
- Warm-up w każdym ładowaniu: macierz warstwy 0 `o` (4 194 304 wartości),
  zdekwantyzowana na GPU, jest bitowo równa rekonstrukcji hosta. To jedna
  macierz na ładowanie, a nie równość wszystkich macierzy GPU.

Ładowania `q8_0`: 22. Strukturalny opis ma 20 z nich (12 bench i 8
benchmark). Dwa ładowania w profilach A/B widać tylko w logu, bo profil nie
zapisuje opisu backendu. Ładowań `none` w S3 nie ma. Razem 28 logów
procesów GPU: 22 `q8_0` i 6 `q4_k`.

## Warunki

| Para | Przebieg | Średni zegar SM [MHz] | Próbki SM < 1200 MHz | Średnia moc [W] | Maks. temp. [°C] | wall_s mixed-c16 [s] |
|---|---|---:|---:|---:|---:|---:|
| A0/B1 | primary/0-A (A) | 1193,65 | 10,10% | 251,78 | 77 | 171,134 |
| A0/B1 | primary/1-B (B) | 1193,36 | 9,80% | 255,81 | 77 | 171,775 |
| B4/A5 | primary/5-A (A) | 1195,98 | 7,64% | 255,77 | 77 | 172,009 |
| B4/A5 | primary/4-B (B) | 1195,25 | 9,10% | 255,08 | 76 | 172,425 |
| A0/B1 (confirmation pair 0) | confirmation/0-A (A) | 1193,21 | 10,60% | 256,30 | 77 | 171,147 |
| A0/B1 (confirmation pair 0) | confirmation/1-B (B) | 1193,76 | 10,87% | 256,09 | 77 | 171,492 |
| A3/B2 (confirmation pair 1) | confirmation/3-A (A) | 1194,22 | 8,95% | 256,60 | 77 | 171,078 |
| A3/B2 (confirmation pair 1) | confirmation/2-B (B) | 1194,36 | 8,72% | 256,39 | 77 | 171,263 |

Wartości pochodzą z całych CSV przebiegów (autor), a nie z okien faz.

- GPU 0, wszystkie 12 CSV:
  - próbki zegara SM: 990–1200 MHz przy żądanych
    1200 MHz, więc zegar nie był stały;
  - moc do 298,04 W przy limicie 300 W;
  - temperatura do 77 °C;
  - maksymalna przerwa między próbkami 0,234 s.
- Zegar pamięci nie był blokowany. W przebiegach benchmark prawie
  wszystkie próbki GPU 0 mają 9501 MHz (P2), a po 3 próbki 10001 MHz (P0).
- Okna runnera mają pokrycie co najmniej 0,999 w każdym przebiegu
  benchmark i 0,992 w przebiegach `ab.py`, bez flagi `missing`. Wartości okien
  benchmark są równe agregatom autora, poza opisaną różnicą liczby próbek
  w confirmation 0-A.
- GPU 1: wykorzystanie maksymalnie 0% i pamięć maksymalnie
  4 MiB we wszystkich 12 CSV.
- Obserwator procesów obliczeniowych runnera: 0 obcych procesów i 0
  nieudanych odczytów w 14 poleceniach. Profile nie mają CSV, tylko po
  10 i 11 próbek tego obserwatora.
- Sonda GEMM `basal bench` przed i po: 104,7–121,3 TFLOPS.

Warunki opisują pomiar. Nie wykluczają przebiegów i nie dowodzą zakłócenia.

## Obserwacje parenta (przeniesione)

Stan z zapisanych rekordów parenta (2026-10-12T12:28:18 UTC i 2026-10-12T12:40:18 UTC):

- żadnej z 11 posiadanych instancji nie ma w bieżącym odczycie
  głównego API;
- każdy z 11 wpisów ledgera ma status `deleted`. Pierwsze 5 starszych
  wpisów nie ma flagi nieobecności, a historycznych flag nie przepisano;
- cztery blokady runtime S3 były wolne.

To obserwacje parenta z chwili zapisu. Autor nie wykonywał zapytań API i
nie twierdzi, że w chmurze nie ma żadnych zasobów.

## Ograniczenia

- Jeden model, jeden typ karty, jedna sesja i GPU 0; wyniki nie są
  uogólniane. Brak CPU, innych zegarów, modeli, GPU oraz nsys.
- Utrzymany sygnał i jego wielkość opisują tę sesję. Wniosek (c) i decyzja
  użytkownika nie dowodzą braku regresji ani szumu.
- Analiza źródła to ograniczony odczyt zmienionej ścieżki `q8_0`, bez
  pomiaru kodu maszynowego i bez ponownego przeglądu poprawności.
- Warunki są agregatami całych przebiegów; fazy nie mają znaczników czasu.
- Profil to jeden proces na rolę, synchronizowany, batch 1. Nie opisuje
  latencji żądań. Rekordy W8 i S1 pochodzą z innych sesji fizycznych i nie
  są parami.
- Kompromis `q4_k` jest opisowy: bez bramki szybkości, bez przyczynowości
  i bez nowej oceny jakości.
- Spójność odpowiedzi benchmarku dotyczy wnętrza przebiegu. Pakiet nie
  zawiera porównania odpowiedzi między binarkami ani z FP32.
- Pamięć: pula tylko stream-ordered, NVML jako próbki, RSS jako próbkowany
  szczyt od startu procesu. Pakiet nie podaje izolowanego szczytu
  ładowania.
- Dekwantyzację GPU `q4_k` zmierzono dla jednej macierzy na ładowanie.
- Pakiet nie zmienia historycznych raportów, kryteriów ani statusów.

## Pliki

- `facts.json`: tożsamości, statusy runnera, protokół, wyniki zbiorcze,
  ograniczona analiza (mapa kodu, mechanizmy, wniosek), decyzja
  użytkownika, pochodzenie wartości, obserwacje parenta i ograniczenia.
- `perf-q8-family.json`: wszystkie pary bench (z 39 ilorazami per pytanie)
  i benchmark (21 metryk), 10 przebiegów (fazy, pule, zasoby, warunki,
  CSV), 16 raportów bench, sygnały, profile z rekordami kontekstowymi,
  sześć ładowań `q4_k` z wierszami P0, opisy `q8_0` i rekordy 14 poleceń.
  Wartości są surowe i przechodzą round trip bitowo.
- `validation.json`: kontrole autora z wynikami, SHA odczytanych plików,
  metoda, obserwacje runnera, kolektora i parenta oraz zakres
  nieprzeliczony.
