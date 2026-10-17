# Wiele GPU i kwantyzacja wag — zestawienie wyników, 2026-10-14

Ten dokument opisuje funkcje dodane w PR #7 i wyniki ich pomiarów z 2026-10-11 – 2026-10-14. Nie jest nowym
pomiarem. Nie jest też formalną akceptacją, certyfikacją ani gwarancją zachowania na innym sprzęcie. Wszystkie
liczby pochodzą z raportów źródłowych wymienionych na końcu. Pełne dane JSON tych raportów są w archiwum
`multigpu-2026-10-14`; pobieranie i sumy kontrolne opisuje
[MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).

## Najważniejsze wnioski

- Bez nowych ustawień każdy model działa na jednym GPU, bez kwantyzacji. Eksporty z domyślnym `none` są bitowo
  równe wersji bazowej w każdej sesji, która to sprawdzała. Wydajność tego toru ma jeden utrzymany sygnał
  regresji (kolejka p99 w L2), przyjęty przez użytkownika.
- Różne modele można przypisać do różnych GPU (`device`). Wyniki pozostają bitowo równe. Przy obciążonym 4.5B na
  drugiej karcie mediana p99 mini zmieniła się o −0,86% do +0,37% (cztery przebiegi na wariant).
- Repliki jednego modelu (`replica_devices`) dały dla mini 2,06× i 2,07× przepustowość jednej repliki przy 16 i 32
  klientach.
- Podział warstw (`layer_split`) pozwala uruchomić model, który nie mieści się na jednej karcie. Wyniki są bitowo
  równe wynikom jednego GPU w tym samym trybie. Pojedyncze żądanie jest z podziałem wolniejsze.
- `q8_0` jest opcjonalne: zmniejsza pamięć wag projekcji, ale wydłuża czas i zmienia część odpowiedzi.
- `q4_k` jest opcjonalne i eksperymentalne: zmniejsza pamięć jeszcze bardziej, ale zmienia odpowiedzi znacznie
  częściej niż `q8_0`. Na 4.5B daje w próbce 900 pytań 685 poprawnych odpowiedzi wobec 702 dla f16.
- Domyślny i zalecany tryb to `none`.
- Kilka pomiarów wydajności zakończyło się utrzymanym sygnałem regresji. Sygnały zostają w raportach. Każdy ma
  zapisane rozstrzygnięcie i jego zakres (sekcja „Sygnały regresji i ich rozstrzygnięcia”).

## Pojęcia

| Pojęcie | Znaczenie |
|---|---|
| PASS / FAIL / GAP / SIGNAL | statusy kontroli runnera: spełniona, niespełniona, niezmierzona lub nieobserwowalna, sygnał wymagający analizy lub decyzji |
| A / B, BASE / CAND | wersja bazowa i kandydat w parach pomiarowych; świeży proces w każdym przebiegu |
| C | ten sam kandydat w innej konfiguracji (np. z podziałem warstw albo innym formatem wag), porównywany z B opisowo |
| reguła bez progu | sygnał utrzymuje się, gdy B jest gorsze we wszystkich parach rundy głównej i rundy potwierdzenia; nie ma progu procentowego ani testu statystycznego |
| pula | bajty `used` puli alokatora strumieniowego CUDA procesu; bez kontekstu CUDA, modułów, uchwytów bibliotek i alokacji synchronicznych, więc nie jest to cały VRAM |
| bitowo równe | pliki wyników (`bench.jsonl`, `systemone.jsonl`) mają identyczne bajty; `basal compare` pokazuje różnicę 0 |
| `none`, `q8_0`, `q4_k` | format wag czterech projekcji każdej warstwy: f16 bez kwantyzacji, bloki GGML Q8_0, bloki GGML Q4_K |

## Sprzęt i zakres pomiarów

Wszystkie sesje GPU wykonano na wynajętych maszynach z dwiema kartami NVIDIA RTX 6000 Ada Generation 48 GB, ze
sterownikiem 580.126.09, CUDA/cuBLASLt 12.9.1 i limitem mocy 300 W. Placement, izolacja i pierwszy pomiar replik
działały z automatycznymi zegarami; od pomiaru kontrolnego replik żądano zegara SM 1200 MHz. Modele: basal-1.5-mini
i basal-1.5-4.5B w f16. Model basal-1.5-max, inne karty (np. RTX 4090), karty różnych typów w jednej grupie, Metal i
Intel nie były mierzone w tych funkcjach. P0 nie było sesją GPU: to jednorazowa diagnostyka błędu rekonstrukcji wag
na lokalnym CPU, bez powtórzeń czasu. Nie jest to pomiar GPU ani kalibracja CPU w chmurze.

Różne sesje to różne wynajęte maszyny tej samej konfiguracji. Dane nie zawierają identyfikatorów fizycznych kart,
więc pary A/B porównujemy tylko w obrębie jednej sesji. Wyniki różnych sesji zestawiamy opisowo.

## Funkcje

| Funkcja | Ustawienie | Co daje | Czego nie daje | Raporty |
|---|---|---|---|---|
| Model na wybranym GPU | `device: N` przy modelu | różne modele na różnych kartach w jednym serwerze i jednym API | nie przyspiesza pojedynczego modelu | [placement](../multigpu-placement-2026-10-11/README.md), [izolacja](../multigpu-isolation-2026-10-11/README.md) |
| Repliki | `replica_devices: [0, 1]`, `--replica-devices 0,1` | niezależna kopia modelu na każdej karcie, jedna nazwa w API, routing do najmniej obciążonej repliki | nie zmniejsza pamięci na kartę; nie ponawia żądania po rozpoczęciu obliczeń | [repliki](../multigpu-replicas-2026-10-11/README.md), [kontrola](../multigpu-replicas-control-2026-10-11/README.md) |
| Podział warstw | `layer_split`, `--layer-split 0:30,1:30` | jeden model rozłożony warstwami na kilka kart jednego procesu; mniej pamięci na kartę | nie przyspiesza; dodaje kopię aktywacji na granicy etapów | [zestawienie L1–L3](../multigpu-layer-revalidation-2026-10-14/README.md) |
| W8A16 | `quantization: q8_0`, `--quantization q8_0` | wagi projekcji jako bloki Q8_0; mniejsza pula pamięci | wolniejsze obliczenia; zmienione części odpowiedzi | [W8A16](../w8a16-2026-10-11/README.md), [jakość W8](../multigpu-w8-quality-2026-10-11/README.md) |
| W4A16 (eksperymentalne) | `quantization: q4_k`, `--quantization q4_k` | wagi projekcji jako bloki Q4_K; najmniejsza pula pamięci | nie spełnia kryterium zgodności z FP32; wolniejsze niż f16 | [P0](../w4a16-p0-2026-10-12/README.md), [S1](../w4a16-s1-2026-10-12/README.md), [S2](../w4a16-s2-2026-10-12/README.md), [S3](../w4a16-s3-2026-10-12/README.md) |

Wszystkie funkcje działają tylko w buildzie CUDA. Metal i Intel obsługują wyłącznie `device: 0`; kwantyzacja i
podział warstw są tam odrzucane przed pobraniem modelu. Kwantyzacja wymaga `dtype: f16` i domyślnych kerneli
`fused`. Podział warstw i repliki wymagają kart jednego typu. Szczegóły konfiguracji opisuje
[README](../../README.md).

## Zgodność wyników

| Pomiar | Wynik |
|---|---|
| Placement (2 modele, 2 GPU) | 24 eksporty bitowo równe wersji bazowej; 1233 odpowiedzi HTTP równe odpowiedziom osobnych serwerów; 167 PASS, 0 FAIL, 0 GAP |
| Izolacja | 104 208 mierzonych odpowiedzi mini zgodnych z referencją; 292 PASS, 0 FAIL, 0 GAP |
| Repliki | 24 eksporty bitowo równe; 1066 odpowiedzi HTTP i wszystkie 165 600 mierzonych odpowiedzi zgodne z referencją; 401 PASS, 0 FAIL, 1 GAP oraz 134 PASS, 0 FAIL, 0 GAP w pomiarze narzutu |
| Podział warstw L1 | 44 pary eksportów bitowo równe (20 par toru jednego GPU CAND wobec BASE, 24 pary podziału wobec jednego GPU, tryby `none`, `q8_0`, `q4_k`); 928 odpowiedzi HTTP CAND porównanych z referencją BASE: 0 różnic; 599 PASS, 0 FAIL, 42 GAP, 0 SIGNAL |
| `q8_0` i `q4_k` między kartami | eksporty bitowo równe między GPU 0 i 1, trybami pakowania, `--no-prefix-cache` i `CUDA_VISIBLE_DEVICES`; repliki dają odpowiedzi równe pojedynczemu serwerowi tego samego trybu |

Zgodność w tej tabeli oznacza równość z poprzednią wersją lub z jednym GPU w tym samym trybie. Nie jest nowym
porównaniem z FP32 upstream. Porównanie z FP32 dla kwantyzacji opisuje następna sekcja.

## Jakość kwantyzacji

Jakość mierzono wobec referencji FP32 upstream na dwóch zestawach:

- ścisłe kryterium zgodności: 44 przykłady bench, 34 żądania System One i 55 pytań endpointu, bez tolerancji i bez
  wyjątków;
- 900 pytań diagnostycznych z etykietą (400 choice, 200 score, 300 noul), na których liczymy zmiany decyzji i
  poprawne odpowiedzi.

### Poprawne odpowiedzi na 900 pytaniach

| Model | FP32 | f16 | `q8_0` | `q4_k` |
|---|---:|---:|---:|---:|
| mini | 664 | 663 | 664 | 668 |
| 4.5B | 703 | 702 | 704 | 685 |

Liczby `q8_0` pochodzą z [pomiaru jakości W8](../multigpu-w8-quality-2026-10-11/README.md), a `q4_k` z [sesji
S1](../w4a16-s1-2026-10-12/README.md). Kontrola f32 w obu sesjach dała 900/900 decyzji równych FP32 upstream, więc
różnice można przypisać formatom wag. To wynik jednej próbki 900 pytań. Nie oznacza ogólnej poprawy ani zachowania
jakości i nie przenosi się na inne zbiory.

### Zmiany decyzji i kryterium FP32

| Wynik | `q8_0` | `q4_k` |
|---|---|---|
| Nowe zmiany decyzji wobec FP32 na 900 pytaniach (bez pytań zmienionych już przez f16) | mini 3, 4.5B 5 | mini 76, 4.5B 81 |
| Nowe zmiany: poprawiające / pogarszające / obie odpowiedzi błędne | mini 2 / 1 / 0, 4.5B 3 / 2 / 0 | mini 38 / 33 / 5, 4.5B 25 / 43 / 13 |
| Naruszenia ścisłego kryterium FP32 | mini 3, 4.5B 2: granice fragmentów evidence w 3 pytaniach endpointu (mini 2, 4.5B 1); kryterium endpointu: FAIL | mini 26 naruszeń w 15 pozycjach, 4.5B 20 w 12; dwa FAIL klasy `quality-criterion` |
| Decyzje 44 przykładów bench | 44/44 jak FP32 | zmienione w 2 (mini) i 4 (4.5B) przykładach |
| f16 bez kwantyzacji w tej samej sesji | zachowuje granice evidence | 0 naruszeń |

Dodatkowe obserwacje `q4_k`:

- na 4.5B nowe zmiany skupiają się w zbiorze emotion: 32 na 100 pytań; poprawne odpowiedzi w tym zbiorze spadają
  z 64 do 53;
- części zmian nie da się opisać jako remisów: największy margines FP32 zmienionego pytania to 0,804 (mini) i
  0,857 (4.5B);
- średnia całkowita odległość wariacyjna rozkładów odpowiedzi wobec FP32 wynosi 0,0807 i 0,0728, a dla `q8_0`
  0,0052 i 0,0042.

W sprawdzonym zakresie nie znaleziono defektu implementacji żadnego z formatów. Dla `q4_k` skala różnic jest
rzędem wielkości zgodna z błędem rekonstrukcji wag tego formatu. To ograniczone wyniki negatywne, a nie dowód
braku defektu.

### Decyzje o trybach

- `q8_0`: użytkownik przyjął opcjonalne `q8_0` z jawnymi różnicami jakości i zalecił `none` jako tryb domyślny.
  Kryterium FP32 endpointu pozostaje FAIL.
- `q4_k`: użytkownik przyjął `q4_k` jako tryb opcjonalny i eksperymentalny, włączany tylko jawnie. Domyślny i
  zalecany pozostaje `none`. FAIL ścisłego kryterium FP32 z S1 pozostaje zapisany.
- Format `q4_k` wybrano przed pomiarem jakości regułą opartą na błędzie rekonstrukcji wag
  ([P0](../w4a16-p0-2026-10-12/README.md)): względny błąd Q4_K jest o 26,8% (mini) i 26,5% (4.5B) mniejszy niż
  Q4_0. To kryterium w przestrzeni wag, a nie ocena odpowiedzi modelu.

Sesje podziału warstw L1–L3 nie mierzyły jakości wobec FP32. Ich równość bitowa dotyczy podziału wobec jednego
GPU w tym samym trybie.

## Pamięć i czas kwantyzacji

Pula po załadowaniu, w bajtach:

| Model | `none` | `q8_0` | `q4_k` | Źródło |
|---|---:|---:|---:|---|
| 4.5B | 9 644 216 320 | 5 309 140 480 | 2 996 968 960 | W8A16 R5, S2, S3 |
| mini | 3 271 867 392 | 1 868 155 904 | 1 119 439 872 | S1 memory-mini (jeden przebieg na format) |

Czas na 4.5B, GPU 0, żądany zegar SM 1200 MHz:

| Porównanie | Wynik |
|---|---|
| `q8_0` wobec `none` (W8A16 R5) | mediana `latency2` w `basal bench` 25,5 → 42,4–42,6 ms (B/A 1,658–1,669 w czterech parach); żądania/s sekwencyjnie 31,9–32,2 → 21,0; p50 ruchu mieszanego przy 16 klientach 586–598 → 673–675 ms |
| `q4_k` wobec `none` (S2) | p50 latencji żądań sekwencyjnych +56,5% i +56,8%; żądania/s −33,2% sekwencyjnie oraz −7,4% i −7,2% w ruchu mieszanym przy 16 klientach; `load_s` 11,5–12,8 → 25,4–25,5 s |
| `q4_k` wobec `q8_0` (S3) | żądania/s +1,37% i +1,70% sekwencyjnie oraz +0,43% i +0,66% przy 16 klientach; mediana C/B czasu pytania w `basal bench` 0,97660; `load_s` 12,47–12,92 → 25,44–26,13 s |

Obie kwantyzacje dekwantyzują wagi przed każdym GEMM projekcji, więc dodają pracę do każdego forwardu. Wyniki
dotyczą tych obciążeń i tego zegara. Nie są stałym mnożnikiem dla innych kart i ustawień.

## Placement, izolacja i repliki

**Izolacja.** Mini na GPU 1, z obciążonym 4.5B na GPU 0 lub bez niego. Mediana p99 zmieniła się o −0,12% (16
klientów), +0,37% (50 żądań/s) i −0,86% (100 żądań/s). Porównanie starej i nowej binarki na jednej karcie dało
+1,40%, +0,61% i −0,23%. Każdy wariant ma cztery przebiegi. To nie dowodzi zerowego wpływu drugiego modelu ani
braku regresji.

**Repliki.** Mini, jedna replika (A) wobec dwóch (B):

| Ruch | p99 A / B (ms) | Odpowiedzi/s A / B |
|---|---:|---:|
| 16 klientów | 246,822 / 140,742 | 83,018 / 170,938 |
| 32 klientów | 444,795 / 252,106 | 83,367 / 172,728 |
| 100 żądań/s | 673,384 / 56,703 | 98,216 / 99,113 |

To wynik tego ruchu i tego sprzętu, a nie gwarancja skalowania. Nie mierzono więcej niż dwóch replik, grup kart
różnych typów ani zachowania po awarii karty.

## Podział warstw

Bieżące źródła mierzono w trzech sesjach (L1 zgodność i pojemność, L2 wydajność `none`, L3 wydajność `q8_0` i
`q4_k`). Szczegóły: [zestawienie L1–L3](../multigpu-layer-revalidation-2026-10-14/README.md).

**Pojemność.** W L1 sztuczna rezerwacja zostawiła 7939 MiB wolnej pamięci na każdej karcie 48 GB. 4.5B `none` na
jednym GPU kończy się wtedy czystym błędem braku pamięci, a podział `0:30,1:30` działa i daje wyniki bitowo równe
referencji. To symulacja mniejszej wolnej pamięci. Nie dowodzi działania na fizycznie mniejszej karcie (np.
RTX 4090), dla modelu max, dużych partii ani maksymalnej pojemności.

**Odmowy.** 38 niepoprawnych konfiguracji kończy się oczekiwanym błędem przed pobraniem i ładowaniem modelu. Brak
alokacji GPU w tych procesach nie jest zmierzony (GAP: proces kończy się przed pierwszą próbką pamięci).

**Pamięć.** Pula po załadowaniu 4.5B, MiB:

| Tryb | Jeden GPU | Podział: GPU 0 / GPU 1 |
|---|---:|---|
| `none` | 9197,4 | 4610,7 / 4618,7 |
| `q8_0` | niezmierzone | 2543,6 / 2551,6 |
| `q4_k` | niezmierzone | 1441,1 / 1449,1 |

**Szybkość.** C = podział `0:30,1:30`, B = jeden GPU, ten sam kandydat, ta sama sesja:

| Tryb | Mediana C/B czasu pytania w `basal bench` | HTTP |
|---|---|---|
| `none` | 1,00384–1,00889 | 1 klient: przepustowość C/B 0,9626–0,9732; 16 klientów ruchu mieszanego: 1,1801 i 1,1740 |
| `q8_0` | 1,12804–1,13927 | jeden przebieg z podziałem, bez pary z jednym GPU |
| `q4_k` | 1,18348–1,20554 | jeden przebieg z podziałem, bez pary z jednym GPU |

Pojedyncze pytanie jest z podziałem wolniejsze. Wyższa przepustowość `none` przy 16 klientach to wynik jednej
sesji: w przebiegach jednego GPU karta 0 miała 80–81 °C, a jej zegar SM spadał do 1035–1095 MHz; z podziałem
karty miały 70 i 74 °C. Pomiar nie ustala mechanizmu i nie obiecuje przyspieszenia. Podział służy pojemności.

## Sygnały regresji i ich rozstrzygnięcia

Pomiary regresji porównują kandydata z wersją bazową na jednym GPU, w tym samym trybie. Od W8 obowiązuje reguła
bez progu; pomiar replik był opisowy.

| Pomiar | Wynik | Rozstrzygnięcie |
|---|---|---|
| Repliki: placement wobec nowej binarki, 16 klientów | mediana przepustowości niższa o 3,04%, p99 niższe (lepsze) o 2,25%; przyczyna nieustalona | pomiar kontrolny tych samych binarek przy żądanym zegarze 1200 MHz: +0,030%, kierunki w parach mieszane; oba wyniki zachowane, brak deklaracji braku regresji |
| W8 `none` | 8 metryk gorszych w rundzie głównej, żadna nie utrzymała się w potwierdzeniu | brak utrzymanego sygnału; nie dowodzi zerowego narzutu |
| W4 S2 `none` | 11 metryk gorszych w rundzie głównej, żadna nie utrzymała się; 87 PASS, 0 FAIL, 1 GAP, 0 SIGNAL | brak utrzymanego sygnału; nie dowodzi zerowego narzutu |
| W4 S3 `q8_0` | utrzymany SIGNAL: żądania/s ruchu mieszanego przy 16 klientach −0,3732%, −0,2413%, −0,2015%, −0,1076%; 86 PASS, 0 FAIL, 1 GAP, 1 SIGNAL | użytkownik przyjął ten spadek jako tolerowany kompromis; spadki rzędu 1–2% uznał za mieszczące się w normie |
| Podział L2 `none` | utrzymany SIGNAL `basal benchmark` `mixed-c1 queue_ms.p99`: +1,237, +2,706, +11,330, +3,896 ms (+1,4565% do +12,7872%); 107 PASS, 0 FAIL, 1 GAP, 1 SIGNAL | użytkownik przyjął ten konkretny wynik ([decyzja](../multigpu-layer-l2-revalidation-2026-10-14/DECISION.md)) |
| Podział L3 `q8_0` | `dec_s` gorsze w 7 z 8 par; sygnał nieutrzymany | brak utrzymanego sygnału; nie dowodzi zerowego narzutu |
| Podział L3 `q4_k` | utrzymany SIGNAL `basal bench` `dec_s`: spadek 0,0104938–0,5561912% w 8 z 8 par; 172 PASS, 0 FAIL, 2 GAP, 1 SIGNAL | koordynator zastosował wcześniejszą instrukcję użytkownika o granicy 1–2%; użytkownik nie oceniał tego wyniku ([rozstrzygnięcie](../multigpu-layer-l3-revalidation-2026-10-14/DECISION.md)) |

Zasady wspólne dla tych rozstrzygnięć:

- SIGNAL pozostaje SIGNAL-em, a GAP GAP-em. Rozstrzygnięcie nie zmienia statusu runnera.
- Rozstrzygnięcie dotyczy tylko jednego wyniku i jednej pary binarek. Nie tworzy progu dla przyszłych pomiarów
  ani nie stwierdza, że różnica jest szumem.
- W L2 `queue_ms.p99` to w każdym przebiegu czas oczekiwania przed obliczeniami najwolniejszego z 18 żądań
  `doc-4k-8k` o największym wejściu. Latencja całych żądań i przepustowość `mixed-c1` nie mają utrzymanego
  sygnału według ustalonej reguły. To nie jest dowód zerowego narzutu.
- W L3 zmiana `dec_s` oznacza przebieg 44 pytań dłuższy o 0,082–4,366 ms przy 778,6–785,0 ms. To czas
  `basal bench`, a nie latencja HTTP. Wydajności HTTP jednego GPU w `q8_0` i `q4_k` dla tego kandydata nie
  zmierzono (GAP).
- Analiza danych i przegląd kodu nie wskazały przyczyny w kodzie żadnego utrzymanego sygnału. Nie wykluczają
  wpływu całej binarki, CPU hosta ani szeregowania wątków; CPU hosta nie profilowano.

## Czas ładowania `load_s`

`load_s` nie jest czasem uruchomienia serwera. Aktualny opis jest w
[erratum](../multigpu-layer-revalidation-2026-10-14/LOAD_S.md):

- w BASE i w CAND zegar kończy się przed wczytaniem głowicy evidence i utworzeniem uchwytu cuBLASLt; żadna
  wersja nie wlicza tych kroków;
- CAND ma dodatkowo synchronizację publikacji, także poza `load_s`;
- dlatego `load_s` BASE i CAND nie są porównaniem pełnego czasu uruchomienia ani pomiarem tego samego przedziału
  pracy.

Wcześniejsze zdania zestawienia, raportu L3, rozstrzygnięcia L3 i faktów L3, że `load_s` w `d42a400` obejmował te
kroki, są błędne. Zapisane wartości `load_s` się nie zmieniają.

## Statusy wszystkich sesji

| Sesja | PASS / FAIL / GAP / SIGNAL | Uwagi |
|---|---|---|
| Placement | 167 / 0 / 0 / – | |
| Izolacja | 292 / 0 / 0 / – | |
| Repliki | 401 / 0 / 1 / – oraz 134 / 0 / 0 / – | pierwszy przebieg przerwany (198 / 3 / 22, złe wejścia kontroli); jego wydajność wykluczono |
| Kontrola replik (1200 MHz) | 162 / 0 / 0 / – | wcześniejszy przebieg przy 1500 MHz przerwany (66 / 0 / 0), niewłączony do porównania |
| W8, pierwsza kampania | 302 / 3 / 15 / 4 | `complete: false`; błąd referencji HTTP poprawiono w R5, statusy tej kampanii pozostają |
| W8A16 R5 | 85 / 0 / 1 / 0 | GAP: niewykonany opcjonalny ślad nsys |
| W4 S1 | 491 / 2 / 38 / 6 | 2 FAIL klasy `quality-criterion` |
| W4 S2 | 87 / 0 / 1 / 0 | |
| W4 S3 | 86 / 0 / 1 / 1 | |
| Podział L1 | 599 / 0 / 42 / 0 | GAP: etykieta źródła BASE (1), pamięć odmów (38), karty różnych typów oraz Metal i Intel (3) |
| Podział L2 | 107 / 0 / 1 / 1 | GAP: etykieta źródła BASE |
| Podział L3 | 172 / 0 / 2 / 1 | GAP: etykieta źródła BASE, HTTP jednego GPU w `q8_0` i `q4_k` |

„–” oznacza, że runner tej sesji nie raportował statusu SIGNAL.

Według audytu koordynatora lokalne kontrole statyczne zmierzonych źródeł podziału warstw (`cargo fmt`, Clippy i
rustdoc dla buildów domyślnego, Intel i CUDA, `cargo-deny`) zakończyły się kodem 0. CI kompiluje i lintuje kod
Linux, Metal i CUDA, ale nie uruchamia GPU i nie sprawdza wyników modelu ([CI.md](../../docs/CI.md)).

## Co z tego wynika w praktyce

- Bez zmian w konfiguracji wyniki modelu są takie jak w wersji bazowej. Nowych funkcji nie trzeba włączać.
- Kilka modeli na kilku kartach: `device` przy każdym modelu.
- Większa przepustowość jednego małego modelu przy wielu klientach: `replica_devices`. Każda karta potrzebuje
  pełnej kopii modelu.
- Model nie mieści się na jednej karcie: `layer_split`. Trzeba liczyć się z wolniejszym pojedynczym żądaniem.
- Mniej pamięci kosztem jakości i czasu: najpierw `q8_0`. `q4_k` tylko świadomie, po sprawdzeniu na własnych
  danych, bo zmienia odpowiedzi znacznie częściej.
- Wyników nie należy przenosić na RTX 4090, model max, karty różnych typów ani inne zegary i limity mocy bez
  nowego pomiaru.

## Czego nie zmierzono

- model basal-1.5-max we wszystkich tych funkcjach;
- inne karty niż RTX 6000 Ada (np. RTX 4090) i grupy kart różnych typów;
- fizyczny brak pamięci (L1 używał sztucznej rezerwacji) i fizyczną trasę kopii między kartami (P2P albo przez
  hosta);
- awarie kart, panikę workera i częściowy start (tylko przegląd kodu);
- Metal i Intel w trybie wielu GPU;
- jakość wobec FP32 w trybie podziału warstw (L1 sprawdzał równość z jednym GPU);
- wydajność HTTP jednego GPU w `q8_0` i `q4_k` dla kandydata podziału warstw;
- profil CPU hosta i pełny czas uruchomienia serwera.

## Raporty źródłowe

| Raport | Zakres |
|---|---|
| [multigpu-placement-2026-10-11](../multigpu-placement-2026-10-11/README.md) | modele na osobnych GPU |
| [multigpu-isolation-2026-10-11](../multigpu-isolation-2026-10-11/README.md) | izolacja dwóch modeli, stara i nowa binarka na jednym GPU |
| [multigpu-replicas-2026-10-11](../multigpu-replicas-2026-10-11/README.md) | repliki mini, spadek −3,04% na jednym GPU |
| [multigpu-replicas-control-2026-10-11](../multigpu-replicas-control-2026-10-11/README.md) | kontrola przy żądanym zegarze 1200 MHz |
| [multigpu-w8-original-2026-10-11](../multigpu-w8-original-2026-10-11/README.md) | pierwsza kampania W8 z zachowanymi lukami |
| [multigpu-w8-quality-2026-10-11](../multigpu-w8-quality-2026-10-11/README.md) | jakość `q8_0` wobec FP32 |
| [multigpu-w8-none-2026-10-11](../multigpu-w8-none-2026-10-11/README.md) | wydajność `none` po zmianach W8 |
| [w8a16-2026-10-11](../w8a16-2026-10-11/README.md) | końcowy pomiar `q8_0`: pamięć, czas, profile |
| [w4a16-p0-2026-10-12](../w4a16-p0-2026-10-12/README.md) | wybór formatu 4-bit z błędu wag |
| [w4a16-s1-2026-10-12](../w4a16-s1-2026-10-12/README.md) | jakość i poprawność `q4_k` |
| [w4a16-s2-2026-10-12](../w4a16-s2-2026-10-12/README.md) | wydajność rodziny `none` i `q4_k` wobec f16 |
| [w4a16-s3-2026-10-12](../w4a16-s3-2026-10-12/README.md) | wydajność rodziny `q8_0` i `q4_k` wobec `q8_0`; decyzja użytkownika |
| [multigpu-layer-revalidation-2026-10-14](../multigpu-layer-revalidation-2026-10-14/README.md) | podział warstw, bieżące źródła: zestawienie L1–L3 i [erratum `load_s`](../multigpu-layer-revalidation-2026-10-14/LOAD_S.md) |
| [multigpu-layer-l1-revalidation-2026-10-13](../multigpu-layer-l1-revalidation-2026-10-13/README.md) | L1: zgodność, odmowy, HTTP, pojemność; [poprawka poleceń](../multigpu-layer-l1-revalidation-2026-10-13/COMMANDS.md) |
| [multigpu-layer-l2-revalidation-2026-10-14](../multigpu-layer-l2-revalidation-2026-10-14/README.md) | L2: `none`; [decyzja użytkownika](../multigpu-layer-l2-revalidation-2026-10-14/DECISION.md) |
| [multigpu-layer-l3-revalidation-2026-10-14](../multigpu-layer-l3-revalidation-2026-10-14/README.md) | L3: `q8_0` i `q4_k`; [rozstrzygnięcie koordynatora](../multigpu-layer-l3-revalidation-2026-10-14/DECISION.md) |
| [multigpu-layer-2026-10-13](../multigpu-layer-2026-10-13/README.md) | podział warstw, poprzedni kandydat (historia); [poprawka zakresu](../multigpu-layer-2026-10-13/SCOPE.md); sesje [L1](../multigpu-layer-l1-2026-10-13/README.md), [L2](../multigpu-layer-l2-2026-10-13/README.md), [L3](../multigpu-layer-l3-2026-10-13/README.md) |
| [multigpu-link-corrections-2026-10-14](../multigpu-link-corrections-2026-10-14/README.md) | publiczne odpowiedniki odsyłaczy do lokalnego `.itsol/` |

Raporty źródłowe zachowują stan z chwili pomiaru. Akapity „Stan”, „decyzja oczekująca” albo zapowiedzi review i
PR w tych raportach są migawkami; bieżący stan opisują ten dokument, pliki `DECISION.md` i erratum `load_s`.
