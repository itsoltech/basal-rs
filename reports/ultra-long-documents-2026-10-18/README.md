# Ultra: duże dokumenty w ruchu mieszanym C32

Pomiar 2026-10-18 UTC, H100 PCIe. Wybrany profil istniejącego silnika skrócił
medianę obserwowanego p95 dużych dokumentów z **32,17 s**
(FIFO) do **17,51 s**;
upstream BF16 eager osiągnął **27,17 s**.
To redukcja o **45,6%** względem FIFO
i **35,6%** względem upstream.
Przepustowość całej fazy: **1,389** wobec
**1,395** i **1,089 żądania/s**.

**Kompromis:** mediana krótkich żądań w ruchu mieszanym jest o 11,5% wyższa niż FIFO; ich p95 jest o 24,2% niższe.

**Status końcowego porównania: AUDITED**, zero problemów audytu.
To **strojenie ustawień**, nie przyspieszenie nowych kerneli: oba warianty
natywne używają dokładnie tej samej zachowanej binarki. Eksperymenty zmian kodu
R1 i R2 wycofano; ich dane i nieukończona seria pozostają w archiwum.
Nie zmieniono ustawień domyślnych serwera.

## Profil i kompromis

Profil: HRRN, partia 16384 tokeny, tor długi powyżej 4096 spakowanych tokenów,
przydział co najmniej 5000 ms przed oddaniem GPU krótkiej partii.
Zwiększenie przydziału ogranicza częstotliwość przerw w dużym forwardzie.
Trzeba oceniać razem duże dokumenty i krótkie zadania: większy przydział
przesuwa czas oczekiwania między klasami. Poniżej zachowano również pogorszenia.

Przykładowa konfiguracja: [serve.ultra-h100.example.yml](../../serve.ultra-h100.example.yml).
Uruchom `basal serve --config serve.ultra-h100.example.yml` na odpowiedniej
karcie z binarką z opisanych źródeł. Plik przypina rewizję modelu i opisuje tekstowy profil pomiarowy;
limity przyjęcia z benchmarku nie są deklaracją produkcyjnej pojemności.
Media w tej kampanii są wyłączone. Równoważne ustawienia CLI:

```sh
basal serve --model Remek/basal-omni-1.0-ultra \
  --dtype auto --inference-mode auto --gemm-table auto \
  --schedule hrrn --max-batch-tokens 16384 \
  --long-tokens 4096 --long-slice-ms 5000 \
  --orders 1 --state-cache-mb 0 --no-prefix-cache
```

Powyższe skrócone polecenie nie przypina rewizji; do odtworzenia użyj modelu
z dokładnej rewizji poniżej albo dostarczonego YAML. Zachowanie przy różnych
liczbach klientów, mediach i innych modelach wymaga osobnych pomiarów.

## Wersje i sprzęt

Binarka obu profili: SHA-256
`f779bc967d6408d7c5fda6483cdcc663b663d3c958fa45c4e29583636fb8b993`.
To zachowana binarka loadera R2 z poprzedniej kampanii.
Sprawdzenie 202 plików budowania potwierdziło zgodność źródeł końcowego silnika
z tą wersją oraz drzewem `e209248`; kod silnika nie zmienia się w tej kampanii.
Eksperymentalne binarki i ich pełne źródła są odrębnie oznaczone w archiwum.

Model `Remek/basal-omni-1.0-ultra`, rewizja
`b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4`.
Upstream v2.1.0, commit `62542d31e8eb69000839c01e8f9b45da1fb80e49`.
Python 3.12.12, PyTorch 2.11.0+cu128, Transformers 5.17.0;
pełny freeze i kontrola sum plików modelu są w danych.
Rust 1.95.0, CUDA 12.9/NVCC 12.9.86; natywne biblioteki CUDA 12.9.1,
tabela GEMM H100 PCIe F16 dla cuBLASLt 120901.

GPU: NVIDIA H100 PCIe, 580.126.20, 81559 MiB, 350.00 W, 9.0. CPU: AMD EPYC 9334 32-Core Processor; 24 logiczne CPU,
affinity 24; RAM 247403824 KiB; Ubuntu 24.04.
OMP/MKL/Rayon=8 jest ustawieniem wątków, nie pinningiem ani limitem cgroup.
Nie zmieniano mocy, zegarów ani chłodzenia. Cache systemowy, bibliotek i
alokatorów nie był czyszczony. Wagi były pobrane przed pomiarem.

## Metoda

Końcowa seria `confirm-profile-r3`: **ABCCBAABC**, A=FIFO, B=tor długi,
C=upstream. Pierwszy start każdego silnika zachowano diagnostycznie;
pozostały **dwa starty na silnik**. Po wyłączeniu pierwszych startów porównania
każdej pary mają układ ABBA, także z odwróceniem ról. Kolejność i mniejszą
liczbę startów ustalono przed obserwacjami końcowej serii, po odrzuceniu
prototypów i z uwzględnieniem końca skończonej sesji. Nie odrzucano wolnych
próbek ani nie zastępowano nieudanych przebiegów skutecznymi.

Każdy osobny serwer: 5 s bezczynności z modelem, pełne 34 wejścia C1 jako
rozgrzewka/kontrola, krótkie zadania C32, mieszany ruch C32, 5 s po opróżnieniu.
32 różne krótkie zadania 2.1 przeplatane dwoma dokumentami o pełnym wejściu modelu po 8192 tokeny
(jeden długi na 16 krótkich). To **32 różne krótkie wejścia i 2 różne duże
wejścia**, nie setki unikalnych dokumentów. Te same ciała HTTP dla obu silników;
kontynuacje `extract` zależą od wyborów modelu, więc nie zakładamy tej samej
liczby gałęzi obliczeń upstream i native.

Minimum fazy krótkiej: 2 pełne przejścia korpusu i 30 s; rzeczywista liczba
może przekroczyć 64 przed opróżnieniem 32 klientów. Faza mieszana: 11 przejść,
374 odpowiedzi, w tym **22 duże dokumenty na start**, **44 w zachowanych
startach każdego silnika**. C oznacza klientów closed loop, odczytujących całą
odpowiedź przed kolejnym żądaniem. Zachowano rzeczywisty szczyt równoczesności,
pełne odpowiedzi, czasy start/koniec i opróżnienie wszystkich żądań.

Pomiar POST `/v1/basal` obejmuje serializację JSON, nowe TCP i odczyt całego HTTP body przez
lokalny HTTP bez TLS/proxy; parsowanie odpowiedzi jest po zatrzymaniu licznika.
P95 to element o indeksie `min(n-1, floor(0.95*n))` po sortowaniu. Dla 22 dużych
próbek jest drugim największym czasem. Tabele podają medianę oraz min–max z
dwóch startów, nie przedział ufności ani p95 po połączeniu wszystkich startów. Procenty dotyczą
ilorazów median przebiegów.
To nadal opis skończonego korpusu, nie gwarancja ogona dowolnego ruchu.

Natywnie: `auto` F16 z readout FP32, GEMM auto, 16384 tokeny, orders=1,
bez cache stanu i prefiksu, media off, max_inflight/max_large_inflight=1024.
Baseline FIFO z wyłączonym torem długim; profil HRRN/4096/5000.
Upstream BF16 eager, 4096 tokeny ustawione w pamięci procesu, orders=1,
bez cache stanu, context-overflow=refuse. Checkout upstream pozostał czysty.
Błędów i 529 nie zaliczamy do poprawnych odpowiedzi/s.

## Przepustowość

Mediany i zakres min–max dwóch zachowanych startów.

| Scenariusz | Silnik | Żądania/s mediana | Min–max | Liczba odpowiedzi w agregacji |
|---|---|---:|---:|---:|
| Krótkie C32 | basal-rs FIFO | 2,146 | 2,142–2,149 | 192 |
| Krótkie C32 | basal-rs tor długi | 2,183 | 2,183–2,184 | 192 |
| Krótkie C32 | upstream BF16 eager | 1,887 | 1,886–1,889 | 192 |
| Mieszany C32 | basal-rs FIFO | 1,395 | 1,395–1,395 | 748 |
| Mieszany C32 | basal-rs tor długi | 1,389 | 1,389–1,389 | 748 |
| Mieszany C32 | upstream BF16 eager | 1,089 | 1,088–1,089 | 748 |

## Opóźnienia obu klas

Mediany z dwóch startów; sekundy, cały czas request→response.

| Scenariusz | Silnik | Klasa | Mediana [s] | P95 [s] | P95 min–max [s] | Odpowiedzi |
|---|---|---|---:|---:|---:|---:|
| Krótkie C32 | basal-rs FIFO | krótkie | 14,79 | 14,95 | 14,94–14,95 | 192 |
| Krótkie C32 | basal-rs tor długi | krótkie | 14,30 | 15,00 | 14,63–15,37 | 192 |
| Krótkie C32 | upstream BF16 eager | krótkie | 16,78 | 18,24 | 18,16–18,32 | 192 |
| Mieszany C32 | basal-rs FIFO | krótkie | 21,40 | 32,29 | 32,24–32,34 | 704 |
| Mieszany C32 | basal-rs FIFO | duże 8192 | 25,58 | 32,17 | 32,17–32,17 | 44 |
| Mieszany C32 | basal-rs tor długi | krótkie | 23,85 | 24,48 | 24,47–24,49 | 704 |
| Mieszany C32 | basal-rs tor długi | duże 8192 | 15,88 | 17,51 | 17,50–17,52 | 44 |
| Mieszany C32 | upstream BF16 eager | krótkie | 31,42 | 38,38 | 38,37–38,38 | 704 |
| Mieszany C32 | upstream BF16 eager | duże 8192 | 24,91 | 27,17 | 27,11–27,22 | 44 |

Porównanie profilu do FIFO oraz upstream dla ruchu mieszanego:

- Względem basal-rs FIFO: przepustowość -0,4%; redukcja mediany/p95 krótkich -11,5% / 24,2%; redukcja mediany/p95 dużych 37,9% / 45,6%. Wartość ujemna oznacza pogorszenie.
- Względem upstream BF16 eager: przepustowość 27,6%; redukcja mediany/p95 krótkich 24,1% / 36,2%; redukcja mediany/p95 dużych 36,3% / 35,6%. Wartość ujemna oznacza pogorszenie.

Wybór profilu oceniamy razem z tym kompromisem; nie deklarujemy dominacji
każdej mediany i każdego percentyla nad wcześniejszym basal-rs. Nagłówki
queue/compute dotyczą całej partii, obejmują także oczekiwanie na gate GPU
i przerwy toru długiego. Nie są czasem samych kerneli jednego żądania.

## CPU, RAM i VRAM

Próbki co sekundę: serwer i potomkowie, bez generatora, samplera i guardu.
Kopiowanie zakończonych przebiegów z kontrolą SHA odbywało się w tle, poza
licznikami serwera. PSS/VRAM to mediany średnich ważonych czasem z dwóch startów; ostatnia
kolumna jest medianą maksimów próbkowanych.
Historia: rozgrzewka C1 → krótkie C32 → mieszany C32. Nie czyszczono alokatorów.

| Scenariusz | Silnik | CPU ms/żądanie | CPU rdzenie średnio | PSS [GiB] | VRAM procesu [GiB] | Szczyt VRAM próbkowany [GiB] |
|---|---|---:|---:|---:|---:|---:|
| Krótkie | basal-rs FIFO | 466,5 | 1,000 | 0,996 | 64,711 | 64,711 |
| Krótkie | basal-rs tor długi | 458,4 | 1,000 | 1,236 | 62,663 | 62,695 |
| Krótkie | upstream BF16 eager | 533,1 | 1,005 | 1,756 | 75,105 | 75,105 |
| Mieszany | basal-rs FIFO | 717,5 | 1,001 | 1,002 | 64,711 | 64,711 |
| Mieszany | basal-rs tor długi | 721,4 | 1,002 | 1,242 | 62,597 | 63,883 |
| Mieszany | upstream BF16 eager | 923,1 | 1,005 | 1,759 | 75,105 | 75,105 |

CPU oznacza czas pracy rdzeni, nie liczbę wymaganych wątków. PSS dzieli wspólne
strony proporcjonalnie; RSS/PSS mogą obejmować rezydentne strony mapowanych wag.
VRAM obejmuje model, runtime i workspace. Maksima są próbkowane. Nie wyznaczono
minimalnej pojemności RAM/VRAM ani czystego kosztu warstwy HTTP.

## Zgodność odpowiedzi

Zachowano **4536 poprawnych odpowiedzi HTTP** w końcowej serii,
łącznie z rozgrzewkami i diagnostycznymi startami. Pierwsze 34 odpowiedzi starej
binarki C1 są niezależną referencją; **2990** dalszych
odpowiedzi natywnych ma identyczne bity payloadu `answers`.
Różnic natywnych: **0**. Metryki czasu w `usage` mają inne
wartości i nie należą do porównania bitowego wyników modelu.

Upstream zmienił payload względem własnej pierwszej odpowiedzi
**544** razy; dane zachowują te
różnice. Nie wymagamy bitowej równości BF16 z natywnym F16 ani nie utożsamiamy
zmiany prawdopodobieństwa ze zmianą decyzji.
W tej kampanii nie tworzono nowej referencji FP32. Zgodność wyników profilu ze
starą wersją zachowuje jej zmierzone odpowiedzi; wcześniejsza ocena odległości
od FP32 jest w [raporcie attention](../ultra-attention-2026-10-17/README.md)
i [raporcie zasobów](../ultra-resources-2026-10-17/README.md).
Nie jest to nowy benchmark semantycznej accuracy.

## Próby odrzucone i nieukończone

| Próba | Stan danych | Decyzja |
|---|---|---|
| R1: FIFO z podobnymi długościami | AUDITED, 510 odpowiedzi, 476 porównań native, zero różnic | Mały screen nie potwierdził użytecznej przewagi dużych dokumentów; kod usunięty |
| Stary silnik: HRRN/4096/5000 | AUDITED, 340 odpowiedzi, 306 porównań, zero różnic | Obiecujący screen parametrów, sam w sobie SIGNAL |
| R2: pierwsze wejście długiego toru przez FIFO resume | AUDITED screen, 340 odpowiedzi, 306 porównań, zero różnic | Wycofany po materialnym przeglądzie priorytetów |
| Początkowa pełna seria R2 | INTERRUPTED_BY_REVIEW; replay FAIL/GAP | Przerwana po znalezieniu ryzyka; nie wchodzi do końcowej agregacji |
| Końcowy profil na starej binarce | AUDITED, 9 startów, po 2 zachowane na silnik | Wyniki i ograniczenia tego raportu |

R2 mógł utrzymywać kolejkę rezerwacji nowych długich torów niepustą przy wielu
modelach i przez to odkładać pilne zadania. To potencjalna regresja polityki,
nie dowód błędu odpowiedzi w wykonanej próbie. Sprawdzenie źródeł wykazało
problem przed wdrożeniem/commitem. Zachowano pełne źródła, kontrole, ekran i
przerwaną serię; kod wrócił do wcześniejszej wersji. Nie przypisujemy wyników
końcowego profilu tej wycofanej zmianie. Różnicy między małymi screenami starego
i R2 nie traktujemy jako osobnego przyspieszenia od kodu gate.

## Weryfikacja i ograniczenia

Dla obu prototypów przeszły fmt, Clippy i Rustdoc CPU oraz CUDA/all-features.
Przeszły cargo deny, kontrola paste i przegląd zależności; nie dodano zależności,
unsafe ani frameworków testowych. Przegląd Rust i samokontrola odrzuciły R2.
Źródła końcowego silnika są zgodne z wcześniej zweryfikowaną binarką;
końcowe zmiany obejmują profil konfiguracji, narzędzia i dokumentację.
Python sprawdzono kompilatorem składni, rzeczywistym pomiarem i niezależnym replay.

To jedna karta/model i tekstowe C32. Nie powtarzano C8/C128, obrazu/dźwięku,
zimnego dysku ani osobnego benchmarku C1. Nie poprawiamy tym wcześniejszych
różnic prawdopodobieństw C128 ani nie twierdzimy, że każda konfiguracja jest
bitowo niezmienna. Tor długi nadal ma dotychczasowe priorytety startu; profil
nie gwarantuje stałej granicy czasu przy dowolnym napływie lub wielu modelach.
Przed szerszym zastosowaniem trzeba ocenić własny rozkład długości, limity
przyjęcia i kompromis obu klas. Nie zmieniono globalnych ustawień domyślnych.

## Artefakty

Raport Markdown zostaje w Gicie. Dane i pełne źródła są pod
`reports/ultra-long-documents-2026-10-18/` w archiwum.
`REPRODUCE.md` podaje replay bez GPU, odbudowę, przypięte wejścia i wymagania
nowego pomiaru. Archiwum obejmuje odrzucone źródła i nieukończone wyniki;
prywatne binarki i dane dostawcy pozostają lokalnie.
Maszynę usunięto po zachowaniu wszystkich plików i sprawdzeniu ich SHA-256.
Nota potwierdzonej publikacji jest dopisywana do końcowej wersji raportu w Gicie.

Dostarczony YAML sprawdzono rzeczywistym uruchomieniem na GPU: 34/34 odpowiedzi
C1 zgodne bitowo z referencją końcowej serii, poprawne zakończenie procesu.
Do lokalnej próby zmieniono wyłącznie port i źródło modelu na pobrany wcześniej
katalog tej samej rewizji; parametry serwowania pozostały identyczne.

## Potwierdzona publikacja

Archiwum publiczne: `hf://buckets/itsoltech/basal-rs/measurements/ultra-long-documents-2026-10-18/c256b4f32d863f1ed54130662d6d3c72a9bf3457f276baaa1d1b4151a81fcdbc/measurement-data.tar.gz`.
SHA-256: `c256b4f32d863f1ed54130662d6d3c72a9bf3457f276baaa1d1b4151a81fcdbc`; 6 482 237 B gzip,
780 plików danych i manifest `SHA256SUMS`.
2026-10-18: świeże pobranie z autoryzacją i anonimowo potwierdziło SHA archiwum
oraz **780/780 sum plików**. Publiczny replay odtworzył audyty wszystkich trzech
screenów, oczekiwany FAIL przerwanej R2 i końcową agregację bez różnic względem
prywatnych oryginałów. Kontrola projekcji potwierdziła zachowanie bitów 6768
odpowiedzi i liczników 5545 próbek zasobów, także odrzuconych prób.

W archiwum, pod `reports/ultra-long-documents-2026-10-18/`:

- `confirm-profile-r3/`, `confirmation-audit.json`, `comparisons.json`: końcowe
  odpowiedzi, czasy, nagłówki, zasoby, metoda i niezależny audyt;
- `screen-length-r1/`, `screen-lane-r1/`, `screen-lane-r2/`, `confirm-lane-r2/`
  i `rejected-r2/`: ekrany oraz wycofany kod, ryzyko priorytetów i nieukończony FAIL;
- `profile-config-check/`, `verification/profile-config-audit.json` i
  `serve.ultra-h100.example.yml`: rzeczywista próba dostarczonej konfiguracji;
- `sources/`, `inputs/`, `metadata/`, `verification/`, `REPRODUCE.md`: dokładne
  źródła, przypięte wejścia, pochodzenie i instrukcje odbudowy/replay.

Z głównego katalogu repozytorium, do nowych katalogów:

```sh
BASAL_LONG_DL=.cache/measurement-download/ultra-long-documents-2026-10-18
BASAL_LONG_ROOT=.cache/measurement-replay/ultra-long-documents-2026-10-18
mkdir -p .cache/measurement-download .cache/measurement-replay &&
  mkdir "$BASAL_LONG_DL" &&
  hf buckets cp "hf://buckets/itsoltech/basal-rs/measurements/ultra-long-documents-2026-10-18/c256b4f32d863f1ed54130662d6d3c72a9bf3457f276baaa1d1b4151a81fcdbc/measurement-data.tar.gz" "$BASAL_LONG_DL/measurement-data.tar.gz" &&
  echo "c256b4f32d863f1ed54130662d6d3c72a9bf3457f276baaa1d1b4151a81fcdbc  $BASAL_LONG_DL/measurement-data.tar.gz" | sha256sum -c - &&
  mkdir "$BASAL_LONG_ROOT" &&
  tar -xzf "$BASAL_LONG_DL/measurement-data.tar.gz" -C "$BASAL_LONG_ROOT" &&
  (cd "$BASAL_LONG_ROOT" && sha256sum -c SHA256SUMS)
```

Na macOS użyj `shasum -a 256 -c`. Z rozpakowanego katalogu
`reports/ultra-long-documents-2026-10-18/` uruchom:

```sh
python -B sources/replay/summarize-omni-long-documents.py \
  --measurement-artifacts confirm-profile-r3 --inputs inputs \
  --out confirmation-replayed.json
python -B sources/replay/aggregate-confirmation.py \
  --measurement-artifacts confirmation-replayed.json --out comparisons-replayed.json
```

Pierwszy JSON musi być identyczny z `confirmation-audit.json`, drugi z
`comparisons.json`. Wyjścia muszą być nowe. Archiwalny `README.md` zachowuje wersję
sprzed tej noty; bieżąca nota publikacji pozostaje w Gicie. Prywatna kopia
archiwum jest zachowana lokalnie; identyfikatory procesów i rzeczywiste ścieżki
zastąpiono etykietami, bez zmiany odpowiedzi, czasów i liczników zasobów.
