# Ładowanie wag Gemma: ponowne użycie bufora

Pomiar 2026-10-18 UTC. Nowy loader Ultra osiągnął medianę **11,51 s do gotowości**
wobec **75,56 s R15** i **19,52 s upstream BF16** na tej samej H100 PCIe.
To **6.56× krótszy start niż R15**
i **41.0% krótszy niż upstream**.
Czas CPU startu spadł o 84.7%,
a szczyt RSS procesu o 47.9% względem R15.

**Status:** starty `AUDITED` (12 przebiegów, zero problemów audytu), końcowa
kontrola strukturalna eksportów `PASS` (561 par rekordów).
Zachowano wcześniejszy **FAIL narzędzia eksportu**: błędny argument ścieżki GEMM,
poprawiony przed końcową kontrolą. Nie dotyczył kodu silnika ani 12 pomiarów startu.

## Zmiana

`GemmaBackend::load_on` czyta kolejne wagi do jednego lokalnego bufora bajtów.
Przy BF16 oraz wcześniejszej konwersji BF16→F16 na GPU pomija dodatkowy własny
tensor CPU i przesyła dane bezpośrednio z bufora. Synchronizuje stream przed
ponownym nadpisaniem danych i zwalnia bufor przed gotowością backendu.
Callback pozostaje synchronicznym `Fn`; lokalne `RefCell::try_borrow_mut`
propaguje błąd ewentualnego nakładania wywołań zamiast panikować.

Próg zapasu VRAM dla konwersji na GPU oraz konwersja CPU przy małym zapasie
pozostają takie same. Obliczenia F32 nadal korzystają z rezydentnych wag BF16,
bezstratnie rozszerzanych przed operacjami. Nie zmieniono attention, RMSNorm,
RoPE, algorytmów inferencji ani schedulera. Bez nowego unsafe, publicznego API
i zależności. Ten bufor nie należy do stanu obsługi żądań.

## Wersje

Bazowa binarka R15: SHA-256
`f23f060cf61ebf5d9d39c4f38835a6152b3823179e102152db2dfe17bc5af8b9`.
Kandydat R2: SHA-256
`f779bc967d6408d7c5fda6483cdcc663b663d3c958fa45c4e29583636fb8b993`.
Sprawdzenie 200 plików potwierdziło pełną zgodność źródeł przed zmianą z R15.
Kandydat zmienia tylko `crates/basal-gpu/src/gemma.rs` w drzewie budowania.

Model startu: `Remek/basal-omni-1.0-ultra`, rewizja
`b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4`. Upstream v2.1.0, commit
`62542d31e8eb69000839c01e8f9b45da1fb80e49`. Dodatkowe kontrole:
`basal-omni-1.1-max` BF16, rewizja `c8ab50cd973c116b3ca6bd9e4d574f713bab1f68`,
oraz `basal-omni-1.0-mini` F32, rewizja `73c993dc3d9ab3cb8887ac0b1ee1cf1c92783240`.
Wagi i konfiguracje sprawdzane po pobraniu według przypiętych sum HF.

## Sprzęt i środowisko

NVIDIA H100 PCIe, 580.126.20, 81559 MiB, 350.00 W, 9.0. CPU AMD EPYC 9334 32-Core Processor, 24 logiczne CPU,
affinity 24; RAM 247403820 KiB.
Ubuntu 24.04, Rust 1.95.0, CUDA 12.9.1/NVCC 12.9.86;
upstream Python 3.12.12, PyTorch 2.11.0+cu128 i Transformers 5.17.0.
Nie zmieniano ustawień mocy, zegarów ani chłodzenia GPU.
CPU w środowisku OMP/MKL/Rayon=8; to nie jest pinning ani limit cgroup.
Pakiety upstream i dokładne źródła są w archiwum.

## Metoda i zakres czasu

Osobne procesy: stary, nowy, nowy, stary, stary, nowy, nowy, stary.
Pierwszy start każdego wariantu zachowany diagnostycznie; agregacja to trzy
pozostałe starty. Potem cztery starty upstream BF16 eager, z pierwszym
odrzuconym z agregacji. Upstream nie jest losowo przeplatany z natywnym ABBA;
porównanie obu rzeczywistych API jest opisowe.

Czas od utworzenia procesu do poprawnej odpowiedzi HTTP health; sprawdzanie
co pół sekundy daje taką rozdzielczość wykrycia gotowości. To czas startu serwera
z modelem, nie czas pojedynczego pytania. Wagi były już pobrane i sprawdzone.
Cache systemowy, bibliotek i alokatorów nie był czyszczony. To nie jest
kontrolowany zimny dysk. Dane I/O i faultów są zachowane dla każdego startu.

Sampler co sekundę obejmuje serwer i potomków, wyklucza generator, guard i sampler.
Osobne liczniki user/system/fault/I/O obejmują proces główny od jego utworzenia.
CPU jest czasem pracy rdzeni, nie liczbą wymaganych rdzeni. RSS może zawierać
wspólne strony; PSS dzieli je proporcjonalnie, USS obejmuje strony prywatne.
Maksima PSS/USS/VRAM są próbkowane. `VmHWM` daje osobno szczyt RSS procesu
od startu do gotowości. Żadne z nich nie określa minimalnej pojemności RAM.
Wszystkie 12 startupów ma ciągłe próbkowanie, największy odstęp około 1,001 s.

Po każdym starcie: 15 s bezczynności, 32 różne krótkie żądania z zadaniami 2.1
oraz dwa dokumenty 8192-tokenowe przy C1, potem 15 s bezczynności.
Local HTTP, bez TLS/proxy, nowe TCP na żądanie. Te żądania kontrolują wyniki;
nie są nowym pomiarem przepustowości. Media są wyłączone w serii startów.
Natywnie: auto F16/FP32 readout, FIFO, batch16384, orders1, state/prefix cache off.
Upstream: BF16 eager, batch4096 ustawiony w pamięci, orders1, state cache off.
Checkout upstream pozostał niezmieniony.

## Starty Ultra

Medianę i zakres min–max policzono z trzech startów po diagnostycznym.
GiB oznacza 2^30 bajtów. Ilorazy dotyczą median, nie przedziałów ufności.

| Silnik | Gotowość mediana [s] | Min–max [s] | CPU do gotowości [s pracy rdzeni] | Szczyt RSS do gotowości [GiB] | Szczyt PSS próbkowany [GiB] |
|---|---:|---:|---:|---:|---:|
| basal-rs R15 | 75.56 | 75.06–76.56 | 75.11 | 5.604 | 4.704 |
| basal-rs R2 | 11.51 | 11.51–12.01 | 11.49 | 2.921 | 2.917 |
| upstream BF16 | 19.52 | 19.52–19.52 | 18.52 | 59.335 | 57.569 |

Szczyt RSS i PSS nie są tą samą metryką. Krótkie skoki między próbkami mogą
wyjaśniać część różnicy; RSS/PSS obejmują także mapowane rezydentne strony plików.
Wysoki RSS upstream nie jest dowodem wymaganego minimum RAM.

| Silnik | CPU użytkownika [s] | CPU jądra [s] | Minor faults [mln] | Major faults | Odczyt `read_bytes` [B] |
|---|---:|---:|---:|---:|---:|
| basal-rs R15 | 11.90 | 63.12 | 30.169 | 0 | 0 |
| basal-rs R2 | 4.09 | 7.40 | 0.957 | 0 | 0 |
| upstream BF16 | 12.52 | 6.00 | 1.387 | 0 | 0 |

Zmiana łączy ponowne użycie bufora z pominięciem kopii CPU. Spadek faultów i CPU
jądra jest zgodny z usunięciem kosztownych powtarzanych alokacji/kopii.
Nie rozdzielono eksperymentalnie kosztu każdego z tych dwóch usprawnień;
nie zebrano profilu stosów pozwalającego przypisać każdą sekundę konkretnej funkcji.

Wszystkie obserwacje pozostają w danych, także pierwsze diagnostyczne:

| Start | W agregacji | Czas [s] | CPU [s pracy rdzeni] | Szczyt RSS [GiB] |
|---|---|---:|---:|---:|
| baseline-r1 | nie, diagnostyczny | 75.571 | 75.46 | 5.691 |
| candidate-r1 | nie, diagnostyczny | 12.512 | 12.48 | 2.957 |
| candidate-r2 | tak | 11.511 | 11.44 | 2.921 |
| baseline-r2 | tak | 75.057 | 74.89 | 5.604 |
| baseline-r3 | tak | 75.558 | 75.11 | 5.604 |
| candidate-r3 | tak | 11.510 | 11.49 | 2.921 |
| candidate-r4 | tak | 12.012 | 11.51 | 2.921 |
| baseline-r4 | tak | 76.560 | 76.48 | 5.604 |
| upstream-eager-r1 | nie, diagnostyczny | 19.518 | 18.58 | 59.335 |
| upstream-eager-r2 | tak | 19.518 | 18.54 | 59.334 |
| upstream-eager-r3 | tak | 19.517 | 18.52 | 59.335 |
| upstream-eager-r4 | tak | 19.516 | 18.50 | 59.335 |

## Bezczynność z modelem i po sprawdzeniu żądań

Mediany średnich ważonych czasem, po trzy zachowane starty. To model i runtime,
nie wyodrębniony koszt HTTP. Alokatory mogą zachować pamięć po długich wejściach.

| Silnik | Etap | CPU [rdzenie] | PSS [GiB] | USS [GiB] | VRAM procesu [GiB] | VRAM całej karty [GiB] |
|---|---|---:|---:|---:|---:|---:|
| basal-rs R15 | po załadowaniu | 0.0007 | 0.902 | 0.900 | 60.287 | 60.296 |
| basal-rs R15 | po żądaniach | 0.0007 | 1.052 | 1.050 | 64.711 | 64.720 |
| basal-rs R2 | po załadowaniu | 0.0000 | 0.845 | 0.843 | 60.287 | 60.296 |
| basal-rs R2 | po żądaniach | 0.0000 | 0.996 | 0.994 | 64.711 | 64.720 |
| upstream BF16 | po załadowaniu | 0.0013 | 1.095 | 1.093 | 59.023 | 59.032 |
| upstream BF16 | po żądaniach | 0.0013 | 1.753 | 1.751 | 75.105 | 75.114 |

## Kontrola bitowa i błędy narzędzia

W serii startów wykonano 408 poprawnych odpowiedzi HTTP: 272 natywne i 136
upstream. Pierwsze 34 natywne odpowiedzi stanowią referencję. Pozostałe 238
porównań, w tym wszystkie 136 odpowiedzi nowego loadera, są bitowo zgodne.
Nie wymagamy bitowej zgodności odpowiedzi upstream BF16 z natywnym F16.

Pełne eksporty stare/nowe objęły 561 par rekordów JSONL:
Ultra F16, max1.1 BF16 i mini F32, każdy z korpusem prymitywów i zadań.
To 12 par plików i 6 konfiguracji model/precyzja/korpus. Rekordy współdzielą
część wejść między korpusami; liczba par nie oznacza tylu unikalnych promptów.
Porównanie zachowuje bity logitów, rozkładów, odpowiedzi, danych wejściowych
oraz niezmienionych odmów. Manifesty wykonania mają różne czasy/ścieżki;
sprawdzono osobno ich zgodną konfigurację. Nie stanowi to niezależnego accuracy
benchmarku ani nowego pomiaru odległości od FP32 upstream.

Media są włączone w eksportach, z czterema zdjęciami COCO. Ultra F16 używa
jawnej zamrożonej tabeli GEMM odpowiadającej mierzonej konfiguracji; BF16 max
oraz F32 mini używają standardowych GEMM Candle bez tabeli. F32 dotyczy obliczeń,
a wagi pozostają BF16 przed rozszerzeniem. Kontrola obejmuje obrazy; nie wykonano
w tej sesji żądań dźwiękowych ani próby z wymuszonym niedoborem VRAM.

| Konfiguracja | Pary rekordów | Różnice bitowe | Żądania z mediami / z odpowiedziami |
|---|---:|---:|---:|
| basal-omni-1.0-ultra-f16 | 187 | 0 | 18 / 18 |
| basal-omni-1.1-max-bf16 | 187 | 0 | 18 / 18 |
| basal-omni-1.0-mini-f32 | 187 | 0 | 18 / 18 |

Początkowa próba `failed-quality-r2/` zakończyła się po załadowaniu starego modelu,
przed eksportem, ponieważ runner przekazał literalne `--gemm-table auto` do CLI
`export`, które oczekuje ścieżki. Zachowano log, wejścia, metadane i źródło
z błędem. Poprawiony `check-quality-r3.py` podał jawną tabelę Ultra, pozostawiając
pozostałe profile bez tabeli. Nie zmieniono binarek ani serii 12 startów.
Nieudana próba nie wchodzi do agregacji i nie została zastąpiona ani skasowana.

## Decyzja i ograniczenia

Przyjmujemy zmianę loadera: dla Ultra na tej H100 skrócenie startu jest potwierdzone,
a komplet badanych eksportów i odpowiedzi C1 zachowuje poprzednie bity.
Pomiar czasu dotyczy jednego modelu na jednej maszynie, z pobranymi wagami.
Pozostałe modele służą kontroli poprawności, bez osobnej serii potwierdzającej
ich czasy ładowania. Nie wyznaczono minimalnego RAM/VRAM ani czasu zimnego dysku.
Bufor zachowuje pojemność największej dotąd wagi do końca ładowania.
Przy presji VRAM pozostaje wcześniejsza konwersja CPU; jej pamięci/czasu nie
zmierzono osobno. Nie przypisujemy tej zmianie nowej przewagi QPS lub p95.

Nie sprawdzano ponownie C128. Wcześniejsze różnice prawdopodobieństw C128
oraz regresja p95 długich żądań pozostają w
[raporcie zasobów](../ultra-resources-2026-10-17/README.md) i
[raporcie R15](../ultra-attention-2026-10-17/README.md); ta zmiana ich nie naprawia.

## Weryfikacja i artefakty

Przeszły formatowanie, workspace Clippy CPU oraz CUDA/all-features, Rustdoc
CPU oraz CUDA/all-features, `cargo deny`, kontrola wyjątku `paste` i kontrola
zduplikowanych zależności. Brak nowych zależności, unsafe i frameworków testowych.
Nie uruchamiano jednostkowych/integracyjnych testów zabronionych przez zasady repo.
Samokontrola objęła lifetime host/GPU, synchronizację przed nadpisaniem bufora,
błędy I/O i borrow, niezmienione progi VRAM, lifecycle procesów oraz źródła/sumy.
Źródła Python sprawdzono kompilatorem składni, rzeczywistym wykonaniem i analizą danych.

Archiwum zawiera pełne źródła R2, patch względem R15, dane startów/eksportów,
nieudaną próbę narzędzia oraz dokładne runnery i audytory. Media pozostają zewnętrzne,
z przypiętymi źródłami i hashami oraz narzędziem przywracającym oryginalne bajty wejść.
Prywatne surowe dane, binarki i informacje maszyny pozostają lokalnie.
Maszynę usunięto po zachowaniu wszystkich wyników i sprawdzeniu ich SHA-256.

Dane w archiwum są pod `reports/ultra-loader-2026-10-18/`.
`REPRODUCE.md` opisuje ponowne przeliczenie danych, odbudowę źródeł oraz wymagania
nowej sesji GPU. Nota potwierdzonej publikacji jest w końcowej wersji raportu w Gicie.


## Potwierdzona publikacja

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/ultra-loader-2026-10-18/1a453f895e4cf547349e240fcd769115040fe793a78495d22c8d13d28df6791c/measurement-data.tar.gz`.
SHA-256: `1a453f895e4cf547349e240fcd769115040fe793a78495d22c8d13d28df6791c`. Rozmiar: 2610101 B.
2026-10-18 pobranie z autoryzacją oraz anonimowo potwierdziło SHA-256 archiwum
oraz **405/405 sum plików** w obu przypadkach. Lokalne archiwum zachowano.
Archiwalny README jest kopią sprzed dopisania tej noty.

Pobranie do nowego katalogu, weryfikacja i ponowny audyt, bez GPU:

```bash
BASAL_LOADER_URI=hf://buckets/itsoltech/basal-rs/measurements/ultra-loader-2026-10-18/1a453f895e4cf547349e240fcd769115040fe793a78495d22c8d13d28df6791c/measurement-data.tar.gz
BASAL_LOADER_DL=.cache/measurement-download/ultra-loader-2026-10-18
BASAL_LOADER_ROOT=.cache/measurement-replay/ultra-loader-2026-10-18
mkdir -p .cache/measurement-download .cache/measurement-replay &&
  mkdir "$BASAL_LOADER_DL" &&
  hf buckets cp "$BASAL_LOADER_URI" "$BASAL_LOADER_DL/measurement-data.tar.gz" &&
  echo "1a453f895e4cf547349e240fcd769115040fe793a78495d22c8d13d28df6791c  $BASAL_LOADER_DL/measurement-data.tar.gz" | sha256sum -c - &&
  mkdir "$BASAL_LOADER_ROOT" &&
  tar -xzf "$BASAL_LOADER_DL/measurement-data.tar.gz" -C "$BASAL_LOADER_ROOT" &&
  (cd "$BASAL_LOADER_ROOT" && sha256sum -c SHA256SUMS)

BASAL_LOADER_DATA="$BASAL_LOADER_ROOT/reports/ultra-loader-2026-10-18"
python -B "$BASAL_LOADER_DATA/sources/summarize-omni-loader.py" \
  --measurement-artifacts "$BASAL_LOADER_DATA/startup" \
  --inputs "$BASAL_LOADER_DATA/inputs" --out "$BASAL_LOADER_DL/startup-audit.json"
python -B "$BASAL_LOADER_DATA/sources/audit-quality.py" \
  --measurement-artifacts "$BASAL_LOADER_DATA/quality" \
  --source-map "$BASAL_LOADER_DATA/public-source-map.json" \
  --startup-method "$BASAL_LOADER_DATA/startup/method.json" \
  --gemm-table "$BASAL_LOADER_DATA/inputs/ultra-gemm-table.json" \
  --out "$BASAL_LOADER_DL/quality-audit.json"
```

Na macOS zamiast `sha256sum -c` użyj `shasum -a 256 -c`.
Jeśli istnieje `~/.local/bin/hf`, użyj tej binarki bezpośrednio.
`REPRODUCE.md` w archiwum opisuje przywrócenie mediów, odbudowę dokładnych źródeł,
powtórzenie pomiarów GPU i zachowaną nieudaną próbę. Publiczne i prywatne audyty
startów oraz jakości dały identyczne wyniki. Przywrócenie czterech zasobów mediów
odtworzyło dokładne sumy czterech źródłowych i 12 przygotowanych plików wejściowych.
