# Pico: wydajność CUDA na RTX 4060

Zoptymalizowany profil `basal-omni-pico.v2` wyprzedza domyślny upstream
`eager` na czterech zbadanych krótkich zadaniach. Nie wygrywa dla każdego
obciążenia: dokument 5080 tokenów pozostaje wolniejszy, a wymuszony upstream
`fast` wygrywa przy C1/C8, lecz przy C32 kończy część żądań błędem OOM.
Wyniki dotyczą powtarzanych wejść syntetycznych na jednej RTX 4060.

## Pomiar potwierdzający

Każda para silników ma osobną serię ABBA: A B B A A B B A, po cztery starty
na silnik. Pierwsze powtórzenie każdego silnika wykluczono; tabela pokazuje
medianę metryk z pozostałych trzech powtórzeń. C oznacza liczbę równoległych
klientów. Każda komórka krótkiego obciążenia obejmuje 192 żądania na silnik
(64 × 3), po 48 na każde z czterech wejść. Dwa punkty długości mają po
18 pomiarów na silnik (6 × 3). Audyt liczy metryki ponownie z odpowiedzi.

### Domyślny upstream `eager`

| Krótkie zadania | basal-rs p50 / p95 [ms] | upstream p50 / p95 [ms] | basal-rs [żądań/s] | upstream [żądań/s] | Iloraz przepustowości |
|---|---:|---:|---:|---:|---:|
| C1 | 36,65 / 57,33 | 46,71 / 89,20 | 25,49 | 18,35 | **1,39×** |
| C8 | 343,94 / 346,13 | 755,49 / 1054,75 | 23,21 | 10,78 | **2,15×** |
| C32 | 1307,23 / 1309,53 | 2436,16 / 2959,15 | 24,42 | 13,12 | **1,86×** |

P50 C1 spadło o 21,5% (przyspieszenie 1,27×). Przepustowość wzrosła o
38,9%, 115,3% i 86,1%. Native oraz eager mają zero niepoprawnych odpowiedzi.
Audyt `audit-confirm-eager.json`: **AUDITED, 0 problemów**. Dla eager C32
p50 poszczególnych powtórzeń wynosiło 1852,57–2484,10 ms; native
1306,86–1307,96 ms. Nie podajemy przedziałów ufności z trzech powtórzeń.

| Pojedynczy dokument / C1 | basal-rs p50 [ms] | eager p50 [ms] | Wynik |
|---|---:|---:|---|
| 255 tokenów, 27 opcji | 26,22 | 35,36 | basal-rs szybszy 1,35× |
| 5080 tokenów, jedno pytanie | 730,63 | 555,72 | **GAP: basal-rs wolniejszy o 31,5%** |

### Wymuszony upstream `fast`

Na Ada upstream `gdn_graph.default_fast()` wybiera `eager`: automatyczne
`fast` wymaga compute capability major 9 oraz FLA. Jawne `fast` działa na
RTX 4060 i zostało zmierzone osobno, z zainstalowanym FLA.

| Obciążenie | basal-rs p50 [ms] | fast p50 [ms] | basal-rs / fast [żądań/s] |
|---|---:|---:|---:|
| Krótkie C1 | 36,20 | **21,73** | 25,76 / **34,23** |
| Krótkie C8 | 342,55 | **271,18** | 23,31 / **29,25** |
| Krótkie C32 | 1307,54 | **FAIL: OOM / HTTP 503** | 24,42 / brak poprawnego pełnego przebiegu |
| 255 tokenów C1 | 25,90 | **13,42** | 38,42 / **73,89** |
| 5080 tokenów C1 | 730,41 | **268,02** | 1,37 / **3,73** |

Cała seria `fast` ma status **FAIL**, zachowany także w audycie
`audit-confirm-fast.json` (216 problemów: 208 niepoprawnych odpowiedzi,
4 niepoprawne fazy, 4 błędy przebiegów). C32 zwróciło odpowiednio
49, 50, 56 i 53 błędy z 64 żądań w czterech startach; w trzech zachowanych
powtórzeniach to 159/192. Komunikaty wskazują próbę alokacji 64 MiB przy
niemal wyczerpanych 7,70 GiB dostępnej pamięci. Nie liczymy przepustowości
z częściowo udanych faz. C1, C8 i oba punkty długości zakończyły się poprawnie
przed OOM i tylko te fazy są w tabeli. Native nie miał błędów ani różnic
bitowych odpowiedzi w żadnej z obu serii.

## Co zmieniono

Profil v2 ma wersjonowany dispatch `qwen35-dispatch.2` i kernel set
`basal-qwen35.2`. Łączy projekcje DeltaNet i gate/up MLP, używa własnych
kerneli CUDA dla norm, konwolucji, przygotowania i rekurencji DeltaNet,
gated RMS i SwiGLU. Przechowuje stan rekurencji FP32 w układzie
`[head,value,key]`; każde rozwidlenie zapisuje nowy stan. BF16/F16 CUDA
używa fragmentów do 512 tokenów. FP32 i Metal zachowują 64.

Attention z głową 256 i kontekstem do 2048 używa softmax online FP32 oraz
stałej kolejności kluczy. Dłuższe wejścia mają attention Candle FP32;
rekurencja nadal przechodzi sekwencyjnie po tokenach. W obrębie wywołania
model współdzieli niemutowalny prefiks i grupuje projekcje do ośmiu promptów.
Dowolny prefiks do 2048 wymaga kwalifikowanej tabeli GEMM; bez pokrycia
tabelą pozostają osobne projekcje i współdzielenie pełnych fragmentów.
Nie ma cache stanu ani prefiksów między żądaniami.

Dodana do Gita tabela BF16 RTX 4060 / cuBLASLt 120901 zawiera 48 wpisów
(6 kształtów, 8 klas M). Wygenerowano ją istniejącym `gemm-search --invariant`.
Znormalizowany identyfikator:
`72fb35fcb899c71c6ed7e52ddb64e20686848f2f7ed8304401147bdac1bdee2e`.
Serwer wybiera ją przez `auto`; inne GPU bez tabeli nie uruchamiają autotuningu.
Profil ogólny Qwen3.5 pozostaje referencyjnym v1. Nie zmieniono zależności Cargo.

## Zgodność i zmiana numeryki

Referencja to upstream FP32 z poprzedniej kampanii, z identycznymi tokenami
i tym samym checkpointem. `basal compare` potwierdził:

| Wariant | Decyzje bench / gałęzie API | Maks. błąd logitu bench | Maks. błąd skalibrowanego prawdopodobieństwa bench |
|---|---:|---:|---:|
| BF16 v2, orders 1 | 44/44 + 12/12 | 0,191184 | 0,052262 |
| BF16 v2, orders 2 | 44/44 + 12/12 | 0,207023 | 0,049012 |
| FP32, orders 2 | 44/44 + 12/12 | 0,000019514 | 0,000005051 |

Osobny zestaw granic zachował 2/2 decyzji (255 opcji i dokument 5080 tokenów)
oraz odmowę przy 256 opcjach. Nie było różnic tokenów. Audyty pól API
potwierdzają identyczne `type`, `choice`, `action`, `answer`, `tool`, `call`
i `selected`; wartości ciągłe, confidence i score mogą się różnić.

**SIGNAL numeryczny:** maksymalny błąd skalibrowanego prawdopodobieństwa BF16
w orders 1 wzrósł z około 0,01863408 w profilu referencyjnym do 0,05226158
(5,23 punktu procentowego) w v2. NLL po kalibracji na 44 pytaniach:
FP32 0,457348, BF16 v2 0,463588; Brier 0,248690 / 0,253247.
Orders 2: NLL 0,484387 / 0,491236; Brier 0,264239 / 0,268501.
Zgodne decyzje na tym niewielkim zestawie nie dowodzą niezmienionej jakości
ani kalibracji na innych danych. Zmiana obliczeń ma osobną wersję profilu.

`cmp` potwierdził bitową równość obu plików eksportu (`bench.jsonl`,
`systemone.jsonl`) single/tree dla BF16 v2 orders 1 i 2, tabeli jawnej
i wbudowanej `auto` dla orders 1 oraz końcowego FP32 orders 2 względem
poprzedniej implementacji. Fuzje referencyjnego fragmentu 64 były bitowo
równe staremu BF16. Szczegóły i pozostałe pary: `validation.json`.
Eksport `bundled-plain-bf16-o1` celowo zachowano: pominięcie
`--gemm-table auto` daje inną ścieżkę niż domyślny serwer.

## Warunki i ślady

GPU: NVIDIA GeForce RTX 4060, 8188 MiB, SM 8.9, sterownik 615.78.08,
limit mocy 115 W; bez zmiany ustawień karty. Telemetria co 200 ms jest w
`telemetry-confirm.csv`. Odczyt faktycznego poboru mocy był niedostępny,
więc nie wyciągamy wniosków o energii. Nie nakładano inferencji silników
ani kompilacji CPU podczas potwierdzających serii.

Model `Remek/basal-omni-pico`, rewizja
`a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`, SHA-256 wag
`ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9`.
Upstream [rkinas/basal](https://github.com/rkinas/basal), commit
`615fe321a818fdf5d2897f758b8d823f919d7f94` (2.4.0), niezmieniony checkout.
Python 3.12.15, Torch 2.11.0+cu128, Transformers 5.17.0,
flash-linear-attention/fla-core 0.6.0. Native: CUDA 12.9.1, Rust 1.95.0,
Candle z lockfile; BF16 i readout FP32. Pełne pakiety: `python-packages.txt`.

Baza źródeł: `db55af9d7cab5f12a18347b0bb27eb4dab4b1531`.
SHA-256 końcowej binarki:
`3fd5f8c9bdf54e508b22ed787bf73dc6f2a752f157907e197c18ab5fc9fdda80`.
SHA-256 zamrożonej wersji początkowej:
`da803da24840c3a34f528e0eecd5ddea5be47c1e2d7dab2600aed90b5af21929`.
Archiwum zawiera patch i nowe źródła obu wariantów, bez binarek i wag.

Pomiar HTTP obejmuje serializację JSON i pełny odczyt body; parsowanie jest
po zatrzymaniu zegara. OMP/MKL/Rayon: 4 wątki, native FIFO, limit 8192 tokenów,
jedna kolejność, cache wyłączone. Rozgrzewka startowa minimum 10 s,
2 próbki przed punktem długości i 8 przed krótkim obciążeniem. Jeden closed-loop
klient tworzy połączenie na każde żądanie. Zachowano ustawienia batchowania
upstream. Korpus to polskie zadanie sześciu typów, dwa pytania z prefiksem,
angielskie 27 opcji oraz Unicode NFC; punkty długości: 255 i 5080 tokenów.
Nie jest to zbiór unikalnych dokumentów produkcyjnych. Powtarzane wejścia
mogą korzystać ze wspólnego prefiksu w obrębie jednej partii. Etykieta
`workload_class: media` w krótkich obserwacjach jest historyczną etykietą
runnera; wszystkie żądania tej kampanii są tekstowe.

`screen-*` to wyłącznie eksploracyjne SIGNAL. Początkowe około 1108 ms
native C1 pochodzi z czterech próbek i nie stanowi potwierdzonego ilorazu
przyspieszenia. `screen-fused` nakładał się z eksportem i jest wykluczony
z wniosków wydajnościowych. Pierwszy pusty start runnera bez mediów przerwał
się błędem dzielenia przez zero; obsługę korpusu tekstowego naprawiono.
Eksperymenty TF32, stałego nietimingowanego algorytmu Lt i buforowania ośmiu
tokenów nie dały uzasadnionej poprawy i usunięto je z implementacji.
`profile-kernels.csv` pochodzi z pośredniego wariantu przed końcowym
współdzieleniem prefiksów; nie opisuje rozkładu czasu końcowej wersji.

## Weryfikacja i artefakty

Przeszły `cargo fmt`, Clippy workspace/all-targets na Linux bez GPU i ze
wszystkimi features CUDA, Rustdoc z prywatnymi elementami i `-D warnings`,
`cargo deny` ze świeżą bazą oraz kontrola wyjątku `paste`. Duplikaty zależności
pozostają istniejącymi ostrzeżeniami. Kernele skompilowano dla SM80/89/90
(także wariant 90a). Przegląd objął stany rozwidleń, kształty, unsafe FFI,
cykl życia wykonania i separację v1/v2. Nie dodano frameworku testowego.
F16, Metal i pozostałe GPU nie mają pomiarów runtime'u tej optymalizacji.

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/pico-speed-rtx4060-2026-10-26/99b1d838e9d2c16344505ca68a815440fe7d30fd14d29412c521bab1eedafd74/measurement-data.tar.gz`.
SHA-256: `99b1d838e9d2c16344505ca68a815440fe7d30fd14d29412c521bab1eedafd74`, 1 121 338 B.
Ponowne pobranie z autoryzacją oraz anonimowo 2026-10-26 potwierdziło
sumę archiwum i **179/179 sum plików** w obu kopiach. Kopia anonimowa
używała pustego HF_HOME i wyłączonego automatycznego tokenu; `hf auth whoami`
potwierdziło brak logowania.

Pobierz przez `hf buckets cp` z powyższego URI, sprawdź SHA-256 archiwum,
rozpakuj do nowego katalogu i uruchom `sha256sum -c SHA256SUMS` w jego
korzeniu. Pełna procedura: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).
Nie nadpisano wcześniejszych prefiksów; lokalną kopię archiwum zachowano.

Dane są pod `reports/pico-speed-rtx4060-2026-10-26/`: `confirm-eager/`,
`confirm-fast/`, audyty, `screen-*/`, eksporty i porównania, wejścia,
kwalifikacja GEMM, telemetria oraz `source/`. `REPRODUCE.md` w archiwum
podaje odtworzenie źródeł, audyt i ponowne pomiary. Prywatne surowe logi,
ślady Nsight, cache kompilatora, identyfikatory maszyn, wagi i binarki
pozostają poza archiwum.
