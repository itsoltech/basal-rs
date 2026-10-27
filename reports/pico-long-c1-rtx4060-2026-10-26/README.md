# Pico: C1 i 5080 tokenów na RTX 4060

Profil CUDA `basal-omni-pico.v3` zmniejszył p50 dla badanego wejścia 5080
tokenów z **727,80 do 672,41 ms**, o **7,61%** (55,39 ms, przyspieszenie
1,082×). To potwierdzone A/B względem v2. Nadal nie wyprzedza upstream dla
tego dokumentu. Wynik dotyczy jednego powtarzanego dokumentu syntetycznego.

## Potwierdzające A/B

ABBA: v2, v3, v3, v2, v2, v3, v3, v2. Po cztery starty na silnik;
pierwszy każdego silnika wykluczono. Pozostały trzy powtórzenia po 12
pomiarów na punkt długości, czyli **36 pomiarów na punkt i silnik**.
Tabela pokazuje mediany metryk z trzech powtórzeń, nie połączony percentyl.

| Wejście, C1 | v2 p50 [ms] | v3 p50 [ms] | v2 p95 [ms] | v3 p95 [ms] |
|---|---:|---:|---:|---:|
| 5080 tokenów, jedno pytanie | 727,80 | **672,41** | 732,13 | **673,49** |
| 255 tokenów, 27 opcji | 25,87 | 25,72 | 26,26 | 25,92 |

Dla 5080 tokenów mediany powtórzeń mieszczą się w 727,49–730,61 ms (v2)
i 669,72–673,03 ms (v3). Przepustowość C1 wzrosła z 1,37279 do 1,48756
żądań/s, o 8,36%. Nie podajemy przedziału ufności z trzech powtórzeń.
`audit-confirm.json`: **AUDITED, 0 problemów**, zero niepoprawnych odpowiedzi
i zero zmian bitowych odpowiedzi wewnątrz każdego wariantu. Różnice
pomiędzy v2 i v3 są zmianą numeryki, opisaną poniżej.

Kontrola upstream eager: **552,27 ms p50 / 553,50 ms p95**, 12 próbek,
zero błędów; audyt ma status SIGNAL i 0 problemów.
To osobna eksploracyjna próba C1, nie kolejna seria potwierdzająca ABBA.
Poprzednia potwierdzająca kampania na tej samej karcie miała eager 555,72 ms
i jawne fast 268,02 ms dla tego wejścia;
[pełne porównanie](../pico-speed-rtx4060-2026-10-26/README.md) zachowuje
również OOM fast przy C32. Fast nie powtarzano w tej kampanii.

## Zmiana wykonania

Nowy profil wybiera `qwen35-dispatch.3` i `basal-qwen35.3`.
CUDA BF16/F16 używa fragmentów do 2048 tokenów dla pełnego promptu
powyżej 2048; krótszy prompt zachowuje fragmenty do 512 i attention online
z v2. FP32 i Metal zachowują referencyjne fragmenty 64.

Długi prompt od pierwszej pozycji używa attention FP32 z batched GEMM,
kafelkami wyników do 64 MiB na bufor oraz istniejącym kernelem GPU
maskowania przyczynowego z softmax. Znikają gęste maski tworzone na CPU
i osobne dodawanie maski dla każdej warstwy. Nie rozszerzamy szeregowego
attention online z krótkich promptów na długi dokument. Rekurencja DeltaNet
pozostaje sekwencyjna, ze stanem FP32; jej algorytmu nie zmieniono.
Usunięto też kopię jednoelementowej listy wyników rekurencji.

Granice fragmentów i wybór attention należą do pełnego promptu. Krótki
prompt nie przechodzi na ścieżkę długą, gdy jest w jednej grupie z długim.
Snapshot prefiksu jest obliczany ze ścieżką jego pełnych kontynuacji;
długość samego prefiksu nie przełącza attention na krótkie. Każda gałąź
nadal zapisuje świeży stan. Zestawy v1/v2 pozostają zaimplementowane.
Nie zmieniono tabeli GEMM, zależności Cargo, ustawień GPU ani checkoutu upstream.

Profil bazowy Nsight wykazał 436,26 ms rekurencji, 220,58 ms krótkiego
attention online oraz wiele kopii i osobnych operacji na wynikach, łącznie
dla trzech forwardów eksportu 5080 tokenów. `profile-v2-kernels.csv`
zawiera agregaty. To profil v2, nie rozkład czasu końcowego v3.
Synchronizowane pomiary etapów i surowy Nsight pozostają prywatne;
nie są podstawą tabeli opóźnień HTTP.

## Zgodność

Porównano końcową binarkę z zamrożonym upstream FP32 z poprzedniej kampanii.
Tokeny, checkpoint i kalibracje referencji są identyczne.

| Zakres | Wynik |
|---|---|
| BF16, orders 1 | 44/44 pytań + 12/12 gałęzi API, bez różnic tokenów |
| BF16, orders 2 | 44/44 + 12/12, bez różnic tokenów |
| FP32, orders 2 | 44/44 + 12/12, bez różnic tokenów |
| Granice BF16 | 2/2: 5080 tokenów i 255 opcji; zgodna odmowa 256 opcji |
| Rozwidlenie długiego prefiksu | Dwa pytania do tego samego dokumentu: single/tree identyczne pliki wynikowe; odpowiedź serwowana vs osobne pytania: różnica 0 |

Na dokumencie 5080 tokenów maksymalna różnica skalibrowanego
prawdopodobieństwa BF16 v3 względem FP32 wynosi **0,001666747**
(0,167 punktu procentowego). Jedna zgodna decyzja nie dowodzi jakości
na innych dokumentach. Pozostają różnice BF16 względem FP32 z v2:
maksymalnie 0,052261576 w bench orders 1 i 0,049012482 w orders 2.
Nie deklarujemy niezmienionej kalibracji ogólnej ani jakości produkcyjnej.

Bench orders 1/2 BF16 oraz orders 2 FP32 są byte-identyczne z v2.
API ma identyczne wartości po kanonicznym uporządkowaniu kluczy JSON;
oryginalne pliki różnią się kolejnością kluczy. `validation.json`
wymienia pary plików i metodę porównania. Dodatkowo mieszanka czterech
krótkich wejść i dokumentu 5080 dała zero błędów i zero zmian bitowych
odpowiedzi przy ustawieniach klientów 1/8/32. To **SIGNAL**, po 10 żądań
na fazę, zatem najwyżej 10 żądań jednocześnie; nie jest to potwierdzenie
przepustowości ani pojemności przy 32 równoległych żądaniach.

## Metoda, wersje i odrzucone próby

HTTP: od serializacji JSON do pełnego odczytu body; parsowanie po zegarze.
Jedna kolejność, BF16, readout FP32, cache wyłączone, FIFO, budżet 8192
tokenów, OMP/MKL/Rayon po 4 wątki. Rozgrzewka minimum 10 s na start,
2 próbki przed każdym punktem długości. Te same wejścia i dopuszczalna
tabela GEMM w obu natywnych wariantach; bez równoległej inferencji i
kompilacji CPU podczas ABBA. Telemetria co 200 ms: `telemetry.csv`.
Pobór mocy był niedostępny; nie wyciągamy wniosków o energii.

NVIDIA GeForce RTX 4060, 8188 MiB, SM 8.9, sterownik 615.78.08, limit 115 W.
Rust 1.95.0, CUDA 12.9.1 / cuBLASLt 120901. Referencja Python 3.12.15,
Torch 2.11.0+cu128, Transformers 5.17.0, FLA/fla-core 0.6.0;
upstream `615fe321a818fdf5d2897f758b8d823f919d7f94` (2.4.0).

Model `Remek/basal-omni-pico`, rewizja
`a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`, SHA-256 wag
`ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9`.
Tabela BF16 RTX 4060, 48 wpisów, identyfikator
`72fb35fcb899c71c6ed7e52ddb64e20686848f2f7ed8304401147bdac1bdee2e`.
Źródła bazują na `db55af9d7cab5f12a18347b0bb27eb4dab4b1531`.
SHA-256 binarki v2:
`3fd5f8c9bdf54e508b22ed787bf73dc6f2a752f157907e197c18ab5fc9fdda80`;
końcowego v3:
`71f6cf96fd8e6c61d9e74309f658e1475288033a18bd043828ac623cf6b5d096`.

Zachowano eksploracje: samo zwiększenie fragmentu do 2048 nie dało zysku
(729,36 ms), pośredni wariant kafelków miał 701,98 ms, pierwszy v3 około
664,07 ms, a fragment 4096 pogorszył wynik do 703,75 ms. To pojedyncze
serie SIGNAL; końcowy zysk pochodzi wyłącznie z ABBA. `screen-chunk2048`
nakładał się z eksportem GPU i jest wykluczony z wniosków czasowych;
osobny `screen-chunk2048-isolated` nie miał tego problemu. Pierwszy
`screen-v2` nakładał się z budowaniem CPU i również nie jest potwierdzeniem.
Krótkie teksty nadal mają historyczną etykietę `media` w runnerze —
w tej kampanii nie ma żądań z mediami.

## Weryfikacja i dane

Przeszły formatowanie, Clippy workspace/all-targets Linux i CUDA/all-features,
Rustdoc z prywatnymi elementami i `-D warnings`, cargo deny ze świeżą bazą,
kontrola wyjątku paste oraz przegląd wersjonowania, stanów i granic CUDA.
Nie dodano frameworków testowych. F16, Metal i inne GPU nie mają nowego
potwierdzenia runtime'u; wyniku RTX 4060 BF16 nie uogólniamy na te ścieżki.

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/pico-long-c1-rtx4060-2026-10-26/2dd5352f02da4f7e31577d5ffd22f7b8b17988f8a3311534b425ec53a25af4ff/measurement-data.tar.gz`.
SHA-256: `2dd5352f02da4f7e31577d5ffd22f7b8b17988f8a3311534b425ec53a25af4ff`, 765 902 B.
Pobranie z autoryzacją i anonimowo 2026-10-26 potwierdziło sumę archiwum
oraz **145/145 sum plików** w obu kopiach. Anonimowe pobranie używało
pustego HF_HOME i wyłączonego automatycznego tokenu; `hf auth whoami`
potwierdziło brak logowania. Lokalną kopię archiwum zachowano.

Pobierz przez `hf buckets cp` z powyższego URI, sprawdź SHA-256,
rozpakuj do nowego katalogu i wykonaj `sha256sum -c SHA256SUMS` w jego
korzeniu. Pełna procedura: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).

Dane są pod `reports/pico-long-c1-rtx4060-2026-10-26/`: `confirm-v2-v3/`,
audyt, kontrole i eksploracje `screen-*/`, eksporty `qualified-*/`,
porównania `compare-qualified-*.json`, wejścia i poprzednie referencje,
profil zbiorczy, telemetria oraz źródła v2/v3. `REPRODUCE.md` opisuje
odtworzenie kodu, audyt bez GPU oraz replay. Wagi, binarki, surowy Nsight,
cache kompilatora i logi zawierające dane maszyn pozostają lokalnie.
