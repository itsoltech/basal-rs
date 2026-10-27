# Pico v4: C1/5080 szybsze od upstream eager na RTX 4060

Końcowy profil `basal-omni-pico.v4` osiągnął **536,42 ms p50** wobec
**553,58 ms upstream eager**: o **3,10% mniej czasu**, 17,17 ms na żądanie.
Potwierdziła to seria ABBA na tym samym dokumencie 5080 tokenów, z jednym
klientem HTTP i jednym pytaniem. To przewaga nad eager, domyślnie wybieranym
przez upstream na SM 8.9. Jawny tryb fast ma osobny wynik; nie deklarujemy
przewagi nad każdym trybem upstream.

## Potwierdzające A/B

Kolejność: eager, v4, v4, eager, eager, v4, v4, eager.
Pierwszy start każdego silnika wykluczono. Pozostały trzy powtórzenia po
12 próbek na punkt: **36 próbek na punkt i silnik**. Tabela pokazuje
mediany metryk z powtórzeń, nie percentyle połączonych próbek.

| Wejście, C1 | eager p50 [ms] | v4 p50 [ms] | eager p95 [ms] | v4 p95 [ms] |
|---|---:|---:|---:|---:|
| 5080 tokenów, jedno pytanie | 553,58 | **536,42** | 555,04 | **538,79** |
| 255 tokenów, 27 opcji | 33,50 | **24,67** | 33,78 | **24,76** |

Dla 5080 tokenów mediany serii mieszczą się w 553,37–553,76 ms (eager)
i 535,35–537,02 ms (v4). Przepustowość C1: 1,80647 i 1,86339 żądań/s.
`audit-confirm-eager-v4.json`: **AUDITED, zero problemów**, zero błędnych
odpowiedzi i zero zmian bitowych odpowiedzi wewnątrz każdego wariantu.
Nie wyznaczamy przedziału ufności z trzech powtórzeń.

Poprzednia kampania v3 dała 672,41 ms; obecne 536,42 ms jest o 20,22%
niższe. To porównanie z historyczną kampanią, nie nowa para ABBA v3/v4.
Osobna bieżąca kontrola jawnego fast: **270,19 ms p50 / 274,25 ms p95**,
12 próbek, zero błędów; audyt SIGNAL, zero problemów. Fast nadal jest
około 1,99× szybszy od v4 w C1/5080. To pojedyncza seria, nie ABBA.
Rozgrzewka/kompilacja fast trwała około 291 s przed gotowością serwera;
ten czas nie wchodzi do pomiaru opóźnienia. Wcześniejsza kampania miała około 268 ms;
[raport v2](../pico-speed-rtx4060-2026-10-26/README.md) zachowuje również
OOM fast przy C32. Tamten OOM nie unieważnia poprawnego pomiaru C1 fast.

## Zmiana wykonania i numeryki

Profil wybiera `qwen35-dispatch.4` i `basal-qwen35.4`.
W CUDA BF16/F16, `inference_mode: auto`, pełny prompt dłuższy niż 2048
tokenów używa iloczynu Q/K na Tensor Cores z wejściami w typie modelu,
akumulacją FP32 i wynikiem FP32. Skala jest nakładana w epilogu GEMM.
Softmax i maska przyczynowa pozostają FP32; wynik jest zaokrąglany do
typu modelu przed P/V, również z akumulacją FP32. Osobny kernel konwersji
zniknął: softmax zapisuje końcowy BF16/F16, ponownie obliczając exp w ostatnim
przebiegu. Nie przechowuje pośredniej macierzy prawdopodobieństw FP32.

To zmiana zaokrągleń względem v3, jawnie zapisana w zamrożonej polityce
wykonania i opisie backendu. `accurate` zachowuje attention FP32 z v3.
FP32 modelu, krótkie attention, granice fragmentów 512/2048 i Metal
zachowują dotychczasowe ścieżki. Historyczne zestawy v1/v2/v3 pozostają.

DeltaNet nadal używa dokładnie tych samych oddzielnie zaokrąglanych operacji
FP32 i drzewa redukcji. Warp liczy teraz dwie niezależne kolumny wartości,
współdzieląc odczyty Q/K. Stan pozostaje FP32 i jest niemutowalnym wejściem
dla każdej gałęzi; każdy wynik ma świeży bufor. Nie ma stanu między żądaniami.
Wybranie ścieżki długiego attention zależy od pełnego promptu, również przy
kontynuowaniu snapshotu prefiksu i w mieszanej partii.

## Zgodność

Końcową binarkę porównano z zamrożonym upstream FP32, na identycznych
tokenach, checkpointcie i kalibracjach. Wynik obejmuje badany zestaw,
bez wniosku o jakości na innych dokumentach.

| Zakres | Wynik |
|---|---|
| BF16 orders 1 i 2 | W każdej kolejności 44/44 pytań i 12/12 gałęzi API; zero różnic tokenów i decyzji kontraktu |
| FP32 orders 2 | 44/44 i 12/12; bench bitowo zgodny z v3 |
| BF16: 5080 tokenów i 255 opcji | 2/2 decyzje; zgodna odmowa 256 opcji |
| BF16 accurate: te same granice | 2/2 decyzje; wartości API identyczne z v3 po uporządkowaniu kluczy JSON |
| Długi prefiks z dwoma pytaniami | Single/tree: bench bajtowo równy, API identyczne po uporządkowaniu kluczy JSON |

Dla 5080 tokenów maksymalna różnica skalibrowanego prawdopodobieństwa
v4 auto względem upstream FP32 wynosi **0,003657080** (0,366 punktu
procentowego); accurate zachowuje **0,001666747** z v3 (0,167 pp).
BF16 krótkich promptów zachowuje większe różnice z v2/v3: maksymalnie
0,052261576 w bench orders 1 i 0,049012482 w orders 2. Nie deklarujemy
zachowanej kalibracji ogólnej. Dla FP32 orders 2 maksymalna różnica
skalibrowanego prawdopodobieństwa w bench wynosi 0,0000050515.

Bench BF16 orders 1 oraz wartości API są identyczne z v3. Eksport długiego
wejścia po zmianie jednej kolumny na dwie w warpie pozostał bajtowo równy
wariantowi Tensor Cores przed tą zmianą; po połączeniu softmaksu z konwersją
również pozostał równy. Te porównania nie są dowodem dla wszystkich wejść.
`validation.json` podaje zakresy i metody porównań.

Mieszanka czterech krótkich dokumentów i jednego 5080 dała zero błędów
i zero zmian bitowych odpowiedzi przy ustawieniach klientów 1/8/32.
To SIGNAL, po 10 żądań na fazę: najwyżej 10 jednocześnie, więc nie jest
to potwierdzenie pojemności ani przepustowości przy 32 równoległych żądaniach.

## Profilowanie i odrzucone warianty

WY16 z trzema iloczynami TF32 miał poprawne skończone wyniki w 162
porównaniach z rekurencją sekwencyjną (maksymalny błąd wyjścia 0,0000064671,
stanu 0,0003261566), lecz C1 wzrosło do **815,72 ms**. Wariant usunięto.
Profil dla trzech forwardów eksportu: 848,88 ms samej rekurencji WY16
plus 78,72 ms przygotowania. To ujemny wynik, nie użyteczna optymalizacja.
Nie zachowano kompletnego pakietu źródłowego tego eksperymentu.

Pośrednie eksploracje C1: same Tensor Cores **572,66 ms**, następnie dwie
kolumny rekurencji **557,89 ms**, a połączony softmax **531,04 ms**.
To pojedyncze serie po sześć próbek bez początkowej rozgrzewki czasowej,
z dwiema próbkami rozgrzewki na punkt. Nie zastępują końcowego ABBA.
Profil wariantu z samymi Tensor Cores: 440,54 ms rekurencji i 165,75 ms
`ucopy_bf16`, łącznie dla trzech forwardów, bez dekompozycji końcowego v4.
Surowy Nsight i logi zawierające dane maszyny pozostają prywatne.

## Metoda i odtworzenie

HTTP: czas od serializacji JSON do odczytu całego body, parsowanie po zegarze.
BF16, orders 1, readout FP32, wyłączone cache stanu/prefiksu, FIFO,
budżet 8192 tokenów, po cztery wątki OMP/MKL/Rayon. Rozgrzewka potwierdzenia
minimum 10 s na start i dwie próbki przed każdym punktem. Jedna seria GPU
naraz, bez eksportów i kompilacji CPU w tle. Korpus to cztery powtarzane
krótkie dokumenty syntetyczne i jeden dokument 5080; etykieta `media`
w klasach krótkich próbek jest nazwą historyczną, nie rzeczywistymi mediami.
Telemetria co 200 ms; pobór mocy niedostępny, bez wniosków o energii.

RTX 4060, 8188 MiB, SM 8.9, sterownik 615.78.08, limit 115 W;
ustawień GPU nie zmieniano. Rust 1.95.0, CUDA 12.9.1, cuBLASLt 120901.
Referencja: Python 3.12.15, Torch 2.11.0+cu128, Transformers 5.17.0,
FLA/fla-core 0.6.0, upstream `615fe321a818fdf5d2897f758b8d823f919d7f94`.
Nie zmieniono zależności Cargo, tabeli GEMM ani checkoutu upstream.

Model `Remek/basal-omni-pico`, rewizja
`a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`, SHA-256 wag
`ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9`.
Tabela BF16 RTX 4060 ma 48 wpisów, identyfikator
`72fb35fcb899c71c6ed7e52ddb64e20686848f2f7ed8304401147bdac1bdee2e`.
Baza źródeł `db55af9d7cab5f12a18347b0bb27eb4dab4b1531`.
Końcowa binarka v4, użyta do kwalifikacji i ABBA, SHA-256:
`2d2bb90841958d4ad84f67581e922260e854dccbd8274cc7145769e2a2caba4b`.

Przeszły fmt, Clippy workspace/all-targets Linux i CUDA/all-features,
Rustdoc z prywatnymi elementami i `-D warnings`, cargo deny ze świeżą bazą,
kontrola wyjątku paste i przegląd granic FFI, wersjonowania oraz stanów.
Nie dodano frameworków testowych. F16, Metal i inne GPU nie mają nowej
kwalifikacji runtime'u. Instrukcje źródeł, audytu i replay: `REPRODUCE.md`
w archiwum; dane i źródła znajdują się pod
`reports/pico-parallel-rtx4060-2026-10-26/`.

## Opublikowane artefakty

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/pico-parallel-rtx4060-2026-10-26/e8dbadd48311f2d9bae58021c01fd8932c8d79b19cd927860ca3d0ff858649a7/measurement-data.tar.gz`.
SHA-256: `e8dbadd48311f2d9bae58021c01fd8932c8d79b19cd927860ca3d0ff858649a7`,
646 249 B. Pobranie z autoryzacją i anonimowo 2026-10-26 potwierdziło
sumę archiwum oraz **123/123 sum plików** w obu kopiach. Anonimowa kontrola
używała pustego HF_HOME, wyłączonego automatycznego tokenu i `whoami`
potwierdzającego brak logowania. Lokalną kopię archiwum zachowano.

Wewnątrz `reports/pico-parallel-rtx4060-2026-10-26/`: `confirm-eager-v4/`,
`screen-fast-control/`, `screen-mixed/`, wcześniejsze `screen-*`, audyty,
`qualified-*`, `compare-*.json`, `explorations/`, wejścia, referencje,
profile zbiorcze, telemetria, walidacja i źródła v3/v4. Dane nie zawierają
identyfikatorów maszyn, adresów, lokalnych ścieżek ani prywatnych logów.

Pobierz przez `hf buckets cp`, sprawdź SHA-256, rozpakuj do nowego katalogu
i wykonaj `sha256sum -c SHA256SUMS` w jego korzeniu. `REPRODUCE.md` opisuje
odtworzenie źródeł i powtórzenie pomiarów. Pełna procedura:
[MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).
