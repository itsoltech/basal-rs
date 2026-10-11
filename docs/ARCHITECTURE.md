# Architektura

## Przegląd

```text
HTTP (axum, tokio)
  -> wątek GPU: walidacja i plan żądania (prompty, tokeny), kolejka, partie
       -> Engine::run_plans: wiersze-drzewa wspólnych prefiksów
            -> Backend::letter_logits: forward Llama, logity liter
       -> uśrednienie porządków, kalibracja, odpowiedź System One
```

| Crate | Odpowiedzialność |
|---|---|
| `basal-core` | kontrakt żądań (`request.rs`), prompt (`prompt.rs`), tokenizer i litery (`tokenize.rs`), JSON jak `json.dumps` Pythona (`pyjson.rs`), pakowanie (`pack.rs`), decyzje i confidence (`decision.rs`), `facts.rs`, `evidence.rs`, Choice 11–255 (`large_choice.rs`), silnik i trait `Backend` (`engine.rs`), współdzielenie GPU przez dwa silniki (`gate.rs`) |
| `basal-gpu` | loader safetensors, forward Llama na candle 0.11, fuzje (`fused.rs`, kernele `kernels.cu` i `kernels.metal`), GEMM przez cuBLASLt (`cublaslt.rs`) lub port GEMM MLX (`gemm.rs`, Metal), głowica `evidence_head.pt` |
| `basal-cli` | polecenie `basal`, serwer (`serve.rs`), eksport i porównania z referencją, pomiary |

`basal client` składa requesty z pytań CLI lub szablonu, albo przyjmuje pełne requesty System One.
`client/input.rs` czyta rekordy z limitem rozmiaru, `client/request.rs` składa i waliduje requesty przez `basal-core`,
`client/http.rs` obsługuje HTTP (połączenia, timeouty, ograniczone ponowienia), a `client.rs` steruje wykonaniem
i wyjściem. Liczba oczekujących/wykonywanych/buforowanych rekordów HTTP jest ograniczona przez `--jobs`.
W trybie lokalnym jeden `Engine` przetwarza rekordy kolejno; ładowanie i inferencja odbywają się poza executorem
Tokio, bez serwera i jego harmonogramu. Wybór tabel GEMM używa konfiguracji `serve`. Kontrakt promptu i odpowiedzi
pozostaje w `basal-core`. Interfejs i przykłady: [CLIENT.md](CLIENT.md).

## Kontrakt promptu

Prompt, szablon czatu, heurystyka języka (polskie znaki), serializacja stanu,
kolejność opcji i litery odtwarzają upstream (`prompt.py`, `server.py`).
Manifest przy ładowaniu porównuje `config.json`, szablony z `basal.json` i
`chat_template.jinja`; inny kontrakt kończy się błędem.

Tokenizer: transformers 5 buduje pre-tokenizer jako
`Metaspace(prepend_scheme="first")`, a `tokenizer.json` zawiera `"always"`.
Runtime stosuje `"first"` jak upstream i wyłącza obcinanie oraz padding z
`tokenizer.json` (obcinanie do 1536 tokenów zmieniałoby prompty długich
stanów).

Każde pytanie to dwa prompty z opcjami w dwóch porządkach. Z logitów liter
liczymy softmax osobno dla każdego porządku, odwracamy permutację, uśredniamy
i stosujemy temperaturę typu pytania:

```text
p = (softmax(l_forward) + unpermute(softmax(l_reverse))) / 2
p_cal = softmax(log(max(p, 1e-12)) / T_type)
```

## Pakowanie

Prompty partii są pakowane w wiersze-drzewa: skompresowane trie wszystkich
promptów wiersza (jak `_pack` upstream 1.5), spłaszczone w głąb. Każdy
wspólny prefiks liczy się raz: szablon, stan wspólny dla wszystkich pytań o
niego (także z różnych żądań jednej partii), treść pytania wspólna dla obu
porządków. Identyczne prompty mają jeden odczyt. Pozycje RoPE to indeksy
tokenów w ich własnych promptach, a token widzi wcześniejsze tokeny swojego
bloku i bloków przodków, co jest równoważne osobnym forwardom pełnych
promptów. Głębokość drzewa jest ograniczona do 8 bloków; poniżej prompty
dostają osobne bloki.

Wiersz ma do 32768 tokenów, forward do 8192 (wiersze dłuższe liczą się
osobno). Pytania o ten sam stan trafiają obok siebie.

Stały prefiks promptu (szablon do stanu, 53 tokeny PL i 66 EN) jest liczony
raz przy starcie. Opcjonalny cache stanu (`--state-cache-mb`) przechowuje K/V
szablonu ze stanem dla kolejnych żądań o ten sam stan (LRU z budżetem
pamięci; CUDA/Metal). Oba działają tylko przy dokładnej zgodności token IDs.
Intel obsługuje cache stałego prefiksu, bez cache stanu między requestami.

## Forward

Model Llama z biasami lub bez (basal-1.5 mini i 4.5B mają biasy, max nie). Domyślna
precyzja to f16: wagi bf16 konwertowane przy ładowaniu, strumień residualny,
normy i MLP w f16, akumulacja GEMM w f32. Precyzja `f32` trzyma wagi bf16 i
rozszerza je przed użyciem (dokładny forward FP32, punkt odniesienia).

- GEMM, normy i MLP działają na zwartej liście tokenów całej partii, bez
  paddingu.
- Projekcje q/k/v oraz gate/up to pojedyncze GEMM; bias, RoPE i podział głów,
  bias+SiLU·up oraz bias+residual to własne kernele.
  CUDA SiLU w wektoryzowanej ścieżce f16 używa bloku 256 wątków na compute
  capability 9.0 w buildzie z PTX 90a; pozostałe ścieżki pozostają przy 1024.
  Zmienia się podział pracy, nie arytmetyka
  ([pomiary H100 i RTX 6000 Ada](../reports/inference-campaign-2026-10-09/README.md)).
  `BASAL_SILU_BLOCK=128|256|512|1024` pozwala przypiąć wariant do pomiarów.
- Attention w f32 po jednostkach drzewa: zapytania jednego bloku i klucze
  jego przodków oraz własnego bloku, kafelkowane według pozycji klucza w
  prompcie. CUDA: `attn_tree_tc` na tensor cores (mma.sync, operandy f16
  rozbite na część wysoką i niską, czyli dokładność bliska f32) dla f16/bf16,
  na H100 (compute capability 9.0, forward f16) `attn_tree_wgp` z tą samą
  arytmetyką na wgmma, z ładowaniem następnego kafelka K/V w trakcie liczenia
  bieżącego, i bitowo tym samym wynikiem
  ([pomiar](../reports/attention-h100-wgmma-2/README.md)); przy krótkich
  promptach w blokach po 64 zamiast 128 wierszy (`attn_tree_wgp64`, te same
  bity, [pomiar](../reports/attention-h100-short/README.md)), `attn_tree_f32` dla
  ścieżki f32. Domyślnie oba rozmiary bloków WGP używają wariantu `_sw128`:
  ciągłych odczytów K/V oraz układu 128 B w pamięci współdzielonej. Kolejność
  arytmetyki pozostaje taka sama; [pomiary i zgodność](../reports/inference-profile-h100-2026-10-09/README.md).
  `BASAL_WGP_LOAD=original|coalesced|sw128` wybiera stary układ, samą zmianę
  odczytów lub pełny swizzling do A/B. Dotyczy wyłącznie Hopper WGP f16;
  ścieżki bf16, f32 i innych GPU zachowują swój wybór kernela.
  K i V przychodzą z `qkv_rope` od razu rozbite na płaszczyzny
  f16, a wynik attention od razu jako scalone głowy w dtype forwardu. Metal: `attn_tree_f32` z `kernels.metal`
  na macierzach `simdgroup_float8x8` w f32 (`BASAL_ATT=sdpa`: poprzednie SDPA
  z MLX po całym wierszu z maską, wynik zależny od pakowania).
- W ostatniej warstwie o_proj i MLP liczą się tylko dla pozycji odczytu, a
  logity tylko dla wierszy `lm_head` odpowiadających literom, w f32.
- CUDA GEMM przez cuBLASLt z tabelą algorytmów z pełnego przeszukania
  (`basal gemm-search`): algorytm, kafelek, liczba etapów, split-K z
  redukcją f32, swizzle; bez tabeli najszybszy z 16 algorytmów heurystyki
  mierzony przy pierwszym użyciu.

### Kwantyzacja W8A16 (`q8_0`)

Opcjonalna, domyślnie wyłączona (`quantization: none`), tylko CUDA z
`dtype: f16` i kernelami `fused` (walidacja przed pobraniem modelu). Cztery
macierze projekcji każdej warstwy (`wqkv`, `wo`, `wgu`, `wdown`) mają w
`Layer` typ `ProjWeight`: `Dense(Tensor)` albo `Q8(Arc<QTensor>)`. Loader
czyta części bf16 tak jak dla gęstych wag (sklejone wzdłuż N), sprawdza
kształt `[N, K]`, `K % 32 == 0`, przepełnienia i dokładną liczbę bloków,
konwertuje wiersze do f32 na wszystkich rdzeniach (bez kopii f32 całej
macierzy), odrzuca wartości nieskończone i skale poza zakresem f16, liczy
bloki ze skalą zerową (`amax/127 ≤ 2⁻²⁵`) i podnormalną, kwantyzuje
`BlockQ8_0::from_float` i wysyła bloki raz przez `load_quantized` Candle.
Bloki muszą leżeć na tym samym urządzeniu Candle (`DeviceId`) co backend;
dekwantyzacja idzie wyłącznie na `q.device()`, więc nie ma transferów.
Wysłania i jednorazowa dekwantyzacja najmniejszej macierzy (ładuje moduł
`QUANTIZED` i kernel przy starcie) są kolejkowane przed synchronizacją
kończącą ładowanie. W forwardzie `matmul_t` dla `Q8` tworzy przejściowy
tensor f16 macierzy (`dequantize_block_q8_0_f16`), podaje go temu samemu GEMM
cuBLASLt f16 i zwalnia go po wywołaniu, na strumieniu wątku silnika. Tor
`Dense` jest kodem sprzed zmiany; jedyną różnicą jest dopasowanie wariantu.
Forki silnika (tor długich żądań) współdzielą bloki przez `Arc`, a repliki
ładują własne. `basal profile` raportuje czas dekwantyzacji osobno
(`q8_dequant_ms_per_item`, część sekcji projekcji, nie dodawana do
`sections_sum_ms`). `QMatMul` Candle nie jest używany (kwantyzuje aktywacje
do Q8_1). Własny kernel W8A16 i formaty 4-bitowe są poza tym etapem.

### Niezależność od partii

Z tabelą `--invariant` (dla każdego kształtu wag algorytmy bez split-K z
jednej grupy dającej bitowo te same wyniki, najszybszy w każdej klasie M), attention kafelkowanym według pozycji klucza i odczytem liter
własnym kernelem wynik pytania jest bitowo ten sam pojedynczo, w dowolnej
partii, w drzewie z innymi pytaniami i z cache prefiksu. Odpowiedź serwera nie
zależy więc od ruchu w tej samej chwili. Na Metal (GEMM MLX niezależny od
liczby wierszy, attention po jednostkach drzewa) wynik jest bitowo ten sam
pojedynczo, w partii i w drzewie
([pomiar](../reports/metal-m2-max-tree/README.md)); z cache stanu tego nie
mierzono.

### Intel Vulkan

Opcjonalny `basal-gpu::intel::IntelBackend` implementuje ten sam trait `Backend`.
Kernele WGSL wykonują forward FP16 przez wgpu/Vulkan, z sumami GEMM, RMSNorm,
RoPE, attention i logitami liter w FP32. Attention czyta ciągłe zakresy przodków
drzewa i aktualizuje online softmax; nie alokuje kwadratowej maski. Granica `Packed`
waliduje tokeny, pozycje, rodziców, ciągłość bloków oraz długość prefiksu przed użyciem cache.

Wiersze partii liczą się kolejno; projekcje wewnątrz wiersza mają zwarte kafelki GEMM.
K/V stałych prefiksów współdzielą forki. Wagi i pipeline'y są niemutowalne i współdzielone
przez `Arc`, a scratch jest lokalny dla forwardu. Wagi embeddingu i głowicy pozostają
w pliku checkpointu (odczyt `pread`, bez mapowania): przesyłane są tylko potrzebne wiersze. Ostatnia warstwa
ogranicza o_proj i MLP do pozycji odczytu, także przy powtórzonych pozycjach evidence.

Wątek długich żądań synchronizuje GPU przed checkpointem bramki między warstwami.
Błędy walidacji i alokacji wgpu są przechwytywane; odczyt GPU ma timeout.
Telemetria używa `VK_EXT_memory_budget` rzeczywistego urządzenia, zachowując jego lifetime.
Szczegóły opcji i ograniczeń: [INTEL.md](INTEL.md).

## Rozszerzenia basal-1.5

- `multi`: osobne pytanie tak/nie dla każdej etykiety, temperatura `noul`;
  gałęzie idą do tego samego drzewa co reszta żądania.
- `act`: odczyt jak `noul` lub `choice`, potem akcja o najmniejszym
  oczekiwanym koszcie; `max_error` porównuje confidence z certyfikowanym
  progiem z `CALIBRATION.json` i w razie potrzeby wybiera odroczenie.
- `facts: "auto"`: port `facts.py` (fakty wyliczone z dat i kwot w stanie,
  dopisane do stanu), wynik zgodny co do bajtu.
- `evidence`: osobny forward promptu w oryginalnej kolejności opcji i głowica
  `evidence_head.pt` (f32) na stanach tokenów stanu; do 3 nienakładających się
  fragmentów do 80 tokenów z offsetami znakowymi.

## Choice 11–255

Opcje dzielone są dwa razy na `ceil(n/10)` grup: blokami i cyklicznie, więc
każda opcja jest w dwóch grupach, a graf grup jest spójny. Każda grupa to
pytanie basal w jednej kolejności opcji (jedno drzewo ze wspólnym stanem).
Wspólny rozkład to model Luce `softmax(theta)` dopasowany metodą największej
wiarygodności do rozkładów grup, uzupełniony rundą finałową z 10 najlepszymi
opcjami w obu kolejnościach. To przybliżenie (model nie widzi wszystkich
opcji naraz); ocena na zbiorach z etykietami i porównanie z TypeSafe w
[reports/choice-sets](../reports/choice-sets/README.md), wcześniejsze w
[reports/large-choice-1.5](../reports/large-choice-1.5/README.md). Inne
strategie (`basal eval-choice-set --strategies`) służą do porównań.

## Serwer

Handlery HTTP (tokio) przekazują żądania do wątku modelu z pola `model`
(`/v1/basal` bez tego pola: do modelu domyślnego). Wątek
tokenizuje i planuje każde żądanie przy przyjściu (błędy walidacji wracają
od razu), szacuje jego koszt liczbą tokenów po współdzieleniu prefiksów i
układa kolejkę:

- `hrrn` (domyślnie): najwyższy stosunek (czekanie + szacowany czas) /
  szacowany czas; krótkie żądania wyprzedzają długie, a długie awansują z
  czasem oczekiwania.
- `fifo`: kolejność przyjścia.

Partia to pierwsze żądanie z kolejki i każde następne, które mieści się w
`--max-batch-tokens`; pytania wszystkich żądań partii dzielą forwardy.
Żądania porzucone przez klienta są pomijane, a ponad `--max-inflight` żądań
w toku dostaje 529 z `Retry-After: 1` (status TypeSafe dla przeciążenia).

Żądania powyżej `--long-tokens` liczy drugi tor: drugi silnik na tym samym
GPU, ze wspólnymi wagami, tabelą GEMM i prefiksami szablonu. Oba tory
korzystają z GPU na zmianę (`basal_core::gate`): tor długich żądań po każdej
warstwie czeka na zakończenie swoich kerneli i oddaje GPU, gdy główny tor ma
partię, a on sam pracował co najmniej `--long-slice-ms`. Po jednej partii
głównego toru wraca do tej samej warstwy. Zatrzymany forward kontynuuje z tymi
samymi tensorami, więc wyniki nie zależą od toru ani przerw. Krótkie żądania
nie czekają w ten sposób na kilkusekundowy forward długiego dokumentu.

Kilka modeli w jednym procesie (`basal serve --config`): każdy ma własny
silnik, kolejkę i tor długich żądań. Model jest przypisany do jednego GPU
(`device`, domyślnie 0); silniki wszystkich modeli tego samego GPU, łącznie z
ich torami długich żądań, korzystają z jednego `gate` tego GPU. Główne tory są
pilne, tory długich żądań oddają GPU między warstwami, więc krótkie partie
dowolnego modelu wyprzedzają długie żądania każdego modelu tego GPU. Silniki
na jednym GPU nie liczą jednocześnie: GPU jest i tak ograniczone mocą, a
kolejność decyduje harmonogram zamiast sterownika. Różne GPU mają osobne
`gate` i liczą jednocześnie; limit `max_inflight`, router HTTP i `/health`
pozostają wspólne dla procesu.

Tożsamością GPU w harmonogramie jest numer `device` z konfiguracji, czyli
indeks CUDA widoczny dla procesu (po `CUDA_VISIBLE_DEVICES`), a nie
identyfikator urządzenia candle: każde `Device::new_cuda` tworzy nowy
identyfikator na tym samym GPU. Ten sam numer wybiera kontekst CUDA przy
ładowaniu, tabelę GEMM (nazwa karty, wyszukiwanie i walidacja algorytmów
cuBLASLt na tym GPU) oraz pomiar pamięci w `basal benchmark`. `basal doctor`
pyta sterownik CUDA (`libcuda.so.1`, bez tworzenia kontekstu) o urządzenia
widoczne dla swojego procesu i według tych numerów sprawdza pamięć oraz
compute capability GPU każdego `device`; lista `nvidia-smi` jest tylko
inwentarzem maszyny, bo jej numeracja może różnić się od CUDA. Gdy sterownika
nie da się zapytać, doctor nie ocenia urządzeń modeli. Wybór PTX, kernela attention
(Hopper albo nie) i liczba SM są zapamiętywane osobno dla każdego GPU, więc
karty różnych architektur w jednym procesie nie dziedziczą wyboru pierwszej.
Ta sama nazwa modelu w dwóch wpisach kończy się błędem przed ładowaniem.
Metal i Intel obsługują tylko GPU 0. Podziału jednego modelu między GPU nie ma.

Repliki (`replica_devices`, CUDA) to niezależne instancje jednego modelu pod
jedną nazwą, po jednej na każdy ordinal z listy. Walidacja konfiguracji
odrzuca pustą listę, powtórzenia, `device` razem z listą (także `device: 0`) i
`gemm_table: none` przy kilku replikach; jeden element listy to zwykła
instancja. Przed pobraniem modelu `basal_gpu::gpu_identity` pyta sterownik,
bez tworzenia kontekstu, o nazwę, compute capability i liczbę SM każdej karty
grupy; różne typy kończą się błędem. Tabela GEMM jest rozwiązywana raz, na
pierwszej karcie grupy (jeden plik w `gemm_cache`), a potem
`check_gemm_table` sprawdza ją na każdej karcie przed załadowaniem
którejkolwiek repliki. Każda replika ładuje wagi na swoją kartę; fork toru
długich żądań współdzieli wagi tylko w obrębie repliki, a scratch i cache ma
własne. Gate'y powstają z płaskiej listy wszystkich instancji (modeli i
replik) według ordinalu, z tą samą regułą co dla modeli, więc niezależne karty
nie czekają na siebie.

`App::routes` trzyma dla modelu jedną instancję (`Route::One`, dotychczasowa
ścieżka: bez kontroli dostępności, bez rezerwacji przy wyłączonych metrykach)
albo grupę (`Route::Group`). Grupa wybiera dostępną replikę z najmniejszą
liczbą rezerwacji (`ReplicaLoad`), skanując od indeksu przesuwanego o jeden
na żądanie, więc remisy rotują. Wybór i zwiększenie licznika to osobne kroki:
równoległe żądania mogą wybrać tę samą replikę, ale liczniki się nie gubią.
Rezerwacja (`Reservation`, RAII) należy do zadania: powstaje przed wysłaniem
do kolejki repliki i znika przy odpowiedzi (przed wysłaniem odpowiedzi
klientowi), błędzie planowania, pominięciu porzuconego żądania, po partii
także dla rozłączonego klienta oraz przy opróżnieniu kolejki zamkniętego
albo spanikowanego workera. `SendError` zwraca całe nieprzyjęte zadanie;
wtedy router zwalnia rezerwację i próbuje innej dostępnej repliki, co
najwyżej raz na replikę. Po przyjęciu zadania przez worker nie ma ponowień.
Bez dostępnej repliki odpowiedź to dotychczasowe „engine stopped”. Globalny
licznik admission handlerów (`max_inflight`) i gauge
`basal_model_inflight_requests` nie zmieniają semantyki.

Replika jest niedostępna, gdy backend zgłosił utratę urządzenia
(`DeviceHealth`) albo jej główny worker zamknął kolejkę; zamknięcie toru
`long` tego nie zmienia, bo tor główny przejmuje jego żądania. CUDA nie
raportuje `DeviceHealth`, więc na CUDA widoczne jest tylko zatrzymanie
workera. Fatalny błąd CUDA może wymagać restartu całego procesu, więc
repliki nie są izolacją awarii sprzętu. `/health` dla modeli z jedną
instancją zachowuje dotychczasową treść i reguły (`device_lost`,
`lost_models`). Grupa z częścią niedostępnych replik daje 200 ze
`status: ok` i `degraded_models`, grupa bez dostępnych replik 503 ze
`status: engine_stopped` i `unavailable_models`; wpis zawiera `model`,
`replicas`, `available` i `unavailable` (`replica`, `device`, `reason`:
`worker_closed` albo `device_lost`).

### Pomiar scenariuszy HTTP

`basal benchmark` uruchamia ten sam router i scheduler na prywatnym,
wcześniej zajętym listenerze loopback z portem przydzielonym przez system.
Własny klient HTTP wykonuje fazę sekwencyjną i profil mieszany przy
zadanej współbieżności; nie symuluje partii przez bezpośrednie `Engine::run`.
Domyślne workloady są osadzone w binarce (`include_str!` z
`crates/basal-cli/benchmark-data/v2`); przed ładowaniem modelu komenda
porównuje ich SHA-256 ze stałymi z locka profilu i liczniki z manifestem.
Payloady planuje `Engine::plan_request` raz na unikalne body, przed fazami.
Właściciel serwera przekazuje sygnał zamknięcia, anuluje klientów i czeka
na zakończenie workerów GPU. Zwykłe `serve` nadal używa SIGINT/SIGTERM.
Monitor pamięci działa na osobnym wątku z uchwytem do rzeczywistego
urządzenia silnika. Metodyka i ograniczenia: [Pomiary](BENCHMARKS.md).

### Metryki Prometheus

Metryki są domyślnie wyłączone. `--metrics` lub `metrics: true` w konfiguracji
włącza rejestr, instrumentację HTTP i torów modelu oraz endpoint `GET /metrics`.
Bez włączenia rejestr i uchwyty metryk nie powstają, middleware nie jest
instalowane, a `/metrics` zwraca 404.

Po włączeniu `GET /metrics` eksportuje prywatny rejestr procesu w formacie tekstowym Prometheus
0.0.4 (`basal-cli/src/serve/metrics.rs`). Middleware obejmuje wyłącznie trasy
inferencji: mierzy czas od wejścia przed odczytem body do zbudowania odpowiedzi,
liczy statusy i aktywne handlery. Handler po parsowaniu przypisuje etykietę modelu
z listy obsługiwanych modeli; pozostałe żądania mają pustą etykietę.

Tory `main` i `long` mają uchwyty do histogramów kolejki i partii. Czas partii
jest obserwowany raz, a czas kolejki osobno dla każdego żądania uruchomionej
partii. Liczniki tokenów i pytań sumują `usage` poprawnie zbudowanych odpowiedzi
na wątku modelu, również gdy klient już nie czeka. Rejestr nie odczytuje GPU;
scrape nie przechodzi przez kolejkę inferencji. Szczegółowa semantyka i PromQL:
[Observability](OBSERVABILITY.md).

## Ograniczenia

- Score powyżej 10 poziomów i pytania z jedną opcją kończą się jawnym błędem.
- Tabela GEMM jest specyficzna dla karty i wersji cuBLASLt. Plik w
  `gemm_cache` jest nazwany modelem i precyzją, bez karty: przeniesienie
  modelu na GPU innego typu generuje tabelę od nowa i zastępuje poprzednią.
- `quantization: q8_0` działa tylko na CUDA z f16 i dodaje dekwantyzację
  wag na każdy forward. Na RTX 6000 Ada zmniejsza zajętość puli pamięci CUDA
  (basal-1.5-mini i 4.5B), wydłuża czas (4.5B) i zmienia część decyzji oraz
  granic evidence wobec FP32 ([raport](../reports/w8a16-2026-10-11/README.md));
  model max, inne karty i obciążenia nie zostały zmierzone.
- Repliki wymagają kart jednego typu i wspólnej tabeli GEMM niezależnej od
  partii; mieszane grupy wymagają osobnej decyzji numerycznej. Zgodność
  konfiguracji nie zastępuje pomiaru równości wyników na obu kartach.
- 4.5B i mini na dwóch RTX 6000 Ada: bitowa zgodność z wersją sprzed zmiany,
  przypisanie urządzeń i ruch równoległy sprawdzone
  ([raport](../reports/multigpu-placement-2026-10-11/README.md)). Nie zmierzono
  skalowania przepustowości ani pracy kart różnych architektur razem.
- Metal sprawdzony na M1 Pro (32 GB) z basal-1.0-4.5B i trzema modelami
  basal-1.5 ([pomiar](../reports/metal-m1-pro-1.5/README.md)) oraz na M2 Max
  (32 GB) z trzema modelami basal-1.5
  ([pomiar](../reports/metal-m2-max-1.5/README.md),
  [attention](../reports/metal-m2-max-tree/README.md)).
