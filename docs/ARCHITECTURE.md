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

Nowe rodziny, format 2, wybór kolejności i kalibracja: [UPSTREAM_2.0.md](UPSTREAM_2.0.md).
Natywne architektury i lifecycle mediów omni: [OMNI.md](OMNI.md).

Opcjonalna sekcja `database:` i moduł `basal-cli/src/database/` posiadają konfigurację
bazy, opcje połączeń, wybór konta oraz moment migracji schematu
([DATABASE.md](DATABASE.md)). Bez tej sekcji i bez konsumenta serwer nie otwiera bazy.
Opcjonalna usługa batchy w `basal-cli/src/serve/batches/` ma osobne planowanie CPU,
trwały magazyn metadanych (w bazie z `database:` albo lokalnym domyślnym SQLite)
oraz artefakty FS/S3. `store::Store` udostępnia typowane
operacje domenowe: SQLite, PostgreSQL i SQL Server korzystają ze wspólnych operacji
z jawnym dialektem SQL, a MongoDB z natywnych transakcji replica set. Lease,
generation i ledger operacji chronią zapis wyników po przejęciu pracy; nie zastępują
wyłączności procesu na fizycznym GPU.

Dedykowany wątek tła posiada osobny fork silnika. `WorkClass`, `StopToken`
i bramki urządzeń w `basal-core` zapewniają kooperacyjne ustępowanie online
między bezpiecznymi checkpointami. Ogłoszenie `OnlineDemand` obejmuje cały lifecycle
żądania online i wszystkie urządzenia instancji. Rezerwacje hosta i urządzeń są
przenoszone razem z pracą; niepotwierdzone cleanup urządzenia blokuje dalsze użycie.
Licznik `StopToken::device_time` mierzy czas posiadania bramki, nie sprzętowy czas
aktywności kerneli. Konfiguracja, kontrakt, granice kwalifikacji i procedury:
[BATCHES.md](BATCHES.md). Wykonanie tła pozostaje domyślnie wyłączone.

Loader Gemma czyta kolejne wagi do jednego ponownie używanego bufora. BF16
przesyła bez pośredniego własnego tensora CPU; przed nadpisaniem bufora czeka
na zakończenie transferu. F16 zachowuje konwersję na GPU przy wystarczającym
zapasie VRAM oraz konwersję CPU przy małym zapasie. Tryb obliczeń F32 nadal
przechowuje wagi BF16 i rozszerza je przed użyciem. Bufor znika przed gotowością
backendu; nie należy do stanu inferencji.

Gemma grupuje krótkie teksty w kubełki 32 tokenów (do 32 promptów / 8192 tokenów).
Projekcje współdzielą GEMM wyłącznie dla kształtów objętych tabelą z bitowo zgodnymi
redukcjami między klasami M; pozostałe zachowują osobne GEMM. Attention domyślnie
wykonuje prompty oddzielnie. CUDA wylicza maskę pozycyjną w softmaxie FP32,
zamiast przesyłać gęstą maskę CPU dla każdej warstwy. Kafel wyniku scores ma limit
128 MiB; nie jest to limit całej pamięci roboczej. Długie prompty zachowują unikalne
K/V GQA. Prefill ma kroki 2048 tokenów i respektuje granice bloków obrazu.

Dla `basal-omni-1.0-ultra`, FP16 z odczytem FP32, tryb `auto` na NVIDIA H100 PCIe
łączy niezależne macierze attention partii, zachowując operandy, iloczyny
i softmax FP32. Jeden kernel układa Q/K/V, powtarza głowy GQA i dokładnie
rozszerza FP16 do FP32, usuwając pośrednie kopie. Sekwencje wyrównuje do
8 tokenów; kubełki 32 i limity partii
pozostają takie same. Bezpośredni RMSNorm używa redukcji FP32, a połączone
RoPE, GELU z mnożeniem oraz skalar warstwy zachowują punkty zaokrąglenia FP16.
Nieafiniczna norma V zmienia sposób redukcji i wymaga porównania z FP32.
K/V warstw bez współdzielenia są zachowywane tylko dla następnego fragmentu
prefillu. `accurate` i pozostałe konfiguracje zachowują poprzednią ścieżkę.
Zakres zgodności i pomiary: [raport Ultra](../reports/ultra-attention-2026-10-17/README.md).

Stałe obroty pozycyjne obrazu i indeksy głów są współdzielone w swoim zakresie
użycia. Pooling obrazu zachowuje kolejność 3×3 i redukcję FP32. Procesor CPU
współdzieli amplitudy FFT między filtrami mel oraz zaokrąglone poziome wiersze
skalowania obrazu; ten drugi cache ma limit 32 MiB i zachowuje poprzednią ścieżkę
po jego przekroczeniu. Te zmiany nie wprowadzają cache żądań ani mediów między
żądaniami. Numeryczne zmiany attention, RMSNorm i Conformera wymagają osobnego
porównania z upstream FP32, opisanego w raportach.

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

W format 1 pytanie domyślnie ma dwa prompty z opcjami w dwóch porządkach.
Format 2 domyślnie wykonuje jeden; `--orders` może zmienić tę politykę. Z logitów liter
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

Wiersz ma do 32768 tokenów, forward domyślnie do 8192 (wiersze dłuższe liczą się
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
- Tryb `inference_mode` jest wybierany dla każdego backendu przed wyliczeniem
  prefiksów i kopiowany do torów wykonania. Dla CUDA Llama `accurate` używa
  attention hi/lo przez przenośny kernel `attn_tree_tc`, również na Hopperze;
  automatyczny dobór WGMMA na Hopperze pozostaje w `auto`. Jawny `accurate`
  nie gwarantuje więc wydajności domyślnej ścieżki H100. `balanced` wybiera specjalizację QK1 dla f16
  z odczytem f32: Q/K w f16, prawdopodobieństwa hi/lo, softmax i akumulacja
  w f32. Specjalizacja wymaga dokładnego V w f16 i scalonego wyniku f16;
  przy innych typach kernel wraca do odpowiadającego mu wariantu ogólnego.
  Na CUDA jawny tryb ma pierwszeństwo przed `BASAL_ATT`; `auto` zachowuje tę zmienną
  do eksperymentów. Wybór nie modyfikuje środowiska procesu ani gotowych cache K/V.
  Automatyczny QK1 dotyczy wyłącznie rozpoznanej RTX 6000 Ada Generation,
  nazw Basal 2.0 mini/4.5B/max oraz CUDA f16 fused z odczytem f32; pozostałe
  konfiguracje zachowują poprzedni wybór. Nieudany opcjonalny odczyt nazwy GPU
  również zachowuje poprzednią politykę.
  [Pomiary i ograniczenia jakości](../reports/precision-performance-ada-2026-10-16/README.md)
  nie są gwarancją zgodności każdego wejścia z FP32.
- `auto` na H100 PCIe dla Ultra w FP16 łączy niezależne macierze attention
  przykładów i głów, zachowując produkty i softmax FP32. RMSNorm z wagami używa
  bezpośrednio kernela Candle z redukcją i mnożeniem FP32 oraz końcowym zapisem
  FP16. Osobny kernel RoPE zachowuje zaokrąglenia FP16 obu mnożeń i ich sumy.
  Model bez współdzielenia K/V między warstwami przechowuje je tylko dla
  rzeczywistego kolejnego chunku. Kubełki długości co 32 tokeny i budżet 8192
  pozostają bez zmian. `accurate` zachowuje poprzednie kernele i FP32 attention;
  inne pary model/karta zachowują poprzednią politykę.
  [Zakres pomiarów i ograniczenia](../reports/ultra-attention-2026-10-17/README.md).
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
`Layer` typ `ProjWeight`: `Dense(Tensor)` albo `Quantized(Arc<QTensor>)`
(typ bloków niesie `QTensor`; ten sam wariant dla `q4_k`). Loader
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
kończącą ładowanie. W forwardzie `matmul_t` dla `Quantized` tworzy przejściowy
tensor f16 macierzy (`dequantize_block_q8_0_f16`), podaje go temu samemu GEMM
cuBLASLt f16 i zwalnia go po wywołaniu, na strumieniu wątku silnika. Tor
`Dense` jest kodem sprzed zmiany; jedyną różnicą jest dopasowanie wariantu.
Forki silnika (tor długich żądań) współdzielą bloki przez `Arc`, a repliki
ładują własne. `basal profile` raportuje czas dekwantyzacji osobno
(`q8_dequant_ms_per_item`, część sekcji projekcji, nie dodawana do
`sections_sum_ms`). `QMatMul` Candle nie jest używany (kwantyzuje aktywacje
do Q8_1). Własny kernel W8A16 jest poza zakresem.

### Kwantyzacja W4A16 (`q4_k`)

Ta sama ścieżka co `q8_0` (walidacja, zakres tensorów, `ProjWeight::Quantized`,
urządzenie, strumienie, forki i repliki, przejściowy f16 przed tym samym GEMM
cuBLASLt), z blokami GGML Q4_K (`BlockQ4K`, 256 wartości wzdłuż K, 144 B,
padding Candle 288 B na macierz). Loader generyczny względem typu bloku
(`load_blocks`) wykonuje dla `q8_0` te same wywołania w tej samej kolejności;
tor hosta Q8 (`quantize_q8_0`) jest bez zmian. Dla `q4_k` (`quantize_q4_k`)
sprawdza przed pracą dodatnie wymiary, `K % 256 == 0`, przepełnienia,
`N·K ≤ i32::MAX` i długość danych, a dla każdego wiersza: skończone wartości
wejścia, brak stałych niezerowych grup 32 wartości (przypięty
`make_qkx1_quants` zwraca dla nich `(0, 0)`, więc rekonstruowałyby się do
zer), długości przed `from_float` i `to_float` (Candle sprawdza je tylko w
buildzie debug), skończone `to_float` i rekonstrukcję f16 (RNE) przed
statystykami i wysłaniem. Błąd wskazuje tensor checkpointu, jego wiersz i
kolumnę (dla q|k|v i gate|up pierwotny tensor) oraz liczby NaN, +inf i −inf
w wierszu. Statystyki f64 (`Σe²`, `Σw²`, `max|e|`, grupy zerowe i
spłaszczone, bez progów) są liczone w stałych kawałkach 16 wierszy,
zapisywane pod indeksem kawałka i redukowane w kolejności kawałków, macierzy
i warstw (q|k|v, o, gate|up, down) — wynik nie zależy od liczby wątków
(co najwyżej `available_parallelism`, każdy z ciągłym zakresem kawałków,
uruchamiane przez `Builder::spawn_scoped` z łączeniem wszystkich przed
zwróceniem błędu). Ta kolejność i wyrażenia odpowiadają diagnostyce P0.
W S1 pełny obiekt `quantization` zapisało 14 z 21 ładowań i jego statystyki
zbiorcze (model i cztery typy projekcji) są równe P0; pozostałe 7 zapisało
tylko linię logu ([raport](../reports/w4a16-s1-2026-10-12/README.md)).
Kernel `dequantize_block_q4_K_f16` nie ma argumentu długości ani kontroli
zakresu, więc przed pierwszym uruchomieniem loader sprawdza na wysłanym
tensorze typ Q4K, kształt, `K % 256`, `N·K ≤ i32::MAX`, liczbę bloków i bajty
(`check_q4k_resident`). Warm-up dekwantyzuje na GPU najmniejszą według
`projection_shapes` macierz warstwy 0 (jej bloki hosta są zachowane tylko do
tej kontroli), kopiuje wynik f16 do hosta i porównuje wzorce bitowe (także znak
zera) z `f16(to_float)` tych bloków; różnica jest błędem ładowania, bez
fallbacku. Kontrola obejmuje jedną macierz, nie wszystkie. `describe` dla
`q4_k` podaje obiekt `quantization` z błędem rekonstrukcji modelu i typów
projekcji (deterministyczny: bez czasów i liczby wątków); `basal profile`
podaje `dequant_ms_per_item`, `dequant_values_per_item`, `dequant_format` i
`dequant_scope` (ten sam zakres co klucze `q8_dequant_*`).

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
`--max-large-inflight` / `max_large_inflight` ogranicza pulę rozwiniętych
promptów powyżej 128 KiB. W serwerze obsługującym media ta sama pula obejmuje
także body powyżej 2 MiB lub bez Content-Length.
Domyślnie są to 4 żądania; zwiększenie wymaga pamięci na równoczesne przygotowanie
i kolejkę. Budżety pojedynczego żądania nadal obowiązują. W pomiarze przepustowości
limit przyjęcia powinien mieć zapas ponad liczbę klientów; odrzuceń 529 nie wolno
zaliczać do poprawnych odpowiedzi/s. Osobno porównujemy przepustowość oraz
czas request→response wraz z oczekiwaniem w kolejce.

Budżet `max_batch_tokens` każdego obsługiwanego modelu ustawia też
`Engine::forward_max_tokens` przed utworzeniem obu lane'ów. Zwiększenie limitu
pozwala więc łączyć wiele spakowanych wierszy jednej grupy językowej w większy
forward. Pojedynczy wiersz przekraczający budżet nadal może zostać wykonany osobno.
Domyślna wartość obu budżetów wynosi 8192. Większe wartości zmieniają jednoczesne
użycie pamięci i czas zajęcia GPU; ich ocenę wykonujemy razem z opóźnieniami
krótkich żądań w ruchu mieszanym.


Żądania powyżej `--long-tokens` liczy drugi tor: drugi silnik na tym samym
GPU, ze wspólnymi wagami, tabelą GEMM i prefiksami szablonu. Oba tory
korzystają z GPU na zmianę (`basal_core::gate`): tor długich żądań po każdej
warstwie czeka na zakończenie swoich kerneli i oddaje GPU, gdy główny tor ma
partię, a on sam pracował co najmniej `--long-slice-ms`. Po jednej partii
głównego toru wraca do tej samej warstwy. Zatrzymany forward kontynuuje z tymi
samymi tensorami. Parametr przydziału steruje kompromisem między czasem
dużych dokumentów a oczekiwaniem krótkich żądań; większa wartość oznacza
rzadsze oddawanie GPU. Niezmienność wyników potwierdzamy pomiarami danej konfiguracji.

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
Metal i Intel obsługują tylko GPU 0. Podział jednego modelu warstwami między
karty CUDA opisuje sekcja [Podział warstw między GPU](#podział-warstw-między-gpu-cuda).

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
`long` tego nie zmienia, bo tor główny przejmuje jego żądania. Backend CUDA
raportuje `DeviceHealth` linii modelu (wspólne dla jej forków, np. toru
długiego). Linia jest oznaczana jako utracona, gdy nie da się potwierdzić
końca jej własnej pracy GPU: dwie nieudane synchronizacje przed zwolnieniem
gate'u albo po błędzie żądania, albo dwa nieudane wiązania kontekstu przed
zwolnieniem pamięci. Kolejne wywołania tej linii są odrzucane przed pracą
GPU, a trwające wywołanie toru długiego, które oddało gate w `checkpoint`,
kończy się tym samym błędem zaraz po jego odzyskaniu. Pamięć, której koniec
użycia nie jest potwierdzony, jest celowo zatrzymana zamiast zwolniona. Zwykły OOM, błąd żądania i timeout nie
oznaczają utraty. To nie jest wykrywanie awarii sprzętu: fatalny błąd CUDA
może wymagać restartu całego procesu, także dla innych kart, więc repliki
nie są izolacją awarii sprzętu. `/health` dla modeli z jedną
instancją zachowuje dotychczasową treść i reguły (`device_lost`,
`lost_models`). Grupa z częścią niedostępnych replik daje 200 ze
`status: ok` i `degraded_models`, grupa bez dostępnych replik 503 ze
`status: engine_stopped` i `unavailable_models`; wpis zawiera `model`,
`replicas`, `available` i `unavailable` (`replica`, `device`, `reason`:
`worker_closed` albo `device_lost`).

### Podział warstw między GPU (CUDA)

`layer_split` (YAML) i `--layer-split` (CLI) opisuje `basal_gpu::LayerSplit`
(`crates/basal-gpu/src/split.rs`): etapy `{device, layers}` w kolejności
warstw. Konstruktor wymaga co najmniej dwóch etapów, co najmniej jednej
warstwy na etap i różnych kart. Zakres etapu `s` wynika z liczb warstw,
`[Σ_{t<s} layers_t, Σ_{t≤s} layers_t)`, więc luki i nakładanie się są
niemożliwe. Kolejność walidacji: parsowanie (clap/serde, nieznane pola
odrzucone), `ModelConfig::placement` (konflikt z `device` i
`replica_devices`; w CLI także `conflicts_with` dla `--device`,
`--replica-devices` i `--config`), `LayerSplit::check` (backend CUDA, `f16`,
kernele `fused`), `check_same_gpus` (każdy ordinal istnieje w procesie,
jeden typ karty; sterownik bez tworzenia kontekstu), pokrycie liczby warstw
z `config.json`, dopiero potem wagi. Serwer i `basal benchmark` rozwiązują
tabelę GEMM na karcie etapu 0 i przed ładowaniem sprawdzają ją na każdej
karcie podziału; pozostałe polecenia wczytują `--gemm-table` jak dla jednego
GPU. Model z podziałem jest w
serwerze jedną instancją (`Replica.devices` z ordinalami etapów); repliki
takiego modelu nie są obsługiwane.

`GpuBackend::load_split` tworzy dla każdego etapu urządzenie
`Device::new_cuda(ordinal)` z własnymi wagami i uchwytem cuBLASLt z
workspace. Etap 0 trzyma też embedding, ostatni końcową normę, `lm_head` i
głowicę evidence. Jedyna kopia wag to `input_layernorm` pierwszej warstwy
następnego etapu, trzymana na etapie przed granicą (`hidden` wartości f16).
Wybór PTX, kernela attention i liczba SM są i tak zapamiętywane per GPU.
Błąd ładowania ma kontekst etapu, karty i warstwy, a już załadowane etapy są
zwalniane, każdy przy własnym kontekście.

Forward (`forward_ragged`, ścieżka `fused`) liczy etapy po kolei. Na
granicy źródło jest opróżniane, bufor celu `[2, M, H]` f16 alokowany i
inicjalizowany na następnej karcie, potem następuje kopia API
urządzenie–urządzenie, potwierdzenie zakończenia na obu kartach, zwolnienie
źródła i wiązanie kontekstu celu. Peer access nie jest włączany, a fizyczna
trasa kopii (P2P albo przez hosta w sterowniku) pozostaje nieustalona. K/V
prefiksów szablonu i cache stanu warstwy `l` leżą na karcie warstwy `l`
(część etapu). Stan końcowy, logity liter, odczyt evidence i ich pośrednie
tensory istnieją tylko na ostatnim etapie i trafiają na hosta; logity nie
wracają na etap 0. Pamięć urządzenia etapu (`OnStage`) jest zwalniana przy
związanym kontekście swojej karty. Gdy wiązania nie da się potwierdzić albo
linia modelu jest utracona, pamięć zostaje celowo zatrzymana zamiast
zwolniona. Bufory granicy z niepotwierdzoną kopią trafiają do kwarantanny
linii i oznaczają ją jako utraconą (`device_lost`, wyżej).

Gate'y powstają z listy instancji; instancja z podziałem liczy się na każdej
swojej karcie, a model z torem długim dostaje gate każdej karty. Wątek trzyma
najwyżej jeden gate naraz i nie czeka na gate, trzymając inny, więc
oczekiwanie między kartami nie tworzy zakleszczenia. Przed zwolnieniem gate'u
etapu jego praca jest opróżniana. Modele, które dzielą kartę z etapem,
korzystają z niej na zmianę z tym etapem, jak modele jednego GPU. Każdy etap
ma własne wagi, K/V i workspace; strumienie są per wątek i kontekst. Tor
główny wchodzi do gate'u każdego etapu jako pilny. Tor długi wznawia
rozpoczęte zadanie na granicy przez `Gate::resume`: bilet w kolejce FIFO,
której nowe żądania pilne ani wywłaszczalne nie wyprzedzają. Czeka więc
najwyżej na resztę bieżącej tury, po jednej turze każdego wcześniejszego
biletu i po jednej turze pilnej na każde oddanie w `checkpoint`. To
ograniczenie wynika z protokołu i przeglądu kodu, a nie z pomiaru. Kolejność
między torami pilnymi jednej karty pozostaje kolejnością `Condvar`, bez
nowej gwarancji sprawiedliwości. Fork toru długiego współdzieli wagi i
`Lt` etapów, liczniki transferów i `DeviceHealth`, ma własny pusty cache
stanu i własne gate'y ustawiane przez serwer (`set_device_gate`); fork
utraconej linii zwraca błąd.

Liczniki transferów istnieją tylko przy podziale: na granicę liczba kopii,
bajty, czas hosta kopii z jej synchronizacjami i czas czekania na gate
następnej karty. Trafiają do `basal profile` i `basal benchmark`, a opis
backendu podaje układ etapów i stały opis protokołu (`stage_transfer`), bez
liczb. Metryk Prometheus dla transferów nie ma
([Observability](OBSERVABILITY.md#podział-warstw)). Dla jednego GPU opis
backendu, manifest eksportu, raporty i odpowiedzi zachowują dotychczasowy
format.

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
urządzenia silnika (przy podziale warstw: do urządzenia każdego etapu).
Metodyka i ograniczenia: [Pomiary](BENCHMARKS.md).

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

Obsługiwane kombinacje architektur i opcji rozmieszczenia (CUDA). Niewspierane są odrzucane
jawnym błędem po odczytaniu `config.json`, przed tabelą GEMM i ładowaniem wag:

| Model | `device` | `replica_devices` | `layer_split` | `quantization` `q8_0` / `q4_k` | `inference_mode` |
|---|---|---|---|---|---|
| basal-1.5 mini/4.5B/max (Llama) | tak | tak, wspólna tabela niezależna od partii | tak, `f16` + `fused` | tak, `f16` + `fused` | `auto`/`accurate`/`balanced` |
| Basal 2.0 mini/4.5B/max (Llama) | tak | tak, wspólna tabela niezależna od partii | tak, `f16` + `fused` | tak, `f16` + `fused`; `q4_k` wymaga `K % 256 == 0` | `auto` wybiera `balanced` na RTX 6000 Ada |
| Omni 1.0 mini/max, Omni 1.1 max i Ultra (Gemma 4) | tak | tak; bez pasującej tabeli stałe GEMM per prompt | Gemma bez shared KV/PLE, jawne `f16` + `fused`; tekst i obrazy | nie (odmowa, tylko `none`) | `auto`/`accurate`; `balanced` odmowa |

Podział Gemma ładuje kolejne zakresy warstw na oddzielne karty. Enkodery mediów i
embedding wejściowy są na pierwszej; końcowa norma oraz powiązany embedding
odczytu na ostatniej. Powiązany embedding ma dwie rezydentne kopie. Cache KV,
rotary, mapy GQA, maski obrazu i uchwyty cuBLASLt należą do karty etapu.
Granice korzystają z tej samej synchronizacji, kopiowania i retencji po błędzie co
Llama. Jedno wywołanie trzyma najwyżej gate jednej karty, a health jest współdzielone
przez wszystkie etapy i forki. Modele z cross-layer KV lub PLE są jawnie odrzucane.

[Kampania Ultra](../reports/ultra-multigpu-2026-10-18/README.md) potwierdza podział
tekstu i obrazów na dwóch RTX 6000 Ada, bitową kontrolę umieszczenia na H100 oraz
skalowanie jednej do dwóch replik H100. Zachowuje FAIL jednej decyzji względem FP32
na Ada i 226 różnic liczbowych replik przy C128 bez zmian decyzji. Nie obejmuje
pozostałych modeli Omni, audio ani trzech lub większej liczby kart.
Kombinacje Basal 2.0 z umieszczeniem na GPU innym niż 0, replikami, podziałem warstw
lub kwantyzacją oraz `auto`/`balanced` razem z kwantyzacją lub podziałem są złożeniem
mechanizmów sprawdzonych osobno; po integracji nie mają jeszcze własnego pomiaru
zgodności ani wydajności na sprzęcie. Wyniki niżej dotyczą basal-1.5.
`auto` wybiera `balanced` dla podziału Llama tylko, gdy każda karta etapu jest RTX 6000 Ada.

- Score powyżej 10 poziomów i pytania z jedną opcją kończą się jawnym błędem.
- Tabela GEMM jest specyficzna dla karty i wersji cuBLASLt. Plik w
  `gemm_cache` jest nazwany modelem i precyzją, bez karty: przeniesienie
  modelu na GPU innego typu generuje tabelę od nowa i zastępuje poprzednią.
- `quantization: q8_0` działa tylko na CUDA z f16 i dodaje dekwantyzację
  wag na każdy forward. Na RTX 6000 Ada zmniejsza zajętość puli pamięci CUDA
  (basal-1.5-mini i 4.5B), wydłuża czas (4.5B) i zmienia część decyzji oraz
  granic evidence wobec FP32 ([raport](../reports/w8a16-2026-10-11/README.md));
  model max, inne karty i obciążenia nie zostały zmierzone.
- `quantization: q4_k` ma te same ograniczenia; dodatkowo wymaga
  `K % 256 == 0` każdej projekcji. To opcjonalny tryb eksperymentalny,
  włączany jawnie; domyślne i zalecane jest `none`. Na RTX 6000 Ada
  (basal-1.5-mini i 4.5B) nie spełnia kryterium FP32 i zmienia decyzje oraz
  granice evidence wyraźnie bardziej niż `q8_0`
  ([raport](../reports/w4a16-s1-2026-10-12/README.md)). Bloki Q4_K są przed
  każdym GEMM dekwantyzowane do f16; nie ma kernela int4 ani rdzeni
  tensorowych int4, a 4 bity dotyczą tylko wag czterech projekcji. Dla 4.5B
  na GPU 0 jednej RTX 6000 Ada zmniejsza pulę alokatora po załadowaniu do
  2 996 968 960 B (f16: 9 644 216 320 B, `q8_0`: 5 309 140 480 B), ale wobec
  f16 obniża żądania/s o około 33% sekwencyjnie i 7% przy 16 klientach
  ([S2](../reports/w4a16-s2-2026-10-12/README.md)); wobec `q8_0` różnice
  żądań/s wynoszą +0,43–1,70%
  ([S3](../reports/w4a16-s3-2026-10-12/README.md)). To opis dwóch osobnych
  sesji, bez kryterium szybkości i ustalonej przyczyny. Pulę mini zmierzono w
  jednym przebiegu, bez szybkości żądań. Czas kwantyzacji na hoście, model max
  i inne karty nie zostały zmierzone.
- Repliki wymagają kart jednego typu i wspólnej tabeli GEMM niezależnej od
  partii; mieszane grupy wymagają osobnej decyzji numerycznej. Zgodność
  konfiguracji nie zastępuje pomiaru równości wyników na obu kartach.
- Podział warstw działa tylko na CUDA z `f16` i kernelami `fused`, na
  kartach jednego typu; etapy liczą po kolei. Na dwóch RTX 6000 Ada (mini i
  4.5B) wyniki z podziałem są bitowo równe jednemu GPU, a pod sztucznym
  limitem wolnej pamięci podział uruchamia 4.5B, którego jeden GPU nie
  mieści ([raport L1](../reports/multigpu-layer-l1-revalidation-2026-10-13/README.md)
  bieżących źródeł). Wydajność 4.5B zmierzono opisowo, po jednej sesji na
  tryb; sesje miały tę samą konfigurację kart, ale ich fizyczna tożsamość
  nie jest ustalona
  ([zestawienie](../reports/multigpu-layer-revalidation-2026-10-14/README.md)):
  jedno żądanie jest z podziałem wolniejsze, a 16 klientów `none` dało
  1,17–1,18× przepustowość jednego GPU przy innych warunkach termicznych
  kart. Tor jednego GPU ma dwa utrzymane sygnały bez wskazanej przyczyny w
  kodzie (`none` `mixed-c1 queue_ms.p99`, `q4_k` `dec_s`), przyjęte jako
  konkretne wyniki, bez progu. Modelu max, innych kart, kart mieszanych ani
  fizycznej trasy kopii nie zmierzono. Pomiary poprzedniego kandydata
  ([zestawienie](../reports/multigpu-layer-2026-10-13/README.md),
  [poprawka zakresu](../reports/multigpu-layer-2026-10-13/SCOPE.md))
  pozostają historią.
- 4.5B i mini na dwóch RTX 6000 Ada: bitowa zgodność z wersją sprzed zmiany,
  przypisanie urządzeń i ruch równoległy sprawdzone
  ([raport](../reports/multigpu-placement-2026-10-11/README.md)). Nie zmierzono
  skalowania przepustowości ani pracy kart różnych architektur razem.
- Metal sprawdzony na M1 Pro (32 GB) z basal-1.0-4.5B i trzema modelami
  basal-1.5 ([pomiar](../reports/metal-m1-pro-1.5/README.md)) oraz na M2 Max
  (32 GB) z trzema modelami basal-1.5
  ([pomiar](../reports/metal-m2-max-1.5/README.md),
  [attention](../reports/metal-m2-max-tree/README.md)).

## Warstwa zadań 2.1

`basal-core::tasks` rozwija classify/verify/compress/extract do zwykłych decyzji.
Plan przechowuje kontekst składania odpowiedzi; extract wykonuje co najwyżej
jedną zależną kontynuację. `Engine::run_plans` grupuje te kontynuacje na tym
samym engine, bez ponownego routingu HTTP. Job zachowuje sloty i rezerwację
repliki przez obie rundy. Plain requests pozostają na dotychczasowej ścieżce.
Budget sprawdza rozwinięty tekst i gałęzie przed ekspansją. Osobne task_thresholds
nie są zastępowane kalibracją prymitywów. Szczegóły: [TASKS.md](TASKS.md).
