# Deklaracja GGUF, companions i częściowa inspekcja źródła

`basal_core::weights::GgufCompanionSnapshot::read(&Path)` odczytuje lokalną
deklarację `gguf-source.json` i pięć stałych plików towarzyszących. Zwraca
sprawdzoną deklarację pochodzenia oraz dokładne bajty plików, których SHA-256
zgadza się z deklaracją. API jest niezależne od backendu.

To ograniczony etap przygotowania źródła. Nie otwiera ani nie parsuje GGUF,
nie wybiera źródła wag w CLI lub runtime i nie dopuszcza modelu do wykonania.
`declared_gguf_sha256()` jest jawnie **niezweryfikowaną deklaracją**.

`basal_core::weights::GgufCompanionSemantics::read(directory, options)` dodatkowo
buduje wspólny `ModelManifest` i `BasalTokenizer` z zachowanych bajtów oraz
zachowuje pliki faktycznie odczytane przez dotychczasowy wybór kalibracji.
`options` ma typ `ModelLoadOptions<'_>`. Wynik posiada snapshot, manifest,
tokenizer i kalibrację; gettery `snapshot()`, `manifest()`, `tokenizer()` oraz
`calibration()` pożyczają rzeczywiste obiekty. Nie klonują całych buforów ani
nie otwierają companions ponownie. Cache tokenizera może być później używany
przez dotychczasowe metody. API nadal nie jest loaderem GGUF ani planem gotowym
do alokacji na urządzeniu.

## Opcjonalny preflight oryginalnego JSON

`GgufCompanionSemantics::read_with_json_envelope(directory, options, lease)`
zwraca ten sam typ właściciela przez `Result<Self, GgufInventoryReadError>`.
`lease` to `CallerCalibrationDirectoryLease` z opisanym niżej obowiązkiem
wykluczenia zapisów. Konstruktor najpierw zachowuje pełny ograniczony inwentarz
kalibracji, następnie wykonuje wspólny odczyt semantyki, cross-binding, kontrolę
pierwotnego stampu katalogu i instalację inwentarza. Nie ma operacji nadawania
tego trybu wcześniej skonstruowanemu właścicielowi ani znacznika gotowości.

Ta ścieżka oraz opisany niżej `read_with_raw_source_budget` sprawdzają oryginalne zachowane bajty JSON przed ich
pierwszym parsowaniem przez serde. Obejmuje `gguf-source.json`, `config.json`,
`basal.json`, `tokenizer.json`, `tokenizer_config.json` i najwyżej dwa faktyczne
odczyty kalibracji: kandydata i zamiennik albo plik jawny, także zewnętrzny.
Zamiennik jest sprawdzany przed konstrukcją jego Value, gdy Value kandydata
może nadal istnieć. Nie ma ponownego odczytu, normalizacji ani serializacji
bajtów do tej kontroli. `chat_template.jinja` i nieodczytane przez selektor
warianty inwentarza pozostają poza skanowaniem JSON.

Stałe limity są liczone osobno dla każdego dokumentu; równość jest dozwolona:

| Zasób | Limit i sposób liczenia |
|---|---|
| Otwarte kontenery | 64 obiekty/tablice jednocześnie; kontener główny ma głębokość 1 |
| Wartości i klucze razem | 1 048 576; każda wartość, w tym kontener, raz, każdy klucz osobno, także duplikaty i później ignorowane pola |
| Surowy string | 1 MiB na ciało, 32 MiB sumy ciał kluczy i wartości; bez otaczających cudzysłowów, ze wszystkimi bajtami escape |
| Surowa liczba | 128 bajtów na token, 8 MiB sumy tokenów; ze znakiem, kropką i wykładnikiem, bez konwersji liczbowej |

Dotychczasowe limity surowych plików i sum pozostają bez zmian. Skaner pożycza
wejście, używa 64 stałych ramek i sprawdzanych liczników, bez heapu, rekurencji,
AST, kopii tokenów ani serde. Sprawdza UTF-8, JSON whitespace, pełną gramatykę
liczb, escape i pary surogatów, dopasowanie kontenerów i dokładnie jeden root
bez niebiałego sufiksu. Root skalarny przechodzi preflight; jego dopuszczalność
nadal określają serde i schema. Kontrola nie narzuca unikalności kluczy.

To jawnie węższa polityka wejścia, bez pomiaru pokrycia pakietów. Istniejące
`GgufCompanionSnapshot::read`, `GgufCompanionSemantics::read` oraz
`read_with_calibration_inventory` wybierają prywatny tryb `Legacy`: nie skanują
JSON i zachowują dotychczasową kolejność odczytów, parserów i błędów. Dla nowej
ścieżki preflight deklaracji poprzedza DTO; kontrola każdego stałego companion
następuje po jego porównaniu digestu, przed odczytem następnego pliku. Może więc
odmówić przed późniejszym błędem pliku lub walidacji semantycznej. Również błędy
leksykalne kalibracji mogą wyprzedzić dotychczasowy błąd serde. Poprawne wejścia
w limitach trafiają jako te same bajty do istniejących parserów, normalizacji,
tokenizera, selektora, kontroli ról i wiązania kalibracji.

Publiczne `JsonEnvelopeResource` rozróżnia Depth, NodesAndKeys, StringBytes,
TotalStringBytes, NumberBytes i TotalNumberBytes. `JsonEnvelopeError` ma
Resource (zasób, zerowy offset bajtowy usize, limit u64), Malformed (offset,
statyczna przyczyna) i Arithmetic (offset). Nie przechowuje wejścia ani ścieżki.
Dla deklaracji i companions łańcuch `source()` prowadzi przez
`GgufInventoryReadError::Semantic`, `GgufSemanticError::Capture` oraz
`WeightSourceError::JsonEnvelope { role, source }`. Dla rzeczywistych odczytów
kalibracji prowadzi przez `GgufSemanticError::Semantic` ze stage
CalibrationSelection i zachowany błąd anyhow do
`GgufSemanticError::CalibrationJsonEnvelope`, następnie `JsonEnvelopeError`.
Nie ma zamiany przyczyn na tekst. Mapowanie open NotFound config do
`MissingConfigG3` pozostaje bez zmian.

**Tylko sam skaner działa bez alokacji.** W `read_with_json_envelope` surowe odczyty, inwentarz, zwykła
propagacja błędów, serde, normalizacja i backend nadal nie mają utrzymywanych
rezerwacji pamięci. Limity leksykalne nie dowodzą rozmiarów capacity, scratch,
serializacji, helper Content, map, automatów, natywnego regex, głębokości
destrukcji trie Unigram ani czasu życia cache BPE TLS. Nie ograniczają RSS,
narzutu allocatorów czy danych ścieżek systemu operacyjnego i nie chronią przed
OOM. API nie daje permitu, admission ani gotowości loadera. REQ-009 pozostaje
częściowe; nie ma dowodu runtime, zgodności bitowej/FP32, jakości, pamięci,
wydajności ani pokrycia pakietów.

## Opcjonalne utrzymywane admission surowych źródeł

`GgufCompanionSemantics::read_with_raw_source_budget(directory, options, lease, &budget)`
zwraca `Result<Self, GgufInventoryReadError>`. Łączy pełny inwentarz i preflight
Envelope z rzeczywistym admission **żądanego layoutu oryginalnych buforów raw**
przed każdą ich alokacją. Nie ma konwersji wcześniej utworzonego właściciela.
Obowiązki trusted-local lease, no-writer i wiązania źródeł pozostają bez zmian.

Aplikacja tworzy jeden `GgufRawSourceBudget::new(limit_bytes: usize)` przed
odczytami i przekazuje ten sam uchwyt albo jego klony do wszystkich równoległych
odczytów i reloadów w ramach tego allowance. Clone współdzieli ledger przez
Arc; oddzielne `new` tworzą niezależne limity. Limit zero odmawia niepustych
buforów; równość jest dozwolona. CLI ani Engine nie dostarczają jeszcze tego
kontekstu. Osobny budżet na każdy load nie oznacza admission całego procesu.
Bootstrap Arc/Mutex jest poza rachunkiem i nie ma obietnicy fallible konstrukcji.

| Właściciel utrzymujący prywatną rezerwację | Pokryte bufory |
|---|---|
| Snapshot | Oryginalna deklaracja i pięć companions, także chat template |
| Inventory | Zawartość każdego pasującego wpisu, także nigdy nieparsowanego |
| CalibrationCapture | Oba faktyczne odczyty selektora, w tym zastąpiony kandydat albo zewnętrzny plik jawny |

Kopie inventory i faktycznych odczytów pozostają oddzielnymi alokacjami i obie
obciążają ten sam ledger, nawet przy identycznych bajtach. Dotychczasowe limity
plików i sum nadal obowiązują. Admission jest przyrostowe, osobno przed każdym
buforem; późniejsza odmowa niszczy wcześniejsze bufory i ich właścicieli.

Po istniejących kontrolach pliku/długości i konwersji usize następuje walidacja
`Layout::array::<u8>(n)`, checked dodawanie do prywatnej rezerwacji i wspólnego
licznika oraz porównanie z limitem. Admission używa `Mutex::try_lock`, bez
alokacji, I/O ani parsowania pod lockiem. Następnie dla n>0 globalne `alloc`
żąda dokładnie tego layoutu; niezerowy wskaźnik jest natychmiast przejmowany
przez prywatny Vec o len=0 i capacity=n. Null daje typowany błąd. Dla n=0
powstaje pusty Vec bez alokacji i bez zajmowania ledgera. Ograniczone odczyty
dopisują najwyżej n bajtów, więc nie wywołują wzrostu Vec. To wąska granica
unsafe z lokalnym uzasadnieniem layoutu, inicjalizacji i globalnej dealokacji;
capacity odczytane po reserve nie stanowi podstawy admission.

Trzy prywatne, nieklonowalne rezerwacje są ostatnimi polami rzeczywistych
właścicieli. Bufory są niszczone przed oddaniem allowance, także przy błędach,
unwind i przenoszeniu semantyki do inspekcji GGUF. Współdzielenie całego
właściciela przez Arc utrzymuje rezerwacje do ostatniego drop. Nowy load
pozyskuje własne rezerwacje z tego samego ledgera. Po null lub innym błędzie
możliwe jest konserwatywne utrzymanie nadmiarowego obciążenia do zniszczenia
właściciela; wcześniejszego refundu nie ma. Gettery pożyczają bajty, bez
publicznego owned Vec, permitu, ręcznego release ani resetu. Kopie wykonane
przez caller’a z pożyczonego slice są poza zakresem loadera.

`GgufRawSourceError` rozróżnia Busy, Poisoned, Overflow,
Limit { requested, held, limit }, Layout { bytes } i Allocation { bytes }.
Nie przechowuje ścieżek ani wejścia. Odmowa zajętego locka nie czeka na
allowance. Cleanup może czekać na krótki lock arytmetyczny; wyłącznie cleanup
odzyskuje poisoned guard, bez ponownego otwarcia admission. Checked odejmowanie
przy niemożliwym underflow zamyka admission zamiast zawijać licznik lub panikować.
Nie ma obietnicy wait-free cleanup ani granicy czasu odczytu.

`WeightSourceError::RawSource { role, source }` oraz
`GgufCalibrationInventoryError::RawSource { index, source }` zachowują typowaną
przyczynę przez `source()`. Kalibracja zachowuje ją przez dotychczasowy anyhow
context i etap CalibrationSelection. Wrapper `GgufInventoryReadError` pozostaje
ten sam. **Dodanie wariantów publicznych enumów błędów wymaga aktualizacji
wyczerpujących matchów downstream**, także przy używaniu starych konstruktorów.
Stare entrypointy pozostają bez rezerwacji i zachowują kolejność odczytów,
alokacji, błędów, hashy, selekcji oraz fingerprint. Nowa ścieżka może odmówić
admission przed odczytem zawartości/preflight/serde, ale po wcześniejszych
kontrolach pliku i limitach długości. Hash deklaracji poprzedza Envelope i DTO;
hash companion jest porównywany przed Envelope. Inwentarz i actual-read binding
oraz końcowy stamp i atomowa instalacja pozostają bez zmian.

Poza rachunkiem są obiekty i control blocks, rekordy/nazwy/ścieżki/błędy,
serde/Value/normalizacja, native/tokenizer/cache, scratch nagłówka/payloadu GGUF,
staging i urządzenie. Nie obejmuje zaokrągleń/metadata/fragmentacji allocatora,
OS, page cache, RSS ani fizycznego OOM. Nie ogranicza liczby zerobajtowych
loadów ani współbieżności semantycznych builderów. REQ-009 pozostaje częściowe:
brak admission całego heapu, owned Engine/Planner, trwałej tożsamości, końcowego
same-fd rehash przed publikacją i materializacji/upload/completion/error retention.
Nie uzyskano dowodu runtime, zgodności bitowej/FP32, jakości ani pomiaru pamięci
lub wydajności. Zaakceptowany thinking FAIL pozostaje bez zmian.

## Zamknięty zakres semantyki G1

Nowe API wymaga zwykłej architektury Llama: `LlamaForCausalLM`, `silu`, osobnej
głowicy (`tie_word_embeddings: false`), jednakowego jawnego `attention_bias`
i `mlp_bias`, pełnego domyślnego RoPE i wymiarów sprawdzanych wspólnym parserem.
Dotychczasowe loadery zachowują własny zakres wejść i kolejność błędów.
G1 odrzuca każde nieznane pole config, nawet o wartości `false`, `0` lub `null`.
Poniższa tabela jest pełną polityką dodatkowych pól; wymagane pola podstawowe
są wymienione w pierwszym wierszu.

| Pola config | Polityka G1 |
|---|---|
| `model_type`, `architectures`, `hidden_act`, `tie_word_embeddings`, `attention_bias`, `mlp_bias`, `num_hidden_layers`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `rms_norm_eps`, `max_position_embeddings` | Wymagane, według wspólnego parsera Llama: dodatnie wymiary, skończone dodatnie epsilon/theta, sprawdzone iloczyny, GQA i parzysty head_dim |
| `pretraining_tp` | Brak albo całkowite 1 |
| `rope_parameters` | Wymagany obiekt: wymagane dodatnie skończone `rope_theta`; opcjonalne `rope_type` i alias `type` wyłącznie string `default`; opcjonalne `partial_rotary_factor`, `factor`, `attention_factor` wyłącznie skończone liczby dokładnie 1; żadnych innych kluczy |
| `rope_theta`, `partial_rotary_factor` na poziomie głównym | Brak albo odpowiednio dodatnia skończona liczba równa zagnieżdżonej theta i liczba dokładnie 1; null odrzucone |
| `rope_scaling`, `quantization_config`, `sliding_window`, `layer_types`, `per_layer_config`, `text_config`, `vision_config`, `audio_config` | Brak lub null; także pusty obiekt/array są odrzucane |
| `is_causal`, `is_decoder` | Brak albo boolean true |
| `is_encoder_decoder`, `add_cross_attention`, `use_bidirectional_attention` | Brak albo boolean false |
| `_attn_implementation`, `attn_implementation` | Brak, null albo string `eager` |
| `attention_dropout` | Brak albo skończona liczba 0 |
| `bos_token_id`, `pad_token_id` | Brak/null albo nieujemne u32 mniejsze od vocab; jawny BOS musi odpowiadać tokenowi BOS manifestu |
| `eos_token_id` | Brak/null, pojedyncze ograniczone u32 albo array takich ID |
| `_name_or_path`, `transformers_version` | Brak albo string; bierne dane pochodzenia, bez otwierania lub uruchamiania wskazanego kodu |
| `torch_dtype`, `dtype` | Brak/null albo string `float16`, `bfloat16`, `float32`; wskazówka nie ustala rzeczywistego typu wag |
| `initializer_range` | Brak albo skończona liczba w [0,1] |
| `use_cache` | Brak albo boolean; polityka cache pozostaje własnością runtime |

Wspólna semantyka `basal.json` zachowuje formaty 1/2, layout, orders, dokładne
template i BOS. Cały tokenizer pozostaje autorytatywny: vocab, merges, scores,
added tokens i ich właściwości, decoder oraz pre-tokenizer. Wspólny konstruktor
zachowuje kontrolę Llama Metaspace i rewrite `always|first` do `first`, wymaga
braku/null normalizera oraz wyłącza truncation i padding. Nie powstaje tokenizer
z samych tokenów GGUF ani drugi równoległy tokenizer.

Surowe ID modelu i added tokens są sprawdzane przed normalizacją i konstrukcją
zależności. Następnie sprawdzane są osobno efektywny model vocabulary, added
vocabulary i decoder map, zgodność deklarowanych added ID z rekonstrukcją
zależności oraz obecność i zgodność BOS. Każde rzeczywiste ID musi mieścić się
w u32 i być mniejsze od `config.vocab`. Dotyczy to też liczbowych BOS/EOS/PAD
w config i tokenizer_config. Unigram ma indeksowane ID i ograniczony unk_id.
Luki w ID i niewykorzystane wiersze embeddingów są dozwolone; liczebność
słownika nie musi być równa vocab i nie wymagamy gęstego słownika. Konstrukcja
nie wykonuje encode ani nie zapełnia cache kodów.

Przed konstrukcją BPE ścieżka captured sprawdza też warunki bezpieczeństwa
merge w przypiętym tokenizers 0.23.2: długość prefiksu w bajtach musi wyznaczać
granicę UTF-8 w drugim tokenie, a łączna długość musi mieścić się w buforze
konstruktora. Kontrola obejmuje pary tokenów i starszy zapis tekstowy merges,
bez przepisywania poprawnych danych i bez zmiany ścieżki legacy. Nie jest to
ogólny dowód bezpieczeństwa zależności ani zachowania tokenizera podczas encode.

To nie dowodzi poprawności kodów w kontekście odpowiedzi. Przyszłe planowanie
żądań musi zachować kontrolę pełnego prefiksu, jednego nowego tokena i unikalności
kodów, a przed backendem ograniczyć każde ID wejściowe, kodu/readout, po ponownym
kodowaniu lub skróceniu, evidence i thinking P2. Dotyczy to również ID podanych
bezpośrednio oraz przyszłych ID generowanych przed decode/embedding.

`GgufSemanticError` zachowuje przyczyny capture, UTF-8, JSON, wspólnych walidacji,
tokenizers i kalibracji. `MissingConfigG3` dotyczy wyłącznie błędu open NotFound
dla `config.json` i zachowuje ten błąd: zestawy bez config wymagają G3. Nie ma
syntezy config z GGUF. Błędy uprawnień, hashy i parserów nie otrzymują tej etykiety.
Top-level Display/Debug pomija bufory, pochodzenie i pełny JSON; szczegółowe
przyczyny mogą zawierać niezaufane dane i ścieżki.

## Zachowana kalibracja i tożsamość

`CalibrationCapture::files()` pożycza najwyżej dwa `CapturedCalibrationFile`:
początkowy Candidate i ewentualny OrdersVariant albo jeden Explicit. Każdy ma
`bytes()`, `sha256()` i `role()`. Zastąpiony kandydat pozostaje zachowany, bo jego
fitted fields wpływają na wybór. `selected()` wskazuje faktycznie wybrane bajty;
ich hash jest równy `manifest.calibration_provenance.content_sha256`. Obie
wartości są puste tylko wtedy, gdy żaden plik nie został wybrany/odczytany.

Wybór współdzieli algorytm legacy. Automatic preferuje istniejący plik layout,
potem sprawdza kandydata; brak daje pustą kalibrację. Najpierw sprawdzany jest
layout. Tylko brak mismatch i różne jawne fitted orders pozwalają automatic
sprawdzić jeden wariant `.o<orders>.json`; wybrany wariant zastępuje kandydata
także przy niezgodnym layout, bez powrotu. Explicit czyta dokładnie podaną
ścieżkę, także poza katalogiem; brak i błędny plik są błędami bez fallbacku.
Off nie czyta plików, ale dla niekanonicznego layout może wykonać dotychczasowy
probe preferowanego pliku. Sam `read` nie skanuje ani nie hashuje wszystkich wariantów.

`decision()` pożycza `CalibrationDecision`: mode, efektywny layout/orders/prompt,
strict, rzeczywiście osiągnięte initial/selected layout/orders checks, decyzję
wariantu i najwyżej trzy faktycznie wykonane probes. Nie zawiera ścieżek.
`NotEvaluated`, `Absent` i `Match` mają różne znaczenia. `Path::exists` może
zwrócić false także przy braku dostępu; wynik nie jest dowodem trwałej nieobecności.
Zniknięcie preferowanego pliku między probes zachowuje dotychczasową pustą
kalibrację; błąd open po ostatnim true przerywa operację.

Mismatch layout/orders zachowuje temperatury i buckets, odrzucając certyfikaty
i multi przed walidacją ich map. Wybrany prompt, temperatury i buckets nadal
podlegają dotychczasowej walidacji; brak temperatury typu daje 1.0. Kandydat
zastąpiony wariantem nie jest przedwcześnie walidowany pod tym kątem.
Jawny mismatch tożsamości ostrzega bez usuwania certyfikatów/temperatur;
strict zwraca istniejący błąd. Pooled format-1 certificates pozostają bez zmian.

Po odczycie `identity_state()` zwraca `CalibrationIdentityState`: NotRequired dla Automatic
i Off, Checked dla Explicit z podaną rzeczywistą served identity oraz Deferred
dla Explicit bez niej, nawet przy braku fitted model. Checked nie jest obietnicą
braku ostrzeżenia ani uwierzytelnienia pochodzenia. Samo
`calibration_provenance.check_identity(actual_served)` nie zapamiętuje wyniku.
Attested origin nie zastępuje served identity.

### Jednorazowe wiązanie logicznej served identity

`GgufCompanionSemantics::bind_served_identity(&mut self, ServedIdentity<'_>)`
zapisuje dokładny logiczny model i opcjonalną revision wraz z wynikiem kontroli
na rzeczywistym właścicielu semantyki. `calibration_binding()` pożycza
`Option<&GgufCalibrationBinding>`; gettery bindingu to `model()`, `revision()`
i kopiowany `outcome()`. Binding ma prywatne pola, bez publicznego konstruktora,
Clone, Default, serializacji, wymiany ani usuwania. Każdy dotychczasowy odczyt
zaczyna z None, także gdy początkowe `ModelLoadOptions.served_identity` dało Checked.

Najpierw odrzucany jest istniejący binding, również identyczne powtórzenie.
Następnie w całości walidowane są model i revision, przed kopiowaniem pól.
Każde obecne pole wymaga 1..=4096 bajtów UTF-8, bez ASCII control (także DEL),
początkowego `/`, dowolnego `\` lub `:`. Revision None jest dozwolone, Some
pustego stringa odrzucane. Zachowane są case, wewnętrzne i końcowe `/` oraz
pozostałe dozwolone bajty; nie ma trimowania, normalizacji ani wnioskowania
z basename, origin, aliasu czy basal.json. Ta opcjonalna metoda odrzuca lokalne
ścieżki absolutne i URI; dotychczasowe API kalibracji, ServedIdentity i CLI
zachowują wcześniejsze zachowanie.

Po fallible `String::try_reserve_exact` i skopiowaniu obu pól do lokalnych
wartości metoda ponownie wywołuje istniejący matcher na zachowanej provenance,
także dla wcześniejszego Checked. Dopiero po sukcesie przenosi ostrzeżenie do
rzeczywistego `manifest.calibration_provenance.identity_warning` i zapisuje
binding oraz stan. Każdy błąd pozostawia core warning/state/binding bez zmian.
Strict mismatch zachowuje oryginalny `CalibrationIdentityError` jako przyczynę;
nonstrict warning nie usuwa temperatur ani rzeczywistych certyfikatów. Pozostałe
mapy, fitted fields, wybór i read-set kalibracji pozostają bez zmian.

`GgufCalibrationBindingOutcome` to NotRequired dla zachowanego Automatic/Off
albo CheckedNoWarning/CheckedWarning dla Explicit, zgodnie z obecnością
zwróconego ostrzeżenia. Stan Explicit po sukcesie to Checked, nigdy Deferred;
Automatic/Off pozostaje NotRequired. Wynik oznacza wykonaną kontrolę, nie dowód
zgodności: istniejący matcher pomija brak fitted model i porównuje revision
tylko, gdy obie są znane. Matcher nadal porównuje końcowy komponent modelu bez
uwzględniania case i końcowych `/`; zapisany binding zachowuje pełne wejście.

Nowe kopie wejścia żądają najwyżej 8192 bajtów plus struktury. Nie jest to limit
capacity, RSS ani admission heapu: niezmieniony matcher nadal alokuje lowercase,
ostrzeżenia i błędy z zachowanych pól. `GgufIdentityBindingError` rozróżnia
AlreadyBound, InvalidField, Allocation i Calibration. Top-level Display/Debug
pomija identity, warning, ścieżki i przyczyny, także dla zewnętrznie utworzonych
field/reason. Pola wariantu pozostają dostępne do inspekcji; `source()` zachowuje
rzeczywistą niezaufaną przyczynę alokacji lub kalibracji bez przepisywania tekstu.

### Opcjonalne zachowanie pełnej przestrzeni nazw kalibracji

`GgufCompanionSemantics::read_with_calibration_inventory(directory, options, lease)`
na 64-bitowym little-endian Linux/macOS najpierw zachowuje pełną ograniczoną
przestrzeń nazw kalibracji, potem wywołuje dotychczasowy `read` dokładnie raz.
Po jego sukcesie wiąże inventory z rzeczywistymi odczytami, osiągniętymi probes
i wybranym digestem manifestu, sprawdza pierwotny stamp katalogu i dopiero wtedy
zapisuje inventory na tym samym właścicielu. Błąd porzuca cały częściowy wynik.
Nie ma operacji dołączania nowego inventory do wcześniej odczytanej semantyki.

`CallerCalibrationDirectoryLease::acknowledge_trusted_local_contract()` potwierdza
rzeczywiste zewnętrzne wykluczenie zapisów, zapewnione **przed wejściem** i utrzymane
przez caller’a do zwrotu sukcesu lub błędu. Osobno obejmuje członkostwo katalogu
(tworzenie, usuwanie, rename, exchange i zastępowanie), rozwiązywanie katalogu,
root i ancestors oraz zapisy treści/metadanych przez inne deskryptory, procesy,
aliasy, hardlinki i writable mmap. Jawny override wymaga osobnego pokrycia jego
ścieżki i celu, również poza katalogiem. Companions nadal wymagają zaufania według
wcześniejszego kontraktu. Gdy caller nie potrafi zapewnić tego układu, musi odmówić
użycia tej ścieżki. Token nie uzyskuje blokady i nie produkuje snapshotu.

Caller musi wcześniej ustalić, że Explicit rozwiązuje się do zaufanego regularnego
pliku. Dziedziczony odczyt otwiera go przed kontrolą typu; FIFO naruszające ten
warunek może blokować. Explicit nadal może być symlinkiem do regularnego celu,
objętym rzeczywistym wykluczeniem zmian rozwiązywania i treści. Nie dodano probe
override ani gwarancji wrogiego systemu plików, czasu wykonania czy odporności
na niekooperującego writer’a. Stampy dev/ino/typ/długość/mtime/ctime nie obejmują
atime odczytów; równe obserwacje nie zastępują rzeczywistego wykluczenia.

| Zasób nowego inventory | Stały limit / polityka |
|---|---|
| Jedno nierekurencyjne przejście katalogu do EOF | 4096 wszystkich wpisów, także niedopasowanych; 4097. odmawia |
| Nazwy spełniające dokładnie `starts_with("CALIBRATION") && ends_with(".json")` | 256 wpisów, 255 bajtów UTF-8 na nazwę i 64 KiB sumy długości; case i bajty zachowane bez normalizacji |
| Typy i kodowanie | Każdy nie-UTF-8 wpis katalogu odrzucany, także niedopasowany; matching symlinki i inne nieregularne wpisy odrzucane przed open; duplikaty po sortowaniu odrzucane |
| Początkowe długości i pomyślnie zachowane capacities bajtów | 64 MiB na plik, 128 MiB łącznie, osobno dla długości i capacities |
| Nowe metadane i scratch | Capacity wektora rekordów razy rozmiar rekordu plus capacities Stringów do 256 KiB; scratch odczytu 8192 bajtów, jeden bajt EOF, jeden hash i otwarty plik naraz; ponownie używany String nazw cross-binding żąda 128 bajtów, przyjmuje capacity do 256 |

To świadomie węższa polityka nowego opt-in: symlinkowane pliki kalibracji,
np. w układzie cache HF, są odrzucane. Dotychczasowe `read` i profile zachowują
wcześniejsze reguły symlinków/nazw. Limity nie mają zmierzonego pokrycia paczek.
Nie ma cichego skracania, deduplikacji, lossy decoding ani pustego sukcesu po błędzie.
Puste `entries()` oznacza ukończony pusty zbiór; `calibration_inventory() == None`
w dotychczasowych konstruktorach oznacza, że pełnego zbioru nigdy nie zachowano.

Kolejność błędów jest jawna: host, otwarty katalog/pierwotny stamp, rezerwacja,
ograniczona enumeracja; traversal/I/O/alokacja przerywają od razu. Po EOF
pierwszeństwo mają niepoprawny UTF-8, limit pojedynczej nazwy, liczby matching
i sumy nazw; potem sort bajtowy i duplikaty. Pliki są sprawdzane i czytane w tej
kolejności: symlink_metadata/typ/długość, budżet, open/zgodność pierwotnego stampu,
fallible reserve/actual capacity, dokładny odczyt/hash, próba EOF i stamp tego
samego fd. Krótkie odczyty są kontynuowane, Interrupted ponawiane tylko w nowych
odczytach capture; wczesny EOF lub wzrost odmawia. Stamp nie jest odświeżany.
Po stampie katalogu następuje istniejący odczyt semantyki; jego błąd, także strict,
wraca bez późniejszego stampu maskującego przyczynę. Dopiero sukces przechodzi
cross-binding i ostatni stamp katalogu. Deskryptor katalogu pozostaje otwarty
przez całą operację. Kolejność błędów OS nie jest deterministycznie gwarantowana.

`GgufCalibrationInventory::entries()` pożycza posortowane rekordy
`GgufCalibrationVariant` z getterami `name()`, `bytes()` i typowanym `sha256()`.
`entry_for_read(Candidate|OrdersVariant)` pożycza osobno powiązane rekordy,
także gdy mają identyczne bajty. Nazwy wynikają wyłącznie z zachowanych
layout/orders i osiągniętych probes. Każdy osiągnięty true/false musi zgadzać
się z obecnością dokładnej nazwy w pełnym zbiorze. Automatic wiąże długość,
bajty i digest rzeczywistych slotów; digest wybranego slotu musi odpowiadać
rzeczywistej provenance manifestu. Explicit zachowuje osobny rzeczywisty odczyt,
również przy aliasie wpisu; `entry_for_read(Explicit)` zawsze zwraca None.
Off nie ma wybranego odczytu/digestu, lecz jego osiągnięty probe preferencji
layout też jest sprawdzany. Missing, NotEvaluated i NotConsidered pozostają
odrębne; nieosiągnięte warianty nie otrzymują wymyślonych wyników probes.

Nieużywane warianty nie są parsowane. Cross-binding nie wywołuje exists, nie
otwiera ponownie ścieżek, nie zmienia wyboru, fitted fields, wartości kalibracji,
certyfikatów, ostrzeżeń ani matchera. Served binding nadal zaczyna od None.
Inventory ma prywatne pola, bez publicznego konstruktora, Clone, Default,
serializacji, mutacji czy into_parts. Cały wzbogacony właściciel companions
przechodzi już przez istniejące inspekcje GPU bez zmiany ich wrapperów.

`GgufInventoryReadError::Inventory` zachowuje typowany
`GgufCalibrationInventoryError`, a `Semantic` oryginalny `GgufSemanticError`.
Inventory rozróżnia host, I/O/operację, katalog, nazwę, limit/zasób, duplikat,
nieregularny wpis, zmianę stampu/fazę/pole, alokację, arytmetykę, read-set,
probe i selected provenance. Top-level Display/Debug obu błędów ma stały,
ograniczony tekst także dla zewnętrznie skonstruowanych wariantów; nie wypisuje
nazw, ścieżek, JSON, identity, warning ani zagnieżdżonych przyczyn. `source()`
zachowuje rzeczywiste niezaufane przyczyny I/O, rezerwacji i semantyki.

Wybrane i zastąpione bajty są celowo przechowywane ponownie: inventory oraz
rzeczywiste odczyty selektora mają osobne bufory. Suma surowych logicznych
długości wynosi najwyżej 384 MiB + 64 KiB deklaracji, plus nowe metadane/scratch
i istniejące obiekty semantyczne. Nie jest to limit capacities całego właściciela,
szczytu/RSS, OOM ani admission. W ścieżkach bez raw budget actual capacity jest sprawdzana **po** fallible
reserve, więc chwilowa nadmiarowa alokacja może już wystąpić. Alokacje OS
enumeracji/nazw/ścieżek, narzut allocatora i amplifikacja JSON/tokenizera są poza
tym rachunkiem. Nie zadeklarowano poprawy pamięci ani wydajności.

Po zwrocie kalibracja używa zachowanych bajtów, więc jej zewnętrzny guard może
się zakończyć. Inventory opisuje chroniony przedział capture, nie aktualny stan
ścieżek. Niezależny obowiązek no-writer GGUF trwa przez końcowe odczyty/rehash
i publikację. Warunkowe świadectwo członkostwa nie daje autentyczności,
snapshotu, trwałej tożsamości, loadera, admission ani uprawnień publikacji.
REQ009 pozostaje częściowe: poza opcjonalnym admission raw otwarte są utrzymywane admission przed pozostałymi alokacjami,
rzeczywista trwała tożsamość źródła/numeryki, materializer/upload/completion
i error retention, owned konsumenci, readout/CLI/runtime. Domyślne zachowanie,
numeryka, legacy fingerprint i zaakceptowany thinking FAIL pozostają bez zmian;
nie dodano nowego źródła fused/split/Intel/CPU/media/evidence/exits/thinking.

## Publiczne API

| Element | Znaczenie |
|---|---|
| `GgufCompanionSnapshot::read(directory)` | Odczyt snapshotu: sześć stałych nazw, bez przeglądania katalogu i wyszukiwania `.gguf` |
| `declaration()` | Pożyczona, sprawdzona `GgufDeclaration`, z prywatnymi polami |
| `declaration_bytes()`, `declaration_sha256()` | Dokładne bajty deklaracji i hash wszystkich tych bajtów |
| `bytes(CompanionRole)` | Pożyczony slice zachowanych bajtów wskazanej roli; bez ponownego otwierania ścieżki |
| `Sha256Digest` | 32 bajty; `FromStr` wymaga dokładnie 64 znaków ASCII hex, `Display` normalizuje do małych liter, `as_bytes()` pożycza bajty |

`CompanionRole` ma dokładnie pięć wariantów:

| Wariant | Nazwa pliku |
|---|---|
| `Config` | `config.json` |
| `Basal` | `basal.json` |
| `Tokenizer` | `tokenizer.json` |
| `TokenizerConfig` | `tokenizer_config.json` |
| `ChatTemplate` | `chat_template.jinja` |

`GgufDeclaration` udostępnia `layout()`, `outtype()`,
`declared_gguf_sha256()`, `companion_sha256(role)`, `origin_model_id()`,
`origin_revision()`, `origin_safetensors()` oraz `metadata_repair()`.
`origin_safetensors()` pożycza nazwy i zwraca typowane digests w zadeklarowanej
kolejności. Żaden getter nie wykonuje I/O ani nie kopiuje całych buforów.
Snapshot nie przechowuje lokalnych ścieżek. Nie definiuje fingerprintu całego
źródła ani serializacji planu wykonania.

## Schemat `basal.gguf-source.1`

Każdy poziom obiektowy wymaga obiektu JSON. Tablice pozycyjne, nieznane
i powtórzone pola są odrzucane. Wszystkie pola są obowiązkowe, także
`metadata_repair`: pominięcie tego pola jest błędem, a jawne `null` oznacza
brak naprawy metadanych. Nie ma domyślnych wartości rozszerzających kontrakt.

| Pole | Dozwolona wartość |
|---|---|
| `schema` | Dokładnie `basal.gguf-source.1` |
| `layout` | Dokładnie `llama-cpp-hf-llama-v1`, typ `GgufLayout::LlamaCppHfLlamaV1` |
| `gguf_sha256` | SHA-256 końcowego GGUF po ewentualnej naprawie metadanych, na tym etapie niezweryfikowany |
| `converter` | Dokładny przypięty kontrakt opisany poniżej |
| `companion_sha256` | Obiekt z dokładnie pięcioma nazwami z tabeli ról i ich SHA-256 |
| `origin` | Logiczne `model_id`, `revision` i niepusta tablica `safetensors` |
| `metadata_repair` | `null` albo dokładny przypięty obiekt naprawy poniżej |

`converter` wymaga repozytorium `ggml-org/llama.cpp`, commitu
`7df8b876b91208a86a6d792e1140b3deec14535b`, klasy
`conversion.llama.LlamaModel`, architektury `LlamaForCausalLM` oraz
`endianness: "little"`. `outtype` to wyłącznie `f16`, `bf16` lub `f32`
(`GgufOuttype::{F16,Bf16,F32}`). `split`, `tensor_overrides` oraz
`weight_postprocessing` muszą być boolean `false`. Deklaracja nie może
rozszerzyć obsługiwanej gałęzi konwertera.

`origin.model_id` i `origin.revision` to niepuste logiczne ciągi bez znaków
kontrolnych, lokalnych ścieżek bezwzględnych i URI. Walidacja odrzuca też
dwukropek, aby wykluczyć schematy URI i ścieżki dysków Windows. Wpisy
`origin.safetensors` zawierają dokładnie `name` oraz `sha256`. Nazwy muszą być
unikalnymi pojedynczymi nazwami plików z końcówką `.safetensors`, bez separatorów,
dwukropka, NUL i innych znaków kontrolnych. Nie mogą być `.` lub `..` ani
ścieżkami bezwzględnymi. API nie otwiera, nie rozwiązuje i nie pobiera tych nazw,
modelu ani rewizji. Kolejność listy pozostaje częścią attestation.

Naprawa metadanych ma dokładnie `kind: "tokenizer-only"`,
`basal_exporter_commit: "b20f0f9b159f71726459aa98ede1356ab65695fd"` oraz
`gguf_copy_commit: "7df8b876b91208a86a6d792e1140b3deec14535b"`.
Ostatni pin wiąże kod kopiowania/zapisu GGUF, którego nie określa sama wersja
pakietu `gguf`. Getter zwraca `Some(MetadataRepair::TokenizerOnly)` albo `None`.

Poniższy kompletny przykład jest wyłącznie ilustracją schematu. Wszystkie
digests złożone z zer są **placeholderami**, a model i rewizja są fikcyjne.
Przykład nie opisuje rzeczywistego checkpointu, konwersji ani zgodnych plików.

```json
{
  "schema": "basal.gguf-source.1",
  "layout": "llama-cpp-hf-llama-v1",
  "gguf_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "converter": {
    "repository": "ggml-org/llama.cpp",
    "commit": "7df8b876b91208a86a6d792e1140b3deec14535b",
    "class": "conversion.llama.LlamaModel",
    "architecture": "LlamaForCausalLM",
    "outtype": "f16",
    "endianness": "little",
    "split": false,
    "tensor_overrides": false,
    "weight_postprocessing": false
  },
  "companion_sha256": {
    "config.json": "0000000000000000000000000000000000000000000000000000000000000000",
    "basal.json": "0000000000000000000000000000000000000000000000000000000000000000",
    "tokenizer.json": "0000000000000000000000000000000000000000000000000000000000000000",
    "tokenizer_config.json": "0000000000000000000000000000000000000000000000000000000000000000",
    "chat_template.jinja": "0000000000000000000000000000000000000000000000000000000000000000"
  },
  "origin": {
    "model_id": "example/model",
    "revision": "example-revision",
    "safetensors": [
      {
        "name": "model.safetensors",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
      }
    ]
  },
  "metadata_repair": {
    "kind": "tokenizer-only",
    "basal_exporter_commit": "b20f0f9b159f71726459aa98ede1356ab65695fd",
    "gguf_copy_commit": "7df8b876b91208a86a6d792e1140b3deec14535b"
  }
}
```

## Limity odczytu i błędy

Deklaracja ma limit 64 KiB, każdy companion 64 MiB, a pięć companions razem
128 MiB. Są to stałe limity inżynierskie. Długości są sumowane z kontrolą
przepełnienia przed rezerwacją bufora; rezerwacje buforów bajtowych są fallible.
Odczyt nie przekracza początkowej długości, po czym używa sondy jednego bajtu
do wykrycia wzrostu. Sonda nie jest zachowywana ani hashowana. EOF przed
początkową długością oznacza zmianę pliku. Błąd po częściowym odczycie przerywa
operację bez zwracania częściowego snapshotu.

Każdy plik otwierany jest raz i musi być regularny. Metadane tego samego
uchwytu są sprawdzane przed i po odczycie: rodzaj, długość i czas modyfikacji.
Brak możliwości odczytania czasu modyfikacji jest błędem I/O. Zmiany wykryte
przez te porównania, skrócenie lub wzrost pliku przerywają operację. Wszystkie
zachowane bajty są hashowane; wynik jest zwracany dopiero po zgodności pięciu
hashy z deklaracją. Pusta zawartość companion nie jest tu semantycznie
odrzucana, jeśli zgadza się jej hash; późniejszy parser musi ją ocenić.

`WeightSourceError` rozróżnia niepoprawny JSON/schemat, niedozwolone pola
kontraktu, składnię digestu z nazwą pola, niezgodność digestu z rolą,
nieregularny lub zmieniony plik, limit zasobów, alokację i I/O.
Przyczyny JSON, digestu, alokacji i I/O zachowują łańcuch `source()`.
`Display` i `Debug` nie wypisują surowych buforów ani tekstu pochodzenia modelu.
Komunikat niezgodności digestu podaje oczekiwany i obliczony hash.
Oryginalna szczegółowa przyczyna parsera JSON może zawierać niezaufane dane
deklaracji; należy traktować ją zgodnie z tym charakterem przy logowaniu.

Limity bajtów nie gwarantują fizycznej dostępności RAM ani ochrony przed OOM.
Nie ograniczają amplifikacji alokacji podczas parsowania JSON deklaracji ani
parsowania konfiguracji lub tokenizera. Sam snapshot pozostawia companions
jako surowe bajty; nowe API semantyczne dodatkowo uruchamia zwykłe parsery
serde_json i tokenizers na zaufanych lokalnych wejściach. Węzły, mapy, stringi,
BPE/Unigram, matchery added tokens, regex i wewnętrzne rezerwacje zależności
mogą powiększać pamięć bez ilościowego limitu. Nie jest to interfejs dowolnego
uploadu. Przed obietnicą ograniczonego host preflight trzeba uwzględnić żywe
alokacje parserów/tokenizera w budżecie hosta albo osobno zweryfikować parse
quotas; mnożnik rozmiaru pliku nie dowodzi takiego ograniczenia.

W podstawowym `read` każdy rzeczywiście czytany plik kalibracji ma limit 64 MiB, najwyżej dwa odczyty
i razem 128 MiB. Companions z kalibracją mają wspólny limit 256 MiB, a deklaracja
osobno 64 KiB. Jeden licznik zaczyna od sprawdzonej sumy rzeczywistych długości
pięciu companions S; jego granica to `min(S.checked_add(128 MiB)?, 256 MiB)`.
Kalibracja używa tego samego czytnika i obciąża sumę przed rezerwacją; opłata za
zastąpionego kandydata nie znika. Łączne zachowane długości raw to najwyżej
256 MiB + 64 KiB, z możliwym narzutem capacity/allocatora oraz dodatkowymi
strukturami semantycznymi. Opcjonalne inventory dodaje osobne, jawnie opisane
wyżej bufory do 128 MiB; podstawowy `read` ich nie tworzy.
Konstrukcja tokenizera przejściowo posiada Value, znormalizowany string i obiekt
zależności; audyt bazowego vocab tworzy też przejściową kopię mapy z API zależności.
Na błędzie częściowy wynik jest odrzucany, a bufory zwalniane normalnie.

## Granica zaufania i dalsza walidacja

Hash wiąże bajty. Nie potwierdza prawdziwości historii konwersji, autora,
wykonania wskazanych commitów ani zgodności z checkpointem źródłowym.
Niezweryfikowane pozostają zarówno `gguf_sha256`, jak i źródłowe digests
safetensors. Wielkość słownika lub podobny kształt tensorów nie zastępują
sprawdzenia pochodzenia i semantyki.

Otwarte uchwyty i porównania metadanych nie stanowią immutable lease.
Nie chronią przed złośliwymi zapisami w miejscu, zmianą i przywróceniem danych
między kontrolami ani atakami na rozwiązywanie ścieżek. Standardowy `File::open`
może podążać za symlinkami; API wymaga zaufanego lokalnego źródła plików i nie
gwarantuje bezpiecznego otwarcia dowolnego złośliwego węzła systemu plików.
Zachowane bajty są stabilne po odczycie także wtedy, gdy ścieżki później się
zmienią. Kolejne parsowanie musi korzystać z tych bajtów albo osobnego
zweryfikowanego immutable snapshotu. Ponowne otwarcie ścieżek nie dziedziczy
tych hashy. Snapshot nie gwarantuje jednoczesnego stanu wszystkich plików.

Opisane niżej API GPU sprawdza nagłówek, metadane oraz inventory i zakresy
tensorów tego samego otwartego GGUF. Dalszy preflight musi zweryfikować payload
i jego finitość, digest tego samego pliku, niezmienność źródła oraz jawne budżety
pamięci. Musi też zachować późniejsze granice ID żądań. API semantyczne
zamyka opisany etap companions/wybranej kalibracji, a nowe opt-in dodaje
warunkowo zachowany pełny zbiór wariantów. Nie definiuje fingerprintu całego
źródła ani trwałej tożsamości. Późniejsza tożsamość musi wiązać również pełne
nazwy/digests inventory, zastąpionego kandydata, wybrany digest, tryb/protokół
oraz rzeczywiście wykonane decyzje/probes, bez lokalnych ścieżek.

Przyszłe konstruktory Engine i RequestPlanner muszą przejąć zwalidowany tokenizer
wraz z manifestem. Obecny `Engine::fork` ponownie używa konstruktora ścieżkowego
i wymaga osobnego dostosowania. Nowa ścieżka nie może otwierać companions przez
`manifest.dir`, zachowując stare hashe. Ownership workerów, source lease/snapshot,
host/device/staging budgets, trwała tożsamość wariantów, materializacja, CLI/runtime i kwalifikacja
numeryczna pozostają otwarte. Nie potwierdzono dystrybucji modeli, zgodności
bitowej, wyników FP32, runtime GPU ani nowej akceptacji thinking.

## Inspekcja nagłówka i inventory na hoście

`basal_gpu::gguf::GgufHeaderInspection::inspect(file, companions, lease)`
przejmuje dokładnie jeden już otwarty `std::fs::File`, istniejący
`GgufCompanionSemantics` oraz jawny `CallerImmutableLease`. Moduł jest dostępny
bez feature GPU na 64-bitowym little-endian Linux i macOS. Pozostałe hosty
otrzymują `GgufInspectionError::UnsupportedHost`; obsługa hosta nie oznacza
inferencji CPU ani Intel.

Caller wybiera plik i zapewnia zaufane lokalne otwarcie. Inspekcja nie otwiera
ścieżek, nie szuka `.gguf`, nie pobiera origin, nie odkrywa siblings i nie
wykonuje fallbacku. Nie parsuje ponownie companions, nie konstruuje ani nie
uruchamia tokenizera. `companions()` pożycza ten sam zachowany obiekt,
`tensor_count()` zwraca sprawdzoną liczbę tensorów, a `source_tensor_bytes()`
sumę bajtów ich gęstych danych źródłowych, bez nagłówka, luk i paddingu.
Ta suma nie jest rozmiarem pamięci rezydentnej.

Wynik nie jest `Clone`, nie ma publicznych pól ani dostępu do `File` lub map
Candle. Zachowuje razem plik, sparsowany `Content`, stamp, companions i token
obowiązku caller’a. Nie ma serializacji, `into_parts`, materializacji ani API
urządzenia. Błąd odrzuca wszystkich przejętych właścicieli bez częściowego
wyniku. Kolejne osobno wdrażane fazy muszą przejąć tych samych właścicieli;
ponowne otwarcie przez `manifest.dir` nie dziedziczy wyniku inspekcji.

### Obowiązek caller’a i obserwacje pliku

`CallerImmutableLease::acknowledge_trusted_local_contract()` jest bezpiecznym,
jawnym potwierdzeniem: caller przed przekazaniem pliku zapewnił wyłączne
wykluczenie zapisów do tego inode i utrzyma je przez wszystkie późniejsze
odczyty oraz publikację wyniku. Caller sam zachowuje ewentualny rzeczywisty
guard. Token nie uzyskuje blokady, nie dowodzi niezmienności, nie jest guardem
systemu plików ani warunkiem bezpieczeństwa Rust. Nie ma `Clone`/`Default`,
a inspekcja nie tworzy go niejawnie. Fałszywe potwierdzenie nie zmienia
bezpiecznego I/O w operację `unsafe`, lecz unieważnia założenie spójności źródła.
Przyszła integracja musi spełnić ten kontrakt, zapewnić osobno zweryfikowany
immutable snapshot albo odmówić. Producent snapshotów nie jest wdrożony.

Plik musi być regularny i mieć od 8 bajtów do 64 GiB. Przed inspekcją i po niej
ten sam descriptor dostarcza `dev`, `ino`, typ, długość, `mtime` i `ctime`
z nanosekundami. Błąd metadata jest błędem I/O; wykryta zmiana odmawia wyniku.
Otwarty descriptor chroni przed podmianą ścieżki, ale nie przed zapisem przez
inny descriptor, proces, mmap lub hardlink. Równe stampy nie dowodzą braku
zmian pomiędzy obserwacjami. Nawet przyszłe powtarzane hashe nie dowiodą braku
wyścigów ani zmian i przywróceń danych. Inspekcja nie oblicza żadnego hasha GGUF.
Zachowane bajty companions/kalibracji są już własnością wcześniejszego etapu
i nie wymagają dalszego lease ich ścieżek.

### Kolejność i limity

Najpierw sprawdzane są host, plik i `N = 3 + (bias ? 16 : 9) * layers`,
z kontrolą przepełnienia i limitem 8195 przed alokacją scratch inventory.
Następnie inspekcja ustawia pozycję zero i odczytuje dokładnie osiem bajtów:
literalne `GGUF` oraz little-endian wersję 3. Po ponownym seek do zera wywołuje
raz `Content::read_with_limits`. Nie wywołuje API odczytu danych tensorów.

| Limit parsera | Wartość |
|---|---:|
| Nagłówek / wpisy metadata | 64 MiB / 4096 |
| Tensory / rank | sprawdzone N / 2 |
| Jeden / wszystkie surowe stringi | 1 MiB / 32 MiB |
| Węzły Value / poziomy zagnieżdżenia array | 1 048 576 / 2 |
| Kumulatywny ledger żądanych alokacji parsera | 256 MiB |

Osobny scratch produktu to `N * size_of::<(u64,u64)>() + 64`, sprawdzany
przed `try_reserve_exact` z limitem 1 MiB, oraz jeden ponownie używany string
nazwy z fallible rezerwacją 128 bajtów. Sortowanie zakresów nie alokuje bufora.
Nie powstaje mapa oczekiwanego inventory ani bufor modelu. Ledger parsera,
scratch produktu i istniejące alokacje companions są odrębne. Stos, diagnostyka
błędów, narzut allocatorów i fizyczne RSS pozostają poza tym lokalnym rachunkiem.
Limity nie zapewniają dostępności RAM ani ochrony przed OOM. Wcześniejsze
ograniczenia ilościowego rachunku semantyki/tokenizera nadal obowiązują.

### Zamknięte metadane GGUF

Poniższe wartości są wymagane w dokładnych wariantach `Value`; nie ma
rozszerzania integerów z U8/U16 ani porównania float z tolerancją. Każdy
integer config musi mieścić się w U32. Epsilon i theta po zaokrągleniu do F32
muszą pozostać dodatnie i skończone; porównanie dotyczy tej wartości F32,
bez zmiany autorytatywnych f64 manifestu.

| Wymagany klucz | Dokładna wartość |
|---|---|
| `general.architecture` | String `llama` |
| `llama.block_count`, `llama.embedding_length`, `llama.feed_forward_length`, `llama.vocab_size` | U32: layers, hidden, intermediate, vocab |
| `llama.context_length`, `llama.attention.head_count`, `llama.attention.head_count_kv` | U32: max_positions, heads, kv_heads |
| `llama.attention.key_length`, `llama.attention.value_length`, `llama.rope.dimension_count` | U32 head_dim, pełne RoPE |
| `llama.attention.layer_norm_rms_epsilon`, `llama.rope.freq_base` | F32: zaokrąglone rms_eps, rope_theta |
| `general.file_type` | U32: 0 dla F32, 1 dla F16, 32 dla BF16 z deklaracji |

| Opcjonalny klucz | Jedyna dopuszczalna wartość, jeśli obecny |
|---|---|
| `general.alignment`, `general.type` | U32 32, String `model` |
| `llama.attention.causal`, `llama.rope.scaling.type` | Bool true, String `none` |
| `split.no`, `split.count`, `split.tensors.count` | U16 0, U16 1, I32 N; inne klucze split odrzucone |
| `general.quantization_version` | U32 2, bierne: przypięty konwerter emituje także dla dense; nie dopuszcza kwantyzacji |

Bierne pola nie konfigurują modelu, ścieżek, tożsamości ani pochodzenia:

- String dla dokładnie `general.{name,author,version,organization,finetune,basename,description,quantized_by,size_label,license,url,doi,uuid,repo_url}`, `general.license.{name,link}` i `general.source.{url,doi,uuid,repo_url}`.
- Array wyłącznie String dla `general.tags` i `general.languages`; puste dozwolone.
- U32 dla `general.base_model.count` i `general.dataset.count`; String dla `general.base_model.<id>.<suffix>` i `general.dataset.<id>.<suffix>`. ID to kanoniczny nieujemny dziesiętny U32 bez zer wiodących poza `0`; suffix to dokładnie `name|author|version|organization|description|url|doi|uuid|repo_url`. Count nie jest wymiarem ani wymogiem pobrania danych.
- `tokenizer.*` przyjmuje dowolne wartości w limitach parsera, całkowicie nieużywane. Nie porównuje się tego tokenizera z companions.
- Wszystkie pozostałe klucze są odrzucane, w tym inne `llama.*`, inne architektury, nieznane `general.*`, `general.sampling.*` i `general.tensor_extra.*`.

Wymagane i opcjonalne kontrole mają stałą kolejność. Spośród wielu nieznanych
kluczy wybierany jest leksykograficznie pierwszy pożyczony klucz. Bierne wartości
o złym typie również są oceniane deterministycznie. Parser może wcześniej
odrzucić wadliwą strukturę lub przekroczenie limitu.

Surowe klucze i nazwy tensorów muszą być UTF-8 bez NUL; duplikaty są odrzucane
przez ograniczony parser. Wartości string zachowują semantykę Candle:
usunięcie końcowych NUL i lossy decoding niepoprawnego UTF-8. Równość
strukturalnych String dotyczy dokładnie wartości **po dekodowaniu**, a nie
kanonicznego kodowania na dysku. Na przykład `llama` z końcowym NUL może
zostać zdekodowane do `llama`. Wstępne osiem bajtów magic/version sprawdza się
surowo, przed Candle; nie powstaje drugi parser GGUF.

### Dokładne inventory i zakresy

Zachowany config nadal wymaga zwykłej Llama, domyślnego pełnego RoPE, GQA,
osobnej głowicy i jednolitej polityki bias. Matryce mają jednolity typ M
z deklaracji: F16, BF16 albo F32. Każdy wektor normy i bias ma F32.
Wymiary są już raz odwrócone przez Candle do kolejności logicznej;
inspekcja nie odwraca ich ponownie i nie transponuje danych.

Niech H=hidden, I=intermediate, V=vocab, K=kv_heads*head_dim, L=layers.

| Wymagana rola GGUF | Dodatni dokładny shape logiczny / typ |
|---|---|
| `token_embd.weight`, `output.weight` | [V,H] / M |
| `output_norm.weight` | [H] / F32 |
| `blk.<b>.attn_norm.weight`, `blk.<b>.ffn_norm.weight` | [H] / F32 |
| `blk.<b>.attn_q.weight`, `blk.<b>.attn_output.weight` | [H,H] / M |
| `blk.<b>.attn_k.weight`, `blk.<b>.attn_v.weight` | [K,H] / M |
| `blk.<b>.ffn_gate.weight`, `blk.<b>.ffn_up.weight` | [I,H] / M |
| `blk.<b>.ffn_down.weight` | [H,I] / M |
| Przy bias: `attn_q.bias`, `attn_output.bias`, `ffn_down.bias` pod `blk.<b>.` | [H] / F32 |
| Przy bias: `attn_k.bias`, `attn_v.bias` pod `blk.<b>.` | [K] / F32 |
| Przy bias: `ffn_gate.bias`, `ffn_up.bias` pod `blk.<b>.` | [I] / F32 |

Dla każdego b z [0,L) wymagane jest dziewięć wag i wszystkie siedem bias iff
config.bias. Dokładna liczebność N i obecność każdej stałej roli dowodzą
braku dodatkowych tensorów. Nie ma tied fallbacku, `rope_freqs.weight`,
fused QKV, evidence, wyjść pomocniczych ani dowolnych mieszanek typów.
Mapowanie payloadu Q/K i materializacja tensorów nie są wykonywane.

Iloczyny wymiarów, szerokości, dodawanie offsetów i sumy używają checked u64;
konwersje do usize przed rezerwacjami także są sprawdzane. Każdy tensor ma
najwyżej 8 GiB. Baza danych musi być równa checked round_up(header_end,32)
i mieścić się w początkowej długości. Offset względny i start są podzielne
przez 32; każdy zakres spełnia `base <= start < end <= initial_EOF`.
Posortowane numerycznie pary zakresów nie mogą się nakładać ani aliasować.
Luki, padding i bajty po ostatnim tensorze są dozwolone: nie wymagamy
`last_end == EOF` ani zerowego paddingu.

Suma rzeczywistych bajtów jest porównywana z zamkniętym wzorem, z kontrolą
każdej operacji (w(M)=2 dla F16/BF16, 4 dla F32):

```text
matrix_elements = 2*V*H + L*(2*H*H + 2*H*K + 3*H*I)
bias_per_layer = 3*H + 2*K + 2*I
source_bytes = w(M)*matrix_elements
             + 4*((2*L+1)*H + (bias ? L*bias_per_layer : 0))
```

`GgufInspectionError` rozróżnia host/plik/zmianę źródła, I/O, magic/version,
parser, limity i arytmetykę, alokację, reprezentację config, brak/typ/wartość
metadata, niedozwolone metadata/split, count/rolę/typ/shape tensorów, alignment,
bazę/zakresy/overlap oraz niezgodność sumy bajtów. `source()` zachowuje
Candle/I/O/TryReserveError; nie klasyfikuje błędów parsera po tekście.
Top-level Display/Debug ma ograniczony tekst kluczy/nazw/wartości i pomija
bufory, pełny JSON, origin oraz ścieżki. Zagnieżdżone przyczyny są niezaufane.

## Digest, payload i koszty żądanej polityki

`GgufHeaderInspection::inspect_payload(self, requested, allowances)` konsumuje
dokładnie zachowanego właściciela nagłówka. `requested` ma typ
`basal_gpu::Precision`, a `GgufInspectionAllowances` wymaga dwóch publicznych
pól u64: `remaining_resident_bytes` i `remaining_staging_bytes`. Nie ma Default;
zero nie oznacza braku limitu. Równość z limitem jest dozwolona. Allowances
służą porównaniu kosztów, nie utrzymują rezerwacji pamięci.

Wynik `GgufPayloadInspection` zachowuje cały właściciel nagłówka, ten sam
deskryptor, Content, początkowy stamp, token zobowiązania i rzeczywiste
companions wraz z kalibracją. Prywatne pola przechowują także żądaną Precision,
zaobserwowany `[u8;32]` digest i `GgufPolicyCosts`. Gettery to `companions()`,
`tensor_count()`, `requested_precision()`, `observed_gguf_sha256()` i `costs()`;
companions, digest i koszty są pożyczane. Wynik nie daje dostępu do File/Content,
mutowalnego config, serializacji, klonowania, rozkładania ani materializacji.
Błąd porzuca wszystkich konsumowanych właścicieli; upload jeszcze nie istnieje.

Kolejność odmów i odczytów jest stała:

1. Porównanie same-fd stamp z pierwotnym, checked koszty i zgodność sumy źródła,
   reprezentowalność rozmiarów w usize, odmowa resident, potem staging.
2. Fallible rezerwacja dokładnie żądanych 1 048 576 bajtów scratch oraz osobnego
   stringa nazwy 128 bajtów. Żądana pojemność nie jest gwarancją RSS ani OOM.
3. Stamp przed hashem, seek tego samego File do zera i SHA-256 dokładnie
   pierwotnej długości: nagłówek, luki, padding, tensory i końcowe bajty.
   Odczyt 0 przed pierwotną długością odmawia; dodatnie częściowe odczyty są
   kontynuowane. Jednobajtowa próba wzrostu nie jest hashowana ani zachowywana.
   Po stampie końcowym digest musi zgadzać się z deklaracją.
4. Stamp przed skanem, następnie odczyt każdego zachowanego zakresu w kolejności
   trzech globalnych ról, warstw rosnąco, dziewięciu wag i siedmiu bias iff bias.
   Każdy fragment mieści się w tym samym scratch, jest wyrównany do szerokości
   skalaru i wypełniany w całości przed dekodowaniem. Wiersz szerszy niż scratch
   nie powoduje dodatkowej alokacji. Odczyty i próby EOF ponawiają Interrupted.
5. Zgodność rzeczywistej sumy skanu, próba wzrostu przy pierwotnym EOF,
   stamp po skanie i stamp bezpośrednio przed sukcesem. Scratch jest zwalniany.

Każdy stamp porównuje się z pierwotnym; nie przyjmuje nowego stanu pliku.
Błędy wracają bezpośrednio, bez maskowania pierwszej odmowy dodatkowym stampem.
Skan obejmuje wyłącznie zakresy tensorów; padding nie jest interpretowany jako
float. Nie powstaje bufor pliku, tensora ani wiersza, mmap ani drugie parsowanie.

### Skończoność i konwersja

Little-endian F16/BF16 dekoduje się przez `from_le_bytes(...).to_f32_const()`,
a F32 przez `f32::from_le_bytes`. Nie ma rzutowań wskaźników ani unsafe.
Najpierw odrzuca się NaN/Inf wejścia, potem nieskończony wynik rzeczywistej
konwersji. Zwracana nazwa i zerowy u64 indeks dotyczą pierwszej wadliwej roli
i pierwszego skalaru w kolejności płaskiego storage źródła, przed permutacją Q/K.

| Żądana Precision | Role poza głowicą | `output.weight` |
|---|---|---|
| F16 | Matryce, normy i obecne bias do F16 przez `half::f16::from_f32_const` | Typ źródłowy M; tylko kontrola skończoności |
| Bf16 | Matryce, normy i obecne bias do BF16 przez `half::bf16::from_f32_const` | Typ źródłowy M; tylko kontrola skończoności |
| F32 | Zachowanie typu źródłowego każdej roli: matryce M, wektory F32 | Typ źródłowy M; tylko kontrola skończoności |

To jawna programowa polityka scalar round-to-nearest, ties-to-even. Akceptuje
skończone zaokrąglenia i underflow, także signed zero i subnormale. Sprawdza
skończoność faktycznego wyniku half, również overflow wskutek zaokrąglenia.
Nie przepisuje danych. Nie używa dotychczasowego `Precision::storage()`.
Przyszły materializer musi konsumować tego właściciela i używać tej samej
zamrożonej polityki scalar; nie może podmienić Precision, użyć niejawnego bulk
cast ani zwęzić głowicy. Poszerzanie do obliczeń F32 jest odrębnym zadaniem.
Syntetyczne bias są zerami w docelowym typie wektorów i nie mają bajtów źródła.

### Koszty i granice rachunku

`GgufPolicyCosts` ma cztery gettery u64: `source_tensor_bytes()`,
`resident_weight_bytes()`, `max_assembly_group_bytes()` i
`two_copy_staging_bytes()`. Wszystkie działania add/mul są checked. Przy
oznaczeniach powyżej i m=w(M):

```text
Ehead = V*H
Elayer_matrix = 2*H*H + 2*H*K + 3*H*I
Ematrix = 2*V*H + L*Elayer_matrix
Enorm = (2*L+1)*H
B = 3*H + 2*K + 2*I
present = bias ? L*B : 0
zero = bias ? 0 : L*B
S = m*Ematrix + 4*(Enorm+present)
Resident(F16/Bf16) = m*Ehead + 2*(V*H+L*Elayer_matrix+Enorm+present+zero)
Resident(F32) = S + 4*zero
```

S zgadza się z zachowaną sumą nagłówka przed alokacją scratch i z rzeczywistą
sumą zakresów po skanie. Resident liczy głowicę źródłową i każdy końcowy bias
dokładnie raz; nie dubluje projekcji i ich konkatenacji jako trwałych wag.
Nie obejmuje compute, KV, RoPE, readout, gather scratch ani narzutu urządzenia.

Dla t=2 przy F16/Bf16, inaczej m, oraz u=2 przy F16/Bf16, inaczej 4,
Amax jest maksimum dokładnych końcowych grup:

| Grupa | Bajty |
|---|---|
| Embedding / głowica / każda norma osobno | V*H*t / V*H*m / H*u |
| QKV / O | (H+2*K)*H*t / H*H*t |
| gate+up / down | 2*I*H*t / H*I*t |
| Bias QKV / O / gate+up / down, obecne albo zerowe | (H+2*K)*u / H*u / 2*I*u / H*u |

`two_copy_staging_bytes = 2*Amax + 1 048 576` jest **PROWIZORYCZNYM planem
inżynierskim dwóch kopii**, sprawdzanym względem
`min(2 147 483 648, remaining_staging_bytes)`, po sprawdzeniu
`Resident <= remaining_resident_bytes`. Inspekcja alokuje tylko scratch i lookup,
nie Amax. Plan nie dowodzi rzeczywistego szczytu materializacji ani admission.
Rzeczywisty CPU staging może nakładać 3A. Przyszły uploader musi policzyć każdą
faktyczną kopię, właścicieli pending i zachowanych po błędzie oraz ponownie
uzyskać utrzymywane allowances przed alokacją modelu. Nie ma gwarancji całego
host RAM, RSS ani ochrony przed OOM; wcześniejszy heap semantyki/tokenizera
i nakładające się temporaries pozostają poza tym rachunkiem.

Nowe odmowy to `GgufDigestMismatch` (expected/actual po 32 bajty, drukowane
wyłącznie jako 64 cyfry hex), `NonFinitePayload` (ograniczona nazwa i indeks),
`NonFiniteConversion` (także Precision), `ResidentBudget` (required/remaining)
i `StagingBudget` (required/effective_limit, jawnie plan prowizoryczny).
Zachowane błędy I/O/alokacji/arytmetyki/źródła utrzymują przyczyny i ograniczone
Display/Debug, także dla wariantów utworzonych zewnętrznie. Przedwczesny EOF
lub wzrost ma osobny statyczny opis fazy w `SourceChanged`.

REQ009 pozostaje częściowe. Zaobserwowany digest nie uwierzytelnia źródła,
nie wymusza lease ani nie dowodzi niezmienności: racing/restored writes mogą
ominąć stampy, a hash sprzed skanu nie certyfikuje późniejszych odczytów.
Zobowiązanie trusted-local no-writer obowiązuje przez każdy późniejszy odczyt
i publikację. Przyszłe odczyty wymagają własnych stampów i końcowego pełnego
rehashu tego samego deskryptora przed publikacją.

Brak nadal ilościowego admission całego heap semantycznego, utrzymywanych
rezerwacji host/resident/staging, trwałej tożsamości źródła wykorzystującej pełny
warunkowo zachowany zbiór wariantów kalibracji, bramki publikacji porównującej
dokładny binding z faktycznie
wybraną logiczną tożsamością i przenoszącej semantykę z zachowanym ostrzeżeniem,
materializera i dowodu kopii/completion/error retention dla uploadu, readout,
owned Engine/RequestPlanner/fork, CLI i runtime. API nie jest loaderem ani
gotowością do alokacji. Nie dostarcza dowodu zgodności bitowej, FP32, jakości,
pamięci, wydajności ani zachowania runtime GPU/Metal modelu.

`GgufPayloadInspection::bind_served_identity(self, ServedIdentity<'_>)` konsumuje
ten sam wynik payloadu. Porównuje pierwotny stamp na wejściu, mutuje rzeczywiste
companions przez powyższą metodę i ponownie porównuje pierwotny stamp przed
sukcesem. Zwraca tego samego właściciela File/Content/stamp/token/companions,
zamrożonej Precision, digestu i kosztów. Nie odczytuje ani nie hashuje payloadu,
nie otwiera plików, nie rekonstruuje tokenizera i nie odświeża stampu.
Core error wraca od razu jako `GgufInspectionError::IdentityBinding`, bez
maskowania dodatkowym stampem. Każdy błąd wrappera porzuca konsumowanego
właściciela, także zmiana pliku wykryta po udanym core bindingu. Top-level
Display/Debug nowego wariantu ma stały tekst; `source()` prowadzi do błędu core.
Udany binding nadal nie daje trwałego fingerprintu, admission ani zgody na
publikację. Obowiązek caller’a oraz stampy późniejszych odczytów i końcowy pełny
rehash tego samego deskryptora pozostają wymagane.
