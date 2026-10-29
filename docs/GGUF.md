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
probe preferowanego pliku. Nie ma skanowania ani hashowania wszystkich wariantów.

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

`identity_state()` zwraca `CalibrationIdentityState`: NotRequired dla Automatic
i Off, Checked dla Explicit z podaną rzeczywistą served identity oraz Deferred
dla Explicit bez niej, nawet przy braku fitted model. Checked nie jest obietnicą
braku ostrzeżenia ani uwierzytelnienia pochodzenia. Przyszły runtime musi wykonać
`calibration_provenance.check_identity(actual_served)` przed publikacją; wywołanie
nie zapamiętuje sukcesu. Attested origin nie zastępuje served identity.

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

Każdy rzeczywiście czytany plik kalibracji ma limit 64 MiB, najwyżej dwa odczyty
i razem 128 MiB. Companions z kalibracją mają wspólny limit 256 MiB, a deklaracja
osobno 64 KiB. Jeden licznik zaczyna od sprawdzonej sumy rzeczywistych długości
pięciu companions S; jego granica to `min(S.checked_add(128 MiB)?, 256 MiB)`.
Kalibracja używa tego samego czytnika i obciąża sumę przed rezerwacją; opłata za
zastąpionego kandydata nie znika. Łączne zachowane długości raw to najwyżej
256 MiB + 64 KiB, z możliwym narzutem capacity/allocatora oraz dodatkowymi
strukturami semantycznymi. Nie ma cichej kopii całych zachowanych buforów.
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
zamyka wyłącznie opisany etap companions/wybranej kalibracji. Nie definiuje
fingerprintu całego źródła ani trwałego inventory wszystkich wariantów kalibracji.
Późniejsza tożsamość musi wiązać również zastąpionego kandydata, wybrany digest,
tryb/protokół oraz rzeczywiście wykonane decyzje/probes, bez lokalnych ścieżek.

Przyszłe konstruktory Engine i RequestPlanner muszą przejąć zwalidowany tokenizer
wraz z manifestem. Obecny `Engine::fork` ponownie używa konstruktora ścieżkowego
i wymaga osobnego dostosowania. Nowa ścieżka nie może otwierać companions przez
`manifest.dir`, zachowując stare hashe. Ownership workerów, source lease/snapshot,
host/device/staging budgets, warianty, materializacja, CLI/runtime i kwalifikacja
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
rezerwacji host/resident/staging, trwałej tożsamości źródła i pełnego zbioru
wariantów kalibracji, sprawdzenia deferred provenance przed publikacją,
materializera i dowodu kopii/completion/error retention dla uploadu, readout,
owned Engine/RequestPlanner/fork, CLI i runtime. API nie jest loaderem ani
gotowością do alokacji. Nie dostarcza dowodu zgodności bitowej, FP32, jakości,
pamięci, wydajności ani zachowania runtime GPU/Metal modelu.
