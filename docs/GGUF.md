# Deklaracja GGUF i semantyka plików towarzyszących

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

Dalszy preflight musi zweryfikować sam GGUF: ograniczony nagłówek i inventory,
metadane, mapowanie, typy i zakresy tensorów, payload i jego finitość, digest
tego samego pliku, niezmienność źródła oraz jawne budżety pamięci. Osobno musi
porównać config z GGUF i zachować późniejsze granice ID żądań. API semantyczne
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
