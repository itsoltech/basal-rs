# Deklaracja GGUF i snapshot plików towarzyszących

`basal_core::weights::GgufCompanionSnapshot::read(&Path)` odczytuje lokalną
deklarację `gguf-source.json` i pięć stałych plików towarzyszących. Zwraca
sprawdzoną deklarację pochodzenia oraz dokładne bajty plików, których SHA-256
zgadza się z deklaracją. API jest niezależne od backendu.

To ograniczony etap przygotowania źródła. Nie otwiera ani nie parsuje GGUF,
nie wybiera źródła wag w CLI lub runtime i nie dopuszcza modelu do wykonania.
`declared_gguf_sha256()` jest jawnie **niezweryfikowaną deklaracją**.

## Publiczne API

| Element | Znaczenie |
|---|---|
| `GgufCompanionSnapshot::read(directory)` | Jedyna operacja odczytu; sześć stałych nazw, bez przeglądania katalogu i wyszukiwania `.gguf` |
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
`Display` i `Debug` nie wypisują wartości deklaracji ani zawartości companions.
Oryginalna szczegółowa przyczyna parsera JSON może zawierać niezaufane dane
deklaracji; należy traktować ją zgodnie z tym charakterem przy logowaniu.

Limity bajtów nie gwarantują fizycznej dostępności RAM ani ochrony przed OOM.
Nie ograniczają amplifikacji alokacji podczas parsowania JSON deklaracji ani
późniejszego parsowania konfiguracji lub tokenizera. Deklaracja jest parsowana
pod limitem wejściowym 64 KiB; duże pliki companion pozostają surowymi bajtami.

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
zastosować dotychczasową semantykę Llama config, `basal.json`, dokładnego
template, BOS i kodów/token IDs oraz odrzucić nieznane semantyki. Kalibracja
wymaga zachowania istniejącego wyboru, walidacji i provenance, także dla
jawnego override. Ta operacja nie wyszukuje kalibracji ani nie rekonstruuje
`ModelManifest` lub tokenizera. Loader, frozen source identity, plan wykonania,
ścieżki odczytu głowicy i upload GPU oraz pomiary numeryczne pozostają osobnymi
warunkami dalszej obsługi GGUF.
