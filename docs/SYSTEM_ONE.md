# Zgodność API

Serwer domyślnie udostępnia dwa kontrakty decyzji:

- `POST /v1/systemone`: kontrakt TypeSafe System One. Na nim opiera się
  integracja z aplikacjami.
- `POST /v1/basal`: konwencje `Server.decide` upstream basal v2.0.0, dla
  klientów napisanych pod serwer upstream.

`basal serve --systemone-dialect upstream` (globalna flaga) lub YAML
`systemone_dialect: upstream` przełącza `/v1/systemone` na rzeczywisty alias
`/v1/basal`: ten sam handler, natywny dekoder JSON, limity i koperta błędów.
Oficjalny klient i most upstream mogą pominąć `model`; wybierany jest
`default_model`, domyślnie pierwszy skonfigurowany model. Jawne `model`
nadal wybiera jeden z wielu serwowanych modeli. Dialekt nie zależy od treści.
Flaga CLI ma pierwszeństwo nad YAML. Domyślne `typesafe` zachowuje kontrakt
opisany poniżej, w tym dekodowanie kompresji i multipart.
Natywny alias przyjmuje ten sam surowy JSON co `/v1/basal`; nie korzysta
z adaptera kompresji, multipart ani `assets` TypeSafe. Limit natywny wynosi
2 MiB, a gdy serwer udostępnia media: 256 MiB ze wspólnym admission dużych ciał.
Metryki zachowują faktyczną ścieżkę endpointu oraz rozwiązany model.
Wybór dotyczy online HTTP; zapisany kontrakt rekordów Batch API nie zmienia się.

`GET /v1/models` zwraca listę TypeSafe `ModelMetadataList` (`name`,
`description`, `release_date`) z polami upstream `mode` i `early_exit`
(pusta lista: runtime nie stosuje wczesnego wyjścia). `release_date` ustawia
`basal serve --release-date`.

Metadane zawierają również `prompt_format`, efektywne `orders`,
`temperature_per_prim` (puste `{}` przy wyłączonej kalibracji), `layout` tylko
gdy niekanoniczny oraz `calibration_warning`, gdy kalibracja jest niezgodna.
Ostrzeżenie jest logowane przed udostępnieniem modelu; API nie ujawnia pełnej
ścieżki pliku kalibracji ani lokalnej tożsamości modelu. Mismatch jawnego pliku
z modelem/rewizją ostrzega i zachowuje certyfikaty; `--strict-calibration`
odmawia startu. Niezgodność layoutu/porządku usuwa certyfikaty i dopasowane
fallbacki. Opcje i pierwszeństwo: [README](../README.md#serwer).

## /v1/systemone

### Wejście

| Pole | Kontrakt |
|---|---|
| `model` | wymagany; nazwa serwowanego modelu, inna: 400 `Unknown model: …` |
| `state` | string, obiekt lub tablica; wewnątrz dowolny JSON; `null` jak brak pola |
| `questions` | mapa ID pytania na pytanie, co najmniej jedno; ID wiąże odpowiedź i nie trafia do promptu |
| `instructions` | string, obiekt, tablica albo brak/null |
| `type` | `choice`, `noul`, `score`; z basal-1.5 także `multi`, `act`; brak: 422, inny: 400 |

- `choice.criteria`: mapa klucz → opis (string, obiekt, tablica lub null;
  null oznacza opcję opisaną kluczem), 1–255 opcji. Przy jednej opcji
  odpowiedź bez modelu: prawdopodobieństwo 1, confidence 1.
- `noul.criteria`: opcjonalne `true` i `false`; puste wartości (`""`, `{}`,
  `[]`) są zachowane, nie zastępowane domyślnymi. Noul bez `instructions` i
  bez `criteria` (brak lub null) to błąd 400.
- `score.criteria`: uporządkowana tablica poziomów 0..n-1 (1–10; jeden
  poziom: score 0, confidence 1, bez modelu); jak w upstream także mapa
  nazwa → opis (odpowiedź po nazwach) i alias `levels`.

### Obrazy TypeSafe

`/v1/systemone` przyjmuje `assets` zgodnie z kontraktem obrazów TypeSafe.
Każdy obraz ma niepustą referencję, używaną w wartościach `state`,
`instructions` lub `criteria`, także w środku zdania i w zagnieżdżonym JSON.
Klucze obiektów pozostają tekstem. Każde wystąpienie referencji umieszcza
obraz w danym miejscu promptu; asset użyty tylko w jednym pytaniu nie
trafia do pozostałych. Permutacja kryteriów zachowuje powiązania obrazów.
Legenda score zawiera oryginalne referencje.

Wymagany jest nagłówek `TypeSafe-Danger-Zone-Images: true`. Można wysłać:

- JSON z `assets: {"photo-id": {"type": "image", "format": "png",
  "data_base64": "..."}}`. `type` jest opcjonalne; `format` to png/jpg/jpeg.
  Base64 ma standardowy alfabet i padding, bez białych znaków i prefiksu `data:`.
- Multipart: pierwsza część `request` ma typ `application/json`; asset
  wskazuje plik przez `part`. Nazwy części muszą być unikatowe, a każda
  część obrazu musi należeć do jednego assetu. `part` ma pierwszeństwo
  przed `data_base64`.

```sh
curl localhost:8000/v1/systemone \
  -H 'TypeSafe-Danger-Zone-Images: true' \
  -F 'request={"model":"basal-omni-1.0-mini","state":{"photo":"camera-frame-01"},"assets":{"camera-frame-01":{"type":"image","part":"photo"}},"questions":{"visible":{"type":"noul","instructions":"Czy na zdjęciu jest człowiek?"}}};type=application/json' \
  -F 'photo=@photo.png;type=image/png'
```

Wybierz nazwę modelu serwowanego przez tę instancję. Obrazy wymagają backendu
Omni z włączonymi mediami. Model tekstowy lub backend bez vision zwraca
`422 {"error":"unsupported","detail":[...]}`. Tak samo odpowiada każdy
obecny model na rozpoznany typ `box`: basal-rs nie implementuje odczytu
współrzędnych. Żądanie jest odrzucane w całości, bez częściowych odpowiedzi.

Obecne ograniczenia adaptera:

- Statyczne PNG/JPEG, plik do 16 MiB, każdy bok 280–1568 pikseli.
  Format jest sprawdzany na podstawie bajtów; animowane PNG są odrzucane.
- Do ośmiu różnych obrazów. Budżet rozwinięcia wynosi 32 MiB tekstu
  i bajtów źródłowych obrazów po dekodowaniu base64, z powtórzeniami
  i użyciem wspólnego stanu przez każde pytanie. Przekroczenie możliwości
  runtime'u zwraca `422 unsupported`; nie dopasowujemy obrazu przez obcięcie
  lub pominięcie referencji. TypeSafe stosuje własne przetwarzanie obrazów,
  więc nie deklarujemy identycznych granic jego rozmiaru po przetworzeniu.
- Z assetami obsługiwane są `choice`, `noul` i `score`. Natywne typy zadań,
  `facts: "auto"`, `evidence` i łączenie `assets` z natywnym `media` zwracają
  `422 unsupported`. Natywny kontrakt mediów nadal działa przez `/v1/basal`.
- Obrazy przechodzą istniejący procesor Omni z budżetem `default`.
  Odpowiedzi zachowują `calibration: "text"` (lub `unvalidated`) oraz
  `usage.media`; nie oznacza to potwierdzonej kalibracji obrazów.

Całe ciało `/v1/systemone` ma limit 32 MiB, także po dekompresji.
Obsługiwane jest jedno `Content-Encoding`: gzip/x-gzip lub zstd z oknem
do 8 MiB. Za duże ciało daje 413, niepoprawna kompresja lub multipart 400,
a nieobsługiwane kodowanie 415. Obowiązują też skończone liczby JSON
i maksymalnie 64 poziomy obiektów/tablic w treści stanu i pytań.
Nagłówek długiego kontekstu TypeSafe nie zwiększa limitu pozycji modelu;
przekroczenie tego limitu nadal daje `422 unsupported`.

SDK basal-rs i endpoint `/v1/basal` zachowują dotychczasowy kontrakt.
Ta zmiana umożliwia przesłanie kontraktu obrazów z klienta TypeSafe;
nie deklaruje zgodności wyników różnych modeli z Jev.

Błędy mają postać i statusy API TypeSafe (sprawdzone na 38 przypadkach
brzegowych, [raport](../reports/typesafe-api-edge-cases/README.md)):

- 422, błąd schematu: `{"detail": [{"type", "loc", "msg", "input", "ctx"}]}`
  jak w pydantic. `loc` zawiera typ pytania po jego ID
  (`["body", "questions", "q", "score", "criteria", 1, "str"]`), wartość
  spoza unii string/obiekt/tablica daje trzy błędy (`string_type`,
  `dict_type`, `list_type`), niepoprawny JSON daje `json_invalid`.
- 400, poprawny schemat, ale żądanie odrzucone: `{"detail": "komunikat"}`
  (choice bez opcji lub z ponad 255, score z ponad 10 poziomami, noul bez
  instrukcji i kryteriów) albo `{"detail": {"error_type": "api_usage_error",
  "message": ...}}` (nieznany model lub typ pytania). Przy błędach obu
  rodzajów odpowiedzią jest 422.
- 529 z `Retry-After: 1`, gdy w toku jest więcej niż `max_inflight` żądań
  (TypeSafe odpowiada 529, gdy jest przeciążone). SDK TypeSafe ponawia 408,
  429 i 5xx z wykładniczym odczekiwaniem.
- 422 `{"error": "unsupported", "detail": [...]}`: żądanie poprawne, którego
  ten runtime nie obsługuje (np. `evidence` w modelu bez głowicy dowodów), w
  tym prompt dłuższy niż limit pozycji modelu (`context_length_exceeded`,
  komunikat zawiera „maximum context length”; sprawdzane przed forwardem).
  `/v1/basal` zwraca wtedy 422 `{"error": ...}` z tym samym komunikatem.

### Odpowiedź

| Typ | Pola |
|---|---|
| `noul` | `type`, `noul` = P(true) |
| `choice` | `type`, `choice`, `probabilities` po kluczach opcji, `confidence` |
| `score` | `type`, `score` = Σ i·p[i], `probabilities` i `legend` po indeksach, `confidence` |
| koperta | `model`, `answers` po ID pytań, `usage` |

`legend` zachowuje oryginalne wartości JSON kryteriów. `usage` zawiera
`input_tokens` (tokeny promptów obu porządków), `output_tokens` = 0,
`questions`, `branches` i `latency_ms` (czas żądania w serwerze).
`longest_prompt_tokens` podaje najdłuższy prompt w obu formatach modelu.
`branches` liczy początkowe gałęzie logiczne (jedną na etykietę `multi`),
bez mnożenia przez liczbę porządków; `second_order_branches` przy `orders: auto`
liczy dodatkowy drugi porządek. Kontynuacje zadań zachowują własne doliczanie
tokenów i gałęzi oraz maksimum długości. Przy aktywnym torze długich żądań
`usage.lane` to `main` lub `long` według toru, który faktycznie wykonał żądanie,
także po fallbacku do głównego toru i na replikach. Pola czasu pozostają dostępne.

Confidence według TypeSafe, liczone z końcowego rozkładu po kalibracji:

```text
choice:  (max(p) - 1/n) / (1 - 1/n)
score:   m = indeks najbardziej prawdopodobnego poziomu
         uniform_mad = Σ |i - (n-1)/2| / n
         max(0, 1 - Σ p[i]·|i - m| / uniform_mad)
```

Remisy rozstrzyga pierwszy klucz (jak `max` w Pythonie). TypeSafe zaokrągla
prawdopodobieństwa, score i confidence do dwóch miejsc po przecinku, a
basal-rs zwraca pełną precyzję.

Rozszerzenia przyjmowane przez basal-rs, które API TypeSafe odrzuca: typy
`multi` i `act`, dodatkowe pola (`facts`, `evidence`, `option_keys`),
kryteria choice jako lista nazw, kryteria score jako mapa. Klient pisany pod
TypeSafe ich nie wysyła, więc nie zmieniają odpowiedzi na jego żądania.

### Różnice względem upstream

| Przypadek | Upstream | /v1/systemone |
|---|---|---|
| `instructions: null` | tekst `null` w prompcie | brak instrukcji |
| Noul z pustym opisem | domyślne Tak/Nie | podana wartość |
| Nieznany lub brak `type` | `choice` | błąd (400 lub 422) |
| Choice/Score z jedną opcją | błąd 2..10 | prawdopodobieństwo 1, jak TypeSafe |
| Choice 11–255 opcji | błąd 2..10 | strategia grupowa ([opis](ARCHITECTURE.md#choice-11255)) |
| Confidence | `max(p)` dla każdego typu | wzory TypeSafe; Noul bez confidence |
| Legenda Score | tekst | oryginalne wartości JSON |

### Rozszerzenia basal-1.5

Nie są częścią TypeSafe System One; odpowiedzi mają format upstream.

- `multi`: dla każdej etykiety pytanie tak/nie (`"{instructions}\nCzy
  dotyczy: „{label}”?"` lub `"...\nDoes this apply: \"{label}\"?"`).
  Odpowiedź `{type, selected, probabilities, threshold, set_confidence}`;
  `threshold` (domyślnie 0,5), `max` i `min` sterują wyborem.
- `act`: odczyt jak `noul` (kryteria `true`/`false` lub `base: "noul"`) albo
  `choice`, następnie akcja o najmniejszym oczekiwanym koszcie (`costs` w
  pełnej postaci lub `{wrong, defer}`). Z `max_error` akcja poniżej
  certyfikowanego progu z `CALIBRATION.json` zostaje zastąpiona odroczeniem.
  Odpowiedź `{type, action, expected_costs, answer, probabilities,
  confidence, [max_error|refused], calibration}`.
- `facts: "auto"` (poziom żądania): fakty wyliczone ze stanu dopisane przed
  wykryciem języka; błąd upstream wraca jako 422 z tym samym komunikatem.
- `evidence` (w pytaniu): do 3 fragmentów stanu z offsetami znakowymi i
  prawdopodobieństwem; przy `facts` fragmenty tylko z oryginalnego stanu.
  `multi` z `evidence` to błąd walidacji; model bez głowicy dowodów zwraca
  błąd `unsupported`.

## /v1/basal

Odczyt jak `to_items` upstream: parser pytań ignoruje pole `model`, ale
router HTTP basal-rs używa go do wyboru modelu (bez niego: `default_model`;
nieznana nazwa: 422). Oryginalny serwer upstream ignoruje to pole. Brak lub nieznany
`type` to `choice` (nieznany typ bez kalibracji), `instructions: null` jako
tekst `"null"`, puste opisy noul zastąpione Tak/Nie, 2–10 opcji. Odpowiedzi
jak upstream: `probabilities` i `confidence = max(p)` dla każdego typu,
legenda score z tekstami opcji. Błędy żądania, także niepoprawny JSON, to 422
`{"error": "..."}`; rozpoznany OOM: 503, inne błędy silnika: 500.
Admission zachowuje 529 i `Retry-After`. Dla modelu basal-1.0 obowiązują reguły v1.5.0 (`type:
"multi"` to multi).

Na 34 żądaniach referencji basal-1.5 błędy występują w tych samych miejscach
co w upstream, a 55/55 pytań ma te same pola i decyzje (maks. różnica
liczbowa 0,0016, FP16 wobec FP32 upstream). `basal export` zapisuje odpowiedź
tego endpointu jako `answer_upstream`, a `basal compare` zestawia ją z
odpowiedzią upstream.

SDK tego endpointu dla Python, TypeScript/JavaScript, C# i Rust:
[SDK natywnego API](SDK.md).

## Layout, evidence i cache

`--layout canonical|qfirst|state_twice` lub per-model YAML `layout` nadpisuje
deklarację modelu; bez deklaracji używane jest canonical. Niekanoniczne layouty
obsługują PL/EN w prompt format 1. Media z niekanonicznym layoutem są odrzucane
z 422. Evidence nadal używa własnego promptu canonical. Dla bounded batch
evidence z niekanonicznym layoutem ma status `unmodelled` i jest odrzucane,
zamiast przyjmować nieudowodniony budżet pamięci.

Cache stanu może działać również dla innych layoutów: ponowne użycie wymaga
identycznego prefiksu tokenów. Skuteczności i wydajności cache w tych layoutach
nie zmierzono. Domyślne auto tau/band pozostają `0.6` i `[0.2, 0.8]`;
walidacja jest ściślejsza od upstream: skończone liczby w `[0,1]`, LO ≤ HI.

## Źródła

- [TypeSafe HTTP API](https://docs.typesafe.ai/api), [OpenAPI](https://api.typesafe.ai/openapi.json);
  migawka w [contracts/typesafe/openapi.json](../contracts/typesafe/openapi.json)
  (`info.version` 0.2.0, aktualizacja 2026-10-19,
  SHA-256 `ee599512…5c057c3`; [opis migawki](../contracts/typesafe/README.md)).
- [Obrazy](https://docs.typesafe.ai/images), [Box](https://docs.typesafe.ai/primitives/box).
- [Confidence](https://docs.typesafe.ai/confidence), [Score](https://docs.typesafe.ai/primitives/score).
- [basal v1.5.0](https://github.com/rkinas/basal/tree/cd63c083636f94ce492ec1e6cc1e02144723b9b5):
  `basal/server.py`, `basal/prompt.py`, `basal/facts.py`.
