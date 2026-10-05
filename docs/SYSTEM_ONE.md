# Zgodność API

Serwer ma dwa endpointy decyzji o tej samej semantyce modelu i różnych
konwencjach żądań oraz odpowiedzi:

- `POST /v1/systemone`: kontrakt TypeSafe System One. Na nim opiera się
  integracja z aplikacjami.
- `POST /v1/basal`: konwencje `Server.decide` upstream basal v1.5.0, dla
  klientów napisanych pod serwer upstream.

`GET /v1/models` zwraca listę TypeSafe `ModelMetadataList` (`name`,
`description`, `release_date`) z polami upstream `mode` i `early_exit`
(pusta lista: runtime nie stosuje wczesnego wyjścia). `release_date` ustawia
`basal serve --release-date`.

## /v1/systemone

### Wejście

| Pole | Kontrakt |
|---|---|
| `model` | wymagany; nazwa serwowanego modelu, inna kończy się błędem `unknown_model` |
| `state` | string, obiekt lub tablica; wewnątrz dowolny JSON |
| `questions` | mapa ID pytania na pytanie; ID wiąże odpowiedź i nie trafia do promptu |
| `instructions` | string, obiekt, tablica albo brak/null |
| `type` | `choice`, `noul`, `score`; z basal-1.5 także `multi`, `act`; inny lub brak to błąd walidacji |

- `choice.criteria`: mapa klucz → opis (string, obiekt, tablica lub null;
  null oznacza opcję opisaną kluczem), 2–255 opcji.
- `noul.criteria`: opcjonalne `true` i `false`; puste wartości (`""`, `{}`,
  `[]`) są zachowane, nie zastępowane domyślnymi.
- `score.criteria`: uporządkowana tablica poziomów 0..n-1 (2–10); jak w
  upstream także mapa nazwa → opis (odpowiedź po nazwach) i alias `levels`.

Błędy walidacji mają postać TypeSafe `{"detail": [{"loc", "msg", "type"}]}`
ze statusem 422.

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

Confidence według TypeSafe, liczone z końcowego rozkładu po kalibracji:

```text
choice:  (max(p) - 1/n) / (1 - 1/n)
score:   m = indeks najbardziej prawdopodobnego poziomu
         uniform_mad = Σ |i - (n-1)/2| / n
         max(0, 1 - Σ p[i]·|i - m| / uniform_mad)
```

Remisy rozstrzyga pierwszy klucz (jak `max` w Pythonie).

### Różnice względem upstream

| Przypadek | Upstream | /v1/systemone |
|---|---|---|
| `instructions: null` | tekst `null` w prompcie | brak instrukcji |
| Noul z pustym opisem | domyślne Tak/Nie | podana wartość |
| Nieznany lub brak `type` | `choice` | błąd walidacji |
| Choice/Score z jedną opcją | błąd 2..10 | błąd `unsupported_single_option` |
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

Odczyt jak `to_items` upstream: pole `model` ignorowane, brak lub nieznany
`type` to `choice` (nieznany typ bez kalibracji), `instructions: null` jako
tekst `"null"`, puste opisy noul zastąpione Tak/Nie, 2–10 opcji. Odpowiedzi
jak upstream: `probabilities` i `confidence = max(p)` dla każdego typu,
legenda score z tekstami opcji. Każdy błąd, także niepoprawny JSON, to 422
`{"error": "..."}`. Dla modelu basal-1.0 obowiązują reguły v1.5.0 (`type:
"multi"` to multi).

Na 34 żądaniach referencji basal-1.5 błędy występują w tych samych miejscach
co w upstream, a 55/55 pytań ma te same pola i decyzje (maks. różnica
liczbowa 0,0016, FP16 wobec FP32 upstream). `basal export` zapisuje odpowiedź
tego endpointu jako `answer_upstream`, a `basal compare` zestawia ją z
odpowiedzią upstream.

## Źródła

- [TypeSafe HTTP API](https://docs.typesafe.ai/api), [OpenAPI](https://api.typesafe.ai/openapi.json);
  migawka w [contracts/typesafe/openapi.json](../contracts/typesafe/openapi.json)
  (`info.version` 0.2.0, SHA-256 `a191f8a7…c0360d5`).
- [Confidence](https://docs.typesafe.ai/confidence), [Score](https://docs.typesafe.ai/primitives/score).
- [basal v1.5.0](https://github.com/rkinas/basal/tree/cd63c083636f94ce492ec1e6cc1e02144723b9b5):
  `basal/server.py`, `basal/prompt.py`, `basal/facts.py`.
