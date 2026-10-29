# Zgodność z OpenAI Decisions API

`basal serve` udostępnia zawsze `POST /v1/decisions` w kształcie
[OpenAI Decisions API](https://developers.openai.com/api/reference/resources/decisions).
Dzięki temu oficjalne SDK OpenAI (`client.decisions.create(...)`, Python ≥ 3.26.0,
JavaScript ≥ 7.30.0) mogą wysyłać pytania do modeli basal. Wystarczy wskazać
serwer jako `base_url`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
decision = client.decisions.create(
    model="basal-2.0-mini",
    input="I was charged twice for my order.",
    questions=[
        {"type": "choice", "name": "route", "instructions": "Which team should handle this?",
         "choices": [{"value": "billing"}, {"value": "technical"}, {"value": "shipping"}]},
        {"type": "predicate", "name": "refund", "instructions": "Does the customer ask for a refund?"},
    ],
)
```

Endpoint tłumaczy żądanie na kontrakt TypeSafe [`/v1/systemone`](SYSTEM_ONE.md).
Potem przechodzi tę samą ścieżkę co `/v1/systemone`: admission, limity,
planowanie, kalibrację i model. Odpowiedź TypeSafe jest zamieniana na obiekt
`Decision` OpenAI. Nie ma osobnej numeryki: przetłumaczone żądanie wysłane do
`/v1/systemone` daje te same prawdopodobieństwa. Wyniki modeli basal nie są
wynikami `gpt-6-luna`; zgodność dotyczy kontraktu HTTP, nie jakości odpowiedzi.

Serwer nie ma uwierzytelniania; nagłówek `Authorization` wysyłany przez SDK jest
ignorowany. Endpoint nie zależy od `systemone_dialect`.

## Żądanie

| Pole | Obsługa |
|---|---|
| `model` | wymagany; nazwa serwowanego modelu basal. Inna nazwa, także `gpt-6-luna`: 404 `model_not_found`. |
| `input` | string albo niepusta tablica elementów (niżej) |
| `questions` | 1–400 pytań |
| `prompt_cache_key`, `prompt_cache_options`, `safety_identifier` | sprawdzany typ i długość, poza tym ignorowane |
| inne pola | 400 `Unrecognized request argument supplied` |

Typy pytań:

| OpenAI | basal (TypeSafe) | Tłumaczenie |
|---|---|---|
| `predicate` | `noul` | `instructions` |
| `choice` | `choice` | `choices[{value, description}]` → `criteria {value: description \| null}`, 2–255 opcji |
| `score` | `score` | `levels[{label, description}]` → uporządkowane `criteria {label: description \| null}`, 2–10 poziomów |

`instructions` jest wymagane w każdym pytaniu. Może być stringiem albo tablicą
części `input_text`, `input_image` i `input_json`. Pytanie dostaje wewnętrzny
identyfikator `q0`, `q1`, …, który nie trafia do promptu. `name` jest tylko
zwracane w odpowiedzi, więc może się powtarzać albo być pominięte (`null`).
Wartość `choice` może być stringiem lub booleanem. Model widzi jej tekst, a
odpowiedź zachowuje typ. Ponieważ klucz basal jest tekstem, `true` i `"true"`
w jednym pytaniu dają 400. Etykiety `score` muszą być unikatowe.

### `input`

String staje się stanem (`state`) TypeSafe bez zmian. Tablica staje się tablicą
JSON w stanie, z jednym elementem na każdy element wejścia:

| Element | Wartość w stanie |
|---|---|
| wiadomość `user` | `{"role": "user", "content": …}`; części `input_text`, `input_image`, `input_json` |
| wiadomość `developer`, `system` | `{"role": "developer", "content": …}`; tylko `input_text`. `system` jest wiadomością developera, jak w OpenAI |
| wiadomość `assistant` | `{"role": "assistant", "content": …}`; `output_text` → tekst, `refusal` → `{"refusal": …}` |
| `function_call` | `{"type", "name", "namespace"?, "arguments", "call_id"}` |
| `custom_tool_call` | `{"type", "name", "namespace"?, "input", "call_id"}` |
| `function_call_output`, `custom_tool_call_output` | `{"type", "call_id", "output"}`; `output` jak `content` z częściami `input_text`, `input_image` |

Treść (`content`, `output`, `instructions`) to string albo tablica części:
`input_text` i `output_text` dają string, `input_json` daje swoje `data`,
a `input_image` daje referencję obrazu. Tablica z jedną częścią tekstową
staje się stringiem. Pola `id`, `status`, `phase`, `detail` i
`prompt_cache_breakpoint` są pomijane. Wynik wywołania narzędzia musi
wskazywać `call_id` wcześniejszego wywołania tego samego rodzaju; inaczej 400.
Stan i instrukcje podlegają limitom `/v1/systemone`: najwyżej 64 poziomy
zagnieżdżenia JSON i tylko skończone liczby.

### Obrazy

`input_image.image_url` musi być URL-em `data:image/png;base64,…` albo
`data:image/jpeg;base64,…`. Serwer nie pobiera adresów `https://`; takie
żądanie dostaje 400 `unsupported_value`. Obraz staje się assetem TypeSafe z
referencją `[image:<16 znaków hex SHA-256 danych base64>]`. Ten sam obraz użyty
kilka razy jest jednym assetem. Obowiązują limity i wymagania obrazów
`/v1/systemone`: backend Omni z mediami, PNG/JPEG do 16 MiB, boki 280–1568 px,
najwyżej osiem różnych obrazów. Nagłówek `TypeSafe-Danger-Zone-Images` nie jest
wymagany. `detail` jest sprawdzany i ignorowany; obrazy przechodzą budżet `default`.

Na tej trasie nie ma kompresji (`Content-Encoding`) ani multipart. Limit ciała
wynosi 32 MiB; ciała powyżej 2 MiB przechodzą admission dużych żądań.

## Odpowiedź

```json
{
  "model": "basal-2.0-mini",
  "answers": [
    {"type": "choice", "name": "route", "choice": "billing",
     "probabilities": [{"value": "billing", "probability": 0.91}, …], "confidence": 0.87},
    {"type": "predicate", "name": "refund", "probability": 0.12}
  ],
  "usage": {"input_tokens": 312, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 312}
}
```

- `answers` mają kolejność pytań.
- `predicate.probability` to `noul` TypeSafe, czyli P(true).
- `choice.probabilities` mają kolejność `choices`.
- `score.score` i `confidence` to wartości TypeSafe.
  `score.probabilities` zawierają indeks poziomu (`value`), `label`
  i prawdopodobieństwo.
- Pola kalibracji, thinking i evidence TypeSafe nie są przenoszone.
  Serwer nie zwraca odpowiedzi `refusal`.
- `input_tokens` to liczba tokenów promptów basal. Wszystkie gałęzie
  i porządki opcji są liczone, więc wartość różni się od liczenia OpenAI.
- `output_tokens` i `reasoning_tokens` to tokeny thinking (bez thinking: 0).
- Liczniki cache są zawsze 0.
- Odpowiedź ma nagłówki czasu `x-basal-*` jak `/v1/systemone`.

## Błędy

Błędy mają kopertę OpenAI `{"error": {"message", "type", "param", "code"}}`:

| Status | `type` / `code` | Kiedy |
|---|---|---|
| 400 | `invalid_request_error` | niepoprawny JSON, schemat, walidacja basal; `param` wskazuje np. `questions[2].choices` |
| 400 | `invalid_request_error` / `unsupported_value` | żądanie, którego ten runtime nie obsługuje (np. URL obrazu, obrazy bez backendu Omni) |
| 400 | `invalid_request_error` / `context_length_exceeded` | prompt dłuższy niż limit pozycji modelu (polityka `context_overflow`) |
| 404 | `invalid_request_error` / `model_not_found` | nieznany model |
| 413 | `invalid_request_error` | ciało ponad 32 MiB; inne błędy odczytu ciała mają status warstwy HTTP i tę samą kopertę |
| 500 | `server_error` | błąd silnika lub zatrzymany worker |
| 503 | `server_error` / `server_overloaded` | przeciążenie admission, także limitu dekoderów i dużych ciał; z `Retry-After: 1`. SDK OpenAI ponawia 5xx |

Błędy walidacji basal zachowują komunikat silnika, który może wymieniać
wewnętrzne pola TypeSafe (`criteria`, `q0`). Błędy obrazów wykrywane przez
adapter assetów TypeSafe (base64, wymiary, liczba obrazów) mają `param: "input"`,
także gdy obraz jest w `instructions`; komunikat podaje referencję `[image:…]`.

## Poza zakresem

- Pobieranie obrazów z URL, `file_id`, pliki i audio.
- Prompt caching OpenAI. Cache prefiksów basal działa niezależnie i nie
  jest raportowany w `usage`.
- Batch API z `url: "/v1/decisions"`. Batch obsługuje natywne kontrakty
  ([BATCHES.md](BATCHES.md)).
- `GET /v1/models` pozostaje listą TypeSafe, a nie listą OpenAI `{"object": "list", "data"}`.
