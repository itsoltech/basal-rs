# OpenAI Decisions API: weryfikacja `POST /v1/decisions` (2026-10-29)

## Cel

Sprawdzić, czy oficjalne SDK OpenAI (`client.decisions.create`) działa z `basal serve`.
Sprawdzić też, czy endpoint `/v1/decisions` daje te same liczby co równoważne
żądanie `/v1/systemone`. Endpoint tylko tłumaczy kontrakt, nie zmienia obliczeń,
więc wynik musi być bitowo równy. Kontrakt endpointu opisuje
[docs/OPENAI_DECISIONS.md](../../docs/OPENAI_DECISIONS.md).

## Wersje i wejścia

- basal-rs: commit bazowy `db3bfbe` ze zmianą wprowadzającą `/v1/decisions`.
  Build `--release --features cuda`.
- Model: `Remek/basal-2.0-mini` z lokalnego cache Hugging Face, z domyślnymi
  ustawieniami `basal serve`.
- Klient: Python `openai` 3.41.0 (wymagane ≥ 3.26.0).
- Runner: [tools/openai-decisions/verify.py](../../tools/openai-decisions/verify.py).

## Sprzęt

NVIDIA GeForce RTX 4060 (8 GB), sterownik 615.78.08, jeden serwer na loopback.

## Metoda

Runner wysyła każdy przypadek przez SDK OpenAI przy użyciu `with_raw_response`.
Sprawdza, że SDK parsuje odpowiedź do swoich typów (`Decision`, odpowiedzi
`predicate`/`choice`/`score`). Następnie niezależnie od kodu Rust buduje równoważne
żądanie TypeSafe według reguł z dokumentacji i wysyła je do `/v1/systemone`.
Liczby obu odpowiedzi są porównywane jako tekst JSON (`parse_float=str`), czyli
bitowo. Porównanie obejmuje też `usage.input_tokens`.

Przypadki sukcesu:

1. `input` jako string; `choice` z opisem jednej opcji, dwa `predicate` (jeden bez
   `name`) i `score` z trzema poziomami.
2. `input` jako tablica: wiadomość `system`, wiadomość `user` z `input_text` i
   `input_json`, `function_call` i jego wynik, wiadomość `assistant` z `phase`,
   `custom_tool_call` i jego wynik jako część. Pytania: `choice` z wartościami
   boolean, w tym `instructions` jako tablica części, oraz `score` z czterema poziomami.

Przypadki błędów, sprawdzające status i kopertę `{"error": {message, type, param, code}}`:
nieznany model (`gpt-6-luna`), obraz `https://`, nieznany typ pytania, `choice`
z jedną opcją, wynik narzędzia bez wywołania, nieznane pole najwyższego poziomu
(`extra_body`) oraz obraz PNG 1×1 jako data URL.

Osobno, ręcznie: ciało 34 MB i niepoprawny JSON.

## Wyniki

| Przypadek | Wynik |
|---|---|
| 1. string input, 4 pytania | **bitowo równy** `/v1/systemone`, `input_tokens` równe |
| 2. wiadomości, JSON, narzędzia, choice boolean | **bitowo równy** `/v1/systemone`, `input_tokens` równe (805) |
| nieznany model | 404 `model_not_found` |
| obraz `https://` | 400 `unsupported_value` |
| nieznany typ, jedna opcja, wynik bez wywołania, nieznane pole | 400 `invalid_request_error` |
| PNG 1×1 | 400 `invalid_request_error`, `param: "input"`: odrzucony przez limit wymiarów 280–1568 px |
| ciało 34 MB | 413 w kopercie OpenAI |
| niepoprawny JSON | 400 w kopercie OpenAI |

Runner zakończył się statusem PASS. SDK sparsowało odpowiedzi do typów
`AnswerAnswerResourceChoice`, `…Predicate` i `…Score`. Wartości boolean
`choice` wróciły jako boolean.

## Ograniczenia

- Jeden model tekstowy i jedna karta. Nie sprawdzono obrazów na backendzie Omni
  z mediami: test PNG kończy się na walidacji wymiarów, przed modelem.
- Nie mierzono wydajności. Endpoint dodaje tylko tłumaczenie JSON, ale jego
  koszt nie był mierzony.
- Nie sprawdzono przeciążenia (503 z `Retry-After`), thinking ani SDK innych
  języków niż Python.
- Zgodność dotyczy kontraktu HTTP. Wyniki modelu basal nie są porównywane
  z `gpt-6-luna`.

## Dane

Wynik runnera (`verify.json`: pełne odpowiedzi, statusy, koperty błędów) leży
lokalnie w archiwum `openai-decisions-2026-10-29.tar.gz`, w ścieżce
`reports/openai-decisions-2026-10-29/verify.json`. SHA-256 archiwum:
`8d7ecf50ca63ffeb8785308e61267b0684fc0a4d5462c7e03c26d919b9e43f2f`.
Archiwum **nie zostało jeszcze opublikowane** w buckecie Hugging Face.

Odtworzenie: uruchom `basal serve --model <basal-2.0-mini> --addr 127.0.0.1:8765`,
a następnie
`python -I tools/openai-decisions/verify.py http://127.0.0.1:8765 basal-2.0-mini verify.json`
w środowisku z `openai>=3.26`.
