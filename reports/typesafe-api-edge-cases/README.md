# /v1/systemone wobec API TypeSafe: przypadki brzegowe

Data: 2026-10-07. Te same 38 żądań wysłane do `https://api.typesafe.ai`
(`jev-latest`, odpowiada `jev-1.13.0`; OpenAPI 0.2.0, ten sam plik co
[contracts/typesafe/openapi.json](../../contracts/typesafe/openapi.json)) i do
`basal serve --model 4.5B` (Metal, M1 Pro): przed zmianą (0.1.3) i po niej.
Narzędzie: [tools/typesafe/probe.py](../../tools/typesafe/probe.py). Dane:
`typesafe.json`, `basal-rs-0.1.3.json`, `basal-rs.json` (żądanie, status,
czas, odpowiedź). Porównanie obejmuje status oraz `type`, `loc` i `msg`
błędów; `input` i `ctx` są zapisane, ale nie porównywane.

| | 0.1.3 | po zmianie |
|---|---:|---:|
| ten sam status co TypeSafe | 23/38 | 34/38 |
| ten sam status i treść | 12/38 | 31/38 |

| Przypadek | TypeSafe | basal-rs 0.1.3 | basal-rs po zmianie |
|---|---|---|---|
| `choice_0` | 400 | 422 | 400 |
| `choice_1` | 200 | 422 | 200 |
| `choice_1_desc` | 200 | 422 | 200 |
| `choice_2_null` | 200 | 200 | 200 |
| `choice_11` | 200 | 200 | 200 |
| `choice_255` | 200 | 200 | 200 |
| `choice_256` | 400 | 422 | 400 |
| `choice_criteria_list` | 422 | 200 | 200 (rozszerzenie) |
| `choice_no_criteria` | 422 | 422 (inna treść) | 422 |
| `score_0` | 422 | 422 (inna treść) | 422 |
| `score_1` | 200 | 422 | 200 |
| `score_2` | 200 | 200 | 200 |
| `score_10` | 200 | 200 | 200 |
| `score_11` | 400 | 422 | 400 |
| `score_null_item` | 422 | 422 (inna treść) | 422 |
| `score_objects` | 200 | 200 | 200 |
| `score_map` | 422 | 200 | 200 (rozszerzenie) |
| `noul_no_instr` | 400 | 200 | 400 |
| `noul_criteria_null` | 200 | 200 | 200 |
| `noul_only_true` | 200 | 200 | 200 |
| `noul_empty_true` | 200 | 200 | 200 |
| `noul_instr_null` | 400 | 200 | 400 |
| `noul_instr_number` | 422 | 422 (inna treść) | 422 |
| `no_type` | 422 | 422 (inna treść) | 422 |
| `type_unknown` | 400 | 422 | 400 (dłuższy komunikat) |
| `type_multi` | 400 | 200 | 200 (rozszerzenie) |
| `questions_empty` | 422 | 422 (inna treść) | 422 |
| `questions_list` | 422 | 422 (inna treść) | 422 |
| `state_number` | 422 | 422 (inna treść) | 422 |
| `state_null` | 422 | 422 (inna treść) | 422 |
| `state_empty` | 200 | 200 | 200 |
| `no_state` | 422 | 422 | 422 |
| `no_model` | 422 | 422 | 422 |
| `model_unknown` | 400 | 422 | 400 |
| `extra_field` | 400 | 200 | 200 (rozszerzenie) |
| `body_not_json` | 422 | 400 | 422 (inna pozycja w `loc`) |
| `body_array` | 422 | 422 (inna treść) | 422 |
| `multi_errors` | 422 | 422 (inna treść) | 422 (dodatkowo błąd `state`) |

Pozostałe różnice:

- Rozszerzenia basal-rs (`multi`, `act`, dodatkowe pola, kryteria choice
  jako lista, score jako mapa) TypeSafe odrzuca; żądania pisane pod TypeSafe
  ich nie zawierają.
- `type_unknown`: ten sam status i kształt, komunikat podaje typ i pytanie
  (TypeSafe: `Invalid request.`).
- `body_not_json`: pozycja błędu z dekodera serde_json (1), Python podaje 0.
- `multi_errors` (`state: null` i dwa błędne pytania): TypeSafe zgłasza tylko
  błędy pytań, choć samo `state: null` zgłasza jako `missing`; basal-rs
  zgłasza wszystkie trzy.

TypeSafe zaokrągla prawdopodobieństwa, score i confidence do dwóch miejsc;
basal-rs zwraca pełną precyzję. Przy jednej opcji (choice, score) obie
implementacje zwracają prawdopodobieństwo 1 i confidence 1; basal-rs nie
uruchamia wtedy modelu.

Oficjalne SDK `typesafe-sdk` 0.7.2 (Python, `base_url` na `basal serve`)
odczytuje odpowiedzi noul, choice (także z jedną opcją) i score. Ponawia
statusy 408, 429 i 5xx (domyślnie 2 razy, z wykładniczym odczekiwaniem i
`Retry-After`). Przy `--max-inflight 1` i 8 równoległych wywołaniach serwer
dostał 22 żądania (14 ponowień po 529); 7 wywołań się powiodło, jedno
wyczerpało ponowienia, czego przy takim limicie należy się spodziewać.

Zmiana nie dotyczy obliczeń: eksport referencji basal-1.5-4.5B
(`reports/reference-basal-1.5-4.5B-fp32`) binarką 0.1.3 i nową daje
`basal compare` 0,0 dla logitów i prawdopodobieństw i 44/44 decyzji bench.
Z 34 żądań System One 16 ma odpowiedź identyczną z 0.1.3, a pozostałe 18 to
błędy w nowym formacie (głównie 400 `Unknown model` zamiast 422, bo żądania
referencji podają model basal-1.0). Odpowiedzi dialektu upstream
(`/v1/basal`, także błędy) są identyczne w 34/34.
