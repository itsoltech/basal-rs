# Natywne thinking

`--think auto` włącza generowanie notatek i drugi readout na **osobnym pełnym
checkpoincie `think/`**. Domyślne `--think off` nie pobiera ani nie ładuje
drugiego modelu. Wspierana para to
`Remek/basal-2.0-4.5B@e5ba9149760d99b73b1963d8e0ccf948b58afacb`.
Loader sprawdza SHA-256 obu opublikowanych plików wag, konfigurację i zgodność
tokenizerów. Mini i Pico nie mają tych wytrenowanych artefaktów; aktywacja
na nich odmawia startu. Nie ma adaptera LoRA, zero-shot ani zastępowania P2
modelem bazowym.

```sh
basal serve --model 2.0-4.5B --think auto --think-tokens 256 --think-max-len 8192
```

W YAML te same opcje modelu mają nazwy `think`, `think_tokens`, `think_max_len`,
`think_calibration`. Jawne flagi CLI zastępują wartości danego modelu w YAML.
Limity muszą być dodatnie, mieścić się w u32, a `think_tokens + 16` musi być
mniejsze od limitu kontekstu; rzeczywisty limit checkpointu dodatkowo ogranicza
`think_max_len`. Walidacja i wybór osobnej kalibracji następują przed GEMM
i ładowaniem wag na GPU. Włączenie może pobrać dodatkowy pełny checkpoint;
każdy z dwóch opublikowanych plików wag ma 9 514 629 496 bajtów.

`--think-calibration PATH` wybiera niezależny plik P2, `off` wyłącza tę
kalibrację, a pominięcie wybiera root `CALIBRATION.think.json`. Jawny brak
pliku jest błędem; automatyczny brak ostrzega i używa T=1 z etykietą `missing`.
Plik jest walidowany dla promptu modelu, canonical/order1. Nigdy nie przejmuje
orders/layout, strict identity ani `--no-calibration` readoutu bazowego.
Temperatury per primitive/bucket i certyfikaty P2 pochodzą wyłącznie z tego
pliku. Opublikowany plik zawiera temperatury, lecz jego certyfikaty per-type
1%/5% są null: to nie potwierdzenie certyfikowanej jakości P2.

## Żądanie i odpowiedź

Oba dialekty HTTP oraz narzędzia MCP przyjmują requestowe `think`:

| Wartość | Zachowanie |
|---|---|
| pominięte | domyślna polityka serwera (`off` albo `auto`) |
| `off` | normalny wynik bazowy |
| `auto` | thinking przy braku certyfikatu 5% lub `max(p_cal) < threshold` |
| `force` | thinking każdego kwalifikującego się pytania |

`null`, inne wartości i niepoprawne typy są błędami. `think_text` musi być
booleanem; domyślnie false. `thinking` nie jest aliasem. Jawne `auto`/`force`
na wyłączonym serwerze odmawia przed forwardem. Tasks classify/verify/compress/
extract wymagają `think: off`; aktywny tryb, również domyślny, odmawia przed GPU.

Najpierw wykonuje się cały przebieg bazowy, razem z orders auto i kalibracją.
Thinking obejmuje single-branch choice/noul/score/act/tool. Nie obejmuje multi,
toolsets, pytań bez readoutu ani grouped choices. Faktycznie skrócony stan
wyłącza thinking całego żądania, również przy force. Próg auto bierze maksimum
skalibrowanego rozkładu P1, a nie przekształcone confidence TypeSafe; równość
z progiem nie uruchamia thinking.

Head zawiera ten sam znormalizowany stan i facts, pytanie i canonical order1
opcji w PL/EN. Końcowy PREFILL zastępuje `<think>\n`. Greedy wybiera największy
logit z całego rzeczywistego słownika, przy remisie najniższy token ID.
Kończy na `</think>` (także przez granicę tokenów), EOS lub limicie tokenów.
Implementacja ponownie oblicza pełny prefiks na każdym kroku.

P2 ponownie koduje `head + notes.strip() + "\n</think>\n" + PREFILL`, sprawdza
code IDs w tym konkretnym kontekście i czyta je na **tych samych wagach think**.
Softmax i kalibracja readoutu są FP32; końcowy builder zachowuje formułę
confidence właściwą dla dialektu. Act/tool odbudowują progi z kalibracji P2,
evidence pozostaje z P1. Przed generacją obowiązuje
`head_tokens + think_tokens + 16 <= min(think_max_len, model_context)`;
rzeczywisty retokenizowany P2 też musi mieścić się w kontekście. Przekroczenie
zachowuje P1 i dodaje `think: {"skipped":"context"}`.

Sukces dodaje `think: {tokens, ms, calibration, no_think: {probabilities,
confidence}}`, a przy `think_text: true` również `text`. Confidence w
`no_think` jest `max(p1)`. Każda odpowiedź włączonego serwera, również off
i niekwalifikująca się, ma `meta.think: {used, tokens}`.
`usage.think_questions` liczy udane P2, a `output_tokens` faktycznie wygenerowane
tokeny, także tokeny stop/EOS. Jeżeli dopiero retokenizacja P2 przekroczy limit,
wygenerowane tokeny nadal są liczone, mimo zachowania P1. `input_tokens`,
branches, longest_prompt_tokens i second_order_branches zachowują znaczenie
przebiegu bazowego. Latency obejmuje cały request. Rejestr decyzji zapisuje
finalny wynik po P2; jego whitelista wyklucza notatki.

## Planowanie i lifecycle

Fork tworzy osobny cache i stan obu backendów, zachowuje ustawienia
i kalibrację. Gate, work class, stop, health oraz drain obejmują oba modele.
Stop jest sprawdzany między tokenami i przed P2; nie przerywa kernela w toku.
Cały request pozostaje w tym samym admission i lane aż do końcowego wyniku.

Planer offline dostaje snapshot capabilities thinking. Dodatkowe host bounds
obejmują P1, notatki, tokenizer/retokenizację, pełny słownik i staging.
Bound notatek pochodzi z maksymalnej długości elementu zwalidowanego dekodera
Metaspace/ByteFallback (z zapasem na zastępowanie błędnych bajtów UTF-8).
Device bounds obejmują największy capped-prefix forward/P2 i pełny słownik;
work cost obejmuje do `think_tokens + 1` pełnych forwardów na pytanie.
Pamięć rezydentna **obu** modeli i ich workspace musi być uwzględniona
osobno w kwalifikacji i rezerwie serwera, tak jak pozostałe resident weights.

Bounded executor wymaga źródłowych host/device bounds, cooperative controls
obu backendów, pokrycia GEMM wszystkich dopuszczonych klas M oraz limitów
slotu/puli i forwardów. Niewymodelowana konfiguracja thinking, np. referencyjne
kernele Candle lub dense-mask SDPA, daje `UNBOUNDED_FEATURE` z konkretną
przyczyną w planowaniu offline; nie wyłącza thinking po cichu. Przekroczenie
skończonego limitu admission daje właściwy błąd budżetu. Zwykłe/off żądania
nie dostają kosztu generacji. Online korzysta z tej samej rzeczywistej ścieżki
generation + P2, bez wymagania background qualification.

`basal-planner.4` rozróżnia nowy kontrakt. Profile batchy i ustawienia benchmarku
zawierają efektywne limity, osobną kalibrację (hash/off/missing), przypiętą
tożsamość i hashe plików obu checkpointów, tokenizerów i konfiguracji oraz
osobny plan obliczeń thinking. Nie zawierają lokalnych ścieżek tych źródeł.

## Zakres kwalifikacji

Kod implementuje natywną generację i drugi readout; kontrole statyczne nie
potwierdzają wyników, stop-token counting referencyjnego vLLM, jakości ani
wydajności. Full-prefix recomputation nie daje obietnicy szybkości lub
parytetu tokenów z vLLM. Pozytywna kwalifikacja wymaga obu przypiętych
checkpointów i odpowiedniej pamięci. Pico/Mini mogą sprawdzić tylko off
i odmowy; lokalna karta 8 GB nie mieści nawet jednego z tych niekwantyzowanych
plików wag. Zmiana precyzji wymaga osobnej kwalifikacji.
