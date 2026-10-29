# MCP: decyzje jako narzędzia agenta

`basal serve --mcp on` udostępnia `/mcp` na tym samym porcie co HTTP.
Równoważne ustawienie YAML to `mcp: "on"`; domyślne `"off"` nie tworzy
transportu ani jego zadań. Jawne `--mcp on|off` zastępuje YAML.
Oficjalny Rust SDK `rmcp 3.8.0` obsługuje streamable HTTP, stateless,
odpowiedzi JSON i starszy handshake `initialize`/`notifications/initialized`,
bez wymagania nowych pól `_meta`.

```sh
basal serve --model pico --addr 127.0.0.1:8000 --mcp on
claude mcp add --transport http basal http://127.0.0.1:8000/mcp
```

## Narzędzia

- `decide(state, question, type="choice", options=null, lang=null)`:
  `state` jest tekstem lub obiektem JSON, `question` tekstem,
  `type` jednym z `choice`, `noul`, `score`, `multi`. `options` to mapa
  klucz → opis tekstowy/null, lista stringów lub null; kolejność jest
  zachowana. `lang` przyjmuje `pl`, `en` lub null (automatyczne wykrywanie).
  Powstaje natywne `questions.answer` z `type`, `instructions` i opcjonalnym
  `criteria`. Wynik zawiera `model` oraz pola `answers.answer`, bez `usage`.
- `decide_many(state, questions, lang=null, facts=null)`: `questions` to mapa
  nazwa → natywny obiekt pytania. Przyjmuje wszystkie typy HTTP, także `act`,
  `tool` i zadania `classify`, `verify`, `compress`, `extract`, z ich natywnymi
  polami. `facts` to `off`, `auto` lub null. Wynik to pełne natywne
  `model`, `answers`, `usage`.

Oba narzędzia przyjmują opcjonalne `think: off|auto|force` oraz boolean
`think_text`, przekazywane do wspólnego dispatchera bez zmiany znaczenia.
Brak `think` wybiera ustawienie serwera; null jest błędem. Aktywne thinking
wymaga załadowanej pary base + think i nie obsługuje tasks (dla nich `off`).
Przy aktywacji model może wygenerować ograniczone notatki i wykonać drugi
readout. `decide` zachowuje pola `think`/`meta.think` pojedynczej odpowiedzi;
`decide_many` także usage. Szczegóły: [THINKING.md](THINKING.md).

Oba narzędzia przekazują też opcjonalne `early_exit` string/null do tego
samego dispatchera. Missing/null wybiera default; jawny string, w tym `off`,
wymaga head-capable backendu i dostępnej polityki. Units/chunks pozostają
izolowane per request; obsługiwany model, opt-in i GAP: [EARLY_EXIT.md](EARLY_EXIT.md).

Oba narzędzia deklarują `readOnlyHint=true`, `destructiveHint=false`,
`idempotentHint=true`, `openWorldHint=false`. SDK zwraca obiekt
`structuredContent` oraz tekstowy JSON dla starszych klientów. Błędy
argumentów, walidacji, inferencji i przeciążenia są wynikami narzędzia
z `isError=true` i przyczyną API, w tym 529/`Retry-After`. Błędy protokołu
pozostają błędami MCP/JSON-RPC. Nie ma argumentu `model`: wybór jest taki
sam jak przy `/v1/basal` bez `model`, a wynik wskazuje faktycznie użyty model.
Instrukcje narzędzi opisują zadawanie pytań i używanie confidence.
Nie zastępują walidacji ani ograniczeń załadowanego modelu.
Natywne pytania i zadania: [SYSTEM_ONE.md](SYSTEM_ONE.md).

## Most stdio

`basal mcp` łączy hosta MCP z już uruchomionym serwerem:

```json
{"mcpServers":{"basal":{"command":"/absolute/path/to/basal","args":["mcp","--url","http://127.0.0.1:8000","--timeout","60"]}}}
```

Domyślny URL to `http://127.0.0.1:8000`, timeout decyzji 60 sekund.
`--timeout` musi być skończony i dodatni. Most dopisuje `/v1/basal`
i `/v1/models` do ścieżki `--url`, zachowując jej segmenty (także
percent-encoded); `http://proxy/basal/` daje `http://proxy/basal/v1/basal`.
URL z query lub fragmentem jest odrzucany przy starcie. Most zawsze wysyła
natywne body do `/v1/basal`, niezależnie od `systemone_dialect`. Nie wymaga
`--mcp on` na zdalnym serwerze. Nie ładuje wag, nie uruchamia serwera i nie
deleguje wykonania do binarki CUDA ani procesu sidecar.

Odczyt `/v1/models` jest best effort z limitem 5 sekund; fallback to `basal`.
Pierwsza nazwa modelu służy tylko instrukcjom MCP i nie zmienia żądania ani
wyboru domyślnego modelu. Stdout zawiera wyłącznie protokół; komunikaty CLI
trafiają na stderr. Ctrl-C i SIGTERM zamykają transport i anulują
oczekujące wywołania HTTP. EOF stdin kończy tylko odbiór: SDK przez najwyżej
5 sekund wysyła odpowiedzi trwających wywołań, potem anuluje pozostałe
i zamyka transport. Linux używa nieblokujących potoków stdio SDK;
fallback odczytu terminala ma ograniczony czas zamykania runtime.

## Limity, lifecycle i dostęp

HTTP `/mcp` ma limit body 2 MiB egzekwowany przez SDK przed parsowaniem JSON.
Współdzieli z TypeSafe permit dekodowania (`max_inflight`) i z online
permit dużego body (`max_large_inflight`): dla MCP już ponad 128 KiB albo
przy braku Content-Length. Odmowa admission to HTTP 529 z `Retry-After: 1`.
Permity żyją również w kontekście zadania SDK, a duże dane przekazane do
silnika zachowują permit do końca pracy. Odłączenie klienta anuluje odbiornik
wyniku; rozpoczęty forward kończy się według istniejącego lifecycle silnika.
Oba narzędzia używają wspólnej kolejki, `/health`, limitów, metryk modelu
i finalnego rejestru decyzji. `decide` rejestruje jeden pełny wynik przed
wybraniem pojedynczej odpowiedzi. Shutdown anuluje transport SDK, zwalnia
nadawców, a writer decyzji kończy drain/join poza executorem.

Most stdio ogranicza pojedynczą linię protokołu oraz odpowiedź API do 2 MiB,
discovery do 64 KiB i najwyżej cztery oczekujące żądania/odpowiedzi. Ma jeden
parser; także zadania powiadomień są ograniczone. Przeciążone wywołanie
narzędzia dostaje błąd narzędzia, inne żądanie błąd protokołu. Zablokowany
stdout stosuje backpressure, bez nieograniczonej kolejki odpowiedzi.

Bez [kluczy API](AUTH.md) endpoint nie ma uwierzytelniania. Z `auth.enabled: true`
wymaga klucza z zakresem `mcp`; `basal mcp` wysyła klucz z `BASAL_API_KEY` (`--key-env`).
SDK sprawdza Host oraz obecny Origin
przeciw jawnym allowlistom, bez zaufania do `X-Forwarded-*`. Dopuszczone są
`localhost`, `127.0.0.1`, `[::1]` oraz konkretny adres bind, z faktycznym
portem nasłuchu. Origin wymaga schematu `http` i dozwolonego hosta/portu;
brak Origin jest dopuszczony dla klientów MCP, brak Host (i authority URI)
jest błędem. Obcy/null Origin, obcy Host i zły port są odrzucane. Host musi
zawierać port także przy nasłuchu na porcie 80.

Wildcard bind `0.0.0.0`/`[::]` nie dodaje dowolnych nazw ani adresów do
allowlisty i nie wyłącza ochrony. Dla bezpośredniego dostępu zdalnego binduj
konkretny adres IP. Gateway powinien zapewniać własne uwierzytelnianie,
weryfikować zewnętrzne Host/Origin i przekazywać zatwierdzone żądania
z wewnętrznym Host/Origin odpowiadającym bind serwera. Same nagłówki
forwarded nie zmieniają allowlisty. Dla agentów lokalnych używaj loopback.
