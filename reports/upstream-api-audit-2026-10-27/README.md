# API względem upstream basal-serve v2.4.1

Stan na 2026-10-27. Najnowsze stabilne wydanie potwierdzone przez GitHub Releases:
[v2.4.1](https://github.com/rkinas/basal/releases/tag/v2.4.1), opublikowane
2026-10-23, commit `b20f0f9b159f71726459aa98ede1356ab65695fd`.
Porównano z basal-rs `729fe47df9ef8117f46b526582941afcc4f97e70`.

Metoda: przegląd kodu tras, parserów żądań, odpowiedzi, konfiguracji i dokumentacji
obu projektów. Nie uruchamiano HTTP, inferencji ani pomiarów. Poniższe ustalenia
opisują kontrakty i dostępne implementacje; nie potwierdzają zgodności numerycznej
ani pełnej zgodności wszystkich kombinacji parametrów. Kod aplikacji nie został
zmieniony.

## Najważniejsze różnice zgodności

1. **Błędy silnika na `/v1/basal` mają niewłaściwy status HTTP.** Upstream
   rozróżnia 422 dla błędu żądania, 503 dla braku pamięci GPU i 500 dla pozostałych
   błędów serwera. Nasze `error_parts` zwraca 422 dla każdego `DecideError`
   w dialekcie upstream, także `Internal`. Klient może uznać awarię silnika za
   błąd danych i zrezygnować z ponowienia. Osobne odpowiedzi admission 529
   i zakończenie workera nie przechodzą przez tę samą regułę.
   Źródła: [upstream http_status](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/server.py#L425),
   [nasze serve.rs](../../crates/basal-cli/src/serve.rs), `error_parts`, linia 762.

2. **Nowe układy promptu i ich kalibracja nie są rozpoznawane.** W v2.4.1
   upstream obsługuje `canonical`, `qfirst`, `state_twice`: wybór przez `--layout`
   lub `basal.json`, pliki `CALIBRATION.<layout>.json` i metadane `layout`.
   Przy niedopasowanej kalibracji usuwa certyfikowane progi i oznacza odpowiedzi
   `calibration: "unvalidated"`. U nas renderer zawsze stosuje dotychczasowy
   układ; loader nie odczytuje `layout` modelu ani kalibracji. Model deklarujący
   inny układ nie zostanie z tego powodu odrzucony i otrzyma inny prompt niż
   upstream. Dotyczy to również pliku kalibracji dopasowanego do innego układu.
   Najmniejsza ochrona przed pełną implementacją to jawne odrzucanie
   nieobsługiwanego układu modelu oraz niedopasowanej kalibracji.
   Źródła: [upstream prompt.py](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/prompt.py#L77),
   [dobór kalibracji](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/server.py#L645),
   nasze [manifest.rs](../../crates/basal-core/src/manifest.rs), linie 137/234,
   [prompt.rs](../../crates/basal-core/src/prompt.rs), `render_model`.

3. **`/v1/systemone` nie jest zamiennikiem endpointu upstream.** Upstream obie
   ścieżki obsługuje identycznie i nie wymaga `model`. U nas `/v1/systemone`
   stosuje świadomie inny kontrakt TypeSafe: wymaga `model`, ma ściślejszą
   walidację oraz inne reguły serializacji odpowiedzi i błędów. Trasą dla
   zgodności upstream jest `/v1/basal`. Oficjalny klient Python, most `basal-mcp`
   i zdalny `basal-calibrate --url` używają jednak `/v1/systemone`; sama zmiana
   adresu serwera nie wystarcza. Dodatkowo nasze serwowanie wielu modeli nadaje
   polu `model` znaczenie routingu, podczas gdy upstream je ignoruje.
   To różnica projektowa, a nie brak podstawowej trasy. Jej rozwiązanie powinno
   zachować kontrakt obecnych klientów TypeSafe, np. przez opcjonalny adapter.
   Źródła: [upstream trasy](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/server.py#L1610),
   [klient](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/client.py#L12),
   nasze [serve.rs](../../crates/basal-cli/src/serve.rs), `systemone`,
   [request.rs](../../crates/basal-core/src/request.rs), `parse_request_for_format`,
   [opis TypeSafe](../../docs/SYSTEM_ONE.md).

## Brakujące funkcje i pola

1. **MCP jako transport serwera.** Brak `/mcp`, narzędzi `decide` i `decide_many`
   oraz odpowiednika `--mcp on`. Upstream oferuje opcjonalne stateless
   streamable HTTP i most stdio `basal-mcp`. Nasza obsługa pytań `type: "tool"`
   i definicji narzędzi MCP dotyczy klasyfikacji narzędzi; nie udostępnia
   protokołu MCP klientom. Most upstream napotyka też różnicę `/v1/systemone`
   opisaną powyżej.
   Źródło: [upstream MCP](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/docs/MCP.md).

2. **Liczniki `/health` i część `usage`.** Upstream zwraca `running` i `queued`
   w `/health`, licząc żądania, także wieloprzebiegowe. U nas są informacje
   o gotowości, modelach i replikach, ale brak tych dwóch pól; metryki Prometheus
   ich nie zastępują w tym kontrakcie. Upstream zawsze emituje
   `usage.longest_prompt_tokens`; u nas pole zależy od formatu promptu 2.
   Opcjonalne `usage.lane` upstream również nie jest emitowane.
   `usage.latency_ms` już mamy. Nie stwierdzono różnicy `usage.branches`
   polegającej na mnożeniu przez liczbę kolejności opcji: kod obu projektów
   liczy gałęzie przed takim powieleniem, mimo opisu w dokumentacji upstream.
   Źródła: [upstream load](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/server.py#L945),
   [assemble](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/server.py#L1405),
   nasze [serve.rs](../../crates/basal-cli/src/serve.rs), `health`,
   [engine.rs](../../crates/basal-core/src/engine.rs), linia 1232.

3. **Konfigurowalne skracanie przekroczonego kontekstu.** Upstream ma
   `--context-overflow truncate|refuse` z domyślnym `truncate`, `--truncate-head`
   oraz `usage.truncated`. Zachowuje początek i koniec stanu, pytania i opcje.
   Nasza implementacja odrzuca za długi prompt przed forwardem i nie ma trybu
   skracania. Jest to świadoma różnica zachowania; dodanie opcji nie wymaga
   zmiany naszego domyślnego trybu, a skracanie wymaga osobnej oceny jakości.
   Źródła: [upstream opcje](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/server.py#L1554),
   nasze [planner.rs](../../crates/basal-core/src/planner.rs), `check_context`.

4. **Operacyjne opcje kalibracji i rejestr decyzji.** Brak odpowiedników
   `--no-calibration`, `--calibration PATH`, `--strict-calibration`, ostrzeżenia
   `calibration_warning` dla obcego modelu/rewizji i `--log-decisions`
   zapisującego prawdopodobieństwa do późniejszej ewaluacji.
   Nasz loader dobiera lokalną kalibrację i pliki dla liczby kolejności opcji,
   ale nie weryfikuje jej `fitted_on.model`/`revision`. Dlatego oficjalny zdalny
   kalibrator wymaga dodatkowego dostosowania także poza samą trasą HTTP:
   oczekuje serwera z wyłączoną kalibracją. Parametry progów adaptacyjnego
   `orders=auto` są u nas stałe, zamiast konfigurowalne jak `--auto-tau` i
   `--auto-band` upstream.
   Źródła: [upstream CLI](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/server.py#L1591),
   [kalibrator HTTP](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/calibrate.py#L1334),
   nasze [manifest.rs](../../crates/basal-core/src/manifest.rs),
   [engine.rs](../../crates/basal-core/src/engine.rs).

5. **Opcjonalne thinking i early exit.** Brak eksperymentalnego `--think auto`,
   pól żądania `think`/`think_text`, drugiego odczytu i metadanych `think`.
   Upstream ogranicza tę ścieżkę do basal-2.0 4.5B i zaleca pozostawienie jej
   wyłączonej. Nasz guard odrzuca pole `thinking`, podczas gdy upstream używa
   `think`. Zwykły serwer upstream z wyłączonym thinking też ignoruje `think`;
   nie jest to różnica względem jego domyślnej ścieżki. Brak także polityk
   `early_exit` i trybu `fast-exit`, który upstream udostępnia dla basal-1.0
   z wytrenowanymi głowami wyjścia. U nas lista polityk jest pusta, a pole
   `early_exit` o wartości innej niż `null` jest odrzucane.
   Nie są to brakujące możliwości Pico.
   Źródła: [upstream thinking](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/docs/THINKING.md),
   [fast-exit CLI](https://github.com/rkinas/basal/blob/b20f0f9b159f71726459aa98ede1356ab65695fd/basal/server.py#L1485),
   nasze [planner.rs](../../crates/basal-core/src/planner.rs), linia 599.

## Co jest już zaimplementowane

Oba endpointy decyzyjne, `/v1/models` i `/health` istnieją. Kod obejmuje
`choice`, `noul`, `score`, `multi`, `act`, `tool` oraz zadania `classify`,
`verify`, `compress`, `extract`. Są format 2 i do 255 opcji, kolejności
`1`/`2`/`auto`/`2leN`, `facts`, PL/EN, `option_keys`, cele multi, certyfikaty,
evidence przy odpowiedniej głowie oraz obrazy/audio dla obsługiwanych modeli.
Definicje narzędzi obejmują OpenAI, MCP i Anthropic. To potwierdzenie obecności
implementacji, nie nowy test wszystkich wariantów.

Nasze dodatkowe możliwości obejmują routing wielu modeli i replik, metryki,
adapter TypeSafe z assets/multipart oraz trwałe files/batches. Brak backendu
vLLM i jego `vllm_preset` nie jest osobnym brakiem kontraktu decyzji.
Upstream nie oferuje tutaj `/v1/chat/completions` ani `/v1/completions`;
nie są to luki tej integracji.

## Priorytet

Najpierw poprawić rozróżnianie błędów HTTP i jawnie obsłużyć lub odrzucać obce
układy promptu/kalibracji. Następnie rozwiązać zgodność oficjalnych klientów
z `/v1/systemone` oraz dodać liczniki `/health`. Pełne layouty, MCP i opcje
kalibracji rozszerzają funkcjonalność. Skracanie kontekstu, thinking i early exit
pozostają osobnymi decyzjami produktowymi. W tym audycie niczego nie wdrożono.
