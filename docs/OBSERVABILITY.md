# Observability: Prometheus

Metryki są **domyślnie wyłączone**. Włącz je flagą:

```sh
basal serve --metrics
# także z jawnym plikiem konfiguracji:
basal serve --config basal-serve.yml --metrics
```

Albo ustaw opcję na poziomie procesu w `basal-serve.yml` (obok `models`):

```yaml
metrics: true
models:
  - repo: Remek/basal-1.5-4.5B
```

Brak `metrics` lub `metrics: false` wyłącza funkcję, o ile nie podano flagi
`--metrics`. Flaga włącza ją także przy `metrics: false` w YAML. Konfiguracja
jest odczytywana przy starcie; zmiana ustawienia wymaga restartu serwera.
Plik wygenerowany przez `basal init` zawiera zakomentowany przykład włączenia.

Gdy metryki są wyłączone, serwer nie tworzy rejestru, nie zbiera liczników ani
histogramów i nie instaluje middleware metryk; `GET /metrics` zwraca 404.
Po włączeniu endpoint działa na tym samym adresie i porcie co API i zwraca
`text/plain; version=0.0.4`. Każdy proces ma własny rejestr; liczniki zerują się
po restarcie. Poniższe definicje i zapytania dotyczą włączonych metryk.

```sh
curl --fail localhost:8000/metrics
```

## Health i błędy API

`GET /health` działa także bez `--metrics`. Oprócz pól gotowości (`status`,
`models`, `default_model` i ewentualnych informacji o niedostępnych modelach
lub replikach) zawsze zwraca `running` i `queued`. Statusy gotowości pozostają
bez zmian: 200 przy gotowości, także z `degraded_models`, oraz 503 przy
`device_lost` lub braku dostępnych replik grupy (`engine_stopped`).

Liczniki sumują żądania decyzji online ze wszystkich modeli i replik, obu
API oraz obu torów kolejek. Jedno żądanie jest liczone raz, niezależnie od
liczby pytań, promptów i gałęzi. Nie obejmują files/batches ani wykonywania
trwałych batches w tle. Liczenie zaczyna się po dekodowaniu body, przy
wejściu do obsługi decyzji, i kończy przy odpowiedzi, błędzie lub porzuceniu
handlera. Odczyt body i dekodowanie nie należą do tych liczników.

`queued` oznacza oczekiwanie na przyjęcie do partii silnika: kanał wejściowy,
planowanie, kolejkę schedulera i przekazanie do toru długiego. `running`
obejmuje pozostałe aktywne żądania, w tym pracę serwera przed kolejką i po
obliczeniach. Para jest odczytywana razem; `running + queued = 0` oznacza brak
aktywnych żądań klientów, niekoniecznie bezczynność GPU. Po porzuceniu handlera
żądanie znika z obu liczników, choć rozpoczęte obliczenia mogą nadal trwać.
Odrzucenie, błąd planowania i zakończenie workera także zwalniają `queued`.
Drugi przebieg `orders=auto` pozostaje `running`: nasz silnik wykonuje go
wewnątrz tego samego wywołania, bez ponownego kolejkowania stosowanego przez
upstream. Wstrzymanie długiego forwardu na gate także pozostaje `running`.

`POST /v1/basal` zwraca błędy w postaci `{"error": "komunikat"}`:
422 dla walidacji, błędnego użycia API i nieobsługiwanych żądań; 503 dla
`Internal`, gdy dowolny komunikat w łańcuchu przyczyn zawiera `out of memory`
bez względu na wielkość liter albo dokładny fragment `OutOfMemory`; 500 dla
pozostałych błędów silnika, także `{"error": "engine stopped"}` przy zakończeniu
workera. To tekstowa reguła upstream: błąd backendu bez pasującego komunikatu
otrzyma 500, nawet jeśli jego przyczyną jest brak pamięci. Przeciążenie nadal
daje 529 z `Retry-After: 1`. Kontrakt błędów TypeSafe `/v1/systemone` i trwałych
rekordów batches pozostaje bez zmian. Opis określa reguły implementacji;
nie stanowi pomiaru ani weryfikacji runtime.

## Zbieranie metryk

Minimalny fragment `prometheus.yml` dla Prometheusa działającego na tym samym
hoście co serwer:

```yaml
scrape_configs:
  - job_name: basal
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:8000"]
```

W osobnych kontenerach zamiast `localhost` użyj nazwy usługi serwera dostępnej
z kontenera Prometheusa, np. `basal:8000`. Dla wielu procesów serwera dodaj osobne targets
lub service discovery. Endpoint podlega tym samym regułom dostępu sieciowego co
API; serwer nie dodaje osobnego uwierzytelniania do metryk. Przy publicznym API
ogranicz dostęp do `/metrics` na reverse proxy do sieci monitoringu.

## Żądania HTTP

| Metryka | Typ | Etykiety | Znaczenie |
|---|---|---|---|
| `basal_http_requests_total` | counter | `endpoint`, `model`, `status` | zakończone żądania do tras inferencji oraz porzucone futures handlerów |
| `basal_http_request_duration_seconds` | histogram | `endpoint`, `model`, `status` | czas od wejścia do middleware do zbudowania odpowiedzi, w sekundach |
| `basal_http_inflight_requests` | gauge | `endpoint` | aktywne handlery, także podczas odczytu body i walidacji |
| `basal_model_inflight_requests` | gauge | `model` | przyjęte żądania oczekujące na odpowiedź wątku modelu |
| `basal_max_inflight_requests` | gauge | — | skonfigurowany limit przyjęć dla wszystkich modeli łącznie |

`endpoint` to `/v1/systemone` lub `/v1/basal`. Metryki obejmują również błędne
metody na tych trasach (405), niepoprawny JSON, przekroczenie limitu body (413),
błędy walidacji, awarie oraz przeciążenie (529). `/metrics`, `/health`,
`/v1/models` i nieznane ścieżki nie są liczone.

`model` jest nazwą obsługiwanego modelu. Gdy żądanie trafia do modelu domyślnego,
otrzymuje jego nazwę. Nieznane modele, niepoprawny JSON i błędy przed odczytaniem
body mają `model=""`. Dowolne nazwy podane przez klientów, treść żądań i query
stringi nie stają się etykietami. Błędy pola `model`, dla których dotychczasowa
ścieżka routingu wybiera model domyślny przed walidacją, mają jego etykietę.

`status` jest kodem HTTP, np. `200`, `422`, `529`. Wartość `cancelled` oznacza
porzucenie future obsługującego żądanie przed zbudowaniem odpowiedzi; nie jest
kodem HTTP. Nie każde zerwanie połączenia musi powodować takie porzucenie.
Anulowane żądania nie trafiają do histogramu opóźnienia. Pomiar HTTP obejmuje
odczyt body, parsowanie, walidację, planowanie, oczekiwanie, obliczenia i
serializację JSON; nie obejmuje wysyłania odpowiedzi po sieci do klienta.

Prompt dłuższy niż kontekst modelu wątek modelu odrzuca już podczas planowania
żądania (`Engine::plan_request_as`, kontrola `check_context`), zanim żądanie
trafi do kolejki partii. Oba API odpowiadają 422: `/v1/systemone` błędem typu
`context_length_exceeded` w `detail`, `/v1/basal` polem `error` z komunikatem
„maximum context length”. Takie żądanie zwiększa `basal_http_requests_total` i
trafia do `basal_http_request_duration_seconds` z etykietą modelu, do którego
zostało skierowane, oraz `status="422"`. Nie zwiększa liczników tokenów ani
pytań i nie trafia do histogramów kolejki ani partii. Metryki nie rozróżniają
przyczyn statusu 422: ten sam status mają błędy walidacji i inne żądania
nieobsługiwane przez model, a przyczynę podaje tylko treść odpowiedzi.

Gauge modelu zwalnia miejsce także po porzuceniu handlera. Obliczenia już
uruchomione na GPU mogą wówczas nadal trwać. Jest to liczba oczekujących
handlerów, a nie pomiar zajętości GPU ani dokładnej długości kolejki.

## Tokeny i pytania

| Metryka | Typ | Etykiety | Znaczenie |
|---|---|---|---|
| `basal_input_tokens_total` | counter | `model` | suma `usage.input_tokens` |
| `basal_output_tokens_total` | counter | `model` | suma `usage.output_tokens` — obecnie zawsze 0 |
| `basal_questions_total` | counter | `model` | suma `usage.questions`, czyli pytań użytkownika, nie wewnętrznych gałęzi |

Liczniki aktualizuje wątek modelu po poprawnym zbudowaniu odpowiedzi. Obejmują
także ukończoną pracę, której klient już nie odbierze. Błędy walidacji,
odrzucone i pominięte przed wykonaniem żądania oraz nieudane odpowiedzi nie
zwiększają tych liczników. Praca częściowo wykonana przed błędem nie jest liczona.

Tokeny wejściowe zachowują semantykę API: sumują długości promptów dla
poszczególnych porządków opcji i kolejnych rund dużych choice. Wspólne prefiksy
są liczone ponownie dla każdego promptu. Nie jest to liczba unikalnych tokenów
po pakowaniu ani fizycznie przetworzonych tokenów GPU po uwzględnieniu cache.
Dodatkowe forwardy evidence nie są uwzględniane w `usage.input_tokens`.
Basal podejmuje decyzje, nie generuje tekstu; dlatego output tokens wynoszą 0.

## Kolejka i partie

| Metryka | Typ | Etykiety | Znaczenie |
|---|---|---|---|
| `basal_queue_duration_seconds` | histogram | `model`, `lane` | czas żądania od wejścia do handlera po odczycie body do początku przyjmowania partii, razem z planowaniem |
| `basal_batch_duration_seconds` | histogram | `model`, `lane`, `outcome` | czas przyjęcia i wykonania partii, jedna obserwacja na partię |
| `basal_batch_requests` | histogram | `model`, `lane` | liczba żądań w wykonanej partii |

`lane` to `main` lub `long`. Tor `long` pojawia się tylko dla modeli, które go
mają. `outcome` to `success` lub `error` wyniku `Engine::run_plans`; późniejszy
błąd budowania odpowiedzi widać w statusie HTTP. Czas partii obejmuje oczekiwanie
na gate GPU i przerwy oddające GPU innemu torowi, nie obejmuje wstępnego
planowania ani budowania odpowiedzi. Nie jest czystym czasem kerneli GPU.

Kolejka ma jedną obserwację na żądanie uruchomionej partii, również w razie błędu
wykonania partii. Pominięte żądania nie trafiają do tego histogramu. Statystyki
kolejki i wielkości partii są publikowane po powrocie `run_plans`.

Histogramy czasu mają granice od 1 ms do 600 s oraz `+Inf`:
`0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600`.
Histogram wielkości partii: `1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, +Inf`.
Każdy histogram eksportuje `_bucket`, `_sum` i `_count`.

## Repliki i instancje modeli

Każdy model ma co najmniej jedną instancję; `replica_devices` daje kilka
replik jednego modelu na osobnych GPU CUDA. Metryki z poprzednich sekcji
zachowują znaczenie zagregowane: wszystkie repliki modelu zapisują do tych
samych serii `model` i `lane`. Osobne serie instancji (także jedynej instancji
modelu bez replik, `replica="0"`) mają etykiety tylko z konfiguracji: nazwę
modelu, indeks repliki w `replica_devices` i ordinal CUDA. Model z podziałem
warstw (`layer_split`) jest jedną instancją (`replica="0"`), a jej `device`
to ordinale etapów w kolejności, rozdzielone przecinkiem, np. `device="0,1"`;
zwykła instancja ma jeden ordinal, np. `device="1"`. Tekst HELP metryki się
nie zmienia.

| Metryka | Typ | Etykiety | Znaczenie |
|---|---|---|---|
| `basal_replica_info` | gauge | `model`, `replica`, `device` | 1 dla każdej skonfigurowanej instancji; wiąże indeks repliki z ordinalem CUDA (przy podziale warstw: ordinale etapów, np. `0,1`) |
| `basal_replica_routed_requests_total` | counter | `model`, `replica` | żądania przekazane do kolejki instancji |
| `basal_replica_inflight_requests` | gauge | `model`, `replica` | żądania zarezerwowane na instancji: w kolejce, planowane, oczekujące i wykonywane |
| `basal_replica_waiting_requests` | gauge | `model`, `replica`, `lane` | zaplanowane żądania pozostałe w torze po ostatnim przyjęciu partii |
| `basal_replica_queue_duration_seconds` | histogram | `model`, `replica`, `lane` | jak `basal_queue_duration_seconds`, dla toru instancji |
| `basal_replica_batch_duration_seconds` | histogram | `model`, `replica`, `lane`, `outcome` | jak `basal_batch_duration_seconds`, dla toru instancji |
| `basal_replica_batch_requests` | histogram | `model`, `replica`, `lane` | jak `basal_batch_requests`, dla toru instancji |
| `basal_replica_available` | gauge | `model`, `replica` | 1, gdy główny worker instancji działa i backend nie zgłosił utraty urządzenia; liczone przy scrape |
| `basal_model_replicas` | gauge | `model` | liczba skonfigurowanych instancji modelu |
| `basal_model_replicas_available` | gauge | `model` | liczba dostępnych instancji modelu; liczone przy scrape |

Rezerwacja należy do zadania od wysłania do kolejki instancji do odpowiedzi
albo usunięcia zadania. Zwalnia się po błędzie planowania, po pominięciu
żądania porzuconego w kolejce, po partii także wtedy, gdy klient rozłączył się
podczas obliczeń, oraz przy zamknięciu albo panice workera. Różni się tym od
`basal_model_inflight_requests`, który liczy handlery HTTP i zwalnia miejsce
od razu po porzuceniu handlera. Router grupy replik wybiera dostępną replikę
z najmniejszą liczbą rezerwacji, a remisy rozstrzyga rotacja; wybór i
rezerwacja nie są jedną operacją atomową, więc równoległe żądania mogą trafić
do tej samej repliki. Liczniki rezerwacji pozostają dokładne. Limit
`max_inflight` nadal liczy handlery wszystkich modeli.

Dostępność to obserwowany stan workera i sygnał backendu, a nie test sprzętu.
CUDA zgłasza utratę (`device_lost`) tylko wtedy, gdy instancja nie
potwierdziła końca własnej pracy GPU (dwie nieudane synchronizacje albo
wiązania kontekstu); poza tym niedostępność na CUDA oznacza zatrzymany
główny worker repliki. Błąd walidacji, alokacji albo timeout żądania nie
wyłącza repliki. Zatrzymanie samego toru `long` nie zmienia
dostępności: jego żądania liczy tor główny.

```promql
sum by (model, replica) (rate(basal_replica_routed_requests_total[5m]))

basal_model_replicas_available < basal_model_replicas
```

## Start serwera

Metryki nie obejmują startu. `basal serve` przed otwarciem portu wybiera
konfigurację, odnajduje lub pobiera pliki modeli, dla każdego modelu ustala
tabelę GEMM (z `gemm_cache`, tabelę wbudowaną w binarkę CUDA po sprawdzeniu
przez cuBLASLt albo nowe wyszukiwanie, jednorazowo do kilkudziesięciu minut),
wczytuje wagi, przygotowuje silnik i sprawdza, czy tabela obejmuje kształty wag
modelu. Rejestr metryk powstaje dopiero potem, w `serve::serve`, przed
uruchomieniem wątków modeli (z drugim silnikiem toru `long`) i otwarciem portu.
Żadna metryka nie mierzy tych etapów ani uruchomienia wątków; ich czas jest
widoczny w logu:
`GEMM table done in … s` (wyszukiwanie), `loaded on GPU … in …s` (przy
podziale `loaded on GPUs 0 (layers 0-29), 1 (layers 30-59) in …s`),
`serving MODEL on GPU …` dla każdej instancji (przy podziale
`on GPUs 0, 1 (layer split)`) oraz `serving on … (ready … s after the
start)`. Do otwarcia portu `/health` i
`/metrics` nie odpowiadają.

Nie dotyczy to wykonywania GEMM podczas inferencji. Mnożenia macierzy są
częścią forwardów, więc ich czas wchodzi do `basal_batch_duration_seconds` i do
czasu żądania HTTP. Przy `gemm_table: none` (CUDA) pierwsze wywołanie danej
klasy liczby wierszy mierzy kandydatów cuBLASLt w trakcie partii, więc pierwsze
partie po starcie mogą być dłuższe.

Przy `quantization: q8_0` ładowanie obejmuje kwantyzację wag projekcji na CPU
i jednorazowe załadowanie kernela dekwantyzacji (log: liczba macierzy, bajty
bloków, liczniki skal zerowych i podnormalnych). Przy `q4_k` ładowanie
dodatkowo rekonstruuje i sprawdza każdą macierz na CPU i porównuje
dekwantyzację jednej macierzy na GPU z CPU (log: liczba macierzy, bajty bloków,
względny RMS błędu rekonstrukcji, liczniki grup, macierz kontroli); czas tego
toru w silniku nie został zmierzony. Dekwantyzacja każdej projekcji w
forwardzie wchodzi do czasu partii i żądania; osobnej metryki Prometheus nie
ma. Rozbicie czasu daje `basal profile` (`q8_dequant_ms_per_item`, dla `q4_k`
`dequant_ms_per_item`), a liczniki puli pamięci CUDA `basal benchmark`
(`memory_pool`, [Pomiary](BENCHMARKS.md)).

## Podział warstw

Dla modelu z `layer_split` metryki z poprzednich sekcji mają te same nazwy i
etykiety co dla modelu jednego GPU. Czas partii i żądania obejmuje wszystkie
etapy, kopie na granicach i czekanie na gate każdej karty. Prometheus nie ma
metryk transferów, etapów ani pamięci GPU. Liczniki transferów istnieją tylko
przy podziale i są dostępne poza serwerem:

- `basal profile`: `stage_transfers`, po jednym obiekcie na granicę
  (`from_stage`, `to_stage`, `from_device`, `to_device`, `copies`, `bytes`,
  `transfer_ms`, `gate_wait_ms`, `scope`), liczone dla profilowanych
  elementów, oraz sekcja czasu `stage_transfer`. Profil synchronizuje
  urządzenia, więc te czasy są diagnostyką, a nie latencją żądań;
- `basal benchmark`: `settings.layer_split` (etapy jak podano) i obiekt
  `layer_split`: `stages`, `transfers` (te same pola, od rozgrzewki do
  zamknięcia) oraz `devices`, po jednym wpisie na etap (`device`, `gpu`,
  `resources`, `estimated_weight_bytes`, `memory_pool`). Pola najwyższego
  poziomu (`memory_pool`, `resources`, okna pamięci faz) opisują kartę
  etapu 0; okna pamięci faz i końcowe `resources` mają dodatkowo
  `per_device` z każdą kartą;
- `basal bench`: `gpu_probe_tflops_per_device` (sonda GEMM przed i po
  pomiarze na każdej karcie), obok `gpu_probe_tflops` karty etapu 0;
- opis backendu (`backend` w manifeście eksportu i raportach):
  `layer_split` (etap, karta, pierwsza i ostatnia warstwa, `holds`: `embed`,
  `norm`, `lm_head`, `evidence_head`) i stały tekst `stage_transfer`, bez
  liczb; manifest eksportu ma też `run.layer_split`.

`transfer_ms` to czas hosta całego protokołu granicy: alokacji i
inicjalizacji celu, opróżnienia, kopii API, potwierdzenia na obu kartach,
zwolnienia źródła i wiązania kontekstu. Nie jest czystym czasem kopii ani
pasmem, a liczniki nie ustalają fizycznej trasy (P2P albo przez hosta). Bez
podziału żadne z tych pól nie występuje, a raporty mają dotychczasowy
format.

`basal doctor` sprawdza `layer_split` statycznie, bez ładowania modelu:
reguły etapów, konflikty z `device` i `replica_devices`, backend i `dtype`,
istnienie i typ kart każdego etapu oraz, przy dostępnym `config.json`,
pokrycie warstw. Pamięć wag każdego etapu jest szacunkiem z arytmetyki
formatu (udział warstw etapu, embedding na pierwszym, `lm_head` i norma na
ostatnim), dodawanym do pamięci danej karty; preflight `basal benchmark`
dolicza na ostatnim etapie plik głowicy evidence. To nie jest pomiar pamięci.

## Zapytania PromQL do Grafany lub Prometheusa

p50 i p99 poprawnych odpowiedzi na model, w sekundach, w oknie 5 minut:

```promql
histogram_quantile(0.50,
  sum by (model, le) (rate(basal_http_request_duration_seconds_bucket{status="200"}[5m]))
)

histogram_quantile(0.99,
  sum by (model, le) (rate(basal_http_request_duration_seconds_bucket{status="200"}[5m]))
)
```

To estymaty z histogramów, interpolowane w granicach bucketów. Brak ruchu daje
`NaN`; przy niewielu żądaniach p99 jest mało stabilne. Kwantyl w `+Inf` jest
ograniczony do najwyższej skończonej granicy (600 s). Nie uśredniaj percentyli
procesów ani replik: sumuj ich buckety, jak powyżej. Możesz dodać `endpoint` do
`sum by`, żeby rozdzielić API, `instance`, żeby rozdzielić procesy serwera, albo
użyć serii `basal_replica_*` z etykietą `replica` dla replik w jednym procesie.
Zobacz [dokumentację histogramów Prometheusa](https://prometheus.io/docs/practices/histograms/).

Żądania/s, liczba żądań z ostatniej godziny i błędy/s, na model i status:

```promql
sum by (model, status) (rate(basal_http_requests_total[5m]))

sum by (model, status) (increase(basal_http_requests_total[1h]))

sum by (model, status) (rate(basal_http_requests_total{status=~"4..|5.."}[5m]))
```

Tokeny wejściowe/s, tokeny wejściowe z ostatniej godziny i pytania/s na model:

```promql
sum by (model) (rate(basal_input_tokens_total[5m]))

sum by (model) (increase(basal_input_tokens_total[1h]))

sum by (model) (rate(basal_questions_total[5m]))
```

p99 oczekiwania na partię na model i tor, p99 poprawnych partii oraz średnia
liczba żądań w partii:

```promql
histogram_quantile(0.99,
  sum by (model, lane, le) (rate(basal_queue_duration_seconds_bucket[5m]))
)

histogram_quantile(0.99,
  sum by (model, lane, le) (rate(basal_batch_duration_seconds_bucket{outcome="success"}[5m]))
)

sum by (model, lane) (rate(basal_batch_requests_sum[5m]))
/
sum by (model, lane) (rate(basal_batch_requests_count[5m]))
```

W metrykach procesów bez ruchu liczniki tokenów, pytań, gauges oraz serie HTTP
ze statusem `200` są dostępne od startu. Pozostałe statusy pojawiają się przy
pierwszym wystąpieniu. `rate` i `increase` wymagają co najmniej dwóch próbek;
`increase` ekstrapoluje do granic okna i może zwracać wartości ułamkowe.
