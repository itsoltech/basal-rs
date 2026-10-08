# Observability: Prometheus

`basal serve` udostępnia `GET /metrics` na tym samym adresie i porcie co API,
bez dodatkowej flagi. Endpoint zwraca `text/plain; version=0.0.4`.
Każdy proces ma własny rejestr; liczniki zerują się po restarcie.

```sh
curl --fail localhost:8000/metrics
```

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
z kontenera Prometheusa, np. `basal:8000`. Dla wielu replik dodaj osobne targets
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
replik: sumuj ich buckety, jak powyżej. Możesz dodać `endpoint` do `sum by`,
żeby rozdzielić API, albo `instance`, żeby rozdzielić repliki.
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
