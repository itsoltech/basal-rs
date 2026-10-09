# Klient do skryptów i przetwarzania danych

`basal client` zamienia dane ze stdin lub pliku na decyzje System One. Domyślnie wysyła requesty HTTP do
`http://127.0.0.1:8000`. Z `--local` ładuje model raz, przetwarza dane do EOF i kończy proces, bez uruchamiania serwera.

Wymaga binarki zawierającej polecenie `client` (`basal client --help`). Przykłady zakładają uruchomienie z katalogu
repozytorium. Serwer uruchamiasz osobno przez `basal serve`; na maszynie wykonującej klienta HTTP GPU nie jest potrzebne.
Przykłady z `jq` wymagają jego instalacji. Modele i semantyka decyzji: [SYSTEM_ONE.md](SYSTEM_ONE.md).

## 1. Jedno pytanie bez pliku JSON

Bez opcji odpowiedzi `--ask` tworzy pytanie `noul` i zwraca prawdopodobieństwo odpowiedzi „tak”.

```bash
cat zgloszenie.txt | basal client --ask 'Czy zgłoszenie opisuje awarię blokującą pracę?'

# Samo prawdopodobieństwo, np. 0.91, wygodne dla jq lub zmiennej Bash.
cat zgloszenie.txt | basal client --ask 'Czy zgłoszenie opisuje awarię blokującą pracę?' --value
```

Z `--choice` wybierasz jedną opcję. Klucz jest wynikiem dla skryptu, opis wyjaśnia jego znaczenie modelowi.
Pierwszy znak `=` oddziela klucz od opisu; bez `=` opcja jest opisana samym kluczem. Powtarzaj flagę dla każdej opcji.

```bash
department=$(basal client --state zgloszenie.txt \
  --ask 'Do którego działu skierować zgłoszenie?' \
  --choice 'returns=Zwroty i reklamacje' \
  --choice 'support=Pomoc techniczna' \
  --choice 'sales=Pytania przed zakupem' \
  --choice 'other=Inna sprawa lub brak informacji' --value)
printf '%s\n' "$department"
```

Przy wielu wynikach dodaj „inne”/„brak danych”, jeżeli mają być dopuszczalne. `choice` zawsze wybiera jedną
z podanych opcji. `--value` wymaga dokładnie jednego pytania: wypisuje klucz choice/action jako tekst, noul/score
jako liczbę, multi jako tablicę JSON. Bez `--value` dostajesz pełną odpowiedź API (`model`, `answers`, `usage`).
Pytanie utworzone przez `--ask` ma ID `result`, np. `.answers.result.choice`.

## 2. Warunek w Bash: błąd i odpowiedź „nie” to różne przypadki

```bash
#!/usr/bin/env bash
set -euo pipefail

# Nie używaj samego kodu wyjścia klienta jako decyzji: oznacza on powodzenie wykonania requestu.
if probability=$(basal client --state zgloszenie.txt \
  --ask 'Czy zgłoszenie opisuje awarię blokującą pracę?' --value); then
  if jq -e '. >= 0.85' <<< "$probability" >/dev/null; then
    printf 'Przekaż zgłoszenie do kolejki pilnych spraw\n'
  else
    printf 'Przekaż zgłoszenie do zwykłej kolejki\n'
  fi
else
  status=$?
  printf 'Nie udało się ocenić zgłoszenia (exit %s)\n' "$status" >&2
  exit "$status"
fi
```

Próg 0.85 jest przykładem polityki automatyzacji, nie gwarancją trafności. `noul` jest P(tak); `choice.confidence`
to inna miara ([definicja](SYSTEM_ONE.md#odpowiedź)).

## 3. JSONL: wiele rekordów, zachowane ID, równoległe requesty

Plik [routing.json](../examples/client/routing.json) definiuje dwa pytania: dział i pilność. To zwykły szablon
requestu zawierający `questions`; nazwa pliku jest dowolna. Klient zastępuje `state` danymi wejściowymi.

```bash
set -o pipefail
basal client --template examples/client/routing.json \
  --state examples/client/tickets.jsonl --input jsonl --state-pointer /text --jobs 8 |
  jq -c 'select(.ok) | {
    id: .input.id,
    department: .response.answers.department.choice,
    urgent: (.response.answers.urgent.noul >= 0.85)
  }'
```

`--state-pointer /text` wybiera pole oceniane przez model. `.input` zachowuje **cały** oryginalny rekord,
więc ID i pozostałe pola nie giną. Bez pointera cały obiekt lub tablica trafia do `state`.
Pointer to standardowy JSON pointer (`/customer/message`, `~1` oznacza `/`, `~0` oznacza `~`).
Brak wskazanego pola jest błędem rekordu.

W formatach strumieniowych każda niepusta linia daje jedną linię wyniku:

```json
{"index":0,"ok":true,"input":{"id":101,"text":"…"},"response":{"model":"…","answers":{},"usage":{}}}
```

Przykład pokazuje kształt koperty; rzeczywisty `response` zawiera odpowiedzi modelu. `index` liczy niepuste rekordy
od zera. Błędny JSON daje `input: null`. Poprawny JSON z błędnym requestem zachowuje oryginalne `.input`.

`--jobs` ogranicza równoległość HTTP (domyślnie 1, maksymalnie 256). Klient zachowuje kolejność wejścia i buforuje
najwyżej tyle rekordów, ile wynosi `--jobs`; wolniejszy pierwszy request może opóźnić wypisanie kolejnych.
Wypisuje i flushuje każdy wynik, również kiedy producent wejścia nadal działa. Nie wczytuje całego JSONL do RAM.

Domyślnie pierwszy błąd w kolejności wyjścia kończy pracę kodem 1. Przy równoległości inne requesty mogły już zostać
wysłane. `--keep-going` przetwarza pozostałe rekordy i kończy kodem 1, jeśli którykolwiek się nie powiódł.
Błąd ma kopertę `{"index":0,"ok":false,"input":...,"error":{"kind":"http","message":"HTTP 529","status":529,"detail":...}}`.

Gotowy skrypt zapisuje pełne wyniki, uproszczone przypisanie działów oraz błędy, zachowując końcowy kod wyjścia:

```bash
bash examples/client/classify-tickets.sh examples/client/tickets.jsonl ./ticket-results
# ticket-results/results.jsonl, routed.jsonl, errors.jsonl
```

Katalog wyników musi być nowy; skrypt nie nadpisuje poprzednich rezultatów. Obsługuje też stdin:

```bash
jq -c '.tickets[]' export.json |
  bash examples/client/classify-tickets.sh - ./ticket-results-2
```

## 4. Tekst, logi i xargs

Wybierz jawnie sposób podziału danych:

- `--input text` (domyślnie z `--ask`/`--template`): cały UTF-8, również wielowierszowy, to jeden stan.
  Zachowuje końcowy znak nowej linii; pusty plik lub stdin daje pusty stan tekstowy. Aby przekazać tekst bez
  końcowego znaku nowej linii, użyj `printf '%s' 'treść'` zamiast `echo`.
- `--input json`: cały input to jeden obiekt, tablica lub string JSON.
- `--input lines`: każda niepusta linia tekstu to osobny stan; CRLF jest obsługiwane.
- `--input jsonl`: każda niepusta linia to osobny JSON. Puste/białe linie są pomijane.

Przykład przesiewania logów; `set -o pipefail` zachowuje błąd klienta pomimo końcowego `jq`:

```bash
set -o pipefail
tail -n 1000 application.log |
  basal client --ask 'Czy wpis wskazuje utratę danych lub niedostępność usługi?' --input lines |
  jq -c 'select(.ok and (.response.answers.result.noul >= 0.85)) | {line:.input,p:.response.answers.result.noul}'
```

Do strumienia działającego bez końca można użyć `tail -F` zamiast `tail -n`. Warto wcześniej odfiltrować
nieistotne wpisy zwykłymi narzędziami, aby nie wysyłać każdego wiersza do modelu.

Klasyfikacja plików, także nazw zawierających spacje:

```bash
find ./documents -type f -name '*.txt' -print0 |
  xargs -0 -n 1 bash examples/client/classify-file.sh
```

Skrypt zwraca JSONL `{path,category}`. Pojedyncze wywołania `xargs` pasują do HTTP. Dla lokalnej inferencji
przesyłaj rekordy w jednym strumieniu, aby nie ładować modelu od nowa dla każdego pliku.

## 5. Score, multi i act

Score: poziomy są uporządkowane od 0; wynik jest średnią ważoną, więc może być ułamkowy:

```bash
cat zgloszenie.txt | basal client --ask 'Oceń wpływ problemu na działanie produktu.' \
  --level 'Brak utrudnień' --level 'Drobne utrudnienia' \
  --level 'Istotne ograniczenia' --level 'Produkt nie działa' --value
```

Multi: dowolna liczba etykiet, każda oceniana osobno. `--threshold` ustala próg wyboru:

```bash
cat zgloszenie.txt | basal client --ask 'Zaznacz wszystkie tematy zgłoszenia.' \
  --label 'damage=Uszkodzenie produktu' --label 'refund=Prośba o zwrot pieniędzy' \
  --label 'delivery=Problem z dostawą' --threshold 0.7 --value
```

`--choice`, `--level` i `--label` są wzajemnie wykluczające. `act` oraz większe reguły zapisuj w szablonie.
[all-types.json](../examples/client/all-types.json) zawiera kompletny przykład wszystkich pięciu typów.

```bash
cat zgloszenie.txt | basal client --template examples/client/all-types.json |
  jq '.answers'
```

W przykładzie `act` szacuje, czy reklamacja spełnia warunki zwrotu, i wybiera działanie o najmniejszym oczekiwanym
koszcie. Koszt błędnego przyznania to 10, błędnej odmowy 20, przekazania człowiekowi 2 (przykładowa polityka).
Warunki zwrotu trzeba dostarczyć w `state`, np. jako obiekt z `policy` i `ticket`.
Wynik `.answers.refund_action.action` to `approve`, `reject` lub `human`; klient sam nie wykonuje tych działań.
`defer_action` wskazuje akcję odroczenia; bez `max_error` nadal uczestniczy ona w zwykłym wyborze według kosztów.
`max_error` wymaga odpowiednich progów w kalibracji modelu; nie jest uniwersalną gwarancją trafności.
`multi` i `act` są rozszerzeniami basal-1.5, nie standardowymi typami API TypeSafe.

Pytania w szablonie są niezależne i oceniają ten sam stan. Jedno pytanie nie korzysta automatycznie z odpowiedzi
na inne; taki następny etap budujesz kolejnym pipe i requestem.

## 6. Pełne requesty i lokalny model

Istniejące requesty System One można wysyłać bez szablonu:

```bash
basal client --request request.json | jq '.answers'
cat requests.jsonl | basal client --request - --input jsonl --jobs 8
```

`--request` domyślnie czyta `json`, dopuszcza też `jsonl`. Nie łącz go z `--ask`, `--template` ani `--state-pointer`.

Lokalny model ładuje się przy pierwszym poprawnym rekordzie i zostaje w pamięci do EOF. Domyślnie wybierany jest
ten sam model co przy `basal serve` (4.5B), chyba że podano `--model`, `BASAL_MODEL` lub model w request/szablonie.
Pierwszy start może pobrać wagi i przygotować tabelę GEMM. Dla wielu rekordów ładowanie następuje tylko raz.
Błąd ładowania modelu zatrzymuje cały lokalny proces także z `--keep-going`, zamiast ponawiać ładowanie dla każdego rekordu.

```bash
basal client --local --model mini --state examples/client/tickets.jsonl \
  --template examples/client/routing.json --input jsonl --state-pointer /text
```

Lokalnie rekordy są przetwarzane kolejno, `--jobs` musi wynosić 1. Obsługiwane są skróty `mini`, `4.5B`, `max`,
repozytorium `owner/name@revision` oraz lokalny katalog, jak w `serve`. Nazwa z manifestu załadowanego modelu trafia
do requestów. Jeden lokalny proces obsługuje jeden model; pełne requesty kierujące następne rekordy do innego modelu
zwracają błąd. `--dtype` i `--gemm-table` dotyczą lokalnej inferencji; automatyczny wybór i kontrola pokrycia tabel
GEMM używają tej samej logiki co serwer. Porównanie odpowiedzi lokalnych i HTTP wymaga tej samej konfiguracji modelu;
`usage.latency_ms` nie jest miarą zgodności numerycznej.

## Konfiguracja i kontrakt automatyzacji

HTTP:

```bash
export BASAL_URL='http://127.0.0.1:8000'
export BASAL_MODEL='basal-1.5-mini'
# Jeśli serwer/reverse proxy wymaga autoryzacji, ustaw BASAL_API_KEY przez swój mechanizm sekretów.
# Alternatywnie --key-env NAZWA_ZMIENNEJ; token nie jest argumentem CLI.
```

`--url` ma pierwszeństwo nad `BASAL_URL`. `--model` ma pierwszeństwo nad `BASAL_MODEL`, następnie nad modelem
w request/szablonie. Skróty znanych modeli są normalizowane (`mini` → `basal-1.5-mini`). Gdy modelu nie podano,
HTTP pobiera `/v1/models` i wybiera model tylko wtedy, gdy lista zawiera dokładnie jeden; przy wielu modelach
trzeba wskazać go jawnie. Brak połączenia nie uruchamia lokalnego modelu. Klient respektuje `HTTP_PROXY`,
`HTTPS_PROXY`, `ALL_PROXY` i `NO_PROXY`, także dla adresu lokalnego. Przy ustawionym proxy dodaj serwer lokalny
do `NO_PROXY` (np. `127.0.0.1,localhost`), aby requesty nie przechodziły przez proxy.

- stdout: kompaktowy JSON/JSONL, albo sama wartość przy `--value`. Logi i błędy pojedynczych requestów: stderr.
- Exit `0`: wszystkie rekordy wykonane poprawnie (także decyzja „nie” i pusty strumień); `1`: co najmniej jeden
  błąd rekordu/requestu; `2`: konfiguracja, wejściowe/wyjściowe I/O lub przerwanie workera; Ctrl-C HTTP: `130`.
- `--value` nie obsługuje `--keep-going`: błąd zatrzymuje pracę, zamiast pomijać rekord i gubić powiązanie.
- `--timeout` to czas jednej próby HTTP razem z body (domyślnie 120 s). `--retries` to dodatkowe próby (domyślnie 0,
  maksymalnie 10) tylko dla problemów połączenia/odbioru i HTTP 408, 429, 500, 502, 503, 504, 529.
  Opóźnienia to 1, 2, 4, 8, potem 16 s; `Retry-After` w sekundach ma pierwszeństwo. Inny format lub wartość powyżej
  30 s kończy request błędem zamiast skracać czas wymagany przez serwer. Walidacja 400/422 nie jest ponawiana.
- `--max-input-bytes` ogranicza rekord i szablon (domyślnie 2 MiB, maksymalnie 64 MiB); odpowiedź HTTP ma limit
  16 MiB. Limit klienta nie zwiększa limitu HTTP serwera/proxy ani kontekstu modelu. Nie ma automatycznego obcinania.
- Pamięć HTTP rośnie z `--jobs`: każdy z najwyżej `--jobs` rekordów w toku trzyma zakodowany request, odpowiedź
  (do 16 MiB) oraz sparsowany rekord wejściowy, jeśli trafia on do koperty. JSON po sparsowaniu zajmuje zwykle
  więcej niż jego tekst. Przy `--jobs 1` klient działa na jednym wątku runtime Tokio.

Zamknięcie dalszego etapu pipe (np. `head`) kończy klienta bez tracebacku, kodem 0. Buforowane HTTP taski są
anulowane, lecz request już przyjęty przez serwer może dokończyć inferencję. Ctrl-C HTTP przerywa oczekiwanie na
stdin/requesty; lokalny proces podlega zwykłej obsłudze sygnałów systemowych. Przekierowania HTTP nie są śledzone.

## Granica weryfikacji

Formatowanie, Clippy, Rustdoc i lint przykładów sprawdzają kod oraz składnię. Nie potwierdzają odpowiedzi modelu,
zgodności lokalnie/HTTP ani wydajności. E2E i inferencja wymagają osobnej, uprzedniej zgody zgodnie z zasadami repozytorium.
