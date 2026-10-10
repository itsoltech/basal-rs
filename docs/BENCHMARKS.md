# Pomiary

## Kryteria

Punktem odniesienia jest forward FP32 upstream (autor ewaluował modele w FP32,
`basal.json: eval_dtype`). Zmianę runtime'u przyjmujemy, gdy:

- prompty, token IDs, litery, permutacje i pakowanie są identyczne z upstream;
- decyzje są takie same jak w FP32 upstream dla każdego przypadku zestawu
  odniesienia, a różnice logitów i prawdopodobieństw są zapisane w raporcie
  (zgodny argmax nie dowodzi zgodnej kalibracji);
- dla optymalizacji, które nie zmieniają obliczeń (pakowanie, cache, harmonogram),
  wynik jest bitowo równy poprzedniej wersji.

Szybkość mierzymy decyzjami (pytanie po dwóch porządkach opcji i kalibracji)
oraz żądaniami, z rozkładem opóźnień. Raport zapisuje warunki: GPU, limit
mocy, zegary, wersje bibliotek, rewizje modelu i upstream.

## Referencja upstream

Repozytorium upstream w przypiętej rewizji i jego środowisko Pythona
umieszczamy w `.baseline/` (poza repozytorium). Eksport referencji:

```sh
# basal-1.5-max: FP32 na CPU (wagi FP32 11B nie mieszczą się w 48 GB GPU razem z aktywacjami)
BASAL_UPSTREAM=.baseline/upstream-1.5 .baseline/upstream-1.5/.venv/bin/python \
  tools/reference/export_reference.py --model .models/basal-1.5-max \
  --repo Remek/basal-1.5-max --revision be1b5ee7e7a9755a931262fa7fab4f59be0fd03c \
  --mode eager --device cpu --dtype float32 \
  --cases tools/reference/systemone_cases_1.5.jsonl --out reports/NOWA-REFERENCJA
```

Wynik: `bench.jsonl` (44 przykłady basal-bench: prompty, token IDs, logity
liter obu porządków, rozkłady), `systemone.jsonl` (żądania System One wraz z
odpowiedzią `Server.decide`) i `manifest.json` (rewizje, wersje, sumy SHA-256).

## Zgodność runtime'u

```sh
# prompty i tokeny bez GPU
./target/release/basal check-prompts --model .models/basal-1.5-max --reference reports/reference-1.5-max-fp32

# wyniki runtime'u w formacie referencji
./target/release/basal export --model .models/basal-1.5-max --dtype f32 \
  --inputs reports/reference-1.5-max-fp32 --out reports/X/export-f32
./target/release/basal export --model .models/basal-1.5-max \
  --gemm-table reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json \
  --batching tree --inputs reports/reference-1.5-max-fp32 --out reports/X/export-f16-tree

# porównanie: tokeny, decyzje, różnice logitów i prawdopodobieństw, pola odpowiedzi
./target/release/basal compare --a reports/reference-1.5-max-fp32 --b reports/X/export-f16-tree --out reports/X/compare.json
```

`--batching single|budget|tree` i `--no-prefix-cache` pozwalają sprawdzić,
że wynik nie zależy od sposobu pakowania (porównanie dwóch eksportów runtime'u
powinno dać różnicę 0,0). Polecenia zapisujące wyniki odmawiają nadpisania
istniejącej ścieżki.

## Wydajność

| Narzędzie | Co mierzy |
|---|---|
| `basal bench` | pojedyncza decyzja metodyką basal-bench: 44 przykłady, oba porządki, batch 1, 39 pomiarów po odrzuceniu 5; przepustowość w grupach po 16 pytań |
| `basal bench-requests` | całe żądania System One, jeden klient (odpowiednik `tools/reference/bench_requests.py` dla upstream) |
| `basal benchmark` | rzeczywisty HTTP: 900 pytań sekwencyjnie, następnie mieszane rozmiary requestów przy różnych współbieżnościach; czasy odpowiedzi i kolejki, przepustowość, błędy oraz zasoby hosta |
| `tools/bench/ab.py` | porównanie dwóch wyników `bench` w parach, kolejność ABBA |
| `tools/bench/loadtest.py` | serwer HTTP, ten sam klient dla runtime'u i upstream: fazy sekwencyjna, zamknięta pętla N klientów i napływ otwarty (Poisson, `--rates` lub `--rate-fractions`); opóźnienia według klas żądań; z `--gpu` moc, zegar SM i J na decyzję z `nvidia-smi` |
| `tools/bench/make_long_states.py` | żądania o dokumentach 1k–16k tokenów |
| `tools/bench/make_mixed.py` | ruch mieszany: krótkie stany, wiele pytań, rozszerzenia 1.5, dokumenty do 16k |
| `tools/bench/split_requests.py` | rozbicie żądań wielopytaniowych na pojedyncze pytania o ten sam stan |
| `tools/perf/client_rss.py` | RSS procesu `basal client` z osobnym syntetycznym serwerem HTTP: szczyt z kernela, próbki w trakcie pracy, pamięć po opróżnieniu kolejki; na macOS opcjonalne skany `leaks` i podsumowanie stref malloc |

Pomiar pamięci klienta HTTP i jego ograniczenia: [reports/client-rss-20261010](../reports/client-rss-20261010/README.md).
Nie obejmuje pamięci lokalnego modelu ani jakości inferencji. Skrypt jest ręcznym narzędziem pomiarowym poza CI.

Przykład obciążenia serwera:

```sh
python3 tools/bench/make_mixed.py --model basal-1.5-max --out tools/bench/mixed.jsonl
python3 tools/bench/loadtest.py --gpu --model basal-1.5-max --requests tools/bench/mixed.jsonl \
  --n-warm 44 --n-seq 400 --n-conc 400 --concurrency 8 32 --rates 0.9 1.3 1.6 \
  --url http://127.0.0.1:8000/v1/systemone --out wynik.json
```

Loadtest porównuje odpowiedzi na to samo żądanie między fazami; różnica inna
niż 0,0 przy tabeli `--invariant` oznacza zależność wyniku od partii.
`answers_differing_vs_first` liczy odpowiedzi, których pełny obiekt `answers`
różni się od pierwszej odpowiedzi na dane żądanie, także w confidence,
evidence lub zestawie pól. Zerowa różnica prawdopodobieństw nie zastępuje
tej kontroli. Porównanie obejmuje powtórzenia wewnątrz jednego uruchomienia
klienta. `--answers-out FILE.json` zapisuje pierwsze pełne odpowiedzi według
ID do porównania między konfiguracjami (po sortowaniu po `id`); zgodność
logitów różnych buildów sprawdzamy osobnymi eksportami. `--gpu` zapisuje
także `gpu_memory_peak_sampled_mib`: najwyższe użycie całej pamięci GPU
zaobserwowane przez `nvidia-smi` co 200 ms, nie gwarantowany peak alokacji.

## Dobór modelu na własnej maszynie

`basal benchmark` uruchamia model i prywatny serwer na wolnym porcie
loopback. Mierzy requesty przez **ten sam router HTTP, kolejkę, batchowanie
i tor długich requestów co `basal serve`**. Serwer i generator ruchu są
zamykane po zakończeniu lub przerwaniu komendy. Nie trzeba uruchamiać
osobnego procesu `serve` ani instalować klienta Pythona.

Domyślnie wykonywane są dwa scenariusze:

1. **Sequential:** wszystkie 900 publicznych pytań, po jednym pełnym
   requeście System One, następny po odebraniu poprzedniej odpowiedzi.
   To zestaw z [porównania z FP32](../reports/decision-sets/README.md):
   po 100 z dziewięciu zbiorów, choice/noul/score, PL/EN. Klasy rozmiaru
   payloadu to ≤2 KiB, 2–16 KiB i >16 KiB.
2. **Mixed:** 900 requestów z powtarzalnego profilu mieszanego, osobny
   przebieg dla każdego poziomu `--concurrency` (domyślnie `1,4,8,16,32`).
   Każdy klient wysyła następny request po poprzedniej odpowiedzi
   (closed loop). Requesty o różnych rozmiarach konkurują w prawdziwej
   kolejce serwera. Kolejność mieszanego profilu jest stała, według SHA-256
   identyfikatorów z ziarnem 7; jest taka sama przy każdym poziomie.

Oba workloady są osadzone w binarce jako profil
`public_900_and_synthetic_mixed_v2`. Mixed korzysta z publicznych pytań i
istniejących syntetycznych fixture'ów `requests_fanout.jsonl`,
`systemone_cases_1.5.jsonl` i `long_states.jsonl`. Udziały odpowiadają
profilowi `make_mixed.py`:

| Klasa | Requesty | Udział | Payload |
|---|---:|---:|---|
| `short-1q` | 360 | 40% | pytania z publicznego zestawu, jedno na request |
| `short-multi` | 180 | 20% | kilka pytań o wspólny stan |
| `features` | 90 | 10% | multi, request mieszany typów, evidence, facts |
| `doc-1k-2k` | 180 | 20% | dokumenty około 1k–2k tokenów, 1 lub 5 pytań |
| `doc-4k-8k` | 72 | 8% | dokumenty około 4k–8k tokenów, 1 lub 5 pytań |
| `doc-16k` | 18 | 2% | dokumenty około 16k tokenów, 1 lub 5 pytań |

Nazwy klas dokumentów określają przybliżoną długość stanu. Raport podaje
rzeczywiste bajty payloadu i liczbę tokenów promptów obu porządków. Część
requestów powtarza te same stany i pytania; `workloads.*.unique_payloads`
pokazuje liczbę różnych payloadów. To profil obciążenia, a nie zestaw
oceny jakości. Osadzony mixed wymaga funkcji basal-1.5. Dla danych własnej
aplikacji należy użyć `--requests`.

Przed rozgrzewką benchmark waliduje pełne prompty dla wybranego modelu.
Requesty przekraczające jego limit kontekstu (`context_length_exceeded`)
są pomijane w całości, także gdy limit przekracza tylko jedno pytanie.
Dotyczy to osadzonego profilu i `--requests`; inne błędy planowania nadal
przerywają benchmark. Nazwa klasy nie decyduje o pominięciu: np. mini może
obsłużyć krótsze requesty `doc-4k-8k`, ale odrzucić dokumenty około 8k
po dodaniu pytania i szablonu promptu. Pusty workload po filtrowaniu
kończy komendę błędem bez uruchamiania pomiaru.

Raport `workloads.*` zawiera `source_requests`, `skipped_requests` oraz
listę `skipped` z ID, klasą i szczegółami błędów. `requests`, statystyki
rozmiarów i `ordered_payloads_sha256` opisują wyłącznie zachowany zestaw,
używany przy każdym poziomie concurrency. Pominięcia nie są błędami HTTP
i nie wchodzą do statystyk pomiaru. Przy porównywaniu modeli należy używać
wspólnego zestawu obsługiwanych requestów; filtrowanie może zmienić udziały
klas i liczbę mierzonych requestów.

Do pomiaru wystarcza sama binarka:

```sh
basal benchmark --model mini --json --out benchmark-mini.json
```

Komenda nie czyta plików z `.cache/`, nie pobiera zbiorów danych i nie
wymaga Pythona; pobiera tylko brakujące wagi modelu. Przed ładowaniem
modelu sprawdza SHA-256 osadzonych plików oraz liczby requestów, pytań,
klas i unikalnych ID z manifestu. `--scenario sequential` uruchamia tylko
bazową fazę, `--scenario mixed` tylko profil mieszany, `all` oba, np.
`basal benchmark --model 4.5B --scenario mixed --concurrency 1,8,32 --json
--out benchmark-4.5B.json`. Domyślny pełny benchmark to 5400 mierzonych
requestów (900 + 5 × 900) oraz rozgrzewka, o ile wszystkie requesty mieszczą
się w kontekście wybranego modelu.

Teksty 900 pytań pochodzą z publicznych zbiorów o własnych licencjach,
w tym niekomercyjnych lub badawczych (źródła w
[build.py](../tools/decision-sets/build.py)). Profil v2 zawiera je w
repozytorium i w binarce.

### Osadzony profil v2

Pliki `sequential.jsonl` (900 publicznych pytań w kolejności zestawu),
`mixed.jsonl` (900 requestów, już w kolejności SHA-256 ID) i
`manifest.json` leżą w
[crates/basal-cli/benchmark-data/v2](../crates/basal-cli/benchmark-data/v2)
i są niezmienne. [benchmark-v2.lock.json](../tools/decision-sets/benchmark-v2.lock.json)
przypina zestaw publiczny (SHA-256
`5b701fe861d626b3983928d3e5e168e452bbf6ba4137d75c76597e71e7691291`),
trzy fixture'y i trzy wynikowe pliki. Te same sumy są stałymi w
`crates/basal-cli/src/benchmark/workload.rs`. Inne dane lub receptura to
nowy profil w nowym katalogu, z nowym lockiem.

Pliki zawierają UTF-8, LF i nie zawierają nazwy modelu. Kolejność kluczy
i literały liczb są takie jak w źródłach, więc kolejność pytań, opcji
podanych obiektem `criteria` i tekst stanów będących obiektami są takie
jak w fixture'ach. Pole `model` z fixture'ów
pozostaje na swojej pozycji z wartością null; benchmark wpisuje tam nazwę
wybranego modelu. Requesty sequential nie mają pola `model`, więc nazwa
trafia na koniec obiektu. Manifest zapisuje sumy SHA-256 zestawu,
fixture'ów i każdego workloadu, recepturę mixed, liczniki klas, pytań i
różnych payloadów oraz sumę uporządkowanych payloadów niezależną od
modelu. Nie zapisuje czasu generowania ani danych maszyny.

Raport zapisuje w `source` wartość `kind: embedded`, nazwę profilu,
SHA-256 manifestu i zestawu oraz `workloads` z manifestu. Te wartości nie
zależą od modelu. `workloads.*.ordered_payloads_sha256` w głównej części
raportu jest liczone z body zawierającego nazwę modelu.

Pliki tworzy narzędzie maintainerów, bez sieci, tokenizera i GPU. Do
sprawdzenia wystarcza odtworzony zestaw publiczny i nowy katalog:

```sh
mkdir -p .cache/decision-sets
python3 tools/decision-sets/build.py .cache/decision-sets/set.jsonl
python3 tools/decision-sets/build_benchmark.py \
  --set .cache/decision-sets/set.jsonl --out .cache/decision-sets/benchmark-v2-check
cmp .cache/decision-sets/benchmark-v2-check/mixed.jsonl \
  crates/basal-cli/benchmark-data/v2/mixed.jsonl
```

Generator odmawia nadpisania katalogu, sprawdza wejścia i wyniki z lockiem
oraz odrzuca powtórzone klucze i liczby, których literał zmieniłby się po
ponownym zapisie.

`--profile v1` odtwarza historyczne pliki v1 bajt w bajt
([benchmark-v1.lock.json](../tools/decision-sets/benchmark-v1.lock.json)).
v1 sortował klucze na każdym poziomie. Prompty `sequential.jsonl` są takie
same jak w v2: stany są tekstem, choice ma klucze `option_0`–`option_5`,
score listę, a noul odczytuje `true`/`false` po nazwie. W `mixed.jsonl` v1
zmienia część requestów względem profilu v2: kolejność pytań w requestach
wielopytaniowych, kolejność opcji podanych obiektem `criteria` (np. skala
score `calm`, `annoyed`, `angry`) i tekst stanów będących obiektami
(102 requesty `short-multi`). Wyników mixed v1 nie porównuje się z v2.

Uczciwe porównanie wymaga tych samych współbieżności, rozgrzewki,
precyzji, ustawień schedulera i cache oraz podobnego obciążenia innych
aplikacji. Osadzony profil ma powtarzane payloady (licznik
`unique_payloads` w manifeście), więc wynik zależy również od cache.
Z `--state-cache-mb` większym od 0 cache stanu przechodzi między fazami:
kolejne poziomy współbieżności startują z cache wypełnionym wcześniej.
Deterministyczne są **dane i kolejność listy requestów**. Przy współbieżności
kolejność dotarcia i zakończenia requestów, skład partii oraz czasy zależą
od sprzętu, klienta HTTP i schedulera.

### Własny ruch

`--requests FILE` zastępuje oba osadzone workloady requestami z pliku.
Każda linia JSONL ma `request`, opcjonalnie `id` i `class`:

```json
{"id":"ticket-1","class":"short","request":{"state":"Paczka nie dotarła.","questions":{"urgent":{"type":"noul","instructions":"Czy sprawa jest pilna?"}}}}
```

```sh
basal benchmark --model mini --requests moje-requesty.jsonl \
  --concurrency 1,4,16 --json --out moje-wyniki.json
```

Pole `model` jest ustawiane na nazwę wybranego modelu: na miejscu
istniejącego pola albo na końcu obiektu. Bez `class` rozmiary
payloadów są klasyfikowane według tych samych progów bajtów co sequential.
Faza sequential zachowuje kolejność pliku, mixed układa jego requesty
deterministycznie i mierzy wszystkie przy każdym poziomie współbieżności.
Nie odtwarzamy timestampów ani niezależnego tempa napływu z logów.
Requesty są walidowane i tokenizowane przed rozpoczęciem pomiaru;
błędny lub nieobsługiwany payload przerywa komendę.

JSON (`schema_version: 2`) raportuje każdy scenariusz i współbieżność
w `phases[]`:

1. `latency_ms` dla wszystkich zakończonych prób i
   `successful_latency_ms` dla poprawnych odpowiedzi: `min`, `p50`, `p95`,
   `p99`, `mean`, `max`, `n`. Czas klienta obejmuje wysłanie requestu,
   oczekiwanie w kolejce, obliczenia i odebranie całego body odpowiedzi.
   Dekodowanie JSON i porównanie odpowiedzi są poza tym czasem.
   Mediana p50 i nearest-rank p99 dotyczą faktycznych próbek.
2. `queue_ms`, `batch_compute_ms`, `batch_requests` z nagłówków serwera.
   Czas compute obejmuje dobór, pakowanie i forwardy całej partii i jest
   współdzielony przez jej requesty; nie jest odizolowanym czasem
   inferencji jednego requestu. Walidacja i tokenizacja requestu na wątku
   GPU wliczają się do `queue_ms`. Brakujące nagłówki pozostają brakującymi
   próbkami.
3. `successful_requests_per_second`, `successful_questions_per_second`,
   `wall_s`, `statuses` (w tym 529), `error_kinds`, `errors`,
   `planned_requests`, `submitted_requests`, `cancelled_requests`.
   Błędy nie podnoszą raportowanej przepustowości poprawnych odpowiedzi.
4. `classes` zawiera te same rozkłady czasów, kolejki i błędy dla każdej
   klasy payloadów. `per_request` zachowuje ID, klasę, rozmiar payloadu,
   tokeny, status, czas i liczbę pytań. Dzięki temu widać, czy duże
   dokumenty pogarszają p99 krótkich requestów.
5. `resources`: najmniejszy zaobserwowany zapas RAM/GPU, szczyt RSS
   procesu oraz średnia liczba zajętych rdzeni CPU z okna próbek fazy.
   Serwer i generator są w jednym procesie, więc CPU/RSS obejmują oba.
   `answer_changes_vs_first_success` liczy różnice pełnych pól answers
   dla identycznego payloadu wobec pierwszej poprawnej odpowiedzi;
   nie stanowi sprawdzenia jakości lub kalibracji względem FP32.

Przed każdym workloadem domyślnie wykonywany jest jeden dodatkowy request
na klasę (`--warmup 0..10`); rozgrzewka nie wchodzi do statystyk faz.
`--max-p99-ms`, `--rate` i `--duty` zostały usunięte: benchmark opisuje
zmierzone zachowanie, nie ocenia modelu według zadanych progów i nie dodaje
przerw ograniczających pomiar przepustowości.

`--model` obsługuje lokalne ścieżki, skróty i przypięte rewizje; brakujące
wagi pobiera do cache. Tabela GEMM jest wybierana automatycznie jak w
`serve` (na CUDA może wymagać wygenerowania i zapisania w cache), chyba że
podano `--gemm-table`. Ustawienia GPU i cache stanu są zapisane w JSON;
tabela GEMM tylko nazwą pliku, bez ścieżki katalogu użytkownika.
Opcje `--max-batch-tokens`, `--max-inflight`, `--schedule`, `--long-tokens`
i `--long-slice-ms` mają takie same domyślne wartości jak `serve`.
Benchmark nie czyta ambientowej konfiguracji `basal-serve.yml`.

Domyślne rezerwy to 2 GiB RAM (`--ram-reserve-gib`) i 1 GiB GPU/Metal
working set (`--gpu-reserve-gib`). Przed ładowaniem dochodzi szacowany
rozmiar wag; dla GPU zunifikowanego także w RAM. Po załadowaniu monitor
sprawdza pamięć mniej więcej co 200 ms, także podczas aktywnych forwardów
i rozgrzewki. Utrata rezerwy, podwyższona presja pamięci macOS albo co
najmniej 64 MiB przyrostu zajętego swapu/zapisów do swapu przerywa ruch.
Swap jest globalny; wpływ innych aplikacji może zatrzymać benchmark.
RAM macOS to szacunek wolnych, nieaktywnych i spekulacyjnych stron.
Linux uwzględnia limity cgroup v1/v2 procesu i rodziców przy standardowym
mount `/sys/fs/cgroup`. CUDA mierzy faktycznie wybrane urządzenie,
z uwzględnieniem `CUDA_VISIBLE_DEVICES`; Metal zalecany working set.
W krótkiej fazie może zabraknąć próbek CPU; wtedy pole ma wartość null.
Okna pamięci obejmują ostatnie 20000 próbek (około 67 minut przy 5 Hz);
zbiorcze minima i wzrost swapu dotyczą całego monitorowanego przebiegu.

Klient i serwer dzielą limit otwartych plików procesu, po dwa gniazda na
klienta. Gdy limit jest mniejszy niż `2 × max(--concurrency) + 64`,
komenda kończy się przed ładowaniem modelu. Domyślny limit 256 w
terminalu macOS wystarcza do współbieżności 96; wyższa wymaga np.
`ulimit -n 4096`.

`--timeout-ms` domyślnie wynosi 120000. Timeout, błąd transportu,
niepoprawna odpowiedź, błąd serwera inny niż przeciążenie 529 albo Ctrl-C
zatrzymuje dalszy ruch. Requesty w toku są anulowane po stronie klienta;
serwer kończy aktywną pracę GPU przed zwolnieniem silników. To może
opóźnić zakończenie komendy. Błędy 4xx/529 są zachowane w statystykach.
Nie mierzymy niezależnego napływu open loop, mocy ani temperatury.
Próbki pamięci nie gwarantują uchwycenia szczytu lub uniknięcia OOM;
profil nie gwarantuje stabilności pod długotrwałym obciążeniem.

Pełny przebieg ma `complete: true` oraz `completed` lub
`completed_with_errors`, kod wyjścia 0. Przerwanie ma `complete: false`,
status `aborted`/`not_run` i kod 1; próbki ukończonych requestów są
zachowane. Aplikacja powinna sprawdzać kompletność i liczniki błędów.
`--out` nie nadpisuje pliku, a `--json` zostawia stdout dla JSON i postęp
na stderr. SIGKILL/OOM, a także Ctrl-C przed startem serwera (pobieranie
i ładowanie modelu, tabela GEMM, tokenizacja workloadów), może pozostawić
pusty plik. Błąd argumentów, limitu otwartych plików, źródła przed
ładowaniem lub zajętej ścieżki kończy komendę bez raportu.

## Pomiary na wynajętym GPU

`tools/cloud/basal-cloud.py` wynajmuje maszynę z GPU w Shadeform (domyślnie RTX
6000 Ada, jak serwer pomiarów; inne karty przez `--gpu`), buduje binarkę CUDA
z bieżącego drzewa na hoście z Dockerem (`tools/release/linux.Dockerfile`,
etap `dev`), kopiuje ją i dane, uruchamia polecenie i pobiera wyniki:

```sh
python3 tools/cloud/basal-cloud.py template                 # raz: szablon basal-rs-test
python3 tools/cloud/basal-cloud.py up --hours 2 --spend 3 --prefetch "4.5B max"
python3 tools/cloud/basal-cloud.py build --host USER@HOST   # równolegle z up
python3 tools/cloud/basal-cloud.py push
python3 tools/cloud/basal-cloud.py sync reports/choice-sets/set.jsonl
python3 tools/cloud/basal-cloud.py run 'basal-dev eval-choice-set --model max --set reports/choice-sets/set.jsonl --out out/x'
python3 tools/cloud/basal-cloud.py pull out/x reports/NOWY-RAPORT
python3 tools/cloud/basal-cloud.py down
```

Każda maszyna ma w Shadeform automatyczne usunięcie po zadanym czasie i
kwocie. Raporty z takiej maszyny podają kartę, sterownik i limit mocy, bez
adresu maszyny. Zasady sesji (budżet, pilnowanie maszyny, skrypty pomiarowe,
pułapki sterowników i profilerów): [CLOUD.md](CLOUD.md).

## Warunki

Pomiary CUDA wykonano na RTX 6000 Ada z limitem mocy 250 W (domyślnie 300 W).
Model 11B pracuje przy tym limicie stale: zegar SM spada do ~750–1000 MHz, więc
bezwzględne czasy są wyższe niż przy pełnej mocy, a porównania dotyczą tych
samych warunków. Upstream mierzony jest w trybie `fast` (BF16, torch.compile,
CUDA graphs), czyli w trybie, w którym serwuje.

Zestawy odniesienia (44 przykłady autora, 34 żądania System One dla
basal-1.5) wystarczają do sprawdzenia zgodności z upstream, ale nie do oceny
jakości na danych konkretnej aplikacji.
