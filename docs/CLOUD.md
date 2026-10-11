# Eksperymenty na wynajętym GPU

[tools/cloud/basal-cloud.py](../tools/cloud/basal-cloud.py) wynajmuje w Shadeform maszynę z GPU, kopiuje na nią
binarkę CUDA z bieżącego drzewa i dane, uruchamia pomiar i pobiera wyniki. Skrypty pomiarowe dla takiej maszyny
leżą w [tools/perf](../tools/perf) i [tools/decision-sets](../tools/decision-sets). Pełna lista opcji jest w
nagłówku `basal-cloud.py`.

## Zgoda i budżet

Każda sesja kosztuje, więc przed jej startem agent pyta użytkownika, jaki budżet przeznacza na eksperyment i na
jakiej karcie. Podaje przy tym kartę, cenę za godzinę z `types`, szacowany czas i plan pomiarów. Domyślna karta to
RTX 6000 Ada (karta pomiarów w [reports](../reports)); inne karty tylko za zgodą użytkownika, osobno dla każdej
sesji. `--hours` i `--spend` przy `up` wynikają z budżetu, więc maszyna sama się usunie, gdy budżet się wyczerpie.

Kwot, cen i kosztów sesji nie zapisujemy w repozytorium: w raportach, dokumentacji ani w commitach. Raport podaje
kartę, sterownik, limit mocy i zegar, bez adresu maszyny, nazwy hosta i identyfikatora instancji.

## Wymagania

- `shade` CLI zalogowane (`shade auth login`) i klucz SSH zapisany w Shadeform pod nazwą `BASAL_CLOUD_SSH_KEY`.
- Host budowania z Dockerem i buildx (`BASAL_BUILD_HOST` lub `build --host USER@HOST`), GPU nie jest potrzebne.
  Adres hosta nie trafia do repozytorium.
- Szablon `basal-rs-test`: `basal-cloud.py template` raz i po każdej zmianie `START_SCRIPT`.

## Przebieg sesji

Maszyna kosztuje od chwili utworzenia, więc wszystko, co nie potrzebuje GPU, dzieje się przed `up` albo
równolegle z nim:

```sh
python3 tools/cloud/basal-cloud.py types --gpu H100                 # oferty teraz, od najtańszej
python3 tools/cloud/basal-cloud.py build --host USER@HOST           # równolegle z up
python3 tools/cloud/basal-cloud.py up --gpu H100 --compat --max-price P --hours H --spend S --prefetch "4.5B max"
python3 tools/cloud/basal-cloud.py push
python3 tools/cloud/basal-cloud.py sync tools/bench tools/perf crates/basal-cli/gemm-tables \
  reports/reference-basal-1.5-4.5B-fp32 reports/reference-1.5-max-fp32
python3 tools/cloud/basal-cloud.py run 'nohup bash tools/perf/run-wg-cloud.sh > wg.log 2>&1 < /dev/null &'
python3 tools/cloud/basal-cloud.py pull out/wg reports/NOWY-RAPORT
python3 tools/cloud/basal-cloud.py down
python3 tools/cloud/basal-cloud.py ls                               # nic nie powinno zostać
```

- `up` czeka, aż skrypt startowy zainstaluje wydanie, biblioteki CUDA 12.9.1 (`basal setup --force`) i modele z
  `--prefetch`, po czym zapisuje `/data/ready`. Gdy dostawca nie uruchomi maszyny w `--boot-timeout` minut
  (status `pending_provider`), usuwa ją i bierze następną ofertę.
- `--max-price` domyślnie przepuszcza tylko tanie karty; dla H100 trzeba go podnieść do ceny z `types`.
- `--compat` dopuszcza obrazy ze starszym sterownikiem (oferty H100 mają zwykle 535 lub 570). Skrypt startowy
  instaluje wtedy biblioteki forward-compatibility NVIDIA.
- Porównanie dwóch buildów (`run-forward-cloud.sh`, `run-ab.sh`): najpierw `push` buildu A i
  `run 'cp /opt/basal-dev/basal-cuda /opt/basal-dev/basal-cuda-base'`, potem `build` i `push` buildu B. Oba
  buildy najlepiej zbudować przed `up`; `build` nadpisuje `.cache/cloud/basal-cuda`, więc A trzeba skopiować obok.
- Długie skrypty uruchamiaj w tle (`nohup ... &`): `run` trzyma połączenie SSH i jego zerwanie przerwałoby pomiar
  na pierwszym planie.
- Kilka maszyn naraz: `BASAL_CLOUD_SESSION=NAZWA` przy każdym poleceniu (osobny plik sesji w `.cache/cloud`).
- `down` usuwa maszynę zaraz po `pull`. Wyniki, których nie pobrano, giną razem z maszyną.

## Skrypty pomiarowe

Każdy skrypt opisuje w nagłówku, co mierzy, jakie ścieżki trzeba wcześniej przesłać przez `sync` i dokąd pisze.

| skrypt | pomiar |
|---|---|
| `tools/perf/run-cloud.sh` | basal-rs wobec upstream `fast`: pojedyncza decyzja, HTTP 1/8/32 klientów, obciążenie mieszane |
| `tools/perf/run-context.sh` | drabina kontekstu 0,5k–16k tokenów wobec upstream, sekcje forwardu |
| `tools/perf/run-forward-cloud.sh` | A/B dwóch buildów: eksporty (0,0), `basal bench`, drabina, zegar zablokowany |
| `tools/perf/run-ab.sh` | A/B dwóch buildów i wariantu attention |
| `tools/perf/run-wg-cloud.sh` | warianty attention jednego buildu (`WG_VARIANTS`), eksporty wobec FP32 i siebie nawzajem |
| `tools/perf/run-gemm-check.sh` | zmiana wyszukiwania tabel GEMM: stara i nowa tabela, heurystyka, zestaw decyzji |
| `tools/perf/run-gemm-tables-cloud.sh` | brakujące tabele f16 z zainstalowanego wydania: generowanie, bajtowa zgodność single/tree/budget, 44 i 900 pytań wobec FP32, benchmark i krótka drabina kontekstu |
| `tools/perf/run-profile-cloud.sh` | Nsight Systems: kategorie kerneli, timeline i API; opcjonalnie Nsight Compute dla wskazanego uruchomienia attention |
| `tools/perf/run-parity-cloud.sh` | identyczność bajtowa eksportów dwóch buildów: single/tree/budget, cache prefiksu i cache stanu |
| `tools/perf/run-http-variants-cloud.sh` | HTTP A/B/B/A jednego buildu: pełne odpowiedzi, latency, throughput, energia i VRAM |
| `tools/perf/run-multigpu-cloud.py` | modele na osobnych GPU (dwie identyczne karty): `doctor` i odrzucane konfiguracje, eksporty baseline i kandydata na GPU 0 i 1 (bajtowo i `basal compare`), jedno API z dwoma modelami na dwóch GPU i na jednym, odwrócone `CUDA_VISIBLE_DEVICES`, zamknięcie serwera |
| `tools/perf/run-isolation-cloud.py` | wpływ obciążonego 4.5B na GPU 0 na mini na GPU 1 w jednym API (identyczny ruch mini: zamknięta pętla i otwarty napływ według stałego harmonogramu, A/B/B/A) oraz mini na jednej karcie w binarce sprzed i po przypisaniu urządzeń |
| `tools/perf/run-replicas-cloud.py` | repliki jednego modelu (`replica_devices`) na dwóch identycznych kartach: eksporty binarki rozmieszczenia i kandydata na GPU 0 i 1, odrzucane konfiguracje, zgodność odpowiedzi HTTP, routing i metryki replik, admission 529, rozłączeni klienci, `CUDA_VISIBLE_DEVICES`, zamknięcie; wydajność jednej i dwóch replik A/B/B/A |
| `tools/perf/run-w8a16-cloud.py` | opcjonalne wagi `q8_0` (W8A16) na dwóch identycznych kartach: tor `none` bajtowo wobec binarki etapu 02, spójność `q8_0` między trybami i kartami, odrzucane konfiguracje, HTTP z replikami, 44+34 i 900 pytań wobec FP32, wydajność `none` (A/B z potwierdzeniem) i `q8_0`, profil dekwantyzacji, pamięć puli |
| `tools/decision-sets/run-cloud.sh` | zestaw decyzji wobec upstream FP32 |

## Pilnowanie maszyny

Maszyna zawieszona albo skrypt, który przerwał się po cichu, kosztują tyle samo co działający pomiar. W trakcie
pomiaru co minutę sprawdzaj przez `run`: liczbę linii logu, to, czy proces skryptu żyje, i obciążenie GPU
(`nvidia-smi --query-gpu=utilization.gpu`). Brak nowych linii przez 20 minut oznacza zawieszenie; wtedy obejrzyj
log i stan procesu zamiast czekać na limit `--hours`. Gdy `run` przestaje odpowiadać, sprawdź `ls`: maszyna mogła
zostać usunięta po przekroczeniu `--hours` lub `--spend`. Pierwsza linia `up` zawiera słowa „deleted at”, więc nie
szukaj w jej wyjściu tego wzorca jako znaku usunięcia.

Procesy na maszynie wybieraj wzorcem z nawiasem, np. `pgrep -f "[r]un-wg-cloud.sh"`, bo zwykły wzorzec pasuje też
do powłoki SSH, która go wykonuje; `pkill` z takim wzorcem zabija własne polecenie. Skrypt opakowujący, który na
końcu woła `down`, usunie maszynę również wtedy, gdy przerwiesz któryś z jego kroków.

## Pułapki

- Sterownik starszy niż 575: wrapper `basal-dev` sam dodaje `/usr/local/cuda-12.9/compat` do
  `LD_LIBRARY_PATH`. Inne binarki (wydanie w `/opt/basal-release`, programy diagnostyczne, `ncu`) potrzebują
  `LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:/data/basal/.local/cuda/12.9.1/lib`; bez tego kończą się błędem
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. `sudo` czyści środowisko, więc pod nim: `sudo env LD_LIBRARY_PATH=... ncu`.
- Obrazy mają własne biblioteki CUDA 12.8; skrypt startowy nadpisuje je przez `basal setup --force`, bo tabele GEMM
  są przypisane do wersji cuBLASLt. Szablon zapisany przed tą zmianą trzeba zapisać ponownie (`template`).
- `sync` nie usuwa plików: plik skasowany lokalnie zostaje w `/work` (np. stara tabela GEMM w
  `crates/basal-cli/gemm-tables`, którą skrypty biorą jako wbudowaną). Usuń go na maszynie przez `run 'rm ŚCIEŻKA'`.
- Nsight 2026.3 nie działa ze sterownikiem 570; działa Nsight Systems 2025.1.3 (pakiet
  `nsight-systems-2025.1.3`). `run-profile-cloud.sh` wymaga wcześniej zainstalowanej zgodnej wersji;
  nie instaluje pakietów. Wersja 2025.1.3 jest wcześniejszym sprawdzonym przykładem, nie uniwersalnym wyborem.
- `ncu --launch-skip` liczy uruchomienia pasujące do filtra kernela. Nie ma stałej wartości gwarantującej
  pominięcie prefiksów i rozgrzewki: wybierz ją z timeline dla konkretnego modelu i wariantu. Runner wymaga
  jawnego `PROF_LAUNCH_SKIP`, gdy włączono `PROF_NCU=1`; domyślnie wykonuje tylko Systems.
  Semantyka filtrów: [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
- Blokada zegara SM (`sudo -n nvidia-smi -lgc`) nie u każdego dostawcy jest dozwolona; skrypty zapisują jej wynik
  w `lock.txt`. Przy pełnej mocy karta schodzi chwilami poniżej zablokowanego zegara, więc zegary trafiają do logu.

## Metodyka

Pomiary czasu: warianty w rotacji (A B B A ...), kilka powtórzeń, mediana bez pierwszego, zegar SM zablokowany,
gdy dostawca na to pozwala. Zmiana bez wpływu na numerykę musi dawać eksporty bitowo równe (`basal compare` 0,0),
zmiana numeryki wymaga porównania z upstream FP32 na zestawie decyzji. Szczegóły zestawów odniesienia:
[BENCHMARKS.md](BENCHMARKS.md).

Runnery `tools/perf/run-*-cloud.sh` opisują wymagane wejścia i opcje w nagłówkach.
Prześlij dane na maszynę przed uruchomieniem. `PROF_LADDER`, `PARITY_INPUT_45B`,
`PARITY_INPUT_MAX` i `HTTP_REQUESTS` wskazują wejścia spoza raportów w Git.
Historyczne wejścia kampanii są w lokalnym archiwum jej autora.
Uruchamiaj pomiary kolejno, bez innych zadań GPU; katalog wyniku musi być nowy.

`run-gemm-tables-cloud.sh` korzysta z zainstalowanego wydania w
`/opt/basal-release/libexec/basal/basal-cuda`; przed sesją sprawdź zgodność jego
kodu wyszukiwania i runtime'u z checkoutem. `GEMM_MODELS="mini 4.5B max"` wybiera
modele mieszczące się w VRAM, a `GEMM_OUT` wskazuje nowy katalog wyniku.
Runner wymaga wszystkich trzech referencji 44 przykładów oraz referencji
900 pytań w `.cache/cloud/refs` wymienionych w nagłówku. Pole `source` wskazuje
raport kampanii z 2026-10-10; przy kolejnej kampanii zmień je przed dodaniem tabel.
Marker `VALIDATED` powstaje osobno dla każdego modelu po zakończeniu wszystkich
kroków. Przed dodaniem tabeli przejrzyj również różnice względem FP32: marker
potwierdza wykonanie porównania, a nie brak różnic numerycznych wobec FP32.

`run-multigpu-cloud.py` wymaga maszyny z dwiema identycznymi kartami NVIDIA,
binarek baseline (bez `--device`) i kandydata, `tools/bench/long_states.jsonl`,
tabel GEMM z `crates/basal-cli/gemm-tables` dla tej karty (wyszukiwania GEMM nie
uruchamia), `crates/basal-cli/src/config.rs` (przypięte rewizje modeli), katalogu
`--inputs` z `bench.jsonl` i `systemone.jsonl` oraz wag `--models` (domyślnie
`4.5B mini`) w `/data/hf`. Uruchamiaj go w `/work`:
`python3 tools/perf/run-multigpu-cloud.py --base /opt/basal-dev/basal-cuda-base --candidate /opt/basal-dev/basal-cuda --inputs KATALOG --out out/multigpu`.
Wszystkim procesom ustawia `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Każde sprawdzenie
zapisuje jako PASS, FAIL lub GAP w `checks.txt`; kod wyjścia 1 oznacza co najmniej
jedno FAIL. Czasy, telemetria `nvidia-smi` osobno dla każdej karty i przedziały
obliczeń z nagłówków odpowiedzi są tylko obserwacją, nie dowodem przyspieszenia.

`run-isolation-cloud.py` korzysta z funkcji `run-multigpu-cloud.py` (oba pliki
z `tools/perf`) i ma te same wymagania co do kart, tabel GEMM, `config.rs` i wag
(mini i 4.5B), a dodatkowo `reports/reference-basal-1.5-mini-fp32` i
`tools/bench/long_states.jsonl`. Uruchamiaj go w `/work`, sumy SHA-256 binarek
bierz z raportu rozmieszczenia:
`python3 tools/perf/run-isolation-cloud.py --base /opt/basal-dev/basal-cuda-base --candidate /opt/basal-dev/basal-cuda --expect-base-sha256 SUMA --expect-candidate-sha256 SUMA --out out/isolation`.
`--plan-only` wypisuje fazy i rozmiary harmonogramów bez GPU. Kampania
`isolation` porównuje mini na GPU 1 bez drugiego modelu (A) z mini na GPU 1 i 4.5B
na GPU 0 w tym samym serwerze (B). 4.5B obciąża osobny proces klienta: 16
zamkniętych pętli długich dokumentów nominalnie 4k i 8k według generatora;
faktyczne tokeny wejścia zależą od modelu. Startuje po rozgrzewce i przed
pierwszym mierzonym żądaniem mini, a kończy się po opróżnieniu ostatniej fazy
mini. Przypadek A czeka przed pierwszą fazą tyle samo (`--bg-lead`).
Kampania `overhead` uruchamia mini pod `CUDA_VISIBLE_DEVICES=1` z tym samym
YAML bez `device` w binarce bazowej (A) i kandydacie (B). Każdy przypadek to
osobny serwer, przypadki idą kolejno w porządku A B B A (`--repeats`).

Ruch mini jest identyczny we wszystkich przypadkach. Zestaw to żądania System
One z odpowiedzią 200 w pierwszej rozgrzewce; pozostałe są wykluczone i wypisane
z kodem. Fazy to zamknięta pętla 16 klientów (`--closed-requests`) oraz otwarty
napływ `--rates` (domyślnie 50 i 100 żądań/s przez `--open-seconds`). Napływ
korzysta ze stałych czasów Poissona; ich SHA-256 jest w `summary.json`.
Przepustowość zamkniętej pętli jest skutkiem szybkości serwera, a nie
kontrolowanym napływem. W otwartym napływie opóźnienie liczy się od planowanego
czasu wysłania, więc obejmuje spóźnienie generatora i kolejkę klienta
(`--max-outstanding`). Zapisywane są też zaległe i trwające żądania. Żadne
zaplanowane żądanie nie jest pomijane; niewysłane, anulowane albo odebrane
dopiero po `--drain-timeout` oznacza FAIL. Spóźnione odpowiedzi zostają w
zapisie, ale faza nie wchodzi do median. Pusty harmonogram (za mała częstość
albo `--open-seconds`) jest odrzucany przed startem. `max_inflight` serwera
(domyślnie 4096) jest wyższy od sumy limitów klientów, a każde 529 oznacza FAIL.
Pierwszy błąd transportu kończy fazę i przypadek: klienci kończą tylko żądania w
toku (najwyżej `--http-timeout`), więc zawieszony serwer kosztuje minuty, a nie
godziny. Klient tła po błędzie transportu przestaje wysyłać.
Każda odpowiedź mini i 4.5B, łącznie z rozgrzewkami, musi mieć status i treść
pierwszej rozgrzewki swojego modelu. Porównanie pomija tylko `usage.latency_ms`.
Faza jest ważną obserwacją tylko wtedy, gdy `/metrics` przed nią i po niej
zwraca 200 z licznikiem żądań, każda odpowiedź 200 ma nagłówki czasu, a każda
oczekiwana karta ma dość pełnych próbek telemetrii (wykorzystanie, pamięć, moc,
zegar SM i temperatura; co najmniej `--telemetry-min-share` interwałów 200 ms,
minimum pięć, bez przerwy dłuższej niż 2 s). Brak tych danych to GAP, osobny od
błędów inferencji. Faza B świadczy o zajętym sąsiedzie tylko przy średnio co
najmniej 90% klientów tła w toku, przerwach najwyżej 1% okna fazy i średnim
`utilization.gpu` GPU 0 nie niższym niż `--busy-util`. Do median A/B wchodzą
tylko ważne fazy. Pokrycie wymaga ważnej fazy we wszystkich 2 × `--repeats`
przypadkach każdego wariantu; mniej to GAP. Kod wyjścia: 0 wszystko PASS,
1 co najmniej jedno FAIL, 3 tylko brakujące dowody (GAP), 2 błąd przygotowania.
Domyślny przebieg obu kampanii (8 przypadków) powinien trwać około 15 minut z
ładowaniem modeli; czas zależy od
przepustowości mini i zaległości przy 100 żądaniach/s.

`summary.json` zawiera odcisk (SHA-256 binarek, runnera, wejść i tabel, rewizje
modeli, sterownik, karty, biblioteki CUDA), parametry, p50/p95/p99 opóźnienia,
nagłówków kolejki i obliczeń, opóźnienia generatora, przepustowość, wyniki
wszystkich żądań, pokrycie metryk i telemetrii, telemetrię każdej karty oraz
mediany A i B. Ścieżki i adresy PCI w opisach kontroli i błędów są zastąpione
przez `<local>` z odsyłaczem do surowego wpisu w `checks.txt`. Zastępowanie
działa według wzorca, więc przed skopiowaniem wyników do `reports/` przejrzyj
plik. Pozostałe pliki są surowe (ścieżki, topologia PCI), więc pobieraj je do
ignorowanego `.cache/` i nie dodawaj do Git. Ograniczenia: klient działa na tym
samym hoście; różnica A/B obejmuje też obsługę HTTP i pamięć drugiego modelu w
procesie oraz obciążenie CPU przez klienta tła. `utilization.gpu` jest zgrubnym
udziałem czasu pracy, a nagłówki opisują partie serwera, nie czas kerneli.
Wynik nie dowodzi pełnej izolacji ani przyspieszenia.

`run-replicas-cloud.py` korzysta z funkcji obu powyższych runnerów (bez ich
zmiany) i ma te same wymagania co do tabel GEMM, `config.rs` i wag (mini i
4.5B), wymaga dokładnie dwóch identycznych kart, a dodatkowo potrzebuje źródeł
Rust (tylko odczyt i SHA-256 do odcisku).
`--base` to zarchiwizowana binarka rozmieszczenia, `--candidate` binarka replik;
obie sumy `--expect-*-sha256` są wymagane. `--plan-only` wypisuje plan bez GPU.
Kampania `correctness` sprawdza `doctor`, odrzucane konfiguracje (przed
ładowaniem wag), eksporty mini i 4.5B na obu kartach względem binarki
rozmieszczenia tej samej karty, odpowiedzi HTTP (także multi, evidence, błędy
400/422 i niepoprawny JSON) jednej instancji na każdej karcie i grupy replik
względem odpowiedzi bazy na GPU 0, liczniki routingu i aktywność obu kart,
presję admission z 529, rozłączenia klientów w kolejce i w trakcie obliczeń
z powrotem rezerwacji do 0, `CUDA_VISIBLE_DEVICES` oraz SIGTERM z żądaniami
w toku. Długie żądania presji, rozłączeń i zamknięcia to `--long-ids`
(domyślnie `long-4000-q1` i `long-4000-q5`). Każde musi dostać 200 w
referencji HTTP obsługiwanego modelu (decyduje jego limit kontekstu), inaczej
kontrola referencji kończy się FAIL; runner niczego nie dobiera sam. Liczba w
identyfikatorze to nominalny rozmiar z `make_long_states.py`, nie liczba
tokenów widziana przez model: mini odpowiada na `long-8000` i `long-16000`
kodem 422 i zostają one w referencji HTTP jako oczekiwane 422. Przy innych
`--models` lub `--long-ids` wskaż żądania, na które pierwszy model odpowiada
200. SIGTERM pada po `--inflight-delay` (domyślnie 0,05 s); gdy wtedy któraś
replika nie ma rezerwacji, ta kontrola to GAP. Mieszanej grupy, niedostępnej
repliki ani nieudanego startu po załadowaniu repliki nie wywołuje: to GAP z
przeglądu statycznego. Kampania
`performance` porównuje mini z jedną (A) i dwiema (B) replikami w porządku
A B B A przy tych samych żądaniach i harmonogramach: zamknięte pętle
`--closed` (domyślnie 1, 16 i 32 klientów) i otwarty napływ `--rates`
(domyślnie 50, 100 i 150 żądań/s). Poza danymi jak w `run-isolation-cloud.py`
zapisuje liczniki każdej repliki, średnią wielkość partii, czas CPU serwera i
klienta na sekundę oraz obciążenie CPU hosta; faza bez kompletnego pomiaru CPU
to GAP i nie wchodzi do median. `--deadline-minutes` (domyślnie 100) zatrzymuje
uruchamianie kolejnych kroków, eksportów i przypadków oraz ogranicza czekanie na
eksport; reszta to GAP. Eksport dłuższy niż `--export-timeout` (podejrzenie
zawieszonego GPU) to FAIL kończący cały przebieg. Przerwany przebieg (sygnał,
także Ctrl-C, nieoczekiwany błąd) ignoruje kolejne sygnały, zapisuje raz do
`requests.jsonl` każdą otrzymaną odpowiedź (także gdy zatrzymanie serwera się
nie powiedzie; odpowiedzi wątków klienta, które nie skończą się w ciągu 10 s po
zatrzymaniu serwera, nie są zapisywane, co oznacza GAP), zapisuje `summary.json` z `complete: false` i nie kończy się
kodem 0. Wynik nie zakłada skalowania 2× ani
izolacji awarii sprzętu.

`run-w8a16-cloud.py` korzysta z funkcji trzech powyższych runnerów i z
`tools/bench/ab.py` (bez ich zmiany). `--plan-only` (bez GPU, chmury i
binarek) sprawdza w pamięci wejście 900 pytań i wypisuje liczbę kroków, żądań
i zgrubny czas ze śledzonych manifestów, także najgorszy przypadek warunkowej
rundy potwierdzenia `perf-none` i sesji rewalidacji; kroków bez śledzonego
czasu nie szacuje, podaje tylko ich limity. `--prepare-900 KATALOG` zapisuje
to wejście (odtworzone z `sequential.jsonl` i plików decyzji, round trip
`public_request` 900/900) poza repozytorium; prawdziwy przebieg buduje je
ponownie i zapisuje sumy SHA-256 w odcisku. `--base` musi być binarką etapu 02
o SHA-256 `289a2eac…` (tekst wersji nie jest tożsamością binarki), a
`--candidate` binarką W8A16; obie sumy `--expect-*-sha256` są wymagane.
Wymaga dwóch identycznych kart, wag mini i 4.5B w przypiętych rewizjach (ich
SHA-256 porównuje z manifestami referencji), referencji FP32 i MLX 8-bit,
plików decyzji, tabel GEMM z `crates/basal-cli/gemm-tables` (podawanych jawnie
każdemu poleceniu f16 i `q8_0`; log z komunikatem wyszukiwania tabeli kończy
polecenie jako FAIL) i źródeł Rust do odcisku. Uruchamiaj w `/work`:
`python3 tools/perf/run-w8a16-cloud.py --base BAZA --expect-base-sha256 SUMA --candidate KANDYDAT --expect-candidate-sha256 SUMA --deadline-minutes MINUTY --out out/w8a16`.
`--deadline-minutes` jest wymagane (bez wartości domyślnej): dobierz je do
najmu z zapasem na pobranie wyników i usunięcie maszyny. Po terminie nie
startuje żaden krok, polecenie ani żądanie HTTP, a trwające są przerywane na
nim; to, co nie zdążyło, jest GAP, nie FAIL. Termin i limity są bezwzględne:
żądanie HTTP (nagłówki i treść razem, także `/health`, `/v1/models`,
`/metrics` i gotowość) jest przerywane, a oczekiwanie na polecenie kończone
najpóźniej w chwili wcześniejszego z nich (z dokładnością do planowania
wątków); zatrzymanie procesu polecenia (SIGTERM, po 60 s SIGKILL) trwa już po
tej chwili. Odpowiedź lub zakończenie zauważone dopiero po niej nie jest
terminowe, także ze statusem 200 (zostaje zapisane jako dowód). Polecenie GPU
(eksport, `decide`, każdy `basal bench` z `ab.py` osobno, `benchmark`, `profile`) albo
żądanie HTTP po własnym limicie czasu to FAIL i koniec przebiegu
(podejrzenie zawieszonej karty, bez dalszej pracy GPU w żadnej kampanii).
Odpowiedzi System One serwera `q8_0` (status i treść) runner porównuje z
natywną referencją `q8_0` tych samych ciał żądań: eksportem wejść pochodnych
w `--out` (ciała HTTP z `model` mini, `bench.jsonl` jako kopia referencji), a
przypadek, który ten eksport odrzuca jako Unsupported, z `basal decide` tego
ciała; referencje i ich eksporty pozostają bez zmian.
Kampanie i reguły interpretacji (sygnał `perf-none` tylko przy B gorszym w
każdej parze, jedna runda potwierdzenia z profilem A/B, wniosek tylko przy
kompletnych parach i wartościach, w `basal bench` dokładnie 39 mierzonych
pytań o indeksach 5..43, bez progów i wykluczeń) opisuje nagłówek pliku.
Wynik ma statusy
PASS, FAIL, GAP oraz SIGNAL (różnica do analizy: utrzymany sygnał `perf-none`,
różnica decyzji f16 lub `q8_0` wobec f16 na 44+34, nowe zmiany decyzji albo
niższa trafność `q8_0` na 900 pytaniach, zmienione eksporty `q8_0` w
rewalidacji; bez znalezionej przyczyny decyzja użytkownika); sygnał samej
rundy podstawowej `perf-none` nie jest SIGNAL. Kod wyjścia 4 oznacza SIGNAL
bez FAIL. Rewalidacja po poprawce kodu: `--campaigns 'correctness revalidate
perf-none perf-q8 profile memory-mini' --steps none --previous-run KATALOG`
(surowy wynik sesji binarki sprzed poprawki, inny niż `--out`). Runner przed
startem sprawdza ten katalog (odcisk w `run.json` i `summary.json`, binarka
etapu 02, znane SHA-256 kandydata inne niż obecnego, te same rewizje modeli,
tabele GEMM i wejścia, ukończony eksport `q8_0` tree na GPU 0 każdego modelu)
i zapisuje SHA-256 kandydata, odcisk źródeł oraz eksportów obu sesji. Surowe
wyniki zawierają ścieżki: pobieraj je do `.cache/cloud`.

Do `reports/` zapisuj krótkie wyniki i zakres weryfikacji. Pełne eksporty, logi
i profile pobieraj do ignorowanego `.cache/` i archiwizuj poza Gitem.
Czasów spod profilera nie używaj jako pomiaru przyspieszenia inferencji.
