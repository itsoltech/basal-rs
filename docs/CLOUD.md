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

Jeśli użytkownik określi wspólny budżet na bieżące i przyszłe sesje, limit obejmuje ich sumę, również nieudane
próby uruchomienia. Nie odnawia się przy nowej maszynie ani checkoutcie. Przed `up` sprawdź prywatny ledger
`~/.local/state/basal-rs/cloud-budget-ledger.json` oraz
`$(git rev-parse --git-common-dir)/basal-cloud-budget-ledger.json`, jeśli istnieją. Ten drugi może być dowiązaniem
do pierwszego; stan poza checkoutem obejmuje kolejne worktree i klony w tym środowisku. Rozbieżnych ledgerów
nie traktuj jako osobnych limitów. Kwoty i dane dostawcy pozostają poza Gitem. Uwzględniaj koszt
zakończonych prób oraz rezerwacje aktywnych maszyn; po usunięciu uzgodnij je z rozliczeniem dostawcy.
Niepotwierdzone koszty oznacz jako szacunek z zapasem. Wybór karty nadal podlega zasadom danej sesji.

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
python3 tools/cloud/basal-cloud.py pull out/wg .cache/cloud/wg
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
| `tools/perf/run-w4a16-cloud.py` | opcjonalne wagi `q4_k` (W4A16) na dwóch identycznych kartach: tory `none` i `q8_0` bajtowo wobec stabilnej binarki W8, spójność `q4_k` między trybami i kartami, diagnostyka ładowania wobec P0, odrzucane konfiguracje, HTTP z replikami, 44+34 i 900 pytań wobec FP32, profil, pamięć puli, rodziny wydajności `none` i `q8_0` (A B C C B A) |
| `tools/perf/run-layer-split-cloud.py` | jeden model podzielony warstwami na dwie identyczne karty (`layer_split`): sesja L1 (tor jednego GPU bajtowo wobec bazy, podział bajtowo wobec jednego GPU, odrzucane konfiguracje, HTTP i `device="0,1"`, pojemność pod sztucznym limitem, profile `stage_transfer`), L2 (regresja `none` A/B i opisowy podział `none`), L3 (regresja `q8_0` i `q4_k` A/B i opisowy podział) |
| `tools/perf/cuda-reserve.py` | pomocnik pojemności uruchamiany przez runner podziału: zajmuje rzeczywistą pamięć urządzenia do zadanego poziomu wolnej pamięci i próbkuje wolną pamięć każdej karty |
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

Wyniki JSON wcześniejszych pomiarów multi-GPU (W8, W4A16, podział warstw)
nie są w Git. Archiwum z tymi plikami, jego SHA-256 i sposób pobrania opisuje
[MEASUREMENT_ARTIFACTS.md](MEASUREMENT_ARTIFACTS.md). Runnery potrzebują
czterech plików z tego archiwum. `run-w4a16-cloud.py` czyta fakty P0
(`reports/w4a16-p0-2026-10-12/facts.json`, SHA-256 przypięte w runnerze) oraz
trzy raporty W8 dla `--plan-only` (`reports/w8a16-2026-10-11/commands.json`,
`reports/multigpu-w8-none-2026-10-11/facts.json`,
`reports/multigpu-w8-original-2026-10-11/perf-q8-original.json`).
`run-layer-split-cloud.py` czyta fakty P0. Oba runnery szukają tych plików pod
tymi samymi ścieżkami `reports/...` w katalogu `--measurement-artifacts
KATALOG`, czyli w katalogu, do którego rozpakowano archiwum. Domyślny katalog
to `.cache/measurement-artifacts/multigpu-2026-10-14`; ścieżka względna
liczy się od katalogu głównego repozytorium (na maszynie `/work`). Runnery
niczego nie pobierają. Brak pliku daje ten sam wynik co dawniej brak pliku w
Git: bez faktów P0 runner W4 zapisuje GAP, a runner podziału warstw odmawia
uruchomienia sesji; bez raportu W8 krok planu nie ma źródła. `--plan-only`
runnera W4 wypisuje, które z czterech plików znalazł, i ich SHA-256. Zapisy
przebiegu podają ścieżkę logiczną `reports/...` i SHA-256, bez lokalnego
katalogu. Przed sesją prześlij rozpakowany katalog na maszynę. Przesyłaj
zwykły katalog, nie dowiązanie symboliczne, bo `rsync` skopiowałby samo
dowiązanie:

```sh
python3 tools/cloud/basal-cloud.py sync .cache/measurement-artifacts/multigpu-2026-10-14
```

`run-w4a16-cloud.py` (W4A16, `quantization: q4_k`) korzysta bez zmian z
`run-w8a16-cloud.py` (a przez niego z trzech powyższych runnerów) i z
`tools/bench/ab.py`; przed startem sprawdza ich SHA-256. Sesje to osobne
wywołania: S1 `--campaigns 'correctness quality profile memory-mini'`
(domyślne), po audycie surowych danych S2 `--campaigns perf-none-family
--s1-run KATALOG_S1` i S3 `--campaigns perf-q8-family --s1-run KATALOG_S1`.
Rodzina nie startuje, gdy przebieg S1 ma FAIL klasy innej niż
`quality-criterion` albo inne binarki lub źródła. Przed pracą GPU porównuje
też zapisane w S1 SHA-256 runnera i helperów, zasoby wbudowane, model rodziny
(nazwa, rewizja, `config.json`, wagi), jego tabelę GEMM, referencję FP32,
decyzje i wejście 900 oraz pliki sequential, lock, long, `config.rs` i fakty
P0: różnica to FAIL i zatrzymanie, brak dowodu w którymkolwiek przebiegu (np.
`--no-weight-hash`) to GAP tego powiązania. Rewalidacja tylko po rzeczywistej
poprawce kodu: `--campaigns revalidate --previous-run KATALOG
--expect-previous-candidate-sha256 SUMA`, potem dotknięte rodziny z `--s1-run`
tego przebiegu. Rewalidacja raz mierzy natywne HTTP i repliki `q4_k` końcowego
kandydata, niezależnie od równości eksportów. `--base` to stabilna binarka W8
ze źródła `750c146`, `--candidate` binarka W4A16. Wymagane są obie sumy
`--expect-*-sha256`, `--base-label` i `--candidate-label` (role i źródła
zapisane jako deklaracja, nie dowód) oraz `--deadline-minutes`; opcjonalne są
rekordy budowy `--base-provenance` i `--candidate-provenance` (JSON z
`binary_sha256` tej binarki; rekord bez niej albo z inną sumą jest odrzucany)
oraz `--lease-end-unix`. Obowiązuje wcześniejszy z tych terminów: koordynator
liczy go z czasu utworzenia maszyny, limitu jej życia (≤ 2 h) i rezerwy na
pobranie wyników i usunięcie. Runner nie przedłuża sesji, nie wynajmuje i nie
usuwa maszyn. Kampanie, statusy, klasy FAIL (`correctness`,
`quality-criterion`, `timeout`), kody wyjścia i limity opisuje nagłówek pliku
(`--help`).

`--plan-only` (bez GPU, chmury, sieci i binarek) buduje w pamięci wejście 900
pytań i dla S1, S2, S3 oraz jednej warunkowej sesji rewalidacji wypisuje
liczby poleceń, serwerów, ładowań modelu i żądań HTTP. Sesję rewalidacji
podaje tak, jak definiuje ją plan: jedna maszyna z wywołaniem `revalidate`
(R1) i dotkniętymi rodzinami (R2 `none`, R3 `q8_0`), w wariantach R1+R2, R1+R3
i R1+R2+R3. Osobne maszyny R1, R2 i R3 są tylko proponowanym podziałem: przed
zakupem wymaga on rewizji planu i prognozy przez koordynatora. Szacunek czasu
każdego kroku ma źródło: (a) zarchiwizowany raport W8 (`--measurement-artifacts`)
z SHA-256 i polem, (b) prywatny surowy rekord W8 (`--w8-raw KATALOG`,
ścieżka względna i SHA-256) albo (c) jawne założenie. Czas procesu `q4_k` to czas tego samego procesu
`q8_0` z W8 plus bezwzględny czas toru hosta Q4_K z P0 na każde ładowanie:
nieskalibrowany zapas, nie zmierzony narzut ani granica. Runner podaje też,
czy sesja z założonym przygotowaniem maszyny i rezerwą (`--bootstrap-minutes`,
`--collect-delete-minutes`) mieści się w 120 min; jeśli nie, przed zakupem
potrzebny jest zaplanowany podział, bez obcinania kroków, par i metryk.
Szacunek nie jest pomiarem ani decyzją o budżecie. `--prepare-900 KATALOG`
zapisuje wejście 900 pytań poza repozytorium. Sesję S1 wykonano
([raport](../reports/w4a16-s1-2026-10-12/README.md)): kryterium FP32 ma FAIL,
a użytkownik dopuścił `q4_k` tylko jako opcjonalny tryb eksperymentalny
(domyślne i zalecane `none`). Ta decyzja jakościowa nie zamykała rodzin
wydajności; wykonano je osobno. S2
([raport](../reports/w4a16-s2-2026-10-12/README.md)) ma wniosek (a) dla
`none` i opisowy kompromis `q4_k` wobec f16. S3
([raport](../reports/w4a16-s3-2026-10-12/README.md)) ma utrzymany sygnał
mixed c16 `q8_0` (−0,1076% do −0,3732%) z nieustaloną przyczyną (wniosek
(c)) i opisowy kompromis `q4_k` wobec `q8_0`. Ten zaobserwowany spadek
zaakceptował użytkownik w osobnej decyzji wydajnościowej, opisanej w
raporcie S3; sygnał i statusy runnera pozostają, a decyzja nie tworzy progu.
Poprawki kodu nie było, więc nie wykonano rewalidacji. Kolejne sesje GPU
wymagają zgody użytkownika na kartę i budżet według
[Zgoda i budżet](#zgoda-i-budżet).

`run-layer-split-cloud.py` (etap 05, `layer_split`) wymaga maszyny z dwiema
identycznymi kartami NVIDIA. Importuje bez zmian `run-w4a16-cloud.py` (a przez
niego runnery W8, replik, izolacji i rozmieszczenia) oraz `tools/bench/ab.py`
i przed załadowaniem sprawdza ich SHA-256; `cuda-reserve.py` uruchamia jako
osobny proces tylko w przypiętej wersji. Wymagania maszyny jak dla runnera
W4 (drzewo w `/work`, wagi w `HF_HOME`, biblioteki CUDA w `BASAL_HOME`,
wbudowane tabele GEMM karty, fakty P0 w `--measurement-artifacts`).
`--base` to przebudowana binarka `d42a400`,
`--candidate` binarka z podziałem, obie z jednego przepisu; wymagane są obie
sumy `--expect-*-sha256`, `--base-label`, `--candidate-label` i
`--deadline-minutes`. Jedno wywołanie to jedna sesja, a koordynator audytuje
surowe dane między sesjami:

```sh
python3 tools/perf/run-layer-split-cloud.py --plan-only        # liczby kroków, źródła, bez GPU
python3 tools/perf/run-layer-split-cloud.py --campaign l1 --base BAZA --expect-base-sha256 SUMA \
  --base-label OPIS --candidate KANDYDAT --expect-candidate-sha256 SUMA --candidate-label OPIS \
  --deadline-minutes MINUTY --out out/layer-split-l1
python3 tools/perf/run-layer-split-cloud.py --campaign l2 --l1-run out/layer-split-l1 ... --out out/layer-split-l2
```

- L1 (jednostki C1–C6): eksporty toru jednego GPU bajtowo wobec bazy
  (`none`, `q8_0`, `q4_k`, także na GPU 1 i dla 900 pytań); eksporty z
  podziałem bajtowo wobec bazy na jednym GPU, z `basal compare` i kontraktem
  manifestu (układy `0:30,1:30`, `0:1,1:59`, `0:59,1:1`, `1:30,0:30`,
  `CUDA_VISIBLE_DEVICES=1,0`, cache, mini); niepoprawne konfiguracje przez
  `serve`, `export`, `benchmark`, `decide`, `profile` i `doctor`; HTTP z
  podziałem (odpowiedzi wobec referencji, `device="0,1"`, SIGTERM z żądaniami
  w toku, równoległy ruch z mini na drugiej karcie); pojemność pod
  sztucznym limitem (C5, metoda w [Pomiarach](BENCHMARKS.md#podział-warstw));
  profile `stage_transfer` (diagnostyka).
- L2: regresja toru domyślnego `none` (A baza, B kandydat na jednym GPU,
  C kandydat z `0:30,1:30`; `basal benchmark` w kolejności A B C C B A i
  `ab.py` A/B oraz B/C). Pola gorsze dla B w każdej parze uruchamiają jedną
  rundę potwierdzenia; gorsze w obu rundach to SIGNAL, bez progu. Pary B/C
  są opisowe.
- L3: regresja `q8_0` i `q4_k` na jednym GPU (A/B, ta sama reguła) oraz
  opisowe porównanie z podziałem w tych trybach.

L2 i L3 wymagają `--l1-run` z tymi samymi binarkami, źródłami i zasobami,
bez FAIL i ze wszystkimi jednostkami C1–C6; GAP pojemności nie blokuje
kolejnych sesji. Każda sesja ma własny twardy limit czasu od żądania
maszyny, a `--deadline-minutes` musi zostawić rezerwę na pobranie wyników.
Każde polecenie, żądanie, zamknięcie serwera i akcja pomocnika ma własny
limit; przekroczenie przez polecenie GPU to FAIL klasy `timeout` i koniec
pracy GPU, a to, czego nie uruchomiono, jest GAP. Statusy PASS, FAIL (klasy
`correctness`, `capacity`, `timeout`), GAP i SIGNAL oraz kody wyjścia (0
same PASS, 1 FAIL, 4 SIGNAL bez FAIL, 3 tylko GAP, 2 błąd preflightu,
128 + sygnał po przerwaniu) opisuje nagłówek pliku (`--help`). Runner nie
wynajmuje, nie przedłuża i nie usuwa maszyn; surowe wyniki zawierają
ścieżki, więc pobieraj je do `.cache/cloud`.

`cuda-reserve.py` używa tylko biblioteki standardowej Pythona i sterownika
CUDA (`libcuda.so.1`). `query` podaje wolną i całkowitą pamięć każdej karty,
`monitor` próbkuje wolną pamięć (domyślnie co 50 ms) do pliku CSV, a
`reserve --leave KARTA=MIB` najpierw alokuje rzeczywistą pamięć, aż sterownik
pokaże zadaną wolną pamięć (tolerancja 64 MiB), potem próbkuje jak
`monitor`. Kończy się sygnałem, po `--max-seconds` albo po zmianie procesu
nadrzędnego i wtedy zwalnia zajętą pamięć (po błędzie sterownika w miarę
możliwości; kody wyjścia w nagłówku). Pliki zawierają tylko ordinale, nazwy
kart i liczby bajtów.

Dla bieżących źródeł kandydata (`831e68e0…`) wykonano wszystkie trzy
sesje, kompletnie: L1
([raport](../reports/multigpu-layer-l1-revalidation-2026-10-13/README.md):
599 PASS, 0 FAIL, 42 GAP, 0 SIGNAL, kod wyjścia 3), L2
([raport](../reports/multigpu-layer-l2-revalidation-2026-10-14/README.md):
107 PASS, 0 FAIL, 1 GAP, 1 SIGNAL, kod 4) i L3
([raport](../reports/multigpu-layer-l3-revalidation-2026-10-14/README.md):
172 PASS, 0 FAIL, 2 GAP, 1 SIGNAL, kod 4). Utrzymane sygnały L2 i L3 nie
mają wskazanej przyczyny w kodzie; rozstrzygnięcia konkretnych wyników są
w plikach `DECISION.md` obu raportów. Wyniki zestawia
[raport zbiorczy](../reports/multigpu-layer-revalidation-2026-10-14/README.md):
osobne sesje miały tę samą konfigurację dwóch kart, ale ich fizyczna
tożsamość nie jest ustalona. Wcześniejsze sesje poprzedniego kandydata
([zestawienie](../reports/multigpu-layer-2026-10-13/README.md) z
[poprawką zakresu](../reports/multigpu-layer-2026-10-13/SCOPE.md))
pozostają historią. Każda sesja GPU wymaga zgody użytkownika na kartę i
budżet według [Zgoda i budżet](#zgoda-i-budżet).

Do `reports/` zapisuj opis wyników i zakres weryfikacji w Markdown. Pełne
eksporty, logi, profile i pliki wyników JSON pobieraj do ignorowanego `.cache/`
i archiwizuj poza Gitem ([MEASUREMENT_ARTIFACTS.md](MEASUREMENT_ARTIFACTS.md)).
Czasów spod profilera nie używaj jako pomiaru przyspieszenia inferencji.

## Porównanie wariantów attention Basal 2.0

`tools/perf/basal2-optimize.py` porównuje zachowaną binarkę bazową i kandydatów na
istniejącej maszynie. Nie wynajmuje GPU ani nie wybiera ustawienia domyślnego.
`BASAL_ATT=tc64`, `tc-sw128`, `tc64-sw128` wybierają eksperymentalne warianty
z mniejszym blokiem lub układem XOR w pamięci współdzielonej. Wymagają osobnego
potwierdzenia zgodności bitowej i wydajności. Te warianty nie są wybierane automatycznie;
profil `auto` ma osobną, ograniczoną do zmierzonego modelu/GPU politykę
opisaną w [raporcie profili](../reports/precision-performance-ada-2026-10-16/README.md).

Runner przyjmuje `--measurement-artifacts` z przygotowanymi wejściami:

- `manifest.json`, `requests.jsonl`, `prompt-audit.json` z
  [drabiny latency](../reports/latency-basal2-ada-2026-10-15/README.md);
- `short.jsonl`, `mixed-mini.jsonl`, `mixed.jsonl` i trzy katalogi
  `basal-2.0-{mini,4.5B,max}-fp32-reference/` z
  [kampanii pięciu GPU](../reports/perf-basal2-fivegpu-2026-10-14/README.md);
- `gemm-tables/<model>-f16.json`: zamrożone tabele wspólne dla wszystkich wariantów,
  zgodne z kartą i bibliotekami CUDA;
- `provenance.json`: obiekt `files`, mapujący każdą względną ścieżkę wejścia na SHA-256.
  SHA binarki bazowej musi odpowiadać `binary_sha256` manifestu drabiny.

Dane pobierz i zweryfikuj według [procedury artefaktów](MEASUREMENT_ARTIFACTS.md).
Przygotuj nowy katalog wejściowy i zachowaj źródłowe manifesty; nie zmieniaj
zakończonego archiwum. Ustaw środowisko CUDA i ścieżki modeli jak w pozostałych
runnerach. Przykład dla przesłanych binarek i wejść:

```bash
python3 tools/perf/basal2-optimize.py \
  --measurement-artifacts inputs/performance \
  --binary /opt/basal-dev/basal-cuda \
  --baseline /opt/basal-dev/basal-cuda-base \
  --models-root /data/models --phase screen \
  --out out/performance-screen --deadline "$BASAL_CAMPAIGN_DEADLINE"
```

`screen` zbiera pojedynczy przebieg kandydatów: drabinę, domyślnie ruch 1/8/32 klientów
oraz cały korpus mieszany przy największej wybranej liczbie klientów.
`--concurrency 1 2 4 8 16 32` pozwala zwiększać obciążenie stopniowo.
Na współdzielonym serwerze użyj `--no-clock-lock`, aby runner nie zmieniał
zegarów GPU. Opcjonalne `--start-temperature 50` czeka na ostygnięcie GPU
przed kształtami drabiny, fazami obciążenia i eksportami; to oczekiwanie
pozostaje poza czasami HTTP. Obciążenie wewnątrz każdej fazy jest ciągłe.
Limit mocy, temperaturę, wolne miejsce, obce procesy GPU i stan
usług musi kontrolować osobny nadzorca; runner sam nie zapewnia tych zabezpieczeń. To **SIGNAL**, nie dowód przyspieszenia.
Po przejrzeniu wyników `confirm --variants baseline <wariant>` wykonuje ABBAABBA,
z r1 obu wariantów poza końcową statystyką. Zachowuje wszystkie odpowiedzi;
jakakolwiek zmiana pełnego obiektu `answers`, także bitów liczb, przerywa pomiar.
Korpus mieszany mierzy przy 8 i 32 klientach. `--short-samples 1056` wydłuża krótkie fazy potwierdzające, ograniczając wpływ
rozruchu klientów na QPS; liczba musi być wielokrotnością rozmiaru korpusu.
Nie wolno utożsamiać wzrostu
przepustowości z poprawą p95 krótkich zapytań; dane zawierają klasy żądań.

Faza `quality` porównuje eksporty single/tree/budget, bez cache prefiksów i z cache
stanów. Zachowuje porównania bitowe do bazowego tree i porównania do FP32 upstream.
Nieoczekiwany błąd eksportu lub różnica bitowa blokuje potwierdzenie wariantu.
Oczekiwane odrzucenia muszą odpowiadać dokładnie identyfikatorom z błędami
w zamrożonej referencji upstream; ich treść musi być bitowo zgodna z bazowym tree.
Sama zgodność
odpowiedzi HTTP nie zastępuje tej fazy.

`--variants baseline gemm-retuned` pozwala oddzielnie porównać nowe tabele
w `retuned-gemm-tables/<model>-f16.json`. `--retuned-attention` wybiera attention
tego wariantu. Tabele wygeneruj istniejącym `basal gemm-search --invariant`,
uwzględniając M pojedynczego promptu po odjęciu prefiksu oraz większe partie.
Runtime najpierw zaokrągla M: do wielokrotności 16 do 512 włącznie, wyżej do
wielokrotności 128. Dopiero dla tej klasy wybiera wpis tabeli. Surowe M=4051
korzysta więc z klasy 4096; nie należy zakładać, że wpis o kluczu 4051 zostanie
użyty dla tego wejścia. Sprawdź także długości sąsiednie i cały zakres zmienionej klasy.
Zamroź nowy komplet wejść i sum kontrolnych. Generowanie tabel samo w sobie nie
potwierdza przewagi ani identyczności z poprzednią tabelą.

Po zebraniu wyników agregację HTTP można odtworzyć bez GPU:

```bash
python3 tools/perf/summarize-basal2-optimize.py \
  --measurement-artifacts out/performance-confirm --out summary-new.json
```

Agregator sprawdza pokrycie, liczbę próbek i powtórzeń, błędy i odpowiedzi. Niekompletne
wyniki oznacza jako GAP, różnice lub błędy jako FAIL. p95 pozostaje orientacyjny.
Wyboru ustawień dokonuj na podstawie potwierdzonych pomiarów latency, throughput,
pamięci i opóźnień klas w ruchu mieszanym. Przed publikacją przeprowadź przegląd
prywatności surowych danych. Wynajętą maszynę usuń zgodnie z procedurą sesji;
na własnym serwerze zatrzymaj wyłącznie zasoby pomiaru i odtwórz poprzednie ustawienia GPU.

Istniejący wariant `--variants baseline tc-pv1` służy oddzielnej ocenie numerycznej:
zaokrągla P do f16 przed P·V. W `method.json` jest jawnie oznaczony jako numeryczny.
Odpowiedzi muszą pozostawać bitowo identyczne wewnątrz wariantu przy zmianie partii
i powtórzeniach; różnice wobec baseline są zachowane w obserwacjach i agregacji.
Faza `quality` wymaga zgodności trybów z własnym tree tego wariantu i zapisuje
pełne porównania do bazowego tree oraz FP32 upstream. PASS wykonania lub zgodności
wewnątrz wariantu nie oznacza akceptacji jakości ani zgody na zmianę ustawień domyślnych.

Dla korpusów z wieloma dużymi/rozszerzonymi żądaniami `--max-large-inflight 128`
ustawia jawną pojemność puli w izolowanym serwerze pomiarowym. Wymaga binarki
z tą opcją; użyj tej samej wersji i limitu po obu stronach porównania, zamrażając
nowy manifest z SHA binarki bazowej. Domyślny limit serwera wynosi 4.
Przebiegu z 529 nie zastępuj skutecznym: zachowaj FAIL i opisz osobno nowe ustawienie.

`tc-qk1` zaokrągla Q/K do hi f16; `tc-f16` łączy to z zaokrągleniem P. Także
te warianty, w tym połączenia `gemm-retuned --retuned-attention tc-f16`, są jawnie
numeryczne i wymagają osobnej oceny FP32.

Dla drabiny nasycenia runner dopuszcza do 512 klientów. Przykład ustawień:
`--concurrency 1 8 32 64 128 --mixed-clients 8 32 64 128 --max-large-inflight 256`.
Pula przyjęcia ma wtedy zapas ponad liczbę klientów; limit ogólny serwera wynosi
1024. Te wartości dotyczą izolowanego korpusu i wymagają osobnej oceny pamięci
przed użyciem w usłudze. Dalsza rampa 256/512 wymaga odpowiedniego zapasu
liczby próbek, limitów przyjęcia, deskryptorów, procesów/wątków i RAM;
nie utożsamiaj największej liczby klientów z najlepszym czasem odpowiedzi.
`--skip-ladder` jawnie pomija drabinę pojedynczych pytań
w przebiegu poświęconym przepustowości. Czekanie na temperaturę ma osobny
`--cooling-timeout` (domyślnie 600 s), aby brak schłodzenia nie zużył całej sesji.

`--variants baseline upstream-fast --reference-python PLIK --upstream KATALOG`
porównuje natywny serwer z niezmienionym checkoutem upstream o commicie zapisanym
w manifeście. Wspólny korpus, modele, liczba klientów i warunki sprzętowe muszą
pozostać te same. Upstream używa `--mode fast`, BF16, bez cache stanów i z odmową
przepełnienia kontekstu; zachowuje domyślny scheduler i batch-window. Jego kolejkę
ogranicza tu liczba klientów closed loop. Natywny limit przyjęcia nie jest opcją
upstream. Różnice odpowiedzi upstream względem baseline oraz między partiami są
zachowane w danych; nie wymagamy od silnika referencyjnego naszej bitowej
niezależności od partii. Odpowiedzi nadal muszą być poprawne, skończone i bez
obcięcia promptu. Nie wolno przenieść tego wyjątku na kandydatów basal-rs.
`method.json` zapisuje wyjątek jawnie, razem z SHA runnera i commitem referencji.

Przy krótkich korpusach raportuj także liczbę **różnych promptów**, nie tylko
liczbę żądań. Wyłączony cache stanów nie wyłącza współdzielenia identycznych
promptów w partii. Duży wzrost QPS przy powtarzanych wejściach nie opisuje
przepustowości różnych dokumentów. Do badania wpływu różnorodności użyj osobnych,
zamrożonych korpusów o takim samym rozkładzie długości tokenowych; zachowaj ich
odpowiedzi i provenance. Syntetyczny identyfikator może zmienić odpowiedź, więc
porównanie bitowe dotyczy tego samego wejścia wewnątrz wariantu.

Uruchomienie `upstream-fast` ma osobny limit gotowości 1200 s, konfigurowany przez
`--reference-startup-timeout` (60–3600 s); ładowanie i kompilacja
pozostają poza czasem żądania. Limit pojedynczego żądania i deadline całej kampanii
nadal obowiązują. Zapisuj pełne niepowodzenie startu; nie zastępuj go nieopisanym
powtórzeniem ani innym trybem upstream.

`--max-batch-tokens N` ustawia jawny budżet tokenów batcha natywnego serwera
(domyślnie 8192). Zachowaj stałą liczbę klientów i korpus; raportuj razem QPS,
opóźnienia, pamięć oraz wpływ na krótkie żądania w ruchu mieszanym. Większy batch
nie daje automatycznie lepszej przepustowości ani czasu odpowiedzi. To osobne
ustawienie od limitów przyjęcia żądań; nie zmienia budżetu batcha upstream.

Opcjonalny `http-reference-answers.json` w zamrożonym katalogu wejść zapisuje
wcześniejsze pełne `answers` natywnego serwera, po jednym rekordzie
`{model,id,variant:"baseline-compatible",answers}`. Musi należeć do `provenance.json`.
Runner wymaga zgodności bitowej tych odpowiedzi dla wariantów strukturalnych,
także po zmianie budżetu batcha. Warianty numeryczne nadal mają własny punkt
odniesienia i osobną ocenę FP32. SHA referencji jest w metodzie pomiaru;
przy odtwarzaniu audytu podaj także
`--http-reference KATALOG_WEJSC/http-reference-answers.json` agregatorowi.

`--variants baseline batch-budget` izoluje poprawkę przekazywania budżetu serwera
do `Engine::forward_max_tokens`: podaj starą binarkę jako `--baseline`, a binarkę
z poprawką jako `--binary`, z tym samym `--max-batch-tokens` po obu stronach.
Wcześniejszy silnik zachowywał niezależne 8192 dla łączenia wielu wierszy,
np. różnych języków. Pojedynczy większy wiersz już wcześniej mógł działać osobno.
Wariant pozostaje strukturalny i musi spełniać tę samą kontrolę bitową.
`method.json` podaje SHA obu binarek oraz rolę tego wariantu.

Profile `accurate`, `auto` i `balanced` przekazują jawnie `--inference-mode`
natywnemu serwerowi. Numeryczne odpowiedzi `balanced` porównuj do FP32;
nie wymagaj ich zgodności bitowej z `accurate`. Każdy wariant natywny nadal musi
zachować własne bity przy zmianie partii i powtórzeniach. Zgodność strukturalnej
specjalizacji kernela sprawdzaj względem jej numerycznego rodzica.

Końcowy pomiar trzech wariantów używa
`--phase confirm --variants accurate balanced upstream-fast --baseline-control`.
Kolejność ABCCBAABCCBA daje ABBAABBA dla każdej pary; pierwsze powtórzenie każdego
wariantu pozostaje diagnostyczne. Osobny zamrożony `--baseline` zbiera poza
pomiarem odpowiedzi kontrolne wszystkich wejść, które `accurate` musi odtworzyć
bitowo. Trzy warianty nie mogą zastąpić tego niezależnego punktu odniesienia.
Agregator wyszukuje wymaganą referencję HTTP według SHA w sąsiednich wejściach
artefaktu, jeśli nie podano jej jawnie; brak referencji kończy audyt błędem.
Pomiary czasu pojedynczego pytania i QPS wymagają oddzielnej interpretacji;
[raport](../reports/precision-performance-ada-2026-10-16/README.md) zachowuje
również gorsze p95 i mediany w ruchu mieszanym.
