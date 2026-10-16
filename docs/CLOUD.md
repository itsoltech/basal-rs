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

Do `reports/` zapisuj krótkie wyniki i zakres weryfikacji. Pełne eksporty, logi
i profile pobieraj do ignorowanego `.cache/` i archiwizuj poza Gitem.
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
