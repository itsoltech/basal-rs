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
