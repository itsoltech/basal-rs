# Instalacja i aktualizacja

Paczki [wydań](https://github.com/itsoltech/basal-rs/releases) powstają z
tagów `v*` ([release.yml](../.github/workflows/release.yml)): macOS na Apple
Silicon (Metal) i Linux x86_64 z kartą NVIDIA (CUDA). Bez instalacji na
hoście: obraz Docker ([README](../README.md#uruchomienie-w-dockerze)).

## macOS (Apple Silicon)

```sh
brew install itsoltech/tap/basal-rs
basal doctor                    # sprawdzenie maszyny
basal serve                     # http://127.0.0.1:8000, basal-1.5-4.5B pobierany przy pierwszym starcie
basal serve --model mini        # inny model (patrz Modele)
brew services start basal-rs    # albo jako usługa w tle (log: $(brew --prefix)/var/log/basal.log)
```

Aktualizacja: `brew upgrade basal-rs`. Usługa Homebrew startuje w
`$(brew --prefix)/etc/basal` i czyta stamtąd `basal-serve.yml`, jeśli
istnieje.

Bez Homebrew: ten sam skrypt co na Linuksie
(`curl -fsSL https://raw.githubusercontent.com/itsoltech/basal-rs/main/install.sh | sh`)
instaluje paczkę macOS do `~/.local/bin` (`--prefix DIR`) i uruchamia
`basal doctor`; aktualizacja: `basal update` albo
ponowne uruchomienie skryptu, usługa w tle: `basal setup --service`
(launchd). Wybrać jeden sposób: przy obu w `PATH` są dwie binarki.

Binarka nie jest podpisana certyfikatem Apple. Homebrew i skrypt pobierają
ją bez atrybutu kwarantanny, więc macOS ją uruchamia; plik pobrany ręcznie
przeglądarką z GitHub Releases trzeba odblokować
(`xattr -d com.apple.quarantine basal`).

## Linux (x86_64, NVIDIA) i macOS bez Homebrew

```sh
curl -fsSL https://raw.githubusercontent.com/itsoltech/basal-rs/main/install.sh | sh
```

Skrypt pobiera paczkę z GitHub Releases, sprawdza SHA-256 z `SHA256SUMS`,
instaluje do `~/.local` (`--prefix DIR`), a potem uruchamia `basal setup` i
`basal doctor`. Ponowne uruchomienie aktualizuje instalację; `--version X.Y.Z`
instaluje wybrane wydanie, `--dry-run` tylko pokazuje kroki. Skrypt sam
wybiera paczkę: macOS na Apple Silicon albo Linux x86_64; dalsza część
dotyczy Linuksa.

Wymagania: glibc 2.28 lub nowsza (Debian 10, Ubuntu 20.04, RHEL 8 i
nowsze), sterownik NVIDIA 575.51 lub nowszy, karta o compute capability 8.0
lub wyższej (A100, RTX 30xx i nowsze). Paczka zawiera `bin/basal` (polecenia
bez GPU) i `libexec/basal/basal-cuda` (kernele dla 8.0, 8.9 i 9.0);
`basal serve` i pozostałe polecenia GPU uruchamiają tę drugą. Biblioteki
CUDA 12.9 (cudart, cuBLAS, cuBLASLt, cuRAND, ~1 GB) `basal setup` pobiera z
serwerów NVIDIA (`developer.download.nvidia.com/compute/cuda/redist`,
SHA-256 z manifestu NVIDIA), jeśli system ich nie ma.

Bez instalacji na hoście: obraz `ghcr.io/itsoltech/basal-rs`
([docker-compose.yml](../docker-compose.yml)).

## Modele

`--model` w `basal serve`, `init`, `doctor` i `setup` przyjmuje:

| `--model` | model |
|---|---|
| `mini`, `4.5B`, `max` albo `basal-1.5-max` | `Remek/basal-1.5-…` w rewizji sprawdzonej z tym wydaniem |
| `basal-1.6-x` | `Remek/basal-1.6-x` (domyślny właściciel Remek), gałąź `main` |
| `Remek/basal-1.5-max` | jak wyżej, pełna nazwa repozytorium |
| `Remek/basal-1.5-max@REF` | gałąź, tag albo commit |
| `./model`, `/ścieżka`, `~/model` | lokalny katalog modelu |

Kilka `--model` serwuje kilka modeli w jednym procesie. Opcje GPU i kolejki
z linii poleceń (`--dtype`, `--max-batch-tokens` itd.) dotyczą wtedy każdego
z nich.

Bez `--model` model wskazuje plik konfiguracji, szukany w tej kolejności:
`--config PLIK`, `BASAL_CONFIG`, `basal-serve.yml` w bieżącym katalogu. Bez
pliku `basal serve` serwuje basal-1.5-4.5B. `basal init [--model ...]`
zapisuje `basal-serve.yml` w bieżącym katalogu; pełny opis opcji:
[serve.example.yml](../serve.example.yml). Obraz Docker domyślnie czyta
`/config/serve.yml`, a argumenty po nazwie obrazu (np. `--model mini`)
zastępują plik.

## Polecenia

| Polecenie | Działanie |
|---|---|
| `basal doctor [--model REF \| --config PLIK] [--json]` | system, GPU, sterownik i biblioteki CUDA, konfiguracja, modele w cache lub do pobrania, pamięć GPU i dysk, dostęp do Hugging Face, port; przy każdym problemie sposób naprawy |
| `basal setup [--model REF \| --config PLIK] [--prefetch] [--service] [--force]` | biblioteki CUDA (Linux), opcjonalnie pobranie modeli i usługa użytkownika (systemd, launchd) z tymi modelami lub plikiem; plików konfiguracji nie zapisuje |
| `basal init [--model REF]... [--out PLIK] [--force]` | `basal-serve.yml` z wybranymi modelami |
| `basal serve [--model REF]... [--config PLIK] [--addr ADRES] [--access-log] [--metrics]` | serwer; modele jak w [Modele](#modele); log podaje czas pobrania lub odczytu modelu z cache, ładowania na GPU i gotowości; `--access-log` (albo `access_log: true`, `BASAL_ACCESS_LOG=1`) dopisuje linię na żądanie: `[POST] 200 /v1/systemone 31 ms (queue 0.5 ms, compute 30 ms, batch 1)`; `--metrics` (albo `metrics: true`) włącza domyślnie wyłączone [metryki Prometheus](OBSERVABILITY.md) |
| `basal update [--check] [--version X]` | najnowsze wydanie w miejsce tej instalacji (Homebrew: `brew upgrade`, obraz: `docker compose pull`) |
| `basal uninstall [--models] [--dry-run] [--yes]` | usuwa usługę, cache, biblioteki CUDA, binarki instalacji ze skryptu, z `--models` modele basal |
| `basal --version` | wersja, commit, backend GPU |
| `--color auto\|always\|never` | kolory w każdym poleceniu; `auto`: tylko w terminalu i bez `NO_COLOR`, `CLICOLOR_FORCE=1` wymusza |

`basal serve` i `basal doctor` raz na dobę sprawdzają, czy jest nowsze
wydanie, i piszą o tym w logu (`BASAL_NO_UPDATE_CHECK=1` wyłącza).

CUDA: wątek serwera domyślnie czeka na GPU aktywnie i zajmuje przy tym cały
rdzeń CPU. Na hoście współdzielonym z innymi usługami
`BASAL_CUDA_SYNC=blocking` zwalnia ten rdzeń kosztem ok. 2,5% czasu odpowiedzi
([pomiar](../reports/choice-sets/README.md#czekanie-na-gpu-cuda)); wynik się
nie zmienia.

## Pliki

| | macOS | Linux |
|---|---|---|
| cache (tabele GEMM, pobrane archiwa) | `~/Library/Caches/basal` | `~/.cache/basal` |
| dane (biblioteki CUDA) | `~/Library/Application Support/basal` | `~/.local/share/basal` |
| modele | `~/.cache/huggingface` (`HF_HOME`) | jak obok |

`BASAL_HOME=DIR` przenosi cache do `DIR/.cache` i dane do `DIR/.local`
(obraz Docker: `/data`). Plik konfiguracji leży tam, gdzie zapisze go
użytkownik ([Modele](#modele)).

## Odinstalowanie

```sh
basal uninstall --models --dry-run   # lista z rozmiarami, nic nie usuwa
basal uninstall --models             # usuwa po potwierdzeniu (--yes bez pytania)
```

`basal uninstall` zatrzymuje i usuwa usługę z `basal setup --service`
(launchd, systemd), usuwa cache (tabele GEMM, pobrane archiwa),
dane (biblioteki CUDA) i binarki instalacji ze skryptu (`bin/basal`,
`libexec/basal`, `share/doc/basal` w prefiksie). Z `--models` także modele
basal (`models--Remek--basal-*`) z cache Hugging Face; inne repozytoria w tym
cache zostają. `basal-serve.yml` i plik z `BASAL_CONFIG` zostają. Przy instalacji z Homebrew na końcu:

```sh
brew services stop basal-rs; brew uninstall basal-rs; brew untap itsoltech/tap
```

Bez binarki (ręcznie): katalogi z tabeli [Pliki](#pliki),
`~/.local/bin/basal`, `~/.local/libexec/basal`, `~/.local/share/doc/basal`,
`~/Library/LaunchAgents/tech.itsol.basal.plist` lub
`~/.config/systemd/user/basal.service` i katalogi
`~/.cache/huggingface/hub/models--Remek--basal-*`.

Docker: `docker compose down -v` (kontener i wolumen `basal-data` z modelami
i tabelami GEMM), potem `docker image rm ghcr.io/itsoltech/basal-rs:<tag>`.

## Wydanie (dla opiekunów)

1. Wersja w `Cargo.toml` (`workspace.package.version`), commit.
2. `git tag -a vX.Y.Z -m ... && git push origin vX.Y.Z`: paczki,
   `SHA256SUMS`, `install.sh` w GitHub Release, potem obraz GHCR (`vX.Y.Z`,
   `vX.Y`, `vX`, `latest` i warianty `-sm80` / `-sm90`) i formuła w
   `itsoltech/homebrew-tap` (sekret `HOMEBREW_TAP_TOKEN`). Bez udanego builda
   obu paczek nic nie jest publikowane. Obraz wydania można opublikować
   ponownie: workflow `docker`, `workflow_dispatch` z tagiem.
3. Bez publikacji: `workflow_dispatch` workflow `release` buduje paczki jako
   artefakty; lokalnie `tools/release/package-macos.sh`,
   `tools/release/package-linux.sh`, a `install.sh` i `basal update`
   instalują z katalogu `BASAL_RELEASE_URL` (paczki i `SHA256SUMS`).
