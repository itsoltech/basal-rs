# Instalacja i aktualizacja

Paczki wydań powstają z tagów `v*` ([release.yml](../.github/workflows/release.yml)):
macOS na Apple Silicon (Metal) i Linux x86_64 z kartą NVIDIA (CUDA). Do
pierwszego wydania (`v0.1.0`) zostaje budowanie ze źródeł albo obraz Docker
([README](../README.md#uruchomienie-w-dockerze)).

## macOS (Apple Silicon)

```sh
brew install itsoltech/tap/basal-rs
basal doctor                    # sprawdzenie maszyny
basal serve                     # http://127.0.0.1:8000, basal-1.5-4.5B pobierany przy pierwszym starcie
brew services start basal-rs    # albo jako usługa w tle (log: $(brew --prefix)/var/log/basal.log)
```

Aktualizacja: `brew upgrade basal-rs`. Binarka nie jest podpisana
certyfikatem Apple; instalacja przez Homebrew i skrypt jej nie wymaga.

## Linux (x86_64, NVIDIA)

```sh
curl -fsSL https://raw.githubusercontent.com/itsoltech/basal-rs/main/install.sh | sh
```

Skrypt pobiera paczkę z GitHub Releases, sprawdza SHA-256 z `SHA256SUMS`,
instaluje do `~/.local` (`--prefix DIR`), a potem uruchamia `basal setup` i
`basal doctor`. Ponowne uruchomienie aktualizuje instalację; `--version X.Y.Z`
instaluje wybrane wydanie, `--dry-run` tylko pokazuje kroki. Skrypt działa
też na macOS.

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

## Polecenia

| Polecenie | Działanie |
|---|---|
| `basal doctor [--json]` | system, GPU, sterownik i biblioteki CUDA, konfiguracja, modele w cache lub do pobrania, pamięć GPU i dysk, dostęp do Hugging Face, port; przy każdym problemie sposób naprawy |
| `basal setup [--prefetch] [--service] [--force]` | biblioteki CUDA (Linux), plik konfiguracji, opcjonalnie pobranie modeli i usługa użytkownika (systemd, launchd) |
| `basal init [--model mini\|4.5B\|max] [--force]` | plik konfiguracji z wybranymi modelami |
| `basal serve` | serwer; bez `--config` i `--model` czyta konfigurację użytkownika, a bez niej serwuje basal-1.5-4.5B |
| `basal update [--check] [--version X]` | najnowsze wydanie w miejsce tej instalacji (Homebrew: `brew upgrade`, obraz: `docker compose pull`) |
| `basal --version` | wersja, commit, backend GPU |

`basal serve` i `basal doctor` raz na dobę sprawdzają, czy jest nowsze
wydanie, i piszą o tym w logu (`BASAL_NO_UPDATE_CHECK=1` wyłącza).

## Pliki

| | macOS | Linux |
|---|---|---|
| konfiguracja (`serve.yml`) | `~/Library/Application Support/basal` | `~/.config/basal` |
| cache (tabele GEMM, pobrane archiwa) | `~/Library/Caches/basal` | `~/.cache/basal` |
| dane (biblioteki CUDA) | `~/Library/Application Support/basal` | `~/.local/share/basal` |
| modele | `~/.cache/huggingface` (`HF_HOME`) | jak obok |

`BASAL_HOME=DIR` przenosi konfigurację do `DIR`, cache do `DIR/.cache`, dane
do `DIR/.local` (obraz Docker: `/data`); `BASAL_CONFIG` wskazuje plik
konfiguracji.

## Odinstalowanie

Homebrew: `brew uninstall basal-rs`. Skrypt: usunąć `~/.local/bin/basal`,
`~/.local/libexec/basal`, `~/.local/share/doc/basal`, a z danymi także
katalogi z tabeli wyżej (modele w cache Hugging Face).

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
