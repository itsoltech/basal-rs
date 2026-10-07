# Paczki instalacyjne: próba bez publikacji

Data: 2026-10-07. Paczki zbudowane lokalnie skryptami `tools/release/`,
zainstalowane `install.sh` z katalogu (`BASAL_RELEASE_URL`), bez GitHub
Releases i bez tapu Homebrew. Opis instalacji: [docs/INSTALL.md](../../docs/INSTALL.md).

## Linux

Serwer z RTX 6000 Ada: Debian 12 (glibc 2.36), sterownik 580.159.03, bez
bibliotek CUDA na hoście. Paczka `basal-0.1.0-x86_64-linux.tar.gz` (29,6 MB)
z obrazu Rocky Linux 8 (glibc 2.28): `bin/basal` i `libexec/basal/basal-cuda`
(kernele dla 8.0, 8.9, 9.0; kernele candle dla 8.0).

- `install.sh --dry-run`: sprawdzenie SHA-256 i lista kroków, nic nie
  zainstalowane.
- `install.sh`: instalacja do prefiksu, `basal setup` pobrał od NVIDIA
  cudart 12.9.79, cuBLAS 12.9.1.4 i cuRAND 10.3.10.19 (1,03 GB archiwów,
  976 MB bibliotek), zapisał konfigurację; `basal doctor`: wszystkie
  biblioteki znalezione, binarka CUDA startuje. Przed `setup` `doctor`
  wskazywał brakujące biblioteki i polecenie naprawy; sama binarka CUDA bez
  nich nie startuje (`libcurand.so.10: cannot open shared object file`).
- Zgodność (limit mocy 250 W na czas testu): eksporty basal-1.5-mini i
  basal-1.5-max (pojedynczo i w drzewie) zainstalowaną paczką są bitowo równe
  eksportom binarki z `main` budowanej w kontenerze CUDA (`linux-exports/`,
  różnica 0,0, 44/44).
- `basal serve` bez argumentów z konfiguracją `basal init --model mini`:
  pobranie modelu, tabela GEMM wygenerowana w 156 s, odpowiedź
  `/v1/systemone`, zakończenie przez SIGTERM z kodem 0.
- `basal update` z pakietem o kolejnym numerze (te same binarki pod nazwą
  0.1.1): wykrycie, pobranie, sprawdzenie SHA-256, podmiana binarek.

## macOS

Mac Studio M2 Max, macOS 27.0.1. Paczka
`basal-0.1.0-aarch64-apple-darwin.tar.gz` (8,4 MB).

- `install.sh`: instalacja, `basal setup` (konfiguracja), `basal doctor`
  (Metal, zestaw roboczy 25,0 GiB, model z cache).
- `basal serve` bez konfiguracji pobrał basal-1.5-4.5B (9,5 GB) i odpowiadał
  na `/v1/models`.
- `basal update`: jak na Linuksie, a pakiet ze zmienioną zawartością
  odrzucony (SHA-256); `basal doctor` zgłasza nowszą wersję; w buildzie z
  repozytorium `update` odmawia (`git pull && cargo build --release`).

## Czego nie sprawdzono

GitHub Actions (`release.yml`) i formuła Homebrew w tapie: wymagają tagu i
repozytorium tapu. Kart innych niż RTX 6000 Ada (PTX 8.0 i 9.0), dystrybucji
innych niż Debian 12, macOS bez dostępu do sieci.
