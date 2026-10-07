# Profil `dist` i buildy wydań

Data: 2026-10-07. Paczki wydań i obraz Docker budowane są profilem `dist`
(`Cargo.toml`: `release` z `lto = "fat"`, `codegen-units = 1`, bez informacji
debugowych), paczka Linux w etapach cargo-chef z cache warstw w rejestrze,
paczka macOS z `Swatinem/rust-cache`, jeden obraz Docker z kernelami 8.0, 8.9
i 9.0 zamiast trzech.

## Wyniki i szybkość (`m2-max/`, Apple M2 Max, zasilanie sieciowe)

`release` (A) i `dist` (B) z tego samego commitu:

- eksporty basal-1.5-mini i basal-1.5-4.5B w drzewie: bitowo równe
  (`cmp-*.json`);
- pojedyncza decyzja, ABBA, 3 rundy (`ab-bench-*`): iloraz B/A mediany lat2
  1,000 (mini, p25–p75 0,999–1,001) i 1,000 (4.5B, 0,999–1,000); 79,7 / 79,7
  ms i 235,0 / 234,8 ms, przepustowość 15,76 / 15,77 i 5,26 / 5,26 decyzji/s;
- HTTP, mini, ABBA (`http-*.json`): sekwencyjnie 10,47 żądania/s w obu,
  32 klientów 13,86–13,87 (A) i 13,85–13,88 (B) żądania/s, p50 2292–2308 ms;
- binarka macOS 25,1 MB (`release`, z informacjami debugowymi) i 16,1 MB
  (`dist`); kompilacja `dist` od zera 144 s.

Czas działania nie zmienia się w granicach pomiaru: forward liczy GPU, a
część CPU (tokenizacja, pakowanie, HTTP) jest mała. Zysk to mniejsze paczki.

## CUDA (`cuda/`)

Paczka Linux zbudowana w GitHub Actions (profil `dist`, kernele candle dla
8.0), zainstalowana `install.sh` na Debianie 12 z RTX 6000 Ada, limit mocy
250 W na czas testu: eksporty basal-1.5-mini i basal-1.5-max (pojedynczo i w
drzewie) bitowo równe eksportom binarki z `main` sprzed zmian buildu. Szybkości
na CUDA nie mierzono. Paczka macOS z CI na M2 Max: eksport mini w drzewie
bitowo równy eksportowi z [metal-m2-max-tree](../metal-m2-max-tree/README.md).

## Czas buildów w GitHub Actions

Workflow `release` uruchamiany ręcznie (`workflow_dispatch`):

| Przebieg | macOS | Linux | Paczka macOS | Paczka Linux |
|---|---:|---:|---:|---:|
| przed zmianą (cargo build w kontenerze, bez cache) | 6 min 26 s | 17 min 51 s | 8,4 MB | 29,6 MB |
| pierwszy z `dist` i pustym cache | 7 min 52 s | 27 min 25 s | 7,0 MB | 16,0 MB |
| kolejny, z cache | 3 min 19 s | 1 min 53 s | 7,0 MB | 16,0 MB |

Cache warstw paczki Linux jest w GHCR (`:buildcache-release-linux`), cache
macOS w cache GitHub Actions gałęzi `main` (przebieg z tagu go odczytuje).
Przebieg bez zmian zależności buduje od nowa tylko workspace; zmiana
`Cargo.lock` przebudowuje warstwę zależności.

Obraz Docker (`docker.yml`): jeden build (14 min z pustym cache `:buildcache`)
zamiast trzech równoległych po 13–14 min; mniej minut runnerów i jeden zestaw
warstw w GHCR.
