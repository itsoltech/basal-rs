# Dane pomiarowe poza Gitem

W `reports/` przechowujemy precyzyjne podsumowania Markdown: cel, wersje i wejścia,
sprzęt, metodę, wyniki, różnice, ograniczenia oraz decyzje. Nowe pliki wynikowe JSON,
JSONL, CSV, logi, profile i eksporty binarne zapisujemy poza Gitem. Tabele GEMM,
referencje upstream i istniejące wejścia wymagane przez kod lub narzędzia zostają w Gicie.
Rozszerzenie pliku samo w sobie nie rozstrzyga, czy jest wynikiem, czy potrzebnym zasobem.

Wspólne zasady są w [AGENTS.md](../AGENTS.md). Procedura pochodzi z
[PR #7](https://github.com/itsoltech/basal-rs/pull/7), z instrukcji przy commicie
`0af3f4e471a044fbbf9bad2f976d4aa8060bd895` z tego niezmergowanego PR. Dane poprzedniej kampanii multiGPU mają
[osobny katalog i instrukcję](https://github.com/itsoltech/basal-rs/blob/0af3f4e471a044fbbf9bad2f976d4aa8060bd895/docs/MEASUREMENT_ARTIFACTS.md).

## Publikacja i integralność

Docelowy magazyn to publiczny [HF Storage Bucket itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs),
URI `hf://buckets/itsoltech/basal-rs/`. Buckety nie mają historii wersji
([dokumentacja HF](https://huggingface.co/docs/huggingface_hub/guides/buckets)).

1. Przygotuj dane w `.cache/measurement-artifacts/<nazwa>-<data>/`, zachowując logiczne
   ścieżki `reports/<katalog>/<plik>`. Prywatne surowe dane sesji zostają lokalnie.
2. Przejrzyj publiczną zawartość: bez sekretów, identyfikatorów maszyn, adresów,
   nazw hostów, lokalnych ścieżek i kosztów sesji.
3. Utwórz `SHA256SUMS` ze ścieżkami względnymi i deterministyczne `tar.gz`
   (kolejność, właściciel i czas plików). Policz SHA-256 archiwum.
4. Korzystaj z globalnego `hf` CLI, bezpośrednio `~/.local/bin/hf`, gdy istnieje.
   Sprawdź `hf auth whoami`; przy braku logowania użyj `hf auth login` w tym samym
   środowisku. Tokenów nie wpisuj do raportów ani czatu.
5. Sprawdź pusty prefiks przez `hf buckets list itsoltech/basal-rs/<prefiks> -R`.
   Publikuj przez `hf buckets cp` pod `measurements/<nazwa>-<data>/<sha256 archiwum>/`.
   Zakończonego prefiksu nie nadpisuj i nie usuwaj; korekta wymaga nowego archiwum.
6. Pobierz archiwum do innego, pustego katalogu, sprawdź jego SHA-256, rozpakuj i
   sprawdź wszystkie wpisy `SHA256SUMS`. Potwierdź też pobranie bez autoryzacji.
7. Dopiero po obu weryfikacjach usuń zbędne kopie wyników z Git i dopisz do raportów
   URI, SHA-256, ścieżki danych i sposób ich odtworzenia. Zachowaj lokalne archiwum.

Dane pobieramy jawnie przed analizą; runtime nie pobiera archiwów wyników. Nowe lub migrowane runnery,
które odczytują archiwizowane wyniki, powinny przyjmować katalog przez
`--measurement-artifacts`, bez automatycznej zależności sieciowej.
To konwencja dla takich runnerów. CLI basal w tej gałęzi nie ma tej opcji.

## Archiwum `upstream-2.0-2026-10-14`

Zawiera 29 plików JSON i 5 oryginalnych raportów Markdown sprzed migracji oraz
`SHA256SUMS`. Wszystkie ścieżki danych zaczynają się od `reports/`.
Prywatne surowe katalogi sesji H100, binarki i `.itsol/` pozostają lokalnie.

| Pole | Wartość |
|---|---|
| URI | `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-2026-10-14/c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536/measurement-data.tar.gz` |
| Rozmiar gzip | 59 198 B |
| SHA-256 | `c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536` |
| Weryfikacja | 2026-10-14: ponowne pobranie z autoryzacją i anonimowo; SHA archiwum i 34/34 sumy plików zgodne |

| Ścieżka w archiwum | Zawartość |
|---|---|
| `reports/upstream-2.0-discovery-2026-10-14/` | 1 JSON z metadanymi/rewizjami modeli i oryginalny README |
| `reports/upstream-2.0-cpu-2026-10-14/` | 8 JSON: prompty/tokeny, procesory, reguły, kalibracja, błędy i YAML; oryginalny README |
| `reports/upstream-2.0-boundaries-2026-10-14/` | 3 JSON: certyfikaty/act, plik o2 i budżet promptów; oryginalny README |
| `reports/upstream-2.0-h100-2026-10-14/` | 13 JSON: readouts, FP32/BF16, media, kolejności, HTTP/lifecycle, bitowe A/B, źródła i kontrole; oryginalny README |
| `reports/upstream-2.0-decisions-2026-10-14/` | 4 JSON: decyzje względem FP32, arytmetyka hosta, alternatywy sumowania i provenance; oryginalny README |

### Pobranie i odtworzenie

Uruchom z katalogu głównego repozytorium. Potrzebny jest `hf` CLI z `hf buckets`.
Katalog docelowy rozpakowania musi być nowy. Polecenia nie zastępują istniejących danych.

```bash
BASAL_ARTIFACT_PREFIX=measurements/upstream-2.0-2026-10-14/c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536
BASAL_ARTIFACT_DL=.cache/measurement-download/upstream-2.0-2026-10-14
BASAL_ARTIFACT_ROOT=.cache/measurement-artifacts/upstream-2.0-2026-10-14
mkdir -p .cache/measurement-download .cache/measurement-artifacts &&
  mkdir "$BASAL_ARTIFACT_DL" &&
  hf buckets cp "hf://buckets/itsoltech/basal-rs/$BASAL_ARTIFACT_PREFIX/measurement-data.tar.gz" "$BASAL_ARTIFACT_DL/measurement-data.tar.gz" &&
  echo "c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536  $BASAL_ARTIFACT_DL/measurement-data.tar.gz" | sha256sum -c - &&
  mkdir "$BASAL_ARTIFACT_ROOT" &&
  tar -xzf "$BASAL_ARTIFACT_DL/measurement-data.tar.gz" -C "$BASAL_ARTIFACT_ROOT" &&
  (cd "$BASAL_ARTIFACT_ROOT" && sha256sum -c SHA256SUMS)
```

Na macOS użyj `shasum -a 256 -c` zamiast `sha256sum -c` w obu miejscach.

Po rozpakowaniu dawny `reports/<katalog>/<plik>.json` jest pod
`$BASAL_ARTIFACT_ROOT/reports/<katalog>/<plik>.json`. Oryginalne README są historycznymi
kopiami sprzed dopisania not o archiwum; aktualne podsumowania i publiczne odsyłacze
pozostają w Gicie. Lokalnego stanu inicjatywy `.itsol/` nie publikujemy.

## Końcowa kampania upstream 2.0

[Podsumowanie](../reports/upstream-2.0-final-2026-10-14/README.md) opisuje CUDA, Metal, Intel, rzeczywiste media, act, kontekst i wydajność. Nowe archiwum: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-final-2026-10-14/3bf4dac1aab9a203d731cd67feae1372ea1bf7eaea46f9baf5b2bfc64c474f61/measurement-data.tar.gz`, SHA-256 `3bf4dac1aab9a203d731cd67feae1372ea1bf7eaea46f9baf5b2bfc64c474f61`, 3 148 341 B. Pobranie z autoryzacją i anonimowo potwierdziło 639/639 sum plików.

Użyj powyższej procedury z nazwą `upstream-2.0-final-2026-10-14`, tym prefiksem i tą sumą. Dane są pod `reports/upstream-2.0-final-2026-10-14/` wewnątrz archiwum. Media base64 zastąpiono hashami i liczbą bajtów; przed ponownym forwardem odtwórz je z `inputs/media-source-manifest.json`. Archiwum nie rozprowadza zdjęć/nagrań ani prywatnych danych maszyn; zachowuje logity, tokeny, prawdopodobieństwa, odpowiedzi i niepowodzenia.

Dodatkowy [benchmark Metal 32768](../reports/upstream-2.0-metal-context-2026-10-14/README.md): `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-metal-context-2026-10-14/e6f30d416ec8037ca6cb78a85ac8b6e48ffd5fc90cead619a35a8c5a0db2bfb2/measurement-data.tar.gz`, SHA-256 `e6f30d416ec8037ca6cb78a85ac8b6e48ffd5fc90cead619a35a8c5a0db2bfb2`, 10 829 B; 15/15 sum potwierdzonych z autoryzacją i anonimowo. Zawiera również pochodną analizę progu noul z poprzedniego loadtestu. Dane pod `reports/upstream-2.0-metal-context-2026-10-14/`; użyj procedury z nowym prefiksem, nazwą i sumą.
