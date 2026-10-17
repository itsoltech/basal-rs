# Dane pomiarowe poza Gitem

W `reports/` przechowujemy precyzyjne podsumowania Markdown: cel, wersje i wejścia,
sprzęt, metodę, wyniki, różnice, ograniczenia oraz decyzje. Nowe pliki wynikowe JSON,
JSONL, CSV, logi, profile i eksporty binarne zapisujemy poza Gitem. Tabele GEMM,
referencje upstream i istniejące wejścia wymagane przez kod lub narzędzia zostają w Gicie.
Rozszerzenie pliku samo w sobie nie rozstrzyga, czy jest wynikiem, czy potrzebnym zasobem.

Wspólne zasady są w [AGENTS.md](../AGENTS.md). Dane kampanii wiele GPU, W8A16 i W4A16 opisuje sekcja
[archiwum `multigpu-2026-10-14`](#archiwum-multigpu-2026-10-14).

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
To konwencja dla takich runnerów, np. `tools/perf/run-w4a16-cloud.py` i
`tools/perf/run-layer-split-cloud.py`; CLI basal nie ma tej opcji.

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

## Wydajność Omni na Ada

[Raport](../reports/omni-performance-ada-2026-10-16/README.md) opisuje pojedyncze żądania tekstowe 128–32768 tokenów, rzeczywiste media, QPS i ogon ruchu mieszanego obu Omni oraz kontrolę jakości FP32. Zachowuje także nieudane warianty i regresje.

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/omni-performance-ada-2026-10-16/82aaf99af87532eaccfaf19e3fe5462b5e0d9b4f1ffcac6ccba7ba987ac8a0a7/measurement-data.tar.gz`, SHA-256 `82aaf99af87532eaccfaf19e3fe5462b5e0d9b4f1ffcac6ccba7ba987ac8a0a7`, 26 718 255 B. Pobranie z autoryzacją i anonimowo potwierdziło 1812/1812 sum plików. Dane, metody i przenośne analizy są pod `reports/omni-performance-ada-2026-10-16/` wewnątrz archiwum; raport podaje pełne instrukcje odtworzenia. Prawdziwe media pozostają zewnętrzne i są odtwarzane z przypiętych źródeł oraz sum.

## Archiwum `multigpu-2026-10-14`

Zawiera wyniki raportów multiGPU, W8A16 i W4A16 z 2026-10-11 – 2026-10-14 (52 pliki JSON) oraz wersje 23 plików
Markdown tych raportów z chwili archiwizacji (commit `6e85375`), przed dopisaniem not o archiwum. Opis wyników:
[zestawienie](../reports/multigpu-results-2026-10-14/README.md).

| Pole | Wartość |
|---|---|
| URI | `hf://buckets/itsoltech/basal-rs/measurements/multigpu-2026-10-14/71acff72913c376a84af653ccb9a5f7df86d9bfbaa1c6f05700d61d8bc2328b2/measurement-data.tar.gz` |
| Strona | [itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs), prefiks `measurements/multigpu-2026-10-14/71acff72…/` |
| Rozmiar | 1 366 026 B (gzip) |
| SHA-256 archiwum | `71acff72913c376a84af653ccb9a5f7df86d9bfbaa1c6f05700d61d8bc2328b2` |
| Zawartość | 75 plików pod `reports/…` (52 JSON, 23 Markdown) i `SHA256SUMS` w katalogu głównym archiwum |
| Stan publikacji | Opublikowane 2026-10-14. Anonimowe pobranie potwierdziło SHA-256 archiwum oraz sumy wszystkich 75 plików. |

`SHA256SUMS` zawiera sumy wszystkich 75 plików, w formacie `sha256sum`, ze ścieżkami względnymi od katalogu głównego
archiwum.

### Pobranie i sprawdzenie `multigpu-2026-10-14`

Polecenia uruchamia się z katalogu głównego repozytorium. Katalog `.cache/` jest ignorowany przez Git. Potrzebny
jest `hf` CLI z poleceniem `hf buckets` (pakiet `huggingface_hub`).

```bash
PREFIX=measurements/multigpu-2026-10-14/71acff72913c376a84af653ccb9a5f7df86d9bfbaa1c6f05700d61d8bc2328b2
DL=.cache/measurement-download/multigpu-2026-10-14
ROOT=.cache/measurement-artifacts/multigpu-2026-10-14
mkdir -p "$DL" .cache/measurement-artifacts
hf buckets cp "hf://buckets/itsoltech/basal-rs/$PREFIX/measurement-data.tar.gz" "$DL/measurement-data.tar.gz"

# 1. suma archiwum przed rozpakowaniem,
# 2. rozpakowanie do nowego katalogu (mkdir kończy się błędem, gdy $ROOT już istnieje),
# 3. sumy wszystkich 75 plików, liczone z katalogu głównego rozpakowania.
echo "71acff72913c376a84af653ccb9a5f7df86d9bfbaa1c6f05700d61d8bc2328b2  $DL/measurement-data.tar.gz" | sha256sum -c - &&
  mkdir "$ROOT" &&
  tar -xzf "$DL/measurement-data.tar.gz" -C "$ROOT" &&
  (cd "$ROOT" && sha256sum -c SHA256SUMS)
```

Po rozpakowaniu plik raportu `reports/<katalog>/<plik>.json` jest pod `$ROOT/reports/<katalog>/<plik>.json`.
Runnery [run-w4a16-cloud.py](../tools/perf/run-w4a16-cloud.py) i
[run-layer-split-cloud.py](../tools/perf/run-layer-split-cloud.py) czytają zarchiwizowane wejścia z katalogu
`--measurement-artifacts` (domyślnie `.cache/measurement-artifacts/multigpu-2026-10-14`). Przypiętą sumę SHA-256 w
runnerze ma tylko plik faktów P0: przy innej sumie runner nie używa tych faktów, a kontrola P0 w zapisie sesji ma
status GAP. Sumy trzech raportów W8, z których `run-w4a16-cloud.py` korzysta przy `--plan-only`, runner tylko
zapisuje i wypisuje; nie porównuje ich z oczekiwaną wartością. Integralność wszystkich czterech plików potwierdza
ręczne sprawdzenie sumy archiwum i `SHA256SUMS` opisane wyżej. Szczegóły w [CLOUD.md](CLOUD.md).

### Pliki JSON w archiwum

Ta sama lista z sumami jest w `SHA256SUMS`. Sumy plików Markdown są tylko tam.

| Ścieżka w archiwum | Bajty | SHA-256 |
|---|---:|---|
| `reports/multigpu-isolation-2026-10-11/measurements.json` | 28192 | `df845c9dead50daf268ea30a820a0ab1b35e0bf0568c37d859da6023f3387e7b` |
| `reports/multigpu-layer-l1-2026-10-13/facts.json` | 137631 | `dc1b16e91eac02322de4330f9957afb6a85d6b04783ba7f1069de96759940169` |
| `reports/multigpu-layer-l1-2026-10-13/provenance.json` | 10985 | `9c8b912dc7403cf82831f35f8b2418d8b492abce39e48224ec0fe285884e0bd9` |
| `reports/multigpu-layer-l1-2026-10-13/validation.json` | 11909 | `6ba9eff86aaa6c12d5b7b3ce66882878adcea52344d14fcc3cebab43a39284ce` |
| `reports/multigpu-layer-l1-revalidation-2026-10-13/facts.json` | 142337 | `68008e9314a046c99197a5d76d56f7afd97d400c0bd073ca4d5884c39eecf5e2` |
| `reports/multigpu-layer-l1-revalidation-2026-10-13/provenance.json` | 15586 | `7178a6e0e12e451213443a10e2fff0c3f9f61ae5fdc70f8092e57d38a58853c9` |
| `reports/multigpu-layer-l1-revalidation-2026-10-13/validation.json` | 16271 | `93ede71fb55f06348627c55fcba13c5416c9b8980d71c089199047540d843cfa` |
| `reports/multigpu-layer-l2-2026-10-13/facts.json` | 121177 | `51d592c90180fe5d81cf76feed940ab4c4d49d3dae5df744f9a8309d5a4e6fc3` |
| `reports/multigpu-layer-l2-2026-10-13/provenance.json` | 9966 | `8469f116255ecd65d46888b6fb4dba9e1a88052ae4cb70474957f04c97d8c92b` |
| `reports/multigpu-layer-l2-2026-10-13/validation.json` | 12023 | `924d4ccf7c4dce54aa641b041fae42d26579cc0cfad23230981fbe39687b7608` |
| `reports/multigpu-layer-l2-revalidation-2026-10-14/facts.json` | 237654 | `fffe3bb796f4fbd60d8f7a154b9e72e6375275a2ec4d469eda5a151e99736933` |
| `reports/multigpu-layer-l2-revalidation-2026-10-14/provenance.json` | 11242 | `fcecef81f2f406453c9b0e7b0647eaa6356786ac103ba7cf468310962bf64dd7` |
| `reports/multigpu-layer-l2-revalidation-2026-10-14/validation.json` | 12149 | `4e7104bc6b5dae067d3d9d524e0fdb63e4210db9d0e7158f9e1e6676d742be70` |
| `reports/multigpu-layer-l3-2026-10-13/facts.json` | 114806 | `d372f9de5168d70f4234a73b92a1e36893295d6888f5e52e83bd523d1ba3a8d3` |
| `reports/multigpu-layer-l3-2026-10-13/provenance.json` | 11645 | `e65c20472ee9a1214e067380b3ff018b57bd1b72e571d556c62b33b0d224328c` |
| `reports/multigpu-layer-l3-2026-10-13/validation.json` | 14528 | `e43d72b84ec3f1db35c1ceddc0d4202df902b863958f8d6d0eaab25d3d1ef84c` |
| `reports/multigpu-layer-l3-revalidation-2026-10-14/facts.json` | 314808 | `5e1f770f36ee6fabd3142b3b4e4298a157471fea08c08c6710046d655b23101d` |
| `reports/multigpu-layer-l3-revalidation-2026-10-14/provenance.json` | 14231 | `975274c52b8bd2e3d7754946a88cf037544702af0c5f94cb513790216d7dec8e` |
| `reports/multigpu-layer-l3-revalidation-2026-10-14/validation.json` | 5607 | `99af0c5ee5f6d17343c9c38df374ff0db85748d851c3413734f3956d5532c684` |
| `reports/multigpu-replicas-2026-10-11/measurements.json` | 412279 | `6ea5921f09f30abd95f1cdc558b777d8916b9596ff5226e8ce7407cb7cdaac10` |
| `reports/multigpu-replicas-control-2026-10-11/measurements.json` | 59578 | `1eabec011cd5db87b9b636202b3acb7d5fdf0da484f0e104132fa1472f3e452c` |
| `reports/multigpu-w8-none-2026-10-11/facts.json` | 407085 | `fda8b2a56484da7f428fce35f9f906b9915e3bbe545681710b6db3ad5d8afe63` |
| `reports/multigpu-w8-original-2026-10-11/correctness.json` | 20165 | `cc488091c7dd3e9522c8a09e9fbae0a025f64a3339aedf771053a3102fe8ad3d` |
| `reports/multigpu-w8-original-2026-10-11/disposition.json` | 7702 | `724ad356ee4b250a5ba0f2986af27882588a7506c59597f8cfd1efd6c27a496e` |
| `reports/multigpu-w8-original-2026-10-11/perf-q8-original.json` | 280817 | `7dbbe7911cd74eb288320597195107feb39e8701936bff95ffc910049a204c8f` |
| `reports/multigpu-w8-original-2026-10-11/quality-primary.json` | 586400 | `7e64a0a0fc385864d4acdbf2be422215506bf639ba3fe7cff02ce198ddd233c7` |
| `reports/multigpu-w8-original-2026-10-11/raw-manifest.json` | 59379 | `df810ad4083489acb7d891a94cfa96cdae40432d64e02aeba73253973b963dbb` |
| `reports/multigpu-w8-quality-2026-10-11/endpoint-answers-4.5B.json` | 62395 | `6b541d7db8e93b14de4e84b762e6d62dbc03a799f4d47c23b1eb71dff02b5b7d` |
| `reports/multigpu-w8-quality-2026-10-11/endpoint-answers-mini.json` | 64128 | `f7e2592943e9999e3a1f359203ff4eb82f991010898da0ea9dff9605550bf7a9` |
| `reports/multigpu-w8-quality-2026-10-11/facts.json` | 23181 | `3f42e9030799c29d999cfa09c1b1e9b2e21ff617be988da098c72fbdca17e733` |
| `reports/multigpu-w8-quality-2026-10-11/quality-900-4.5B.json` | 623219 | `eada70210850b8a6a3b7d9a64fae2fd1e169c6d5af2f2c83d847e6890ddff1d9` |
| `reports/multigpu-w8-quality-2026-10-11/quality-900-mini.json` | 621956 | `a2132e85a8fc3c5fa87ecf9fea1b6c7411a59a588986c6ebcc5b645d0e025814` |
| `reports/w4a16-p0-2026-10-12/facts.json` | 117934 | `44005ecc347b52b5f2295ee9b54edbebc7f014e006b2c2fd86e9dcef7c5cbf68` |
| `reports/w4a16-p0-2026-10-12/provenance.json` | 4342 | `eed48f799993c8aceddcb4ab332efb93fe968c4382712dec2e95ac7ea9aafa2f` |
| `reports/w4a16-p0-2026-10-12/validation.json` | 558 | `8ea0b379288b6c2580c2bb83cf36cba80d91de29d6a7107551a92b158416c38e` |
| `reports/w4a16-s1-2026-10-12/facts.json` | 63152 | `558f8b61ef0874cae3514b17d8a19e109d6c0edafebd5ccc76fd7544105273c2` |
| `reports/w4a16-s1-2026-10-12/quality-44-34.json` | 97009 | `b4a475268f300bc28f83745d0dcd4ed0c49e7bf8329e40cd80d13f8289fa385c` |
| `reports/w4a16-s1-2026-10-12/quality-900.json` | 637310 | `4bf766517480f5cb4f3a6307ed3fce4d6cc47aa92e2500c9687b7f5c9433ed62` |
| `reports/w4a16-s1-2026-10-12/validation.json` | 57794 | `51dd92cc9867117c13c9102d06bf99a242a40d248432a2cf196cd602e856d8da` |
| `reports/w4a16-s2-2026-10-12/facts.json` | 33386 | `3ddf6755b073201687d294750608d556e546e8951b53d79fc143c72f6599bbc0` |
| `reports/w4a16-s2-2026-10-12/perf-none-family.json` | 288724 | `24ed890eccb6863ac0bd1edea481a596573312a17d714aa243ee25416d1340fa` |
| `reports/w4a16-s2-2026-10-12/validation.json` | 33982 | `e4a6884e076a46c428f6f3c3ddcaa300017304ad5aef876c95340acf0a9f07e3` |
| `reports/w4a16-s3-2026-10-12/facts.json` | 32489 | `4d9aeb312bf739f7071b8c64b9b3084240baa95aee6eeb6f307bd683e5925777` |
| `reports/w4a16-s3-2026-10-12/perf-q8-family.json` | 244220 | `adc219844d39d1003e47e36e06b293da715fbe739a8020eecc131f61fe0fb083` |
| `reports/w4a16-s3-2026-10-12/validation.json` | 36169 | `f158c396a268df5df8eaa08f37b7d63384fed0679b999956be44a213d3fbcb17` |
| `reports/w8a16-2026-10-11/commands.json` | 6494 | `fba7196884db81f6d182d99ba488b4e1ce3d6b8786993269fed60b39b0d08769` |
| `reports/w8a16-2026-10-11/correctness.json` | 24762 | `8732bde14a773502114656191627ae9af7ee498085fba291a2a367181001d0cd` |
| `reports/w8a16-2026-10-11/perf-q8.json` | 321976 | `5c6eb815bb7b13a8e9ec01ac9c5d68631b06847dd2da8dc96e798f7832f19f8d` |
| `reports/w8a16-2026-10-11/profile-memory.json` | 227000 | `42d09b1249c1af2d85f6904b3df71cd9edfa21837a8d44d097045679cef3583d` |
| `reports/w8a16-2026-10-11/provenance.json` | 23355 | `3a145d4877d105d8b79448c46385ea0c84f02708f9b8b30c9650a3d992691948` |
| `reports/w8a16-2026-10-11/raw-manifest.json` | 17337 | `9ccd9d51d5042b7f95a6631db6abaa036fad46bd234dfe77f4f2eadf08f73d95` |
| `reports/w8a16-2026-10-11/static-validation.json` | 3912 | `d021ee7c3e774aad353a798e67c033c694918ab2f7f4b502215b9284551e5500` |

## Upstream 2.1, SDK i nowe Omni

[Raport](../reports/upstream-2.1-2026-10-17/README.md) opisuje zadania 2.1,
cztery SDK, Omni 1.1 max i Ultra, jakość względem FP32 oraz HTTP względem BF16.
Zachowuje różnicę starszego `act`, przegrane Ultra w sekwencyjnych zadaniach,
OOM upstream przy C8, nieukończone próby i krótką serię obrazów jako SIGNAL.

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.1-2026-10-17/9a8dae4bc02b08a4f2bcaf50aec712710706fb746820aff89961d4924ac1b10d/measurement-data.tar.gz`,
SHA-256 `9a8dae4bc02b08a4f2bcaf50aec712710706fb746820aff89961d4924ac1b10d`,
7 352 849 B. Pobranie z autoryzacją i anonimowo potwierdziło 896/896 sum.
Dane są pod `reports/upstream-2.1-2026-10-17/`; raport oraz archiwalny
`REPRODUCE.md` podają polecenia odtworzenia. Media są zewnętrzne, przypięte
hashami. Archiwum zawiera też dokładne źródła pomiarów i patch kodu do `8adebaa`.
