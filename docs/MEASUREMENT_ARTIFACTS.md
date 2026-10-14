# Dane pomiarowe poza Gitem

Raporty w [reports](../reports/README.md) opisują wyniki w Markdown. Pełne dane maszynowe nowych pomiarów (pliki
JSON z faktami, parami A/B, rozkładami i sumami kontrolnymi) nie są przechowywane w Gicie. Trzymamy je jako
archiwa w publicznym Hugging Face Storage Bucket
[itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs).

Ta zasada dotyczy plików wyników. Dane, z których korzysta kod lub które są wejściem pomiarów, zostają w Gicie bez
zmian: tabele GEMM w `crates/basal-cli/gemm-tables`, referencje upstream `reports/reference-*` i dane starszych
raportów.

## Zasady

- Bucket jest zmiennym magazynem obiektów, a nie repozytorium z historią wersji. Dokumentacja Hugging Face opisuje
  buckety jako „mutable storage” bez kontroli wersji, w przeciwieństwie do repozytoriów Git
  ([Buckets](https://huggingface.co/docs/huggingface_hub/guides/buckets)). Dlatego:
  - każdy zestaw danych ma własny prefiks `measurements/<nazwa>-<data>/<sha256 archiwum>/`;
  - zakończonego prefiksu nie nadpisujemy i nie usuwamy; poprawka danych to nowe archiwum pod nowym prefiksem;
  - przed użyciem danych zawsze sprawdzamy SHA-256 archiwum i sumy plików w nim.
- Raport w Gicie podaje URI archiwum i jego SHA-256. Ścieżki plików wewnątrz archiwum są takie same jak ścieżki,
  pod którymi te pliki były w repozytorium (`reports/<katalog>/<plik>`).
- Do bucketu trafiają tylko dane, które mogłyby być publiczne w repozytorium: bez adresów, nazw hostów,
  identyfikatorów maszyn i instancji, lokalnych ścieżek, kwot ani kosztów. Prywatne dane sesji (surowe katalogi z
  maszyn, rekordy wynajmu) nie są publikowane.
- Kod repozytorium nie zależy w czasie działania od Hugging Face. Dane pobiera się ręcznie przed analizą.

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

### Pobranie i sprawdzenie

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

## Publikacja nowego zestawu danych

1. Nowe wyniki pomiaru zapisuj poza Gitem, w `.cache/measurement-artifacts/<nazwa>-<data>/`, w układzie
   `reports/<katalog raportu>/<plik>`. Raport Markdown w `reports/<katalog raportu>/` zostaje w Gicie.
2. Przejrzyj dane przed publikacją: usuń adresy, nazwy hostów, identyfikatory maszyn i instancji, lokalne ścieżki,
   kwoty i koszty. Prywatne surowe dane sesji zostają lokalnie.
3. Utwórz w katalogu danych `SHA256SUMS` (`sha256sum` ścieżek względnych) i archiwum `tar.gz` z deterministyczną
   kolejnością, właścicielem i czasem plików. Policz SHA-256 archiwum.
4. Sprawdź, że prefiks `measurements/<nazwa>-<data>/<sha256 archiwum>/` jest pusty
   (`hf buckets list itsoltech/basal-rs/<prefiks> -R`). Jeśli nie jest pusty, nie wysyłaj.
5. Wyślij archiwum: `hf buckets cp <archiwum> hf://buckets/itsoltech/basal-rs/<prefiks>/<archiwum>`.
6. Pobierz je z powrotem do innego, pustego katalogu w `.cache/` i powtórz kroki z sekcji „Pobranie i
   sprawdzenie”: suma archiwum i `sha256sum -c SHA256SUMS`.
7. Dopiero po kroku 6 dopisz w raporcie URI, SHA-256 archiwum i stan publikacji. Lokalną kopię archiwum zachowaj.
