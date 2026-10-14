# Etap 05, podział warstw między GPU — sesja L1, 2026-10-13

Stan: raport danych z jednej zakończonej sesji L1 (zgodność, odmowy, HTTP,
pojemność, transfery). To nie jest akceptacja etapu 05 ani końcowe review.
Sesje L2 (regresja toru domyślnego i podział `none`) i L3 (tryby
kwantyzowane) są w toku albo przed nami. Raport nie zawiera żadnych
twierdzeń o szybkości.

## Wynik w skrócie

- **Runner:** `complete: true`, `stopped: null`, kod wyjścia 3 (według
  reguły runnera: brak FAIL i SIGNAL, co najmniej jeden GAP). 641 kontroli:
  PASS 599, FAIL 0, GAP 42, SIGNAL 0. Wszystkie sześć jednostek C1–C6
  wystartowało i się zakończyło.
- **C1, tor jednego GPU:** 20 par CAND wobec BASE (34 eksporty, mini i 4.5B,
  `none`/`q8_0`/`q4_k`, w tym 900 pytań). Wszystkie pary są bajtowo równe
  (`bench.jsonl` i `systemone.jsonl`), a `basal compare` nie pokazuje żadnej
  różnicy. CAND na GPU 1 daje te same bajty co BASE na GPU 0.
- **C2, podział:** 24 pary podziału CAND wobec BASE na jednym GPU (31
  eksportów). Pary obejmują układy, cache, `CUDA_VISIBLE_DEVICES=1,0` i 900
  pytań. Wszystkie 24 pary są bajtowo równe, `c2_differing` jest puste, a
  `manifest_problems` puste w 44 z 44 par. Warunkowa diagnostyka FP32 nie
  była potrzebna.
- **C3, odmowy:** 38 niepoprawnych konfiguracji zakończyło się kodem ≠ 0 z
  oczekiwanym komunikatem, przed pobieraniem, wyszukiwaniem tabeli i
  ładowaniem (38 PASS). Pięć kontroli `doctor` to PASS. Kontrola pamięci GPU
  procesu jest GAP we wszystkich 38 przypadkach, bo proces skończył się po
  0,062–0,246 s, przed pierwszą próbką NVML co 0,5 s. To brak obserwacji, a
  nie zmierzone zero.
- **C4, HTTP:** odpowiedzi CAND (jeden GPU i podział `none`/`q8_0`/`q4_k`)
  są równe referencji BASE w każdym żądaniu: sekwencyjnie, przy 16 klientach
  i w ruchu równoległym do 4.5B z podziałem oraz mini na GPU 1. SIGTERM z
  dwoma długimi żądaniami w toku zakończył się kodem 0. Oba żądania dostały
  odpowiedź referencyjną, a po wyjściu na żadnym GPU nie została pamięć
  procesu.
- **C5, pojemność pod sztucznym limitem:** PASS. Przy rezerwacji, która na
  każdej z dwóch kart 48 GB zostawia 7939 MiB wolnej pamięci, 4.5B `none` na
  jednym GPU kończy się OOM bez paniki (CAND i kontrola BASE). Podział
  `0:30,1:30` pod tym samym limitem eksportuje i serwuje wyniki bitowo równe
  referencji. Ciaśniejszy limit GPU 1 daje czysty błąd etapu 1 z kontekstem.
- **C6, profile:** sekcja `stage_transfer` i `stage_transfers` są obecne i
  skończone w 6 profilach. To diagnostyka, a nie pomiar szybkości.

## Zakres i pochodzenie

| Element | Wartość |
|---|---|
| Sprzęt | 2 × NVIDIA RTX 6000 Ada Generation, 49140 MiB każda, sterownik 580.126.09, limit mocy 300 W |
| Biblioteki | cuBLAS/cuBLASLt 12.9.1.4, cudart 12.9.79, tabele GEMM cuBLASLt 120901 |
| Środowisko procesów | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `HF_HUB_OFFLINE=1`, `BASAL_NO_UPDATE_CHECK=1` |
| Okno runnera (UTC) | 10:53–11:23 |
| BASE | `d8a6a968…`, przebudowane `d42a400`, źródła `94ef2131…` (65 plików) |
| CAND | `dfd13b7e…`, `d42a400` + zmiany etapu 05, źródła `77b629ed…` (66 plików) |
| Build | ten sam builder i wspólny przepis `42fa4a4e…` (25 kroków), manifest buildu `6ae67566…` (375 plików) |
| Runner / helper | `tools/perf/run-layer-split-cloud.py` `81437a1c…`, `tools/perf/cuda-reserve.py` `ebea01ed…` |
| Modele | basal-1.5-mini `1978d070…` (32 warstwy), basal-1.5-4.5B `784a683b…` (60 warstw) |
| Wejścia | 44 + 34 (`reports/reference-basal-1.5-*-fp32`), 900 pytań (`sequential.jsonl` `81756119…`), długie stany `38f3d7f8…` |
| Surowe dane | manifest SHA-256 `2fec9dc683dda9acafd7550cbb6b06068314dcfc1b53d5c3a457c65acd9f30ac`: 525 plików, 92 katalogi, 92 472 852 B |

Pełne sumy znajdują się w `provenance.json`. Etykieta źródła BASE jest
deklaracją z rekordu buildu. Runner jej nie weryfikuje, więc pozostaje GAP
(niżej). Raport nie twierdzi, że binarki są odtwarzalne bit w bit, ani że
to te same binarki co w innych sesjach.

## C1 i C2 — równość bitowa

Kryterium: na dwóch identycznych kartach podział wykonuje te same
obliczenia, więc każdy eksport musi być bajtowo równy BASE na jednym GPU w
tym samym trybie i z tymi samymi flagami. Równość policzono ponownie z
surowych plików dla wszystkich 44 par (`facts.json` → `pairs`).

| Jednostka | Pary | Zakres | Bajtowo równe |
|---|---:|---|---:|
| C1 | 7 | mini 44 + 34: `none` tree (GPU 0 i 1), `none` single, `q8_0` i `q4_k` tree (GPU 0 i 1) | 7 |
| C1 | 7 | 4.5B 44 + 34: jak wyżej | 7 |
| C1 | 6 | 900 pytań tree, mini i 4.5B × `none`/`q8_0`/`q4_k` | 6 |
| C2 | 9 | 4.5B `none`: `0:30,1:30` tree/single/budget/single + `--state-cache-mb 512`/tree bez prefiksu, `0:1,1:59`, `0:59,1:1`, `1:30,0:30`, `0:30,1:30` z `CUDA_VISIBLE_DEVICES=1,0` | 9 |
| C2 | 8 | 4.5B `q8_0` i `q4_k`: `0:30,1:30` tree/single/tree bez prefiksu, `1:30,0:30` tree | 8 |
| C2 | 4 | mini: `0:16,1:16` i `1:16,0:16` `none`, `0:16,1:16` `q8_0` i `q4_k` | 4 |
| C2 | 3 | 900 pytań 4.5B `0:30,1:30` × `none`/`q8_0`/`q4_k` | 3 |

- `basal compare` we wszystkich 44 parach: 0 brakujących wejść, 0 niezgodności
  tokenów, maksimum każdej statystyki różnic równe 0,0, brak zmian argmax.
- W eksportach 900 pytań `systemone.jsonl` jest pusty (0 B, zbiór 900 nie
  zawiera wejść systemone). W tych 9 parach dane niesie tylko
  `bench.jsonl` (900 wierszy).
- Manifesty spełniają kontrakt porównania (20 × wariant „jeden GPU”, 24 ×
  „podział”: tekst `device`, `run.layer_split`, `backend.layer_split`,
  `stage_transfer`).
- `q4_k`: jeden obiekt kwantyzacji na model we wszystkich ładowaniach (mini
  6, 4.5B 12). Statystyki są równe faktom P0 w 18 eksportach.
- NVML pokazuje pamięć procesu na obu fizycznych GPU w każdym z 24
  eksportów podziału. Przykłady (szczyt procesu, MiB): `0:30,1:30` `none`
  5214/5596, `0:1,1:59` 894/9884, `0:59,1:1` 9918/958. Przy
  `CUDA_VISIBLE_DEVICES=1,0` ordinal procesu 0 to fizyczne GPU 1.

Zakres: dwie identyczne karty RTX 6000 Ada, te modele, wejścia i flagi.
Równość bitowa nie jest ogólną gwarancją jakości. Nie przenosimy jej na
inne architektury, mieszane karty ani inne platformy.

## C3 — odmowy i `doctor`

| Polecenie | Przypadki | Kod | Komunikat |
|---|---:|---|---|
| `serve` YAML | 14 | 1 | m.in. „a layer split needs at least two stages, got 1”, „covers 61 of 60 layers”, „CUDA device 2 does not exist”, „give either `device` or `layer_split`”, „replicas of a layer-split model are not supported”, „needs dtype f16, not bf16/f32”, nieznane i brakujące pola |
| `serve` CLI | 8 | 2 | „cannot be used with”, „invalid value” |
| `export` | 6 | 1 lub 2 | zero warstw, pokrycie 59, `candle`, `bf16`, brak urządzenia 2, `--device` z podziałem |
| `benchmark` | 4 | 1 lub 2 | duplikat, pokrycie 61, `f32`, `CUDA_VISIBLE_DEVICES=0` |
| `decide` | 3 | 1 lub 2 | jeden etap, litera w liczbie, `candle` |
| `profile` | 3 | 1 lub 2 | pokrycie 59, `--device` z podziałem, `bf16` |
| `doctor` | 5 | 1 / 0 | cztery błędy zgłoszone jako porażka; poprawny wpis z szacunkiem „weights device 0 ~4.8 GB, device 1 ~4.8 GB (estimate)” |

Wszystkie 38 odmów to PASS: oczekiwany kod i komunikat, brak komunikatu
pobierania, wyszukiwania tabeli i ładowania. Kontrola „no GPU memory of the
process” jest GAP w każdym z 38 przypadków, także przy pokryciu 61 warstw
(`serve` YAML 0,22 s, `benchmark` 0,25 s, zaokr.). Proces kończył się przed
pierwszą próbką NVML (0 próbek), więc pomiar nie stwierdza braku alokacji
GPU. Odmowę przed pobieraniem, wyszukiwaniem tabeli i ładowaniem pokazują
komunikaty tej próbki. Brak alokacji GPU na tych ścieżkach opiera się tylko
na przeglądzie kolejności walidacji w kodzie etapu 05, a nie na tym pomiarze.
Nie jest to ogólny dowód braku pamięci GPU na każdej ścieżce błędu.

## C4 — HTTP

Porównanie obejmuje status i kanoniczne ciało bez `usage.latency_ms`, dla
każdego identyfikatora żądania. Wartości policzono ponownie z rekordów żądań.
Referencje to BASE `none`/`q8_0`/`q4_k` na GPU 0 i mini na GPU 1, po 85
żądań: 79 × 200, 2 × 400, 4 × 422.

| Serwer CAND | Przebieg | Żądania | Różne | Długie | Maks. batch serwera |
|---|---|---:|---:|---:|---:|
| `none` GPU 0 | sekwencyjnie | 85 | 0 | 2 | 1 |
| `none` `0:30,1:30` | sekwencyjnie; 16 klientów × 2 rundy (długie najpierw) | 85 + 85 + 85 | 0 | 2 na przebieg | 14 |
| `none` `0:30,1:30` | SIGTERM z żądaniami w toku | 2 | 0 | 2 | 1 |
| `q8_0` `0:30,1:30` | sekwencyjnie; 16 klientów × 1 runda | 85 + 85 | 0 | 2 na przebieg | 14 |
| `q4_k` `0:30,1:30` | sekwencyjnie; 16 klientów × 1 runda | 85 + 85 | 0 | 2 na przebieg | 14 |
| 4.5B `0:30,1:30` + mini `device: 1` | 16 klientów, oba modele na przemian | 163 | 0 | 4 | 9 |

- CAND na jednym GPU: `/health`, `/v1/models` i 42 linie rodzin metryk
  (HELP/TYPE) są równe BASE. W serwerach podziału `basal_replica_info` ma
  etykietę `device="0,1"`, a mini w procesie mieszanym `device="1"`.
- W chwili SIGTERM metryka `basal_model_inflight_requests` wynosiła 1.
  Proces zakończył się kodem 0, a oba długie żądania dostały odpowiedź
  równą referencji.
- Każdy serwer zakończył się kodem 0, a po wyjściu na żadnym GPU nie
  została pamięć procesu. Nie było błędów transportu ani 5xx.
- W procesie mieszanym pominięto trzy rodzaje żądań: `basal-default-model`,
  `err-systemone-unknown-model` i błędne żądania mini. Wynika to z konstrukcji
  procesu, który serwuje dwa modele.
- Pamięć procesu (NVML, szczyt, MiB): `none` jeden GPU 9726; podział `none`
  5150/5084; `q8_0` 3198/3100; `q4_k` 2046/1948.

Zakres: skończona próbka. Nie jest dowodem sprawiedliwości kolejkowania,
ograniczenia oczekiwania toru długiego ani anulowania w każdym przypadku.
Ograniczenie oczekiwania wynika z protokołu `Gate::resume` i przeglądu
kodu. Procesy obce sprawdzano tylko przed startem serwera. Okresów żądań nie
próbkowano.

## C5 — pojemność pod sztucznym limitem wolnej pamięci

Model 4.5B `none` (f16), 44 + 34 tree. Helper `cuda-reserve.py` alokuje
rzeczywistą pamięć urządzenia, aby na każdej karcie została żądana wolna
pamięć.

| Krok | Wynik |
|---|---|
| Kalibracja (spadek wolnej pamięci `cuMemGetInfo`) | `U_single` = 10211,3125 MiB (GPU 0); `U_split` = 5635,3125 / 5667,3125 MiB (GPU 0 / 1) |
| Limit | `A = floor((10211,3125 + 5667,3125) / 2)` = 7939 MiB; okno [6691,3125; 9187,3125] spełnione |
| Rezerwacja `A` na obu GPU | wolne 8 323 268 608 B (7938 MiB, zaokr.), odchylenie −1 376 256 B (tolerancja ±64 MiB) |
| CAND jeden GPU pod `A` | kod 1, OOM przy ładowaniu, brak paniki; pamięć GPU 0 wróciła do poziomu `A` w 0,05 s |
| BASE jeden GPU pod `A` (kontrola) | kod 1, OOM, brak paniki; łańcuch bez kontekstu modelu i warstwy (obserwacja `d42a400`) |
| CAND `0:30,1:30` eksport pod `A` | kod 0, 44 + 34; bajtowo równe BASE jednego GPU i podziałowi bez limitu; kontrakt manifestu spełniony |
| CAND `0:30,1:30` serwer pod `A` | 83 żądania sekwencyjnie (`long_tokens: 0`), 0 różnych, 0 OOM; wyjście z kodem 0 |
| Ciasno: GPU 0 `A`, GPU 1 `A1` = 2833 MiB | kod 1, OOM na etapie 1 i urządzeniu 1, brak paniki; oba GPU wróciły do poziomów rezerwacji w 0,01 s |
| Helpery | monitor i obie rezerwacje: kod 0, 0 błędów zwolnienia |

Łańcuchy błędów (stderr, dosłownie):

```text
CAND jeden GPU:  error: loading basal-1.5-4.5B
                   caused by: stage 0 (layers 0–59)
                   caused by: layer 50 of stage 0 on CUDA device 0
                   caused by: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
BASE jeden GPU:  error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
CAND ciasno:     error: loading basal-1.5-4.5B with layer split 0:30,1:30
                   caused by: stage 1 (layers 30–59)
                   caused by: layer 46 of stage 1 on CUDA device 1
                   caused by: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

Źródła pamięci są różne i ich nie mieszamy:

- `U_*`, `A` i poziomy rezerwacji to wolna pamięć urządzenia z
  `cuMemGetInfo`. Helper próbkuje ją co 50 ms i obejmuje ona wszystkie
  procesy na karcie.
- Szczyt pamięci procesu pochodzi z NVML (`used_memory` drzewa procesu,
  próbki co 0,5 s). Przykładowo CAND jeden GPU pod `A`: 7830 MiB przed OOM;
  podział pod `A`: 5630/5662 MiB.
- Rozmiarów pul alokatora nie ma w danych L1.

Jedynymi obcymi procesami obliczeniowymi widocznymi w 6 krokach C5 były
zarejestrowane procesy helpera. W żadnym poleceniu nie zauważono innego
procesu.

Zakres twierdzenia: bez rezerwacji 4.5B mieści się na jednej karcie 48 GB.
Pomiar pokazuje tylko, że pod jawnym, sztucznym limitem wolnej pamięci na
kartę jeden GPU nie uruchamia modelu z tymi wejściami, a podział tak. To nie
jest pomiar RTX 4090, modelu max, fizycznego braku pamięci karty 48 GB ani
dużych partii. Okno dobrano dla 44 + 34. Przy 900 pytaniach szczyt NVML
podziału `none` wynosił 9086/9150 MiB (GPU 0/1). Raport nie ustala fizycznej
trasy kopii (P2P ani przez hosta).

## C6 — profile `stage_transfer` (diagnostyka)

`basal profile` 4.5B `0:30,1:30`, `--n 40`, średnio 239,8 tokenu na
element. Profil synchronizuje urządzenia, więc te czasy nie są opóźnieniem
żądania ani szybkością. Czas obejmuje alokację i inicjalizację celu,
opróżnienie, kopię API urządzenie–urządzenie, potwierdzenie na obu GPU,
zwolnienie źródła i wiązanie kontekstu. Fizyczna trasa nie jest ustalona.

| Profil | Kopie 0→1 | Bajty | `transfer_ms` (suma) | `gate_wait_ms` | Sekcja `stage_transfer` ms/element (zaokr.) |
|---|---:|---:|---:|---:|---:|
| `none` b1 | 40 | 59 187 200 | 5,825905 | 0,021748 | 0,158 |
| `none` b16 | 6 | 59 187 200 | 3,759749 | 0,008160 | 0,378 |
| `q8_0` b1 | 40 | 59 187 200 | 32,821935 | 0,030605 | 0,837 |
| `q8_0` b16 | 6 | 59 187 200 | 7,960451 | 0,007739 | 0,499 |
| `q4_k` b1 | 40 | 59 187 200 | 42,218416 | 0,041383 | 1,449 |
| `q4_k` b16 | 6 | 59 187 200 | 10,287879 | 0,008916 | 0,659 |

Wydajność (latency, throughput, `load_s` w parach, pule) mierzą L2 i L3.
Ten raport niczego o niej nie twierdzi.

## GAP (42, bez zmiany nazw)

| Grupa | Liczba | Treść runnera |
|---|---:|---|
| Etykieta źródła BASE | 1 | „base binary: declared source label” — deklaracja z rekordu buildu, niezweryfikowana przez runner |
| Krótkie niepoprawne polecenia | 38 | „invalid …: no GPU memory of the process” — „ended after 0.06–0.25 s, before a 0.5 s placement sample (the log shows no load)” |
| Nieobserwowalne na tej parze | 3 | mieszane typy GPU, podział na Metal, podział na Intel — „not observable on this pair of identical CUDA cards; static review and the CI compile checks only” |

Pełna lista z numerami kontroli jest w `facts.json` → `gaps.items`. Te GAP
zostają GAP-ami. Późniejszy wynik ich nie usuwa.

## Ograniczenia

- Jedna sesja, jedna maszyna, dwie identyczne karty RTX 6000 Ada 48 GB.
  Nie wnioskujemy o innych kartach, mieszanych parach, Metal ani Intel.
- Wcześniejsze nieudane lub niewykorzystane sesje L1 nie są dowodem tego
  raportu.
- Brak twierdzeń o szybkości, narzucie podziału ani o fizycznej trasie
  transferu.
- Odmowy C3 nie mają obserwacji pamięci GPU (GAP). C5 dotyczy sztucznego
  limitu i jednego zestawu wejść.

## Polecenia (ilustracyjne)

Rzeczywiste argv, ścieżki i środowisko zostają w prywatnych surowych
danych. `<candidate>`, `<base>`, `<models>`, `<out>` i `<table>` to
symbole zastępcze.

```sh
# C2: eksport z podziałem (CAND) i referencja jednego GPU (BASE)
<candidate> export --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --batching tree --layer-split 0:30,1:30 --inputs reports/reference-basal-1.5-4.5B-fp32 --out <out>/split
<base> export --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --batching tree --device 0 --inputs reports/reference-basal-1.5-4.5B-fp32 --out <out>/base
cmp <out>/split/bench.jsonl <out>/base/bench.jsonl
<candidate> compare <out>/base <out>/split

# C6: profil (diagnostyka)
<candidate> profile --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --layer-split 0:30,1:30 --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16 --n 40 --batch 16 --quantization q8_0
```

Serwer z podziałem (fragment YAML): `layer_split: [{device: 0, layers: 30},
{device: 1, layers: 30}]`, `dtype: f16`, `long_tokens: 3000`,
`state_cache_mb: 512`, `metrics: true`.

## Pliki

- `facts.json` — statusy i liczności (z kontroli i `checks.txt`), 44 pary z
  sumami SHA-256 i ponownie policzoną równością bajtową, eksporty (kod,
  `load_s`, czasy manifestu, NVML), C3 per przypadek, C4 per przebieg, C5
  (kroki, reguła okna policzona ponownie z szeregu monitora, łańcuchy
  błędów), C6, lista GAP. Liczby mają pełne bity, a zaokrąglenia
  występują tylko w README.
- `provenance.json` — sumy binarek, źródeł, runnera i helperów, modeli,
  tabel GEMM, wejść i referencji FP32, środowisko (karta, sterownik,
  biblioteki), manifest surowych danych.
- `validation.json` — kontrole autora: wyprowadzone fakty wobec surowych
  danych, bieżące sumy źródeł i wejść, brak danych identyfikujących maszynę
  i danych kosztowych.
