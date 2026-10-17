# Jeden model na kilku GPU (podział warstw) — zestawienie, 2026-10-13

Raport danych. Zestawia trzy sesje pomiarowe podziału warstw (L1, L2, L3) na
jednej maszynie z dwiema NVIDIA RTX 6000 Ada 48 GB. Pełne tabele, pary i
GAP-y są w raportach sesji:

- [L1](../multigpu-layer-l1-2026-10-13/README.md): zgodność, odmowy, HTTP,
  pojemność i profile transferów (poprawka poleceń ilustracyjnych:
  [COMMANDS.md](../multigpu-layer-l1-2026-10-13/COMMANDS.md));
- [L2](../multigpu-layer-l2-2026-10-13/README.md): regresja toru jednego GPU
  `none` i opisowy kompromis podziału `none`;
- [L3](../multigpu-layer-l3-2026-10-13/README.md): regresja toru jednego GPU
  `q8_0` i `q4_k` i opisowy podział w tych trybach.

## Co daje funkcja

Wcześniejsze etapy pozwalają uruchomić różne modele na różnych kartach
(`device`, [raport](../multigpu-placement-2026-10-11/README.md)), repliki
jednego modelu (`replica_devices`,
[raport](../multigpu-replicas-2026-10-11/README.md)) i opcjonalną
kwantyzację wag projekcji (`q8_0`,
[raport](../w8a16-2026-10-11/README.md); eksperymentalne `q4_k`,
[raport](../w4a16-s1-2026-10-12/README.md)). Ten etap dodaje podział jednego
modelu warstwami między karty jednego procesu:

```yaml
models:
  - path: .models/basal-1.5-4.5B
    dtype: f16
    layer_split:
      - {device: 0, layers: 30}   # warstwy 0–29 i embedding
      - {device: 1, layers: 30}   # warstwy 30–59, norma, lm_head, głowica evidence
```

albo `--layer-split 0:30,1:30` w `serve --model`, `export`, `decide`,
`bench`, `bench-requests`, `benchmark`, `profile` i `eval-*`.

- Tylko CUDA, `dtype: f16` i kernele `fused`, karty jednego typu (nazwa,
  compute capability, liczba SM). Co najmniej dwa etapy na różnych kartach,
  każdy z co najmniej jedną warstwą; kolejne zakresy w kolejności warstw, a
  suma równa liczbie warstw modelu (mini 32, 4.5B 60, max 50).
- Numery kart to ordinale procesu po `CUDA_VISIBLE_DEVICES`
  (z `CUDA_VISIBLE_DEVICES=1,0` etap `0:30` trafia na fizyczną kartę 1).
- `layer_split` wyklucza `device` (także `device: 0`) i `replica_devices`;
  `--layer-split` wyklucza `--device`, `--replica-devices` i `--config`.
  Błędy są zgłaszane przed pobraniem modelu, a pokrycie warstw po odczycie
  `config.json`, przed wagami. `basal doctor` sprawdza te reguły i szacuje
  wagi każdego etapu z arytmetyki formatu.
- Etapy liczą po kolei te same kernele i GEMM co jeden GPU; granica kopiuje
  tylko aktywacje. To nie jest równoległość tensorowa. Podział służy
  pojemności: wagi i cache K/V warstwy leżą na karcie jej etapu.
- `quantization` dotyczy każdego etapu. Domyślne jest `none`, `q8_0` jest
  opcjonalne, `q4_k` eksperymentalne; ich różnice jakości wobec FP32 opisują
  raporty kwantyzacji i podział ich nie zmienia.
- W serwerze model z podziałem to jedna instancja bez replik, z etykietą
  `device="0,1"` w `basal_replica_info`. Liczniki transferów są w
  `basal profile` i `basal benchmark`, nie w Prometheusie.

Opis użytkownika: [README](../../README.md), szczegóły:
[Architektura](../../docs/ARCHITECTURE.md#podział-warstw-między-gpu-cuda),
[Observability](../../docs/OBSERVABILITY.md#podział-warstw),
[Pomiary](../../docs/BENCHMARKS.md#podział-warstw).

## Wyniki w skrócie

| Sesja | Kontrole (PASS / FAIL / GAP / SIGNAL) | Wynik |
|---|---|---|
| L1 | 599 / 0 / 42 / 0 | 44 pary eksportów bajtowo równe (20 toru jednego GPU wobec bazy, 24 podziału wobec jednego GPU); odpowiedzi HTTP równe referencji; 38 odmów poprawnych; pojemność pod sztucznym limitem PASS |
| L2 | 123 / 0 / 1 / 0 | `none`, jeden GPU: brak powtarzalnego spowolnienia przy rozdzielczości pomiaru; podział `none` opisowo |
| L3 | 154 / 0 / 2 / 0 | `q8_0` i `q4_k`, jeden GPU: brak powtarzalnego spowolnienia przy rozdzielczości `basal bench`; podział opisowo |

Każda sesja: `complete`, bez przerwania, kod wyjścia runnera 3 (brak FAIL i
SIGNAL, co najmniej jeden GAP).

### Zgodność (L1)

Na dwóch identycznych kartach podział daje te same bajty co jeden GPU:
eksporty mini i 4.5B w `none`, `q8_0` i `q4_k`, w tym układy `0:30,1:30`,
`0:1,1:59`, `0:59,1:1`, `1:30,0:30`, `CUDA_VISIBLE_DEVICES=1,0` i 900 pytań.
`basal compare` pokazuje różnicę 0,0 we wszystkich 44 parach. Odpowiedzi HTTP
z podziałem są równe referencji jednego GPU (sekwencyjnie, 16 klientów,
SIGTERM z żądaniami w toku, ruch równoległy z mini na drugiej karcie). To
równość arytmetyki na tej parze kart, a nie ogólna gwarancja jakości:
jakość `q8_0` i `q4_k` wobec FP32 pozostaje jak w raportach kwantyzacji.

38 niepoprawnych konfiguracji kończy się błędem z oczekiwanym komunikatem
przed pobieraniem i ładowaniem. Brak alokacji GPU w tych procesach nie jest
zmierzony (GAP: proces kończył się przed pierwszą próbką NVML).

### Pojemność (L1, C5)

Bez limitu 4.5B mieści się na jednej karcie 48 GB, więc pojemność
sprawdzono pod jawnym, sztucznym limitem. Rezerwacja zostawiała na każdej
karcie 7939 MiB wolnej pamięci (`U_single` 10211,3125 MiB, `U_split`
5635,3125 / 5667,3125 MiB). Pod tym limitem:

- 4.5B `none` na jednym GPU kończy się czystym OOM, bez paniki;
- `0:30,1:30` eksportuje i serwuje wyniki bitowo równe referencji bez limitu;
- ciaśniejszy limit karty 1 daje czysty OOM etapu 1 z kontekstem etapu,
  karty i warstwy.

To nie jest pomiar fizycznie mniejszej karty (np. RTX 4090), modelu max ani
dużych partii.

### Wydajność (L2, L3)

Model 4.5B, `basal bench` przez `tools/bench/ab.py` (4 pary, 39 pytań) i
`basal benchmark` (HTTP w jednym procesie, 1 i 16 klientów). Regresja toru
jednego GPU (A = baza, B = kandydat) używa reguły bez progu: pole z B
gorszym we wszystkich parach dostaje dokładnie jedną rundę potwierdzenia,
a utrzymany sygnał wymaga B gorszego we wszystkich parach obu rund.

| Tryb | Sygnały rundy głównej | Potwierdzenie | Utrzymane |
|---|---|---|---:|
| `none` (L2) | `dec_s` (B/A 0,9977–0,9998), `mixed-c16` p99 kolejki i latencji | po jednej rundzie; B gorsze w 6 z 8 par (`dec_s`) i 3 z 4 (`benchmark`) | 0 |
| `q8_0` (L3) | brak (`dec_s` B/A 0,9976–1,0004, B gorsze w 3 z 4 par) | niepotrzebne | 0 |
| `q4_k` (L3) | `dec_s` (B/A 0,9984–0,9999) | jedna runda i profile A/B batch 1; `dec_s` B/A 0,9982–1,0037, B gorsze w 2 z 4 par | 0 |

Wniosek każdego trybu: brak powtarzalnego spowolnienia przy zmierzonej
rozdzielczości. To nie dowód braku narzutu: w `none` większość par ma B
gorsze. W `q8_0` i `q4_k` dowodem są tylko pary `basal bench`; HTTP jednego
GPU nowego kandydata w tych trybach nie był mierzony (GAP L3).

Podział wobec jednego GPU tego samego kandydata, opisowo, bez reguły:

| Tryb | `basal bench`: mediana stosunku latencji pytania (zakres, szybsze z 39) | `dec_s` podział / jeden GPU | HTTP |
|---|---|---|---|
| `none` (L2) | 1,0061 (1,0001–1,0087; 0) | 0,9426–0,9459 | 1 klient: przepustowość −2,75…−2,83%, latencja p50 +1,6…4,0%; 16 klientów mieszanego ruchu: przepustowość 1,1804 i 1,1823 |
| `q8_0` (L3) | 1,1359 (1,0576–1,1791; 0) | 0,9121–0,9243 | jeden przebieg z podziałem, bez pary jednego GPU |
| `q4_k` (L3) | 1,1798 (1,1132–1,2258; 0) | 0,9151–0,9192 | jeden przebieg z podziałem, bez pary jednego GPU |

- Pojedyncze żądanie jest z podziałem wolniejsze; w trybach kwantyzowanych
  różnica w `basal bench` jest większa niż w `none`. Pomiar nie ustala
  przyczyny.
- Wyższa przepustowość `none` przy 16 klientach to wynik jednej sesji. W
  przebiegach jednego GPU karta 0 miała 84–85 °C, a zegar SM spadał do
  1005–1050 MHz przy żądanych 1200 MHz; z podziałem karty miały 71–74 °C.
  Pomiar nie ustala mechanizmu i nie jest obietnicą przyspieszenia.
- Pula alokatora po ładowaniu 4.5B `none`: jeden GPU 9197,4 MiB, podział
  4610,7 / 4618,7 MiB (L2).

## GAP

Wszystkie GAP-y pozostają GAP-ami; późniejszy wynik ich nie usuwa.

| Sesja | GAP | Treść |
|---|---:|---|
| L1, L2, L3 | po 1 | etykieta źródła binarki bazowej jest deklaracją z rekordu buildu, runner jej nie weryfikuje |
| L1 | 38 | brak próbki pamięci GPU w krótkich procesach odmów |
| L1 | 3 | karty różnych typów, podział na Metal i Intel: nieobserwowalne na tej parze |
| L3 | 1 | HTTP (`basal benchmark`, 1 i 16 klientów) jednego GPU nowego kandydata w `q8_0` i `q4_k`: niemierzone z planu |

## Zakres i pochodzenie

| Element | Wartość |
|---|---|
| Sprzęt | 2 × NVIDIA RTX 6000 Ada Generation, 49140 MiB każda, sterownik 580.126.09, limit mocy 300 W |
| Biblioteki | cuBLAS/cuBLASLt 12.9.1.4, cudart 12.9.79, tabele GEMM cuBLASLt 120901 |
| Baza | binarka `d8a6a968…`, przebudowane `d42a400`, źródła `94ef2131…` (65 plików) |
| Kandydat | binarka `dfd13b7e…`, `d42a400` + zmiany etapu, źródła `77b629ed…` (66 plików) |
| Build | wspólny manifest buildu `6ae67566…`; te same binarki we wszystkich trzech sesjach (sumy sprawdzone przez runner) |
| Runner | `tools/perf/run-layer-split-cloud.py` `81437a1c…`, `tools/bench/ab.py` `17b188dc…`, `tools/perf/cuda-reserve.py` `ebea01ed…` |
| Modele | basal-1.5-4.5B `784a683b…`; w L1 także basal-1.5-mini `1978d070…` |
| Surowe dane (manifest SHA-256) | L1 `2fec9dc6…` (525 plików), L2 `fb0dd50f…` (105 plików), L3 `138a9487…` (119 plików) |

Raport nie twierdzi, że binarki są odtwarzalne bit w bit.

## Ograniczenia

- Jedna maszyna, dwie identyczne karty RTX 6000 Ada 48 GB, po jednej sesji.
  Nie zmierzono modelu max, innych kart (np. RTX 4090), kart mieszanych,
  Metal, Intel, fizycznego braku pamięci ani fizycznej trasy kopii (P2P albo
  przez hosta).
- Wyniki wydajności dotyczą 4.5B i tych obciążeń. Nie są ogólną obietnicą
  szybkości ani narzutu.
- Wydajność toru jednego GPU w `q8_0` i `q4_k` przez HTTP nie jest
  zmierzona; L3 nie daje też nowej oceny jakości wobec FP32.
