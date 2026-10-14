# Podział warstw — zestawienie pomiarów bieżących źródeł, 2026-10-14

> **Dane JSON w archiwum.** Pliki `facts.json`, `provenance.json` i `validation.json` sesji L1, L2 i L3 z tabeli
> „Pliki (SHA-256)” są poza Gitem, w archiwum pomiarów `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod tymi samymi
> ścieżkami w `reports/`. Sumy plików tych sesji w tabeli dotyczą wersji z chwili zestawienia. Dokładnie te wersje,
> także plików Markdown, są w archiwum. Pliki Markdown sesji mają w Gicie później dopisaną notę o archiwum, więc ich
> bieżące sumy są inne. Dwóch ostatnich plików tabeli (`multigpu-layer-2026-10-13/README.md` i `SCOPE.md`) nie ma w
> archiwum; są w Gicie bez zmian. Opis końca zegara `load_s` w tym pliku poprawia [erratum `load_s`](LOAD_S.md).
> Pobieranie i sumy kontrolne: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Archiwum zawiera też
> wersję tego pliku sprzed dopisania tej noty. Przeniesienie danych nie jest nowym pomiarem: liczby, statusy i
> rozstrzygnięcia raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

Stan: zestawienie trzech zakończonych sesji pomiarowych (L1, L2, L3)
bieżącego kandydata etapu 05, źródła `831e68e0…` (66 plików). To nie jest
nowy pomiar, akceptacja etapu 05 ani końcowe review. Niezależne końcowe
review, commit i PR są jeszcze przed nami. Pełne tabele, pary, GAP-y i
analizy są w raportach sesji:

- [L1](../multigpu-layer-l1-revalidation-2026-10-13/README.md): zgodność,
  odmowy, HTTP, pojemność i profile transferów; poprawka poleceń
  ilustracyjnych: [COMMANDS.md](../multigpu-layer-l1-revalidation-2026-10-13/COMMANDS.md);
- [L2](../multigpu-layer-l2-revalidation-2026-10-14/README.md): regresja
  toru jednego GPU `none` i opisowy kompromis podziału `none`; decyzja
  użytkownika: [DECISION.md](../multigpu-layer-l2-revalidation-2026-10-14/DECISION.md);
- [L3](../multigpu-layer-l3-revalidation-2026-10-14/README.md): regresja
  toru jednego GPU `q8_0` i `q4_k` i opisowy podział w tych trybach;
  rozstrzygnięcie koordynatora: [DECISION.md](../multigpu-layer-l3-revalidation-2026-10-14/DECISION.md).

[Wcześniejsze zestawienie](../multigpu-layer-2026-10-13/README.md) z
[poprawką zakresu](../multigpu-layer-2026-10-13/SCOPE.md) dotyczy
poprzedniego kandydata (źródła `77b629ed…`, binarka `dfd13b7e…`). Pozostaje
bez zmian jako historia i nie jest dowodem tego zestawienia. Pomiary
powtórzono, bo po nich doszła poprawka sprawdzania stanu urządzeń po
końcowym review R1 i scalenie `main` (`4f68eb2`).

## Co zmierzono

| Element | Wartość |
|---|---|
| Kandydat (CAND) | binarka `169aedfc…`, `d42a400` + zmiany etapu 05 + poprawka po review R1 + scalenie `4f68eb2`, źródła `831e68e0…` (66 plików) |
| Baza (BASE) | binarka `44c8e399…`, przebudowane `d42a400`, źródła `94ef2131…` (65 plików); etykieta źródła to deklaracja z rekordu buildu (GAP w każdej sesji) |
| Build | ten sam builder i wspólny przepis `2e37cb45…` (25 kroków), manifest pary `2efaeca1…` (390 plików); te same binarki we wszystkich trzech sesjach; bez twierdzenia o odtwarzalności bit w bit |
| Runner | `tools/perf/run-layer-split-cloud.py` `81437a1c…`, `tools/bench/ab.py` `17b188dc…`, `tools/perf/cuda-reserve.py` `ebea01ed…` |
| Sprzęt każdej sesji | 2 × NVIDIA RTX 6000 Ada Generation 48 GB, sterownik 580.126.09, limit mocy 300 W, żądany zegar SM 1200 MHz; cuBLAS/cuBLASLt 12.9.1.4, tabele GEMM cuBLASLt 120901 |
| Sesje (UTC) | L1 2026-10-13 20:11–20:44, L2 2026-10-13 20:57–22:08, L3 2026-10-14 05:06–05:51 |
| Modele | basal-1.5-4.5B `784a683b…` (60 warstw); w L1 także basal-1.5-mini `1978d070…` |

Każda sesja to osobna wynajęta maszyna o tej samej konfiguracji dwóch kart.
Dane nie zawierają identyfikatorów fizycznych kart ani maszyn, więc nie
ustalają, czy sesje działały na tych samych kartach. Pary A/B i podział
wobec jednego GPU porównujemy w obrębie jednej sesji. Wyniki różnych sesji
zestawiamy tylko opisowo.

## Wyniki w skrócie

| Sesja | PASS / FAIL / GAP / SIGNAL | Kod | Wynik |
|---|---|---:|---|
| L1 | 599 / 0 / 42 / 0 | 3 | 44 pary eksportów bajtowo równe; 928 odpowiedzi HTTP CAND równych referencji; 38 odmów poprawnych; pojemność pod sztucznym limitem PASS |
| L2 | 107 / 0 / 1 / 1 | 4 | `none`, jeden GPU: utrzymany SIGNAL `basal benchmark` `mixed-c1 queue_ms.p99`, przyjęty przez użytkownika |
| L3 | 172 / 0 / 2 / 1 | 4 | `q8_0` bez utrzymanego sygnału; `q4_k`: utrzymany SIGNAL `bench: dec_s`, rozstrzygnięty przez koordynatora w granicy podanej wcześniej przez użytkownika |

Każda sesja: `complete: true`, `stopped: null`. Koordynator przeliczył dane
każdego raportu z surowych plików bez błędów.

### Zgodność, odmowy i pojemność (L1)

- 44 pary eksportów są bajtowo równe (`bench.jsonl`, `systemone.jsonl`): 20
  par toru jednego GPU CAND wobec BASE (mini i 4.5B; `none`, `q8_0`, `q4_k`;
  GPU 0 i 1; 900 pytań) i 24 pary podziału wobec BASE na jednym GPU (m.in.
  `0:30,1:30`, `0:1,1:59`, `0:59,1:1`, `1:30,0:30`,
  `CUDA_VISIBLE_DEVICES=1,0`). `basal compare --a … --b … --out …`
  wywołano 44 razy, zawsze z kodem 0 i różnicą 0,0.
- HTTP: 1268 rekordów żądań ma spójne sumy odpowiedzi. 928 odpowiedzi CAND
  (jeden GPU, podział `none`/`q8_0`/`q4_k`, 16 klientów, SIGTERM z
  żądaniami w toku, ruch równoległy z mini na GPU 1, serwer pod limitem C5)
  porównano z referencją BASE: 0 różnic.
- 38 niepoprawnych konfiguracji kończy się oczekiwanym błędem przed
  pobraniem i ładowaniem. Brak alokacji GPU w tych procesach nie jest
  zmierzony (GAP: 0 próbek NVML).
- C5: rezerwacja zostawia 7939 MiB wolnej pamięci na każdej karcie 48 GB.
  4.5B `none` na jednym GPU kończy się czystym OOM (CAND i BASE), podział
  `0:30,1:30` eksportuje i serwuje wyniki bitowo równe referencji, a
  ciaśniejszy limit karty 1 daje OOM etapu 1 z kontekstem etapu, karty i
  warstwy. To sztuczny limit, a nie pomiar fizycznie mniejszej karty (np.
  RTX 4090), modelu max, dużych partii ani maksymalnej pojemności.
- Wszystkie serwery były zdrowe. Dane pokazują tylko zdrową ścieżkę
  poprawionego `/health`; utraty urządzenia nie wywoływano (bez
  wstrzykiwania błędów).

### Regresja toru jednego GPU wobec `d42a400` (L2, L3)

Reguła bez progu: pole z B gorszym we wszystkich parach rundy głównej
dostaje jedną rundę potwierdzenia. Utrzymany SIGNAL wymaga B gorszego we
wszystkich parach obu rund.

| Tryb | Narzędzie | Sygnały rundy głównej | Po potwierdzeniu | Utrzymany SIGNAL |
|---|---|---|---|---|
| `none` (L2) | ab.py (`basal bench`) | brak | — | brak |
| `none` (L2) | `basal benchmark` | 9 z 21 pól | 8 pól ma parę z B niegorszym (w wielu B gorsze w 3 z 4 par) | `mixed-c1 queue_ms.p99`: +1,237, +2,706, +11,330, +3,896 ms (+1,4565% do +12,7872%) |
| `q8_0` (L3) | ab.py | `dec_s` (4 z 4) | B gorsze w 3 z 4 (7 z 8 łącznie) | brak |
| `q4_k` (L3) | ab.py | `dec_s`, `lat2_ms` (4 z 4) | `dec_s` 4 z 4; `lat2_ms` 3 z 4 (7 z 8) | `dec_s`: B/A 0,994438–0,999895 (−0,0105% do −0,5562%), 8 z 8 par |

- W obu przypadkach analiza surowych danych i przegląd kodu tylko do
  odczytu nie wskazały przyczyny w kodzie. Nie znaczy to, że przyczyny nie
  ma. Analizy nie wykluczają wpływu całej binarki, CPU hosta i szeregowania
  wątków. Profilowania CPU hosta nie wykonano, a telemetria dotyczy tylko
  GPU.
- `mixed-c1 queue_ms.p99` to w każdym przebiegu najwolniejsze z 18 żądań
  `doc-4k-8k` o największym wejściu, w oknie przed obliczeniami. Latencja i
  przepustowość `mixed-c1` nie mają utrzymanego sygnału.
- `q4_k` `dec_s`: przebieg 44 pytań trwa w B dłużej o +0,082 do +4,366 ms
  (mediana par +0,878 ms) przy 778,6–785,0 ms. To czas `basal bench`, a nie
  latencja HTTP.
- Nieutrzymane pola nie dowodzą braku narzutu: w wielu z nich B jest
  gorsze w większości par.
- HTTP jednego GPU nowego kandydata w `q8_0` i `q4_k` nie był mierzony
  (GAP L3). Dla tych trybów dowodem są tylko pary ab.py.

### Kompromis podziału `0:30,1:30` (opisowo, bez reguły)

C = CAND z podziałem, B = CAND na GPU 0, ta sama sesja i ten sam tryb.
Stosunki C/B per para.

| Tryb | `basal bench`: mediana C/B per pytanie | `basal bench`: `dec_s` C/B | HTTP |
|---|---|---|---|
| `none` (L2) | 1,00384–1,00889 (C szybsze w 0–3 z 39) | 0,9415–0,9445 | 1 klient (`sequential-c1`, `mixed-c1`): przepustowość 0,9626–0,9732; 16 klientów (`mixed-c16`): przepustowość 1,1801 i 1,1740, latencja p95 0,7478 i 0,7398 |
| `q8_0` (L3) | 1,12804–1,13927 (0 z 39) | 0,9160–0,9229 | jeden przebieg z podziałem, bez pary z jednym GPU |
| `q4_k` (L3) | 1,18348–1,20554 (0 z 39) | 0,9152–0,9173 | jeden przebieg z podziałem, bez pary z jednym GPU |

- Pojedyncze pytanie i przebieg `basal bench` są z podziałem wolniejsze.
  Podział służy pojemności, a nie przyspieszeniu.
- Wyższa przepustowość `none` przy 16 klientach to wynik jednej sesji. W
  przebiegach jednego GPU karta 0 miała 80–81 °C, a zegar SM spadał do
  1035–1095 MHz; z podziałem karty miały 70 i 74 °C. Pomiar nie ustala
  mechanizmu i nie jest obietnicą przyspieszenia.
- HTTP z podziałem w `q8_0` i `q4_k`: po jednym przebiegu na tryb, w każdej
  z 3 faz 900 ukończonych żądań, 0 błędów i 0 zmian odpowiedzi.
- Profile `basal profile` i dwie grupy wartości sondy GEMM (ok. 104 i 121
  TFLOPS) w L3 to diagnostyka bez ustalonej przyczyny. Nie wynika z nich
  przyspieszenie ani spowolnienie.

### Pamięć

Pula alokatora strumieniowego CUDA procesu po ładowaniu 4.5B (`used`, MiB),
z `basal benchmark`. Nie obejmuje kontekstu CUDA, modułów, uchwytów
bibliotek ani alokacji synchronicznych.

| Tryb | Jeden GPU | Podział: GPU 0 / GPU 1 | Sesja |
|---|---:|---|---|
| `none` | 9197,4 | 4610,7 / 4618,7 | L2 |
| `q8_0` | niezmierzone | 2543,6 / 2551,6 | L3 |
| `q4_k` | niezmierzone | 1441,1 / 1449,1 | L3 |

NVML, wolna pamięć urządzenia i szacunki wag mają inne zakresy. Są w
raportach sesji i nie mieszamy ich z pulą.

## Rozstrzygnięcia utrzymanych SIGNAL

| Sesja | Wynik | Kto | Plik |
|---|---|---|---|
| L2 | `none` `mixed-c1 queue_ms.p99` | użytkownik przyjął ten konkretny wynik i dopuścił L3 | [DECISION.md](../multigpu-layer-l2-revalidation-2026-10-14/DECISION.md) |
| L3 | `q4_k` `bench: dec_s` | koordynator, na podstawie wcześniejszej instrukcji użytkownika, że spadek do 1–2% mieści się w normie; użytkownik nie oceniał tego wyniku | [DECISION.md](../multigpu-layer-l3-revalidation-2026-10-14/DECISION.md) |

Oba rozstrzygnięcia dotyczą tylko swojego wyniku i tej pary binarek. Nie
zmieniają statusów runnera, nie ustanawiają progu, nie stwierdzają szumu ani
braku regresji i nie akceptują etapu 05. Decyzja L2 nie przenosi się na L3.
Akapity „decyzja oczekująca” w raportach L2 i L3 są migawką sprzed
rozstrzygnięć.

## Zakres twierdzeń

- Tryby: domyślny i zalecany `none`; `q8_0` opcjonalny, `q4_k` opcjonalny i
  eksperymentalny, oba z jawnymi różnicami jakości wobec FP32. Sesje L1–L3
  nie mierzą nowej jakości wobec FP32. Równość bitowa L1 dotyczy podziału
  wobec jednego GPU w tym samym trybie.
- Alokacje: zdanie raportu L3, że CAND na ścieżce `dec_s` nie dodaje
  alokacji, dotyczy pamięci GPU. Po stronie hosta każde wywołanie
  `letter_logits` buduje dwa wektory właścicieli bramek i klonuje uchwyty
  urządzenia i stanu zdrowia. Ich czasu nie mierzono.
- `load_s`: w CAND zegar zatrzymuje się przed wczytaniem głowicy evidence,
  utworzeniem uchwytu cuBLASLt każdego etapu i synchronizacją publikacji, a
  w `d42a400` po nich. `load_s` BASE i CAND nie są porównaniem czasu
  uruchomienia.
- Nie zmierzono: modelu max, innych kart (np. RTX 4090), kart mieszanych,
  podziału na Metal i Intel (GAP L1), fizycznego braku pamięci ani fizycznej
  trasy kopii (P2P albo przez hosta).
- Wyniki wydajności dotyczą 4.5B, tych obciążeń i tych sesji. Nie są ogólną
  obietnicą szybkości, braku narzutu ani przyspieszenia.
- Kontrole statyczne: według audytu koordynatora 19 lokalnych kontroli dla
  źródeł `831e68e0…` (m.in. `cargo fmt`, Clippy i rustdoc dla buildów
  domyślnego, Intel i CUDA, `cargo-deny`) zakończyło się kodem 0. CI, w tym
  build Metal, zostanie uruchomione dopiero w PR.

## GAP i SIGNAL (bez zmiany nazw)

| Sesja | Status | Liczba | Treść |
|---|---|---:|---|
| L1, L2, L3 | GAP | po 1 | etykieta źródła BASE to deklaracja z rekordu buildu, runner jej nie weryfikuje |
| L1 | GAP | 38 | brak próbki pamięci GPU w krótkich procesach odmów |
| L1 | GAP | 3 | karty różnych typów, podział na Metal i Intel: nieobserwowalne na tej parze |
| L3 | GAP | 1 | HTTP jednego GPU CAND w `q8_0` i `q4_k`: niemierzone z planu |
| L2 | SIGNAL | 1 | `benchmark: mixed-c1 queue_ms.p99` |
| L3 | SIGNAL | 1 | `q4_k` `bench: dec_s` |

GAP pozostaje GAP-em, a SIGNAL SIGNAL-em. Rozstrzygnięcia ich nie usuwają.

## Pliki (SHA-256)

Sumy w chwili zestawienia. Zestawienie nie zmienia raportów sesji.

| Plik | SHA-256 |
|---|---|
| `../multigpu-layer-l1-revalidation-2026-10-13/README.md` | `e40283b01988c690062fe57b6c28d940df344fd4c426b9dd5e1a61e1c8f29214` |
| `../multigpu-layer-l1-revalidation-2026-10-13/COMMANDS.md` | `31ed046e98a36f39b750cdc4cbe57fc2a670bc6c96154c2d427ca5f5447a6770` |
| `../multigpu-layer-l1-revalidation-2026-10-13/facts.json` | `68008e9314a046c99197a5d76d56f7afd97d400c0bd073ca4d5884c39eecf5e2` |
| `../multigpu-layer-l1-revalidation-2026-10-13/provenance.json` | `7178a6e0e12e451213443a10e2fff0c3f9f61ae5fdc70f8092e57d38a58853c9` |
| `../multigpu-layer-l1-revalidation-2026-10-13/validation.json` | `93ede71fb55f06348627c55fcba13c5416c9b8980d71c089199047540d843cfa` |
| `../multigpu-layer-l2-revalidation-2026-10-14/README.md` | `9d83e57a24d146d65d87e856427d45e16f3722b47018d8cd7df62f41a59098f4` |
| `../multigpu-layer-l2-revalidation-2026-10-14/DECISION.md` | `0e87e6da8c0015711838a49e64a696068aa15e3be631301513d6aabb88d3be8c` |
| `../multigpu-layer-l2-revalidation-2026-10-14/facts.json` | `fffe3bb796f4fbd60d8f7a154b9e72e6375275a2ec4d469eda5a151e99736933` |
| `../multigpu-layer-l2-revalidation-2026-10-14/provenance.json` | `fcecef81f2f406453c9b0e7b0647eaa6356786ac103ba7cf468310962bf64dd7` |
| `../multigpu-layer-l2-revalidation-2026-10-14/validation.json` | `4e7104bc6b5dae067d3d9d524e0fdb63e4210db9d0e7158f9e1e6676d742be70` |
| `../multigpu-layer-l3-revalidation-2026-10-14/README.md` | `41b7c4771bb0c86af280c1f9e1660b03eff14fc86119277dc02296fedb54d13d` |
| `../multigpu-layer-l3-revalidation-2026-10-14/DECISION.md` | `7a64692ab3940068f8d41762f26e0ba9a62dc95fcb417a93200b0f88811abc46` |
| `../multigpu-layer-l3-revalidation-2026-10-14/facts.json` | `5e1f770f36ee6fabd3142b3b4e4298a157471fea08c08c6710046d655b23101d` |
| `../multigpu-layer-l3-revalidation-2026-10-14/provenance.json` | `975274c52b8bd2e3d7754946a88cf037544702af0c5f94cb513790216d7dec8e` |
| `../multigpu-layer-l3-revalidation-2026-10-14/validation.json` | `99af0c5ee5f6d17343c9c38df374ff0db85748d851c3413734f3956d5532c684` |
| `../multigpu-layer-2026-10-13/README.md` | `339b01edad875f0f7ef7179d053ec2772de42417a5164bbee9653f1197cfb5d3` |
| `../multigpu-layer-2026-10-13/SCOPE.md` | `5c9d728fd67e4e26b66e616c16eb11f6ae16ff2603495d25fa564d8c140980b5` |
