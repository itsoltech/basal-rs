# W4A16 S1: jakość i poprawność q4_k, 2026-10-12

> **Dane JSON w archiwum.** Pliki `facts.json`, `quality-44-34.json`, `quality-900.json` i `validation.json` z
> tego katalogu są poza Gitem, w archiwum pomiarów `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod ścieżką
> `reports/w4a16-s1-2026-10-12/<plik>`. Odwołania do nich w tekście dotyczą plików z archiwum. Pobieranie i
> sumy kontrolne: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Archiwum zawiera też wersję
> tego pliku sprzed dopisania tej noty. Przeniesienie danych nie jest nowym pomiarem: liczby, statusy i
> rozstrzygnięcia raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

Stan: zamrożony pakiet danych do niezależnego review. To nie jest akceptacja
etapu 04 ani decyzja M2. Ścisłe kryterium FP32 dla `q4_k` ma wynik FAIL na
obu modelach (klasa `quality-criterion`). Wymagana jest decyzja użytkownika
M2. Opcjonalna decyzja M2 dla Q8 nie obejmuje W4.

Rewizja R2 poprawia wyłącznie opis zakresu dowodów: pochodzenie wartości,
liczbę ładowań `q4_k` i opis RSS hosta. Liczby, statusy i oba pliki
`quality-*.json` są bez zmian.

## Zakres i pochodzenie

Sesja S1 trwała od 06:59 do 07:25 UTC i obejmowała kampanie correctness,
quality, profile oraz memory-mini. Sprzęt: 2 × NVIDIA RTX 6000 Ada Generation
48 GB, sterownik 580.126.09, cuBLASLt 120901. Warunki: żądany zegar SM
1200 MHz, limit 300 W, maska 16 logicznych CPU.

Pliki wykonywalne:

- kandydat: SHA-256 `a3ed4ade…`, 65 plików źródłowych, odcisk `94ef2131…`;
- baza: przebudowa `750c146`, SHA-256 `3bdea953…`. Runner przyjmuje
  etykietę bazy w deklarowanej postaci (GAP); jej pochodzenie sprawdzono poza
  runnerem;
- runner: `97e9df05…`.

Pełne odciski są w `facts.json`.

Runner zakończył pracę statusem `complete` i kodem wyjścia 1, ponieważ
wystąpiły FAIL. Liczniki: PASS 491, FAIL 2, GAP 38, SIGNAL 6. Oba FAIL mają
klasę `quality-criterion`. Nie wystąpiły FAIL klas `correctness` ani
`timeout`.

Manifest surowych danych `w4-s1-20261012-01-raw-sha256-2026-10-12T072505.json`
(`963713d4…`) obejmuje 444 pliki o łącznym rozmiarze 79 299 066 B. Rodzaj,
rozmiar i SHA każdego pliku są zgodne. Surowe dane pozostają prywatne.

Pochodzenie wartości (szczegóły w `validation.json`, `value_provenance`).
Skrypty autora wyłącznie odczytywały pliki. Niezależnie od runnera przeliczyły
z surowych eksportów, śledzonych referencji FP32 i plików decyzji:

- listy naruszeń 44+34 dla pięciu porównań na model;
- dane 900: mapowanie wierszy i `gold`, tokeny, matematykę hosta, decyzje,
  trafność, zmiany, pary, marginesy, TV, logity i rozkłady;
- liczniki statusów z `summary.json` i `checks.txt`;
- pary równości bajtowej (SHA surowych eksportów);
- statystyki obiektu `quantization` wobec P0 (13 manifestów eksportów).

Wyniki są równe wynikom runnera. Pozostałe wartości to obserwacje zapisane
przez runner (`summary.json`, `checks.txt`), przeniesione bez przeliczenia:

- wyniki HTTP i replik, w tym `/health`, `/v1/models`, rodziny metryk
  i SIGTERM;
- licznik routingu replik;
- czas dekwantyzacji profilu;
- część pól memory-mini;
- szacunki doctor i czasy zakończenia odmów.

Autor sprawdził z surowych plików tylko część z nich: liczby i statusy
rekordów żądań, sekcje profilu i pulę pamięci. Dwa niezależne review danych
R1 porównały z surowymi rekordami m.in.:

- odpowiedzi HTTP;
- 34 natywne body;
- repliki i liczniki 130/137;
- wartości profilu i memory-mini.

Nie znalazły rozbieżności. `/health`, `/v1/models` i rodziny metryk opierają
się wyłącznie na kontrolach runnera; ich surowych body nie porównano ani tutaj,
ani w review. SIGTERM opiera się na kontroli runnera i jego źródle.

## Kryterium FP32: 44 bench, 34 System One i 55 pytań endpointu

Obowiązuje istniejący predykat W8 (`docs/BENCHMARKS.md`): bez tolerancji i
bez wyjątków po ID.

| Model | Naruszenia | Unikalne pozycje | Bench (cal+avg) | Pozycje SO | Endpoint | Pola `*_same` = false |
|---|---:|---:|---:|---:|---:|---|
| mini | 26 | 15 (2 bench, 13 pytań) | 4 | 2 | 6 | choice 1, action 1, answer 1, spans 7, first 4 |
| 4.5B | 20 | 12 (4 bench, 8 pytań) | 8 | 0 | 3 | choice 2, selected 1, spans 4, first 2 |

Kolumna „Pozycje SO” zawiera zmiany `argmax(p_cal)` natywnych pozycji
System One o identycznych tokenach (14 na model). Kontrola f16 wobec FP32
daje 0 naruszeń na obu modelach.

Diagnostyka `q4_k`:

- wobec f16 tej sesji: 26 (mini) i 19 (4.5B);
- wobec `q8_0` tej sesji: 24 i 21;
- kontekst: `q8_0` wobec FP32: 3 i 2.

Jedno pytanie może dać kilka naruszeń, np. zmianę pozycji SO, decyzji
endpointu i pól typowanych. Liczba naruszeń nie jest więc liczbą pytań.
Odpowiedzi FP32, f16, q8 i q4 zmienionych pytań, ze spanami evidence,
zawiera `quality-44-34.json`.

## 900 pytań

Kontrola f32 tej sesji:

- 900/900 decyzji jest równych zapisanemu upstream FP32;
- `p_cal` jest bitowo równe zapisanemu `basal_rs_f32` w 900/900 pytaniach;
- maks. |Δp| wobec upstream: 1,27e-5 (mini) i 8,89e-6 (4.5B).

Porównania z zapisanym FP32 są więc przypisywalne.

| Model | Zmiany wobec FP32: f16 / q8 / q4 | q4: nowe / naprawione / wspólne | q4 wobec f16 / q8 sesji | Trafność FP32 / f16 / q8 / q4 |
|---|---|---|---|---|
| mini | 1 / 4 / 77 | 76 / 0 / 1 | 77 / 78 | 664 / 663 / 664 / 668 |
| 4.5B | 1 / 5 / 81 | 81 / 1 / 0 | 82 / 80 | 703 / 702 / 704 / 685 |

### Pary wobec `gold`

Porównanie `q4_k` z f16 tej sesji:

| Model | Obie poprawne | Tylko f16 | Tylko q4 | Obie błędne |
|---|---:|---:|---:|---:|
| mini | 630 | 33 | 38 | 199 |
| 4.5B | 659 | 43 | 26 | 172 |

Nowe zmiany wobec FP32, ocenione względem `gold`:

| Model | q4 poprawne, FP32 błędne | q4 błędne, FP32 poprawne | Obie błędne |
|---|---:|---:|---:|
| mini | 38 | 33 | 5 |
| 4.5B | 25 | 43 | 13 |

Trafność według typu, w kolejności FP32 / f16 / q4:

| Model | choice | score | noul |
|---|---|---|---|
| mini | 285 / 285 / 297 | 122 / 121 / 120 | 257 / 257 / 251 |
| 4.5B | 317 / 317 / 308 | 119 / 119 / 116 | 267 / 266 / 261 |

Różnica +5 na mini to wynik tej próbki. Nie oznacza poprawy, równoważności
ani uogólnienia jakości. Na 4.5B `q4_k` daje 685, a f16 702.

### Zbiory danych i marginesy

W 4.5B nowe zmiany koncentrują się w zbiorze emotion: 32 na 100 pytań,
a trafność spada z 64 do 53. W mini nowe zmiany rozkładają się tak: sst5 16,
cdsc-e 14 (trafność z 57 do 67), emotion 12, allegro-reviews 11. Prompty i
tokeny są identyczne we wszystkich formatach. Nie zidentyfikowano mechanizmu
zależnego od zbioru.

Zmienna `margines FP32` oznacza różnicę dwóch najwyższych prawdopodobieństw
FP32:

- p50 (nearest-rank) marginesu nowych zmian: 0,141 (mini) i 0,149 (4.5B);
  p50 (nearest-rank) wszystkich pytań: 0,710 i 0,790;
- wśród 100 najmniejszych marginesów leży 38 z 76 (mini) i 40 z 81 (4.5B)
  nowych zmian;
- największy margines nowej zmiany: 0,804 (mini) i 0,857 (4.5B), więc nie są
  to wyłącznie remisy;
- zwycięzca `q4_k` jest drugim indeksem FP32 w 71 z 76 (mini) i 57 z 81
  (4.5B) nowych zmian.

### Rozkłady różnic

Zmienna `TV` oznacza całkowitą odległość wariacyjną rozkładów odpowiedzi.

| Porównanie | Model | mean | p50 | p90 | p99 | max |
|---|---|---:|---:|---:|---:|---:|
| TV q4 wobec FP32 | mini | 0,0807 | 0,0514 | 0,2023 | 0,3980 | 0,6635 |
| TV q4 wobec FP32 | 4.5B | 0,0728 | 0,0287 | 0,2031 | 0,4757 | 0,6633 |
| TV q8 wobec FP32 | mini | 0,0052 | 0,0025 | 0,0133 | 0,0340 | 0,0625 |
| TV q8 wobec FP32 | 4.5B | 0,0042 | 0,0020 | 0,0114 | 0,0257 | 0,0486 |
| \|Δ logit liter\| q4 wobec f16 | mini | 0,460 | 0,361 | 0,953 | 1,930 | 4,182 |
| \|Δ logit liter\| q4 wobec f16 | 4.5B | 0,847 | 0,621 | 1,871 | 3,460 | 6,710 |
| \|Δ logit liter\| q8 wobec f16 | mini | 0,030 | 0,022 | 0,063 | 0,130 | 0,437 |
| \|Δ logit liter\| q8 wobec f16 | 4.5B | 0,038 | 0,032 | 0,078 | 0,136 | 0,238 |

Percentyle liczono metodą nearest-rank. Logity porównano dla 6600 logitów
liter na model. `quality-900.json` zawiera:

- wiersze zmian: marginesy, rangi, TV, przesunięcie luki według W8 R2;
- grupy według zbioru i typu;
- rozkłady `p_cal` f16, q8 i q4 dla każdego z 900 pytań.

Pozostałe dane pochodzą z śledzonego pliku decyzji.

## Analiza defektów

W sprawdzonym zakresie nie znaleziono defektu źródeł ani danych. To
ograniczony wynik negatywny, a nie dowód braku defektu. Sprawdzono:

- **Wejścia 900, mapowanie wierszy i `gold`:** wiersze 0–899 zgadzają się z
  `sequential.jsonl` i plikami decyzji (0 rozbieżności na obu modelach).
  Cztery pliki JSONL wejścia są bajtowo równe przygotowanym.
- **`inputs-900.json`:** różni się od przygotowanego (`d4f6c37e…`) wyłącznie
  kolejnością dwóch kluczy modeli. Sparsowany JSON jest równy, także po
  sortowaniu kluczy. Zachowujemy oryginalne `prepared_900_equal: false`
  kolektora; to różnica serializacji, nie danych.
- **Model, config, kalibracja, tokeny:** rewizje i SHA wag oraz configu są
  równe manifestom referencji FP32 i tożsamości P0. Temperatury kalibracji
  w każdym eksporcie są równe referencji. Prompt, `input_ids`, `letter_ids`,
  `perm` i `pack` są identyczne w sześciu eksportach 900 (900/900).
  Pozycje 44 bench mają tokeny identyczne z referencją FP32.
- **Matematyka hosta:** `probs`, `p_avg`, `p_cal` i argmaxy odtworzono
  bitowo z zapisanych logitów f32 dla 10 800 rekordów. Różnice powstają więc
  w logitach liter forwardu GPU.
- **Ścieżki `none` i `q8_0`:** 10 + 10 par eksportów i pary 900 są bajtowo
  równe bazie, a odpowiedzi HTTP są równe bazie. Liczby 900 dla `q8_0` są
  równe pakietowi W8.
- **Wagi `q4_k`:** S1 zawiera 21 ładowań `q4_k` (mini 13, 4.5B 8).
  - Pełny obiekt `quantization` zapisało 14 z nich (mini 8, 4.5B 6):
    13 manifestów eksportów i memory-mini. W obrębie modelu obiekty są
    identyczne. Statystyki rekonstrukcji (model i cztery typy projekcji) są
    równe P0 w bitach f64.
  - Skrypt autora porównał z P0 13 manifestów. Obiekt memory-mini
    sprawdzono w R2 odczytem: jest równy obiektowi mini. Jedno z review R1
    potwierdziło 70 porównań (14 obiektów × model i 4 projekcje).
  - Pozostałe 7 ładowań zapisało tylko linię logu: mini 5 (pojedynczy
    serwer 1, repliki 2, rezerwowe ładowania `basal decide` 2) i 4.5B 2
    (profil b1 i b16). Linia podaje to samo względne RMS i PASS warm-upu.
    Równości obiektu i P0 dla nich nie wykazano.
  - Eksporty `q4_k` są bajtowo równe między GPU, single/tree,
    `--no-prefix-cache` i `CUDA_VISIBLE_DEVICES`.
- **Dekwantyzacja GPU:**
  - kontrola warm-upu we wszystkich 21 ładowaniach (linia logu): macierz
    warstwy 0 `o` jest bitowo równa rekonstrukcji hosta;
  - kernel `dequantize_block_q4_K` i `BlockQ4K::to_float` w przypiętym
    Candle 0.11.0 mają tę samą kolejność wartości w superbloku;
  - mają też tę samą arytmetykę. `d·sc` i `dmin·m` są dokładne w f32
    (≤ 17 bitów znaczących), a `d1·q` jest dokładne (≤ 21 bity). Wynik
    `d1·q − m1` ma jedno zaokrąglenie z FMA i bez niego, a potem obie strony
    zaokrąglają f32 do f16 metodą RNE;
  - bitowa równość jest więc oczekiwana dla każdej macierzy, ale zmierzono ją
    dla jednej macierzy na ładowanie.
- **GEMM:** zdekwantyzowana macierz f16 trafia do tego samego cuBLASLt
  `matmul_nt` co gęsta waga f16. Pole `gemm` i tabela bundled są wspólne dla
  f16, q8 i q4.
- **Kwantyzator:** Candle `BlockQ4K::from_float` używa `make_qkx1_quants(15, 5)`
  na grupę 32 wartości, bez wag ważności. P0 zmierzył dokładnie ten
  kwantyzator, a M1-F go wybrał. Inny kwantyzator byłby zmianą formatu lub
  zakresu (rewizja planu lub M2), nie poprawką defektu. Jego wpływu nie
  zmierzono.

Zmierzona zmiana jest rzędem wielkości zgodna z numeryką kwantyzacji
`q4_k`. P0 daje R `q4_k` / R `q8_0` = 188 na obu modelach (√ ≈ 13,7).
Średnie |Δ logit| wobec f16 tej sesji są 15,5 (mini) i 22,2 (4.5B) raza
większe niż dla `q8_0`, a średnie TV 15,9 i 17,2 raza. To opis bez progu i
bez dowodu przyczyny. Wyjaśnienie mechanizmu nie zamyka M2.

Poza zakresem analizy:

- bitowa dekwantyzacja GPU pozostałych macierzy (tylko argument źródłowy);
- porównanie aktywacji warstw;
- referencja upstream 4-bit, której nie ma;
- powtórzenie kwantyzacji lub P0;
- inne karty i modele.

## Poprawność S1: dokładny zakres

**`none`:**

- 10 par eksportów 44+34 jest bajtowo równych bazie: tree i single, GPU 0
  i 1, bez flagi oraz z `--quantization none`;
- 900 f16 kandydata jest bajtowo równe bazie;
- HTTP: 93/93 odpowiedzi równe bazie; `/health`, `/v1/models` i rodziny
  metryk są równe według kontroli runnera (bez porównania surowych body w
  tym pakiecie i w review).

**`q8_0`:**

- 10 par jest bajtowo równych bazie tree GPU 0, w tym `--no-prefix-cache`
  i `CUDA_VISIBLE_DEVICES=1 --device 0`;
- 900 `q8_0` jest bajtowo równe bazie;
- HTTP: 93/93 odpowiedzi równe bazie.

**Spójność `q4_k`:**

- 8 par jest bajtowo równych tree GPU 0;
- warm-up przechodzi we wszystkich 21 ładowaniach (jedna macierz na
  ładowanie);
- równość P0 statystyk zbiorczych wykazano dla 14 ładowań z pełnym obiektem
  `quantization`, nie dla wszystkich macierzy.

**HTTP `q4_k` (mini):**

- 34/34 statusów i natywnych body jest równych natywnej referencji `q4_k`
  dla tych samych bajtów żądania: 32 z eksportu i 2 z `basal decide`;
  statusy 200 × 32, 400 i 422;
- to spójność serwowania `q4_k`, a nie porównanie z FP32;
- repliki na GPU 0 i 1: 93/93 odpowiedzi sekwencyjnie oraz 93/93 w każdej
  z dwóch rund po 16 klientów są równe pojedynczemu serwerowi;
- licznik zakolejkowanych żądań replik wynosi 130 i 137. Suma 267 nie jest
  utożsamiana z 279 żądaniami HTTP;
- SIGTERM kończy proces kodem 0, a próbki po wyjściu nie pokazują pamięci
  procesu (kontrola runnera);
- równości odpowiedzi i licznik routingu pochodzą z rekordów runnera.
  Review R1 porównały je z surowymi żądaniami, body i `metrics.txt`.

**Odmowy:**

- 35 nieprawidłowych poleceń zostało odrzuconych przed pobraniem modelu i
  ładowaniem (PASS);
- brak pamięci GPU procesu ma status GAP, bo proces kończy się po
  0,07–0,08 s, przed pierwszą próbką po 0,5 s.

**Doctor** (kontrole runnera): szacunek wag `q4_k` wynosi ok. 1,0 GB.
Szacunek `q8_0` wynosi ok. 1,8 GB i jest równy w kandydacie i bazie. Błędne konfiguracje są
zgłaszane.

**GAP 38:**

- etykieta źródła bazy: 1;
- odmowy przed pierwszą próbką: 35;
- Metal i Intel, tylko statycznie: 1;
- nsys, którego nie planowano: 1.

Nie stosujemy bramki zero GAP. Oryginalne statusy pozostają bez zmian.

## Profil (4.5B, 40 pozycji, jeden przebieg na format)

Wartości w ms na pozycję.

| Format | b1 razem | b1 dekwant. | b1 gate_up | b16 razem | b16 dekwant. | b16 gate_up |
|---|---:|---:|---:|---:|---:|---:|
| f16 | 34,83 | — | 10,11 | 34,58 | — | 7,79 |
| q8_0 | 232,80 | 95,45 | 146,09 | 56,36 | 8,14 | 18,85 |
| q4_k | 187,72 | 49,02 | 142,70 | 55,94 | 7,59 | 20,11 |

Dekwantyzacja jest synchronizowana i wliczona do sekcji projekcji. Różnica
sekcji `gate_up` wobec f16 przy b1 (132,6 ms dla `q4_k` i 136,0 ms dla
`q8_0`) przekracza cały czas dekwantyzacji. Ten profil nie przypisuje
reszty. To obserwacja wydajności, nie defekt jakości. Wyniki nie opisują
latencji żądań, przepustowości ani ogólnej szybkości. Rodziny wydajności
S2 i S3 to przyszłe sesje.

## memory-mini (mini, jeden przebieg na format)

Każdy przebieg obejmuje 900 żądań sekwencyjnych c1, 846 mieszanych c1 i 846
mieszanych c16. Wszystkie zakończyły się bez błędów. W każdym przebiegu
pominięto tych samych 54 żądań mieszanych ponad limitem kontekstu (36
doc-4k-8k i 18 doc-16k), więc nie wszystkie 900 żądań mieszanych otrzymało
odpowiedź.

| Format | Pula po załadowaniu [B] | Szczyt puli w pracy [B] | Szacunek wag [B] | load_s | Żądania/s: seq / mixed c1 / mixed c16 | Próbka NVML [MiB] |
|---|---:|---:|---:|---:|---|---:|
| f16 | 3 271 867 392 | 4 409 372 132 | 3 193 073 112 | 4,39 | 76,4 / 24,7 / 35,6 | 5032 |
| q8_0 | 1 868 155 904 | 3 005 660 644 | 1 789 361 624 | 4,16 | 53,9 / 21,7 / 34,1 | 3720 |
| q4_k | 1 119 439 872 | 2 331 161 112 | 1 040 645 592 | 9,40 | 54,7 / 21,8 / 34,3 | 3080 |

Kolumny „Pula” podają wartości `used` puli stream-ordered. Pula nie obejmuje
kontekstu CUDA, modułów, uchwytów bibliotek ani alokacji synchronicznych. Po
zamknięciu pula wynosi 0 we wszystkich przebiegach. NVML to próbki urządzenia
co 200 ms, nie szczyty.

RSS hosta: nie zmierzono izolowanego szczytu RSS ładowania ani kwantyzacji.
Fazy benchmarku zapisują natomiast próbkowany co 200 ms rekord
`lifetime_peak_rss_bytes`. To `ru_maxrss` z `getrusage(RUSAGE_SELF)`,
przeliczone z KiB na bajty. Dotyczy jednego procesu: serwera HTTP i
generatora obciążenia. Wartość fazy to maksimum próbek w oknie fazy.
`ru_maxrss` jest szczytem od startu procesu, więc obejmuje wcześniejszą
aktywność, w tym ładowanie modelu. Nie jest to szczyt samego ładowania,
przyrost w fazie, zużycie całego hosta ani końcowy szczyt procesu.

| Format | seq c1 [B] | mixed c1 [B] | mixed c16 [B] |
|---|---:|---:|---:|
| f16 | 699 322 368 | 813 699 072 | 825 626 624 |
| q8_0 | 712 388 608 | 827 617 280 | 837 414 912 |
| q4_k | 711 753 728 | 826 343 424 | 839 761 920 |

Pole runnera `host_rss_peak: "not measured"` w `summary.json` pozostaje bez
zmian. Ten pakiet odczytuje je jako brak izolowanego szczytu.

Bloki `q4_k` projekcji mini zajmują 842 305 536 B wraz z paddingiem. Gęste
f16 tych samych projekcji zajmowałoby 2 994 733 056 B. To pamięć wyłącznie
projekcji, nie udział w pamięci całego modelu.

## Pliki

- `facts.json`: zawiera:
  - powiązania SHA;
  - liczniki kontroli;
  - wyniki 44+34 i 900;
  - kwantyzację i P0;
  - pary równości bajtowej;
  - HTTP, profil i pamięć;
  - analizę defektów i ograniczenia.
- `validation.json`: zawiera:
  - weryfikację manifestu surowych danych i plików chronionych;
  - SHA użytych plików surowych i śledzonych;
  - porównanie przeliczenia z runnerem;
  - pochodzenie wartości: przeliczone, zapisane przez runner, sprawdzone w
    review;
  - spis 21 ładowań `q4_k` i zmiany rewizji R2;
  - wyjaśnienie `inputs-900.json`.
- `quality-44-34.json`: zawiera:
  - naruszenia predykatu dla pięciu porównań;
  - odpowiedzi zmienionych pytań;
  - flipy bench z rozkładami.
- `quality-900.json`: zawiera:
  - wiersze zmian;
  - grupy;
  - rozkłady f16, q8 i q4 każdego pytania.

## Ograniczenia

- Pakiet nie zmienia kryterium. Nie dodaje tolerancji, wyjątków po ID, nowej
  bramki jakości ani testu statystycznego.
- Dane obejmują jedną kartę (RTX 6000 Ada), dwa modele i jedną sesję. Pakiet
  nie uogólnia wyników.
- Pakiet nie zawiera decyzji M2 ani wyników S2/S3. Nie wykonano powtórzeń.
- Wartości zapisane przez runner nie są niezależnym przeliczeniem. Ich
  porównanie z surowymi danymi ma zakres opisany w `validation.json`.
