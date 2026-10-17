# W8 — zamknięta pierwsza kampania, z zachowanymi lukami

> **Dane JSON w archiwum.** Pliki `correctness.json`, `disposition.json`, `perf-q8-original.json`,
> `quality-primary.json` i `raw-manifest.json` z tego katalogu są poza Gitem, w archiwum pomiarów
> `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod ścieżką `reports/multigpu-w8-original-2026-10-11/<plik>`.
> Odwołania do nich w tekście dotyczą plików z archiwum. Pobieranie i sumy kontrolne:
> [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Archiwum zawiera też wersję tego pliku
> sprzed dopisania tej noty. Przeniesienie danych nie jest nowym pomiarem: liczby, statusy i rozstrzygnięcia
> raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

Ten pakiet opisuje historyczną kampanię z 2026-10-11, zamrożony runner R4
i binarkę W8. Nie jest akceptacją całego etapu. Oryginalny wynik pozostaje
`complete: false`, `exit_status: 1`: 302 PASS, 3 FAIL, 15 GAP, 4 SIGNAL.
Status zapisany w summary nie jest zaobserwowanym kodem wyjścia zewnętrznego
procesu runnera. Pobrano i ponownie sprawdzono wszystkie 318 plików
(55 662 897 B); maszynę usunięto, a jej brak potwierdzono przez API.

## Pochodzenie i metoda

2× NVIDIA RTX 6000 Ada Generation 48 GB, sterownik 580.126.09,
limit mocy 300 W. Wydajność mierzono na GPU 0, repliki na GPU 0 i 1.
Żądano SM 1200 MHz i skonfigurowano maskę 16 logicznych CPU.
Maskę potwierdza konfiguracja oraz pojedyncza późniejsza obserwacja;
nie ma ciągłego zapisu affinity każdego dziecka. CSV opisują rzeczywiste
warunki, nie dowodzą idealnie stałego zegara ani braku wszystkich zakłóceń.

Role binarek, SHA-256 65 plików wejściowych budowy (także Cargo.lock
i manifestów), runnera, źródeł i całego zbioru
są w `disposition.json`, `correctness.json` i `raw-manifest.json`.
Baza to kod etapu replik `4894c5e`; jej wbudowana etykieta wersji
pozostaje `6ef86665` z budowy przed commitem. Kandydat nie ma nowego commita.
Zmierzony runner R4 ma SHA `cd57da8dd1977522dfb8b4af4493aae43152e99e010c7972a0b1fe3c16da2a3a`.
Późniejszy R5 poprawia referencję HTTP, bez zmiany binarki i źródeł Rust;
dowód jego runtime'u należy do osobnej kampanii. Pakiet nie zawiera
adresów, nazw hostów, identyfikatorów maszyn ani ich ścieżek.

## Poprawność i ograniczenia kontroli

14 eksportów `none` (10 kandydata i 4 bazy; mini, 4.5B; domyślne i jawne none, tree i single,
GPU 0 i 1) daje identyczne bajty z binarką etapu replik. Każdy zawiera
44 pozycje bench i 34 System One. Dziesięć eksportów Q8 jest bitowo
zgodnych we własnych rodzinach między urządzeniami i konfiguracjami,
w tym bez prefix cache oraz przy `CUDA_VISIBLE_DEVICES=1`, logical 0.
Nie oznacza to równości Q8 z FP32 lub F16. Wszystkie 10 porównań none
i 8 Q8 pozostaje w raw przypisanym przez SHA; pakiet zachowuje ich
ścieżki względne, SHA eksportów i dowód równości bajtów. Sekcja `upstream_endpoint` porównuje różne dialekty
i nie służy do dowodzenia równości bajtowej eksportów natywnych.

HTTP none: 93/93 identyczne statusy i pełne kanoniczne body między bazą
i kandydatem. Q8: 279/279 odpowiedzi replik równych pojedynczemu Q8
(93 sekwencyjne i dwa przebiegi po 93 przy 16 klientach).
Statusy jednej serii: 83×200, 8×422, 2×400; replik: 249/24/6.
Routing replik wynosił 139/128 dla 267 żądań z rozpoznanym modelem
(249×200 i 18×400/422). Pozostałych 12 odrzucono przed wyborem modelu,
więc nie trafiły do routingu; łącznie było 249×200 i 30×4xx. Zapisano zakończenie serwerów exit 0
i kontrolę braku pamięci procesu po zakończeniu. To zapis obserwatora
procesu, nie dowód z ostatniego odczytu całego urządzenia: ostatni CSV
pojedynczego Q8 nadal pokazuje 510 MiB, a replik 2 MiB na obu GPU.

Zgodność `/health` i `/v1/models` jest zapisana w kontrolach runnera;
nie zachowano surowych body pozwalających ponownie je porównać.
21 rodzin metryk HELP/TYPE jest zgodnych. Jedenaście niepoprawnych
konfiguracji kończy się exit 1 z oczekiwanym komunikatem przed ładowaniem,
ale po 0,063–0,066 s, przed pierwszą próbką placement co 0,5 s:
alokacji GPU nie potwierdzono. Metal i Intel nie miały kontroli runtime.

Oryginalne HTTP Q8 względem eksportu pozostaje FAIL 18/34.
16 natywnych body wskazywało stary model `basal-1.0-4.5B`, podczas gdy HTTP przepisywał
model na mini. To błąd referencji runnera; nie zmieniamy starego wyniku
w PASS. Poprawiona kontrola 34 body wymaga osobnego dowodu R5.

## Jakość

`quality-primary.json` zachowuje pełne osiem porównań: dwa modele razy
FP32/F16, FP32/Q8, F16/Q8 oraz FP32/MLX-Q8 (kontekst porównawczy).
W bench avg i cal: 44/44 zgodnych decyzji, mini 31/44 poprawnych,
4.5B 35/44 dla FP32, F16 i Q8. JSON zawiera także różnice logitów,
prawdopodobieństw, TV, NLL, Brier i ECE; zgodne argmaxy nie znaczą
zgodnych rozkładów ani ogólnego zachowania jakości.
Osiem różnic tokenizacji/formatu względem upstream występuje już w F16;
między F16 i Q8 nie ma tych różnic. Zachowano również błędy natywnych
przypadków i pełne wyniki System One; nie wykluczamy ich z raportu.

[Osobny pakiet jakości](../multigpu-w8-quality-2026-10-11/README.md)
opisuje 55 kompatybilnych pytań endpointu na model oraz 900 pytań.
Q8 zmienia granice evidence w trzech pytaniach oraz wprowadza 3 nowe
zmiany decyzji mini i 5 dla 4.5B względem FP32. To nie dowodzi ogólnej
jakości. [Rzeczywista decyzja M2](../../.itsol/initiatives/basal-multigpu/phases/03-w8a16/quality-decision.md)
akceptuje Q8 jako opcjonalny tryb z jawnymi różnicami, zaleca domyślne
`none`. Oba pierwotne kryteria endpointu FP32 pozostają FAIL;
decyzja nie nadaje im równości ani statusu PASS.

## Wydajność i pamięć Q8 — zachowany niepełny pomiar

A: F16 none kandydata, B: F16 Q8 kandydata, model 4.5B.
Bench wykonał cztery pary AB/BA w kolejności ABBAABBA.
Każdy świeży proces: 44 przykłady, 5 rozgrzewkowych, 39 mierzonych
unikalnych indeksów 5..43. Wszystkie wartości A/B i każda pozycja
są w `perf-q8-original.json`. Mediany latency2 A: 25,679; 25,639;
25,635; 25,647 ms; B: 42,417; 42,376; 42,371; 42,366 ms.
To opis dodatkowej pracy Q8, około 1,65× czasu tej metryki,
bez bramki przyspieszenia i bez wniosku dla innych modeli/partii/GPU.

HTTP benchmark planował A B B A po 900 żądań sequential c1,
mixed c1 i mixed c16 na proces. Zakończyły się A0, B1 i B2:
8100 odpowiedzi 200, bez błędów, anulowań i zmian względem pierwszej
odpowiedzi we własnym przebiegu. Ostatnie A3 zatrzymano po 375,317 s
na deadline; JSON ma zero bajtów i nie ma końcowych metryk.
Jedynie pierwsza para A0/B1 jest kompletna. Zachowano B2 i wszystkie
braki drugiej pary. Nie parujemy B2 ze starym A0 ani z nową sesją.

Pool resident used po ładowaniu: A0 9 644 216 320 B,
B1/B2 5 309 140 480 B; reserved: A0 9 663 676 416 B,
B1/B2 5 368 709 120 B. Runtime used high: A0 12 254 394 932 B,
B1 8 044 867 864 B, B2 8 196 516 988 B. Wszystkie load/runtime/resident,
used/reserved i after_shutdown (0/0) zachowano w JSON.
To liczniki puli cuMemAllocAsync, obejmujące m.in. tensory i workspace,
wyłączające kontekst, moduły, uchwyty bibliotek i alokacje synchroniczne;
nie są pełnym szczytem pamięci urządzenia ani samymi wagami.
CSV NVML to oddzielna telemetria próbkowana. Ustawienia benchmarku
są haszowane przez UTF-8 `json.dumps(settings, sort_keys=True,
separators=(',', ':'))`, domyślne ensure_ascii=True, bez końcowej nowej linii.

Zapisano pary, pełne agregaty faz i klas, warunki, obserwacje obcych
procesów i źródłowe SHA. Identyczny blok `source` trzech zakończonych
przebiegów HTTP zachowano raz w `benchmark_source`, wraz z haszami
manifestu, profilu danych, publicznego zestawu i poszczególnych workloads.
Ścieżkę `model.reference` zastąpiono `<local-checkpoint>` i publicznym
revision; pełne lokalne referencje pozostają w raw. Dla prywatności lub
rozmiaru pominięto `argv`/`env`, pełne `resources`, pojedyncze
`phases[].per_request` i ścieżki `a`/`b` porównań. Są tylko w raw
przypisanym przez SHA, z jawną listą `omitted_fields` w JSON.
318 zweryfikowanych plików to 315 w rawroot i 3 sąsiednie;
Astra niezależnie sprawdziła 315, parent i Opus wszystkie 318. Profile Q8/F16 i mini memory nie zostały uruchomione
w tej kampanii. [Zamknięty pakiet none](../multigpu-w8-none-2026-10-11/README.md)
pozostaje osobnym dowodem braku powtarzalnego sygnału przy tej rozdzielczości,
z ograniczeniami. Całe W8 czeka na osobną kampanię R5 uzupełniającą
HTTP, pełne nowe Q8 A/B, profile i pamięć mini oraz na review tych danych.
