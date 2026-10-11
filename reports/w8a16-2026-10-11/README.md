# W8A16 — końcowy pomiar opcjonalnego Q8

`q8_0` zmniejsza rezydentną pamięć projekcji kosztem czasu i jawnych
różnic jakości. Zalecanym domyślnym trybem pozostaje `none`.
Ten pakiet łączy odnośniki do zamkniętych wcześniejszych dowodów
z osobnym pełnym pomiarem R5. Nie zmienia statusów historycznych
ani nie łączy przebiegów różnych sesji w nowe pary.

## Zakres i pochodzenie

2026-10-11, 2× NVIDIA RTX 6000 Ada Generation 48 GB,
sterownik 580.126.09, CUDA/cuBLASLt 12.9.1 (identyfikator tabel 120901).
GPU 0: wydajność/profile/pamięć, GPU 0 i 1: repliki i wcześniejsza
zgodność eksportów. Żądany SM 1200 MHz, limit 300 W, skonfigurowane
16 logicznych CPU. Rzeczywiste zegary, moc, temperatura, partie,
obserwator obcych procesów i zakresy próbek pozostają w JSON;
nie służą do wykluczania wyników. Nie ma ciągłego historycznego
śladu affinity każdego procesu.

Binarka bazowa etapu replik: źródło `4894c5e`, SHA
`289a2eaca3ac96762b7acd309f9dbf39f43041c7dab37ae89eb968125f82dfef`.
Jej wbudowana etykieta `6ef86665` pochodzi z budowy przed commitem.
Kandydat W8, niezatwierdzony kod na tym etapie: SHA
`21bdb702b2cb4c4bb59908c772eca8ce6599cafb7851a0b731a02b3de3cb8bce`.
65 plików wejściowych budowy (także Cargo.lock i manifesty) ma
fingerprint `5d8a70247f87f55c0271e8ad9dc4df2eb63601d5d1ab9b33f0c5400c62202ea5`.
Rust i binarki pozostają te same między R4 i R5.
R5 zmienia referencję kontroli HTTP; SHA runnera
`92b47c9e341f47fe7a538cbd42a0f0e152ae7199720bda2b5d15f79f83d945ea`.
Pełne revisions/SHA wag, źródeł, danych, tabel GEMM, helperów,
wersje binarek i bibliotek są w `provenance.json`.

R5 ukończył zaplanowane kampanie: 85 PASS, 0 FAIL, 1 GAP, 0 SIGNAL,
`complete: true`, summary `exit_status: 3`. GAP to niewykonany
opcjonalny ślad nsys. Kod z summary nie jest osobno zaobserwowanym
kodem wyjścia zewnętrznego procesu runnera. Parent ponownie sprawdził
95 pobranych plików (10 456 985 B) przez SHA-256 i potwierdził brak
usuniętej dokładnie tej maszyny w API. Nie ma aktywnej maszyny pomiarowej.

R5 używa `--models mini` dla poprawności; jego fingerprint wejść
obejmuje ten model. Referencję 4.5B używaną przez bench/profile wiąże
osobno pełny manifest wysłanych plików i sprawdzenie SHA wszystkich
230 plików (szczegóły w provenance). Weryfikacja SHA ma znacznik
czasu parenta, a start perf — zegara maszyny pomiarowej; nie zmierzono
ich przesunięcia. Same znaczniki nie dowodzą dokładnej kolejności
na wspólnej osi czasu. Powiązanie treści przez SHA nie zależy od tej kolejności.
Pierwsza pełna kontrola uploadu wykryła brak redundantnego, nieużywanego
archiwum binarki. Dodano je po starcie poprawności; rzeczywiście
wykonywane binarki i runner były sprawdzone przed ich uruchomieniem.
Nie twierdzimy, że wszystkie 230 plików sprawdzono przed startem runnera.
Nie zmieniano kodu Rust, modelu, referencji ani runnera podczas pomiaru.

## Poprawność, none i jakość

[Pierwsza zamknięta kampania](../multigpu-w8-original-2026-10-11/README.md)
zawiera pełne osiem porównań, 14 eksportów none i 10 Q8,
bitową równość none z poprzednim etapem oraz spójność Q8 między
GPU, trybami i `CUDA_VISIBLE_DEVICES`. Zachowuje 302 PASS, 3 FAIL,
15 GAP, 4 SIGNAL, `complete:false`, summary exit 1 oraz zero-bajtowe
ostatnie A3. Tych statusów nie zastępują wyniki nowej sesji.

R5 naprawia zakres referencji HTTP: 34/34 natywnych body i statusów
zgodnych z Q8 (32 z eksportu, 2 z ogólnej ścieżki `basal decide` dla
klasy Unsupported eksportera). Osiemnaście requestów otrzymało model
serwowanego mini, pozostałe 16 się nie zmieniło. Parent niezależnie
porównał pełne kanoniczne odpowiedzi/statusy z rzeczywistymi źródłami,
bez zastępowania ich `answer_upstream` i bez wyjątków po ID.
Pierwotny FAIL 18/34 pozostaje historycznym błędem referencji R4.

R5 none: 93/93 odpowiedzi bazy i kandydata identycznych; Q8 repliki:
279/279 równych pojedynczemu Q8, w serii sekwencyjnej i dwóch seriach
po 93 przy 16 klientach. Routing 135/132 obejmuje 267 zadań z modelem,
w tym 249×200 i 18 błędów; 12 pozostałych odrzucono przed wyborem modelu.
Łącznie repliki: 249×200, 24×422, 6×400. Nowy zwykły eksport Q8 mini
jest też bitowo równy oryginalnemu eksportowi R4.
Serwery mają zapis graceful exit 0 i braku pamięci procesu po wyjściu.
Launch records w `commands.json` nie zawierają własnych pól zakończenia;
dowodem tego są kontrole i logi serwerów. Surowych udanych body
health/models nie zachowano, tylko ich kontrole zgodności.

[Wydajność none](../multigpu-w8-none-2026-10-11/README.md): pierwotne
sygnały nie utrzymały się w ustalonym potwierdzeniu. Nie wykazano
powtarzalnego spadku przy tej rozdzielczości; nie jest to dowód zerowego
narzutu ani ogólnego braku regresji. Wszystkie pary i mieszane kierunki
pozostają w osobnym zaakceptowanym pakiecie.

[Jakość](../multigpu-w8-quality-2026-10-11/README.md): Q8 zmienia granice
evidence w trzech pytaniach. Na 900 pytaniach nowe zmiany decyzji względem
FP32: mini 3, 4.5B 5; fixed/shared: mini 0/1, 4.5B 1/0.
FP32/F16/Q8: mini 664/663/664 poprawnych, 4.5B 703/702/704.
Na bench 44 avg/cal decyzje są zgodne 44/44; poprawne wobec gold:
mini 31/44, 4.5B 35/44 (FP32, F16 i Q8).
Zgodność argmaxów nie oznacza równości rozkładów i nie dowodzi ogólnej
jakości. Natywne przypadki i kompatybilne 55 pytań endpointu nie są
zamiennymi zakresami porównania z upstream.
[Decyzja użytkownika M2](../../.itsol/initiatives/basal-multigpu/phases/03-w8a16/quality-decision.md)
akceptuje wyłącznie opcjonalne Q8 z jawnymi różnicami i zaleca none.
Oryginalne kryteria endpointu FP32 pozostają FAIL, nie stają się PASS.

## Pełne nowe A/B Q8 — koszt czasu

A: kandydat F16 none, B: ten sam kandydat F16 q8_0, model 4.5B,
identyczne ustawienia poza quantization. Świeże procesy, ta sama tabela GEMM.
Bench: osiem przebiegów ABBAABBA, po 44 przykłady, 5 rozgrzewkowych,
39 mierzonych indeksów 5..43. Latency2 median A/B (ms):
25,505/42,564; 25,545/42,363; 25,463/42,422; 25,482/42,537.
Ilorazy B/A 1,658–1,669. Każda para i każda mierzona pozycja pozostaje
w `perf-q8.json`. To opis metryki tego obciążenia, bez bramki przyspieszenia.

HTTP benchmark: pełne A B B A, dwie ustalone pary A0/B1 i A3/B2,
po 900 sequential c1, mixed c1 i mixed c16 na proces.
10 800 odpowiedzi 200, bez błędów, anulowań i zmian we własnym przebiegu.
Wszystkie 21 metryk każdej pary, wartości A/B, B−A, B/A, pełne statystyki
faz i klas oraz warunki są w JSON. Poniżej wybrane latency/przepustowość;
żadnej pozostałej metryki nie wykluczono z danych.

| Metryka | Para 0 A / B | Para 1 A / B |
|---|---:|---:|
| sequential c1 latency p50 ms | 26,980 / 43,260 | 26,783 / 42,964 |
| sequential c1 RPS | 31,949 / 20,953 | 32,217 / 21,021 |
| mixed c1 latency p50 ms | 45,921 / 61,942 | 45,456 / 61,919 |
| mixed c1 RPS | 4,895 / 4,481 | 4,917 / 4,483 |
| mixed c16 latency p50 ms | 597,837 / 674,999 | 585,846 / 672,929 |
| mixed c16 RPS | 5,709 / 5,283 | 5,760 / 5,277 |

Nie parujemy starych trzech przebiegów z nowymi. Wszystkie stare dane,
w tym B2 i brak A3, pozostają w osobnym historycznym pakiecie.

## Pamięć — zakres liczników

Wszystkie liczby tabeli to bajty `used` puli cuMemAllocAsync, a czas to
`load_s`; pełne `reserved`, watermarks, metadata i NVML są w JSON.
Pool obejmuje tensory, bloki Q8, przejściowe F16 i workspace,
wyłącza kontekst CUDA, moduły, uchwyty bibliotek i alokacje synchroniczne.
Rezydentne to odczyt po ładowaniu i synchronizacji, przed requestami,
z prefixami/workspace; nie tylko wagi. Runtime high obejmuje wszystkie
fazy, warm-up i drenaż razem, nie osobne fazy. NVML jest próbkowanym
odczytem całego urządzenia, nie prawdziwym szczytem alokacji.

| Przebieg | load used high B | resident used B | runtime used high B | load s |
|---|---:|---:|---:|---:|
| 4.5B A0 none | 9 644 216 320 | 9 644 216 320 | 12 531 592 828 | 12,714 |
| 4.5B B1 Q8 | 5 395 567 368 | 5 309 140 480 | 8 044 867 864 | 16,001 |
| 4.5B B2 Q8 | 5 395 567 368 | 5 309 140 480 | 8 044 867 864 | 14,602 |
| 4.5B A3 none | 9 644 216 320 | 9 644 216 320 | 12 379 943 592 | 12,243 |
| mini F16 | 3 272 813 832 | 3 271 867 392 | 4 409 372 132 | 4,347 |
| mini Q8 | 1 922 361 608 | 1 868 155 904 | 3 005 660 644 | 4,261 |

After shutdown used/reserved wynosi 0/0 dla wszystkich sześciu procesów;
to diagnostyka puli po zwolnieniu silnika i synchronizacji.
Mini ma po jednym przebiegu na tryb, nie statystyczny dowód szybkości.
Sequential mini: 900; mixed c1/c16: po 846. Z tych samych 900 wejść
przed fazami pominięto identyczne 54 żądania mixed (klasy doc-4k-8k
i doc-16k) przekraczające kontekst 8192 tokenów. Wszystkie ID i szczegółowe powody są w workloads.
Łącznie mini: 5184 odpowiedzi 200, bez błędów/anulowań/zmian we własnym
przebiegu. Nie opisujemy mini jako pełnych 3×900.
Szacunek preflight z arytmetyki formatu jest osobno w model metadata;
nie jest zmierzoną pamięcią ani gwarancją ochrony przed OOM.

## Profile diagnostyczne

Po jednym F16 i Q8 dla 40 przykładów, batch 1 oraz batch 16.
`profile-memory.json` zawiera wszystkie sekcje i ich różnice.

| batch | F16 profiled ms/item | Q8 profiled ms/item | Q8 dequant ms/item | dequant effective GB/s | F16 pure-GEMM weight-read effective GB/s |
|---|---:|---:|---:|---:|---:|
| 1 | 34,681 | 245,486 | 97,108 | 145,834 | 488,808 |
| 16 | 34,535 | 66,754 | 9,358 | 226,988 | 102,756 |

To osobne zsynchronizowane profile, nie latencja produkcyjnego requestu.
W Q8 profil synchronizuje przed/po dekwantyzacji; licznik obejmuje
wywołanie alokującego API, dekwantyzację i synchronizację końcową,
jako część już wliczoną w sekcje projekcji. Nie dodajemy go drugi raz
do sumy i nie przypisujemy jego 97 ms do zwykłego kosztu około 17 ms
bench. Synchronizacje mogą zmieniać oddawanie pamięci puli i kolejność
pracy; bez nsys nie ustalamy udziału każdej przyczyny narzutu.

Efektywne GB/s to arytmetyka wartości/czas: 3,0625 B na dequantized
value, a dla F16 2 B wag tylko qkv_proj i gate_up_proj. Liczba forwards
z liczników Q8: 1/item dla batch1, 0,15/item dla batch16.
Nie są to bezpośrednie liczniki transferów DRAM ani fizyczna przepustowość.
Znana dodatkowa praca Q8 to dekwantyzacja i przejściowe macierze;
profile opisują jej koszt w trybie synchronizowanym. A8/A9 lub nowy
fused kernel pozostają przyszłymi, niezmierzonymi alternatywami.

## Kontrole i luki

`static-validation.json`: fmt, locked Clippy i Rustdoc default/Intel/CUDA,
cargo-deny oraz kontrola wyjątków/zależności z zachowanymi hashami logów.
65 wejść budowy nie zmieniło się od tych kontroli. Metal lokalnie:
review źródeł, bez świeżej kompilacji; macOS CI nadal musi go sprawdzić.
Nie dodano testów projektu i nie uruchamiano cargo test.

Wczesne odmowy błędnych konfiguracji z pierwszej sesji mają lukę alokacji:
kończą się przed pierwszą próbką 0,5 s. Metal/Intel runtime, model max,
inne GPU i inne zegary nie zostały zmierzone. Historyczna jakość Q8
i none performance nie są nowym pomiarem R5, lecz dowodami tej samej
niezmienionej binarki. Ogólnej jakości ani braku regresji nie udowodniono.

`commands.json` zachowuje wszystkie rekordy bez lokalnych argv/env;
część launch records serwerów nie ma pól czasu/wyjścia. Pominięto
pojedyncze per-request rows dla rozmiaru, zastąpiono ścieżkę modelu
neutralną referencją/publicznym revision, deduplikowano pełny source
benchmarku. Wszystko pozostało w SHA-bound raw; pełne fazy/klasy,
workload hashes, skips i zakresy są w pakiecie. Hashe settings to UTF-8
Python json.dumps(sort_keys=True, separators=(',', ':')), ensure_ascii=True,
bez końcowej nowej linii.

Przed lokalnym commitem dane i dokumentacja podlegają dwóm niezależnym
końcowym review. Wynik oraz domknięcie etapu zapisuje
[progress inicjatywy](../../.itsol/initiatives/basal-multigpu/progress.md).
PR powstanie po zakończeniu wszystkich etapów inicjatywy.
