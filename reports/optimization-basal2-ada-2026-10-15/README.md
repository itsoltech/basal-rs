# basal-2.0 — optymalizacja czasu HTTP i przepustowości na RTX 6000 Ada

Celem było poprawienie czasu pojedynczego pytania oraz przepustowości bez ukrywania błędów i bez zmiany wyników obliczeń wariantów strukturalnych. To osobna kampania na innym CPU i sterowniku niż [wcześniejszy pomiar pojedynczego żądania](../latency-basal2-ada-2026-10-15/README.md). Czasów z obu maszyn nie łączymy.

Kampania: 15–16 października 2026. Czasy zdarzeń w surowych danych są zapisane w UTC; nazwa katalogu oznacza dzień rozpoczęcia.

## Zmiany i decyzje

- **Tablica GEMM mini:** zachowano 97/100 wpisów, zmieniono trzy algorytmy dla klasy M4096. Na 300 W mediana przy 4096 tokenach spadła ze 148,55 do 129,34 ms, czyli o 12,93%. Dla sześciu długości 3200–4097 redukcja wynosi 11,8–13,8%. Osobne potwierdzenie na 250 W dało 11,7–13,6%. Wszystkie zapisane odpowiedzi obu potwierdzeń są bitowo zgodne. Obliczenia modelu pozostają domyślne.
- **Limit dużych żądań:** dodano `--max-large-inflight` oraz pole YAML `max_large_inflight`. Domyślne 4 pozostaje bez zmian. To kontrola liczby przyjętych żądań i zajętej pamięci; zwiększenie limitu pozwala mierzyć GPU zamiast szybkości odrzucania HTTP 529. Nie przyspiesza pojedynczego forwardu.
- **Attention:** nowe warianty układu pamięci i wielkości bloku pozostają jawnie eksperymentalne. Pomiary mini nie uzasadniły zastąpienia domyślnego kernela. Istniejące `tc-pv1`, `tc-qk1`, `tc-f16` zmieniają numerykę; ich osobne wyniki oraz porównanie z FP32 podano niżej. Żaden nie został domyślnym wyborem.
- **Konfiguracja i narzędzia:** runner zapisuje wejścia, odpowiedzi, statusy, limity, sumy źródeł i binarek; odróżnia pojedynczy przegląd SIGNAL od potwierdzenia ABBA. Podwyższone limity są ustawieniem izolowanego benchmarku, nie nową konfiguracją produkcji.

Nie wdrażano tych zmian do działającej usługi.

## Metoda

Czas HTTP zaczyna się przed serializacją JSON i wysłaniem żądania. Kończy się po odebraniu całego body odpowiedzi, przed parsowaniem JSON klienta. Obejmuje tokenizację, kolejkę, wykonanie modelu i odpowiedź serwera. Nie obejmuje ładowania modelu, kompilacji silnika ani rozgrzewki. Klient i serwer są na tej samej maszynie; nie mierzymy opóźnienia Internetu.

Domyślne obliczenia basal-rs to FP16 z dokładniejszym attention hi/lo oraz akumulacją FP32. Upstream v2.0.0 `fast` używa BF16, `torch.compile` i CUDA Graphs. To porównanie ustawień produktów, nie identycznej arytmetyki. W obu silnikach: jedno pytanie w drabinie długości, jedna kolejność opcji, cache KV stanów wyłączony, brak skracania wejść. Oba korzystają z tego samego przypiętego `CALIBRATION.json`, dopasowanego do jednej kolejności opcji; dodatkowy audyt metadanych zachowuje sumy plików. Preobliczenie prefiksu i pozostałe mechanizmy produktu mogą działać. Upstream: scheduler `auto`, max batch 64, okno 3 ms, kolejka domyślna; jej liczbę oczekujących żądań ogranicza liczba klientów benchmarku.

Potwierdzenie tablicy używa ABBAABBA: A to nowa binarka kontrolna ze starą tablicą, B ta sama binarka z tablicą hybrydową. Pierwszy przebieg każdego wariantu zachowano, lecz wyłączono ze statystyki. Każda długość ma dwie rozgrzewki i dziesięć próbek na przebieg. Wynik to mediana trzech median. Zakresy median nie są przedziałami ufności. p95 z małych prób jest orientacyjny. Pomiary przepustowości oznaczone SIGNAL to pojedyncze przejścia po obciążeniach, nie potwierdzony stan ustalony ani absolutny limit sprzętu.

Liczba klientów oznacza zamkniętą pętlę: klient wysyła kolejne żądanie po odebraniu poprzedniej odpowiedzi. QPS podaje liczbę kompletnych, poprawnych formatowo odpowiedzi HTTP 200 na sekundę. Walidacja techniczna odpowiedzi nie oznacza zgodności decyzji z prawidłową etykietą. Błędów i szybkich odrzuceń nie liczymy jako wykonanych inferencji. Przy większej liczbie klientów czas HTTP zawiera dłuższe oczekiwanie w kolejce; wyższa przepustowość sama nie oznacza niższego opóźnienia.

Korpus podstawowy: 44 krótkie pytania choice PL/EN oraz 400 żądań mieszanych; mini korzysta z 371 mieszanych żądań mieszczących się w jego kontekście. Krótkie pytania są powtarzane. Nawet przy wyłączonym cache stanów identyczne prompty mogą współdzielić obliczenia w jednym batchu; ich wysokiego QPS nie należy przenosić na ruch z różnymi dokumentami. Osobny, syntetyczny przegląd różnorodności dodaje na początku dokumentu osiem rozdzielonych cyfr: 1024 różne prompty wobec kontroli z 44 powtarzanymi promptami i takim samym rozkładem długości tokenowych. Nonce może zmieniać odpowiedź modelu; porównujemy niezmienność odpowiedzi wewnątrz danego wejścia, a nie równość odpowiedzi dwóch różnych korpusów.

Drabina pojedynczego pytania zawiera pierwotnych 25 promptów: mini 128–8192, większe modele 128–32768 tokenów całego promptu. To polskie syntetyczne dokumenty, jedno pytanie o formę reklamacji. Ich prompty i tokeny były zgodne z upstream w poprzedniej kampanii. Dodano sześć długości mini 3200/3584/3840/4095/4097/4608; świeży audyt CPU wszystkich 31 wejść (25 pierwotnych i sześć sąsiednich) potwierdził równość pełnych promptów, tokenów oraz ich zachowanych natywnych kotwic z przypiętym upstream. Wynik jest w `runs/input-ladder-upstream-audit/`; pierwotnych plików audytu wejść nie zmieniano. Kontrole FP32 mają 49 przypadków bench i korpus endpointu z zachowanymi oczekiwanymi odrzuceniami. Używamy zachowanych eksportów upstream `float32` z przypiętej rewizji i ich manifestów; nie jest to świeża pełna drabina FP32 długich kontekstów na tej karcie. Mała próba nie dowodzi zachowania jakości dla wszystkich dokumentów ani długich kontekstów.

Bieżące dopasowane potwierdzenia ABBA obu silników dotyczą mini 4096/8192 i większych modeli 32768 tokenów. Pozostałe długości mają natywny przegląd SIGNAL; pełną wcześniejszą drabinę obu silników opisuje osobny raport z innej maszyny, wskazany na początku. Nie łączymy ich w jedną porównawczą serię.

## Sprzęt i wersje

RTX 6000 Ada Generation, 48 GB VRAM (49140 MiB), compute capability 8.9; sterownik 580.159.03. Native: CUDA/cuBLASLt 12.9.1, identyfikator cuBLASLt 120901. Reference: PyTorch 2.11.0+cu128, Transformers 5.17.0; pełny zestaw pakietów jest w zamrożonym `requirements.txt`. CPU Intel Xeon Gold 5412U; kontener ma limit 8 CPU, 64 GiB RAM, bez swapu i 768 procesów/wątków. Limit 8 CPU jest kwotą kontenera, bez przypisania osobnych rdzeni (`cpuset` pusty); pozostałe usługi serwera pozostawały aktywne. Część wag była w tmpfs, przy tym samym limicie RAM. Ładowanie jest poza czasem odpowiedzi.

Główne nowe pomiary: fabryczny limit 300 W, wentylator tymczasowo 100%, cel sterownika 85°C, zatrzymanie benchmarku przy 80°C. Bez blokady zegara SM. Kontrola stanu obejmowała temperaturę, moc, innych klientów GPU, zapas RAM/dysku, ciągłość uruchomienia hosta, nowe błędy GPU/OOM/sprzętu oraz zdrowie czterech istniejących usług. Aktywne chłodzenie utrzymywało temperaturę poniżej progu przerwania. Na zakończenie przywracano 300 W, cel 85°C i automatyczne sterowanie wentylatorem.

Wcześniejsze przebiegi diagnostyczne 150/250 W, automatyczny wentylator oraz próba 85% mają osobne katalogi. Nie łączymy ich statystyk z 300 W/100%. Próg 80°C jest konserwatywną granicą benchmarku, nie rozpoznaniem przyczyny historycznych restartów serwera. W tej kampanii nie potwierdzono takiej przyczyny.

Kod bazowy: `9c0762f2971e0970345c2ddc7dad03a189d11658`, z patchami i sumami plików w archiwum. Pierwotna binarka `169ce1…` pochodzi z `3fa10de179b0ac8fb4bec024065aa5e90f543096`; między tym commitem a bazą `9c0762…` nie ma zmian w `crates` (zmieniała się dokumentacja). Upstream bez lokalnych zmian: `af1ce43a1770dd4fdfc1611e3fe720b93d42fc73` (v2.0.0).

| Rola binarki CUDA | SHA-256 |
|---|---|
| Pierwotna referencja | `169ce19a4be027f9026c5be8b2d86bb41a4374d92f4667fa373cb3c5ddf02f30` |
| Pierwotna binarka eksperymentalnych układów attention | `7fa8daa241a4122592aff5b2b4b04f77e7c24ce3f226d8637719d4064393bf51` |
| Kontrola z konfigurowalnym limitem przyjęcia | `473d0108af6eb2596a0511f4203453e6aaf69fd7577e0f172e9c5a07bb738dbf` |
| Zintegrowana tablica i limit przyjęcia, przed spięciem budżetu forward | `07620ab3f6d5fbc567cbd128007814b53bbf41e106f3fb8cccae338518e15877` |
| Końcowa binarka z budżetem forward zgodnym z konfiguracją serwera | `f318abbba2df1519aa2b1e3e60d0be49d7eef87ed0573124efdd0badd29c4d0e` |

Checkpointy BF16 identyczne dla silników: mini `cb81b32ce89b0b87e62fcdad28f0435e25c953f3`; 4.5B `e5ba9149760d99b73b1963d8e0ccf948b58afacb`; max `2fffc5cc41e5cf7a2f752f38f17e7b4d9a4a8252`. Pełne zależności reference są w artefaktach instalacji. Dokładne warunki każdego pomiaru są w jego `method.json` i telemetryce.

## Potwierdzenie tablicy mini na 300 W

| Tokeny całego promptu | Stara tablica ms | Hybrydowa ms | Skrócenie czasu |
|---:|---:|---:|---:|
| 1024 | 31,11 | 30,85 | 0,8% |
| 2048 | 63,19 | 63,12 | 0,1% |
| 3200 | 112,74 | 99,33 | 11,9% |
| 3584 | 128,08 | 111,81 | 12,7% |
| 3840 | 138,09 | 119,43 | 13,5% |
| 4095 | 148,00 | 127,60 | 13,8% |
| 4096 | 148,55 | 129,34 | 12,9% |
| 4097 | 157,96 | 139,32 | 11,8% |
| 4608 | 163,83 | 163,61 | 0,1% |
| 8192 | 328,72 | 328,17 | 0,2% |

Pełne 40 wierszy obejmuje także 128/256/512 tokenów oraz krótkie i mieszane obciążenia 1/8/32/128 klientów. Występuje rozrzut czasów także tam, gdzie algorytm GEMM się nie zmienił: np. 128 tokenów 8,03→7,15 ms, zakresy median 8,03–8,08 i 6,79–8,10 ms. Nie przypisujemy tej różnicy zmianie klasy M4096. Główny wniosek dotyczy zakresu długości objętego zmienioną klasą i jest spójny z osobnym potwierdzeniem 250 W. Przy 32 klientach krótki QPS 276,71→278,22 i mieszany 58,63→58,83 praktycznie się nie zmieniły. Przy 128 klientach krótki korpus 44 powtarzanych promptów poprawił medianę QPS z 375,30 do 417,92 (+11,4%) i p50 z 306,47 do 268,15 ms (−12,5%); zakresy median p50 296,37–320,45 wobec 261,68–295,66 ms. To potwierdzenie ABBA dla powtarzanych wejść, które mogą współdzielić pracę w batchu; nie jest potwierdzeniem takiego samego zysku na 1024 różnych promptach. Nie deklarujemy ogólnego przyspieszenia wszystkich żądań.

Runtime zaokrągla M do wielokrotności 16 do 512 włącznie, wyżej do 128, a następnie wybiera klasę tablicy. Zmiana M4096 dotyczy więc zakresu faktycznych rozmiarów i pakowania; liczba tokenów dokumentu nie jest bezpośrednio kluczem tablicy. Potwierdzono sześć sąsiednich kształtów, pełne rekordy eksportów i osobną zgodność grupy algorytmów: 792 porównania, cztery kształty wag, trzy ziarna, 33 rozmiary M, obie tablice. Wszystkie wartości skończone i zgodne bitowo. To audyt zmierzonej grupy, nie formalny dowód dla wszystkich macierzy.

Wpisy mają wspólne kształty dla mini 1.5 i 2.0, dlatego zachowano istniejącą nazwę zasobu `basal-1.5-mini`. Dla 1.5 sprawdzono 45 obsługiwanych przypadków bench i 24 przypadki endpointu, pięć sposobów wykonania, po obu stronach zmiany. Dziesięć pełnych eksportów jest zgodnych bitowo. Zintegrowana binarka poprawnie dobiera nową tablicę dla świeżego cache obu generacji mini i zachowuje zgodne odpowiedzi. Istniejący poprawny cache GEMM pozostaje używany; jawne wskazanie nowej tablicy przez `--gemm-table`/YAML pozwala jej użyć bez usuwania starego cache.

## Kontrola przyjmowania żądań

Dwa osobne zalewy po 32 jednoczesne duże rozwinięte żądania dały: limit 4 — cztery przyjęte, 28 odpowiedzi 529; limit 8 — osiem przyjętych, 24 odpowiedzi 529. Health oraz małe i duże żądania po opróżnieniu kolejki dawały 200. Odpowiedzi przyjętych żądań były zgodne bitowo. To oczekiwane zachowanie limitu, a nie pomiar maksymalnego QPS.

Nowe pomiary obciążenia ustawiają limit dużych żądań ponad liczbę klientów: 128/256/768 zależnie od przebiegu, przy limicie ogólnym 1024. Nadal działają budżety pojedynczego żądania: do 4096 gałęzi i 32 MiB rozwiniętego wejścia, oraz osobna pula mediów 4. Granica dużego rozwiniętego wejścia wynosi 128 KiB; surowe body powyżej 2 MiB lub chunked jest objęte wcześniejszą bramką tylko przy serwerze obsługującym media. Zwiększenie limitu wymaga pamięci hosta i wyboru akceptowanego opóźnienia kolejki.

## Pojedyncze pytanie: dopasowane porównania z upstream

Mediana i w nawiasie zakres trzech median przebiegów, w ms. Różnica czasu to basal-rs/upstream−1; wartość ujemna oznacza krótszy czas basal-rs. Każdy wiersz korzysta z referencji B swojej serii ABBA. Pierwsze przebiegi zachowano poza statystyką.

| Model | Tokeny | Attention basal-rs | basal-rs ms | upstream fast ms | Różnica czasu |
|---|---:|---|---:|---:|---:|
| mini | 4096 | domyślne | 127,51 (127,15–127,76) | 131,87 (131,70–131,90) | -3,3% |
| mini | 8192 | domyślne | 322,73 (322,09–323,01) | 301,45 (299,88–301,81) | 7,1% |
| mini | 8192 | tc-pv1 | 311,63 (310,51–313,02) | 301,45 (299,88–301,81) | 3,4% |
| mini | 8192 | tc-qk1 | 291,42 (287,93–291,46) | 301,45 (299,88–301,81) | -3,3% |
| mini | 8192 | tc-f16 | 279,69 (279,67–280,24) | 301,48 (300,55–301,78) | -7,2% |
| 4.5B | 32768 | domyślne | 6541,79 (6528,63–6552,91) | 5504,23 (5496,65–5504,85) | 18,9% |
| 4.5B | 32768 | tc-f16 | 4868,47 (4860,14–4878,46) | 5504,23 (5496,65–5504,85) | -11,6% |
| max | 32768 | domyślne | 11926,01 (11861,27–11932,34) | 10728,51 (10712,26–10729,26) | 11,2% |
| max | 32768 | tc-f16 | 9314,87 (9313,87–9327,04) | 10728,51 (10712,26–10729,26) | -13,2% |

Serie źródłowe: `paired-4.5B-attention-300w`, `paired-max-attention-300w`, `paired-mini-default-300w`, `paired-mini-f16-8192-300w`, `paired-mini-pv1-qk1-8192-300w`.

Tryby tc-pv1/tc-qk1/tc-f16 zmieniają numerykę i pozostają opt-in. Dla mini tc-f16 zachowano regresję bench/37 (48/49 decyzji FP32); sam pomiar czasu nie stanowi akceptacji jakości. P95 z trzydziestu próbek i te zakresy median nie są przedziałami ufności.

## Gotowość procesu, osobno od czasu HTTP

Zakres zaobserwowanych startów serwera w docelowych seriach ABBA, przy istniejących cache sesji. Obejmuje przygotowanie modelu i silnika, poprzedza gotowość API. To nie pomiar pierwszej instalacji, pobierania wag ani czasu żądania. Upstream wykonuje kompilację i przygotowanie grafów podczas startu.

| Model | basal-rs gotowość s | upstream fast gotowość s |
|---|---:|---:|
| mini | 5,5–5,6 | 157,7–169,7 |
| 4.5B | 13,0–14,1 | 380,5–600,4 |
| max | 24,5–27,6 | 654,9–805,1 |

## Przepustowość obu silników na tych samych wejściach

Każdy punkt to pojedynczy przegląd SIGNAL, te same ograniczenia CPU/RAM/GPU oraz domyślne attention basal-rs wobec upstream fast BF16. Krótkie wejścia: 1024 różne prompty, 2048 odpowiedzi na punkt. Korpus mieszany jest osobny; 371 żądań dla mini, 400 dla większych modeli.

### basal-2.0-mini

Źródło: `screen-throughput-mini-upstream-fan100-300w`; SIGNAL, błędy 0, braki 0.

Różne krótkie prompty:

| Klienci | basal-rs odp./s | upstream odp./s | basal-rs p50 ms | upstream p50 ms | basal-rs p95 ms | upstream p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128,63 | 95,66 | 7,51 | 10,24 | 8,07 | 11,11 |
| 8 | 225,44 | 130,40 | 33,18 | 61,78 | 36,82 | 72,79 |
| 32 | 267,79 | 120,87 | 112,37 | 263,25 | 119,71 | 295,62 |
| 64 | 259,64 | 107,36 | 230,47 | 597,99 | 250,04 | 672,18 |
| 128 | 308,49 | 132,30 | 383,72 | 935,94 | 405,42 | 1206,10 |

Mieszane żądania:

| Klienci | basal-rs odp./s | upstream odp./s | basal-rs p50 ms | upstream p50 ms | basal-rs p95 ms | upstream p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 46,34 | 25,35 | 154,46 | 284,42 | 335,26 | 644,32 |
| 32 | 60,25 | 24,99 | 481,37 | 1221,73 | 757,52 | 2079,86 |
| 128 | 114,68 | 26,36 | 743,22 | 2542,43 | 1636,98 | 6673,46 |

### basal-2.0-4.5B

Źródło: `screen-throughput-4.5B-upstream-fan100-300w`; SIGNAL, błędy 0, braki 0.

Różne krótkie prompty:

| Klienci | basal-rs odp./s | upstream odp./s | basal-rs p50 ms | upstream p50 ms | basal-rs p95 ms | upstream p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51,08 | 40,86 | 19,32 | 23,07 | 20,24 | 27,98 |
| 8 | 75,51 | 41,32 | 103,55 | 192,97 | 113,10 | 234,10 |
| 32 | 90,46 | 35,86 | 346,96 | 885,11 | 357,44 | 1056,73 |
| 64 | 94,52 | 39,54 | 659,83 | 1570,29 | 678,28 | 2254,17 |
| 128 | 101,27 | 43,56 | 1230,49 | 2946,24 | 1267,16 | 3716,91 |

Mieszane żądania:

| Klienci | basal-rs odp./s | upstream odp./s | basal-rs p50 ms | upstream p50 ms | basal-rs p95 ms | upstream p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6,45 | 4,71 | 329,22 | 1382,03 | 5779,31 | 3954,93 |
| 32 | 7,19 | 4,53 | 1104,73 | 3678,19 | 10832,63 | 18328,14 |
| 128 | 8,42 | 4,60 | 2569,84 | 7434,62 | 14748,78 | 44122,33 |

### basal-2.0-max

Źródło: `screen-throughput-max-upstream-fan100-300w`; SIGNAL, błędy 0, braki 0.

Różne krótkie prompty:

| Klienci | basal-rs odp./s | upstream odp./s | basal-rs p50 ms | upstream p50 ms | basal-rs p95 ms | upstream p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23,17 | 17,26 | 43,81 | 53,94 | 45,75 | 66,65 |
| 8 | 32,05 | 17,51 | 245,46 | 461,11 | 264,08 | 549,21 |
| 32 | 40,19 | 15,77 | 790,13 | 2045,26 | 814,13 | 2244,57 |
| 64 | 45,77 | 17,68 | 1383,94 | 3497,18 | 1406,85 | 5063,61 |
| 128 | 47,82 | 19,50 | 2644,36 | 6508,46 | 2679,13 | 8283,77 |

Mieszane żądania:

| Klienci | basal-rs odp./s | upstream odp./s | basal-rs p50 ms | upstream p50 ms | basal-rs p95 ms | upstream p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3,12 | 2,14 | 548,69 | 3052,99 | 13944,70 | 8095,17 |
| 32 | 3,38 | 2,06 | 1883,71 | 8339,09 | 27611,28 | 40223,70 |
| 128 | 4,05 | 2,16 | 3744,03 | 16233,49 | 31745,06 | 93883,95 |

## Różnice odpowiedzi i ogon opóźnień

Porównanie przepustowości jest przeglądem SIGNAL, po jednym przejściu każdego silnika. Przy 128 klientach i 1024 różnych krótkich promptach basal-rs uzyskał około 2,33×/2,32×/2,45× QPS upstream dla mini/4.5B/max. Modele były uruchamiane osobno na tej samej karcie; tych wartości nie sumujemy. Każda krótka faza zapisuje 2048 mierzonych odpowiedzi. Fazy mieszane mają 371 żądań mini lub 400 większego modelu i zawierają rozpędzenie oraz opróżnienie kolejki.

Wyższy QPS i lepsze p50 nie oznaczają przewagi każdego percentyla. W mieszanym punkcie ośmiu klientów oba większe modele mają **gorszy p95 basal-rs**, mimo lepszej przepustowości i mediany:

| Model | basal-rs p95 ms | upstream p95 ms |
|---|---:|---:|
| 4.5B | 5779 | 3955 |
| max | 13945 | 8095 |

Przy 32 i 128 klientach ich mieszane p95 jest już niższe niż upstream, zgodnie z pełnymi tabelami wyżej. Wynik punktu ośmiu klientów pozostaje przegraną; nie deklarujemy niższego czasu każdego żądania ani każdego obciążenia.

Silniki dają różne prawdopodobieństwa. Ścisły audyt pól dyskretnych liczy również kolejność listy `selected`. Poniższy podział zachowuje wszystkie takie różnice, a osobno pokazuje, czy zmienił się wybór/wartość pola lub przynależność etykiet, czy wyłącznie kolejność tej samej listy etykiet:

| Model | Odpowiedzi z różnicą porównywanych pól JSON | Zmiana wartości/etykiet poza kolejnością | Wyłącznie kolejność `selected` |
|---|---:|---:|---:|
| mini | 4 | 2 | 2 |
| 4.5B | 12 | 8 | 4 |
| max | 1 | 0 | 1 |

Łącznie to 17 odpowiedzi: 10 zmian wartości/wyboru i siedem zmian samej kolejności. Powtórzenia tego samego wejścia są liczone ponownie, więc nie jest to liczba unikalnych zadań ani błędnych odpowiedzi. W mini rzeczywiste zmiany porównywanych wartości dotyczą `answer` w `features/5` oraz `choice` w `unique-0363`. W 4.5B osiem zmian dotyczy `choice`, m.in. powtarzanego `unique-0426`. W max jedyna różnica to `features/16`, pytanie `topics`: `['login', 'card', 'address']` wobec `['card', 'login', 'address']`. Zestaw etykiet pozostaje ten sam, lecz kolejność może mieć znaczenie dla klienta API; nie usuwamy jej z wyniku ścisłego porównania.

Maksymalne różnice prawdopodobieństwa w tych porównaniach wynoszą około 0,0514 mini, 0,094237 modelu 4.5B i 0,046959 max. Audyt zachowuje także osobne zmiany argmax prawdopodobieństw oraz różnice wartości `noul` i `score`. Nie mamy etykiet FP32 dla wszystkich nowych wejść z nonce. Różnica wyboru nie wskazuje sama, który silnik wybrał prawidłową odpowiedź. Wszystkie grupy, ID i przykłady są w `http-decision-summary.json`. Nie przypisujemy całej różnicy wyłącznie precyzji: silniki różnią się też kernelami, pakowaniem i harmonogramem pracy.

Dodatkowy audyt CPU potwierdził równość tokenów wszystkich 3072 krótkich promptów (1024 na model) względem zachowanych natywnych hashy, z sześcioma bezpośrednimi kontrolami przygotowania na model. Dla wszystkich 1171 żądań mieszanych (371/400/400) porównał bezpośrednio prompt i tokeny każdej gałęzi. Uwzględnia upstreamowe `facts=auto`, nazwy opcji oraz wybór języka. Lokalny CLI przygotowania był starszą binarką CPU o zapisanej sumie; zmiany tej kampanii nie modyfikują kodu przygotowania. Kotwice 1024 krótkich wejść pochodzą dodatkowo z zamrożonej binarki benchmarku. Szczegóły są w `runs/input-prompt-upstream-audit-r2/`; kontrola nie wykonuje inferencji i nie mierzy jakości. Pierwszy niepoprawny audytor pominął `facts=auto`; jego zatrzymanie zachowano osobno.

## Przepustowość a kolejka — różne i powtarzane dokumenty

Osobne, dopasowane długością korpusy mają 1024 wiersze: kontrola zawiera 44 różne prompty, wariant różnorodny 1024. Każdy punkt wykonuje 4096 żądań po rozgrzewce 1024 żądań jednego klienta. Zakres całych promptów: 143–228 tokenów. W tabelach obie strony to basal-rs z tą samą binarką, tablicą hybrydową mini i domyślnym attention. Każdy przebieg ma status SIGNAL, zero błędów, braków i różnic bitowych wewnątrz tego samego wejścia.

### basal-2.0-mini

| Klienci | Powtarzane QPS | Różne QPS | Powtarzane p50 ms | Różne p50 ms | Różne p95 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 131,46 | 127,83 | 7,41 | 7,57 | 8,17 |
| 128 | 419,48 | 310,79 | 277,38 | 380,96 | 404,70 |
| 512 | 981,81 | 298,35 | 331,96 | 1644,78 | 1908,38 |

### basal-2.0-4.5B

| Klienci | Powtarzane QPS | Różne QPS | Powtarzane p50 ms | Różne p50 ms | Różne p95 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 51,31 | 50,76 | 19,22 | 19,45 | 20,45 |
| 128 | 190,46 | 101,45 | 631,61 | 1232,32 | 1265,11 |
| 512 | 498,08 | 101,06 | 868,39 | 5037,88 | 5708,89 |

Przy 512 klientach różne dokumenty nie zyskują przepustowości względem 128; ich mediana jest około cztery razy dłuższa. Dalsze zwiększanie przyjętej kolejki w tym pomiarze pogarsza czas odpowiedzi. Powtarzane prompty nadal dają duży wzrost QPS dzięki współdzieleniu pracy w batchu. To odrębne scenariusze.

Wcześniejszy korpus bez nonce ilustruje pytanie o „3× wolniej”: mini przy 32 klientach miało 281,27 QPS i p50 107,0 ms; przy 128 — 387,66 QPS i 299,5 ms. Cztery razy więcej klientów dało 38% większy QPS i 2,8× dłuższą medianę. To zmiana obciążenia tej samej wersji aplikacji. Zysk optymalizacji sprawdzamy przy tym samym obciążeniu.

Pełna drabina podstawowych 44 powtarzanych pytań 1/8/32/64/128 oraz wyższa rampa 1/128/256/512 pozostają w archiwum. Najwyższy zmierzony krótki QPS tej drugiej rampy to 1146,71 dla mini i 510,39 dla 4.5B przy 512 klientach. Ten korpus nie ma nonce i różni się od dopasowanego eksperymentu powyżej. Nie są to wyniki dla 512 różnych dokumentów ani dowód absolutnej maksymalnej przepustowości. Fazy mają końcowe opróżnianie kolejki i ograniczoną liczbę żądań na klienta; nie zastępują długiego pomiaru stanu ustalonego.


## Max: natywny przegląd pojemności

Osobny przegląd SIGNAL, 18 wierszy, zero błędów i braków. Krótkie żądania powtarzają pierwotne 44 prompty, po 3520 odpowiedzi na punkt. To krótsze wejścia bez dodatkowego nonce; nie porównujemy ich bezpośrednio jako izolowanej zmiany różnorodności z korpusem 1024 różnych promptów. Korpus mieszany ma 400 żądań.

| Obciążenie | Klienci | Odp./s | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| krótkie, powtarzane | 1 | 25,23 | 37,19 | 46,15 |
| krótkie, powtarzane | 8 | 34,23 | 228,24 | 251,43 |
| krótkie, powtarzane | 32 | 41,82 | 755,99 | 786,98 |
| krótkie, powtarzane | 64 | 47,25 | 1339,43 | 1367,07 |
| krótkie, powtarzane | 128 | 81,91 | 1507,94 | 1538,07 |
| mieszane | 8 | 3,11 | 559,52 | 13347,65 |
| mieszane | 32 | 3,32 | 2010,51 | 27672,39 |
| mieszane | 64 | 3,69 | 3098,04 | 33065,22 |
| mieszane | 128 | 4,03 | 3989,05 | 33401,37 |

Źródło: `capacity-max-128clients-fan100-300w`. Przy większej liczbie klientów koszt kolejki rośnie; poprawa QPS nie jest obietnicą niższego czasu pojedynczego żądania. Odpowiedzi natywne pozostały bitowo niezmienne w zapisanych powtórzeniach i fazach.


### Max: wyższa współbieżność na powtarzanych wejściach

Osobny przegląd SIGNAL, po 5632 krótkie odpowiedzi dla każdego punktu; te same 44 pierwotne prompty. Audyt: pięć wierszy, zero błędów i braków. Mieszany punkt 128 klientów to osobne powtórzenie 400 żądań, nie łączenie statystyki z poprzednim przeglądem.

| Obciążenie | Klienci | Odp./s | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| krótkie, powtarzane | 1 | 25,42 | 37,06 | 45,74 |
| krótkie, powtarzane | 128 | 88,34 | 1412,35 | 1450,05 |
| krótkie, powtarzane | 256 | 152,14 | 1600,20 | 1662,24 |
| krótkie, powtarzane | 512 | 278,80 | 1634,37 | 1787,92 |
| mieszane | 128 | 4,06 | 3566,81 | 32597,47 |

Źródło: `capacity-high-max-300w`. Maksimum oznacza najwyższy wynik w zbadanych punktach tego powtarzanego korpusu, a nie absolutny limit GPU ani pojemność ruchu z różnymi dokumentami.

## Numeryka attention i różnice decyzji

Domyślne `tc` rozkłada Q/K i P na części hi/lo. `tc-pv1` zaokrągla P do FP16, `tc-qk1` używa tylko części hi Q/K, a `tc-f16` łączy oba uproszczenia. Softmax i akumulacja pozostają FP32. Są to istniejące, jawnie wybierane tryby, nie nowe domyślne obliczenia.

Audyt pięciu sposobów wykonania (`tree`, `single`, ograniczony budżet, brak prefiksu, cache) porównuje pełne rekordy danego wariantu z jego własnym wynikiem `tree`. PASS takiego audytu oznacza zgodność wewnątrz wariantu i poprawną obsługę oczekiwanych odrzuceń. Nie oznacza bitowej zgodności wariantów liczbowych z domyślnym trybem ani takiej samej jakości jak FP32.

| Model i wariant | Argmax wobec FP32 | Maks. różnica skalibrowanego prawdopodobieństwa | Zmienione dyskretne pola korpusu endpointu |
|---|---:|---:|---:|
| mini, domyślny | 49/49 | 0,003854 | 0 |
| mini, tc-pv1, potwierdzony świeżym audytem | 49/49 | 0,005460 | 0 |
| mini, tc-qk1 | 49/49 | 0,004682 | 0 |
| mini, tc-f16 | 48/49 | 0,004752 | 0 |
| 4.5B, domyślny | 49/49 | 0,005878 | 0 |
| 4.5B, tc-pv1 | 49/49 | 0,003451 | 0 |
| 4.5B, tc-qk1 | 49/49 | 0,006856 | 0 |
| 4.5B, tc-f16 | 49/49 | 0,003744 | 0 |
| max, domyślny | 49/49 | 0,001878 | 0 |
| max, tc-pv1 | 49/49 | 0,001856 | 0 |
| max, tc-qk1 | 49/49 | 0,002794 | 0 |
| max, tc-f16 | 49/49 | 0,000907 | 0 |

Mini `tc-f16` zmieniło `bench/37`: wybór z indeksu 1 na 0, przy poprawnej etykiecie 1. Skalibrowane prawdopodobieństwa pierwszych dwóch opcji FP32 wynoszą 0,466185/0,466550, a wariantu FP16 0,466558/0,466096. Mała różnica liczbowa odwraca decyzję przy niewielkiej przewadze jednej opcji. Ten wynik pozostaje regresją w zachowanym przypadku i wyklucza przedstawianie trybu jako zamiennika bez różnicy jakości.

Wcześniejszy przegląd 4.5B na 250 W z wentylatorem 100% mierzył przy 32768 tokenach: domyślny 7500,91 ms, `tc-pv1` 7102,31 ms, `tc-qk1` 6067,05 ms i `tc-f16` 5610,48 ms. To SIGNAL, po dwie próbki na kształt. Odrębne potwierdzenie na 300 W i porównanie z upstream podajemy osobno; czasów nie łączymy. Maksymalna różnica względem FP32 w małym korpusie może być niższa po uproszczeniu, lecz nie jest to dowód poprawy jakości dla innych wejść.

Kontrola 49 przypadków i małego korpusu endpointu nie dowodzi zachowania jakości długich dokumentów, rzadkich przypadków ani wszystkich modeli. Dlatego żaden wariant liczbowy nie zastępuje domyślnego attention. Szczegółowe logity, prawdopodobieństwa, zmiany decyzji i oczekiwane odrzucenia są zachowane w archiwum.

Dla max świeży audyt 20 pełnych eksportów (cztery tryby × pięć sposobów wykonania) jest PASS: pełne rekordy każdego trybu są bitowo zgodne z jego własnym `tree`, bez nieoczekiwanych odrzuceń, braków i nieskończonych wartości. Wszystkie cztery tryby zachowały 49/49 argmax FP32 i dyskretne pola obsługiwanego korpusu endpointu. Nie jest to pełna kontrola jakości długiego kontekstu 32768.

W osobnym przeglądzie max na 300 W długość 32768 miała 11726,16 ms domyślnie, 11161,97 ms w tc-pv1, 9926,77 ms w tc-qk1 i 9306,95 ms w tc-f16. To SIGNAL, dwie próbki po dwóch rozgrzewkach. Pełny audyt przeglądu: 64 wiersze, 0 błędów/braków. Opisowy audyt 24 grup odpowiedzi wariantów liczbowych nie znalazł w tym korpusie zmian dyskretnych decyzji ani argmax względem natywnego domyślnego trybu; prawdopodobieństwa są różne. Nie jest to dowód jakości wszystkich długich kontekstów. Wyniki ABBA z upstream podajemy oddzielnie.

Dopasowany ABBA max na 300 W jest PASS: 144 odpowiedzi, trzy wiersze końcowe, zero błędów i braków. Dla 32768 domyślny tryb ma medianę 11926,01 ms (zakres median 11861,27–11932,34), tc-f16 9314,87 ms (9313,87–9327,04), a upstream fast 10728,51 ms (10712,26–10729,26). Domyślny basal-rs jest o 11,2% wolniejszy od upstream; tc-f16 jest o 13,2% szybszy od upstream i 21,9% szybszy od domyślnego trybu. Wynik 49/49 FP32 dotyczy osobnego małego korpusu; nie promujemy na jego podstawie uproszczonego attention do trybu domyślnego.

## Budżet tokenów forward

Przegląd kodu ujawnił, że serwer używał `max_batch_tokens` do składania i dopuszczania pracy, a niezależny `Engine.forward_max_tokens` pozostawał na 8192. Końcowa zmiana przekazuje budżet do silnika przed utworzeniem długiej kolejki. Domyślne 8192 się nie zmienia. Wpływ dotyczy łączenia kilku spakowanych wierszy w obrębie jednej grupy językowej; pojedynczy większy wiersz już wcześniej mógł być wykonany osobno ponad ten budżet.

Dwa nowe przeglądy mini porównują starą i końcową binarkę przy identycznym jawnym budżecie 16384 lub 32768, klientach 1/128/512 i wcześniejszych zamrożonych pełnych odpowiedziach. Oba audyty SIGNAL: po osiem wierszy, zero błędów, braków i różnic odpowiedzi względem poprzedniej referencji 8192. Nie wykazano wzrostu szczytowej przepustowości: krótkie żądania przy 128 klientach około 286→305 QPS dla 16384 i 306→304 QPS dla 32768; przy 512 klientach około 297→296 i 296→293 QPS. Jednoprzebiegowy zysk krótkiego N1 nie wynika z samego łączenia wierszy i pozostaje zmiennością warunków. Mieszany przegląd 32768 pogorszył QPS ze 103,88 do 94,41; wynik zachowano. Nie podnosimy domyślnego budżetu na podstawie tych danych.

Dla 4.5B oba analogiczne przeglądy również pozostały bez błędów, braków i różnic bitowych. Przy budżecie 16384 krótki QPS starej/końcowej binarki wynosił 102,81/102,89 przy 128 klientach i 103,42/103,47 przy 512; mieszany 11,83/11,16. Przy 32768 odpowiednio 103,78/102,81, 103,30/102,26 i 19,81/16,99. Różnica między dwoma mieszanymi przeglądami starej binarki pokazuje wpływ samego składania kolejki i zmienności krótkiej próby. Nie przypisujemy jej poprawce forward. Końcowa binarka respektuje konfigurację; pomiary nie uzasadniają zwiększenia domyślnych 8192.

### Max: zgodność budżetów 16k i 32k

Oba przeglądy SIGNAL są kompletne: po osiem wierszy, zero błędów, braków i różnic pełnych odpowiedzi względem zamrożonej referencji 8192. W każdym A/B **obie binarki dostają ten sam jawny budżet serwera**: 16384 albo 32768. Stara zachowuje wewnętrzny budżet forward 8192, końcowa respektuje żądaną wartość. To porównanie poprawki przekazywania konfiguracji, a nie kontrolowane ABBA trzech wartości budżetu.

| Budżet | Obciążenie | Klienci | Stara odp./s | Końcowa odp./s | Stara p50 ms | Końcowa p50 ms | Stara p95 ms | Końcowa p95 ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16384 | krótkie, różne | 1 | 23,29 | 23,13 | 43,56 | 43,90 | 45,35 | 45,75 |
| 16384 | krótkie, różne | 128 | 45,38 | 45,24 | 2782,08 | 2792,45 | 2820,52 | 2828,61 |
| 16384 | krótkie, różne | 512 | 46,60 | 46,64 | 11110,70 | 11097,23 | 11604,54 | 11623,32 |
| 16384 | mieszane | 128 | 5,77 | 5,78 | 5709,74 | 5681,63 | 17650,80 | 17569,17 |
| 32768 | krótkie, różne | 1 | 23,13 | 23,09 | 43,91 | 44,19 | 45,76 | 46,36 |
| 32768 | krótkie, różne | 128 | 45,20 | 45,15 | 2792,79 | 2794,49 | 2829,42 | 2835,82 |
| 32768 | krótkie, różne | 512 | 46,75 | 46,78 | 10746,82 | 10743,28 | 12488,02 | 11400,12 |
| 32768 | mieszane | 128 | 9,15 | 9,17 | 5417,82 | 5751,21 | 21263,28 | 19374,64 |

Nie ma istotnego zysku QPS samej poprawki forward przy stałej wartości budżetu. Krótkie 512 klientów także tutaj wydłuża kolejkę przy zbliżonym QPS do 128 klientów.

W osobnych pojedynczych przeglądach mieszany max przy 128 klientach miał około 4,05 odp./s i p50 3,74 s z budżetem 8192, 5,78 odp./s i p50 5,68 s z 16384 oraz 9,17 odp./s i p50 5,75 s z 32768. P95 wynosiło odpowiednio 31,75/17,57/19,37 s. To sygnał kompromisu: większy pakiet może zwiększać przepustowość ciężkiego korpusu i skracać jego ogon, a wydłużać medianę. Nie jest to potwierdzony ABBA wpływ samej wartości budżetu; różnią się też uruchomienia i składanie skończonej kolejki. Nie twierdzimy, że większe budżety nigdy nie pomagają, ani że ustalono uniwersalne optimum. Domyślne 8192 pozostaje bez zmian; wybór pod konkretny rozkład dokumentów i cel czasu odpowiedzi wymaga osobnego potwierdzenia.

Źródła: `screen-batch16384-max-300w`, `screen-batch32768-max-300w`, `inputs-batch-budget-max/http-reference-answers.json`.

## Bezpośrednia kontrola budżetu forward

Forwardy nie łączą różnych języków. Przy domyślnym limicie wiersza drzewa 32768 część partii z budżetami 16384/32768 mieści się w jednym wierszu danego języka, więc wcześniejsze przeglądy nie zawsze aktywują różnicę grupowania. Osobna sonda wymusza wiersze drzewa do 1024, budżet forward 16384 i długi tor od 4096. Jedno żądanie ma 128 pytań o wspólny stan, poniżej niezmienionych limitów wejścia.

Dwa powtórzenia na każdej binarce dały 64 spakowane wiersze. Stara binarka wykonała osiem forwardów po osiem wierszy; końcowa cztery, z 17/17/17/13 wierszami. Wszystkie 512 odpowiedzi na pytania są bitowo równe. Ponowny audyt CPU sprawdza pełne pola odpowiedzi, sumy i liczbę wierszy oraz odpowiadające im wpisy logu. Oryginalny runner wymagał HTTP200 przed zapisem, lecz nie zapisał osobnego pola statusu — to ograniczenie surowego formatu.

Sonda z `BASAL_HOST_TIMING` potwierdza semantykę budżetu i jej dziedziczenie przez długi tor. Wydajność opisują oddzielne pomiary HTTP; z mniejszej liczby wywołań nie wyprowadzamy skrócenia czasu odpowiedzi.

## Decyzje i kierunek dalszych prac

Włączona zmiana mini dotyczy trzech sprawdzonych wpisów GEMM. Zachowuje obliczenia i daje potwierdzone skrócenie czasu w zmierzonym zakresie M4096, bez zastępowania całej tablicy. Poprawka budżetu forward zapewnia zgodność konfiguracji z wykonaniem; pomiary nie uzasadniają podnoszenia domyślnych 8192 tokenów.

Dla ruchu produkcyjnego należy dobierać współbieżność do różnorodnych wejść i akceptowanego czasu kolejki. W mini/4.5B zwiększenie 128 do 512 klientów na 1024 różnych promptach nie podniosło QPS, a wydłużyło p50 około czterokrotnie. Wysokie QPS 44 powtarzanych promptów jest osobnym wynikiem współdzielenia pracy w partii i nie służy do ustalania pojemności ruchu z różnymi dokumentami. Limit dużych wejść pozostaje jawny; podniesienie go zapobiega 529 w benchmarku, lecz nie usuwa kosztu oczekiwania.

Dalsza optymalizacja pojedynczych długich żądań powinna skupić się na dokładnym attention. Wyniki trybów z uproszczoną numeryką wskazują mierzalny koszt dokładniejszej ścieżki, lecz nie dowodzą, że można go usunąć bez zmiany obliczeń. Osobne uproszczenie Q/K jest interesującym kierunkiem mini; pełne tc-f16 ma zachowaną regresję jednej kontrolnej decyzji i nie może być przedstawiane jako równoważny domyślny tryb. Każda nowa zmiana kernela lub numeryki wymaga nowych wyników zgodności i oddzielnego pomiaru.

Nie włączono domyślnie nowych układów TC64/SW128: przeglądy nie wykazały przekonującego ogólnego zysku. Nie wprowadzano szerszej przebudowy planowania ani nowych mechanizmów przechwytywania grafów GPU. Nie wdrażano ani nie testowano zmian w usłudze produkcyjnej. Kampania nie potwierdza jeszcze pojemności dla otwartego napływu żądań ani stabilności w wielogodzinnym rzeczywistym ruchu.

Mieszane p95 przy ośmiu klientach jest konkretnym obszarem dalszych prac: 4.5B i max przegrywają tę miarę mimo lepszego QPS i mediany. Kolejny profil powinien rozdzielić czas obliczeń długich dokumentów od oczekiwania w długim torze i ocenić politykę współdzielenia GPU z krótkimi zadaniami. Ta kampania nie wyznacza ustawienia gwarantującego najlepszy kompromis dla dowolnego ruchu; potrzebny jest cel czasu odpowiedzi i docelowy rozkład dokumentów. Nie nazywamy tej przegranej usuniętą przez zmianę tablicy GEMM lub limitu przyjęć.

## Zachowane niepowodzenia i ograniczenia

- Pierwszy audyt jakości mini błędnie potraktował dwa oczekiwane odrzucenia walidacji upstream jako awarię; poprawiono kontrolę dokładnych ID i kolizję nazw logów. Pierwotny FAIL pozostał.
- Pierwszy pomiar mieszany 4.5B przy 32 klientach dostał 529 przy domyślnych czterech slotach. To błąd ustawienia eksperymentu względem celu pomiaru GPU, nie wynik przepustowości.
- Pierwsza próba surowego body dla modelu tekstowego oczekiwała bramki instalowanej tylko dla mediów i przekroczyła timeout. Osobny poprawny audyt dotyczy rozwiniętego wejścia.
- Pierwszy korpus 1.5-mini miał pytania z 11 opcjami przy limicie starego modelu 10. Zachowano FAIL; osobny nowy pomiar jawnie filtruje nieobsługiwane wejścia i podaje ich ID.
- Pierwsza próba integracji nie miała katalogu tmpfs i zakończyła się przed właściwym pomiarem. Poprawione uruchomienie ma nowy katalog wyniku.
- Wcześniejsze guardy zatrzymały sesje po osiągnięciu własnego progu temperatury. Próby na 250 W z automatycznym wentylatorem, w tym z niższym celem sterownika, pozostały częściowe. Próba z wentylatorem 85% została jawnie przerwana przy zmianie na 100%. Ich dane nie zostały zastąpione ani połączone z nowymi statystykami.
- Pełne przestrojenie tablicy mini dawało zysk przy 4096, lecz regresje krótszych żądań i przepustowości; odrzucono je. Włączono tylko trzy sprawdzone wpisy.
- Mini `tc-pv1` miało niejednoznaczny mały zysk przy zakłóceniach przygotowaniem danych CPU; nie promowano tego wyniku. Warianty liczbowe mają odrębne kontrole jakości i nie zastępują domyślnych obliczeń.

Nie wykonano dopasowanego A/B starego basal-1.5-max na tych samych wejściach i warunkach. Różnicy wobec historycznej tabeli 1.5 nie nazywamy regresją. Wyniki nie obejmują wszystkich modeli omni, mediów, innych GPU, otwartego napływu żądań ani wielogodzinnego produkcyjnego ruchu. Wysoki QPS powtarzanych promptów nie jest dowodem takiej samej przepustowości różnych dokumentów. Zachowanie dyskretnych decyzji w małym korpusie FP32 nie dowodzi ogólnej równości jakości.

Część wcześniejszych metod nie zapisywała SHA-256 runnera. Ich parametry, odpowiedzi i zachowane skrypty pozostają w archiwum, a `run-index.json` jawnie pokazuje brak zapisanego odcisku źródła. Dla metod z zapisanym SHA przygotowanie archiwum wymaga obecności źródła o dokładnie tej sumie. Nie przypisujemy starszym próbom nieodnotowanej tożsamości runnera.

Modele były uruchamiane osobno, z zatrzymaniem poprzedniego serwera. Wyników QPS różnych modeli nie sumujemy; kampania nie mierzy jednoczesnego ruchu wielu modeli rezydujących na tej samej karcie.

## Weryfikacja kodu i przebiegu

Końcowa binarka `f318…` ma zgodne SHA-256 wszystkich ośmiu zapisanych plików źródłowych i konfiguracji. Zachowano kontrole PASS: `cargo fmt`, Clippy i Rustdoc dla Linux bez GPU oraz CUDA ze wszystkimi features, kompilację kerneli dla 80/89/90/90a, `cargo deny` i kontrolę wyjątku `paste`. Kontrole zależności wykorzystano ponownie przy niezmienionych manifestach i lockfile. Pełne komendy, logi i sumy są w `verification/`. Kompilacja nie jest dowodem zachowania modelu na innych kartach; nie wykonano nowego runtime Metal ani nowego pomiaru H100.

Końcowy audyt pięciu sesji z fabrycznym limitem mocy jest PASS: wszystkie zakończyły zadania normalnie, zatrzymały własny kontener i przywróciły 300 W, cel sterownika 85°C oraz automatyczne sterowanie wentylatorem. Najwyższa zarejestrowana temperatura tych sesji wyniosła 69°C. Stan czterech monitorowanych usług pozostawał prawidłowy; nie odnotowano nowego błędu GPU/OOM/sprzętu ani restartu hosta. To dowód dla tej kampanii, a nie rozpoznanie przyczyny wcześniejszych restartów lub gwarancja przyszłej stabilności.

Interaktywne porównanie domyślnych trybów jest w `reproduce/benchmark-comparison.html`. Pełne dane przeszły kontrolę widoku desktop 728 px w ciemnym motywie i telefonu 360 px w jasnym; selektor liczby klientów działał, bez błędów konsoli. Przed archiwizacją stronę odtworzono z opublikowanych podsumowań i potwierdzono zgodność bajtową.

## Archiwum i odtworzenie wyników

Dane są opublikowane w publicznym HF Storage Bucket `itsoltech/basal-rs`, pod nowym, wcześniej pustym prefiksem. Nie nadpisywano istniejących archiwów. Lokalna kopia archiwum pozostaje zachowana.

| Pole | Wartość |
|---|---|
| URI | `hf://buckets/itsoltech/basal-rs/measurements/optimization-basal2-ada-2026-10-15/7138d61f57f85ef7e0ef48a20a435d28ae2871ea87093e336db6b473a27a17b1/measurement-data.tar.gz` |
| SHA-256 archiwum | `7138d61f57f85ef7e0ef48a20a435d28ae2871ea87093e336db6b473a27a17b1` |
| Rozmiar gzip | 53 713 571 B |
| Weryfikacja 2026-10-16 | Pobranie z autoryzacją i anonimowo: SHA archiwum i 2436/2436 sum plików zgodne; 46 podsumowań odtworzonych identycznie, porównanie odtworzone identycznie bajtowo |

Archiwum zachowuje logiczny katalog `reports/optimization-basal2-ada-2026-10-15/`:

| Ścieżka względem tego katalogu | Zawartość |
|---|---|
| `runs/`, `execution-logs/`, `run-index.json` | 74 katalogi uruchomień, odpowiedzi, eksporty, protokoły, także pierwotne FAIL/GAP i mapowanie zapisanych SHA runnerów do zachowanych źródeł |
| `inputs*/` | Zamrożone wejścia, tablice użyte w pomiarach, referencje FP32 i odpowiedzi kontrolne, sumy i przypięte rewizje modeli |
| `quality-summary.json`, `*-summary.json` | Agregaty zgodności i wydajności do ponownego obliczenia bez GPU |
| `http-decision-summary.json` | Pełne różnice porównywanych pól, podział zmian wartości/etykiet i samej kolejności `selected`, prawdopodobieństwa i przykłady |
| `guards/`, `guards-summary.json` | Telemetria i potwierdzenia zakończenia/przywrócenia ustawień, z usuniętymi prywatnymi danymi hosta |
| `verification/` | Logi kontroli, sumy źródeł i PTX do odtworzenia porównania domyślnych kerneli |
| `reproduce/`, `protocols/` | Dokładne zachowane wersje runnerów, patche od wskazanej bazy, audytory, źródła sond i instrukcje; `reproduce/benchmark-comparison.html` jest interaktywnym podsumowaniem |

Publiczne dane przeszły przegląd prywatności: nie zawierają adresu ani nazwy hosta, prywatnych ścieżek, identyfikatorów maszyny, sekretów ani cen/kosztów sesji. Kwoty w syntetycznych pytaniach są częścią danych modelu. Wersje pakietów przypominające IPv4 oraz ogólne literały bind/loopback w dokładnych źródłach runnerów nie identyfikują maszyny. Wagi modeli, binarki, prywatne surowe diagnostyki hosta i cache kompilatora nie są rozprowadzane.

Pobranie do nowych katalogów; potrzebne są globalny `hf` CLI i Python. Nie uruchamiaj w już istniejących katalogach docelowych:

```bash
set -e
BASAL_OPT_DL=.cache/measurement-download/optimization-basal2-ada-2026-10-15
BASAL_OPT_ROOT=.cache/measurement-artifacts/optimization-basal2-ada-restored-2026-10-15
mkdir -p .cache/measurement-download .cache/measurement-artifacts
mkdir "$BASAL_OPT_DL" "$BASAL_OPT_ROOT"
hf buckets cp "hf://buckets/itsoltech/basal-rs/measurements/optimization-basal2-ada-2026-10-15/7138d61f57f85ef7e0ef48a20a435d28ae2871ea87093e336db6b473a27a17b1/measurement-data.tar.gz" "$BASAL_OPT_DL/measurement-data.tar.gz"
printf '%s  %s\n' "7138d61f57f85ef7e0ef48a20a435d28ae2871ea87093e336db6b473a27a17b1" "$BASAL_OPT_DL/measurement-data.tar.gz" | sha256sum -c -
tar -xzf "$BASAL_OPT_DL/measurement-data.tar.gz" -C "$BASAL_OPT_ROOT"
(cd "$BASAL_OPT_ROOT" && sha256sum -c SHA256SUMS)
BASAL_OPT_DATA="$BASAL_OPT_ROOT/reports/optimization-basal2-ada-2026-10-15"
python3 "$BASAL_OPT_DATA/reproduce/summarize-paired-targets.py" --measurement-artifacts "$BASAL_OPT_DATA/runs/paired-max-attention-300w" --out max-paired-audit.json
python3 "$BASAL_OPT_DATA/reproduce/summarize-basal2-optimize.py" --measurement-artifacts "$BASAL_OPT_DATA/runs/screen-batch32768-max-300w" --http-reference "$BASAL_OPT_DATA/inputs-batch-budget-max/http-reference-answers.json" --out max-budget-audit.json
```

Na macOS użyj `shasum -a 256 -c` zamiast `sha256sum -c`. Pozostałe polecenia audytu jakości, odpowiedzi, PTX, sondy forward i telemetrii są w `reproduce/README.md`. `--measurement-artifacts` jest konwencją tych skryptów Python, nie opcją CLI basal. Nowy pomiar wymaga nowego katalogu i manifestu rzeczywiście zbudowanych binarek; starych manifestów i zakończonego prefiksu HF nie zmieniamy.

## Zakończenie i zasoby

Po potwierdzeniu obu pobrań HF usunięto wyłącznie własny, zatrzymany kontener benchmarku i katalog roboczy sesji, po sprawdzeniu ich tożsamości, znacznika własności oraz braku montowań w innych kontenerach. Identyfikatory, obrazy, stany uruchomienia i montowania 12 pozostałych kontenerów pozostały zgodne z kontrolą przed tą operacją. Nie usuwano ich wolumenów ani danych. Współdzielony obraz pozostał zachowany.

Końcowa kontrola potwierdziła brak klientów GPU, limit 300 W, cel sterownika 85°C i automatyczne sterowanie wentylatorem. Cztery monitorowane usługi były zdrowe. Zastany nieaktywny backend produkcji pozostał nieaktywny; kampania nie wdrażała nowej wersji usługi. Prywatne surowe dane i kopia archiwum pozostają lokalnie, poza Gitem i publicznym magazynem.
