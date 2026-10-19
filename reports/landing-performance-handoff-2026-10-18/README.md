# Wydajność i zasoby basal-rs względem upstream do landing page

Raport do przygotowania treści i wykresów landing page. Stan dowodów: 18 października 2026. Kod zebrany w `main` jako `0da3769`; poszczególne pomiary pochodzą z opisanych niżej wcześniejszych binarek. To zestawienie istniejących badań, bez nowej sesji GPU.

## Najważniejsze wyniki

- **Ultra na H100, ruch mieszany C32:** o **27,6% więcej żądań/s**, o **35,6% niższe p95 dużych dokumentów**, o **29,4% mniej RAM PSS**, o **16,7% mniej VRAM procesu** i o **21,8% mniej czasu CPU na żądanie** niż upstream BF16 eager. To jeden jawny profil i jeden zamrożony korpus.
- **Basal 2.0 na RTX 6000 Ada:** pojedyncze pytania mają o **6–27% krótszy czas** w zmierzonych punktach; krótkie wejścia przy C32 osiągają **2,22–2,58× przepustowości** upstream fast BF16.
- **Omni 1.0 na RTX 6000 Ada:** **2,83× krótszy czas** dla mini i **2,97×** dla max dla wejść PL po 32768 tokenów. To konkretne punkty drabiny, nie mnożnik dla każdego requestu.
- **Start Ultra po optymalizacji loadera:** **11,51 s zamiast 19,52 s** upstream, przy wagach już obecnych lokalnie. Start serwera i czas odpowiedzi API to osobne metryki.

Własne wartości FP16 i referencje BF16 opisują kompletne konfiguracje produktów. Zgodność z FP32 oznacza odtworzenie wyników referencji na badanych wejściach, a nie „100% trafności” modelu na dowolnych danych.

**Zakres zasobów:** porównywalną pełną drabinę CPU/RAM/VRAM mamy dla Ultra. Nie przenosimy tych oszczędności na inne modele.

Źródło: [ultra-long-documents-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-long-documents-2026-10-18/README.md), [precision-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/precision-performance-ada-2026-10-16/README.md), [omni-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/omni-performance-ada-2026-10-16/README.md), [ultra-loader-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-loader-2026-10-18/README.md).

## Ultra na H100 — bieżący profil C32

Model `basal-omni-1.0-ultra`, NVIDIA H100 PCIe 80 GB, tekst. 32 różne krótkie zadania 2.1 i dwa dokumenty po 8192 tokeny, jeden duży na 16 krótkich. „C32” oznacza 32 klientów wysyłających następne żądanie po odebraniu poprzedniej odpowiedzi.

basal-rs: HRRN, partia 16384 tokeny, tor długi od progu 4096, przydział 5000 ms; FP16 z odczytem FP32. Upstream v2.1.0: BF16 eager, partia 4096. Cache stanu i prefiksu wyłączone, orders=1. [Gotowy YAML](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/serve.ultra-h100.example.yml).

Wartości są medianą metryk dwóch zachowanych startów na silnik; pierwszy start każdego silnika wyłączono zgodnie z metodą. RAM/VRAM to mediany średnich ważonych czasem próbek fazy. P95 dużych: 22 odpowiedzi na start, 44 zachowane na silnik, tylko dwa różne długie wejścia.

**Kompromis względem naszego FIFO:** nowy profil ma o 45,6% niższe p95 dużych dokumentów, ale mediana krótkich jest o 11,5% wyższa; p95 krótkich spada o 24,2%, QPS o 0,4%. Oba profile używają tej samej binarki. Nie jest to przyspieszenie samych kerneli ani zmiana globalnych ustawień domyślnych.

Audyt końcowy: AUDITED, 4536 poprawnych odpowiedzi HTTP, 2990 natywnych porównań odpowiedzi bez różnic bitowych. Odrzucone eksperymenty kodu i przerwana seria pozostają w źródłowym archiwum.

Źródło: [ultra-long-documents-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-long-documents-2026-10-18/README.md).

| Metryka | basal-rs profil | upstream BF16 eager | Różnica vs upstream |
| --- | --- | --- | --- |
| Przepustowość całego miksu | 1,389 żądań/s | 1,089 żądań/s | +27,6% przepustowości |
| P95 dużych dokumentów 8192 | 17,51 s | 27,17 s | −35,6% |
| Mediana dużych dokumentów 8192 | 15,88 s | 24,91 s | −36,3% |
| P95 krótkich zadań w miksie | 24,48 s | 38,38 s | −36,2% |
| Mediana krótkich zadań w miksie | 23,85 s | 31,42 s | −24,1% |
| CPU na ukończone żądanie | 721,4 ms CPU | 923,1 ms CPU | −21,8% |
| RAM procesu i potomków | 1,242 GiB PSS | 1,759 GiB PSS | −29,4% |
| VRAM procesu | 62,597 GiB | 75,105 GiB | −16,7% |
| Średnie zajęcie CPU | 1,002 rdzenie | 1,005 rdzenie | około jednego rdzenia po obu stronach |

## Ultra — drabina C1, C2, C8, C16, C32, C128

Poniższa osobna kampania używa **wcześniejszej binarki R15, FIFO i wyłączonego toru długiego**; nie opisuje profilu HRRN/5000 z poprzedniej sekcji. GPU H100 PCIe 80 GB; oba silniki danej pary działały na tej samej maszynie. Tekst, batch native/upstream 16384/4096, dwa powtórzenia. To osobna sesja od późniejszego profilu C32. Pełną serię ponownie wykonano do C128; do każdej fazy wliczono opróżnienie wszystkich rozpoczętych żądań.

Każda para liczb poniżej ma kolejność **basal-rs / upstream BF16 eager**. Oba silniki zajmowały średnio około jednego rdzenia podczas obciążenia. Oszczędność CPU dotyczy pracy na ukończone żądanie, nie radykalnie mniejszej liczby rdzeni.

**Status całej kampanii: FAIL ze znanym zakresem.** Wszystkie 48 okien obciążenia i odpowiedzi HTTP są kompletne; brak 529 i OOM. Przy C128 wykryto 80 odpowiedzi z innymi bitami prawdopodobieństw, bez zmiany decyzji/tekstu/danych wynikowych. Brakuje też trzech odcinków końcowej bezczynności. Nie wolno opisywać tej kampanii jako „zero regresji” lub „bitowo identyczne do C128”. C1–C32 zachowały porównywane bity.

C128 jest próbą nasycenia, nie rekomendowaną konfiguracją latencji produkcyjnej.

Źródło: [ultra-resources-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-resources-2026-10-17/README.md).

| Ruch | C | Żądania/s B / U | Zysk QPS | CPU ms/żąd. B / U | PSS GiB B / U | VRAM GiB B / U |
| --- | --- | --- | --- | --- | --- | --- |
| krótkie | 1 | 1,920 / 1,596 | +20,3% | 520,7 / 629,6 | 1,128 / 1,752 | 60,961 / 65,105 |
| krótkie | 2 | 1,913 / 1,557 | +22,9% | 523,6 / 645,8 | 1,128 / 1,755 | 60,961 / 65,105 |
| krótkie | 8 | 2,068 / 1,686 | +22,7% | 484,1 / 596,1 | 1,129 / 1,758 | 62,197 / 67,940 |
| krótkie | 16 | 2,159 / 1,797 | +20,1% | 463,7 / 559,5 | 1,130 / 1,760 | 62,508 / 68,094 |
| krótkie | 32 | 2,203 / 1,921 | +14,7% | 454,4 / 523,7 | 1,133 / 1,761 | 62,523 / 68,094 |
| krótkie | 128 | 2,262 / 2,021 | +11,9% | 442,4 / 498,0 | 1,143 / 1,770 | 63,008 / 68,094 |
| mieszany | 1 | 1,323 / 1,007 | +31,4% | 755,9 / 996,6 | 1,052 / 1,757 | 64,711 / 75,105 |
| mieszany | 2 | 1,319 / 0,991 | +33,1% | 759,0 / 1012,8 | 1,053 / 1,760 | 64,711 / 75,105 |
| mieszany | 8 | 1,386 / 1,036 | +33,8% | 722,1 / 969,5 | 1,054 / 1,764 | 64,711 / 75,105 |
| mieszany | 16 | 1,416 / 1,080 | +31,1% | 706,5 / 929,4 | 1,056 / 1,766 | 64,711 / 75,105 |
| mieszany | 32 | 1,436 / 1,121 | +28,1% | 696,8 / 895,4 | 1,059 / 1,767 | 64,711 / 75,105 |
| mieszany | 128 | 1,464 / 1,150 | +27,3% | 683,4 / 873,9 | 1,078 / 1,776 | 64,711 / 75,105 |

### Czas odpowiedzi w tej samej drabinie

Milisekundy; para **basal-rs / upstream**. To pełny czas HTTP z kolejką, a nie czas pojedynczego forwardu na GPU. Wzrost C zwiększa przepustowość, lecz także oczekiwanie. Nie przeliczaj QPS jako odwrotności mediany.

Źródło: [ultra-resources-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-resources-2026-10-17/README.md).

| Ruch | C | p50 ms B / U | p95 ms B / U |
| --- | --- | --- | --- |
| krótkie | 1 | 425 / 475 | 990 / 1307 |
| krótkie | 2 | 907 / 1330 | 1544 / 2258 |
| krótkie | 8 | 3797 / 5191 | 4172 / 5761 |
| krótkie | 16 | 7282 / 9419 | 7662 / 10144 |
| krótkie | 32 | 14504 / 17121 | 14687 / 18361 |
| krótkie | 128 | 50755 / 69345 | 72488 / 73301 |
| mieszany | 1 | 426 / 477 | 4487 / 6823 |
| mieszany | 2 | 907 / 1338 | 5478 / 7731 |
| mieszany | 8 | 4102 / 5386 | 8377 / 12080 |
| mieszany | 16 | 11272 / 15687 | 13746 / 21806 |
| mieszany | 32 | 18471 / 29697 | 31069 / 37370 |
| mieszany | 128 | 75292 / 121783 | 121466 / 133047 |

## Basal 2.0 na RTX 6000 Ada

### Jedno pytanie, jeden klient, model rozgrzany

Profil native balanced FP16, porównanie z **upstream fast BF16**, który używa kompilacji i CUDA Graphs. RTX 6000 Ada 48 GB, 300 W. Kolumna tokenów oznacza cały prompt modelu, a nie liczbę znaków dokumentu. Podano wybrane długości; pełna drabina ma 25 punktów, wszystkie z niższą medianą native. Mediany trzech zachowanych powtórzeń; 30 zmierzonych odpowiedzi na silnik/punkt. Dane pochodzą z 16 października; nie powtarzano wszystkich punktów na końcowym commicie `main`.

Czas zaczyna się przed serializacją żądania, kończy po odczycie pełnego body; wyłącza ładowanie, kompilację i przygotowanie CUDA Graphs. Jednostka: **ms**.

Źródło: [precision-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/precision-performance-ada-2026-10-16/README.md).

| Model | Tokeny | basal-rs ms | upstream fast BF16 ms | Redukcja czasu |
| --- | --- | --- | --- | --- |
| basal-2.0-mini | 128 | 7,437 | 9,415 | 21,0% |
| basal-2.0-mini | 512 | 14,282 | 16,929 | 15,6% |
| basal-2.0-mini | 2048 | 58,781 | 62,657 | 6,2% |
| basal-2.0-mini | 8192 | 282,400 | 303,512 | 7,0% |
| basal-2.0-4.5B | 128 | 16,605 | 19,215 | 13,6% |
| basal-2.0-4.5B | 512 | 43,568 | 51,058 | 14,7% |
| basal-2.0-4.5B | 2048 | 164,333 | 205,647 | 20,1% |
| basal-2.0-4.5B | 8192 | 768,250 | 901,503 | 14,8% |
| basal-2.0-4.5B | 32768 | 5052,832 | 5521,409 | 8,5% |
| basal-2.0-max | 128 | 33,048 | 37,507 | 11,9% |
| basal-2.0-max | 512 | 96,069 | 122,173 | 21,4% |
| basal-2.0-max | 2048 | 350,638 | 474,900 | 26,2% |
| basal-2.0-max | 8192 | 1641,471 | 1980,929 | 17,1% |
| basal-2.0-max | 32768 | 9624,414 | 10775,159 | 10,7% |

### Przepustowość Basal 2.0 względem fast BF16

Ten sam profil balanced i GPU. Krótkie wejścia: 1024 różne payloady z 44 stanów i nonce; nie 1024 niezależne przykłady jakości. Mediana trzech zachowanych powtórzeń. Ruch mieszany jest innym korpusem niż zadania 2.1 Ultra, dlatego nie porównuj QPS między tabelami jako rankingu modeli.

**Zachowane przegrane:** w ruchu mieszanym C8 p95 4.5B wyniosło 5219,989 vs 4128,307 ms, a max 12053,225 vs 8501,476 ms; najdłuższe dokumenty także miały gorsze p95. Większy QPS nie oznacza krótszej każdej odpowiedzi. Punkt mini 4,536× przy mieszanym C128 wolno cytować wyłącznie z takim opisem warunków.

Źródło: [precision-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/precision-performance-ada-2026-10-16/README.md).

| Model | Ruch | C | basal-rs ż./s | upstream ż./s | Iloraz QPS |
| --- | --- | --- | --- | --- | --- |
| basal-2.0-mini | krótkie | 1 | 139,982 | 91,731 | 1,526× |
| basal-2.0-mini | krótkie | 8 | 215,354 | 130,961 | 1,644× |
| basal-2.0-mini | krótkie | 32 | 269,391 | 121,500 | 2,217× |
| basal-2.0-mini | krótkie | 128 | 313,885 | 134,397 | 2,336× |
| basal-2.0-mini | mieszany | 8 | 49,089 | 26,189 | 1,874× |
| basal-2.0-mini | mieszany | 128 | 119,620 | 26,369 | 4,536× |
| basal-2.0-4.5B | krótkie | 1 | 53,051 | 41,104 | 1,291× |
| basal-2.0-4.5B | krótkie | 8 | 76,435 | 41,633 | 1,836× |
| basal-2.0-4.5B | krótkie | 32 | 92,010 | 36,675 | 2,509× |
| basal-2.0-4.5B | krótkie | 128 | 102,990 | 43,622 | 2,361× |
| basal-2.0-4.5B | mieszany | 8 | 7,210 | 4,601 | 1,567× |
| basal-2.0-4.5B | mieszany | 128 | 9,488 | 4,725 | 2,008× |
| basal-2.0-max | krótkie | 1 | 23,215 | 17,334 | 1,339× |
| basal-2.0-max | krótkie | 8 | 32,398 | 17,742 | 1,826× |
| basal-2.0-max | krótkie | 32 | 40,831 | 15,804 | 2,584× |
| basal-2.0-max | krótkie | 128 | 48,883 | 19,805 | 2,468× |
| basal-2.0-max | mieszany | 8 | 3,398 | 2,146 | 1,583× |
| basal-2.0-max | mieszany | 128 | 4,541 | 2,131 | 2,131× |

## Omni na RTX 6000 Ada

### Omni 1.0 mini i max — jedno pytanie PL

Decoder native FP16 z akumulacją/softmaxem FP32; wieże mediów BF16. Upstream **eager BF16**; sprawdzone oficjalne warianty fast nie uruchamiały tych modeli. Nie nazywaj tego porównaniem z fast.

128–32768 tokenów, cache stanu/prefiksu wyłączone, orders=1. Trzy zachowane powtórzenia, cztery odpowiedzi na punkt i powtórzenie. Wybrane punkty PL poniżej; raport źródłowy zawiera też EN i długości pośrednie. **ms**, pełny czas HTTP. Nie potwierdzono tym pełnych limitów kontekstu 131072/262144 ani jakości FP32 każdego punktu drabiny.

Źródło: [omni-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/omni-performance-ada-2026-10-16/README.md).

| Model | Tokeny | basal-rs ms | upstream BF16 ms | Redukcja czasu |
| --- | --- | --- | --- | --- |
| basal-omni-1.0-mini | 128 | 22,20 | 38,60 | 42,5% |
| basal-omni-1.0-mini | 512 | 51,60 | 59,30 | 13,0% |
| basal-omni-1.0-mini | 2048 | 212,40 | 289,10 | 26,5% |
| basal-omni-1.0-mini | 8192 | 1355,10 | 1829,00 | 25,9% |
| basal-omni-1.0-mini | 32768 | 6620,50 | 18736,10 | 64,7% |
| basal-omni-1.0-max | 128 | 44,80 | 58,40 | 23,3% |
| basal-omni-1.0-max | 512 | 119,30 | 136,40 | 12,5% |
| basal-omni-1.0-max | 2048 | 502,00 | 742,30 | 32,4% |
| basal-omni-1.0-max | 8192 | 3031,40 | 4529,40 | 33,1% |
| basal-omni-1.0-max | 32768 | 14856,70 | 44103,10 | 66,3% |

### Omni 1.0 — przepustowość krótkich wejść

128 różnych wejść, 512 żądań na serię i poziom C, mediana trzech zachowanych powtórzeń. To proste pytania, a nie wieloetapowe zadania 2.1.

W osobnej serii FIFO ruchu mieszanego C8/C32 osiągnięto 1,72–1,80× QPS upstream. P95 dużych przy C32 pozostało jednak gorsze o około 3,4% dla mini i 2,8% dla max. Nie wolno przenosić poprawy toru dużych dokumentów Ultra na te modele.

Źródło: [omni-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/omni-performance-ada-2026-10-16/README.md).

| Model | C | basal-rs ż./s | upstream ż./s | Iloraz QPS |
| --- | --- | --- | --- | --- |
| basal-omni-1.0-mini | 1 | 45,80 | 25,58 | 1,79× |
| basal-omni-1.0-mini | 8 | 79,27 | 69,28 | 1,14× |
| basal-omni-1.0-mini | 32 | 87,50 | 81,88 | 1,07× |
| basal-omni-1.0-max | 1 | 22,69 | 17,65 | 1,29× |
| basal-omni-1.0-max | 8 | 33,64 | 31,19 | 1,08× |
| basal-omni-1.0-max | 32 | 38,28 | 32,95 | 1,16× |

### Omni 1.1 max i Ultra — jedno pytanie PL

Native FP16, upstream v2.1.0 BF16 eager. Trzy zachowane powtórzenia; **ms**, pełny HTTP request→response, model rozgrzany. 1.1 max: cztery mierzone odpowiedzi na punkt/start, Ultra: pięć. To osobne serie pojedynczych pytań; nie są pomiarem profilu długiego toru C32.

Omni 1.1 max w zadaniach 2.1 osiągnął C1: 1,778 vs 1,588 ż./s; C8: 1,908 vs 1,160; C32: 2,004 vs 1,508. Odpowiada to około 12%, 64% i 33% więcej QPS. Korpus ma 32 różne stany; zadanie może obejmować wiele decyzji i drugą rundę extract.

Źródło: [upstream-2.1-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/upstream-2.1-2026-10-17/README.md), [ultra-attention-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-attention-2026-10-17/README.md).

| Model | GPU | Tokeny | basal-rs ms | upstream BF16 ms | Redukcja czasu |
| --- | --- | --- | --- | --- | --- |
| basal-omni-1.1-max | RTX 6000 Ada | 128 | 47,24 | 58,83 | 19,7% |
| basal-omni-1.1-max | RTX 6000 Ada | 512 | 130,26 | 141,04 | 7,6% |
| basal-omni-1.1-max | RTX 6000 Ada | 2048 | 620,27 | 756,99 | 18,1% |
| basal-omni-1.1-max | RTX 6000 Ada | 8192 | 3011,23 | 4587,24 | 34,4% |
| basal-omni-1.0-ultra | H100 PCIe | 128 | 54,42 | 85,64 | 36,5% |
| basal-omni-1.0-ultra | H100 PCIe | 512 | 138,20 | 168,26 | 17,9% |
| basal-omni-1.0-ultra | H100 PCIe | 2048 | 824,67 | 977,23 | 15,6% |
| basal-omni-1.0-ultra | H100 PCIe | 8192 | 4631,73 | 7083,43 | 34,6% |

### Obrazy i dźwięk

**ms**, sekwencyjny request→response. Dane obejmują różne korpusy; tabela pokazuje osobne pary, nie ranking modeli. Ultra obsługuje w tej wersji tekst i obraz, **nie audio**. Jednego obrazu Ultra nie przedstawiaj jako dowodu „każdy obraz 2× szybciej”.

Źródło: [omni-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/omni-performance-ada-2026-10-16/README.md), [upstream-2.1-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/upstream-2.1-2026-10-17/README.md), [ultra-attention-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-attention-2026-10-17/README.md).

| Model | GPU | Zakres | basal-rs ms | upstream BF16 ms | Interpretacja |
| --- | --- | --- | --- | --- | --- |
| Omni 1.0 mini | RTX 6000 Ada | 34 wejścia obraz/audio/miks, C1 | 127,4 | 201,0 | zbiorcza mediana serii; 34/34 niższe |
| Omni 1.0 max | RTX 6000 Ada | 34 wejścia obraz/audio/miks, C1 | 146,2 | 168,3 | zbiorcza mediana serii; 34/34 niższe |
| Omni 1.1 max | RTX 6000 Ada | obraz + zadania, C1 | 515,6 | 553,8 | jedno z 3 wejść serii |
| Omni 1.1 max | RTX 6000 Ada | audio + zadania, C1 | 554,6 | 608,5 | jedno z 3 wejść serii |
| Omni 1.1 max | RTX 6000 Ada | obraz + audio + zadania, C1 | 991,8 | 1012,7 | jedno z 3 wejść serii |
| Ultra | H100 PCIe | jeden obraz COCO + pytanie, C1 | 699,28 | 1339,33 | 48 zachowanych powtórzeń jednego wejścia |

## Start serwera Ultra

H100 PCIe, trzy zachowane starty na wariant. Wagi były już lokalnie; nie czyszczono cache dysku/bibliotek. Licznik: utworzenie procesu → poprawna odpowiedź health, odpytywanie co około 0,5 s. Starty upstream nastąpiły po przeplatanym porównaniu starego i nowego native; jest to opisowe porównanie, nie pełne losowe przeplatanie trzech wariantów.

Nowy loader: **41,0% krótszy start**, około **38,0% mniej czasu CPU startu** i **95,1% niższy obserwowany szczyt RSS** względem upstream. Ostatnia wartość opisuje szczyt procesu obejmujący mapowane/rezydentne strony wag; **nie oznacza 95% mniejszego wymaganego RAM ani pamięci gorącego API**.

Nie publikuj starego wyniku 74,8–75,6 s jako bieżącego czasu startu basal-rs. Nowy loader zachował 561 par rekordów eksportu; kontrolne odpowiedzi HTTP również były bitowo zgodne.

Źródło: [ultra-loader-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-loader-2026-10-18/README.md).

| Silnik | Gotowość s | CPU startu s pracy rdzeni | Szczyt RSS GiB |
| --- | --- | --- | --- |
| basal-rs nowy loader | 11,51 | 11,49 | 2,921 |
| upstream BF16 eager | 19,52 | 18,52 | 59,335 |
| basal-rs poprzedni loader R15 | 75,56 | 75,11 | 5,604 |

## Jakość odpowiedzi względem FP32

Kolejność odległości: **basal-rs / upstream BF16**. MAE to średni błąd bezwzględny, TV to odległość rozkładów prawdopodobieństw; niższe = bliżej FP32. To różne metryki i różne korpusy, **nie wolno ich połączyć w jeden „wynik jakości”**. Mniejszy średni błąd nie oznacza poprawy każdego wejścia.

FP32 jest referencją tych samych checkpointów, nie zbiorem bezbłędnych odpowiedzi człowieka. Wynik 140/140 albo 199/199 oznacza zgodność na skończonej próbie. Nie jest to „100% accuracy” produktu. Dla Omni 1.1 max sama zgodność argmax nie obejmuje całej decyzji starszego act.

Końcowa kampania profilu Ultra C32 nie tworzyła nowej referencji FP32: sprawdziła zachowanie bitów odpowiedzi wcześniejszej binarki. Celowana kontrola 14 wejść zmiennych przy C128 dała funkcjonalną zgodność Ultra 14/14 vs BF16 13/14, ale nie jest niezależnym benchmarkiem jakości całej rodziny.

Źródło: [precision-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/precision-performance-ada-2026-10-16/README.md), [omni-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/omni-performance-ada-2026-10-16/README.md), [upstream-2.1-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/upstream-2.1-2026-10-17/README.md), [ultra-attention-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-attention-2026-10-17/README.md), [ultra-resources-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-resources-2026-10-17/README.md).

| Model | Próba | Zgodność z FP32 | Odległość B / U | Warunek / ograniczenie |
| --- | --- | --- | --- | --- |
| Basal 2.0 mini | niezależna rezerwa | 140/140 kategorii; BF16 139/140 | MAE prawd. 0,000403722 / 0,003126164 | 178 wejść; 38 score poza mianownikiem kategorii |
| Basal 2.0 4.5B | niezależna rezerwa | 140/140; BF16 139/140 | MAE 0,000329934 / 0,002167563 | bezpośredni FP32 długich: 6/6 |
| Basal 2.0 max | niezależna rezerwa | 140/140; BF16 140/140 | MAE 0,000155223 / 0,001060019 | bezpośredni upstream FP32 długich: GAP/OOM |
| Omni 1.0 mini | 199 głowic kontrolnych | 199/199; BF16 także 199/199 | średnia TV 0,00060193 / 0,00212067 | nie każdy punkt drabiny ma kontrolę FP32 |
| Omni 1.0 max | 199 głowic kontrolnych | 199/199; BF16 także 199/199 | TV 0,00065548 / 0,00180935 | długi FP32 z wcześniejszej H100; świeży Ada OOM |
| Omni 1.1 max | 191 pierwszych odczytów | 191/191 argmax; pełna zgodność API ma FAIL/GAP | TV 0,003017056 / 0,003495285 | starsze image-default/act: human vs accept |
| Ultra R15 | 161 pierwszych odczytów | 161/161; BF16 160/161 | TV 0,0006319363 / 0,0019138364 | BF16 zmienia jeden verify; skończony korpus |

## Sformułowania gotowe do użycia

Proponowany nagłówek: **„Więcej żądań na tej samej karcie, mniejsze zużycie zasobów”**.

Proponowany opis: „W pomiarze Omni Ultra na H100 basal-rs obsłużył o 27,6% więcej żądań na sekundę niż upstream BF16 eager, przy 29,4% mniejszym RAM PSS i 16,7% mniejszym VRAM procesu. Profil dużych dokumentów skrócił ich p95 o 35,6%.” Bezpośrednio obok zachowaj dopisek: „Tekstowy miks krótkich zadań i dokumentów 8192 tokeny, 32 klientów, dwa zachowane starty na silnik; szczegóły i dane pomiarowe w metodologii”.

Na osobnych kafelkach i wykresach użyj następujących ograniczonych twierdzeń. Do każdego wykresu przypnij raport źródłowy i konfigurację, a nie sam procent.

Źródło: [ultra-long-documents-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-long-documents-2026-10-18/README.md), [precision-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/precision-performance-ada-2026-10-16/README.md), [ultra-loader-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-loader-2026-10-18/README.md).

| Tekst | Wymagany kontekst | Dowód liczbowy |
| --- | --- | --- |
| 27,6% więcej żądań/s w ruchu mieszanym | Ultra, H100 PCIe, tekst, C32, HRRN/4096/5000, BF16 eager | 1,389 vs 1,089 ż./s; 2 zachowane starty |
| 35,6% niższe p95 dużych dokumentów | Ultra, H100, 8192 tokeny w tym samym miksie C32 | 17,51 vs 27,17 s; 44 duże obserwacje z 2 wejść |
| 29,4% mniej RAM i 16,7% mniej VRAM podczas pracy | Ultra w tym samym miksie C32; RAM = PSS, VRAM = proces | 1,242 vs 1,759 GiB PSS; 62,597 vs 75,105 GiB VRAM |
| 21,8% mniej czasu CPU na żądanie | Ultra, ten sam miks C32; obie strony około 1 rdzenia | 721,4 vs 923,1 ms CPU/żądanie |
| Do 2,58× przepustowości dla krótkich żądań C32 | Basal 2.0 na RTX 6000 Ada, balanced vs fast BF16 | max 40,831 vs 15,804 ż./s; mini 2,22×, 4.5B 2,51× |
| Ultra gotowe w 11,51 s zamiast 19,52 s | H100, wagi obecne lokalnie, proces → health | 41,0% krótszy start; nie zimny download |

## Jak opisywać i rysować porównania

- Wykres żądań/s: więcej = lepiej, procent = `(basal-rs / upstream − 1) × 100%`. Wykres czasu/pamięci/CPU na żądanie: mniej = lepiej, redukcja = `(1 − basal-rs / upstream) × 100%`.
- „2× przepustowości” to **100% więcej**, a „2× krótszy czas” to **50% mniej czasu**. Nie pisz „200% szybciej” dla mnożnika 2×.
- RAM podczas pracy = **PSS procesu i potomków**; VRAM = alokacja procesu z modelem/runtime/workspace. To nie minimalne wymagania pamięci i nie sam koszt HTTP. GiB = 2^30 bajtów.
- CPU = czas pracy rdzeni. CPU ms/żądanie opisuje koszt obsłużonej pracy; średnie zajęcie rdzeni i przydzielony limit wątków to osobne informacje.
- Czasy API: lokalny HTTP, bez TLS/proxy/Internetu, model już załadowany. Obejmują kolejkę i odczyt body; parsowanie JSON poza licznikiem. To nie czas generowania tokenów LLM; nie podawaj tokens/s.
- Nie łącz punktów różnych kampanii jedną linią „skalowania aktualnego profilu”. R15 C1–C128 i późniejszy HRRN/5000 C32 to osobne serie. Nowy loader zmierzył start, nie ponownie całą drabinę obciążenia.
- Basal 2.0 porównano do działającego **fast BF16**, Omni do działającego **eager BF16**. Native to określone FP16/FP32 profile bez zmiany checkpointów. Nie deklaruj porównania identycznej arytmetyki.
- Bezpieczny zakres zasobów w szerokiej historycznej drabinie Ultra: około **35–40% mniej PSS**, **6–14% mniej VRAM procesu**, **11–26% mniej CPU/żądanie**. To inne wyniki niż bieżący profil C32 i trzeba je tak podpisać.
- Nie publikuj: „zawsze szybszy”, „mniej RAM dla każdego modelu”, „100% accuracy”, „bitowo identyczny do C128”, „pełne 32k FP32 quality”, „95% mniej RAM wymagane do pracy”, „C128 bez wpływu na latencję”.

Pełne porównania pięciu kart dla Basal 2.0 są osobnym [raportem historycznym](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/perf-basal2-fivegpu-2026-10-14/README.md). Obejmują także przegrane mini na H100; nie zastępują nowszego pomiaru balanced na Ada i nie potwierdzają uniwersalnego zysku na każdej karcie.

Źródło: [precision-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/precision-performance-ada-2026-10-16/README.md), [omni-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/omni-performance-ada-2026-10-16/README.md), [ultra-long-documents-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-long-documents-2026-10-18/README.md), [ultra-resources-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-resources-2026-10-17/README.md), [ultra-loader-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-loader-2026-10-18/README.md).

## Źródła i odtworzenie

Każdy raport źródłowy zawiera przypięte rewizje modeli, wersje silników, pełną metodę, zachowane FAIL/GAP i publiczne archiwum z SHA-256. Poniższe odsyłacze GitHub są przypięte do commita zmergowanego do `main`; nowe zestawienie nie zastępuje raportów źródłowych.

- [precision-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/precision-performance-ada-2026-10-16/README.md): archiwum `hf://buckets/itsoltech/basal-rs/measurements/precision-performance-ada-2026-10-16/62a636ab0f1417603dbd8f3189e03121df24b662d777aaef70d9cd4411b1446d/measurement-data.tar.gz`.
- [omni-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/omni-performance-ada-2026-10-16/README.md): archiwum `hf://buckets/itsoltech/basal-rs/measurements/omni-performance-ada-2026-10-16/82aaf99af87532eaccfaf19e3fe5462b5e0d9b4f1ffcac6ccba7ba987ac8a0a7/measurement-data.tar.gz`.
- [upstream-2.1-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/upstream-2.1-2026-10-17/README.md): archiwum `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.1-2026-10-17/9a8dae4bc02b08a4f2bcaf50aec712710706fb746820aff89961d4924ac1b10d/measurement-data.tar.gz`.
- [ultra-attention-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-attention-2026-10-17/README.md): archiwum `hf://buckets/itsoltech/basal-rs/measurements/ultra-attention-2026-10-17/c298d99630c826cf6a1972aaf5ba477060f9f0c9e88427996213adad8b55f095/measurement-data.tar.gz`.
- [ultra-resources-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-resources-2026-10-17/README.md): archiwum `hf://buckets/itsoltech/basal-rs/measurements/ultra-resources-2026-10-17/47926913c632019d9d05c5461025c397ce45fb79ec5aec29cdff83d139c2cd02/measurement-data.tar.gz`.
- [ultra-loader-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-loader-2026-10-18/README.md): archiwum `hf://buckets/itsoltech/basal-rs/measurements/ultra-loader-2026-10-18/1a453f895e4cf547349e240fcd769115040fe793a78495d22c8d13d28df6791c/measurement-data.tar.gz`.
- [ultra-long-documents-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-long-documents-2026-10-18/README.md): archiwum `hf://buckets/itsoltech/basal-rs/measurements/ultra-long-documents-2026-10-18/c256b4f32d863f1ed54130662d6d3c72a9bf3457f276baaa1d1b4151a81fcdbc/measurement-data.tar.gz`.

Archiwa źródłowe były pobrane i zweryfikowane z logowaniem oraz anonimowo w poszczególnych kampaniach. Instrukcje: [docs/MEASUREMENT_ARTIFACTS.md](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/docs/MEASUREMENT_ARTIFACTS.md). To zestawienie nie uruchamia nowych pomiarów ani nie twierdzi, że powtórzono runtime każdego modelu na końcowym `main`.


Źródło: [precision-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/precision-performance-ada-2026-10-16/README.md), [omni-performance-ada-2026-10-16](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/omni-performance-ada-2026-10-16/README.md), [upstream-2.1-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/upstream-2.1-2026-10-17/README.md), [ultra-attention-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-attention-2026-10-17/README.md), [ultra-resources-2026-10-17](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-resources-2026-10-17/README.md), [ultra-loader-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-loader-2026-10-18/README.md), [ultra-long-documents-2026-10-18](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/ultra-long-documents-2026-10-18/README.md).
