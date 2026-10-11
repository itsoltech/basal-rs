# W8: wydajność none, zamknięty wycinek pomiaru

Nie wykazano powtarzalnego spadku przy zmierzonej rozdzielczości.
Osiem metryk gorszych dla B w obu parach rundy podstawowej przestało
być jednokierunkowo gorsze po wymaganej rundzie potwierdzenia.
To nie dowodzi zerowego narzutu ani ogólnego braku regresji.

## Metoda i zakres

2026-10-11, NVIDIA RTX 6000 Ada Generation 48 GB, GPU 0;
GPU 1 miał w dziewięciu CSV wykorzystanie maksymalnie 0% i pamięć
maksymalnie 4 MiB; profile nie mają CSV. A: binarka etapu replik; B: kandydat W8 bez
kwantyzacji, oba F16/fused z tą samą tabelą GEMM. Żądany zegar SM
1200 MHz, limit mocy 300 W, skonfigurowana maska 16 logicznych CPU.
Zamknięte run.json i rekord kontroli potwierdzają limit oraz żądanie
zegara; późniejsza pojedyncza obserwacja oryginalnego runnera potwierdza
tę samą maskę. Procesy potomne dziedziczą maskę; nie ma ciągłego
historycznego zapisu affinity każdego procesu. Wszystkie dziewięć CSV
GPU 0 ma maksimum 1200 MHz, a minimum wynosi 1185–1200 MHz.
Nie nazywamy zegara literalnie stałym.
Odciski binarek i zamkniętych plików znajdują się w `facts.json`.

`basal bench`: cztery sąsiednie pary AB/BA, świeży proces na przypadek.
44 przykłady, pierwsze 5 stanowi rozgrzewkę, 39 mierzone pozycje 5..43;
wszystkie indeksy unikalne, kompletne, z dodatnim skończonym czasem.
Żadna z trzech głównych metryk ani mediana ilorazów per pytanie nie
była gorsza dla B we wszystkich czterech parach, więc nie było
potwierdzenia bench. Nie zastępujemy tego zgodnością wszystkich czasów. Trafność bench
wynosiła 35/44 w każdym z ośmiu przypadków; wartości obu stron
zapisano przy każdej parze.

`basal benchmark`: dwie pary AB/BA w podstawowej kolejności A B B A,
po sygnale dokładnie jedna dodatkowa sekwencja A B B A. Każdy ze świeżych
procesów mierzył po 900 żądań sekwencyjnych c1, mieszanych c1 i
mieszanych c16. Łącznie 21 600 mierzonych odpowiedzi 200, bez błędów,
anulowań i zmian odpowiedzi względem pierwszego sukcesu we własnym
przebiegu. To stabilność wewnątrz przebiegu; bajtowa zgodność eksportów
none z poprzednią binarką należy do osobnej kampanii poprawności.

Reguła ustalona przed pomiarem: utrzymany sygnał wymaga gorszego B
we wszystkich ustalonych parach obu rund. Nie stosowano progu,
p-value, wykluczania przebiegów, wybierania grup ani nowego parowania.
W `facts.json` zapisano wszystkie metryki, każdą wartość A i B,
B−A i B/A dla każdej pary, również gdy B jest lepsze.

## Sygnały podstawowe i potwierdzenie

Tabela pokazuje zmianę B/A−1 dla wszystkich ośmiu sygnałów podstawowych.
RPS/QPS jest gorsze, gdy ujemne; latency i kolejka, gdy dodatnie.
Pozostałe metryki, które nie dały pierwotnego sygnału, pozostają w JSON.

| Metryka | Podstawowa para 0 | Podstawowa para 1 | Potwierdzenie para 0 | Potwierdzenie para 1 |
|---|---:|---:|---:|---:|
| mixed-c16 successful_latency_ms.p95 | +0.3027% | +0.0377% | -3.4473% | -0.7059% |
| mixed-c16 queue_ms.p50 | +0.1871% | +3.2531% | -18.7690% | -22.1395% |
| mixed-c16 queue_ms.p99 | +0.0588% | +0.6337% | -0.3036% | -0.8107% |
| mixed-c16 successful_requests_per_second | -0.0055% | -0.2647% | +0.5722% | +1.1220% |
| mixed-c16 successful_questions_per_second | -0.0055% | -0.2647% | +0.5722% | +1.1220% |
| sequential-c1 queue_ms.p50 | +11.9420% | +10.3018% | -33.1663% | -7.0929% |
| sequential-c1 successful_requests_per_second | -0.0005% | -0.5849% | +0.5945% | -0.4271% |
| sequential-c1 successful_questions_per_second | -0.0005% | -0.5849% | +0.5945% | -0.4271% |

Żaden z tych ośmiu sygnałów nie utrzymał się we wszystkich czterech
parach. Zmienność nie służy do rozgrzeszania utrzymanego sygnału;
tutaj warunek utrzymania faktycznie nie zaszedł. Nie ustalono przyczyny
każdej pojedynczej różnicy ani wpływu kodu na każdą partię. Mediany
latency mixed-c16 i mixed-c1 oraz część metryk bench były gorsze dla B
w 3/4 parach; pełne liczby kierunków i wszystkie wartości pozostają
w JSON. Nie zmienia to ustalonej wcześniej reguły utrzymania sygnału.

## Profilowanie i telemetria

Poza głównymi pomiarami wykonano po jednym profilu A i B, 40 przykładów,
batch 1. Czas profilowany na przykład: A 35,09518245 ms,
B 34,95190515 ms. Suma sekcji: A 35,006973675 ms, B 34,86638685 ms.
Wszystkie sekcje i ich różnice są zapisane w JSON. Profil pojedynczy
nie stanowi dowodu przyczynowego ani odrębnej bramki szybkości.

W JSON zapisano wszystkie 11 rekordów poleceń: wyjścia, zatrzymania,
czasy wall/CPU, liczby próbek, nieudanych odczytów, obcych procesów
i zakończeń przed przypisaniem. Dodano również telemetrię całego bench.
Oba profile mają po 7 próbek obserwatora procesów obliczeniowych,
ale nie mają CSV parametrów GPU.

Telemetria w JSON opisuje całe świeże procesy: ładowanie, rozgrzewkę
i wszystkie fazy razem, bez izolowania granic faz. Własny licznik
runnera nie obserwował obcych procesów obliczeniowych w tych poleceniach;
liczby próbek i nieudanych odczytów pozostają w danych. Zegary, moc,
temperatura i tworzenie partii tylko opisują warunki, nie wykluczają
wyników ani nie dowodzą niezależnego zakłócenia.

## Pochodzenie i ograniczenia

47 zamkniętych plików pobrano i porównano SHA-256 z ich zamkniętymi
odpowiednikami z maszyny. Osobno pobrano i sprawdzono SHA run.json
i rekordu kontroli; odciski tych źródeł oraz obserwacji affinity
znajdują się w source_records. Rekord run.json wiąże role obu binarek
i runnera z ich skrótami. Zapisane runtime_version każdego przebiegu
potwierdza etykietę A/B; ustawienia wszystkich ośmiu przebiegów są
równe po usunięciu jedynie metadanych quantization: none kandydata. Ten wycinek nie zastępuje końcowego manifestu
całej kampanii ani potwierdzenia usunięcia maszyny. Pomiar jest przypisany
do zamrożonego runnera R4 i binarki W8; późniejsza poprawka runnera R5
dotyczy referencji HTTP i nie zmienia tego silnika ani pomiaru none.

Jeden model 4.5B, jedna konfiguracja i jeden typ karty. Brak pomiaru
CPU, innych zegarów, innych modeli i innych GPU. Jakość Q8, wydajność Q8,
VRAM/profile Q8, poprawiona referencja HTTP i akceptacja całego W8
pozostają poza tym częściowym pakietem.
