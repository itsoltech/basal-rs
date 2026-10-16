# Precyzja i wydajność attention na RTX 6000 Ada

Wynik: balanced skraca pełny czas pojedynczego HTTP we wszystkich 25 punktach
i zwiększa poprawne QPS we wszystkich 18 punktach względem upstream fast BF16.
Zakresy trzech powtórzeń nie nakładają się względem BF16. Na niezależnej rezerwie
wszystkie trzy modele zachowują 140/140 decyzji kategorycznych FP32 i mają mniejszy
średni błąd prawdopodobieństw oraz score niż BF16. Pozostają gorsze ogony
opóźnień dużych dokumentów przy 8 klientach oraz brak bezpośredniej referencji
upstream FP32 dla długich wejść Max (OOM). Nie jest to obietnica poprawy każdego
żądania ani każdej wartości prawdopodobieństwa.

## Cel i warianty

Celem jest krótszy czas pełnego HTTP request–response niż upstream fast BF16 oraz
większa liczba poprawnie zakończonych żądań na sekundę, przy odpowiedziach bliskich
upstream FP32. Te trzy warunki sprawdzamy osobno. Nie utożsamiamy lepszego QPS
z krótszym czasem pojedynczego żądania ani zgodności z nauczycielem z jakością
względem ludzkich etykiet.

`accurate` zachowuje dotychczasowe attention hi/lo. `balanced` używa iloczynów Q/K
w f16, prawdopodobieństw hi/lo oraz softmaxu i akumulacji w f32. Specjalizacja
`tc-qk1-static` usuwa obsługę alternatywnych typów V i wyjścia tam, gdzie host
potwierdza dokładne V f16 i scalone wyjście f16. Ta specjalizacja musi zachować
wszystkie bity wyniku rodzica `tc-qk1`. Zmiana Q/K jest zmianą numeryki.

Sprzęt: jedna NVIDIA RTX 6000 Ada Generation, 48 GB, limit 300 W, bez blokady
zegarów SM. Kontener ma limit 8 CPU i 64 GiB RAM. Kolejne silniki nie działają
jednocześnie. Obciążenie przerywa kontrola temperatury 80°C, błędów GPU/pamięci,
zasobów i zdrowia istniejących usług. Ten pomiar nie uzasadnia polityki dla innych
GPU, Metal, Intel ani innych modeli.

Upstream: `af1ce43a1770dd4fdfc1611e3fe720b93d42fc73`, checkout bez modyfikacji.
Modele: mini `cb81b32ce89b0b87e62fcdad28f0435e25c953f3`,
4.5B `e5ba9149760d99b73b1963d8e0ccf948b58afacb`,
max `2fffc5cc41e5cf7a2f752f38f17e7b4d9a4a8252`.
Zamrożony numeryczny kandydat: SHA-256 binarki
`90383ed2348eaca052ffdb6cff793acb0bff871848a72180e297fedc9bea8ba2`.

## Dane i rozdział doboru od potwierdzenia

Rozwój: 233 żądania, 265 pytań; część to powtórzenia 44 bazowych stanów z nonce.
To materiał doboru wariantów, nie niezależny test końcowy. Dane publiczne to
900 rekordów z `benchmark-data/v2/sequential.jsonl`, SHA-256
`81756119f5fe3a4e44aefa1c1cc56e5cd3153001141fb715ec502bb3ee193374`.
Po grupowaniu znormalizowanego stanu (NFKC, casefold, białe znaki) pozostaje 898
unikalnych rekordów. Zamrożony podział: 360 walidacji, 360 potwierdzenia i 178
rezerwy. Dziewięć zbiorów: ag-news, allegro-reviews, boolq, cbd, cdsc-e, dyk,
emotion, polemo2-in i sst5. Nie ma wspólnych znormalizowanych stanów z materiałem
rozwojowym. Publiczny charakter danych nie daje gwarancji braku w treningu.

Krótkie przykłady mają do 1024 tokenów. Osobno sprawdzono 2 poprawne długie wejścia z drabiny długości
oraz 4 wejścia z neutralnym dopełnieniem na model: 4k/8k mini, 16k/32k większych.
Pierwszym nauczycielem był natywny FP32; sam most do upstream FP32 na krótkich
wejściach nie dowodzi zgodności obu nauczycieli na długich dokumentach. Dlatego
osobna seria próbuje zebrać bezpośrednie upstream eager FP32 na tych samych
sześciu długich wejściach oraz 18 krótkich kotwicach na model. Używa oficjalnego
`--deterministic`, chunkowania upstream i obserwatora polityki precyzji, bez zmian
w checkout upstream. Każdy brak referencji pozostaje GAP. Dopełnienie wybrano
z walidacji bez etykiet późniejszego potwierdzenia.

Bezpośrednia seria r2 potwierdziła w tym samym procesie TF32=false, precision=highest i deterministyczne obliczenia. Wszystkie 18 krótkich kotwic/model jest bitowo zgodnych z wcześniejszym upstream FP32.

| Model | Długie referencje | Kategorie balanced / BF16 | MAE prawdop. accurate | balanced | BF16 | Status |
|---|---:|---:|---:|---:|---:|---|
| mini | 6/6 | 6/6 / 6/6 | 0.000312641 | 0.000524913 | 0.003010340 | PASS |
| 4.5B | 6/6 | 6/6 / 6/6 | 0.000372874 | 0.000424786 | 0.004714132 | PASS |
| max | 0/6 | — | — | — | — | GAP |

Dla Max pierwszy długi request 16k otrzymał 422 z komunikatem CUDA out of
memory; następnych pięciu długich wejść nie wykonano. Wszystkie sześć braków
pozostaje GAP. Nie zastąpiono brakującej referencji upstream wynikiem natywnym.
Natywne FP32 jest osobnym, uzupełniającym dowodem: sześć długich przypadków/model
zachowuje decyzje. Jego różnica prawdopodobieństw względem bezpośredniego FP32
na sześciu wejściach wynosi około 5,24e-6 dla Mini i 2,36e-5 dla 4.5B.
Dla Max taka różnica na długich wejściach pozostaje niezmierzona. Nie wynajmowano
dodatkowej karty ani nie dostrajano numeryki do tych wyników.

## Korekta metryki `score`: zachowane FAIL i niezależna rezerwa

Pierwotna bramka wymagała tego samego argmax także dla `score`. Mini uzyskało
359/360 względem 360/360 accurate; kandydat zmienił modalny poziom w
`allegro-reviews-40`. Max ma analogiczną zmianę w `sst5-45`: 359/360 względem
360/360 accurate. Wynik pierwotnej bramki dla tych modeli pozostaje FAIL.

API `score` zwraca jednak sumę `i*p_i`, a nie modalny poziom. Dla wskazanego Mini
FP32 zwraca 0,83228709, balanced 0,83273896, a BF16 0,83403965. Dla Max:
FP32 2,03407711, balanced 2,03540723, BF16 2,03735507. Oba wyniki balanced są
bliższe zwracanej wartości FP32 mimo różnicy argmax.

Kryterium skorygowano jawnie po zobaczeniu tego problemu pomiaru. Zachowano
wszystkie stare diagnostyki. Decyzje kategoryczne wyłączają ciągłe `score`;
wartość zwracana przez `score` podlega osobnemu MAE względem FP32. Kryteria
choice/multi/act i binarny próg noul pozostały bez zmian. Skorygowane wyniki starego
potwierdzenia nie są nazywane nietkniętym testem. Nowe kryterium zamrożono przed
pobraniem jakichkolwiek etykiet FP32 dla niewykorzystanej rezerwy 178. Kandydat,
wagi, kalibracja i progi pozostają niezmienione. Wynik rezerwy podajemy
osobno. Wszystkie trzy modele przeszły tę bramkę; nie dostrajano do niej kernela.

| Model | Kategorie balanced / BF16 | MAE prawdop. balanced / BF16 | MAE score balanced / BF16 |
|---|---:|---:|---:|
| mini | 140/140 / 139/140 | 0.000403722 / 0.003126164 | 0.001162171 / 0.011861399 |
| 4.5B | 140/140 / 139/140 | 0.000329934 / 0.002167563 | 0.000953463 / 0.007086853 |
| max | 140/140 / 140/140 | 0.000155223 / 0.001060019 | 0.000445515 / 0.002607093 |

Na rezerwie balanced jest bliżej FP32 pod względem średniego błędu
prawdopodobieństw w 170/178, 169/178 i 170/178 wejść (Mini/4.5B/Max).
BF16 jest bliżej w pozostałych 8/9/8. Dla wartości score balanced jest bliżej
w 35/38, 33/38 i 30/38 przypadków. Lepsza średnia nie oznacza poprawy każdego
wejścia. Te same 898 publicznych tekstów służyły trzem modelom; nie jest to
2694 niezależnych stanów.

Walidacja i potwierdzenie obejmują choice, noul i score. Nie zebrano nowego,
niezależnego holdoutu multi/act; te typy występują w materiale rozwojowym,
a porównania profili sprawdzają bity numerycznego rodzica. Nie rozszerzamy
wniosku o zgodności kategorycznej na wszystkie możliwe typy i wejścia.

Zmieniona metryka na starym potwierdzeniu (360 żądań/model, z czego 280
kategorycznych i 80 score):

| Model | balanced: kategorie FP32 | BF16: kategorie FP32 | balanced: MAE prawdop. | BF16: MAE prawdop. | balanced: MAE score | BF16: MAE score |
|---|---:|---:|---:|---:|---:|---:|
| mini | 280/280 | 277/280 | 0.00044226 | 0.00293913 | 0.00144666 | 0.00752218 |
| 4.5B | 280/280 | 279/280 | 0.00032538 | 0.00239487 | 0.00095808 | 0.00624914 |
| max | 280/280 | 280/280 | 0.00014565 | 0.00107792 | 0.00042781 | 0.00318861 |

Każde żądanie zostało porównane z rzeczywistą odpowiedzią upstream eager FP32.
Upstream fast jest rzeczywistym BF16 z kompilacją i CUDA Graphs. To odległość
numeryczna od FP32, nie pomiar trafności względem semantycznych etykiet zbiorów.

## Dowody strukturalne i ograniczenia

Końcowe 72 przypadki kernela zachowują bity rodzica; pomiar CUDA event daje
około 7–20% krótszy czas samego kernela QK1-static. Nie jest to dowód przyspieszenia
pełnego HTTP. Warianty async, przeładowania Q i wstępnego pakowania Q zachowano
jako wyniki negatywne; nie weszły do ścieżki produkcyjnej.

Profil API przed końcową selekcją: 30 porównań eksportów (3 modele, 5 sposobów
pakowania/cache, 2 pliki) zachowało wszystkie bity. Sonda YAML sprawdziła 72
współbieżne odpowiedzi, w tym kopiowanie trybu do torów długich żądań, oraz
lokalne CLI i automatyczne tabele GEMM. Pierwsza sonda zgłosiła FAIL przez
porównanie confidence dwóch różnych dialektów API; zachowano tę porażkę.
Poprawiona sonda porównuje ten sam dialekt z zamrożonym rodzicem i przechodzi.
Nie zmieniono kontraktu confidence aplikacji.

Końcowy profil zachował bity 30 eksportów zamrożonego rodzica i 1080 odpowiedzi
HTTP na potwierdzeniu (360/model). Końcowa sonda konfiguracji przeszła 73
porównania. Wszystkie podsumowania bramek odtworzono niezależnie na CPU.

## Pełny czas pojedynczego pytania

Jeden klient HTTP, jedno pytanie, załadowany model, cache stanu wyłączony,
`orders=1`. Oba endpointy mierzono po loopback, bez opóźnienia Internetu.
Pomiar zaczyna się przed serializacją i wysłaniem żądania, a kończy po
odczycie całego ciała odpowiedzi. Parsowanie JSON jest poza pomiarem. Ładowanie
modelu, kompilacja i przygotowanie CUDA Graphs są poza czasem request–response.

Oba profile natywne działają naprzemiennie, a zewnętrzna kolejność względem
upstream to ABBAABBA. Są cztery powtórzenia na wariant; pierwsze zachowano jako
diagnostykę i wyłączono z agregacji. Każdy punkt ma dwie rozgrzewki i dziesięć
pomiarów na powtórzenie: poniżej mediana trzech median, 30 próbek na silnik.
Zakresy trzech median są w danych i porównaniu interaktywnym. Nie są przedziałem
ufności. Silniki nie działają jednocześnie; nie ma ucinania wejść.

Wszystkie 25 punktów balanced mają niższą medianę niż upstream BF16; zakresy
median obu silników nie nakładają się w żadnym z tych punktów. Redukcja czasu:
Mini 6,2–21,0%, 4.5B 8,5–22,2%, Max 10,7–26,9%.

| Model | Tokeny promptu | basal-rs accurate ms | basal-rs balanced ms | upstream fast BF16 ms | Mniej czasu vs BF16 |
|---|---:|---:|---:|---:|---:|
| mini | 128 | 7.908 | 7.437 | 9.415 | 21.01% |
| mini | 256 | 9.663 | 9.071 | 11.429 | 20.63% |
| mini | 512 | 15.032 | 14.282 | 16.929 | 15.63% |
| mini | 1024 | 30.570 | 28.989 | 31.051 | 6.64% |
| mini | 2048 | 62.347 | 58.781 | 62.657 | 6.19% |
| mini | 4096 | 127.724 | 115.952 | 133.649 | 13.24% |
| mini | 8192 | 324.651 | 282.400 | 303.512 | 6.96% |
| 4.5B | 128 | 17.002 | 16.605 | 19.215 | 13.58% |
| 4.5B | 256 | 23.907 | 22.866 | 27.803 | 17.76% |
| 4.5B | 512 | 43.999 | 43.568 | 51.058 | 14.67% |
| 4.5B | 1024 | 83.118 | 81.360 | 104.525 | 22.16% |
| 4.5B | 2048 | 170.997 | 164.333 | 205.647 | 20.09% |
| 4.5B | 4096 | 374.441 | 352.467 | 433.773 | 18.74% |
| 4.5B | 8192 | 871.003 | 768.250 | 901.503 | 14.78% |
| 4.5B | 16384 | 2250.222 | 1866.904 | 2257.627 | 17.31% |
| 4.5B | 32768 | 6617.449 | 5052.832 | 5521.409 | 8.49% |
| max | 128 | 33.550 | 33.048 | 37.507 | 11.89% |
| max | 256 | 53.786 | 53.675 | 67.297 | 20.24% |
| max | 512 | 96.725 | 96.069 | 122.173 | 21.37% |
| max | 1024 | 182.723 | 179.105 | 238.839 | 25.01% |
| max | 2048 | 361.984 | 350.638 | 474.900 | 26.17% |
| max | 4096 | 747.666 | 705.368 | 964.593 | 26.87% |
| max | 8192 | 1817.265 | 1641.471 | 1980.929 | 17.14% |
| max | 16384 | 4487.772 | 3815.849 | 4657.583 | 18.07% |
| max | 32768 | 11996.728 | 9624.414 | 10775.159 | 10.68% |

Balanced ma także niższe mediany od accurate w tych 25 punktach. Nie wszystkie
małe różnice względem accurate są rozdzielone zakresem powtórzeń: np. Max256
ma 53,67 ms wobec 53,79 ms i nakładające się zakresy. Nie przypisujemy tak małej
różnicy pewnego zysku optymalizacji.

Upstream fast ma domyślne okno adaptacyjne 3 ms po wcześniejszej partii większej
niż jeden. Przy jednym kliencie i jednym pytaniu efektywne okno pozostaje 0 ms;
wyniku sekwencyjnego nie przypisujemy sztucznemu oczekiwaniu upstream.

## Przepustowość i ogon opóźnień

Przed pomiarem osobny zamrożony f318 zbiera odpowiedzi kontrolne dla wszystkich
1395/1424/1424 wejść Mini/4.5B/Max przy 128 klientach. Są wyłączone z agregatów
wydajności. Accurate musi zachować bity tych odpowiedzi także przy innych partiach.
Trzy warianty uruchomiono w kolejności ABCCBAABCCBA, co dla każdej pary daje
ABBAABBA. Pierwsze powtórzenie każdego wariantu wyłączono, pozostawiając trzy.
Krótkie obciążenie zawiera 1024 różne wejścia, utworzone z 44 bazowych stanów
i nonce; to różnorodność wejść dla pomiaru serwowania, nie 1024 niezależne
przykłady jakości. Na fazę przypada 2048 żądań Mini/4.5B i 1024 Max. Ruch
mieszany przechodzi cały skończony korpus: 371 żądań Mini, 400 większych modeli.
Klienci zamykają pętlę po odpowiedzi. Cache stanu wyłączony, `orders=1`,
budżet partii 8192, limit expanded/large 256 i limit przyjętych żądań 1024.
QPS to poprawne odpowiedzi podzielone przez czas całej fazy, z jej opróżnieniem.

| Model | Ruch | Klienci | accurate QPS | balanced QPS | upstream BF16 QPS | balanced / BF16 |
|---|---|---:|---:|---:|---:|---:|
| mini | short | 1 | 128.512 | 139.982 | 91.731 | 1.526× |
| mini | short | 8 | 212.818 | 215.354 | 130.961 | 1.644× |
| mini | short | 32 | 268.111 | 269.391 | 121.500 | 2.217× |
| mini | short | 128 | 309.706 | 313.885 | 134.397 | 2.335× |
| mini | mixed | 8 | 46.849 | 49.089 | 26.189 | 1.874× |
| mini | mixed | 128 | 113.614 | 119.620 | 26.369 | 4.536× |
| 4.5B | short | 1 | 50.943 | 53.051 | 41.104 | 1.291× |
| 4.5B | short | 8 | 75.573 | 76.435 | 41.633 | 1.836× |
| 4.5B | short | 32 | 91.212 | 92.010 | 36.675 | 2.509× |
| 4.5B | short | 128 | 102.043 | 102.990 | 43.622 | 2.361× |
| 4.5B | mixed | 8 | 6.542 | 7.210 | 4.601 | 1.567× |
| 4.5B | mixed | 128 | 8.463 | 9.488 | 4.725 | 2.008× |
| max | short | 1 | 23.149 | 23.215 | 17.334 | 1.339× |
| max | short | 8 | 32.286 | 32.398 | 17.742 | 1.826× |
| max | short | 32 | 40.809 | 40.831 | 15.804 | 2.584× |
| max | short | 128 | 48.292 | 48.883 | 19.805 | 2.468× |
| max | mixed | 8 | 3.098 | 3.398 | 2.146 | 1.584× |
| max | mixed | 128 | 4.044 | 4.541 | 2.131 | 2.131× |

Balanced ma wyższy QPS niż BF16 w 18/18 punktach; zakresy trzech powtórzeń nakładają się w 0/18 punktach. To zakres skończonego pomiaru, nie przedział ufności ani dowód granicznej pojemności serwera.

Poniżej mediana trzech median oraz p95 połączonych próbek ruchu mieszanego, w milisekundach. P50 i p95 odpowiadają różnym agregacjom; nie należy ich łączyć w jedną dystrybucję produkcji.

| Model | Klienci | accurate p50 / p95 | balanced p50 / p95 | upstream BF16 p50 / p95 |
|---|---:|---:|---:|---:|
| mini | 8 | 151.793 / 338.944 | 143.236 / 339.968 | 278.778 / 606.368 |
| mini | 128 | 793.514 / 1653.855 | 785.656 / 1523.856 | 2631.562 / 6826.036 |
| 4.5B | 8 | 297.453 / 6020.310 | 345.716 / 5219.989 | 1387.575 / 4128.307 |
| 4.5B | 128 | 2150.568 / 14731.010 | 2460.394 / 13102.450 | 7152.375 / 40992.409 |
| max | 8 | 552.572 / 13179.467 | 543.461 / 12053.225 | 3144.946 / 8501.476 |
| max | 128 | 3810.336 / 31647.901 | 3693.130 / 29124.825 | 16844.329 / 95976.485 |

Wyższy QPS nie skraca każdej odpowiedzi. Pooled p95 balanced jest gorsze od BF16 dla: 4.5B, N=8, max, N=8. Mediana balanced jest gorsza od accurate dla: 4.5B, N=8, 4.5B, N=128.

Ogon każdej klasy poniżej to mediana trzech p95 liczonych osobno dla przebiegów. Najdłuższa klasa ma tylko 8 żądań na przebieg (24 łącznie), więc wynik jest wrażliwy na pojedyncze żądania. Znak „tak” zachowuje regresję względem BF16.

| Model | Klienci | Klasa | accurate p95 ms | balanced p95 ms | BF16 p95 ms | balanced wolniej od BF16 |
|---|---:|---|---:|---:|---:|---|
| mini | 8 | doc-1k-2k | 358.716 | 344.089 | 736.352 | nie |
| mini | 8 | doc-4k-8k | 576.004 | 500.672 | 716.025 | nie |
| mini | 8 | features | 263.349 | 250.564 | 591.353 | nie |
| mini | 8 | short-1q | 316.346 | 307.296 | 552.028 | nie |
| mini | 8 | short-multi | 367.479 | 365.756 | 714.356 | nie |
| mini | 128 | doc-1k-2k | 2520.814 | 2379.647 | 7686.934 | nie |
| mini | 128 | doc-4k-8k | 1185.676 | 1156.210 | 10847.873 | nie |
| mini | 128 | features | 895.598 | 845.340 | 3440.660 | nie |
| mini | 128 | short-1q | 885.306 | 844.314 | 3423.858 | nie |
| mini | 128 | short-multi | 1150.032 | 1122.768 | 3451.548 | nie |
| 4.5B | 8 | doc-16k | 17873.171 | 15060.557 | 5833.996 | tak |
| 4.5B | 8 | doc-1k-2k | 832.690 | 814.705 | 3806.431 | nie |
| 4.5B | 8 | doc-4k-8k | 9785.160 | 9946.276 | 4949.811 | tak |
| 4.5B | 8 | features | 668.570 | 741.078 | 3171.553 | nie |
| 4.5B | 8 | short-1q | 664.958 | 668.202 | 3947.658 | nie |
| 4.5B | 8 | short-multi | 637.939 | 702.432 | 3738.097 | nie |
| 4.5B | 128 | doc-16k | 38562.225 | 34069.928 | 67016.764 | nie |
| 4.5B | 128 | doc-1k-2k | 5469.845 | 6732.100 | 20707.721 | nie |
| 4.5B | 128 | doc-4k-8k | 19553.865 | 17885.669 | 44282.174 | nie |
| 4.5B | 128 | features | 2709.678 | 2633.921 | 10289.644 | nie |
| 4.5B | 128 | short-1q | 2702.522 | 2753.366 | 10292.008 | nie |
| 4.5B | 128 | short-multi | 2789.189 | 2755.495 | 10304.671 | nie |
| max | 8 | doc-16k | 38401.691 | 35161.896 | 11972.964 | tak |
| max | 8 | doc-1k-2k | 1483.464 | 1503.034 | 8853.340 | nie |
| max | 8 | doc-4k-8k | 20599.422 | 18703.878 | 11311.116 | tak |
| max | 8 | features | 1367.460 | 1237.501 | 8191.902 | nie |
| max | 8 | short-1q | 1306.892 | 1169.858 | 7871.730 | nie |
| max | 8 | short-multi | 1229.201 | 1172.235 | 8456.775 | nie |
| max | 128 | doc-16k | 80235.568 | 73077.538 | 150915.639 | nie |
| max | 128 | doc-1k-2k | 11862.456 | 8831.070 | 51570.464 | nie |
| max | 128 | doc-4k-8k | 41352.033 | 39778.409 | 104110.609 | nie |
| max | 128 | features | 4184.964 | 3928.401 | 20712.197 | nie |
| max | 128 | short-1q | 4187.384 | 3918.870 | 24723.949 | nie |
| max | 128 | short-multi | 4195.122 | 3922.139 | 20721.321 | nie |

P95 jest opisem tego skończonego korpusu i liczby próbek. Nie jest oszacowaniem
SLO całej produkcji ani dowodem maksymalnej wydajności przy dowolnej liczbie
klientów. Bramka odrzuca pomiar z błędami HTTP, 529, brakami pokrycia lub
naruszeniem wymaganej niezmienności odpowiedzi. Zmienność upstream pozostaje
osobną diagnostyką, bez przypisywania mu obietnicy niezależności od partii.

## Wersje i końcowy wybór

Końcowy kandydat z API profili ma SHA-256
`20222c5eb3650a5044a9d49aea0c32a5e392fca2e152d54338c1c90b08804e68`.
Źródło odtwarza patch względem commita
`9c0762f2971e0970345c2ddc7dad03a189d11658`; pełne źródła numerycznego
rodzica 90383 i pośrednich wariantów pozostają w archiwum.

Rust 1.95.0, toolkit CUDA 12.9.1, sterownik 580.159.03. Natywny silnik używa
cuBLASLt 120901; upstream: PyTorch 2.11.0+cu128, cuBLASLt 120804, runtime CUDA
12.8.90, Transformers 5.17 i Triton 3.6. Porównujemy kompletne silniki z ich
rzeczywistymi bibliotekami oraz f16/BF16, nie same frameworki przy identycznej
arytmetyce. Te same przypięte checkpointy BF16 są przeliczane w odpowiednim dtype;
FP32 oznacza precyzję obliczeń referencyjnych tych checkpointów.

Zwykłe eager FP32 we wcześniejszych danych miało obserwowane ustawienia PyTorch
TF32=false i precision=highest. Dawny opis runnera „server disables TF32” był
zbyt szeroki: upstream ustawia to jawnie w funkcji dla `--deterministic`.
Zachowano stare źródła i metadane; dopisano wyjaśnienie i bezpośredni obserwator
polityki dla nowej serii. Most native FP32 do upstream FP32 na 360 walidacjach/model
zachowuje decyzje, ze średnią różnicą prawdopodobieństw około 1e-6 lub mniejszą.

W końcowym źródle `auto` wybiera balanced dla nazw `basal-2.0-mini`,
`basal-2.0-4.5B` i `basal-2.0-max` wyłącznie na dokładnie rozpoznanej
`NVIDIA RTX 6000 Ada Generation`, przy CUDA f16, fused kernels i odczycie f32.
W pozostałym zakresie zachowuje poprzednią politykę. Jawne `accurate` pozwala
zachować pierwotną arytmetykę, a `balanced` jest dostępnym wyborem użytkownika.

Wszystkie trzy modele mają PASS bramki produktu na świeżej rezerwie, bitowej
dziedziczności profilu oraz własnej drabiny latency i QPS. Mini i 4.5B mają
również PASS bezpośredniego FP32 na sześciu długich wejściach. Max przyjęto
w zakresie zebranego dowodu: upstream FP32 na krótkich danych oraz oddzielne
natywne FP32 na długich danych. Nie zaliczamy jego braku referencji upstream
jako PASS ani nie twierdzimy, że potwierdzono pełną jakość Max względem upstream
FP32 dla długich dokumentów. Jest to jawne ograniczenie dowodu dla automatycznej
polityki opartej na tych pomiarach, a nie wyjątek numeryczny dla wybranych żądań.

`accurate` zachowuje pierwotną arytmetykę attention; nie oznacza całego forwardu
w FP32. Pełny FP32 wymaga dtype f32. `balanced` jest wybierany przed cache K/V,
nie zmienia środowiska procesu i jest kopiowany do torów wykonania. Nie zmieniono
wag, kalibracji, progów ani semantyki confidence. Jawny `BASAL_ATT` pozostaje
wyjątkiem eksperymentalnym dla `auto`. Nie wykonano wdrożenia produkcyjnego.

## Kontrole i niepowodzenia

Formatowanie, Clippy oraz Rustdoc: CPU, Intel i CUDA ze wszystkimi features PASS.
`cargo deny --locked --all-features`, wyjątek paste i przegląd zależności PASS.
CUDA kompiluje własne kernele dla 80/89/90 oraz Hopper WGMMA. To nie dowodzi
runtime na tych GPU. Metal nie był kompilowany ani uruchamiany w tej kampanii.
Nie dodano frameworków ani bramek testowych.

Zachowano także nieudane próby harnessu: brak wykonywalnego bitu skopiowanej
binarki, błędną liczbę krótkich próbek oraz w guard11 przekazanie request-only
zamiast paczki eksportów FP32. Ostatnia próba zakończyła się przed inferencją;
naprawiono ścieżkę w nowej sesji, zachowując oryginalny FAIL. Błąd dialektu
confidence opisano wyżej. Nie zaliczamy tych prób do osiągów ani jakości modelu. Przed startem końcowego QPS
poprawiono także stary warunek baseline-only odrzucający nazwę accurate; dodano
niezależne odpowiedzi kontrolne i ich odtwarzalną walidację. Wykrycie nastąpiło
przed jakąkolwiek obserwacją końcowego QPS; zachowano niewykonane wersje planów.

Pierwszy końcowy QPS 4.5B osiągnął limit zadania 90 minut. Nie był to błąd GPU:
guard zakończył proces, zatrzymał kontener i przywrócił ustawienia. Oryginalny
run pozostaje niekompletny (GAP), a timeout jest zachowany jako FAIL wykonania.
Nowy run kontynuuje wyłącznie ciągły prefiks dziesięciu w pełni zakończonych
wariantów, po sprawdzeniu 101 744 odpowiedzi na CPU. Nieukończony wariant jest
wykonywany od początku; ostatnie dwa warianty zebrano w nowej sesji.

Kontynuacja zachowuje przypięte wejścia, binarki, kolejność wariantów,
numery powtórzeń, rozgrzewki, limity mocy i chłodzenie. Zachowano stare źródło
runnera, jego SHA-256, oryginalny postęp i skopiowane pliki; audyt odtwarza
łańcuch danych. Granica sesji pozostaje ograniczeniem tego porównania, nawet
przy tych samych zadeklarowanych warunkach. Globalny limit pojedynczej sesji
trzech godzin nie został zwiększony.

Pierwsza bezpośrednia seria FP32 zakończyła serwery, lecz rejestrator polityki
precyzji zgłosił `NameError: __file__` przy wyjściu. Jej odpowiedzi pozostają
w archiwum, a audyty oznaczają je jako niezweryfikowane (GAP); nie zaliczono ich
jako bezpośredniego dowodu FP32. Poprawka przechwytuje SHA obserwatora przed
uruchomieniem modułu upstream. Nowy obserwator przeszedł sondę `--help` bez GPU,
a świeża seria używa tych samych 24 wejść/model i oficjalnych flag. Nie zmieniono
kodu upstream ani numeryki kandydata. Stare i nowe źródła oraz wyniki zachowano
oddzielnie.

Zachowano 18 zakończonych sesji nadzorcy; wszystkie mają potwierdzone
przywrócenie ustawień i zatrzymanie kontenera. Najwyższa temperatura wyniosła
71°C. Końcowe sesje 15, 16, 20, 18, 19 i 21 zakończyły plan zadań;
dozwolone niepowodzenie aplikacyjne Max FP32 w 19/21 pozostaje osobnym GAP.
Nie wykryto restartu hosta, nowych błędów sprzętu/Xid ani host OOM. CUDA OOM
procesu referencyjnego nie jest pomijany ani nazywany udanym pomiarem jakości.

W trakcie Max wystąpiła przerwa zewnętrznego dostępu SSH/HTTPS. Po odzyskaniu
połączenia potwierdzono brak restartu, ciągłość telemetrii (największa przerwa
między próbkami około 3,20 s), brak nowych wpisów o błędach GPU/OOM/sprzętu
i zdrowie czterech kontrolowanych usług. HTTP pomiaru działa po loopback wewnątrz
kontenera i trwało dalej. Przyczyna problemu dostępu pozostaje nieustalona;
lokalny odbiór wyników wznowiono w nowym katalogu, bez restartowania kampanii.

Po potwierdzeniu obu pobrań archiwum usunięto wyłącznie zatrzymany kontener
pomiarowy i jego katalog. Zachowano 12 kontenerów użytkownika, ich stany oraz
współdzielony obraz; cztery kontrolowane usługi pozostały zdrowe. GPU nie ma
zadań pomiarowych, limit wynosi 300 W, wentylator działa automatycznie,
a przywrócenie temperatury docelowej 85°C potwierdza nadzorca. Host nie został
zrestartowany. Nie wykonano wdrożenia produkcyjnego.

## Dane i odtworzenie

| Pole | Wartość |
|---|---|
| URI | `hf://buckets/itsoltech/basal-rs/measurements/precision-performance-ada-2026-10-16/62a636ab0f1417603dbd8f3189e03121df24b662d777aaef70d9cd4411b1446d/measurement-data.tar.gz` |
| SHA-256 archiwum | `62a636ab0f1417603dbd8f3189e03121df24b662d777aaef70d9cd4411b1446d` |
| Rozmiar gzip | 64 202 436 B |
| Weryfikacja | Pobranie z autoryzacją i anonimowo: SHA archiwum oraz 1933/1933 sumy plików zgodne; w obu kopiach 64 odtworzenia CPU PASS |

W obu kopiach sprawdzono 85 dokładnych źródeł i 12 paczek provenance wejść.
Zachowano 619 227 zapisanych odpowiedzi HTTP 200; ta liczba obejmuje powtórzenia
i nie jest liczbą niezależnych przykładów jakości. Lokalna kopia archiwum pozostaje
zachowana. Zakończony prefiks bucketa nie był nadpisywany.

Dane są pod `reports/precision-performance-ada-2026-10-16/` wewnątrz archiwum:
`runs/` zachowuje udane i nieudane próby, `inputs/` zamrożone wejścia,
`guards/` telemetrię bez identyfikatorów hosta, `summaries/` agregaty CPU,
`verification/` kontrole i kryteria, a `reproduce/` dokładne źródła runnerów,
patch źródła runtime i porównanie interaktywne. Odpowiedzi HTTP 200 oraz liczbowe
wartości JSON zachowują wszystkie pola i bity. Redakcje metadanych mają osobny
rejestr źródłowych/publicznych SHA. Prywatne surowe kopie i binarki pozostają
lokalnie; nie publikujemy wag modeli ani cache kompilacji. README wewnątrz
archiwum jest niezmienną kopią sprzed dopisania końcowego URI i SHA.

Pobierz do nowych katalogów, używając globalnego `hf` CLI:

```bash
BASAL_ARTIFACT_DL=.cache/measurement-download/precision-performance-ada-2026-10-16
BASAL_ARTIFACT_ROOT=.cache/measurement-artifacts/precision-performance-ada-2026-10-16-download
mkdir -p .cache/measurement-download .cache/measurement-artifacts &&
  mkdir "$BASAL_ARTIFACT_DL" "$BASAL_ARTIFACT_ROOT" &&
  hf buckets cp "hf://buckets/itsoltech/basal-rs/measurements/precision-performance-ada-2026-10-16/62a636ab0f1417603dbd8f3189e03121df24b662d777aaef70d9cd4411b1446d/measurement-data.tar.gz" "$BASAL_ARTIFACT_DL/measurement-data.tar.gz" &&
  echo "62a636ab0f1417603dbd8f3189e03121df24b662d777aaef70d9cd4411b1446d  $BASAL_ARTIFACT_DL/measurement-data.tar.gz" | sha256sum -c - &&
  tar -xzf "$BASAL_ARTIFACT_DL/measurement-data.tar.gz" -C "$BASAL_ARTIFACT_ROOT" &&
  python3 "$BASAL_ARTIFACT_ROOT/reports/precision-performance-ada-2026-10-16/reproduce/verify-artifact-tree-r1.py" \
    --measurement-artifacts "$BASAL_ARTIFACT_ROOT" --out verification-new.json
```

Kontrola działa bez GPU, sieci i poświadczeń. Sprawdza wszystkie wpisy
`SHA256SUMS`, dokładne źródła, provenance wejść, dozwolone redakcje metadanych,
łańcuch wznowienia 4.5B, agregaty wydajności, bramki produktu, oba audyty FP32
(oryginalny GAP i poprawioną serię), przywrócenie ustawień oraz bajtowe
odtworzenie porównania HTML. Odtwarza analizę zapisanych odpowiedzi; nie wykonuje
ponownie modeli i nie dowodzi semantycznej trafności.
