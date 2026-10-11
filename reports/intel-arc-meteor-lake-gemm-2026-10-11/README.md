# Intel Arc Meteor Lake: optymalizacja GEMM, 2026-10-11

Cel: krótsza inferencja i większa przepustowość przy zachowaniu wyników poprzedniego
kernela oraz dotychczasowej zgodności z FP32 upstream.

Sprzęt: zintegrowany Intel Arc Meteor Lake, Mesa 26.2.2, wgpu 30.0.1, Rust 1.95.0.
Punktem odniesienia jest build Intel z commita
`87da7867657dd052ffa9aac5f7c40cbf348fa6f6`. Porównywany build różni się wyłącznie
kernelem GEMM. SHA-256 nowego `gemm.wgsl`:
`51eb4a57b36453d3ef85cf64c6499fceda419ef963c1bbf14f2cbb8d86313720`.

## Kernel

W wewnętrznej pętli GEMM zapisano jawnie aktualizacje czterech akumulatorów
oraz dwa kolejne kroki K. Kafel pozostaje 32×32×32, grupa ma 64 wątki,
pamięć współdzielona przechowuje FP16, a akumulatory FP32. Dla każdego elementu
wyniku kolejność FMA nadal biegnie po rosnącym K. Maskowanie niepełnych kafli,
zaokrąglenie do FP16 oraz bariery pozostają takie same. Zmiana nie dodaje `unsafe`.

Sprawdzono 16 wariantów: głębokości K, rozwinięcia pętli, podziały pracy między
wątki i kolejność bloków. Większe K oraz alternatywne podziały między wątki
zwykle pogarszały wyniki. Samo rozwinięcie pętli K nie dawało stabilnego zysku.
Wybrano jawne aktualizacje wierszy połączone z dwoma krokami K; dalsze
rozwinięcie do czterech kroków było słabsze.

Końcowy mikrobenchmark obejmuje QKV, output, gate/up i down wszystkich modeli,
M = 4, 53, 240, 1024 oraz macierz z niepełnymi wymiarami N = 39, K = 37.
Łącznie 52 kształty, po trzy przeplatane rundy starego i nowego kernela,
z rozgrzewką oraz trzema pomiarami na wariant w rundzie. Po każdej rundzie
porównano cały bufor wyniku: **156/156 porównań nowego kernela jest bitowo zgodnych**.
Dane wejściowe były deterministycznymi, niezerowymi wartościami FP16.

[gemm.csv](gemm.csv) zawiera wymiary i mediany dziewięciu pomiarów na wariant.
Czas obejmuje dispatch, submit i synchronizację. To izolowane projekcje
z rezydentnymi buforami, a nie czas inferencji modelu.

## Krótkie HTTP: przed i po

Ruch przez produkcyjny router `/v1/systemone`, z domyślnym cache prefiksów
i FP16. Żądanie: stan „Ala ma kota.”, pytanie choice „Co ma Ala?”, opcje
„kota” i „psa”. Każdy przebieg ma rozgrzewkę, sześć żądań sekwencyjnych
i sześć żądań przy dwóch klientach. Kolejność buildów: stary–nowy–nowy–stary.
Modele uruchamiano osobno, bez innych pomiarów GPU i kompilacji w tle.

Mediany poniżej łączą 12 próbek na wersję i poziom współbieżności. Przepustowość
to suma udanych żądań podzielona przez sumę czasów odpowiednich faz.
Opóźnienie obejmuje wysłanie HTTP i odczyt całej odpowiedzi; nie obejmuje
ładowania modelu ani rozgrzewki.

| Model | Klienci | Mediana przed → po | Spadek mediany | Żądania/s przed → po | Wzrost przepustowości |
|---|---:|---:|---:|---:|---:|
| mini | 1 | 0,899 → 0,798 s | 11,2% | 1,064 → 1,175 | 10,4% |
| mini | 2 | 1,907 → 1,771 s | 7,1% | 1,008 → 1,098 | 9,0% |
| 4.5B | 1 | 2,875 → 2,253 s | 21,6% | 0,348 → 0,446 | 28,1% |
| 4.5B | 2 | 5,434 → 4,613 s | 15,1% | 0,360 → 0,424 | 17,9% |

Wszystkie 96 mierzonych żądań zakończyły się powodzeniem. Porównanie odpowiedzi
z pierwszą udaną odpowiedzią w obrębie każdego przebiegu nie wykazało zmian.
Nie zastępuje to porównania modeli z FP32.

[http-short.csv](http-short.csv) zachowuje wynik każdego przebiegu,
a [http-samples.csv](http-samples.csv) wszystkie mierzone opóźnienia HTTP.
Mini ma zauważalny rozrzut: mediana sekwencyjna w pierwszej parze wyniosła
0,877 → 0,916 s, a w drugiej 0,942 → 0,797 s. Wzrost przepustowości przy dwóch
klientach wystąpił w obu parach. Dla 4.5B obie pary poprawiły obie metryki.
W tej próbce dwa równoczesne żądania nie dawały większej przepustowości niż
jedno, a zwiększały opóźnienie.

## Dłuższe wejścia mini

Osobna, sekwencyjna para buildów: po dwa żądania każdej klasy po rozgrzewce.
`shared` zawiera wspólny stan zamówienia i cztery pytania różnych typów;
`document` dłuższy dokument. Pole `input_tokens` benchmarku wynosi odpowiednio
258, 1830 i 2208 dla short, shared i document; nie jest to liczba wierszy
po usunięciu cache prefiksu w pojedynczej projekcji.

| Klasa | Mediana przed | Mediana po |
|---|---:|---:|
| short | 0,871 s | 0,747 s |
| shared | 4,280 s | 3,613 s |
| document | 8,550 s | 7,491 s |

Wszystkie 12 żądań zakończyło się powodzeniem, bez zmian odpowiedzi w obrębie
przebiegów. [http-varied.csv](http-varied.csv) zawiera te dane. Dwie próbki
na klasę i wersję są obserwacją pomocniczą, nie estymacją percentyli ogona.

## Zgodność bitowa i FP32

| Model | Zakres eksportu | Bitowo identyczne z poprzednim kernelem |
|---|---|---:|
| mini | 44 bench Tree + 34 przypadki API | 78/78 |
| 4.5B | 44 bench Single | 44/44 |

Porównano pliki JSONL bajt po bajcie. Dla mini punktem odniesienia jest zachowany
eksport kodu opisanego w [raporcie poprawek](../intel-arc-meteor-lake-fixes-2026-10-11/README.md),
zgodnego ze źródłami bazowego commita. API obejmuje wyniki pytań Single oraz
odpowiedzi serwowane z pakowaniem Tree. Dla 4.5B wykonano dwa świeże eksporty
pełnego zestawu 44 zadań. Nie powtarzano przypadków API 4.5B.

Osobne `basal compare` z FP32 upstream potwierdziło po 44/44 zgodnych decyzji
avg i cal oraz brak różnic tokenów w bench obu modeli:

| Model | max \|Δlogit\| | max \|Δp_cal\| | max TV avg |
|---|---:|---:|---:|
| mini | 0,021352 | 0,004326 | 0,004997 |
| 4.5B | 0,039001 | 0,005328 | 0,006454 |

To nadal forward z wagami i aktywacjami FP16, a nie pełny FP32. Optymalizacja
nie powiększyła błędu w sprawdzonym zbiorze. Powyższa tabela dotyczy bench;
przypadki API wykorzystano do porównania bitowego między wariantami kernela.
Istniejące różnice przygotowania wejść i dialektów API względem surowej
referencji upstream nie są nowym wynikiem tej optymalizacji.

## Kontrole kodu

Przeszły `cargo fmt --all --check`, Clippy i Rustdoc z `-D warnings` dla
Intel oraz Linux bez GPU, build release Intel z walidacją WGSL przez Naga,
`cargo deny` 0.19.8, kontrola wyjątku `paste` i `git diff --check`.
Polecenia Cargo dotyczące zależności używały `--locked`. `cargo deny` zgłosił
dotychczasowe ostrzeżenia o duplikatach zależności; wszystkie jego polityki przeszły.

Przegląd kernela potwierdził tę samą kolejność FMA dla każdego akumulatora,
indeksy w zakresie 0–1023 dla obu kafli, te same miejsca synchronizacji
i rozłączne zapisy wyniku. Nie zmieniono interfejsów ani zarządzania pamięcią
w Rust. Nie dodano testów, frameworków, zależności ani opcji konfiguracji.

## Ograniczenia

- **Max:** benchmark zatrzymał się przed ładowaniem modelu. RAM dostępny po
  uwzględnieniu limitu cgroup wynosił około 19,0 GiB, poniżej checkpointu
  około 20,8 GiB i rezerwy 2 GiB. Nie zmniejszano rezerw. Dla max ta sesja
  potwierdza wyłącznie izolowane projekcje GEMM; nie potwierdza czasu ani
  jakości całej inferencji nowym kernelem.
- GPU współdzieli desktop; nie kontrolowano zegarów ani długotrwałych limitów
  termicznych. Wyniki dotyczą tej karty, sterownika i wejść. Nie dowodzą
  przyspieszenia na innych Arc, dedykowanych GPU ani każdym rozmiarze partii.
- Małe próbki i ruch closed-loop nie opisują produkcyjnego p95/p99 ani
  zachowania przy niezależnym napływie żądań. Nie mierzono maksymalnego kontekstu.
- Pomiary HTTP skończono przed kontrolami kompilatora. Kontrole uruchomiono
  obok późniejszych eksportów jakości; ich czasy nie są pomiarem wydajności.
- Surowe logi, binarki, eksporty i pomocnicze programy pomiarowe pozostają
  lokalnie poza Gitem. Wersjonowane CSV zawierają próbki opóźnień i podsumowania bez
  danych identyfikujących maszynę.
