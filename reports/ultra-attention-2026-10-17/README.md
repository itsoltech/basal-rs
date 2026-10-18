# Ultra: normowanie, RoPE, GELU i attention na H100

**Status pomiarów: zakończone.** Tekst, obraz i wybrane FIFO mają osobne,
kompletne ABBA. Eksporty jakości, audyty i przegląd kodu są zakończone;
maszyna pomiarowa jest usunięta. Publikację potwierdzono ponownym pobraniem z autoryzacją i anonimowo.
Ten raport uzupełnia [poprzednią kampanię](../upstream-2.1-2026-10-17/README.md).

Na zmierzonej parze Ultra/H100 pojedyncze pytania są szybsze o 16–37% od
upstream BF16, a pełne zadania przy C1 mają medianę 432 zamiast 487 ms.
FIFO/16384 daje o 14–23% więcej zakończonych żądań/s przy C1/8/32.
Decyzje native pozostają zgodne z FP32 w badanych korpusach. Pozostaje
ograniczenie: p95 długich dokumentów w mieszanym ruchu C32 jest o 4,7% wyższe
od upstream, a średnia TV długości pogarsza się względem starego FP16,
choć nadal jest mniejsza od BF16.

## Cel i zakres

Skrócić czas pojedynczego HTTP requestu i zwiększyć liczbę poprawnych odpowiedzi
na sekundę dla Ultra względem upstream BF16, zachowując odpowiedzi referencji
FP32 oraz mniejsze odchylenie prawdopodobieństw niż BF16. Nie utożsamiamy
zgodności z FP32 z poprawnością semantyczną na dowolnym dokumencie.

Badamy `Remek/basal-omni-1.0-ultra`, rewizję
`b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4`, wagi bez kwantyzacji.
W tej kampanii Ultra obsługuje tekst i obrazy. Pozostałe modele rodziny
zostały opisane w poprzednim raporcie; ta sesja nie powtarza ich pomiarów.

## Wersje i sprzęt

- Baza źródeł: `f6b54cf`. Kod crate'ów jest identyczny z `c11ab46`, z którego
  pochodzi poprzednia binarka kontrolna. Nowe warianty mają osobne manifesty
  źródeł i SHA-256 binarek.
- Końcowa binarka CUDA R15: SHA-256
  `f23f060cf61ebf5d9d39c4f38835a6152b3823179e102152db2dfe17bc5af8b9`.
- Upstream: `62542d31e8eb69000839c01e8f9b45da1fb80e49`;
  Transformers 5.17.0, PyTorch 2.11.0 z CUDA 12.8.
- H100 PCIe, 80 GiB VRAM, około 180 GB RAM, sterownik 570.195.03.
  Native: CUDA 12.9.1, cuBLASLt 120901.
- Limit mocy 350 W. Blokada SM zażądana na 1755 MHz; rzeczywiste zegary
  rejestruje telemetria i mogą spadać pod limitem mocy. Nie zakładamy,
  że samo udane ustawienie blokady oznacza stały zegar.
- Oba serwery HTTP używają OMP/MKL/Rayon=8, bez przypinania rdzeni CPU.
  Pracują kolejno na tej samej karcie. Budowanie i lint CUDA odbywają się bez inferencji na GPU.

Telemetria obejmuje 6761 próbek. Maksymalna temperatura całej sesji wyniosła
72°C; minimalna dostępna pamięć RAM około 55 GiB, wolny dysk około 629 GiB.
Poniższe zakresy obejmują zachowane, mierzone fazy końcowego FIFO przy
wykorzystaniu GPU co najmniej 50%, z próbkowaniem około trzech sekund:

| Wariant | SM min / mediana / max [MHz] | Moc mediana / max [W] | VRAM max [MiB] |
|---|---:|---:|---:|
| Native R15 | 1110 / 1260 / 1560 | 345,13 / 353,96 | 64031 |
| Upstream BF16 | 1125 / 1335 / 1590 | 346,39 / 354,47 | 79975 |

Limit mocy obu wariantów pozostawał równy 350 W. Odczyt chwilowej mocy może
przekraczać ustawiony limit; tabela podaje obserwacje. Rzeczywiste zegary
są niższe od żądanych 1755 MHz. Nie opisujemy tej sesji jako stałozegarowej.

## Metoda

Zamrożony korpus HTTP zawiera osiem pojedynczych pytań: PL i EN po
128, 512, 2048 i 8192 tokeny. Drugi korpus to 32 różne requesty dokumentowe: 18 z samym extract,
7 z czterema pytaniami classify/verify/compress/extract oraz 7 z sześcioma
pytaniami, w tym wariantami classify i verify. Liczymy całe requesty HTTP;
zależny wybór fragmentu w extract może zmienić liczbę operacji wewnętrznych.
Korpus obrazów jest osobny.

Ustawienia: jedna kolejność opcji, wyłączony cache stanu i prefiksu,
wyłączona osobna długa kolejka native. Admission jest wyższe niż liczba
klientów; błędy i 529 nie są traktowane jako szybkie odpowiedzi.
Mierzymy czas od wysłania requestu do odczytania odpowiedzi na loopbacku.
Przepustowość to poprawne, zakończone requesty podzielone przez czas fazy.

Screen: dwie rozgrzewki i trzy próbki na punkt drabiny, 32 requesty przy
1/8/32 klientach. Audyt sprawdza kompletność faz, liczbę tokenów, poprawność
HTTP i niezmienność pełnych natywnych odpowiedzi przy różnym obciążeniu.

Upstream domyślnie ma `MAX_BATCH_TOKENS=16384`. Dodatkowo sprawdzamy
4096 i 8192 ustawione w pamięci procesu. Checkout upstream pozostaje
niezmieniony. Wyniki konfiguracji domyślnej i dostrojonej rozróżniamy.

Potwierdzenie HTTP R15 używa kolejności A B B A A B B A. Pierwsze uruchomienie
każdego silnika jest odrzucane; pozostają trzy powtórzenia. Punkt drabiny ma
pięć próbek po dwóch rozgrzewkach. Fazy zadań mają po 128 requestów przy
C1/8/32 i 32 dokumenty rozgrzewki. Ruch mieszany ma 64 krótkie dokumenty
z czterema pytaniami o dokumenty 8192-tokenowe, przy C8/32. Media mają osobne
ABBA: jeden obraz COCO, osiem rozgrzewek, 16 próbek C1 i 32 próbki C4 na
uruchomienie. To ograniczony korpus obrazów, a nie wynik dla dowolnych mediów.

Podstawowa seria wyłącza osobny tor długich żądań (`long_tokens=0`), aby
zmierzyć wykonanie i dotychczasową kolejkę. Domyślny serwer ma próg 4096
i przydział 100 ms. Dodatkowe, osobne ekrany porównują ten tor, przydział
5000 ms oraz pojedynczy tor FIFO w ruchu mieszanym. Runner `--text-phases`
pozwala jawnie mierzyć wybrane obciążenia; ich audyt nie obejmuje pominiętych
faz. Takie serie rozgrzewają także oba długie dokumenty przed pomiarem.
Wyników tych ustawień nie łączymy z tabelą głównego ABBA.

## Potwierdzone HTTP: FIFO i budżet 16384

Wybrane ustawienie jest osobną konfiguracją serwera: `--schedule fifo`,
`--long-tokens 0`, `--max-batch-tokens 16384`. Nie zmienia globalnych wartości
domyślnych. Natywny backend nadal ogranicza własne grupy promptów do 8192
padded tokenów. To inny limit niż budżet zewnętrznej partii żądań serwera.

A B B A A B B A, pierwsza para odrzucona, po trzy zachowane powtórzenia.
Każde obejmuje 128 zadań przy C1/8/32 i 68 requestów mieszanych przy C8/32.
Audyty potwierdzają kompletność faz, brak błędów i stabilność bitów native.
Upstream to BF16 eager z budżetem 4096; binarka native jest tą samą R15 co
w głównym ABBA. Drabinę pojedynczych pytań i obraz mierzono w osobnych seriach.

| Klienci, zadania 2.1 | basal-rs [żądań/s] | upstream [żądań/s] | Mediana basal-rs / upstream [ms] | p95 basal-rs / upstream [ms] |
|---:|---:|---:|---:|---:|
| 1 | 1,884 | 1,567 | 432 / 487 | 1006 / 1325 |
| 8 | 2,035 | 1,659 | 3853 / 5244 | 4232 / 5862 |
| 32 | 2,166 | 1,900 | 14722 / 17347 | 14876 / 17744 |

Przewaga przepustowości zadań: C1 20,2%, C8 22,7%, C32 14,0%.

| Ruch mieszany | Klienci | basal-rs mediana / p95 [ms] | upstream mediana / p95 [ms] |
|---|---:|---:|---:|
| Zadania | 8 | 3855 / 8519 | 5308 / 12175 |
| Dokumenty 8192 | 8 | 8338 / 8519 | 10316 / 11026 |
| Zadania | 32 | 17643 / 28954 | 23391 / 30635 |
| Dokumenty 8192 | 32 | 18953 / 28420 | 17042 / 27135 |

Przepustowość całej fazy mieszanej: C8 1,351 / 1,017 żądań/s (native / upstream), C32 1,400 / 1,097 żądań/s (native / upstream).

Powyższe p95 długich dokumentów jest maksimum czterech próbek w każdym
powtórzeniu. Zakresy między powtórzeniami są w `analyses/audit-confirm-production-final-r15.json`.

Długie dokumenty C8: p95 native jest o 22,7% niższe niż upstream.
Długie dokumenty C32: p95 native jest o 4,7% wyższe niż upstream.

Uruchomienie wybranego ustawienia na zmierzonej parze model/karta:

```sh
basal serve --model "$BASAL_ULTRA_MODEL" --dtype auto --inference-mode auto \
  --gemm-table auto --schedule fifo --long-tokens 0 --max-batch-tokens 16384 \
  --orders 1 --state-cache-mb 0 --no-prefix-cache \
  --max-inflight 1024 --max-large-inflight 128
```

`BASAL_ULTRA_MODEL` wskazuje lokalny katalog przypiętej rewizji. Pomiar używał
budżetu OMP/MKL/Rayon=8 i jednego GPU H100 PCIe. Nie przenosimy ustawienia
partii na inne modele lub karty bez pomiaru pamięci i zgodności.

## Potwierdzone HTTP: jeden tor, HRRN

Audyt `analyses/audit-confirm-text-final-r15.json`: **AUDITED**, zero problemów,
104 kompletne fazy. Tabele podają medianę metryk trzech zachowanych powtórzeń.
Niższy czas jest lepszy; wyższa liczba zakończonych poprawnych żądań/s jest lepsza.
Upstream to BF16 eager z limitem 4096, native to R15 z automatycznym FP16,
tabelą GEMM i readoutem FP32. Oba serwery używają tych samych wejść.

| Język | Tokeny | basal-rs [ms] | upstream BF16 [ms] | Krótszy czas |
|---|---:|---:|---:|---:|
| EN | 128 | 53,79 | 85,84 | 37,3% |
| EN | 512 | 137,55 | 167,13 | 17,7% |
| EN | 2048 | 820,55 | 972,15 | 15,6% |
| EN | 8192 | 4602,95 | 7080,00 | 35,0% |
| PL | 128 | 54,42 | 85,64 | 36,4% |
| PL | 512 | 138,20 | 168,26 | 17,9% |
| PL | 2048 | 824,67 | 977,23 | 15,6% |
| PL | 8192 | 4631,73 | 7083,43 | 34,6% |

| Klienci, zadania 2.1 | basal-rs [żądań/s] | upstream [żądań/s] | Mediana basal-rs / upstream [ms] | p95 basal-rs / upstream [ms] |
|---:|---:|---:|---:|---:|
| 1 | 1,875 | 1,563 | 435 / 487 | 1008 / 1325 |
| 8 | 2,034 | 1,655 | 3856 / 5161 | 4228 / 5571 |
| 32 | 2,173 | 1,896 | 12501 / 17090 | 19123 / 17952 |

Requesty w korpusie mają od 8 do 25 gałęzi modelu. Przykładowy
request w zachowanym powtórzeniu C32 miał około 18,5 s kolejki i 6,6 s
obliczeń partii 13 żądań; ten sam request przy C1 trwał około 1,01 s wobec
1,33 s upstream. Nagłówki obejmują obliczenie całej partii, więc nie są
osobnym czasem wykonania jednego requestu na GPU. Pokazują także koszt
oczekiwania i zakończenia dużej partii, poza samym forwardem.

Wzrost przepustowości względem upstream: C1 19,9%, C8 22,9%, C32 14,6%.
Mediana zadań C1 jest niższa o 10,7%. Wszystkie osiem punktów pojedynczego
pytania ma niższą medianę oraz obserwowane p95. p95 pojedynczych punktów ma
tylko pięć próbek na powtórzenie, więc jest maksimum tej małej próby.

Przy C32 zadań p95 native jest **o 6,5% wyższe** mimo większej przepustowości
i niższej mediany. W ruchu mieszanym native osiąga około 1,352 i 1,413
żądań/s przy C8/32 wobec 1,015 i 1,095 upstream. Krótkie dokumenty kończą
się szybciej, lecz długie mają gorszy czas oczekiwania. To zachowany punkt
do oceny kolejki; nie deklarujemy przewagi każdego percentyla.

| Ruch mieszany | Klienci | basal-rs mediana / p95 [ms] | upstream mediana / p95 [ms] |
|---|---:|---:|---:|
| Zadania | 8 | 3863 / 7870 | 5301 / 12221 |
| Dokumenty 8192 | 8 | 11919 / 16575 | 10468 / 11005 |
| Zadania | 32 | 11673 / 17718 | 23445 / 30600 |
| Dokumenty 8192 | 32 | 35615 / 41700 | 17020 / 27178 |

Faza mieszana zawiera cztery długie żądania na powtórzenie. Jej p95 dla
tej klasy jest maksimum czterech próbek; nie jest estymacją ogólnego ogona
dla dowolnego ruchu. Pełne rozkłady, zakresy między uruchomieniami, tokeny,
gałęzie oraz odpowiedzi zachowano w archiwum.

## Dobór kolejki i budżetu partii

Osobne ekrany pozostają **SIGNAL**: po jednym uruchomieniu, 34 requesty
mieszane przy C8/32. Ekrany FIFO obejmują też 128 requestów zadań przy C1/8/32.
Nie zastępują potwierdzenia na 68 requestach mieszanych. Audyty nie wykazały
błędów ani zmian bitów odpowiedzi native względem głównej serii.

| Ustawienie | Budżet partii | Zadania C32: żądania/s / p95 [ms] | Mieszane C32: żądania/s | Długie C32: p95 [ms] |
|---|---:|---:|---:|---:|
| HRRN, długi tor 4096 / 100 ms | 8192 | nie mierzono w tym ekranie | 1,411 | 19243 |
| HRRN, długi tor 4096 / 5000 ms | 8192 | nie mierzono w tym ekranie | 1,400 | 10176 |
| FIFO, jeden tor | 4096 | 2,055 / 18330 | 1,366 | 21370 |
| FIFO, jeden tor | 8192 | 2,143 / 18169 | 1,384 | 17785 |
| FIFO, jeden tor | 16384 | 2,175 / 15039 | 1,399 | 19632 |

Dłuższy przydział długiego toru zmniejsza czas dużych dokumentów kosztem krótkich. Zmniejszenie
budżetu partii do 4096 pogorszyło przepustowość i p95 samych zadań. FIFO z
budżetem 16384 wybrano do osobnego ABBA: poprawia p95 zadań C32 przy zachowanej
przepustowości. Długie żądania w małym screenie nadal mają inny rozkład niż
przy FIFO 8192, więc dopiero pełne porównanie rozstrzyga ten wybór.

Kontrola upstream z budżetem 8192 zamiast 4096, przy 64 requestach zadań,
dała około 1,591 / 1,477 / 1,891 żądania/s przy C1/8/32. Native w tej samej
krótkiej parze osiągnął 1,894 / 2,042 / 2,181. Jest to **SIGNAL**;
nie ma podstaw, by uznać 8192 za szybszą konfigurację upstream w tym korpusie.
Domyślne 16384 wcześniej skończyło się OOM. Wszystkie próby pozostają w danych.

## Potwierdzone HTTP: obraz

Audyt `analyses/audit-confirm-media-final-r15.json`: **AUDITED**, zero problemów,
16 kompletnych faz. Zachowano po trzy powtórzenia silnika: 48 próbek C1
i 96 próbek C4 na wariant, po odrzuceniu pierwszych uruchomień.

| Klienci | basal-rs mediana / p95 [ms] | upstream mediana / p95 [ms] | basal-rs / upstream [żądań/s] |
|---:|---:|---:|---:|
| 1 | 699,28 / 701,98 | 1339,33 / 1343,28 | 1,429 / 0,747 |
| 4 | 2806,32 / 2814,14 | 5345,57 / 5355,72 | 1,424 / 0,748 |

Mediana jest niższa o około 47,8% przy C1 i 47,5% przy C4. Przepustowość
C4 jest wyższa o około 90%. Obserwowane p95 także jest niższe. Pełne
natywne odpowiedzi zachowały bity między C1/C4 i powtórzeniami.

To wynik dla **jednego obrazu COCO** i tego samego pytania. Powtórzenia
potwierdzają stabilność tego przypadku; nie zastępują zróżnicowanego korpusu
obrazów, wielu obrazów w żądaniu ani innych modalności. Jakość modelu
wobec FP32 kontrolowano osobno, w tym na czterech obrazach.

## Zgodność i jakość

Główna referencja FP32 jest kompletnym, zachowanym eksportem poprzedniej
kampanii, liczonym z offloadem CPU/GPU. Wejścia i rewizja modelu są te same.
Poprzednia kontrola wobec CPU FP32 obejmowała 69 odczytów i zachowała
wszystkie wybrane odpowiedzi. W tej sesji dochodzi osobna referencja FP32
PL/EN dla długich wejść drabiny; dziewięć pierwszych odczytów i osiem
odpowiedzi API zachowało decyzje FP32.

R13 zachował bitowo pełne eksporty R12: 49 rekordów benchmarku,
49 rekordów API i osobne 1+8 rekordów długości do 8192. R12 jest bitowo równy
R11. R11 celowo zmienia nieafiniczne normowanie V. Wcześniejsze R7–R10
zachowały bity starego native; kontrola `accurate` R13 zachowała 49+49 rekordów
poprzedniej ścieżki. `verification/final-parity-r13.json` zawiera wszystkie
liczności i dziesięć kontroli, każda PASS. R14 łączy układanie, powtarzanie
GQA i konwersję Q/K/V do docelowych macierzy FP32; eksporty główne i długości
są bitowo identyczne z R13. R15 poprawia wyłącznie formatowanie komentarza
Rustdoc. Końcowa binarka R15 zachowała bitowo eksporty główne i długości R14.
Tryb `accurate` zachował 49+49 rekordów R13. Sześć końcowych kontroli ma PASS
w `verification/final-parity-r15.json`.

| Korpus i wariant | Argmax pierwszych odczytów | Średnia TV skalibrowanych prawdopodobieństw | Decyzje i dane API względem FP32 |
|---|---:|---:|---|
| Główny, stary FP16 / R8–R10 | 161/161 | 0,0006366214 | PASS |
| Główny, końcowy R15 | 161/161 | 0,0006319363 | PASS; 65 pytań bez zmian |
| Główny, upstream BF16 | 160/161 | 0,0019138364 | FAIL; różnica verify |
| Długości, stary FP16 / R8–R10 | 9/9 | 0,00000302216 | PASS |
| Długości, końcowy R15 | 9/9 | 0,00000343703 | PASS; 8 pytań bez zmian |
| Długości, upstream BF16 | 9/9 | 0,00000894150 | PASS |

Końcowy native ma 3,03 razy mniejszą średnią TV od BF16 w głównym korpusie
oraz 2,60 razy mniejszą w kontroli długości. W głównym korpusie TV poprawia
się o 0,74% względem starego FP16, w kontroli długości pogarsza o 13,7%.
To zachowana regresja numeryczna, mimo zgodnych decyzji i przewagi nad BF16.
Nie ma zmian automatyzacji, błędów, zakresów extract ani sparsowanych danych
w porównaniu końcowego native z FP32 w tych korpusach.

Ponowny eksport BF16 w bieżącej sesji dał te same metryki i różnicę decyzji
co zachowany BF16. Końcowy audyt to `runtime/quality-summary-current-bf16-r15/`;
przedstawiona tabela używa tego nowego eksportu.

Upstream BF16 zmienił w `tasks-pl/verify` odpowiedź `not_stated` na
`contradicted`. Prawdopodobieństwa native nie są bitowo identyczne z FP32.
Tabela nie jest miarą ogólnej dokładności modelu. Surowe porównania zachowują
NLL, Brier, kalibrację i pełne odpowiedzi po kontynuacji extract.

Jakość BF16 w tabeli dotyczy eksportu pojedynczych wejść. W seriach HTTP
upstream zmienia bity części odpowiedzi wraz z partią; nie utożsamiamy tego
licznika ze zmianą decyzji. Native musi zachować całe odpowiedzi bitowo między
klientami i powtórzeniami. Dane zachowują odpowiedzi oraz liczbę tokenów i gałęzi
każdego silnika; nie zakładamy identycznej kontynuacji extract dla każdego batcha.

## Rozpoznanie wydajności i wybór zmian

Profilowanie wykonano po załadowaniu i rozgrzaniu serwera: opóźnione
przechwycenie 30 sekund, cztery różne requesty z zadaniami albo dwa punkty
512/2048 tokenów. Czasy profilera są diagnostyką; nie zastępują pomiaru HTTP.

W bazowym profilu zadań `bmul_f16` zajmował 16,9% czasu kerneli,
`ucopy_f16` 9,6%, a dwa kierunki konwersji FP16/FP32 łącznie 8,9%.
W profilu drabiny jeden z kerneli FP32 GEMM zajmował 28,1%.
To wskazało także operacje elementowe i układ danych, poza samym attention.

Końcowe profile R15 mają status completed: 54 pełne requesty zadań i 61
requestów drabiny wewnątrz przechwycenia. Poniżej udział w czasie kerneli
każdej osobnej sesji profilera, nie stosunek czasów całych requestów:

| Grupa kerneli | Zadania: baza / R15 | Drabina: baza / R15 |
|---|---:|---:|
| Osobne `bmul_f16` | 16,94% / 0,00% | 13,50% / 0,00% |
| Kopie `ucopy` i `copy2d` | 13,93% / 3,97% | 11,46% / 5,37% |
| Konwersje FP16–FP32 | 8,82% / 1,25% | 7,43% / 0,88% |
| Nowe układanie attention | 0,00% / 10,13% | 0,00% / 6,53% |
| Nowe połączone GELU/RoPE/skalar | 0,00% / 6,74% | 0,00% / 4,51% |

Największy kernel FP32 GEMM stanowi teraz 41,07% czasu kerneli drabiny.
To wskazuje dalszy koszt attention po usunięciu części obsługi danych;
nie dowodzi, że jego bezpośrednie zastąpienie zachowa numerykę. Różne liczby
requestów i narzut profilera wykluczają liczenie przyspieszenia HTTP z tych
udziałów. `profiles-summary-final.json` oraz CSV zachowują pełne wartości.

Sprawdzone warianty:

- R2: wspólne batchowanie macierzy attention. Tor FP32 zachował bitowo
  eksporty. Wariant FP16 z wynikami GEMM FP32, softmax FP32 i podziałem
  prawdopodobieństw na dwie części zachował wszystkie decyzje FP32, lecz
  średnia TV wzrosła do 0,0006421656. Mały zysk przy 2048 tokenach i regresja
  przy 8192 nie uzasadniały pozostawienia dodatkowego modułu cuBLAS.
- R3: szerokie łączenie długości zachowało wynik wariantu R2, ale zwiększyło
  pracę na paddingu. Czas C1 zadań wzrósł z około 659 do 721 ms.
- R7: bezpośredni RMSNorm FP16 z wewnętrzną redukcją i mnożeniem FP32
  usuwa trzy konwersje. Eksporty główne i długie są bitowo zgodne z bazą.
- R8/R9: połączone RoPE zachowuje osobne zaokrąglenia FP16 obu mnożeń
  i ich sumy. Attention zachowuje FP32. Niepotrzebne K/V modelu bez
  współdzielenia między warstwami są przechowywane tylko dla kontynuacji.
  Przywrócono dotychczasowe kubełki długości i limit batcha.
- R10: wyrównanie sekwencji do 8 zamiast 32 i kernel skalara warstwy.
  Kubełki i limity pozostają bez zmian. Eksporty główne i długości zachowują
  bity R9; audyt krótkiego HTTP zachowuje pełne odpowiedzi przy C1/8/32.
- R11: nieafiniczny RMSNorm V przez kernel z jednostkową wagą i redukcją
  FP32. Zmienia numerykę; oba porównania z FP32 są w tabeli powyżej.
- R12: GELU połączone z mnożeniem przez projekcję up. Zachowano kolejność
  i zaokrąglenia FP16 przypiętego Candle 0.11, również przed końcowym
  mnożeniem. Eksporty główne i długości są bitowo równe R11.
- R13: końcowe ograniczenie automatycznego wyboru do Ultra, FP16, readout
  FP32 i dokładnej nazwy NVIDIA H100 PCIe. Wyjścia kerneli używają jawnego
  zapisu w cudarc, aby prawidłowo śledzić dostęp między strumieniami.
  `accurate` zachowuje poprzednią implementację; inne pary nie są objęte
  nowym automatycznym wyborem.
- R14: układanie Q/K/V bez pośrednich tensorów gather/permute/cast. Kernel
  wykonuje tylko mapowanie indeksów i dokładną konwersję FP16 do FP32.
  Wynik zachowuje dotychczasowy układ macierzy i arytmetykę attention.
  Eksporty 49+49 oraz 1+8 są bitowo identyczne z R13. Krótki screen poprawił
  C1 z około 475 do 431 ms i C32 z około 1,96 do 2,20 odpowiedzi/s.
- R15: poprawka zapisu kształtu w komentarzu Rustdoc po wykryciu błędnego
  odsyłacza. Kod obliczeń i kernele CUDA są takie same jak w R14; binarka
  i kontrola końcowa mają własne pochodzenie i sumy.

Krótki screen R8: EN 512 około 166 ms wobec około 201 ms starego native;
EN 8192 około 5122 ms wobec około 5672 ms. Komplet zadań przy C1: około
548 ms wobec 699 ms starego native i 487 ms upstream. Przy C8 native
osiągnął medianę około 4792 ms. Te dane pozostają **SIGNAL**.

Domyślny upstream w screenie przy C8 odpowiedział 31 błędami HTTP 422
z OOM i jedną poprawną odpowiedzią. C32 nie wykonano. To **FAIL**,
a nie przewaga czasu obsługi błędów. Kontrola 4096 przeszła fazy 1/8/32.

Serię ABBA R13 przerwano po pierwszej parze i części kolejnego uruchomienia,
aby wybrać wariant z mniejszą liczbą kopii. Zachowano obserwacje jako **GAP**;
nie są potwierdzonym wynikiem końcowym. Pełne ABBA dotyczy osobnej serii R15.
Pierwsza próba profilu używała błędnego endpointu i nie dostarczyła poprawnych
próbek; zastąpiły ją nowe, osobne profile. Zachowano też nieudaną analizę
z powtórzoną etykietą kandydata (`long-quality-summary-r3`) i skorygowaną
analizę w nowym katalogu. Te błędy narzędzi nie są wynikami inferencji.

## Kontrole kodu

Nie dodano testów ani bramek CI. Dla końcowego R15 przeszły formatowanie, Clippy i
Rustdoc CPU oraz Clippy i Rustdoc CUDA ze wszystkimi features. CUDA buduje
PTX dla SM 80, 89 i 90. Graf zależności jest niezmieniony; zachowano wynik
`cargo deny` i kontroli wyjątku `paste`. Przegląd objął granice własnych kerneli,
kontrolę kształtów i urządzeń, zaokrąglenia, retencję K/V, zakres `auto`
i zachowanie dotychczasowego trybu `accurate`. Nie zmieniono zależności ani
publicznych sygnatur API Rust. Diagnostyka backendu opisuje wybrane operacje
i wyrównanie tekstu. Linty nie potwierdzają runtime GPU; od tego są eksporty
i obserwacje HTTP. Osobna kompilacja CPU `--release --locked --offline`
z kompletnego, zredagowanego drzewa archiwalnego także przeszła; jego 204
pliki pozostały identyczne z R15. Nie jest to powtórny build CUDA z archiwum.

## Decyzja i ograniczenia

Pozostawiamy zoptymalizowane `auto` wyłącznie dla Ultra FP16 z readoutem FP32
na H100 PCIe. `accurate` zachowuje wcześniejsze operacje. Do zmierzonego
obciążenia zadań wybieramy jawne FIFO/16384; nie zmieniamy globalnej polityki
serwera. Zyskuje p95 zadań C32 kosztem wyższej mediany niż w osobnej serii HRRN.
Różnica p95 długich dokumentów C32 pozostaje otwarta; nie deklarujemy przewagi
każdego percentyla ani każdego rodzaju ruchu.

Dalsze hipotezy wynikające z pomiaru to ograniczenie kosztu FP32 attention
w drabinie oraz lepszy podział czasu GPU między długie i krótkie żądania.
Każda zmiana redukcji wymaga nowej kontroli FP32, a kolejki — ponownego
potwierdzenia całych klas requestów. Nie ma tu zmierzonego zysku dla tych
hipotez, innych kart, innych modeli, zimnego startu ani innych modalności.

Przed zamknięciem potwierdzono SHA-256 732 plików z maszyny pomiarowej
(około 1,01 GB) oraz 37 logów budowania. Przy braku aktywnych procesów GPU
przywrócono ustawienie zegara i usunięto instancję; lista dostawcy potwierdziła
jej brak. Własny katalog i tag obrazu CPU zostały usunięte po retencji logów.
Stany istniejących kontenerów aplikacji pozostały niezmienione.
`retention-proof-public.json`, `cleanup-proof-public.json` i `review-final.json`
zawierają zminimalizowane potwierdzenia bez identyfikatorów maszyn.

## Dane i odtworzenie

Archiwum opublikowano pod nowym, niezmiennym prefiksem. Pobranie z autoryzacją
oraz anonimowo potwierdziło SHA-256 archiwum i **1133/1133 sum plików**.
Lokalna kopia archiwum pozostaje zachowana. Dane identyfikujące maszyny,
rozliczenie i surowe ślady Nsight pozostają prywatne. Zdjęcia zastąpiono
hashami i odtwarza się je osobno z przypiętych źródeł COCO.

| Pole | Wartość |
|---|---|
| URI | `hf://buckets/itsoltech/basal-rs/measurements/ultra-attention-2026-10-17/c298d99630c826cf6a1972aaf5ba477060f9f0c9e88427996213adad8b55f095/measurement-data.tar.gz` |
| SHA-256 | `c298d99630c826cf6a1972aaf5ba477060f9f0c9e88427996213adad8b55f095` |
| Rozmiar gzip | 11 170 513 B |
| Katalog kampanii w archiwum | `reports/ultra-attention-2026-10-17/` |

Ścieżki poniżej są względne do katalogu kampanii:

| Ścieżka | Zawartość |
|---|---|
| `reference-fp32/`, `quality/exports/` | Główna referencja, nowy BF16, eksporty wariantów native |
| `long-quality/exports/` | Referencje FP32/BF16 i warianty długości |
| `runtime/confirm-text-final-r15/`, `runtime/confirm-production-final-r15/`, `runtime/confirm-media-final-r15/` | Metody, pełne obserwacje i postęp trzech potwierdzeń HTTP |
| `runtime/screen-*/`, `analyses/` | Kontrole, odrzucone warianty i audyty, w tym FAIL/GAP/SIGNAL |
| `runtime/profile-*/`, `profiles-summary-final.json` | Diagnostyczne CSV kerneli/API i zestawienie udziałów |
| `verification/`, `cross-settings-bits-r15.json` | Pochodzenie, parytet eksportów, linty i zgodność między kolejkami |
| `sources/build-tree-r15/`, `sources/r*/` | Kompletne końcowe drzewo budowania i źródła wariantów |
| `tools/`, `method-sources/`, `REPRODUCE.md` | Przenośne analizy, dokładne wykonane skrypty i polecenia |

Publiczne odtworzenie potwierdziło oryginalne sumy czterech wejść HTTP,
identyczne audyty dziewięciu serii oraz metryki pięciu wariantów jakości
w obu korpusach. `public-analysis-replay-final.json` zapisuje tę kontrolę.
Wszystkie dane i narzędzia użyte w tym odtworzeniu pozostały identyczne
w końcowym archiwum. 204 pliki drzewa budowania są identyczne z R15.

Pobranie i kontrola integralności, z katalogu głównego repozytorium.
Katalogi docelowe muszą być nowe:

```sh
BASAL_ULTRA_PREFIX=measurements/ultra-attention-2026-10-17/c298d99630c826cf6a1972aaf5ba477060f9f0c9e88427996213adad8b55f095
BASAL_ULTRA_DL=.cache/measurement-download/ultra-attention-2026-10-17
BASAL_ULTRA_ARTIFACTS=.cache/measurement-artifacts/ultra-attention-2026-10-17-replay
mkdir -p .cache/measurement-download .cache/measurement-artifacts &&
  mkdir "$BASAL_ULTRA_DL" &&
  hf buckets cp "hf://buckets/itsoltech/basal-rs/$BASAL_ULTRA_PREFIX/measurement-data.tar.gz" "$BASAL_ULTRA_DL/measurement-data.tar.gz" &&
  echo "c298d99630c826cf6a1972aaf5ba477060f9f0c9e88427996213adad8b55f095  $BASAL_ULTRA_DL/measurement-data.tar.gz" | sha256sum -c - &&
  mkdir "$BASAL_ULTRA_ARTIFACTS" &&
  tar -xzf "$BASAL_ULTRA_DL/measurement-data.tar.gz" -C "$BASAL_ULTRA_ARTIFACTS" &&
  (cd "$BASAL_ULTRA_ARTIFACTS" && sha256sum -c SHA256SUMS)
```

Następnie użyj `REPRODUCE.md` wewnątrz kampanii. Runnery analityczne czytają
wyniki przez `--measurement-artifacts`; nie jest to opcja CLI basal. Przywróć
media narzędziem `tools/reference/tasks-v21/restore-inputs.py` z osobno pobranych,
zweryfikowanych zasobów przed audytem HTTP. Porównanie jakości i audyt obserwacji
nie uruchamiają modelu ani nowej maszyny GPU.

`README-measurement-snapshot.md` wewnątrz archiwum jest kopią sprzed publikacji
i dopisania jego własnej sumy. Aktualny raport w Gicie zawiera potwierdzony URI,
SHA-256 i stan pobrania. Zakończonego prefiksu nie nadpisujemy.
