# Ultra: repliki H100 i podział warstw dla tekstu oraz obrazów

Pomiar zakończony 2026-10-18. Ultra działa z podziałem 60 warstw dekodera między
**dwie RTX 6000 Ada 48 GB**, dla tekstu i obrazów. Najwyższe próbkowane użycie
VRAM procesu wyniosło **36,38 GiB na kartę**, **71,22 GiB łącznie**; PSS API
**1,68 GiB**. To potwierdzenie pojemności i zgodności badanego zakresu.

Osobny pomiar **replik** na H100 wykazał przy C128 **1,98× przepustowości** dla
krótkich zadań: **4,30 zamiast 2,16 poprawnej odpowiedzi/s**. Ruch mieszany
z dokumentami 8k osiągnął **1,91×**, 2,70 zamiast 1,41 odpowiedzi/s.
Porównujemy tutaj basal-rs na jednej i dwóch kartach. Ta kampania nie mierzy
ponownie przepustowości upstream.

**Ograniczenia jakości pozostają jawne.** Kontrola umieszczenia warstw na H100
jest zgodna bitowo ze starą binarką; wszystkie 234 odpowiedzi Ada są zgodne z
jej sekwencyjną kontrolą. Jednak kontrola Ada względem FP32 ma **FAIL: 160/161**
zgodnych argmaxów głowic i **64/65** zgodnych odpowiedzi kategorycznych.
Upstream BF16 zmienia tę samą decyzję. Natywny średni błąd TV jest 2,20× mniejszy
niż BF16, ale pełny cel zgodności decyzji FP32 nie został osiągnięty.
W pomiarze replik przy C128 są też **226 różnic liczbowych odpowiedzi**, bez
zmian decyzji; ścisła kontrola niezmienności bitowej ma FAIL.

Materiał dla strony: [metryki i dozwolone sformułowania](landing-page.md).

## Co zmienia silnik

Gemma może rozmieścić kolejne zakresy warstw na oddzielnych kartach CUDA.
Pierwsza karta ma enkodery mediów i embedding wejściowy; ostatnia końcową normę
oraz drugą kopię powiązanego embeddingu do odczytu kodów. Każdy etap ma własne
wagi warstw, KV, rotary, mapy GQA, maski obrazu i uchwyt cuBLASLt. Etapy
przekazują pełne aktywacje w kolejności warstw. Synchronizacja, retencja po
niepotwierdzonym kopiowaniu i współdzielone health używają mechanizmu Llama.
Jedno wywołanie trzyma najwyżej bramkę jednej karty.

Podział wymaga kart jednego typu, jawnego FP16, gęstych wag (`quantization:
none`), fused kernels i dokładnego pokrycia warstw. PLE (per-layer input
embeddings) i współdzielone KV między warstwami są odrzucane przed wagami.
Ultra w tej rewizji spełnia te warunki: Gemma 4, 60 warstw, `unified: false`,
bez PLE i bez cross-layer shared KV. Enkoder obrazu zachowuje BF16.
Dotychczasowa sygnatura publicznego `load_split` pozostaje dostępna;
nowe wejście `load_split_with_media` obsługuje media.

```bash
basal serve --model omni-ultra --dtype f16 --media --layer-split 0:30,1:30
```

To tryb zwiększający dostępną pamięć dla jednego modelu. Repliki trzymają
pełne wagi na każdej karcie i rozdzielają niezależne żądania. Podział warstw
nie oznacza automatycznie przyspieszenia pojedynczego pytania ani równoległego
wykonania warstw jednego pytania.

## Wersje i sprzęt

Checkpoint: `Remek/basal-omni-1.0-ultra` rewizja
`b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4`. Upstream użyty do przygotowania
wejść i zachowanych referencji jakości:
`62542d31e8eb69000839c01e8f9b45da1fb80e49`.

| Wariant | Źródła w archiwum | SHA-256 binarki CUDA |
|---|---|---|
| bazowy, main `0da3769c95a0e20e53b3717eb132954d458a89ed`; repliki i stara kontrola | `sources/baseline-tree` | `f91b91d5b7277c905cf008529db90fc8ddc12305859b246e0a51993853d23795` |
| podział R5, pojemność Ada | `sources/ada-capacity-tree-r5` | `eb2cb8866406e11f770873bd7f8177f5c96ca22afae6a0ff87c2f4622d59cb2c` |
| końcowy R6, kontrola H100/Llama i eksport jakości Ada | `sources/final-tree-r6` | `7b70ebc35305e6de3bab950e0edcf8a7f08e3d15179f81d5ac1bfb64492512a5` |

Każde drzewo ma 202 pliki i `SOURCE-SHA256SUMS`. R6 dodaje do R5 walidację
nagłówków checkpointu: długość, liczbę i granice offsetów. Nie zmienia arytmetyki.
Końcowe źródła Rust w drzewie roboczym są zgodne z R6. Rust 1.95.0,
CUDA 12.9.1, kompilacja dla capabilities 80/89/90; dokładne wersje i komendy
są w archiwum. Binarne programy, biblioteki i checkpointy pozostają poza
publicznym archiwum; ich hashe są zachowane.

| Cecha | kampania H100 | kampania Ada |
|---|---|---|
| GPU | 2 × NVIDIA H100 PCIe, 81 559 MiB każda | 2 × RTX 6000 Ada Generation, 49 140 MiB każda |
| sterownik | 580.126.20 | 580.126.09 |
| fabryczny limit mocy/kartę | 350 W | 300 W |
| maks. SM / pamięć | 1755 / 1593 MHz | 3105 / 10001 MHz |
| CPU | AMD EPYC 9334 32-Core; 48 widocznych CPU logicznych | Intel Xeon Gold 6338 @ 2 GHz; 26 widocznych CPU logicznych |
| RAM hosta | 507 001 946 112 B | 152 073 560 064 B |

Ustawienia mocy i zegarów nie były zmieniane. Rejestrowano rzeczywiste zegary,
wykorzystanie, temperaturę i moc. Limit wątków bibliotek CPU wynosił 16.
Maszyny różnią się CPU i platformą; tych dwóch kampanii nie należy traktować
jako kontrolowanego porównania szybkości H100 i Ada.

## Repliki H100: metoda

Jedna i dwie repliki używają tej samej zamrożonej binarki i checkpointu.
Kolejność **ABBA: jedna, dwie, dwie, jedna** daje dwa zachowane starty na wariant.
Każdy start obejmuje C1/C2/C8/C16/C32/C128 i dwa obciążenia:

- `tasks`: 32 różne krótkie dokumenty, sześć zadań na żądanie:
  classify, multilabel, verify, binary, compress i extract;
- `mixed`: te same krótkie dokumenty oraz dokument 8192 tokeny na każde
  16 krótkich; razem 34 różne wejścia, dwa różne długie dokumenty.

Jedno HTTP żądanie w tabelach krótkich zadań obejmuje sześć pytań. QPS nie
oznacza liczby pojedynczych pytań na sekundę. Długości dotyczą zamrożonych
wejść/tokenizacji, a nie bajtów dokumentu.

Każda faza trwa co najmniej 30 s, cztery pełne obiegi korpusu i cztery żądania
na klienta. Zlecone żądania są rozliczane do końca. Łącznie **48 faz, 9632
mierzone odpowiedzi**, 272 odpowiedzi rozgrzewki i 68 niezależnych kontroli:
**9972 odpowiedzi HTTP**, wszystkie poprawne strukturalnie, bez 529.
Klienci działają w pętli zamkniętej: kolejna odpowiedź uruchamia kolejne żądanie.

FP16, odczyt FP32, `inference_mode: auto`, niezmienna względem partii tabela
H100, `orders: 1`, FIFO, bez prefix/state cache, bez oddzielnego toru długich
żądań, limit partii 16384 tokeny na replikę. `max_inflight` i
`max_large_inflight` wynoszą 1024, aby limit przyjęć nie zastępował pomiaru
pojemności. Liczniki routingu, opróżnienie kolejek oraz ciągłość próbkowania
mają kontrolę PASS. Niezależna kontrola GPU1 ma 34/34 wyniki bitowo zgodne z GPU0.

Latencja obejmuje serializację JSON, połączenie/HTTP oraz pełny odczyt i
zdekodowanie odpowiedzi. W tabelach podajemy **medianę dwóch metryk fazowych**;
kwantyle nie są liczone z połączonej próby obu startów. Skalowanie to iloraz
median QPS. Dwa starty nie dają mocnej oceny istotności ani stabilności w czasie.

### Przepustowość i czas odpowiedzi H100

C oznacza liczbę równoległych klientów. Wszystkie czasy są w milisekundach.

| Ruch | C | 1 GPU odp./s | 2 GPU odp./s | skalowanie | p50 1 GPU | p50 2 GPU | p95 1 GPU | p95 2 GPU | p99 1 GPU | p99 2 GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tasks | 1 | 1.841 | 1.913 | 1.039× | 442.2 | 425.2 | 1033.4 | 1005.8 | 1035.3 | 1008.2 |
| tasks | 2 | 1.836 | 3.749 | 2.042× | 940.8 | 436.5 | 1606.7 | 1023.4 | 1608.1 | 1028.7 |
| tasks | 8 | 1.989 | 3.797 | 1.909× | 3945.9 | 1954.6 | 4322.3 | 3001.9 | 4846.3 | 3005.3 |
| tasks | 16 | 2.072 | 3.931 | 1.897× | 7505.8 | 4009.3 | 7965.5 | 4576.5 | 7968.0 | 4811.0 |
| tasks | 32 | 2.107 | 4.119 | 1.955× | 15167.4 | 7367.6 | 15187.2 | 8145.7 | 16990.6 | 8148.5 |
| tasks | 128 | 2.165 | 4.296 | 1.984× | 53975.1 | 27363.7 | 74846.5 | 32723.1 | 91186.8 | 36330.6 |
| mixed | 1 | 1.273 | 1.317 | 1.035× | 442.9 | 425.8 | 4665.6 | 4524.2 | 4678.4 | 4591.5 |
| mixed | 2 | 1.271 | 2.474 | 1.947× | 940.3 | 439.1 | 5707.7 | 4609.3 | 5714.0 | 4649.7 |
| mixed | 8 | 1.337 | 2.491 | 1.864× | 4298.9 | 2146.0 | 8659.0 | 6649.0 | 8660.5 | 6948.9 |
| mixed | 16 | 1.364 | 2.547 | 1.866× | 11668.1 | 4741.4 | 12660.0 | 8484.1 | 12664.9 | 8551.3 |
| mixed | 32 | 1.382 | 2.682 | 1.941× | 19525.8 | 11280.6 | 32616.3 | 14032.5 | 32687.1 | 14565.3 |
| mixed | 128 | 1.412 | 2.700 | 1.913× | 78783.6 | 38531.4 | 128251.6 | 64749.5 | 148743.2 | 78086.4 |

Przy C1 druga karta ma mało pracy: wzrost QPS wynosi około 3–4%. Przy C2–C128
krótkie zadania osiągają 1,90–2,04×. Nie oznacza to dwukrotnie krótszego
wykonania pojedynczego pytania bez kolejki. Duże C daje bardzo długie kolejki;
p95 przy C128 nadal wynosi około 32,7 s dla krótkich zadań i 64,7 s dla mieszanych.

### Zasoby H100

CPU to średnia liczba zajętych rdzeni logicznych: 1,00 oznacza około 100%
jednego CPU, a nie 100% całej maszyny. PSS to proporcjonalnie rozliczona pamięć
procesów API. Pamięć i CPU generatora oraz próbnika nie należą do API.
W tabeli są mediany dwóch średnich ważonych czasem; VRAM to suma pamięci procesu
na kartach. GiB = 2³⁰ B.

| Ruch | C | CPU 1 GPU | CPU 2 GPU | PSS GiB 1 GPU | PSS GiB 2 GPU | VRAM GiB 1 GPU | VRAM GiB 2 GPU | CPU s/odp. 1 GPU | CPU s/odp. 2 GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tasks | 1 | 0.998 | 0.998 | 0.999 | 1.368 | 63.055 | 125.922 | 0.5430 | 0.5225 |
| tasks | 2 | 1.000 | 1.993 | 0.999 | 1.368 | 63.055 | 125.922 | 0.5453 | 0.5328 |
| tasks | 8 | 0.999 | 1.967 | 0.999 | 1.369 | 63.055 | 125.922 | 0.5034 | 0.5195 |
| tasks | 16 | 0.999 | 1.952 | 1.000 | 1.369 | 63.055 | 125.922 | 0.4830 | 0.4976 |
| tasks | 32 | 0.999 | 1.957 | 1.001 | 1.370 | 63.055 | 125.922 | 0.4749 | 0.4761 |
| tasks | 128 | 1.001 | 1.981 | 1.016 | 1.379 | 63.055 | 126.072 | 0.4623 | 0.4616 |
| mixed | 1 | 0.999 | 0.999 | 1.000 | 1.369 | 63.174 | 126.538 | 0.7855 | 0.7594 |
| mixed | 2 | 1.000 | 1.913 | 1.000 | 1.372 | 63.211 | 126.891 | 0.7877 | 0.7744 |
| mixed | 8 | 1.000 | 1.897 | 1.002 | 1.373 | 63.211 | 126.891 | 0.7489 | 0.7630 |
| mixed | 16 | 0.999 | 1.885 | 1.004 | 1.375 | 63.211 | 126.891 | 0.7333 | 0.7419 |
| mixed | 32 | 1.000 | 1.941 | 1.006 | 1.378 | 63.211 | 126.891 | 0.7243 | 0.7250 |
| mixed | 128 | 1.000 | 1.912 | 1.025 | 1.396 | 63.211 | 126.916 | 0.7089 | 0.7085 |

Przy krótkich zadaniach C128 dwie repliki zużywają około 1,98 rdzenia zamiast
1,00 i 1,38 GiB PSS zamiast 1,02 GiB. CPU na odpowiedź pozostaje niemal takie
samo: 0,4616 wobec 0,4623 s. VRAM rośnie około dwukrotnie, bo każda replika
trzyma pełny model. Nie jest to oszczędność VRAM ani dwukrotnie tańsza inferencja.

Próbkowanie odbywało się co około sekundę. Dane zawierają również RSS, USS,
swap, maksima próbkowane, per-GPU VRAM/utilization/power/temperature/SM oraz
okna bezczynności. Suma mocy obejmuje obie zainstalowane karty, także nieaktywną
w wariancie jednej karty. Utilization i zegar SM w agregacie dotyczą aktywnych
kart; temperaturę agreguje maksimum aktywnych kart. Globalny page cache hosta
nie jest RAM procesu. Rosnąca drabina C zachowuje historię alokatora; bez
resetowania pamięci między punktami. Maksima próbkowane mogą ominąć krótkie piki.

### Niezmienność odpowiedzi H100

Audyt struktury, zasobów i końcowego opróżnienia: **PASS**. Ścisłe porównanie
pełnych odpowiedzi: **FAIL**, 226 odpowiedzi z różnicami liczbowymi względem
niezależnej kontroli C1, wszystkie przy C128. Projekcja decyzji zachowuje typy,
kategorie, dane, wartości, offsety i automatyzację, pomijając confidence oraz
prawdopodobieństwa: **zero zmian, PASS**. Nie zmieniamy FAIL na PASS tylko
dlatego, że decyzje są zgodne. To kontrola niezmienności rodzimej inferencji,
nie nowy pomiar zgodności z FP32. Surowe różnice są w analizie i obserwacjach.

## Podział warstw: tekst i obrazy

Korpus zawiera 39 wejść: 32 krótkie dokumenty z sześcioma zadaniami, dwa
dokumenty 8192 tokeny, cztery pojedyncze zdjęcia COCO i parę dwóch obrazów
`high`. Ta para daje najdłuższy pojedynczy prompt 2273 tokeny, więc przechodzi
przez granicę prefilla 2048. `usage.input_tokens: 9085` jest sumą wielu promptów
w żądaniu, a nie długością pojedynczego promptu; `media_tokens: 8684` i dwa obrazy.
Źródła obrazów i SHA-256 są przypięte; zdjęcia pozostają zewnętrzne.

Dla podziału użyto `long_tokens: 4096`, odczytu FP32, `orders: 1`, bez prefix/state
cache. H100 używa tabeli niezmiennej względem partii; Ada przy braku pasującej
tabeli zachowuje stałe GEMM na prompt. Zachowano tę samą kolejność obliczeń.

### Kontrola bitowa na H100

Stara binarka na jednym GPU oraz R6: jedna karta, podział 30/30, odwrócone
karty 30/30 i nierówny podział 1/59. Każdy wariant wykonał po 39 żądań C1:
**195 odpowiedzi**, **156 porównań ze starą binarką**, wszystkie pełne odpowiedzi
zgodne bitowo, wszystkie żądania poprawne. To ograniczona kontrola poprawności,
nie powtarzane ABBA wydajności.

| Wariant | odpowiedzi | zgodność ze starą kontrolą |
|---|---:|---|
| stara, jedna karta | 39 | kontrola |
| R6, jedna karta | 39 | 39/39 |
| R6, 30/30 | 39 | 39/39 |
| R6, odwrócone 30/30 | 39 | 39/39 |
| R6, 1/59 | 39 | 39/39 |

### Pojemność na dwóch Ada 48 GB

Dwa starty, z podziałem 30/30 i odwróconą kolejnością kart; każdy obejmuje
C1/C2/C8, jeden obieg 39 wejść na punkt. **234/234 odpowiedzi poprawne**, wszystkie
zgodne bitowo z pierwszą sekwencyjną kontrolą dla danego wejścia. Nie uruchamiano
wariantu Ultra na jednej Ada do OOM: gęste wagi FP16 przekraczają pojemność jednej
48 GB karty. Pozytywna próba dwóch kart potwierdza badany zakres, a nie pełny limit
kontekstu modelu lub każdą możliwą liczbę obrazów.

Serwer ma istniejący limit czterech jednoczesnych żądań mediów. Końcowy runner
jawnie ogranicza klienta do czterech takich żądań. C8 oznacza osiem klientów łącznie,
nie osiem jednoczesnych żądań obrazowych. Czekanie na pozwolenie klienta jest
zapisane osobno; latencja HTTP zaczyna się po jego otrzymaniu. QPS fazy obejmuje
także ten czas. Żadne dane z tej próby nie wyznaczają maksymalnej przepustowości
obrazów ani stabilnego wyniku długiego obciążenia.

| Kolejność kart | C | odp./s | p50 ms | p95 ms | p99 ms | CPU rdzenie | PSS max GiB | VRAM max suma GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0:30,1:30 | 1 | 0.402 | 1336.9 | 10469.7 | 15310.6 | 0.998 | 1.604 | 68.215 |
| 0:30,1:30 | 2 | 0.457 | 2162.8 | 25513.1 | 27774.1 | 1.167 | 1.606 | 71.096 |
| 0:30,1:30 | 8 | 0.439 | 13457.8 | 31137.6 | 31490.9 | 1.173 | 1.631 | 71.221 |
| 1:30,0:30 | 1 | 0.395 | 1366.4 | 10670.5 | 15479.9 | 0.998 | 1.603 | 68.309 |
| 1:30,0:30 | 2 | 0.453 | 2187.6 | 25009.6 | 27920.1 | 1.165 | 1.606 | 70.346 |
| 1:30,0:30 | 8 | 0.440 | 13437.4 | 30954.9 | 32565.5 | 1.174 | 1.681 | 71.189 |

Najwyższe próbkowane użycie VRAM procesu na dowolnej karcie wyniosło
**36,37890625 GiB**, maksymalna próbka sumy **71,220703125 GiB**. Nie sumujemy
niejednoczesnych maksimów kart. Maksymalne próbkowane PSS **1,680728912 GiB**
dotyczy pracy API, a nie pamięci szczytowej ładowania. Każdy punkt ma tylko
39 odpowiedzi, w tym pięć żądań obrazowych. Kwantyle ogona są bardzo słabym
oszacowaniem produkcyjnego ogona. Nie porównujemy tych czasów bezpośrednio z
QPS krótkich zadań H100: to inny korpus i inna metoda.

## Jakość Ada względem FP32 i upstream BF16

Końcowy eksport R6 obejmuje 49 rekordów bench i 49 System One, 65 odpowiedzi
na pytania oraz 161 głowic pierwszej rundy. Tokeny i pokrycie są zgodne.
Zachowane FP32 i BF16 pochodzą z tego samego przypiętego checkpointu i upstream;
zweryfikowano je względem niezmiennego wcześniejszego archiwum attention.
FP32 używało głównych wag FP32 na CPU z warstwami strumieniowanymi na H100,
TF32 wyłączone; nie był to forward wykonywany wyłącznie na CPU.
Ponowne porównanie eksportów wykonano lokalnym CLI bez GPU.

TV to odległość rozkładów prawdopodobieństw; mniej oznacza bliżej referencji.
Średnia dotyczy skalibrowanych głowic pierwszej rundy, nie czasu ani accuracy.

| Miara względem FP32 | basal-rs Ada split | upstream BF16 |
|---|---:|---:|
| średnia TV | 0,0008707790684740179 | 0,0019138364215758967 |
| zgodne argmax głowic | 160/161 | 160/161 |
| zgodne odpowiedzi kategoryczne | 64/65 | 64/65 |
| zmiany automatyzacji / błędów | 0 / 0 | 0 / 0 |
| ścisły status kategoryczny | **FAIL** | **FAIL** |

Średnia TV native jest **2,19784× mniejsza** niż BF16. Jedyna różna decyzja:
`tasks-pl`, `verify`, twierdzenie `1`. FP32 wybiera `not_stated`, native i BF16
wybierają `contradicted`:

| Wariant | P(not_stated) | P(contradicted) |
|---|---:|---:|
| FP32 | 0,5121334791 | 0,4869082570 |
| native Ada | 0,4971296787 | 0,5019030571 |
| upstream BF16 | 0,4803265929 | 0,5187345147 |

Zbiorcze `all_supported: false` i `not_supported: ["1"]` pozostają takie same;
nie ma certyfikowanej automatycznej decyzji w tej pozycji. To nie usuwa różnicy
etykiety. Brakuje kontroli FP32 z jednokartowym native na Ada, więc nie
przypisujemy różnicy samemu podziałowi. Na H100 kontrola starej i nowej
binarki oraz podziałów jest bitowo zgodna. Mały bench ma 49/49 zgodności ze
złotą etykietą, ale nie uprawnia do deklarowania 100% jakości modelu.
**Cel identycznych decyzji FP32 na Ada wymaga dalszej pracy.**

## Kontrola wspólnego transferu Llama i kontrole kodu

Model `Remek/basal-1.5-4.5B` rewizja
`784a683bfadcc8865238fc0fc74a83b4000269bc`. Pięć eksportów: stara binarka single,
stary podział, R6 single, R6 podział i odwrócony podział. Każdy zawiera osiem
rekordów bench i osiem System One. Pełne pliki eksportów są bajtowo identyczne
we wszystkich pięciu wariantach. FP16, accurate, tree, bez prefix/state cache,
ta sama niezmienna tabela H100 4.5B. To 80 rekordów w pięciu eksportach,
ograniczona kontrola współdzielonego transferu, a nie cała rodzina modeli.

Sprawdzenia zakończone PASS: `cargo fmt --all --check`, release Clippy
workspace/all-targets domyślnie i CUDA/all-features, Rustdoc z prywatnymi
pozycjami i `-D warnings` domyślnie i CUDA, `cargo deny --locked` z all-features
oraz kontrola wyjątku paste. Istniejące ostrzeżenia o zdublowanych wersjach
zależności pozostają w logu. Pięć ręcznych syntetycznych przypadków nagłówków
checkpointu przeszło kontrolę CPU: cztery niepoprawne wejścia zwracają błąd,
kontrolne poprawne wejście jest akceptowane. Nie dodano testów, frameworka,
nowych zależności, unsafe ani bramek CI. Te kontrole nie zastępują pomiaru GPU.

## Zachowane niepowodzenia i granice wniosków

- Pierwsza próba wynajęcia H100 miała niezgodny klucz SSH. Zasób usunięto przed inferencją.
- Na Ada nie pasowała glibc hosta; dodano prywatny userspace loader zgodny z binarką.
  Nie podmieniano bibliotek systemowych. Hashe wersji i bibliotek zachowano.
  Stara binarka wydania nie rozpoznawała aliasu Ultra i nie pobrała shardów;
  użyto nowej binarki i jawnie przypiętego checkpointu. Logi prób są zachowane.
- **Ada R1 FAIL:** 117 odpowiedzi, w tym jedno 529 przy piątym żądaniu mediów.
  Timer liczył od startu fazy zamiast żądania. Pole `elapsed_ms` jest nieważne;
  oryginalne czasy startu/końca pozostają. Dokładna wersja błędnego runnera jest
  w archiwum. R2 naprawia timer, ogranicza media do czterech i ponawia wszystkie punkty.
- Pierwszy eksport jakości Ada podał `--gemm-table auto`, które `export`
  zinterpretował jako nazwę pliku. Kolejny eksport pomija opcję; nie używamy
  nieudanego eksportu jako wyniku jakości.
- Kompilacja z zachowanym starym mtime źródeł użyła nieaktualnego cache Cargo.
  Poprawiono mtime i zbudowano R6 ponownie; nie ma pomiaru GPU z błędnej kompilacji.
- Repliki: **FAIL niezmienności bitowej C128**; Ada: **FAIL decyzji FP32**.
  Błędy, nieukończone przygotowania i różnice są zachowane, a nie wykluczone po pomiarze.
- Brak pomiaru trzech lub większej liczby GPU, audio, pełnego kontekstu,
  wszystkich proporcji podziału oraz wszystkich modeli Omni. PLE/shared KV
  pozostają odrzucone. Nie deklarujemy nowego wyniku mini/max ani Omni 1.1 max.
- Wynik replik dotyczy bazowej binarki; R6 ma ograniczoną kontrolę C1 i Ada C1/C2/C8,
  bez powtórzenia pełnego ABBA H100 C128 po refaktorze. Nie przenosimy wyniku
  skalowania na każdą konfigurację nowego silnika.

## Dane, odtworzenie i zamknięcie sesji

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/ultra-multigpu-2026-10-18/2411d4f8eb1761fdb0a7caf54b57831f8ce566b75651c6ea465afd4a76d5e982/measurement-data.tar.gz`.

SHA-256: `2411d4f8eb1761fdb0a7caf54b57831f8ce566b75651c6ea465afd4a76d5e982`. Rozmiar **8 472 492 B**,
**950 plików oraz SHA256SUMS**. Pobranie z autoryzacją i anonimowo
potwierdziło SHA archiwum i **950/950 sum plików**.

| Ścieżka pod `reports/ultra-multigpu-2026-10-18/` w archiwum | Zawartość |
|---|---|
| `h100/replicas-r1/` | ABBA, wszystkie odpowiedzi/czasy/zasoby, preflight, metody i routing |
| `h100/gemma-split-h100-r1/` | pięć kontroli umieszczenia tekstu i obrazów |
| `h100/llama-transfer-r1/` | pięć eksportów Llama i audyt bajtowy |
| `ada/gemma-split-ada-r2/` | końcowe C1/C2/C8 i per-GPU zasoby |
| `ada/gemma-split-ada-r1/` | zachowane 529 i błędny timer |
| `quality/`, `quality-inputs/`, `analyses/quality/` | referencje FP32/BF16, eksport R6 i pełne różnice |
| `analyses/` | wyniki replik i podziałów; audyt nagłówków |
| `sources/`, `inputs/`, `metadata/`, `verification/` | dokładne źródła, dane wejściowe, provenance, buildy, replay i zamknięcie |
| `REPRODUCE.md`, `public-source-map.json`, `media-placeholders.json` | polecenia replay i odtwarzanie mediów z hashami |

Obrazy, checkpointy i biblioteki nie są redystrybuowane. Pozostają publiczne
źródła/hash obrazów, wyniki i dokładne źródła programu. Obserwacje i liczby
zachowują dane; identyfikatory procesów zanonimizowano, prywatne ścieżki i dane
maszyn zredagowano. Publiczny replay odtwarza prywatne analizy i wszystkie FAIL.

Pobranie do nowych katalogów, z katalogu głównego repozytorium:

```bash
BASAL_ULTRA_PREFIX=measurements/ultra-multigpu-2026-10-18/2411d4f8eb1761fdb0a7caf54b57831f8ce566b75651c6ea465afd4a76d5e982
BASAL_ULTRA_DL=.cache/measurement-download/ultra-multigpu-2026-10-18
BASAL_ULTRA_ROOT=.cache/measurement-artifacts/ultra-multigpu-2026-10-18
mkdir -p .cache/measurement-download .cache/measurement-artifacts
mkdir "$BASAL_ULTRA_DL" &&
  ~/.local/bin/hf buckets cp "hf://buckets/itsoltech/basal-rs/$BASAL_ULTRA_PREFIX/measurement-data.tar.gz" "$BASAL_ULTRA_DL/measurement-data.tar.gz" &&
  echo "2411d4f8eb1761fdb0a7caf54b57831f8ce566b75651c6ea465afd4a76d5e982  $BASAL_ULTRA_DL/measurement-data.tar.gz" | sha256sum -c - &&
  mkdir "$BASAL_ULTRA_ROOT" &&
  tar -xzf "$BASAL_ULTRA_DL/measurement-data.tar.gz" -C "$BASAL_ULTRA_ROOT" &&
  (cd "$BASAL_ULTRA_ROOT" && sha256sum -c SHA256SUMS)
BASAL_ULTRA_DATA="$BASAL_ULTRA_ROOT/reports/ultra-multigpu-2026-10-18"
python "$BASAL_ULTRA_DATA/sources/tools/perf/summarize-omni-replicas.py" \
  --measurement-artifacts "$BASAL_ULTRA_DATA/h100/replicas-r1" \
  --inputs "$BASAL_ULTRA_DATA/inputs" --out .cache/ultra-replicas-replay.json
python "$BASAL_ULTRA_DATA/sources/tools/perf/summarize-omni-layer-split.py" \
  --measurement-artifacts "$BASAL_ULTRA_DATA/ada/gemma-split-ada-r2" \
  --out .cache/ultra-ada-replay.json
```

Na macOS użyj `shasum -a 256 -c`. `--measurement-artifacts` jest opcją runnerów,
nie CLI basal. Pozostałe polecenia, budowa CPU CLI do porównania jakości i
odtworzenie prawdziwych wejść GPU są w archiwalnym `REPRODUCE.md`. Replay
samych analiz nie wymaga GPU ani obrazów. Nowy pomiar GPU wymaga odtworzenia
obrazów z przypiętych źródeł i weryfikacji oryginalnych sum wejść.

Przed usunięciem zasobów zachowano lokalnie **291 plików wyjściowych** z obu
maszyn GPU i potwierdzono ich rozmiary/SHA-256. Wszystkie wynajęte maszyny,
także nieudana pierwsza próba, są usunięte z potwierdzeniem nieobecności u
dostawcy. Usunięto własne kontenery i katalog roboczy budowy CPU; stany
uruchomionych usług produkcyjnych pozostały niezmienione. Lokalne kopie
prywatnych danych i opublikowanego archiwum są zachowane.
