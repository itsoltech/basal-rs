# Upstream 2.1, SDK i nowe modele Omni

## Zakres i wersje

Implementacja bazuje na `d32264e`, po scaleniu obsługi wielu GPU. Referencją jest
upstream v2.1.0, commit `62542d31e8eb69000839c01e8f9b45da1fb80e49`.
Zmiany 2.1 względem 2.0 dotyczą warstwy zadań i serwera, a nie nowych kerneli
inferencji. Checkout upstream używany do porównań pozostał niezmieniony.

Dodano eksperymentalne `classify`, `verify`, `compress` i `extract` w CLI oraz
na obu endpointach HTTP. Ekstrakcja tekstu ma najwyżej dwie rundy; ten sam job
i rezerwacja repliki obejmują całość. Obowiązują limity rozwinięcia promptów,
schematu i liczby kandydatów. Szczegóły kontraktu, offsetów Unicode i kalibracji:
[TASKS.md](../../docs/TASKS.md).

SDK Python, TypeScript i .NET mają wersję 0.3.0, Rust 0.4.0. Pakiety dodają
fabryki zadań, typowane wyniki, wartości JSON ekstrakcji i nullable `automated`.
Przygotowanie i sprawdzenie pakietów nie oznacza ich publikacji do rejestrów.

| Model | Przypięta rewizja | Zakres adaptacji |
|---|---|---|
| `basal-omni-1.1-max` | `c8ab50cd973c116b3ca6bd9e4d574f713bab1f68` | Unified: tekst, obraz i audio; alias `omni-1.1-max` |
| `basal-omni-1.0-ultra` | `b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4` | Tekst i obraz, dwa shardy BF16, większy encoder obrazu; alias `omni-ultra` |

Ultra nie ma encodera audio. Konfiguracja procesora audio nie wystarcza do
udostępnienia tej modalności: poprawne wejście audio jest odrzucane kodem 422,
a metadane modelu wymieniają obraz. ModelOpt FP8 pozostaje jawnie odrzucany.
Ta kampania nie implementuje podziału warstw Gemma ani kwantyzacji FP8.

## Sprzęt i środowisko

Omni 1.1 max i mniejsze modele sprawdzono na RTX 6000 Ada 48 GiB, z limitem
300 W, tymczasowym wentylatorem 100% oraz kontenerem ograniczonym do 8 CPU
i 72 GiB RAM. Ultra sprawdzono na H100 PCIe 80 GiB, z fabrycznym limitem 350 W
i około 180 GiB RAM. Na H100 ustawiono po 8 wątków OMP/MKL/Rayon; nie był to
limit całego systemu operacyjnego. Oba silniki danego porównania korzystały
z tej samej maszyny i tych samych ustawień.

Referencja: PyTorch 2.11.0+cu128, Transformers 5.17.0, tokenizers 0.23.3;
Python 3.12.3 na Ada i 3.12.12 na H100. Natywny build korzystał z Rust 1.95.0,
CUDA 12.9.1 i cuBLASLt 120901. Kernele Candle skompilowano dla capability 89
na Ada i 80 w przenośnym buildzie H100; własne PTX obejmowały 80/89/90.
Upstream używał wspieranego przez Gemma trybu **eager BF16**. Nie oznaczamy go
jako `fast`: ten backend upstream nie obsługuje tych modeli.

## Metoda jakości

Porównujemy przypięte wejścia PL/EN, 49 pytań z etykietami i korpus API z
zadaniami, przypadkami braku wartości, liczbami i Unicode. Media pochodzą
z poprzedniej kampanii z przypiętymi źródłami i sumami kontrolnymi. Nowe
zadania sprawdzamy także na rzeczywistych mediach.

Średnia TV to średnia odległość między skalibrowanymi rozkładami na unikatowych
readoutach **pierwszej rundy**. Mniejsza wartość oznacza większą bliskość FP32.
Osobno porównujemy gotowe odpowiedzi, również wynik drugiej rundy ekstrakcji:
etykiety, werdykty, wybrane jednostki, wartości i fragmenty źródła. Nie eksportujemy
surowych readoutów kontynuacji. Zgodność z FP32 na tym skończonym korpusie nie
jest miarą semantycznej trafności na dowolnych danych.

| Model i profil basal-rs | Readouty | Średnia TV basal-rs | Średnia TV upstream BF16 | Zgodność argmax z FP32 |
|---|---:|---:|---:|---:|
| Basal 2.0 mini, FP16 ze stałą tabelą GEMM | 142 | 0,000683794 | 0,004621410 | 142/142 |
| Omni 1.0 mini, auto FP16 na Ada | 191 | 0,000772820 | 0,002345693 | 191/191 |
| Omni 1.1 max, FP16 ze stałą tabelą GEMM | 191 | 0,003017056 | 0,003495285 | 191/191 |
| Omni Ultra, FP16 ze stałą tabelą GEMM | 161 | 0,000636621 | 0,001913836 | 161/161 |

Referencje mini i Omni mini wykonano na GPU w FP32; dla 1.1 max użyto pełnej
referencji CPU FP32. Dla Ultra użyto opisanego niżej FP32 z przesyłaniem warstw
z RAM. Precyzja natywnego profilu opisuje decoder; wieże mediów pozostają BF16,
również przy diagnostycznym decoderze FP32.

W 49 pytaniach z etykietami uzyskano odpowiednio 39/49, 46/49, 48/49 i 49/49
poprawnych decyzji — tyle samo co właściwa referencja FP32. Ten mały, wcześniej
używany korpus nie jest niezależnym benchmarkiem ogólnej jakości modeli.

Dla pierwszych dwóch modeli gotowe decyzje również zgadzają się z FP32.
Dla 1.1 max wszystkie nowe zadania mają te same decyzje, ale starsze
`image-default/act` ma rzeczywistą różnicę: FP32 CPU wybiera `human`, natomiast
basal-rs FP16, basal-rs FP32 i upstream BF16 wybierają `accept`; odpowiedź
`true` pozostaje taka sama. Ten przypadek ma status **FAIL/GAP** dla pełnej
zgodności decyzji, mimo zgodności wszystkich argmax. Nie ukrywamy go przez
ograniczenie porównania do samych etykiet.

`automated` dla nowych zadań z mediami celowo wynosi `null`: opublikowany próg
tekstowy nie certyfikuje jakości multimodalnej. Upstream potrafi zastosować
próg tekstowy również do tych zadań. Te różnice polityki raportujemy osobno.

Pełny Ultra FP32 nie mieści się w 80 GB VRAM. Na CPU ukończono 49 pytań
kontrolnych i 11 przypadków API; cały przebieg przerwano ze względu na czas.
Referencja z warstwami przesyłanymi z RAM wykonuje oryginalny upstream na GPU
w FP32, z wyłączonym TF32 i bez konwersji do FP16/BF16. Jej wspólny podzbiór
z CPU obejmuje 69 readoutów, również obraz: 69/69 argmax, bez różnic gotowych
decyzji; średnia TV 0,000001025, maksymalna TV 0,000022419.
To kontrola zmiany urządzenia dla referencji, a nie kompletny eksport CPU
ani pomiar szybkości serwowania FP32. Przebiegi referencyjne są oddzielone
od czasów HTTP porównywanych z upstream BF16 na GPU.

Ultra FP16 zachowało również wszystkie gotowe decyzje zadań. Upstream BF16
oraz natywny BF16 zmieniały jeden werdykt `tasks-pl/verify`: twierdzenie o
imieniu sprzedawcy dostawało `contradicted`, a FP32 wybierało `not_stated`.
FP16 bez tabeli miało TV 0,000854652; zmierzona tabela poprawiła ją do
0,000636621. To około 3,01× bliżej FP32 niż upstream BF16 na tych readoutach.

Profil domyślny wybiera teraz FP16 dla 1.1 max na RTX 6000 Ada oraz Ultra na
H100 PCIe. Ultra ma wbudowaną tabelę cuBLASLt obejmującą osiem geometrii wag
i sześć klas M: 64, 128, 256, 512, 2048, 8192. Pełna tabela źródłowa ma SHA-256
`1ff81e025124ea5e70a9981485e10d891baa2ef9dfc93035ad7cb6c41498d63c`.
Build usuwa z niej metadane czasowe; porównanie profilu używanego przez `auto`
musi dotyczyć tych samych geometrii, klas i słów algorytmów, a nie samego SHA
zredukowanego pliku cache. Dotychczasowe aliasy i pozostałe pary model/karta
zachowują wcześniejszy wybór precyzji.

## HTTP: Omni 1.1 max na RTX 6000 Ada

W tabelach **basal-rs** oznacza decoder FP16 ze zweryfikowaną tabelą GEMM,
a **upstream** oznacza eager BF16. Czas mierzymy od rozpoczęcia serializacji
requestu do odczytania całego body odpowiedzi HTTP. Parsowanie odpowiedzi
jest poza pomiarem. Wagi są już załadowane.

Drabina zawiera jedno pytanie `choice`, jednego klienta oraz dokładnie
128/512/2048/8192 tokenów gotowego promptu. Wszystkie odpowiedzi wskazują
oczekiwaną opcję `blue`. Seria ma kolejność ABBAABBA; pierwsze powtórzenie
każdego silnika jest odrzucone. Podajemy medianę metryk trzech pozostałych
powtórzeń, po cztery zmierzone żądania na punkt i powtórzenie.

| Język | Tokeny | basal-rs [ms] | upstream [ms] | upstream / basal-rs |
|---|---:|---:|---:|---:|
| EN | 128 | 46,83 | 58,63 | 1,25× |
| EN | 512 | 129,10 | 138,95 | 1,08× |
| EN | 2048 | 611,13 | 750,36 | 1,23× |
| EN | 8192 | 2979,69 | 4564,46 | 1,53× |
| PL | 128 | 47,24 | 58,83 | 1,25× |
| PL | 512 | 130,26 | 141,04 | 1,08× |
| PL | 2048 | 620,27 | 756,99 | 1,22× |
| PL | 8192 | 3011,23 | 4587,24 | 1,52× |

Większy iloraz oznacza krótszy czas basal-rs. Korpus obciążeniowy to 32 różne
stany dokumentów, obejmujące całe zadania i zależne kontynuacje ekstrakcji.
Każda faza wykonuje 64 żądania. Sloty 1024 i limit dużych żądań 128 przewyższają
liczbę klientów, więc 529 nie ogranicza pomiaru. Cache stanu, prefix cache
i osobne pasmo długich żądań są wyłączone.

| Klienci | basal-rs [żądań/s] | upstream [żądań/s] | Przewaga przepustowości | Mediana basal-rs / upstream [ms] | p95 basal-rs / upstream [ms] |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,778 | 1,588 | 1,12× | 467 / 476 | 1023 / 1302 |
| 8 | 1,908 | 1,160 | 1,64× | 4101 / 6822 | 4519 / 8330 |
| 32 | 2,004 | 1,508 | 1,33× | 12449 / 20284 | 19843 / 22328 |

Czas pod obciążeniem obejmuje czekanie w kolejce. Porównujemy silniki w tym
samym wierszu. Jedno żądanie zadań może obejmować wiele decyzji modelu i dwie
rundy; te QPS opisują inny zakres pracy niż wcześniejsze proste `choice`/`noul`.

W osobnej pełnej serii mediów zastosowano ten sam porządek ośmiu uruchomień.
Po odrzuceniu pierwszych pozostało dziewięć obserwacji każdego przypadku
na silnik. Pod obciążeniem C4 pozostały po 72 żądania na silnik.

| Przypadek, C1 | basal-rs [ms] | upstream [ms] |
|---|---:|---:|
| Obraz i zadania | 515,6 | 553,8 |
| Audio i zadania | 554,6 | 608,5 |
| Obraz + audio i zadania | 991,8 | 1012,7 |

Przy C4 uzyskano 1,513 vs 1,403 żądania/s; mediana 2505 vs 2721 ms i p95
2916 vs 3122 ms. To około 8% większa przepustowość na tych trzech wejściach.
Nie wyciągamy wniosku o dowolnych obrazach i nagraniach. Wstępny pojedynczy
screen sygnalizował niewielką przegraną mieszanego przypadku; zachowano go
obok powtórzonego pomiaru.

Pierwotny wspólny run tekstu i mediów zatrzymał guard po limicie 2400 s,
tuż przed ostatnią fazą obciążenia mediów. Wszystkie osiem serii tekstowych
było już ukończone i zostało osobno zweryfikowane przez `--surfaces text`.
Status nieukończonego parenta pozostaje w audycie. Media powtórzono jako nowy,
kompletny run; poprzedniego nie przepisano na sukces.

## HTTP: Ultra na H100 PCIe

Końcowy build `c11ab46` używał `auto`: faktycznego FP16 i wbudowanej tabeli
GEMM opisanej powyżej. Suma przenośnej binarki:
`534235a0f66c890987a7cb5b711df55f6a03e752e0e4e7dc505ebf7ce7239d41`.
Przebieg ABBAABBA ukończył wszystkie osiem uruchomień. Po odrzuceniu pierwszego
uruchomienia każdego silnika pozostały trzy powtórzenia. Drabina ma po dwa
mierzone żądania i jedno rozgrzewkowe na punkt; korpus zadań ma 32 żądania
na fazę, poprzedzone ośmioma żądaniami rozgrzewkowymi. Dodatkowa rozgrzewka
czasowa była wyłączona. To krótszy protokół niż seria Ada, ograniczony czasem
sesji; nie porównujemy bezpośrednio przepustowości obu kart.

| Język | Tokeny | basal-rs [ms] | upstream BF16 [ms] | upstream / basal-rs |
|---|---:|---:|---:|---:|
| EN | 128 | 77,31 | 84,91 | 1,10× |
| EN | 512 | 196,36 | 166,14 | 0,85× |
| EN | 2048 | 1055,07 | 961,78 | 0,91× |
| EN | 8192 | 5518,31 | 6920,06 | 1,25× |
| PL | 128 | 77,92 | 84,89 | 1,09× |
| PL | 512 | 197,93 | 167,65 | 0,85× |
| PL | 2048 | 1059,94 | 970,22 | 0,92× |
| PL | 8192 | 5549,51 | 6954,79 | 1,25× |

Każda zmierzona odpowiedź drabiny miała oczekiwane `blue`. Basal-rs skraca czas
dla 128 i 8192 tokenów, lecz jest około 18% wolniejszy przy 512 i 9–10%
przy 2048. Klasy M tabeli wybierają algorytm GEMM; nie powodują paddingu
wykonywanego mnożenia do rozmiaru klasy.

| Klienci, zadania 2.1 | basal-rs [żądań/s] | upstream [żądań/s] | Mediana basal-rs / upstream [ms] | p95 basal-rs / upstream [ms] |
|---:|---:|---:|---:|---:|
| 1 | 1,183 | 1,583 | 688 / 490 | 1615 / 1284 |
| 8 | 1,252 | **FAIL: CUDA OOM** | 6305 / brak poprawnej fazy | 6822 / brak poprawnej fazy |
| 32 | 1,284 | **GAP: faza pominięta po OOM** | 10614 / brak pomiaru | 24912 / brak pomiaru |

W sekwencyjnych zadaniach basal-rs ma około 25% niższą przepustowość i 40%
wyższą medianę opóźnienia. Cel przewagi dla wszystkich długości i zadań
**nie został osiągnięty** dla Ultra.

Przy C8 każde z czterech uruchomień upstream zwróciło 31 błędów HTTP 422
z CUDA OOM i jedną odpowiedź 200. Następująca faza C32 nie została wykonana.
Basal-rs wykonał po 32/32 poprawne żądania w każdej fazie C1/C8/C32 wszystkich
czterech uruchomień. Odrzucenie pierwszych powtórzeń nie usuwa niepowodzenia:
pełny audyt ma status **FAIL**. Nie przeliczamy szybkich błędów na przepustowość
ani nie podajemy ilorazu QPS dla niepoprawnej fazy upstream.

To wynik niezmodyfikowanego backendu eager z limitem batcha 16384 tokenów.
Nie wykonano dodatkowego strojenia tego limitu ani ponownego porównania po
zmianie ustawień upstream. OOM dotyczył procesu modelu; nie zaobserwowano
restartu maszyny ani błędów Xid. W poprawnych odpowiedziach tekstowych nie
było zmian decyzji ani automatyzacji względem C1 tego samego silnika:
448 obserwacji basal-rs i 196 upstream. Pełne odpowiedzi basal-rs zachowały
bitową zgodność pod obciążeniem. Ta kontrola nie zastępuje porównania z FP32.

Osobny pomiar obrazów ma status **SIGNAL**, po jednym uruchomieniu silnika
na jednym obrazie. C1: 913 ms basal-rs vs 1314 ms upstream, około 1,44× szybciej.
Krótki burst C4 obejmował tylko cztery żądania na silnik: 1,096 vs 0,760
żądania/s, ale mediana basal-rs była gorsza: 3646 vs 3291 ms. p95 wyniosło
3647 vs 5262 ms. Większa przepustowość w tym burstcie nie oznacza krótszej
mediany. Potrzebna jest dłuższa powtórzona seria obrazów.

Następny krok optymalizacji to profilowanie Ultra dla 512–2048 tokenów
oraz poszczególnych zadań i drugiej rundy ekstrakcji. Należy oddzielić czas
przygotowania zadania, transferów i forwardu, sprawdzić algorytmy GEMM dla
rzeczywistych rozmiarów i porównać batchowanie z upstream. Każdy wariant
zmieniający numerykę wymaga ponownej kontroli FP32. Są to kierunki dalszych
pomiarów, nie potwierdzone przyczyny obecnej różnicy.

## Ładowanie wag

CUDA może teraz konwertować BF16 do FP16 po przesłaniu wagi na GPU. Gdy brakuje
marginesu VRAM lub odczyt telemetrii się nie uda, pozostaje wcześniejsza
konwersja na CPU. FP32 zachowuje poprzedni sposób ładowania.

Pełne linie eksportów były bitowo zgodne przed i po zmianie: 49 + 61 rekordów
dla Omni mini i 1.1 max oraz 49 + 49 dla Ultra, osobno bez tabeli i ze zmierzoną
tabelą. Końcowy build Ultra wybierający `auto` potwierdził faktyczne FP16.
Na H100 zaobserwowano skrócenie `load_s` z 145,22 do 76,89 s bez tabeli
i z 160,55 do 76,90 s z tabelą. Na Ada czasy 20,23→19,75 s oraz 30,55→31,37 s
nie wykazują jednoznacznej poprawy. Cache plików nie był kontrolowany,
dlatego są to obserwacje startu, oddzielne od wyników HTTP.

## Poprawki znalezione podczas pomiarów

- Zadania na `/v1/systemone` potrzebują prawdopodobieństwa wybranej opcji.
  Znormalizowana pewność pierwotnego `choice` ma inne znaczenie. Warstwa zadań
  korzysta teraz z tego samego prawdopodobieństwa na obu endpointach.
- Ultra wymaga zamkniętego pustego bloku `thought`. Wybór renderera na podstawie
  samej architektury pomijał ten blok. Renderer wybieramy przez zweryfikowany
  hash szablonu; po poprawce wszystkie 142 porównania tekstowych promptów/tokenów
  oraz komplet readoutów eksportu GPU mają zgodne tokeny.
- Pierwsza wersja drabiny HTTP dodawała nazwy opcji i przekraczała zadane długości.
  Poprawiony generator używa `option_keys: hide` i sprawdza finalny request przez
  parser upstream. Natywne przygotowanie potwierdziło 8/8 długości dla obu nowych
  modeli. Wcześniejsze serie zachowano jako nieważne dla wniosków wydajnościowych.

## SDK, zgodność i lifecycle

Zainstalowane, spakowane SDK wykonano przeciw serwerowi GPU, a nie wyłącznie
przeciw fixture'om. Dla Basal 2.0 mini uzyskano 48 obserwacji, dla Omni 1.0 mini
69, dla Omni 1.1 max 69, a dla Ultra 55: łącznie 241. Końcowe serie 1.1 max i Ultra sprawdzają
również domyślny profil `auto` FP16 na buildzie `c11ab46`.
W każdej serii uczestniczyły Python, TypeScript, .NET i Rust. Kontrole obejmowały
nowe zadania, starsze typy pytań i modalności obsługiwane przez dany model.
Konsument .NET skompilowano dla net8.0 i uruchomiono na .NET 10 z dopuszczonym
przejściem na nowszy główny runtime.

Ręczny, hostowy runner uzyskał 88/88 zgodnych obserwacji ekspansji i składania
zadań na dostarczonych prawdopodobieństwach; osobna seria formatu 1 z limitem
10 opcji również uzyskała 88/88. Nie są to pomiary jakości modelu.

Kontrola lifecycle wykazała bitową zgodność odpowiedzi zadań między endpointami,
4 przyjęte i 12 celowo odrzuconych żądań przy limicie 4 oraz zwolnienie rezerwacji
po odłączeniu 8 klientów. Kolejne żądania zakończyły się poprawnie. Ta seria
admission z celowymi 529 jest oddzielna od pomiaru przepustowości.

Bitowe A/B istniejących obliczeń potwierdzono dla Basal 2.0 mini ze stałą tabelą
GEMM oraz starszych Omni mini i max. Pierwsza próba Llama bez stałej tabeli
wykazała różnice przez wybór innych algorytmów autotunera; poprawiona seria
z zamrożonym profilem była bitowo zgodna. Pierwsze eksporty Ultra miały błędny
prompt i nie stanowią dowodu zgodności z upstream.

## Zakres weryfikacji

Formatowanie, Clippy, Rustdoc z ostrzeżeniami jako błędami, kontrola wersji SDK
i `cargo deny` przeszły. CUDA sprawdzono kompilacyjnie dla całego workspace,
wszystkich targetów i features oraz uruchomiono na rzeczywistym GPU. Nie dodano
frameworku testowego ani bramki testów do CI.

Nie wykonano w tej kampanii runtime'u Metal ani nowych zadań jednocześnie na
dwóch GPU. Basal 2.0 4.5B i max nie były ponownie mierzone; niniejszy raport
nie zastępuje ich wcześniejszych wyników. Średnia TV, zgodność decyzji i czasy
HTTP są odrębnymi obserwacjami, a nie zamiennymi dowodami przewagi.

## Zakończenie zasobów

Przed usunięciem środowisk zachowano lokalne kopie wyników i sprawdzono
SHA-256 wszystkich 460 plików środowiska Ada oraz 215 plików środowiska H100.
Wynajęty H100 został usunięty; lista instancji dostawcy potwierdziła jego brak.
Na własnym serwerze usunięto wyłącznie kontener i katalog tej kampanii,
przywrócono automatyczny wentylator, a chronione usługi produkcyjne pozostały
zdrowe przed i po sprzątaniu. Prywatne identyfikatory zasobów i logi dostawcy
nie są częścią publicznego archiwum.

## Archiwum i odtworzenie

Publikację potwierdzono 2026-10-17 po pobraniu z autoryzacją oraz anonimowo,
z nowym pustym katalogiem poświadczeń. Oba pobrania potwierdziły SHA-256
archiwum i **896/896** sum plików. Archiwum ma 7 352 849 B.

- URI: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.1-2026-10-17/9a8dae4bc02b08a4f2bcaf50aec712710706fb746820aff89961d4924ac1b10d/measurement-data.tar.gz`
- SHA-256: `9a8dae4bc02b08a4f2bcaf50aec712710706fb746820aff89961d4924ac1b10d`
- Magazyn: [itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs).

Wszystkie dane są pod `reports/upstream-2.1-2026-10-17/` wewnątrz archiwum:

| Ścieżka względem katalogu kampanii | Zawartość |
|---|---|
| `ada/exports/`, `h100/exports/` | Eksporty referencji FP32/BF16 i natywne, w tym nieukończone i wcześniejsze błędne warianty |
| `ada/runtime/`, `h100/runtime/` | Obserwacje HTTP, odpowiedzi, logi błędów, telemetria guardów, profile i lifecycle |
| `ada/analyses/`, `h100/analyses/` | Porównania jakości, niezależne audyty HTTP i kontroli decyzji |
| `ada/perf-inputs/`, `h100/perf-inputs/` | Zamrożone poprawione wejścia pomiarów; media jako znaczniki hashy |
| `ada/historical-perf-inputs/`, `h100/historical-perf-inputs/` | Zachowane wejścia wcześniejszych prób, w tym niepoprawnej drabiny |
| `ada/sdk/`, `h100/sdk/`, `*/sdk-inputs/` | Wyniki konsumentów wszystkich SDK oraz wejścia i sumy mediów |
| `host/` | Porównania mechaniki zadań, również zachowane nieudane próby |
| `method-sources/`, `tools/` | Dokładne wersje wykonanych exporterów, runnery i audytory; patch kodu względem `d32264e` |
| `verification/` | Kontrole źródeł SDK, retencji, sprzątania, odtworzenia analiz i patcha |
| `REPRODUCE.md`, `public-source-map.json`, `media-placeholders.json`, `inputs/` | Instrukcje, oryginalne/publiczne sumy i przypięte źródła mediów |

Przykład pobrania z katalogu głównego repozytorium; katalogi docelowe muszą być nowe:

```sh
BASAL_DL=.cache/measurement-download/upstream-2.1-final
BASAL_ROOT=.cache/measurement-artifacts/upstream-2.1-final
BASAL_SHA=9a8dae4bc02b08a4f2bcaf50aec712710706fb746820aff89961d4924ac1b10d
mkdir -p .cache/measurement-download .cache/measurement-artifacts &&
  mkdir "$BASAL_DL" &&
  hf buckets cp "hf://buckets/itsoltech/basal-rs/measurements/upstream-2.1-2026-10-17/$BASAL_SHA/measurement-data.tar.gz" "$BASAL_DL/measurement-data.tar.gz" &&
  echo "$BASAL_SHA  $BASAL_DL/measurement-data.tar.gz" | sha256sum -c - &&
  mkdir "$BASAL_ROOT" &&
  tar -xzf "$BASAL_DL/measurement-data.tar.gz" -C "$BASAL_ROOT" &&
  (cd "$BASAL_ROOT" && sha256sum -c SHA256SUMS)
```

Na macOS użyj `shasum -a 256 -c`. Dalsze polecenia analizy są w archiwalnym
`REPRODUCE.md`. Porównanie jakości nie wymaga wykonania modelu. Audyt HTTP
wymaga odtworzenia oryginalnych bajtów wejść przez `restore-inputs.py`, z mediami
pobranymi osobno i zweryfikowanymi SHA-256. Powtórny forward wymaga modeli,
środowiska GPU i osobno nadzorowanej sesji.

Odtworzenie z publicznych danych dało dokładnie te same cztery audyty HTTP,
włącznie z **FAIL** upstream Ultra i **SIGNAL** obrazów. Ponowna analiza Ultra
zachowała metryki jakości i różnice decyzji. Patch odtworzył dokładne drzewo
Git `8adebaa` z bazy `d32264e`; jego kod silnika odpowiada końcowemu `c11ab46`.
Późniejsze commity dotyczą narzędzi pomiarowych i dokumentacji.

Zdjęcia, nagrania, wagi, binarki oraz prywatne logi zasobów pozostały poza
archiwum. Publiczne obserwacje mają zanonimizowane ścieżki i endpointy;
mapa sum dokumentuje transformacje. Archiwalny snapshot tego raportu powstał
przed dopisaniem niniejszej noty o publikacji. Lokalna kopia archiwum została
zachowana, a zakończony prefiks nie będzie nadpisywany.
