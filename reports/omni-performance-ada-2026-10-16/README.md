# Wydajność basal-omni: RTX 6000 Ada, 16–17 października 2026

**Status: końcowe powtórzenia trwają.** Wyniki wydajności i dane archiwum
zostaną dołączone dopiero po audycie; opis jakości dotyczy zakończonych kontroli.

## Cel i konfiguracja

Porównujemy `basal-omni-1.0-mini` i `basal-omni-1.0-max`: pełny czas HTTP request
→ response przy jednym kliencie oraz poprawnie zakończone żądania/s pod obciążeniem.
Wagi modeli pozostają niezmienione. Rozdzielamy tekst, obraz, dźwięk i wejścia
mieszane; zgodność z FP32 jest osobnym warunkiem przyjęcia optymalizacji.

W każdej tabeli **basal-rs** oznacza Auto FP16 dekodera, iloczyny attention FP16
z akumulacją i softmaxem FP32 oraz wieże mediów BF16. **Upstream** oznacza
**eager BF16**, najszybszy działający oficjalny tryb spośród sprawdzonych.
Upstream `fast` nie uruchamia obu modeli tekstowych: `Gemma4Model` nie ma `layers`
w `merge_projections`. Media jawnie wymagają `eager`. Zachowujemy te niepowodzenia;
nie przypisujemy niedziałającemu `fast` wyników działającego `eager`.
Oficjalne `fast --merge off` również sprawdzono: mini i max kończą start
błędem braku `num_key_value_heads` w konfiguracji Gemmy przy alokacji KV.

| Składnik | Wersja / identyfikator |
|---|---|
| Upstream | v2.1.0, `62542d31e8eb69000839c01e8f9b45da1fb80e49` |
| Punkt wyjścia basal-rs | `1a590d95b0cb7b6a075e128d2e1a5e7234433691` |
| Wybrany wariant | r13; źródła i tabele powiązane sumami z metadanymi kompilacji |
| SHA-256 mierzonej binarki | `eaf493a9dcaa8f2f98ebd0a2c79c2afc54a56332b57b189dbc514d453404861d` |
| Mini: rewizja checkpointu | `73c993dc3d9ab3cb8887ac0b1ee1cf1c92783240` |
| Mini: SHA-256 checkpointu | `dd79e50cad8858c1593efab8b301a538059d68e3b70dd8768ade2f790ae6ac9b` |
| Max: rewizja checkpointu | `e538228c4f3ffd96dde298a073ed10247da0d637` |
| Max: SHA-256 checkpointu | `69a2600a5eb11e3c96bb4913d7e1e8abfc1c7ec0f7f0443d447954c54dd5b20d` |
| Rust / CUDA toolkit / cuBLASLt | 1.95.0 / 12.9.1 / 120901 |
| PyTorch / Transformers / tokenizers upstream | 2.11.0+cu128 / 5.17.0 / 0.23.3 |
| tokenizers basal-rs / sterownik NVIDIA | 0.23.2 / 580.159.03 |

NVIDIA RTX 6000 Ada 48 GiB, Intel Xeon Gold 5412U. Oba silniki działają kolejno
na tej samej maszynie, z limitem 8 CPU i 64 GiB RAM. Pilotaż porównał 1 i 8 wątków
upstream: końcowy tekst mini używa 1, pozostałe powierzchnie upstream 8; basal-rs
używa 8. Dobór i oba limity są zapisane w metadanych. Nie jest to porównanie
przy identycznej liczbie wątków, lecz z szybszym z dwóch sprawdzonych ustawień
upstream dla danej powierzchni.

Limit mocy pozostaje 300 W. Końcowe serie używają wentylatora 100%; kontroler
przerywa przy 80°C, błędach GPU, zmianie uruchomienia systemu, braku zasobów
lub pogorszeniu zdrowia istniejących usług. Po każdej sesji zatrzymuje własny
kontener i przywraca automatyczne chłodzenie. Wcześniejsze serie z innym
chłodzeniem zachowujemy jako ekrany, bez zaliczania do końcowych powtórzeń.

## Metoda i wejścia

Model jest załadowany i rozgrzany. Mierzony odcinek zaczyna się przed serializacją
żądania i kończy po odczycie całego body HTTP; parsowanie i walidacja odpowiedzi
następują później. QPS dzieli liczbę poprawnych odpowiedzi przez czas całej fazy
klienta, obejmujący również pracę klienta. Pobieranie, start procesu i ładowanie
wag nie należą do podanych czasów. Cache żądań, stanu i prefiksów są wyłączone,
a `orders=1`. Ruch idzie przez interfejs loopback, bez zewnętrznej sieci.

Kolejność procesów to ABBAABBA: cztery serie każdego silnika. Pierwszą serię
każdego wykluczamy zgodnie z metodą przed pomiarem; zostają trzy. Wynik jest
medianą metryki trzech zachowanych serii; min–max opisuje ich rozrzut, a nie
przedział ufności. Zachowujemy wszystkie próbki, również wykluczone rozgrzewkowe,
błędy i odpowiedzi. Audyt ponownie oblicza metryki, sprawdza kompletność, tokeny,
klucze pytań, błędy i bitową zgodność odpowiedzi basal-rs między seriami,
procesami i poziomami współbieżności. Status HTTP sam nie wystarcza do zaliczenia
odpowiedzi jako poprawnej.

- Tekst: dokładna liczba tokenów pełnego promptu 128, 256, 512, 1024, 2048, 4096,
  8192, 16384 i 32768, osobno PL/EN; cztery mierzone próbki punktu na serię po
  rozgrzewce. To nie jest pomiar pełnych limitów kontekstu 131072/262144.
- Krótkie dokumenty: 128 różnych wejść, 16 wariantów semantycznych PL/EN,
  512 żądań na serię i poziom N=1/8/32. Starsze ekrany z dwoma dokumentami
  pozostają wyłącznie sygnałem pomocniczym.
- Ruch mieszany: jeden dokument 8192 tokenów na 16 krótkich; 272 żądania na
  serię przy N=8/32. Osobno podajemy p95 globalne, krótkich i długich żądań.
- Media: 34 żądania / 38 pytań, cztery zdjęcia COCO, osiem wypowiedzi FLEURS,
  warianty kodeków i obie kolejności obrazu z dźwiękiem. Na serię 102 żądania
  sekwencyjne i 272 przy N=4. Wielopytaniowe wejścia liczymy jako jedno żądanie.

Basal-rs używa `--long-tokens 4096 --long-slice-ms 10`, dużych limitów przyjęć,
aby nie mierzyć ograniczenia 529 zamiast przepustowości. Domyślny slice procesu
nadal wynosi 100 ms; podane wyniki odpowiadają jawnemu ustawieniu 10 ms. Pełne
argumenty i hashe wejść są w `method.json`. Nie zmieniamy checkoutu upstream.

## Porównanie implementacji i przyjęte zmiany

Upstream Gemma wyłącza Flash, memory-efficient i cuDNN SDPA ze względu na
stwierdzone tam błędne wyniki. Jego działająca ścieżka używa math attention,
prefillu 2048, sortowania po długości i do 16384 dopełnionych tokenów partii.
To działająca implementacja referencyjna, której ograniczeń nie obchodzimy
modyfikacją kodu. Upstream ma adaptacyjne zbieranie żądań także przy `wait-ms=0`.

Basal-rs ogranicza koszt wykonania pojedynczego żądania przez połączony RMSNorm,
maskowanie pozycyjne i softmax FP32 na GPU, ponowne użycie obrotów pozycyjnych
oraz iloczyny FP16 z akumulacją FP32. Długie prompty zachowują unikalne K/V GQA,
a kafel scores ogranicza pojedynczy taki bufor do 128 MiB; nie jest to limit
całej pamięci roboczej. Długie zadania przechowują współrzędne odczytu i
rekonstruują tylko wykonywany prompt, zamiast powielać cały stan dla każdego
pytania.

Projekcje krótkich tekstów współdzielą GEMM tylko dla kształtów objętych tabelą
cuBLASLt zweryfikowaną pod względem niezmienności bitów. Opaque algorytmy mogą
różnić się między klasami M, lecz należą do sprawdzonej grupy zgodnych redukcji.
Mini ma 253 wpisy, max 184; brak pasującej tabeli oznacza osobne GEMM promptów.
Attention pozostaje osobne dla każdego promptu. Zwykłe zmienne batchowanie,
które zmieniało odpowiedź zależnie od sąsiednich żądań, zostało odrzucone.

W max po równoległej partii serwer zbiera powracających klientów przez najwyżej
10 ms; po bezczynności lub przy gotowej kolejce rusza od razu. Mini nie dodaje
tego oczekiwania. Zbieranie działa na dedykowanym wątku GPU, poza executorem
Tokio. Kolejkowanie jest częścią przewagi pod obciążeniem, ale zmiany kerneli,
projekcji i przygotowania mediów służą także pojedynczej inferencji.

W mediach zachowujemy BF16 wież przy dekoderze FP16. CPU współdzieli amplitudy
FFT między filtrami mel, pomija wyłącznie zerowe wagi przy zachowaniu kolejności
sumowania i starej ścieżki dla niefinitywnych wartości. Skalowanie obrazu używa
ograniczonego cache identycznych zaokrąglonych wierszy, z dawną ścieżką po
przekroczeniu 32 MiB. Vision współdzieli obroty i zachowuje kolejność poolingu
3×3; Conformer łączy uwagę i pięć przyczynowych tapów depthwise. Wyniki CPU
cech, pozycji, kształtów i flag obu modeli były bitowo równe w 76/76 kontrolach.
Lokalne przyspieszenie procesora audio nie jest samo w sobie wynikiem HTTP.

Automatyczny dobór FP16 i attention dotyczy dokładnie obu modeli Omni na
rozpoznanej RTX 6000 Ada. Jawny BF16/F32 i `accurate` zachowują iloczyny FP32;
`accurate` nie zmienia całego dekodera na FP32. Nie deklarujemy tych samych
wyników na innych kartach. Biblioteczna domyślna precyzja Gemmy pozostaje BF16;
wybór Auto opisuje politykę CLI.

## Jakość względem upstream FP32

Korpus ma 199 głowic/model, 398 łącznie: 49 benchmarku, 63 głowice API,
38 z mediami, 48 nowych pytań oraz jeden długi odczyt. Nowe pytania zamroziliśmy
przed wynikami; pytania o media używają tych samych zasobów, więc nie są nowym
zbiorem zdjęć/nagrań. Historyczne referencje FP32 bazowego korpusu pochodzą
z poprzedniej zweryfikowanej kampanii, nowe krótkie referencje z tej samej Ada.
`engine.py`, `prompt.py`, `media.py` i `bench.py` są identyczne w użytym upstream
2.0 i 2.1; nowe typy zadań 2.1 pozostają poza zakresem tej kampanii.

W wybranym profilu wszystkie **398/398 decyzji** zgadzają się z FP32.
Upstream BF16 również ma tutaj 398/398; nie wykazaliśmy wzrostu trafności
kategorycznej ponad BF16. Porównujemy dodatkowo odległość rozkładów
prawdopodobieństwa TV, gdzie mniejsza wartość oznacza bliżej FP32.

| Model | Średnia TV basal-rs / upstream BF16 | Bliżej FP32 średnio | p95 TV basal-rs / upstream | Maksimum TV basal-rs / upstream | Głowice bliżej / dalej |
|---|---:|---:|---:|---:|---:|
| mini | 0,00060193 / 0,00212067 | 3,52× | 0,00308486 / 0,01150923 | 0,02402226 / 0,04369982 | 167 / 32 |
| max | 0,00065548 / 0,00180935 | 2,76× | 0,00252721 / 0,01058809 | 0,02273206 / 0,03526604 | 149 / 50 |

Zachowujemy wszystkie pojedyncze regresje. Lepsza średnia nie oznacza przewagi
każdej głowicy ani poprawności semantycznej. Długi przykład mini zachowuje
błędny wybór RED referencji FP32 mimo BLUE w stanie.

Na zamrożonych 48 nowych pytaniach/model ze świeżym FP32 na tej samej Ada
średnia TV wyniosła 0,00078606 vs 0,00105546 dla mini (1,34× bliżej) oraz
0,00041563 vs 0,00092250 dla max (2,22×). To skończony zbiór kontrolny.

Wybrany wariant r11 przeszedł wszystkie tryby single/budget/tree: 326 unikalnych
pełnych rekordów × 3, czyli 978 porównań rekordów. Finalny r13 zmienia tylko
retencję identyfikatorów długich promptów i dwa zbędne zapożyczenia; wszystkie
326 rekordów r13 są bitowo równe r11, z bezpośrednimi porównaniami FP32.
Nie przedstawiamy tego jako świeżego pełnego przebiegu trzech trybów r13.
Osobno końcowy runner HTTP sprawdza niezmienność odpowiedzi między obciążeniami
oraz uruchomieniami serwera.

Pełny natywny FP32 r11 również przeszedł wszystkie 398 głowic i trzy tryby
z niezmiennością bitów; średnie TV mini 0,00003461 i max 0,00018935. To profil
kontrolny, a nie profil wydajnościowy tabel HTTP. Czas eksportu całego korpusu
nie jest czasem pojedynczego pytania.

**GAP:** świeży upstream FP32 max dla 8192 tokenów nie mieści się w 48 GiB.
Dodatkowa próba z `expandable_segments:True` także kończy się CUDA OOM.
Zgodnie z uprzednio zapisaną regułą kompletności zachowujemy historyczny
FP32 H100 dla tego jednego długiego punktu; świeży mini FP32 i oba BF16 działają.
Nie zastępujemy nieudanych wyników ani nie deklarujemy pełnej zgodności długiego
max ze świeżym FP32 na tej Ada.

## Kontrole i ograniczenia

CPU: fmt, Clippy release wszystkich targetów oraz Rustdoc prywatnych elementów
z `-D warnings` przeszły. CUDA: Clippy release wszystkich features/targetów
oraz Rustdoc przeszły; kompilacja PTX obejmuje 80/89/90/90a, a pomiary runtime
wyłącznie Ada. Brak pomiaru runtime Metal/Intel oraz pozostałych GPU.

Rzeczywista kontrola lifecycle r11 przeszła: bitowo zgodne odpowiedzi trzech
wejść Basal 2 mini względem poprzedniej binarki, równoległe odpowiedzi max,
32 rozłączenia klientów z poprawnym odzyskaniem obsługi, błędne media z poprawnym
kolejnym żądaniem oraz SIGINT z dokończeniem ośmiu przyjętych odpowiedzi.
Są to kontrole zachowania, nie pomiary przepustowości. Nie dodano frameworków
ani bramek testowych.

**FAIL zależności:** `cargo deny --locked --all-features` blokują istniejące
już w punkcie wyjścia `zip 8.6.0` przez Candle 0.11.0: RUSTSEC-2026-0367,
RUSTSEC-2026-0368 i RUSTSEC-2026-0369. Wymagane poprawki zip są w >=9.0.0;
sprawdzona aktualna wersja Candle nadal wymaga 8.6. Nie zmieniamy zależności,
nie dodajemy wyjątku i nie opisujemy CI jako w pełni zielonego. Kontrole licencji,
źródeł, zakazów oraz istniejącego wyjątku paste przeszły.

Zachowujemy odrzucone warianty zmiennego batchowania i wszystkie FAIL/GAP/SIGNAL.
Próbę r9 Auto z niezaładowanymi tabelami odrzucono jako FAIL_CONFIG; przyczyną
był cache build-scriptu skopiowanego checkoutu, usunięty w kolejnych budowach
przez czyszczenie artefaktów samego CLI i kontrolę wygenerowanych tabel.
Wcześniejszy ekran przerwany na granicy termicznej nie jest końcowym powtórzeniem.
Zachowujemy także nieudane próby profilera z pustymi/błędnymi wejściami.
Nie stwierdzono restartu serwera ani pogorszenia produkcyjnych usług w tych próbach.

Dodatkowa kontrola finalnej binarki bez tabel GEMM potwierdziła 398/398 wyborów
FP32 i bitową zgodność single/budget/tree także dla jawnego FP16 oraz BF16.
FP16 bez tabel ma średnią TV 0,00063166 (mini) i 0,00061414 (max), mniejszą niż
upstream BF16. Jawny BF16 bez tabel ma 0,00199013 i 0,00215612: **FAIL przewagi
średniej TV dla max** względem upstream 0,00180935, mimo tych samych wyborów.
Dlatego przewaga jakości opisana wyżej dotyczy wybranego profilu Auto FP16,
a nie dowolnego ustawienia precyzji. Logi i osobne audyty diagnostyczne zachowują
ten wynik zamiast utożsamiać zgodność decyzji z przewagą całych rozkładów.

BF16 z osobną tabelą diagnostyczną również zachowuje wszystkie decyzje i bity
między trybami. Jego średnia TV wynosi 0,00150184 (mini) i 0,00179758 (max),
więc przechodzi kontrolę lepszej średniej od upstream BF16, choć dla max różnica
jest mała. Te tabele BF16 nie są dołączonymi tabelami produkcyjnego Auto FP16.
