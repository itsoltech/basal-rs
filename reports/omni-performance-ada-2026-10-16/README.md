# Wydajność basal-omni: RTX 6000 Ada, 16–17 października 2026

**Potwierdzona przewaga na zamrożonym korpusie i jednej RTX 6000 Ada.**
Basal-rs ma niższą medianę w 36/36 punktach tekstowych i 68/68 przypadkach mediów.
Mediany QPS są wyższe dla obu modeli we wszystkich poziomach czterech głównych
porównań. Wszystkie 398 decyzji kontrolnych zgadzają się z referencją FP32;
średnia odległość rozkładów jest mniejsza niż w upstream BF16.

FIFO bez osobnego toru długich dodatkowo poprawia miks: 1,72–1,80× QPS upstream
i niższe globalne p95. Pozostaje niewielka regresja p95 długich przy N=32:
około 3,4% mini i 2,8% max. Wyniki nie oznaczają przewagi każdego żądania,
pełnego kontekstu ani wszystkich urządzeń; szczegółowe ograniczenia są poniżej.

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
| Wybrany wariant | r13; commit `c7b7f7c232e5c144861520646fa35d15f07484a5`; źródła i tabele powiązane sumami z metadanymi kompilacji |
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
upstream dla danej powierzchni. Tekst max używa dodatkowo
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: domyślny allocator zakończył
32768 tokenów CUDA OOM; to ustawienie pozwoliło wykonać dwa kolejne żądania
32k. W kontrolnym punkcie 16k odpowiedzi obu ustawień były bitowo równe.
Pełna seria max używa tego ustawienia przez zarchiwizowany wrapper interpretera;
kod upstream pozostaje niezmieniony. Dotychczasowy OOM zostaje w archiwum.

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
odpowiedzi jako poprawnej. Upstream może zmieniać bity prawdopodobieństw przy
batchowaniu; sam licznik takich zmian nie oznacza zmiany wyboru ani błędu HTTP.

- Tekst: dokładna liczba tokenów pełnego promptu 128, 256, 512, 1024, 2048, 4096,
  8192, 16384 i 32768, osobno PL/EN; cztery mierzone próbki punktu na serię po
  rozgrzewce. To nie jest pomiar pełnych limitów kontekstu 131072/262144.
- Krótkie dokumenty: 128 różnych wejść, 16 wariantów semantycznych PL/EN,
  512 żądań na serię i poziom N=1/8/32. Starsze ekrany z dwoma dokumentami
  pozostają wyłącznie sygnałem pomocniczym. N=32 to ostatni zmierzony poziom
  tekstu, a nie potwierdzony punkt nasycenia przepustowości serwera.
- Ruch mieszany: jeden dokument 8192 tokenów na 16 krótkich, naprzemiennie te same dwa prompty PL/EN;
  272 żądania na
  serię przy N=8/32. Osobno podajemy p95 globalne, krótkich i długich żądań.
- Media: 34 żądania / 38 pytań, cztery zdjęcia COCO, osiem wypowiedzi FLEURS,
  warianty kodeków i obie kolejności obrazu z dźwiękiem. Na serię 102 żądania
  sekwencyjne i 272 przy N=4. Wielopytaniowe wejścia liczymy jako jedno żądanie.

Natywny budżet partii wynosi 8192 tokeny liczone po współdzieleniu stanu;
upstream używa 16384 dopełnionych tokenów krótkiej partii i wykonuje prompty
powyżej 2048 osobno w kawałkach. Są to różne reguły, a nie identyczny budżet.
Pojedyncze żądanie ponad natywny budżet jest dopuszczane samo.

Basal-rs używa `--long-tokens 4096 --long-slice-ms 10`, dużych limitów przyjęć,
aby nie mierzyć ograniczenia 529 zamiast przepustowości. Domyślny slice procesu
nadal wynosi 100 ms; podane wyniki odpowiadają jawnemu ustawieniu 10 ms. Pełne
argumenty i hashe wejść są w `method.json`. Nie zmieniamy checkoutu upstream.

## Pojedyncze żądania tekstowe

Czas HTTP request → pełna odpowiedź w ms; model jest załadowany i rozgrzany. Mniejsza wartość jest lepsza. Każda komórka jest medianą median trzech zachowanych serii (po cztery próbki danego promptu w serii).

### Omni mini

| Tokeny pełnego promptu | basal-rs PL | upstream PL | basal-rs EN | upstream EN |
|---:|---:|---:|---:|---:|
| 128 | 22.2 | 38.6 | 25.1 | 39.1 |
| 256 | 30.7 | 40.5 | 33.7 | 41.0 |
| 512 | 51.6 | 59.3 | 55.0 | 60.1 |
| 1024 | 100.2 | 116.2 | 105.5 | 119.0 |
| 2048 | 212.4 | 289.1 | 215.7 | 291.8 |
| 4096 | 456.7 | 686.4 | 463.0 | 698.6 |
| 8192 | 1355.1 | 1829.0 | 1295.2 | 1858.8 |
| 16384 | 2808.6 | 5496.3 | 2825.2 | 5548.4 |
| 32768 | 6620.5 | 18736.1 | 6621.0 | 18771.6 |

Mediana basal-rs niższa w **18/18** punktach PL/EN. Zakresy między seriami są w audytach JSON; nie są przedziałami ufności. Długości ponad 32768 nie były mierzone.

### Omni max

| Tokeny pełnego promptu | basal-rs PL | upstream PL | basal-rs EN | upstream EN |
|---:|---:|---:|---:|---:|
| 128 | 44.8 | 58.4 | 46.6 | 58.3 |
| 256 | 63.2 | 73.2 | 66.4 | 74.1 |
| 512 | 119.3 | 136.4 | 125.5 | 139.0 |
| 1024 | 235.2 | 304.3 | 245.4 | 310.6 |
| 2048 | 502.0 | 742.3 | 519.0 | 754.3 |
| 4096 | 1108.2 | 1734.5 | 1140.5 | 1758.9 |
| 8192 | 3031.4 | 4529.4 | 3064.2 | 4578.6 |
| 16384 | 6514.5 | 13410.6 | 6545.3 | 13433.0 |
| 32768 | 14856.7 | 44103.1 | 14871.3 | 44125.4 |

Mediana basal-rs niższa w **18/18** punktach PL/EN. Zakresy między seriami są w audytach JSON; nie są przedziałami ufności. Długości ponad 32768 nie były mierzone.

## Przepustowość krótkich dokumentów

128 różnych dokumentów, bez cache żądań ani prefiksów. QPS oznacza poprawnie zakończone żądania/s; większa wartość jest lepsza. Nawiasy pokazują zakres między trzema zachowanymi seriami.

| Model | Klienci | basal-rs QPS [min–max] | upstream QPS [min–max] | Przewaga QPS | Mediana ms basal-rs / upstream |
|---|---:|---:|---:|---:|---:|
| mini | 1 | 45.80 [44.60–46.03] | 25.58 [24.97–25.83] | 1.79× | 21.7 / 38.5 |
| mini | 8 | 79.27 [69.31–79.50] | 69.28 [68.25–69.34] | 1.14× | 99.3 / 116.1 |
| mini | 32 | 87.50 [87.05–87.73] | 81.88 [79.06–83.85] | 1.07× | 359.7 / 390.9 |
| max | 1 | 22.69 [22.69–22.71] | 17.65 [17.41–17.77] | 1.29× | 43.7 / 56.5 |
| max | 8 | 33.64 [33.59–33.99] | 31.19 [30.45–31.43] | 1.08× | 236.3 / 252.2 |
| max | 32 | 38.28 [37.77–38.80] | 32.95 [32.34–33.35] | 1.16× | 831.6 / 965.9 |

## Ruch mieszany i czasy oczekiwania

Jedno żądanie 8192 tokenów na 16 krótkich; 272 żądania/serię, w tym 16 długich. Podajemy również p95 osobnych klas, żeby większa przepustowość nie ukryła regresji długich żądań. P95 długiej klasy przy 16 próbkach jest jej maksimum w danej serii; nie traktujemy go jako oszacowania produkcyjnego SLO.

| Model | Klienci | QPS basal-rs / upstream | p95 globalnie ms | p95 krótkie ms | p95 długie ms |
|---|---:|---:|---:|---:|---:|
| mini | 8 | 9.70 / 8.29 | 7582 / 1919 | 178 / 1919 | 17267 / 1920 |
| mini | 32 | 11.72 / 8.45 | 10979 / 4012 | 456 / 4012 | 22755 / 4013 |
| max | 8 | 4.67 / 3.35 | 15465 / 4792 | 255 / 4792 | 33667 / 4799 |
| max | 32 | 4.95 / 3.36 | 27679 / 10098 | 898 / 10098 | 54080 / 10100 |

W każdej parze pierwsza liczba należy do basal-rs, druga do upstream. Obciążenie jest zamkniętą pętlą klientów: szybsze odpowiedzi zmieniają moment przyjścia kolejnych żądań, więc to porównanie całej obsługi ruchu, a nie izolowanej kolejki przy identycznym strumieniu czasowym.

## FIFO i długie żądania w ruchu mieszanym

Porównanie tego samego ruchu: 272 żądania/serię, 16 długich z dwóch naprzemiennych promptów PL/EN 8192 tokenów i 256 krótkich ze 128 różnych dokumentów. Każdy wiersz ma własne trzy zachowane serie basal-rs i sparowane serie upstream. W każdej parze pierwsza liczba należy do basal-rs.

- HRRN: osobny tor długich, próg 4096 i slice 10 ms, jak w pełnej drabinie powyżej.
- FIFO: `--schedule fifo --long-tokens 0`, bez osobnego toru. Ta dodatkowa seria dotyczy ruchu mieszanego; jej drabina zawiera tylko 128 tokenów, a krótkie fazy po 128 żądań. Nie zastępuje pełnej drabiny i 512-żądaniowych pomiarów HRRN.

| Model | Kolejka | Klienci | QPS basal-rs / upstream | Iloraz QPS | p95 globalnie ms | p95 krótkie ms | p95 długie ms |
|---|---|---:|---:|---:|---:|---:|---:|
| mini | HRRN | 8 | 9.70 / 8.29 | 1.17× | 7582 / 1919 | 178 / 1919 | 17267 / 1920 |
| mini | FIFO | 8 | 14.47 / 8.42 | 1.72× | 1074 / 1892 | 1120 / 1892 | 1052 / 1934 |
| mini | HRRN | 32 | 11.72 / 8.45 | 1.39× | 10979 / 4012 | 456 / 4012 | 22755 / 4013 |
| mini | FIFO | 32 | 14.90 / 8.46 | 1.76× | 3202 / 4030 | 3199 / 4030 | 4169 / 4032 |
| max | HRRN | 8 | 4.67 / 3.35 | 1.39× | 15465 / 4792 | 255 / 4792 | 33667 / 4799 |
| max | FIFO | 8 | 5.94 / 3.36 | 1.77× | 2591 / 4778 | 2590 / 4777 | 2613 / 4794 |
| max | HRRN | 32 | 4.95 / 3.36 | 1.47× | 27679 / 10098 | 898 / 10098 | 54080 / 10100 |
| max | FIFO | 32 | 6.06 / 3.36 | 1.80× | 7937 / 10064 | 7935 / 10064 | 10349 / 10066 |

P95 długich przy 16 próbkach jest maksimum danej serii; zakresy między seriami są w audytach JSON. To zamknięta pętla klientów, więc kolejność i szybkość odpowiedzi wpływają na moment nadejścia kolejnych żądań. Powtórzenia długich kontekstów i współdzielenie pracy wewnątrz batcha ograniczają uogólnianie wyniku na różne dokumenty.

Pełne odpowiedzi basal-rs, z pominięciem wyłącznie pola usage.latency_ms, muszą być bitowo równe podstawowej konfiguracji. Ta kontrola jest osobna od jakości FP32.

### Wybór profilu

Dla zmierzonego miksu FIFO daje większą przepustowość i niższe globalne p95 niż
osobny tor długich. Krótkie żądania tracą część priorytetu względem HRRN:
p95 rośnie z 178/456 ms do 1120/3199 ms w mini i z 255/898 ms do 2590/7935 ms
w max przy N=8/32. Nadal jest niższe od sparowanego upstream. Przy N=32 p95
samych długich pozostaje wyższe o około 3,4% mini i 2,8% max; to nie jest
przewaga we wszystkich metrykach. HRRN pozostaje wariantem dla priorytetu krótkich
żądań, a FIFO dla większego QPS i bardziej równomiernego rozkładu tego miksu.

Jawne włączenie konfiguracji FIFO dla tekstu:

```bash
basal serve --model omni-mini --dtype auto --gemm-table auto --schedule fifo --long-tokens 0
basal serve --model omni-max --dtype auto --gemm-table auto --schedule fifo --long-tokens 0
```

Domyślne HRRN i próg 4096 pozostają ustawieniami procesu; powyższe argumenty
wybierają zmierzony dodatkowy profil. Pełne argumenty benchmarków są w metodach.
Kontrola obejmuje 6832 obserwacje mini i 5808 max: wszystkie pełne odpowiedzi
native, po pominięciu wyłącznie `usage.latency_ms`, są bitowo równe profilowi
głównemu. To 132 różne wejścia/model, w tym dwa długie; drabina FIFO nie obejmuje
pełnych 32768 tokenów. Nie przenosimy jej wyników na niezmierzone długości.

## Obraz, dźwięk i wejścia mieszane

Te same 34 żądania / 38 pytań. W serii: 102 żądania sekwencyjne i 272 przy czterech klientach.

| Model | Mediana sekwencyjna ms basal-rs / upstream | Sekwencyjne QPS basal-rs / upstream | QPS przy N=4 basal-rs / upstream | Niższa mediana w przypadkach |
|---|---:|---:|---:|---:|
| mini | 127.4 / 201.0 | 3.35 / 2.01 | 3.70 / 2.09 | 34/34 |
| max | 146.2 / 168.3 | 4.56 / 3.94 | 5.20 / 4.21 | 34/34 |

### Przypadki multimedialne — mini

| Przypadek | basal-rs ms | upstream ms | upstream / basal-rs |
|---|---:|---:|---:|
| real-image-000000000139-en-default | 125.9 | 199.5 | 1.58× |
| real-image-000000000139-en-high | 923.4 | 1400.7 | 1.52× |
| real-image-000000000139-pl-default | 126.6 | 200.9 | 1.59× |
| real-image-000000000139-pl-high | 934.8 | 1403.3 | 1.50× |
| real-image-000000000285-en-default | 147.5 | 219.3 | 1.49× |
| real-image-000000000285-en-high | 829.4 | 1412.1 | 1.70× |
| real-image-000000000285-pl-default | 154.3 | 218.8 | 1.42× |
| real-image-000000000285-pl-high | 832.9 | 1415.2 | 1.70× |
| real-image-000000000632-en-default | 119.4 | 201.6 | 1.69× |
| real-image-000000000632-en-high | 851.3 | 1400.2 | 1.64× |
| real-image-000000000632-pl-default | 122.2 | 202.1 | 1.65× |
| real-image-000000000632-pl-high | 859.8 | 1404.0 | 1.63× |
| real-image-000000039769-en-default | 121.0 | 201.8 | 1.67× |
| real-image-000000039769-en-high | 813.4 | 1396.6 | 1.72× |
| real-image-000000039769-pl-default | 133.8 | 201.2 | 1.50× |
| real-image-000000039769-pl-high | 810.4 | 1397.3 | 1.72× |
| real-mixed-en-audio-image | 270.1 | 483.2 | 1.79× |
| real-mixed-en-image-audio | 272.1 | 487.6 | 1.79× |
| real-mixed-pl-audio-image | 242.6 | 453.5 | 1.87× |
| real-mixed-pl-image-audio | 244.1 | 453.8 | 1.86× |
| real-speech-en-1544-wav | 127.5 | 174.9 | 1.37× |
| real-speech-en-1557-wav | 99.5 | 159.1 | 1.60× |
| real-speech-en-1590-flac | 109.5 | 154.0 | 1.41× |
| real-speech-en-1590-mp3 | 68.6 | 147.7 | 2.15× |
| real-speech-en-1590-ogg | 70.8 | 141.9 | 2.00× |
| real-speech-en-1590-wav | 123.9 | 172.9 | 1.40× |
| real-speech-en-1646-wav | 93.2 | 155.5 | 1.67× |
| real-speech-pl-1542-wav | 125.4 | 173.5 | 1.38× |
| real-speech-pl-1554-flac | 66.4 | 147.8 | 2.23× |
| real-speech-pl-1554-mp3 | 48.4 | 128.7 | 2.66× |
| real-speech-pl-1554-ogg | 49.2 | 139.7 | 2.84× |
| real-speech-pl-1554-wav | 87.2 | 152.6 | 1.75× |
| real-speech-pl-1584-wav | 96.5 | 161.9 | 1.68× |
| real-speech-pl-1599-wav | 114.0 | 166.3 | 1.46× |

### Przypadki multimedialne — max

| Przypadek | basal-rs ms | upstream ms | upstream / basal-rs |
|---|---:|---:|---:|
| real-image-000000000139-en-default | 130.8 | 148.9 | 1.14× |
| real-image-000000000139-en-high | 380.6 | 450.4 | 1.18× |
| real-image-000000000139-pl-default | 136.9 | 152.2 | 1.11× |
| real-image-000000000139-pl-high | 393.8 | 456.6 | 1.16× |
| real-image-000000000285-en-default | 149.0 | 165.8 | 1.11× |
| real-image-000000000285-en-high | 398.9 | 463.8 | 1.16× |
| real-image-000000000285-pl-default | 161.2 | 179.2 | 1.11× |
| real-image-000000000285-pl-high | 409.4 | 469.6 | 1.15× |
| real-image-000000000632-en-default | 127.1 | 145.4 | 1.14× |
| real-image-000000000632-en-high | 373.6 | 445.2 | 1.19× |
| real-image-000000000632-pl-default | 133.8 | 157.6 | 1.18× |
| real-image-000000000632-pl-high | 380.3 | 453.0 | 1.19× |
| real-image-000000039769-en-default | 125.8 | 150.7 | 1.20× |
| real-image-000000039769-en-high | 368.0 | 441.7 | 1.20× |
| real-image-000000039769-pl-default | 146.6 | 161.3 | 1.10× |
| real-image-000000039769-pl-high | 390.2 | 446.5 | 1.14× |
| real-mixed-en-audio-image | 420.2 | 475.4 | 1.13× |
| real-mixed-en-image-audio | 422.2 | 485.7 | 1.15× |
| real-mixed-pl-audio-image | 340.4 | 364.0 | 1.07× |
| real-mixed-pl-image-audio | 345.1 | 366.0 | 1.06× |
| real-speech-en-1544-wav | 153.8 | 181.7 | 1.18× |
| real-speech-en-1557-wav | 120.1 | 146.6 | 1.22× |
| real-speech-en-1590-flac | 138.2 | 164.8 | 1.19× |
| real-speech-en-1590-mp3 | 119.2 | 152.4 | 1.28× |
| real-speech-en-1590-ogg | 120.7 | 157.9 | 1.31× |
| real-speech-en-1590-wav | 145.9 | 173.8 | 1.19× |
| real-speech-en-1646-wav | 111.2 | 134.7 | 1.21× |
| real-speech-pl-1542-wav | 153.9 | 186.3 | 1.21× |
| real-speech-pl-1554-flac | 97.7 | 129.5 | 1.33× |
| real-speech-pl-1554-mp3 | 90.6 | 114.7 | 1.27× |
| real-speech-pl-1554-ogg | 91.5 | 120.7 | 1.32× |
| real-speech-pl-1554-wav | 106.5 | 134.6 | 1.26× |
| real-speech-pl-1584-wav | 118.7 | 144.8 | 1.22× |
| real-speech-pl-1599-wav | 140.9 | 166.8 | 1.18× |

## Porównanie implementacji i przyjęte zmiany

Upstream Gemma wyłącza Flash, memory-efficient i cuDNN SDPA ze względu na
stwierdzone tam błędne wyniki. Jego działająca ścieżka używa math attention,
prefillu 2048, sortowania po długości i do 16384 dopełnionych tokenów partii.
To działająca implementacja referencyjna, której ograniczeń nie obchodzimy
modyfikacją kodu. Upstream ma adaptacyjne zbieranie żądań także przy `wait-ms=0`.
Źródła przypiętej wersji: [EagerBackend i GraphBackend](https://github.com/rkinas/basal/blob/62542d31e8eb69000839c01e8f9b45da1fb80e49/basal/engine.py#L80)
oraz [worker i harmonogram](https://github.com/rkinas/basal/blob/62542d31e8eb69000839c01e8f9b45da1fb80e49/basal/server.py#L829).

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
kontrolny, a nie profil wydajnościowy tabel HTTP. Oryginalne wagi BF16 pozostają
na GPU i są dokładnie rozszerzane do FP32 podczas użycia; aktywne obliczenia
wykonują się w FP32. Czas eksportu całego korpusu
nie jest czasem pojedynczego pytania.

**GAP:** świeży upstream FP32 max dla 8192 tokenów nie mieści się w 48 GiB.
Dodatkowa próba z `expandable_segments:True` także kończy się CUDA OOM.
Zgodnie z uprzednio zapisaną regułą kompletności zachowujemy historyczny
FP32 H100 dla tego jednego długiego punktu; świeży mini FP32 i oba BF16 działają.
Nie zastępujemy nieudanych wyników ani nie deklarujemy pełnej zgodności długiego
max ze świeżym FP32 na tej Ada. Drabina czasowa do 32768 nie oznacza kontroli
FP32 każdego jej punktu: długi punkt korpusu jakości ma 8192 tokenów, a jakość
względem FP32 przy dłuższych wejściach pozostaje tutaj niepotwierdzona.

## Kontrole i ograniczenia

CPU: fmt, Clippy release wszystkich targetów oraz Rustdoc prywatnych elementów
z `-D warnings` przeszły. CUDA: Clippy release wszystkich features/targetów
oraz Rustdoc przeszły; kompilacja PTX obejmuje 80/89/90/90a, a pomiary runtime
wyłącznie Ada. CI commita `c7b7f7c` dodatkowo zakończyło sukcesem kontrole Rust Linux,
Intel Vulkan, Metal i CUDA, formatowanie, workflow oraz pakiety trzech SDK.
Bramka zależności i zbiorcza lint pozostają FAIL. Brak pomiaru runtime
Metal/Intel oraz pozostałych GPU.

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
Kontrole uruchomienia systemu i zdrowia istniejących usług nie zgłosiły błędów
w tych próbach; nie jest to osobny pomiar opóźnień usług produkcyjnych.

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

## Zachowanie danych i sprzątanie

Przed usunięciem zasobów porównano wszystkie 1409 plików surowego katalogu
wynikowego: 1 856 826 529 B, SHA każdego pliku i rozmiary zgodne między serwerem
a lokalną kopią; manifest serwera był identyczny przed kopiowaniem i po nim.
Prywatne oryginalne ślady profilera, binarki bazowa/r11/r13 i wejścia pozostają
lokalnie. Publiczne archiwum zawiera wyłącznie przeznaczone do publikacji dane.

Własny kontener i katalog kampanii usunięto po sprawdzeniu tożsamości i retencji.
Zwolniono około 65 GiB. Pozostałe kontenery zachowały stan, cztery istniejące
usługi były zdrowe przed sprzątaniem i po nim; współdzielony obraz zachowano.
GPU pozostało bezczynne, sterowanie wentylatorem automatyczne, limit mocy bez zmian.
Nie było wdrożenia produkcyjnego w ramach tej kampanii.

## Dane i odtworzenie

Publikację potwierdzono 2026-10-17: nowe pobrania z autoryzacją i anonimowo
mają ten sam SHA-256 archiwum i zgodne wszystkie 1812/1812 sum plików.
Anonimowe pobranie używało nowego środowiska HF bez tokenów i potwierdzonego
stanu niezalogowanego. Lokalna kopia archiwum pozostaje zachowana.

Przed publikacją przywrócono wszystkie 22 pliki zamrożonego korpusu, 18 payloadów
mediów i sześć kodeków z oryginalnymi SHA. Z publicznej wersji ponownie policzono
sześć audytów wydajności, oba porównania bitowe, obie tabele i audyt jakości:
wyniki liczbowe są identyczne z prywatnymi oryginałami.

| Pole | Wartość |
|---|---|
| URI | `hf://buckets/itsoltech/basal-rs/measurements/omni-performance-ada-2026-10-16/82aaf99af87532eaccfaf19e3fe5462b5e0d9b4f1ffcac6ccba7ba987ac8a0a7/measurement-data.tar.gz` |
| SHA-256 archiwum | `82aaf99af87532eaccfaf19e3fe5462b5e0d9b4f1ffcac6ccba7ba987ac8a0a7` |
| Zweryfikowane pliki | 1812/1812 w obu pobraniach |
| Rozmiar gzip | 26 718 255 B |

Pod `reports/omni-performance-ada-2026-10-16/` znajdują się:

- `runs/confirm-{text,media}-r13-{mini,max}-r1/`: metody, wszystkie odpowiedzi,
  fazy i logi ośmiu uruchomień każdego porównania; cache tabel zawiera jedynie
  zasoby GEMM, bez mediów i odpowiedzi.
- `audits/audit-confirm-*.json`: ponownie obliczone metryki, zakresy między seriami,
  wszystkie klasy ruchu i 34 indywidualne przypadki mediów na model.
- `runs/quality-gate-r11/`, `runs/structural-borrow-r13/`,
  `runs/quality-all-omni-lt-r11-auto-table/` oraz dodatkowe `quality-all-*`:
  porównania FP32 i pełne eksporty poszczególnych konfiguracji, w tym FAIL BF16
  bez tabel dla średniej TV max.
- `inputs-confirm-r1/`, `quality-all-inputs-r1/` i pozostałe zamrożone wejścia:
  tokeny, pytania, manifesty źródeł i hashy. Prawdziwe media zastąpiono markerami
  SHA-256; archiwum nie rozprowadza zdjęć ani nagrań.
- `variants/lt-candidate-r13/`: źródła i tabele finalnego wariantu względem
  podanego commita bazowego. `runners/`, `tools/perf/` i `method-sources/`:
  rzeczywiście użyte wersje runnerów oraz analizy. `environment/` zawiera
  przypięte pakiety upstream, `checks/` logi kompilatora i zależności.
- Profile zminimalizowano do nazw kerneli i czasów, bez metadanych maszyny/procesu.
  Nieudane ekrany i profile, błędy startu `fast` oraz oba OOM upstream FP32 max
  pozostają w `runs/`. Oryginalne prywatne ślady Nsight zachowano lokalnie.

`curation.json`, `media-registry.json`, `codec-replay.json` i `SHA256SUMS` są
w korzeniu archiwum. Curation zachowuje osobno hashe oryginalnych i opublikowanych
plików; numeryczny tekst nie był zmieniany. README w archiwum jest kopią sprzed
dopisania jego własnego URI; aktualny raport z potwierdzeniem publikacji jest
w Gicie. Nie umieszczamy binarek ani wag modeli w publicznym archiwum.

```bash
BASAL_ARTIFACT_SHA=82aaf99af87532eaccfaf19e3fe5462b5e0d9b4f1ffcac6ccba7ba987ac8a0a7
BASAL_ARTIFACT_DL=.cache/measurement-download/omni-performance-ada-2026-10-16
BASAL_ARTIFACT_ROOT=.cache/measurement-artifacts/omni-performance-ada-replay
mkdir -p .cache/measurement-download .cache/measurement-artifacts
mkdir "$BASAL_ARTIFACT_DL"
hf buckets cp "hf://buckets/itsoltech/basal-rs/measurements/omni-performance-ada-2026-10-16/$BASAL_ARTIFACT_SHA/measurement-data.tar.gz" "$BASAL_ARTIFACT_DL/measurement-data.tar.gz"
echo "$BASAL_ARTIFACT_SHA  $BASAL_ARTIFACT_DL/measurement-data.tar.gz" | sha256sum -c -
mkdir "$BASAL_ARTIFACT_ROOT"
tar -xzf "$BASAL_ARTIFACT_DL/measurement-data.tar.gz" -C "$BASAL_ARTIFACT_ROOT"
(cd "$BASAL_ARTIFACT_ROOT" && sha256sum -c SHA256SUMS)
```

Katalogi muszą być nowe. Na macOS zamiast `sha256sum -c` użyj `shasum -a 256 -c`.
Korzystaj z globalnego `hf`; jeśli istnieje `~/.local/bin/hf`, wywołaj tę binarkę.

Do ponownego audytu HTTP odtwórz media z czterech JPEG COCO i dwóch przypiętych
archiwów FLEURS dev, wskazanych w `inputs-confirm-r1/media-source-manifest.json`.
Umieść je w `.cache/omni-replay-assets`; przy odtwarzaniu sześciu wariantów kodeków
potrzebny jest dokładny FFmpeg zapisany w `codec-replay.json` (n9.0.1).
Można zamiast tego dostarczyć gotowe pliki o wymaganych SHA-256. Ogg wymaga
zapisanych numerów seryjnych i CRC; helper odtwarza je i wymaga oryginalnego hasha.
Dwie własne syntetyczne próbki historycznego korpusu są zapisane bezstratnie
w `codec-replay.json`; prawdziwe media pozostają zewnętrzne.

```bash
BASAL_REPORT_ROOT="$BASAL_ARTIFACT_ROOT/reports/omni-performance-ada-2026-10-16"
BASAL_RESTORED_INPUTS=.cache/omni-replay-inputs
python3 "$BASAL_REPORT_ROOT/tools/perf/restore-omni-inputs.py" \
  --measurement-artifacts "$BASAL_ARTIFACT_ROOT" \
  --input-subdir reports/omni-performance-ada-2026-10-16/inputs-confirm-r1 \
  --assets .cache/omni-replay-assets --out "$BASAL_RESTORED_INPUTS"
mkdir .cache/omni-replay-audits
for BASAL_SURFACE in text media; do
  for BASAL_MODEL in mini max; do
    python3 "$BASAL_REPORT_ROOT/tools/perf/summarize-omni.py" \
      --measurement-artifacts "$BASAL_REPORT_ROOT/runs/confirm-$BASAL_SURFACE-r13-$BASAL_MODEL-r1" \
      --inputs "$BASAL_RESTORED_INPUTS" \
      --out ".cache/omni-replay-audits/audit-confirm-$BASAL_SURFACE-r13-$BASAL_MODEL-r1.json"
  done
done
python3 "$BASAL_REPORT_ROOT/method-sources/render-final-tables-r13.py" \
  --measurement-artifacts .cache/omni-replay-audits --out .cache/omni-replayed-tables.md
```

Ta opcja `--measurement-artifacts` należy do runnerów i analiz, nie do CLI basal.
Odtworzenie wymaga oryginalnych hashy wszystkich wejść; helper niczego nie
pobiera i nie nadpisuje. Do nowego pomiaru GPU użyj zarchiwizowanej metody
z nowym katalogiem wynikowym oraz przypiętymi modelami i upstream. Nie porównuj
czasów z różnych maszyn jako bezpośredniego pomiaru przyspieszenia.

Dla ponownego pomiaru tekstu max do 32768 tokenów nadaj uprawnienie wykonania
`method-sources/reference-python-expandable-r1` i dostosuj ścieżkę interpretera
do odtworzonego środowiska. Wrapper zmienia wyłącznie allocator PyTorch.
Archiwum zapisuje pliki źródłowe z trybem 0644, więc samo rozpakowanie nie nadaje
wrapperowi uprawnienia wykonania. Zachowaj tę zmianę konfiguracyjną jawnie
w nowej metodzie; nie zastępuj nią historycznych metadanych.

Dodatkowe potwierdzenia FIFO zachowują własne serie upstream. Ich audyty i pełną
zgodność odpowiedzi między profilami odtworzysz w tym samym katalogu audytów:

```bash
for BASAL_MODEL in mini max; do
  python3 "$BASAL_REPORT_ROOT/tools/perf/summarize-omni.py" \
    --measurement-artifacts "$BASAL_REPORT_ROOT/runs/confirm-fifo-r13-$BASAL_MODEL-r1" \
    --inputs "$BASAL_RESTORED_INPUTS" \
    --out ".cache/omni-replay-audits/audit-confirm-fifo-r13-$BASAL_MODEL-r1.json"
  python3 "$BASAL_REPORT_ROOT/method-sources/profile-bit-parity-r1.py" \
    --measurement-artifacts "$BASAL_REPORT_ROOT" \
    --reference-run "runs/confirm-text-r13-$BASAL_MODEL-r1" \
    --candidate-run "runs/confirm-fifo-r13-$BASAL_MODEL-r1" \
    --out ".cache/omni-replay-audits/audit-fifo-bits-$BASAL_MODEL-r1.json"
done
python3 "$BASAL_REPORT_ROOT/method-sources/render-fifo-comparison-r1.py" \
  --measurement-artifacts .cache/omni-replay-audits --out .cache/omni-replayed-fifo.md
```

Audyt 398 decyzji oraz średniej TV wobec upstream BF16 korzysta z zapisanych
porównań numerycznych i nie wymaga ponownej inferencji:

```bash
python3 "$BASAL_REPORT_ROOT/tools/perf/audit-omni-quality.py" \
  --measurement-artifacts "$BASAL_REPORT_ROOT/runs/quality-all-omni-lt-r11-auto-table" \
  --bf16-comparison "mini=$BASAL_REPORT_ROOT/runs/quality-gate-r11/mini-upstream-bf16.json" \
  --bf16-comparison "max=$BASAL_REPORT_ROOT/runs/quality-gate-r11/max-upstream-bf16.json" \
  --out .cache/omni-replay-audits/audit-quality-r11.json
```

To odtwarza kontrolę numeryki r11; osobny `runs/structural-borrow-r13/` dokumentuje
bitową równość finalnego r13. Publiczne metadane mają etykiety zamiast prywatnych
ścieżek. Dlatego ponownie obliczony hash `method.json` może różnić się od hasha
oryginału zachowanego w historycznym audycie. `curation.json` podaje obie sumy;
wyniki numeryczne pozostają niezmienione.
