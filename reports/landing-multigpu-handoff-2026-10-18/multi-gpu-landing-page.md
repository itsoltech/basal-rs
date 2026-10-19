# Multi-GPU w basal-rs — materiały dla agenta landing page

Stan opisu: **18 października 2026**. Implementacja opisana według `main`, commit `0da3769c95a0e20e53b3717eb132954d458a89ed`. Liczby pochodzą z wcześniejszych pomiarów z 11–14 października 2026, na wskazanych niżej modelach i binarkach. Ten dokument zestawia istniejące dowody; nie opisuje nowego benchmarku.

## Zadanie dla agenta

Dodaj sekcję o trzech trybach multi-GPU: przypisaniu modeli do kart, replikach oraz podziale warstw. Pokaż korzyść każdego trybu i osobno wynik skalowania replik. Zachowaj przy metryce nazwę modelu, sprzęt, obciążenie i podstawę porównania. Gotowe teksty poniżej można skracać, zachowując ich zakres.

**Najważniejsze rozróżnienie:** obsługa konfiguracji jest zaimplementowana dla szerszej rodziny modeli niż rodzina objęta benchmarkami. Wynik **2,07×** dotyczy **Basal 1.5 mini**, a porównanie to **basal-rs na jednym GPU → basal-rs na dwóch GPU**.

## Gotowe copy

### Nagłówek sekcji

**Jeden serwer. Wiele GPU.**

Rozdzielaj modele między karty, zwiększaj przepustowość przez repliki i rozkładaj obsługiwane modele warstwami, gdy potrzebujesz więcej pamięci. Wszystko za jednym API.

### Trzy karty funkcjonalności

**Różne modele, równoległa praca**  
Przypisz modele do wybranych GPU. Każda karta obsługuje swoje obciążenie, a aplikacja korzysta ze wspólnego API.

**Więcej żądań dzięki replikom**  
Uruchom kopię tego samego modelu na każdej karcie. Serwer rozdziela żądania między dostępne repliki według ich obciążenia, zachowując jedną nazwę modelu w API.

**Większa pojemność dzięki podziałowi warstw**  
Rozłóż warstwy obsługiwanego modelu między kilka GPU, aby zmniejszyć zapotrzebowanie na pamięć na pojedynczej karcie. Dostępne dla modeli Basal 1.5 i Basal 2.0 na CUDA.

### Karta z wynikiem

**2,07× przepustowości na dwóch GPU**

83,4 → 172,7 odpowiedzi/s przy 32 równoległych klientach.

**Obowiązkowy podpis przy wyniku:** Basal 1.5 mini, F16, bez kwantyzacji, 1 vs 2 × NVIDIA RTX 6000 Ada 48 GB. Porównanie jednej i dwóch replik basal-rs w tym samym benchmarku. Mediany czterech przebiegów na wariant; wynik zależy od modelu, sprzętu i ruchu.

W rozwinięciu lub tooltipie można dodać: **p99 czasu odpowiedzi 444,8 → 252,1 ms** w tym samym profilu C32. P99 to czas, w którym zakończyło się 99% żądań danego przebiegu. Opublikowana liczba jest medianą tych percentyli z czterech przebiegów.

### Krótki opis techniczny pod sekcją

Multi-GPU działa w jednym procesie serwera z backendem NVIDIA CUDA. Przypisanie kart i repliki obejmują rodziny Basal 1.5, Basal 2.0 i Omni. Podział warstw obejmuje modele Basal 1.5 i Basal 2.0; wymaga F16 i kerneli `fused`. Grupa replik oraz podział warstw wymagają kart tego samego typu.

## Jak działają poszczególne tryby

### 1. Przypisanie modeli do GPU — `device`

- Każdy model jest ładowany na wybraną kartę. Przykład: mini na GPU 0, max na GPU 1.
- Modele na różnych kartach wykonują obliczenia równocześnie, we wspólnym serwerze.
- Żądanie wskazuje model; użytkownik SDK nie wybiera karty dla każdego wywołania.
- Pamięć danego GPU musi pomieścić przypisane do niego modele i ich pamięć roboczą.
- Ten tryb służy rozdzieleniu obciążenia różnych modeli. Samo przypisanie modelu do innej karty nie rozkłada jego obliczeń między GPU.
- Pomiary izolacji obejmowały Basal 1.5 mini i 4.5B na dwóch RTX 6000 Ada. Nie stanowią gwarancji całkowitej izolacji zasobów hosta.

### 2. Repliki jednego modelu — `replica_devices`

- Każda karta otrzymuje pełną kopię wag tego samego modelu oraz własne kolejki, cache i harmonogram przetwarzania.
- Model nadal występuje pod jedną nazwą w API i discovery.
- Router wybiera dostępną replikę z najmniejszą liczbą zarezerwowanych żądań, obejmującą kolejkę i żądania w toku. Przy remisie rotuje wybór. Równoległe wybory dają przybliżone równoważenie obciążenia.
- Niezależne GPU mogą jednocześnie wykonywać różne żądania. To tryb zwiększania przepustowości pod obciążeniem.
- Pełna kopia modelu musi zmieścić się na każdej karcie. Repliki zwiększają łączną pamięć zajętą przez wagi.
- Serwer udostępnia metryki replik i ich dostępność w `/health`. Może omijać repliki oznaczone jako niedostępne; po przyjęciu żądania do kolejki repliki nie przenosi go automatycznie na inną.
- Repliki nie zapewniają izolacji awarii sprzętu. Niektóre błędy CUDA mogą wymagać restartu całego procesu. Awarii karty nie wywoływano w benchmarku.

### 3. Podział modelu warstwami — `layer_split`

- Jeden model jest dzielony na kolejne zakresy warstw. Każdy zakres trafia na inną kartę.
- Wagi i cache K/V warstwy pozostają na karcie jej etapu. Aktywacje są kopiowane na granicach etapów.
- Etapy pojedynczego żądania wykonują się kolejno. Ten mechanizm nie implementuje równoległości tensorowej ani nakładania etapów wielu żądań jako pipeline parallelism.
- Celem jest rozłożenie zapotrzebowania na pamięć. Transfery między etapami dodają koszt; w pomiarach pojedyncze żądanie było wolniejsze.
- Obecnie nie można połączyć `layer_split` i replik tego samego modelu.
- Tryb działa dla architektury Llama: Basal 1.5 i Basal 2.0. Omni, w tym Ultra, odrzuca podział warstw.

## Zakres implementacji i dowodów

| Rodzina | Model na wybranym GPU | Repliki modelu | Podział warstw |
|---|---|---|---|
| Basal 1.5 mini / 4.5B / max | Zaimplementowane | Zaimplementowane | Zaimplementowane: CUDA, F16, `fused` |
| Basal 2.0 mini / 4.5B / max | Zaimplementowane | Zaimplementowane | Zaimplementowane: CUDA, F16, `fused` |
| Omni 1.0 mini / max, Omni 1.1 max, Ultra | Zaimplementowane | Zaimplementowane | Nieobsługiwane |

Macierz opisuje możliwości kodu. Historyczne pomiary multi-GPU obejmują **Basal 1.5 mini i 4.5B**. Skalowanie przepustowości replik zmierzono dla **mini**. Nowe kombinacje multi-GPU Basal 2.0 i Omni po integracji nie mają jeszcze pomiaru zgodności i wydajności na sprzęcie. Pomiary Ultra na jednej H100 nie wypełniają tej luki.

Wszystkie opisane tryby wielu kart wymagają CUDA. Metal i Intel obsługują w tym zakresie wyłącznie urządzenie 0. Grupy replik i podział warstw wymagają zgodnego typu kart: nazwy GPU, compute capability i liczby SM. Opis dotyczy jednego serwera, bez skalowania między maszynami.

Źródło: [macierz architektury](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/docs/ARCHITECTURE.md#ograniczenia), [raport integracji](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/multigpu-main-integration-2026-10-17/README.md).

## Liczby dopuszczone do wykorzystania

### Repliki: jeden GPU względem dwóch GPU

Model **Basal 1.5 mini**, F16, bez kwantyzacji; NVIDIA RTX 6000 Ada Generation 48 GB, limit mocy 300 W na kartę, CUDA/cuBLASLt 12.9.1, automatyczne zegary GPU. Jedna binarka, wspólna tabela GEMM niezależna od wielkości partii, metryki włączone. Ruch z 32 poprawnych wejść System One. C16 i C32 oznaczają odpowiednio 16 i 32 klientów w pętli zamkniętej: klient wysyła następne żądanie po otrzymaniu odpowiedzi.

Dwie sekwencje A/B/B/A, świeży serwer i rozgrzewka w każdym przypadku. A: jedna instancja na GPU 0; B: dwie repliki na GPU 0 i 1. Tabela przedstawia mediany statystyk czterech przebiegów na wariant. Jednostką przepustowości jest kompletna odpowiedź HTTP.

| Ruch | Jedna replika: odpowiedzi/s | Dwie repliki: odpowiedzi/s | Skalowanie | Jedna replika: p99 | Dwie repliki: p99 |
|---|---:|---:|---:|---:|---:|
| C16 | 83,018 | 170,938 | 2,06× | 246,822 ms | 140,742 ms |
| C32 | 83,367 | 172,728 | 2,07× | 444,795 ms | 252,106 ms |

Źródło: [benchmark replik — metoda, tabela i ograniczenia](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/multigpu-replicas-2026-10-11/README.md).

Nie wyznaczono skalowania dla więcej niż dwóch kart. Wyniki dotyczą tych wejść i tych wersji kodu. Kontrole zgodności replik porównywały odpowiedzi z referencją pojedynczego serwera; nie są nowym pomiarem jakości względem FP32 upstream.

### Podział warstw: dowód rozłożenia pamięci

Basal 1.5 4.5B, bez kwantyzacji, dwie RTX 6000 Ada 48 GB. Pula alokatora CUDA po załadowaniu wynosiła **9197,4 MiB** na jednej karcie, a przy podziale 30/30 warstw **4610,7 MiB / 4618,7 MiB** na dwóch kartach. To pula alokatora, obejmująca część pamięci procesu; nie pełny VRAM procesu ani zużycie w szczycie obciążenia.

W osobnej kontroli sztucznie ograniczono wolną pamięć do **7939 MiB na każdej karcie**. Jeden GPU kończył ładowanie błędem OOM, a podział działał i dawał eksporty bitowo równe referencji jednego GPU. Była to symulacja ograniczonej wolnej pamięci na kartach 48 GB. Nie potwierdza konkretnej konfiguracji fizycznie mniejszych kart ani maksymalnego rozmiaru obsługiwanego modelu.

W F16 bez kwantyzacji przepustowość HTTP przy jednym kliencie z podziałem wyniosła **96,3–97,3%** wyniku jednego GPU. Nie używaj podziału jako obietnicy krótszej inferencji.

Źródło: [zestawienie — pamięć i szybkość](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/multigpu-results-2026-10-14/README.md#podział-warstw), [kontrola pojemności i zgodności L1](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/multigpu-layer-l1-revalidation-2026-10-13/README.md).

## Zasady publikacji i granice twierdzeń

- **2,07× opisuje skalowanie basal-rs z jednej do dwóch kart.** Nie podpisuj tego wyniku jako przewagi nad upstream.
- Nie przypisuj tej liczby Basal 2.0, Omni ani Ultra. Nie łącz jej z metrykami CPU/RAM/VRAM Ultra z H100 w jeden profil pomiarowy.
- Nie obiecuj liniowego skalowania na dowolnej liczbie GPU, krótszej pojedynczej inferencji dzięki replikom ani braku narzutu.
- W historycznym pomiarze narzutu na jednym GPU odnotowano spadek mediany przepustowości o 3,04%. Osobna kontrola przy żądanym zegarze 1200 MHz dała +0,030% i mieszane kierunki różnic. Przyczyna pierwszego wyniku pozostała nieustalona. [Pomiar kontrolny](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/multigpu-replicas-control-2026-10-11/README.md).
- Podział warstw służy pojemności. Nie reklamuj go jako pełnego sumowania VRAM bez narzutu, tensor parallelism ani obsługi podziału Omni/Ultra.
- Nie deklaruj odporności na awarię GPU, automatycznego odtworzenia utraconego żądania ani niezmierzonej dostępności usługi.
- Kwantyzacja `q8_0` / `q4_k` jest osobną funkcją. Nie jest warunkiem multi-GPU ani dowodem zachowania jakości FP32. Domyślny i zalecany format pozostaje bez kwantyzacji (`none`).
- Historyczne raporty zachowują FAIL/GAP/SIGNAL i rozstrzygnięcia regresji. Ten skrót nie zmienia ich statusów. Szczegóły, w tym sygnały toru jednego GPU przy zmianach podziału warstw, pozostają w [pełnym zestawieniu](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/multigpu-results-2026-10-14/README.md#sygnały-regresji-i-ich-rozstrzygnięcia).

## Źródła i odtworzenie danych

Odsyłacze w dokumencie są przypięte do commitu, aby działały również po przekazaniu samego pliku Markdown agentowi bez dostępu do bieżącego checkoutu.

- [Konfiguracja i zachowanie multi-GPU](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/README.md).
- [Architektura: routing, dostępność replik i podział warstw](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/docs/ARCHITECTURE.md).
- [Zestawienie historycznych pomiarów multi-GPU](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/reports/multigpu-results-2026-10-14/README.md).
- [Procedura pobrania i sprawdzenia archiwum](https://github.com/itsoltech/basal-rs/blob/0da3769c95a0e20e53b3717eb132954d458a89ed/docs/MEASUREMENT_ARTIFACTS.md#archiwum-multigpu-2026-10-14).

Istniejące publiczne archiwum danych:

```text
URI: hf://buckets/itsoltech/basal-rs/measurements/multigpu-2026-10-14/71acff72913c376a84af653ccb9a5f7df86d9bfbaa1c6f05700d61d8bc2328b2/measurement-data.tar.gz
SHA-256: 71acff72913c376a84af653ccb9a5f7df86d9bfbaa1c6f05700d61d8bc2328b2
```

Zawiera m.in. `reports/multigpu-replicas-2026-10-11/measurements.json`, `reports/multigpu-replicas-control-2026-10-11/measurements.json` oraz `reports/multigpu-layer-l1-revalidation-2026-10-13/{facts,provenance,validation}.json`. Według zapisanej procedury archiwum opublikowano 14 października 2026; anonimowe pobranie potwierdziło SHA-256 archiwum i wszystkie 75 sum plików. To stan weryfikacji z archiwizacji, bez nowego pobrania w ramach przygotowania tego dokumentu.

Aby odtworzyć dane, pobierz wskazane archiwum przez `hf buckets cp` do nowego katalogu, sprawdź SHA-256 archiwum, rozpakuj je, a następnie wykonaj `sha256sum -c SHA256SUMS` w katalogu rozpakowania. Szczegółowe polecenia i reguły są w podlinkowanej procedurze.
