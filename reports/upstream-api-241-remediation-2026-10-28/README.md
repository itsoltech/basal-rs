# API 2.4.1: zamknięcie FAIL Pico i uzupełnienie lokalnych GAP

Oba wskazane problemy Pico są zamknięte w zbadanym zakresie. `serve --batches`
startuje po poprawce walidacji BOS. F32 mieści się w progu maksymalnej różnicy
skalibrowanego prawdopodobieństwa **≤0,0001**, kiedy referencja Qwen3.5 używa
funkcji PyTorch zamiast automatycznie wybranych kerneli FLA/Triton z TF32.
Nie zmieniono natywnej arytmetyki, wag, kalibracji ani progu.

Nowy pomiar potwierdza też walidację offline Mini oraz ograniczony zestaw
rzeczywistych wykonań w tle, anulowanie i odzyskanie pracy po restarcie.
Nie jest to kwalifikacja produkcyjnego profilu GPU ani pełnego zakresu modeli.
[Poprzedni raport](../upstream-api-241-local-2026-10-28/README.md) i jego FAIL
pozostają niezmienione jako zapis wcześniejszych pomiarów.

## Wersje i metoda

Kod: baza `73cfc9710280b74e7d8480e8ca137ee776e10dee`; poprawka BOS
`bc5ad0aa` oraz eksporter `1d34cfa` i uproszczenie `b4cdb078251369fa40f6c123996297f0ef23e2a5`.
Każda poprawka przeszła niezależny przegląd poprawności i osobny etap
upraszczania przez Opus 5.5 przed lokalną integracją. Uproszczenie BOS nie
wymagało dalszej zmiany. Nie wykonano push, wydania ani publikacji danych.

Binarna wersja CUDA została zbudowana przy `bc5ad0aa`; późniejsze dwa commity
zmieniają wyłącznie Python i dokumentację. SHA-256 binarki:
`e6d4acb3efbcd444c89756561f47716b4e71dad303368a469801455e2afd9331`.
Fmt, Clippy i prywatny Rustdoc default/Intel/CUDA dla poprawki Rust przeszły;
kontrole zależności odziedziczono po potwierdzeniu niezmienionych wejść.
Python przeszedł `py_compile`, a obie zmiany `git diff --check`.

Sprzęt: NVIDIA GeForce RTX 4060, 8188 MiB, sterownik 615.78.08, toolkit
CUDA 12.9.1. Referencja: Python 3.12.15, PyTorch 2.11.0+cu128,
Transformers 5.17.0, tokenizers 0.23.3, FLA 0.6.0, Triton 3.6.0.
Upstream 2.4.1: `b20f0f9b159f71726459aa98ede1356ab65695fd`.
Checkout upstream i oryginalne środowisko pakietów pozostały niezmienione.

Modele i korpus są te same co poprzednio: Pico rewizja
`a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`, Mini 2.0 rewizja
`cb81b32ce89b0b87e62fcdad28f0435e25c953f3`. Nie pobierano nowych wag.
Mini używa przypiętej tabeli F16 GEMM SHA-256
`2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96`.
Inferencję wykonywał jeden proces GPU naraz. Surowe dane i źródła diagnostyki
są lokalnie poza Gitem; [odtworzenie](REPRODUCE.md).

## H1: pusty BOS w planerze Pico — PASS

`RequestPlanner::load` wymagał BOS obecnego w słowniku także dla Qwen3.5,
chociaż manifest tego modelu celowo nie ma BOS. Konstruktor online już
uwzględniał ten wyjątek. Planer stosuje teraz identyczną regułę. Pozostałe
style nadal zgłaszają `MissingBos`.

Pico z `qfirst/auto`, tau 0,7, band [0,1;0,9], `truncate`, head 0,25
i `--batches` uruchomiło się poprawnie. `/health/batches` podał
`profiles_ready=true`, pustą listę błędów i gotową bazę. Upload i utworzenie
batcha działają; anulowanie kończy się `cancelled`. Operator zaobserwował
exit 0 po SIGTERM w odpowiedzi narzędzia terminalowego; kod nie został
osobno zapisany w logu serwera.

**Ograniczenie funkcjonalne:** Qwen Pico nie dostarcza granicy host staging
ani współpracującego wykonania tła. Health poprawnie podaje `validates=false`
i `background_supported=false`. Poprawny batch pozostaje w `validating`
do anulowania. Naprawa startu nie oznacza dodatniej kwalifikacji walidacji
offline lub inferencji Batch dla Pico.

Nowa binarka wyeksportowała cały podstawowy korpus Pico F32/orders 1:
44 rekordy bench i 7 żądań API. Oba pliki `bench.jsonl` i `systemone.jsonl`
są **bajtowo identyczne** z eksportem sprzed poprawki, bez maskowania pól.
Pierwsza próba eksportu nie miała katalogu nadrzędnego; zachowano błąd
przygotowania i udane ponowienie po jego utworzeniu.

## H2: rzeczywista ścieżka referencji F32 — PASS

Poprzednia referencja z `dtype=float32` automatycznie wybierała FLA. Triton
wykonywał część iloczynów DeltaNet w TF32. Samo
`NVIDIA_TF32_OVERRIDE=0` nie zmieniło wyników, ponieważ nie steruje tą
arytmetyką Triton. Prywatna kopia środowiska bez importowalnego FLA
przywróciła zgodność bez zmiany kodu natywnego.

Eksporter dla F32 przypina cztery oryginalne funkcje DeltaNet/splotu PyTorch
z `modeling_qwen3_5.py` przed utworzeniem modelu. Weryfikuje ich pochodzenie
i zapisuje funkcje domyślne oraz użyte, wersje pakietów, SHA źródła i flagi
precyzji. BF16 zachowuje automatyczny wybór. Jawne `--qwen-delta auto`
pozwala odtworzyć poprzednią ścieżkę. Globalnego cuDNN TF32 nie wyłączono;
raport nie obiecuje ścisłego IEEE FP32 każdej operacji.

Maksymalna różnica skalibrowanego prawdopodobieństwa nowych pełnych
referencji wobec niezmienionych eksportów natywnych:

- Canonical: bench **0,000002041459**, API **0,000004529953**.
- Qfirst: bench **0,000001847744**, API **0,000000357628**.
- State_twice: bench **0,000001810491**, API **0,000000596046**.

Każdy wynik jest poniżej niezmienionego progu 0,0001. Tokeny zgadzają się;
argmax: 44/44 bench w każdym układzie oraz 14/14 odczytów API canonical,
10/10 w każdym pozostałym układzie. To nie jest deklaracja bitowej równości
Rust/PyTorch ani certyfikacja kalibracji nowych układów. Ostrzeżenia
`calibration unvalidated` nadal obowiązują.

Końcowe porównania korzystają bezpośrednio z trzech referencji przypiętego
eksportera. Eksporty natywne pochodzą z wcześniejszej binarki; nową binarką
potwierdzono bitowo canonical, lecz nie powtarzano qfirst/state_twice.
Jedyna zmiana Rust dotyczy walidacji BOS podczas ładowania planera.

Eksport canonical z przypiętą ścieżką w środowisku z FLA jest bajtowo
identyczny z diagnostycznym eksportem bez FLA dla bench. Dla API porównano
cały JSON po usunięciu wyłącznie `upstream_answer.usage.latency_ms`.
Po metadanych zmienionych w uproszczeniu wykonano dodatkowy pojedynczy
eksport: identyczny pierwszy rekord bench, policy `torch`, poprawne
`kernels=null` i `triton=3.6.0`. Pełnej serii nie powtarzano po tej zmianie.
Wcześniejszych wyników FLA i ich FAIL nie nadpisano.

## Mini: walidacja offline i ograniczony runtime tła

Mini `qfirst/auto/truncate` zgłosił `validates=true` i
`background_supported=true`. Przy domyślnie wyłączonym tle poprawny rekord
przeszedł do `in_progress` z total=1. Osobny plik zawierał poprawny wiersz
i wiersz z `think:null`: błąd `invalid_body` w wierszu 2 odrzucił cały batch
(status `failed`, total=0). Anulowanie poprawnego
zlecenia przeszło przez `cancelling` do `cancelled`.

Do rzeczywistego eksperymentu tła wykonano nowy bootstrap Mini na tej karcie:
canonical, AUTO, tau 0,6, band [0,2;0,8], refuse, bez prefix/state cache
i długiej linii. Sześć zamrożonych przypadków obejmuje sześć typów,
wspólny prefiks, Unicode, 27 i 255 opcji oraz dokument 4130 tokenów.
Benchmark wykonał 6 żądań sekwencyjnych, 6 przy C1 i 6 przy C4,
wszystkie poprawne, bez zmiany answers między obserwacjami tego benchmarku.

Żywa pamięć puli CUDA po ładowaniu: 3 226 569 728 B. Szczyt runtime:
3 663 878 928 B, przyrost **437 309 200 B**. Po shutdown used/reserved=0.
Pula nie obejmuje całej pamięci kontekstu, modułów i synchronicznych alokacji.
To pomiar tego korpusu, nie ogólna granica pamięci.

Eksperymentalny profil solo przeznaczony tylko do tego lokalnego QA używał:

- Limit tła i rezerwa online: po **905 969 664 B**. To dwukrotność zmierzonego
  przyrostu zaokrąglona w górę do 32 MiB, zatem 100% marginesu plus zaokrąglenie.
- Konserwatywne obciążenie każdej estymowanej alokacji: **2 MiB**, na podstawie
  zapytania o minimalną/rekomendowaną ziarnistość VMM sterownika CUDA.
  Próbnik 16 alokacji async we własnej nowej puli zwrócił used=requested
  i reserved=32 MiB; nie używał domyślnej puli basal i nie dowodzi
  fizycznego narzutu każdej alokacji.
- Współczynnik **2 252 298 ns/token**, jako konserwatywna inicjalizacja z
  maksimum compute fazy sekwencyjnej 650,914 ms i najmniejszego
  `input_tokens` benchmarku w korpusie: 289. Inna faza solo C1 miała
  maksimum 653,309 ms, a współdzielony batch C4 689,107 ms.
  Licznik benchmarku różni się od `usage.input_tokens` API dla trzech
  przypadków; minimum 289 jest wspólne. Nie jest to packed-token count
  ani bezpośredni pomiar czasu gate na packed token.

`qualified=true` w health oznacza zaakceptowaną deklarację operatora,
nie automatyczną akceptację naukową. Nie zapisano produkcyjnego profilu w repo.
Cały model nadal dopuszcza kontekst 8192; zbadano wyłącznie wskazany korpus.

Wyniki rzeczywistego tła:

1. Batch sześciu przypadków ukończył **6/6**, bez błędów, z czasem gate
   1 509 594 100 ns i rezerwacją urządzenia 0 po zakończeniu.
2. Podczas 60-rekordowego batcha wykonano sześć kolejnych żądań online.
   Wszystkie dały HTTP 200 i identyczne odpowiedzi jak online solo po
   usunięciu wyłącznie `usage.latency_ms`. Czasy 24,159–659,364 ms są
   pojedynczymi obserwacjami, bez wniosku o SLO lub regresji opóźnień.
3. Anulowanie zatrzymało ten batch po **16/60** wyników. Końcowy stan
   `cancelled`, 16 poprawnych wyników, bez błędów wyjściowych; rezerwacja 0.
   Health zachował jeden odrzucony spóźniony wynik i `abandoned_requeued=1`.
4. Następne 60-rekordowe zlecenie przerwano łagodnym SIGTERM przy
   `in_progress`, 11 gotowych wynikach i niezerowej rezerwacji. Serwer
   według obserwacji operatora zakończył się kodem 0 (nie zapisano osobnego
   rekordu exit). Liczba 11 pochodzi ze statusu batcha; snapshot health
   nie jest atomowy względem niego. Po restarcie z tą samą bazą i artefaktami zlecenie
   doszło do **60/60**, bez błędów i duplikatów identyfikatorów.

Wszystkie **82 wyniki** tła (6 + 16 + 60) odpowiadają całym odpowiedziom
online solo po usunięciu wyłącznie `usage.latency_ms`. Zachowano wszystkie
prawdopodobieństwa, odpowiedzi i pozostałe metadane. Sześć dodatkowych
odpowiedzi online w ruchu mieszanym również się zgadza. Liczniki health
resetują się przy restarcie; nie sumowano ich jako liczby unikalnych wyników.
Nie instrumentowano przekazania gate, więc nie ogłaszamy dowodu wywłaszczenia
kerneli ani konkretnych czasów oddawania urządzenia.

## Budżety refuse względem fazy D

Prywatny program CPU wywołał rzeczywiste publiczne estymatory planera
z bazy `1da2dd9` i końcowego źródła. Użył tego samego tokenizera Mini,
wyłączonego thinking/early-exit i identycznych jawnych możliwości planera.
Nie zastępował estymatorów własnymi równaniami. CLI nie udostępnia tych
wartości, dlatego diagnostyka pozostaje poza kodem produktu.

Zbadano **243 kombinacje przypadku, układu i orders**, łącznie
**3915 par prób budżetu**. Domyślne ustawienie końcowego planera i jawne
`refuse` dają identyczne całe pliki wyników. Wobec bazy zgodne są wyniki
akceptacji/odmowy, komunikaty błędów, zużycie i szczyt `PlanBudget`,
przygotowane prompty/tokeny, koszty planu i `result_bound`.
Było 1573 par poprawnych planów i 2342 par odmów, w tym 1730 odmów budżetu.

**Nie ma pełnej równości estymatorów bajtowych.** `RequestPlan` urósł
z 224 do 424 B, a `retained_bound` prawidłowo liczy `size_of::<RequestPlan>()`.
Daje to +200 B w każdym poprawnym planie. Dla 849 poprawnych par
`exec_bound` jest bez zmian, a `host_bound` rośnie o 200 B. Dla 724 par
z uwzględnionym planem kontynuacji również `exec_bound` rośnie o 200 B,
więc łączny `host_bound` rośnie o 400 B. Pozostałe pola porównania są
identyczne po wydzieleniu dokładnie tych czterech różniących się estymatorów.

To rzeczywisty, wyjaśniony wzrost rachunku pamięci nowych pól. Nie obniżano
poprawnego oszacowania dla uzyskania PASS. Jeśli pula wykonania leży między
starym a nowym `host_bound`, odmowa może nastąpić o 200 lub 400 B wcześniej.
Zgodność `PlanBudget` nie dowodzi identycznej granicy tej osobnej puli.
Diagnostyka używa jawnego syntetycznego host staging; nie kwalifikuje
rzeczywistego host/GPU boundu, media ani aktywnych funkcji 4.5B.

## Pozostałe GAP i granice wyniku

Pozytywne trained thinking wymaga obu przypiętych checkpointów
`Remek/basal-2.0-4.5B@e5ba9149760d99b73b1963d8e0ccf948b58afacb`,
w tym `think/`. Early-exit wymaga
`Remek/basal-1.0-4.5B@b95288041cb1975bb930c6f0410819273fd2f54f`
z `exit_heads/`. Lokalna karta 8 GB nie mieści nawet jednego wymaganego
niekwantyzowanego modelu. Kolejny pomiar powinien otrzymać dostępną kartę
CUDA 48 GB jako zapas roboczy; nie jest to zmierzony minimalny wymóg VRAM.
Nie pobierano wag, nie zamawiano maszyny i nie wykonywano wydatków.

Pozostają również Metal, Gemma4/media, pełny oryginalny kontekst Pico
262 144 tokenów, rzeczywisty OOM/device loss i odczyt mocy. Nie wywoływano
celowo OOM lub utraty urządzenia. Finite QA tła Mini nie zamyka ogólnego
boundu fizycznego alokatora, pełnego concurrent memory peak, długotrwałego
obciążenia ani kwalifikacji produkcyjnego profilu i wszystkich modeli.

Operator zaobserwował zakończenie wszystkich własnych serwerów kodem 0
w odpowiedziach narzędzia terminalowego; logi serwerów nie utrwalają kodów.
Końcowe osobno zapisane odczyty potwierdzają zamknięty port kampanii
i pustą listę procesów obliczeniowych GPU. Dane są lokalne,
bez publikacji HF; integralność archiwum jest opisana w `REPRODUCE.md`.
