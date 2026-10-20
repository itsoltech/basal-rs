# Basal mini: obserwowane przekazanie gate i pamięć — 2026-10-20

Osiem celowo wywołanych przekazań gate zakończyło się wznowieniem pracy tła, poprawną odpowiedzią online i ukończeniem Batch. Wszystkie osiem ciężkich odpowiedzi Batch jest dokładnie zgodnych z referencją online z tego uruchomienia oraz wcześniejszą referencją nieinstrumentowanego kandydata po usunięciu wyłącznie `usage.latency_ms`. Osobny, izolowany operator pamięci wykonał 17 przypadków: żaden zaobserwowany przyrost żywej pamięci puli CUDA nie przekroczył odpowiedniej estymacji źródłowej. Dwa pozostałe przypadki odrzucono podczas walidacji; zachowano je jako niewykonane.

To wąska kwalifikacja jednej konfiguracji. Nie potwierdza globalnego SLO, wywłaszczenia kernela GPU, ogólnej granicy pamięci ani wystarczającego zapasu online przy współbieżnym obciążeniu. Ten pakiet jest analizą ukończonego pomiaru, bez nowego uruchomienia GPU i bez kolejnego przeglądu implementacji.

## Wersje, wejścia i metoda

- Model `Remek/basal-2.0-mini`, rewizja `cb81b32ce89b0b87e62fcdad28f0435e25c953f3`; CUDA, NVIDIA RTX 6000 Ada Generation, 49 140 MiB, urządzenie 0. Wagi f16 skonwertowane z bf16; Candle 0.11.0, fused `attn_tree_tc_qk1_static`, cuBLASLt z przypiętą tabelą, akumulacja f32, bez TF32. Arytmetyka i zakresy są opisane także w rekordzie `resident_online_and_background_fork`.
- Tree, limit forward 2048, state cache 0, `long_tokens: 0`, profile `solo`, maksymalnie 1 żądanie tła. Prywatny profil kwalifikacyjny używa 2 GiB limitu urządzenia i 6 GiB rezerwy online; nie jest promocją profilu publicznego. Plik `inputs/lab-calibration.json` zachowuje rzeczywistą deklarację laboratorium, a `inputs/baseline-calibration.json` wcześniejsze wejście.
- `inputs/cases-mini-plus-boundary.json` zawiera 19 przypadków operatora pamięci, `inputs/cases-boundary-noul-r2.json` wejście przekazań gate. Zachowano 8 rzeczywistych wejść JSONL i 8 wyników Batch. Wagi modelu i binarka nie są redystrybuowane; konfiguracje, tabela GEMM, źródła operatorów i identyfikator binarki są w pakiecie.
- SHA-256 użytej binarki: `5202060f0a696c25b186c4ad1098f3085d7a06dc3a2509b98c488d83e574619c`. Zdalna kontrola porównała 269 plików źródłowych bez różnic. Pakiet zawiera wszystkie 269 dokładnych plików i manifest. Zamrożony produkt to 267 plików poprzedniego kandydata; rodzic osobno potwierdził 267/267 bez zmian. Instrumentacja zmienia tylko `gate.rs` i prywatny punkt wejścia CLI w `main.rs` oraz dodaje moduły `qualification_trace.rs` i `qualification_memory.rs`; zapisano obie łatki i manifest bazowy. Etykieta bazowego commitu `beb85627` nie zastępuje pełnych manifestów.
- Ślad gate dodaje zdarzenia hosta przez ograniczony kanał 65 536 zdarzeń, `try_send`, osobny wątek zapisu i licznik utraconych zdarzeń. Nie zmienia istniejącej synchronizacji checkpoint, ale może zmieniać czas hosta. Nie przeprowadzono równoległego pomiaru czasu bez instrumentacji.
- Operator HTTP czeka na zarejestrowane zajęcie gate przez tło, odczekuje 5 ms, wysyła krótki wniosek online i odczytuje status oraz artefakt Batch. Ścieżki online naprzemiennie `/v1/basal` i `/v1/systemone`. Każda próba zawiera jedno ciężkie żądanie Batch.

## Gate, wznowienie i wyniki

Ślad zawiera 260 zdarzeń jednego gate, ciągłe numery 0–259 i `dropped_before = 0` we wszystkich rekordach. W każdej próbie `demand_add` zapisuje właściciela `background`, następnie występują dokładnie jedno `yield_online`, zwolnienie tła, zajęcie przez `online_main`, zwolnienie online i wznowienie tła. Po zajęciu online następuje 11 kolejnych zajęć tła. Wszystkie Batch kończą się `completed`, total/completed/failed = 1/1/0, HTTP online i wyników 200, błąd wyniku null.

| Próba | Online | Demand→yield ms | Demand→online acquire ms | Demand→BG resume ms | HTTP online ms | Device health ns |
|---:|---|---:|---:|---:|---:|---:|
| 0 | /v1/basal | 0.786696 | 0.923881 | 7.585774 | 9.697806 | 543046199 |
| 1 | /v1/systemone | 0.778045 | 0.940606 | 7.580201 | 10.394055 | 550501632 |
| 2 | /v1/basal | 0.212746 | 0.679472 | 7.327579 | 9.339122 | 547952707 |
| 3 | /v1/systemone | 0.326374 | 0.698041 | 7.435809 | 10.064366 | 550749877 |
| 4 | /v1/basal | 0.933957 | 1.073430 | 7.727046 | 9.616998 | 552760803 |
| 5 | /v1/systemone | 0.929047 | 1.070598 | 7.696293 | 10.573916 | 550982467 |
| 6 | /v1/basal | 0.291153 | 0.697691 | 7.451026 | 9.078276 | 546105649 |
| 7 | /v1/systemone | 1.134734 | 1.273616 | 8.799790 | 11.607295 | 554387240 |

Czasy zdarzeń pochodzą z jednego `Instant` śladu i są różnicami `elapsed_ns`. HTTP mierzy osobny zegar monotoniczny klienta od wysłania do odczytu body. Nie odejmujemy tych dwóch osi czasu. Zdarzenie yield pochodzi z checkpoint po istniejącym drain/synchronizacji. Ślad potwierdza zajęty gate aktywnego forwardu tła przy zgłoszeniu popytu online, a następnie oddanie własności; nie zebrano śladu kerneli, więc nie dowodzi, że konkretny kernel fizycznie wykonywał się dokładnie w chwili nadejścia ani że nastąpiło sprzętowe wywłaszczenie. Zakres 0,679472–1,273616 ms dotyczy hostowego demand→acquire tych ośmiu prób, nie globalnego limitu. HTTP wyniósł 9,078276–11,607295 ms; osiem krótkich żądań nie stanowi nowego badania p99.

Health zwiększył `commits` i `accepted_results` o 8, `device_ns` o 4 396 486 574 ns. `interruptions`, `failed_groups`, `discarded_wall_ms`, `discarded_device_ns`, `failed_group_device_ns`, `requeued_results`, `rejected_results` i `commit_replays` miały przyrost zero. W każdym odczycie po próbie pula hosta była pełna, `reserved_bytes` urządzenia zero, urządzenie nie było quarantined. Suma hostowych interwałów acquire→release tła wynosi 4 396 261 354 ns, o 225 220 ns mniej niż licznik ownership. Zdarzenia i metryka mają różne punkty pobrania czasu; nie utożsamiamy ich z czasem kerneli ani z czasem HTTP. Nie zebrano osobnego ledgeru kosztu zarezerwowanego/refundu SQL: z tych zer nie wynika potwierdzenie ścieżki refundu.

Dokładne porównanie body używa kanonicznego JSON z uporządkowanymi kluczami i usuwa tylko `body.usage.latency_ms`. Zachowuje model, wszystkie odpowiedzi, prawdopodobieństwa, confidence i pozostałe usage; nie używa tolerancji float. Referencja ciężka tego uruchomienia i oryginalna `packed-32-near-limit.body` są zgodne; wszystkie 8 wyników Batch zgadza się z obiema. Koperty wyniku z ID/request_id nie są częścią body i pozostają w surowych zanonimizowanych wynikach.

Cztery krótkie odpowiedzi Basal mają pełne body zgodne z krótką referencją Basal bez latency. Cztery odpowiedzi SystemOne celowo zawierają lean schema bez probabilities/confidence. Ich porównanie do całego body Basal daje **false** i jest jawnie zachowane w analizie; to różnica kontraktu ścieżek. Osobna porównywana projekcja obu ścieżek zachowuje model, komplet kluczy odpowiedzi, type/noul i wszystkie nietimingowe pola usage: 8/8 dokładnie zgodnych. Wartość noul wynosi 0,5429822206497192, input_tokens90, output_tokens0, questions1, branches1, longest_prompt_tokens90.

## Pamięć: 17 wykonanych, 2 odrzucone

Operator pamięci tworzy rozgrzany online engine i jego fork tła; online pozostaje bezczynny. Prywatny gate przyłączono wyłącznie do pracy tła. Przed i po pracy operator synchronizuje urządzenie i resetuje high watermark puli, dlatego jego odcinki czasu nie są benchmarkiem latency. Poniższy przyrost to `used_high_bytes - used_bytes_before`, bez resident weights/prefixów, porównany z odpowiednią estymacją źródłową. Rezerwowana pojemność puli i całkowity peak mają inne znaczenie.

| Przypadek | Przyrost live B | Source B | Rounded B | Live/source | Próbki RSS podczas pracy |
|---|---:|---:|---:|---:|---:|
| so-structured-state | 23294932 | 60525828 | 115051754 | 0.384876 | 1 |
| so-array-state-en | 15882908 | 35178800 | 89704726 | 0.451491 | 1 |
| so-null-and-missing-instructions | 7893328 | 22007628 | 76533554 | 0.358663 | 0 |
| so-empty-structures | 8759668 | 35522932 | 90048858 | 0.246592 | 2 |
| so-duplicate-descriptions-hide | 9048448 | 27142172 | 81668098 | 0.333372 | 0 |
| so-unicode | 16075428 | 34955560 | 89481486 | 0.459882 | 1 |
| so-polish-from-instructions | 5679344 | 11617716 | 66143642 | 0.488852 | 1 |
| so-ten-options | 23872484 | 32713960 | 87239886 | 0.729734 | 0 |
| so-score-two-levels | 5775604 | 11729336 | 66255262 | 0.492407 | 1 |
| so-eleven-options | 10299824 | 16975548 | 71501474 | 0.606745 | 1 |
| so-unknown-type | 7315768 | 20556568 | 75082494 | 0.355885 | 0 |
| so-missing-type | 3369104 | 8938836 | 63464762 | 0.376906 | 1 |
| v2-multi | 13861452 | 36415876 | 90941802 | 0.380643 | 1 |
| long-text | 112816724 | 134622956 | 189148882 | 0.838020 | 3 |
| continuation-choice | 12225024 | 19207956 | 73733882 | 0.636456 | 0 |
| near-limit-single | 184337904 | 218784436 | 273310362 | 0.842555 | 4 |
| packed-32-near-limit | 190594812 | 228901248 | 283427174 | 0.832651 | 54 |

W każdym z 17 przypadków wynik `ok:true`, po jednym PlanOutput, live poniżej raw i rounded. Największy stosunek live/source to 0,842555 (`near-limit-single`), live/rounded0,674464. Wszystkie 17 po resecie wraca dokładnie do tego samego live i reserved przed pracą: 3 252 456 448 B oraz 3 254 779 904 B. Maksymalny przyrost live190 594 812 B, reserved268 435 456 B; całkowite high live3 443 051 260 B i reserved3 523 215 360 B. Po usunięciu obu engine i synchronizacji current puli wynosi0/0; high watermark pozostaje historyczny.

Każda estymacja źródłowa liczy 26 alokacji. Nie jest to pomiar liczby wywołań allocator. Rounded = source + 26 × (2 097 152 − 1); 2 MiB to założenie operatora, nie dowód rzeczywistego zaokrąglania lub fragmentacji. Live pool nie obejmuje kontekstu CUDA, modułów, wewnętrznych alokacji bibliotek ani wszystkich alokacji synchronicznych. Nie porównujemy całego resident-inclusive reserved z granicą transient live.

**Korekta etykiety wcześniejszego raportu:** dla 32 pytań **62 902 to suma tokenów promptów**, nie liczba packed ani pojedynczego forward. `packed_tokens = 2864` po współdzieleniu prefiksu, longest_prompt1966, konserwatywne forward_tokens2048, readouts32/letters64. Dla single odpowiednie wartości to1960/1960/1960. Poprzedni raport pozostaje historyczny; ta nota koryguje interpretację bez nadpisywania wyników.

Odrzucone i niewykonane: `so-score-one-level` — pojedyncza opcja, wymagane2..255; `v2-span` — extract schema nie jest obiektem. Nie liczymy ich jako sukcesów inferencji. Analiza zachowuje pełne komunikaty walidacji.

RSS:1224 rekordy próbkowania co10 ms,1218 z RSS/HWM i6 bez tych wartości przy końcu procesu. Pięć krótkich przypadków nie ma próbki wewnątrz pracy (zera w tabeli): jest to **GAP**, nie zero pamięci. Najbliższe próbki przed/po zachowano osobno; nie zastępują brakujących. Wewnątrz obserwowanych odcinków RSS i lifetime VmHWM wynosiły688 160 768 B. Maksymalny RSS/HWM całego procesu771 948 544 B wystąpił12,167648 ms po markerze engine drop, poza każdym mierzonym requestem; przyczyny nie zmierzono. RSS to host, nie pamięć GPU; lifetime HWM nie jest per-request telemetry allocator.

Guard pamięci:32 próbki, minimum free9379 MiB, maksimum własnego procesu3838 MiB. Guard serwera:208 próbek, minimum free9347 MiB, maksimum własnego procesu3870 MiB. Brak zdarzeń stop. Progi guard minfree5120/maxown7680 MiB, próg wejścia launchera12000 MiB. To próbkowana telemetria NVML; zegar elapsed guard nie ma wspólnego UTC anchor do join per-case. Współbieżnej granicy full-service i rezerwy online ten izolowany pomiar nie dowodzi.

Rodzic powtórzył redukcję pamięci w świeżym katalogu: JSON/TXT bitowo identyczne, niezależnie17 przypadków poniżej raw source. Pakiet zachowuje to potwierdzenie. Rodzic zakończył oba procesy i operator HTTP kodem0, listener zamknięty, własnych kontenerów0; obserwował powrót do wcześniejszego VRAM baseline35 288 MiB used/13 222 MiB free i niezmieniony proces produkcyjny. Dane w archiwum zachowują tylko te wartości i booleany, bez identyfikatorów maszyn/procesów. Samo current pool0/0 nie potwierdza pełnego zwolnienia drivera.

## Zachowany FAIL i granice decyzji

Wcześniejsza próba boundary-no-continuation miała32 obserwacje przy concurrency2: baseline p99 **1251,0019890032709 ms**, mixed **1265,462988987565 ms**, oryginalny próg baseline+10 ms = **1261,0019890032709 ms**. Status progu pozostaje **FAIL**: nadmiar4,4609999842942 ms, wzrost względem baseline14,4609999842942 ms (~1,16%). Użytkownik zaakceptował2026-10-20 wyłącznie ten konkretny nadmiar (`ACCEPTED_DEVIATION`); nie zmieniono progu, nie awansowano profilu i nie ogłoszono nowego SLO. Zapis decyzji i obie oryginalne summary oraz threshold są w `references/`. Te osiem dodatkowych przekazań nie zastępuje pierwotnego eksperymentu ani nie dowodzi przyczyny tamtej różnicy.

Pierwsze dwa buildy prywatnego operatora nie przeszły: przeniesiony cache uruchomił CLI build-script z osadzoną przez `env!` ścieżką poprzedniego katalogu, co zakończyło się brakiem `INVARIANT_SEARCH_VERSION`. Zachowano oba logi i kody101. Trzecie podejście po `cargo clean --release -p basal-cli` zbudowało pakiet poprawnie, kod0,1m27s. Są to niepowodzenia przygotowania środowiska/cache, nie zmierzona regresja produktu. Nie dodano testów/frameworków ani bramek CI.

Nie wykazano ogólnego lub ciasnego bound, innej precyzji/modelu/backendu, multi-device, produkcyjnego profilu, pełnego concurrent online memory peak, liczby runtime allocation calls, numerycznej zgodności z FP32 upstream w operatorze pamięci ani nieinstrumentowanego SLO. To raport wąskich obserwacji; wcześniejsze FAIL/GAP pozostają zachowane.

## Dane i odtworzenie analizy

Pakiet lokalny, bez publikacji HF. Logiczne dane `reports/batches-mini-followup-2026-10-20/`: `inputs/`, `handoff/`, `memory/`, `analysis/`, `operators/`, `references/`, `provenance/`, `source/`, `logs/`. `SHA256SUMS` obejmuje wszystkie pliki. Manifest sanitizacji zachowuje oryginalne SHA i wskazuje zmienione teksty. Uruchomienia hosta mają przenośne aliasy `/qualification/...`; identyfikatory, adresy i fizyczne ścieżki zostały usunięte. Dokładne źródła produktu zachowują zwykłe generyczne literały loopback/wildcard z kodu i przykładów; nie są danymi pomiarowej maszyny. Logi nie zawierają rzeczywistych adresów. Sanitizowane launch recipes wymagają jawnego przypięcia modelu/runtime do aliasów; nie są automatycznym ponownym uruchomieniem na cudzym hoście.

Po rozpakowaniu do nowego katalogu i `sha256sum -c SHA256SUMS` można odtworzyć wyłącznie analizę, bez GPU/HTTP:

```bash
python3 "$DATA/operators/analyze-handoff.py" --measurement-artifacts "$DATA" --output "$NEW/handoff-analysis.json"
```

`DATA` oznacza powyższy katalog logiczny, `NEW` nowy pusty katalog. Wynik porównaj z `analysis/handoff-analysis.json`. Dla pamięci skopiuj niezmieniony `memory-analysis.py` i jego cztery wejścia (`memory-only.jsonl`, `memory-rss.jsonl`, `memory-guard.jsonl`, `remote-source-verification.json`) z `memory/` do osobnego świeżego katalogu i uruchom tam skrypt; tworzy JSON/TXT wyłącznie jako nowe pliki. Porównaj je bitowo z zawartością pakietu. Pełny forward wymaga zewnętrznych wag i środowiska CUDA; nie jest potrzebny do reprodukcji tej redukcji.


Archiwum: `.cache/measurement-artifacts/batches-gpu-followup-packet-20261020/measurement-data-qualified.tar.gz`, **1,744,180 B**, SHA-256 **`b8dbff8d8caa2e882aee151d742f90086d30f9c0f593fb4a5018ecdb1e0f4a61`**. Świeże lokalne rozpakowanie: **354/354** sum zgodnych, zero różnic; wszystkie pliki objęte manifestem (w tym vendored SHA256SUMS). Pełne269 plików source zgodne z manifestem, redukcja gate i pamięci odtworzona bitowo identycznie. README/METHOD wewnątrz archiwum zawierają treść sprzed dopisania tej noty integralności. URI HF: brak, **nie opublikowano**. Niezależna kontrola rodzica tego nowego archiwum pozostaje osobnym krokiem.
