# Batch API — poprawki R2 i obserwacje operatora

Status: **PASS opisanych prób magazynu i helperów; PENDING pełnego serwisu i długiego eksportu; GAP pełnej kwalifikacji**.
Data: 2026-10-19. Raport dotyczy poprawek pięciu uwag niezależnego przeglądu R2.
Nie oznacza gotowości produkcyjnej ani zamknięcia inicjatywy. Brak inferencji GPU,
pomiaru jakości modelu, szczytu RSS, przyspieszenia i kwalifikacji SQL Server runtime.
Raport pozostaje do końcowego uzupełnienia przez rodzica po trwającym zbieraniu dowodów pełnego serwisu.

## Cel, wersje i wejścia

Sprawdzić przypięcie katalogu po rolling update, ograniczone czyszczenie ledgerów,
granice multipart, czas życia slotów I/O i zamykanie połączeń HTTP. Kod to commit bazowy
`beb856275fbcb811eff5ce4446479bd74dafa176` oraz zamrożone zmiany drzewa roboczego;
nie jest to pomiar opublikowanego commitu. Archiwum przypina konkretne bajty źródeł
przez `SHA256SUMS` i manifesty pochodzenia. Nie uruchamiano testów projektu,
frameworków, snapshotów testowych ani nowych bramek CI.

Próby CPU wykonano na Linux. Serwery zgłosiły PostgreSQL **18.6** i MongoDB **8.0.34**;
Mongo działało jako pojedynczy lokalny replica set z cache 0.25 GiB. SQLx **0.9.0**,
Mongo driver **3.9.1**, libsqlite3-sys **0.37.0** (bundled), Axum **0.8.9**,
Multer **3.1.0**, Hyper **1.11.1**, hyper-util **0.1.21**, Tokio **1.53.2**.
Dokładne grafy operatorów zachowano w ich Cargo.lock. To nie jest benchmark sprzętu.

Store operator korzysta z dokładnej kopii rzeczywistych adapterów. Tworzy katalogi
`m1/fp1` i `m2/fp2`, syntetyczny plik wejściowy i batch, normalne validation errors
oraz syntetyczną historię zerowych, już rozliczonych retry. Nie wykonuje modelu.
Helper operator używa dokładnych upload/http helperów i sekcji ArtifactReader
z minimalnym scaffoldingiem typów. Nie uruchamia pełnego Batch API ani jego store.

## F1–F5: źródło, obserwacja i granica dowodu

| Uwaga R2 | Poprawka w źródle | Obserwowane | Pozostałe ograniczenie |
| --- | --- | --- | --- |
| F1: nieograniczone framing/buforowanie multipart | Adapter przed Multer ogranicza raw total i outstanding bytes; framing nie otrzymuje kredytu za payload. Yield pomiędzy gotowymi transport frames pozwala parserowi przekazać payload konsumentowi. | Helper: poprawny plik 300000 B przed/po purpose daje 200; oversized frame, rzeczywisty raw total, niedomknięty preamble/header i nadmierny epilogue dają 413. | PENDING pełnego serwisu; brak pomiaru RSS. Frame większy niż upload_buffer_bytes celowo daje 413. |
| F2: nowy katalog odrzuca poprawne starsze batche | SQL/Mongo czytają dokładną immutable revision z limitem 1024 wpisów, walidacją i digestem; ingest bierze przypięty default. Brak oczekiwanej revision jest typowanym Corrupt. | SQLite/PG/Mongo: revision 1 nadal zwraca m1 przy latest m2; discovery m1 zostaje przy fp1; missing lookup to None, błędny digest to Corrupt. | PENDING pełnej walidacji JSONL z pierwszym jawnym modelem i późniejszym defaultem; lookup operatora nie wykonuje ingest/modelu. SQL Server runtime GAP. |
| F3: sloty I/O i shutdown nie mają skutecznego końca | Jeden deadline obejmuje cały upload; reader watchdog oddaje źródło i permit niezależnie od polling. HTTP helper posiada połączenia i zadania H2, po grace abortuje i joinuje pozostałe. | Helper: silent upload 408 po 35–36 ms dla limitu 35 ms, shutdown 503; slot readera wraca bez polling. Prawdziwe izolowane połączenia H1/H2 zamykają pending/backpressured odpowiedzi po 50–52 ms dla grace 50 ms, guard drop=1. | PENDING pełnego serwisu, sesji upload/quota i długiego eksportu; osobny helper socketowy nie kwalifikuje całego serwera. |
| F4: długi eksport zatrzymuje lifecycle | Jeden stały finalizer jest osobnym, posiadanym zadaniem serwisu; lifecycle ma niezależną pętlę. Graceful shutdown zachowuje handle, abort/join i fencing. | Potwierdzone kodem i kompilacją; zachowano ograniczenie do jednego eksportu na proces. | **PENDING** rzeczywistego wolnego eksportu jednocześnie z expiry/drain/settlement/recovery/GC i shutdown. Brak twierdzenia o zmierzonym postępie z samego rozdzielenia tasków. |
| F5: nieograniczony terminalny cleanup | Settlement → claims → internal operations: najwyżej 512 ledger rows w transakcji. Metadata errors/operations również stronami; done dopiero po pustych ledgerach i accounting_pending=0. Quota/batch locks i idempotency retention zachowane. | SQLite/PG/Mongo: strony, reconnect, accounting barrier, replay, client idempotency i quota drift 0; serwery potwierdziły ograniczone zapytania. | SQL Server runtime GAP; brak process-kill/failover między stronami oraz benchmarku dużych produkcyjnych ledgerów. |

## Store: wynik native operations

W każdym z trzech silników batch przyjęto przy revision 1/default `m1`, potem
zainstalowano revision 2/default `m2`. Exact lookup i jawne discovery pozostały przy
starym modelu/fingerprint. Nieistniejąca revision zwraca `None` w API
`Result<Option<Catalog>, StoreError>`; błędny digest istniejącej revision daje
`Corrupt("immutable catalog digest mismatch")`. Operator przywraca digest przed dalszą pracą.
Mapowanie braku **oczekiwanej przypiętej** revision na Corrupt w ingest jest potwierdzone
źródłowo, a nie tym standalone przebiegiem.

Początkowo istniało 1201 settlements z jednym claim_id i różnymi line_no, 1103 claims,
1301 internal operations, jeden prawdziwy create operation oraz 1000 errors zapisanych
przez normalne `fail_validation`. Przy `accounting_pending=1` cleanup niczego nie usunął.
Potem transakcje usuwały kolejno:

| Ledger | Usunięte w kolejnych transakcjach |
| --- | --- |
| Settlements | 512 / 512 / 177 |
| Claims | 512 / 512 / 79 |
| Internal operations | 512 / 512 / 277 |
| Validation errors, dopiero po retention | 512 / 488 |
| Client create operation, dopiero po retention | 1 |

Store zamknięto i ponownie otwarto po drugiej stronie cleanup. Dalsze strony zachowały
postęp. Done pojawiło się po potwierdzeniu pustych ledgerów; create operation pozostał
do metadata retention. Pełny metadata charge i retained slot zwolniono dopiero
w końcowej transakcji usuwającej batch. Quota drift wynosił **0 przed purge,
po każdej jego stronie oraz po zakończeniu**. Powtórny cleanup zwrócił done,
powtórny purge usunął 0 batchy.

PostgreSQL zarejestrował 27 bounded DELETE pięciu typów. Każde ma owner/batch filter,
pełny klucz i podzapytanie `LIMIT 512`, również composite `(claim_id,line_no)` settlements.
Mongo profiler zarejestrował 27 bezpośrednich page aggregates:
`$match owner_id/batch_id → $project _id → $limit 512`.
Dwanaście poleceń usuwania miało najwyżej 512 `_id` w `$in`; każde `ndeleted`
było równe liczbie wybranych id. Repeated originatingCommand/getMore nie są liczone
jako nowe polecenia aggregate. Zapytania SQL Server korzystają z `DELETE TOP (512)`;
to ocena źródła, bez obserwacji na rzeczywistym SQL Server.

Prywatne instancje baz zatrzymano jawnie po sesji; nie pozostawiono działających usług.
Próby nie obejmują HA, multihost, sharding, S3 ani opóźnień i przepustowości.

## Upload / reader / HTTP helper

Końcowy przebieg helpera r5 dostarczył dokładnie 300005 B pól z pliku 300000 B
oraz purpose, niezależnie od kolejności tych pól. Niedomknięty preamble/header
200000 B i epilogue 70000 B zwróciły 413. Silent preamble/header/text/file oraz
stream po closing boundary zwróciły 408. Deadline nie jest restartowany przy file part.
Watchdog oddał slot zarówno po deadline bez polling, po shutdown bez polling,
jak i po dołączeniu permitu do readera już zamkniętego przez watchdog.

Izolowane rzeczywiste H1 i H2 połączenia obejmowały pending handler i trwale
backpressured response. Shutdown zakończył się po 50–52 ms dla grace 50 ms,
a odpowiednie guardy handlera/body zostały zwolnione po jednym razie. To dowód
transport helpera; pełny Batch service jest nadal osobnym PENDING.

**Ograniczenie kompatybilności:** pojedynczy raw transport frame większy niż
`upload_buffer_bytes` jest odrzucany 413, nawet gdy zawiera poprawne multipart.
Adapter nie zachowuje slices do nadmiernie dużej alokacji wejściowej. Domyślne 1 MiB
mieści typowe transport frames; konfiguracja 64 KiB może odrzucać większe frames H1.
To świadomie przyjęta granica pamięci, nie deklaracja zgodności każdego klienta HTTP.
Outstanding parser bytes nie są szczytem RSS: osobne alokacje mają transport,
parsed headers, tekstowe pola i backend artefaktów.

Zachowano nieudane początkowe próby: r1/r2 operatora helpera nie kompilowały przez
niepełne izolowane mock types digest/Copy; po poprawieniu scaffolding zachowano r3/r4/r5.
Nie jest to ukryta porażka kompilacji produktu. SQLite operator początkowo użył
niedozwolonego w SQLx 0.9 dynamicznego String zapytania diagnostycznego, potem zastąpił go
stałymi literałami. PostgreSQL przygotowanie wymagało skrócenia prywatnego Unix socketu
i skorygowania URI z pustym hostem. Nieudane próby pozostają prywatnie; archiwum zawiera
ich opis oraz zredagowane diagnostyki, bez identyfikatorów środowiska.

## Weryfikacja i granice zamknięcia

Scoped release Clippy basal-cli/all-targets z `-D warnings`, Rustdoc z prywatnymi
pozycjami i `-D warnings` oraz fmt przeszły w opisanych zamrożonych pakietach.
Ostateczna macierz całego workspace, Intel/CUDA/Metal i kontroli zależności
jest odpowiedzialnością rodzica; ten raport jej nie zastępuje. Wcześniejsze próby
podczas równoległych edycji miały przejściowy locked-manifest/import Duration błąd;
nie są przedstawiane jako finalny wynik całej integracji.

PENDING: trwający operator pełnego serwisu, quota/session cleanup dla HTTP,
pierwszy jawny model/późniejszy default podczas rolling catalog update,
prawdziwy wolny eksport z niezależnym lifecycle i shutdown.
GAP: SQL Server runtime, GPU/model/numerical equivalence, S3, production RSS,
process-kill między stronami, aktualne HA/multihost oraz benchmark wydajności.
Nie wykonano trzeciego pełnego cyklu przeglądu ani publikacji.

## Dane i odtworzenie

Lokalny, niepublikowany pakiet ma logiczną ścieżkę `reports/batches-r2-fixes-2026-10-19/`.
Nie ma URI HF ani potwierdzenia pobrania z autoryzacją/anonimowo. Nie wysyłano danych.
Raw logi serwerów, DSN, adresy sesji i prywatne ścieżki pozostają poza pakietem.
Kod zachowuje wyłącznie ogólne stałe loopback wymagane przez lokalną implementację;
nie zawiera adresów ani nazw rzeczywistych maszyn z sesji.

W archiwum znajdują się:

- `observations/store/{sqlite,postgresql,mongodb}.log`, `observations/upload/observations-r3..r5.log`;
- `observations/store/postgresql-bounded-queries.log` i `mongodb-bounded-filters.json`;
- `sources/product/` — przypięte bajty zmienionych granic i wspólnego store;
- `operators/{sqlite,postgresql,mongodb,upload-helper}/` — Cargo.lock i źródła ręcznego operatora;
- `provenance/` — mapowanie oryginalnych i archiwalnych hashy oraz nieudane przygotowania;
- `REPRODUCE.md`, `REPORT-SNAPSHOT.md` oraz root `SHA256SUMS`.

Operatorów uruchamia się ręcznie na świeżych prywatnych bazach po jawnej weryfikacji
`SHA256SUMS`; dokładne kroki i granice są w `REPRODUCE.md`. Nie są częścią CI ani bramką
testową. Archiwalny REPORT-SNAPSHOT jest migawką tego raportu sprzed dopisania SHA
samego archiwum. Raport w Gicie może otrzymać późniejszy dodatek rodzica; zakończonego
archiwum nie wolno nadpisywać.

Lokalne archiwum: `.cache/measurement-artifacts/batches-r2-fixes-20261019/measurement-data.tar.gz`.
SHA-256: `e5a9d6cdcdca4a6c786c8c4cee731800b100c1d49a93033b630ab89e06564c83`. Rozmiar: 670083 B.
Osobne świeże rozpakowanie potwierdziło wszystkie **194/194** sum z root `SHA256SUMS`;
SHA archiwum sprawdzono ponownie po rozpakowaniu. To weryfikacja lokalna; publikacji i pobrania HF nie wykonano.
`ARCHIVE_SHA256SUMS` i `verification.json` pozostają obok lokalnego archiwum.

Końcowy raport pełnej usługi i F4 pozostaje **PENDING**; ten niezmienny pakiet obejmuje wyłącznie
zakończone próby store i helperów opisane powyżej. Dalsze wyniki wymagają nowego archiwum.


## Niezależna kontrola rodzica

2026-10-19: niezależnie sprawdzono SHA-256 archiwum `e5a9d6cdcdca4a6c786c8c4cee731800b100c1d49a93033b630ab89e06564c83`
oraz wszystkie **194/194** wpisów `SHA256SUMS` po nowym rozpakowaniu.
Opisane wyniki i ograniczenia pozostają bez zmian. Weryfikacja jest lokalna;
nie potwierdza publikacji ani pobrania HF.
