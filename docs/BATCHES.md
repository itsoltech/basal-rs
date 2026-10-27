# Batch i Files API

Od serwera 0.1.13 wszystkie endpointy API działają bez uwierzytelniania kluczem API.
Batch/Files udostępniają wspólne zasoby wszystkim klientom.

Funkcja jest domyślnie wyłączona. Po włączeniu udostępnia trwałe zlecenia JSONL,
Files API i `/health/batches`. Format rekordu i lifecycle nawiązują do OpenAI Batch API;
obsługiwanym endpointem rekordu jest obecnie wyłącznie `POST /v1/basal`.
`/v1/systemone` w batchu zwraca jawny błąd. Kontrakt maszynowy:
[contracts/batches/openapi.json](../contracts/batches/openapi.json).

Klienty Python, TypeScript/JavaScript i C# od 0.4.0 oraz Rust od 0.5.0
udostępniają Batch/Files API. Przepływ aplikacji i przykłady:
[SDK — Batch i Files](SDK.md#batch-i-files-api).

Ruch online `/v1/basal` i `/v1/systemone` ma pierwszeństwo przed batchami,
także przed ich zbliżającym się terminem. Okno wynosi 24 godziny od utworzenia
batcha i obejmuje walidację oraz oczekiwanie w kolejce. To termin wygaśnięcia,
nie bezwarunkowa gwarancja ukończenia. Przy stałym obciążeniu online batch może
wygasnąć z częściowymi wynikami.

Implementacja jest w trakcie kwalifikacji. Włączenie API nie włącza wykonywania
GPU: model wymaga osobnego `batch_profile` popartego pomiarami. Bez niego można
obsługiwać pliki i lifecycle, ale żądania nie uzyskają wyników inferencji w tle.
Kompilacja, próby magazynu i syntetyczne wyniki nie potwierdzają zgodności numerycznej
ani opóźnień na GPU. Aktualne dowody magazynów:
[PostgreSQL](../reports/batches-postgresql-store-2026-10-19/README.md),
[MongoDB](../reports/batches-mongodb-store-2026-10-19/README.md),
[HA MongoDB](../reports/batches-mongodb-failover-2026-10-19/README.md),
[HTTP i artefakty](../reports/batches-http-store-2026-10-19/README.md),
[poprawki po przeglądzie](../reports/batches-review-fixes-2026-10-19/README.md),
[poprawki R2](../reports/batches-r2-fixes-2026-10-19/README.md),
[natywne HTTP i wolny eksport](../reports/batches-native-http-r2-2026-10-19/README.md),
[odzyskanie walidacji po zmianie defaultu](../reports/batches-pinned-default-r2-2026-10-19/README.md),
[SQL Server 2022](../reports/batches-mssql-runtime-2026-10-19/README.md),
[aktualny klient S3](../reports/batches-s3-r2-2026-10-20/README.md),
[konfigurowalne migracje startowe](../reports/batches-auto-migrate-2026-10-20/README.md).
Próby baz obejmują opisane transakcje i konfiguracje; nie stanowią kwalifikacji
każdej wersji, topologii HA ani obciążenia produkcyjnego.

[Pomiar `basal-2.0-mini` na CUDA](../reports/batches-mini-gpu-2026-10-19/README.md)
potwierdził zgodność odpowiedzi i opisane krótkie przebiegi mieszane. Próba przy
granicy tokenów przekroczyła ustalony próg p99 online; profil laboratoryjny nie
otrzymał kwalifikacji produkcyjnej. Raport zachowuje ten FAIL i ograniczenia
pamięci, zadań z kontynuacjami oraz metody pomiaru.

Użytkownik zaakceptował konkretne przekroczenie progu o 4,461 ms; pierwotny
FAIL pozostaje w raporcie. [Pomiar uzupełniający](../reports/batches-mini-followup-2026-10-20/README.md)
potwierdził osiem przejęć GPU przez ruch online podczas pracy tła oraz powrót
puli CUDA do poziomu rezydentnego po 17 wykonaniach. Przyrost zajętych bajtów
puli mieścił się w oszacowaniu źródłowym dla każdego z tych wejść. To dowody
dla opisanego profilu mini CUDA f16 solo; nie kwalifikują wszystkich kształtów,
współbieżnej pamięci online/tła ani innych modeli i topologii.

## Szybki start lokalny

```sh
basal serve --model mini --batches --batch-data-dir ./batch-data
```

Profil `local` używa SQLite i plików w prywatnym katalogu. Serwer domyślnie nasłuchuje
na loopback. Nie tworzy ani nie odczytuje tokenów. Nie dodawaj bazy ani artefaktów do Gita.
Nie zmieniaj katalogu danych pomiędzy restartami, jeżeli chcesz zachować kolejkę i wyniki.

Bez sekcji głównej `database:` profil `local` używa dotychczasowego pliku
`<katalog danych>/batches.sqlite3` z automatyczną migracją. Jawna sekcja
`database:` z `backend: sqlite` i własnym `path` zastępuje ten plik; wtedy
`--batch-data-dir` dotyczy już tylko artefaktów. Przy
`database.enabled: false` opcja `--batches` nie wybiera innej bazy: serwer
odmawia startu przed ładowaniem modelu. Szczegóły bazy: [DATABASE.md](DATABASE.md).

Domyślna lokalna quota: 10 GiB, 10 aktywnych batchy, 1000 przechowywanych
batchy i plików oraz 2 jednoczesne sesje uploadu. Rezerwacje obejmują także
metadane, przyszłe wyniki i eksport, więc quota nie jest samym rozmiarem plików.

Przygotuj `input.jsonl`, po jednym rekordzie w wierszu:

```jsonl
{"custom_id":"ticket-1","method":"POST","url":"/v1/basal","body":{"state":"Klient prosi o zwrot pieniędzy.","questions":{"dept":{"type":"choice","instructions":"Do którego działu skierować zgłoszenie?","criteria":{"returns":"Zwroty","support":"Wsparcie"}}}}}
```

`body` ma ten sam kontrakt co online [Basal API](../contracts/basal/openapi.json).
`custom_id` musi być unikalne w pliku. Pominięte `body.model` wskazuje model
domyślny z przypiętej rewizji katalogu, a nie aktualny model w chwili wykonania.
Upload i utworzenie zlecenia to dwie osobne operacje:

```sh
curl --fail-with-body http://localhost:8000/v1/files \
  -F purpose=batch -F file=@input.jsonl

# Zastąp file_... identyfikatorem z odpowiedzi uploadu.
curl --fail-with-body http://localhost:8000/v1/batches \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: import-1' \
  -d '{"input_file_id":"file_...","endpoint":"/v1/basal","completion_window":"24h"}'
```

Ten sam `Idempotency-Key` i to samo wejście odtwarzają utworzony batch;
zmiana wejścia dla istniejącego klucza daje 409. Sam upload nie jest tą operacją
idempotentną. Klucz należy do uwierzytelnionego właściciela.

## Trasy i lifecycle

Poniższe trasy zasobów nie wymagają klucza API.
Nagłówki proxy ani pole właściciela w body nie nadają dostępu do danych.
Nieistniejący zasób i zasób innego właściciela zwracają ten sam 404.
Uwierzytelnianie dotychczasowych tras inferencji pozostaje osobną konfiguracją.

| Metoda i ścieżka | Operacja |
|---|---|
| `POST /v1/files` | Upload multipart, `purpose=batch` |
| `GET /v1/files` | Lista plików, paginacja `after`, `limit`, opcjonalnie `purpose` |
| `GET /v1/files/{id}` | Metadane pliku |
| `GET /v1/files/{id}/content` | Strumień zawartości |
| `DELETE /v1/files/{id}` | Logiczne usunięcie; aktywny input przypięty przez batch daje 409 |
| `POST /v1/batches` | Utworzenie zlecenia |
| `GET /v1/batches` | Lista batchy właściciela |
| `GET /v1/batches/{id}` | Status, liczniki, identyfikatory wyników |
| `POST /v1/batches/{id}/cancel` | Rozpoczęcie anulowania |
| `GET /health/batches` | Gotowość zależności, katalog, profile i wykonanie; bez danych rekordów |

Główna ścieżka statusów to `validating → in_progress → finalizing → completed`.
Błąd walidacji kończy batch jako `failed`. Anulowanie przechodzi przez
`cancelling` do `cancelled`. Po terminie batch kończy się jako `expired`.
Anulowanie nie cofa opublikowanych wyników i nie przerywa kernela w połowie.
Wyniki zaakceptowane przed zamknięciem batcha pozostają dostępne.

`output_file_id` wskazuje poprawne odpowiedzi, a `error_file_id` błędy rekordów.
Oba wygenerowane pliki mają `purpose=batch_output`. Kojarz wyniki po
`custom_id`, bez zakładania kolejności wierszy. `response.body` zachowuje natywną
odpowiedź `/v1/basal`. Wynik pojedynczego rekordu może być błędem przy poprawnie
ukończonym batchu.

Limity domyślne: plik 200 000 000 bajtów, 50 000 rekordów i 64 MiB na wiersz
wyniku; konfiguracja może ograniczyć wejście. Dodatkowo obowiązują limity planowania,
kształtu i pamięci modelu. Nadmiarowy wynik staje się `result_too_large`.
Przekroczenie quota lub współbieżności daje 429 z `Retry-After`, a niedostępna
usługa batchy daje 503. Nie stosuj nieograniczonych ponowień.

Multipart ma osobny limit bufora `batches.host.upload_buffer_bytes` (domyślnie
1 MiB) oraz 64 KiB narzutu na framing, nagłówki i epilogue. Limity dotyczą
rzeczywiście odebranych bajtów także bez `Content-Length`. Pojedyncza ramka
transportu większa niż bufor daje 413, nawet jeżeli zawiera poprawne multipart;
bardzo mały bufor może więc odrzucić poprawny upload. Deadline
`batches.host.upload_max_duration_s` obejmuje całe odczytanie żądania, w tym
oczekiwanie na pierwszą część i dane po ostatniej granicy. Przekroczenie daje
408 `upload_timeout`; klient powinien rozpocząć nowy upload.

Timeout lub przerwanie zwalnia slot strumienia. Rezerwacja porzuconego uploadu
pozostaje w quota do bezpiecznego GC po okresie ochronnym I/O. Pobieranie ma
osobny deadline `batches.retention.download_max_duration_s`, który zamyka reader
i oddaje slot również wtedy, gdy klient przestał odbierać dane. Przy wyłączaniu
serwera pozostałe połączenia HTTP są zamykane po `batches.leases.shutdown_grace_ms`.

Pliki domyślnie wygasają po 30 dniach. `expires_after` uploadu oraz
`output_expires_after` zlecenia przyjmują `anchor=created_at` i czas
3600–2 592 000 sekund. W multipart pola `expires_after[anchor]` i
`expires_after[seconds]` umieść przed częścią `file`; `purpose` może być przed
nią lub po niej. Fizyczne GC czeka na okres ochronny pobrań i writerów.

## Magazyny i produkcja

| Backend | Zastosowanie i wymagania |
|---|---|
| SQLite | Jeden lokalny proces i trwały lokalny dysk; profil `local` |
| PostgreSQL | Wspólna kolejka produkcyjna, transakcje i blokady wierszy, zweryfikowany TLS |
| SQL Server (`mssql`) | Natywny TDS/Tiberius; operacje na SQL Server 2022, RCSI OFF/ON i Required TLS z weryfikacją CA/hosta sprawdzone w opisanym zakresie |
| MongoDB | Replica set lub obsługiwany klaster z transakcjami; standalone jest odrzucany; zweryfikowany TLS |

Nowe wybieranie kandydatów używa `$lookup` wewnątrz transakcji. Dla topologii
sharded wymagamy MongoDB ≥ 8.0; starszy `mongos` jest odrzucany przy połączeniu.
To ograniczenie [transakcyjnego `$lookup`](https://www.mongodb.com/docs/v8.0/reference/operator/aggregation/lookup/),
nie dowód kwalifikacji sharded. Przebieg na takim klastrze pozostaje do wykonania.

Wybierasz jeden backend metadanych oraz filesystem lub S3 dla artefaktów.
Zmiana silnika nie migruje danych między bazami. Wszystkie procesy wspólnej
kolejki muszą widzieć te same artefakty, katalog i politykę quota. Filesystem
w takim wdrożeniu wymaga współdzielonego trwałego wolumenu z właściwą semantyką
operacji plikowych. Dla S3 skonfiguruj również lifecycle usuwający porzucone
uploady multipart.

Profil `production` wymaga sekcji głównej `database:` z bazą zdalną oraz jawnych
artefaktów, przestrzeni danych i quota w `batches:`. Profil `local` wymaga SQLite.
Szablony ze wstrzymanym przyjmowaniem i wykonaniem:
[PostgreSQL](../examples/batches/postgresql.yml),
[SQL Server](../examples/batches/mssql.yml),
[MongoDB](../examples/batches/mongodb.yml). Podmień ścieżkę modelu, źródła sekretów,
certyfikat, bucket i właścicieli; żaden szablon nie nadaje kwalifikacji GPU.
Backend, poświadczenia runtime i migracji, TLS, pula oraz `auto_migrate` są
w sekcji głównej `database:` ([DATABASE.md](DATABASE.md)), a nie w `batches:`.
Dawne `batches.storage` i `batches.auto_migrate` są odrzucane z komunikatem
wskazującym nową sekcję; tabela przeniesienia pól jest w DATABASE.md.
Serwer nie przełącza się awaryjnie na SQLite. Zdalne bazy wymagają
`tls.mode: verify_full`; parametry URI/ADO.NET nie mogą wyłączyć weryfikacji.

`database.auto_migrate` domyślnie wynosi `true` dla SQLite i `false` dla baz
zdalnych. `true` migruje schemat przed udostępnieniem magazynu workerom
batchowym; dla bazy zdalnej wymaga osobnego konta migracyjnego, po czym serwer
otwiera zwykłe połączenie kontem runtime. `false` wymaga istniejącego,
aktualnego schematu i nie tworzy pliku SQLite ani nie uruchamia DDL.
Przykład dla pojedynczej instancji produkcyjnej:

```yaml
database:
  backend: postgresql
  auto_migrate: true
  connection_string_env: BASAL_DATABASE_URL
  migration_connection_string_env: BASAL_DATABASE_MIGRATION_URL
  tls: {mode: verify_full}
batches:
  enabled: true
  profile: production
  # artifacts, owner i quota jak w szablonie wybranego backendu
```

Przygotowanie produkcyjnej bazy batchy obejmuje schemat oraz zapis limitów
klientów i katalogu modeli. Wykonuje je jedno polecenie
`basal batches migrate --config serve.yml --catalog batch-catalog.json`.
Szczegółowa kolejność jest w sekcji [Przygotowanie i uruchomienie batchy](#przygotowanie-i-uruchomienie-batchy).
Lokalny start `basal serve --batches` wykonuje inicjalizację automatycznie.

Dla wielu replik ustaw `auto_migrate: false` i wykonuj migracje jednym zadaniem
wdrożeniowym przed uruchomieniem nowych replik. PostgreSQL i SQL Server mają
blokady synchronizujące migratorów, lecz nie chronią one starych procesów API
przed niezgodną zmianą schematu. MongoDB używa wznawialnych operacji DDL;
koordynuj jeden migrator. Automatyczny tryb nie gwarantuje zgodności z działającą
starszą wersją serwera. Przy błędzie schematu moduł batchy pozostaje niedostępny;
standardowa inferencja nadal działa. Po ręcznej naprawie schematu zrestartuj API.
Ręczne `basal database migrate` działa także przy `auto_migrate: false`.
Przy wyłączonym Batch API i bez sekcji `database:` serwer nie otwiera bazy.

`batches.accepting: false` wstrzymuje nowe uploady i tworzenie batchy.
`batches.processing: false` wstrzymuje nowe claimy discovery, walidacji
i wykonania. Obie flagi domyślnie mają wartość `true` i są odczytywane przy
starcie. Odczyt, pobieranie i anulowanie pozostają dostępne; przyjęte claimy
kończą swoją pracę. Wygaśnięcia, finalizacja, rozliczenia i GC nadal działają.
Pozwala to zatrzymać przyjęcia i opróżnić kolejkę przed zmianą katalogu.

Jeden osobny worker eksportu na proces składa pliki wynikowe, dzięki czemu
długi eksport nie zatrzymuje obsługi wygaśnięć, anulowań i rozliczeń innych
batchy. Terminalne ledgery i metadane są usuwane porcjami do 512 wierszy na
transakcję; klucz utworzenia zlecenia pozostaje do końca retencji metadanych.
Limit porcji nie gwarantuje czasu zapytania ani kosztu skanowania bazy.

SQL Server dodaje `tls.ca_file` do systemowych korzeni zaufania, zgodnie
z ograniczeniem Tiberius. PostgreSQL z podanym plikiem CA używa tego CA jako
magazynu zaufania. Nazwa hosta jest sprawdzana w obu przypadkach.
Nie używaj `TrustServerCertificate` jako obejścia konfiguracji certyfikatów.

### Wspólna przestrzeń danych i aktualizacja z 0.1.12

`batches.owner` wybiera jedną przestrzeń przechowywania dla wszystkich żądań tego serwera;
domyślnie `local`. To nazwa w bazie i polityce quota, nie tożsamość klienta ani uprawnienie.
Musi występować w `batches.quota.owners`; lokalne limity powstają automatycznie dla wybranej nazwy.
Repliki obsługujące ten sam zestaw zasobów powinny mieć tę samą wartość.
Klient nie może zmienić tej przestrzeni nagłówkiem ani polem żądania.

Po aktualizacji usuń sekcję `batches.auth` z konfiguracji: jest teraz nieznanym polem.
Nie ma komendy `basal batches token` ani pliku tokenu tworzonego przy starcie.
Stare pliki tokenów pozostają nietknięte i nie są odczytywane.
Dla dotychczasowego lokalnego wdrożenia zachowaj `owner: local` (wartość domyślna).
Dla istniejącej produkcyjnej bazy ustaw `owner` na dotychczasową nazwę właściciela, np. `acme`;
w ten sposób zachowasz dostęp do jego plików i batchy bez migracji danych.
Dane pozostałych przestrzeni pozostają w bazie; nie są automatycznie łączone ani usuwane.
Nagłówek `Authorization` nie jest sprawdzany i nie wybiera zasobów.

### Przygotowanie i uruchomienie batchy

Lokalnie wystarczy `basal serve --batches`: baza, domyślne limity klienta
i katalog modeli są inicjalizowane automatycznie.

Przy wdrożeniu produkcyjnym postępuj w tej kolejności:

1. Przygotuj `serve.yml`: bazę i poświadczenia, artefakty, przestrzeń `batches.owner`,
   limity `batches.quota` oraz modele i kwalifikowane profile wykonania.
   Przygotuj też `batch-catalog.json` z nazwami i fingerprintami modeli
   tego wdrożenia, zgodnie z opisem poniżej. Dla wielu replik ustaw
   `database.auto_migrate: false`.
2. Jednym zadaniem wdrożeniowym przygotuj bazę i zapisz konfigurację batchy:

   ```sh
   basal batches migrate --config serve.yml --catalog batch-catalog.json
   ```

   Polecenie kolejno aktualizuje schemat, zapisuje wspólne limity klientów
   z `batches.quota` i rejestruje katalog modeli. Nie ładuje modeli ani GPU.
   Przy aktualizacji istniejącego wdrożenia najpierw wykonaj backup i wstrzymaj
   przyjmowanie nowych batchy, zgodnie z procedurą wdrożenia.
3. Uruchom serwer i sprawdź magazyn:

   ```sh
   basal serve --config serve.yml
   basal batches doctor --config serve.yml --json
   ```

   `serve` działa w osobnym procesie lub pod nadzorem systemu usług;
   `doctor` uruchom w drugim terminalu albo kolejnym kroku wdrożenia.
   Szablony mają `accepting: false` i `processing: false`; włącz przyjęcia
   i przetwarzanie po sprawdzeniu konfiguracji oraz kwalifikacji GPU.

### Cel `batches install` i osobne operacje administracyjne

`basal batches install` zapisuje **konfigurację batchy w istniejącej bazie**:
limity miejsca, liczby plików i batchy dla klientów oraz, z `--catalog`, katalog
modeli dopuszczonych do wykonania. Wszystkie repliki korzystają z tych samych
zapisanych limitów i rewizji katalogu. Polecenie nie pobiera wag ani nie ładuje
modeli; nazwa `install` dotyczy rekordów konfiguracji.

W zwykłym przygotowaniu wdrożenia ten krok jest już częścią `batches migrate`.
Osobne `install` służy do późniejszej aktualizacji limitów lub katalogu bez
migracji schematu. Przy zmianie limitów wstrzymaj przyjmowanie nowych batchy;
nowa rewizja katalogu dotyczy nowych batchy, a przyjęte zachowują poprzednią.

| Polecenie | Cel |
|---|---|
| `basal batches migrate --config serve.yml --catalog batch-catalog.json` | pełne przygotowanie bazy dla batchy: schemat, limity klientów i katalog |
| `basal database migrate --config serve.yml` | tylko aktualizacja schematu wspólnej bazy |
| `basal batches install --config serve.yml` | zapis zmienionych limitów klientów; z `--catalog FILE` także katalogu |

Gdy wdrożenie koordynuje schemat bazy i konfigurację modułów w osobnych
krokach, odpowiednikiem jednego `batches migrate` są kolejno:

```sh
basal database migrate --config serve.yml                         # sam schemat
basal batches install --config serve.yml --catalog batch-catalog.json
basal batches doctor --config serve.yml --json
```

Operacje administracyjne nie wymagają sekcji `models`. Migracja i zapis
konfiguracji używają konta migracyjnego. `install` wymaga aktualnego schematu
i nie wykonuje DDL; niezmienione limity i katalog zachowują obecną rewizję.
`batches migrate` działa też dla lokalnego pliku domyślnego bez sekcji
`database:` (np. z `--batch-data-dir`). `doctor` łączy się kontem runtime
i nie wykonuje DDL.

Katalog JSON zawiera `version: 1`, `default_model` i `models` z polami `name`,
`fingerprint`, `planner_version`. Fingerprint podaje start serwera oraz health.
Obejmuje zawartość modelu, ustawienia obliczeń i granice admission; numer GPU
nie jest tożsamością modelu. Nowy katalog tworzy rewizję. Przyjęte batche
pozostają przypięte do poprzedniej rewizji; utrzymuj zgodnych workerów do ich
zakończenia lub świadomego wygaśnięcia.

Wersja planera `basal-planner.3` rozróżnia również skracanie kontekstu i jego usage.
Fingerprint obejmuje efektywne `orders`, layout (także override), SHA-256
wybranej kalibracji lub stan off/brak, strict, auto tau/band oraz
`context_overflow` i `truncate_head` (także przy polityce refuse).
Zachowuje też hashowanie `basal.json` i plików `CALIBRATION*.json` w katalogu
modelu. Pełna ścieżka jawnego pliku, lokalna tożsamość i ostrzeżenie o niej
nie zastępują hasha treści i nie wchodzą do odcisku; numer urządzenia również nie.
Planner CPU dostaje migawkę manifestu i opcji faktycznie załadowanego silnika;
fork tła dziedziczy te same opcje. Bounded evidence przy niekanonicznym layoutcie
jest odrzucane jako `unmodelled`, ponieważ jego budżet nie jest zamodelowany.
Wybór online `systemone_dialect` nie zmienia kontraktu trwałych rekordów batchy.
Procesowy `log_decisions` nie jest częścią odcisku i nie zapisuje rekordów
batchy w tle. Planer CPU stosuje tę samą politykę skracania co online;
przejściowe kopie/retokenizacja, oryginalny stan kontynuacji i wynikowe usage
pozostają objęte budżetami hosta. Przekroczenie budżetu nadal odmawia rekordu,
nawet gdy krótszy prompt zmieściłby się w kontekście modelu.

## Wykonanie i kwalifikacja GPU

Tylko jeden lokalny scheduler może zarządzać danym fizycznym GPU. Lease bazy
chroni zapisy wyników, ale nie zatrzymuje kerneli starego procesu. Deployment
musi odciąć starego właściciela urządzenia przed uruchomieniem następnego.

Kolejka korzysta z DRR najpierw między właścicielami, potem między ich batchami.
Rezerwuje koszt przed claimem, a po wykonaniu rozlicza zmierzony czas posiadania
bramki urządzenia. Jednostką kosztu są nanosekundy; `cost_ns_per_token` jest
zmierzonym współczynnikiem estymacji, nie ceną. Czas oczekiwania za online nie
jest naliczany jako zajęcie urządzenia. Pomiar obejmuje przygotowanie i drain
wewnątrz bramki, więc nie oznacza sprzętowego czasu aktywności kerneli.
Przerwana praca ma osobne liczniki; po awarii niepotwierdzony koszt rozliczany
jest konserwatywnie przez zarezerwowaną estymację.

Przerwane rekordy otrzymują trwałe odroczenie ponowienia od 1 do 30 sekund.
Nie zużywa ono limitu rzeczywistych błędów wykonania. Lokalny, współdzielony
limit pamięci powoduje rozbicie grupy, a dla pojedynczego rekordu odroczenie.
Przy utrzymującym się braku zasobów rekord może wygasnąć po 24 godzinach.
Przejściowy błąd odczytu telemetrii również odracza pracę.

Zadania z dalszymi rundami, np. `extract`, wymagają konserwatywnego boundu
obejmującego przyszłe prompty aż do limitów kontekstu modelu. Profil z mniejszym
limitem forwardu może więc odrzucić także krótki rekord obsługiwany online.
Samo dopasowanie początkowego promptu do limitu nie kwalifikuje takiego zadania.

Planowanie CPU ma wydzielone wątki i limit `R` na plan. Pula `P` obejmuje
plany wykonania oraz bufory wyników; claim następuje dopiero po rezerwacji hosta.
Mikroserie mają ograniczone tokeny i liczbę rekordów. Domyślnie wykonanie modelu
w tle jest wyłączone. Grupowanie wymaga osobnego dowodu niezmienności wyniku
względem składu i kolejności mikroserii. Zdrowy błąd grupy prowadzi do zapisania
`requires_solo` i diagnozy osobno, bez nieograniczonych ponowień grupy.

Do włączenia `models[].batch_profile.background: true` wymagane są:

- `evidence`: raport kwalifikacji dla tej konfiguracji modelu i backendu;
- `device_bytes`: zmierzony budżet pracy tła na urządzenie;
- `online_reserve_bytes`: zmierzony zapas pozostawiany online;
- `allocation_granularity_bytes`: ziarnistość zaokrąglania alokacji;
- `cost_ns_per_token`: zmierzony koszt, dodatnia liczba całkowita ≤ 1 000 000 000;
- `composition: solo` albo zakwalifikowane `group` z `max_group_requests`.

Wartości nie są automatycznie wyliczane z wolnego VRAM. Telemetria pamięci
może odmówić startu, ale nie zastępuje kwalifikacji. `evidence` jest deklaracją
operatora: serwer sprawdza konfigurację, a nie treść ani wiarygodność raportu.
Rzeczywiste próby muszą
obejmować zgodność bitową z online solo, granice pamięci, media w deklarowanym
zakresie, przerwania, anulowanie, shutdown i obciążenie mieszane z pomiarem
opóźnień online. Procedura wyników: [BENCHMARKS.md](BENCHMARKS.md),
[MEASUREMENT_ARTIFACTS.md](MEASUREMENT_ARTIFACTS.md).

Bezpieczne wycofanie zachowuje bazę i artefakty, zatrzymuje przyjęcia oraz
wykonanie, a następnie czeka na zakończenie pracy i zwolnienie lease.
Nie uruchamiaj starszej binarki do zapisu schematu nowszej wersji.
Przed migracją wykonaj spójny backup bazy i artefaktów oraz sprawdź odtworzenie.
