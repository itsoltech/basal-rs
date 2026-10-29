# Baza danych serwera

Zwykła inferencja nie potrzebuje bazy danych. Bez sekcji `database:` i bez
włączonego konsumenta `basal serve` nie otwiera połączeń, nie tworzy plików bazy
i nie uruchamia migracji.

Sekcja główna `database:` opisuje jedną bazę procesu: backend, plik SQLite,
poświadczenia runtime i migracji, TLS, pulę oraz `auto_migrate`. Konsumentami są
[Batch i Files API](BATCHES.md) oraz [klucze API](AUTH.md). Ustawienia bazy nie należą do
sekcji `batches:`. Kolejne moduły serwera mogą korzystać z tej samej sekcji.

## Konfiguracja

```yaml
database:
  enabled: true                      # domyślnie true, gdy sekcja istnieje
  backend: postgresql                # sqlite | postgresql | mssql | mongodb
  auto_migrate: false                # domyślnie: sqlite true, bazy zdalne false
  connection_string_env: BASAL_DATABASE_URL            # albo connection_string_file
  migration_connection_string_env: BASAL_DATABASE_MIGRATION_URL
  tls:
    mode: verify_full
    ca_file: /run/secrets/basal-db-ca.pem
  pool:
    max_connections: 16
```

SQLite przyjmuje tylko `path` (ścieżka bezwzględna) oraz `enabled` i
`auto_migrate`:

```yaml
database:
  backend: sqlite
  path: /var/lib/basal/basal.sqlite3
```

| Pole | Zachowanie |
|---|---|
| sekcja pominięta | brak bazy; konsument może użyć własnego domyślnego pliku lokalnego (Batch: niżej) |
| `enabled: false` | brak bazy i brak dalszej walidacji pól; konsument wymagający bazy odmawia startu |
| `backend` | wymagany przy włączonej sekcji |
| `auto_migrate` | pominięte: `true` dla SQLite, `false` dla PostgreSQL, SQL Server i MongoDB |
| `connection_string_env` / `_file` | dokładnie jedno źródło konta runtime (bazy zdalne) |
| `migration_connection_string_env` / `_file` | osobne konto migracyjne; wymagane przy `auto_migrate: true` i do administracji |
| `tls.mode` | wymagane `verify_full` dla baz zdalnych; `ca_file` opcjonalny |
| `pool` | skończone limity puli i timeoutów (domyślnie 16 połączeń, statement 15 s, lock 5 s) |

Sekrety są czytane dopiero przy otwieraniu połączenia, z jednej zmiennej
środowiskowej albo jednego pliku (zwykły plik do 64 KiB, bez zapisu dla grupy
i innych). Nie trafiają do logów ani komunikatów błędów. Konto migracyjne musi
mieć inne źródło niż konto runtime. Konto runtime nigdy nie wykonuje DDL.
Parametry URI/ADO.NET nie mogą wyłączyć szyfrowania ani weryfikacji certyfikatu
i hosta. Nie ma przełączenia awaryjnego na SQLite ani na inną bazę.

## Migracje schematu

Ręczna migracja nie ładuje modeli ani GPU i nie wymaga sekcji `models`,
`batches`, tokenów ani artefaktów:

```sh
basal database migrate --config basal.yml
```

Polecenie zawsze używa konta migracyjnego, niezależnie od `auto_migrate`.
Tworzy brakujący plik SQLite (`0600`, katalog `0700`) lub schemat, stosuje
brakujące migracje i pozostawia aktualny schemat bez zmian. Nowszy lub
uszkodzony schemat jest odrzucany i nie jest modyfikowany. Polecenie przygotowuje
strukturę bazy. Przy wdrożeniu Batch API trzeba również zapisać w niej limity
klientów i katalog modeli. Pełne przygotowanie batchy wykonuje jedno polecenie:

```sh
basal batches migrate --config serve.yml --catalog batch-catalog.json
```

Obejmuje ono migrację schematu oraz zapis konfiguracji batchy. Kolejność
przygotowania plików, uruchomienia i sprawdzenia serwera opisuje
[instrukcja wdrożenia batchy](BATCHES.md#przygotowanie-i-uruchomienie-batchy).
Osobne `basal batches install` służy do aktualizacji limitów i katalogu
w już przygotowanej bazie albo do wdrożeń rozdzielających te kroki.

`auto_migrate: true` uruchamia tę samą migrację przy starcie serwera, najwyżej
raz z sukcesem na proces. Gdy z bazy korzysta Batch API, migracja kończy się przed
udostępnieniem magazynu jego workerom. Błąd migracji nie jest zapamiętywany:
po błędzie przejściowym (np. niedostępna baza) moduł batchy ponawia próbę
z backoffem i do tego czasu odpowiada 503. Błąd schematu lub konfiguracji
pozostawia moduł niedostępnym do naprawy i restartu. Gdy żaden
konsument nie jest włączony, serwer wykonuje jedną próbę w tle; zwykła
inferencja na nią nie czeka, a błąd trafia do logu. Klucze API (`auth.enabled`)
czekają na migrację przy starcie; błąd migracji lub połączenia przerywa start serwera.
Przy `auto_migrate: false`
serwer niczego nie tworzy, a połączenie runtime odrzuca brakujący lub nieaktualny
schemat.

PostgreSQL (advisory lock) i SQL Server (`sp_getapplock`) szeregują równoczesne
migratory. MongoDB wznawia przerwaną migrację, ale wymaga jednego migratora
naraz. Blokady nie chronią starszych procesów API przed niezgodnym schematem.
Dla wielu replik ustaw `auto_migrate: false` i migruj jednym zadaniem
wdrożeniowym przed uruchomieniem nowych replik. Przed migracją wykonaj spójny
backup.

## Granica odpowiedzialności

Moduł `crates/basal-cli/src/database/` posiada konfigurację, walidację, opcje
połączeń (TLS, pula, timeouty), wybór konta oraz moment i sposób migracji.
Jedynym zarejestrowanym schematem jest schemat magazynu batchy; od wersji 3
zawiera też tabelę (kolekcję MongoDB) `api_keys`. Jego tabele, ledger wersji,
sumy kontrolne i natywne migratory pozostają w adapterach `serve/batches/store`,
podobnie jak wszystkie zapytania domenowe. Starsza binarka odrzuca schemat 3 jako nowszy. Każdy konsument
otwiera własną, ograniczoną pulę z tych samych ustawień; serwer nie ma jednej
wspólnej puli połączeń dla wielu modułów.

## Migracja konfiguracji z `batches.storage`

`batches.storage` i `batches.auto_migrate` zostały usunięte. Plik, który je
zawiera, jest odrzucany z komunikatem wskazującym tę sekcję, także przy
`batches.enabled: false`. Przenieś pola bez zmiany nazw:

| Poprzednio | Teraz |
|---|---|
| `batches.storage.backend` | `database.backend` |
| `batches.storage.path` | `database.path` |
| `batches.storage.connection_string_env` / `_file` | `database.connection_string_env` / `_file` |
| `batches.storage.migration_connection_string_env` / `_file` | `database.migration_connection_string_env` / `_file` |
| `batches.storage.tls`, `batches.storage.pool` | `database.tls`, `database.pool` |
| `batches.auto_migrate` | `database.auto_migrate` |

Szablony w `examples/batches/` używają teraz zmiennych `BASAL_DATABASE_URL`
i `BASAL_DATABASE_MIGRATION_URL` zamiast `BASAL_BATCH_DB` i
`BASAL_BATCH_MIGRATION_DB`; nazwy zmiennych wybiera operator, więc istniejące
wdrożenie może zachować własne. `basal batches migrate` nadal wykonuje pełne
przygotowanie batchy. Wdrożenia rozdzielające schemat i konfigurację modułów
mogą używać kolejno `basal database migrate` oraz `basal batches install`.

Domyślne `auto_migrate` zależy teraz od backendu, a nie od profilu batchy. Dla
dotychczas dopuszczalnych par (`local` + SQLite, `production` + baza zdalna)
efektywna wartość domyślna się nie zmienia.
