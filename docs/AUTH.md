# Klucze API

Klucze API są domyślnie wyłączone. Bez sekcji `auth:` serwer działa jak dotąd:
nie sprawdza nagłówka `Authorization` i nie otwiera bazy z powodu kluczy.

Po włączeniu każda trasa poza `/health` i `/health/batches` wymaga nagłówka
`Authorization: Bearer <token>`, także na adresie loopback. Klucze są
przechowywane w bazie z sekcji [`database:`](DATABASE.md) (SQLite, PostgreSQL,
SQL Server albo MongoDB). Operator zarządza nimi poleceniem `basal keys`;
serwer klucze tylko odczytuje.

## Konfiguracja

```yaml
database:
  backend: postgresql                # sqlite | postgresql | mssql | mongodb
  connection_string_env: BASAL_DATABASE_URL
  migration_connection_string_env: BASAL_DATABASE_MIGRATION_URL
  tls: {mode: verify_full}

auth:
  enabled: true                      # domyślnie false
  protect_metrics: false             # true: GET /metrics wymaga zakresu metrics
```

`basal serve --auth` włącza klucze tak samo jak `auth.enabled: true`.
Klucze używają tej samej bazy co Batch API. Bez sekcji `database:` mogą użyć
lokalnego pliku SQLite Batch API (`basal serve --batches --auth`). Bez bazy serwer
odmawia startu. Nie ma trybu awaryjnego, który przepuszczałby żądania bez klucza.

## Przygotowanie i zarządzanie

1. Przygotuj schemat bazy (wersja 3 zawiera tabelę `api_keys`):

   ```sh
   basal database migrate --config serve.yml
   ```

   SQLite z `auto_migrate: true` (domyślnie) tworzy schemat automatycznie.
   Konto runtime potrzebuje uprawnień `SELECT`, `INSERT` i `UPDATE` do `api_keys`.

2. Utwórz klucz dla klienta:

   ```sh
   basal keys create --config serve.yml --owner acme --name "Zespół A" \
     --scopes inference,batches --expires-in 90d
   ```

   Token trafia na stdout tylko raz. Komunikaty trafiają na stderr. Zapisz token
   w menedżerze sekretów klienta. Baza przechowuje wyłącznie SHA-256 tokenu.
   `--json` wypisuje jeden obiekt JSON z tokenem i metadanymi.

3. Pozostałe operacje:

   | Polecenie | Działanie |
   |---|---|
   | `basal keys list [--owner O] [--all] [--json]` | lista kluczy bez tokenów; `--all` dodaje odwołane |
   | `basal keys revoke KEY_ID` | odwołanie; ponowne odwołanie zachowuje pierwszy czas |
   | `basal keys rotate KEY_ID [--expire-old-in 24h] [--expires-in 90d]` | nowy klucz z tym samym ownerem, nazwą i zakresami |

   `KEY_ID` to 16 znaków szesnastkowych po prefiksie `bsk_`. Rotacja nie odwołuje
   starego klucza. Z `--expire-old-in` stary klucz wygasa po podanym czasie.
   Bez tej opcji pozostaje ważny do `basal keys revoke`.

`basal keys` wybiera bazę tak jak `basal batches`: sekcja `database:` z `--config`,
`BASAL_CONFIG` albo `basal-serve.yml`, a bez niej lokalny plik SQLite Batch API.
Polecenie nie ładuje modelu ani GPU.

## Token, zakresy i owner

Token ma postać `bsk_<key_id>_<secret>`: 16 znaków szesnastkowych identyfikatora
i 43 znaki base64url z 256 losowych bitów. Prefiks pozwala skanerom sekretów
rozpoznać wyciek. Serwer porównuje SHA-256 całego tokenu w stałym czasie.

| Zakres | Trasy |
|---|---|
| `inference` | `/v1/basal`, `/v1/systemone`, `/v1/models` |
| `batches` | `/v1/files…`, `/v1/batches…` |
| `mcp` | `/mcp`; narzędzia MCP wykonują inferencję, więc ten zakres też do niej uprawnia |
| `metrics` | `/metrics`, tylko z `auth.protect_metrics: true` |

Domyślne zakresy nowego klucza to `inference,batches`. Inne ścieżki wymagają
dowolnego ważnego klucza, więc bez klucza nie da się sprawdzić, które trasy istnieją.

`--owner` wskazuje przestrzeń danych Batch API: pliki, batche i quota. Klucz
zastępuje `batches.owner`, więc klienci z różnymi ownerami nie widzą swoich
zasobów. Owner musi mieć limity w `batches.quota.owners`; w przeciwnym razie
jego żądania Batch/Files są odrzucane z powodu braku quota. `basal keys create`
ostrzega, gdy zainstalowana polityka quota nie zawiera ownera.

## Odpowiedzi

| Sytuacja | Odpowiedź |
|---|---|
| brak, zły format, nieznany, odwołany lub wygasły klucz | 401, `invalid_api_key`, `WWW-Authenticate: Bearer` |
| ważny klucz bez zakresu trasy | 403, `insufficient_scope` |
| baza kluczy niedostępna | 503 z `Retry-After` |

Wszystkie przyczyny 401 dają tę samą odpowiedź, więc nie ujawniają stanu klucza.

## Opóźnienia i obciążenie bazy

- Proces odczytuje dany klucz z bazy najwyżej raz na 30 s. Odwołanie lub wygaśnięcie
  działa więc na każdym serwerze najpóźniej po 30 s.
- Gdy baza jest niedostępna dłużej niż ten czas, żądania kończą się kodem 503.
  Serwer nie przepuszcza ich bez sprawdzenia klucza.
- Czas wygaśnięcia jest liczony zegarem bazy.
- `last_used_at` jest zapisywane w tle najwyżej raz na 60 s na klucz. Przy pełnej
  kolejce zapis jest pomijany; kolejne użycie po 60 s zapisze go ponownie.
- Najwyżej 16 równoczesnych odczytów kluczy na proces. Żądanie czeka na odczyt
  najwyżej 5 s, potem dostaje 503.
- Cache mieści do 10 000 identyfikatorów, także nieznanych. Gdy jest pełny, najpierw
  usuwa nieznane i nieaktualne wpisy. Zalew losowych identyfikatorów nie wypycha więc
  aktualnych wpisów istniejących kluczy, ale każdy nowy identyfikator to jeden odczyt bazy.

## Klienci

- `basal client` i most `basal mcp` czytają klucz ze zmiennej `BASAL_API_KEY`
  (inną nazwę wskazuje `--key-env`). Token nie jest argumentem wiersza poleceń.
- SDK przyjmują klucz w konstruktorze; szczegóły w [SDK.md](SDK.md).
- Przez HTTP bez TLS token jest widoczny w sieci. Wystawiaj serwer za proxy
  z TLS albo w zaufanej sieci.

## Ograniczenia

- Brak HTTP API do zarządzania kluczami; tylko `basal keys` z dostępem do bazy.
- Brak limitów liczby żądań na klucz i limitu nieudanych prób z jednego adresu.
- `/health/batches` pozostaje publiczne i pokazuje stan magazynu batchy.
- Rotacja nie jest atomowa: gdy ustawienie wygaśnięcia starego klucza się nie powiedzie,
  nowy token jest już wypisany, a stary klucz pozostaje ważny.
- Odmowy uwierzytelnienia nie są osobno logowane. Z `access_log` widać je jako
  odpowiedzi 401/403.
