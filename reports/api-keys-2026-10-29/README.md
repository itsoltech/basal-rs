# Użytkownicy i klucze API w bazie — 2026-10-29

## Cel i zakres

Opcjonalny moduł `auth:` dla wdrożeń `basal serve`: użytkownicy (`api_users`)
i ich klucze (`api_keys`) w bazie z sekcji `database:` (SQLite, PostgreSQL,
SQL Server, MongoDB). Zarządzanie przez `basal users` i `basal keys`. Klucze
mają zakresy, wygasanie i `last_used_at`. Wszystkie klucze użytkownika
współdzielą jego dane Batch API. Klienci: `basal client`/`basal mcp`
(`BASAL_API_KEY`), cztery SDK (Python/npm/C# 0.7.0, Rust 0.8.0), kontrakt
Batch 0.3.0. Opis zachowania: [docs/AUTH.md](../../docs/AUTH.md).

Moduł jest domyślnie wyłączony. Bez `auth:` serwer nie wymaga bazy i nie
sprawdza nagłówka `Authorization`. Wcześniejsze tokeny plikowe usunięto
w 0.1.13 ([raport](../remove-api-key-2026-10-21/README.md)); nowy moduł ich
nie przywraca. Schemat bazy ma teraz wersję 3. Starsze binarki odrzucają go jako nowszy.

## Środowisko i metoda

Linux x86_64, RTX 4060, build release z CUDA (`--features cuda`, cap 89),
model `Remek/basal-2.0-mini` z lokalnego cache HF. Serwer na loopback,
`batches.profile: local`, SQLite. Ręczne żądania `curl` i polecenia CLI;
bez testów automatycznych (zgodnie z AGENTS.md). Nie zmieniono obliczeń
modelu, więc nie wykonano porównania numerycznego ani pomiaru wydajności.

## Wyniki

- PASS, CLI na SQLite: tworzenie, lista, rotacja z `--expire-old-in` i odwołanie
  kluczy; idempotentne odwołanie i błędy dla złego zakresu, czasu, id i użytkownika.
  Tworzenie, duplikat, wyłączanie i włączanie użytkowników. Klucz dla
  nieistniejącego lub wyłączonego użytkownika jest odrzucany.
- PASS, lokalna baza serwera (izolowany `BASAL_HOME`):
  - Bez sekcji `database:` polecenia i `basal serve --auth` (bez `--batches`) używają
    `<katalog danych>/db/basal.sqlite3`, katalog 0700.
  - Istniejący plik `<katalog batchy>/batches.sqlite3` z wcześniejszych wersji
    pozostaje w użyciu, z notą o jego położeniu.
  - `basal serve` bez `auth` i bez batchy nie tworzy żadnej bazy, a `/v1/models` daje 200 bez klucza.
- PASS, migracja: baza SQLite w wersji 2 jest odrzucana przez `basal keys`
  z błędem schematu. Po `basal database migrate` ma wersję 3 i działa.
- PASS, HTTP z `auth.enabled`:
  - 401 `invalid_api_key` z `WWW-Authenticate: Bearer` dla braku klucza, złego
    sekretu, dwóch nagłówków i nieznanej trasy.
  - 403 `insufficient_scope` dla klucza bez zakresu (`/v1/files`, `/mcp`, `/v1/decisions`).
  - 200 dla `/health` i `/health/batches` bez klucza.
- PASS, dane Batch:
  - Dwa klucze jednego użytkownika widzą ten sam plik.
  - Inny użytkownik dostaje 404.
  - Użytkownik bez limitów quota dostaje 403 `quota_not_configured`.
- PASS, opóźnienie: odwołany klucz i klucze wyłączonego użytkownika działały
  jeszcze z cache. Po 31 s dostały 401. Po włączeniu użytkownika i 31 s znów 200.
  `last_used_at` jest zapisywane.
- PASS, kontrole: `cargo fmt --check`, Clippy `-p basal-cli --all-targets -D warnings`,
  rustdoc z prywatnymi elementami `-D warnings`, `cargo deny --all-features`.
  SDK: build/lint Rust, npm build/pack, Python wheel/sdist, `dotnet build/pack`
  oraz walidacja JSON kontraktu.

## Ograniczenia (GAP)

- PostgreSQL, SQL Server i MongoDB: tylko kompilacja i przegląd kodu. Lokalnie
  nie było tych baz ani dostępu do Dockera. Migracje i zapytania tych silników
  wymagają sprawdzenia na prawdziwych instancjach przed wdrożeniem.
- Clippy i rustdoc uruchomiono w profilu debug dla `basal-cli`, nie dla całego
  workspace w release ani z funkcjami CUDA/Intel. Pełny zakres zapewnia CI.
- SDK nie sprawdzono na serwerze z kluczami. C# zbudowano lokalnym .NET 10.0.300
  (CI: 10.0.401).
- `tools/openai-decisions/verify.py` nie wysyła klucza. Działa tylko z serwerem bez `auth`.
- Brak limitów żądań na klucz, HTTP API do zarządzania i usuwania użytkowników.

## Dane

Obserwacje to wyniki poleceń w tym raporcie; nie zachowano osobnych plików
danych, baz ani tokenów. Do odtworzenia potrzebny jest build z CUDA, model
z cache HF i kroki z docs/AUTH.md.
