# Konta basal-ui na PostgreSQL, SQL Server i MongoDB — 2026-10-30

## Cel i zakres

Uzupełnienie [raportu basal-ui](../basal-ui-2026-10-30/README.md), w którym
migracja 0004 i operacje kont były sprawdzone tylko na SQLite. Ten przebieg
sprawdza je na trzech bazach zdalnych:

- migrację schematu 3 do 4 na danych z poprzedniej wersji;
- użytkowników, logowanie, sesje, zmianę hasła, blokowanie i usuwanie;
- klucze API: tworzenie, zakresy i odwołanie.

Opis zachowania: [docs/UI.md](../../docs/UI.md), [docs/AUTH.md](../../docs/AUTH.md).
Kod nie zmienił się w tym przebiegu.

## Środowisko i metoda

Linux x86_64, RTX 4060, build release z CUDA tej gałęzi, model
`Remek/basal-2.0-mini` z lokalnego cache HF. Bazy uruchomiono lokalnie
w Dockerze:

- PostgreSQL 17, SQL Server 2022, MongoDB 7.0 (replica set z jednym węzłem);
- TLS z prywatnym CA i certyfikatem z SAN `localhost`, połączenia
  `tls.mode: verify_full`;
- PostgreSQL: osobne konto runtime z uprawnieniami dokładnie z docs/AUTH.md
  (schemat 3, a po migracji schemat 4);
- SQL Server i MongoDB: jedno konto administracyjne w obu rolach; MongoDB bez
  uwierzytelniania (tylko loopback i TLS).

Kroki dla każdej bazy:

1. Binarka z `main` (schemat 3) migruje bazę, tworzy użytkownika `legacy`
   i jego klucz `inference`.
2. Binarka tej gałęzi wykonuje `basal database migrate` dwa razy, potem
   `basal users create admin --admin --password-stdin`.
3. `basal serve --auth --ui` i skrypt HTTP (Python, stdlib) z 37 kontrolami API
   kont `/ui/api` i tras `/v1`.
4. CLI: `users password`, `users password --clear`, `users delete` przed
   upływem 30 s i po nim, `keys list`, `users list`.
5. Odmowa usunięcia przy danych Batch: wiersz `batch_quota` użytkownika z
   `files = 1` wpisany bezpośrednio do bazy, potem wyzerowany.

Zgodnie z AGENTS.md skrypt nie trafił do repozytorium ani do CI.

## Wyniki

| Kontrola | PostgreSQL | SQL Server | MongoDB |
|---|---|---|---|
| migracja 3 → 4, powtórzona migracja bez zmian | PASS | PASS | PASS |
| binarka ze schematem 3 odrzuca schemat 4 (`TooNew`) | PASS | PASS | PASS |
| użytkownik ze schematu 3: rola `user`, bez hasła; jego klucz działa | PASS | PASS | PASS |
| 37 kontroli HTTP kont i kluczy | 37/37 PASS | 37/37 PASS | 37/37 PASS |
| CLI: hasło, usunięcie dopiero po blokadzie starszej niż 30 s | PASS | PASS | PASS |
| odmowa usunięcia przy danych Batch, usunięcie po ich wyzerowaniu | PASS | PASS | PASS |

37 kontroli HTTP obejmuje:

- logowanie:
  - złe hasło: 401 `invalid_credentials`;
  - sesja bez `X-Basal-UI`: 403;
  - wylogowanie czyści ciasteczko.
- konta:
  - administrator tworzy użytkownika; duplikat daje 409;
  - użytkownik na trasie administratora dostaje 403;
  - inferencja `/v1/systemone` sesją: 200;
  - administrator nie może zablokować własnego konta (409).
- klucze:
  - użytkownik tworzy klucz `inference` z wygasaniem; klucza `metrics` nie może utworzyć;
  - nowy klucz: `/v1/models` 200, `/v1/files` 403 `insufficient_scope`, `/ui/api/me` 401;
  - odwołanie przez użytkownika i przez administratora.
- hasła:
  - złe obecne hasło: 400 `wrong_password`;
  - zmiana hasła zachowuje bieżącą sesję i kończy pozostałe;
  - stare hasło przestaje działać;
  - reset przez administratora kończy sesje użytkownika.
- blokada i usunięcie:
  - blokada kończy sesję od razu i blokuje logowanie;
  - usunięcie przed 30 s: 409 `user_not_disabled`;
  - po odblokowaniu użytkownik znów się loguje;
  - usunięcie po 31 s kasuje konto i klucze, także odwołane;
  - ten sam identyfikator można utworzyć ponownie.

Uprawnienia konta runtime PostgreSQL opisane w docs/AUTH.md wystarczyły do
wszystkich operacji serwera i CLI, także do usuwania z blokadą wiersza quota.
Bez `SELECT` do polityki quota `basal users create` tylko ostrzega, zgodnie
z dokumentacją.

Problemy po stronie przygotowania testu, nie kodu:

- Binarka z `main` nie zna sekcji `ui:`; dla niej użyto pliku bez tej sekcji.
- SQL Server odrzuca connection string z `TrustServerCertificate` w każdej
  wartości; usunięto ten parametr, jak wymaga docs/DATABASE.md.
- MongoDB: replica set musi ogłaszać adres osiągalny z hosta. Pierwszy kontener
  ogłaszał port wewnętrzny 27017 przy mapowaniu na 17017, więc obie binarki
  dostawały `Unavailable`. Poprawiono to przez `mongod --port 17017`.

## Ograniczenia (GAP)

- SQL Server i MongoDB nie były sprawdzane na kontach z minimalnymi
  uprawnieniami.
- Dane Batch w teście usuwania to ręcznie ustawiony licznik quota. Prawdziwe
  pliki i batche oraz ich wygasanie nie były wykonywane w tym przebiegu.
- Tylko pojedyncze instancje baz na jednej maszynie. Nie sprawdzano wielu
  replik serwera, PgBouncera ani klastrów.
- Interfejs w przeglądarce na tych bazach nie był oglądany. Strony korzystają
  z tego samego API, sprawdzonego tu przez HTTP.

## Dane

Obserwacje to wyniki poleceń w tym raporcie. Nie zachowano baz, haseł ani
tokenów; kontenery i dane testowe usunięto po przebiegu. Do odtworzenia
potrzebne są obrazy baz z TLS jak wyżej, build z CUDA i kroki z docs/UI.md
oraz docs/DATABASE.md.
