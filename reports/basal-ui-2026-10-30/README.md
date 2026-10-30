# Interfejs w przeglądarce basal-ui — 2026-10-30

## Cel i zakres

Opcjonalny interfejs w przeglądarce dla `basal serve`, odpowiednik upstream
basal-ui (rkinas/basal#45) oparty na mechanizmach basal-rs: kluczach API
i użytkownikach w bazie (`auth:`), Batch/Files API i `/v1/systemone`.
Opis zachowania: [docs/UI.md](../../docs/UI.md), [docs/AUTH.md](../../docs/AUTH.md).

- `basal serve --ui` albo YAML `ui.enabled: true`; domyślnie wyłączone. Bez tej
  opcji trasa `/ui` nie istnieje i API zachowuje się jak dotąd.
- Strony to HTML, CSS i moduły JavaScript wkompilowane w binarkę
  (`crates/basal-cli/ui/`). Nie ma kroku budowania, zależności npm ani zasobów
  z innych domen. Odpowiedzi `/ui` mają ścisły CSP. Ikona pochodzi z repozytorium
  basal-rs-landing.
- Konta: schemat bazy 4 dodaje rolę (`user`/`admin`), hasło argon2id i tabelę
  `api_sessions` na SQLite, PostgreSQL, SQL Server i MongoDB.
- Sesja jest w ciasteczku HttpOnly/SameSite=Strict. Żądania z sesją wymagają
  nagłówka `X-Basal-UI: 1` (ochrona CSRF).
- Nowe API kont `/ui/api/...`: logowanie, własne klucze, zmiana hasła oraz
  administracja użytkownikami i kluczami.
- CLI: `basal users create --admin --password-stdin`, `basal users password`
  i `basal users delete`.
- Widoki:
  - Playground z kreatorem pytań `noul`, `choice`, `score`, `multi` i `act`
    oraz szablonami;
  - batche i pliki użytkownika;
  - własne klucze i konto;
  - administracja: użytkownicy, klucze API i stan systemu (`/health`,
    `/health/batches`, `/v1/models`, `/metrics`).

Nie zmieniono obliczeń modelu, więc nie wykonano porównania numerycznego ani
pomiaru wydajności.

## Środowisko i metoda

Linux x86_64, RTX 4060, build release z CUDA (`--features cuda`, cap 89), model
`Remek/basal-2.0-mini` z lokalnego cache HF. Serwer na loopback z `--auth`,
`--ui`, `--metrics`, Batch API w profilu `local` i SQLite w izolowanym
`BASAL_HOME`. Przeglądarka Chromium (podgląd T3 Code) przy 1440×900.

Na prośbę o testy w tej pracy dodano testy Rust `#[cfg(test)]` dla nowej logiki,
bez nowych frameworków i bez bramki w CI. Pozostałe zasady AGENTS.md pozostają
bez zmian.

## Wyniki

- PASS, `cargo test -p basal-cli --bin basal -- serve::ui serve::auth`: 19 testów.
  - Jednostkowe: hasła argon2id i polityka długości, format i odczyt ciasteczka
    sesji, nagłówek UI, limit nieudanych logowań, zakresy sesji, dopasowanie tras
    w `need()`, walidacja zapytań i ciał, konfiguracja `ui:`, wbudowane pliki
    i ich typy.
  - HTTP z prawdziwym listenerem, tymczasową bazą SQLite i middleware `auth`:
    - logowanie: wymagany nagłówek UI, ta sama odpowiedź 401 dla nieznanego
      użytkownika i złego hasła, wylogowanie, także przy zakończonej sesji;
    - sesja na `/v1/models` z nagłówkiem i 403 bez niego;
    - klucz API nie otwiera `/ui/api/me`;
    - użytkownik nie widzi ani nie odwołuje cudzego klucza, nie tworzy klucza
      `metrics` i nie wchodzi na trasy administratora;
    - zmiana hasła kończy inne sesje;
    - administrator: tworzenie użytkownika i klucza, duplikat 409, słabe hasło
      400; ochrona własnego konta przed blokadą, usunięciem i resetem hasła;
    - blokada kończy sesje od razu; usunięcie przed upływem 30 s daje 409;
      usunięcie w magazynie kasuje klucze;
    - 429 po 10 nieudanych logowaniach.
- PASS, ręcznie w przeglądarce z modelem na GPU:
  - logowanie administratora i użytkownika;
  - Playground: `/v1/systemone` z pięcioma typami pytań, HTTP 200; skrót
    Ctrl+Enter;
  - lista użytkowników z zużyciem quota;
  - lista kluczy; utworzenie klucza z jednorazowym tokenem;
  - stan systemu z metrykami i backlogiem;
  - pliki i batche użytkownika, w tym upload, utworzenie batcha i inferencja
    sesją przez `curl`;
  - użytkownik nie widzi sekcji administracji.
- PASS, przegląd poprawności i bezpieczeństwa (osobny agent, tylko odczyt). Nie
  znalazł dostępu do cudzych danych, eskalacji do administratora ani obejścia
  nagłówka UI. Poprawiono zgłoszone problemy średnie i niskie:
  - throttle tylko dla poprawnych identyfikatorów, usuwa najstarsze wpisy;
  - usuwanie użytkownika blokuje wiersz quota;
  - czas wpisu cache kluczy liczony przed odczytem;
  - sesja tworzona tylko dla sprawdzonego hasła;
  - administrator zmienia własne hasło wyłącznie przez Account;
  - wylogowanie czyści ciasteczko wygasłej sesji;
  - dummy hash liczony przy starcie.
- PASS, etap upraszczania (Opus 5.5): wspólne funkcje haszowania i limitu
  odczytów, mniejsza widoczność stałych. Kontrakt i testy bez zmian.
- Kontrole: wynik `cargo fmt --check`, Clippy, rustdoc i `cargo deny` podaje
  opis PR.

Zrzuty ekranu: [docs/assets/ui](../../docs/assets/ui/).

## Ograniczenia (GAP)

- Migracja 0004 i operacje kont były uruchamiane tylko na SQLite. PostgreSQL,
  SQL Server i MongoDB sprawdzono kompilacją i przeglądem kodu. Lokalny Docker
  był niedostępny. Składni `ALTER TABLE` SQL Server z nazwanymi ograniczeniami
  nie wykonano na serwerze.
- Batche demo nie zostały wykonane, bo model nie miał kwalifikowanego profilu
  pracy w tle. Wykonanie batchy nie zmieniało się w tej pracy.
- Brak testów E2E przeglądarki w repozytorium; widoki sprawdzono ręcznie
  w jednej szerokości (1440 px) i w ciemnym motywie systemu.
- Limit nieudanych logowań jest w pamięci procesu i działa per identyfikator,
  nie per adres. Brak MFA, SSO i resetu hasła e-mailem.
- Lista użytkowników administratora odczytuje zużycie quota osobnym zapytaniem
  na użytkownika (do 1000) i liczy aktywne klucze z najwyżej 100 000 kluczy.

## Dane

Obserwacje to wyniki poleceń i testów opisane w tym raporcie. Nie zachowano
osobnych plików danych, baz, haseł ani tokenów. Do odtworzenia potrzebny jest
build z CUDA, model z cache HF i kroki z docs/UI.md.
