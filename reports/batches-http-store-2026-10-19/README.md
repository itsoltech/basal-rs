# Batch API: HTTP, lifecycle i artefakty — przebieg operacyjny bez GPU

Status: **PASS w opisanym zakresie; GAP pełnej kwalifikacji wydania**.
Data: 2026-10-19. Bez inferencji modelu, bez pomiaru wydajności i bez GPU.

## Cel

Sprawdzić na rzeczywistym `BatchService` (HTTP, uwierzytelnienie, walidacja,
supervisor lifecycle, finalizacja, GC, retencja), magazynie artefaktów i typowanym
`Store`, że ścieżki niezależne od GPU działają na SQLite, PostgreSQL 18.6 i S3.
Wyniki rekordów były **syntetycznymi bajtami fixture**: przechodziły przez
prawdziwe claimy i commity magazynu, ale nie są wynikami modelu. Ten raport nie
dotyczy numeryki, priorytetu online ani wykonania na GPU.

## Wersje i wejścia

- Kod: commit bazowy `beb856275fbcb811eff5ce4446479bd74dafa176` z niezatwierdzonymi
  zmianami inicjatywy Batch API. Manifest SHA-256 źródeł pakietu HTTP/lifecycle
  (19 plików: `serve/batches/*` bez executor/bg_lane/scheduler/device_budget,
  `serve/batches/artifacts/*`, `src/batches.rs`, `contracts/batches/openapi.json`):
  plik `SOURCE_SHA256SUMS`, jego suma
  `b11fefb3cd69113afac9726298af012a33ab4bee79e53a0052e7ba3c3b803ed7`.
  Wynik nie dotyczy automatycznie późniejszych zmian.
- Rust 1.95.0, axum 0.8.9, sqlx 0.9.0, object_store 0.14.2, Linux x86_64.
- PostgreSQL 18.6 z podpisanego pakietu `postgresql-18.6-2` (prywatnie rozpakowany,
  bez instalacji systemowej), świeże bazy diagnostyczne, TLS verify-full z prywatnym
  CA, SCRAM-SHA-256, osobne role migracji i runtime.
- S3: SeaweedFS 4.48 z pakietu Arch `seaweedfs-4.48-1` (podpis sprawdzony
  `pacman-key --verify`; suma pakietu w `checks/seaweedfs-package.sha256`),
  prywatnie rozpakowany, nasłuch tylko na loopback, S3 przez TLS z tym samym
  prywatnym CA, statyczne poświadczenia z plików prywatnych. Oficjalne binarki MinIO
  community są wycofane (HTTP 410, projekt zarchiwizowany) i nie były używane.
- Planer CPU: lokalne pliki modelu `basal-1.5-mini` (tokenizer, prompty, plan).
  Współczynniki `HostStaging` w driverze są wartościami diagnostycznymi, nie pomiarem.

## Metoda

Jednorazowy driver operacyjny (`harness/trace3.rs.txt`, poza Gitem projektu, nie
jest testem ani bramką CI) uruchamia `BatchService` z prawdziwym routerem na
prywatnym listenerze i dwoma właścicielami (`alice`, `bob`) z tokenami bearer.
Driver działa jako klient HTTP oraz jako wykonawca: pobiera claimy przez
`Store::claim_items`, zapisuje wyniki przez `Store::commit_results`, a duży wynik
przez sesję wyniku i artefakt. Upływ czasu (deadline 24h, lease, I/O guard, grace
pobrania, retencja metadanych) wywoływano przesunięciem znaczników czasu w świeżej
bazie diagnostycznej, bez czekania. Każdy z czterech przebiegów wykonano na
końcowym kodzie z manifestu; wcześniejsze przebiegi robocze dały te same wyniki.

| Przebieg | Magazyn | Artefakty |
|---|---|---|
| `traces/sqlite.txt` | SQLite (profil local, tokeny) | system plików |
| `traces/pg-fs.txt` | PostgreSQL 18.6, TLS verify-full | system plików |
| `traces/pg-s3.txt` | PostgreSQL 18.6, TLS verify-full | S3 (SeaweedFS, TLS) |
| `traces/s3-artifacts.txt` | — | S3: semantyka magazynu artefaktów |
| `traces/pg-cli.txt` | PostgreSQL 18.6 | `basal batches migrate/doctor` |

## Wyniki

Wyniki są identyczne we wszystkich trzech przebiegach usługi (SQLite, PG+FS, PG+S3).

| Scenariusz | Obserwacja |
|---|---|
| S1 izolacja ownerów | `bob` dla pliku i batcha `alice`: GET/content/DELETE/create/cancel → 404, identyczne ciało jak dla nieistniejącego id; listy `bob` puste; kursor `alice` → 400 |
| S2 expiry z mieszanymi wynikami | 6 rekordów: wynik inline, duży wynik przez artefakt (300 342 B), błąd natywny 422, 3 bez wyniku. Po deadline: `expired`, liczniki total 6 / completed 2 / failed 4 |
| Bajtowa zgodność eksportu | Plik output równy bajt w bajt zatwierdzonym liniom 1 i 5 w kolejności (300 700 B); plik error zaczyna się zatwierdzoną linią 422, potem 3 linie `batch_expired` |
| Replay commitu | Ten sam operation id → `replayed=true`, ten sam wynik |
| Cleanup po publikacji | Źródłowy artefakt dużego wyniku usunięty po publikacji |
| S3 cancel z częściowym wynikiem | `cancelled`, output = zatwierdzona linia, brak pliku error |
| S4 awaria finalizatora | Finalizator „po awarii” zapisał częściowy eksport; drugi finalizator przy ważnym lease: `Idle`; po końcu lease: `NeedsFencing`, adopcja przed I/O guard odrzucona (`FencingPending`); po przesunięciu poza guard nowa instancja usługi usunęła częściowy obiekt, przejęła rezerwę i opublikowała `completed` z poprawnymi bajtami |
| S5 GC pliku i retencja | DELETE → 404 przy pobraniu; po `gc_after` liczba plików 8 → 7, `used_bytes` zmniejszone; purge metadanych → 404, ten sam Idempotency-Key tworzy nowy batch |
| S6 osierocone sesje | Przerwany upload i niezatwierdzony obiekt wyniku: po I/O guard obiekt usunięty, sesje 2 → 0, `open_sessions` 0; dryf quota 0 po S1–S6 |
| S7 shutdown | Aktywny upload otrzymał 503; pobieranie 64 MiB zatrzymanego klienta przerwane po ok. 3 MiB; shutdown < 0,1 s przy grace 30 s; brak plików staging |
| CLI na PostgreSQL | `migrate` (konto migracji) i `doctor` (konto runtime) bez problemów; obce CA → redagowany `Unavailable`; `sslmode=disable` w URI odrzucone przed połączeniem |
| S3: semantyka artefaktów | Pojedynczy PUT z SHA-256; zakres; zła suma i zła długość wykryte przy odczycie; ten sam klucz odrzucony (pre-check i `If-None-Match` serwera); multipart 12 MiB w częściach 5 MiB zgodny bajtowo, zakres przez granicę części; ETag nie jest SHA-256; przekroczony limit i porzucony writer nie zostawiły obiektu; termin zapisu i odczytu egzekwowany; podwójne DELETE poprawne; 0 obiektów pozostawionych |
| Sekrety i ścieżki | 0 wystąpień haseł i kluczy w zapisanych logach; brak ścieżek lokalnych w archiwum |

Kontrole kodu na lustrze bieżącego drzewa: Clippy (`--all-targets`) bez uwag w
plikach pakietu (21 uwag w plikach innych właścicieli), Rustdoc `-D warnings`
zakończony, `rustfmt --check` bez różnic w plikach pakietu (`checks/`).

## Ograniczenia (GAP, nie PASS)

- Brak inferencji i GPU: syntetyczne wyniki nie potwierdzają executora, mikroserii,
  numeryki, priorytetu online, czasu oddania GPU ani pamięci (T8–T11 pozostają GAP).
- MongoDB i SQL Server nie były uruchamiane przez usługę w tym przebiegu; adapter
  SQL Server jest tylko skompilowany (brak serwera).
- S3 sprawdzono tylko na SeaweedFS 4.48, nie na AWS S3 ani innych implementacjach;
  łańcuch poświadczeń `aws_default_chain` i reguła cyklu życia dla niedokończonych
  multipart nie były sprawdzane. Odczyt usuniętego obiektu S3 zgłasza ogólny błąd
  odczytu przy pierwszym fragmencie, nie `NotFound` przy otwarciu.
- Shutdown podczas niedokończonej mutacji bazy (oczekujący handler) nie został
  zaobserwowany: nowe połączenie po rozpoczęciu shutdown było odrzucane przez serwer
  HTTP przed handlerem.
- Flagi `batches.accepting=false` i `batches.processing=false` nie były uruchamiane
  (przebiegi używały wartości domyślnych `true`).
- Upływ czasu symulowano zmianą znaczników w bazie; nie czekano 24h ani 2h.
- Brak testu awarii procesu w trakcie COMMIT, failoveru PostgreSQL, wielu procesów
  usługi i obciążenia.

## Decyzje

- Pliki wyników (output i error) mają publicznie `purpose=batch_output`, jak
  OpenAI FileObject; wcześniejsze lokalne rozszerzenie `batch_error` usunięto
  z API i kontraktu. Filtr `purpose=batch_output` obejmuje oba pliki, ponieważ
  magazyn zapisuje je z tym celem.
- Kod nie wymagał poprawek po przebiegach.

## Dane i odtworzenie

Dane są lokalne, **nieopublikowane** (brak URI HF):
`.cache/measurement-artifacts/batches-http-store-2026-10-19/`, archiwum
`measurement-data.tar.gz` (20 925 B), SHA-256
`d65d35d3dfd1c34aef914931c7987704c66e149fe1aed67175f2f8477c35118c`, 17 plików
w `SHA256SUMS` (sprawdzone po rozpakowaniu do nowego katalogu). Ścieżki w archiwum:
`reports/batches-http-store-2026-10-19/{traces,harness,checks}/` oraz
`SOURCE_SHA256SUMS`. Ścieżki lokalne w kopiach zastąpiono `$REPO`, `$SCRATCH`
i `$HOME`.

Odtworzenie wymaga lokalnej kopii drzewa zgodnej z `SOURCE_SHA256SUMS`, plików
modelu `basal-1.5-mini`, prywatnej instancji PostgreSQL 18.6 z nową bazą na
przebieg oraz prywatnego SeaweedFS z TLS. Driver dołącza się do kopii roboczej
skryptem `harness/apply_trace3.sh.txt`, a `harness/run_r3_all.sh.txt` uruchamia
cztery przebiegi i zatrzymuje serwery. Hasła, klucze S3 i URI przekazuje się
wyłącznie przez środowisko. Prywatnych kluczy TLS, danych baz i logów serwerów
nie publikować.
