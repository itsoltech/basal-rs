# Batch store: PostgreSQL 18.6 — przebieg operacyjny

Status: **PASS w opisanym zakresie; GAP pełnej kwalifikacji wydania**.
Data: 2026-10-19. Bez pomiaru wydajności i bez inferencji modelu.

## Cel i wersje

Sprawdzić operacje trwałego magazynu batchy na rzeczywistym PostgreSQL,
po wcześniejszym przebiegu SQLite. Użyto PostgreSQL 18.6, sqlx 0.9.0,
Rust 1.95.0 i UUID 1.27.0, na Linux x86_64. Sprzęt nie był przedmiotem pomiaru;
GPU nie uczestniczyło w przebiegu.

Źródłem był snapshot nowego modułu `serve/batches/store` podczas implementacji
na bazie commitu `beb856275fbcb811eff5ce4446479bd74dafa176`, przed podłączeniem
MongoDB i zmianami dialektu MSSQL. Suma manifestu SHA-256 wszystkich plików
tego snapshotu: `d48147a432df7f5e986ef53f0697de05bb568f19e023de09b59aff43e786a006`. Wynik nie dotyczy automatycznie późniejszych zmian.

## Metoda

Przypięte pakiety dystrybucji PostgreSQL 18.6-2 i numactl 2.0.19-1 rozpakowano
w prywatnym katalogu po sprawdzeniu podpisów pacman-key. Bez instalacji systemowej.
Utworzono świeżą lokalną bazę i uruchomiono zachowany poza Gitem driver operacyjny
magazynu, scenariusze A–I. Driver nie należy do zestawu testów ani CI projektu.

Pierwszy przebieg używał prywatnego socketu Unix. Kolejne używały TLS,
własnego jednodniowego CA i certyfikatu serwera oraz SCRAM-SHA-256.
Osobny przebieg przekazał `sslmode=disable` w URI adaptera, ale jawne
`PgTls::VerifyFull` nadal wymagało poprawnego CA i nazwy serwera. Techniczne
połączenie drivera do sterowania czasem używało osobnego URI verify-full.
Po zakończeniu zatrzymano prywatny serwer.

Expiry, utratę lease i retencję wywoływano przez zmianę timestampów w świeżej
bazie diagnostycznej; nie czekano 24 godzin. Artefakty wejścia i wyników były
opisami magazynu oraz syntetycznymi bajtami. Przebieg nie zapisywał rzeczywistych
obiektów do Filesystem/S3 i nie uruchamiał HTTP ani modelu.

## Wyniki

| Kontrola | Wynik |
|---|---|
| Migracja i otwarcie schematu | Current, wersja 1 |
| Lifecycle, idempotency replay/conflict, izolacja ownerów, przypięty plik | Przebieg zakończony |
| Walidacja, przejęcie lease, ponowny append, duplikat custom_id | Przebieg zakończony |
| Commit/replay, retry, cancel/drain, expiry i publikacja | Przebieg zakończony |
| Osierocony wynik, przejęcie finalizatora, fencing i GC | Przebieg zakończony |
| Quota, owner quota_blocked, requires_solo, retencja i purge | Przebieg zakończony |
| Rozliczenia quota po krokach A–I | 23/23 punkty z zerową różnicą w każdym pełnym przebiegu |
| 6 równoległych workerów, 40 rekordów | 40 akceptacji, 40 unikalnych numerów, 0 odrzuceń, 0 błędów busy po retry |
| Pełny przebieg przez poprawne TLS | Zakończony |
| VerifyFull przy URI sslmode=disable | Pełny przebieg zakończony; wymóg TLS zachowany |
| Obce CA / błędna nazwa serwera | Oba połączenia odrzucone jako redagowany Unavailable przed migracją |

Nie było konieczności poprawiania kodu SQL w badanym snapshotcie. Początkowa
próba połączenia bez jawnej nazwy roli została odrzucona; po wskazaniu roli
lokalnej przebieg przeszedł. Nie jest to błąd adaptera ani wynik kwalifikacji.

## Ograniczenia i decyzja

To dowód wybranych operacji adaptera PostgreSQL, nie dowód kompletności systemu.
Nie wykonano awarii procesu podczas COMMIT, zerwania sieci w niepewnym commicie,
failoveru PostgreSQL, obciążenia wieloma procesami, benchmarku, HTTP/SDK,
inferencji ani kontroli bitowej. Nie potwierdza to zachowania MSSQL, MongoDB,
S3, współpracy z GPU ani dotrzymania 24h. Po integracji wymagane są kompilacja,
Clippy, Rustdoc i ponowny przebieg dla końcowego kodu oraz brakujące scenariusze.

## Dane i odtworzenie

Dane są lokalne, nieopublikowane: `.cache/measurement-artifacts/batches-implementation-20261019/postgresql/`.
Nie utworzono publicznego archiwum ani URI HF. Snapshot źródeł i drivera jest
w `probe-r1/`, manifest źródeł w `probe-r1/SOURCE_SHA256SUMS`, sumy logów
w `SHA256SUMS-r1`. Logi pełnych przebiegów: `operator-trace-r2.txt`,
`operator-trace-tls-r1.txt`, `operator-trace-tls-override-r2.txt`; odmowy TLS:
`tls-refusal-untrusted-ca-r1.txt`, `tls-refusal-wrong-host-r1.txt`.

Odtworzenie wymaga świeżej prywatnej instancji PostgreSQL 18.6 i nowej bazy
na każdy przebieg. Driver buduje się z manifestu w `probe-r1/`; przyjmuje
`postgresql NAZWA_ZMIENNEJ_URI [NAZWA_ZMIENNEJ_CA]`. Opcjonalna zmienna
`BASAL_BATCH_PROBE_RAW_PG` podaje połączenie techniczne drivera do sterowania
czasem. URI i hasła przekazuje się wyłącznie przez środowisko, poza logami.
Przed użyciem snapshotu należy sprawdzić `SOURCE_SHA256SUMS` i sumę jego manifestu.
Prywatnych konfiguracji, kluczy TLS, danych bazy i logów serwera nie publikować.
