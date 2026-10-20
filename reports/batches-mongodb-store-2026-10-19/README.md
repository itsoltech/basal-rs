# Batch store: MongoDB 8.0.34 — przebieg operacyjny natywnego adaptera

Status: **PASS w opisanym zakresie; GAP pełnej kwalifikacji wydania**
(jednowęzłowy replica set, bez failoveru i bez shardingu).
Data: 2026-10-19. Bez pomiaru wydajności i bez inferencji modelu.

## Cel i wersje

Sprawdzić natywny adapter MongoDB modułu `serve/batches/store` na rzeczywistym
serwerze: te same operacje domenowe i niezmienniki co SQLite/PostgreSQL,
transakcje, fencing commitów, nadzorowane anulowanie operacji drivera oraz TLS.

| Składnik | Wersja |
|---|---|
| Serwer | MongoDB 8.0.34 (`mongodb-linux-x86_64-ubuntu2404-8.0.34.tgz`), jednowęzłowy replica set `rs0` |
| Driver | mongodb 3.9.1 (async, rustls), bson 3 |
| Pozostałe | Rust 1.95.0, sqlx 0.9.0 (porównanie z SQLite), tokio-util 0.7.19 |
| Platforma | Linux x86_64; GPU i sprzęt nie były przedmiotem pomiaru |

Źródłem był snapshot modułu `serve/batches/store` z drzewa roboczego
(baza: commit `beb856275fbcb811eff5ce4446479bd74dafa176`). Suma SHA-256 manifestu
wszystkich 33 plików snapshotu: `a1cb6a913765c1f303b8aa82b811bb80db3a1c1f1c1da31fb6821b4f2d5659fc`.
Po przebiegu w drzewie zmienił się wyłącznie `mssql.rs`, którego `mod.rs` jeszcze
nie deklaruje. Wszystkie kompilowane pliki magazynu są identyczne ze snapshotem.
Wcześniejszy przebieg (r1) dotyczył starszego snapshotu sprzed zmian `ops/**`
autora MSSQL; zachowano go wyłącznie historycznie i nie jest podstawą statusu.

## Metoda

Binarkę serwera pobrano z oficjalnego `fastdl.mongodb.org`. Weryfikacja:
SHA-256 z manifestu `downloads.mongodb.org/current.json` i pliku `.sha256` oraz
podpis `.sig` kluczem „MongoDB 8.0 Release Signing Key”. Odcisk klucza:
`4B07 52C1 BCA2 38C0 B4EE 14DC 41DE 058A 4E7D CA05`. Klucz z `pgp.mongodb.com`
i z `www.mongodb.org/static/pgp/` był identyczny bajtowo. Archiwum rozpakowano
prywatnie pod `.cache/tools`, bez instalacji systemowej i bez Dockera.

Każdy przebieg używał świeżego, prywatnego katalogu danych i serwera nasłuchującego
tylko na loopback, z osobną bazą na każdy scenariusz. Polecenia testowe serwera
(`enableTestCommands`) włączono wyłącznie w tej jednorazowej instancji, na potrzeby
failpointów. Po zakończeniu zatrzymano wszystkie serwery.

Zachowany poza Gitem driver operacyjny (nie test projektu ani CI) wykonał:
- migrację dwukrotnie, połączenie runtime i scenariusze A–I identyczne jak dla SQLite/PostgreSQL;
- przebieg obciążeniowy: 16 workerów, 400 rekordów;
- scenariusze awarii F1–F6 przez failpoint `failCommand`;
- macierz TLS T1–T11 z własnym jednodniowym CA A, obcym CA B i certyfikatem serwera wyłącznie z SAN `DNS:localhost`.

Expiry, utratę lease i retencję wywoływano zmianą timestampów w bazie
diagnostycznej; nie czekano 24 godzin. Artefakty były syntetycznymi bajtami;
przebieg nie zapisywał obiektów do Filesystem/S3 i nie uruchamiał HTTP ani modelu.

## Wyniki

| Kontrola | Wynik |
|---|---|
| Migracja dwukrotnie, otwarcie runtime | Current, wersja 1; druga migracja bez zmian |
| Scenariusze A–I | Przebieg zakończony |
| Rozliczenia quota po krokach A–I | 23/23 punkty z zerową różnicą, w obu przebiegach |
| Parytet z SQLite (ten sam snapshot) | Znormalizowane ślady identyczne: 206 linii, poza losowymi ID i liniami backend/doctor; liczniki doctor zgodne |
| 6 workerów, 40 rekordów | 40 akceptacji, 40 unikalnych, 0 odrzuceń, 0 błędów busy po retry |
| 16 workerów, 400 rekordów | 400 akceptacji, 400 unikalnych, 0 odrzuceń, 0 błędów busy; 23/23 punkty quota |
| F1: wywołujący porzuca future po 300 ms, serwer blokuje `commitTransaction` 3 s | Operacja dokończona w nadzorowanym zadaniu; ponowienie tego samego operation id: `replayed=true`, te same wyniki |
| F2: jednorazowy `UnknownTransactionCommitResult` | Ograniczony retry commitu, wynik zaakceptowany |
| F3: trwały `UnknownTransactionCommitResult` | `CommitUnknown`; rekoncyliacja tym samym operation id zapisuje wynik raz, następna zwraca `replayed=true` |
| F4: dwukrotny `TransientTransactionError` (112) | Transakcja ponowiona i zakończona |
| F5: stały `TransientTransactionError` | Ograniczone retry kończą się retryowalnym `Busy` po ok. 170 ms, bez nieskończonej pętli |
| F6: `close()` przy operacji zablokowanej 2 s | `close()` czeka ok. 1,7 s na zakończenie operacji, która się udaje; potem nowe wywołania dają `Unavailable` |
| Otwarte transakcje po przebiegach | `currentOpen=0` |
| T1 VerifyFull z prywatnym CA A, host `localhost` | Połączenie, migracja i operacje |
| T2 obce CA B / T3 publiczne korzenie | Odmowa (przyczyna po stronie drivera: `UnknownIssuer`) |
| T4 zła nazwa serwera (`127.0.0.1`) | Odmowa (certyfikat nieważny dla nazwy) |
| T5 `tlsInsecure=true` / T6 `tlsAllowInvalidCertificates=true` z obcym CA | Odmowa: konfiguracja adaptera wygrywa z URI |
| T7 `tlsAllowInvalidHostnames=true` | Odmowa: driver z rustls odrzuca opcję jako niepoprawny connection string |
| T8 URI `tls=false` przy VerifyFull CA A | Połączenie przez TLS: URI nie wyłącza szyfrowania |
| T9 URI `tlsCAFile` = CA A, konfiguracja CA B | Odmowa: obowiązuje skonfigurowane CA |
| T10 plaintext do serwera wymagającego TLS | Odmowa |
| T11 plaintext do hosta spoza loopback | Odmowa konfiguracji przed połączeniem |

Odmowy zwracają redagowany `Unavailable` bez treści błędu drivera. Przyczyny T2–T5
potwierdzono osobną diagnostyką drivera, która wypisuje wyłącznie klasyfikację
błędu, bez hostów i ścieżek. Liczniki serwera po wszystkich przebiegach instancji
r2: 1022 zatwierdzone i 2011 przerwanych transakcji. Przerwania obejmują konflikty
zapisu rozwiązane przez ograniczone retry oraz wymuszone failpointy. Wyniki
domenowe pozostały dokładne. Nie mierzono wpływu na przepustowość.

Nie było konieczności zmiany kodu adaptera MongoDB w badanym snapshotcie.

Kontrola integracji: `cargo clippy --locked -p basal-cli --release --all-targets
-- -D warnings` w bieżącym drzewie, z osobnym katalogiem docelowym. Kompilacja
typów przeszła; zgłoszono wyłącznie lint martwego kodu i nieużywanych importów.
Po stronie adaptera MongoDB są to `renew_upload` bez wywołującego oraz
niekonstruowany wariant `MongoTls::LocalPlaintext`. Pełne Clippy integracji
nie jest więc jeszcze czyste.

## Ograniczenia i decyzja

To dowód wybranych operacji natywnego adaptera MongoDB, nie kwalifikacja wydania.
Jednowęzłowy replica set nie sprawdza:
- failoveru, zapisu majority na wielu członkach ani przesunięcia zegarów członków;
- zachowania `$$NOW` przy zmianie primary;
- klastra shardowanego.

`failCommand` przerywa commit przed wykonaniem, więc nie odtworzono przypadku
zatwierdzonego commitu z utraconą odpowiedzią. Ścieżkę ledgera dla wyniku
zapisanego wcześniej pokazuje F1 i drugie ponowienie w F3. Nie wykonano
benchmarku, HTTP/SDK, zapisu do magazynu artefaktów, inferencji, Rustdoc ani
kontroli bitowej.

Wymagane dalej: przebieg na replica secie wielowęzłowym z failoverem (MongoDB 8.0),
klaster shardowany (jeśli ma być deklarowany), pomiar kosztu konfliktów przy
wielu workerach oraz ponowny przebieg po zakończeniu integracji i czystym Clippy.

## Dane i odtworzenie

Dane są lokalne, nieopublikowane: `.cache/measurement-artifacts/batches-mongodb-store-20261019/`.
Archiwum `measurement-data.tar.gz` ma 27 128 B i SHA-256
`7d42981171c6e8d76fa6af006b0db295d4cbff8ec6877744c9b234ada941283b`. Zawiera
`SHA256SUMS` dla 20 plików pod `reports/batches-mongodb-store-2026-10-19/`.
Nie utworzono URI HF.

| Ścieżka w archiwum | Zawartość |
|---|---|
| `snapshot-r2/mongodb-trace.txt`, `mongodb-stress.txt` | A–I oraz A–I z 16 workerami / 400 rekordami |
| `snapshot-r2/mongodb-faults.txt`, `mongodb-txstats.txt` | F1–F6 i liczniki transakcji serwera |
| `snapshot-r2/mongodb-tls.txt`, `mongodb-tls-diagnose.txt` | Macierz TLS i klasyfikacja przyczyn odmów |
| `snapshot-r2/sqlite-trace-same-snapshot.txt`, `parity-sqlite-vs-mongodb.txt` | Ślad SQLite z tego samego snapshotu i wynik porównania |
| `snapshot-r2/SOURCE_SHA256SUMS` | Manifest źródeł snapshotu |
| `earlier-snapshot-r1/` | Historyczny przebieg starszego snapshotu |
| `driver/` | Driver (bez prywatnych ścieżek), `Cargo.toml`, skrypt uruchomienia serwera |
| `server-binary-verification.txt`, `locked-clippy-current-tree.txt` | Weryfikacja binarki i kontrola integracji |

Odtworzenie:
1. Pobierz i zweryfikuj binarkę jak wyżej.
2. Uruchom `driver/start-rs.sh <nazwa>`: świeży katalog prywatny, loopback, losowy port.
3. Zainicjuj replica set: `storeprobe admin initiate ZMIENNA_URI` ze zmienną
   `BASAL_MONGO_MEMBER=host:port`.
4. Uruchom `storeprobe mongodb ZMIENNA_URI`.
5. Uruchom `storeprobe mongo-faults ZMIENNA_URI ZMIENNA_URI_ADMIN`.
6. Dla TLS: serwer z `--tlsMode requireTLS` i członek `localhost:port`, potem
   `storeprobe mongo-tls` ze zmiennymi `BASAL_TLS_PORT`, `BASAL_TLS_CA_A`, `BASAL_TLS_CA_B`.

URI przekazuje się wyłącznie przez środowisko. Przed użyciem snapshotu sprawdź
`SOURCE_SHA256SUMS` i sumę manifestu.
