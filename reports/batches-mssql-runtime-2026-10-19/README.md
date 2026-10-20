# Batch Store — rzeczywisty SQL Server 2022, rootless

Status: **PASS opisanych native operations, R2 ledger i TLS; GAP pełnej kwalifikacji**.
Data: 2026-10-19. Ten nowy raport uzupełnia wcześniejszy SQL Server runtime GAP.
Nie nadpisuje historycznych raportów ani lokalnego archiwum R2 store/helper.
Brak zmiany źródeł produktu, testów projektu, instalacji usługi systemowej,
produkcji, GPU, publikacji i deklaracji gotowości całej inicjatywy.

## Wersja i metoda

Rzeczywisty serwer zgłosił **SQL Server 2022 RTM-CU27-GDR, 16.0.4305.5,
Developer Edition 64-bit**, z oficjalnego obrazu Ubuntu 22.04.5.
Pinned manifest `mcr.microsoft.com/mssql/server`:
`sha256:cf8f0ab2ebe61feec2eed2629061a07d247f791c2a8ddda74a4ac9f06a291718`.
Trzy oficjalne warstwy mają razem 626708787 B skompresowanych danych;
sprawdzono ich rozmiary i SHA-256. Pobranie zajęło 29.48 s, a zakończenie ekstrakcji
39.27 s od początku. Są to czasy przygotowania środowiska, nie benchmark SQL Server.

Wybrano wyłącznie darmową edycję Developer do diagnostyki nieprodukcyjnej,
zgodnie z [oficjalną dokumentacją zmiennych SQL Server](https://learn.microsoft.com/en-us/sql/linux/configure/environment-variables?view=sql-server-ver16).
Nie akceptowano płatnej edycji/licencji. Bwrap użył namespace użytkownika i prywatnego
rootfs/data; transient user units miały MemoryMax 5 GiB, MemorySwapMax 0,
CPUQuota 200%, RuntimeMax 180 s. Wszystkie listenery były wyłącznie lokalne.
Nie korzystano z sudo, Docker socketu ani zmian konfiguracji/security hosta.
Rootless przygotowanie z bwrap jest lokalną metodą kwalifikacji, nie deklaracją
wspieranego przez Microsoft sposobu wdrożenia produkcyjnego.

Serwer nie działał po zakończeniu: trzy owned units były inactive, port własnej
instancji zamknięty. Pierwsza sesja zakończyła się ustawionym limitem czasu w trakcie
ostatniej diagnostyki administracyjnej; oba właściwe operatory ukończyły się wcześniej.
Kolejne kontrolowane sesje TLS i RCSI jawnie zatrzymano; zachowano logi zakończenia.

## Źródło i operator

Kod to baseline `beb856275fbcb811eff5ce4446479bd74dafa176` oraz zamrożone zmiany
Batch API. Operator używał dokładnej kopii bieżącego Store: **37/37 plików Rust
oraz migracji zgodne z produktem** według SOURCE_SHA256SUMS. SQLx 0.9.0,
Tiberius 0.13.0, bb8 0.9.1; pełny Cargo.lock jest zachowany prywatnie.
Nie modyfikowano żadnego pliku źródłowego produktu.

Istniejący ręczny operator A–I wymagał adaptacji wyłącznie swojego scaffolding:
usunięto logi pól, które zniknęły z publicznego dla crate API, referencje nieeksportowanych
helperów oraz wywołania usuniętego renew_upload. Ten ostatni przypadek jest jawnie
pominięty; aktualny upload ma stały deadline. Zachowano początkowy błąd kompilacji
starego operatora. Dodano prywatny tryb R2 ledger z już wykonanym scenariuszem innych
adapterów oraz minimalny native Tiberius admin tryb do odczytu/ustawienia RCSI
w nowych prywatnych bazach. Nie powstał framework ani lasting assertion gate.

Scenariusze używają syntetycznych validated records, wyników i sztucznego przesunięcia
terminów w prywatnej bazie. Nie uruchamiają modelu ani rzeczywistych artefaktów GPU/S3.

## Native operations, RCSI OFF i ON

| Tryb | Migracja | Quota checkpoints | Konkurencja |
| --- | --- | --- | --- |
| RCSI OFF, local plaintext | Current schema 2, identyczny drugi migrate | 23/23 z drift 0 | 6 workerów, 40 rekordów: 40 accepted, 40 unique, 0 rejected, 0 busy po retries |
| RCSI ON, Required TLS/private CA/pełna walidacja hosta | Current schema 2, identyczny drugi migrate | 23/23 z drift 0 | 6 workerów, 40 rekordów: 40 accepted, 40 unique, 0 rejected, 0 busy po retries |

Odczyt sys.databases potwierdził RCSI OFF w dwóch pierwotnych bazach.
W nowej, osobnej bazie ustawiono READ_COMMITTED_SNAPSHOT ON i potwierdzono bit ON
przed uruchomieniem drugiego operatora. Nie zmieniano baz cudzej aplikacji.

A–I obejmuje create/replay, append/seal, accepted/stale commit i jego replay,
cancel barrier/drain/partial publication, duplicate custom_id, walidacyjne cancel,
expired lease/recovery i upload GC, deadline expiry z zaakceptowanym wynikiem,
result-upload fencing/adoption, quota-blocked owner i zakaz zmniejszenia limitów
poniżej commitments, solo marking, file/metadata expiry oraz concurrent claim/commit.
W tych operacjach nie zaobserwowano quota drift. Końcowy doctor może nadal pokazywać
pozostawione przez scenariusze finalizing/in-progress/pending-cleanup obiekty;
operator nie twierdzi, że wszystkie utworzone batche zostały terminalnie usunięte.
Nie wymuszano rzeczywistego deadlocka, awarii COMMIT ani utraty odpowiedzi serwera.

## R2: przypięty katalog i bounded cleanup

Na osobnej bazie rzeczywisty SQL Server wykonał oba migrate do schema 2.
Przy batch revision 1/default m1 zainstalowano revision 2/default m2.
Exact revision 1 nadal zwracała m1, a jawne discovery wskazało fp1.
Brak revision zwrócił None w domenowym Option<Catalog>; uszkodzony digest
zwrócił typowany Corrupt. Mapowanie braku oczekiwanej przypiętej revision w ingest
pozostaje oceną źródła, ponieważ ten operator nie wykonuje pełnej walidacji JSONL.

Historia sterująca: 1201 już rozliczonych settlements z jednym claim_id i różnymi
line_no, 1103 claims, 1301 internal operations i jeden normalny create operation.
Accounting barrier zatrzymała cleanup. Następne rzeczywiste transakcje usuwały
settlements **512/512/177**, claims **512/512/79**, internal operations **512/512/277**.
Store zamknięto i ponownie otwarto po drugiej stronie; postęp zachowano.
Client operation pozostał do retention, a done wymagało pustych ledgerów.
To bezpośrednia obserwacja skutecznego DELETE TOP (512) w adapterze SQL Server.

1000 errors zapisano normalnym fail_validation z naliczeniem quota.
Po retention metadata purge usuwał **512/488** errors, potem client operation,
potem batch/retained slot. Quota drift był 0 przed purge, po każdej jego stronie
i po końcu. Cleanup replay zwrócił done; metadata replay usunął 0 batchy.
Nie wykonywano process-kill ani failover pomiędzy stronami.

## Rzeczywisty TDS Required TLS

Prywatny CA podpisał certyfikat wyłącznie z SAN DNS:localhost.
W prywatnym mssql.conf ustawiono network.tlscert/tlskey, forceencryption=1 i TLS 1.2,
zgodnie z [oficjalną instrukcją szyfrowania SQL Server na Linux](https://learn.microsoft.com/en-us/sql/linux/sql-server-linux-encrypted-connections?view=sql-server-ver16).
Serwer potwierdził załadowanie certyfikatu i TLS 1.2.

Rzeczywisty Store::connect_mssql użył VerifyFull i Tiberius EncryptionLevel::Required
(TDS 7.4, TLS wewnątrz pre-login), bez TrustServerCertificate, zmiany weryfikatora,
wyłączenia walidacji hosta ani fallbacku plaintext:

| Parametry | Obserwacja |
| --- | --- |
| Właściwy CA, właściwy DNS host | Store connect zakończony poprawnie |
| Inny CA, ten sam DNS host | Typowany Unavailable; brak przyjętej sesji |
| Właściwy CA, host nieobecny w SAN | Typowany Unavailable; brak przyjętej sesji |

Adapter redaguje szczegóły błędu, dlatego log negatywny pokazuje Unavailable zamiast
wire alertu. Zmieniano wyłącznie CA albo host przy działającym tym samym serwerze.
Cały operator A–I w RCSI ON następnie przeszedł z właściwym CA/host przez Required TLS.
Tiberius przy ca_file dodaje ten CA do system roots; nie jest to tryb CA-only.
TDS 8.0 Strict na prawdziwym SQL Server nie było wykonywane.

Pomocnicze SQLcmd admin próby miały problem z rozwiązywaniem nazwy i lokalizacją
zaufanego CA w prywatnym rootfs. Nie wyłączono walidacji certyfikatu, aby je obejść.
Ostateczny odczyt/ustawienie RCSI wykonano native Tiberius przez Required TLS,
korzystając z tego samego poprawnego CA i hosta. Nieudane przygotowania zachowano.

## Dowody i ograniczenia

Prywatne dane są w `.cache/measurement-artifacts/batches-mssql-rootless-r2-20261019/`:
`native-scenarios-r1.log`, `native-scenarios-rcsi-tls-r1.log`, `native-r2-ledger-r1.log`,
`tls-native-*.log`, `rcsi-native-admin-r1.log`, `summary.json`, operator/source manifest
oraz osobne start/stop/build logs. Raw rootfs/data, klucze prywatne, hasło SA,
adresy/porty i logi środowiska nie są przeznaczone do publikacji.
Nie wysyłano danych i nie ma URI HF ani anonimowej weryfikacji pobrania.

Zakres PASS dotyczy wyłącznie tego wydania SQL Server 2022 i opisanych operacji.
Pozostają GAP: SQL Server 2025, HA/redirect wielohostowy, commit-unknown/reply-lost,
wymuszony deadlock/lock timeout/session-cancel fault, TDS 8.0 Strict, obciążenie
produkcyjne, długie duże ledger benchmarki oraz GPU/model/S3. Rootless sesja nie
zastępuje produkcyjnej kwalifikacji hosta ani aktualnej pełnej macierzy integracji.


## Lokalne archiwum i odtworzenie

Osobny immutable, sanitizowany pakiet: `.cache/measurement-artifacts/batches-mssql-runtime-sanitized-20261019/measurement-data.tar.gz`.
SHA-256: `4d464b64be3d6ec6e5973f1e4987c4f146dd39fc8797b82d6f981626b2fb294e`. Rozmiar: **179099 B**.
Świeże osobne rozpakowanie potwierdziło **57/57** wpisów root `SHA256SUMS`
oraz dokładne bajty wszystkich **37/37** źródeł Store/migracji.
SHA archiwum sprawdzono ponownie po rozpakowaniu. `ARCHIVE_SHA256SUMS`, `verification.json`
i katalog `fresh-extraction` pozostają obok lokalnego archiwum do niezależnej kontroli rodzica.

Logiczna ścieżka danych to `reports/batches-mssql-runtime-2026-10-19/`.
Pakiet zawiera selected native A–I/R2/TLS/RCSI traces, sanitized summary/failures/GAP,
pinned MCR manifest metadata, exact source37files, portable env-only operator i `REPRODUCE.md`.
Nie zawiera SA secret, cert/private keys, danych SQL, warstw obrazu/rootfs ani raw unit/server logs.
Dane sieciowe/DSN/host paths/PID/identyfikatory transient units nie zostały przeniesione z sesji.
Jedynie dokładne źródła zachowują standardowe loopback constants polityki lokalnego adaptera;
nie są to zapisane identyfikatory maszyn. `REPORT-SNAPSHOT.md` jest sanitizowaną kopią
raportu sprzed tego append, z certificate-name fixture zastąpionym etykietą.
`provenance/file-origins.json` i `transformations.json` opisują źródłowe/archiwalne SHA oraz transformacje.

Odtworzenie: rozpakuj do nowego katalogu, sprawdź root `SHA256SUMS`, następnie przeczytaj
`REPRODUCE.md`. Operator kompiluje się przez `cargo check --locked --release` po relokacji modułów;
nie zawiera connection secret ani endpointu, pobiera je przez nazwane zmienne środowiska.
Ręczne native runs wymagają nowych własnych prywatnych baz i świadomie przygotowanego
SQL Server Developer; archive nie uruchamia ani nie instaluje usługi. Odtworzenie nie jest bramką CI.

Status pakietu: **lokalnie przygotowany i zweryfikowany; niezależna kontrola rodzica oczekiwana**.
Nie wykonano HF upload, pobrania ani anonimowej weryfikacji; brak publicznego URI.
Raw81pliki audit i ich wcześniejsze manifesty pozostały prywatnie bez nadpisania.
Zakończonego archiwum nie nadpisywać; nowe dane wymagają nowego pakietu.
Opisane obserwacje i wcześniejsze GAP pozostają bez zmian.

## Niezależna kontrola rodzica

2026-10-19: niezależnie sprawdzono SHA-256 archiwum `4d464b64be3d6ec6e5973f1e4987c4f146dd39fc8797b82d6f981626b2fb294e`
oraz wszystkie **57/57** wpisów `SHA256SUMS` po nowym rozpakowaniu.
Opisane wyniki i ograniczenia pozostają bez zmian. Weryfikacja jest lokalna;
nie potwierdza publikacji ani pobrania HF.
