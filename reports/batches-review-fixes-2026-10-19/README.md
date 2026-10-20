# Batch API — poprawki po pierwszym przeglądzie

Status: **PASS opisanych prób CPU i magazynu; GAP pełnej kwalifikacji**.
Data: 2026-10-19. Brak inferencji GPU, pomiaru przyspieszenia i deklaracji gotowości produkcyjnej.

## Cel i wersje

Sprawdzić poprawki wyboru pracy, rozliczania porzuconych claimów, izolacji
uszkodzonego rekordu oraz sterowania przyjęciami i przetwarzaniem. Przegląd
Opus 5.5 R1 znalazł pięć problemów poprawności oraz dwa ryzyka odporności.
Rodzic wdrożył poprawki i zweryfikował opisane niżej obserwacje. R2 nie dostarczył
raportu z powodu limitu API; poprawki nie mają jeszcze niezależnego zatwierdzenia.

Kod: bazowy commit `beb856275fbcb811eff5ce4446479bd74dafa176` i zmiany Batch API
w drzewie roboczym. Nie jest to wynik opublikowanego commitu. Rust 1.95.0,
SQLx 0.9.0, MongoDB driver 3.9.1, PostgreSQL 18.6, MongoDB 8.0.34, Linux x86_64.
SQLite korzysta z przypiętego grafu Cargo. Środowisko CPU; nie mierzyliśmy
jego wydajności ani przepustowości baz.

Każdy przebieg używał osobnej kopii źródeł. Manifesta źródeł i kod driverów
są w archiwum. Sumy oryginalnych manifestów:

| Zakres | SHA-256 manifestu |
|---|---|
| SQLite | `89ed446e19dd14b8f2cb21cdd7e1fb460147cd7884af9efcf2788d0100a7f1be` |
| PostgreSQL i MongoDB | `edf66ac2d4aa39aea33651a080e5ebf4abe7a4b5203e7cac6410ddda6adf025c` |
| Scheduler | `bbcac85732b2642c872fa29053c9b53520538fc206ea3f2743f7031343559c5a` |

Rodzic sprawdził zgodność wszystkich wpisów tych manifestów z surowymi kopiami:
37/37 dla każdego magazynu i 1/1 dla kopiowanego schedulera. Pełne manifesty
HTTP R1/R2 i końcowych kontroli są w `provenance.json` oraz plikach
`ORIGINAL_SOURCE_SHA256SUMS` w archiwum. Wyniki starszych przebiegów HA/S3
nie potwierdzają automatycznie późniejszych zmian.

## Metoda i obserwacje

Jednorazowe drivery operacyjne działały poza Gitem. Korzystały z rzeczywistych
operacji Store i prywatnych, świeżych baz. Dane walidowanych rekordów i koszty
były syntetyczne. Nie tworzyliśmy frameworka ani bramki testowej.
Próby PostgreSQL używały prywatnego socketu Unix, MongoDB pojedynczego replica set
na loopback. Nie jest to nowy dowód TLS, HA ani działania klastra sharded.
Wszystkie procesy baz i prywatne serwery HTTP zakończyły pracę.

| Próba | Zaobserwowany wynik |
|---|---|
| 1001 batchy zablokowanego wcześniejszego ownera | SQLite, PG i Mongo wybrały `bb-active`, którego nie było w dawnym oknie pierwszych 1000 batchy |
| 1001 historycznych wierszy deficytów | Zachowano dług `zz-live`; przesunięcie o 90 001 tur, bez wyzerowania kredytu |
| Porzucony claim | Syntetyczny koszt rzeczywisty 170 zapisany raz; po settlement kredyt −70; powtórzenie release → `Stale` |
| Natychmiastowe ponowienie | `Idle`; rekord miał przyszły `eligible_at` |
| Kolejne porzucenia | Odroczenia 2, 4, 8, 16, 30, 30 s dla prób 2–7; licznik rzeczywistych błędów wykonania nadal 0 |
| Bilans quota | 0 wpisów dryfu we wszystkich trzech magazynach |
| Poprawny duży plan | Prawdziwy tokenizer/planner mini: 24 pytania, 31 259 tokenów łącznie, najdłuższy prompt 1339, forward Single 2678 ≤ 8192; scheduler wybrał rekord bez wyłączenia lane |

Do sprawdzenia rosnących odroczeń driver przesuwał `eligible_at` do zera
w prywatnej bazie. Nie czekaliśmy pełnych okresów. Raportowane różnice
znaczników czasu uwzględniają czas zapytań oraz zaokrąglenie SQLite do sekundy.
170 i −70 to wartości fixture rozliczenia, nie pomiar czasu GPU.

Osobny driver uruchamiał rzeczywisty BatchService/HTTP, planner i executor,
ale **bez silnika GPU**. Pierwszy adapter miał zamknięty kanał wykonania;
drugi zwracał syntetyczne `MemoryLimited` z zerowym czasem urządzenia.

| Próba CPU/HTTP | Zaobserwowany wynik |
|---|---|
| Zmiana bajtu pierwszego rekordu po walidacji | Pierwszy rekord → `input_changed`; następny poprawny rekord dotarł do kanału wykonania; failed 1, queued 1 |
| Współdzielony lokalny limit pamięci | Próby grup [2, 1], potem odroczenie; oba rekordy pending i requires_solo, 0 failure_attempts, po 1 abandon_attempt; brak trwałego błędu rekordu |
| `processing: false`, `accepting: true` | Upload/create 200/200; batch pozostał validating; cancel zakończył go jako cancelled, osobny batch przeszedł do expired |
| `accepting: false`, `processing: true` | Nowy upload/create 503/503; odczyt, cancel, delete istniejących obiektów 200/200/200 |
| Bilans po próbach HTTP | 0 wpisów dryfu quota |
| Trzy produkcyjne szablony YAML | Rzeczywisty loader i walidator zaakceptowały strukturę, oba przełączniki false |

Walidacja szablonów zastąpiła wyłącznie ścieżki plików wdrożeniowych
prywatnymi fixture. Plik CA sprawdzano jako istniejący plik, bez handshake
TLS i bez połączenia z bazą/S3. Profile nie zawierają fikcyjnej kwalifikacji GPU.

## Zmiany i ich granice

- Kandydaci są wybierani najpierw według ownera: obracane okno 32 ownerów
  i 8 batchy na ownera. Wykluczenia i blokady quota działają przed limitem.
  Mongo używa najwyżej dwóch agregacji na widok, zamiast odczytu każdego head
  osobnym żądaniem. Ograniczone są rozmiary odpowiedzi i liczba żądań;
  koszt skanu oraz plan wykonania bazy wymagają osobnego pomiaru.
- Release zapisuje zmierzony koszt porzucenia, a bez potwierdzonego pomiaru
  używa estymacji. Migracja 2 dodaje trwałe odroczenia i licznik porzuceń,
  zachowując sumy migracji 1. SQLite/PG/Mongo wykonały migrację 2 w tych próbach.
- Scheduler nie myli całkowitej liczby tokenów planu z pojedynczym forwardem.
  Błąd integralności danych jest izolowany do rekordu; przejściowe I/O odracza claim.
- Fazy workerów rozdzielają CPU, GPU i cleanup. Awaria CPU nie powinna oznaczać
  utraty wspólnego urządzenia. Niepotwierdzony cleanup zatrzymuje background
  i kwarantannuje rezerwację. Ta część wymaga jeszcze fault injection na GPU.
- Fingerprint obejmuje kwalifikowane limity pamięci i skład mikroserii.
  Bardziej restrykcyjny lokalny limit współdzielonego GPU odracza pojedynczy
  rekord. Przejściowa awaria telemetrii również odracza pracę.

Pierwsza próba SQLite zatrzymała się na zduplikowanym wpisie fixture deficytu;
driver zmieniono na UPDATE istniejącego wpisu. Pierwsza próba PG miała błędnie
dobraną lokalną rolę klienta; poprawny przebieg podał ją jawnie. HTTP R1
potwierdził izolację/sterowanie, po czym zatrzymał walidację przykładu na
nieobecnym wdrożeniowym CA. HTTP R2 z podmienionymi ścieżkami zakończył całość.
To problemy przygotowania driverów, nie obserwowane błędy produktu.

## Kontrole i pozostałe luki

PASS: `cargo fmt --all --check`; Clippy całego workspace, release/all-targets,
`-D warnings`, domyślnie i z `basal-cli/intel`; Rustdoc obu wariantów,
private-items/no-deps, `-D warnings`; Cargo deny all-features z odświeżeniem
advisories; zakres wyjątku `paste`. Ostrzeżenia o duplikatach wersji pozostają
w logu deny. Wyjątki licencji ograniczono do dwóch wersji zatwierdzonych przez użytkownika.

Przegląd rodzica obejmował zmienione granice scheduler→worker,
CPU/GPU/cleanup, ownership rezerwacji, klasyfikację błędów wejścia,
rozliczenie release oraz trwałą kwalifikację ponowienia w trzech adapterach.
Nie zastępuje to niezależnego R2 dla całej powierzchni HTTP/auth/artefaktów/migracji.

GAP: niezależny Opus R2; rzeczywisty SQL Server; lokalna kompilacja CUDA/Metal;
numeryka przed/po i upstream; runtime GPU, szczyty RAM/VRAM, media,
priorytet i opóźnienia online, przepustowość oraz topologie multi-GPU.
Nowy kolektor [tools/batches](../../tools/batches/README.md) ma sprawdzoną
składnię i `--help`; przebieg z GPU nadal wymaga kwalifikacji. Nie dodano
ani nie uruchomiono testów projektu.

Decyzja: Batch API pozostaje domyślnie wyłączone; brak profilu GPU
zakwalifikowanego tym raportem. Termin 24h nadal dopuszcza wygaśnięcie
przy ciągłym obciążeniu online.

## Archiwum i odtworzenie

Publiczne URI: **brak — archiwum lokalne, nieopublikowane**.
Kopia lokalna: `.cache/measurement-artifacts/batches-review-fixes-20261019/measurement-data.tar.gz`.
SHA-256: `1802914ab91be23ab67496a1680ac1fcb8ade2484cd64a0c663717ecc6c1e4af`.
Rozmiar: 3 164 127 B. Rodzic rozpakował archiwum do nowego katalogu
i sprawdził **511/511** wpisów `SHA256SUMS`.

Ścieżki w archiwum: `reports/batches-review-fixes-2026-10-19/`:
`sqlite/`, `postgresql/`, `mongodb/`, `scheduler/`, `http-r1/`, `http-r2/`,
`checks/`, `examples/`, `provenance.json`, `REPRODUCE.md`.
Zawierają źródła operatorów, kopie źródeł produktu i przebiegi bez danych
uwierzytelniających. Prywatne ścieżki zamieniono na symbole.
Oryginalne manifesty opisują bajty sprzed tej zamiany; `SHA256SUMS` w katalogu
głównym archiwum opisuje zanonimizowaną zawartość.

Po otrzymaniu archiwum sprawdź jego SHA-256, rozpakuj do nowego katalogu,
uruchom `sha256sum -c SHA256SUMS` i postępuj według `REPRODUCE.md`.
Potrzebne są świeże prywatne bazy oraz przypięte wejścia mini; nie używaj
danych produkcyjnych. Kolejne zmiany wymagają nowego przebiegu i raportu.
