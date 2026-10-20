# Batch HTTP: limity uploadu, cleanup i współbieżny finalizer — 2026-10-19

Status: **PASS w opisanym zakresie CPU/SQLite/filesystem**. Rzeczywisty HTTP, `BatchService`,
router, bounded `serve/http.rs`, store i artefakty działały w prywatnych procesach. Nie uruchamiano
GPU ani inferencji. Próba dużego wyniku w R1 została poprawnie odrzucona; wynik R2 dotyczy jawnego
syntetycznego limitu przyjętego przez typed store. Pełny runtime odzyskania walidacji JSONL po zmianie
rewizji/defaultu katalogu pozostaje **GAP**.

## Cel, źródła i wejścia

Sprawdzono, czy nowe limity multipart/deadline kończą także niepełne dane przed plikiem, czy cleanup
zachowuje rezerwację do bezpiecznego GC oraz czy długi eksport nie zatrzymuje expiry, cancel i
rozliczenia innego ownera. To ręczna diagnostyka implementacyjna; nie jest frameworkiem testów,
bramką CI, trzecim przeglądem ani pomiarem przepustowości.

Źródła pochodzą z roboczej implementacji na bazie `beb856275fbcb811eff5ce4446479bd74dafa176`.
Sam commit nie identyfikuje niezacommitowanych zmian: pełne drzewa kompilacji przypinają manifesty
`full-source-SHA256SUMS-r1/r2` w archiwum. `source-provenance-r1.json` porównuje źródła produkcyjne
z kopią diagnostyczną. Rust `1.95.0`, Cargo `1.95.0`; build `--locked --release`, domyślne features
Linux, bez CUDA. Procesor hosta Linux x86_64 wykonywał wyłącznie planowanie, HTTP, SQLite i I/O;
nie kwalifikujemy konkretnej maszyny, jej pamięci ani opóźnień produkcyjnych.

Użyto wyłącznie rzeczywistych tokenizer/config/CALIBRATION/prompt modelu **basal-2.0-mini**,
bez wag, z prywatnym profilem CPU. Hash każdego wejścia jest w `model-input-sha256s-r1.json`.
Obliczony fingerprint jest identyfikatorem tego fixture'u; nie potwierdza tożsamości pełnego
modelu ani zgodności numerycznej. Wynik 300 356 B jest syntetycznym JSONL, nie wynikiem modelu.

Niezmienione pliki krytycznych granic mają SHA-256:

| Plik | SHA-256 |
|---|---|
| `crates/basal-cli/src/serve/batches/api.rs` | `3becf36075a370d043e1fa6c92a6019b74032e5ea59cdc6e76d4fae3bb92a96c` |
| `crates/basal-cli/src/serve/batches/upload.rs` | `726a7bfcc3bc86808fe2bcf6e6197b8e0afe04e95d010ac0d1c1b25e242e2dad` |
| `crates/basal-cli/src/serve/batches/artifacts/mod.rs` | `824e45debf46c88925ccb4eac34ad6978a1568e2626150afc8df601340ec8318` |
| `crates/basal-cli/src/serve/http.rs` | `48503410af85e3827cf44650e118cb25ffb30bd951985e57efbfae5e780e8807` |

Prywatne różnice są jawne w `private-fixture-patch-r1/r2.diff`: wejście operatora, profil CPU,
prywatne kopiowanie dużego stored-resultu 1024 B/20 ms, a w R2 także jeden syntetyczny
`ValidatedRecord`. Produkcyjnych źródeł podczas zbierania tych danych nie zmieniano.

## Metoda i wyniki uploadu — R1

Konfiguracja została rzeczywiście zwalidowana: `upload_max_duration_s=60`, buffer 65 536 B,
max file 262 144 B, I/O guard 7200 s. Dwa uwierzytelnione, niekończące się chunked body działały
równolegle: pole przed plikiem oraz rozpoczęty plik. Zewnętrzny obserwator tylko do odczytu
zbierał stan SQL i liczbę fizycznych obiektów/staging co 200 ms.

| Obserwacja | Wynik |
|---|---|
| Poprawny upload, 219 B | HTTP 200; plik opublikowany; open sessions 0; reserved 0 |
| Cisza w polu przed plikiem | HTTP 408 po 60 002 ms; bez sesji/rezerwacji DB dla tego requestu |
| Cisza w rozpoczętym pliku | HTTP 408 po 60 002 ms; staging usunięty, sesja `abandoned` |
| Niedokończony preamble / nagłówek partu | HTTP 413 po 37 / 36 ms; brak sesji DB |
| Niedokończony nagłówek po pliku | HTTP 413 po 36 ms; sesja `abandoned`, staging 0 |
| Shutdown podczas rozpoczętego pliku | HTTP 503; staging 0, sesja `abandoned`; service join 1 ms |

W trakcie rozpoczętego uploadu jedna sesja ownera rezerwowała 266 240 B i jeden slot pliku.
Po odmowie rezerwacja pozostała do upływu I/O guard; to oczekiwane zabezpieczenie cleanupu.
Dopiero po zaobserwowanym porzuceniu i usunięciu staging przesunięto `lease_until` porzuconej
sesji w **jednorazowej bazie** poza guard. Natywny lifecycle sam usunął obiekt i zwolnił quota
po 708–1013 ms. Nie czekano rzeczywistych 7200 s i nie skracano ograniczeń produkcyjnego kodu.

Po usunięciu poprawnego pliku przez HTTP oraz analogicznym przesunięciu jego terminu GC:
wszyscy ownerzy mieli used/reserved/files/open sessions = 0; sesje SQL = 0; fizyczne obiekty = 0;
staging = 0. `Store::quota_drift` zwracał 0 we wszystkich zapisanych punktach. Po shutdown i
ponownym uruchomieniu serwisu guarded GC zakończył również pozostawioną sesję; stan wrócił do zera.

## Długi eksport i kontrola innego ownera — R2

R1 poprawnie odrzucił syntetyczny wynik 300 356 B jako `ExceedsResultBound`; replay tego commitu
zwrócił ten sam wynik. Próba R1 nie kwalifikuje długiego eksportu. Zachowano cały sygnał, w tym
nieukończony batch i orphaned result objęty rezerwacją, zamiast oznaczać tę część jako PASS.

Osobna baza i kopia R2 przyjęły tylko dla `custom_id=large-result` syntetyczny
`ValidatedRecord.result_bound=1 MiB` i host bound powiększony o 4 MiB. Rzeczywisty planner podał
16 573 B; nie twierdzimy, że prawdziwa inferencja mogłaby wygenerować/admitować ten duży wynik.
Typed store policzył rezerwację z podanego boundu. Result session, zapis obiektu, typed claim,
commit i replay przebiegły przez rzeczywistą implementację. Spowolnienie dotyczyło jedynie
kopiowania stored-resultu >256 KiB w prywatnym lifecycle; finalizer/control split pozostał bez zmian.

| Obserwacja | Wynik |
|---|---|
| Commit dużego stored-resultu | `Accepted`; replay idempotentny, ten sam outcome |
| Alice rozpoczyna eksport | faza `finalizing`, 300 356 B |
| Po 638 ms eksportu | Alice nadal `finalizing`; Bob expiry i cancel już `finalizing`; Bob pending=false, unsettled=0 |
| Koniec eksportu | `completed` po 7921 ms; output dokładnie równy wszystkim 300 356 B commitu |
| Terminalne batche Boba | `expired` oraz `cancelled` |
| Stan po cleanup | sesje 0, reserved 0, open sessions 0, active batches 0, staging 0, pending settlements 0, drift 0 |
| Fizyczne obiekty | 6, równe 6 zachowanym plikom DB |

To dowód postępu control duties przy celowo powolnym eksporcie, nie przyspieszenia ani gwarancji
produkcyjnej latencji. Normalne quota zużyte przez zachowane wejścia, metadane i pliki wynikowe
pozostały naliczone. Oba własne procesy HTTP zakończyły się kodem 0; końcowy service join R2 trwał
2 ms. SQLite nie uruchamiał osobnego serwera DB. Nie uruchamiano GPU, SSH ani usług zewnętrznych.

## Ograniczenia i decyzja

- **GAP:** pełna rzeczywista walidacja/odzyskanie JSONL: pierwsza linia explicit model, druga bez
  modelu, zmiana rewizji/defaultu katalogu przed wznowieniem. Istniejący binarny harness nie miał
  kontrolowanego hooka; trzecia ciężka przebudowa CLI została pominięta według jawnego limitu
  bieżącego zadania. Przegląd źródeł nie zastępuje tej obserwacji runtime.
- W tym raporcie nie badano PostgreSQL, MongoDB, SQL Server, S3, GPU, jakości modelu, RSS ani
  przepustowości. Nie rozszerzamy wyniku na inne profile/model inputs.
- GC zakwalifikowano po kontrolowanym przesunięciu czasu porzuconych writerów w prywatnej DB;
  nie zmieniano quota bezpośrednio i nie zwalniano rezerwacji przed guard.
- R2 kwalifikuje ścieżkę finalizacji syntetycznie przyjętego wyniku. To nie pomiar rzeczywistych
  boundów planera lub modelu. Natywny negatywny wynik R1 pozostaje zachowany.
- Native service używał nowego bounded HTTP helpera bezpośrednio. Root GPU `serve_inner` nie był
  uruchamiany; jego integrację sprawdzają osobne kompilacje/kwalifikacje.
- Nie dodano ani nie uruchomiono testów projektu, frameworków, snapshotów testowych lub bramek CI.

Wyniki wspierają zamknięcie runtime luki upload/session cleanup oraz control interleaving przy
długim eksporcie w opisanej topologii. Pinned-default recovery pozostaje jawnie niezakwalifikowane.

## Dane, integralność i odtworzenie

Archiwum lokalne: `.cache/measurement-artifacts/batches-native-http-r2-20261019/measurement-data.tar.gz`
(54 065 B), SHA-256:

```text
dd0abd05ed8a67c96417eb928fc5fb9abd6c9c7b4289ce75f96942c5ec72b7de
```

**Nie wysłano do HF.** Docelowy URI, jeżeli później zostanie opublikowane i niezależnie sprawdzone:
`hf://buckets/itsoltech/basal-rs/measurements/batches-native-http-r2-20261019/dd0abd05ed8a67c96417eb928fc5fb9abd6c9c7b4289ce75f96942c5ec72b7de/measurement-data.tar.gz`.
Ten URI nie jest potwierdzeniem dostępności. Nie wykonano pobrania HF ani pobrania anonimowego.

Świeże lokalne rozpakowanie przez autora potwierdziło **16/16** sum `SHA256SUMS`; rodzic wykona
osobną weryfikację. Zachowano lokalne archiwum. Dane w nim mają logiczny katalog
`reports/batches-native-http-r2-2026-10-19/`:

- `native-r1.log`, `native-r2.log`, `native-observations-r1/r2.jsonl`: pełne obserwacje obu prób;
- `final-state-r1/r2.json`: końcowy SQL i fizyczne obiekty, także negatywny stan R1;
- `metadata.json`, manifesty source/provenance/model-input SHA, fixture patches: wersje i różnice;
- `driver/trace-r1.rs`, `driver/trace-r2.rs`, `METHOD.md`: portable kopie driverów, metoda i komendy.

Portable drivery mają wyłącznie neutralny zapis loopback bind i syntetyczny nagłówek Host;
oryginały użyte przy kompilacji przypinają pełne manifesty źródeł i pozostały lokalnie do kontroli.
Nie są one bitowo identyczne z portable kopiami, co jawnie opisuje `metadata.json`.

Archiwum rozpakuj do nowego katalogu, sprawdź SHA-256 archiwum i wszystkie wpisy `SHA256SUMS`.
Dokładne flagi kompilacji i uruchomienia są w `METHOD.md`; każda próba wymaga nowej prywatnej
kopii przypiętego drzewa, zastosowania odpowiedniego fixture patch i wyłącznie wejść CPU mini
bez wag. Dalsze udostępnienie wymaga procedury [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).

## Niezależna kontrola rodzica

2026-10-19: niezależnie sprawdzono SHA-256 archiwum `dd0abd05ed8a67c96417eb928fc5fb9abd6c9c7b4289ce75f96942c5ec72b7de`
oraz wszystkie **16/16** wpisów `SHA256SUMS` po nowym rozpakowaniu.
Opisane wyniki i ograniczenia pozostają bez zmian. Weryfikacja jest lokalna;
nie potwierdza publikacji ani pobrania HF.
