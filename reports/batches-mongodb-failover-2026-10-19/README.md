# Batch store: MongoDB 8.0.34 — failover replica setu i utracona odpowiedź po commicie

Status: **PASS w opisanym zakresie; GAP kwalifikacji HA, zegarów i shardingu.**
Data: 2026-10-19. Bez pomiaru wydajności i bez inferencji modelu.

## Cel i wersje

Uzupełnić [przebieg natywnego adaptera MongoDB](../batches-mongodb-store-2026-10-19/README.md)
o trzy elementy:
1. Ślad A–I na najnowszym drzewie.
2. Wybory i failover trzywęzłowego replica setu podczas obciążenia z identyfikatorami operacji.
3. Rzeczywista utrata odpowiedzi na commit, który serwer zatwierdził, i rekoncyliacja tym samym operation id.

| Składnik | Wersja |
|---|---|
| Serwer | MongoDB 8.0.34, ta sama zweryfikowana binarka co w poprzednim raporcie |
| Topologie | jednowęzłowy replica set `rs0`; trzywęzłowy `rs0` na tym samym hoście |
| Driver | mongodb 3.9.1 (async, rustls); Rust 1.95.0; sqlx 0.9.0 do porównania z SQLite |
| Platforma | Linux x86_64; sprzęt i GPU nie były przedmiotem pomiaru |

Źródłem jest snapshot `serve/batches/store` z drzewa roboczego (baza: commit
`beb856275fbcb811eff5ce4446479bd74dafa176`).

| Snapshot | Manifest SHA-256 (33 pliki) | Zawartość |
|---|---|---|
| Najnowszy (podstawa statusu) | `8d4cdc6c43d860ad2c12183566cad46a7a8451103aba3b46cc9df53a14373a73` | Plik błędów publikowany z `purpose=batch_output` (Mongo i SQL), poprawka dzielenia w `policy.rs::drr_pick`, fasada MSSQL w `mod.rs`/`types.rs`/`error.rs` |
| Wcześniejszy | `1332f5ce2f9b905c437b14fbde8561202e05dbbfa6f482f700867d4df4307ed7` | Zmiana `batch_output` bez poprawki `drr_pick` i bez fasady MSSQL |

Moduł `mongodb/**` jest identyczny w obu snapshotach. Poprawka zaokrąglania
w górę w `drr_pick` (`missing / quantum + (missing % quantum != 0)`) zmienia wynik
tylko przy bardzo dużych kosztach. Rekordy o koszcie 10 w tym przebiegu jej
nie testują.

## Metoda

Wszystkie procesy `mongod` uruchamiano z `--wiredTigerCacheSizeGB 0.25`,
wyłącznie na loopback, w nowych prywatnych katalogach i z osobną bazą
na scenariusz. Polecenia testowe włączono tylko w instancjach jednorazowych.
Po każdym scenariuszu zatrzymano procesy, sprawdzając PID. Na końcu nie działał
żaden `mongod`. Driver operacyjny jest narzędziem poza Gitem, nie testem projektu ani CI.

**Ślad A–I.** Te same scenariusze co wcześniej. Driver dodatkowo odczytuje
`purpose` opublikowanych plików wyników. Ślad SQLite z tego samego snapshotu
porównano po normalizacji losowych identyfikatorów.

**Failover.** Trzy procesy, priorytety równe, zapis majority i journal.
- Obciążenie: 720 rekordów, 6 workerów, do 4 rekordów na claim, 2 s przerwy
  między claimem a commitem, lease wykonania 20 s.
- Każdy błąd commitu skutkuje ponowieniem z tym samym operation id i tymi samymi
  wynikami. Claimy o nieznanym wyniku wracają po wygaśnięciu lease przez
  `recover_expired_claims`.
- Oś czasu:

  | Czas | Zdarzenie |
  |---|---|
  | +10 s | `replSetStepDown` na primary |
  | +23 s | `SIGKILL` bieżącego primary |
  | +47 s | restart zabitego węzła (wraca jako secondary) |
  | +62 s | ponowny `replSetStepDown` |

- Na końcu: rozliczenia, ponowienie każdej zapisanej operacji tym samym id,
  kontrola quota, finalizacja, publikacja i cleanup.

**Utracona odpowiedź.** Prywatne proxy na loopback między adapterem
a jednowęzłowym serwerem, z `directConnection=true`, więc driver nie omija proxy
przez wykrywanie topologii. Proxy rozpoznaje `commitTransaction` w ramkach OP_MSG.
Gdy jest uzbrojone, przekazuje polecenie serwerowi, odczytuje jego odpowiedź,
zapisuje wartość `ok` i zamyka połączenie klienta bez przekazania odpowiedzi.
Serwer zatwierdził więc transakcję, a klient tego nie wie. Kompresja wiadomości
nie była włączona.

Nie jest to failpoint `failCommand`: failpoint przerywa commit przed wykonaniem
i nie dowodzi utraty odpowiedzi po zatwierdzeniu.

## Wyniki (najnowszy snapshot)

| Kontrola | Wynik |
|---|---|
| A–I, MongoDB jednowęzłowy | 23/23 punkty quota bez różnicy; 6 workerów, 40 rekordów: 40 akceptacji, 40 unikalnych |
| Parytet z SQLite, ten sam snapshot | Znormalizowane ślady identyczne: 208 linii |
| `purpose` opublikowanych plików wyników i błędów | `batch_output` w MongoDB i SQLite |
| Failover: wybory | 4 kadencje (inicjalna i 3 przejęcia), 0 rollbacków |
| Failover: wyniki | 720 akceptacji, 720 unikalnych rekordów, 0 innych wyników pozycji |
| Failover: błędy klienta | `Busy` ×3 przy commicie, ×1 przy rozliczeniu; każdy commit ponowiony tym samym operation id i przyjęty raz |
| Failover: ledger | 216/216 operacji zwraca przy ponowieniu `replayed=true` i identyczne wyniki |
| Failover: quota i cykl życia | 0 różnic po rozliczeniu i po cleanup; publikacja `Completed`, 720/720 |
| R1: wstrzymana pierwsza odpowiedź `ok: 1` | Driver ponawia `commitTransaction` raz i dostaje sukces; wynik zapisany raz; ponowienie operacji: `replayed=true` |
| R2: wstrzymane wszystkie odpowiedzi (8 × `ok: 1`) | Adapter zwraca `CommitUnknown`; serwer ma już 2/2 rekordy zakończone |
| R2: rekoncyliacja | Ten sam operation id: `replayed=true` dwukrotnie; nowy operation id: `AlreadyTerminal`; brak duplikatu; quota bez różnicy |
| R3: utracona odpowiedź commitu claimu | `CommitUnknown`; kolejny claim: `Idle` (rekordy zajęte na serwerze); po 3 s lease odzyskano 1 claim, 3/3 rekordy zatwierdzone raz; quota bez różnicy |
| Transakcje serwera w R1–R3 | 48 zatwierdzonych, 0 przerwanych, 0 otwartych |

Wcześniejszy snapshot dał te same wyniki A–I, R1–R3 i failoveru: 720/720,
216/216 identycznych powtórzeń, `Busy` ×2. Porządek przejęć w drugim przebiegu
był inny (zdarzenia obejmowały n1, n2, n3).

Uwagi dla właścicieli kodu:
- Komunikat `CommitUnknown` brzmi „reconcile with the same operation id” także
  przy claimie, który nie ma operation id i wraca przez wygaśnięcie lease.
  Zachowanie jest poprawne; dotyczy to tylko treści komunikatu.
- Clippy sterownika na najnowszym snapshotcie zgłasza wyłącznie nieużywany
  re-export `MssqlTls` w `store/mod.rs`. To sprawa integracji rodzica.

## Ograniczenia i decyzja

To dowód wybranych operacji natywnego adaptera, nie kwalifikacja wysokiej
dostępności. Trzy procesy na jednym hoście nie sprawdzają opóźnień sieci między
hostami, partycji sieci ani przesunięcia zegarów członków (`$$NOW` po zmianie
primary). Nie sprawdzają też awarii dysku, klastra shardowanego ani zachowania
przy obciążeniu produkcyjnym. W failoverze błędy klienta przyjęły postać `Busy`.
`CommitUnknown` pojawił się tylko w scenariuszach proxy, bo sam failover go nie
wywołał. Nie mierzono przepustowości ani czasu przełączenia. Wyniki nie dotyczą
HTTP, magazynu artefaktów, MSSQL, PostgreSQL ani GPU.

Wyniki dotyczą wyłącznie przypiętych snapshotów. Po przebiegu przegląd R1
potwierdził w DRR głodzenie kandydatów (F2) i niezwolnienie kosztu porzuconych
claimów (F4); poprawki w `store/**` są w toku u innego autora i nie są objęte
tym raportem. Parytet z najnowszym źródłem pozostaje jawnym GAP do końcowego
przebiegu rodzica po poprawkach.

Dalej potrzebne: replica set na osobnych hostach z partycją sieci i kontrolą
zegarów, sharding (jeśli ma być deklarowany) oraz ponowny przebieg po czystej
integracji i Clippy całego crate'a.

## Dane i odtworzenie

Dane są lokalne, nieopublikowane: `.cache/measurement-artifacts/batches-mongodb-failover-20261019/`.
Archiwum `measurement-data.tar.gz` ma 29 639 B i SHA-256
`bb9da12908b51287e6a6230a2a7aaae3c0500ee08b45e1946a56c9c7e35c5b69`. Zawiera
`SHA256SUMS` dla 21 plików pod `reports/batches-mongodb-failover-2026-10-19/`.
Nie utworzono URI HF.

| Ścieżka w archiwum | Zawartość |
|---|---|
| `latest-snapshot/mongodb-trace.txt`, `sqlite-trace.txt`, `parity-sqlite-vs-mongodb.txt` | A–I i parytet |
| `latest-snapshot/mongodb-failover-workload.txt`, `-timeline.txt`, `-elections.txt` | Obciążenie, oś czasu wyborów, liczba wyborów i rollbacków |
| `latest-snapshot/mongodb-replylost.txt`, `mongodb-replylost-txstats.txt` | R1–R3 i liczniki transakcji |
| `latest-snapshot/SOURCE_SHA256SUMS`, `earlier-snapshot/SOURCE_SHA256SUMS` | Manifesty źródeł |
| `earlier-snapshot/` | Te same scenariusze na wcześniejszym snapshotcie |
| `driver/` | Driver bez prywatnych ścieżek, `Cargo.toml`, skrypty `start-rs.sh`, `start-rs3.sh`, `chaos-rs3.sh` |

Odtworzenie:
1. Przygotuj binarkę jak w poprzednim raporcie.
2. Ślad A–I: `start-rs.sh <nazwa>`, potem `storeprobe admin initiate ZMIENNA_URI`
   z `BASAL_MONGO_MEMBER`, potem `storeprobe mongodb ZMIENNA_URI`.
3. Utracona odpowiedź: `BASAL_UPSTREAM=host:port storeprobe mongo-replylost` (proxy uruchamia się samo).
4. Failover: `start-rs3.sh <nazwa>`, potem inicjacja trzech członków przez
   `BASAL_MONGO_MEMBER=h1,h2,h3`, potem w tle `storeprobe mongo-failover ZMIENNA_URI`
   ze zmiennymi `BASAL_FAILOVER_RECORDS=720`, `BASAL_FAILOVER_WORKERS=6`,
   `BASAL_FAILOVER_PACE_MS=2000`, równolegle `chaos-rs3.sh <nazwa>`.

URI przekazuje się wyłącznie przez środowisko. Przed użyciem sprawdź
`SOURCE_SHA256SUMS` i sumę manifestu.
