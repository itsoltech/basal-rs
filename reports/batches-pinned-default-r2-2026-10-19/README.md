# Batch ingest: przypięty default katalogu po odzyskaniu walidacji — 2026-10-19

Status: **PASS w opisanym zakresie CPU/SQLite/filesystem**. Rzeczywista walidacja JSONL przyjęła
pierwszą linię z jawnym `basal-2.0-mini` i drugą bez `model`, mimo zmiany defaultu katalogu oraz
defaultu uruchomionego hosta. Prawdziwa mieszanka dwóch logicznych nazw modeli została odrzucona
na linii 2. Jest to osobny pomiar uzupełniający runtime GAP z wcześniejszego
[raportu native HTTP](../batches-native-http-r2-2026-10-19/README.md); wcześniejszego raportu ani
archiwum nie nadpisano.

## Zakres i wejścia

Próba używa rzeczywistych `BatchService`, routera, bounded `serve/http.rs`, store, SQLite,
filesystem artifacts i natywnego ingest/plannera CPU. Jedynymi rzeczywistymi wejściami modelu
były tokenizer/config/CALIBRATION/prompt **basal-2.0-mini**, bez wag. Nie wykonano inferencji
ani pracy GPU. Fingerprint jest tożsamością fixture'u CPU, nie pełnego modelu.

`synthetic-default` jest drugim profilem CPU opartym na **tych samych wejściach mini**, z inną
logiczną nazwą i znacznikiem fixture'u w numeric profile. Nie załadowano danych innego modelu.
Obie nazwy są wpisane do katalogu, a dotychczasowy fingerprint/planner mini pozostaje zgodny.
W tej próbie użyto rzeczywistych bounds planera CPU; nie nadpisywano `ValidatedRecord` bounds.

Źródła: robocza implementacja na bazie `beb856275fbcb811eff5ce4446479bd74dafa176`, Rust/Cargo
`1.95.0`, domyślne Linux features, build `--locked --release`. Pełne źródła przypina
`full-source-SHA256SUMS`; sam commit nie obejmuje wszystkich zmian roboczych. Sprzęt: CPU hosta
Linux x86_64 i lokalne I/O; to nie kwalifikacja sprzętu, RSS, throughput lub latencji produkcyjnej.

`source-provenance.json` i `private-fixture-patch.diff` dokumentują prywatne zmiany:

- zwalidowane `processing=false` zatrzymuje automatyczne discovery/validation;
- dodatkowe prywatne wrappery wywołują niezmienione ciała `ingest::discover` i `ingest::validate`
  oraz odczytują rzeczywiste `Shared.default_model`;
- dwa profile CPU używają tej samej kopii wejść mini;
- operator trzyma typed validation lease, zmienia katalog przez `Store::install_catalog`,
  zatrzymuje i uruchamia serwis z nowym defaultem; dopiero po zatrzymaniu starego hosta przesuwa
  termin jego nierozpoczętej walidacji w jednorazowej bazie.

Produkcji i źródeł implementacji nie edytowano. Nie dodano frameworka/testów/bramki CI.
To ręczna diagnostyka implementacyjna, nie trzeci niezależny review.

## Obserwacje: rzeczywisty JSONL i recovery

Plik 471 B został wysłany przez HTTP. Pierwszy rekord jawnie wybiera `basal-2.0-mini`, drugi
pomija `model`. HTTP upload/create odpowiedziały 200; utworzony batch przypiął katalog 1,
którego defaultem był mini.

| Etap | Obserwacja |
|---|---|
| Discovery | Rzeczywisty odczyt pierwszej linii rozwiązał target mini |
| Pierwsza validation lease | generation 2, ingested 0; pełna walidacja jeszcze nieuruchomiona |
| Zmiana katalogu | `Installed(Revision(2))`, default `synthetic-default` |
| Restart hosta | rzeczywisty `Shared.default_model=synthetic-default`; latest katalog 2 |
| Przypięcie batcha | nadal katalog 1, default `basal-2.0-mini` |
| Recovery | generation 3, stary fingerprint mini nadal zgodny |
| Próba starej generacji | pusty append zwrócił `LeaseStale` |
| Pełne `ingest::validate` | `Ok(())`, odczyt i seal rzeczywistego JSONL |
| Stan przez HTTP i SQL | `in_progress`, total 2, errors null, dokładnie 2 zapisane rekordy |

To rozdziela przypięty default od aktualnego katalogu i od zmienionego defaultu procesu.
Nie polega wyłącznie na zachowaniu starej konfiguracji hosta. Przesunięcie zegara dotyczyło
nierozpoczętej validation lease po zatrzymaniu starego procesu, bez równoległego starego I/O.

Negatywny plik używał jawnie mini na pierwszej linii i `synthetic-default` na drugiej.
Rzeczywisty ingest zakończył batch jako `failed` z `mixed_models` na linii 2. Obie logiczne
nazwy były wpisane do katalogu; nie był to przypadek brakującego modelu w katalogu.

Po obserwacji dodatni batch anulowano przez HTTP i zaczekano na lifecycle. Stan końcowy:
sesje 0, reserved 0, active batches 0, open sessions 0, pending settlements 0, staging 0,
`quota_drift` 0; fizyczne obiekty 2 odpowiadają 2 zachowanym plikom wejściowym. Użyte quota
metadanych i wejść pozostało naliczone. Oba hosty zatrzymały się poprawnie; końcowy join 1 ms.
Własny proces diagnostyczny zakończył się kodem 0 po 7,23 s. SQLite nie miał osobnego serwera.

## Istniejący kolektor tylko do odczytu

Na tym samym już uruchomionym batchu/listenerze wywołano istniejący
[`tools/batches/capture.py`](../../tools/batches/capture.py) dokładnie raz, z `--duration 3`
i `--interval 1`. Token miał mode 0600, nowy katalog danych 0700; token usunięto po próbie.
Nie uruchomiono dodatkowego serwera ani generatora pracy.

Kolektor zakończył się kodem 0. Zapisano 8 zdarzeń: cztery odczyty batcha 200 i cztery health
200, wszystkie bez błędu. Health potwierdził ready/accepting true, processing false, katalog 2,
zgodność obu profili oraz brak kwalifikacji GPU. To obserwacja lokalnego pollingu batch/health;
nie kwalifikuje downloadu plików terminalnych, redirectów, zdalnego origin ani TLS.

## Ograniczenia

- Recovery nastąpiło przed pierwszym zapisanym rekordem; nie badano wznowienia po częściowo
  zapisanej stronie 512 rekordów, wielokrotnych awarii lub równoległych validatorów.
- Katalog/default i host faktycznie się zmieniły, ale drugi profil jest jawnym aliasem tego
  samego fixture'u CPU, nie innym modelem z wagami.
- Brak pracy GPU, inferencji, zgodności numerycznej, RSS, throughput i kwalifikacji produkcyjnej.
- Inne adaptery DB, S3 i inne modele nie są objęte tym pomiarem.
- Wrappery kontroli są wyłącznie w prywatnej kopii. Natywne discovery/validation pozostały
  niezmienione; `processing=false` i ręczne lease sterowały kolejnością diagnostyki.

W opisanym scenariuszu runtime luka pinned-default recovery jest uzupełniona; szersze warianty
odzyskiwania nie otrzymują automatycznie statusu PASS.

## Artefakty i odtworzenie

Nowe lokalne archiwum:
`.cache/measurement-artifacts/batches-pinned-default-r2-20261019/measurement-data-final.tar.gz`
(33 593 B), SHA-256:

```text
a7117014276b009a4cf8524c464821a861710ad2279e64fb4c9abc1153947880
```

**Nie opublikowano na HF.** Przewidziany docelowy URI, jeżeli później zostanie wysłane i
niezależnie zweryfikowane:
`hf://buckets/itsoltech/basal-rs/measurements/batches-pinned-default-r2-20261019/a7117014276b009a4cf8524c464821a861710ad2279e64fb4c9abc1153947880/measurement-data.tar.gz`.
URI nie potwierdza dostępności; nie wykonano pobrania HF ani anonimowego.

Autor sprawdził świeże lokalne rozpakowanie i **11/11** wpisów `SHA256SUMS`; osobna weryfikacja
rodzica pozostaje oddzielna. Lokalne archiwum zachowano. W archiwum ścieżki zaczynają się od
`reports/batches-pinned-default-r2-2026-10-19/`:

- `pinned-r1.log`, `pinned-observations-r1.jsonl`, `final-state.json`: pełne obserwacje i cleanup;
- `capture-observations.jsonl`: osiem rzeczywistych odczytów istniejącego kolektora;
- `metadata.json`, manifest źródeł/provenance/model-input SHA, fixture patch: wersje i zmiany;
- `driver/trace-pinned-r1.rs`, `METHOD.md`: portable driver i dokładne komendy.

Portable driver zmienia tylko neutralny zapis loopback bind i syntetyczny Host header;
oryginał użyty do kompilacji pozostaje lokalnie i jest przypięty manifestem. Ta różnica jest
jawna w `metadata.json`. Nie nadpisano wcześniejszego archiwum; wstępny, nieopublikowany pakiet
z literówkami w opisie komend pozostał lokalnie, a raport wskazuje nowy plik `-final`.

Archiwum rozpakuj do nowego katalogu i sprawdź SHA-256 archiwum oraz wszystkie `SHA256SUMS`.
`METHOD.md` opisuje zastosowanie prywatnego patcha w nowej kopii przypiętych źródeł, kompilację
oraz uruchomienie z nowym katalogiem i wejściami CPU mini bez wag. Dalsze udostępnienie wymaga
[procedury artefaktów pomiarowych](../../docs/MEASUREMENT_ARTIFACTS.md).

## Niezależna kontrola rodzica

2026-10-19: niezależnie sprawdzono SHA-256 archiwum `a7117014276b009a4cf8524c464821a861710ad2279e64fb4c9abc1153947880`
oraz wszystkie **11/11** wpisów `SHA256SUMS` po nowym rozpakowaniu.
Opisane wyniki i ograniczenia pozostają bez zmian. Weryfikacja jest lokalna;
nie potwierdza publikacji ani pobrania HF.
