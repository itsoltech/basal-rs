# Wspólna konfiguracja bazy danych — 2026-10-20

Wydzielono sekcję główną `database:` i moduł `basal-cli/src/database/`.
Backend, ścieżka SQLite, źródła poświadczeń, TLS, pula i `auto_migrate` nie
należą już do konfiguracji batchy. Zachowanie operatora opisuje
[DATABASE.md](../../docs/DATABASE.md), a użycie przez Batch API —
[BATCHES.md](../../docs/BATCHES.md).

## Zakres i wersja źródeł

Zmiana obejmuje osiem istniejących plików produktu i cztery nowe moduły Rust.
259 z 267 plików wcześniejszego obrazu produktu pozostało identycznych;
obecny manifest obejmuje 271 plików produktu, zasobów i budowania. Osobno
zmieniono dokumentację i trzy szablony konfiguracji. Porównaniem jest obraz
bezpośrednio sprzed wydzielenia, zawierający wcześniejsze niezatwierdzone
w Gicie prace, a nie `git HEAD`.

Nie zmieniono adapterów magazynu, SQL, ledgerów ani sum migracji, zależności,
obliczeń modelu, bramek GPU i schedulera. Dotychczasowe pomiary są związane
z ich historycznymi źródłami; ten raport nie rozszerza kwalifikacji numeryki,
wydajności ani topologii GPU.

Moduł bazy posiada konfigurację, opcje połączenia, wybór konta i uruchamianie
migracji. Aktualnie rejestruje schemat magazynu batchy, którego implementacja
pozostaje w jego adapterach. Konsumenci otwierają własne ograniczone pule;
nie wprowadzono jednej wspólnej puli dla całego procesu.

## Metoda i wyniki

Lokalne środowisko Linux x86_64, Rust 1.95.0, build release bez GPU.
Diagnostyka korzystała z rzeczywistego loadera YAML, `Database::prepare`,
`BatchService` i jednorazowych plików SQLite. Nie ładowano modelu.

| Sprawdzenie | Wynik i granica dowodu |
|---|---|
| Operator źródłowy | 38 obserwacji, exit 0; rzeczywisty start usługi SQLite oraz walidacja konfiguracji |
| Produkt CLI | 20/20 oczekiwanych kodów zakończenia; rodzic powtórzył wszystkie 20 na nowych danych |
| SQLite | automatyczne utworzenie i v1→v2; tryb ręczny nie tworzy brakującego pliku i nie zmienia v1; aktualny v2 łączy się |
| Nowszy schemat | v3 odrzucony; niezależny odczyt zachował ledger 1,2,3 |
| Wspólny uchwyt | udana migracja zapamiętana; kolejne `prepare` nie tworzy ponownie usuniętego pliku; nowy uchwyt może go utworzyć |
| Opcjonalność | brak sekcji i wyłączone batche nie wybierają bazy; jawnie wyłączona baza z włączonymi batchami daje błąd konfiguracji |
| Administracja | `database migrate` działa bez modeli/batchy; migruje również przy `auto_migrate: false`; `batches install` wymaga aktualnego schematu |
| Konfiguracja zdalna | trzy silniki × pięć wariantów; domyślnie automigracja wyłączona, włączenie wymaga osobnego źródła konta migracyjnego |
| TLS i sekrety | PostgreSQL z parametrem osłabiającym TLS odrzucony przed połączeniem; komunikat i Debug nie zawierają fikcyjnego sekretu; brak fallbacku konta migracyjnego |
| Kontrole Rust | fmt, release Clippy wszystkich targetów i prywatny Rustdoc z `-D warnings`: build domyślny i Intel; build produktu, odświeżony cargo-deny, polityka paste i diff-check: exit 0 |

Rodzic sprawdził sumy obrazu wejściowego, wszystkich 20 zmienionych/nowych
plików i pozostałych plików produktu. Przyrostowy patch odtworzono bez fuzz.
Operator różni się od produktu tylko hookiem uruchomienia, jednym modułem
diagnostycznym oraz komentarzem w `config.rs`. Zwykły produkt nie zawiera hooka.
Ledgery odczytano niezależnie przez Python `sqlite3`.

Nie uruchamiano ani nie dodawano testów, frameworków, snapshotów lub bramek CI.
Nie zmieniano produkcji, checkoutów upstream ani wcześniejszych raportów.

## Ograniczenia i decyzja

Nie uruchomiono rzeczywistych PostgreSQL, SQL Server ani MongoDB dla tej
zmiany. Dowód dotyczy ich konfiguracji i nie potwierdza nowych połączeń,
równoległych migratorów ani uprawnień kont na serwerze. Walidacja wymusza
różne źródła poświadczeń; operator odpowiada za rzeczywiście odrębne konta
i ich uprawnienia.

Gałąź `serve()` wykonująca migrację w tle przy bazie bez konsumenta została
przejrzana, ale nie uruchomiona end-to-end z modelem. Obserwowano bezpośrednio
używane przez nią `prepare`. Nie wykonywano CUDA/Metal ani pomiarów modeli.
Brak tych pomiarów nie jest deklaracją zgodności lub braku regresji runtime.

Zmiana przyjęta w opisanym zakresie. Stare `batches.storage` i
`batches.auto_migrate` wymagają przeniesienia do `database:`; program odrzuca
je z komunikatem. Lokalny start bez sekcji bazy zachowuje dotychczasową
ścieżkę `batches.sqlite3`. Dla wielu replik pozostaje zalecana jedna ręczna
migracja podczas wdrożenia i `auto_migrate: false`.

## Archiwum i odtworzenie

Archiwum jest **lokalne, nieopublikowane**. Nie potwierdzono pobrania z HF.

- Plik: `.cache/measurement-artifacts/batches-database-extraction-20261020/measurement-data.tar.gz`.
- SHA-256: `c952b3fac953067d14ee3e41c35673cb503f4a5ee54ce443fe356e224ead069b`.
- Rozmiar: 1 720 037 B. Nowe rozpakowanie zweryfikowało 344/344 sumy i kompletność manifestu.
- Planowane URI po osobno autoryzowanej publikacji: `hf://buckets/itsoltech/basal-rs/measurements/batches-database-extraction-20261020/c952b3fac953067d14ee3e41c35673cb503f4a5ee54ce443fe356e224ead069b/measurement-data.tar.gz`.

Dane pod `reports/batches-database-extraction-2026-10-20/` w archiwum:
`observations.jsonl`, `child-cli/`, `parent-cli/`, ledgery, `checks.json`,
logi z usuniętymi ścieżkami sesji, `scripts/`, `source/before/`, manifesty,
`incremental.diff`, `parent-operator.diff` i szczegółowe `METHOD.md`.
Bez sekretów sesji, tokenów, plików baz, binarek i identyfikatorów maszyny.

Przed rozpakowaniem sprawdź SHA-256 archiwum, a po rozpakowaniu uruchom
`sha256sum -c SHA256SUMS` z jego katalogu głównego. `METHOD.md` opisuje
odtworzenie produktu z obrazu wejściowego i patcha, a następnie prywatnego
operatora z osobnego patcha. Rodzic odtworzył i sprawdził wszystkie 271
plików produktu oraz 272 pliki operatora. Ponowne uruchomienie wymaga
nowych katalogów danych; dotychczasowych dowodów nie należy zastępować.
