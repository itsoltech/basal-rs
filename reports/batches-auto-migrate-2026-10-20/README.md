# Konfigurowalne migracje przy starcie — 2026-10-20

Dodano `batches.auto_migrate`. Pominięta flaga zachowuje dotychczasowe domyślne
wartości: `true` lokalnie, `false` w produkcji. Wyłączony moduł pomija walidację
zależności i start magazynu także przy `auto_migrate: true`.

Zmiana obejmuje `config.rs` i `service.rs`, dokumentację oraz szablony. Nie zmienia
skryptów migracyjnych, obliczeń modelu, priorytetów online, kontraktu HTTP ani
zależności. Porównano wszystkie 267 plików wejściowego drzewa: 265 bez zmian,
dwa zmienione. Źródła po wcześniejszym uproszczeniu Opusa są punktem odniesienia;
historyczne raporty GPU zachowują własne wersje kodu.

## Zachowanie

- `true`: migracja schematu przed opublikowaniem połączenia workerom batchowym.
  Zdalne bazy wymagają osobnego źródła konta migracyjnego; następnie serwer
  otwiera połączenie kontem runtime. Brak źródła jest błędem konfiguracji.
- `false`: sprawdzenie istniejącego schematu bez tworzenia pliku SQLite i DDL.
  Jawne `basal batches migrate` nadal może utworzyć lub zaktualizować schemat.
- Produkcyjne quota i katalog pozostają jawnie instalowane przez administrację.
  Lokalny bootstrap zachowuje dotychczasowe zachowanie. Nie ma fallbacku bazy.
- PostgreSQL i MSSQL korzystają z istniejących blokad migratorów. MongoDB ma
  wznawialne DDL; należy koordynować jeden migrator. Blokada migracji nie zapewnia
  zgodności ze starszymi replikami API; dla wielu replik domyślny tryb jest ręczny.

## Rzeczywiste obserwacje CPU / SQLite

Użyto prawdziwego `BatchService`, migratora SQLx z dołączonym SQLite i jego
workerów, bez modeli, GPU, sieci ani zewnętrznych baz. Prywatne wejście CLI
tylko uruchamia operator; pliki konfiguracji i usługi są identyczne z produktem.

| Przypadek | Obserwacja |
|---|---|
| Local, flaga pominięta | automatycznie utworzony schemat v2, magazyn podłączony |
| Jawne `true`, brak pliku | schemat v2, magazyn podłączony |
| `false`, brak pliku | brak połączenia i brak utworzonego pliku |
| `false`, schemat v1 | brak połączenia; pozostaje v1 |
| `true`, schemat v1 | aktualizacja do v2, magazyn podłączony |
| `false`, schemat v2 | połączenie bez migracji |
| `true`, nieznana wersja v3 | odmowa `TooNew`; pozostają wpisy v1, v2, v3 |
| Moduł wyłączony, `true` | walidacja zwraca `None`, bez ustawień/startu modułu |
| Jawne CLI przy `false` | tworzy brakującą bazę i schemat v2, exit 0 |

Niezależny odczyt plików SQLite potwierdził wersje. Dla każdego z trzech zdalnych
backendów sprawdzono cztery konfiguracje: pominięcie i jawne `false` dają tryb
ręczny; `true` bez konta migracyjnego jest odrzucane; `true` z osobnym źródłem
konta jest przyjmowane. To **12 prób walidacji konfiguracji**, bez odczytu
poświadczeń i bez połączenia ze zdalną bazą. Operator wyemitował 20 obserwacji,
zakończył się kodem 0 i zamknął wszystkie uruchomione przez siebie usługi.

## Kontrole i ograniczenia

Formatowanie, release workspace Clippy all-targets oraz prywatny Rustdoc
z `-D warnings` dla default i Intel: PASS. Odświeżone `cargo deny`, kontrola
wyjątku `paste`, zwykły build produktu i `git diff --check`: PASS. Nie dodano
ani nie uruchamiano testów projektu, frameworków, snapshotów lub bramek CI.

Nie wykonano nowej migracji startowej na PostgreSQL/MSSQL/MongoDB ani próby
równoczesnego startu replik. Ponownie użyte natywne migratory mają wcześniejsze,
opisane osobno dowody; nie rozszerzamy ich na nowy przebieg startowy. Nie badano
CUDA/Metal, numeryki ani wydajności modelu. Produkcji nie zmieniano.

Dane i dokładny kod operatora pozostają w lokalnym archiwum; sposób odtworzenia
opisuje jego `METHOD.md`. Nota integralności poniżej identyfikuje zamknięty pakiet.


## Integralność i odtworzenie

Lokalne archiwum: `.cache/measurement-artifacts/batches-auto-migrate-20261020/measurement-data.tar.gz`,
1,626,647 B. SHA-256: `2fc6a79289fe76700e4e94c0d34bbfbf01d6249c3478fb131c630f07d79979c7`.
Świeże rozpakowanie potwierdziło **298/298** sum i pełne pokrycie plików,
w tym 268/268 źródeł operatora. Dane: `reports/batches-auto-migrate-2026-10-20/`
wewnątrz archiwum; instrukcje w `METHOD.md`, wyniki w `observations.jsonl`
i `native-summary.json`, provenance i kontrolne logi w tym samym katalogu.

**Nie opublikowano w HF.** Docelowy URI przy ewentualnej późniejszej publikacji:
`hf://buckets/itsoltech/basal-rs/measurements/batches-auto-migrate-20261020/2fc6a79289fe76700e4e94c0d34bbfbf01d6249c3478fb131c630f07d79979c7/measurement-data.tar.gz`.
URI nie potwierdza dostępności; nie wykonywano pobrania zdalnego ani anonimowego.
Przed odtworzeniem sprawdź SHA archiwum i wszystkie wpisy `SHA256SUMS`. Zachowano
lokalne archiwum i prywatne surowe dane; nie nadpisano wcześniejszych raportów.
