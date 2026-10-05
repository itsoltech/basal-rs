# Zasady pracy w repozytorium

Opis projektu: [README.md](README.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Testy

- Na obecnym etapie projekt nie ma testów jednostkowych, integracyjnych,
  regresyjnych ani E2E. Nie dodawaj frameworków testowych, snapshotów ani
  bramek testowych do CI.
- Poprawność sprawdzamy porównaniem wyników modelu z referencją upstream
  (`basal export`, `basal compare`, [docs/BENCHMARKS.md](docs/BENCHMARKS.md)),
  a wydajność pomiarami. Zapisuj ich dane i ograniczenia w `reports/`.
- Można korzystać z kompilatora, `cargo fmt`, `cargo clippy` i przeglądu kodu.
  Brak testów nie uprawnia do deklarowania niezmierzonej zgodności, zachowania
  jakości ani przyspieszenia.

## Pomiary i raporty

- Nie nadpisuj istniejących raportów; nowy pomiar to nowy plik lub katalog.
- Zmiana, która nie powinna zmieniać obliczeń, musi dawać wynik bitowo równy
  poprzedniej wersji; zmiana numeryki wymaga porównania z FP32 upstream.
- Nie modyfikuj checkoutów upstream w `.baseline/`.
- Repozytorium nie zawiera danych pozwalających zidentyfikować maszyny, na
  których wykonano pomiary (adresy, ścieżki, nazwy hostów).
