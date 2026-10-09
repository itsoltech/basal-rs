# Zasady pracy w repozytorium

Opis projektu: [README.md](README.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Jakość kodu Rust

- Przy zmianach ownership, publicznego API, błędów, async/współbieżności,
  `unsafe`, serializacji lub wydajności stosuj `itsolpowers:rust-implementation`.
  Przegląd tych zmian wykonuj według `itsolpowers:rust-review`, a przed
  zakończeniem stosuj `itsolpowers:itsol-self-review`.
  Zasady projektu, w tym ograniczenia testów i pomiarów, mają pierwszeństwo.
- Najpierw poprawność i czytelność. Optymalizacje, dodatkowe alokacje i kopie
  oceniaj na podstawie wymagań i pomiarów, bez komplikowania API na zapas.
- Dla danych tylko odczytywanych preferuj referencje i slice'y. Ograniczaj
  widoczność API, waliduj dane na granicach i dokumentuj invariants.
- W publicznym API bibliotek preferuj typowane błędy; w CLI `anyhow` z kontekstem.
  Zachowuj łańcuch przyczyn. Błędy wejścia i I/O nie mogą prowadzić do paniki;
  `unwrap`/`expect` wymagają jasnego invariantu.
- Każdy blok i impl `unsafe` wymaga komentarza `SAFETY:` opisującego warunki
  bezpieczeństwa, ownership, czas życia i synchronizację. Operacje unsafe
  wewnątrz unsafe fn zamykaj w jawnych blokach. Stosuj bezpieczne abstrakcje FFI.
- Nie blokuj executora Tokio ani nie trzymaj locków przez `.await`. Kontroluj
  limity kolejek, timeouty, lifecycle workerów, paniki i graceful shutdown.
- Wyjątki od lintów ograniczaj do najmniejszego zakresu i podawaj `reason`.
  Preferuj `#[expect(..., reason = "...")]`; nie włączaj całego `pedantic`
  ani globalnych wyciszeń. Każdy crate dziedziczy `[workspace.lints]`.
- Przed zakończeniem sprawdź zmienione granice według powyższych zasad oraz
  uruchom formatowanie, Clippy, lint dokumentacji i kontrolę zależności.
  Zakres CI i kontrole lokalne: [docs/CI.md](docs/CI.md). CI obejmuje kompilację
  i lint kodu Linux, Metal i CUDA, ale nie potwierdza zachowania runtime'u GPU
  ani wyników modelu.

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
- Pomiary na wynajętym GPU: [docs/CLOUD.md](docs/CLOUD.md). Przed każdą sesją
  zapytaj użytkownika o budżet i kartę; kwot ani kosztów sesji nie zapisuj w
  repozytorium. Po pomiarze usuń maszynę i pilnuj jej w trakcie pracy.
