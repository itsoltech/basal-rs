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
  a wydajność pomiarami. Opis wyników i ograniczeń zapisuj w `reports/`;
  dane pomiarowe przechowuj według zasad poniżej.
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

## Artefakty pomiarowe — Hugging Face

- Przed przygotowaniem lub publikacją wyników przeczytaj
  [docs/MEASUREMENT_ARTIFACTS.md](docs/MEASUREMENT_ARTIFACTS.md).
  To wspólna procedura archiwizacji, wysyłania, pobierania i weryfikacji.
- W Gicie przechowuj szczegółowe podsumowania Markdown w `reports/`.
  Opisuj prostym, precyzyjnym językiem: cel, wersje i wejścia, sprzęt,
  metodę, wyniki, ograniczenia oraz decyzje. Zachowuj regresje, różnice
  jakości i statusy FAIL/GAP/SIGNAL; nie wyciągaj niezmierzonych wniosków.
- Nowe dane wynikowe (JSON, JSONL, CSV, logi, profile i eksporty binarne)
  zapisuj lokalnie poza Gitem, w `.cache/measurement-artifacts/<nazwa>-<data>/`.
  W archiwum zachowuj logiczne ścieżki `reports/<katalog>/<plik>`.
- Tabele GEMM w `crates/basal-cli/gemm-tables`, referencje upstream i inne
  istniejące wejścia wymagane przez kod lub narzędzia pozostają w Gicie.
  Rozróżniaj wynik pomiaru od potrzebnego zasobu; rozszerzenie JSON samo
  w sobie nie jest powodem do usunięcia pliku ani ignorowania go w Git.
- Docelowy magazyn to publiczny HF Storage Bucket
  [itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs),
  URI `hf://buckets/itsoltech/basal-rs/`. Korzystaj z globalnego `hf` CLI;
  jeśli istnieje `~/.local/bin/hf`, wywołuj tę binarkę bezpośrednio.
  Sprawdź dostęp przez `hf auth whoami`. Przy braku logowania poproś
  użytkownika o `hf auth login` w tym samym środowisku, bez tokenów w czacie.
- Do publicznego bucketa wysyłaj tylko dane przeznaczone do publikacji:
  bez sekretów, identyfikatorów
  maszyn, adresów, nazw hostów, lokalnych ścieżek, kwot i kosztów.
  Prywatne surowe dane sesji zachowuj lokalnie.
- Utwórz `SHA256SUMS` plików i archiwum `tar.gz`; policz SHA-256 archiwum.
  Publikuj przez `hf buckets cp` pod nowym, pustym prefiksem
  `measurements/<nazwa>-<data>/<sha256 archiwum>/`.
  Bucket nie ma historii wersji: zakończonych prefiksów nie nadpisuj ani
  nie usuwaj. Korekta danych wymaga nowego archiwum i prefiksu.
- Po wysłaniu pobierz archiwum do osobnego katalogu i sprawdź jego SHA-256
  oraz wszystkie wpisy `SHA256SUMS`; dla publicznych danych potwierdź też
  pobranie bez autoryzacji. Dopiero wtedy usuń zbędne kopie wyników z Git
  i opisz publikację jako potwierdzoną. Zachowaj lokalną kopię archiwum.
- Raport Markdown musi podawać URI, SHA-256 archiwum, ścieżki danych w nim
  i sposób ich odtworzenia. Runnery, które czytają takie dane, przyjmują ich
  katalog przez `--measurement-artifacts`. To konwencja runnerów w `tools/`,
  nie opcja CLI basal. Szczegóły są w procedurze i `docs/CLOUD.md`.
