# Rozdzielenie wydań silnika i SDK

Data: 2026-10-15. Zakres: workflow wydania oraz kontekst budowania Docker.

## Przyczyna i skutki

Utworzenie stron GitHub Release dla czterech SDK wygenerowało zdarzenia
`release: published`. Filtr `push.tags: v*` nie ogranicza osobnego zdarzenia
`release`. Joby macos/linux nie miały warunku, więc rozpoczęły budowanie silnika.
Job publish miał już filtr tagu silnika: publish, image i homebrew zostały
pominięte. Nie opublikowano nowego wydania silnika ani obrazu serwera.

Przykładowy nieprawidłowy przebieg:
[TypeScript SDK](https://github.com/itsoltech/basal-rs/actions/runs/38805382063).
Linux zakończył się na cargo-chef: workspace wymieniał sdk/rust, ale etap
planner kopiował wyłącznie crates. Ten sam brak występował w kolejnych etapach
budowania i w Dockerfile obrazu serwera.

## Poprawka

- macos/linux mają warunek na poziomie joba: ręczne uruchomienie albo tag v*.
  Zdarzenia release SDK pomijają joby przed przydzieleniem runnerów.
- publish dodatkowo odrzuca workflow_dispatch. Ręczny przebieg pozostaje
  budowaniem bez publikacji, również przy wybraniu tagu silnika.
- Pięć etapów planner/build/dev-build kopiuje sdk/rust, zgodnie z workspace.
  SDK nie jest nową zależnością linkowanego serwera; nie zmieniano zależności,
  obrazów bazowych, sekretów, uprawnień ani plików kopiowanych do runtime.

Zachowano release: published, aby nadal obsługiwać publikację wydania silnika
z interfejsu GitHub. To zdarzenie nie obsługuje filtra tagów jak push: w Actions
nadal może pojawić się przebieg dla SDK, ale wszystkie jego joby będą pominięte.
[Zdarzenia GitHub Actions](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release).
Historyczne przebiegi i niezmienne tagi SDK nie są przepisywane.

## Weryfikacja i przegląd

- actionlint 1.7.11 z ShellCheck oraz git diff --check: sukces.
- Izolowana diagnostyka oceniła aktualne wyrażenia if dla 11 kombinacji
  zdarzenia i ref: cztery release SDK pomijają joby; push/release v* buduje
  i dopuszcza publikację; ręczny przebieg buduje bez publikacji.
  [Macierz wyników](event-matrix.json).
- Odtworzono lokalnie pliki kopiowane przez pięć etapów Dockerfile.
  cargo metadata --locked --offline --no-deps odczytuje wszystkie cztery
  elementy workspace. Usunięcie sdk/rust reprodukuje błąd brakującego manifestu;
  przywrócenie katalogu usuwa ten błąd.
  [Wyniki etapów](docker-context-metadata.json).
- Self-review oraz przegląd granic supply chain/kontenera: filtr jest przed
  uruchomieniem runnerów; pominięcie propaguje się przez needs; ręczne uruchomienie
  nie uzyskuje ścieżki publikacji. Zmiana COPY dotyczy wyłącznie etapów budujących.

Nie dodano testów ani bramek testowych. Nie wykonano pełnego budowania obrazu:
sesja nie miała dostępu do demona Docker. Odczyt metadanych nie potwierdza
kompilacji obrazu CUDA ani zachowania runtime GPU. Nie zmieniano kodu Rust,
obliczeń modelu, wersji paczek ani istniejących wydań.
