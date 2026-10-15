# Poprawki po review Opusa: Python dict/class

Data: 2026-10-13. SDK 0.1.0, przed publikacją. Ten raport uzupełnia
[pierwotny pomiar](../README.md), bez zmiany wcześniejszych danych i hashów.

Opus 5.5 XHigh zakończył przegląd Python/TypeScript. Znalazł jeden problem
średniej wagi i dwie uwagi niskiej wagi. Rodzic odtworzył problem i sprawdził
poprawki w ponownie zbudowanej paczce.

## Ustalenia i rozstrzygnięcie

- **Medium: definicje poza schematem.** Deskryptor QuestionSpec traktował odczyt
  przez instancję zwykłej klasy jak odczyt niewypełnionego schematu Questions.
  `class Router: routing = choice(...)` i `Router().routing` kończyły się
  AttributeError; zbyt wąski typ owner powodował również błędy statyczne przy
  ClassVar, polach instancji i dataclasses. Rozszerzono typ owner, dodano
  przeciążenie zwracające definicję dla zwykłego obiektu oraz ograniczono
  zachowanie schematu do instancji Questions. Odczyt klasy schematu nadal daje
  definicję, a wypełnionej instancji schematu — konkretny typ odpowiedzi.
- **Low: copy/pickle.** Udokumentowano, że typowane kontenery odpowiedzi nie
  obsługują protokołów kopiowania/pickle. Do cache i komunikacji między procesami
  służy natywny JSON `response.raw`; do niezależnej kopii danych
  `copy.deepcopy(response.raw)`. Nie dodawano protokołu odbudowy obiektów ani
  nie osłabiano ochrony mapy odpowiedzi przed zapisem.
- **Low: dziedziczenie.** Dokumentacja opisuje kolejność zgodną z odwróconym MRO,
  pozycję zachowywaną przez nadpisane pole oraz nadpisywanie tym samym rodzajem
  pytania dla zgodności z mypy i Pyright. Mypy odrzuca zmianę rodzaju pola oraz
  usunięcie go przez None, nawet jeśli Pyright akceptuje takie przypisanie.

Review TypeScriptu po poprawce kluczy liczbowych nie wykazał nowych ustaleń.
Opus potwierdził odczyt po kluczach, symbole, specjalne nazwy właściwości,
brakujące wyniki, wymagane typy i brak nowych zależności. Nie przeglądał
równoległej poprawki .NET; ma ona osobny przegląd.

## Sprawdzenie poprawki

Wheel i sdist zbudowano ponownie przypiętym Flit Core. Wheel zainstalowano
w środowisku konsumenta CPython 3.11.17. Bezpośredni odczyt Router().routing
odtwarzał błąd przed zmianą, a po niej zwraca oryginalną definicję.

- Mypy 2.4.0 strict: źródła SDK przechodzą bez błędów.
- Mypy i Pyright strict: konsument z klasą namespace, ClassVar, polem instancji,
  dataclass i schematem Questions przechodzi bez błędów. Assert_type potwierdza
  ChoiceQuestion poza schematem i ChoiceAnswer/str w odpowiedzi.
- Wcześniejszy konsument pięciu rodzajów odpowiedzi, dziedziczenia, wspólnej
  definicji, sync/async i słownika nadal przechodzi obydwa checkery.
- Diagnostyka gotowej paczki potwierdza niezależne odpowiedzi wspólnej definicji,
  wymagane pola parserów, odrzucenie błędnych odpowiedzi, snapshot mapy pytań,
  równoległe wywołania async, ograniczenia modyfikacji oraz zgodność raw API.
- Osobno sprawdzono opisaną kolejność MRO/nadpisania oraz kopiowanie i serializację
  natywnego JSON. Collector/verifier potwierdziły dwa hashe dystrybucji;
  checker wersji i git diff --check przeszły.

Kontrola Pyright dotyczy konsumentów. Nie deklarujemy przejścia Pyright strict
dla całych źródeł SDK; Opus zgłosił tam diagnostyki dotyczące defensywnych kontroli
JSON, a źródła w tej iteracji weryfikowano przez mypy strict.

Nie dodano zależności, trwałych testów ani nowych bramek CI. Diagnostyki wykonywano
tymczasowo poza repozytorium. Nie zmieniono HTTP, parserów ani payloadu API;
nie powtarzano pomiaru live, którego dane pozostają w raporcie nadrzędnym.
Nie twierdzimy, że Opus wykonał drugi przegląd poprawki — akceptacja opiera się
na odtworzeniu problemu i weryfikacji rodzica. Brak otwartych materialnych ustaleń
z tego review Python/TS; ograniczenie copy/pickle pozostaje jawnie udokumentowane.

[Aktualne SHA-256 dystrybucji](distribution-SHA256SUMS), ścieżki względem sdk/.
Paczki nie zostały opublikowane. Wynik osobnego review .NET przez Opusa pozostaje
oczekiwany w chwili zapisu raportu.
