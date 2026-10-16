# .NET SDK: odpowiedzi po kluczu i typowany obiekt

Data: 2026-10-13. SDK 0.1.0, przed publikacją. Raport uzupełnia
[wariant Python/TS](../sdk-dict-class-2026-10-13/README.md). Wcześniejsze raporty
i hashe pozostawiono bez zmian.

## Zmiana

Poprzednie `Answer(question)` w .NET identyfikowało wynik przez obiekt definicji
pytania i wymagało osobnego obiektu dla każdego ID. Teraz definicję można użyć
pod wieloma kluczami, a odpowiedź rozróżnia ID w konkretnym żądaniu.

- Wariant słownika zwraca mapę tylko do odczytu `Answers`, indeksowaną przez
  ordinalne klucze tekstowe. `Answer("department")` zwraca `AnswerResult`,
  który można zawęzić przez pattern matching. Nieznany klucz daje
  KeyNotFoundException. `Answer(question)` usunięto przed pierwszym wydaniem.
- Nowe przeciążenie `DecideAsync(state, q => new { ... })` zachowuje typ obiektu
  z fabryki. `Department = q.Add("department", routing)` tworzy `ChoiceField`,
  więc wynik czyta się przez `response.Answers.Department.Choice`, bez rzutowań
  i operatorów uciszających nullable. To samo działa z własną klasą zwróconą
  przez fabrykę. Nie potrzeba osobnej klasy odpowiedzi ani generatora kodu.
- Każdy `Add` wiąże definicję z kluczem i tworzy nowe pole odpowiedzi. Pola pięciu
  rodzajów udostępniają nieopcjonalne wartości oraz `.Result` z rekordem wyniku.
  Istniejące opcjonalne komunikaty act pozostają opcjonalne.
- Fabryka działa raz, synchronicznie, przed I/O. Builder zamyka się w `finally`.
  Pola są udostępniane przez wspólny punkt publikacji dopiero po walidacji całej
  odpowiedzi. Wcześniejszy odczyt, również po błędzie/anulowaniu, rzuca
  InvalidOperationException. Nie ma częściowo dostępnej odpowiedzi.
- Publiczny `QuestionSpec<TAnswer>` wymaga teraz `TAnswer : AnswerResult`,
  a `Parse` zwraca wspólny typ wyników. To również zmiana przed pierwszym wydaniem.

Fabryka powinna zwracać świeży obiekt z polami utworzonymi przez swój builder.
Nie należy przenosić pól z innego żądania ani używać buildera współbieżnie.
To kontrakt lokalnego kodu aplikacji, opisany w README. Nested DTOs i kolekcje
wyników nie są głęboko zamrożone. Namespace pozostaje `ITSOL.Basal`.
Nie zmieniono transportu, pozostałych SDK ani procedury publikacji.

## Kompilacja i lokalne diagnostyki

Biblioteka `net8.0`, przykład i paczka NuGet zbudowały się bez ostrzeżeń i błędów.
Gotową paczkę zainstalowano poza repozytorium do oddzielnego cache. Konsument
uruchamiał się na .NET 10.0.12 z Nullable i TreatWarningsAsErrors; nie uruchamiano
osobnego runtime .NET 8 w tej iteracji.

Sprawdzono anonimowy obiekt, własną klasę i słownik, pięć rodzajów wyników,
evidence oraz zgodność zbudowanego requestu z natywnym JSON. Lokalny handler
zwracał różne choice pod dwoma ID tej samej definicji; każda ścieżka zachowała
właściwy wynik. Sprawdzono osiem równoległych wywołań, jednokrotne wykonanie
fabryki, odrzucenie powtórzonego ID przed I/O, zamknięcie buildera także po
wyjątku fabryki, brak modyfikacji mapy odpowiedzi i nieznany klucz.

Brak pola wymaganego w każdym z pięciu rodzajów, zarówno dla klasy jak i
słownika, dawał BasalException: dziesięć przypadków. Po nieudanym żądaniu
przechwycone pola pozostawały nieczytelne, także te z poprawnymi wynikami.
Anulowanie zachowało OperationCanceledException i nie udostępniło pól.

Negatywny konsument został odrzucony z czterema oczekiwanymi błędami kompilacji:
pole Noul dla ChoiceField, nieistniejąca nazwa w anonimowym obiekcie,
przekazanie definicji zamiast klucza do Answer oraz przypisanie do Choice.
Nie stwierdzono dodatkowych ostrzeżeń. Diagnostyki pozostały poza repo;
nie dodano trwałych testów, frameworków ani bramek testowych CI.

## Działający Basal

Użyto istniejącego serwera przez tunel SSH z wcześniej potwierdzonym kluczem.
Trzy warianty z zainstalowanej paczki — anonimowy obiekt, własna klasa i słownik —
wysłały po sześć pytań obejmujących pięć rodzajów. Dwa ID korzystały ze wspólnej
definicji choice. Dane wejściowe były syntetyczne, z polskimi znakami i emoji.
Model: basal-1.5-mini.

**3 odpowiedzi HTTP, 18 wyników pytań, zero różnic** wobec bezpośredniego HTTP.
Porównano wszystkie klucze i wartości, liczby dokładnie jako bity IEEE-754
binary64; pominięto wyłącznie `usage.latency_ms`. Nie jest to ocena jakości
decyzji ani wydajności modelu. Nie zmieniano usługi ani konfiguracji serwera.
Po kontroli zamknięto tunel i potwierdzono brak jego lokalnego listenera.

Dowody: [request](request.json), [znormalizowana referencja](reference.json),
[znormalizowane wyniki](client-results.json), [podsumowanie](comparison-summary.json).

## Review i zależności

GPT-6 Astra XHigh zakończył odczytowy review bez Blocker/Should. Potwierdził
ordinalne ID, atomowe udostępnienie odpowiedzi oraz zamknięcie buildera.
Własna diagnostyka recenzenta na .NET 10 obejmowała różne wyniki dla `x`/`X`,
fabrykę uruchamianą raz, zamknięcie buildera, mapę tylko do odczytu, brak
częściowych wyników i anulowanie. Rodzic osobno sprawdził gotową paczkę i API.

Opus 5.5 XHigh otrzymał niezależny review tej poprawki. W chwili zapisu raportu
wynik pozostaje oczekiwany, podobnie jak jego poprzedni review Python/TS.
Nie uznajemy tych przeglądów za zatwierdzenie. Wyniki i ewentualne poprawki
zostaną zapisane osobno, bez nadpisywania tego pomiaru.

Metadane NuGet potwierdzają **zero zewnętrznych zależności runtime**, bez
assetów tools/build/buildTransitive. Nie dodano zależności build, reflection,
kompilowania wyrażeń ani generatora kodu. Collector i verifier potwierdziły
hash paczki, checker wersji potwierdził 0.1.0, `git diff --check` przeszedł.
[SHA-256 dystrybucji](distribution-SHA256SUMS) używa ścieżki względem sdk/.
Paczki nie opublikowano; konfiguracja kont i uprawnień rejestrów nadal jest
oddzielnym krokiem.
