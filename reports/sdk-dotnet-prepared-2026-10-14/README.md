# .NET SDK: pytania przygotowane wcześniej

Data: 2026-10-14. SDK 0.1.0, przed publikacją. Ta iteracja uzupełnia
[automatyczne ID](../sdk-dotnet-inferred-2026-10-13/README.md); wcześniejsze raporty pozostają bez zmian.

## Zachowanie

`QuestionBuilder.Prepare(q => new { Department = q.Choice(...), Urgent = q.Noul(...) })`
tworzy szablon raz, po zakończeniu odczytu danych. Następnie
`client.DecideAsync(state, new { Department = questions.Department })`
wybiera pytania i zwraca dokładnie typowany `response.Answers.Department.Choice`.
Obsługiwane są cały anonimowy szablon, wybór pól, zmiana nazw, powtórzenie definicji
pod różnymi nazwami i składanie pól z kilku szablonów.

Każdy request dostaje nowe pola i osobny punkt publikacji odpowiedzi. Szablon
nigdy nie przechowuje wyników, także przy współbieżnych wywołaniach i błędzie
requestu. Wbudowane definicje zachowują kopię kryteriów. Odczyt wyniku z szablonu
oraz użycie pól wcześniejszej odpowiedzi jako definicji są odrzucane.

Metoda przyjmująca obiekt jest rozszerzeniem w `ITSOL.Basal`, dzięki czemu istniejące
przeciążenia słownika i fabryki zachowują pierwszeństwo. Nie dodano zależności,
generatora, kompilacji dynamicznej ani zapisu do prywatnych/readonly pól.
Refleksja odtwarza kształt przez jeden pasujący publiczny konstruktor. Obsługuje
anonimowe obiekty i positional records; klasy zależne od buildera mogą wybierać
pola do obiektu anonimowego. Kolejność pytań wynika z parametrów konstruktora.

Konstruktor musi zachować podane pola pod odpowiadającymi im nazwami; zamiana
lub zastąpienie pól powoduje ArgumentException przed HTTP. Typ zmiennej musi
odpowiadać konkretnemu typowi obiektu, aby nie pominąć pól typu pochodnego.
Metadane parametrów konstruktora są kopiowane płytko i ponownie przechodzą przez
konstruktor: powinien przypisywać je bez przekształceń. Pozostałe członki metadanych
nie są kopiowane. Gettery pól powinny zwracać stabilne wartości. Jawne ID z Add
pozostają jawne i nie można powtórzyć jednego ID przez dwa aliasy.

## Review i poprawki

GPT-6 Astra XHigh zakończył odczytowy review bez Blocker/Should. Sprawdził między
innymi niezależność odpowiedzi, kompozycję, konstruktor record i odrzucanie obcych
pól oraz niejednoznacznych konstruktorów.

Opus 5.5 XHigh zakończył niezależny review i wskazał dwa przypadki cichej zmiany
kształtu: konstruktor zamieniający pola oraz zmienną typu bazowego gubiącą pytania
klasy pochodnej. Dodano kontrolę tożsamości odtworzonych pól oraz zgodności typu
konkretnego. Rodzic potwierdził odrzucenie obu przypadków przed HTTP w gotowej paczce.

Pozostałe uwagi dotyczyły komunikatów i dokumentacji. Doprecyzowano zakres kopiowania
metadanych, jawne ID, cache pól z Prepare versus inline, `cancellationToken: token`
i odrzucanie async Prepare przez kompilator. Dodano brakujący przykład inline oraz
osobne komunikaty dla definicji zamiast pól i nieodpowiedniego słownika.
Poprawki po review Opusa zweryfikował rodzic; nie wykonywano drugiej rundy recenzentów.

Recenzenci korzystali z tymczasowych projektów i lokalnych handlerów, bez serwera.
Opus dodatkowo włączył analizatory trim/AOT w kopii: nie zgłosiły uwag dla nowego
fragmentu, ale zgłaszały istniejące wywołania serializacji JSON. Nie deklarujemy
zgodności całego SDK z trimming/NativeAOT ani wykonania takiej publikacji.

## Sprawdzenie końcowej paczki

Bibliotekę net8.0, przykład i NuGet zbudowano bez ostrzeżeń i błędów. Paczkę
zainstalowano poza repozytorium do nowego cache. Konsument C# 12 z nullable
i TreatWarningsAsErrors działał na .NET 10; osobnego runtime .NET 8 nie uruchamiano.

Diagnostyka potwierdziła jednorazowe Prepare, zamrożenie kryteriów, wybór/zmianę
nazw/powtarzanie/łączenie pól, anonimowy obiekt i record, wybór pól klasy zależnej
od buildera, osiem niezależnych równoległych odpowiedzi i reuse po błędzie protokołu.
Nieprawidłowe kształty, zamienione pola konstruktora, typ bazowy, null i pola
wcześniejszej odpowiedzi nie wysłały HTTP. Zachowano przeciążenia słownika i inline.
Kompilator odrzucił nieznane pole, błędny rodzaj odpowiedzi oraz async Prepare:
[diagnostyki](compiler-rejections.json), [podsumowanie](verification-summary.json).

NuGet ma zero zewnętrznych zależności i nie zawiera install/build hooks. README
w paczce odpowiada aktualnemu README. Sprawdzono hashe artefaktu, wspólną wersję
SDK oraz git diff --check. [Hashe źródeł i paczki](SHA256SUMS) są względem katalogu repozytorium.
Nie dodano trwałych testów, frameworków ani bramek testowych CI.

## Działające API

Końcową paczkę po poprawkach sprawdzono na istniejącym Basalu przez SSH
z wcześniej potwierdzonym kluczem. Pełny szablon anonimowy, wybrany obiekt anonimowy
i positional record wysłały po sześć pytań z pięciu rodzajów, w tym dwa ID tej samej
definicji choice. Dane syntetyczne zawierają polskie znaki i emoji; model basal-1.5-mini.

**3 odpowiedzi HTTP, 18 wyników pytań, zero różnic** względem bezpośredniego HTTP.
Porównano wszystkie klucze i wartości; liczby bitowo jako IEEE-754 binary64.
Wyłączono wyłącznie usage.latency_ms. Nie jest to pomiar jakości ani wydajności modelu.
Nie wykonywano rzeczywistego odczytu z bazy; konfigurację pobrano z lokalnego fixture.
Nie zmieniano serwera, nie publikowano paczki. Tunel zamknięto i potwierdzono brak listenera.

Dane: [request](request.json), [referencja](reference.json),
[wyniki paczki](client-results.json), [porównanie](comparison-summary.json),
[referencja po normalizacji](normalized-reference.json),
[wyniki po normalizacji](normalized-results.json).
