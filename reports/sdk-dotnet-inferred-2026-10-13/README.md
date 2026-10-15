# .NET SDK: uproszczone fabryki i nazwy pól jako ID

Data: 2026-10-13. SDK 0.1.0, przed publikacją. Nowa iteracja uzupełnia
[wcześniejsze review .NET](../sdk-dotnet-keyed-2026-10-13/post-review/README.md).
Poprzednie raporty i hashe pozostają bez zmian.

## Zachowanie

Domyślny przykład używa teraz `Department = q.Choice(...)`, `Urgent = q.Noul(...)`
i analogicznych fabryk Score/Multi/Act. Nazwa zwracanej publicznej właściwości
lub pola jest dokładnym ID requestu, z zachowaniem wielkości liter. Nie trzeba
powtarzać jej w osobnym stringu. Typ odpowiedzi pozostaje zachowany, np.
`response.Answers.Department.Choice` jest string, a `.Urgent.Noul` jest double.

Fabryki obsługują instrukcje, kryteria i typowane opcje oraz przyjmują istniejącą
definicję, np. `Department = q.Choice(routing)` i `SecondDepartment = q.Choice(routing)`.
Każde wywołanie tworzy osobne pole; wspólna definicja nie łączy odpowiedzi.
Własne definicje generyczne można przypisać przez `q.From(definition)` i czytać
`.Result`. Anonimowe obiekty, własne klasy, records i structs zachowują kształt
odpowiedzi. Słownik nadal udostępnia sprawdzone rekordy po kluczach.

SDK odczytuje nazwy z publicznych członków typu zwróconego przez synchroniczną
fabrykę, używając wbudowanej refleksji .NET. Czyta tylko członków o typie pola
odpowiedzi; pozostałe gettery nie są uruchamiane. Getter odpowiedzi jest odczytywany
raz, przed HTTP. Kolejność pytań wynika z tworzenia pól, nie z kolejności odczytu
metadanych. Adnotacje DynamicallyAccessedMembers zachowują publiczne pola/właściwości
w łańcuchu metod; jest to standardowy mechanizm opisany w
[dokumentacji Microsoft](https://learn.microsoft.com/en-us/dotnet/core/deploying/trimming/fixing-warnings).

Pola spoza bieżącego buildera, puste/null pola, pominięte pola automatyczne,
alias jednego pola pod różnymi nazwami i kolizje ID są odrzucane przed I/O.
Jawne `q.Add("custom_id", definition)` pozostaje dostępne dla niestandardowego ID
i ma pierwszeństwo przed nazwą członka. Kontrola wykrywa także ponowne użycie
pola z wcześniejszego requestu w publicznych typowanych członkach kształtu.
Wszystkie wyniki są publikowane atomowo po sprawdzeniu całej odpowiedzi.

## Review

GPT-6 Astra XHigh zakończył odczytowy review bez Blocker/Should. Własna diagnostyka
potwierdziła wielkość liter, kolejność tworzenia, publiczne pola i dziedziczenie,
jednorazowy getter, niezależność wspólnej definicji oraz odrzucenie aliasów,
obcych/pominiętych pól, kolizji ID i tworzenia pola w getterze.

Opus 5.5 XHigh zakończył niezależny review. Zgłosił jedną uwagę średniej wagi
i trzy uwagi niskiej wagi. Rodzic zweryfikował i rozstrzygnął je:

- **Krotki:** aliasy named tuples nie istnieją w runtime; wcześniejsza refleksja
  wysyłała Item1/Item2. Kształty implementujące ITuple są teraz odrzucane przed
  HTTP, jeśli zawierają automatyczne pola. Krotki z wyłącznie jawnymi ID pozostają
  zgodne z wcześniejszym API. README i SDK.md opisują ten warunek.
- **Wczesny odczyt:** kontrola zakończenia requestu poprzedza teraz odczyt ID,
  więc błąd wskazuje brak udanego DecideAsync, a nie brak nazwy pola.
- **Interfejsy/ukryte pola:** opisano użycie publicznych członków zwracanego typu
  i klas bazowych. Dziedziczenie interfejsów oraz ukryte pola o powtórzonych
  nazwach nie należą do obsługiwanego zakresu inferencji; niepowodzenie pozostaje
  lokalne, przed HTTP.
- **Dokumentacja:** główny przykład i README używają fabryk bez ręcznych ID.
  Doprecyzowano pierwszeństwo jawnego Add oraz dodano ten raport.

Recenzenci pracowali na źródłach/skopiowanych źródłach w tymczasowych katalogach,
bez publikacji i bez połączenia z serwerem. Opus dodatkowo uruchomił analizator
trim/AOT w kopii: nowy fragment nie wygenerował IL2xxx/IL3xxx, natomiast istniejące
wywołania System.Text.Json nadal je generują. Nie deklarujemy zgodności całego
SDK z trimming/NativeAOT ani wykonania takiej publikacji. Poprawek po ustaleniach
Opusa nie poddawano drugiemu review; zweryfikował je rodzic w gotowej paczce.

## Sprawdzenie gotowej paczki

Biblioteka net8.0, przykład i NuGet zbudowane bez ostrzeżeń i błędów. Paczkę
zainstalowano poza repozytorium do nowego cache. Konsument C# 12 z nullable
i TreatWarningsAsErrors uruchamiał się na .NET 10; nie wykonywano osobnego
runtime .NET 8.

Lokalna diagnostyka paczki potwierdziła pięć fabryk, kształt anonimowy,
dziedziczoną klasę i publiczne pola, rozróżnienie wspólnej definicji, zachowanie
wielkości liter, kolejność tworzenia niezależną od deklaracji członków,
jednorazowy getter i pomijanie innych getterów. Sześć niepoprawnych kształtów
zostało odrzuconych przed HTTP; osiem współbieżnych wywołań zachowało niezależność.
Sprawdzono też poprawny komunikat wczesnego odczytu i krotkę z jawnymi ID.

Diagnostyka wcześniejszej ścieżki zachowała zachowanie słownika, jawnego Add,
walidacji brakujących pól, zamknięcia buildera i anulowania. Negatywny konsument
odrzuca błędny rodzaj pola, nieznaną nazwę i zapis do pola readonly; async lambda
nadal jest odrzucana. [Diagnostyki kompilatora](compiler-rejections.json).
Nie dodano trwałych testów, frameworków ani bramek testowych CI.

## Działające natywne API

Po poprawce krotek i wczesnego odczytu aktualną paczkę ponownie sprawdzono
na istniejącym Basalu przez SSH z wcześniej potwierdzonym kluczem serwera.
Anonimowy obiekt, dziedziczona klasa i słownik wysłały po sześć pytań z pięciu
rodzajów, z dwoma ID jednej definicji choice. Dane były syntetyczne, z polskimi
znakami i emoji; model basal-1.5-mini. Inherited class w diagnostyce tworzyła
pola w konstruktorze po klasie bazowej, zachowując kolejność referencji.

**3 odpowiedzi HTTP, 18 wyników pytań, zero różnic** z bezpośrednim HTTP.
Sprawdzono wszystkie klucze i wartości, liczby bitowo jako IEEE-754 binary64;
wyłączono wyłącznie usage.latency_ms. Nie jest to ocena jakości ani wydajności
modelu. Nie zmieniano serwera. Tunel zamknięto i potwierdzono brak listenera.

Dowody: [request](request.json), [znormalizowana referencja](reference.json),
[znormalizowane wyniki](client-results.json), [podsumowanie](comparison-summary.json).

NuGet nadal ma **zero zewnętrznych zależności**, bez tools/build/buildTransitive.
Nie dodano generatora ani dynamicznej kompilacji. Collector/verifier, kontrola
wersji i git diff --check przeszły. README w paczce odpowiada aktualnemu źródłu.
[SHA-256 dystrybucji](distribution-SHA256SUMS), ścieżka względem sdk/.
Paczka pozostaje nieopublikowana.
