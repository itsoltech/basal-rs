# SDK: słownik i deklaratywna klasa pytań

Data: 2026-10-13. SDK 0.1.0, przed publikacją. Nowy pomiar uzupełnia
[poprzedni raport](../sdk-typed-builders-2026-10-13/post-review/README.md).
Nie nadpisano wcześniejszych danych ani hashów.

## Zakres zmiany

Python przyjmuje `decide(state, questions)` w dwóch formach:

- `dict` / `Mapping`: sprawdzone odpowiedzi pod kluczami requestu,
  `response.answers[key]` albo `response.answer(key)`. Typ wartości to
  unia `AnswerResult`; konkretny wariant można zawęzić przez `isinstance`.
- Podklasa `Questions`, przekazywana bez tworzenia instancji: pola deklarowane
  przez `choice`, `noul`, `score`, `multi`, `act` wyznaczają typy pól odpowiedzi.
  `response.answers.department.choice` jest `str`, a `.urgent.noul` jest `float`.
  Użytkownik nie deklaruje osobnej klasy odpowiedzi ani adnotacji pól.

Obie ścieżki działają synchronicznie i asynchronicznie. Jedna definicja pytania
może wystąpić pod kilkoma kluczami, a odpowiedzi pozostają niezależne. Klasy
obsługują dziedziczenie i przesłanianie pól; nazwy zaczynające się od `_`
są zarezerwowane. SDK nie wywołuje konstruktorów użytkownika. Najpierw waliduje
wszystkie odpowiedzi, następnie umieszcza je pod kluczami instancji schematu.
Definicje są deskryptorami bez `__set__`, więc nie przechowują pojedynczej nazwy
ani odpowiedzi i mogą być współdzielone. Przypisanie i usunięcie pola instancji
odpowiedzi jest blokowane. Dane zagnieżdżone nie są głęboko zamrożone.

W TypeScripcie współdzielone definicje także mają niezależne wyniki po kluczu.
`response.answer("department")` zachowuje wywnioskowany typ, podobnie jak
`response.answers.department`. Klucze liczbowe są normalizowane do tekstu;
symbole w mapie pytań są odrzucane przed transportem. Jest to zmiana API przed
pierwszym wydaniem: Python/TS `answer()` przyjmuje klucz, zamiast obiektu pytania.
Surowe jednoargumentowe `decide(request)` pozostaje dostępne.

Nie zmieniano w tej iteracji C#, Rust, transportów ani konfiguracji publikacji.
W C#/Rust istniejący odczyt po tożsamości pytania nadal wymaga osobnych definicji
dla osobnych ID. Python nie wyprowadza literal union kluczy choice z dict;
wartość choice pozostaje `str`. Schematów klas nie należy mutować w trakcie żądań.

## Weryfikacja typów i zachowania lokalnego

Sprawdzano gotowy wheel i archiwum npm zainstalowane poza repozytorium:

- CPython 3.11.17, mypy 2.4.0 strict i Pyright 1.1.414 strict: poprawny konsument
  klasy, dziedziczenia, wspólnej definicji, pięciu rodzajów odpowiedzi, sync/async
  i słownika przechodzi bez błędów. Mypy strict przechodzi także dla źródeł SDK.
- Negatywny konsument Python: nieznane pole, niewłaściwy rodzaj odpowiedzi
  i dawne `answer(question)` są odrzucane. Mypy zgłasza cztery oczekiwane błędy;
  Pyright siedem diagnostyk dotyczących tych samych niepoprawnych odczytów.
- TypeScript 6.0.3: strict, noUncheckedIndexedAccess, exactOptionalPropertyTypes;
  inferencja kluczy, literal union choice, wspólna definicja i klucz liczbowy.
  Dyrektywy oczekiwanych błędów dla nieznanego klucza, dawnego obiektu pytania
  oraz odczytu pola niewłaściwego rodzaju zostały wykorzystane przez kompilator.
- Lokalne adaptery HTTP zwracały celowo różne odpowiedzi pod dwoma kluczami
  tej samej definicji. Python, npm ESM i CommonJS zachowały właściwe przypisanie.
  Sprawdzono snapshot mapy wejściowej, zgodność raw, brak wymaganego wyniku,
  niezależne odpowiedzi oraz specjalne klucze tekstowe JS.
- Python: osiem współbieżnych wywołań tej samej klasy nie mieszało odpowiedzi;
  dziedziczenie, przesłanianie, kolizje nazw z metadanymi wrappera i blokada
  modyfikowania odpowiedzi działały. Brak wymaganego pola każdego z pięciu
  rodzajów w obu formach powodował BasalProtocolError, łącznie dziesięć przypadków.
  Niepoprawne/puste schematy były odrzucane przed I/O.

Diagnostyki wykonano tymczasowo poza repo. Nie dodano testów, frameworków ani
bramek testowych do CI. Narzędzia do sprawdzania typów nie są zależnościami SDK.

## Porównanie z działającym API

Gotowe dystrybucje wywołały istniejący serwer Basal przez tunel SSH z wcześniej
potwierdzonym kluczem. Użyto syntetycznego tekstu z polskimi znakami i emoji,
sześciu pytań obejmujących pięć rodzajów, dwóch kluczy jednej definicji choice,
evidence noul, progu multi i macierzy kosztów act. Model: basal-1.5-mini.

**9 odpowiedzi HTTP, 54 wyniki pytań, zero różnic** wobec bezpośredniego HTTP.
Warianty: Python class/dict sync/async, Node ESM, adapter Request/Response,
CommonJS oraz przeglądarka z Fetch i adapterem. Porównano wszystkie klucze
i wartości; liczby porównano bitowo jako IEEE-754 binary64. Wyłączono wyłącznie
`usage.latency_ms`. To porównanie zachowania klientów, nie ocena jakości modelu
ani pomiar wydajności. Kontrola Node/przeglądarki została powtórzona po poprawce
klucza liczbowego w TS.

Przeglądarka: Chromium 152.0.7977.130, Electron 44.4.5. Ładowała moduł ESM
z zainstalowanej paczki. Lokalny helper przekazywał wyłącznie natywny endpoint
do tunelu i udostępniał pliki diagnostyki. Nie zmieniano serwera ani jego CORS.
Po pomiarze zamknięto tunel i helper oraz potwierdzono brak ich listenerów.

Dowody: [request](request.json), [znormalizowana referencja](reference.json),
[znormalizowane odpowiedzi](client-results.json), [podsumowanie](comparison-summary.json).

## Review i dystrybucje

GPT-6 Astra XHigh zakończył odczytowy przegląd źródeł i dokumentacji.
Znaleziony Should/medium: liczbowy klucz przechodził kompilację, ale wyszukiwanie
go w mapie o tekstowych kluczach kończyło się błędem. Poprawiono normalizację
klucza i ograniczenie sygnatury, dodano odrzucanie symboli. Recenzent potwierdził
poprawkę lokalnie; nie pozostawił otwartych Blocker/Should. Sprawdził też
deskryptory, diamond MRO, snapshoty, niezależność wyników, pomijanie konstruktorów
i anulowanie. Rodzic zweryfikował poprawkę w zbudowanej paczce.

Opus 5.5 XHigh otrzymał niezależny przegląd tej iteracji. **W chwili zapisania
raportu jego wynik pozostaje oczekiwany**; nie traktujemy go jako zatwierdzenia.
Wynik oraz ewentualne poprawki zostaną zapisane osobno, bez zmiany tego pomiaru.

Zbudowano wheel/sdist i npm ESM/CommonJS/deklaracje. Collector i verifier
potwierdziły trzy dystrybucje, checker wersji potwierdził 0.1.0, `git diff --check`
przeszedł. [SHA-256 dystrybucji](distribution-SHA256SUMS) podaje ścieżki względem sdk/.
Metadane wheel nadal wymagają jedynie HTTPX; npm ma zero zależności runtime.
Nie ma plików .pth w wheel ani hooków instalacyjnych npm. Nie dodano zależności
build/runtime, generatora, pluginu type-checkera ani dynamicznego wykonywania kodu.

Nie publikowano paczek. Konta i uprawnienia rejestrów pozostają do skonfigurowania.
Pozostałe granice audytu platform i grafu zależności są opisane we wcześniejszych
raportach. Pełny workspace nadal ma znany advisory unsafe-libyaml silnika, poza
niezależnym grafem SDK; nie osłabiano polityki zależności.
