# Typowane fabryki pytań i odpowiedzi SDK

Data: 2026-10-13. SDK `0.1.0`, przed pierwszą publikacją.

## Dostarczone API

W czterech SDK dodano `choice`, `noul`, `score`, `multi`, `act` (PascalCase w C#)
oraz odpowiadające im konkretne typy pytań i opcje. Fabryki przechowują kopię
danych wejściowych. Choice/score wymaga 2–10 kryteriów/poziomów, multi co najmniej
jednego kryterium. Multi nie przyjmuje evidence. Act wymaga pełnej macierzy
skończonych kosztów pokrywającej wszystkie outcomes; bez criteria są to true/false.
Nie dodano zależności runtime ani zależności budowania.

Python/TS/C# przyjmują osobno state i mapę pytań. Rust przyjmuje state i ordered
builder `Questions`. Opcje routingu/facts mają osobny typ lub zamknięte argumenty
keyword. Mapę utrwala się przed I/O. Ta sama instancja pytania pod kilkoma ID jest
odrzucana, aby typowany dostęp przez obiekt pytania był jednoznaczny.

Odpowiedzi mają wymagane pola dla danego rodzaju pytania: choice/confidence,
noul/confidence, score/legend/confidence, selected/threshold/set_confidence,
action/outcome/expected_costs/confidence/calibration. Nie wypełnia się brakujących
pól fikcyjnymi zerami/null; wadliwa odpowiedź daje błąd SDK przed zwróceniem wyniku.
Brak evidence jest pustą kolekcją w typowanych wynikach obsługujących evidence;
natywna odpowiedź nadal zachowuje oryginalną obecność pól.

| Język | Typowany dostęp bez ręcznych castów/nonnull |
|---|---|
| TypeScript | `response.answers.department.choice`, literal union kryteriów; `response.answers.urgent.noul`; probabilities i koszty po znanych kluczach |
| Python | `response.answer(department).choice`, dataclass wywnioskowany z generycznego obiektu pytania |
| C# | `response.Answer(department).Choice`, wywnioskowany record z wymaganymi polami |
| Rust | `response.answer(&department)?.choice`, pożyczony widok z wymaganymi polami; `?` propaguje błąd obcego pytania, nie unwrapuje opcjonalnego pola |

TS może mapować statycznie znane klucze obiektu na indywidualne typy odpowiedzi.
Python/C#/Rust nie potrafią automatycznie utworzyć takich named properties z
dowolnego słownika; tam zachowuje się referencję do obiektu pytania. Pomyłka
polegająca na użyciu pytania spoza requestu jest jawnie odrzucana w runtime.
Nie deklarujemy, że kompilator tych języków udowadnia przynależność każdego
handle do konkretnego requestu. Rust lifetime wiąże pożyczony widok z odpowiedzią.

Namespace .NET: **`ITSOL.Basal`**, również RootNamespace projektu. Identyfikator
NuGet pozostaje `Itsoltech.Basal.Client`. `Questions.Create` i Choice/Multi z
named tuples pozwalają tworzyć mapę bez ręcznych generycznych adnotacji.

Surowy kontrakt nadal jest dostępny: `decide(request)` Python/TS,
`DecideAsync(request)` C#, **`decide_request(&request)` Rust**. Rust nie ma
przeciążania metod, więc wcześniejsze jednoparametrowe `decide` przeniesiono pod
tę nazwę. Wspólna wersja pozostaje 0.1.0, która nie była publikowana.
Helpery obejmują tekstowe instrukcje/opisy, ordered list score i pełną macierz act;
pozostałe natywne formaty/rozszerzenia nadal obsługuje raw API.

## Weryfikacja na działającym API

Zbudowane paczki Python/npm/NuGet zainstalowano w oddzielnych środowiskach
konsumentów. Rust korzystał z zawartości zweryfikowanego .crate poza workspace
silnika. Każdy wariant zbudował ten sam natywny request z pięcioma typami pytań,
polskim tekstem/emoji, evidence dla noul, progami/limitami multi i macierzą act.
Zserializowane helpery dały semantycznie ten sam JSON co fixture, z zachowaną
kolejnością kryteriów. Odpowiedzi raw porównano z bezpośrednim API, pomijając
wyłącznie usage.latency_ms i sprawdzając dokładne bity binary64.

| Wariant | Porównane odpowiedzi z pięcioma pytaniami | Różnice |
|---|---:|---:|
| Python sync/async | 2 | 0 |
| Node ESM native/adapted Fetch i CommonJS z własnymi fabrykami CJS | 3 | 0 |
| C# NuGet | 1 | 0 |
| Rust packaged crate | 1 | 0 |
| Chromium 154 native/adapted Fetch | 2 | 0 |

Łącznie **9 odpowiedzi, 45 wyników pytań, zero różnic**. Przeglądarka użyła
lokalnego pośrednika pod tym samym originem oraz modułów ESM z zainstalowanego npm.
Połączenie z istniejącym Basalem prowadziło przez SSH ze ścisłą weryfikacją
wcześniej potwierdzonego klucza hosta. Nie zmieniano serwera, konfiguracji CORS
ani danych produkcyjnych. Użyto wyłącznie syntetycznego tekstu.

Dowody: [request](request.json), [reference](reference.json),
[Python](python-results.json), [Node](node-results.json), [C#](csharp-results.json),
[Rust](rust-results.json), [przeglądarka](browser-results.json),
[podsumowanie](comparison-summary.json), [sumy dystrybucji](distribution-SHA256SUMS).
Nie nadpisano wcześniejszych raportów.

## Typy, walidacja i review

Próby kompilacyjne konsumentów nie zawierały rzutowań ani nonnull assertions
dla pól odpowiedzi. TS strict + noUncheckedIndexedAccess + exactOptionalPropertyTypes
potwierdził klucze requestu, union choice/selected/outcomes/actions, wymagane
probabilities/expected_costs i evidence.map bez optional chaining. Mypy strict
potwierdził dataclasses i wywnioskowane float/string.

Celowo niepoprawne fragmenty poza repo zostały odrzucone przez cztery kompilatory:
choice na odpowiedzi noul i evidence na opcjach multi. TS odrzucił także nieznany
ID oraz nieznany outcome probability. Nie dodano frameworków, trwałych testów
ani bramek testowych do CI.

Kontrole transportu poza repo potwierdziły odrzucenie brakujących podstawowych
pól wszystkich pięciu typów przed zwróceniem odpowiedzi w Python sync/async,
C# i Rust. Lokalny standardowy Response adapter TS potwierdził odrzucenie:
brakującego choice, mismatched type, nieznanego choice/outcome, wadliwego evidence,
brakującego znanego probability oraz kosztu akcji. Response spoza HTTP success
nadal trafia do istniejącej hierarchii błędów HTTP; pełny raw tryb pozostaje dostępny.

GPT-6 Astra XHigh wykonała odczytowy, skupiony review nowej funkcji, obejmujący
fabryki, parsing, snapshoty, identity, Rust GAT/lifetimes, cancellation i przykłady.
Znalazła dwie luki TS: act.answer spoza literalnych outcomes oraz niewalidowane
evidence. Obie naprawiono, dodano wymagane klucze probabilities/expected_costs
i osobny typowany widok z zachowanym .raw. Astra potwierdziła odczytem poprawki:
brak otwartych Blocker/Should w sprawdzonym zakresie. Rodzic potwierdził je
diagnostykami runtime i ponowną kompilacją. Poprzedni review Opusa dotyczy
supply chain, nie jest deklarowany jako review tej późniejszej funkcji.

Przeszły: fmt, Clippy workspace release/all-targets Linux i Intel, Rustdoc -D warnings
tych wariantów, weryfikowane cargo package, oddzielny cargo deny SDK, OSV dla
dziewięciu przypiętych paczek Python/npm, actionlint, check-paste-exception,
checker wersji, mypy strict, ESM/CommonJS/deklaracje, wheel z sdist,
locked restore i build przykładu C# z TreatWarningsAsErrors, NuGet pack oraz
weryfikacja hashów artefaktów. Lint/deny nie osłabiono. Zależności są bez zmian.

## Ograniczenia

To kontrola transportu i ergonomii, nie jakości modelu/upstream FP32, GPU ani
wydajności. Nie sprawdzono wszystkich semantycznych/numerical invariants każdego
wyniku lub wszystkich przyszłych rozszerzeń API. Typy i parsing chronią wymagane
pola oraz znane rodzaje/klucze; model capabilities i kalibracja należą do serwera.
Wyjątkowo opcjonalne metadane pozostają w raw, zamiast udawać zawsze obecne dane.

Python 3.11, Node 26, biblioteka net8.0 w runtime .NET 10, Rust 1.95 i Chromium 154.
Nie potwierdzono minimalnego Node 22, osobnego runtime .NET 8, runtime macOS/Windows
ani Deno/Bun/Workers. ESM i CommonJS mają oddzielne klasy: klient i specy powinny
pochodzić z tego samego importu/instancji SDK. W requestach stosowano model mini.

Nie publikowano paczek, tagów, commitów ani zmian kont. Konta, registry permissions,
zdalny OIDC i ochrony GitHub pozostają do konfiguracji. Istniejący advisory
unsafe-libyaml silnika nadal jest znanym problemem pełnego cargo deny workspace;
nie należy do grafu SDK i nie został zmieniony w tej funkcji.

Instrukcje i przykłady: [docs/SDK.md](../../docs/SDK.md).
