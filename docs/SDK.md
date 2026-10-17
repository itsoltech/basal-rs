# SDK natywnego API Basala

Cztery klienty HTTP obsługują **`POST /v1/basal`**. Nie korzystają z kontraktu
TypeSafe System One. Nie wymagają GPU ani wag modelu po stronie aplikacji.
Serwer uruchamiasz osobno przez `basal serve`.

Pierwsze wydanie każdego SDK ma wersję `0.1.0`. Kolejne wersje i wydania
są niezależne dla każdej paczki. Paczki instaluje się z rejestrów poniższymi
poleceniami; samo zbudowanie repozytorium nie publikuje paczek.

| Język | Rejestr / paczka | Instalacja po publikacji | Kod |
|---|---|---|---|
| Python 3.11+ | PyPI: `basal-client` | `pip install basal-client` | [sdk/python](../sdk/python/README.md) |
| TypeScript / JavaScript | npm: `@itsoltech/basal-client` | `npm i @itsoltech/basal-client` | [sdk/typescript](../sdk/typescript/README.md) |
| C# / .NET 8+ | NuGet: `ITSOL.Basal.Client` | `dotnet add package ITSOL.Basal.Client` | [sdk/csharp](../sdk/csharp/README.md) |
| Rust 1.95+ | crates.io: `basal-client` | `cargo add basal-client` | [sdk/rust](../sdk/rust/README.md) |

Każde SDK ma niezależną wersję w swoim manifeście. Poprawka TypeScriptu może
wydać tylko npm 0.1.1, gdy Python, C# i Rust pozostają na 0.1.0.
Wersja opisu kontraktu `contracts/basal/openapi.json` oraz wersja silnika są
oddzielne od wersji paczek. Rust SDK jest osobnym, publikowalnym crate'em;
nie zależy od `basal-core`, `basal-gpu` ani `basal-cli`.

| SDK | Źródło wersji | Tag wydania | Changelog |
|---|---|---|---|
| Python | `sdk/python/pyproject.toml` | `sdk-python-vX.Y.Z` | [Python](../sdk/python/CHANGELOG.md) |
| TypeScript/JavaScript | `sdk/typescript/package.json` | `sdk-typescript-vX.Y.Z` | [npm](../sdk/typescript/CHANGELOG.md) |
| C# | `sdk/csharp/ITSOL.Basal.Client/ITSOL.Basal.Client.csproj` | `sdk-csharp-vX.Y.Z` | [NuGet](../sdk/csharp/CHANGELOG.md) |
| Rust | `sdk/rust/Cargo.toml` | `sdk-rust-vX.Y.Z` | [crates.io](../sdk/rust/CHANGELOG.md) |

## Basal 2.0 i Omni

SDK obsługują typowane `tool`, 2–255 opcji choice/score/act (zależnie od modelu),
cele multi `threshold`/`exact`/`iou`/`f1` i ograniczenia `exclusive`/`fallback`.
Przy celach innych niż threshold odpowiedź nie wymaga `threshold`; zwraca
`objective`, `expected_score` i diagnostykę ograniczeń. Tool zwraca decyzję,
ranking albo zbiór narzędzi i odroczenie; SDK nie wykonuje narzędzi.

Opcje żądania obejmują `lang`, `media` i `detail`. Pomocniki kodują bajty do
base64; nie czytają plików ani nie pobierają URL. Obraz i audio wymagają serwera
Omni z CUDA i włączonymi mediami. Status kalibracji `text` jest akceptowany,
ale nie oznacza potwierdzonej kalibracji mediów. Typowane usage i metadane
obejmują liczbę tokenów/mediów, kolejności oraz format promptu.

Nowe wersje źródłowe: Python/npm/.NET **0.2.0**, Rust **0.3.0**. Zmiany modeli
odpowiedzi (opcjonalny threshold i nowe warianty) są opisane w changelogach.
Samo scalenie kodu nie publikuje paczek. Runtime nadal sprawdza możliwości
starszych modeli; SDK nie uruchamiają inferencji lokalnie.

## Python

Wariant klasy daje dokładne typy nazwanych pól, bez osobnej klasy odpowiedzi:

```python
from basal_client import BasalClient, Questions, choice, noul

routing = choice("Do którego działu skierować zgłoszenie?", {"returns": "Zwroty", "support": "Wsparcie techniczne"})

class TicketQuestions(Questions):
    department = routing
    second_department = routing
    urgent = noul("Czy sprawa jest pilna?")

with BasalClient("http://127.0.0.1:8000") as client:
    response = client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", TicketQuestions)
    print(response.answers.department.choice)
    print(response.answers.second_department.choice)
    print(response.answers.urgent.noul)
```

Można też przekazać zwykły słownik z tymi samymi fabrykami:

```python
with BasalClient() as client:
    response = client.decide("Telefon jest uszkodzony.", {
        "department": routing,
        "second_department": routing,
    })
    print(response.answers["department"])
    print(response.answer("second_department"))
```

Wariant słownika zwraca dataclasses pod kluczami requestu. Statyczny typ wartości to
unia `AnswerResult`; można ją zawęzić przez `isinstance`. Wariant klasy pozwala od razu
odczytać `.choice`/`.noul` itd. bez zawężania. Klasa jest przekazywana bez `()`; pola
można dziedziczyć, a nazwy zaczynające się od `_` są zarezerwowane. SDK nie wywołuje
konstruktorów użytkownika przy tworzeniu odpowiedzi. Nazwy opcji choice pozostają
w Pythonie typu `str`. Nie są wywnioskowanym literal union.
`AsyncBasalClient` obsługuje oba warianty z `await` i `async with`.

Dla zgodności z mypy i Pyright nadpisuj odziedziczone pole pytaniem tego samego rodzaju.
Mypy nie pozwala zmienić rodzaju ani usunąć pola przez `None`. Kolejność pól wynika
z odwróconego MRO Pythona: w `class D(M1, M2)` pola `M2` poprzedzają pola `M1`,
a nadpisane pole zachowuje pierwotną pozycję. Definicje można też przechowywać
w zwykłych klasach i dataclasses; poza `Questions` nadal są definicjami.
Typowane kontenery odpowiedzi nie obsługują protokołów copy/pickle. Do cache
i przekazywania wyniku między procesami serializuj `response.raw` jako JSON;
`copy.deepcopy(response.raw)` daje niezależną kopię natywnego wyniku.

## TypeScript / JavaScript

```typescript
import { BasalClient, choice, noul } from "@itsoltech/basal-client";

const client = new BasalClient({ baseUrl: "http://127.0.0.1:8000" });
const response = await client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", {
  department: choice("Do którego działu skierować zgłoszenie?", { returns: "Zwroty", support: "Wsparcie techniczne" }),
  urgent: noul("Czy sprawa jest pilna?"),
});
console.log(response.answers.department.choice); // "returns" | "support"
console.log(response.answers.urgent.noul);        // number
```

Paczka zawiera ESM, CommonJS i deklaracje typów. CommonJS:
`const { BasalClient } = require("@itsoltech/basal-client")`.
Wymaga Node.js 22+ albo przeglądarki z `fetch` i `AbortController`.
Transport można wstrzyknąć przez `fetch: adapter`, zachowując kontrakt Web Fetch API:
`(input: RequestInput, init?: RequestInit) => Promise<Response>`, gdzie
`RequestInput = string | URL | Request`. Adapter innego klienta HTTP zwraca standardowy
`Response` z body i nagłówkami; musi respektować `signal`, `redirect: "error"`,
`credentials: "omit"` i weryfikację TLS. SDK nie instaluje dodatkowego klienta HTTP.
Przykład: [README npm](../sdk/typescript/README.md).
Bezpośrednie wywołanie z przeglądarki wymaga CORS w reverse proxy; basal-rs
domyślnie nie włącza CORS. Klucze prywatne trzymaj po stronie serwera aplikacji.

## C#

```csharp
using ITSOL.Basal;

using var client = new BasalClient();
// These criteria may be loaded asynchronously from a database before preparation.
var criteria = new Dictionary<string, string> { ["returns"] = "Zwroty", ["support"] = "Wsparcie techniczne" };
var questions = QuestionBuilder.Prepare(q => new
{
    Department = q.Choice("Do którego działu skierować zgłoszenie?", criteria),
    Urgent = q.Noul("Czy sprawa jest pilna?"),
});
var response = await client.DecideAsync("Klient prosi o zwrot pieniędzy za uszkodzony telefon.",
    new { questions.Department, questions.Urgent });
Console.WriteLine($"Dział: {response.Answers.Department.Choice}, P(tak): {response.Answers.Urgent.Noul}");
```

`QuestionBuilder.Prepare` pozwala przygotować pytania wcześniej, po asynchronicznym
odczycie enumów/kryteriów z bazy. Gotowy obiekt jest szablonem, który można współdzielić.
Wywołanie `client.DecideAsync(state, new { Department = questions.Department })` wybiera
pole i tworzy nowy, dokładnie typowany obiekt odpowiedzi. Można przekazać cały anonimowy
szablon, zmienić nazwę pola lub przypisać tę samą definicję pod kilkoma nazwami.
Każde wywołanie ma niezależne pola odpowiedzi; szablon nie przechowuje wyników.

Przeciążenie obiektowe jest metodą rozszerzającą w `ITSOL.Basal`. Dla nazwanych kształtów
wymaga publicznego konstruktora z parametrami odpowiadającymi publicznym polom odpowiedzi,
np. positional record. Inną klasę przygotowanych pytań można przekazać przez wybór jej
pól do anonimowego obiektu. Konstruktor odpowiedzi uruchamia się przed HTTP, a kolejność
pytań odpowiada parametrom konstruktora wybranego kształtu. Konstruktor musi zachować
otrzymane pola pod odpowiadającymi im nazwami; zamiana pól jest odrzucana przed HTTP.
Przekaż konkretny typ szablonu; zmienna typu bazowego/interfejsu jest odrzucana, aby
nie zgubić pytań z typu pochodnego. Jawnego ID z `q.Add` nie można powtórzyć przez aliasy.
Token anulowania przekazuj jako `cancellationToken: token`. Przykłady i zakres kopiowania:
[README .NET](../sdk/csharp/README.md). Nadal dostępna jest fabryka inline:

```csharp
var response = await client.DecideAsync(state, q => new
{
    Department = q.Choice("Do którego działu skierować zgłoszenie?", criteria),
    Urgent = q.Noul("Czy sprawa jest pilna?"),
});
```

Fabryka uruchamia się raz i może zwracać anonimowy obiekt albo własną klasę.
Musi działać synchronicznie; przygotowanie asynchroniczne wykonaj przed `DecideAsync`.
Fabryki zwracające `Task` albo `ValueTask` są odrzucane przez kompilator.
Klucze requestu wynikają z nazw pól: tutaj `Department` i `Urgent`, z zachowaniem wielkości liter.
Nie trzeba powtarzać nazwy w osobnym stringu. Fabryki `q.Choice`, `q.Noul`, `q.Score`,
`q.Multi`, `q.Act` zwracają dokładnie typowane pola. Pola czyta się po `await`;
wszystkie stają się dostępne dopiero po poprawnej walidacji całej odpowiedzi.
Tę samą definicję można przypisać przez `Department = q.Choice(routing)` i
`SecondDepartment = q.Choice(routing)`; każde wywołanie tworzy odrębne pole odpowiedzi. Wariant słownika przez
`Create(...)` albo `IReadOnlyDictionary<string, QuestionSpec>` daje
`response.Answers["department"]` / `response.Answer("department")` typu `AnswerResult`,
który można zawęzić przez pattern matching. Przykład własnej klasy i zasady
lifecycle buildera: [README .NET](../sdk/csharp/README.md).
W fabryce inline współdziel definicje pytań, ale twórz pola odpowiedzi od nowa przy każdym wywołaniu.
SDK przed HTTP odrzuca pola z innej fabryki oraz jedno pole użyte pod dwiema różnymi nazwami.
Nazwy odczytywane są przez wbudowaną refleksję .NET z publicznych pól/właściwości zwracanego typu;
nie dodano zależności ani generatora. Kolejność pytań wynika z tworzenia pól w fabryce.
`q.Add("custom_id", definition)` pozostaje opcją dla jawnego ID. Opcje decyzji i CancellationToken
są dostępne zarówno dla słownika, jak i fabryki obiektu.
Jawne ID z `q.Add` ma pierwszeństwo przed nazwą pola. Dla automatycznych nazw
użyj anonimowego obiektu, record, klasy albo struct. Krotki z automatycznymi ID
są odrzucane, ponieważ aliasy ich elementów nie są dostępne w runtime.
Obsługiwane są publiczne pola i właściwości klas bazowych; odziedziczone właściwości
interfejsów i ukryte pola o powtórzonych nazwach nie są obsługiwane.

Klient przyjmuje też zewnętrzny `HttpClient`, np. z `IHttpClientFactory`,
i nie przejmuje jego ownership. Ustaw `AllowAutoRedirect=false` w jego handlerze.
Stan obiektowy, kryteria i macierz kosztów tworzysz przez `JsonObject` / `JsonArray`.
Przeciążenie `DecideAsync(JsonObject)` pozwala wysłać pełny natywny JSON, także
jawne `instructions: null`; null w opcjonalnych polach DTO oznacza ich pominięcie.
Przykład do kompilacji: [Decide](../sdk/csharp/examples/Decide/Program.cs).

## Rust

```rust,no_run
use basal_client::{choice, noul, questions, Client};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://127.0.0.1:8000")?;
    let department = choice(
        "Do którego działu skierować zgłoszenie?",
        &[("returns", "Zwroty"), ("support", "Wsparcie techniczne")],
    )?;
    let urgent = noul("Czy sprawa jest pilna?")?;
    let questions = questions! {
        department: &department,
        second_department: &department,
        urgent: &urgent,
    }?;
    let response = client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", &questions).await?;
    let answers = response.answers()?;
    println!("Dział: {}, P(tak): {}", answers.department.choice, answers.urgent.noul);
    println!("Drugie pytanie: {}", answers.second_department.choice);
    Ok(())
}
```

Ten wariant wymaga Rust SDK **0.2.0**. Dodaj runtime: `cargo add tokio --features macros,rt-multi-thread`.
Makro `questions!` wyprowadza typy pól odpowiedzi; nazwy pól stają się kluczami requestu.
Dla dynamicznych kluczy użyj `Questions::new().with("department", &department)?` i
`response.choice("department")?.choice` albo dopasuj enum z `response.answer("department")?`.
`Question.criteria` przyjmuje obiekt/tablicę JSON; `Question.extra` i
`DecisionRequest.extra` obsługują pola rozszerzeń. Nie powtarzaj w `extra`
pól nazwanych w strukturze. Przykład do kompilacji:
[decide.rs](../sdk/rust/examples/decide.rs).

## Typowane pytania i odpowiedzi

Fabryki mają wspólny schemat: instrukcja, wymagane kryteria/poziomy/koszty oraz opcje
konkretnego typu. Odpowiadają im klasy/typy ChoiceQuestion, NoulQuestion, ScoreQuestion,
MultiQuestion, ActQuestion. Choice/score wymaga 2–10 opcji; multi co najmniej jednej.
Opcje multi nie zawierają evidence. Act przyjmuje pełną macierz kosztów z wierszem
na każdą akcję i kosztem dla każdego outcome; bez criteria używa true/false.
TypeScript wyprowadza klucze outcomes z criteria (domyślnie true/false), wymaga
ich w każdym wierszu kosztów i ogranicza defer_action do kluczy akcji.
Helpery obejmują tekstowe instrukcje i opisy; pełny JSON nadal obsługuje natywne rozszerzenia.

Każdy typ pytania określa typ wyniku. W TS `response.answers.department.choice` jest
wywnioskowany jako union kluczy kryteriów, a `response.answers.urgent.noul` jako number.
Python obsługuje dwa warianty: klasa `Questions` daje nazwane, dokładnie typowane
pola w `response.answers`, a słownik daje mapę sprawdzonych odpowiedzi z typem-union.
C# zachowuje kształt obiektu z fabryki i daje np. `response.Answers.Department.Choice`;
pola wiążą się z kluczami, nie z tożsamością definicji. Rust od 0.2.0 zachowuje nazwane
pola przez `questions!`: `response.answers()?.department.choice`. Dynamiczna mapa daje enum
z `response.answer("department")?` lub sprawdzony widok z `response.choice("department")?`.
Wyniki mają wymagane pola bez opcjonalnego choice/noul/score itd. W Rust końcowe `?`
propaguje błąd walidacji lub nieznanego klucza; nie jest unwrapem opcjonalnego pola.
W Python wyniki są dataclass, w C# record, w Rust pożyczonymi widokami.
Evidence w typowanych wynikach ma pustą kolekcję zamiast null, gdy nie zwrócono spanów.
Wyniki act w Python/C# udostępniają też opcjonalne `refused`/`max_error`
(`Refused`/`MaxError`): komunikaty serwera o odmowie lub niedostępnym certyfikowanym progu.
Wymagane pola wyniku pozostają nieopcjonalne. Dodatkowe pola pozostają w odpowiedzi raw;
TS zachowuje ją w `response.raw`.

Typowana ścieżka waliduje wymagane odpowiedzi przed ich zwróceniem; pominięty field,
zły typ pytania albo nieznana choice/akcja daje błąd SDK. Nie zastępuje to walidacji
capabilities, kalibracji ani jakości decyzji po stronie modelu. Nie zmienia liczb ani
kolejności kryteriów. Wszystkie SDK (Rust od 0.2.0) pozwalają użyć jednej definicji pod
kilkoma kluczami; odpowiedzi rozróżnia klucz. `answer()` / `Answer()` przyjmuje klucz
(C# udostępnia tę metodę w wariancie słownika). Rust 0.1.0 odrzucał ponowne użycie
tej samej definicji; migracja do 0.2.0 wymaga zastąpienia `answer(&question)` odczytem
po kluczu lub przez nazwane pola z `questions!`.
Obiekty przechowują kopię wejścia, a mapę pytań utrwala się przed wysłaniem.
Klas pytań nie należy modyfikować podczas oczekiwania na odpowiedź. Nie dodano zależności.

Pełny request: `decide(request)` w Python/TS, `DecideAsync(request)` w C#,
`decide_request(&request)` w Rust. Namespace C#: `ITSOL.Basal`.

## Kontrakt i zachowanie klientów

[contracts/basal/openapi.json](../contracts/basal/openapi.json) opisuje natywny
kontrakt na podstawie handlerów basal-rs, parsera i budowania odpowiedzi.
Jest utrzymywanym opisem kanonicznych pól, z dopuszczeniem rozszerzeń, a nie
schematem wygenerowanym z OpenAPI TypeSafe. Przy zmianie natywnego API
aktualizuj kontrakt i DTO we wszystkich czterech SDK.

- `decide` / `DecideAsync`: `POST /v1/basal`, pełna odpowiedź `model`, `answers`, `usage`.
- `models` / `ModelsAsync`: `GET /v1/models`.
- Wymagane jest HTTPS. HTTP jest dopuszczalne dla loopback, np. lokalnego serwera lub
  tunelu SSH. URL nie może zawierać danych logowania, query ani fragmentu.
- URL to root serwera, opcjonalnie z prefiksem proxy, np. `https://example.com/basal`.
  SDK dopisuje `v1/basal` lub `v1/models`; nie podawaj końcowego `/v1`.
- `model` jest opcjonalny. W basal-rs wybiera model po dokładnej nazwie z `/v1/models`;
  bez niego działa `default_model`. Oryginalny serwer upstream ignoruje to pole.
- SDK wysyła stan jako JSON; renderowanie stanu, kalibracja i decyzje należą do serwera.
- Typy pytań: `choice`, `noul`, `score`, `multi`, `act`; obsługiwane są `facts` i `evidence`.
  Natywne choice/score ma limit 2–10 opcji. Multi nie obsługuje evidence.
- SDK zachowuje natywne prawdopodobieństwa i confidence. Dla choice/noul/score/act
  `confidence = max(p)`. Multi ma niezależne P(tak) i `set_confidence`.
- Kolejność kryteriów wpływa na obliczenia. Nie sortuj kluczy. W JS numeryczne klucze
  obiektu są enumerowane rosnąco; używaj nazw tekstowych lub uporządkowanej tablicy score.
- `evidence.start/end` liczą znaki Unicode. Nie są offsetami UTF-8 ani UTF-16.
  Najłatwiej korzystać z gotowego `evidence.text`.
  Krótki tekst może dać pusty span z prawdopodobieństwem 0 i `end < start`;
  SDK zachowuje go zgodnie z natywną odpowiedzią.
- Klient nie wykonuje akcji wskazanej przez `act`.

Domyślny timeout wynosi 120 sekund. JS, C# i Rust obejmują nim całe żądanie
wraz z odczytem odpowiedzi. Python stosuje timeout dla faz connect/read/write/pool
HTTPX ([semantyka timeoutów](https://www.python-httpx.org/advanced/timeouts/)).
JS przyjmuje `AbortSignal`, C# `CancellationToken`; w Rust anulowanie polega na
porzuceniu future, a w Pythonie async na anulowaniu zadania. Anulowanie po stronie
klienta nie gwarantuje przerwania inferencji na serwerze.

Błędy zachowują status HTTP, body oraz surowe `Retry-After`; dla body bez JSON
zachowują tekst. Natywna walidacja daje 422, przeciążenie 529 z `Retry-After: 1`.
Brak połączenia, timeout i niepoprawna odpowiedź JSON są oddzielone od błędów
HTTP. Nie ma automatycznych retry: ponowienie może uruchomić kolejną inferencję.
Domyślne transporty nie podążają za przekierowaniami. Opcjonalny klucz API
ustawia `Authorization: Bearer ...` dla proxy; basal-rs nie implementuje uwierzytelniania.

Domyślny limit odpowiedzi wynosi 8 MiB i obejmuje również body błędów. Klienty
liczą rzeczywiste bajty strumienia; parametr limitu opisują README paczek.
Python żąda `Accept-Encoding: identity` i odrzuca skompresowane odpowiedzi przed
uruchomieniem dekompresora. Proxy musi zwracać temu klientowi nieskompresowany JSON.
Python ignoruje konfigurację proxy/certyfikatów ze środowiska (`trust_env=False`);
włączenie jej wymaga `trust_env=True`. Domyślny C# nie używa proxy środowiskowego.
Rust używa Hyper HTTP/1, Rustls z ring i platformowej weryfikacji certyfikatów,
bez proxy środowiskowego, AWS-LC ani CMake. Nie instaluje globalnego providera kryptografii.
TS używa ustawień sieciowych środowiska Fetch; żądania pomijają cookies i cache.

Python i TS zachowują odpowiedź jako obiekt JSON i sprawdzają tylko taki kształt
koperty. Rust i C# deserializują DTO. Wszystkie klienty zachowują dodatkowe pola
odpowiedzi; surowa ścieżka TypeScript/Python nie wykonuje pełnej walidacji schematu runtime.
Typowana ścieżka dodatkowo sprawdza wymagane pola wyników zgodnie z pytaniami.

## Lokalne budowanie paczek

```sh
python tools/sdk/check-versions.py
python -m pip install --require-hashes --only-binary=:all: -r tools/sdk/requirements-build.txt
python tools/sdk/build-python.py
python tools/sdk/audit-locks.py
npm --prefix sdk/typescript ci --ignore-scripts
npm --prefix sdk/typescript pack --pack-destination "$PWD/sdk/typescript/dist"
dotnet restore sdk/csharp/examples/Decide/Decide.csproj --locked-mode --configfile sdk/csharp/nuget.config
dotnet pack sdk/csharp/ITSOL.Basal.Client/ITSOL.Basal.Client.csproj --no-restore --configuration Release -p:ContinuousIntegrationBuild=true --output sdk/csharp/dist
cargo package --locked -p basal-client
```

Python można zainstalować lokalnie przez `pip install ./sdk/python`.
C#: dodaj `sdk/csharp/dist` jako lokalne źródło NuGet.
Rust: użyj zależności ścieżkowej na `sdk/rust` przed publikacją.

Lokalna instalacja npm wybiera archiwum na podstawie wersji paczki:

```sh
npm_sdk_version=$(python tools/sdk/check-versions.py --sdk typescript --version-only)
npm i "./sdk/typescript/dist/itsoltech-basal-client-$npm_sdk_version.tgz"
```

Po zbudowaniu `.crate` zależności samego SDK można sprawdzić przez
poniższe polecenia:

```sh
rust_sdk_version=$(python tools/sdk/check-versions.py --sdk rust --version-only)
cargo deny --manifest-path "target/package/basal-client-$rust_sdk_version/Cargo.toml" --locked check --config sdk/rust/deny.toml
```

Ta konfiguracja zachowuje reguły workspace bez wyjątku `paste`, którego graf SDK nie potrzebuje.

CI kompiluje SDK i przykłady, sprawdza wersje i metadane dystrybucji. Rust SDK
dziedziczy politykę lintów workspace. Projekt nie dodaje testów, frameworków
testowych ani bramek testowych. Kompilacja i pakowanie nie potwierdzają jakości
decyzji modelu, działania GPU ani udanej publikacji do rejestrów.
Wyniki pierwszej kontroli lokalnej opisuje [raport SDK](../reports/sdk-native-api-2026-10-13/README.md).
Porównanie z działającym natywnym API, weryfikację przeglądarki i poprawki bezpieczeństwa
opisuje [raport live](../reports/sdk-live-and-hardening-2026-10-13/README.md).

## Zależności i granice audytu

| Paczka | Zewnętrzne zależności runtime / rozwiązanego grafu | Budowanie dystrybucji |
|---|---|---|
| npm | 0 | jeden kompilator TypeScript 6.0.3; brak hooków instalacyjnych |
| NuGet | 0 | SDK .NET; brak skryptów i assetów build/buildTransitive w paczce |
| Python | HTTPX i jego graf: 7 paczek dla przypiętego Python 3.11 | jeden hash-locked backend Flit Core 4.1.0 |
| Rust | 56 Linux / 59 macOS / 56 Windows, z makrami i build dependencies | Hyper HTTP/1, Rustls i ring; bez Reqwest, AWS-LC, CMake, silnika i GPU |

Liczby Rust dotyczą niezależnie spakowanego locka, bez crate'a SDK i zależności dev.
Rozwiązanie grafu dla macOS/Windows nie oznacza wykonania na tych platformach.
SDK nie wymusza `serde_json/arbitrary_precision`; `float_roundtrip` zachowuje liczby binary64.
Grafy, wersje i ograniczenia audytu są zapisane w raporcie live. Aplikacje powinny utrzymywać
własne locki: audyt przypiętego grafu nie obejmuje wszystkich przyszłych wersji dopuszczonych
przez zakresy bibliotek. Mniejsza liczba zależności ogranicza powierzchnię ataku,
ale nie zastępuje ochrony kont, runnerów i review aktualizacji.

Wyniki dla typowanych fabryk i odpowiedzi:
[raport typed SDK](../reports/sdk-typed-builders-2026-10-13/README.md),
[poprawki po review Opusa](../reports/sdk-typed-builders-2026-10-13/post-review/README.md).
Obsługę słownika i deklaratywnej klasy Python oraz odczytu po kluczu Python/TS
opisuje [raport dict/class](../reports/sdk-dict-class-2026-10-13/README.md).
Zakończenie review Opusa i poprawkę definicji w zwykłych klasach opisuje
[uzupełnienie raportu](../reports/sdk-dict-class-2026-10-13/post-review/README.md).
Odpowiedzi .NET po kluczach oraz przez typowany obiekt/klasę opisuje
[raport .NET](../reports/sdk-dotnet-keyed-2026-10-13/README.md).
Zakończenie review Opusa i zabezpieczenia fabryk opisuje
[uzupełnienie .NET](../reports/sdk-dotnet-keyed-2026-10-13/post-review/README.md).
Uproszczone fabryki .NET z nazwami pól jako ID, review i pomiar aktualnej paczki opisuje
[raport automatycznych ID](../reports/sdk-dotnet-inferred-2026-10-13/README.md).
Przygotowywane wcześniej szablony, wybór pól i walidację konstruktorów opisuje
[raport przygotowanych pytań](../reports/sdk-dotnet-prepared-2026-10-14/README.md).

## Przygotowanie kont i pierwsze wydanie

[sdk-release.yml](../.github/workflows/sdk-release.yml) wybiera jedno SDK z tagu.
Buduje tylko jego paczkę w jobie bez sekretów i uprawnień OIDC. Publikujący job
zależy wyłącznie od metadata i właściwego joba budowania; pominięte inne SDK
nie blokują publikacji. Python/npm audytują wyłącznie własny przypięty graf.
Zwykłe CI pull requestów nadal sprawdza wszystkie SDK i ich niezależne wersje.

Publisher pobiera swój artefakt po niezmiennym ID zwróconym przez job budowania
(przez `needs`), a następnie weryfikuje SHA-256. Podmiana artefaktu o tej samej nazwie
nie zmienia zapisanego ID; usunięcie oryginału blokuje publikację. npm i NuGet
publikują gotowe archiwa, PyPI korzysta z obrazu przypiętego digestem.
Cargo w jobie OIDC przepakowuje bez kompilacji (`--no-verify`) i porównuje
archiwum z już skompilowanym artefaktem; nie uruchamia kodu zależności.

Tagi `sdk-python-vX.Y.Z`, `sdk-typescript-vX.Y.Z`, `sdk-csharp-vX.Y.Z` i
`sdk-rust-vX.Y.Z` publikują odpowiednią paczkę. Tagi `vX.Y.Z` pozostają wydaniami
silnika. Stare wspólne tagi `sdk-vX.Y.Z` nie uruchamiają tego workflow.
Obecna wersja procesu obsługuje stabilne `X.Y.Z`; tagi prerelease są odrzucane.

Ręczny `workflow_dispatch` domyślnie tylko buduje. Wybierz jedno wejście `sdk`:
`python`, `typescript`, `csharp` albo `rust`. Publikacja ręczna wymaga `publish=true`
i uruchomienia na pasującym tagu. SDK wybrane ręcznie musi odpowiadać SDK z tagu,
a wersja tagu musi dokładnie zgadzać się z manifestem. Publikacja z gałęzi oraz
rozbieżności są odrzucane przed budowaniem i uzyskaniem uprawnień publikacji.

W GitHub utwórz środowiska `sdk-npm`, `sdk-pypi`, `sdk-nuget`, `sdk-crates`.
Skonfiguruj trusted publishers na właściciela `itsoltech`, repozytorium
`basal-rs`, workflow **`sdk-release.yml`** i odpowiednie środowisko.
Konta, rezerwację nazw i dostęp do organizacji przygotowuje się poza kodem.

| Środowisko | Dozwolone tagi |
|---|---|
| `sdk-npm` | `sdk-typescript-v*` |
| `sdk-pypi` | `sdk-python-v*` |
| `sdk-nuget` | `sdk-csharp-v*` |
| `sdk-crates` | `sdk-rust-v*` |

Chroń tworzenie, zmianę i usuwanie tych tagów rulesetem. Chroń workflow i skrypty
wydawnicze przez review zmian. Taguj zatwierdzony commit z głównej gałęzi.
Skonfiguruj 2FA i minimalne uprawnienia w rejestrach. Ustawienia ochrony repo,
środowisk i kont nie są zmieniane przez workflow.

1. **npm:** zapewnij dostęp do scope `@itsoltech`. Pierwszą paczkę opublikuj
   tokenem z prawem tworzenia publicznej paczki, ustawionym jako `NPM_TOKEN`
   w środowisku `sdk-npm`, albo lokalnie przez `npm publish --access public`.
   Następnie skonfiguruj [trusted publisher](https://docs.npmjs.com/trusted-publishers/)
   dla paczki z uprawnieniem bezpośredniej publikacji i usuń bootstrap token.
   Workflow używa Node 24 / npm obsługującego OIDC.
2. **PyPI:** dla nowej paczki `basal-client` skonfiguruj
   [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
   Wskaż środowisko `sdk-pypi`. Pierwsze i kolejne wydania użyją OIDC bez tokenu PyPI.
3. **NuGet:** skonfiguruj [trusted publishing](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing)
   z prawem publikowania `ITSOL.Basal.Client`, również tworzenia nowej paczki.
   Ustaw w środowisku `sdk-nuget` sekret `NUGET_USER` z nazwą profilu NuGet,
   nie adresem e-mail. `NuGet/login` pobiera krótkotrwały klucz przez OIDC.
4. **crates.io:** zapewnij właściciela nowej paczki `basal-client`. Pierwszą
   publikację można wykonać tokenem jako `CARGO_REGISTRY_TOKEN` w `sdk-crates`
   albo lokalnie przez `cargo publish -p basal-client`. Następnie skonfiguruj
   [trusted publishing](https://crates.io/docs/trusted-publishing) dla paczki,
   usuń bootstrap token i korzystaj z
   [crates-io-auth-action](https://github.com/rust-lang/crates-io-auth-action).

## Wydanie jednej paczki

W PR wydania zmień wyłącznie wersję odpowiedniego manifestu i jego changelog.
Dla npm zaktualizuj też oba pola wersji w `sdk/typescript/package-lock.json`
(np. `npm version 0.1.1 --no-git-tag-version --ignore-scripts` w katalogu npm).
Dla Rust odśwież wpis paczki w głównym `Cargo.lock` po zmianie manifestu.
Pozostałe wersje paczek, wersja silnika i wersja opisu kontraktu pozostają niezależne.
Jeśli zmienia się wspólny kontrakt, sprawdź wszystkie SDK; publikuj tylko te,
które wymagają zmian.

Po merge uruchom `SDK release` przez `workflow_dispatch` dla tego SDK z
`publish=false`, na głównej gałęzi. Poczekaj na udane budowanie i weryfikację
artefaktu. Taguj ten sam sprawdzony commit. Ten przebieg obejmuje również
kontrole pakowania i grafu spakowanego Rust SDK wykonywane w workflow wydawniczym.

Przykład wydania poprawki TypeScriptu po merge PR z wersją 0.1.1:

```sh
python tools/sdk/check-versions.py --sdk typescript --tag sdk-typescript-v0.1.1
git tag sdk-typescript-v0.1.1
git push origin sdk-typescript-v0.1.1
```

Każdy tag SDK wysyłaj w osobnym `git push` i potwierdź rozpoczęcie jego workflow.
Nie używaj zbiorczego `git push --tags` przy wydawaniu czterech SDK:
GitHub nie tworzy zdarzeń tagów, gdy jednocześnie wysyłane są więcej niż trzy.
[Ograniczenie zdarzenia push](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#push).
Jeśli przebieg nie wystartował, można uruchomić go ręcznie na właściwym tagu,
wybierając pasujące `sdk` i `publish=true`.

`python tools/sdk/check-versions.py` sprawdza wszystkie manifesty niezależnie,
bez wymogu równych numerów, oraz poprawny format wersji opisu kontraktu.
`--sdk` sprawdza tylko wybraną paczkę i jej lock. `--tag` może sam wybrać SDK;
`--version-only` zwraca jego wersję dla skryptów budowania. Pakowanie artefaktów
również odczytuje wersję z właściwego manifestu.

W fazie 0.x używaj patch dla zgodnych poprawek, minor dla nowych funkcji i zmian
niezgodnych. Od 1.0 stosuj SemVer: patch dla poprawek, minor dla zgodnych nowych
funkcji, major dla zmian niezgodnych. Opublikowana wersja pozostaje niezmienna.

Gdy publikacja się nie powiedzie, ponów tylko jej job w istniejącym runie;
pobierze ten sam artefakt po ID. Nie ponawiaj udanej publikacji tej samej wersji.
Nowy `workflow_dispatch` z tym samym tagiem, odpowiednim `sdk` i `publish=true`
buduje nowy artefakt: nie jest wznowieniem z oryginalnego runu. Wydanie danej
paczki jest zakończone dopiero po potwierdzeniu dostępności wersji w jej rejestrze.

Jeśli PyPI przyjęło tylko część plików wheel/sdist, ponowienie może zostać
odrzucone z powodu już istniejącego pliku. Nie włączaj pomijania istniejących
plików bez sprawdzenia zgodności ich hashów z pierwotnym artefaktem. Najpierw
ustal stan publikacji; do rejestru należy przesłać tylko brakujące pliki
z tego samego zweryfikowanego artefaktu. Ten przypadek wymaga obsługi operatora.
