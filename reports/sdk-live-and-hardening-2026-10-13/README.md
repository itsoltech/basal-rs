# SDK: działające natywne API, Fetch i ograniczenie zależności

Data: 2026-10-13. SDK `0.1.0`. Weryfikacja obejmuje natywne `/v1/basal`,
`/v1/models` i `/health`, bez kontraktu System One.

## Wynik porównań

W zapisanym zestawie **64 odpowiedzi SDK nie ma różnic względem bezpośredniego API**.
Porównanie obejmuje status, strukturę, tekst, kolejność elementów tablic i bity liczb
binary64. Pominięto wyłącznie `usage.latency_ms`, ponieważ jest czasem konkretnego
wywołania. Nie zaokrąglano prawdopodobieństw ani confidence.

| Dystrybucja i sposób wywołania | Zapisane porównania | Różnice |
|---|---:|---:|
| Python wheel: sync i native async | 16 | 0 |
| npm ESM w Node: native Fetch i wstrzyknięty adapter | 16 | 0 |
| NuGet: typowane DTO i pełny JSON dla jawnego null | 8 | 0 |
| Rust: klient ze zweryfikowanego, rozpakowanego `.crate` | 8 | 0 |
| npm ESM w Chromium: native Fetch i adapter | 16 | 0 |

Każdy wariant wykonał te same osiem przypadków: wszystkie pięć typów pytań,
stan obiektowy i tekstowy z polskimi znakami/emoji, evidence, `facts: auto`,
jawne `instructions: null`, wybór modelu, pominięty model oraz trzy błędy 422
(nieznany model, za mało opcji choice, evidence w multi). Jest pięć odpowiedzi 200
i trzy 422 na wariant: 40 poprawnych decyzji i 24 zachowane błędy HTTP.
Przypadki są syntetyczne; nie wysyłano danych użytkowników.

Serwer reklamował `basal-1.5-max`, `basal-1.5-4.5B`, `basal-1.5-mini`.
Decyzje z wybranym modelem używały mini; pominięty model zwracał max.
Odkrywanie modeli i health sprawdzono osobno. Nie wykonano decyzji modelem 4.5B.
Zbudowane archiwa Python/npm/NuGet zainstalowano w oddzielnych środowiskach konsumenta;
Rust użył zawartości spakowanej biblioteki, poza workspace silnika.

Dowody: [żądania](requests.json), [bezpośrednie API](reference.json),
[Python](python-results.json), [Node](node-results.json), [C#](csharp-results.json),
[Rust](rust-results.json), [przeglądarka](browser-results.json),
[podsumowanie porównania](comparison-summary.json).
Zapisane odpowiedzi nie zawierają zmiennego czasu inferencji.

## Transport i przeglądarka

Połączenie z istniejącym serwerem prowadziło przez SSH z kluczem hosta
potwierdzonym przez użytkownika i ścisłą weryfikacją. Tunel nasłuchiwał wyłącznie
na loopback. Nie zmieniano konfiguracji ani plików serwera. Po weryfikacji zamknięto
tunel i lokalne serwery pomocnicze; działającego Basala pozostawiono uruchomionego.
Raport nie zawiera identyfikatorów maszyny, adresu serwera ani jego ścieżek.

Przeglądarka pobrała ESM z zainstalowanego archiwum npm. Lokalny pośrednik pod tym
samym originem przekazywał wyłącznie trzy stałe ścieżki API do tunelu. Nie zmieniano
CORS Basala. Drugi wariant adaptera tworzył standardowy Web `Request`, respektował
`RequestInit` i zwracał standardowy `Response`. Bezpośrednie wywołanie między originami
zostało zablokowane przez CORS. Aplikacja wymaga odpowiedniego reverse proxy/CORS albo
pośrednika pod własnym originem; SDK nie omija tych ograniczeń.

Kontrakt TS jest taki sam dla przeglądarki i backendu:
`RequestInput = string | URL | Request`,
`Fetch = (input: RequestInput, init?: RequestInit) => Promise<Response>`.
Adapter musi respektować anulowanie, brak przekierowań, pomijanie cookies i TLS.
Deadline SDK odrzuca również adapter ignorujący signal; taki adapter nadal może
prowadzić własne I/O. SDK nie może narzucić implementacji wewnętrznej obcego klienta.

Diagnostyki transportu wykonano poza repozytorium, bez dodawania frameworków i bramek testowych:

- Wszystkie paczki odrzuciły lokalny niezaufany certyfikat TLS. Python zwrócił przyczynę
  `CERTIFICATE_VERIFY_FAILED`, Node `DEPTH_ZERO_SELF_SIGNED_CERT`, C#
  `AuthenticationException`, Rust błąd weryfikacji peer certificate `CaUsedAsEndEntity`.
  To negatywna kontrola certyfikatu; nie jest pełną macierzą przypadków TLS.
- Limit rzeczywistych bajtów odpowiedzi działa także dla discovery; domyślnie wynosi 8 MiB.
  Nie opiera się wyłącznie na Content-Length. Limit ten nie oznacza identycznego limitu
  całej pamięci transportu, dekompresora i parsera.
- Powolne body uruchomiło deadline JS/C#/Rust i timeout odczytu HTTPX. Python ma timeout
  faz sieciowych, nie całkowitego czasu. Przetwarzanie/serializacja JSON nie jest osobno przerywalne.
- Anulowanie przed wysłaniem i w trakcie I/O sprawdzono w JS; także CancellationToken C#,
  anulowanie zadania Python async i porzucenie future Rust. Nie potwierdza to zatrzymania
  inferencji po stronie serwera.
- Domyślne klienty nie podążyły za przekierowaniem. Zachowano HTTP 529 i surowe Retry-After;
  wadliwy JSON powodował błąd SDK. Błąd UTF-8 JS anuluje niedokończony strumień.
- Konfiguracja HTTP poza loopback jest odrzucana. Domyślne Python/C#/Rust ignorują proxy
  środowiskowe; Python może jawnie włączyć `trust_env`. Transporty zewnętrzne pozostają
  odpowiedzialnością konfigurującej je aplikacji.

Dowód zbiorczy: [kontrole transportu](transport-checks.json).

## Zależności i poprawki review

| Paczka | Sprawdzony graf | Zależności budowania paczki |
|---|---|---|
| TypeScript/JavaScript | 0 zewnętrznych runtime | TypeScript 6.0.3, jedna paczka JS |
| C# | 0 zewnętrznych runtime | SDK .NET; pusty lock zależności biblioteki |
| Python 3.11 | 7: httpx, httpcore, anyio, certifi, h11, idna, typing_extensions | Flit Core 4.1.0, jedna paczka z hashem |
| Rust Linux x64 | 56 normal + build, bez SDK i dev | wliczone makra i build dependencies |
| Rust macOS ARM64 | 59 normal + build | graf rozwiązany na Linuxie |
| Rust Windows MSVC | 56 normal + build | graf rozwiązany na Linuxie |

Graf Rust obejmuje pięć paczek w gałęzi makr i trzy dodatkowe build dependencies.
Świeża aplikacja diagnostyczna rozwiązała 58 zależności z własnymi features Tokio
i nowszymi wersjami; lock spakowanego SDK ma 56 na Linuxie. Nie utożsamiano pełnego
locka wszystkich platform, zależności dev ani nieaktywnych features z grafem runtime.
Grafy: [Linux](rust-graph-x86_64-unknown-linux-gnu.txt),
[macOS](rust-graph-aarch64-apple-darwin.txt),
[Windows](rust-graph-x86_64-pc-windows-msvc.txt).

Rust przeszedł z Reqwest na Hyper HTTP/1, Rustls, platformowy verifier i ring:
bez AWS-LC, CMake, IDNA/ICU, silnika i GPU. Pozostaje natywny build ring.
Provider kryptografii jest jawny i lokalny dla klienta. Pool ma limit ośmiu
bezczynnych połączeń oraz timer do ich sprzątania.

Live ujawnił trzy różnice jednego bitu w domyślnym parserze f64 serde_json.
Włączenie `float_roundtrip` usunęło je bez nowej zależności. SDK nie dziedziczy
`arbitrary_precision` z silnika; odrębny konsument potwierdził poprawne untagged/flatten
dla f64. `Error::Debug` nie ujawnia body HTTP ani szczegółów transportu.

Buildy czterech paczek są rozdzielone i nie mają sekretów/OIDC. Publisher weryfikuje
tylko własny artefakt SHA-256. Rust nie kompiluje zależności w publisherze:
`cargo package --no-verify` daje archiwum bitowo równe wcześniej skompilowanemu,
zweryfikowanemu `.crate`; `cargo publish --no-verify` nie wykonuje makr i build.rs.
Bootstrap token nie jest zmienną całego joba. PyPI używa obrazu przypiętego digestem.
Python buduje wheel z sdist przez jeden hash-locked backend; npm ma jedynie przypięty
kompilator, NuGet locked restore i mapowanie źródeł.

Archiwa sprawdzono odczytowo: wheel nie ma plików .pth, npm nie ma hooków instalacyjnych
ani zależności runtime, NuGet nie ma tools/build/buildTransitive. Zweryfikowano sumy
pięciu dystrybucji: [SHA-256](distribution-SHA256SUMS).
Podsumowanie niezależnych przeglądów Astra XHigh i Opus 5.5 XHigh:
[review](review-r2.md). Historyczne uwagi sprzed poprawek pozostają w
[raporcie pierwszej rundy](../sdk-supply-chain-review-2026-10-13/README.md).

## Kontrole jakości i ograniczenia

Przeszły: `cargo fmt`, Clippy całego workspace release/all-targets Linux oraz Intel,
Rustdoc tych wariantów z `-D warnings`, `cargo package` z weryfikacją kompilacji,
niezależny `cargo deny` SDK obejmujący Linux/macOS/Windows, check-paste-exception,
actionlint i ShellCheck, checker wersji, mypy strict, budowanie ESM/CommonJS i deklaracji,
wheel z sdist, locked restore/build przykładu C# i NuGet pack, kontrola hashów artefaktów.
OSV nie wykazał znanych advisory w dziewięciu przypiętych paczkach Python/npm.
Nie wykonano publikacji jako sposobu kontroli workflow.

`cargo deny` całego workspace zwraca błąd dla istniejącego w silniku
`unsafe-libyaml 0.2.11`, RUSTSEC-2024-0449 (brak utrzymania). Biblioteka nie występuje
w grafie SDK. Nie zmieniano reguł deny ani nie dodawano wyjątku. Pełna bramka CI
workspace pozostaje z tego powodu problemem do osobnego rozwiązania.

Środowiska konsumenta: Python 3.11.17, Node 26.8.2, biblioteka C# net8.0 uruchomiona
w aplikacji .NET 10 z SDK 10.0.401, Rust 1.95.0, Chromium 152. Nie potwierdzono
osobnego runtime .NET 8 ani minimalnego Node 22. Rozwiązanie grafów i audyt licencji
macOS/Windows nie zastępują kompilacji lub runtime na tych systemach. Nie uruchamiano
lokalnie Metal/CUDA ani porównania jakości modelu z upstream FP32.

To porównanie transportu SDK z tym samym działającym serwerem, nie benchmark,
audyt implementacji inference ani gwarancja bezpieczeństwa wszystkich przyszłych zależności.
Kontrola OSV/RustSec dotyczy zapisanych wersji i znanych advisory w dniu sprawdzenia.
Aplikacje konsumentów rozwiązują własne zakresy wersji i powinny utrzymywać swoje locki.

Pierwsza publikacja nadal wymaga kont/uprawnień i konfiguracji trusted publishing,
ochrony tagów/środowisk w GitHub oraz review zmian workflow. Publiczne API rejestrów
zwróciły 404 dla wskazanych nazw w momencie kontroli; nie potwierdza to rezerwacji nazw
ani prawa publikacji. Nie publikowano paczek, nie wykonywano push/tagów ani zmian kont.
Instrukcje konfiguracji: [docs/SDK.md](../../docs/SDK.md).
