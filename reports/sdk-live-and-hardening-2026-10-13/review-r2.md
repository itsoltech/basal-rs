# Astra XHigh i Opus 5.5 XHigh — runda 2 i poprawki

Data: 2026-10-13. Przeglądy były odczytowe. Nie publikowano paczek ani nie zmieniano
ustawień serwera i rejestrów. Ustalenia zweryfikował rodzic; nie wykonano pełnej trzeciej rundy.

## Astra GPT-6 XHigh

Potwierdziła zamknięcie istotnych uwag pierwszej rundy: kompilacja Cargo oddzielona
od OIDC/tokenów, hash-locked Flit Core, rozdzielenie builderów, jeden kompilator TS,
obraz PyPI po digest, audyt Python/npm/Windows, redagowany Error Debug i usunięcie
arbitrary_precision z SDK. Nie wskazała nowego Blockera.

Znalazła dwa nieblokujące błędy lifecycle: brak timera do idle pool Hyper oraz brak
anulowania niedokończonego Response po wadliwym UTF-8. Oba naprawiono. Rodzic
potwierdził anulowanie strumienia, kompilację i pakowanie Rust. Astra odczytała
poprawione źródła, dist i .crate, potwierdzając brak otwartych Blockerów i Should
w swoim zakresie.

## Opus 5.5 XHigh

Potwierdził usunięcie wykonania zależności Cargo z tokenem, floating Python build tools,
arbitrary_precision, mutowalnego obrazu PyPI, Rust Debug body i nadmiernego grafu Reqwest.
Sprawdził także konsumenta, który sam włącza arbitrary_precision: typowana odpowiedź
SDK nadal deserializuje się poprawnie. Werdykt: brak hard blockerów, dodatkowe uwagi
do zamknięcia przed pierwszym tagiem.

| Ustalenie Opusa | Poprawka i dowód rodzica |
|---|---|
| Medium: builder może podmienić artefakt innego języka pod tą samą nazwą | Builder eksportuje artifact-id z upload-artifact; publisher wymaga niepustego numerycznego ID z właściwego `needs.build-*` i pobiera po tym ID, nie nazwie. Pinned action.yml potwierdza obsługę artifact-ids; actionlint przechodzi. Podmiana nazwy tworzy inny ID; usunięcie oryginału blokuje pobieranie/publikację. |
| Low–Medium: HTTPX dekompresuje chunk przed sprawdzeniem limitu | `Accept-Encoding: identity` oraz odrzucenie Content-Encoding przed iter_bytes/aiter_bytes. Chroni także przed proxy ignorującym nagłówek. Kontrole rzeczywistego lokalnego HTTP sync/async potwierdziły nagłówek i odrzucenie gzip przed dekodowaniem. Nie deklarujemy zmierzonego maksimum pamięci. |
| Low: TS body błędu pojawia się w domyślnym logowaniu Node | Pole body jest nieenumerowalne, z zachowanym jawnym dostępem. Kontrola z zainstalowanego npm potwierdziła brak syntetycznego sekretu w util.inspect i JSON.stringify. |
| Low: głęboki JSON Pythona ujawnia RecursionError poza hierarchią SDK | RecursionError jest przechwytywany razem z błędami parsowania. Sync/async zwracają BasalProtocolError z zachowaną przyczyną dla odpowiedzi 200. |
| Nit: digest PyPI był snapshotem release-v1 | Manifest GHCR nazwanej wersji v1.14.2 pobrano anonimowo, hash policzono i porównano z Docker-Content-Digest. Workflow używa tego digestu i komentarza wersji. |

Rodzic ujednolicił też nazwy paczek PyPI w zapytaniu OSV zgodnie z regułą
normalizacji (`[-_.]+` → `-`, małe litery), usuwając wątpliwość dotyczącą underscore.
OSV nadal nie wskazuje znanych advisory w dziewięciu przypiętych paczkach.

Po tych poprawkach przebudowano i zainstalowano Python/npm. Ponowiono **48 porównań**
Python sync/async, Node native/adapter i Chromium native/adapter: zero różnic,
w tym binary64. Nie ponawiano niezmienionych C#/Rust. Wcześniejsze 16 porównań tych
dwóch języków pozostaje aktualne, co daje 64 zgodne przypadki końcowego zestawu.
Nie nadpisano wcześniejszych danych raportu; najnowsze dowody i sumy dystrybucji są w
[post-review](post-review/comparison-summary.json),
[kontrolach](post-review/checks.json) i
[końcowych SHA-256](post-review/distribution-SHA256SUMS).

## Granice werdyktu

Rodzic zamknął wszystkie wskazane materialne ustalenia w zakresie SDK i workflow.
Wiązanie artifact-id sprawdzono statycznie względem przypiętych akcji; nie wykonano
ataku na GitHub ani rzeczywistej publikacji/OIDC. Bitową powtarzalność .crate
sprawdzono lokalnie między package z weryfikacją i no-verify, na tym samym brudnym
checkoutie. Nie zastępuje to przebiegu na czystym, chronionym tagu.

Deno/Bun/Workers, runtime macOS/Windows, minimalny Node 22 i osobny runtime .NET 8
pozostają niezweryfikowane. Opus nie przeprowadził pełnego ponownego audytu OpenAPI
względem wszystkich handlerów. Zakresy bibliotek mogą w przyszłości rozwiązać inne
wersje niż przypięte grafy. Ustawienia kont, ochrony tagów/środowisk i zdalne
uprawnienia rejestrów pozostają do konfiguracji. Istniejący advisory unsafe-libyaml
w silniku blokuje pełny check workspace, ale nie dotyczy grafu SDK.
