# Batch i Files API w czterech SDK — 2026-10-20

Uzupełniono klientów Python i async Python, TypeScript/JavaScript, C# oraz Rust
o dziewięć operacji zasobów: upload/list/get/delete/download plików oraz
create/list/get/cancel batchy. Wersje: Python/npm/.NET 0.4.0, Rust 0.5.0;
serwer 0.1.12. Przykłady: [SDK.md](../../docs/SDK.md#batch-i-files-api).

Transport zachowuje bearer, prefiks proxy, weryfikację TLS, timeouty,
anulowanie i ograniczenia odpowiedzi. Tworzenie obsługuje `Idempotency-Key`,
metadane i retencję wyników. Pobieranie oddaje bajty bez interpretacji JSON;
SDK nie wykonują automatycznych ponowień ani odpytywania statusu. DTO zachowują
dodatkowe pola odpowiedzi. Typy statusów nie powodują odrzucenia nowego statusu
przez deserializację transportu. Rust dodatkowo udostępnia odczyt health.

## Metoda i wynik

Rodzic uruchomił rzeczywisty `BatchService`, Axum, SQLite i filesystem pod
prefiksem proxy. Prywatny operator inicjalizował syntetyczny katalog oraz
wyłączał przetwarzanie. Nie ładował silnika modelu i nie zastępował odpowiedzi
HTTP fikcyjnymi wynikami. Każdy klient używał tokenu i rzeczywistych plików
oraz rekordów w jednorazowej bazie.

| Wariant | Obserwowany wynik |
|---|---|
| Python sync | wszystkie 9 operacji; replay tego samego klucza zwraca ten sam batch; Unicode i bajty pliku zgodne; upload bytes i chunked |
| Python async | wszystkie 9 operacji; zgodne bajty; upload async iterable i bytes |
| TypeScript ESM | wszystkie 9 operacji; replay, Unicode i zgodne bajty; upload bytes i ReadableStream |
| JavaScript CJS | te same operacje i wyniki co ESM |
| C# | wszystkie 9 operacji; replay, Unicode i zgodne bajty; upload ze strumienia oraz pobranie do strumienia |
| Rust | wszystkie 9 operacji; replay, Unicode i zgodne bajty |

Anulowanie przyjmowano jako `cancelling` lub `cancelled`. Przetwarzanie było
wyłączone, więc nie potwierdzano inferencji ani generowania wyników modelu.
Syntetyczny katalog służył wyłącznie operacjom kontrolnym.

Osobne prywatne obserwatory HTTP rejestrowały żądania i odpowiedzi błędów:
Python 35 obserwacji, C# 34, TypeScript zapis wire oraz zachowania anulowania
uploadu, Rust 15 żądań obejmujących zasoby, health i przypadki transportowe.
Sprawdzono framing multipart, kolejność pól retencji przed plikiem, token,
prefiks, kodowanie identyfikatorów/query, blokowanie wstrzyknięć nagłówków,
bajty odpowiedzi, limity, odmowę przekierowań, timeouty i `Retry-After`.
Te obserwatory używały kontrolowanych odpowiedzi, nie modeli.

Rodzic poprawił dwa szczegóły po integracji: TypeScript zachowuje zwykły limit
koperty błędu przy mniejszym limicie pobieranego pliku (rzeczywiste 404 przy
`maxBytes: 1` pozostaje błędem HTTP); C# żąda `identity` i odrzuca pozostałe
kodowania odpowiedzi (potwierdzono kontrolowanym nagłówkiem gzip).

## Kompilacja i pakowanie

- Rust 1.95.0: release Clippy wszystkich targetów i prywatny Rustdoc z `-D warnings`, workspace domyślny i Intel: PASS; `cargo package` z kompilacją paczki: PASS.
- Python: compile/import, pyright, przypięty Flit Core, sdist i wheel oraz import wheel na 3.13 i 3.15: PASS.
- TypeScript 6.0.3: ESM/CJS, deklaracje, kompilacja konsumenta i npm pack dry-run: PASS; po korekcie rodzica ponownie build i oba warianty z natywnym serwerem: PASS.
- .NET SDK 10.0.401, target net8.0: restore locked, build przykładu i pack bez ostrzeżeń: PASS; po korekcie rodzica build operatora i pack: PASS.
- Manifesty/locki wersji, fmt/diff, cargo-deny, wyjątek paste oraz OSV dla dziewięciu przypiętych pakietów Python/npm: PASS.

Native C# uruchamiano na runtime .NET 10 przez `DOTNET_ROLL_FORWARD=Major`,
ponieważ lokalnie brak runtime .NET 8. Python 3.11/3.12 i przeglądarkowy
upload strumieniowy TypeScript nie były uruchamiane lokalnie. Bieżące CI
wydania ma oddzielny zakres platform; nie zastępuje pomiarów GPU.

Nie dodano zależności ani testów, frameworków, snapshotów i bramek testowych.
Operator HTTP i usługi pomocnicze zakończono. Nie zmieniono produkcji ani
obliczeń modelu. Publikacja SDK jest osobnym etapem workflow; lokalne pakowanie
nie jest dowodem obecności paczki w rejestrze.

## Ograniczenia

Pobieranie do bufora wymaga jawnego limitu pamięci (domyślnie 8 MiB);
duży plik wymaga jego zwiększenia. C# oferuje także strumień z limitem.
Rust upload tworzy kopię multipart wejścia; Python, C# i TypeScript mają
również warianty strumieniowe. Wczesna odmowa dużego uploadu może w zależności
od biblioteki transportowej ujawnić się jako błąd transportu przed odebraniem
statusu HTTP. Nie zmierzono przepustowości ani zużycia pamięci SDK w skali.
Brak nowej kwalifikacji GPU, modeli, baz zdalnych, wykonania 24h i HA.

## Archiwum lokalne

Archiwum jest nieopublikowane; nie potwierdzono pobrania z HF.

- Plik: `.cache/measurement-artifacts/batches-sdk-20261020/measurement-data.tar.gz`.
- SHA-256: `da186327e48bb959a3f04063e64ee513c1644602e7895921e32f4bb45ca0bcad`.
- Rozmiar: 1 870 915 B; nowe rozpakowanie: 430/430 sum i kompletność manifestu zgodne.
- Planowane URI po osobnej publikacji: `hf://buckets/itsoltech/basal-rs/measurements/batches-sdk-20261020/da186327e48bb959a3f04063e64ee513c1644602e7895921e32f4bb45ca0bcad/measurement-data.tar.gz`.

Pod `reports/batches-sdk-2026-10-20/` są: `sdk-source/` (83 pliki wraz
z lockami, narzędziami i dokumentacją), `sdk-source-sha256.json`, dokładny
`native-server-source/`, `evidence/` oraz `METHOD.md`. Sesyjne tokeny, bazy,
ready/config, pakiety binarne i środowiska zależności nie są dołączone.
Diagnostyka ma zastąpione ścieżki sesji. Przed rozpakowaniem zweryfikuj hash
archiwum, potem uruchom `sha256sum -c SHA256SUMS`; `METHOD.md` opisuje
odtworzenie z nowych prywatnych katalogów. Wcześniejszych danych nie zastępuj.
