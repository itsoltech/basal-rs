# Usunięcie health z SDK

Data: 2026-10-14, SDK 0.1.0 przed publikacją. Health jest operacją orkiestracji
usługi. Usunięto metody sync/async oraz publiczne HealthResponse we wszystkich
czterech SDK, wpis i schemat w kontrakcie SDK OpenAPI oraz wzmianki w aktualnych
README i SDK.md. Zachowano decide i models. Nie zmieniano serwera ani jego
endpointu; to usunięcie powierzchni SDK, bez zmiany konfiguracji dostępu HTTP.

Zbudowano sdist/wheel, npm ESM/CommonJS, NuGet i crate. Inspekcja zainstalowanego
wheelu i NuGet oraz obu formatów npm potwierdziła brak health API. Przeskanowano
również źródła i zawartość dystrybucji. Kontrakt ma tylko /v1/basal i /v1/models,
a wszystkie jego lokalne odsyłacze rozwiązywane są poprawnie.

Przeszły cargo fmt, Clippy dla basal-client ze wszystkimi targetami, Rustdoc
z ostrzeżeniami jako błędami, cargo deny grafu SDK, kompilacja przykładu C#,
kontrola wspólnej wersji, hashe wszystkich dystrybucji i git diff --check.
Crate pakowano z --no-verify; kompilację źródeł obejmuje Clippy. Pełny cargo deny
workspace nadal zgłasza RUSTSEC-2024-0449 w unsafe-libyaml silnika; ta zależność
nie występuje w grafie SDK. Nie zmieniano polityki ani wyjątków zależności.

Zmianę przejrzano lokalnie według rust-implementation, rust-review i self-review.
Usunięcia nie zmieniają implementacji pozostałych metod, ownership transportu,
timeoutów ani walidacji. Nie wykonano inferencji ani pomiarów modelu dla tej
iteracji. Nie dodano testów/frameworków, zależności ani publikacji.
Wcześniejsze raporty pozostają bez zmian.

[Wyniki kontroli](verification.json), [hashe dystrybucji](SHA256SUMS)
względem katalogu repozytorium.
