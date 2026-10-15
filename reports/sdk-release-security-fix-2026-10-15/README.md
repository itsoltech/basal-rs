# Usunięcie advisory blokujących wydanie SDK

Data: 2026-10-15. PR #10, pierwsze wydania SDK 0.1.0. Kontrola dependencies
wykryła cztery advisory Rustls wydane 2026-10-14 oraz brak utrzymania
unsafe-libyaml w CLI. Nie zmieniano polityki audytu ani wyjątków.

Rustls zaktualizowano z 0.23.45 do 0.23.46; manifest SDK wymaga co najmniej
0.23.46, aby poprawka dotyczyła również użytkowników z istniejącym lockfile.
Oficjalne advisory wskazują ten próg dla wszystkich czterech problemów:
[RUSTSEC-2026-0346](https://rustsec.org/advisories/RUSTSEC-2026-0346.html),
[RUSTSEC-2026-0347](https://rustsec.org/advisories/RUSTSEC-2026-0347.html),
[RUSTSEC-2026-0348](https://rustsec.org/advisories/RUSTSEC-2026-0348.html),
[RUSTSEC-2026-0349](https://rustsec.org/advisories/RUSTSEC-2026-0349.html).

Tylko CLI zmienia serde_yaml na serde-saphyr 2.0.0 z default-features=false
i jedyną funkcją deserialize. Usunięto serde_yaml/unsafe-libyaml z workspace.
Nie włączano include, include_fs, properties, komentarzy ani serializacji.
Rust SDK nadal nie ma parsera YAML w swoim grafie. Źródło wyboru:
[RustSec](https://rustsec.org/advisories/RUSTSEC-2024-0449.html) i
[dokumentacja serde-saphyr](https://docs.rs/serde-saphyr/2.0.0/serde_saphyr/).

## Sprawdzenie

Przeszły cargo fmt, Clippy całego workspace/all-targets/release, Rustdoc całego
workspace z ostrzeżeniami jako błędami, cargo deny workspace/all-features,
cargo deny samego SDK, kontrola dotychczasowego wyjątku paste i git diff --check.
Duplikaty istniejących wersji oraz niewykorzystane wpisy dozwolonych licencji
pozostają ostrzeżeniami. Nie dodano nowych wyjątków ani trwałych testów.

Izolowany program porównał stare i nowe deserializowanie rzeczywistych definicji
ServeConfig/ModelConfig z dodanym wyłącznie Serialize dla porównania wartości.
Default cache znormalizowano do stałego placeholdera; nie odczytywano modeli
ani GPU. Pięć poprawnych konfiguracji (domyślna, serve.example.yml, wszystkie
pola, alias, komentarze/Unicode) dało identyczne wartości. Pięć błędnych
(nieznane pole, duplikat, błędny typ, składnia, wiele dokumentów) odrzuciły
oba parsery. Nie porównywano tekstów diagnostyk; mogą się różnić. To sprawdzenie
ograniczonego zestawu konfiguracji, bez deklaracji pełnej zgodności parserów.
[Wyniki](config-parser-comparison.json), [graf SDK](sdk-dependency-summary.json).

Zmiana nie dotyczy obliczeń modelu; nie wykonywano pomiarów jakości,
wydajności ani runtime GPU. Zmieniona zależność CLI nie jest publikowana
jako część SDK. Nie modyfikowano działającego serwera.
