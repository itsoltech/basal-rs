# Kontrole jakości Rust w CI

[lint.yml](../.github/workflows/lint.yml) uruchamia się dla każdego PR oraz po zmianach kodu lub konfiguracji CI na
`main`. Check `lint` jest zbiorczą bramką: wymaga sukcesu wszystkich poniższych jobów, także obu wariantów macierzy
Rust. Błąd, anulowanie lub pominięcie wymaganego joba oznacza błąd bramki. Nowszy commit anuluje poprzedni przebieg.

## Kontrole automatyczne

- **Formatowanie:** `cargo fmt --all --check` dla całego workspace.
- **Linux bez GPU oraz Metal:** Clippy dla wszystkich crate'ów i targetów na Linux oraz Apple Silicon.
- **CUDA:** Clippy ze wszystkimi features na Linux z toolkitem CUDA 12.9.1. Kompilacja nie wymaga GPU;
  `CUDA_COMPUTE_CAP=80` ustala architekturę kerneli Candle, a `CUDA_COMPUTE_CAPS=80,89,90` sprawdza kompilację
  własnych kerneli PTX dla wszystkich architektur dystrybuowanych w wydaniu.
  Pobieranie przypiętych zależności jest osobnym krokiem; w kontenerze Cargo używa HTTP/1.1 i pięciu ponowień
  błędów sieciowych. Błąd pobierania blokuje job, zanim rozpocznie się kontrola kodu.
- **Dokumentacja:** Rustdoc na każdej z tych platform, z prywatnymi elementami, bez dokumentowania zależności.
  Ostrzeżenia, w tym błędne odsyłacze, są błędami CI.
- **Zależności:** `cargo deny` dla Linux i Apple Silicon, ze wszystkimi features, według [deny.toml](../deny.toml).
  Blokuje advisory o podatnościach, unsoundness i braku utrzymania, wycofane wersje, niedopuszczone licencje,
  nieznane źródła oraz wildcardy zależności z registry. Własne, niepublikowane crate'y mogą mieć zależności
  ścieżkowe bez wersji. Duplikaty wersji są raportowane jako ostrzeżenia i przez `cargo tree --duplicates`;
  ich sensowność wymaga przeglądu zależności upstream.
- **Zakres wyjątku:** dodatkowa kontrola blokuje zmianę wersji `paste`, nową bibliotekę zależną od niego lub
  wygaśnięcie zatwierdzonego wyjątku. Szczegóły poniżej.
- **Workflow:** przypięte wersje `actionlint` i ShellCheck sprawdzają wszystkie workflow, osadzony kod powłoki
  oraz skrypty polityki CI.
  `actionlint` pochodzi z binarnego wydania upstream i jest weryfikowany przez przypiętą sumę SHA-256;
  `taiki-e/install-action` instaluje tylko obsługiwany przez nią ShellCheck, bez fallbacku do Cargo.

Wszystkie polecenia Cargo dotyczące zależności używają `--locked`. Wersja Rust jest ustalona na 1.95.0 w
[rust-toolchain.toml](../rust-toolchain.toml), a crate'y dziedziczą `rust-version` i politykę lintów z workspace.
Akcje workflow jakości są przypięte do commitów; joby mają timeouty, token z `contents: read`, bez utrwalania
poświadczeń checkoutu. Aktualizacja toolchainu obejmuje też wersje w workflow oraz obrazach budujących wydania.

Job CUDA pobiera gotowy obraz `nvidia/cuda:12.9.1-devel-ubuntu24.04`. PR nie buduje ani nie publikuje obrazu Docker
aplikacji. Publikacja obrazu pozostaje częścią wydania lub ręcznego przebiegu z tagiem wydania.

## Reguły lintów

[Cargo.toml](../Cargo.toml) definiuje wybrane reguły dziedziczone przez każdy crate:

- `unsafe_op_in_unsafe_fn`: operacje unsafe muszą mieć jawne bloki także wewnątrz unsafe fn;
- `clippy::undocumented_unsafe_blocks`: bloki i implementacje unsafe wymagają komentarza `SAFETY:`;
- `clippy::allow_attributes_without_reason`: wyjątki od lintów wymagają `reason`;
- `clippy::dbg_macro`, `clippy::todo`, `clippy::unimplemented`: kod diagnostyczny i niewypełnione implementacje
  nie mogą pozostać w produkcyjnym kodzie workspace.

Clippy traktuje wszystkie pozostałe ostrzeżenia jako błędy. Nie włączamy całych grup `pedantic` ani `restriction`.
Preferujemy lokalne `#[expect(..., reason = "...")]`, gdy lint rzeczywiście występuje; nieaktualne oczekiwanie
powoduje błąd. Wyjątki zależne od backendu pozostają lokalnymi `cfg_attr(..., allow(..., reason = "..."))`.

## Sprawdzanie lokalne

Na macOS te polecenia obejmują Metal; na Linux bez toolkitu obejmują kod bez CUDA:

```sh
cargo fmt --all --check
cargo clippy --locked --workspace --release --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --release --no-deps --document-private-items
cargo deny --locked --all-features check --hide-inclusion-graph
bash tools/ci/check-paste-exception.sh
cargo tree --locked --workspace --all-features --duplicates
actionlint
```

Narzędzia użyte w CI: `cargo-deny 0.19.8`, `actionlint 1.7.11` i `ShellCheck 0.11.0`. ShellCheck musi być na `PATH`,
aby actionlint kontrolował kod powłoki. Cargo deny pobiera aktualną bazę RustSec; nie wyłączamy jej odświeżania.

Na Linux z CUDA 12.9.1 dodatkowo:

```sh
export CUDA_COMPUTE_CAP=80
export CUDA_COMPUTE_CAPS=80,89,90
cargo clippy --locked --workspace --release --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --release --all-features --no-deps --document-private-items
```

## Przegląd i ograniczenia

Automatyczne kontrole uzupełniają `itsolpowers:rust-implementation`, `itsolpowers:rust-review` oraz
`itsolpowers:itsol-self-review`. Nadal przeglądamy:

- ownership, koszt kopii i alokacji oraz minimalną powierzchnię publicznego API;
- typowane błędy, zachowanie łańcucha przyczyn i walidację danych na granicach;
- rzeczywistą poprawność warunków `SAFETY:`, lifetimes FFI i implementacji `Send`/`Sync`;
- limity kolejek, timeouty, locki, lifecycle zadań, obsługę panik i shutdown;
- zasadność optymalizacji, nowych zależności i wyjątków od lintów lub polityki zależności.

CI nie wykonuje inferencji ani nie sprawdza zgodności wyników z upstream i wydajności. Te wymagania pozostają
opisane w [BENCHMARKS.md](BENCHMARKS.md); pomiary zapisujemy w nowych raportach. Projekt nie ma testów, więc CI
nie uruchamia `cargo test`, doctestów, nextest, coverage ani Miri. Nie ma też SQLx ani własnych procedural macros,
które wymagałyby bramek `cargo sqlx prepare` lub trybuild.

Wyjątek od advisory wymaga konkretnego ID, uzasadnienia, ograniczenia do istniejących zależności oraz terminu
ponownego przeglądu. Nie wyłączamy całej kategorii advisory, żeby przepuścić pojedynczy przypadek.

Zatwierdzony wyjątek: [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436.html) dla `paste 1.0.15`.
To informacja o braku utrzymania, bez wskazanej podatności i bez poprawionej wersji. `paste` jest makrem proceduralnym
wykorzystywanym podczas kompilacji przez przypięte wersje `tokenizers`, `gemm` i `pulp`; nie używamy go bezpośrednio
w kodzie projektu. Wyjątek obejmuje tylko obecne wersje bibliotek zależnych od `paste`, wymienione w
[.github/paste-dependents.txt](../.github/paste-dependents.txt).
Nowa biblioteka lub aktualizacja tych wersji wymaga ponownego przeglądu zamiast automatycznego rozszerzenia wyjątku.

Termin przeglądu i wygaśnięcia: **2027-01-08 (UTC)**. Przypięte `cargo-deny 0.19.8` obsługuje dla wyjątku tylko
ID i uzasadnienie, dlatego [tools/ci/check-paste-exception.sh](../tools/ci/check-paste-exception.sh) sprawdza w CI
termin i dokładną listę wersji bibliotek zależnych od `paste` oraz wersję samego `paste`. Od wskazanego dnia
skrypt blokuje CI. Samodzielne `cargo deny check` nie egzekwuje tego terminu ani listy; lokalnie uruchamiamy też skrypt.
Gdy `paste` zniknie z zależności, `unused-ignored-advisory = "deny"` wymusi usunięcie niepotrzebnego wyjątku.
Pozostałe advisory, także przyszłe advisory bezpieczeństwa dotyczące `paste`, nadal podlegają kontroli.
