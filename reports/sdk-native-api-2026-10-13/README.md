# SDK natywnego API: kontrola lokalna, 2026-10-13

Sprawdzono cztery paczki SDK `0.1.0` dla `/v1/basal`. Kod serwera,
inferencji, kalibracji i kerneli nie został zmieniony.

## Wyniki

- Python: zbudowane wheel i sdist, `twine check` bez błędów; `mypy --strict`
  bez błędów na Pythonie 3.11. Paczka zawiera `py.typed`, LICENSE i NOTICE.
- npm: zbudowane ESM, CommonJS i deklaracje typów; `npm pack` bez błędów.
  Import ESM i `require` sprawdzone po instalacji archiwum.
- C#: przykład z targetem .NET 8 skompilowany bez ostrzeżeń; `dotnet pack`
  utworzył NuGet. Zainstalowaną paczkę uruchomiono na runtime .NET 10.
- Rust: `cargo package --locked --allow-dirty -p basal-client` zbudował
  i zweryfikował samodzielny crate; pakiet nie zależy od silnika ani bibliotek GPU.
- `cargo fmt --all --check`, Clippy całego workspace z `--release --all-targets -D warnings`
  i Rustdoc z `RUSTDOCFLAGS="-D warnings"` przeszły na Linux bez GPU oraz
  z feature `basal-cli/intel`.
- OpenAPI 3.1 zwalidowane przez `openapi-spec-validator`; wersje czterech
  manifestów, lock npm, `sdk/VERSION` i kontraktu zgodne z `sdk-v0.1.0`.
- `actionlint 1.7.11` z `ShellCheck 0.11.0` oraz kontrola zakresu wyjątku `paste`
  przeszły. `cargo tree --duplicates` uruchomione; nowe SDK nie wprowadza
  nowych wersji zależności do workspace, tylko feature serde dla indexmap.
- `cargo deny 0.19.8` dla samodzielnie spakowanego Rust SDK z konfiguracją
  `sdk/rust/deny.toml`: advisories, bans, licenses i sources OK. Ostrzeżenia
  dotyczą niewystępujących w tym mniejszym grafie licencji z listy workspace.
- **`cargo deny` całego workspace nie przeszedł:** istniejące
  `serde_yaml → unsafe-libyaml 0.2.11` podlega nowemu advisory
  [RUSTSEC-2024-0449](https://rustsec.org/advisories/RUSTSEC-2024-0449.html).
  Nie dodano wyjątku ani nie osłabiono polityki zależności.

## Ręczna kontrola transportu

Po lokalnej instalacji wheel, tarball npm i NuGet oraz użyciu samodzielnie
spakowanego crate'a wykonano wywołania do tymczasowego serwera HTTP z
syntetycznymi odpowiedziami. Sprawdzono trzy ścieżki API z prefiksem proxy,
stan obiektowy z polskimi znakami i emoji, kolejność kryteriów, bearer token,
odpowiedź noul z evidence i zachowanie dodatkowych pól JSON.

Wszystkie klienty rozróżniły odpowiedzi 422/529/503, zachowały `Retry-After`,
tekst błędu bez JSON oraz zgłosiły niepoprawny JSON odpowiedzi i timeout.
W tych przypadkach nie zaobserwowano automatycznych retry. Sprawdzono także
Python sync/async, anulowanie JS i C#, ownership wstrzykniętego `HttpClient`
oraz zachowanie jawnego null przez przeciążenie C# przyjmujące `JsonObject`.
Diagnostyka była tymczasowa; do repozytorium nie dodano testów ani frameworków.

## Zakres i ograniczenia

Nie wykonano inferencji, porównania wyników z FP32 upstream ani pomiaru
wydajności. Kontrola transportu nie potwierdza jakości modelu, bitowej
zgodności wyników ani działania rzeczywistego serwera pod obciążeniem.
Nie uruchomiono lokalnie Metal/CUDA. Node wykorzystany w kontroli: 26.8.2;
minimalny deklarowany Node 22 i runtime .NET 8 nie zostały osobno uruchomione.
Nie uruchomiono workflow w GitHub Actions i nie opublikowano paczek w
rejestrach; konfiguracja kont i pierwsza publikacja pozostają osobnym krokiem.
