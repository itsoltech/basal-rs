# Uproszczenia konfiguracji po przeglądzie Opus 5.5

Przegląd read-only obejmował cały zmieniony kod Rust/CUDA i cztery runnery
Python w commicie `7dfc6aac10135bfabd4b8fac79775d03d04bcca3`. Rodzic
zweryfikował wskazane miejsca w źródłach. Agent nie uruchamiał kompilatora,
modeli ani pomiarów GPU.

## Wprowadzone zmiany

- Osobna funkcja `default_inference_mode` zwraca dotychczasowe `auto`.
  Domyślny tryb nie zależy już od funkcji domyślnej precyzji.
- Wspólna lista `INFERENCE_MODES` zasila walidację obu wejść CLI i komunikat
  błędu parsera konfiguracji. Nazwy, kolejność i komunikaty pozostają takie same.
- Połączenie z `main` (`f34247e`) zachowuje niezależne SDK i przypiętą wersję
  `serde-saphyr =2.0.0`. Jedyny konflikt dotyczył zakresu wersji tej zależności;
  lockfile nadal wybiera 2.0.0.

## Pozostałe uwagi przeglądu

Jawny `accurate` wybiera przenośny `attn_tree_tc`, także na H100, zamiast
automatycznej ścieżki WGMMA. To ograniczenie opisano w architekturze.
Przywrócenie doboru WGMMA wymaga nowego porównania bitowego i pomiaru H100;
tego porównania nie wykonano. Wyniki kampanii Ada nie potwierdzają wydajności
tego trybu na H100.

Pozostawiono eksperymentalne TC64/SW128, ponieważ istniejące runnery używają
ich do A/B. Usunięcie wymagałoby również porównania generowanego PTX.
Nie przenoszono logiki z runnerów do wspólnego modułu: ich zapisany hash
obejmuje pojedynczy plik. Nie zmieniano wyboru kernela w pętli warstw,
kontroli admission ani zabezpieczeń specjalizacji f16.

## Weryfikacja i ograniczenia

Lokalnie na Linux wykonano:

- `cargo fmt --all --check` i release build CLI;
- Clippy workspace/all-targets oraz Rustdoc/private-items z ostrzeżeniami
  jako błędami, bez GPU i z feature Intel;
- `cargo-deny 0.19.8`, kontrolę wyjątku `paste` i przegląd duplikatów
  zależności; aktualna baza advisory nie zgłosiła błędów;
- kontrolę wersji SDK po scaleniu `main`;
- 33 porównania rodzic/kandydat: help i błędy CLI, 12 wygenerowanych
  konfiguracji init oraz odrzucenie trzech nieznanych trybów YAML przed
  rozwiązywaniem modelu. Wyjścia i kody zakończenia były równe po ujednoliceniu
  nazwy pliku wykonywalnego w komunikatach Clap.

Źródła `basal-core`, `basal-gpu` i `tools/perf` są identyczne z `7dfc6aa`.
Nowe zmiany dotyczą wyłącznie konfiguracji i walidacji CLI. Nie wykonano
nowego forwardu modelu ani pomiaru wydajności; kontrole CLI nie są dowodem
bitowej zgodności wyników modelu. Dotychczasowe dowody i ich ograniczenia
pozostają w [raporcie kampanii Ada](../precision-performance-ada-2026-10-16/README.md).
Kompilację CUDA i Metal sprawdza CI finalnego PR; nie zastępuje ona
weryfikacji runtime GPU. Przegląd ownership, błędów, API oraz self-review
nie wykazał dodatkowych problemów w tym wąskim refaktorze.
