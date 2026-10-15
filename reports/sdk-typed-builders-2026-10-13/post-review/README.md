# Poprawki po review typowanego SDK

Data: 2026-10-13. SDK 0.1.0, przed publikacją. Ten raport uzupełnia wcześniejszy
[raport typowanych SDK](../README.md); nie nadpisuje jego pomiarów ani hashów.

## Review i zamknięte uwagi

Zakończył się dodatkowy odczytowy review **Opus 5.5 XHigh**, dotyczący fabryk,
typowanych odpowiedzi i ergonomii czterech SDK. Poprzedni review nowej funkcji
wykonała **GPT-6 Astra XHigh**; jej uwagi i sprawdzenie poprawek opisuje raport główny.
Rodzic zweryfikował uwagi Opusa względem kodu serwera, kontraktu i gotowych paczek.

- **Should: evidence w TS/C#.** Usunięto warunek `end >= start`, zachowując
  sprawdzenie nieujemnych offsetów i pozostałe kontrole. Natywne `best_spans`
  w `crates/basal-core/src/evidence.rs` uwzględnia maskowane pary, a pusty zakres
  może dać `text: ""`, `probability: 0`, `start > end`. Kontrakt EvidenceSpan
  wymaga nieujemnych offsetów, bez relacji między nimi. Python/Rust już to
  przyjmowały. Nie zmieniano serwera ani obliczeń evidence.
- **API act Python/C#.** Typowane wyniki zachowują teraz `refused`/`max_error`
  (`Refused`/`MaxError`). To opcjonalne komunikaty serwera; wymagane wartości
  akcji/outcome/kosztów pozostają nieopcjonalne. Ostrzeżenia o niedostępnym
  certyfikowanym progu nie giną przy korzystaniu z ergonomiczną odpowiedzią.
- **Python.** OverflowError przy ogromnej liczbie odpowiedzi jest zamieniany
  na BasalProtocolError, z zachowaną przyczyną.
- **Rust.** Publiczne opcje, pytania, builder i typowana odpowiedź mają Debug.
  Pożyczone widoki odpowiedzi mają również Clone/Copy. Nie dodano unsafe,
  alokacji w ścieżce odczytu ani wymuszenia kopiowania danych odpowiedzi.
- **Typy act TS.** Dodatkowo ograniczono inferencję outcomes do criteria,
  z domyślnymi true/false. Koszty wymagają wszystkich tych kluczy, a defer_action
  wskazuje jedną ze zdefiniowanych akcji. Brak wierszowego outcome i nieznana
  akcja są odrzucane przez kompilator. Walidacja runtime pozostaje potrzebna
  dla dynamicznego JSON i danych z JavaScript.

Opus potwierdził poprawność sprawdzonych inferencji, lifetimes, identity,
snapshotów, rozpoznawania raw mode i budowania CommonJS. Nie deklarujemy,
że Opus ponownie przejrzał poprawki: ich przyjęcie opiera się na kontroli rodzica
oraz opisanych poniżej dowodach. Nie pozostały otwarte materialne uwagi w tym zakresie.

## Sprawdzenie gotowych dystrybucji

Ponownie zbudowano wheel/sdist, npm ESM/CommonJS/deklaracje, NuGet i zweryfikowany
crate. Konsumenci korzystali z paczek zainstalowanych poza repo/silnikiem,
Rust z zawartości zweryfikowanego .crate. Nie dodano żadnych zależności.

Działający Basal przez SSH z wcześniej potwierdzonym kluczem hosta otrzymał
syntetyczny state `Paid today`, evidence noul i act z max_error. Zwrócił ostrzeżenie
`no certified threshold for this target`; Python/C# zachowały je w typowanym wyniku.

Porównanie: **7 odpowiedzi, 14 wyników pytań, zero różnic** z bezpośrednim HTTP.
Warianty: Python sync/async, Node ESM/Request-Response adapter/CommonJS,
C# i Rust. Sprawdzono wszystkie klucze i wartości oraz dokładne bity binary64;
pominięto tylko usage.latency_ms. Dowody: [request](request.json),
[reference](reference.json), [wyniki](client-results.json),
[podsumowanie](comparison-summary.json).

Ta konkretna odpowiedź live nie miała odwróconego spanu. Osobny lokalny adapter
HTTP podał identyczną odpowiedź ze spanem `{text:"", start:5, end:4, probability:0}`
oraz refusal act. Wszystkie cztery paczki przyjęły go i zachowały metadane.
Osobne wadliwe odpowiedzi z end=-1 i prawdopodobieństwem 10**400 zostały odrzucone
jako błędy SDK; również przez async Python i trzy warianty npm. Są to diagnostyki
kontraktu, nie dodatkowy pomiar częstości tego przypadku w modelu.

Przeszły fmt, Clippy workspace release/all-targets Linux i Intel, Rustdoc obu
wariantów z -D warnings, cargo package z weryfikacją, oddzielny cargo deny SDK,
checker wersji, mypy strict, build/pack C# bez ostrzeżeń i kontrola hashów dystrybucji.
TS strict/noUncheckedIndexedAccess/exactOptionalPropertyTypes potwierdził
oryginalne wywnioskowane typy odpowiedzi oraz nowe ograniczenia act w ESM i CommonJS,
dla fabryki i konstruktora. Konsument Rust kompiluje Debug pytań w własnej strukturze,
Debug odpowiedzi/opcji/buildera i Debug+Copy wszystkich pięciu widoków.

Ostatnie doprecyzowanie typów TS nie zmieniło plików JS ESM/CommonJS:
porównanie bajtowe z wersją sprawdzoną live potwierdziło to przed przepakowaniem.
Aktualne hashe: [distribution-SHA256SUMS](distribution-SHA256SUMS), ścieżki względem sdk/.
Poprzednie hashe dotyczą wcześniejszych dystrybucji i pozostają bez zmian.

## Granice dowodów

Nie dodano trwałych testów, frameworków ani bramek testowych. Diagnostyki uruchamiano
poza repo. Nie zmieniano usługi, konfiguracji ani danych użytkownika; zamknięto
lokalny tunel i helper HTTP. Przeglądarkę sprawdzono w wcześniejszym pomiarze;
ten dodatkowy pomiar obejmował Node z tym samym modułem SDK, bez ponownego browser run.
Pozostałe ograniczenia platform, inferencji i jakości modelu pozostają takie jak
w raporcie głównym. Nie publikowano paczek; konfiguracja kont i uprawnień pozostaje
po stronie użytkownika. Znany advisory unsafe-libyaml silnika nadal dotyczy pełnego
cargo deny workspace, poza niezależnym grafem SDK. Polityki zależności nie osłabiono.
