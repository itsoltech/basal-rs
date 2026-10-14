# Review i model domyślny — 2026-10-14

Cel: sprawdzić spójność feature’u i precyzję podsumowań oraz ustawić Basal 2.0 jako model domyślny.
GPT-6 Astra i Claude Opus 5.5 wykonały po dwie rundy przeglądu. Oba końcowe review
nie pozostawiły konkretnych błędów. Przegląd źródeł nie zastępuje pomiaru modelu.
Raporty używają prostego języka technicznego; nie deklarujemy formalnej zgodności polskich tekstów z ASD-STE100.

- Default: `basal-2.0-4.5B`, rewizja `e5ba9149760d99b73b1963d8e0ccf948b58afacb`.
  Bez konfiguracji wybierają go serve/init/setup oraz lokalny klient. Jawny YAML i `--model` zachowują wybór;
  aliasy mini/4.5B/max nadal wskazują 1.5. Zweryfikowano domyślny i jawny nowy `init`, stary alias `init`
  oraz wybór konfiguracji w `doctor`, bez pobierania wag. Pełny doctor zgłasza brak GPU na hoście.
- Budżet tekstu: `score.levels` jest teraz uwzględniane przed kopiowaniem i tokenizacją.
  Jednorazowy program porównał rzeczywisty core ze starym modułem budżetu z `7db591c`.
  8/8 przypadków ma oczekiwany wynik: pięć małych wejść przyjętych; levels 18 MiB,
  criteria 18 MiB i 5000 gałęzi odrzucone. Stary budżet przyjmował levels 18 MiB.
  Bez pola levels estymaty w tych przypadkach pozostały identyczne. To kontrola przyjęcia wejścia, bez inferencji.
- Korekta opisów: 450 porównań obejmuje choice/noul/score/multi/tool, bez act.
  Dla noul porównano stronę progu, dla score najbardziej prawdopodobny poziom.
  Act ma osobne 17/17 porównań reguł hosta dla zadanych prawdopodobieństw.
  Liczby i historyczne archiwum pozostają niezmienione; poprawiono zbyt szeroki opis w oryginalnym Markdown.
- Zależności: rustls 0.23.45 → 0.23.46 usuwa cztery nowe advisory wykryte przez cargo-deny.
  Solver wybrał dla tempfile getrandom 0.3.4 zamiast 0.4.3, w dopuszczonym zakresie tej zależności.
- Kontrole końcowych źródeł: fmt; Clippy i Rustdoc z `-D warnings` dla Linux, Intel i CUDA all-features;
  cargo-deny, wyjątek paste i diff check — wszystkie zakończone kodem 0.

Nie wykonano nowej sesji GPU ani porównania forwardu po tych zmianach.
Kod forwardu, procesorów i arytmetyki odczytu nie zmienił się względem fingerprintu sesji H100;
zmieniły się wybór domyślnego modelu, budżet wejścia, komunikaty CLI i Cargo.lock.
Metal nie został lokalnie skompilowany, runtime Metal/Intel pozostaje niezmierzony, GitHub CI nieuruchomione.
Poprzednie [pomiary H100](../upstream-2.0-h100-2026-10-14/README.md) zachowują swój opisany zakres.

## Dane poza Gitem

- URI: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-review-2026-10-14/748b7f2b2c2cb64455bb4b5214b4e1b0defc20f97d99119a36bc8e4e87bd2063/measurement-data.tar.gz`.
- SHA-256: `748b7f2b2c2cb64455bb4b5214b4e1b0defc20f97d99119a36bc8e4e87bd2063`; gzip 1413 B.
- Ścieżki: `reports/upstream-2.0-review-2026-10-14/prompt-budget.json` i `verification.json` w tym samym katalogu.
- Pobranie z autoryzacją i anonimowo: SHA archiwum i 2/2 wpisy SHA256SUMS zgodne.
- Odtworzenie według [procedury](../../docs/MEASUREMENT_ARTIFACTS.md): użyj powyższego prefiksu HF
  i nazwy `upstream-2.0-review-2026-10-14` w zmiennych katalogów oraz powyższej sumy archiwum.
