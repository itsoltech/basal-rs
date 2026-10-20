# Zbieranie przebiegów Batch API

`capture.py` pobiera stan **istniejącego** batcha, `/health/batches` i końcowe
pliki wyników. Służy operatorowi do zbierania danych. Nie tworzy pracy,
nie generuje ruchu online, nie przydziela GPU i nie wystawia werdyktu
kwalifikacji. To nie jest zestaw testów ani bramka CI.

Uruchom z katalogu repozytorium, po utworzeniu batcha według
[docs/BATCHES.md](../../docs/BATCHES.md):

```sh
python3 tools/batches/capture.py \
  --base-url http://127.0.0.1:8000 \
  --token-file ./batch-data/local-token \
  --batch-id batch_REPLACE \
  --duration 60 --interval 1 \
  --output .cache/measurement-artifacts/batches-session-YYYYMMDD/capture-01
```

Katalog wyjściowy musi być nowy. Narzędzie nadaje mu prawa `0700`, a plikom
`0600`. Nie podawaj tokenu w argumentach ani w adresie URL. Dla adresu innego
niż loopback wymagane jest HTTPS z weryfikacją certyfikatu. Przekierowania
są odrzucane, aby token nie trafił do innego serwera.

`observations.jsonl` zawiera kolejne odpowiedzi API z czasem monotonicznym
od startu. Po stanie terminalnym narzędzie pobiera `output.jsonl` i
`error.jsonl`, jeśli batch je wskazuje. Domyślny limit każdego pobrania to
200 MiB; nieukończony plik zachowuje końcówkę `.part`. `--duration` ogranicza
okno próbkowania; trwające operacje sieciowe i pobrania mogą je przedłużyć.
`--timeout` jest timeoutem operacji socketu, a nie limitem całej sesji GPU.

Dane są **prywatnymi surowymi wynikami**. Mogą zawierać treść żądań, odpowiedzi
oraz szczegóły diagnostyczne usługi. Przed archiwizacją przejrzyj je według
[MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).

Sam zapis health i JSONL nie dowodzi zgodności bitowej, szczytu VRAM ani
priorytetu online. Numerykę porównujemy istniejącymi `basal export` i
`basal compare`. Opóźnienia online wymagają identycznego harmonogramu ruchu
w baseline i wariancie z background, a RAM/VRAM osobnego pomiaru. Raport
musi podać konfigurację, wejścia, wersję kodu, skład mikroserii i luki.

Sprawdzono składnię, `--help` i rzeczywisty lokalny polling batch/health:
osiem odczytów 200, prywatny token i katalog wyjściowy. Dowód:
[odzyskanie walidacji](../../reports/batches-pinned-default-r2-2026-10-19/README.md).
Pobrania terminalnych plików, zdalny origin i pełny przebieg kolektora z GPU
pozostają poza tą próbą.
