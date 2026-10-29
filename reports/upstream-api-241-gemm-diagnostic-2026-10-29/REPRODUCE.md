# Odtworzenie diagnostyki GEMM J4

Archiwum jest prywatne: zawiera surowe dane środowiska, ścieżki i logi.
Nie publikuj go bez osobnej redakcji. Nie zawiera wag, binarek ani cache buildów.

## Dane i źródła

1. Sprawdź SHA-256 lokalnego `measurement-data.tar.gz` podane w README.
2. Rozpakuj do nowego katalogu i wykonaj w nim `sha256sum -c SHA256SUMS`.
3. Ustaw `EVIDENCE` na rozpakowane
   `reports/upstream-api-241-gemm-diagnostic-2026-10-29/evidence/private/`.

`source.tar` i `source-commit.txt` identyfikują aplikację. `candidate.diff`
zachowuje zmianę względem poprzedniej wersji. `probe/` zawiera dokładny program
Rust i lockfile; `static-evidence/` zawiera handoff, review oraz logi kontroli.
Przed kompilacją przeniesionego probe popraw trzy ścieżki w `Cargo.toml` na
rozpakowane `crates/basal-core`, `crates/basal-gpu` i `vendor/candle-core`.
Użyj zachowanego lockfile i środowiska z logów. W tej sesji użyto dokładnej
binarki z końcowego builda simplifiera, a CLI zbudowano ponownie po integracji.

`score-input.json`, `inputs/` i `reference-inputs/` zachowują wejścia i starsze
odpowiedzi. Nie tokenizuj ponownie promptów probe. `remote/out/` zawiera siedem
nowych wyników probe, każdy z `.json`, `.log` i `.exit`.
`remote/http-runner/out/j4-heuristic-{01,02}/` zawiera nowe odpowiedzi HTTP,
checks, polecenie serwera i exit. Runner jest identyczny z użytym w J2.

## Replay GPU

Wymagane są przypięte wagi i kalibracja, środowisko opisane w README oraz
odizolowana karta. Surowe polecenia sesji są zapisane w `remote-session-01.log`
i `http-session-01.log`. Nie kopiuj z nich operacji zatrzymujących usługę na
inną instalację. Nowy pomiar wymaga nowych katalogów i własnego przywrócenia.

Probe przyjmuje model, wejście, nowy plik wyniku i profil:

```bash
"$PROBE" "$THINK_MODEL" "$EVIDENCE/score-input.json" "$NEW/timed-01.json" pointwise
BASAL_THINK_GEMM=heuristic "$PROBE" "$THINK_MODEL" \
  "$EVIDENCE/score-input.json" "$NEW/heur-01.json" pointwise
"$PROBE" "$THINK_MODEL" "$EVIDENCE/score-input.json" "$NEW/control-01.json" inherit-f32
```

Kolejność: trzy timed, dwa heuristic, FP32 bez zmiennej, FP32 ze zmienną.
Są to świeże procesy; poza wskazaną zmienną środowisko nie ma override'ów
`BASAL_*`. Wyjść nie nadpisuj. Kontroluj exit 0, `gemm_check.valid`, wszystkie
dziewięć atrybutów, surowe hex i `.plan.overrides`. W heuristic wszystkie `ms`
są null; w timed liczbami. Wersja biblioteki jest w `preflight-01.log`.

Pełne HTTP uruchamiaj dopiero po przejściu D0/D2, dokładnym P2 score w progu
1e-4 i 25/25 top-1 kroków 0–24. Runner wymaga katalogu `models/` obok siebie,
z przypiętym modelem. Polecenie dla każdego z dwóch nowych tagów:

```bash
BASAL_THINK_GEMM=heuristic python3 "$RUNNER/native_probe.py" think \
  --binary "$BASAL" --tokens 256 --max-len 2048 \
  --think-numerics upstream-bf16-pointwise --repeat-all --tag "$NEW_TAG"
```

## Analiza bez modelu

`analysis-01.log` zapisuje pełne polecenie jq i exit; `analysis-01.json` zawiera
G, D0, D1, D2, P2 i kroki generacji. `timed-detail-02.*` rozdziela zmiany
top-1, zapisanych logitów i wyborów algorytmów. Pierwszy eksploracyjny query
timed-detail miał błąd składni; nie był użyty do wnioskowania.
`input-output-validation-01.log` sprawdza tokeny, temperaturę i długości.

`compare-j4-heuristic-*-upstream-think-*.log` zawierają dokładne porównania
19 plików z obiema referencjami i asercje ich kompletności. Obok są wyniki
JSON; `http-reference-summary.json` agreguje progi, notatki i argmax.
`http-restart-02.*` porównuje 42 body między procesami po usunięciu tylko
`usage.latency_ms` i `answers.*.think.ms`; pierwszy query nie obsługiwał pliku
checks będącego tablicą i zakończył się błędem. `http-off-preserved.*`
porównuje dziewięć odpowiedzi off do J2. `j2-vs-j4-thresholds.*` zachowuje
regresje i poprawę poszczególnych przypadków. Analizę wykonano jq 1.8.2,
z dowodem dokładnego porównywania dużych liczb; opaque algos porównuje się
przez ciągi hex niezależnie od reprezentacji liczb jq.

D1 jest wyłącznie obserwacją: timed records nie dokumentują wszystkich
wykonań dla dokładnego M. Równe lub różne rekordy nie rozstrzygają przyczyny.
D2 dotyczy zapisanych odczytów i dwóch procesów, nie wszystkich tensorów
czy dowolnego korpusu. Logproby upstream porównuj przez różnice względem
tego samego tokenu, nigdy z absolutnymi effective logits native.
