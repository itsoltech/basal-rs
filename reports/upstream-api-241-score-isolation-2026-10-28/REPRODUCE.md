# Odtworzenie izolacji score

Archiwum jest lokalne i prywatne. Zawiera surowe ścieżki i logi środowiska;
nie jest przeznaczone do bezpośredniej publikacji. Nie zawiera wag, binarek,
katalogu kompilacji ani plików tymczasowych.

## Dane i źródła

1. Sprawdź SHA-256 `measurement-data.tar.gz` podane w README.
2. Rozpakuj archiwum do nowego katalogu.
3. W katalogu rozpakowania wykonaj `sha256sum -c SHA256SUMS`.

Logiczny katalog danych wewnątrz archiwum:
`reports/upstream-api-241-score-isolation-2026-10-28/evidence/private/`.
W dalszych poleceniach ustaw `EVIDENCE` na jego lokalną ścieżkę.
`source.tar` zawiera dokładne źródła aplikacji identyfikowane przez
`source-commit.txt`. `probe/` zawiera prywatny program Rust oraz Cargo.lock.
`inputs/` zachowuje upstream score i jego powtórzenie w grupie, logproby
generacji, native J2, diagnostykę Torch FP32 oraz oryginalną kalibrację.
`prepare-input-01.log` dokumentuje pochodzenie i ekstrakcję wejścia.

`score-input.json` zawiera oba prompty P2, prompt generacji, wymuszone tokeny,
kody, kalibrację i kontrole zgodności. Nie zastępuj tokenów ponowną tokenizacją.
Wyniki są w `remote/out/`; każdemu z sześciu procesów odpowiadają `.json`,
`.log` i `.exit`. Logi `restored-*` oraz `restored-fresh-01.log` dokumentują
przywrócenie oryginalnej usługi.

## Replay GPU

Wymagana jest ta sama przypięta rewizja modelu, metadane i kalibracja katalogu
`think/`, Rust 1.95.0, CUDA 12.9.1 i środowisko opisane w
`probe-check-build-01.log`. Ten log zapisuje wszystkie polecenia kompilacji,
Clippy i Rustdoc. Przed kompilacją po przeniesieniu źródeł popraw trzy lokalne
ścieżki zależności w `probe/Cargo.toml` na rozpakowane `crates/basal-core`,
`crates/basal-gpu` i `vendor/candle-core`. Użyj zachowanego lockfile.

Probe przyjmuje dokładnie cztery argumenty:

```bash
"$PROBE_BINARY" "$THINK_MODEL" "$EVIDENCE/score-input.json" \
  "$NEW_OUTPUT/inherit-f32-01.json" inherit-f32
"$PROBE_BINARY" "$THINK_MODEL" "$EVIDENCE/score-input.json" \
  "$NEW_OUTPUT/f32-rounded-01.json" f32-rounded
"$PROBE_BINARY" "$THINK_MODEL" "$EVIDENCE/score-input.json" \
  "$NEW_OUTPUT/pointwise-01.json" pointwise
```

Każdy proces sam wykonuje po dwa P2, 26 kroków teacher forcing i jeden odczyt
stateless po zamknięciu KV. Uruchom drugi proces każdego wariantu z nowym
plikiem `-02.json`. Probe odmawia nadpisania istniejącego wyjścia. Rzeczywista
kolejność serii to oba procesy FP32, oba FP32-rounded, oba pointwise.
Ustawienia środowiska i limity czasu są w `remote-session-01.log`.
Nie odtwarzaj poleceń zatrzymujących usługę z surowego logu na innej instalacji.

## Ponowna analiza bez modelu

`analysis-01.log` zawiera dokładne polecenie jq, wyjście i kod zakończenia.
`analysis.json` jest tym samym wyjściem JSON po usunięciu nagłówka i stopki
narzędzia `script`. `restart-analysis-01.log` porównuje zapisane P2, generację
i tożsamości planów między procesami; nie usuwa żadnych pól liczbowych.
Przy przenoszeniu danych zastąp stary prefiks lokalny wartością `EVIDENCE`.

Podstawowe odczyty:

```bash
jq '{profile,p2:[.p2[]|select(.repeat==0)|del(.input_ids)],
  fork:.generation.steps[25],stateless:.generation.stateless_fork}' \
  "$EVIDENCE/remote/out/pointwise-01.json"
jq -s 'group_by(.profile)|map({profile:.[0].profile,
  p2_equal:(.[0].p2==.[1].p2),
  generation_equal:(.[0].generation==.[1].generation),
  plan_equal:(.[0].plan==.[1].plan)})' "$EVIDENCE"/remote/out/*.json
```

`input-result-validation-01.log` sprawdza tokeny, kody, temperaturę, długości,
liczby odczytów i równość powtórzeń wewnątrz procesów. `process-exits-01.log`
sprawdza sześć kodów zakończenia. `plan-hash-*.log` hashuje kompaktowy JSON
tożsamości bez końcowego newline, tak jak `plan_summary` aplikacji.

Logproby upstream porównuj przez różnice względem tego samego tokenu.
Analiza generacji używa oczekiwanego tokenu jako kotwicy i tylko wspólnych ID
z zapisanych top-8. Nie jest to pełne porównanie słownika ani trace warstw.
Dokładne dopasowanie P2 oznacza równość trzech zapisanych prawdopodobieństw,
nie zgodność całego żądania HTTP lub kompletnej generacji.
