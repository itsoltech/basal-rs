# Odtworzenie lokalnej kampanii API 2.4.1

To kampania lokalna, bez opublikowanego URI HF. Dane są w
`.cache/measurement-artifacts/upstream-api-241-local-2026-10-28/`,
z logicznym korzeniem `reports/upstream-api-241-local-2026-10-28/`.
Ścieżki w raporcie odnoszą się do tego logicznego korzenia.
Surowe logi zawierają prywatne ścieżki; nie nadają się do bezpośredniej publikacji.
Suma i weryfikacja lokalnego archiwum są zapisane w [RESULTS.md](RESULTS.md).

Sprawdzenie zachowanego archiwum, z katalogu kampanii:

```bash
sha256sum -c measurement-data.tar.gz.sha256
mkdir replay-check
tar -xzf measurement-data.tar.gz -C replay-check
(cd replay-check && sha256sum -c SHA256SUMS)
```

`replay-check` musi być nowym katalogiem. Odtworzona lokalna kopia była
sprawdzona w ten sposób; nie deklarujemy pobrania z publicznego magazynu.

## Wejścia i wykonanie

1. Użyj dokładnych commitów aplikacji i upstream oraz rewizji modeli z
   [README.md](README.md). Checkout upstream musi pozostać czysty. Wagi i binarki
   nie są częścią archiwum wyników. Obie binarki zbudowano z `--locked --release
   -p basal-cli --features cuda`, w osobnych targetach i TMPDIR.
2. Przywróć katalog danych kampanii. W `inputs/` są 44 bench i 7 żądań API;
   `task-inputs/` zawiera osobne zadania, `layout-inputs/` nowe layouty,
   `sdk-ready/inputs/` wejścia odpowiednie dla ergonomicznych konstruktorów SDK.
   Manifest archiwum przypina wszystkie zachowane regularne pliki.
3. Uruchamiaj tylko jeden proces GPU naraz. Ustaw ścieżkę bibliotek CUDA
   właściwą dla środowiska i `RAYON_NUM_THREADS=2`. Mini AUTO wymaga
   `inputs/mini-f16-invariant.json` po obu stronach. Dla Pico BF16 użyto
   `--gemm-table none`. Dokładne komendy i stderr są w `evidence/private/`.
4. Wykonaj eksporty `basal export` i porównania `basal compare` z zachowanymi
   korpusami. Referencje utworzono istniejącymi runnerami w `tools/reference/`
   przeciwko przypiętemu upstream. Sposób FP32 Mini nie jest pomiarem wydajności.
5. Kontrole HTTP uruchamiano skończonym helperem `evidence/private/probe-api241.py`
   na lokalnym serwerze. Helper uzupełnia brak istniejących operacji MCP,
   hashów loggera i izolacji późnych odmów kontekstu; nie jest frameworkiem
   ani bramką CI. Pakiety `diagnostics/` mają własne polecenia, wejścia i wyniki.

Porównanie bitowe nie wymaga modelu:

```bash
BASAL_DATA=.cache/measurement-artifacts/upstream-api-241-local-2026-10-28/reports/upstream-api-241-local-2026-10-28
cmp "$BASAL_DATA/exports/baseline-pico-bf16-o1/bench.jsonl" \
    "$BASAL_DATA/exports/final-pico-bf16-o1/bench.jsonl"
cmp "$BASAL_DATA/exports/baseline-pico-bf16-o1/systemone.jsonl" \
    "$BASAL_DATA/exports/final-pico-bf16-o1/systemone.jsonl"
```

Pozostałe dziewięć par wymienia `comparisons/exact-canonical.json`.
Wszystkie 20 pełnych plików przeszło `cmp`; nie maskowano JSON.

## Wydajność i ponowny audyt

Istniejący runner: `tools/perf/omni-optimize.py`; niezależne przeliczenie
zapisanych faz: `tools/perf/summarize-omni.py`. Nie zmieniono tych narzędzi.
Każdy `method.json` zapisuje pełne flagi, hashe narzędzi i wejść.

Podstawowe kampanie: `performance/pico-abba-02` i `mini-abba-01`, wejścia
`perf-inputs-v2/`. Cztery krótkie dokumenty: sześć typów pytań, wspólny prefiks,
27 opcji, Unicode. Osobny długi dokument: 5080 tokenów Pico, 4130 Mini.
Dodatkowy korpus `perf-options255-inputs/` obejmuje 255 opcji;
to osobny pomiar niepokrytego wcześniej przypadku.

Wspólne ustawienia: orders 1, inference auto, state cache 0, prefix cache off,
long lane 4096 tokenów / 100 ms, max_batch_tokens 8192, HRRN, CPU threads 2.
Każda kampania confirm ma kolejność ABBAABBA, cztery starty na wariant,
pierwsze powtórzenie każdego wariantu wykluczone, trzy zachowane. Podstawowy
pomiar: warmup 10 s, długi dokument 2 próby warmup + 10 mierzonych,
krótkie 64 żądania na C1 i C4 w każdym powtórzeniu. Daje to po 48 obserwacji
na krótki dokument/client/wariant i 30 na długi. Dla osobnych 255 opcji:
warmup 5 s, 16 żądań na C1 i C4/powtórzenie, również 48 zachowanych na komórkę.

Przykład ponownego audytu do **nowego** pliku:

```bash
env PYTHONDONTWRITEBYTECODE=1 python3 tools/perf/summarize-omni.py \
  --measurement-artifacts "$BASAL_DATA/performance/pico-abba-02" \
  --inputs "$BASAL_DATA/perf-inputs-v2" \
  --out "$BASAL_DATA/performance/audit-pico-replay.json"
```

Dla Mini zamień katalog kampanii. Dla 255 opcji użyj odpowiadającego katalogu
`*-options255-abba-01` oraz `perf-options255-inputs`.
Audyt sprawdza kompletność faz, ich liczebność, HTTP, niezmienność answers,
liczbę tokenów i ponownie oblicza czasy. `AUDITED` nie oznacza spełnienia progu
FP32; ten ma osobne porównania w `comparisons/`.

## Zachowane nieudane próby

Nie zastąpiono ich udanymi logami. Dotyczy to pierwszego helpera z kolizją
nazwy modułu `http`, skrótów wejść niewspieranych przez konstruktory SDK,
prób formatu 1 na Pico, native format1 z 11 opcjami, za małego limitu large
inflight przy długim Mini oraz pierwotnych zdublowanych ID wejścia perf.
Pierwszy odczyt profili Mini nastąpił przed zakończeniem ich hashowania,
a wysłane wtedy body miało błędną nazwę modelu. Te wyniki nie są dodatnim
dowodem inferencji. Pierwsza próba CLI adaptive nie miała wymaganego `model`;
poprawne wejścia i wyniki mają sufiks `-02`. Przebiegi przyjęte do wniosków
wskazano w README/RESULTS.

Położenie oryginalnych modeli w prywatnych fixture jest zapisane w symlinkach;
archiwum nie zawiera ich wag. Fixture prompt1 Mini i limitu 512 są wyłącznie
diagnostyczne. Nie odtwarzaj z nich benchmarku jakości oryginalnego modelu.
