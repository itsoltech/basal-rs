# Wytrenowane early-exit

Opcjonalny backend wykonuje wytrenowaną głowicę po pośredniej warstwie Llama.
Jeżeli każde odczytanie w chunku spełnia próg polityki, zwraca jej rzeczywiste
logity i pomija pozostałe warstwy. Obsługiwany artefakt to wyłącznie
`Remek/basal-1.0-4.5B@b95288041cb1975bb930c6f0410819273fd2f54f`.
Pico, Mini, basal-1.0-1.5B oraz checkpoint thinking 2.0 nie są jego zamiennikami.
Pozytywna kwalifikacja runtime, numeryki, pamięci i wydajności pozostaje **GAP**.

## Włączenie i konfiguracja

```sh
basal serve --model /path/to/published-basal-1.0-4.5B \
  --kernels candle --readout f32 --inference-mode accurate --early-exit off
```

Obsługiwana ścieżka to pojedynczy CUDA Llama, jawne `--kernels candle`,
`--readout f32`, bez kwantyzacji i bez media. Dtype f16/bf16/f32 jest nadal
jawnym ustawieniem operatora; nie zmieniamy go dla early-exit. `balanced`
wymaga fused i jest niezgodne z tą konfiguracją; `auto` zachowuje istniejące
rozstrzygnięcie planu. Fused/ragged, layer split, Metal, Intel, Gemma i Qwen
odmawiają aktywacji przy starcie, przed GPU/GEMM. Kompilacja tych wariantów nie
oznacza wsparcia runtime. Thinking i early-exit nie mogą być razem włączone.

Per-model YAML przyjmuje `early_exit: "off"` lub nazwę polityki z pliku oraz
opcjonalne `exit_heads: /path/to/exit_heads`. Globalne flagi CLI
`--early-exit` i `--exit-heads` zastępują odpowiednie wartości YAML.
`--kernels candle` jest istniejącą globalną flagą CLI, nie polem YAML.
Pominięcie obu nowych opcji pozostawia plain backend: nie czyta ani nie
pobiera głowic. Samo `early_exit: "off"` włącza backend z głowicami i domyślną
pełną głębokością; samo `exit_heads` robi to samo. Nie mieszamy tego ustawienia
z istniejącym `inference_mode`, który wybiera arytmetykę jakościową.
Jest to świadomie inny model konfiguracji CLI niż upstream fast-exit.

Dla konfiguracji HF opt-in wymaga dokładnego repo i revision powyżej.
Bez `exit_heads` acquisition dołącza tylko `exit_heads/heads.pt` oraz
`exit_heads/thresholds.json`; z jawnym katalogiem używa tych lokalnych plików.
Nie ma fallbacku na inny checkpoint. Polecenia prepare-only nie aktywują
głowic: jawnie odmawiają opcji early-exit; planowanie offline serwera dostaje
snapshot z faktycznie załadowanego backendu.

## Żądanie i metadane

`early_exit` na najwyższym poziomie requestu jest niezmiennym override.
Brak i null wybierają domyślną politykę backendu. Jawne `"off"` wymaga
backendu z głowicami; plain backend odmawia również tej wartości. Jawna
polityka musi być dostępna. Native API normalizuje liczby JSON przez skończone
`f64` do najkrótszego zapisu: `0.990` i `9.9e-1` wybierają `0.99`. Wartości
tekstowe pozostają dosłowne, więc `"0.990"` nie wybiera `"0.99"`. Pozostałe
wartości nietekstowe podlegają string coercion; TypeSafe przyjmuje wyłącznie
string lub null. Jawne ustawienie
przy media/assets jest niedozwolone. Polityka nie zmienia globalnego defaultu.

Inventory pochodzi z przypiętego JSON i obejmuje `0.98`, `0.99`, `0.995`,
`0.999` oraz `off`. Są to etykiety docelowej zgodności z pełnym modelem
podczas fit, bez gwarancji accuracy lub confidence na danych użytkownika.
`GET /v1/models` pokazuje `early_exit: []` dla plain backendu; po aktywacji
pokazuje rzeczywiste available oraz `early_exit_default`. Narzędzia MCP
przekazują to samo opcjonalne pole. Native OpenAPI i cztery typed SDK opisują
te pola w lokalnych, nieopublikowanych wersjach ([SDK](SDK.md)); runtime
nowych pól w klientach pozostaje do weryfikacji.

## Obliczenia, grupowanie i cache

Po pełnym residual update warstwy o głębokości L, przed następną normą,
readout wykonuje FP32 RMSNorm z wytrenowaną wagą i epsilon modelu,
`x + b(a(x))`, następnie projekcję zamrożonymi bazowymi wierszami LM.
Softmax FP32 obejmuje wyłącznie dozwolone kody konkretnego readoutu.
Gate bierze minimum z maksimów WSZYSTKICH readoutów chunka, przed inverse
permutation, uśrednieniem, kalibracją i składaniem odpowiedzi. Próg >1 wyłącza
daną warstwę. Warstwy są sprawdzane rosnąco. Zachowujemy dotychczasowe
TypeSafe formuły po readoucie i wybór bazowej kalibracji, bez plików
CALIBRATION zależnych od polityki. Pełne `off` używa dotychczasowej arytmetyki.
Nieskończone lub NaN logity głowicy przerywają całe wywołanie silnika
błędem wewnętrznym. Online serve zwraca wtedy `Internal` wszystkim zadaniom
tego batcha, również współbatchowanym żądaniom off i zwykłym. Background
lane nie wykonuje aktywnej polityki: bounded planowanie odmawia z
`UNBOUNDED_FEATURE`. To zamierzona różnica względem upstream, gdzie nieudany
gate dla NaN pozwala kontynuować do głębszych warstw.

Aktywne żądanie ma własną grupę wszystkich pytań, orders i label branches.
Dokładny upstream `_pack` tworzy units; `_units` bisekuje zakres readoutów,
gdy ma kilka roots albo ponad 8192 tokeny pod root. Units sortujemy stabilnie
według packed length; chunking ma batch buckets 1/2/4/8/16/32, grid długości
128/192/256/320/384/448/512/640/768/896/1024/1280/1536/1792/2048/2304/2560/3072
oraz padded budget 12288. Dummy padding nie ma readoutów. Unit ponad 3072
packed tokens wykonuje pełny forward bez exit, z zachowanym mapowaniem.
Limit dotyczy packed length, nie najdłuższego promptu. Auto-orders,
grouped rounds i tasks/continuations niosą tę samą efektywną politykę.

Niskopoziomowe `Engine::run` z aktywnym defaultem traktuje każdy element
`Prepared` jako osobną grupę jednego pytania, ze wszystkimi jego orders razem.
Zachowuje kolejność wejść i wyników; skład i kolejność pozostałych elementów
zbioru nie wpływają na gate tego pytania. Tak działają niezależne elementy
`bench.jsonl` w export, bench i profile. Surowe readouty przypadku
`systemone.jsonl` w export (`run_prepared` bez media) tworzą natomiast jedną
grupę całego żądania, jak `run_decision_plans`. Oba diagnostyczne readouty
używają defaultu backendu; jawny `early_exit` w żądaniu przypadku steruje
tylko porównywaną odpowiedzią decide, więc surowe readouty mogą użyć innej
polityki niż ta odpowiedź. `run_decision_plans` zachowuje grupę całego
żądania i jego jawną, zamrożoną politykę we wszystkich rundach.

Inne żądanie nie zmienia wybranej głowicy: aktywne plany rozdzielamy przed
flatten. To odpowiednik izolacji upstream `--deterministic`; nie obiecujemy
zgodności z niedeterministycznym grupowaniem klientów upstream. Off zachowuje
zwykłe wspólne batching i packing. Aktywny forward nie używa prefix/state
cache i nie zapisuje niepełnego KV. Fork współdzieli niezmienne tensorowe
głowice i progi; zachowuje własne cache, gate i lifecycle pracy. Każdy aktywny
layer checkpoint i early return przechodzi istniejący drained release/health.

## Tożsamość, bounds i kwalifikacja

Loader używa istniejącego czytnika tensorów Candle pickle z kluczem `state`,
bez wykonywania Python pickle. Waliduje kompletne triplets
`<L>.norm.weight[H]`, `a.weight[r,H]`, `b.weight[H,r]`, dodatni rzeczywisty
rank, zakres warstw, obecność wszystkich warstw progów i finite wartości.
Nie zakłada rank 256 jako dowodu zawartości binarnej. Tożsamość wiążą SHA-256:

| Plik | SHA-256 |
|---|---|
| model.safetensors | `3cfe36727cbacc54b3f85f382319c877a3fa07689591cbf661ec5e90dbf8f0f3` |
| heads.pt | `4e6752762132ffe97f2d21cfe63ffd5f59b281451d583c989fe5c65cff49eb65` |
| thresholds.json | `5cf1b71833ce85bb5ea39ac5b02fb7c84f7b20c52a78810b4ede9f36430fc6bc` |

Sprawdzamy też opublikowane config.json i basal.json. Sama zgodność hidden
dimension nie wystarcza. Pliki modelu i głowic muszą pozostawać niezmienne
podczas ładowania. Progi są odczytywane raz; SHA-256 i parser używają tego
samego bufora bajtów. Profile batchy i benchmark zachowują hashe, default,
inventory oraz kontrakt izolacji bez lokalnych ścieżek. `basal-planner.5`
oddziela ten kontrakt od .4 thinking.

Benchmark preflight liczy rzeczywiste rezydentne bajty FP32 głowic po
walidacji tensorów CPU. Dla aktywnego Candle nie ma jeszcze source-derived
host/device bound obejmującego grid, dense masks, staging, adaptery FP32
i retained logits. Planowanie bounded odmawia z `UNBOUNDED_FEATURE`;
estymatory host/device również zgłaszają brak boundu. Nie przypisujemy savings
pomijaniu warstw i nie używamy starego ordinary Tree boundu jako dowodu.
Online wykonuje rzeczywiste głowice i skip-tail; runtime może odmówić alokacji.

Statyczne fmt/Clippy/Rustdoc nie potwierdzają obliczeń. Dodatnia kwalifikacja
wymaga uprawnionych artefaktów 4.5B, odpowiedniej pamięci i porównania
z przypiętym upstream FP32: każda polityka, wybrana głębokość, rozkłady,
brak wykonania tail, kolejne rundy i izolacja ruchu. Pico/Mini kwalifikują
jedynie brak/null, odmowy niedostępnych polityk i zwykłe regresje.
