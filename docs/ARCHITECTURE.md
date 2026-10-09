# Architektura

## Przegląd

```text
HTTP (axum, tokio)
  -> wątek GPU: walidacja i plan żądania (prompty, tokeny), kolejka, partie
       -> Engine::run_plans: wiersze-drzewa wspólnych prefiksów
            -> Backend::letter_logits: forward Llama, logity liter
       -> uśrednienie porządków, kalibracja, odpowiedź System One
```

| Crate | Odpowiedzialność |
|---|---|
| `basal-core` | kontrakt żądań (`request.rs`), prompt (`prompt.rs`), tokenizer i litery (`tokenize.rs`), JSON jak `json.dumps` Pythona (`pyjson.rs`), pakowanie (`pack.rs`), decyzje i confidence (`decision.rs`), `facts.rs`, `evidence.rs`, Choice 11–255 (`large_choice.rs`), silnik i trait `Backend` (`engine.rs`), współdzielenie GPU przez dwa silniki (`gate.rs`) |
| `basal-gpu` | loader safetensors, forward Llama na candle 0.11, fuzje (`fused.rs`, kernele `kernels.cu` i `kernels.metal`), GEMM przez cuBLASLt (`cublaslt.rs`) lub port GEMM MLX (`gemm.rs`, Metal), głowica `evidence_head.pt` |
| `basal-cli` | polecenie `basal`, serwer (`serve.rs`), eksport i porównania z referencją, pomiary |

`basal client` składa requesty z pytań CLI lub szablonu, albo przyjmuje pełne requesty System One.
`client/input.rs` czyta rekordy z limitem rozmiaru, `client/request.rs` składa i waliduje requesty przez `basal-core`,
`client/http.rs` obsługuje HTTP (połączenia, timeouty, ograniczone ponowienia), a `client.rs` steruje wykonaniem
i wyjściem. Liczba oczekujących/wykonywanych/buforowanych rekordów HTTP jest ograniczona przez `--jobs`.
W trybie lokalnym jeden `Engine` przetwarza rekordy kolejno; ładowanie i inferencja odbywają się poza executorem
Tokio, bez serwera i jego harmonogramu. Wybór tabel GEMM używa konfiguracji `serve`. Kontrakt promptu i odpowiedzi
pozostaje w `basal-core`. Interfejs i przykłady: [CLIENT.md](CLIENT.md).

## Kontrakt promptu

Prompt, szablon czatu, heurystyka języka (polskie znaki), serializacja stanu,
kolejność opcji i litery odtwarzają upstream (`prompt.py`, `server.py`).
Manifest przy ładowaniu porównuje `config.json`, szablony z `basal.json` i
`chat_template.jinja`; inny kontrakt kończy się błędem.

Tokenizer: transformers 5 buduje pre-tokenizer jako
`Metaspace(prepend_scheme="first")`, a `tokenizer.json` zawiera `"always"`.
Runtime stosuje `"first"` jak upstream i wyłącza obcinanie oraz padding z
`tokenizer.json` (obcinanie do 1536 tokenów zmieniałoby prompty długich
stanów).

Każde pytanie to dwa prompty z opcjami w dwóch porządkach. Z logitów liter
liczymy softmax osobno dla każdego porządku, odwracamy permutację, uśredniamy
i stosujemy temperaturę typu pytania:

```text
p = (softmax(l_forward) + unpermute(softmax(l_reverse))) / 2
p_cal = softmax(log(max(p, 1e-12)) / T_type)
```

## Pakowanie

Prompty partii są pakowane w wiersze-drzewa: skompresowane trie wszystkich
promptów wiersza (jak `_pack` upstream 1.5), spłaszczone w głąb. Każdy
wspólny prefiks liczy się raz: szablon, stan wspólny dla wszystkich pytań o
niego (także z różnych żądań jednej partii), treść pytania wspólna dla obu
porządków. Identyczne prompty mają jeden odczyt. Pozycje RoPE to indeksy
tokenów w ich własnych promptach, a token widzi wcześniejsze tokeny swojego
bloku i bloków przodków, co jest równoważne osobnym forwardom pełnych
promptów. Głębokość drzewa jest ograniczona do 8 bloków; poniżej prompty
dostają osobne bloki.

Wiersz ma do 32768 tokenów, forward do 8192 (wiersze dłuższe liczą się
osobno). Pytania o ten sam stan trafiają obok siebie.

Stały prefiks promptu (szablon do stanu, 53 tokeny PL i 66 EN) jest liczony
raz przy starcie. Opcjonalny cache stanu (`--state-cache-mb`) przechowuje K/V
szablonu ze stanem dla kolejnych żądań o ten sam stan (LRU z budżetem
pamięci). Oba działają tylko przy dokładnej zgodności token IDs.

## Forward

Model Llama z biasami lub bez (basal-1.0 ma biasy, basal-1.5 nie). Domyślna
precyzja to f16: wagi bf16 konwertowane przy ładowaniu, strumień residualny,
normy i MLP w f16, akumulacja GEMM w f32. Precyzja `f32` trzyma wagi bf16 i
rozszerza je przed użyciem (dokładny forward FP32, punkt odniesienia).

- GEMM, normy i MLP działają na zwartej liście tokenów całej partii, bez
  paddingu.
- Projekcje q/k/v oraz gate/up to pojedyncze GEMM; bias, RoPE i podział głów,
  bias+SiLU·up oraz bias+residual to własne kernele.
- Attention w f32 po jednostkach drzewa: zapytania jednego bloku i klucze
  jego przodków oraz własnego bloku, kafelkowane według pozycji klucza w
  prompcie. CUDA: `attn_tree_tc` na tensor cores (mma.sync, operandy f16
  rozbite na część wysoką i niską, czyli dokładność bliska f32) dla f16/bf16,
  na H100 (compute capability 9.0, forward f16) `attn_tree_wgp` z tą samą
  arytmetyką na wgmma, z ładowaniem następnego kafelka K/V w trakcie liczenia
  bieżącego, i bitowo tym samym wynikiem
  ([pomiar](../reports/attention-h100-wgmma-2/README.md)); przy krótkich
  promptach w blokach po 64 zamiast 128 wierszy (`attn_tree_wgp64`, te same
  bity, [pomiar](../reports/attention-h100-short/README.md)), `attn_tree_f32` dla
  ścieżki f32. K i V przychodzą z `qkv_rope` od razu rozbite na płaszczyzny
  f16, a wynik attention od razu jako scalone głowy w dtype forwardu. Metal: `attn_tree_f32` z `kernels.metal`
  na macierzach `simdgroup_float8x8` w f32 (`BASAL_ATT=sdpa`: poprzednie SDPA
  z MLX po całym wierszu z maską, wynik zależny od pakowania).
- W ostatniej warstwie o_proj i MLP liczą się tylko dla pozycji odczytu, a
  logity tylko dla wierszy `lm_head` odpowiadających literom, w f32.
- CUDA GEMM przez cuBLASLt z tabelą algorytmów z pełnego przeszukania
  (`basal gemm-search`): algorytm, kafelek, liczba etapów, split-K z
  redukcją f32, swizzle; bez tabeli najszybszy z 16 algorytmów heurystyki
  mierzony przy pierwszym użyciu.

### Niezależność od partii

Z tabelą `--invariant` (dla każdego kształtu wag algorytmy bez split-K z
jednej grupy dającej bitowo te same wyniki, najszybszy w każdej klasie M), attention kafelkowanym według pozycji klucza i odczytem liter
własnym kernelem wynik pytania jest bitowo ten sam pojedynczo, w dowolnej
partii, w drzewie z innymi pytaniami i z cache prefiksu. Odpowiedź serwera nie
zależy więc od ruchu w tej samej chwili. Na Metal (GEMM MLX niezależny od
liczby wierszy, attention po jednostkach drzewa) wynik jest bitowo ten sam
pojedynczo, w partii i w drzewie
([pomiar](../reports/metal-m2-max-tree/README.md)); z cache stanu tego nie
mierzono.

## Rozszerzenia basal-1.5

- `multi`: osobne pytanie tak/nie dla każdej etykiety, temperatura `noul`;
  gałęzie idą do tego samego drzewa co reszta żądania.
- `act`: odczyt jak `noul` lub `choice`, potem akcja o najmniejszym
  oczekiwanym koszcie; `max_error` porównuje confidence z certyfikowanym
  progiem z `CALIBRATION.json` i w razie potrzeby wybiera odroczenie.
- `facts: "auto"`: port `facts.py` (fakty wyliczone z dat i kwot w stanie,
  dopisane do stanu), wynik zgodny co do bajtu.
- `evidence`: osobny forward promptu w oryginalnej kolejności opcji i głowica
  `evidence_head.pt` (f32) na stanach tokenów stanu; do 3 nienakładających się
  fragmentów do 80 tokenów z offsetami znakowymi.

## Choice 11–255

Opcje dzielone są dwa razy na `ceil(n/10)` grup: blokami i cyklicznie, więc
każda opcja jest w dwóch grupach, a graf grup jest spójny. Każda grupa to
pytanie basal w jednej kolejności opcji (jedno drzewo ze wspólnym stanem).
Wspólny rozkład to model Luce `softmax(theta)` dopasowany metodą największej
wiarygodności do rozkładów grup, uzupełniony rundą finałową z 10 najlepszymi
opcjami w obu kolejnościach. To przybliżenie (model nie widzi wszystkich
opcji naraz); ocena na zbiorach z etykietami i porównanie z TypeSafe w
[reports/choice-sets](../reports/choice-sets/README.md), wcześniejsze w
[reports/large-choice-1.5](../reports/large-choice-1.5/README.md). Inne
strategie (`basal eval-choice-set --strategies`) służą do porównań.

## Serwer

Handlery HTTP (tokio) przekazują żądania do wątku modelu z pola `model`
(`/v1/basal` bez tego pola: do modelu domyślnego). Wątek
tokenizuje i planuje każde żądanie przy przyjściu (błędy walidacji wracają
od razu), szacuje jego koszt liczbą tokenów po współdzieleniu prefiksów i
układa kolejkę:

- `hrrn` (domyślnie): najwyższy stosunek (czekanie + szacowany czas) /
  szacowany czas; krótkie żądania wyprzedzają długie, a długie awansują z
  czasem oczekiwania.
- `fifo`: kolejność przyjścia.

Partia to pierwsze żądanie z kolejki i każde następne, które mieści się w
`--max-batch-tokens`; pytania wszystkich żądań partii dzielą forwardy.
Żądania porzucone przez klienta są pomijane, a ponad `--max-inflight` żądań
w toku dostaje 529 z `Retry-After: 1` (status TypeSafe dla przeciążenia).

Żądania powyżej `--long-tokens` liczy drugi tor: drugi silnik na tym samym
GPU, ze wspólnymi wagami, tabelą GEMM i prefiksami szablonu. Oba tory
korzystają z GPU na zmianę (`basal_core::gate`): tor długich żądań po każdej
warstwie czeka na zakończenie swoich kerneli i oddaje GPU, gdy główny tor ma
partię, a on sam pracował co najmniej `--long-slice-ms`. Po jednej partii
głównego toru wraca do tej samej warstwy. Zatrzymany forward kontynuuje z tymi
samymi tensorami, więc wyniki nie zależą od toru ani przerw. Krótkie żądania
nie czekają w ten sposób na kilkusekundowy forward długiego dokumentu.

Kilka modeli w jednym procesie (`basal serve --config`): każdy ma własny
silnik, kolejkę i tor długich żądań, a wszystkie silniki korzystają z jednego
`gate`. Główne tory wszystkich modeli są pilne, tory długich żądań oddają GPU
między warstwami, więc krótkie partie dowolnego modelu wyprzedzają długie
żądania każdego modelu. Silniki różnych modeli nie liczą jednocześnie: GPU
jest i tak ograniczone mocą, a kolejność decyduje harmonogram zamiast
sterownika.

### Metryki Prometheus

Metryki są domyślnie wyłączone. `--metrics` lub `metrics: true` w konfiguracji
włącza rejestr, instrumentację HTTP i torów modelu oraz endpoint `GET /metrics`.
Bez włączenia rejestr i uchwyty metryk nie powstają, middleware nie jest
instalowane, a `/metrics` zwraca 404.

Po włączeniu `GET /metrics` eksportuje prywatny rejestr procesu w formacie tekstowym Prometheus
0.0.4 (`basal-cli/src/serve/metrics.rs`). Middleware obejmuje wyłącznie trasy
inferencji: mierzy czas od wejścia przed odczytem body do zbudowania odpowiedzi,
liczy statusy i aktywne handlery. Handler po parsowaniu przypisuje etykietę modelu
z listy obsługiwanych modeli; pozostałe żądania mają pustą etykietę.

Tory `main` i `long` mają uchwyty do histogramów kolejki i partii. Czas partii
jest obserwowany raz, a czas kolejki osobno dla każdego żądania uruchomionej
partii. Liczniki tokenów i pytań sumują `usage` poprawnie zbudowanych odpowiedzi
na wątku modelu, również gdy klient już nie czeka. Rejestr nie odczytuje GPU;
scrape nie przechodzi przez kolejkę inferencji. Szczegółowa semantyka i PromQL:
[Observability](OBSERVABILITY.md).

## Ograniczenia

- Score powyżej 10 poziomów i pytania z jedną opcją kończą się jawnym błędem.
- Tabela GEMM jest specyficzna dla karty i wersji cuBLASLt.
- Metal sprawdzony na M1 Pro (32 GB) z basal-1.0-4.5B i trzema modelami
  basal-1.5 ([pomiar](../reports/metal-m1-pro-1.5/README.md)) oraz na M2 Max
  (32 GB) z trzema modelami basal-1.5
  ([pomiar](../reports/metal-m2-max-1.5/README.md),
  [attention](../reports/metal-m2-max-tree/README.md)).
