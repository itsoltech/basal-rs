# Natywne thinking

`--think auto` włącza generowanie notatek i drugi readout na **osobnym pełnym
checkpoincie `think/`**. Domyślne `--think off` nie pobiera ani nie ładuje
drugiego modelu. Wspierana para to
`Remek/basal-2.0-4.5B@e5ba9149760d99b73b1963d8e0ccf948b58afacb`.
Loader sprawdza SHA-256 obu opublikowanych plików wag, konfigurację, zgodność
tokenizerów i efektywny zbiór EOS `[4, 2]`; hash pliku generation config
pozostaje w tożsamości źródeł. Mini i Pico nie mają tych wytrenowanych
artefaktów; aktywacja na nich odmawia startu. Nie ma adaptera LoRA, zero-shot
ani zastępowania P2 modelem bazowym. Własny pełny checkpoint Llama z lokalnego
katalogu można wskazać jawnie (`think_source`, sekcja
[Własny checkpoint thinking](#własny-checkpoint-thinking)); wtedy para
opublikowana nie jest pobierana ani używana.

```sh
basal serve --model 2.0-4.5B --think auto --think-tokens 256 --think-max-len 8192
```

W YAML te same opcje modelu mają nazwy `think`, `think_tokens`, `think_max_len`,
`think_calibration`, `think_numerics`, `think_source`, `think_integrity`,
`think_gemm_table`. Jawne flagi CLI zastępują wartości danego modelu w YAML;
`--think-source` i `--think-integrity` zastępują je zawsze jako para.
Limity muszą być dodatnie, mieścić się w u32, a `think_tokens + 16` musi być
mniejsze od limitu kontekstu; rzeczywisty limit checkpointu dodatkowo ogranicza
`think_max_len`. Walidacja i wybór osobnej kalibracji następują przed GEMM
i ładowaniem wag na GPU. Włączenie może pobrać dodatkowy pełny checkpoint;
każdy z dwóch opublikowanych plików wag ma 9 514 629 496 bajtów.

`--think-calibration PATH` wybiera niezależny plik P2, `off` wyłącza tę
kalibrację, a pominięcie wybiera root `CALIBRATION.think.json`. Jawny brak
pliku jest błędem; automatyczny brak ostrzega i używa T=1 z etykietą `missing`.
Plik jest walidowany dla promptu modelu, canonical/order1. Nigdy nie przejmuje
orders/layout, strict identity ani `--no-calibration` readoutu bazowego.
Temperatury per primitive/bucket i certyfikaty P2 pochodzą wyłącznie z tego
pliku. Opublikowany plik zawiera temperatury, lecz jego certyfikaty per-type
1%/5% są null: to nie potwierdzenie certyfikowanej jakości P2.

## Własny checkpoint thinking

`--think-source DIR --think-integrity FILE` (YAML `think_source`,
`think_integrity`) ładuje jako backend thinking własny pełny checkpoint Llama
z lokalnego katalogu, opisany deskryptorem integralności operatora. Katalog może
być lokalnym snapshotem HF, ale jest czytany wyłącznie jako ścieżka: basal nic
dla niego nie pobiera, nie stosuje ograniczenia do pary 4.5B, wzorców `think/`
ani automatycznego `CALIBRATION.think.json`. Model bazowy pobiera się jak
zwykły model. Nie ma fallbacku do `think/`, modelu bazowego, adaptera ani
zero-shot. Bazowy model może mieć dowolną architekturę obsługiwaną przez GPU
tego buildu; tokenizer i wymiary obu modeli nie muszą być równe, ale wejście
media bazowego modelu jest odrzucane.

```sh
basal serve --model ./base --think auto \
  --think-source ./my-think --think-integrity ./my-think.integrity.json \
  --think-gemm-table auto
```

### Deskryptor `basal-think-source.1`

```json
{
  "schema": "basal-think-source.1",
  "files": { "<nazwa>": {"sha256": "<64 małe znaki hex>", "bytes": <u64>} },
  "calibration": "off"
}
```

`calibration` to `"off"` albo `{"file": "<nazwa>", "source_model_id": "<64 hex>"}`.
Plik ma najwyżej 1 MiB, jest ścisłym UTF-8 JSON-em, a nieznane pola, powtórzone
pola i powtórzone nazwy plików są odrzucane. Nazwa to 1–255 bajtów
`[A-Za-z0-9_.-]`, bez `/` i bez kropki na początku; dowiązania symboliczne
snapshotów HF są dopuszczalne, jeśli wskazują zwykły plik.

`files` musi zawierać **dokładnie** czytane pliki (7–72 wpisy), wszystkie
obowiązkowe:

| Rola | Pliki | Limit bajtów |
|---|---|---|
| struktura | `config.json`, `basal.json`, `tokenizer_config.json`, `chat_template.jinja` | 1 MiB każdy |
| tokenizer | `tokenizer.json` | 64 MiB |
| generacja | `generation_config.json` | 1 MiB |
| korzeń | dokładnie jeden z `model.safetensors` (zarazem jedyny plik wag) albo `model.safetensors.index.json` | 1 TiB / 16 MiB |
| shardy (tryb indeksu) | każda nazwa z `weight_map`, 1–64 pliki `.safetensors` | każdy plik i suma ≤ 1 TiB |
| kalibracja | `calibration.file`, jeśli podany | 16 MiB |

O zbiorze wag decyduje deskryptor, nie zawartość katalogu: brakujący wpis
i wpis pliku, którego basal nie czyta, są błędem. Inne pliki katalogu
(`evidence_head.pt`, `CALIBRATION*.json`, `think/`, dodatkowe shardy) nie są
otwierane. Plik kalibracji nie może mieć innej roli.

### Walidacja przed GPU i GEMM

Statyczne odmowy w `validate()`, przed pobieraniem modelu: `think_source`
i `think_integrity` wymagają `think: auto` i występują razem;
`think_calibration` jest z nimi sprzeczne; `think_gemm_table` wymaga
`think_source`; `think_numerics` musi być `inherit`; tabela musi mieć składnię
`gemm_table`; build musi mieć backend Llama CUDA albo Metal (Intel i build bez
GPU odmawiają). Przed odczytem deskryptora odmawiane są: backend CPU,
kwantyzacja inna niż `none`, wejście media, reprodukowalny plan bazowy bez
`think_gemm_table` `pinned`/`sha256:` oraz kilka `replica_devices` przy
`think_gemm_table: none`.

Potem kolejno: deskryptor i zbiór plików; każdy plik spoza wag jest otwierany
raz, porównywany z zadeklarowanym rozmiarem i limitem, czytany do końca,
haszowany i zamykany. `config.json` musi opisywać Llamę z `head_dim` 128
(jądra fused attention), co najwyżej 256 głowami, `hidden_size` ≤ 32768,
256 warstwami, `intermediate_size` ≤ 262 144, słownikiem i kontekstem
≤ 1 048 576. Manifest jest czytany z canonical/order1 i wyłącznie z kalibracją
deskryptora; konfiguracja musi być równa wcześniej zahaszowanej. Format promptu
thinking musi mieścić co najmniej tyle opcji co bazowy. Tokenizer musi mieć
słownik równy konfiguracji, dekoder Llama o znanym limicie bajtów i BOS będący
jednym tokenem (zgodny z `bos_token_id` w `config.json`, jeśli podano).
`generation_config.json` jest czytany z zahaszowanych bajtów: `eos_token_id`
to liczba albo lista 1–64 unikalnych ID poniżej rozmiaru słownika, a
`bos_token_id`, jeśli podany, równa się BOS tokenizera; nie ma stałej `[4, 2]`.
Następnie nagłówki safetensors, routing tensorów, dokładny zbiór i kształty
tensorów Llama, SHA-256 każdego pliku wag, tożsamość, wszystkie reguły core
thinking bez backendu (canonical Llama order1, gęsty słownik, EOS, dekoder)
i fence. Każdy błąd następuje przed wyborem tabeli GEMM, zapytaniem
o urządzenie i ładowaniem.

### Wagi: jeden uchwyt na plik

Każdy plik wag jest otwierany dokładnie raz i ten sam uchwyt służy do odczytu
nagłówka, SHA-256 i każdego tensora; nie ma mapowania ani drugiego otwarcia.
Nagłówek ma najwyżej 100 000 000 B (suma nagłówków 256 MiB), każdy tensor jest
BF16, offsety pokrywają sekcję danych ciągle od zera, rozmiar odpowiada
kształtowi, powtórzony klucz w pliku i ten sam tensor w dwóch shardach są
błędem (żaden wpis nie wygrywa „ostatni”). W trybie indeksu `weight_map` musi
być dokładnie równy faktycznemu routingowi. Złożenie wag (`qkv`, `gate/up`,
zerowe biasy, lm_head BF16) jest takie jak w domyślnym loaderze; nie ma
evidence head.

### Plan, profil i tabela GEMM

Checkpoint thinking używa zawsze profilu `llama-generic.v1`, nigdy profilu
katalogu wybranego po nazwie z `basal.json`. Precyzja, kernele, tryb inferencji
i `readout_layer` (rozwiązany dla konfiguracji thinking) pochodzą z instancji
bazowej, readout jest FP32. Tabela GEMM jest własna: `think_gemm_table`
(`none` domyślnie, `auto`, `pinned`, `sha256:<hex>` albo ścieżka) jest
rozwiązywana dla kształtów checkpointu thinking; tabela bazowa nie jest
kopiowana. `auto` może uruchomić wyszukiwanie dopiero po całej walidacji
i admission. Przy layer split liczba warstw thinking musi być pokryta przez
ten sam split; niezależny split thinking nie istnieje.

### Tożsamość i kalibracja

`model_id` to SHA-256 zwartego JSON
`{"files":{"<nazwa>":{"bytes":N,"sha256":"<hex>"},…},"schema":"basal-think-model.1"}`
po wszystkich czytanych plikach oprócz kalibracji, z nazwami w kolejności
bajtowej. `sources` zawiera `kind: operator-local`, `provenance:
operator-supplied`, `schema`, `descriptor_sha256` (hash bajtów deskryptora:
zmiana samego formatowania zmienia odcisk), `model_id`, `profile`, hash
każdego pliku, `checkpoint` (`layout` single/index, liczba shardów, tensorów,
bajtów), `calibration` (`{"off": true}` albo plik, jego hash i
`source_model_id`), `numerics: inherit` oraz plan wykonania. Nie zawiera
lokalnych ścieżek, nazw katalogów ani nazwy z `basal.json`; trafia do profilu
batchy, benchmarku i forków jak dla pary opublikowanej.

Kalibracja `off` daje prawdopodobieństwa bez kalibracji (T=1). Plik kalibracji
jest czytany przez core z katalogu; jego `content_sha256` musi równać się
hashowi z deskryptora, a `source_model_id` — `model_id` checkpointu. To
deklaracje rodowodu operatora związane hashami: nie dowodzą, że kalibrację
dopasowano do tego modelu, ani jakości. Hashe potwierdzają tylko, że basal
przeczytał zadeklarowane bajty, nie autentyczność wydawcy, pochodzenie
treningu ani jakość.

### Kontrakt stabilnego katalogu

Deskryptor i każdy wskazany plik muszą pozostać niezmienione, nieobcięte,
niezastąpione i nieprzemianowane przez cały czas działania procesu.
`tokenizer.json` jest czytany ponownie po ścieżce przy każdym forku
(także dla pary opublikowanej), a `config.json`, `basal.json`,
`tokenizer_config.json`, szablon i tokenizer — przez core przy ładowaniu.
Wiązanie bajtami obejmuje wagi (ten sam uchwyt haszowany i czytany), deskryptor,
indeks i `generation_config.json` (parsowane z zahaszowanego bufora) oraz
kalibrację (równość `content_sha256`). Pozostałe ponowne odczyty po ścieżce
osłania tylko fence: porównanie `fstat` uchwytów wag i `stat` każdej ścieżki
(urządzenie, inode, rozmiar, mtime, ctime) po weryfikacji i po każdym
utworzeniu thinking. Fence nie wykrywa zapisu w miejscu o tym samym rozmiarze
z zachowanymi mtime/ctime, podmiany i przywrócenia między sprawdzeniami ani
zmian po ostatnim fence. To nie jest niezmienny ani odporny na atak snapshot;
operator zapewnia kontrakt np. snapshotem tylko do odczytu albo montowaniem
read-only. Uchwyty wag są zamykane po załadowaniu (w serve po ostatniej replice
danego modelu).

### Szacunki zasobów (admission)

Przed tabelami GEMM i ładowaniem basal sprawdza nazwane szacunki źródłowe
z ograniczonej konfiguracji i nagłówków: na hoście metadane inwentarza plus
największe z: składania jednego tensora (3S dla wielu części, 2S dla jednej,
lm_head 2S), budowy tabeli RoPE (`2 * 4 * max_positions * 128` B plus
odwrotności częstotliwości) i największego nagłówka; w serve dodatkowo RoPE
zatrzymywany przez każdą instancję. Na każdym urządzeniu suma etapów:
wagi po 2 B (z zerowymi biasami), embedding na pierwszym etapie, norma na
każdym, lm_head na ostatnim i 32 MiB workspace cuBLASLt na etap w CUDA (bez
`BASAL_GEMM=cublas`); przy pamięci zunifikowanej ta suma dolicza się też do
hosta. Admission jest pierwszym zapytaniem o urządzenie na tej ścieżce;
porównanie typów GPU splitu lub replik następuje po nim. Szacunki nie obejmują
wzrostu `Vec` ponad podane współczynniki, narzutu metadanych i allocatora,
stagingu sterownika, cache puli candle, modelu bazowego (ładowanego po
thinking), K/V ani pracy generacji. Nie są to granice allocatora, RSS ani
ochrona przed OOM. Benchmark dolicza dla własnego checkpointu jego szacunek
etapów zamiast drugiej kopii modelu bazowego.

### Ograniczenia

Ta ścieżka nie została uruchomiona: brak pomiaru generacji, P2, odmów na
uszkodzonych plikach, sprzątania przy przerwaniu, pamięci i wydajności, a także
bitowej równości pary opublikowanej po wspólnej refaktoryzacji loadera. Nie
ma deklaracji FP32, jakości ani kalibracji żadnego własnego modelu. Otwarte
pozostają: LoRA, gramatyki Gemma/Qwen, pobieranie z sieci, słowniki
z paddingiem, `head_dim` inny niż 128, kwantyzacja i
`upstream-bf16-pointwise`, CPU i Intel, niezależny split thinking, formaty
GGUF/ModelOpt, szacowanie przez `doctor`, Windows oraz wiązanie bajtami
plików czytanych przez core po ścieżce. Pamięć tokenizera nie jest szacowana.

## Opcjonalna numeryka BF16

`--think-numerics inherit` (YAML `think_numerics: inherit`) jest domyślne:
zachowuje numerykę po J1, precyzję modelu bazowego i readout FP32. Diagnostyczne
thinking FP32 pozostaje dostępne przez `--dtype f32`. Pominięcie opcji i jawne
`inherit` wybierają ten sam plan; jawna opcja wymaga `think: auto`.

`--think-numerics upstream-bf16-pointwise` wybiera wyłącznie dla osobnego
backendu think zestaw `basal-llama-think-bf16-pointwise.1`, BF16 obliczeń
i istniejący readout FP32 zaokrąglany na hoście do BF16, także w sesji generacji.
Nie zmienia modelu bazowego ani domyślnego wyboru. Profil deklaruje:

1. Tabelę RoPE z odwrotnością częstotliwości i kątem w FP32, hostowym sin/cos
   zaokrąglanym przez FP32 do BF16; te wartości są przechowywane jako FP32.
2. Bias czterech projekcji w epilogu cuBLASLt przed końcowym zapisem BF16;
   konsumenci fused nie dodają go ponownie. Heurystyczny split-K może wcześniej
   zaokrąglać sumy częściowe do BF16. Algorytm/redukcja GEMM pozostają odroczone.
3. Q/K zaokrąglone do BF16 przy zapisie RoPE, bez pośredniego zaokrąglania
   iloczynów. V pochodzi z projekcji BF16; backing KV pozostaje FP32 z hi/lo
   zależnie od faktycznego attention.
4. SiLU w FP32 aż do mnożenia przez up, z jednym końcowym zapisem BF16.
5. Wewnętrzny residual `s0 = f32(o + R)` aż przez MLP. Koniec warstwy liczy
   `s1 = f32(s0 + down)`, normę z niezaokrąglonego `s1` i dopiero zapisuje
   residual oraz wynik normy jako BF16. Pierwsza norma pozostaje bez zmian.

Przed ładowaniem wag odmawiane są: opcja przy `think: off` (także `inherit`),
inny model niż wspierana para Llama, backend bez CUDA (Metal/Intel), split,
kwantyzacja, kernele Candle, balanced, pinned, `BASAL_NORM=register`
i `BASAL_GEMM=cublas`. Nie ma cichego fallbacku po błędzie epilogu bias.
Profil używa istniejącego Lt każdego etapu osobnego backendu think wyłącznie
do biasu: bez drugiego handle/workspace, tabel, invariant search ani pinned.
Jego plan ma `table: null` i `runtime_timing`; diagnostyka GEMM etykietuje
wyniki jako czasy epilogu bias, nie wpisy tabel. Pierwszy wybór może się różnić
między procesami i klasami M. Deskryptory hosta są zachowywane dla dokładnych
M i kształtów projekcji; ich liczba zależy od liczby użytych długości.
Profil nie ma tabel GEMM, a background nie stroi GEMM w czasie działania,
dlatego włączenie profilu wyłącza cały background executor tego silnika,
także dla requestów z `think: off`. Ścieżka online nie wymaga background
qualification; praca w tle wymaga `inherit` i warunków opisanych niżej.

Diagnostyczne `BASAL_THINK_GEMM=heuristic` zastępuje w tym profilu pomiar czasu
pierwszym wynikiem heurystyki cuBLASLt dla dokładnego M, bez pomiaru czasu i bez
użycia algorytmu wybranego dla innego M. Powtarzalność tego wyboru między
procesami wymaga pomiaru.
Inne wartości są odrzucane przy planie profilu; pozostałe plany nie zmieniają
obliczeń, ale zmienna trafia do tożsamości każdego planu procesu.
`gemm_tuning()` podaje surowe bajty i konfigurację wybranych algorytmów (ID, tile,
split-K, schemat redukcji); atrybut, którego cuBLASLt nie zwrócił, ma wartość
`null`. W trybie z pomiarem czasu wpis opisuje zwycięzcę pierwszego pomiaru
klasy M, a nie algorytm wykonany dla każdego późniejszego M. Nie jest to
potwierdzony wybór upstream ani tryb parytetu.

To **nie jest tryb parytetu**. Redukcje GEMM (w tym split-K), FA2, normy
i lm_head oraz implementacje pow/sin/cos/exp/div pozostają źródłami różnic.
Zgodność nowych kerneli z deklaracją wymaga operatorowego replay. Wyniki wobec
upstream BF16 i FP32, deterministyczność po restarcie, pamięć i latency trzeba
kwalifikować osobno dla danej konfiguracji i korpusu.
Historyczny FAIL pozostaje FAIL; kompilacja nie dowodzi jakości ani parytetu.

[Pomiar J1/J2](../reports/upstream-api-241-thinking-rebuild-2026-10-28/README.md)
potwierdził na skończonym korpusie zachowanie default/inherit oraz powtarzalność
profilu po restarcie. Profil osiągnął 14/19 odpowiedzi w progu 1e-4 względem
rzeczywistego upstream, ale nadal tylko 4/9 identycznych notatek i 7/9 argmax.
To nadal FAIL parytetu; nie jest kwalifikacją operatorów ani granic alokatora.

[Izolacja score](../reports/upstream-api-241-score-isolation-2026-10-28/README.md)
pokazała wpływ treści notatki na wybór poziomu i różnicę top-1 między KV
a pełnym prefiksem na identycznych tokenach. W nowej kolejności odczytów P2
profil BF16 dał też różne prawdopodobieństwa w dwóch procesach z tą samą
tożsamością planu. Wcześniejsza powtarzalność dotyczy zatem wyłącznie
zmierzonej serii HTTP; profil nie gwarantuje powtarzalności po restarcie.

[Diagnostyka GEMM J4](../reports/upstream-api-241-gemm-diagnostic-2026-10-29/README.md)
potwierdziła powtarzalność jawnego trybu `BASAL_THINK_GEMM=heuristic` w dwóch
procesach na badanym korpusie. Pełne HTTP dało 13/19 odpowiedzi w progu 1e-4,
4/9 zgodnych notatek i 7/9 argmax względem rzeczywistego upstream. To nadal
FAIL parytetu; tryb pozostaje diagnostyczny, bez zmiany ustawień domyślnych.

[Decyzja z 2026-10-29](../reports/upstream-api-241-gemm-diagnostic-2026-10-29/ACCEPTANCE.md)
akceptuje obecne thinking Rust/CUDA z tymi różnicami i zamyka próbę J1–J4.
Dokładne odtworzenie notatek i prawdopodobieństw upstream nie jest warunkiem
zamknięcia tej próby. Wyniki FAIL parytetu pozostają udokumentowane;
domyślny `inherit` i opcjonalne diagnostyki thinking pozostają bez zmian.

## Żądanie i odpowiedź

Oba dialekty HTTP oraz narzędzia MCP przyjmują requestowe `think`:

| Wartość | Zachowanie |
|---|---|
| pominięte | domyślna polityka serwera (`off` albo `auto`) |
| `off` | normalny wynik bazowy |
| `auto` | thinking przy braku certyfikatu 5% lub `max(p_cal) < threshold` |
| `force` | thinking każdego kwalifikującego się pytania |

`null`, inne wartości i niepoprawne typy są błędami. `think_text` musi być
booleanem; domyślnie false. `thinking` nie jest aliasem. Jawne `auto`/`force`
na wyłączonym serwerze odmawia przed forwardem. Tasks classify/verify/compress/
extract wymagają `think: off`; aktywny tryb, również domyślny, odmawia przed GPU.

Najpierw wykonuje się cały przebieg bazowy, razem z orders auto i kalibracją.
Thinking obejmuje single-branch choice/noul/score/act/tool. Nie obejmuje multi,
toolsets, pytań bez readoutu, grouped choices ani nieznanych typów natywnego
dialektu mapowanych na nieskalibrowane choice. Faktycznie skrócony stan
wyłącza thinking całego żądania, również przy force. Próg auto bierze maksimum
skalibrowanego rozkładu P1, a nie przekształcone confidence TypeSafe; równość
z progiem nie uruchamia thinking.

Head zawiera ten sam znormalizowany stan i facts, pytanie i canonical order1
opcji w PL/EN. Końcowy PREFILL zastępuje `<think>\n`. Greedy wybiera największy
logit z całego rzeczywistego słownika, przy remisie najniższy token ID.
Kończy na `</think>` (także przez granicę tokenów), EOS lub limicie tokenów.
Notatki są dekodowane z `skip_special_tokens=true`; ten tekst służy do szukania
stopu i budowy P2. Licznik nadal obejmuje surowe wygenerowane ID, także special
tokens. Dekodowanie stanu przy skracaniu kontekstu zachowuje special tokens.
Llama CUDA z kernelami fused wykonuje jeden prefill, a potem przesyła przez
decoder tylko nowe tokeny. Każda warstwa zachowuje K/V całego dotychczasowego
prefiksu w osobnej sesji pytania. Kolejny krok musi ściśle rozszerzać dokładną
sekwencję ID poprzedniego kroku; zgodność samej długości nie wystarcza.
Pozostałe kombinacje backendu i kerneli, w tym Candle, Metal i Intel,
zachowują fallback obliczający pełny prefiks na każdym kroku.

P2 ponownie koduje `head + notes.strip() + "\n</think>\n" + PREFILL`, z zakresem
białych znaków Python `strip()` (wspólny helper promptu), sprawdza
code IDs w tym konkretnym kontekście i czyta je na **tych samych wagach think**.
Sesja generacji jest zamknięta przed P2. P2 otrzymuje niezależny pełny prompt
i używa zwykłej ścieżki readoutu; nie odczytuje KV wygenerowanych notatek.
Softmax i kalibracja readoutu są FP32; końcowy builder zachowuje formułę
confidence właściwą dla dialektu. Act/tool odbudowują progi z kalibracji P2,
evidence pozostaje z P1. Przed generacją obowiązuje
`head_tokens + think_tokens + 16 <= min(think_max_len, model_context)`;
rzeczywisty retokenizowany P2 też musi mieścić się w kontekście. Przekroczenie
zachowuje P1 i dodaje `think: {"skipped":"context"}`.

Sukces dodaje `think: {tokens, ms, calibration, no_think: {probabilities,
confidence}}`, a przy `think_text: true` również `text`. Confidence w
`no_think` jest `max(p1)`. Każda odpowiedź włączonego serwera, również off
i niekwalifikująca się, ma `meta.think: {used, tokens}`.
`usage.think_questions` liczy udane P2, a `output_tokens` faktycznie wygenerowane
tokeny, także tokeny stop/EOS. Jeżeli dopiero retokenizacja P2 przekroczy limit,
wygenerowane tokeny nadal są liczone, mimo zachowania P1. `input_tokens`,
branches, longest_prompt_tokens i second_order_branches zachowują znaczenie
przebiegu bazowego. Latency obejmuje cały request. Rejestr decyzji zapisuje
finalny wynik po P2; jego whitelista wyklucza notatki.

## Planowanie i lifecycle

Fork tworzy osobny cache i stan obu backendów, zachowuje ustawienia
i kalibrację; jego sesja generacji jest pusta. KV generacji nie przechodzi
między pytaniami, żądaniami ani forkami i nie trafia do zwykłego cache prefiksów.
Scoped cleanup zamyka sesję po sukcesie, błędzie, stopie i przy unwinding,
zanim można rozpocząć P2. Reset/release również usuwa stan sesji. Każda część
KV jest zwalniana z kontekstem własnego etapu; niepotwierdzony drain lub bind
zachowuje dotychczasową regułę lost/quarantine zamiast zwalniać używaną pamięć.
Gate, work class, stop, health oraz drain obejmują oba modele.
Stop jest sprawdzany między tokenami i przed P2; nie przerywa kernela w toku.
Cały request pozostaje w tym samym admission i lane aż do końcowego wyniku.
Generacja jest sekwencyjna: także żądanie off w tym samym microbatchu czeka
na zakończenie myślenia pozostałych żądań przed wysłaniem odpowiedzi. To źródło
opóźnienia head-of-line, bez zmierzonej wartości latency. Błąd thinking jednego
żądania kończy jego dalsze thoughts i daje błąd wewnętrzny z pełnym opisem
przyczyn. Dialekt upstream odpowiada 500, a 503 tylko wtedy, gdy ten opis
zawiera rozpoznany natywny OOM; TypeSafe zawsze odpowiada 500. W trwałym
batchu rekord jest ponawiany, a po `batches.execution.max_attempts` próbach
dostaje wynik błędu `execution_failed`. Poprawne żądania współdzielące
microbatch mogą otrzymać odpowiedź. Typowane przerwanie i utrata urządzenia
nadal kończą wywołanie, z dotychczasowym drain i quarantine.

Planer offline dostaje snapshot capabilities thinking. Dodatkowe host bounds
obejmują P1, notatki, tokenizer/retokenizację, pełny słownik i staging.
Bound notatek pochodzi z maksymalnej długości elementu zwalidowanego dekodera
Metaspace/ByteFallback (z zapasem na zastępowanie błędnych bajtów UTF-8).
Device bounds obejmują największy capped-prefix forward/P2 i pełny słownik.
Liczą jeden wiersz o długości efektywnego `think_max_len` z jednym readoutem
pełnego słownika, niezależnie od kształtu planów bazowych w microbatchu;
z bazowym boundem łączy je maksimum na każdym urządzeniu, bo modele działają sekwencyjnie.
Dla inkrementalnego CUDA doliczają jednocześnie stare i nowe K/V wszystkich
warstw danego etapu do tego limitu kontekstu (`context`), również
bufory hi/lo oraz pamięć Q współdzieloną przez widoki K/V z prefillu.
Dodatkowy konserwatywny bound na etap wynosi
`2 * layers * context * (heads + 2 * kv_heads) * head_dim * 4` bajtów,
a przy attention na tensor cores dodatkowo
`4 * layers * context * kv_heads * head_dim * 4` bajtów.
To zapas ponad dotychczasowy forward scratch, z osobnym licznikiem alokacji
dla granulatów allocatora; nie jest zmierzonym szczytem. Host bound dolicza
uchwyty tensorów i metadane starych/nowych KV; kopie ID mieszczą się w
dotychczasowym składniku `context * 128`. Zwykłe/off bounds są bez zmian.
Work cost zachowuje konserwatywny koszt do `think_tokens + 1` pełnych forwardów na kwalifikujące
się pytanie Direct; inne pytania nie zwiększają tego kosztu.
Pamięć rezydentna **obu** modeli i ich workspace musi być uwzględniona
osobno w kwalifikacji i rezerwie serwera, tak jak pozostałe resident weights.
Profil BF16 dolicza w `stage_work` konserwatywne `4 * M * hidden` bajtów
na f32 `s0` i jeden slot alokacji. Nie zmienia konserwatywnego boundu J1
dla starego/nowego KV z backingiem FP32 i hi/lo według attention. Istniejący
workspace Lt to 32 MiB na etap, współdzielone przez lanes tego modelu pod
tym samym gate/drain. Transient `stage_work` go nie dolicza: to pamięć
rezydentna, wymagająca osobnej rezerwy. Pokrycie rzeczywistych szczytów
allocatora i całej rezerwy rezydentnej pozostaje GAP.

Na włączonym silniku `background_support` backendu think musi pokrywać pełne
`think_max_len`, także gdy konkretny request wybiera off. Brak tego wsparcia
wyłącza cały background executor danego silnika. Silnik uruchomiony z off
zachowuje dotychczasowe wymagania bazowego backendu.

Bounded executor wymaga źródłowych host/device bounds, cooperative controls
obu backendów, pokrycia GEMM wszystkich dopuszczonych klas M oraz limitów
slotu/puli i forwardów. Niewymodelowana konfiguracja thinking, np. referencyjne
kernele Candle lub dense-mask SDPA, daje `UNBOUNDED_FEATURE` z konkretną
przyczyną w planowaniu offline; nie wyłącza thinking po cichu. Przekroczenie
skończonego limitu admission daje właściwy błąd budżetu. Zwykłe/off żądania
nie dostają kosztu generacji. Online korzysta z tej samej rzeczywistej ścieżki
generation + P2, bez wymagania background qualification.

Background zachowuje odziedziczony limit polityki DRR:
`max_record_cost = floor(exec_pool_bytes / max(1, host_staging.per_token)) * cost_ns_per_token`.
Najpierw sprawdza bazowe `max(1, packed_tokens)` jak wcześniej
(`HOST_BUDGET_EXCEEDED`). Potem sprawdza całkowitą pracę
`max(1, packed_tokens) + eligible_direct * (think_tokens + 1) * think_max_len`,
pomnożoną przez `cost_ns_per_token`, wobec tego samego limitu. Odmowa tej
dodatkowej pracy ma kod `BUDGET_EXCEEDED` i podaje pracę, koszt oraz cap.
To limit polityki planowania powiązany z pulą/staging, nie pomiar bajtów
hosta zużywanych przez thinking. Przy domyślnych 256/8192 i staging 1280 B/token
sama dodatkowa praca jednego eligible Direct wymaga puli co najmniej
2 694 840 320 B (około 2,7 GB), plus udział promptu bazowego. Dlatego limit
jest praktycznie restrykcyjny. Osobny rzeczywisty check `host_bound <= pool`
nadal odmawia `HOST_BUDGET_EXCEEDED`, gdy źródłowy bound bajtów hosta nie mieści
się w puli. Żaden z tych wzorów nie jest zmierzonym kosztem ani szczytem pamięci.

`basal-planner.5` zachowuje kontrakt thinking wprowadzony w .4 i dodaje
[trained early-exit](EARLY_EXIT.md). Głowice exit pochodzą z innego modelu
1.0; loader odmawia łączenia obu opt-in i nie używa ich w think/ 2.0.
Profile batchy i ustawienia benchmarku
zawierają efektywne limity, osobną kalibrację (hash/off/missing), przypiętą
tożsamość i hashe plików obu checkpointów, tokenizerów i konfiguracji oraz
osobny plan obliczeń thinking. Nie zawierają lokalnych ścieżek tych źródeł.

## Zakres kwalifikacji

[Pomiar RTX 6000 Ada](../reports/upstream-api-241-trained-2026-10-28/README.md)
dotyczy wcześniejszej generacji z ponownym obliczaniem pełnych prefiksów
i P2 na przypiętej parze 4.5B: native FP32 zgadza się
z diagnostycznym sidecarem PyTorch FP32 w 19/19 przypadków przy progu
0,0001, z równym tekstem i liczbą tokenów dziewięciu notatek. Adapter
referencji i konfiguracja orders 1 / context 2048 są jawnie opisane w raporcie.

**Parytet z oryginalnym sidecarem vLLM BF16 ma status FAIL:** siedem z 19
przypadków przekracza próg, a argmax P2 zgadza się w siedmiu z dziewięciu
wygenerowanych odpowiedzi. Osobna kontrola native BF16 nie usuwa różnicy.
Zachowano też FAIL bitowej równości P1 solo/batch w diagnostycznej konfiguracji
FP32, odtworzony bez thinking. Nie potwierdza to regresji samego thinking.

Inkrementalne KV zbliża strukturę generacji do cached decode, ale nie
gwarantuje identycznych tokenów ani wyników z vLLM. Różne harmonogramy
prefillu i decode mogą zmieniać wyniki numeryczne. FP32 pozostaje wybieralne;
ścieżka `inherit` zachowuje dotychczasowe kernele BF16, RoPE i SiLU.
Opcjonalny profil powyżej ma własne kernele i wymaga pomiarów. Dotychczasowe FAIL pozostają
FAIL. Ścieżkę inkrementalną zmierzono w skończonych seriach J1/J2 i J4 opisanych
wyżej. W J2 domyślne `inherit` przy FP32 zgadza się z diagnostyką Torch FP32
w 19/19 odpowiedzi. Parytet z vLLM pozostaje FAIL; decyzja z 2026-10-29
akceptuje tę różnicę. Wyniki dotyczą binarek i korpusu z tych raportów.
Split, limit generacji, błędy, przerwania, utrata urządzenia, długie obciążenie
i Metal/macOS pozostają niezakwalifikowane.
Nie ma zmierzonej obietnicy przyspieszenia.
Ogólny parytet detokenizacji, stop/EOS i ich liczenia poza badanym korpusem,
jakość, bounds i wydajność pozostają **GAP**. Czas `think.ms` pozostaje bez
zaokrąglenia do dwóch miejsc stosowanego przez Python upstream. Pico/Mini
mogą sprawdzić tylko off i odmowy; karta 8 GB nie mieści nawet jednego
z tych niekwantyzowanych plików wag. Zmiana precyzji wymaga osobnej kwalifikacji.
