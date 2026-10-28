# Thinking: izolacja rozbieżności score

**Inna notatka zmienia końcowy wybór poziomu; pozostaje też osobna różnica
numeryczna.** Przy identycznej notatce upstream wszystkie trzy badane warianty
Rust/CUDA wybierają poziom 2. Przy notatce native wszystkie wybierają poziom 1.
Profil J2 BF16 na identycznym wejściu P2 raz przekracza próg 1e-4, a po restarcie
daje dokładnie te same prawdopodobieństwa co upstream. Nie jest to naprawa
parytetu. Nie zmieniono kodu aplikacji, konfiguracji domyślnej ani produkcji.

## Zakres i wejścia

Pomiar 2026-10-28, NVIDIA RTX 6000 Ada Generation 48 GB, SM 8.9,
sterownik 580.159.03, Rust 1.95.0, CUDA 12.9.1. Źródła aplikacji:
`af56e6e7b638fea3e70c804fcfd594f5c39074ce`; względem wcześniejszego `8f135a3`
różnią się tylko dokumentacją. Model thinking:
`Remek/basal-2.0-4.5B@e5ba9149760d99b73b1963d8e0ccf948b58afacb`, katalog `think/`.
Użyto tej samej instalacji wag co w J2; całych wag ponownie nie hashowano.
SHA-256 użytej `CALIBRATION.think.json`:
`8938f99af2d23223e197cfbd52e3941627934cf33dd132fd310d1a84db9b217d`.

Badano jedno pytanie `score`: pilność sprawy klienta, który nie może zalogować
się do bankowości internetowej, mimo sprawnej karty. Powtórzenie tego pytania
w grupie ma identyczne tokeny generacji i P2; nie jest niezależnym przypadkiem.
Kolejność kodów `[31928, 31936, 31935]` odpowiada poziomom 0/1/2.
Temperatura kalibracji score wynosi `4.756828460010884`.

Referencją są zachowane odpowiedzi i logproby rzeczywistego upstream
Basal 2.4.1 (`b20f0f9b159f71726459aa98ede1356ab65695fd`), korzystającego z
vLLM 0.26.0; Torch 2.11.0, Transformers 5.17.0. Ich wcześniejsze potwierdzenie
przez oryginalny HTTP opisuje [raport diagnozy](../upstream-api-241-thinking-diagnosis-2026-10-28/README.md).
W tej sesji nie uruchamiano upstream ponownie. Torch FP32 jest dodatkową
diagnostyką, nie zastępuje rzeczywistej referencji.

## Metoda

Prywatny probe Rust wywołuje niezmienione publiczne metody
`RuntimeBackend::load_planned`, `letter_logits`, `generation_logits`
i `end_generation`, następnie produkcyjne `softmax32` i `calibrate32`.
Nie jest częścią aplikacji ani nową zależnością runtime. Każdy z trzech
wariantów uruchomiono w dwóch osobnych procesach, kolejno na odizolowanej karcie.

| Wariant probe | Ciało modelu | Readout |
|---|---|---|
| `inherit-f32` | produkcyjne V1 FP32 | FP32 |
| `f32-rounded` | diagnostyczne V1 FP32 | zaokrąglony do BF16 |
| `pointwise` | produkcyjny profil J2 BF16 | zaokrąglony do BF16 |

Plany używają accurate/fused, adaptive i braku tabel GEMM/kwantyzacji.
Hash tożsamości planu `pointwise` zaczyna się od `cb843f6942296027`,
a `inherit-f32` od `22387106e1f2e5f3`: identycznie jak plany aplikacji w J2.
Nie przypięto algorytmów wybieranych przez runtime timing.

W każdym procesie najpierw wykonano po dwa odczyty P2 dla notatki upstream
(61 tokenów, prompt P2 160) i native (73 tokeny, prompt P2 172).
Następnie wymuszano kolejne zapisane tokeny upstream od promptu generacji
92 tokeny do kroku 25 włącznie, używając request-local KV. Indeksy są od zera:
krok 25 otrzymuje 117 tokenów prefiksu. Po zamknięciu KV wykonano jeden
stateless readout pełnego, identycznego prefiksu. Liczba 92 opisuje rzeczywisty
prompt generatora; pole `usage.input_tokens` odpowiedzi HTTP wynosi 93.

Łącznie: 24 odczyty P2, 156 kroków teacher forcing i 6 odczytów stateless.
Probe zachowuje top-8 i parę tokenów rozgałęzienia, nie wszystkie logity.
Pole `effective_logits` po BF16 readout zawiera wartości już zaokrąglone;
nie jest zrzutem niezaokrąglonych logitów ciała BF16.

## P2 na identycznej notatce upstream

Prawdopodobieństwa upstream dla poziomów 0/1/2:
`[0.1520159841, 0.2029671073, 0.6450169086]`.
Wszystkie wiersze poniżej wybierają poziom 2. W każdym procesie dwa odczyty
są dokładnie równe. Różnica oznacza maksimum po trzech skalibrowanych wartościach.

| Wariant / proces | Effective logits 0/1/2 | Maks. różnica do upstream | Próg 1e-4 |
|---|---|---:|---|
| inherit-f32, oba | 20.97519 / 22.314 / 27.833055 | 0.0007583052 | FAIL |
| f32-rounded, oba | 21 / 22.375 / 27.875 | 0 | PASS tego odczytu |
| pointwise-01 | 21 / 22.25 / 27.875 | 0.0042178780 | FAIL |
| pointwise-02 | 21 / 22.375 / 27.875 | 0 | PASS tego odczytu |

Upstream nie zapisał absolutnych logitów P2. Jego logproby dają różnice
względem kodu 31928: `[0, 1.375, 6.87499995]`. Nie porównujemy absolutnych
logitów native z logprobami upstream. Dokładna zgodność prawdopodobieństw
`f32-rounded` potwierdza dla tego wejścia poprawność kolejności kodów,
softmaxu i kalibracji; nie dowodzi zgodności całego backendu.

## P2 na notatce native

Przy niezmienionym kodzie i podmianie wyłącznie wejściowych tokenów notatki
wszystkie warianty wybierają poziom 1. Dla `pointwise` oba procesy dają
logity `[19.75, 25.375, 24.875]` i prawdopodobieństwa
`[0.1388969123, 0.4531588256, 0.4079442918]`.
Zmiana notatki jest zatem wystarczająca, aby zmienić wybór w tym replay.
Nie oznacza to, że odpowiada za każdą różnicę liczbową końcowego HTTP.

Odczyt pointwise różni się od wcześniejszego HTTP J2 przy tych samych
tokenach notatki o maksymalnie `0.0078267455`. Poprzedni HTTP zwracał
`[0.1435833722, 0.4562990963, 0.4001175463]`. Probe najpierw odczytuje P2,
a aplikacja wykonuje inne wcześniejsze operacje i generację; nie są to
identyczne historie wykonania. Nie przenosimy bitowej zgodności z poprzedniej
serii na tę nową kolejność. Native FP32 dla notatki native różni się od
zapisanych logitów Torch FP32 o około `1.71e-5`, zachowując wybór poziomu 1.

## Pierwsze rozgałęzienie generacji

Przy wymuszonym identycznym prefiksie pierwsza różnica top-1 FP32 i J2
względem tokenów upstream występuje w kroku 25. Kroki 0–24 zachowują top-1.
Różnice samych odstępów między logitami widać już w kroku 0: dla J2 maksimum
na wspólnych top-8 wynosi około 0.125. Jest to pierwsza obserwowana granica
wyjścia, nie lokalizacja pierwszego różnego operatora wewnątrz modelu.

| Wariant / harmonogram, oba procesy | `danych` 1847 | `konta` 2695 | Wybrany ID |
|---|---:|---:|---:|
| upstream decode, logproby | -1.2298276424 | -1.2298276424 | 1847 |
| inherit-f32, KV | 19.200867 | 19.266417 | 2695 |
| f32-rounded, KV | 19.25 | 19.25 | 1847 |
| pointwise, KV | 19.125 | 19.25 | 2695 |
| pointwise, pełny prefiks po zamknięciu KV | 19.25 | 19.25 | 1847 |

Wiersze native to effective logits, wiersz upstream to logproby; znaczenie ma
remis lub różnica w obrębie wiersza. Remis native rozstrzyga w stronę niższego
ID, zgodnie z upstream. Dla FP32 pełny prefiks daje 19.200863 / 19.266417,
więc wybór się nie zmienia. `f32-rounded` zachowuje 26/26 tokenów upstream
w badanym wymuszonym odcinku, ale dalszej generacji nie wykonano.

**Sam native J2 zmienia top-1 zależnie od sposobu przeliczenia tego samego
prefiksu.** Wynik KV i stateless powtórzył się w obu procesach. Różnica obejmuje
harmonogram, kształty GEMM i ścieżkę attention; nie dowodzi uszkodzenia KV
ani konkretnego błędu kernela. Nie dodano przełącznika recompute ani reguły
zmieniającej wybór przy bliskich logitach.

## Powtarzalność i wniosek

Wszystkie 12 par powtórzeń P2 wewnątrz procesów jest dokładnie równych.
Między procesami FP32 i f32-rounded zachowują wszystkie zapisane odczyty.
Pointwise zachowuje całą zapisaną generację i P2 notatki native, ale zmienia
P2 notatki upstream: logit poziomu 1 różni się o 0.125, a maksymalna różnica
prawdopodobieństw wynosi `0.0042178780`. To kontrolowany kontrprzykład dla
ogólnej powtarzalności profilu po restarcie; nie unieważnia historycznej
równości dwóch procesów na wcześniejszej kolejności HTTP.

Runtime timing cuBLASLt jest możliwym źródłem tej różnicy. Probe nie zapisał
ID wybranych algorytmów ani tensorów po operatorach, więc przyczyna pozostaje
**GAP**. Nie wskazano potwierdzonego defektu aplikacji, który uzasadniałby
poprawkę arytmetyki. Kolejny cel diagnostyczny to rozdzielenie wyboru algorytmu
GEMM od wpływu harmonogramu, z zapisem faktycznie wybranych konfiguracji;
tej dalszej próby nie wykonano w obecnym etapie.

Status zgodności z rzeczywistym upstream pozostaje **FAIL**. PASS jednego
odczytu i zgodny pierwszy remis diagnostycznego FP32-rounded nie stanowią
podstawy do zmiany profilu produkcyjnego ani deklarowania 9/9 zgodnych notatek.
Nie mierzono tu wydajności, fizycznego maksimum pamięci, nowych przypadków,
EOS, limitu generacji, błędów, split ani Metal.

## Weryfikacja i dane

Prywatny probe przeszedł formatowanie, release Clippy z `-D warnings`,
prywatny release Rustdoc i release CUDA build, wszystkie exit 0, offline.
Zwalidowano identyczne tokeny, kody, temperaturę i długości wszystkich
zapisanych odczytów. Sześć procesów i sesja nadzorująca zakończyły się kodem 0.
Produkcję przywrócono; świeży dowód potwierdza healthy, HTTP 200 i jeden
oryginalny proces GPU. Nie wdrożono binarki probe.

SHA-256 binarki probe:
`0217c0a0ee321d3413236a498e7bfd613fea4f52a64032e24c9953db165f3310`.
SHA-256 wejścia `score-input.json`:
`09b22e1f6a625ae5f6347842b28ff926b4bbc1d0c5bc0a8a280de994a1783242`.
Źródła probe, lockfile, dokładne polecenia i surowe wyniki zachowano prywatnie.
[REPRODUCE.md](REPRODUCE.md) opisuje ich odtworzenie. Nie ma publicznego URI;
nie publikowano danych ani nie wykonywano push.

Lokalne prywatne archiwum `measurement-data.tar.gz`: **14 541 718 B**,
SHA-256 `c3e7b173333cbde2daff70361ed8ca52055af7cc9d5cceebf97e4057578bd7e6`.
Osobna kopia archiwum i świeże rozpakowanie potwierdziły SHA archiwum
oraz **58/58** sum plików. Dane są pod
`reports/upstream-api-241-score-isolation-2026-10-28/evidence/private/`.
README wewnątrz archiwum poprzedza dopisanie tej noty.
