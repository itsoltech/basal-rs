# Czy upstream fast około 270 ms zachowuje zgodność z FP32?

**Nie spełnia progu ≤0,0001.** Około 270 ms to pomiar `fast` z wagami
i aktywacjami BF16, nie pełnego FP32. Ponownie przeanalizowano zachowane
odpowiedzi HTTP, bez uruchamiania GPU i bez zmiany kodu inferencji.

W dokładnej serii **270,19 ms** dla 5080 tokenów maksymalna różnica
końcowego skalibrowanego prawdopodobieństwa względem upstream FP32 wyniosła
**0,001065552**, czyli **0,106555 punktu procentowego i 10,66× uzgodniony
limit**. Decyzja została zachowana. Dla 255 tokenów ta seria dała różnicę
0,000718415 i zachowała wybór `label-26`. Oba wejścia przekraczają próg.

## Ta sama kontrola: fast, eager i v4 względem FP32

Poniższe silniki pochodzą z jednej późniejszej kontroli, wykonywanej
kolejno na RTX 4060. Liczby czasu to wcześniejszy SIGNAL, nie nowy ABBA.
Referencja FP32 pochodzi z zamrożonych eksportów upstream na tej samej karcie.

| Silnik | C1/5080 p50 [ms] | Maks. różnica prawdopodobieństwa dla 5080 | Maks. na 5 dokumentach / 12 odpowiedziach |
|---|---|---|---|
| Upstream fast BF16 | 269,06 | **0,002960145** | **0,005885273** |
| Upstream eager BF16 | 554,50 | 0,000664786 | 0,004665762 |
| Basal-rs v4 BF16 | 535,49 | 0,003657080 | 0,010523021 |

W każdym z tych trzech przebiegów pola decyzji wszystkich **12 odpowiedzi**
są zgodne z FP32: wybory, zbiory multi, akcje, typy i pozostałe porównywane
pola dyskretne. Nie ma różnic nazw pytań ani schematów odpowiedzi.
Wynik `score` jest ciągły i zmienia się wraz z prawdopodobieństwami.
Największa różnica fast w tym korpusie, 0,005885273, dotyczy pytania
`severity` w `pico-bank-six-types`: 0,588527 punktu procentowego, 58,85× limit.

Nie dowodzi to spadku trafności decyzji modelu. Dowodzi mniejszej zgodności
numerycznej z FP32. Nie wyznaczano tu NLL, Brier ani trafności na dużym
etykietowanym zbiorze. Nie należy na tej podstawie twierdzić, że fast jest
ogólnie gorszym modelem, ani że BF16 zapewnia próg bliskości FP32.

Eager BF16 również przekracza limit, a v4 nie jest od fast bliższy FP32
na tym zestawie. Zatem nie cały problem można przypisać wyłącznie trybowi
fast. Jego implementacja oprócz BF16 stosuje grafy CUDA, kompilację/fuzję
oraz kernele FLA; część przyspieszenia pochodzi z organizacji wykonania.
Ten audyt nie rozdziela ilościowo wpływu każdej optymalizacji.

Poprzednie porównanie około 269 ms fast z około 776 ms naszego prototypu
bliskiego FP32 zestawiało dwa różne poziomy zgodności numerycznej.
270 ms jest poprawnym punktem szybkości BF16 fast, lecz **nie jest
zmierzonym czasem konfiguracji spełniającej próg FP32 użytkownika**.
Nie zmienia to wcześniejszej decyzji o odrzuceniu wolniejszego prototypu.

## Zakres i odtworzenie

Model: `Remek/basal-omni-pico`, rewizja
`a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`; upstream
`615fe321a818fdf5d2897f758b8d823f919d7f94`, Transformers 5.17.0,
Torch 2.11.0+cu128. Jedna kolejność, te same kalibracje, bez cache prefiksu
i stanu. GPU RTX 4060, SM 8.9. Nie zmieniono ani nie uruchomiono ponownie
checkoutu upstream.

Źródła: [kontrola v4](../pico-parallel-rtx4060-2026-10-26/README.md)
oraz [kontrola późniejszego prototypu](../pico-f32-fast-rtx4060-2026-10-26/README.md).
Sprawdzono zgodność 5/5 żądań po ujednoliceniu wyłącznie nazwy modelu
oraz liczby wejściowych tokenów: 918, 701, 255, 256 i 5080.
W nowej serii fast wszystkie zachowane mierzone odpowiedzi danego dokumentu
są identyczne: odpowiednio 6, 6, 18, 6 i 12 obserwacji. W starszej serii
270,19 ms oba dokumenty mają po 12 identycznych odpowiedzi. Wybrano po
jednym reprezentancie, bez uśredniania prawdopodobieństw.

Do istniejącego `basal compare` przekształcono odpowiedzi HTTP przez `jq`:
`{id, items: [], answer_upstream: .response}`. Korzystamy wyłącznie z sekcji
`upstream_endpoint`. HTTP nie zawiera wewnętrznych logitów ani tablic
tokenów, dlatego metryki benchmarku i tokenów w tym porównaniu są puste;
`0/0` nie jest wynikiem kwalifikacji całego benchmarku 44 pytań.

`compare-fast-270-scoped.json` porównuje dokładnie dwa wejścia starej serii.
Wcześniejszy plik `compare-fast-270.json` zachowuje GAP trzech brakujących
dokumentów wobec pełnego nowszego korpusu; nie są to zmienione decyzje.
`compare-fast-269.json`, `compare-eager.json` i `compare-v4.json` obejmują
wszystkie pięć dokumentów. Brak nowego pomiaru czasu lub nowej inferencji.

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/pico-fast-fp32-audit-rtx4060-2026-10-26/4835989bc7c268fa187c9779f0235eb9dd9e639e7ebbd18213a384c10c78e2b2/measurement-data.tar.gz`.

SHA-256 `4835989bc7c268fa187c9779f0235eb9dd9e639e7ebbd18213a384c10c78e2b2`, rozmiar 51 835 B.
Ponowne pobranie z autoryzacją i anonimowo potwierdziło sumę archiwum
oraz **26/26 sum plików w obu kopiach**. Lokalna kopia archiwum pozostaje
zachowana. Dane mają logiczną ścieżkę
`reports/pico-fast-fp32-audit-rtx4060-2026-10-26/`.
`REPRODUCE.md` w archiwum podaje odtworzenie porównań istniejącym
`basal compare` oraz kontrolę pól decyzji i argmax. Prywatnych ścieżek,
wag, binarek ani surowych logów maszyn nie publikowano.
