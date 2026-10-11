# Repliki: kontrola porównania na jednym GPU

Te same binarki placementu i replik dały mediany 70,564 / 70,585
odpowiedzi/s (+0,030%) przy 16 klientach i żądaniu zegara SM 1200 MHz.
W ośmiu sąsiednich parach przepustowość B była wyższa sześć razy, niższa
dwa. Kierunki różnic latency i kolejki również były mieszane.
Ten pomiar nie odtworzył jednokierunkowego spadku przepustowości;
nie dowodzi zerowego narzutu ani równoważności przy zegarach automatycznych.
Wcześniejszy [wynik −3,04%](../multigpu-replicas-2026-10-11/README.md)
pozostaje obserwacją o niewyjaśnionej przyczynie.

## Metoda i zakres

2026-10-11, 2 × RTX 6000 Ada Generation, 49140 MiB, 300 W,
sterownik 580.126.09, cuBLASLt 12.9.1.4. CPU Xeon Gold 6338,
26 dostępnych CPU. Mini F16, ta sama tabela GEMM i YAML bez `device`;
różni się tylko katalog wyjściowy cache GEMM danego przypadku.
A: placement `ae15ad9`; B: binarka replik bez zmiany Rust od głównego
pomiaru. Fizyczny GPU 1 przez `CUDA_VISIBLE_DEVICES=1`, metryki włączone.
GPU 0 pozostawał bez obciążenia.

Niezmieniony [run-isolation-cloud.py](../../tools/perf/run-isolation-cloud.py):
`--campaigns overhead --repeats 4 --rates '' --closed-requests 2000
--closed-clients 16`. Cztery sekwencje A/B/B/A, świeży serwer w każdym
przypadku, 34 requesty rozgrzewki sekwencyjnej i 64 współbieżnej.
Wszystkie przypadki mają ten sam harmonogram 2000 wejść. Klient,
runner i serwery dziedziczą tę samą maskę 16 logicznych CPU;
pięć obserwacji aktywnego procesu potwierdziło tę maskę. To kontrola
próbkowana, nie ciągły zapis wszystkich procesów.

Żądanie `nvidia-smi -lgc 1200,1200` zakończyło się powodzeniem.
Rzeczywiste zegary były zmienne: średnie A 1179,4–1196,1 MHz,
B 1183,4–1194,2 MHz, minimum 1095 MHz w obu wariantach.
Nie nazywamy tego literalnie stałym zegarem. Limit mocy pozostał 300 W.
Nie usunięto żadnego przypadku ukończonej kampanii ani nie wybierano
przypadków według wielkości partii.

Percentyle nearest-rank liczone osobno w przypadku; tabela poniżej
pokazuje mediany ośmiu statystyk A i ośmiu B.

| Metryka | A | B | B/A − 1 |
|---|---:|---:|---:|
| Odpowiedzi/s | 70.564 | 70.585 | +0.030% |
| Latency p50 (ms) | 226.466 | 226.627 | +0.071% |
| Latency p95 (ms) | 266.406 | 267.289 | +0.332% |
| Latency p99 (ms) | 279.636 | 280.288 | +0.233% |
| Kolejka p50 (ms) | 24.115 | 21.269 | −11.802% |
| Kolejka p99 (ms) | 228.692 | 229.649 | +0.418% |

## Wszystkie przypadki

Średnia partii jest ważona requestami. Wartości SM pochodzą z telemetrii
fazy; nie są izolowanym pomiarem kerneli.

| Przypadek | Wariant | Odpowiedzi/s | Latency p99 (ms) | Średnia partii | SM średni (MHz) | SM min (MHz) |
|---|---|---:|---:|---:|---:|---:|
| 01-base | A | 70.487 | 277.403 | 12.878 | 1196.1 | 1155 |
| 02-cand | B | 70.833 | 278.767 | 14.065 | 1189.4 | 1125 |
| 03-cand | B | 70.376 | 281.374 | 13.854 | 1191.4 | 1125 |
| 04-base | A | 69.014 | 282.743 | 12.500 | 1192.1 | 1110 |
| 05-base | A | 70.672 | 278.200 | 14.065 | 1188.4 | 1110 |
| 06-cand | B | 70.794 | 279.501 | 14.065 | 1185.5 | 1110 |
| 07-cand | B | 69.235 | 281.074 | 12.500 | 1183.4 | 1095 |
| 08-base | A | 70.641 | 278.646 | 14.065 | 1187.1 | 1110 |
| 09-base | A | 69.222 | 286.592 | 11.579 | 1189.3 | 1110 |
| 10-cand | B | 70.951 | 278.735 | 14.065 | 1183.6 | 1095 |
| 11-cand | B | 69.259 | 282.465 | 11.125 | 1188.1 | 1125 |
| 12-base | A | 69.215 | 281.141 | 12.500 | 1184.9 | 1095 |
| 13-base | A | 70.840 | 280.626 | 14.065 | 1185.9 | 1095 |
| 14-cand | B | 70.018 | 285.331 | 9.355 | 1194.2 | 1110 |
| 15-cand | B | 71.010 | 276.805 | 14.065 | 1184.5 | 1095 |
| 16-base | A | 70.871 | 277.288 | 14.065 | 1179.4 | 1110 |

Sąsiednie pary AB/BA: zmiana przepustowości od −1,990% do +2,498%,
p50 latency od −2,045% do +2,007%, p95 od −4,052% do +1,651%,
p99 od −2,742% do +1,677%. B było gorsze w 2/8 parach przepustowości,
3/8 p50 latency, 4/8 p95, 5/8 p99 oraz 4/8 p50 i 6/8 p99 kolejki.
Wszystkie wartości, różnice par i rozkłady partii są w
[measurements.json](measurements.json). Nie stosowano progu akceptacji.

Opisowo porównano również dokładnie równe histogramy wielkości partii.
Histogram `{15: 1860, 10: 10, 1: 130}` wystąpił w 4/8 przypadków A
i 4/8 B; mediany przepustowości tej grupy 70,756 / 70,892/s (+0,192%).
Histogram `{14: 1750, 2: 250}` wystąpił w dwóch A (69,014 i 69,215/s)
i jednym B (69,235/s). Wszystkie 16 przypadków przypisano do siedmiu
grup w JSON, również do grup bez odpowiednika drugiego wariantu.
To dodatkowy opis, bez wyłączania obserwacji z głównego porównania.
Równość histogramu nie oznacza tej samej kolejności partii, podziału
wejść ani zegarów; nie dowodzi braku narzutu lub przyczyny wcześniejszego
spadku. Najczęstszy histogram miał tutaj równą częstość w A i B;
częstość rozkładów przy automatycznych zegarach pozostaje nieustalona.

## Poprawność i przerwany przebieg

Ukończona kampania: 162 PASS, 0 FAIL, 0 GAP, exit 0. Wszystkie 32 000
mierzone odpowiedzi 200 były zgodne z pełną referencją starej binarki
po pominięciu `usage.latency_ms`; nie było błędów transportu, 529 ani
brakujących odpowiedzi. Rozgrzewki: 1536 × 200, 16 × 400, 16 × 422,
również zgodne. Wszystkie 16 serwerów zakończyło się kodem 0 i zwolniło
VRAM. Po załadowaniu 3646 MiB, próbkowane szczyty faz 4040–4072 MiB.
To zgodność z poprzednią binarką, nie nowy pomiar FP32 upstream.

W tej samej sesji wcześniejsze żądanie 1500 MHz nie dało porównywalnych
rzeczywistych zegarów: pierwszy A średnio 1452,6 MHz, następne B
1357,8 i 1369,5 MHz. Przebieg przerwano po sześciu zakończonych
przypadkach: incomplete, 66 PASS, 0 FAIL, 0 GAP, exit 1; proces nadrzędny
otrzymał SIGINT. Zachowano surowe dane oraz wszystkie sześć wyników,
lecz nie włączono ich do ukończonego porównania 1200 MHz.
Brak FAIL w tym przerwanym przebiegu nie oznacza sukcesu kampanii.

Pobrano oba przebiegi, ponownie sprawdzono SHA-256 wszystkich 206
plików, usunięto maszynę i potwierdzono jej brak w API. Niezależny audyt
sprawdził skróty odpowiedzi, kompletność indeksów, harmonogram, formuły
czasów, percentyle, mediany, przyrosty metryk i pokrycie telemetrii.
Runner nie utrwala dokładnych t0/t1 fazy: dokładnej przepustowości z
samych requestów nie odtworzono; sprawdzono mediany zapisanych wyników.
Zaokrąglony czas telemetrii pozwala sprawdzić próbki wewnętrzne,
nie odtworzyć co do próbki pełnego okna runnera.

## Ograniczenia

Jedna karta użyta do obliczeń, jeden model i jedna pętla zamknięta,
metryki włączone, obniżony cel zegara. Nie mierzono CPU ani trybu
bez metryk. Maska CPU i typ procesora różnią się od pierwotnej sesji;
liczby służą do porównania A/B w tej kontroli, nie między sesjami.
Pozostaje nieustalony wpływ kodu na rozkład partii i zegary w trybie
automatycznym. Żaden wynik nie potwierdza bezwarunkowego braku regresji.
