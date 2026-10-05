# GEMM niezależny od partii z algorytmem dobranym do każdej klasy M

Data: 2026-10-05, basal-1.5-max, RTX 6000 Ada, limit 250 W.

Dotąd tryb niezależny od partii używał jednego algorytmu cuBLASLt na kształt
wag dla każdej liczby wierszy M, bo inny algorytm mógłby sumować iloczyny
po K w innej kolejności. Eksperyment (`BASAL_GEMM_EQUIV=1 basal gemm-search
--invariant`, `equiv.log`, `search-groups.log`): każdy kandydat liczy GEMM na
tych samych danych w każdej klasie M, a kandydaci z bitowo identycznymi
wynikami tworzą grupę. Dla każdego kształtu wychodzą 2–3 grupy; w
największej (29–35 algorytmów) są najszybsze algorytmy prawie każdej klasy.

| Kształt (N×K) | Jeden algorytm na kształt | Najlepszy algorytm grupy w każdej klasie |
|---|---:|---:|
| qkv 6144×4096 | 1,160 | 1,004 |
| o_proj 4096×4096 | 1,140 | 1,000 |
| gate/up 28672×4096 | 1,133 | 1,000 |
| down 4096×14336 | 1,218 | 1,000 |

(średnia geometryczna spowolnienia względem najszybszego kandydata klasy,
25 klas M od 16 do 16384)

`gemm-search --invariant` wybiera teraz grupę o najmniejszym takim
spowolnieniu i zapisuje dla każdej klasy jej najszybszy algorytm
(`gemm-algos-f16-invariant-groups.json`, 100 wpisów, 31 algorytmów). W
czasie wykonania M trafia do najbliższej klasy nie mniejszej niż jego
własna; wszystkie algorytmy kształtu dają te same wyniki, więc wynik wiersza
nadal nie zależy od partii.

## A/B (`run-ab.sh`) wobec tabeli `gemm-retune/gemm-algos-f16-invariant-ladder.json`

| Pomiar | Jeden algorytm na kształt | Algorytm na klasę M |
|---|---:|---:|
| Pojedyncza decyzja, basal-bench, mediana lat2 (ABBA) | 62,0–63,0 ms | 63,4–64,0 ms |
| basal-bench, przepustowość w grupach po 16 | 21,3–21,6 dec/s | 23,6 dec/s |
| HTTP, 1 klient, p50 | 67–69 ms | 73 ms |
| HTTP, 32 klientów (ABBA) | 18,44–18,82 żądania/s, p50 1692–1723 ms | 21,13–21,21 żądania/s, p50 1491–1496 ms |
| Energia przy 32 klientach | 13,3–13,5 J/decyzję | 11,8 J/decyzję |
| Stan 16k, 1 / 5 pytań | 6462 / 7037 ms | 5838 / 6375 ms |
| Stan 2k, 1 / 5 pytań | 560 / 734 ms | 493 / 647 ms |

Pod obciążeniem +13% żądań na sekundę i −12% energii na decyzję, długie
stany 9–12% szybciej; pojedyncze żądanie bez kolejki 2–6% wolniej (przy
jednym kliencie zegar SM był niższy: 851–875 MHz wobec 1069–1148 MHz).

Zgodność: pojedynczo = drzewo (`compare-new-single-vs-tree.json`: 0,0),
odpowiedzi pod obciążeniem identyczne we wszystkich fazach. Wobec FP32
upstream 44/44; różnice logitów liter: mediana 0,0127, p95 0,057, maks.
0,271 (poprzednia tabela 0,0121 / 0,054 / 0,171); różnice prawdopodobieństw
po kalibracji maks. 0,0008 (poprzednio 0,0053). Część kształtów używa innej
grupy niż poprzednia tabela, więc wyniki różnią się od niej na poziomie szumu
FP16 (`compare-old-vs-new.json`: maks. różnica logitu 0,167).
