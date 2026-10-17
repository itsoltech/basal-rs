# basal-2.0 na pięciu kartach NVIDIA — 2026-10-14

Cel: porównać domyślne serwowanie trzech tekstowych modeli basal-2.0 w basal-rs z upstream v2.0.0 `fast`, na tej samej karcie i tych samych wejściach. Osobno sprawdzamy decyzje basal-rs względem zachowanej referencji FP32. Kampania nie obejmuje omni, obrazu, audio ani formatów kwantyzowanych.

Data kampanii i nazwa katalogu odnoszą się do UTC; sesje rozpoczęto 2026-10-14. Model max na obu RTX-ach powtórzono 2026-10-15 po wyczerpaniu czasu pierwszych sesji. Dane pierwotne zachowano; `selection.json` wybiera pełny nowy przebieg max, bez łączenia powtórzeń między maszynami.

HTTP przy 32 klientach: pełne powtórzenia 15/15 par; basal-rs ma większą medianę przepustowości w 14/15 porównanych par. Zakres stosunku basal-rs/upstream: 0,82–2,57×.
SIGNAL — niższa przepustowość basal-rs: h100/mini: 0,82×. To wynik porównania silników w tej kampanii, nie dowód regresji kodu względem starszej wersji.

SIGNAL — przy 1 kliencie niższa przepustowość basal-rs: h100/mini: 0,91×; h100/4.5B: 0,93×; l40s/mini: 0,95×.
SIGNAL — przy 8 klientach niższa przepustowość basal-rs: h100/mini: 0,75×; a100/mini: 0,85×; l40s/mini: 0,94×.

Zgodność basal-rs z FP32: 15/15 par ma kompletne 49/49 argmax, 41/41 odpowiedzi API i zero różnic tokenów. Te mianowniki nie obejmują nieudanych odpowiedzi ruchu mieszanego.

## Wersje, sprzęt i wejścia

Kod basal-rs: `3fa10de179b0ac8fb4bec024065aa5e90f543096`; binarka CUDA SHA-256 `169ce19a4be027f9026c5be8b2d86bb41a4374d92f4667fa373cb3c5ddf02f30`. Upstream: `af1ce43a1770dd4fdfc1611e3fe720b93d42fc73` (v2.0.0). Oba silniki czytają checkpointy BF16; basal-rs oblicza w FP16, upstream fast w BF16. To porównanie ustawień produktu, z różnymi formatami obliczeń pośrednich.

| Model | Przypięta rewizja Remek/ |
| --- | --- |
| basal-2.0-mini | cb81b32ce89b0b87e62fcdad28f0435e25c953f3 |
| basal-2.0-4.5B | e5ba9149760d99b73b1963d8e0ccf948b58afacb |
| basal-2.0-max | 2fffc5cc41e5cf7a2f752f38f17e7b4d9a4a8252 |

| GPU | Sterownik / CC | Pamięć / limit mocy | CPU / vCPU | Blokada SM |
| --- | --- | --- | --- | --- |
| NVIDIA H100 PCIe | 570.195.03 / 9.0 | 81559 MiB / 350.00 W | AMD EPYC 9554 64-Core Processor / 28 | 1755 MHz / PASS |
| NVIDIA A100-SXM4-80GB | 580.126.09 / 8.0 | 81920 MiB / 500.00 W | Intel(R) Xeon(R) Platinum 8462Y+ / 14 | 1410 MHz / PASS |
| NVIDIA L40S | 580.126.09 / 8.9 | 46068 MiB / 350.00 W | AMD EPYC 9224 24-Core Processor / 12 | 2520 MHz / PASS |
| NVIDIA RTX 6000 Ada Generation | 580.126.09 / 8.9 | 49140 MiB / 300.00 W | Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz / 12 | 3105 MHz / PASS |
| NVIDIA RTX A6000 | 570.195.03 / 8.6 | 49140 MiB / 300.00 W | AMD EPYC 7763 64-Core Processor / 28 | 2100 MHz / PASS |

| Ponowny pomiar max | Sterownik / CC | Pamięć / limit mocy | CPU / vCPU | Blokada SM |
| --- | --- | --- | --- | --- |
| ada | 580.126.09 / 8.9 | 49140 MiB / 300.00 W | Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz / 12 | 3105 MHz / PASS |
| a6000 | 570.195.03 / 8.6 | 49140 MiB / 300.00 W | AMD EPYC 7763 64-Core Processor / 28 | 2100 MHz / PASS |

Obecny H100 jest wariantem PCIe 350 W; historyczny H100 miał limit 700 W. A100-SXM4 ma teraz zastany limit 500 W, poprzednio 400 W. Zmieniły się też przydziały CPU. Rodziny kart są te same co w [kampanii 1.5](../perf-gpus/README.md), ale są to nowe maszyny. Porównanie dwóch silników w bieżącej parze używa tej samej maszyny. Nie wyprowadzamy przyspieszenia między kampaniami z tych danych.

Python 3.13.16, PyTorch 2.11.0+cu128, transformers 5.17.0; pełne zależności w plikach `requirements.txt` archiwum. Native używa bibliotek CUDA 12.9.1; upstream koła PyTorch CUDA 12.8. Zmapowane biblioteki cuBLASLt mają wersje 120901 w native i 120804 w upstream; zapisano je w `runtime-libraries.json`. Na H100 i RTX A6000 użytkowa biblioteka libcuda 575.57.08 pochodzi z pakietu zgodności, przy sterowniku jądra 570.195.03. Karta pracuje bez innych zadań GPU tej kampanii. Zapisano wynik próby blokady SM i rzeczywiste zegary, moc, temperaturę oraz próbki VRAM; blokada nie oznacza stałej częstotliwości pod obciążeniem.

Wejście wydajnościowe: te same 44 krótkie pytania PL/EN co w historycznej referencji `reports/reference-1.5-max-fp32/bench.jsonl`, przepisane na żądania v2. Każde zawiera jedno pytanie choice. Referencja zgodności decyzji to inny zestaw: 49 pytań benchmarku i 26 rekordów żądań, z których 24 poprawne rekordy obejmują 41 odpowiedzi (17 choice, 12 noul, 5 score, 5 multi, 2 tool). Dwa błędy referencji są wyłączone z mianownika 41, zachowane w danych. Referencje FP32 pochodzą z [końcowej kampanii 2.0](../upstream-2.0-final-2026-10-14/README.md); nie wykonujemy nowego FP32 na każdej karcie.

## Metoda

- HTTP: `/v1/basal`, jawne `orders=1` w obu serwerach; pozostałe ustawienia serwera domyślne. Osobny proces serwera dla każdego przebiegu, kolejność ABBAABBA (A=native, B=upstream fast). Cztery uruchomienia na silnik; pierwsze nie wchodzi do statystyki, również gdy przekroczy limit startu. Trzy kolejne muszą się zakończyć poprawnie; tabela podaje medianę z trzech oraz zakres min–max. To zakres obserwacji, nie przedział ufności. p50 i p99 w tabelach to również mediany trzech percentyli obliczonych osobno dla przebiegów, nie percentyle połączonych 900 żądań. Czas uruchomienia, kompilacja i 44 żądania rozgrzewki są poza pomiarem.
- Domyślny AutoGEMM dobiera tabelę według GPU, dtype, wersji cuBLASLt i kształtów macierzy. Źródło tabeli jest w logu serwera, jej kopia w archiwum. Nazwa `basal-1.5-*` w nagłówku ponownie użytej tabeli dotyczy jej pochodzenia i kształtów, a nie uruchomionych wag; w tej kampanii ładowane są przypięte modele 2.0.
- Każdy krótki przebieg: 44 żądania sekwencyjne, następnie po 300 w zamkniętej pętli przy 1, 8 i 32 klientach. Tabele poniżej pokazują fazy po 300 żądań; sekwencyjne 44 są w archiwum. Ten sam klient i harmonogram po obu stronach. Klient nie ponawia żądań zakończonych błędem HTTP. Dodano zapis wszystkich poprawnych odpowiedzi do audytu; licznik różnic pełnych obiektów answers nie jest licznikiem zmienionych decyzji.
- Bez HTTP: 44 pytania, 39 czasów po odrzuceniu pierwszych pięciu; każda forma pytania rozgrzana. `lat2` używa dwóch kolejności opcji. Native wlicza przygotowanie promptu; upstream mierzy backend po tokenizacji. Dlatego czasy nie mają identycznego zakresu. Native zapisuje również osobny czas forwardu. Throughput: grupy po 16, dwie kolejności; native mediana trzech pomiarów wewnętrznych, upstream jeden pomiar. Wyniki tych narzędzi zachowujemy osobno od porównania pełnego HTTP.
- Ruch mieszany: tylko native, jeden przebieg/model/kartę, 44 warmup, 300 sekwencyjnych i 300 przy 32 klientach. Korpus zawiera 400 żądań choice/noul/score/multi/act; usunięto 14 nieobsługiwanych pól evidence/facts z oryginału 1.5. Mini używa filtra długości stanu i 371 żądań. Fazy wykonują pierwsze 300 wejść korpusu; to 300 różnych identyfikatorów na przebieg, a nie pełne 400/371. Kontrola CPU największego przygotowanego promptu całego korpusu wykazała: mini 4405 tokenów, pozostałe 17323. Nie jest to identyczny korpus jak 1.5 ani drabina limitu kontekstu.
- Decyzje: porównujemy pola choice, selected, tool/tools, call, action, answer, odmowę/odroczenie oraz ranking narzędzi. Dla noul pomocniczo stronę progu 0,5, dla score najbardziej prawdopodobny poziom. Confidence i liczbowa wartość score nie są kryterium równości. Eksport 49 pytań sprawdza też tokeny i argmax. Zgodność na tym małym zestawie nie oznacza trafności na innych danych.

## HTTP: 1 klient

| GPU | Model | upstream: żądania/s [min–max]; p50 / p99 ms | basal-rs: żądania/s [min–max]; p50 / p99 ms |
| --- | --- | --- | --- |
| h100 | mini | 187,1 [184,1–187,5]; 5,3 / 6,1; n=3, błędy=0 | 170,5 [169,5–171,2]; 5,8 / 6,6; n=3, błędy=0 |
| h100 | 4.5B | 86,9 [86,5–87,5]; 11,4 / 12,8; n=3, błędy=0 | 80,8 [80,6–81,4]; 12,2 / 13,4; n=3, błędy=0 |
| h100 | max | 52,2 [52,1–52,2]; 19,1 / 20,8; n=3, błędy=0 | 52,9 [52,6–53,3]; 18,7 / 20,6; n=3, błędy=0 |
| a100 | mini | 122,2 [122,2–124,7]; 8,2 / 9,5; n=3, błędy=0 | 124,9 [124,1–125,8]; 7,9 / 9,0; n=3, błędy=0 |
| a100 | 4.5B | 57,9 [57,8–57,9]; 17,4 / 19,0; n=3, błędy=0 | 60,7 [60,7–60,8]; 16,0 / 18,4; n=3, błędy=0 |
| a100 | max | 31,7 [31,4–31,7]; 32,2 / 33,4; n=3, błędy=0 | 39,7 [39,6–39,9]; 24,1 / 29,0; n=3, błędy=0 |
| l40s | mini | 116,0 [115,6–120,0]; 8,4 / 9,8; n=3, błędy=0 | 109,9 [106,4–116,7]; 8,9 / 10,4; n=3, błędy=0 |
| l40s | 4.5B | 46,5 [46,3–47,8]; 21,3 / 24,9; n=3, błędy=0 | 47,9 [47,5–48,1]; 20,6 / 22,6; n=3, błędy=0 |
| l40s | max | 22,4 [22,3–22,6]; 44,9 / 46,7; n=3, błędy=0 | 24,9 [24,8–25,0]; 39,7 / 43,0; n=3, błędy=0 |
| ada | mini | 122,8 [121,1–122,9]; 8,1 / 9,4; n=3, błędy=0 | 129,2 [127,9–129,6]; 7,7 / 8,7; n=3, błędy=0 |
| ada | 4.5B | 40,2 [39,6–40,2]; 24,6 / 32,3; n=3, błędy=0 | 53,0 [52,9–53,2]; 18,4 / 21,0; n=3, błędy=0 |
| ada | max | 16,4 [16,3–16,4]; 61,2 / 78,1; n=3, błędy=0 | 23,7 [23,7–24,0]; 40,1 / 51,5; n=3, błędy=0 |
| a6000 | mini | 82,0 [81,8–82,4]; 12,3 / 13,5; n=3, błędy=0 | 94,0 [94,0–94,2]; 10,3 / 12,3; n=3, błędy=0 |
| a6000 | 4.5B | 30,7 [30,4–30,8]; 32,9 / 36,7; n=3, błędy=0 | 38,1 [37,7–38,6]; 25,2 / 30,5; n=3, błędy=0 |
| a6000 | max | 16,6 [16,5–16,6]; 60,8 / 75,0; n=3, błędy=0 | 22,6 [22,5–22,6]; 43,1 / 48,5; n=3, błędy=0 |

## HTTP: 8 klientów

| GPU | Model | upstream: żądania/s [min–max]; p50 / p99 ms | basal-rs: żądania/s [min–max]; p50 / p99 ms |
| --- | --- | --- | --- |
| h100 | mini | 460,8 [459,2–463,2]; 16,7 / 27,1; n=3, błędy=0 | 346,6 [337,5–351,9]; 22,4 / 35,8; n=3, błędy=0 |
| h100 | 4.5B | 160,8 [159,8–161,2]; 48,2 / 57,9; n=3, błędy=0 | 163,3 [163,3–164,3]; 47,3 / 60,6; n=3, błędy=0 |
| h100 | max | 81,7 [79,5–82,4]; 93,6 / 110,3; n=3, błędy=0 | 93,4 [93,2–94,1]; 84,8 / 92,0; n=3, błędy=0 |
| a100 | mini | 282,6 [281,2–282,9]; 27,1 / 40,2; n=3, błędy=0 | 239,7 [230,0–247,7]; 33,8 / 39,3; n=3, błędy=0 |
| a100 | 4.5B | 102,5 [102,3–102,5]; 73,9 / 89,0; n=3, błędy=0 | 110,6 [110,6–110,9]; 70,7 / 76,9; n=3, błędy=0 |
| a100 | max | 50,2 [50,0–50,2]; 151,4 / 177,4; n=3, błędy=0 | 62,4 [62,2–62,4]; 126,3 / 137,1; n=3, błędy=0 |
| l40s | mini | 251,4 [249,4–255,6]; 30,6 / 45,0; n=3, błędy=0 | 237,3 [222,2–238,0]; 32,0 / 40,1; n=3, błędy=0 |
| l40s | 4.5B | 84,8 [84,6–85,1]; 89,9 / 117,2; n=3, błędy=0 | 92,0 [91,8–92,1]; 85,5 / 91,4; n=3, błędy=0 |
| l40s | max | 37,4 [36,7–37,8]; 198,9 / 267,2; n=3, błędy=0 | 45,3 [45,2–45,4]; 173,1 / 189,8; n=3, błędy=0 |
| ada | mini | 139,8 [139,0–140,2]; 56,0 / 68,7; n=3, błędy=0 | 219,2 [216,5–220,0]; 36,6 / 42,9; n=3, błędy=0 |
| ada | 4.5B | 42,1 [41,9–42,1]; 179,6 / 239,6; n=3, błędy=0 | 76,0 [75,7–76,1]; 103,1 / 112,0; n=3, błędy=0 |
| ada | max | 17,9 [17,7–18,0]; 427,4 / 531,6; n=3, błędy=0 | 31,9 [31,9–32,1]; 245,0 / 273,1; n=3, błędy=0 |
| a6000 | mini | 128,4 [127,9–129,3]; 59,8 / 78,1; n=3, błędy=0 | 151,1 [151,0–155,7]; 52,0 / 61,6; n=3, błędy=0 |
| a6000 | 4.5B | 45,2 [43,0–45,3]; 175,6 / 211,2; n=3, błędy=0 | 60,5 [60,3–60,8]; 129,3 / 143,4; n=3, błędy=0 |
| a6000 | max | 22,1 [21,8–22,2]; 350,7 / 448,2; n=3, błędy=0 | 30,9 [30,9–31,0]; 255,2 / 279,3; n=3, błędy=0 |

## HTTP: 32 klientów

| GPU | Model | upstream: żądania/s [min–max]; p50 / p99 ms | basal-rs: żądania/s [min–max]; p50 / p99 ms |
| --- | --- | --- | --- |
| h100 | mini | 382,5 [367,2–403,3]; 81,4 / 117,1; n=3, błędy=0 | 314,8 [246,8–320,7]; 77,8 / 360,2; n=3, błędy=0 |
| h100 | 4.5B | 135,5 [131,5–139,1]; 233,7 / 269,3; n=3, błędy=0 | 228,6 [227,2–231,4]; 134,1 / 167,5; n=3, błędy=0 |
| h100 | max | 62,6 [61,9–62,7]; 503,0 / 590,8; n=3, błędy=0 | 114,0 [113,2–114,4]; 278,1 / 297,4; n=3, błędy=0 |
| a100 | mini | 241,9 [222,9–251,5]; 133,6 / 172,9; n=3, błędy=0 | 353,8 [351,4–360,6]; 83,0 / 148,1; n=3, błędy=0 |
| a100 | 4.5B | 83,4 [82,0–88,6]; 388,7 / 450,4; n=3, błędy=0 | 146,6 [145,0–148,0]; 211,2 / 246,7; n=3, błędy=0 |
| a100 | max | 40,8 [40,6–40,9]; 820,1 / 908,2; n=3, błędy=0 | 73,5 [72,3–74,5]; 432,7 / 481,3; n=3, błędy=0 |
| l40s | mini | 187,9 [181,9–197,4]; 173,7 / 225,9; n=3, błędy=0 | 327,1 [323,9–332,8]; 90,5 / 183,8; n=3, błędy=0 |
| l40s | 4.5B | 66,8 [64,9–68,0]; 491,1 / 558,9; n=3, błędy=0 | 122,8 [121,6–123,1]; 254,6 / 280,0; n=3, błędy=0 |
| l40s | max | 32,0 [30,5–32,4]; 1041,3 / 1157,3; n=3, błędy=0 | 57,4 [56,5–58,6]; 548,6 / 597,0; n=3, błędy=0 |
| ada | mini | 120,6 [101,8–121,8]; 262,2 / 339,7; n=3, błędy=0 | 256,1 [255,4–256,8]; 119,5 / 150,2; n=3, błędy=0 |
| ada | 4.5B | 34,5 [34,3–34,9]; 923,1 / 1107,4; n=3, błędy=0 | 87,9 [87,6–88,8]; 357,3 / 382,7; n=3, błędy=0 |
| ada | max | 15,3 [15,3–15,4]; 2112,6 / 2352,4; n=3, błędy=0 | 39,3 [38,6–39,6]; 808,3 / 871,9; n=3, błędy=0 |
| a6000 | mini | 117,6 [110,8–123,6]; 265,5 / 373,8; n=3, błędy=0 | 195,1 [193,6–195,9]; 155,3 / 276,1; n=3, błędy=0 |
| a6000 | 4.5B | 36,6 [35,9–37,3]; 861,8 / 1015,8; n=3, błędy=0 | 69,2 [69,2–69,6]; 454,2 / 497,0; n=3, błędy=0 |
| a6000 | max | 17,6 [17,6–18,1]; 1796,7 / 2163,7; n=3, błędy=0 | 34,8 [34,8–34,9]; 906,8 / 974,1; n=3, błędy=0 |

## Status wykonania wybranych przebiegów

| GPU | Model | Etap | Status | Czas s |
| --- | --- | --- | --- | --- |
| ada | max | basal-2.0-max-upstream-fast-server-r1-startup | TIMEOUT_STARTUP | 900,5 |
| ada | max | basal-2.0-max-upstream-fast-offline | TIMEOUT | 600,2 |
| a6000 | max | basal-2.0-max-upstream-fast-server-r1-startup | TIMEOUT_STARTUP | 900,8 |
| a6000 | max | basal-2.0-max-upstream-fast-offline | TIMEOUT | 568,2 |

Nieudany pierwszy start nie dostarcza odpowiedzi do audytu. Mianowniki w tabeli zgodności obejmują tylko zapisane poprawne odpowiedzi; trzy kompletne powtórzenia 2/3/4 są warunkiem mediany HTTP.

## Bez HTTP i energia

| GPU | Model | upstream lat1 / lat2 ms / dec/s | native lat1 / lat2 ms / dec/s | HTTP32 J/decyzję: upstream / native |
| --- | --- | --- | --- | --- |
| h100 | mini | 4,2 / 10,9 / 234,1 | 4,7 / 4,9 / 552,5 | 0,9 / 1,0 |
| h100 | 4.5B | 10,1 / 24,5 / 89,2 | 10,9 / 11,3 / 213,2 | 2,6 / 1,5 |
| h100 | max | 17,2 / 38,9 / 46,9 | 17,3 / 18,2 / 100,6 | 5,6 / 3,0 |
| a100 | mini | 6,9 / 13,3 / 154,3 | 6,8 / 7,1 / 340,9 | 1,7 / 1,0 |
| a100 | 4.5B | 16,2 / 32,0 / 57,4 | 14,9 / 16,4 / 130,6 | 5,2 / 2,8 |
| a100 | max | 31,0 / 53,9 / 30,0 | 22,5 / 26,6 / 63,9 | 12,1 / 6,5 |
| l40s | mini | 7,0 / 15,1 / 148,2 | 7,4 / 7,6 / 297,9 | 1,8 / 0,9 |
| l40s | 4.5B | 19,7 / 40,1 / 52,7 | 19,1 / 19,7 / 108,7 | 5,2 / 2,6 |
| l40s | max | 42,5 / 81,1 / 24,8 | 37,7 / 39,6 / 49,6 | 10,9 / 5,9 |
| ada | mini | 6,6 / 13,7 / 93,4 | 6,6 / 6,7 / 229,2 | 2,5 / 1,1 |
| ada | 4.5B | 19,9 / 42,7 / 28,0 | 18,0 / 19,1 / 77,7 | 8,7 / 3,4 |
| ada | max | 50,2 / 92,7 / 12,2 | 37,7 / 45,7 / 34,9 | 19,6 / 7,6 |
| a6000 | mini | 10,5 / 18,5 / 76,4 | 8,8 / 9,9 / 176,9 | 2,5 / 1,5 |
| a6000 | 4.5B | 29,8 / 52,7 / 28,0 | 23,1 / 27,0 / 63,0 | 8,2 / 4,3 |
| a6000 | max | 56,8 / 87,3 / 13,7 | 41,1 / 45,3 / 30,0 | 16,9 / 8,6 |

ada/basal-2.0-max: upstream bez HTTP pochodzi z ponownej próby `ada-recovery/diagnostic/basal-2.0-max-upstream-fast-offline.json` po nieukończeniu pierwotnego pomiaru; pierwsze niepowodzenie pozostaje w tabeli statusów i logach. Użyto tych samych przypiętych wag, wejść i środowiska. Pierwotny limit wynosił do 600 s, ponowny do 800 s; czas przygotowania nie jest składnikiem podanych opóźnień decyzji.
a6000/basal-2.0-max: upstream bez HTTP pochodzi z ponownej próby `a6000-recovery/diagnostic/basal-2.0-max-upstream-fast-offline.json` po nieukończeniu pierwotnego pomiaru; pierwsze niepowodzenie pozostaje w tabeli statusów i logach. Użyto tych samych przypiętych wag, wejść i środowiska. Pierwotny limit wynosił do 600 s, ponowny do 800 s; czas przygotowania nie jest składnikiem podanych opóźnień decyzji.

Energia to średnia moc GPU w oknie fazy pomnożona przez jej czas i podzielona przez liczbę odpowiedzi, próbkowanie co 200 ms. Nie obejmuje CPU ani całej maszyny. Krótkie fazy i próbkowanie ograniczają dokładność; zapisany szczyt VRAM jest próbkowany, nie gwarantuje wykrycia każdego maksimum.

## Ruch mieszany, native

| GPU | Model | sekwencyjnie: żądania/min; p50 / p95 s | 32 klientów: żądania/min; p50 / p95 s |
| --- | --- | --- | --- |
| h100 | mini | 4065,7; 0,007 / 0,048; błędy=0 | 6992,7; 0,260 / 0,340; błędy=0 |
| h100 | 4.5B | 763,4; 0,015 / 0,432; błędy=0 | 1197,2; 0,754 / 2,606; błędy=8 |
| h100 | max | 412,5; 0,024 / 0,814; błędy=0 | 673,8; 1,188 / 5,298; błędy=10 |
| a100 | mini | 2584,0; 0,010 / 0,077; błędy=0 | 4447,7; 0,443 / 0,537; błędy=0 |
| a100 | 4.5B | 416,4; 0,022 / 0,834; błędy=0 | 709,0; 0,973 / 5,703; błędy=9 |
| a100 | max | 235,1; 0,035 / 1,491; błędy=0 | 395,1; 1,450 / 8,767; błędy=10 |
| l40s | mini | 2473,1; 0,010 / 0,079; błędy=0 | 3953,9; 0,475 / 0,611; błędy=0 |
| l40s | 4.5B | 406,3; 0,025 / 0,834; błędy=0 | 684,2; 1,018 / 5,578; błędy=10 |
| l40s | max | 213,6; 0,048 / 1,524; błędy=0 | 366,7; 1,677 / 10,363; błędy=10 |
| ada | mini | 2097,0; 0,011 / 0,101; błędy=0 | 3253,9; 0,561 / 0,777; błędy=0 |
| ada | 4.5B | 324,1; 0,030 / 1,040; błędy=0 | 522,0; 1,363 / 5,331; błędy=10 |
| ada | max | 156,3; 0,063 / 2,235; błędy=0 | 245,3; 2,414 / 14,388; błędy=10 |
| a6000 | mini | 1551,8; 0,015 / 0,133; błędy=0 | 2468,1; 0,748 / 0,991; błędy=0 |
| a6000 | 4.5B | 223,0; 0,037 / 1,545; błędy=0 | 368,4; 1,685 / 8,617; błędy=10 |
| a6000 | max | 118,4; 0,067 / 2,986; błędy=0 | 206,3; 2,524 / 15,806; błędy=11 |

FAIL — ruch mieszany przy 32 klientach: większe modele zwróciły nieudane odpowiedzi; licznik błędów jest w każdym wierszu. Przepustowość dotyczy poprawnych odpowiedzi, p50/p95 tylko poprawnych żądań. Nie traktujemy tych wyników jako bezbłędnej pojemności serwera. Klient pierwotnego pomiaru nie zachował kodów i treści błędnych odpowiedzi; osobna diagnostyka na nowych sesjach RTX jest pod `diagnostic/`. Jej wyników nie włączamy do median HTTP ani pierwotnych tabel ruchu mieszanego. Diagnostyka używa tych samych wejść, 44 żądań rozgrzewki, 1 sekwencyjnego i 300 przy 32 klientach; zachowuje kody i treści odpowiedzi non-200.

| Osobna diagnostyka | Model | HTTP: liczba błędów | Kod lub treść błędu |
| --- | --- | --- | --- |
| ada | 4.5B | 529: 9 | overloaded: too many expanded requests in flight |
| ada | max | 529: 11 | overloaded: too many expanded requests in flight |
| a6000 | 4.5B | 529: 10 | overloaded: too many expanded requests in flight |
| a6000 | max | 529: 11 | overloaded: too many expanded requests in flight |

Zapisane komunikaty `overloaded: too many expanded requests in flight` oznaczają odmowę przyjęcia żądania. W mierzonym [kodzie serwera](https://github.com/itsoltech/basal-rs/blob/3fa10de179b0ac8fb4bec024065aa5e90f543096/crates/basal-cli/src/serve.rs#L430) rozwinięte wejście powyżej 128 KiB wymaga miejsca we wspólnej puli czterech dużych żądań. Brak wolnego miejsca zwraca 529. Nie zmieniano tego limitu podczas kampanii. Pierwotny klient nie zapisał kodów błędów, więc tej przyczyny nie przypisujemy automatycznie wszystkim kartom.

Tabela diagnostyczna dotyczy osobnych 300 żądań przy 32 klientach, po rozgrzewce i jednym żądaniu sekwencyjnym. Nie zastępuje liczników z pierwotnego pomiaru i nie rozstrzyga kodów błędów na kartach, dla których ich nie zapisano. Pełne odpowiedzi są w `*-answers.json.errors.json`.

## Zgodność decyzji i zachowanie pod obciążeniem

| GPU | Model | 49 argmax vs FP32 | 41 odpowiedzi vs FP32 | różnice tokenów | zmienione odpowiedzi HTTP / obserwacje: upstream / native | mieszane native: zmiany / obserwacje |
| --- | --- | --- | --- | --- | --- | --- |
| h100 | mini | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/644 |
| h100 | 4.5B | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/636 |
| h100 | max | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/634 |
| a100 | mini | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/644 |
| a100 | 4.5B | 49/49 | 41/41 PASS | 0 | 32/3952 / 0/3952 | 0/635 |
| a100 | max | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/634 |
| l40s | mini | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/644 |
| l40s | 4.5B | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/634 |
| l40s | max | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/634 |
| ada | mini | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/644 |
| ada | 4.5B | 49/49 | 41/41 PASS | 0 | 33/3952 / 0/3952 | 0/634 |
| ada | max | 49/49 | 41/41 PASS | 0 | 0/2964 / 0/3952 | 0/634 |
| a6000 | mini | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/644 |
| a6000 | 4.5B | 49/49 | 41/41 PASS | 0 | 0/3952 / 0/3952 | 0/634 |
| a6000 | max | 49/49 | 41/41 PASS | 0 | 0/2964 / 0/3952 | 0/633 |

FAIL — powtarzalność upstream fast 4.5B: A100 32/3952, RTX 6000 Ada 33/3952 obserwacji zmieniło choice względem pierwszej odpowiedzi na ten sam identyfikator. Dotyczy `bench-31`, opisu niedziałającego systemu płatności: option_3 (wysoka pilność) → option_2 (średnia pilność), przy 8 lub 32 klientach. To zmiana decyzji, nie tylko confidence. Te statusy nie oznaczają błędu wykonania pomiaru; są zachowane w `validation.json`.

Obserwacje obejmują rozgrzewkę i pierwszy odrzucony przebieg; każdą odpowiedź porównujemy z pierwszą odpowiedzią na ten sam identyfikator w tym samym silniku. Mianownik obserwacji jest liczbą żądań, nie pytań. Brak zmian pod obciążeniem to powtarzalność w obrębie silnika; zgodność z FP32 ma osobne mianowniki 49 i 41. Szczegóły, różnice i statusy są w `summary.json`, a wszystkie odpowiedzi w plikach `*.observations.json`. Nie utożsamiamy różnic liczbowych w answers ze zmianą decyzji.

## Ograniczenia i decyzja

Wnioski dotyczą wskazanych wejść, wersji, sprzętu i ustawień. Krótkie żądania są cyklicznie powtarzane; konfiguracje domyślne mogą korzystać z cache. Nie jest to pomiar zimnych, unikalnych dokumentów, ruchu otwartego, długiego testu stabilności ani jakości na 900 pytaniach. Ruch mieszany i benchmark bez HTTP mają pojedyncze przebiegi. Nie ma pomiaru mieszanego upstream ani mediów omni. Zestaw 41 odpowiedzi nie obejmuje act; act sprawdzamy tutaj tylko pod kątem powtarzalności w ruchu mieszanym, bez nowego porównania z FP32. Dwie wstępne próby uruchomienia runnera (RTX 6000 Ada i L40S) przerwano z powodu obsługi SSH i sprawdzania procesu; nie dostarczyły ukończonych pomiarów i zostały powtórzone. W pierwszym przebiegu max na RTX 6000 Ada ruch mieszany przerwano po 176,2 s z powodu końca sesji; pominięto eksport jakości i oba pomiary bez HTTP. Na RTX A6000 czwarty upstream nie zdążył się uruchomić, pominięto czwarty native i końcowe etapy. Nie są to potwierdzone błędy modeli. W tabelach max obu RTX-ów użyto nowych pełnych sesji; oryginalne niepowodzenia pozostają w `ada/progress.json` i `a6000/progress.json`. Czas uruchamiania nie wchodzi do tabel HTTP. Nie zmieniono numeryki runtime’u ani kalibracji na podstawie tych wyników. Wyników historycznych nie nadpisujemy.

## Artefakty i odtworzenie

URI: `hf://buckets/itsoltech/basal-rs/measurements/perf-basal2-fivegpu-2026-10-14/5e3a38733bcda4c9d2342857820ac53c045687298db9a021971874b2be7bf234/measurement-data.tar.gz`. SHA-256 archiwum: `5e3a38733bcda4c9d2342857820ac53c045687298db9a021971874b2be7bf234`. Pobranie i weryfikacja: [wspólna procedura](../../docs/MEASUREMENT_ARTIFACTS.md).

Weryfikacja 2026-10-15: pobranie z autoryzacją i anonimowo potwierdziło SHA-256 archiwum oraz 1003/1003 sum plików w obu kopiach. Rozmiar archiwum: 4311790 B.

W archiwum katalog `reports/perf-basal2-fivegpu-2026-10-14/` zawiera `inputs/` (manifest, korpusy, referencje FP32), po jednym katalogu `h100`, `a100`, `l40s`, `ada`, `a6000` (metryki, odpowiedzi, logi, eksporty, sprzęt i zależności), `ada-recovery` i `a6000-recovery` (nowe pełne przebiegi max), `selection.json`, `summary.json`, `validation.json` oraz `reproduce/` (zamrożone skrypty). Ścieżki wykonania i lokalny adres endpointu w wynikach zastąpiono nazwami logicznymi; nie zmieniono liczb. Manifest wejść zachowuje hashe sprzed tej zamiany, a `SHA256SUMS` opisuje publiczne pliki. Runnery zachowują ustalone punkty montowania, potrzebne do ponownego uruchomienia.

Po pobraniu, sprawdzeniu SHA archiwum i wszystkich wpisów `SHA256SUMS`, agregację można odtworzyć bez GPU:

```bash
BASAL_REPORT_ROOT="$BASAL_ARTIFACT_ROOT/reports/perf-basal2-fivegpu-2026-10-14"
python3 "$BASAL_REPORT_ROOT/reproduce/summarize.py" \
  --measurement-artifacts "$BASAL_REPORT_ROOT" --out /tmp/basal2-summary.json
```

Ponowny pomiar wymaga CUDA, zbudowania podanego commita, instalacji zapisanych zależności i pobrania przypiętych wag. `reproduce/setup.sh` opisuje środowisko `/work` i `/data`; wymaga `uv` 0.13.0. Korpusy umieść pod `/work/.cache/cloud/basal2-fivegpu/inputs`, binarkę pod `/opt/basal-dev/basal-cuda`, kod upstream pod `/work/upstream`. Uruchom `reproduce/campaign.py --measurement-artifacts <katalog-wejść> --deadline <czas-UNIX>`. Katalog wyników musi być nowy. Polecenia te nie wynajmują ani nie usuwają GPU; sesją zarządza operator zgodnie z [CLOUD.md](../../docs/CLOUD.md).
