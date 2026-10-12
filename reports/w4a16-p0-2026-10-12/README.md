# W4A16 P0 — diagnostyka wag, 2026-10-12

Stan: pakiet do niezależnego review danych. Narzędzie wskazuje q4_k; koordynator zastosuje delegowaną regułę wyboru po zamknięciu review. Etap W4 i jakość modelu pozostają otwarte.

Q4_K ma niższe R niż Q4_0 na obu przypiętych modelach. R = suma kwadratów błędu rekonstrukcji / suma kwadratów dokładnych wag bf16, po wszystkich czterech projekcjach każdej warstwy. Rekonstrukcja to przypięty Candle to_float → f16 RNE → f32; sumy f64 w kanonicznej kolejności z C=16. To wybór w przestrzeni wag; trafność i zachowanie odpowiedzi wymagają osobnej kampanii GPU.

## Wyniki

| Model | Format | R | Czas toru hosta [s] | Grupy spłaszczone |
|---|---|---:|---:|---:|
| mini | q8_0 | 3.088302523e-05 | 1.817552 | 0 |
| mini | q4_0 | 0.007932102043 | 1.663077 | 0 |
| mini | q4_1 | 0.006390076625 | 1.512112 | 0 |
| mini | q4_k | 0.005807692642 | 4.590151 | 0 |
| 4.5B | q8_0 | 3.089537348e-05 | 6.637877 | 0 |
| 4.5B | q4_0 | 0.00790283135 | 6.472730 | 0 |
| 4.5B | q4_1 | 0.00639803156 | 5.845659 | 0 |
| 4.5B | q4_k | 0.005808291323 | 17.706247 | 9 |

R Q4_K wobec Q4_0 jest niższe o 26.782% dla mini i 26.504% dla 4.5B. Q4_0 i Q4_K są dopuszczalne na obu modelach; Q4_0 nie jest ściśle lepsze na obu, więc ustalona funkcja wybiera Q4_K (both_eligible_otherwise_q4_k), bez rozbieżności. Q8_0 i Q4_1 są kontekstem; nie są kandydatami tej reguły.

## Zakres i kompletność

Jedno normalne wykonanie CPU: exit 0, complete:true, 60,636679719 s według narzędzia (60,67 s monotonicznego czasu procesu). 16 workerów, 22 dostępne logiczne CPU; C=16 wierszy. Nie kontrolowano zegarów CPU ani izolacji obciążenia i nie wykonano powtórzeń czasu. Czasy obejmują pełny tor kwantyzacji, rekonstrukcji, kontroli, statystyk, tworzenia workerów i redukcji; czytanie macierzy i SHA są zapisane osobno. P0 Q8 również rekonstruuje i liczy statystyki, więc różnica czasów formatów nie jest narzutem ładowania silnika. Czasy nie są kalibracją CPU w chmurze, pomiarem GPU, request latency ani gwarancją granicy czasu.

Mini: 32 warstwy, 128 macierzy, 1 497 366 528 wartości, 46 792 704 grupy po 32. 4.5B: 60 warstw, 240 macierzy, 4 624 220 160 wartości, 144 506 880 grup. Każda macierz przeszła wszystkie cztery formaty: 1472 przebiegi macierz–format. Wszystkie wartości uwzględnione; zero pominiętych macierzy/wierszy, NaN i ±inf wejścia oraz obu rekonstrukcji, zerowych i niezerowych stałych grup źródła. Wszystkie R ważne. Nie oznacza to braku pojedynczych zer w wagach.

Max: wyłącznie kontekst kształtów (50 warstw, 200 macierzy, podzielność K i limit n spełnione); bez kwantyzacji, bez hasha wag, bez udziału w wyborze.

## Diagnostyka brzegowa

W 4.5B Q4_K wykryto 9 grup z niestałym źródłem i stałą skończoną rekonstrukcją f16; mini: 0. Ich błąd jest uwzględniony w R. Pierwsza pozycja (indeksy od 0): warstwa 3, gate|up, model.layers.3.mlp.gate_proj.weight, wiersz 1681, kolumna 256. Liczniki macierzy ze spłaszczonymi grupami i ich pierwsze pozycje w facts.json; dokładnej przyczyny nie przypisujemy bez prywatnych metadanych skal. Grupy spłaszczone nie mają progu i nie wyłączają dopuszczalności według planu.

Q4_0 4.5B: 20 429 podnormalnych skal f16; mini: 0. Skale zerowe niezerowych bloków i niefinitywne: 0 dla obu. Skala jest odtworzona według przypiętego kwantyzatora i sprawdzona na rekonstrukcji maksimum, bez czytania prywatnych pól. Podnormalne d/dmin Q4_K pozostają nieobserwowalne przez publiczne API.

## Weryfikacja i odtwarzalność

Koordynator niezależnie sprawdził exit, długość/SHA artefaktu i summary stdout, wszystkie siedem osadzonych SHA źródeł/locka, binarkę i przypięty Rust 1.95.0. Po wykonaniu ponownie policzył pełne SHA wag i configów mini/4.5B zgodne z manifestami FP32. Sprawdzenie długości/mtime i hashe końców przebiegu wspierają założenie zaufanego cache tylko do odczytu; nie są niezmiennym snapshotem ani dowodem przeciw chwilowej zmianie i przywróceniu danych.

Wszystkie statystyki, R i sumy czasu modelu, warstw oraz typów projekcji niezależnie przeliczono z rzeczywistych rekordów macierzy, łącznie z porównaniem bitów f64, kompletnością zakresu i mapowaniem pierwszych pozycji. To weryfikacja arytmetyki agregacji, nie niezależne powtórzenie kwantyzacji ani empiryczny pomiar zgodności liczby workerów. Surowy JSON (4 993 303 B), stdout/stderr, kontrola oraz tymczasowe źródła/binarka są zachowane prywatnie poza Gitem; SHA i zakresy w provenance.json. Wersje i wagi są przypięte; do dokładnego odtworzenia tego jednorazowego narzędzia potrzebne są zachowane prywatne źródła.

## Pozostałe bramki

Review danych P0 i delegowany wybór formatu poprzedzają implementację W4. Dokładna zgodność diagnostyki P0 i silnika, GPU dekwantyzacja i spójność trybów/kart, natywny HTTP/repliki, 44+34 i endpoint FP32, 900 z kontrolą f32, pamięć i wydajność są nadal niezmierzone. Q8-owa decyzja jakości nie dotyczy W4. M2 wymaga rzeczywistych danych W4; domyślne none i obowiązkowa walidacja GPU pozostają. Techniczny Ready nie upoważnia do wynajmu; przed S1 potrzebna jest konkretna prognoza i odpowiednia dodatkowa autoryzacja użytkownika. Etap pozostaje w toku; brak commitu etapu i przejścia do 05.
