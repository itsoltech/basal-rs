# Izolacja dwóch modeli i koszt obsługi wielu GPU

> **Dane JSON w archiwum.** Plik `measurements.json` z tego katalogu jest poza Gitem, w archiwum pomiarów
> `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod ścieżką `reports/multigpu-isolation-2026-10-11/<plik>`.
> Odwołania do nich w tekście dotyczą plików z archiwum. Pobieranie i sumy kontrolne:
> [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Archiwum zawiera też wersję tego pliku
> sprzed dopisania tej noty. Przeniesienie danych nie jest nowym pomiarem: liczby, statusy i rozstrzygnięcia
> raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

Na dwóch RTX 6000 Ada mini na GPU 1 zachowało zbliżone opóźnienia przy
obciążonym 4.5B na GPU 0: zmiana mediany p99 wyniosła −0,12%, +0,37%
i −0,86% dla trzech rodzajów ruchu. Porównanie starej i nowej binarki
na jednym GPU dało odpowiednio +1,40%, +0,61% i −0,23%. To obserwacje
czterech przebiegów każdego wariantu, bez wniosku o zerowym wpływie
drugiego modelu lub statystycznie potwierdzonym braku regresji.

## Zakres i warunki

2026-10-11, 2 × NVIDIA RTX 6000 Ada Generation, 48 GB (49140 MiB),
limit mocy 300 W, sterownik 580.126.09. CUDA: cuBLAS/cuBLASLt 12.9.1.4,
cudart 12.9.79; F16, wspólne tabele GEMM batch-invariant. CPU: Xeon
Platinum 8352Y, dostępne 26 CPU. Python 3.10.12. Klient i serwer działały
na tej samej maszynie. Zegary GPU nie były zablokowane.

Runner [run-isolation-cloud.py](../../tools/perf/run-isolation-cloud.py),
`--closed-requests 2000 --repeats 2`, pozostałe parametry domyślne.
Każda kampania: dwie sekwencje A/B/B/A, świeży serwer w każdym z ośmiu
przypadków, rozgrzewka 34 sekwencyjnych i 64 współbieżnych requestów mini.
Łącznie 16 przypadków i 48 ważnych faz. Dwie błędne pozycje rozgrzewki
(400 i 422) wykluczono z mierzonego ruchu; pozostały 32 różne requesty.

We wszystkich przypadkach wykorzystano identyczny porządek requestów:
2000 w zamkniętej pętli 16 klientów oraz 1487 i 3026 w otwartych fazach
Poissona 50 i 100 requestów/s przez 30 s, ze stałym seedem `20261011`.
Limit klienta: 256 requestów w toku; globalny admission: 4096. Otwarta
latency zaczyna się w zaplanowanym czasie napływu i obejmuje opóźnienie
generatora oraz kolejkę klienta. Requesty po zakończeniu napływu są
rozliczane do końca; limit drain wynosi 120 s. Przepustowość to wszystkie
poprawne odpowiedzi podzielone przez czas od początku fazy do ostatniej
odpowiedzi. Wartości poniżej są medianami czterech statystyk przebiegów,
a percentyle w przebiegu obliczono metodą nearest-rank.

## Mini bez tła i z obciążonym 4.5B

A: mini na GPU 1. B: mini na GPU 1 i 4.5B na GPU 0, pod jednym API.
Tło ma 16 klientów i dokumenty `long-4000`/`long-8000` z 1/5 pytaniami.
4000/8000 to nominalne rozmiary generatora; 4.5B raportuje 8338/16442
tokenów wejścia przy jednym pytaniu oraz 58322/115050 łącznie na siedem
gałęzi przy pięciu pytaniach. Tło startuje 10 s przed pierwszą fazą mini
i kończy po wszystkich jej odpowiedziach.

| Ruch | p50 A / B (ms) | p95 A / B (ms) | p99 A / B (ms) | Zmiana p99 | Kolejka p99 A / B (ms) | Odpowiedzi/s A / B |
|---|---:|---:|---:|---:|---:|---:|
| 16 klientów | 192,897 / 192,584 | 225,300 / 225,895 | 240,183 / 239,901 | −0,12% | 197,873 / 190,596 | 83,152 / 83,623 |
| 50/s | 19,766 / 19,459 | 71,951 / 70,356 | 105,136 / 105,525 | +0,37% | 48,099 / 49,361 | 49,566 / 49,566 |
| 100/s | 411,808 / 413,216 | 588,149 / 583,707 | 669,839 / 664,048 | −0,86% | 387,021 / 382,789 | 99,660 / 99,674 |

W każdej fazie B GPU 0 było rzeczywiście zajęte: średnie wykorzystanie
98,3–99,8%, średnio 15,998–15,999 klientów tła w toku, zero czasu bez
requestu tła. Zajętość pamięci po załadowaniu: mini 3646 MiB na GPU 1;
z tłem dodatkowo 9726 MiB na GPU 0. Szczyt pamięci GPU 1 podczas faz:
4040–4232 MiB zarówno A, jak i B.

W zamkniętej pętli mediana p50 kolejki wyniosła 15,916 / 30,371 ms,
przy nakładających się zakresach 15,677–81,500 / 18,313–68,719 ms.
Mediana p50 nagłówka czasu obliczeń wyniosła 170,119 / 136,875 ms,
zakresy 95,581–170,459 / 110,089–167,043 ms. Pierwszy przebieg A sam
wykazał najwyższe p50 kolejki spośród obu wariantów. Czasy i rozmiary
partii różniły się między przebiegami; nie ustalono przyczyny tych
różnic ani ich związku z tłem. Nagłówki opisują
pracę partii i oczekiwanie serwera, a nie sam czas kerneli. p99 końcowej
latency zamkniętej pętli miało zakresy 238,746–242,195 ms w A oraz
233,202–243,975 ms w B. Zbliżona końcowa latency nie dowodzi identycznego
harmonogramu ani izolacji każdej składowej czasu.

## Stara i nowa binarka na jednej karcie

A: wersja bazowa `87da786`. B: placement `ae15ad9`. Obie obsługują
wyłącznie mini na tym samym fizycznym GPU 1 pod `CUDA_VISIBLE_DEVICES=1`,
z identycznym YAML bez pola `device`. Porównanie obejmuje całą zmianę
silnika; nie izoluje kosztu pojedynczego locka wyboru kernela.

| Ruch | p50 A / B (ms) | p95 A / B (ms) | p99 A / B (ms) | Zmiana p99 | Kolejka p99 A / B (ms) | Odpowiedzi/s A / B |
|---|---:|---:|---:|---:|---:|---:|
| 16 klientów | 196,268 / 196,880 | 231,113 / 231,573 | 240,233 / 243,587 | +1,40% | 196,404 / 198,549 | 81,807 / 81,463 |
| 50/s | 20,039 / 20,092 | 72,493 / 72,259 | 106,016 / 106,666 | +0,61% | 48,938 / 49,465 | 49,566 / 49,566 |
| 100/s | 418,010 / 417,696 | 595,734 / 593,148 | 678,042 / 676,479 | −0,23% | 401,311 / 396,539 | 99,606 / 99,636 |

Przepustowość zamkniętej pętli zmieniła się o −0,42%. Zakresy jej p99:
236,600–247,001 ms w A i 237,584–247,110 ms w B. Mediana p99 opóźnienia
generatora przy 100/s wzrosła z 1,259 do 1,834 ms; przy 50/s wyniosła
0,557 / 0,561 ms. W izolacji wynosiła 0,558 / 0,373 ms przy 50/s
i 1,637 / 1,756 ms przy 100/s. Nie mierzono wykorzystania CPU.

## Poprawność i dowody

Wszystkie 104 208 mierzone odpowiedzi mini miały 200 i zgodną treść;
tak samo 927 odpowiedzi klientów tła. Rozgrzewki: 1536 odpowiedzi mini
200, 16 oczekiwanych 400, 16 oczekiwanych 422 i 16 odpowiedzi 4.5B 200.
Porównanie całej kanonicznej odpowiedzi pomijało wyłącznie
`usage.latency_ms`. Referencje pochodziły z pierwszej rozgrzewki danego
modelu; to porównanie starej/nowej binarki oraz różnych obciążeń, nie
nowy pomiar względem FP32 upstream. Eksporty dla tych samych binarek
są w [raporcie placementu](../multigpu-placement-2026-10-11/README.md).

Zero błędów transportu, 529, 5xx, utraconych requestów, anulowań przez
drain lub odpowiedzi po jego terminie. Każda faza miała poprawne metryki
przed i po, komplet nagłówków czasu oraz wymagane próbki GPU. Minimum
116 pełnych próbek na wymaganą kartę w fazie; maksymalna przerwa 0,207 s.
Wszystkie 16 serwerów zakończyło się kodem 0 i zwolniło pamięć GPU.
Runner: 292 PASS, 0 FAIL, 0 GAP, kod 0. Maszyna została usunięta po
pobraniu danych, a jej brak potwierdzono przez API.

Koordynator osobno sprawdził surowe statusy i skróty odpowiedzi względem
pełnych referencji, kompletność indeksów i zgodność harmonogramów,
percentyle latency/kolejki/obliczeń, metryki, pokrycie telemetrii oraz
mediany A/B. SHA runnera, helpera, binarek i wejść odpowiadają zapisowi
pomiaru. Skrót zmierzonego diffu Rust względem `87da786` to
`d7422df254bfa644484e61a8db6d9ac10b653de517faa9ada31b2f416b1eb899`.
Diff z `git diff --src-prefix=c/ --dst-prefix=w/ 87da786 ae15ad9 -- crates`
jest bajtowo równy zachowanemu diffowi z budowy binarki. Prefiksy diffu
wpływają na jego skrót. Binarka kandydata była zbudowana przed commitem,
więc jej tekst wersji zawiera bazowy SHA; identyfikuje ją SHA-256 binarki.

Odciski modeli, binarek, tabel, wejść i narzędzi oraz cztery wartości
każdej statystyki A/B są w `measurements.json`.
Surowe dane pozostają poza Gitem. Runner przeszedł trzy rundy review
Opus 5.5 high i GPT 6 Astra xhigh; ostatnia bez konkretnych uwag.
Dwa review dowodów i raportu przez te same modele zakończyły się
akceptacją po dwóch korektach opisu; dane pomiarowe pozostały bez zmian.
Sprawdzono Python compile/help/plan-only i diff check. Rust nie zmienił
się względem placementu; zakres jego kontroli jest w tamtym raporcie.

## Ograniczenia

Jeden typ stacji, dwa modele F16, określony zestaw requestów, cztery
obserwacje każdego wariantu, wspólny klient/CPU i automatyczne zegary.
Średnie zegary SM mini w fazach wynosiły 1429,7–2406,0 MHz, średnia moc
207,9–279,7 W. CPU, router i limit admission pozostają wspólne.
Nie badano przeciążenia globalnego limitu, innych kart, replik,
kwantyzacji ani podziału modelu. Wynik nie dowodzi zachowania na dwóch
RTX 4090, pełnej izolacji lub konkretnego przyspieszenia wielu GPU.
