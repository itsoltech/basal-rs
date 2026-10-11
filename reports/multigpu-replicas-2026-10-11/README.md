# Repliki jednego modelu na dwóch GPU

Dwie repliki mini na RTX 6000 Ada dały 2,06× i 2,07× przepustowość jednej
repliki przy 16 i 32 klientach. p99 zmalało odpowiednio z 246,822 do
140,742 ms i z 444,795 do 252,106 ms. Wszystkie mierzone odpowiedzi były
zgodne z referencją. To wynik tego ruchu i sprzętu, nie gwarancja skalowania.

Porównanie placementu i nowej binarki na jednym GPU z metrykami wykazało
spadek mediany przepustowości o 3,04% przy 16 klientach. p99 było niższe
o 2,25%, a fazy otwarte miały zbliżoną przepustowość. Przyczyna różnicy
zamkniętej pętli nie została ustalona; nie deklarujemy braku regresji.
Osobny [pomiar kontrolny](../multigpu-replicas-control-2026-10-11/README.md)
tych samych binarek z żądanym zegarem 1200 MHz dał +0,030% mediany
przepustowości i mieszane kierunki różnic w parach; zachowujemy oba wyniki.

## Warunki i metoda

2026-10-11, 2 × NVIDIA RTX 6000 Ada Generation, 49140 MiB na kartę,
limit mocy 300 W, sterownik 580.126.09, cuBLAS/cuBLASLt 12.9.1.4,
cudart 12.9.79. F16, wspólna tabela GEMM batch-invariant. CPU: Xeon
Platinum 8352Y, 26 dostępnych CPU. Python 3.10.12. Klient i serwer na
tej samej stacji, zegary GPU automatyczne.

[run-replicas-cloud.py](../../tools/perf/run-replicas-cloud.py): zmierzona
wersja R4 z jawnymi `--steps serve --long-ids 'long-4000-q1 long-4000-q5'
--inflight-delay 0.05 --repeats 2`. Eksporty, doctor i błędne konfiguracje
pochodzą z wcześniejszego przebiegu tej samej binarki i źródeł.

Każda kampania ma dwie sekwencje A/B/B/A, świeży serwer w każdym
przypadku, 34 requesty rozgrzewki sekwencyjnej i 64 współbieżnej.
32 wejścia System One z odpowiedzią 200 tworzą mierzony ruch;
`so-null-and-missing-instructions` (400) i `so-missing-type` (422)
pozostają w rozgrzewce, poza ruchem mierzonym. Wszystkie przypadki danej
kampanii mają te same wejścia i harmonogramy. Globalny admission 4096,
limit klienta 256 requestów w toku, drain 120 s.

Percentyle nearest-rank liczone osobno w przebiegu; tabela pokazuje
mediany czterech statystyk A i czterech B. Otwarta latency zaczyna się
w zaplanowanym czasie napływu, obejmuje opóźnienie generatora i kolejkę
klienta. Przepustowość obejmuje drain do ostatniej odpowiedzi. Nagłówki
kolejki i obliczeń opisują serwer i partię, nie sam czas kerneli.

## Jedna i dwie repliki

A: nowa binarka, `device: 0`. B: ta sama binarka,
`replica_devices: [0, 1]`, jedna nazwa modelu API. Metryki włączone.
48 ważnych faz w ośmiu przypadkach; 113 496 mierzone odpowiedzi 200.
Pętle zamknięte: 300/2000/3000 requestów przy 1/16/32 klientach.
Otwarte: Poisson przez 30 s przy 50/100/150 requestach/s, seed `20261012`,
odpowiednio 1542/2975/4370 napływów w każdym przypadku.

| Ruch | p50 A / B (ms) | p95 A / B (ms) | p99 A / B (ms) | Kolejka p99 A / B (ms) | Odpowiedzi/s A / B |
|---|---:|---:|---:|---:|---:|
| closed-c1 | 12.848 / 11.945 | 23.806 / 22.336 | 37.218 / 32.696 | 1.253 / 1.224 | 74.875 / 81.129 |
| closed-c16 | 192.842 / 91.174 | 232.568 / 128.221 | 246.822 / 140.742 | 192.825 / 105.501 | 83.018 / 170.938 |
| closed-c32 | 384.273 / 184.846 | 427.494 / 231.482 | 444.795 / 252.106 | 255.781 / 167.300 | 83.367 / 172.728 |
| open-50rps | 20.969 / 13.005 | 65.508 / 26.200 | 104.603 / 34.538 | 47.122 / 13.648 | 51.417 / 51.417 |
| open-100rps | 399.510 / 15.412 | 578.039 / 39.925 | 673.384 / 56.703 | 398.144 / 28.489 | 98.216 / 99.113 |
| open-150rps | 550.856 / 33.582 | 821.375 / 103.153 | 941.162 / 153.500 | 614.271 / 75.099 | 143.954 / 145.694 |

Przy jednym kliencie obie karty są używane naprzemiennie, bez równoczesnej
pracy nad tym samym requestem; +8,35% obserwowanej przepustowości nie
jest dowodem przyspieszenia pojedynczego requestu przez dwie karty.
W ruchu otwartym tempo napływu ogranicza przepustowość: przy 50/s jest
ona identyczna, przy 100/150/s różni się o +0,91%/+1,21%. Zysk dotyczy
głównie latency: p99 przy 100/s wyniosło 673,384 / 56,703 ms.
Wyniki zamknięte i otwarte mają inne partie; nie wyznaczają jednego
uniwersalnego limitu przepustowości.

Liczniki routingu w każdej fazie sumują się do liczby wysłanych requestów,
a każda replika otrzymuje ruch. Przy 16 klientach podział B wynosił
973–990 / 1010–1027 z 2000 requestów, przy 32 klientach
1457–1492 / 1508–1543 z 3000. Średnie wykorzystanie obu kart przy tych
fazach B: 89,0–91,1%. Po każdej fazie rezerwacje wracały do zera.
Po załadowaniu: 3646 MiB na każdej użytej karcie.

CPU serwera (sekundy CPU na sekundę ścienną) przy 16 klientach wyniósł
1,0145 / 2,021, przy 32 klientach 1,014 / 2,0255. To pomiar procesu
między odczytami `/proc`, nie rozbicie kosztów routera. Mediana p99
opóźnienia generatora przy 100/s: 1,399 / 0,549 ms, przy 150/s:
3,123 / 0,500 ms. Klient i serwer współdzielą CPU.

## Placement i repliki na jednym GPU

[run-isolation-cloud.py](../../tools/perf/run-isolation-cloud.py),
`--campaigns overhead --repeats 2`, pozostałe parametry domyślne.
A: placement `ae15ad9`. B: nowa binarka replik. Mini na fizycznym GPU 1,
`CUDA_VISIBLE_DEVICES=1`, identyczny YAML bez `device`, metryki włączone.
24 ważne fazy, 52 104 mierzone odpowiedzi 200. Zamknięte: 2000 requestów,
16 klientów; otwarte Poisson 50/100/s przez 30 s, seed `20261011`,
1487/3026 napływów. Porównanie obejmuje całą zmianę silnika.

| Ruch | p50 A / B (ms) | p95 A / B (ms) | p99 A / B (ms) | Kolejka p99 A / B (ms) | Odpowiedzi/s A / B |
|---|---:|---:|---:|---:|---:|
| closed-c16 | 187.291 / 191.987 | 223.665 / 224.974 | 237.214 / 231.882 | 160.600 / 190.998 | 86.120 / 83.504 |
| open-50rps | 19.378 / 19.381 | 70.154 / 70.223 | 104.599 / 105.168 | 49.062 / 48.002 | 49.566 / 49.566 |
| open-100rps | 396.007 / 393.663 | 572.411 / 565.673 | 653.974 / 643.298 | 375.704 / 371.370 | 99.793 / 99.816 |

Zamknięta przepustowość A: 83,080–87,555/s, B: 83,163–84,436/s;
mediany 86,121 / 83,504/s (−3,04%). p50 latency wzrosło o 2,51%, p99
zmalało o 2,25%; p99 otwarte zmieniło się o +0,54% i −1,63%.
Zakresy nakładają się, ale to nie usuwa obserwowanego spadku mediany.

W zamkniętej pętli zmieniła się organizacja partii: średni rozmiar
partii ważony requestami A 8,405–11,678, B 11,185–13,474; mediana p50
kolejki 74,441 / 29,073 ms, p50 czasu obliczeń 101,255 / 154,100 ms.
Średnie zegary SM A 1551,8–1679,9 MHz, B 1531,4–1559,6 MHz.
Ostatni A miał przepustowość i partie podobne do B. To współwystępujące
obserwacje; nie ustalono, czy źródłem różnicy jest instrumentacja,
harmonogram, zegary czy inne czynniki. Ten runner nie mierzył CPU.
Trybu z wyłączonymi metrykami nie mierzono.

Rozbicie surowych liczników: 251–252 partie na przypadek, mediana sumy
czasu partii 22,327 / 23,046 s. Rozpiętość mierzonych requestów to
`max(t_done) − min(t_start)`, nie pełny czas t0/t1 fazy runnera.
Mediana różnicy między tą rozpiętością a sumą czasu partii to w obu
wariantach około 0,898 s. Te pomiary obejmują
admission i `run_plans`, nie sam czas GPU ani profil CPU. Różnica czasu
partii współwystępuje z różnymi rozkładami ich wielkości; pozostaje otwarte,
czy kod wpływa na częstość takich rozkładów. Podobny czas poza partiami
nie dowodzi braku narzutu. Dane i metoda rozbicia są w `measurements.json`.
W osobnym pomiarze kontrolnym tych samych binarek nie odtworzono
jednokierunkowego spadku przepustowości. Nie ustala to przyczyny
pierwotnego wyniku ani nie zastępuje obserwacji przy zegarach automatycznych.

## Poprawność i pochodzenie danych

24 eksporty: mini i 4.5B, single/tree/tree bez prefix-cache, placement
oraz kandydat na obu kartach; po 44 wpisy bench i 34 System One.
Dla każdego modelu wynik dwóch plików jest bitowo identyczny między
wszystkimi wariantami i trybami; 18 porównań `basal compare` bez różnic.
To zgodność z poprzednią binarką, nie nowy pomiar FP32 upstream.

HTTP: referencja 93 wejść ze starej binarki, pełna kanoniczna odpowiedź
z pominięciem tylko `usage.latency_ms`. Sprawdzono jedną replikę jako
`device`, listę jednoelementową, obie repliki z 16 klientami i czterema
powtórzeniami, `CUDA_VISIBLE_DEVICES=1` oraz odwrócenie `1,0`.
1066 kompletnych odpowiedzi miało status i body zgodne z referencją.
Osobno 40 oczekiwanych 529 miało body TypeSafe i `Retry-After: 1`.
Anulowania: 25 odpowiedzi przed rozłączeniem, 23 zaplanowane rozłączenia;
rezerwacje obejmowały nadal pracę GPU i wróciły do zera. Po SIGTERM
wszystkie 16 requestów dostało zgodne pełne odpowiedzi, a przy wysłaniu
sygnału obie repliki miały rezerwacje. Worker panic, utrata GPU i
częściowy start nie były prowokowane. Grupa różnych kart również nie
była mierzona; te zachowania obejmuje tylko review statyczne.

Pierwszy przebieg przerwano: mini zwraca 422 dla nominalnych
`long-8000`/`long-16000` już w referencji, więc były złymi wejściami
kontroli wymagających 200. Podsumowanie tego przebiegu: incomplete,
198 PASS, 3 FAIL, 22 GAP; jego wydajność wykluczono. Zachowano wyłącznie
zakończone doctor, 10 odrzuconych konfiguracji i eksporty. Drugi przebieg
użył jawnie `long-4000-q1/q5` (mini raportuje 8338/58322 input_tokens,
odpowiednio 1/7 gałęzi); wcześniejsze etykiety są nominalne.

Drugi przebieg: complete, 401 PASS, 0 FAIL, 1 GAP (nieprowokowane awarie).
Pomiar narzutu: complete, 134 PASS, 0 FAIL, 0 GAP. Wszystkie mierzone
165 600 odpowiedzi 200 były zgodne; rozgrzewki obu kampanii: 1536 × 200,
16 × 400, 16 × 422. Nie było błędów transportu, 529 ani 5xx w fazach
wydajności. Wszystkie 16 serwerów tych faz zakończyło się kodem 0 i
zwolniło VRAM. Pobrano dane, zweryfikowano SHA-256 wszystkich 553 plików,
usunięto instancję i potwierdzono jej brak w API.

Niezależny audyt sprawdził surowe skróty odpowiedzi względem pełnych
referencji, indeksy i harmonogramy, formuły czasów, percentyle,
mediany, metryki routingu/partii/rezerwacji, obecność danych CPU oraz
pokrycie telemetrii i próbki GPU. Nie odtworzono odczytów CPU z dwóch
punktów `/proc`; zachowano ich wynik w danych. SHA binarek, 64 plików
źródeł, wejść i helperów są powiązane z pomiarem.

Zmierzonego runnera R4 identyfikuje `99cddb4c…`; końcowy R5 to
`cb22ac7a…`. Różnica to docstring i domyślne `long_ids`/`inflight_delay`.
Przy jawnych flagach tego pomiaru wszystkie argumenty są identyczne;
struktura kodu po pominięciu tych trzech fragmentów jest identyczna.
Końcowy R5 sam nie był wykonany na GPU. Rust oraz binarki nie zmieniły
się przy tej korekcie. Diff Rust względem `6ef86665`, z prefiksami c/w,
ma SHA `959a0c8d…`; binarka bazowa ma SHA `f8528cc9…`, kandydat
`289a2eac…`. Wersja tekstowa wskazuje HEAD z chwili budowy przed
commitem, dlatego binarki identyfikuje pełny SHA-256 w
[measurements.json](measurements.json).

Kod i runner przeszły review Opus 5.5 high i GPT 6 Astra xhigh;
ostatnie R5 bez konkretnych uwag. Kontrole: fmt, Clippy, Rustdoc,
kontrola zależności i build CUDA; runtime Metal/Intel nie mierzono.
Po dodatkowej kontroli nie wykazano regresji wymagającej poprawki kodu.
Wpływ kodu na częstość rozkładów partii przy zegarach automatycznych
pozostaje nieustalony; nie deklarujemy braku narzutu.

## Ograniczenia

Jeden typ sprzętu, mini dla serwera i wydajności, dwa modele dla eksportów,
cztery obserwacje wariantu, automatyczne zegary i wspólny CPU klienta.
Nie mierzono max, heterogenicznych grup, faktycznych awarii, więcej niż
dwóch replik, kwantyzacji ani podziału modelu. Brak dowodu dla RTX 4090
lub HA po fatalnym błędzie CUDA. CPU, router i admission pozostają wspólne.
