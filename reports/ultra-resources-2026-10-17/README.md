# Ultra: zasoby API pod obciążeniem

Zmierzone 2026-10-17 UTC. basal-rs obsłużył o **12–23% więcej krótkich żądań/s**
i **27–34% więcej żądań/s w ruchu mieszanym** niż upstream BF16. Jednocześnie
zużywał o **35–40% mniej RAM PSS**, **6–14% mniej VRAM procesu** oraz
**11–26% mniej czasu CPU na żądanie**. Oba serwery zajmowały średnio około
jednego rdzenia CPU podczas obciążenia. Oszczędność dotyczy pracy na ukończone
żądanie, a nie wielokrotnie mniejszej liczby jednocześnie zajętych rdzeni.

To wynik jednego modelu Ultra i tekstowych wejść. Ruch mieszany oznacza krótkie
zadania przeplatane długimi dokumentami; obrazy i dźwięk były wyłączone.
Nie rozciągamy wyników na pozostałe modele rodziny ani ruch multimodalny.

**Status audytu całości: FAIL.** Wszystkie 48 okien obciążenia mają kompletną
telemetrię i poprawne odpowiedzi HTTP. Ścisła kontrola bitowa wykryła jednak
80 odpowiedzi basal-rs z innymi prawdopodobieństwami przy C128, bez zmiany
odpowiedzi funkcjonalnych. Trzy starty zakończyły się sygnałem błędu po zapisaniu
ostatniej fazy; zabrakło ich końcowej bezczynności C128. To nie są błędy HTTP
ani niekompletne fazy obciążenia. Szczegóły jakości i GAP pozostają poniżej.

## Cel i wersje

Porównanie niezmienionej binarki R15 basal-rs z upstream 2.1 BF16 eager na tej
samej karcie H100 80 GB. Model `Remek/basal-omni-1.0-ultra`, rewizja
`b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4`; upstream commit
`62542d31e8eb69000839c01e8f9b45da1fb80e49`. SHA-256 binarki:
`f23f060cf61ebf5d9d39c4f38835a6152b3823179e102152db2dfe17bc5af8b9`.

## Metoda

Dwa bloki ruchu: 32 różne krótkie żądania z zadaniami 2.1 oraz ruch mieszany
z dwoma dokumentami po 8192 tokeny na każde 32 krótkie żądania. Każdy blok
uruchamia osobne serwery w kolejności basal-rs, upstream, upstream, basal-rs.
Każdy serwer przechodzi pełną rozgrzewkę i poziomy C1, C2, C8, C16, C32, C128.
Zachowujemy dwa powtórzenia po rozgrzewce, bez odrzucania całego pierwszego
startu serwera. To krótszy protokół niż standardowe serie wydajności z osobnym
odrzucanym powtórzeniem; podajemy też rozrzut obu obserwacji.
C oznacza liczbę klientów wysyłających kolejne żądanie po odebraniu odpowiedzi.
Faza trwa przynajmniej 60 sekund, cztery pełne przejścia korpusu i cztery
żądania na klienta. Obejmuje opróżnienie wszystkich rozpoczętych żądań.
Żaden błąd HTTP nie stanowi poprawnie obsłużonego żądania.
Żądania idą przez lokalny HTTP, bez TLS i proxy, każde z nowym połączeniem TCP.
Latencja obejmuje serializację JSON i odczyt całej odpowiedzi; jej parsowanie
odbywa się po zatrzymaniu licznika.

Próbkowanie co sekundę obejmuje wyłącznie proces serwera i jego potomków.
Generator ruchu, sampler i proces pilnujący sesji są poza licznikami serwera;
korzystają jednak z tej samej maszyny. CPU mierzymy w czasie pracy rdzeni,
podając średnią liczbę zajętych rdzeni i milisekundy CPU na żądanie.
RSS może wielokrotnie liczyć wspólne strony; PSS dzieli je proporcjonalnie,
a USS obejmuje strony prywatne. VRAM procesu i zajętość całej karty są osobne.
Maksima RAM/VRAM są próbkowane, więc mogą pomijać krótkie skoki alokacji.

Oddzielamy ładowanie, bezczynność z modelem, bezczynność po rozgrzewce,
fazy obciążenia i 15 sekund bezczynności po każdej fazie. Szczyt RSS procesu
od startu pokazuje koszt ładowania; nie jest szczytem konkretnej fazy.
RSS/PSS obejmują też rezydentne strony mapowanych plików. Szczyt nie określa
minimalnej pamięci RAM, przy której silnik może działać.
Alokatory zachowują pamięć między rosnącymi poziomami C. To pokazuje historię
pracy serwera; nie wyznacza minimalnej pamięci świeżego procesu dla każdego C.
Bezczynność z załadowanym modelem obejmuje runtime i wagi, więc nie oznacza
samego kosztu warstwy HTTP.

basal-rs: auto F16 z odczytem FP32, FIFO, batch 16384 tokeny. Upstream:
BF16 eager, batch 4096 tokenów. Oba: orders=1, wyłączone cache stanu i prefiksu,
środowisko wątków CPU=8. To nie jest twardy limit ani przypisanie ośmiu rdzeni.
Limity przyjmowania basal-rs są wyższe od C128. OOM, 529, błędy odpowiedzi
i luki telemetrii pozostają w danych i nie tworzą pozornej przewagi.

## Sprzęt i środowisko

NVIDIA H100 PCIe, 81559 MiB VRAM, limit mocy 350 W, sterownik 580.126.20,
compute capability 9.0. CPU AMD EPYC 9334, w maszynie pomiarowej 24 logiczne CPU
widoczne i dostępne w affinity; RAM 247403824 KiB, około 236 GiB.
Ubuntu 24.04.4 LTS, Python 3.12.12, PyTorch 2.11.0+cu128,
Transformers 5.17.0; binarka CUDA 12.9.1 i cuBLASLt 120901.
Nie zmieniano zegarów, limitu mocy ani chłodzenia. Maksymalna obserwowana
temperatura podczas faz obciążenia wyniosła 78°C. Pełne zależności upstream
są w `upstream-requirements.txt` archiwum.

## Jak czytać wyniki

`basal-rs` to niezmieniona binarka R15, a `upstream BF16` to przypięty serwer
upstream w trybie eager. C oznacza liczbę klientów, a nie wielkość batcha modelu.
Większe C może zwiększać QPS, ale zwiększa też kolejkę i czas odpowiedzi.
Na przykład krótkie basal-rs C1 → C128 daje 1,920 → 2,262 żądania/s,
lecz p50 rośnie z 425 ms do około 50,8 s. C128 jest próbą nasycenia, nie
proponowanym ustawieniem latencji produkcyjnej.

GiB oznacza 2^30 bajtów. CPU w rdzeniach to czas pracy CPU podzielony przez
czas fazy. RAM i VRAM poniżej są średnimi ważonymi czasem z próbek fazy,
a następnie medianą dwóch powtórzeń. QPS jest liczbą poprawnych żądań
podzieloną przez cały czas fazy, razem z opróżnieniem kolejki. p50/p95 obejmują
czekanie w kolejce. Nie należy traktować odwrotności p50 jako QPS.

## Krótkie zadania 2.1

Wartości to mediany dwóch powtórzeń. CPU i pamięć są średnimi w fazie.

| Klienci | Silnik | CPU [rdzenie] | CPU [ms/żądanie] | RAM PSS [GiB] | VRAM procesu [GiB] |
|---:|---|---:|---:|---:|---:|
| 1 | basal-rs | 1.00 | 520.7 | 1.128 | 60.961 |
| 1 | upstream BF16 | 1.00 | 629.6 | 1.752 | 65.105 |
| 2 | basal-rs | 1.00 | 523.6 | 1.128 | 60.961 |
| 2 | upstream BF16 | 1.00 | 645.8 | 1.755 | 65.105 |
| 8 | basal-rs | 1.00 | 484.1 | 1.129 | 62.197 |
| 8 | upstream BF16 | 1.00 | 596.1 | 1.758 | 67.940 |
| 16 | basal-rs | 1.00 | 463.7 | 1.130 | 62.508 |
| 16 | upstream BF16 | 1.00 | 559.5 | 1.760 | 68.094 |
| 32 | basal-rs | 1.00 | 454.4 | 1.133 | 62.523 |
| 32 | upstream BF16 | 1.01 | 523.7 | 1.761 | 68.094 |
| 128 | basal-rs | 1.00 | 442.4 | 1.143 | 63.008 |
| 128 | upstream BF16 | 1.01 | 498.0 | 1.770 | 68.094 |

| Klienci | Silnik | Żądania/s | p50 HTTP [ms] | p95 HTTP [ms] |
|---:|---|---:|---:|---:|
| 1 | basal-rs | 1.920 | 425 | 990 |
| 1 | upstream BF16 | 1.596 | 475 | 1307 |
| 2 | basal-rs | 1.913 | 907 | 1544 |
| 2 | upstream BF16 | 1.557 | 1330 | 2258 |
| 8 | basal-rs | 2.068 | 3797 | 4172 |
| 8 | upstream BF16 | 1.686 | 5191 | 5761 |
| 16 | basal-rs | 2.159 | 7282 | 7662 |
| 16 | upstream BF16 | 1.797 | 9419 | 10144 |
| 32 | basal-rs | 2.203 | 14504 | 14687 |
| 32 | upstream BF16 | 1.921 | 17121 | 18361 |
| 128 | basal-rs | 2.262 | 50755 | 72488 |
| 128 | upstream BF16 | 2.021 | 69345 | 73301 |

## Krótkie zadania i dokumenty 8192-tokenowe

Wartości to mediany dwóch powtórzeń. CPU i pamięć są średnimi w fazie.

| Klienci | Silnik | CPU [rdzenie] | CPU [ms/żądanie] | RAM PSS [GiB] | VRAM procesu [GiB] |
|---:|---|---:|---:|---:|---:|
| 1 | basal-rs | 1.00 | 755.9 | 1.052 | 64.711 |
| 1 | upstream BF16 | 1.00 | 996.6 | 1.757 | 75.105 |
| 2 | basal-rs | 1.00 | 759.0 | 1.053 | 64.711 |
| 2 | upstream BF16 | 1.00 | 1012.8 | 1.760 | 75.105 |
| 8 | basal-rs | 1.00 | 722.1 | 1.054 | 64.711 |
| 8 | upstream BF16 | 1.00 | 969.5 | 1.764 | 75.105 |
| 16 | basal-rs | 1.00 | 706.5 | 1.056 | 64.711 |
| 16 | upstream BF16 | 1.00 | 929.4 | 1.766 | 75.105 |
| 32 | basal-rs | 1.00 | 696.8 | 1.059 | 64.711 |
| 32 | upstream BF16 | 1.00 | 895.4 | 1.767 | 75.105 |
| 128 | basal-rs | 1.00 | 683.4 | 1.078 | 64.711 |
| 128 | upstream BF16 | 1.00 | 873.9 | 1.776 | 75.105 |

| Klienci | Silnik | Żądania/s | p50 HTTP [ms] | p95 HTTP [ms] |
|---:|---|---:|---:|---:|
| 1 | basal-rs | 1.323 | 426 | 4487 |
| 1 | upstream BF16 | 1.007 | 477 | 6823 |
| 2 | basal-rs | 1.319 | 907 | 5478 |
| 2 | upstream BF16 | 0.991 | 1338 | 7731 |
| 8 | basal-rs | 1.386 | 4102 | 8377 |
| 8 | upstream BF16 | 1.036 | 5386 | 12080 |
| 16 | basal-rs | 1.416 | 11272 | 13746 |
| 16 | upstream BF16 | 1.080 | 15687 | 21806 |
| 32 | basal-rs | 1.436 | 18471 | 31069 |
| 32 | upstream BF16 | 1.121 | 29697 | 37370 |
| 128 | basal-rs | 1.464 | 75292 | 121466 |
| 128 | upstream BF16 | 1.150 | 121783 | 133047 |

## Obserwowane ładowanie modelu

Wagi były już pobrane i sprawdzone; cache systemu plików i bibliotek nie był czyszczony.

| Silnik | Liczba startów | Model gotowy [s] | CPU do gotowości [s pracy rdzeni] | Szczyt RSS do gotowości [GiB] |
|---|---:|---:|---:|---:|
| basal-rs | 4 | 74.80 | 74.62 | 5.604 |
| upstream BF16 | 4 | 19.52 | 18.45 | 59.335 |

Szczyt RSS obejmuje rezydentne strony mapowanych plików i nie wyznacza minimalnego limitu RAM.

## Rozrzut, maksima i pamięć po opróżnieniu

Dwa powtórzenia nie dają przedziału ufności. Poniższy zakres to minimum–maksimum
ich rzeczywistych obserwacji; wszystkie pozostałe metryki z zakresem są
w `audit.json → summary[].metrics`. Szczególnie RAM krótkiego basal-rs różni się
między startami, około 1,04–1,23 GiB PSS; pojedyncza mediana nie jest stałym
kosztem serwera.

| Ruch | C | QPS basal-rs min–max | QPS upstream min–max | PSS basal-rs min–max [GiB] | PSS upstream min–max [GiB] |
|---|---:|---:|---:|---:|---:|
| krótkie | 1 | 1.914–1.925 | 1.595–1.597 | 1.035–1.220 | 1.752–1.753 |
| krótkie | 2 | 1.912–1.913 | 1.536–1.577 | 1.035–1.221 | 1.754–1.755 |
| krótkie | 8 | 2.067–2.069 | 1.686–1.687 | 1.037–1.221 | 1.758–1.758 |
| krótkie | 16 | 2.158–2.159 | 1.793–1.800 | 1.039–1.222 | 1.760–1.760 |
| krótkie | 32 | 2.198–2.208 | 1.918–1.923 | 1.042–1.223 | 1.761–1.762 |
| krótkie | 128 | 2.260–2.265 | 2.020–2.022 | 1.059–1.228 | 1.769–1.770 |
| mieszany | 1 | 1.316–1.330 | 1.002–1.012 | 1.052–1.052 | 1.753–1.761 |
| mieszany | 2 | 1.314–1.324 | 0.991–0.991 | 1.052–1.053 | 1.756–1.764 |
| mieszany | 8 | 1.382–1.390 | 1.034–1.037 | 1.054–1.055 | 1.759–1.769 |
| mieszany | 16 | 1.411–1.421 | 1.078–1.082 | 1.056–1.057 | 1.761–1.771 |
| mieszany | 32 | 1.433–1.439 | 1.121–1.122 | 1.058–1.060 | 1.762–1.772 |
| mieszany | 128 | 1.460–1.468 | 1.149–1.151 | 1.077–1.078 | 1.771–1.780 |

Maksima poniżej pochodzą ze wszystkich faz obciążenia danego silnika i ruchu;
nie z ładowania. CPU to maksymalna obserwowana prędkość przyrostu licznika
w oknie co najmniej pół sekundy, a pamięć to największa próbka.

| Ruch | Silnik | CPU max [rdzenie] | RSS max [GiB] | PSS max [GiB] | USS max [GiB] | VRAM całej karty max [GiB] |
|---|---|---:|---:|---:|---:|---:|
| krótkie | basal-rs | 1.040 | 1.233 | 1.229 | 1.228 | 63.032 |
| krótkie | upstream BF16 | 1.212 | 1.775 | 1.771 | 1.769 | 68.103 |
| mieszany | basal-rs | 1.040 | 1.085 | 1.081 | 1.080 | 64.720 |
| mieszany | upstream BF16 | 1.496 | 1.785 | 1.781 | 1.779 | 75.114 |

Bezczynność: mediana średnich z dwóch startów, chyba że podano `n=1`.
Pamięć alokatora po rozgrzewce/obciążeniu pozostaje zajęta. Dane pozostałych
C są w `audit.json → idle`; brakującego C128 nie zastępujemy poprzednią fazą.

| Ruch | Silnik | Etap | n | CPU [rdzenie] | PSS [GiB] | VRAM procesu [GiB] |
|---|---|---|---:|---:|---:|---:|
| krótkie | basal-rs | po załadowaniu | 2 | 0.0003 | 0.912 | 60.287 |
| krótkie | basal-rs | po rozgrzewce | 2 | 0.0003 | 1.128 | 60.961 |
| krótkie | basal-rs | po C32 | 2 | 0.0007 | 1.133 | 62.523 |
| krótkie | basal-rs | po C128 | 1 | 0.0000 | 1.061 | 63.023 |
| krótkie | upstream BF16 | po załadowaniu | 2 | 0.0010 | 1.095 | 59.023 |
| krótkie | upstream BF16 | po rozgrzewce | 2 | 0.0010 | 1.752 | 65.105 |
| krótkie | upstream BF16 | po C32 | 2 | 0.0010 | 1.762 | 68.094 |
| krótkie | upstream BF16 | po C128 | 2 | 0.0013 | 1.770 | 68.094 |
| mieszany | basal-rs | po załadowaniu | 2 | 0.0007 | 0.902 | 60.287 |
| mieszany | basal-rs | po rozgrzewce | 2 | 0.0003 | 1.052 | 64.711 |
| mieszany | basal-rs | po C32 | 2 | 0.0007 | 1.059 | 64.711 |
| mieszany | basal-rs | po C128 | 0 | GAP | GAP | GAP |
| mieszany | upstream BF16 | po załadowaniu | 2 | 0.0010 | 1.099 | 59.023 |
| mieszany | upstream BF16 | po rozgrzewce | 2 | 0.0010 | 1.757 | 75.105 |
| mieszany | upstream BF16 | po C32 | 2 | 0.0013 | 1.767 | 75.105 |
| mieszany | upstream BF16 | po C128 | 2 | 0.0007 | 1.776 | 75.105 |

## Jakość i statusy FAIL/GAP

Zapisano 10056 poprawnych odpowiedzi HTTP, w tym 264 w rozgrzewce;
9792 odpowiedzi pochodzą z 48 faz obciążenia. Brak HTTP 529, innych błędów HTTP,
OOM i niepoprawnych formatów odpowiedzi w tej kampanii. Każdy z 24 punktów
silnik × ruch × C ma dwa kompletne powtórzenia. Niezależne odtworzenie potwierdza
48/48 okien zasobów, wejścia, rzeczywiste C oraz pełne opróżnienie żądań.

Kontrola odpowiedzi basal-rs względem zachowanej referencji tej samej binarki:

| Faza C128 | Inne bity odpowiedzi / liczba żądań | Największa zmiana prawdopodobieństwa |
|---|---:|---:|
| krótkie r1 | 35/512 | 8,35657e-5 |
| krótkie r2 | 0/512 | 0 |
| mieszany r1 | 30/544 | 1,31130e-6 |
| mieszany r2 | 15/544 | 3,24249e-5 |

Łącznie 80 z 5108 odpowiedzi basal-rs (wliczając rozgrzewkę) różni się bitowo.
Żadna nie zmieniła decyzji, tekstu, skrótu, spanów ani danych wynikowych.
Pozostałe poziomy C były bitowo zgodne z zachowaną referencją. Także upstream
zmienia prawdopodobieństwa między poziomami, bez zmian odpowiedzi funkcjonalnych
wewnątrz jego własnych pomiarów. To nie oznacza zgodności upstream z FP32.

Surowy audyt ma 166 wpisów: 80 `native_answer_changed`, 80
`native_reference_mismatch` opisujących te same 80 odpowiedzi, trzy błędy
końcowego guardu oraz trzy `idle_resource_gap`. Guard pozostaje restrykcyjny;
nie zamieniamy jego FAIL na PASS na podstawie małej różnicy liczb. Zatrzymał
starty `tasks-candidate-r1`, `mixed-candidate-r1` i `mixed-candidate-r2` po
zapisaniu pełnego C128, przed planowanymi 15 sekundami końcowej bezczynności.
Ich obciążenie ma komplet danych, ich bezczynność C128 ma GAP.

Dodatkowa referencja upstream FP32 objęła 14 unikalnych wejść, których odpowiedzi
basal-rs miały powyższą zmienność. Wagi FP32 przechowywano w RAM i przesyłano
warstwy na GPU; PyTorch `highest`, TF32 wyłączony. To osobny przebieg jakościowy,
wyłączony z wszystkich metryk zasobów, ładowania i czasu HTTP.
14/14 odpowiedzi referencyjnych ukończono i zweryfikowano.

basal-rs zachował odpowiedź funkcjonalną FP32 dla 14/14 wejść we wszystkich
sprawdzonych obserwacjach; upstream BF16 dla 13/14. W `distinct-task-030`
BF16 zachował segmenty skrótu `[1,2,3,4]`, FP32 i basal-rs `[0,1,2,3]`:
zmienił wynikowy tekst, liczbę znaków i współczynnik skrótu. Nie jest to tylko
różnica raportowanej pewności.

Dla porównywalnej ekspozycji C128 (224 odpowiedzi na każdy start, te same
14 wejść po 16 razy), średni bezwzględny błąd pól prawdopodobieństw względem
FP32 wyniósł około **0,000248 basal-rs vs 0,000866 upstream BF16**.
Maksymalny błąd to **0,0132794 vs 0,0339737**. Agregacja obejmuje także
`set_confidence`; zachowuje literalne `data`/`value` jako dane odpowiedzi.
Dokładne liczby, ścieżki i zakresy są w `quality.json` i źródle analizatora.

Jest to celowana kontrola zmiennych wejść, nie niezależny benchmark jakości ani
oszacowanie accuracy całej rodziny. Nie sprawdza ponownie FP32 dla pozostałych
wejść, długich dokumentów ani mediów. Wcześniejsze pełniejsze kontrole są
w [raporcie R15](../ultra-attention-2026-10-17/README.md).

## Decyzje i dalsze usprawnienia

Przewaga gorącego API jest potwierdzona dla tego korpusu na H100. Ruch mieszany
przy C128 potrzebował średnio około 64,71 GiB VRAM procesu basal-rs wobec
75,11 GiB upstream; oszczędność około 10,39 GiB. Mniejsza pamięć nie oznacza,
że zmierzono minimalny rozmiar GPU lub RAM. Nie wykonywano prób z limitem RAM,
cgroup ani kontrolowanym page cache. Nie wyznaczono kosztu samej warstwy HTTP,
bo mierzone procesy obejmują model, runtime i wagi.

Najwyraźniejsza przegrana to start modelu: mediana 74,80 s basal-rs wobec
19,52 s upstream, około 3,8× dłużej. Zakres czterech startów wynosił
74,05–75,57 s oraz 19,5155–19,5175 s. To nie jest test zimnego dysku;
cache systemowy i bibliotek nie był czyszczony.

Dodatkowy obserwator ostatniego startu basal-rs wykazał 11,82 s CPU użytkownika
i 63,20 s CPU jądra, około 84% czasu CPU w jądrze, około 30,2 mln minor faults,
zero major faults i zero `read_bytes` pomimo około 61,4 GB `rchar`.
Sugeruje to koszt przygotowania buforów i kopiowania wag z cache. Ten późno
włączony obserwator nie obejmuje wszystkich startów i nie stanowi profilu stosów.
Kod loadera tworzy świeży bufor, tensor CPU i transfer GPU dla kolejnych wag.
Konwersja BF16→F16 już odbywa się na GPU; nie przypisujemy opóźnienia rzekomej
konwersji CPU. Następny konkretny krok to profilowanie loadera i pomiar
współdzielenia/reużycia buforów przy zachowaniu bitowej zgodności.
Nie zmieniano loadera ani obliczeń silnika w tej kampanii.

C128 daje niewielki dodatkowy QPS kosztem dużej kolejki. Dobór C do produkcji
wymaga własnego limitu p95 i rzeczywistego rozkładu wejść. Poprzedni raport
zachowuje regresję p95 długich żądań C32 w innym przebiegu; obecna łączna p95
ruchu mieszanego nie unieważnia tamtej obserwacji. Dwa powtórzenia, lokalny HTTP,
wyłączone media i brak limitowania RAM/CPU ograniczają zakres wniosków.

## Weryfikacja i dane

Runnery: `tools/perf/omni-resources.py`, `api_resources.py` i niezależne
przeliczenie `summarize-omni-resources.py`. Źródła pomiaru są zamrożone
w archiwum. Ponowne przeliczenie publicznej projekcji dało identyczny JSON
audytu i analizy startu jak dane prywatne. Anonimizacja identyfikatorów procesów
i ścieżek nie zmieniła liczników, odpowiedzi ani obliczonych wyników.

Samokontrola objęła granice procesów, PID reuse, drenaż kolejki, timeouty,
cleanup, niewliczanie błędów do QPS, kompletność faz i prywatność publikacji.
Nie dodawano frameworków testowych ani bramek CI. Ta kampania nie zmienia
kodu Rust ani numeryki; wcześniejsze kompilacje/linty R15 są opisane
w jego raporcie. Źródła Python sprawdzono kompilatorem składni,
a działanie narzędzi przez tę rzeczywistą sesję i odtworzenie jej wyników.

Maszynę pomiarową usunięto po zachowaniu i sprawdzeniu SHA-256 wszystkich
64 plików wynikowych sesji. Prywatne surowe dane pozostają lokalnie.

Archiwum opublikowano i zweryfikowano pobraniem z autoryzacją oraz anonimowo:
SHA-256 archiwum i wszystkie 74/74 sumy plików były zgodne.
W archiwum ścieżka bazowa danych to `reports/ultra-resources-2026-10-17/`.
Instrukcje odtworzenia są w `REPRODUCE.md` obok danych.

| Pole | Wartość |
|---|---|
| URI | `hf://buckets/itsoltech/basal-rs/measurements/ultra-resources-2026-10-17/47926913c632019d9d05c5461025c397ce45fb79ec5aec29cdff83d139c2cd02/measurement-data.tar.gz` |
| SHA-256 | `47926913c632019d9d05c5461025c397ce45fb79ec5aec29cdff83d139c2cd02` |
| Rozmiar gzip | 3729273 B |
| Pliki | 74 pliki danych/źródeł/raportów oraz `SHA256SUMS` |

Pobierz do nowego katalogu, sprawdź sumę przed rozpakowaniem, potem sumy plików:

```bash
BASAL_RESOURCE_ARCHIVE=hf://buckets/itsoltech/basal-rs/measurements/ultra-resources-2026-10-17/47926913c632019d9d05c5461025c397ce45fb79ec5aec29cdff83d139c2cd02/measurement-data.tar.gz
BASAL_RESOURCE_SHA=47926913c632019d9d05c5461025c397ce45fb79ec5aec29cdff83d139c2cd02
BASAL_RESOURCE_DL=.cache/measurement-download/ultra-resources-2026-10-17
mkdir -p .cache/measurement-download
mkdir "$BASAL_RESOURCE_DL" &&
  hf buckets cp "$BASAL_RESOURCE_ARCHIVE" "$BASAL_RESOURCE_DL/measurement-data.tar.gz" &&
  echo "$BASAL_RESOURCE_SHA  $BASAL_RESOURCE_DL/measurement-data.tar.gz" | sha256sum -c - &&
  mkdir "$BASAL_RESOURCE_DL/unpacked" &&
  tar -xzf "$BASAL_RESOURCE_DL/measurement-data.tar.gz" -C "$BASAL_RESOURCE_DL/unpacked" &&
  (cd "$BASAL_RESOURCE_DL/unpacked" && sha256sum -c SHA256SUMS)
```

`REPRODUCE.md` w archiwum opisuje lokalne odtworzenie audytu, jakości, startu
i tabel. Audyt świadomie zwraca kod 2 i FAIL; oczekiwane są 166 wpisy i 48 faz.
Oryginalna kopia README w archiwum pochodzi sprzed dopisania tej noty publikacji.
