# RSS klienta HTTP — 2026-10-10

Małe rekordy utrzymywały RSS około **13–16 MiB**. Nie zaobserwowano nieograniczonego wzrostu pamięci:
120 000 małych rekordów w jednym procesie utrzymywało około 14 MiB, a dwadzieścia cykli dużych i małych
rekordów osiągnęło plateau około 85 MiB. Cztery skany żywych procesów przez macOS `leaks` zgłosiły
**0 wycieków / 0 leaked bytes**. To wynik tych obciążeń i czasu obserwacji, nie dowód braku każdego wycieku.

Duże JSON-y i równoległość mają znaczący koszt. Odpowiedzi 12 MiB przy `--jobs 8` dały medianę szczytu
219 MiB; przy `--jobs 1` było to 43,5 MiB. Po dużych rekordach allocator zatrzymuje pamięć i RSS nie wraca
od razu do poziomu początkowego. Nie zmieniono implementacji klienta ani obliczeń modelu.

## Warunki i metoda

- Rewizja klienta: `273150c7c665af4fa2718328fdcc4009ffa3b4f7`; release, `--locked`, domyślny backend Metal,
  bez ładowania modelu. SHA-256 identycznej binarki we wszystkich seriach:
  `4e3ad35dc877c854c43daa4e98ef5461bceafc925631d11033a8859201532107`.
- macOS 27.0.1, arm64, Apple M1 Pro, 32 GiB RAM; Python 3.14.7. Wyników nie przenosimy na Linux/CUDA.
- Ręczne narzędzie: [client_rss.py](../../tools/perf/client_rss.py). Serwer HTTP, generator wejścia,
  konsument stdout i sampler są w osobnym procesie Pythona, poza RSS mierzonego procesu CLI.
- Szczyt RSS: kernelowe `os.wait4(...).ru_maxrss` w bajtach na macOS. Próbki: `ps -o rss=` co nominalnie
  100 ms, KiB przeliczone na bajty. **MiB = 1 048 576 bajtów**. Szczyt pochodzi z kernela, więc sampler
  nie musi trafić w krótkotrwałą alokację.
- Przed wejściem stdin pozostaje otwarte przez 0,5 s. Po każdej porcji strumienia czekamy na wszystkie
  oczekiwane linie stdout, zostawiamy stdin otwarte przez 0,8 s i bierzemy medianę RSS próbek po pierwszych
  0,2 s tej przerwy. Dla pojedynczego `text` EOF jest konieczny do wykonania requestu, więc nie ma próbki
  po opróżnieniu kolejki w tym samym żywym procesie.

Fixture obsługuje HTTP/1.1 i keep-alive, zwraca syntetyczną odpowiedź `noul: 0.75` na jedno pytanie,
nie wykonuje inferencji. Ma kolejkę 256 połączeń: domyślne pięć połączeń serwera Pythona powodowało
resety w pilocie przy `--jobs 8`. Nieudane piloty nie są włączone do wyników. Z klienta usunięto odziedziczone
ustawienia proxy, autoryzacji i backtrace. Model podano jawnie, więc nie mierzono autodiscovery ani TLS.

Łącznie wykonano **67 przebiegów i 740 990 rekordów**, w tym celowo błędne wejścia i HTTP 503.
Kontrolowano liczbę i kolejność wyników, liczbę błędnych rekordów oraz kod wyjścia. Nie przechowywano całego
stdout w pamięci. Obciążenia w tabeli mają po trzy osobne procesy; długie 20 cykli ma dwa procesy,
a dodatkowe skany `leaks` są pojedyncze. Dodatkowa seria `native-leaks` częściowo nakładała się na końcowe
obciążenia pierwszego powtórzenia `main`; drugie i trzecie powtórzenie oraz kolejne serie były wykonywane
bez drugiego procesu pomiarowego klienta. Nie jest to kontrolowany benchmark przepustowości.

## Szczyt RSS

Mediana i zakres trzech przebiegów. `1 MiB` w wejściu oznacza długość tekstowego pola stanu;
rekord JSONL ma również niewielki narzut struktury. Strumienie bez `--value` zachowują wejście w kopercie.

| Obciążenie | Rekordy na proces | jobs | Mediana MiB | Min–max MiB |
|---|---:|---:|---:|---:|
| Jeden tekst 64 B, `--value` | 1 | 1 | 12,94 | 12,89–12,98 |
| Jeden tekst 1 MiB, `--value` | 1 | 1 | 18,09 | 18,05–18,20 |
| Linie 256 B | 5 000 | 1 | 13,36 | 13,34–13,45 |
| Linie 256 B | 5 000 | 8 | 14,22 | 14,14–14,34 |
| Linie 256 B | 5 000 | 32 | 16,06 | 16,02–16,25 |
| JSONL, stan 1 KiB | 5 000 | 8 | 14,31 | 14,28–14,34 |
| JSONL, stan 1 MiB | 64 | 1 | 27,06 | 27,03–30,66 |
| JSONL, stan 1 MiB | 64 | 8 | 72,11 | 70,25–74,36 |
| JSONL, stan 1 MiB | 64 | 32 | 77,48 | 73,19–95,02 |
| JSONL, stan 1 MiB, `--value` | 64 | 8 | 60,06 | 60,05–64,70 |
| JSONL: 20 000 małych obiektów, 628 908 B na linię | 32 | 8 | 84,64 | 74,66–91,41 |
| Odpowiedź z polem padding 1 MiB | 64 | 8 | 36,70 | 35,86–39,06 |
| Odpowiedź z polem padding 12 MiB | 32 | 1 | 43,50 | 43,06–44,22 |
| Odpowiedź z polem padding 12 MiB | 32 | 8 | 218,89 | 215,58–221,66 |
| JSONL 1 MiB, pierwszy request opóźniony o 2 s | 64 | 32 | 91,31 | 85,53–91,70 |
| JSONL 1 MiB, każdy request opóźniony o 50 ms | 64 | 32 | 114,67 | 112,95–121,75 |
| Niepoprawny JSON, `--keep-going` | 20 000 | 8 | 12,23 | 12,12–12,28 |
| HTTP 503, `--keep-going`, bez retry | 5 000 | 8 | 14,41 | 14,20–14,42 |
| Osiem linii 3 MiB ponad limit, potem 5 000 linii 256 B | 5 008 | 8 | 18,25 | 18,20–20,12 |
| Sześć porcji po 20 000 linii 256 B | 120 000 | 8 | 14,39 | 14,31–14,39 |
| Trzy cykle dużych i małych JSONL po rozgrzewce | 16 192 | 8 | 80,83 | 74,72–83,70 |

Różne przebiegi dużych rekordów mają różną kolejność alokacji i stopień jednoczesnego wykonywania requestów.
Samo `--jobs 32` ustala górny limit, nie gwarantuje 32 dużych requestów naraz; opóźnienie serwera zwiększa
nakładanie się alokacji. Nie jest to górna granica RSS dla wszystkich poprawnych danych. Liczba węzłów JSON,
escape'owanie tekstu, liczba pytań, wielkość odpowiedzi i wolny konsument stdout mogą zwiększyć koszt.

## Długi strumień i pamięć po opróżnieniu kolejki

W drugim powtórzeniu `soak-120k-j8` RSS po 20 000 rekordów wyniósł 14,203 MiB, po 60 000 — 14,188 MiB,
a po 120 000 — 14,188 MiB. Trzecie powtórzenie utrzymywało 14,141 MiB we wszystkich sześciu punktach.
Pierwsze powtórzenie wzrosło z 13,969 do 14,156 MiB. Nie widać wzrostu proporcjonalnego do liczby rekordów.

Wydłużona seria: rozgrzewka 1 000 JSONL po 1 KiB, potem 20 razy: 64 JSONL po 1 MiB i 1 000 po 1 KiB.
Łącznie 22 280 rekordów i około 42 s na proces. RSS po małej porcji:

| Zakończony cykl | Przebieg 1 (MiB) | Przebieg 2 (MiB) |
|---|---:|---:|
| 1 | 76,469 | 69,453 |
| 5 | 82,094 | 78,797 |
| 10 | 84,703 | 84,703 |
| 15 | 84,719 | 84,719 |
| 20 | 84,766 | 84,734 |

Oba procesy osiągnęły szczyt 84,953 MiB. Po 20 cyklach `vmmap` pokazał:

| Miara | Przebieg 1 | Przebieg 2 |
|---|---:|---:|
| Aktywne alokacje malloc, wszystkie strefy | 2 783 KiB | 2 781 KiB |
| Resident głównej strefy malloc | 65,1 MiB | 65,0 MiB |
| Fragmentacja głównej strefy malloc | 96% | 96% |
| Physical footprint | 44,3 MiB | 48,1 MiB |
| `leaks`: leaked bytes | 0 | 0 |

Te miary nie są zamienne: RSS, footprint, resident strefy malloc i jej aktywne alokacje liczą różne rzeczy.
Mała liczba żywych alokacji, duża fragmentacja i plateau wskazują na zatrzymane strony allocatora,
a nie narastającą retencję wszystkich przetworzonych rekordów. Mimo braku wykrytego wycieku proces może
utrzymywać około 85 MiB RSS po zakończeniu dużej porcji. Nie dodano wymuszania zwalniania stron allocatora,
które wymagałoby osobnego pomiaru kosztu i przenośności.

## Wnioski praktyczne

- Dla małych rekordów zmierzony narzut procesu jest niewielki: około 14 MiB przy `--jobs 8`.
- Dla dużych payloadów zacznij od `--jobs 1`. Dla odpowiedzi 12 MiB zmierzony szczyt był około pięć razy
  mniejszy niż przy ośmiu jobach. Nie wyciągamy z tego wniosku o szybkości na prawdziwym serwerze.
- Gdy potrzebna jest jedna odpowiedź bez koperty wejściowej, `--value` zmniejszyło medianę dla JSONL 1 MiB,
  jobs 8, z 72,11 do 60,06 MiB. Nadal trzeba odebrać i sparsować odpowiedź API.
- Limity bajtów wejścia i odpowiedzi nie są budżetem RSS. W szczególności mały JSON z wieloma obiektami
  może kosztować więcej niż większy JSON zawierający jeden długi tekst.
- Nie wykryto wycieku w tych przebiegach. Nie mierzono `--local`, wag/aktywacji modelu, prawdziwej inferencji,
  TLS, retry, długich timeoutów, wielogodzinnego działania ani zachowania na Linux. `leaks` bez historii
  stacków alokacji może pominąć problemy; stabilny RSS sam nie dowodzi braku wycieku.

## Dane i odtworzenie

- [main/summary.json](main/summary.json): 17 obciążeń × 3 procesy; per-run JSON zawiera pełną serię RSS.
- [extended/summary.json](extended/summary.json) i [options/summary.json](options/summary.json): duże
  odpowiedzi, gęsty JSON i wpływ `--jobs`/`--value`, po trzy procesy.
- [native-leaks/summary.json](native-leaks/summary.json): dwa dodatkowe skany `leaks` po opróżnieniu kolejki.
- [long-cycles/summary.json](long-cycles/summary.json): dwa wydłużone przebiegi wraz z `.heap.txt` i `.leaks.txt`.

```sh
cargo build --locked --release -p basal-cli --bin basal
python3 tools/perf/client_rss.py --binary target/release/basal --repeats 3 \
  --case lines-256B-j1 --case lines-256B-j8 --case jsonl-1MiB-j8 \
  --case response-12MiB-j1 --case response-12MiB-j8 --case soak-120k-j8 \
  --out reports/NOWY-POMIAR-RSS
python3 tools/perf/client_rss.py --binary target/release/basal --repeats 2 \
  --case large-small-20-cycles-j8 --heap --leaks --out reports/NOWY-POMIAR-ALLOCATORA
```

Każdy pomiar wymaga nowego katalogu. Narzędzie odmawia nadpisania wyników. `--heap` i `--leaks` wymagają macOS;
sam pomiar RSS obsługuje również Linux, ale nie został tu wykonany na Linux.
JSON zapisuje czas porcji, rekordy/s i `wall_ms_per_record` jako amortyzowany czas na rekord,
**nie latency pojedynczego requestu**. Są zależne od fixture, generowania danych i parsowania stdout,
więc nie służą do porównania szybkości inferencji.

Weryfikacja narzędzia: wszystkie kompletne przebiegi zakończyły się oczekiwanym kodem i liczbą wyników;
`ruff check`, `ruff format --check` i kontrola whitespace diffu przeszły. Nie dodano frameworków testowych
ani bramek CI. Kod Rust klienta pozostaje bez zmian względem mierzonej rewizji.
