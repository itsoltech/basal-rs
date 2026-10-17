# basal-2.0 — czas pojedynczego żądania HTTP na RTX 6000 Ada

Cel: zmierzyć czas od wysłania jednego pytania do odebrania pełnej odpowiedzi, dla basal-rs i upstream v2.0.0 fast, od krótkiego tekstu do limitu kontekstu modelu. Każde żądanie zawiera dokładnie jedno pytanie choice i jedną kolejność opcji; w danej chwili pracuje jeden klient i jeden silnik.

Kompletne pary długość/model: **25/25**. Liczba zapisanych niepoprawnych obserwacji, łącznie z rozgrzewką i pierwszym odrzucanym przebiegiem: **0**. Błędy, różnice odpowiedzi i braki danych pozostają w archiwum; nie zastępujemy ich skutecznymi ponowieniami.

- basal-2.0-mini: basal-rs ma niższą medianę w 4/7 kompletnych długości. Niższa mediana upstream: 128, 4096, 8192 tokenów. Zakresy median przebiegów zachodzą na siebie przy 128 tokenach; zakresy nie są przedziałami ufności.
- basal-2.0-4.5B: basal-rs ma niższą medianę w 8/9 kompletnych długości. Niższa mediana upstream: 32768 tokenów.
- basal-2.0-max: basal-rs ma niższą medianę w 8/9 kompletnych długości. Niższa mediana upstream: 32768 tokenów.

## Metoda i zakres czasu

- Wspólny klient HTTP/1.1 i trwałe połączenie lokalne. Zegar zaczyna biec przed serializacją JSON i wysłaniem żądania, a kończy po odebraniu całego body odpowiedzi. Obejmuje obsługę serwera, tokenizację, kolejkę, wykonanie modelu i składanie odpowiedzi. Parsowanie JSON po stronie klienta odbywa się po zatrzymaniu zegara. To czas HTTP request–response na tej samej maszynie, bez opóźnienia Internetu, nie czas samego forwardu GPU.
- Osobne uruchomienia serwerów w rotacji ABBAABBA, gdzie A=basal-rs, B=upstream fast. Pierwszy przebieg każdego silnika jest zachowany, ale odrzucony ze statystyki. W każdym uruchomieniu: dwie rozgrzewki każdej długości i dziesięć żądań pomiarowych. Do kompletnej pary wymagamy trzech pełnych zachowanych przebiegów dla obu silników. Czas ładowania, kompilacji i rozgrzewki jest poza wynikiem; zdarzenia startu zapisano osobno.
- Mediana w tabeli to mediana trzech median przebiegów. Zakres min–max dotyczy tych trzech median, nie jest przedziałem ufności. p95 liczony łącznie z 30 poprawnych próbek jest orientacyjny; tak mała próba nie potwierdza stabilnego ogona rozkładu.
- Liczba tokenów dotyczy **całego promptu**, razem z szablonem, pytaniem i opcjami. `state_tokens` w danych podaje sam dokument. Mini: 128–8192; 4.5B/max: 128–32768. Native i upstream mają identyczne prompty i identyfikatory tokenów dla wszystkich 25 wejść; `usage.input_tokens` każdej poprawnej odpowiedzi musi zgadzać się z wejściem. Nie akceptujemy skróconych stanów: upstream ma jawne `--context-overflow refuse`.
- Audyt poprawnej próbki wymaga HTTP 200, zgodnej liczby tokenów i najdłuższego promptu, jednej odpowiedzi choice z kluczem a/b/c oraz skończonych prawdopodobieństw i confidence w zakresie 0–1. Suma trzech prawdopodobieństw musi różnić się od 1 o mniej niż 0,001. To kontrola odpowiedzi i danych pomiarowych, odrębna od oceny trafności modelu.
- Stany syntetyczne: ponumerowane akapity polskiej umowy i jedno pytanie o formę reklamacji. Te same dokumenty są powtarzane. Cache KV stanów wyłączony jawnie w obu silnikach; pozostałe ustawienia produktu domyślne. Stałe prefiksy, cache algorytmów GEMM, CUDA Graphs i cache kompilacji mogą być używane przez dany silnik. Wynik nie opisuje zimnego startu ani rozkładu wszystkich rzeczywistych dokumentów.
- Basal-rs preoblicza stałe prefiksy szablonów: logi podają 34/45 tokenów, w tym zestawie używamy polskiego szablonu. Upstream na Ada zgłasza attention `fa2`; domyślne `--template-prefix auto` w przypiętym GraphBackend włącza preobliczenie szablonu dla trybu compact/fa3, więc tutaj go nie włącza. To różnica domyślnych ustawień produktów, odrębna od wyłączonego cache stanów. Wynik krótkiego wejścia nie izoluje wydajności samych kerneli.
- Na GPU nie uruchamiamy obu silników równolegle. Zachowano próbę blokady SM i telemetryczne próbki mocy, zegara, temperatury oraz VRAM. Diagnostyka z instrumentacją, jeśli wykonana, jest osobna i nie wchodzi do tabel czasu.
- Po zamknięciu głównych pomiarów dodatkowy runner `profile-local.py` zebrał natywny ślad Nsight Systems z `BASAL_HOST_TIMING`: mini 128/4096, 4.5B 32768 i max 32768 tokenów. Dwie rozgrzewki i trzy profilowane żądania na kształt. To oddzielna diagnostyka CPU/GPU; narzut instrumentacji wyklucza jej użycie jako pomiaru przewagi szybkości. Statusy i odpowiedzi są w `ada/extra-diagnostic/`, a surowe binarne profile pozostają prywatnie lokalnie.
- Pierwsza próba techniczna zakończyła się przed pomiarem: uruchomienie procesu w tle utrzymało połączenie SSH i przekroczyło limit klienta nadzorującego. Maszynę usunięto. Poprawiony launcher odłącza stdin/stdout/stderr procesu przez `subprocess.Popen`; właściwe pomiary pochodzą z jednej nowej sesji, bez łączenia próbek między maszynami.

## Wersje i sprzęt

GPU: NVIDIA RTX 6000 Ada Generation; VRAM 49140 MiB; limit mocy 300.00 W; sterownik 580.126.09; compute capability 8.9. Maksymalne zgłaszane zegary: SM 3105 MHz, pamięć 10001 MHz. CPU: Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz; vCPU: 12.

Basal-rs: kod `3fa10de179b0ac8fb4bec024065aa5e90f543096`, binarka CUDA SHA-256 `169ce19a4be027f9026c5be8b2d86bb41a4374d92f4667fa373cb3c5ddf02f30`. Drzewo dokumentacji przed kampanią: `9c0762f2971e0970345c2ddc7dad03a189d11658`; brak zmian w crates względem mierzonego kodu. Upstream: `af1ce43a1770dd4fdfc1611e3fe720b93d42fc73` (v2.0.0). Obie implementacje korzystają z tych samych przypiętych checkpointów BF16; basal-rs oblicza w FP16, upstream fast w BF16. To porównanie ustawień produktów.

Python 3.13.16, PyTorch 2.11.0+cu128, CUDA koła PyTorch 12.8, transformers 5.17.0. Pełne pakiety w `ada/requirements.txt`. Native używa bibliotek CUDA 12.9.1 z procedury uruchomienia.

- `Remek/basal-2.0-mini`: `cb81b32ce89b0b87e62fcdad28f0435e25c953f3`.
- `Remek/basal-2.0-4.5B`: `e5ba9149760d99b73b1963d8e0ccf948b58afacb`.
- `Remek/basal-2.0-max`: `2fffc5cc41e5cf7a2f752f38f17e7b4d9a4a8252`.

## basal-2.0-mini

Mniej ms oznacza szybciej. Współczynnik = czas upstream / czas basal-rs; powyżej 1 wygrywa basal-rs, poniżej 1 upstream.

| Tokeny promptu / dokumentu | upstream mediana [min–max] ms | basal-rs mediana [min–max] ms | p95 upstream / basal-rs ms | upstream / basal-rs |
|---:|---:|---:|---:|---:|
| 128 / 43 | 6,31 [6,27–6,66] | 6,45 [6,35–6,46] | 6,78 / 6,54 | 0,98× |
| 256 / 171 | 9,22 [9,10–9,26] | 8,63 [8,54–8,67] | 10,10 / 8,85 | 1,07× |
| 512 / 427 | 17,09 [16,78–17,15] | 15,77 [15,64–15,83] | 17,31 / 16,10 | 1,08× |
| 1024 / 939 | 34,80 [33,85–34,97] | 33,63 [33,04–33,69] | 35,82 / 34,15 | 1,03× |
| 2048 / 1963 | 67,47 [67,03–67,64] | 66,61 [66,25–66,71] | 71,81 / 67,40 | 1,01× |
| 4096 / 4011 | 142,77 [142,66–143,58] | 156,28 [156,16–156,31] | 146,03 / 157,17 | 0,91× |
| 8192 / 8107 | 328,60 [328,47–329,13] | 332,18 [331,86–332,56] | 330,37 / 335,57 | 0,99× |

## basal-2.0-4.5B

Mniej ms oznacza szybciej. Współczynnik = czas upstream / czas basal-rs; powyżej 1 wygrywa basal-rs, poniżej 1 upstream.

| Tokeny promptu / dokumentu | upstream mediana [min–max] ms | basal-rs mediana [min–max] ms | p95 upstream / basal-rs ms | upstream / basal-rs |
|---:|---:|---:|---:|---:|
| 128 / 43 | 17,34 [17,04–17,38] | 16,80 [16,67–16,89] | 17,60 / 17,03 | 1,03× |
| 256 / 171 | 31,42 [31,40–31,82] | 26,92 [26,60–26,93] | 32,63 / 27,85 | 1,17× |
| 512 / 427 | 56,40 [56,29–56,47] | 47,64 [47,49–49,04] | 60,49 / 51,58 | 1,18× |
| 1024 / 939 | 118,32 [117,10–118,94] | 92,38 [91,54–92,79] | 122,55 / 95,47 | 1,28× |
| 2048 / 1963 | 228,65 [228,54–228,83] | 187,72 [187,12–188,68] | 231,62 / 193,20 | 1,22× |
| 4096 / 4011 | 486,22 [486,08–486,80] | 410,39 [409,01–414,41] | 490,43 / 417,18 | 1,18× |
| 8192 / 8107 | 1007,19 [1006,67–1007,57] | 925,40 [924,54–933,81] | 1010,22 / 939,60 | 1,09× |
| 16384 / 16299 | 2499,70 [2499,31–2501,21] | 2419,09 [2409,02–2419,29] | 2523,37 / 2419,66 | 1,03× |
| 32768 / 32683 | 6152,71 [6151,42–6156,07] | 7207,90 [7206,92–7208,26] | 6165,93 / 7219,08 | 0,85× |

## basal-2.0-max

Mniej ms oznacza szybciej. Współczynnik = czas upstream / czas basal-rs; powyżej 1 wygrywa basal-rs, poniżej 1 upstream.

| Tokeny promptu / dokumentu | upstream mediana [min–max] ms | basal-rs mediana [min–max] ms | p95 upstream / basal-rs ms | upstream / basal-rs |
|---:|---:|---:|---:|---:|
| 128 / 43 | 43,21 [43,05–43,27] | 35,13 [35,08–35,60] | 43,83 / 36,32 | 1,23× |
| 256 / 171 | 76,80 [76,39–76,91] | 59,97 [59,38–60,04] | 83,86 / 64,61 | 1,28× |
| 512 / 427 | 143,56 [143,19–144,00] | 106,59 [105,72–106,80] | 147,36 / 111,62 | 1,35× |
| 1024 / 939 | 271,66 [271,11–273,20] | 200,29 [197,85–200,97] | 276,66 / 204,21 | 1,36× |
| 2048 / 1963 | 540,13 [538,58–540,19] | 393,34 [392,66–396,06] | 543,42 / 397,87 | 1,37× |
| 4096 / 4011 | 1095,36 [1091,90–1096,35] | 812,84 [809,52–818,98] | 1097,65 / 822,41 | 1,35× |
| 8192 / 8107 | 2232,64 [2227,48–2239,35] | 1979,31 [1979,29–1979,34] | 2242,24 / 1979,80 | 1,13× |
| 16384 / 16299 | 5221,03 [5211,55–5228,87] | 4948,01 [4928,66–4958,35] | 5242,53 / 4978,77 | 1,06× |
| 32768 / 32683 | 12109,94 [12059,85–12111,08] | 13223,62 [13202,98–13238,19] | 12124,35 / 13257,91 | 0,92× |

## Gotowość serwera — osobno od czasu inferencji HTTP

Czasy poniżej obejmują uruchomienie procesu do zaobserwowania odpowiedzi health (odpytywanie co sekundę), w tym ładowanie wag i przygotowanie silnika. Nie obejmują pobierania wag i instalacji. Pierwszy start modelu w tej sesji nie jest gwarantowanym zimnym startem maszyny: runtime, pliki i wspólny cache kompilacji mogą być już rozgrzane przez poprzednie modele. Dwa żądania rozgrzewki każdej długości odbywają się dopiero po gotowości serwera. Tych wartości nie dodajemy do tabel czasu odpowiedzi.

| Model | basal-rs: pierwszy start s | upstream: pierwszy start s | basal-rs: zakres startów 2–4 s | upstream: zakres startów 2–4 s |
|---|---:|---:|---:|---:|
| basal-2.0-mini | 12,01 | 309,15 | 6,00–6,00 | 190,08–199,10 |
| basal-2.0-4.5B | 14,01 | 629,30 | 14,01–15,01 | 453,21–457,24 |
| basal-2.0-max | 33,02 | 907,45 | 31,01–34,02 | 755,31–759,32 |

## Odpowiedzi, błędy i ograniczenia

Zgodny zbiór obserwowanych wyborów obu silników: 25/25 kompletnych par. Szczegółowe liczniki decyzji i liczba różnych pełnych obiektów odpowiedzi są w `summary.json`; wszystkie odpowiedzi, błędy HTTP i wyjątki klienta są zachowane per żądanie. To kontrola zachowania w tej kampanii, nie porównanie ogólnej jakości ani nowy benchmark względem FP32. Poprzednia zgodność z FP32 jest opisana w [kampanii pięciu GPU](../perf-basal2-fivegpu-2026-10-14/README.md).

Wyniki dotyczą RTX 6000 Ada i tych wejść. Nie wyjaśniają automatycznie wcześniejszej przewagi mini upstream na H100 ani nie dowodzą przewagi na innych GPU. Nie jest to pomiar przepustowości wielu klientów, otwartego ruchu, mediów omni ani kwantyzacji. Główne tabele czasu HTTP wykluczają start serwera. Przypadki graniczne kontekstu i długie wejścia nie reprezentują wszystkich typów dokumentów. Różne formaty obliczeń FP16/BF16 pozostają częścią porównania produktu.
Drabina używa dokładnych potęg dwójki dla długości całego promptu. Do 8192 są to także obsługiwane długości plain CUDA Graphs przypiętego upstream; nie mierzymy tu kosztu dopełniania dla długości pomiędzy tymi punktami. Ranking na rozkładzie rzeczywistych, zróżnicowanych dokumentów może wyglądać inaczej.

## Oddzielny profil basal-rs — instrumentacja Nsight Systems

Zebrano trzy profilowane żądania po dwóch rozgrzewkach na każdy poniższy kształt. Profil dotyczy wyłącznie basal-rs. Nie służy do odejmowania narzutu ani wyjaśnienia różnicy względem upstream. Nsight zgłasza włączone śledzenie zdarzeń CUDA i niedostępne próbkowanie CPU oraz context-switch tracing; instrumentacja może zmieniać czasy i zależności strumieni.

| Wejście | span GPU ms | aktywność GPU ms | GEMM ms | attention ms | kernele | host enqueue / wait ms |
|---|---:|---:|---:|---:|---:|---:|
| basal-2.0-mini/ctx-128 | 5,89 | 5,58 | 3,96 | 1,23 | 293 | 3,17 / 2,74 |
| basal-2.0-mini/ctx-4096 | 145,83 | 145,49 | 74,22 | 56,85 | 293 | 3,56 / 142,31 |
| basal-2.0-4.5B/ctx-32768 | 6952,48 | 6947,07 | 1383,60 | 5293,03 | 545 | 6858,87 / 93,69 |
| basal-2.0-max/ctx-32768 | 12872,02 | 12863,47 | 4240,59 | 8160,58 | 455 | 12689,33 / 182,15 |

Span GPU biegnie od pierwszej aktywności po poprzednim wyniku do zakończenia kopii własnych trzech logitów (12 B) do CPU. Aktywność GPU to suma przedziałów po usunięciu nakładania. Liczba kopii wyników musi odpowiadać liczbie żądań, a każdy zachowany forward zawiera dokładnie jeden kernel letter_logits. Czasy kategorii są sumami czasu kerneli; kolumny są medianami obliczanymi osobno. Host enqueue i wait pochodzą z BASAL_HOST_TIMING: enqueue obejmuje pracę CPU i ewentualne oczekiwanie w trakcie forwardu, wait końcowe oczekiwanie na wynik. GPU pracuje równolegle z enqueue; tych kolumn nie dodajemy do aktywności GPU. Przerwy w instrumentowanym śladzie nie są prognozą możliwego przyspieszenia.

Publiczne `profile-events/*-gpu-events.jsonl` zachowują wyłącznie względne przedziały, nazwy kerneli, rodzaj i rozmiar transferu. Nie zawierają adresów pamięci, identyfikatorów procesów, kontekstów ani urządzeń. `profile-summary.json` zawiera próbki i agregację. Odtworzenie: `python3 reproduce/summarize-profile.py --measurement-artifacts <katalog-raportu> --out <nowy-plik.json>`. Surowe SQLite i nsys-rep pozostają prywatnie; `extract-profile.py` dokumentuje eksport wybranych pól.

## Artefakty i odtworzenie

URI: `hf://buckets/itsoltech/basal-rs/measurements/latency-basal2-ada-2026-10-15/78fea1fc5eaf59b6790830c3871f0057c88598609f23910bf18f3147c8024f90/measurement-data.tar.gz`. SHA-256 archiwum: `78fea1fc5eaf59b6790830c3871f0057c88598609f23910bf18f3147c8024f90`. Publikacja pod nowym, niezmiennym prefiksem; procedura pobrania i weryfikacji: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).

Weryfikacja 2026-10-15: archiwum 602 148 B pobrano z autoryzacją i anonimowo. W obu przypadkach SHA-256 archiwum i 111/111 sum plików były zgodne; ponowna agregacja głównego pomiaru i profili dała identyczne obiekty JSON. Maszynę usunięto po zebraniu danych, a status `deleted` potwierdzono w API.

Archiwum zawiera `reports/latency-basal2-ada-2026-10-15/`: `inputs/requests.jsonl`, `inputs/manifest.json`, `inputs/prompt-audit.json`, `ada/*-observations.jsonl` (ciepłe i mierzone żądania), logi serwerów, starty i statusy w `ada/progress.json`, sprzęt, oprogramowanie, telemetrykę, `summary.json`, `reproduce/` ze skryptami. Prywatnych danych maszyny, rozliczeń i surowych binarnych profili z identyfikatorami hosta nie publikujemy. Ścieżki wykonania w logach zastąpiono nazwami logicznymi; wartości pomiarowe pozostały bez zmian.

Po pobraniu i sprawdzeniu archiwum oraz wszystkich `SHA256SUMS`, odtwórz agregację bez GPU:

```bash
BASAL_REPORT_ROOT="$BASAL_ARTIFACT_ROOT/reports/latency-basal2-ada-2026-10-15"
python3 "$BASAL_REPORT_ROOT/reproduce/summarize.py" \
  --measurement-artifacts "$BASAL_REPORT_ROOT" --out /tmp/basal2-latency-summary.json
```

Ponowny pomiar wymaga osobnej zatwierdzonej sesji CUDA. `reproduce/setup.sh` opisuje przypięte zależności i pobranie checkpointów. `reproduce/campaign.py --measurement-artifacts <katalog-wejść> --out <nowy-katalog> --deadline <czas-UNIX>` uruchamia serwery i klienta na istniejącej maszynie, bez wynajmowania. Zapisane punkty montowania to konwencja runnera. `make-inputs.py` wymaga tokenizerów, checkoutu upstream, lokalnych manifestów modeli i CPU-capable CLI `basal prepare`; pełne wejścia i wynik audytu są już w archiwum.
