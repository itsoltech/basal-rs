# Pico: refaktor bez zmiany obliczeń

## Zakres i kryteria

Refaktor wykonany wspólnie z Claude Opus 5.5, na prośbę użytkownika.
Celem jest uproszczenie dodanego backendu Pico przy zachowaniu wyników,
flag, domyślnych ustawień, kolejności operacji GPU, granic fragmentów,
checkpointów i niemutowalnego stanu prefiksów. Bez nowych algorytmów,
zmiany precyzji, profili modeli i modyfikacji upstream.

Baza to końcowy build z kampanii
[`pico-singlepass-rtx4060-2026-10-27`](../pico-singlepass-rtx4060-2026-10-27/README.md),
SHA-256 `514f08cde1874fa13c4676d659c5e49a7201d1286dee894e10fd102ad2970a81`.
Nie używamy wcześniejszej binarki z wyłączoną diagnostyką emulacji jako
zamiennika tej bazy. Źródła przed refaktorem zostały zachowane.

RTX 4060 8 GiB, SM89, CUDA 12.9.1. Checkpoint
`Remek/basal-omni-pico`, rewizja
`a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`, SHA-256 wag
`ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9`.
Upstream referencyjny 2.4.0, commit
`615fe321a818fdf5d2897f758b8d823f919d7f94`.
Kod bazuje na `db55af9d7cab5f12a18347b0bb27eb4dab4b1531` z wcześniejszą
niezatwierdzoną implementacją Pico. Archiwum zachowuje pełne zmienione
źródła, nie tylko identyfikator commita.

## Uproszczenia

1. Główny plik backendu ma 286 zamiast 1091 linii; pięć prywatnych modułów
   rozdziela ładowanie wag, attention, DeltaNet, prefill i prefiksy.
   Kod CUDA tej rodziny jest w osobnym `fused/qwen35.rs`; ogólny `fused.rs`
   ma 1575 zamiast 2086 linii. Wspólny masked softmax pozostał w module
   ogólnym. Jest to podział odpowiedzialności, nie usunięcie tych operacji.
2. `shared_rows` zbiera dotychczas powtarzane sprawdzanie pokrycia tabelą
   GEMM, łączenie wejść i rozdzielanie wyniku. MLP korzysta ze wspólnej
   implementacji dla pojedynczych i grupowanych wierszy.
3. Ładowanie ma jawny `Loader`, a attention i DeltaNet krótsze funkcje
   poszczególnych etapów. `Prepared` nazywa operandy rekurencji i zachowuje
   ich niemutowalne widoki. Bez nowych bibliotek i publicznego API.
4. Wspólne warunki wyboru kerneli i stałe długości mają jedną definicję.
   Prefill prefiksu oraz walidacja promptów i kodów liter nie są powielone
   w ścieżkach grupowanych i pojedynczych.
5. Przegląd lokalny dodał jawny warunek urządzenia CUDA w krótkim
   attention i przenoszenie ownership bufora konwolucji, aby zachować jego
   wcześniejszy czas życia. Pierwszą próbę powtórnego przeglądu Opus przerwał
   limit API; wznowiony przegląd zakończył się bez uwag blokujących.
   [Uzupełnienie przeglądu](REVIEW.md) wyjaśnia też korektę wcześniejszego
   określenia „przywrócił”. Implementację przejrzał również główny agent
   według `rust-review` i `itsol-self-review`; pomiary są osobną weryfikacją.

`plan.rs`, `kernels.cu`, profile i domyślne flagi są identyczne z bazą.
Przeniesione kernele zachowały walidację, konfiguracje launch i komentarze
`SAFETY`. Zmieniono organizację kodu i drobne struktury hosta, nie arytmetykę.

## Kwalifikacja wyników

Wykonano świeże eksporty tej samej bazy i końcowego builda dla 13 konfiguracji:
BF16 domyślne i z gotowym buforem (orders=1/2 i granice dla obu), FP32
domyślne (orders=2), FP32 rezydentne (orders=2 i granice), FP32 rezydentne
z gotowym buforem (orders=1/2 i granice) oraz FP16 domyślne (orders=1).
To te same przykłady uruchomione w różnych konfiguracjach, nie 13
niezależnych korpusów.

`cmp` potwierdził **13/13 par bitowo identycznych plików `bench.jsonl`
i `systemone.jsonl`**. `basal compare` wykazał zero różnic logitów
i prawdopodobieństw, brak różnic tokenów i brak zmian pól odpowiedzi.
Pełny zestaw ma 44 pytania benchmarku i 12 gałęzi liczbowych API;
granice to 5080 tokenów, 255 opcji i zachowana odmowa 256 opcji.
FP32 zachowuje wcześniejszy próg ≤0,0001 względem upstream przez bitową
równość z zakwalifikowaną bazą. Nie zmienia to wcześniejszych ograniczeń
jakości BF16/FP16 względem FP32.

Build CUDA, `cargo fmt`, Clippy Linux i CUDA all-features oraz Rustdoc
Linux i CUDA all-features (`--document-private-items`, `-D warnings`): PASS.
Kontrola wyjątku `paste` i składni runnera Python: PASS. Zależności nie
zostały zmienione. `cargo deny` nadal ma wcześniejszy **FAIL** dla
`zstd 0.13.3` i `zstd-safe 7.3.0`, RUSTSEC-2026-0391 i 0395–0406;
nie dodano wyciszeń. Istniejące duplikaty `cargo tree` przejrzano.

## Pomiary szybkości

Osobne ABBA dla domyślnego BF16 oraz FP32 z rezydentnymi wagami i gotowym
buforem po obu stronach. HTTP C1, FIFO, budżet 8192 tokeny, bez cache,
OMP/MKL/Rayon=4. Osiem startów A/B/B/A/A/B/B/A, pierwsza seria każdej
wersji odrzucona. Minimum 10 s rozgrzewki silnika; dwa żądania rozgrzewki
przed każdym punktem drabiny, 12 próbek 255 i 5080 tokenów oraz 64
krótkie żądania (po 16 na dokument) w każdym starcie. Wynik to mediana
median trzech zachowanych startów; zakresy pokazują zmienność między nimi.

Oba audyty: **AUDITED, zero problemów**, bez błędów HTTP i zmian odpowiedzi.

| Dokument | BF16 przed, ms | BF16 po, ms | FP32 przed, ms | FP32 po, ms |
|---|---:|---:|---:|---:|
| 5080 tokenów, osobny punkt C1 | 536,78 | 536,77 | 1548,56 | 1548,61 |
| 27 opcji, krótki korpus | 24,49 | 24,47 | 68,29 | 68,29 |
| bank, sześć typów pytań | 55,01 | 54,99 | 163,03 | 163,01 |
| wspólny prefiks | 44,71 | 44,67 | 131,79 | 131,84 |
| Unicode | 25,18 | 25,18 | 62,70 | 62,69 |

Zakresy median startów dla 5080: BF16 przed 536,54–537,19 ms,
po 535,68–537,02 ms; FP32 przed 1547,81–1548,99 ms,
po 1547,19–1548,99 ms. Największy wzrost mediany krótkiego dokumentu
to około 0,045% (prefiks FP32, +0,0585 ms); jego zakresy przed/po
131,59–131,82 i 131,67–132,00 ms nakładają się.
Nie zaobserwowano istotnej regresji median. Nie jest to dowód
matematycznie identycznego czasu każdego żądania ani pomiar przyspieszenia.

Mediany p95 poszczególnych startów, również z zachowaniem niewielkich wzrostów:

| Dokument | BF16 przed, ms | BF16 po, ms | FP32 przed, ms | FP32 po, ms |
|---|---:|---:|---:|---:|
| 5080 tokenów | 538,02 | 538,56 | 1550,31 | 1550,35 |
| 27 opcji | 25,55 | 24,70 | 68,98 | 69,08 |
| bank | 56,25 | 55,85 | 164,00 | 164,65 |
| wspólny prefiks | 45,73 | 44,75 | 133,36 | 133,12 |
| Unicode | 26,07 | 25,39 | 63,28 | 63,58 |

Największy wzrost mediany p95 wyniósł około 0,482% (+0,305 ms,
Unicode FP32). Przy 16 próbkach dokumentu w starcie p95 jest maksimum
tego startu; tabela nie daje mocnej gwarancji o ogonie rozkładu.
Pełne zakresy i wszystkie obserwacje pozostają w audytach/JSONL.
Nie wykonano formalnego testu równoważności statystycznej.

Wnioski dotyczą hot HTTP C1 dla dwóch zmierzonych konfiguracji i tego
korpusu na RTX 4060. Pozostałe konfiguracje przeszły kwalifikację bitową,
ale nie osobne ABBA. Metal i inne GPU nie były uruchamiane. Flagi
diagnostycznego profilowania i synchronizowanego timingu przejrzano
w kodzie, bez osobnego pomiaru ich runtime'u.

## Artefakty i odtworzenie

Dane mają logiczny katalog `reports/pico-refactor-rtx4060-2026-10-27/`:
`exports/` zawiera eksporty przed/po, `compare-*.json` porównania,
`references/` referencje upstream, `http/` metody, odpowiedzi i audyty,
`inputs/` korpus i tabelę GEMM, `source/` źródła przed/po, a
`validation.json` zakres i status kontroli. `telemetry.csv` zawiera
odczyty temperatury, zegarów, obciążenia i łącznej pamięci GPU co sekundę;
nie jest pomiarem pamięci samego backendu ani modyfikacją zegarów.

Binarka po refaktorze: SHA-256
`dbb9d80a0b10ad920c9300a38eb514816badf2b4b52093090c7006c006a58b0d`.
`REPRODUCE.md` opisuje odbudowę, kwalifikację i audyt przez
`--measurement-artifacts`. Lokalne archiwum zostaje zachowane.

Publiczne archiwum:
`hf://buckets/itsoltech/basal-rs/measurements/pico-refactor-rtx4060-2026-10-27/12bc0b1ad6c3fb4b1dfe5f1c871a9bbcd89ea8f3c665f277ced53f251e44e2e8/measurement-data.tar.gz`.
SHA-256 `12bc0b1ad6c3fb4b1dfe5f1c871a9bbcd89ea8f3c665f277ced53f251e44e2e8`,
1 167 423 B, 148 plików. Pobrania z logowaniem i anonimowo potwierdziły
SHA archiwum oraz 148/148 sum plików w obu kopiach. Anonimowe środowisko
potwierdziło brak logowania. Publicznego prefiksu nie nadpisano.

Pobierz przez `hf buckets cp` do nowego katalogu, sprawdź powyższy SHA-256,
rozpakuj i uruchom `sha256sum -c SHA256SUMS`, a następnie korzystaj
z `REPRODUCE.md`. Archiwalne `SUMMARY.md` jest kopią tego raportu sprzed
dopisania noty publikacyjnej.

Później dostarczony wynik wznowionego przeglądu jest opisany w `REVIEW.md`.
Pierwotne archiwum pozostaje bez zmian; jego `validation.json` odnotowuje
stan przeglądu z chwili publikacji. Korektę opisu warunku CUDA wyjaśnia
uzupełnienie, bez zmiany mierzonego kodu i wyników.
