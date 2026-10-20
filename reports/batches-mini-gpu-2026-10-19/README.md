# Batch mini CUDA — obserwacje GPU, 2026-10-19

Status: **PASS zgodności A/B i body; PASS krótkiego korpusu C2/C8;
FAIL progu online przy granicy tokenów; GAP pełnej kwalifikacji**. To pomiar jednego profilu mini na jednym GPU,
bez promocji konfiguracji produkcyjnej ani zamknięcia całej inicjatywy Batch API.

## Cel i pochodzenie

Sprawdzić, czy ekstrakcja planera i wykonanie Batch API zachowują wyniki mini,
czy ruch obu dialektów online nadal uzyskuje odpowiedzi podczas pracy tła,
oraz czy aktywne cancel i restart własnego serwisu zachowują trwały stan.
Przebiegi wykonał rodzic na własnym GPU; przygotowanie tego raportu i archiwum
odtwarza analizę lokalnych wyników, bez nowych wywołań modelu lub serwera.

| Element | Wersja / ustawienie |
| --- | --- |
| Baseline kodu | `beb856275fbcb811eff5ce4446479bd74dafa176` |
| Candidate | przypięty, niecommitowany snapshot R2; 267 plików zgodnych z manifestem |
| Model | wyłącznie `basal-2.0-mini`, revision `cb81b32ce89b0b87e62fcdad28f0435e25c953f3` |
| Obliczenia | CUDA, f16 z wag bf16, bez kwantyzacji, prefix cache on, state cache 0 |
| Batching | request Tree, limit 2048 tokenów; primitive export Single; tło solo |
| Sprzęt | NVIDIA RTX 6000 Ada Generation 48 GB; urządzenie współdzielone |
| Toolchain | Rust 1.95.0, CUDA 12.9, driver 580.159.03 |
| Użyty GEMM | `inputs/gemm-mini-f16.json`, SHA-256 `6ca59cc5dc56eb06c19f5dfcb161b9c3875c5326ab4b731ed9ea0b7681f0ba29` |

Tabela GEMM ma metadata `basal-1.5-mini` i wspólne kształty projekcji. We wszystkich
przebiegach użyto tego samego niezmienionego pliku; nie ładowano wag 1.5.
Nie zastępuje go tracked table o podobnej nazwie. Archiwalne snapshoty źródeł
zachowują także inne istniejące tabele kodu, ale nie są one wejściami tych pomiarów.

Admission było laboratoryjne: 2 GiB pracy tła, 6 GiB zapasu online,
granularity 2 MiB z minimum VMM drivera oraz koszt 40000 ns/token z zaokrąglonego
bootstrapu dwóch kalibracyjnych żądań online. To nie jest ścisły pomiar każdej
alokacji, uniwersalny koszt GPU ani kwalifikacja dowolnego rekordu. Ustawienia
pozostają opisem eksperymentu; nie zmieniano konfiguracji produkcyjnej.

## Wyniki numeryczne

| Porównanie | Rekordy | Wynik |
| --- | ---: | --- |
| Primitive export baseline → candidate | 16 | pliki identyczne bajt po bajcie |
| Request export baseline → candidate | 17 | pliki identyczne bajt po bajcie |
| Token orders w `basal compare` | 42 | 0 mismatches; max różnica logitów 0 |
| Batch solo względem online solo | 64 | 0 różnic canonical body |
| Batch podczas C2 online | 128 | 0 różnic canonical body |
| Batch podczas C8 online | 256 | 0 różnic canonical body |
| Wyniki zachowane po aktywnym cancel | 4 | 0 różnic canonical body |
| Wyniki po SIGINT i restarcie store | 512 | 0 różnic canonical body |
| Batch blisko granicy tokenów, bez kontynuacji | 8 | 0 różnic canonical body |

Łącznie odtworzono **972/972** porównania body, bez różnic i brakujących rekordów.
Porównanie zachowuje wszystkie pola odpowiedzi i reprezentacje liczb; usuwa
wyłącznie `latency_ms`, rekurencyjnie. Identyfikatory envelope są poza body.
Sposób canonicalizacji i hashe każdej pary są w archiwum; niezależny od runtime'u
`analyze.py --measurement-artifacts` ponownie wylicza wszystkie pary.
Nie jest to porównanie jakości z upstream FP32.

SHA-256 identycznych primitive JSONL: `6002237bc5c9294b52b9ec843e623d0702e477327b7cddf394200ec49f09d5ef`.
SHA-256 identycznych request JSONL: `5411a3bcae204ca39657e14aa2dfc4424f48d2be7adecec38609a7284f330289`.

## Ruch online podczas Batch API

Ruch timed był closed-loop, naprzemiennie `/v1/basal` i `/v1/systemone`.
P99 używa nearest-rank i został ponownie wyliczony z zapisanych timed responses.
Osobne referencje i celowe błędne wejścia nie należą do timed próbek w tabeli.
Przed każdym mixed przyjęto próg baseline p99 + 10 ms i zero błędów timed online.
To sesyjny próg eksperymentu, nie produkcyjne SLO.

| Concurrency | Online-only / mixed próbek | Baseline p99 | Mixed p99 | Próg przed mixed | Błędy timed online |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2 | 200 / 200 | 38,435 ms | 39,311 ms | 48,435 ms | 0 / 0 |
| C8 | 1000 / 1000 | 88,532 ms | 89,961 ms | 98,532 ms | 0 / 0 |

W obu krótkich przebiegach C2/C8 spełniono wcześniej ustalony próg. Nie wyciągamy
wniosku o przyspieszeniu: po jednym przebiegu, współdzielony GPU, krótki
syntetyczny korpus i closed-loop nie są dowodem dowolnego długotrwałego obciążenia.

W C8 health zapisał 2 przerwania `pause_limit`, 10 ponownie zakolejkowanych
rekordów porzuconej pracy i 0 rzeczywistych failure_attempts. Następnie ukończono
256/256. Te dwa przerwania miały zapisany koszt urządzenia 0; nie stanowią dowodu
oddania GPU w środku rozpoczętego długiego forwardu.

## Granica tokenów: zachowany FAIL i zakres profilu

Osobne, eksploracyjne próby użyły dłuższych wejść na tym samym modelu i profilu:
single miał 1960 tokenów, a 32 wybory 62902 tokeny packed / 1966 najdłuższego
forwardu wobec limitu 2048. Ich przyjęcie i ukończenie pokazują dla tych danych,
że suma packed większa od 2048 nie jest mylona z długością jednego forwardu.
To nadal solo dla jednego rekordu API, a nie grouping wielu rekordów Batch API.

Pierwszy batch z 12 rekordami zawierał także poprawny native schema extract:
online HTTP200 (595 tokenów, 479 najdłuższego forwardu). Cały batch zakończył
się **FAIL walidacji**, `budget_exceeded` na liniach 3/6/9/12: konserwatywna
granica kontynuacji to 16384, więcej niż deklarowane 2048 profilu. Nie wykonano
pracy tła; mixed p99 1254,229 ms wobec progu 1254,766 ms nie jest dowodem
priorytetu. Native extract działa online, ale nie jest zakwalifikowany w tym
profilu Batch API. Zachowano też HTTP422 dla native evidence: revision mini
nie ma `evidence_head.pt`, więc ta funkcja pozostaje niewspierana i niekwalifikowana.
To ograniczenia profilu/wejścia, bez wykazanej regresji numerycznej.

Osobny korpus bez kontynuacji wykonał 8/8 rekordów tła, wszystkie body zgodne.
Online rozpoczęto po zaobserwowaniu pierwszego ukończenia i opóźnieniu 10 ms;
nie dowodzi to, że następny forward już trwał w chwili nadejścia online.

| Próba bez kontynuacji C2 | Baseline | Mixed | Próg ustalony przed mixed |
| --- | ---: | ---: | ---: |
| Timed odpowiedzi / błędy | 32 / 0 | 32 / 0 | zero błędów |
| P99 nearest-rank | 1251,002 ms | 1265,463 ms | 1261,002 ms |

**FAIL: mixed przekroczył próg o 4,461 ms** (+14,461 ms wobec baseline).
Przy 32 próbkach p99 nearest-rank jest maksimum. Tylko mixed miał watcher health
co 50 ms; baseline nie miał watchera. To czynnik zakłócający, bez rozstrzygnięcia
przyczyny przekroczenia. Nie zmieniano progu ani nie powtarzano próby w celu
ukrycia FAIL. Health pokazał 3 `pause_limit`, 5 ponownie zakolejkowanych rekordów,
0 failure_attempts i failed groups, 0 discarded device ns. Nie zmierzono
bezpośrednio latencji przekazania urządzenia ani przerwania rozpoczętego forwardu.
Ten wynik nie kwalifikuje priorytetu online dla całego profilu lub wszystkich
funkcji Basal.

## Cancel, restart i zasoby

Aktywny cancel rozpoczynał od 512 rekordów i 3 zaobserwowanych ukończeń przed
żądaniem anulowania. Stan terminalny to cancelled; zachowano 4 odpowiedzi
zaakceptowane przed zamknięciem bariery. Każda zgadzała się z online solo.

W osobnym przebiegu własny serwis otrzymał SIGINT przy 512 rekordach i 3
zaobserwowanych ukończeniach. Po ponownym uruchomieniu z tym samym store uzyskano
512/512 sukcesów. Doctor pokazał unaccounted settlements 15 → 0, quota drift 0,
reserved bytes 0, open sessions 0 i pending cleanup 0. To graceful stop/restart,
nie process-kill, power loss, utrata GPU ani failover bazy.

Guard próbkował co 250 ms. Pierwsze pięć szeregów ma 907 próbek, max własnego
workloadu 3774 MiB i min wolnego VRAM 9443 MiB; bez zatrzymania przez guard.
Dodatkowy boundary ma 587 próbek: max własnego workloadu 3870 MiB i min wolnego
VRAM 9347 MiB. Łącznie 1494 próbki. Osobny terminal event boundary to
`telemetry_unavailable` przy 240,322 s, po zakończeniu własnego kontenera przez
jego limit 240 s (exit 124, OOM false, aplikacja raportowała shutdown). Wszystkie
pomiary skończyły się wcześniej. Nie klasyfikujemy tego eventu jako alarmu VRAM. To próbki,
nie rzeczywisty peak alokacji ani dowód bezpieczeństwa dowolnej kombinacji modeli.
Rodzic potwierdził usunięcie własnych kontenerów i powrót pamięci do stanu wyjściowego;
produkcja nie otrzymywała żądań, restartów ani zmian w tej sesji.

## SIGNAL i ograniczenia

Pierwszy `online-r2` jest zachowany jako **SIGNAL błędu harnessu**: 100/200
sukcesów, pozostałe 100 odpowiedzi System One 422 przez pominięty model.
Poprawiono operator, bez zmiany produktu; baseline akceptacji to `online-r2-valid`.
Nie usunięto tej nieudanej próby ani jej odpowiedzi z archiwum.

`compare-r2.json` zachowuje dwa errors_agree=false (`so-score-one-level`, `v2-span`).
Część upstream_endpoint oczekuje pola `upstream_error` z Python reference A,
a Rust baseline używa `error` i `answer_upstream.error`. Są to **SIGNAL kontraktu
wejściowego komparatora**, przy identycznych bajtach A/B; nie dowód rzeczywistej
różnicy z upstream. Nie zmieniano komparatora w tym zadaniu.

GAP: media, grupowanie wielu rekordów, repliki, split, inne modele, Metal runtime
i kompilacja, Intel runtime, długotrwałe nasycenie online, szczyt RSS oraz ścisła
granica alokatora. Ten pakiet nie kwalifikuje SQL Server ani topologii Mongo/HA;
ich wyniki mają odrębne raporty. Nie ma nowej oceny jakości FP32 upstream.

Zamrożony candidate przeszedł default/Intel/CUDA Clippy i Rustdoc oraz fmt,
cargo deny i kontrolę zależności w pracy rodzica. Metal nie był kompilowany.
Snapshot 267 plików przypina źródła użytej binarki; późniejsze uzupełnienie opisu
OpenAPI odpowiedzi upload 408 nie jest częścią zmierzonego snapshotu.
Sanitized compilation summary przypina hashe prywatnych logów; kompilacja nie
zastępuje obserwacji modelu. Nie dodano i nie uruchomiono testów projektu,
frameworków, snapshotów testowych ani bramek CI.

## Dane i odtworzenie

Pakiet ma logiczną ścieżkę `reports/batches-mini-gpu-2026-10-19/` w archiwum.
Zawiera wejścia i rzeczywisty GEMM, byte-exact eksporty, odpowiedzi numeryczne,
aliasowane stany/health, serie zasobów, predeclared limits, canonical parity,
pełne przypięte źródła baseline/candidate, hashe binarek i analizator.
Wagi i binarki nie są redystrybuowane. Surowe identyfikatory, adresy, lokalne
ścieżki, konfiguracje, tokeny, DSN, PIDs i polecenia zdalne pozostają prywatne.
Zasoby mają aliasy, czasy statusów przesunięto wspólną stałą; różnice zachowano.

Stan publikacji: **lokalne, nieopublikowane**. Nie ma potwierdzonego URI HF
ani pobrania autoryzowanego/anonimowego. Docelowy magazyn po osobnej publikacji
to `hf://buckets/itsoltech/basal-rs/`; nie jest to URI opublikowanego archiwum.
Szczegółowa procedura jest w `REPRODUCE.md` wewnątrz pakietu oraz
[MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).

## Integralność lokalnego pakietu

Archiwum: `.cache/measurement-artifacts/batches-mini-gpu-20261019/measurement-data.tar.gz`.
SHA-256: `b78208c4166d477e535a0841013ee68271fb4d7b12c125b7fc73593c3ed79fc7`.
Rozmiar: 3377750 B; 623 pliki danych i źródeł w `SHA256SUMS`,
624 pliki łącznie z samym manifestem. Archiwum zachowuje raport
jako `reports/batches-mini-gpu-2026-10-19/REPORT.md` przed dodaniem niniejszego
akapitu z hashem archiwum, aby uniknąć odwołania hasha do samego siebie.

Potwierdzono lokalną kopię archiwum o tym samym SHA-256, świeże rozpakowanie,
wszystkie wpisy `SHA256SUMS` i identyczny bajtowo wynik ponownej analizy.
To weryfikacja lokalna; wysyłanie do HF oraz pobranie publiczne pozostają niewykonane.
Po rozpakowaniu uruchom `sha256sum -c SHA256SUMS`, a następnie
`python reports/batches-mini-gpu-2026-10-19/analyze.py --measurement-artifacts .`.

## Korekta odtworzenia — właściwe repozytorium wag

Właściwe repozytorium wag to **`Remek/basal-2.0-mini`**, revision
`cb81b32ce89b0b87e62fcdad28f0435e25c953f3`. Poprzedni pakiet błędnie podawał
`itsoltech/basal-2.0-mini` w `REPRODUCE.md` i `provenance/source.json`.
Poprawiono wyłącznie tę nazwę repozytorium oraz słowo „pięć” na „sześć” przy
wyliczeniu sześciu porównań body w `REPRODUCE.md`. Dane pomiarowe, przypięte
źródła, wyniki PASS/FAIL/SIGNAL/GAP i analiza pozostały identyczne bajtowo.

Nowy pakiet do odtworzenia:
`.cache/measurement-artifacts/batches-mini-gpu-corrected-20261019/measurement-data.tar.gz`.
SHA-256: `5844f56ce0dd90d0b1ed6366db70c04764ca12edca5805c29432f3c87a90298d`.
Rozmiar: 3377748 B; 623 wpisy danych i źródeł w `SHA256SUMS`,
624 pliki łącznie z manifestem. Ponownie potwierdzono lokalną kopię
archiwum, świeże rozpakowanie, wszystkie wpisy manifestu i identyczny bajtowo wynik
analizatora. Ten pakiet również jest lokalny i nieopublikowany.

Poprzednie archiwum SHA-256
`b78208c4166d477e535a0841013ee68271fb4d7b12c125b7fc73593c3ed79fc7`
zachowano bez zmian; nowy pakiet zastępuje je na potrzeby odtworzenia z poprawnym
adresem repozytorium wag. Archiwalny `REPORT.md` nadal zachowuje pierwotny opis
obserwacji przed akapitami integralności i niniejszą korektą.

## Niezależna kontrola rodzica

2026-10-19: potwierdzono SHA-256 skorygowanego archiwum `5844f56ce0dd90d0b1ed6366db70c04764ca12edca5805c29432f3c87a90298d`,
wszystkie **623/623** wpisów po nowym rozpakowaniu i identyczny bajtowo wynik
ponownej analizy. Względem poprzedniego pakietu zmieniły się wyłącznie dwa pliki
opisu pochodzenia/odtworzenia i ich manifest; wyniki pomiarów pozostały identyczne.
Weryfikacja lokalna nie oznacza publikacji HF ani kwalifikacji produkcyjnej.
