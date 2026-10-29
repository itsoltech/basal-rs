# Upstream API 2.4.1 — końcowe sprawdzenie na dwóch GPU

Zakres REQ-001–013 jest zaimplementowany i zamknięty z opisanymi niżej
ograniczeniami oraz wcześniej zaakceptowaną różnicą thinking względem vLLM.
Przebiegi wykonano 2026-10-29 na RTX 4060 i RTX 6000 Ada Generation.
Nie oznacza to pełnego parytetu vLLM, kwalifikacji wszystkich konfiguracji
ani oceny prawdziwości odpowiedzi modelu.

## Wersje i wejścia

| Element | Tożsamość |
|---|---|
| Końcowy kod aplikacji | `4d9a9c0681feaa38ea7922822962f33e695b1fb7` |
| Drzewo tego commitu | `b44771fcae9995b455a26397765fb5c496384fbb` |
| Końcowa binarka CUDA, użyta na obu GPU | SHA-256 `555893900360ab2d7892db21cde4d0d111cff2544803d5d63d01751029a0e8c2` |
| Pierwsza binarka kampanii, kod `f9f9156` | SHA-256 `9648d47279ba9c80ed98a04e01ed8d237a8a55703d639188f46b9a9f2187fac5` |
| Toolchain | Rust 1.95.0, CUDA 12.9.1; build release offline, locked |
| `Cargo.lock` | SHA-256 `9d44123bc164b9d7fc9229ed7934f060817dc0fc50d371b401abeb0bd9749154` |
| `rust-toolchain.toml` | SHA-256 `24ef3b9d3edbd850aa386cb0a98e10450b0030991a4537cb359f54d49dbbb33a` |
| RTX 4060 | 8188 MiB raportowanej pamięci, Mini 2.0 i Omni Pico |
| RTX 6000 Ada Generation | 49140 MiB raportowanej pamięci, Mini 2.0 oraz modele 4.5B z thinking i wytrenowanymi głowicami early-exit |

Między `f9f9156` a końcowym kodem zmieniły się dokumentacja i lokalny cykl
życia batchy w CLI. Źródła GPU, core, vendor, SDK i kontraktów są identyczne.
Raport rozróżnia pomiary obu binarek; przeniesienie wyniku dotyczy wyłącznie
niezmienionych ścieżek. Późniejszy commit samego raportu nie zmienia binarki.

Podstawowe wejścia są w `reports/upstream-api-241-final-2026-10-29/inputs/`
wewnątrz prywatnego archiwum. SHA-256:

| Wejście | SHA-256 |
|---|---|
| `bench.jsonl` | `bc4cfe8cc6a2786d2f832b25c4aeb632668b4a5e5811bc66ccc6689581f807ed` |
| `systemone.jsonl` | `3dc27c6b16a2512a350cdfc7690ed467b54949d79e69fd054de31f59483e31ea` |
| `cases.jsonl` | `dc78bbcfa704556e2741ba50f2c842e873ed6c88f43cb55f135c137e670131cf` |
| `layout-cases.jsonl` | `ad42c01935c5fb62b86f6202dc651a9281b0a2e542280a8a352f7845bdda794b` |
| Sześć rekordów background Mini | `0e21fd5ff69656282d2fd8886de3f2390258b5f77bae4d9717441c6328927f86` |

Surowe żądania trained HTTP, wariantów batcha i restartu są zachowane wraz
z odpowiedziami i SHA-256 każdego pliku. Wagi nie są częścią archiwum.
Mini pochodzi z `Remek/basal-2.0-mini@cb81b32ce89b0b87e62fcdad28f0435e25c953f3`;
SHA-256 jego wag to `b902691d00cdc97946f70bd9dccd4988adf06de123fa1865c0f1b316c033d744`.
Rodzic porównał sumy ośmiu plików snapshotu Mini po transferze;
zapis tego porównania nie jest częścią archiwum.
Modele 4.5B i referencje pochodzą z przypiętych kampanii
[trained](../upstream-api-241-trained-2026-10-28/README.md) i
[thinking](../upstream-api-241-thinking-rebuild-2026-10-28/README.md).
Nie pobierano ponownie ani nie zmieniano checkpointów upstream.

## Macierz wymagań

PASS oznacza spełnienie wymagania w opisanym zakresie, a nie sprawdzenie
każdej platformy lub możliwego błędu sprzętowego.

| REQ | Wynik | Dowód i granica |
|---|---|---|
| 001: błędy API | PASS w badanym zakresie | Pierwsza binarka, kod niezmieniony: oba modele 422, przeciążenie 529 z Retry-After i 500 po ENOSPC loggera. Odmowy 422 thinking/early-exit także na końcowej binarce. Mapowanie rzeczywistego OOM na 503 pozostaje sprawdzone źródłowo. |
| 002: running/queued w health | PASS | Pierwsza binarka, kod niezmieniony: admission obu modeli 11/11 kontroli, liczniki pod obciążeniem oraz drain po anulowaniu połączeń. |
| 003: layouty i wybór modelu | PASS dla tekstu | Kanoniczne i alternatywne layouty Mini/Pico zachowują zapisane eksporty bitowo. Odmowa alternatywnego layoutu mediów jest sprawdzona źródłowo. |
| 004: kalibracja i tożsamość | PASS | Niezmieniona implementacja i wcześniejszy pakiet CPU, bieżące manifesty eksportów oraz discovery SDK; szczegóły poprzedniego pakietu w raporcie local. |
| 005: dialekt upstream i klienci | PASS | Pierwsza binarka, kod niezmieniony: oficjalny klient Python na obu modelach; niezmieniony `basal-mcp` 2.4.1: 8/8 kontroli, initialize/tools/decide/decide_many, EOF exit 0. |
| 006: usage, lane, branches | PASS | Bieżące HTTP i audyty czterech spakowanych SDK na obu modelach. |
| 007: orders auto i konfiguracja | PASS | Eksporty orders 1/2 bitowo równe poprzednim; wcześniejsze próby wymuszenia auto zachowują ważność dla niezmienionego kodu; zmiana tau zmienia fingerprint w próbie K1. |
| 008: overflow | PASS w badanych limitach | Pierwsza binarka, kod niezmieniony: oba modele z limitem 512, Mini z oryginalnym 8192 — refuse/truncate; późniejsze grupy 7/7. Jedna obserwacja współbatchowania przy Mini/refuse/512 pozostaje GAP. |
| 009: decision log | PASS | Końcowa binarka: po 86 rekordów i niezależna kontrola hash/whitelist/append dla Mini i Pico. Pierwsza binarka (kod loggera niezmieniony): ENOSPC daje jawne 500 i shutdown exit 1. |
| 010: MCP HTTP/stdio | PASS | Końcowa binarka: po 13/13 kontroli Mini i Pico, odmowy Host/Origin, błędy narzędzia i zamknięcie stdio. |
| 011: thinking | PASS z zaakceptowanym odstępstwem | Ada, końcowa binarka: 73/73 kontrole; 42/42 odpowiedzi równe pierwszej serii tej kampanii. Parytet vLLM pozostaje FAIL, zgodnie z wcześniejszą akceptacją. |
| 012: early-exit | PASS w badanej konfiguracji | Ada: 42/42 kontrole; 41/41 odpowiedzi identyczne z pierwszą serią; 32/32 w progu 1e-4 względem zapisanej referencji upstream. |
| 013: docs, kontrakty, SDK, raport | PASS | Zaktualizowane opisy kwalifikacji, cztery faktycznie spakowane SDK, przypięty kod/binarka/wejścia i ten raport. |

Szczegóły niezmienionych kontraktów i wcześniejszych dowodów:
[local](../upstream-api-241-local-2026-10-28/README.md),
[remediation](../upstream-api-241-remediation-2026-10-28/README.md).

## Wyniki lokalne — RTX 4060

Pierwsza binarka kampanii wyeksportowała dziesięć konfiguracji Mini/Pico:
domyślną precyzję i FP32, orders 1/2 oraz zadania. Wszystkie 20 całych plików
`bench.jsonl`/`systemone.jsonl` jest identycznych bajtowo z G2.
Końcowa binarka powtórzyła kanoniczny orders 1 obu modeli (4 pliki) oraz
FP32 `qfirst`/`state_twice` obu modeli (8 plików): wszystkie identyczne.

Kanoniczne porównania FP32 pierwszej binarki z zachowanym upstream, po 44 przypadki bench
i 14 API na model, dały zgodny argmax 58/58 i zero różnic tokenów.
Największe różnice skalibrowanych prawdopodobieństw:

| Model | Bench | API |
|---|---:|---:|
| Mini | 0.000013381242752075195 | 0.000006735324859619141 |
| Pico | 0.000002041459083557129 | 0.0000045299530029296875 |

Na końcowej binarce każdy model przeszedł native 21/21 i MCP 13/13.
Każdy burst 80 żądań dał 80 HTTP 200 i identyczne odpowiedzi.
Python, TypeScript, C# i Rust z faktycznych paczek przeszły dla każdego modelu
69 obserwacji, 7 discovery oraz 14 oczekiwanych błędów.

Skończony loadtest Mini na pierwszej binarce obejmował 3160 mierzonych żądań
oraz 44 warmup: zero błędów i zero różnic odpowiedzi. W fazach concurrency
1/4/16/64 obserwowano odpowiednio 42.05/49.70/61.35/70.91 żądań/s,
a p95 29.93/90.05/273.46/947.18 ms. To wynik konkretnego harmonogramu
i korpusu, bez deklaracji SLO lub przyspieszenia względem innej wersji.

## Wyniki trained — RTX 6000 Ada

Pierwsza seria kampanii: 248/248 kontroli w pięciu świeżych procesach.
Restartowe porównania thinking 42/42 i early-exit 41/41 były dokładnie równe
po usunięciu wyłącznie `usage.latency_ms` i `answers.*.think.ms`.
Końcowa binarka przeszła thinking 73/73, early-exit 42/42 i zwykły control 18/18;
jej odpowiedzi thinking 42/42 i heads 41/41 są równe pierwszej serii.

Early-exit jest także bitowo równy zapisanym wynikom sprzed J1–J4 (41/41).
Porównanie z upstream daje 32/32 w progu 1e-4, maksimum
`0.000004112071362083469`. Wcześniejszy profil wykonania potwierdzający
pomijanie warstw pozostaje dowodem historycznym; nie uruchamiano go ponownie
w tej kampanii.

Thinking `inherit` FP32 zachował 19/19 wyników J2. Względem niezależnej
diagnostyki Torch FP32: 19/19 w progu 1e-4, maksimum
`0.0000011026859283447266`, notatki 9/9 dokładnie równe.
To zgodność implementacji na tym korpusie, bez rozstrzygnięcia jakości
semantycznej odpowiedzi. Historyczny FAIL względem vLLM i jego
[akceptacja](../upstream-api-241-gemm-diagnostic-2026-10-29/ACCEPTANCE.md)
pozostają bez zmian. Nie uruchamiano vLLM ponownie.

Próby przerwania połączenia odzyskały obsługę żądań. SIGTERM zakończył oba
serwery kodem 0; zaobserwowany drain wyniósł 363.93 ms dla heads i 2920.19 ms
dla thinking. To dwie obserwacje, nie gwarantowany czas zamknięcia.

## Batch: poprawki cyklu życia i numeryka

K1 ma niezależny przegląd poprawności i osobny etap upraszczania Opus 5.5.
Tryb `local` odświeża katalog po zmianie fingerprintu; stare batche zachowują
przypiętą rewizję. Lokalna reprodukcja starej binarki: zmiana tau pozostawiała
rewizję 1 i `lists_this_host=false`. Końcowa binarka: rewizja 1 → 2,
nowy batch waliduje, stare/nowe przypięcie SQLite pozostaje 1/2.
Pico batch kończy się teraz jawnie `failed/unbounded_feature`; health nadal
raportuje brak wsparcia background. Nie dodano wykonywania batchy Pico.

Mini na końcowej binarce przeszedł na obu kartach trzy rundy po sześć rekordów,
sześć przeplatanych żądań online, anulowanie nieukończonego batcha i restart
z aktywną rezerwacją. Anulowanie zachowało poprawne wyjścia 3/60 lokalnie
i 4/60 na Ada. Po SIGTERM SQLite nadal wskazywał `in_progress` bez
`terminal_at`; po restarcie uzyskano 60/60, bez duplikatów, z poprawnymi ID
i odpowiedziami równymi online po usunięciu tylko `latency_ms`.
Po drain rezerwacje wynosiły zero, pula hosta była cała wolna, a liczniki
failed groups/unavailable lanes/quarantine pozostały zerowe lub false.

Prywatny runner przeszedł dwa przeglądy. Lokalny background użył wersji
`2d5bc312580b65f11beb5309ae31b38db1ec5f5a549f02d002f6b431647fd7f5`, sprzed
ostatniej jednozdaniowej asercji `cancel.incomplete`; rodzic sprawdził ją
na zapisanych danych (3/60 i istniejący, sprawdzony output). Ada użyła już
wersji z asercją, SHA-256
`7257022450617477afd1b4c1833cc7699f6129455deeca74e760ba467315090a`.

Profil background był laboratoryjny: solo, device/reserve 905969664 B,
granularność 2097152 B, 2252298 ns/token, host pool 536870912 B,
model max_batch_tokens 8192. Te same liczby przeniesiono na Ada.
`qualified=true` było deklaracją próby, bez produkcyjnej kwalifikacji profilu
i bez dowodu rzeczywistego maksimum alokatora lub zachowania przy OOM.

Tabele Mini: lokalnie digest
`2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96`,
Ada `cf5ba1a8a57955a2c97e10cfe0ecdb814ac3413a86a9d59e44aa290e1b776575`
(współdzielona tabela 1.5-mini z builda).
Zwykła 4.5B na Ada użyła współdzielonej tabeli 1.5-4.5B z builda:
`6d90b8cbe6a87fc40ea45f36a6660c834b9e7823b79553ab1ab75e1231b7d6c2`.

Kontrola niezależności od batcha obejmowała identyczne pytania 2/15 w jednym
żądaniu, grupę, ruch concurrent i powtórzenie. Wyniki obu binarek rozdzielono:
Mini/4060 — pierwsza, 4.5B/Ada — końcowa; źródła obliczeń są niezmienione.

| Konfiguracja | Mini / RTX 4060 | 4.5B / RTX 6000 Ada |
|---|---:|---:|
| Default F16, tabela | 46/46 PASS | 46/46 PASS |
| F16 pinned | 46/46 PASS | 46/46 PASS |
| F16 bez tabeli, diagnostyka | 35/46 kontroli FAIL | 3/46 FAIL |
| FP32, `gemm_table none` | 5/46 FAIL | 34/46 FAIL |
| FP32, `gemm_table auto` | 5/46 FAIL | 34/46 FAIL |

Wszystkie serwery tych prób zakończyły się kodem 0, bez błędu harnessu.
FP32 auto i none dają 25/25 równych odpowiedzi na każdej karcie.
FP32 nie korzysta z tabel GEMM. Gwarancja niezależności batcha dotyczy
ścieżki z invariant F16/BF16, nie diagnostycznego FP32 lub braku tabeli.
FAIL pozostają w danych; nie zaokrąglano prawdopodobieństw i nie zmieniano defaultów.

Nsight na Ada, przez `basal decide` z tymi samymi flagami FP32 co warianty
serwera, potwierdził zmianę kerneli dla solo i dwóch identycznych pytań.
Zapisane prawdopodobieństwa z CLI są bitowo równe wynikom serwera.
Trzy wywołania GEMV w solo zastępują trzy SGEMM w duplicate-2
(`ampere_sgemm_128x32_sliced1x4_tn` raz i `ampere_sgemm_32x32_sliced1x4_tn`
dwa razy). Jest to bezpośrednia obserwacja zależności wykonania od kształtu.
Pasuje do hipotezy innej kolejności sumowania, ale sam profil nie przypisuje
każdej różnicy prawdopodobieństw do konkretnego kernela.

## Kontrole statyczne, zasoby i ograniczenia

Końcowy K1 przeszedł fmt, Clippy i prywatny rustdoc dla default/Intel/CUDA.
Uproszczenie K1 zmieniło tylko komentarz i dokumentację; ponowiono fmt.
Rodzic wykonał również `cargo deny --locked --all-features check
--hide-inclusion-graph`: advisories/bans/licenses/sources OK.
Pełny release CUDA build zakończył się kodem 0. Logi autora i kompilacji są
w prywatnych dowodach. Nie dodawano frameworków testowych ani bramek CI.

Próbkowanie NVML co około 50 ms wskazało maksymalnie 3662 MiB lokalnie
podczas serii K1, 19114 MiB dla thinking i 9890 MiB dla heads na Ada.
To pamięć całego urządzenia widoczna w próbkach, nie maksimum alokatora;
krótkie piki mogą umknąć. Moc na 4060 była N/A. Brak kwalifikacji energii,
długiego soak i ogólnego braku wycieków.

Nie wykonywano macOS/Metal, pełnego kontekstu Pico 262144, awarii urządzenia,
rzeczywistego OOM ani zamknięcia workera przez fault injection.
Media-layout refusal pozostaje statyczny. Split thinking, wszystkie ścieżki
błędów i przerwań oraz pozostałe konfiguracje nie zyskują kwalifikacji z tej
próby. Nie oceniano semantycznej trafności modelu na oznaczonym zbiorze.
Opisane wcześniej +200/+400 B accounting pamięci planera pozostaje jawne.

Nieudane wywołania diagnostyki i runnerów zachowano, lecz nie zaliczono do PASS.
Pierwsza końcowa sesja Ada wykonała thinking, heads, control, lifecycle i
batch numerics 4.5B; runnery wierszy FAIL zakończyły się kodem 1 bez błędu
harnessu, pozostałe etapy kodem 0. Tylko etap Nsight zakończył się kodem 2
z powodu brakującego pola `model` w wejściu CLI, stąd kod 2 całej sesji;
produkcja została przywrócona. Nowa sesja z poprawnym wejściem wykonała oba
profile i background, wszystkie etapy exit 0.
Oryginalna usługa produkcyjna została ponownie uruchomiona i sprawdzona jako
healthy/HTTP 200. Procesy kampanii zakończono; nie wdrożono nowej produkcji.

## Prywatne archiwum i odtworzenie

Archiwum jest lokalne i nieopublikowane. Surowe dane zawierają prywatne
ścieżki i metadane sesji; nie nadają się do publicznego HF bez osobnego
przygotowania. Nie podajemy fikcyjnego URI publikacji.

Lokalizacja względna od repozytorium:
`.cache/measurement-artifacts/upstream-api-241-final-2026-10-29/measurement-data-private.tar.gz`.
SHA-256 archiwum:
`1c98057b937bb671c4388aeea3c2dadeb2de2c024b72c1a15b7e1e4ce8c908b6`.
Sprawdzono sumę osobnej kopii archiwum, rozpakowano ją do nowego katalogu
i potwierdzono wszystkie **3190/3190** wpisów `SHA256SUMS` (exit 0).
Oryginalne archiwum i kopia weryfikacyjna pozostają lokalnie.

Ścieżki logiczne w archiwum:

| Ścieżka | Zawartość |
|---|---|
| `reports/upstream-api-241-final-2026-10-29/` | Wejścia, eksporty, porównania, HTTP/MCP, SDK i batch numerics lokalne |
| `reports/upstream-api-241-final-2026-10-29/evidence/` | Prywatne komendy, logi, wersje runnerów, lifecycle, wyniki Ada i profile |
| `reports/upstream-api-241-final-2026-10-29/references/` | Kopie wcześniejszych eksportów i referencji użytych do porównań |
| `SHA256SUMS` | Sumy każdego pliku ze ścieżkami względnymi |

Rozpakować do nowego katalogu i wykonać `sha256sum -c SHA256SUMS` w jego
katalogu głównym. Porównania eksportów odtwarza `basal compare --a <ref>
--b <export> --out <nowy-plik>`; bitową zgodność całych JSONL sprawdza `cmp`.
Samego exit 0 z `basal compare` nie należy utożsamiać z progiem jakości:
sprawdzić różnice, argmax i tokeny w JSON.
Komendy runtime, konfiguracje i źródła prywatnych runnerów są w `evidence/`;
ich zapisane ścieżki trzeba zastąpić miejscem rozpakowania, modeli i binarki.
Odtworzenie GPU wymaga dokładnej wersji kodu, CUDA i wag oraz autoryzowanej
sesji. Dane można analizować bez ponownej inferencji.
