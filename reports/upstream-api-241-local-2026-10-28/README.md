# API upstream 2.4.1 — lokalna kwalifikacja Pico i Mini

Status: **PARTIAL**. Implementacja A–G1 jest lokalnie zintegrowana po niezależnych
przeglądach i osobnych uproszczeniach. Dostępna kwalifikacja obejmuje CUDA na
RTX 4060 oraz dwa wymienione niżej modele. Nie stanowi akceptacji dodatnich
ścieżek trained thinking ani early-exit 4.5B.

## Tożsamość pomiaru

| Element | Wersja |
|---|---|
| Kod końcowy | `ac7c9bf2d07c675fc48c759f38148d2a06b7af03` |
| Baza sprzed inicjatywy | `729fe47df9ef8117f46b526582941afcc4f97e70` |
| Referencja basal-serve v2.4.1 | `b20f0f9b159f71726459aa98ede1356ab65695fd` |
| Binarka końcowa SHA-256 | `61f39e03a291139fab51deb31fc6e70482d213527aa98ca9c738c8d6e9815f9c` |
| Binarka bazowa SHA-256 | `b01a797d7cfa4f236265e36a8c1311477b067281d1b6697d49e5962f350a48ef` |

Obie binarki: basal 0.1.13, release CUDA, Rust 1.95.0, toolkit CUDA 12.9.1.
Sprzęt: RTX 4060, 8188 MiB VRAM, sterownik 615.78.08. Praca GPU odbywała się
sekwencyjnie. Referencje: Python 3.12.15, PyTorch 2.11.0+cu128,
Transformers 5.17.0, tokenizers 0.23.3. Checkout upstream pozostał niezmieniony.

| Model | Rewizja HF | SHA-256 wag |
|---|---|---|
| `Remek/basal-omni-pico` | `a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9` | `ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9` |
| `Remek/basal-2.0-mini` | `cb81b32ce89b0b87e62fcdad28f0435e25c953f3` | `b902691d00cdc97946f70bd9dccd4988adf06de123fa1865c0f1b316c033d744` |

Mini oznacza tutaj wersję **2.0**, nie alias CLI wskazujący 1.5. Użyto istniejących
lokalnych wag, bez pobierania kolejnych modeli. Mini AUTO używało tej samej
stałej tabeli F16 GEMM po obu stronach porównania: SHA-256
`2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96`.
Autotuning przy `--gemm-table none` nie jest podstawą bitowej kwalifikacji Mini.

## Niezmienione obliczenia — PASS

Wykonano dziesięć par eksportów bazowych i końcowych:

- Pico BF16, orders 1 i 2; Pico F32, orders 1 i 2.
- Mini AUTO ze stałą tabelą, orders 1 i 2; Mini F32, orders 1 i 2.
- Pico BF16 i Mini AUTO: osobny korpus classify/verify/compress/extract, orders 1.

W każdej parze **całe** `bench.jsonl` i `systemone.jsonl` są identyczne bajt w bajt:
**20/20 porównań `cmp` PASS**, bez usuwania pól. Obejmuje to prompty, identyfikatory
tokenów, logity, prawdopodobieństwa, końcowe odpowiedzi i usage.
Korpus podstawowy zawiera 44 pozycje bench oraz 7 żądań API, w tym sześć typów
primitive, PL/EN, Unicode, 255 opcji, odmowę 256 opcji i długi stan. Osobny
korpus zadań zawiera 4 żądania i 5 pytań zadaniowych.

Dowód: `comparisons/exact-canonical.json` oraz `exports/{baseline,final}-*/`.
Zgodność z bazą jest odrębnym wynikiem od zgodności z FP32 upstream.

## Jakość względem świeżej referencji FP32

Każdy model i layout mają nowy eksport upstream z przypiętego v2.4.1,
z jedną kolejnością. Mini korzystało z wag głównych FP32 na CPU, przesyłanych
do CUDA; te przebiegi służą ocenie jakości, nie czasu.

Prompty i tokeny zgadzają się. Zgodność argmax: canonical 44/44 bench i 14/14
odczytów API; każdy nowy layout 44/44 bench i 10/10 odczytów API, osobno dla
każdego modelu. Nie oznacza to równości ciągłych wartości score ani całych answers.

| Model / layout | Maks. różnica skalibrowanego p, bench | API | Próg ≤0,0001 |
|---|---:|---:|---|
| Mini canonical | 0,000013381243 | 0,000006735325 | PASS |
| Mini qfirst | 0,000010788441 | 0,000001922250 | PASS |
| Mini state_twice | 0,000009477139 | 0,000002413988 | PASS |
| Pico canonical | 0,002739176154 | 0,001105844975 | **FAIL** |
| Pico qfirst | 0,003685355186 | 0,001973450184 | **FAIL** |
| Pico state_twice | 0,001135051251 | 0,000313580036 | **FAIL** |

Canonical Pico daje identyczne pliki w bazie i finale, także przy osobnym
uruchomieniu z `BASAL_QWEN_F32_RESIDENT=1` i
`BASAL_QWEN_PACKED_RECURRENCE=1`. Te flagi nie zmniejszyły zaobserwowanych
różnic. Jest to odziedziczony FAIL progu FP32; ta kampania nie naprawia numeryki
Pico i nie ogłasza pełnej zgodności FP32.

Dowód: `comparisons/final-*-f32-vs-upstream.json`, dodatkowe porównania
`*-f32-resident-packed-vs-upstream.json`, `references/` i `exports/`.

## API, admission, logowanie i MCP

| Powierzchnia | Zaobserwowany wynik na obu modelach |
|---|---|
| Domyślne API | Health ready, discovery, natywny endpoint i domyślny TypeSafe; lean odpowiedź noul zachowana |
| F na plain backend | `think=off`, `early_exit=null` i `facts=off` zachowują odpowiedzi; aktywne think/exit, jawne exit=off oraz null think/think_text dają 422 |
| Przeciążenie | Przy max_inflight=4: 16 jednoczesnych żądań daje 4×200 i 12×529, `Retry-After: 1`; liczniki nieujemne, suma ≤4, zaobserwowano main i long lane |
| Anulowanie | 8 gwałtownie zamkniętych połączeń, opróżnienie liczników do zera i niezmieniona odpowiedź po anulowaniu |
| Alias upstream | Natywne odpowiedzi zgodne z aliasem; przypięty oficjalny klient Python `basal.client.Basal` daje te same answers |

Logger: przy max_inflight=128 burst **80/80 odpowiedzi 200**. Każdy przebieg
native dopisał dokładnie **86/86** oczekiwanych końcowych rekordów. Hash zgadza
się z Python `json.dumps([state, questions], sort_keys=True, ensure_ascii=False)`;
sprawdzono whitelist pól i dopisywanie. Awaria zapisu do `/dev/full` daje jawne
500 na native i TypeSafe, a health pozostaje ready. Shutdown kończy się kodem
1 i zachowuje pierwotną przyczynę `No space left on device (os error 28)`.

MCP na rzeczywistych modelach: HTTP initialize, tools/list, decide,
decide_many, błąd narzędzia i stdio initialize/call/EOF. Answers są bitowo zgodne
z bezpośrednim API. Obce Host oraz null/obce/HTTPS Origin odrzucono 403;
brak Host daje 400, zły port Origin daje 403. MCP off daje 404.

Oddzielny skończony pakiet mock bridge-wire: **25/25 scenariuszy,
282 asercje PASS**. Obejmuje prefiks URL, końcowy slash, percent-encoding,
odmowę query/fragment, wersje protokołu 2024-11-05/2025-11-25/2026-07-28,
limit czterech wywołań, anulowanie, numeric 422/529 i cleanup. Opóźniona odpowiedź
po EOF dotarła po około 1,02 s. Zablokowany stdout zakończył proces po
**5,034 s od EOF**. Mock nie zastępuje dodatniego testu inferencji.

Dowód rzeczywisty: `http/pico-native-02`, `mini-native-01`, `*-admission-01`,
`*-mcp-01`, `*-sink-full-01`; transport: `diagnostics/mcp-bridge-wire/`.

## Kontekst i izolacja odmów

Oryginalne modele używają prompt format 2: **11 opcji jest Direct**, nie Groups.
Nie traktowano takiego żądania jako dowodu obsługi późniejszych rund Groups.

Oryginalny Mini: stan 18 016 tokenów przy limicie 8192. Truncate zachowało
2695 tokenów początku i 5392 końca, odrzuciło 9929; najdłuższy wynikowy prompt
noul miał 8167 tokenów. Refuse odrzuciło przekroczenie. Sprawdzono Unicode,
zadania, odmowę evidence po cięciu i przeplatanie dobrych/złych żądań.

Dodatkowe prywatne kopie konfiguracji ograniczają kontekst do 512 tokenów,
z tymi samymi wagami i tokenizerami. Są dowodem mechaniki granicy, nie jakości
zmodyfikowanego modelu. W Mini osobny **diagnostyczny prompt format 1**, litery
ABCDEFGHIJ i `--no-calibration` umożliwia rzeczywiste rundy Groups. Prompt jest
niewytrenowany dla tego checkpointu; nie wyciągamy wniosków jakościowych.
Pico jawnie odmawia formatu 1.

W celowanej próbie późnego overflow pierwsze grupy 6/5/6/5 opcji miały
505/481/505/481 tokenów i mieściły się w limicie 512. Finał dziesięciu opcji miał
601 tokenów. Pełne żądanie zostało odrzucone jako 422 `context_length_exceeded`
na właściwym pytaniu. W serii ośmiu mieszanych żądań dobre dostały 200,
złe 422; nagłówki dobrych odpowiedzi wykazały wspólny batch o rozmiarze 7.
**Późna odmowa nie zmieniła się w batch-wide 500.** Sprawdzono również dwa
grouped pytania w jednym planie i truncate.

Dowód: `http/mini-original8192-*-05`, `*-cap512-*`,
`mini-grouped512-v2-*-05`, `mini-later-group-01`.

Ograniczenia: nie osiągnięto oryginalnej granicy Pico 262 144 tokenów.
Przy refuse odrzucenie może nastąpić przed batchowaniem, więc taki przypadek
nie dowodzi wspólnego wywołania silnika. Nie zmierzono osobno wszystkich
budżetów refuse względem pośredniej bazy fazy D `1da2dd9`; istnieje dowód
statyczny i bitowy względem oryginalnej bazy całej inicjatywy.

## Cztery SDK i konfiguracja CPU

Lokalne, niepublikowane pakiety: Python/TypeScript/C# 0.7.0, Rust 0.8.0;
native OpenAPI 0.4.0. Użyto zainstalowanego wheel i tarball, zbudowanych
artefaktów C#/Rust oraz istniejących adapterów live.
Środowisko SDK: Python 3.14.8, Node 26.8.2, .NET SDK 10.0.401 / runtime 10.0.12
z jawnym roll-forward Major dla przykładu net8. Metadane pakowania i sumy
artefaktów zachowano w `diagnostics/sdk-packaging/`.

Oba audyty SDK: **69 obserwacji odpowiedzi, 7 discovery i 14 błędów, PASS**;
wszystkie 7 trybów SDK kończy się kodem 0, dodatkowo 6 trybów named.
Audyt porównuje obecność opcjonalnych pól i wartości binary64 z bezpośrednim
API; wyłącza jedynie zmienny `think.ms`. Dodatniego think tutaj nie było.

Korpus zawiera 9 przypadków. Dwa serwerowe skróty wejść nie mają odpowiednika
w ergonomicznych konstruktorach SDK: skrócona macierz kosztów act i tablicowe
criteria choice. Zachowano nieudane próby adapterów, a w **nowym** korpusie
zastosowano jawną macierz kosztów i mapy criteria. Identyczne znormalizowane
body przesłano bezpośrednio i przez SDK. Nie zmieniono implementacji SDK.
SHA-256 korpusu: `1ce76314d7308a962795f80f9bf8d9bdbf01a2b19687bbd552fd906dede7fe8f`.

Dowód: `sdk-ready/runs/pico-final-03`, `sdk-ready/runs/mini-final-01`.

Pakiet CPU: **124 wywołania binarki i 16 kontroli wyjść**, bez timeoutów
i niewyjaśnionych różnic. Obejmuje prepare-only, CLI/YAML precedence,
layout, kalibrację auto/explicit/off/strict, odmowy identity/revision/order
mismatch, granice tau/band/truncate_head oraz odmowy aktywnego F przed wagami.
Rodzic zweryfikował 645 sum plików tego pakietu i 249 sum pakietu MCP.
Dowód: `diagnostics/config-cpu/`. Prepare-only nie jest inferencją ani
dowodem działania fingerprintów w uruchomionym serwerze.

## Wydajność

Wyniki i uzupełniające obserwacje runtime znajdują się w
[RESULTS.md](RESULTS.md). Metodę i lokalne odtworzenie opisuje
[REPRODUCE.md](REPRODUCE.md).

## Granice akceptacji

- **FAIL:** próg FP32 Pico, odziedziczony także przez canonical sprzed inicjatywy.
  Zgodne decyzje i niezmienność bitowa nie zastępują tego progu.
- **GAP:** dodatnie trained thinking i early-exit wymagające checkpointów 4.5B:
  ładowanie heads, tokeny, numeryka, tail skip, off-bitwise na tym checkpointcie,
  pamięć, anulowanie i czas. Lokalnych Pico/Mini nie można użyć w ich zastępstwie.
- **GAP:** Metal, Gemma4/media, rzeczywisty OOM/lost device i pełny oryginalny
  kontekst Pico. Nie wywoływano celowego wyczerpania pamięci.
- **GAP:** dodatnie durable background execution bez zmierzonego profilu
  admission; wygenerowany fingerprint nie jest kwalifikacją budżetu urządzenia.
  Pico dodatkowo nie startuje z Batch API z powodu pustego BOS; ten sam błąd
  odtworzono na oryginalnej bazie. Szczegóły w RESULTS.md.

Wyniki nie uzasadniają pełnej akceptacji całej inicjatywy ani publikacji.
Nie wykonano push, release, tagowania, wdrożenia, chmury ani wysyłania danych.
Archiwum pozostaje lokalne; zawiera prywatne ścieżki w surowych logach i wymaga
osobnego przygotowania przed jakąkolwiek publikacją.
