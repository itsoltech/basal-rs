# SDK: Basal 2.0 i Omni — 2026-10-17

Status: **PASS w zmierzonym zakresie**. Python, TypeScript, .NET i Rust obsługują
nowe funkcje istniejącego API basal-rs. Końcowe cztery serie dały **178/178
zgodnych odpowiedzi HTTP SDK**, obejmujących **1154 odpowiedzi na pytania**,
**28/28 kontroli discovery** i **56/56 kontroli błędów**. Brak różnic w
porównywanych polach odpowiedzi i metadanych użycia. Liczby porównano po bitach
IEEE-754 binary64, bez tolerancji.

## Zmiany i wersje

Python, npm i NuGet: **0.2.0**; Rust: **0.3.0**; niezależny opis kontraktu: **0.2.0**.
Paczki zbudowano lokalnie; nie opublikowano nowych wersji w rejestrach.

- Choice, score i act przyjmują do 255 wyników. Możliwości konkretnego modelu
  nadal sprawdza serwer.
- Tool: definicje OpenAI Chat/Responses, MCP i Anthropic, dodatkowe wybory,
  ranking, pojedynczy lub wielokrotny wybór oraz odroczenie decyzji. SDK nie
  wykonują narzędzi. Pominięte `extra` zachowuje domyślne wybory serwera;
  jawne `[]` je wyłącza.
- Multi: threshold, exact, IoU i F1, limity liczności, exclusive i fallback.
  `threshold` w wyniku jest opcjonalny; pozostałe cele mają `expected_score`.
- Żądania: `lang`, uporządkowane media inline, `detail`, helpery bajtów,
  base64 i data URL. Helpery nie pobierają adresów URL. Lista mediów wymaga 1–8 elementów.
- Odpowiedzi: kalibracja `text`, deferral, metadane mediów,
  `longest_prompt_tokens`, `second_order_branches`, możliwości modelu i temperatury.

Zmiana typu `threshold` i rozszerzenie publicznych struktur Rust wymagają
uwagi przy aktualizacji kodu klientów. Zachowano dotychczasowe style wywołań
oraz dostęp do surowych odpowiedzi.

## Przegląd Opus i wersje źródeł

Po implementacji `c294a82` wykonano żądany przegląd **Claude Opus 5.5, xhigh**
w osobnym checkoutcie. Włączono jego cztery commity jako `8b8ea39`, `27437e2`,
`90a3e2c` i `d83132e`. Poprawiono domyślne `extra` w Pythonie, możliwość jawnego
pustego `extra` w Rust, typowanie jawnych pól tool w .NET, błąd null w rankingu
.NET, walidację pustych mediów i dokumentację. Rust współdzieli teraz parser
wyników multi i multi-tool. Przegląd nie wykonywał inferencji GPU; opisane
poniżej wywołania przeprowadzono po włączeniu poprawek.

Źródła SDK: `d83132e`; runnery: `46b9c19`. Serwer CUDA zbudowano z `c294a82`;
źródła silnika i vendor nie zmieniły się między nim a sprawdzonym SDK.
SHA-256 binarki serwera:
`58f02a8ca1ab68e695b56093c66b3bc0e72de578053da519171fab708bb34504`.
Archiwum zawiera również hashe wykonanych artefaktów SDK.

| Model | Przypięta rewizja checkpointu |
|---|---|
| basal-2.0-mini | `cb81b32ce89b0b87e62fcdad28f0435e25c953f3` |
| basal-omni-1.0-mini | `73c993dc3d9ab3cb8887ac0b1ee1cf1c92783240` |
| basal-omni-1.0-max | `e538228c4f3ffd96dde298a073ed10247da0d637` |

Pełne źródła i sumy plików modeli: `model-sources.json` w archiwum.

## Sprzęt i metoda

NVIDIA RTX 6000 Ada, 48 GiB, CUDA 12.9.1, sterownik 580.159.03, Rust 1.95.0.
Serwer używał `dtype=auto` na GPU, order 1 oraz dodatkowej serii `orders=auto`.
Wyłączono cache stanu, prefix cache i autotuning GEMM. Limit kontenera:
8 CPU i 64 GiB RAM. Jeden model, jeden klient, sekwencyjne żądania.

Klienci: Python 3.15.0 z zainstalowanego wheel; Node 26.8.2 z zainstalowanego
lokalnego tarballa npm; .NET SDK 10.0.401 z targetem net8.0 uruchomionym na
zainstalowanym runtime .NET 10 przez roll-forward; Rust 1.95.0, przykład release.
.NET runner odwołuje się do projektu biblioteki; osobno zbudowano pakiet NuGet.

Dla każdego przypadku najpierw wysłano bezpośrednie żądanie HTTP do tego samego
serwera. Potem wykonano je przez Python sync/async/schema, TypeScript ESM/CJS,
.NET map i Rust map. Dodatkowe wywołania nazwanych pytań obejmowały TS ESM/CJS,
.NET inline/prepared/explicit Add i makro Rust `questions!`. Sprawdzono także
kompilowaną inferencję typów nazwanych odpowiedzi.

Audyt porównuje wszystkie zwracane klucze odpowiedzi, decyzje, prawdopodobieństwa,
confidence, score i legendę, koszty act, kalibrację, deferral, rankingi, cele
multi, expected score, diagnostykę constraints, model oraz znane pola usage.
Kontrola discovery obejmuje name, prompt_format, orders, media i temperatury.
Nie porównuje tekstowej serializacji JSON, latency ani nieznanych pól.
Rust zapisuje sprawdzone projekcje typów i usage, a nie pełną serializację raw DTO.

## Wyniki

| Seria | Przypadki wejściowe | Odpowiedzi HTTP SDK | Odpowiedzi na pytania | Discovery | Kontrole błędów | Wynik |
|---|---:|---:|---:|---:|---:|---|
| Basal 2.0 mini, order 1 (`basal2-mini-r2`) | 3 | 27 | 201 | 7 | 14 | PASS |
| Basal 2.0 mini, auto (`basal2-mini-auto-r1`) | 3 | 27 | 201 | 7 | 14 | PASS |
| Omni mini (`omni-mini-r1`) | 8 | 62 | 376 | 7 | 14 | PASS |
| Omni max (`omni-max-r1`) | 8 | 62 | 376 | 7 | 14 | PASS |
| **Razem, końcowe serie** | **22** | **178** | **1154** | **28** | **56** | **PASS** |

Przypadek tekstowy EN zawiera 23 pytania: choice o 10/11/26/27/255 opcjach,
score 255, act 27/255, noul, wszystkie cztery cele multi i constraints, cztery
formaty definicji tool, ranking, deferral, multi-tool, domyślne `extra` i jawne
puste `extra`. Osobne przypadki sprawdzają język PL oraz nazwane API.

Omni dodatkowo przetwarza rzeczywiste zdjęcie COCO i mowę FLEURS: obraz default,
obraz high, audio oraz oba porządki obraz/dźwięk. Wykorzystano helpery bytes,
base64 i data URL. Odpowiedzi act z mediami miały kalibrację `text`.
Przypadek tool deferral zwracał `tool=null`, `deferred=true` i metadane odmowy.
Seria auto zwróciła `second_order_branches=11` dla tekstu EN oraz 0 dla
pozostałych dwóch przypadków; wszystkie tryby SDK zachowały te wartości.

Kontrole błędów w każdym z siedmiu głównych trybów SDK: typowany błąd HTTP 422
przy nieznanym modelu oraz lokalne odrzucenie pustej listy mediów. To nie jest
pełna kampania wszystkich błędnych wejść i problemów sieciowych.

Zachowano również wcześniejszą poprawną serię `basal2-mini-r1`: 27 odpowiedzi
HTTP / 201 odpowiedzi na pytania / 7 discovery, przed dodaniem kontroli błędów.
Całe archiwum zawiera zatem 205 obserwacji odpowiedzi HTTP SDK i 1355 odpowiedzi
na pytania. Nie doliczamy tej powtórki do końcowego wyniku powyżej.

## Nadzór i ograniczenia

Temperatura maksymalna: 62°C; VRAM maksymalnie 25 358 MiB. Limit mocy 300 W
pozostał bez zmian. Wentylator tymczasowo ustawiono na 100%, a po zakończeniu
potwierdzono powrót do automatu. Wszystkie cztery zadania serwera zakończyły się
kodem 0. Nadzór nie wykrył nowych Xid, OOM, błędów sprzętu ani krytycznej temperatury;
cztery istniejące usługi zachowały zdrowie. Własny kontener, katalog pomiarowy
i tunel usunięto po zachowaniu danych lokalnych.

Zachowano dwa nieudane etapy przygotowania: brak pobranego `chat_template.jinja`
oraz odmowę nadpisania starego logu przez następny start nadzoru. Żaden nie
uruchomił inferencji. Uzupełniono przypięty szablon i użyto nowych identyfikatorów
zadań; oba zakończenia przywróciły automat wentylatora. Szczegóły w `session.json`.

To potwierdzenie kontraktu SDK względem bezpośredniego basal-rs, **nie nowy pomiar
jakości względem FP32 ani przewagi szybkości nad upstream**. Nie zmieniano
obliczeń silnika. Wyniki wcześniejszej kampanii jakości i wydajności są
[w osobnym raporcie Omni](../omni-performance-ada-2026-10-16/README.md).
Nie powtarzano tutaj modeli tekstowych 4.5B/max, pełnej drabiny kontekstu,
obciążenia wielu klientów ani runtime Metal.

Upstream 2.1 classify/verify/compress/extract i nowe architektury pozostają poza
zakresem tej aktualizacji. Opcjonalny parametr tool `description_chars` pozostaje
w surowych żądaniach. Poprawność `max_error` dla trybu narzędzi, w tym wymóg
0 < x < 1 i brak łączenia z multi, sprawdza serwer. Rust ma zamknięte enumy dla
bieżących wartości media/detail/objective; nowe wartości serwera mogą wymagać
kolejnej aktualizacji SDK.

## Kontrole kodu i paczek

PASS: workspace CPU Clippy release/all-targets z `-D warnings`, Rustdoc release
z prywatnymi elementami i `-D warnings`, fmt, cargo-deny all-features, graf
wyjątku paste, wersje SDK, Python sdist→wheel, npm ESM/CJS i ścisła kompilacja
runnera, .NET Release build/pack z zerem ostrzeżeń oraz Rust release runner.
OSV nie znalazł znanych podatności w dziewięciu przypiętych paczkach Python/npm.
Cargo-deny zachował istniejące ostrzeżenia o duplikatach wersji; advisories,
bans, licenses i sources zakończyły się OK. Serwer przeszedł build CUDA.
Nie dodano frameworków ani bramek testowych do CI.

## Archiwum i odtworzenie

| Pole | Wartość |
|---|---|
| URI | `hf://buckets/itsoltech/basal-rs/measurements/sdk-basal2-omni-2026-10-17/a366390eb67bf366da57e527046b896a6fb828020c52514bbab84e85d995b780/measurement-data.tar.gz` |
| SHA-256 | `a366390eb67bf366da57e527046b896a6fb828020c52514bbab84e85d995b780` |
| Rozmiar | 1 012 253 B |
| Weryfikacja | Pobranie z autoryzacją i anonimowo: SHA archiwum oraz 118/118 sum plików zgodne; ponowny audyt pięciu pobranych serii PASS |

Dane są pod `reports/sdk-basal2-omni-2026-10-17/`: `inputs/`, `runs/`,
`summary.json`, `provenance.json`, `model-sources.json`, `checks.json`,
`session.json`, odpersonalizowana telemetria/zdarzenia i `method-sources/`.
Przykład pobrania do nowego katalogu:

```bash
BASAL_SDK_DATA=.cache/measurement-download/sdk-basal2-omni-2026-10-17
mkdir -p "$BASAL_SDK_DATA"
hf buckets cp hf://buckets/itsoltech/basal-rs/measurements/sdk-basal2-omni-2026-10-17/a366390eb67bf366da57e527046b896a6fb828020c52514bbab84e85d995b780/measurement-data.tar.gz "$BASAL_SDK_DATA/measurement-data.tar.gz"
echo "a366390eb67bf366da57e527046b896a6fb828020c52514bbab84e85d995b780  $BASAL_SDK_DATA/measurement-data.tar.gz" | sha256sum -c -
mkdir "$BASAL_SDK_DATA/restored"
tar -xzf "$BASAL_SDK_DATA/measurement-data.tar.gz" -C "$BASAL_SDK_DATA/restored"
(cd "$BASAL_SDK_DATA/restored" && sha256sum -c SHA256SUMS)
python tools/sdk/live-basal2-omni/audit.py --measurement-artifacts "$BASAL_SDK_DATA/restored/reports/sdk-basal2-omni-2026-10-17/runs/omni-max-r1"
```

Do nowej inferencji odtwórz rzeczywiste media według poprzedniego raportu Omni,
którego archiwum ma SHA-256
`82aaf99af87532eaccfaf19e3fe5462b5e0d9b4f1ffcac6ccba7ba987ac8a0a7`.
`inputs/media-sources.json` wskazuje dokładne przypadki oraz sumy dwóch zasobów.
Następnie użyj `restore-media.py` i instrukcji w
[README runnerów](../../tools/sdk/live-basal2-omni/README.md).
Nowe pomiary wymagają nowej nazwy serii; zakończonych danych nie nadpisuj.
Archiwum nie zawiera zdjęć, nagrań, wag modeli, identyfikatorów maszyn ani sekretów.
