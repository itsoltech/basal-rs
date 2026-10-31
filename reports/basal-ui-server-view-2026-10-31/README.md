# Widok Server w basal-ui — 2026-10-31

## Cel i zakres

Widok administratora z działaniem serwera na żywo: statystyki przetwarzania,
obciążenie sprzętu, stan cache prefiksów i ostatnie żądania z odpowiedziami.
Opis zachowania i źródeł danych: [docs/UI.md](../../docs/UI.md#widok-server).

- Widok istnieje tylko z `ui` i `auth`. Bez kont serwer nie zbiera statystyk
  i nie ma ich tras. Dane widzą tylko administratorzy.
- Treść żądań i odpowiedzi (ostatnie 200, w pamięci procesu) wymaga osobnej opcji
  `ui.capture_requests` albo `--ui-capture-requests` oraz `auth`.
- Nowe dane silnika: `Backend::cache_stats` (basal-core) i liczniki `GpuBackend`:
  chybienia i usunięcia state cache oraz trafienia prefiksów szablonu. To same
  inkrementacje liczników, bez zmiany obliczeń.
- GPU z NVML przez FFI ładowane w trakcie działania. Nie doszła żadna zależność Cargo ani npm.
- Kontrakt `/v1/*` i nazwy metryk Prometheus bez zmian.

## Wersje, wejścia, sprzęt

- Baza: `548b697` (main). Wersja badana: ta gałąź, ten sam profil `release`, CUDA,
  kernele dla compute capability 89. Po pomiarze zmieniły się tylko strony UI
  i pole `reset_at` migawki; kod ścieżki żądania, wątków modelu i silnika jest
  ten sam co w pomiarze.
- Model `Remek/basal-2.0-mini` z lokalnego cache HF, tabela GEMM wbudowana
  w build dla RTX 4060, `state_cache_mb: 256`, `long_tokens` domyślne.
- Żądania: 44 pozycje `reports/reference-1.5-max-fp32/bench.jsonl`, każda jako
  żądanie System One z jednym pytaniem `choice`, cyklicznie.
- NVIDIA GeForce RTX 4060 8 GB (sterownik 615.78.08, limit mocy 115 W),
  AMD Ryzen 5 5600X (12 wątków), Linux x86_64.

## Metoda

Trzy konfiguracje, w trzech rundach z naprzemienną kolejnością, każda w świeżym
procesie z tą samą bazą SQLite, kluczem API, `auth`, `ui` i `metrics`:

- `baseline`: binarka `548b697` (bez statystyk widoku);
- `stats`: ta gałąź, statystyki widoku włączone (`ui` + `auth`);
- `capture`: jak `stats` oraz `ui.capture_requests: true`.

W każdej rundzie: rozgrzewka (44 żądania sekwencyjnie), 220 żądań przy
współbieżności 1, 1600 żądań przy współbieżności 16 (pętla zamknięta). Klient
zapisuje pierwszą kompletną odpowiedź `answers` dla każdej pozycji w obu fazach.

## Wyniki

Mediana z trzech rund, żądania/s (w nawiasach wartości rund):

| Konfiguracja | c=1 req/s | c=1 p50 ms | c=16 req/s | c=16 p50 ms | Błędy |
|---|---|---|---|---|---|
| baseline | 51,25 (51,62; 51,25; 47,31) | 17,93 | 64,81 (65,63; 64,69; 64,81) | 246,90 | 0 |
| stats | 50,98 (50,98; 51,51; 46,96) | 18,08 | 65,18 (65,33; 64,90; 65,18) | 243,68 | 0 |
| capture | 49,76 (49,76; 46,74; 50,31) | 18,57 | 64,94 (65,24; 63,73; 64,94) | 243,58 | 0 |

- Zgodność bitowa: PASS. Wszystkie 18 zestawów odpowiedzi (3 konfiguracje × 3 rundy
  × fazy c=1 i c=16, po 44 pozycje) są identyczne z `baseline` runda 1. Porównano
  całe obiekty `answers`, w tym prawdopodobieństwa i `confidence`.
- Przepustowość: przy c=16 różnice median mieszczą się w 0,6 %. Przy c=1 rozrzut
  między rundami tej samej konfiguracji sięga 8 % (47,31 do 51,62). `capture` ma
  medianę o 2,9 % niższą od `baseline`, w zakresie tego rozrzutu. Pomiar nie
  wykazał narzutu większego niż szum. Nie potwierdza też jego braku poniżej tego poziomu.
- Ręczna kontrola widoku na tej maszynie:
  - NVML podał nazwę karty, wykorzystanie, szynę pamięci, pamięć całkowitą i pamięć
    procesu, temperaturę, zegar SM oraz limit mocy.
  - Pobór mocy jest pusty, bo sterownik zwraca dla tej karty `[N/A]`
    (`nvidia-smi --query-gpu=power.draw`).
  - State cache przy pozycjach bench nie miał wstawień: stany są krótsze niż
    32 tokeny ponad prefiks szablonu. Z `tools/bench/long_states.jsonl` widać
    wstawienia i trafienia.

## Ograniczenia

- Metal i Vulkan nie były uruchamiane. CI kompiluje kod Metal, ale nie sprawdza
  odczytów. Na Metal widok pokazuje tylko pamięć z sondy backendu, bez wykorzystania,
  mocy i temperatury. Host na macOS: CPU procesu, load, liczba rdzeni i RAM całkowity.
- Jedna karta, jeden mały model. Nie mierzono replik, `layer_split` ani modeli
  Gemma i Qwen3.5, które nie mają cache prefiksów.
- Narzut przy przepustowości rzędu tysięcy żądań/s nie był mierzony. Ścieżka
  żądania ma jedną blokadę na żądanie i odczyt `usage` odpowiedzi 200 do 64 KiB.
- Percentyle od startu są przybliżone (histogram logarytmiczny, błąd względny
  poniżej 5 %).

## Dane

Archiwum `hf://buckets/itsoltech/basal-rs/measurements/basal-ui-server-view-2026-10-31/e5520f46b9283ae7adda778fa7ac755f6173977d016160c04ab5cf556e9dd132/measurement-data.tar.gz`,
SHA-256 `e5520f46b9283ae7adda778fa7ac755f6173977d016160c04ab5cf556e9dd132`,
11 326 B. Obok leży `SHA256SUMS` (ten sam plik co w archiwum). Pobranie z logowaniem
i anonimowo potwierdziło SHA-256 archiwum oraz 49/49 sum `SHA256SUMS` w obu kopiach.
Archiwum nie zawiera adresów, nazw hostów, lokalnych ścieżek ani tokenów.
Lokalna kopia jest zachowana.

| Ścieżka w archiwum | Zawartość |
|---|---|
| `reports/basal-ui-server-view-2026-10-31/results/<konfiguracja>-r<N>-{warm,c1,c16}.json` | req/s, p50/p95/p99, błędy każdej fazy |
| `reports/basal-ui-server-view-2026-10-31/results/<konfiguracja>-r<N>-answers-{seq,c16}.json` | pierwsze `answers` każdej pozycji |
| `reports/basal-ui-server-view-2026-10-31/{measure.sh,load.py}` | przebieg pomiaru i klient HTTP (biblioteka standardowa Pythona) |
| `reports/basal-ui-server-view-2026-10-31/serve*.yml` | konfiguracje serwera, ścieżka bazy zastąpiona opisem |

Odtworzenie: zbuduj obie wersje (`cargo build --release -p basal-cli --features cuda`),
umieść binarki jako `basal-baseline` i `basal-branch` katalog wyżej niż `measure.sh`.
W tym samym katalogu połóż `cuda-env.sh` ze zmiennymi środowiska CUDA. Utwórz bazę
z `basal database migrate`, użytkownika `basal users create` i klucz
`basal keys create --scopes inference` zapisany w pliku `token`. Potem uruchom
`measure.sh` z katalogu repozytorium, tak aby ścieżka
`../../reports/reference-1.5-max-fp32/bench.jsonl` wskazywała plik bench.
