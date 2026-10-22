# basal-2.0-mini na RTX 4060 — lokalna próba uruchomienia

## Cel i konfiguracja

Sprawdzenie, czy model mieści się na NVIDIA GeForce RTX 4060 8 GB i wykonuje
krótkie żądania przez produkcyjny serwer HTTP. Pomiar z 2026-10-22.

- basal-rs 0.1.13, commit `0758538037377d93dbeee42bb24de79b64cdff79`;
  gotowa binarka CUDA, odpowiadająca commitowi checkoutu.
- `Remek/basal-2.0-mini@cb81b32ce89b0b87e62fcdad28f0435e25c953f3`,
  podstawowy checkpoint BF16, wykonanie FP16, bez kwantyzacji, orders=1.
- RTX 4060, compute capability 8.9, 8188 MiB według nvidia-smi;
  sterownik 615.78.08, biblioteki CUDA 12.9.1; około 15,5 GiB RAM hosta.
- Domyślne parametry serwera, w tym 8192 tokenów partii, HRRN,
  tor długich żądań powyżej 4096 tokenów i wyłączony cache stanu.
  Włączony access log. Bez zmiany zegarów i limitu mocy.

## Metoda i wyniki

Wejścia: `examples/client/routing.json` i cztery rekordy
`examples/client/tickets.jsonl`. Każde żądanie zawiera dwa pytania:
choice o dział oraz noul o pilność. Najdłuższe prompty mają 140–144 tokeny.
Po czterech rozgrzewkowych żądaniach wykonano 40 sekwencyjnych
oraz 40 przy czterech równoległych klientach. Czas HTTP mierzono w kliencie
Python przez `perf_counter`, razem z odczytem i dekodowaniem odpowiedzi.
Każde żądanie otwierało połączenie przez urllib; klient działał obok serwera.

- Wszystkie 84 żądania zwróciły HTTP 200.
- Sekwencyjnie: p50 **23,21 ms**, p95 **24,31 ms**
  (p95: 38. uporządkowany wynik z 40).
- Czterech klientów: **48,72 żądania/s**, 40 żądań w **0,821 s**.
  To krótka próba funkcjonalna, nie pomiar trwałej przepustowości.
- Najwyższa zaobserwowana zajętość całej karty: **3304 MiB**, około **3,23 GiB**.
  Próbkowanie nvidia-smi co około 200 ms, od przygotowania tabeli do końca próby;
  może pomijać krótsze szczyty. Po próbie również 3304 MiB, około 4884 MiB wolne.
- Pierwszy start: **447,5 s**, z czego pobranie około 3,19 GB zajęło **27 s**,
  a dobór tabeli GEMM **415 s**. Samo ładowanie modelu: **1,6 s**.
  Model i tabela zostały zapisane w cache. Kolejnego startu nie mierzono.
- Pola `answers` dla danego zgłoszenia były dokładnie równe jako wartości JSON
  we wszystkich powtórzeniach, także przy współbieżności. To nie jest porównanie
  bitowe logitów ani kontrola względem upstream.

Odpowiedzi modelu:

- 101, uszkodzony telefon i żądanie zwrotu: `returns`, P(pilne)=0,1011.
- 102, konfiguracja Wi-Fi drukarki: `support`, P(pilne)=0,0672.
- 103, pytanie o zakup monitora: `sales`, P(pilne)=0,0519.
- 104, system po aktualizacji nie startuje i blokuje zespół:
  **`other`**, P(pilne)=0,9317. Pilność została rozpoznana, ale routing
  do „inne” wymaga uwagi; nie zaliczamy tego do potwierdzenia trafności routingu.

## Zakres potwierdzenia

Model mieści się bez kwantyzacji i wykonuje opisane krótkie żądania na tej karcie.
Nie sprawdzano maksymalnego kontekstu modelu (8192 pozycji), dużych partii,
stabilności długotrwałego obciążenia ani pozostałych typów pytań.
Nie wykonywano porównania z upstream FP32; nie deklarujemy zgodności numerycznej
ani jakości modelu na podstawie tej próby. Kod Rust i zależności nie zmieniły się;
kontrole kompilacji, Clippy i rustdoc nie były częścią próby runtime.

## Dane i odtworzenie

Dane pozostają lokalnie poza Gitem. **Nie publikowano ich do HF**; publiczny URI
nie został utworzony. Archiwum:
`.cache/measurement-artifacts/rtx4060-basal2-mini-2026-10-22/measurement-data.tar.gz`.
SHA-256: `e9275d2f151d9883e2f33725a05bf798dfa0aa96816a891d6e7c375b45646232`.
Zawiera `SHA256SUMS` i dwa pliki:

- `reports/rtx4060-basal2-mini-2026-10-22/results.json`: wejścia, odpowiedzi,
  czasy i próbki GPU;
- `reports/rtx4060-basal2-mini-2026-10-22/models.json`: metadane API.

Po rozpakowaniu do nowego katalogu sprawdź `sha256sum -c SHA256SUMS`.
Prywatny log startu i skrypt sesji pozostają poza archiwum, w lokalnym
podkatalogu `private` obok danych. Skrypt odczytuje wymienione wejścia repozytorium,
czeka na gotowość serwera i wykonuje opisaną serię HTTP.

Uruchomienie z katalogu repozytorium z użyciem dostępnej lokalnej binarki:

```bash
BASAL_NO_UPDATE_CHECK=1 .cache/local-basal/bin/basal serve --model 2.0-mini --access-log
```

W drugim terminalu przykład klienta dla tych samych wejść:

```bash
.cache/local-basal/bin/basal client --template examples/client/routing.json \
  --state examples/client/tickets.jsonl --input jsonl --state-pointer /text --jobs 4
```

Serwer pozostawiono uruchomiony lokalnie po próbie. Nie utworzono usługi systemowej.
