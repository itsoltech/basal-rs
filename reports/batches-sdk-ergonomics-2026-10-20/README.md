# SDK: dwie ścieżki Batch API i pomocniki — 2026-10-20

## Cel i zakres

Dodano do wszystkich czterech SDK zasoby Files i Batches oraz dwie ścieżki tworzenia:
plik JSONL → upload → create z `input_file_id`, albo `create` z rekordami
`{custom_id, body}`, które SDK koduje i przesyła samodzielnie. Dodano jawne `wait`
i odczyt rekordów `results`. Treść rekordów pozostaje natywnym żądaniem `/v1/basal`.
Zgodność z OpenAI dotyczy nazw zasobów i przepływu, nie formatu Chat Completions/Responses.

Punkt bazowy: `e00bb4b0862fd84f91b7596fc92e730d1e348449`. Zmiany tego raportu są
niezatwierdzoną jeszcze w Gicie poprawką **Unreleased**. Wersje manifestów pozostają:
Python/npm/C# 0.4.0, Rust 0.5.0. Nie opublikowano ponownie tych numerów ani nie zmieniono tagów.
Dotychczasowe publiczne metody pozostają dostępne. Nie zmieniono kodu obliczeń modelu.

## Zachowanie

- `files`: create/retrieve/list/delete/content; `batches`: create/retrieve/list/cancel/wait/results.
  Nazwy i argumenty dostosowano do języka. Rust używa `files()`/`batches()` i konwersji
  `CreateBatchRequest`/`CreateBatchFromRequests`; C# ma przeciążenia `CreateAsync`.
- Uproszczone create przygotowuje JSONL w pamięci. Sprawdza kopertę, identyfikatory,
  duplikaty, zakaz `stream:true`, limit 50 000 rekordów/200 000 000 bajtów oraz klucz.
  Serwer nadal waliduje treść, możliwości modelu, część parametrów i swoje niższe limity.
  Narzut obejmuje buforowanie i kodowanie pojedynczego rekordu; to nie jest pomiar peak RSS.
- Po udanym uploadzie błąd tworzenia zachowuje ID pliku, klucz idempotencji i przyczynę.
  Ponowienie z tym plikiem i kluczem nie wymaga uploadu. SDK nie ponawia operacji ani nie usuwa
  pliku po niepewnym wyniku. Przerwany upload może nie udostępnić jego ID.
- `wait` ma domyślnie limit 600 s i przerwę 5 s. Obejmuje pobieranie statusu i oczekiwanie;
  zwraca completed/failed/expired/cancelled. Timeout nie wysyła anulowania do serwera.
  Nie ponawia błędów HTTP. Python sync używa własnego klienta async i dołączanego wątku;
  jego pula połączeń jest osobna dla wywołania wait. Opóźnienie systemowego DNS może opóźnić
  końcowe zamknięcie/dołączenie wątku ponad deadline — nie ma gwarancji przerwania resolvera OS.
- `results` wymaga stanu terminalnego. Czyta output, potem error, deduplikując ID plików.
  Uwzględnia częściowe wyniki. Buforuje jeden plik z limitem (domyślnie 8 MiB), a następnie
  parsuje pojedyncze wiersze; to nie jest nieograniczony strumień w stałej pamięci.
  Błędny UTF-8/JSON/koperta przerywa iterację z kontekstem pliku i wiersza. Błędy walidacji
  całego zadania są na obiekcie Batch; brak plików oznacza pustą iterację.

## Metoda i środowisko

Lokalny Linux x86_64; Rust 1.95.0, Node 26.8.2, Python 3.15.0 w obserwacji rodzica.
C# kompilowano dla net8.0, uruchamiano na .NET 10.0.12 przez RollForward, ponieważ lokalnie
nie ma runtime .NET 8. Dodatkowe wersje Pythona i kontrole typów są opisane w danych dzieci.
Nie uruchamiano GPU, modelu, upstream ani produkcji. Nie wykonywano testów projektu,
nie dodano frameworka ani bramki testowej.

Jednorazowe programy diagnostyczne uruchomiono poza Gitem. Rodzic użył rzeczywistego
BatchService, routera Axum, SQLite i artefaktów FS z osobnym tymczasowym stanem,
syntetycznym katalogiem `operator-model` oraz `processing:false`. Ta prywatna instrumentacja
ma jedynie uruchomić API bez silnika modelu; nie weszła do kodu produktu. Osobne kontrolowane
odpowiedzi HTTP/adaptatory służą sprawdzeniu błędów, deadline'u i parsowania wyników.

## Wyniki

| Obszar | Wynik i granica dowodu |
|---|---|
| Natywne API | PASS: obie ścieżki create i dziewięć operacji zasobów w sześciu wariantach: Python sync/async, JS ESM/CJS, C#, Rust. Dokładne bajty wejścia, nazwa Unicode, timeout bez anulowania, not-ready, jawne anulowanie, wait do cancelled i pusty odczyt terminalny. |
| TypeScript | PASS: build ESM/CJS z deklaracjami i pakowanie; 72/72 obserwacje adaptatora oraz dodatkowe odrzucenie streamingowego wejścia i pięciu błędnych kopert/pustego wiersza. Obejmuje przerwanie HTTP, niekooperujący adapter, deduplikację plików, recovery i limity. |
| Python | PASS: Flit wheel/sdist i instalacja; pyright bez błędów. 48 obserwacji R1 ponowiono po poprawkach; 31 obserwacji R2, w tym wolne nagłówki/body po jednym bajcie. Dla limitu 0,8 s timeout zaobserwowano około 0,80–0,81 s; nie pozostał wątek oczekujący. Zachowano recovery po Ctrl-C/cancel. To kontrolowana obserwacja, nie SLO. |
| C# | PASS: build z warnings-as-errors i pack; poprzednia powierzchnia API zachowana. 32 obserwacje dziecka oraz cztery dodatkowe przypadki błędnych wyników u rodzica. Timeout obejmuje HTTP; anulowanie po uploadzie zachowuje OperationCanceledException i recovery. |
| Rust | PASS: domyślny i Intel Clippy całego workspace, Rustdoc z prywatnymi elementami, fmt, cargo package z weryfikacją i cargo deny paczki. Obserwacje HTTP: częściowy output/error, kontekst błędu, limit bajtów, deadline podczas HTTP, przyszły status i recovery. |
| Zależności | PASS: pełne cargo deny, polityka paste i OSV dziewięciu przypiętych pakietów Python/npm po poprawkach TLS opisanych poniżej. |
| Dokumentacja | PASS: lokalne odsyłacze, git diff --check, check-versions; przykłady C#/Rust kompilują się w istniejących kontrolach. README/snippety nie są osobnym wykonywanym zestawem testów. |

Mypy strict nadal zgłasza siedem istniejących błędów re-eksportów `tasks` obecnych także w bazie;
nie nazywamy tego kontroli zakończonej sukcesem. Pyright pakietu i sondy konsumenta przechodzi.
W danych zachowano wcześniejsze nieudane próby i logi kompilacji, następnie poprawione:
kontrola TLS/licencji, lokalne wywołanie niewłaściwej ścieżki projektu C#, typ pola stream
w TypeScript i lokalna kolizja nazwy zmiennej C#. Końcowe pliki mają nazwy wskazane w
`validation-summary.json`; historyczne próby nie są zaliczone jako PASS.

## Poprawki zależności TLS

Podczas wymaganej kontroli ujawniono RUSTSEC-2026-0381/0382 i GHSA-6q48-6c7w-9gw2.
Przypięcia zaktualizowano do rustls-webpki 0.103.16, webpki-roots/root-certs 1.0.10
oraz certifi 2026.10.19 z weryfikacją hasha wheel i licencji MPL-2.0.
Wyjątek licencyjny CDLA-Permissive-2.0 przeniesiono wyłącznie z webpki-roots =1.0.9
na =1.0.10 po jawnej zgodzie użytkownika. Nie wyłączono advisory ani kategorii licencji.

Źródła zmian: [webpki](https://github.com/rustls/webpki/pull/546),
[webpki-roots/root-certs](https://github.com/rustls/webpki-roots/pull/133),
[advisory certifi](https://github.com/advisories/GHSA-6q48-6c7w-9gw2).
Kontrole potwierdzają graf i kompilację; nie przeprowadzono własnej próby ataku TLS.
Poprawki nie zostały wdrożone na produkcji ani opublikowane jako nowe wydanie.

## Ograniczenia i decyzja

Zakres SDK jest zweryfikowany lokalnie. Brak nowej walidacji modeli, numeryki, wydajności,
backendów produkcyjnych, topologii GPU oraz wygenerowanych przez inferencję wyników.
Częściowe wyniki parserów obserwowano na kontrolowanych odpowiedziach, nie w modelu.
Brak runtime Windows/macOS, przeglądarki i .NET 8; nie uruchamiano dla tej poprawki nowego CI
Metal/CUDA. Starsze raporty i ich luki pozostają niezmienione. Jawny upload pozostaje
właściwy dla aplikacji, która musi trwale zapisać każdy etap lub obsłużyć duże dane strumieniowo.

## Dane i odtworzenie

Archiwum pozostaje lokalnie; **brak publicznego URI** i brak publikacji HF.
Docelowy magazyn po osobnej publikacji to `hf://buckets/itsoltech/basal-rs/`.

- Lokalnie: `.cache/measurement-artifacts/batches-sdk-ergonomics-20261020/measurement-data.tar.gz`.
- SHA-256: `3fb5ee6ebe8c516027349d235db5e8bd28d6070be0af6dd59a84cdf2ca9cf6a7`.
- Rozmiar: 289 083 B; świeże rozpakowanie potwierdziło **178/178** sum plików.
- Ścieżka danych: `reports/batches-sdk-ergonomics-2026-10-20/` — `source/`, `source-manifest.json`,
  `diagnostics/validation-summary.json`, logi/obserwacje, `operators/`, `METHOD.md`.
- Snapshot obejmuje 73 pliki źródłowe/konfiguracyjne. Dane maszyn, tokeny, bazy,
  ready/config instancji, środowiska pakietów i binarki pominięto; ścieżki logów zanonimizowano.

Sprawdź SHA archiwum, rozpakuj do nowego katalogu, uruchom `sha256sum -c SHA256SUMS`.
Na osobnym checkoucie punktu bazowego nałóż `source/`; wykonaj polecenia z `docs/CI.md`
oraz build/pack z README SDK. `operators/` i `METHOD.md` opisują prywatną instrumentację
API bez modelu. Powtórzenie obserwacji wymaga własnych tymczasowych listenerów i stanu;
parametry instancji są celowo wykluczone. Nie ma runnera, który pobiera te dane automatycznie.
