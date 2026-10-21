# Usunięcie uwierzytelniania kluczem API — 2026-10-21

## Zakres i zgodność

Na jawną prośbę użytkownika usunięto klucze API z serwera, czterech SDK i klienta CLI.
Punkt bazowy: `bd397ebed78c98f6ca57b84c2ebc7a5a31e0d4e9`.
Wersje przygotowane do wydania: serwer 0.1.13, Python/npm/C# 0.6.0, Rust 0.7.0.
Publikacja jest osobnym etapem po tych lokalnych kontrolach.

Batch/Files, podobnie jak istniejąca inferencja online, nie uwierzytelniają klientów.
Nie ma middleware tokenów, generowania ani odczytu ich plików, komendy `batches token`,
opcji CLI `--key-env`, parametrów SDK `apiKey`/`api_key` ani wysyłania nagłówka Bearer.
Własny kontrakt Batch OpenAPI ma wersję 0.2.0, bez securitySchemes i odpowiedzi 401.
Kontrakty upstream oraz poświadczenia baz, S3, Hugging Face i publikacji paczek pozostają osobne.
Usunięto dwie zbędne bezpośrednie zależności CLI; nie dodano zależności ani wyjątków licencyjnych.

Wszystkie żądania HTTP używają `batches.owner` (domyślnie `local`). Jest to wspólna
przestrzeń danych i quota, ustawiana wyłącznie w konfiguracji serwera, bez tożsamości
klienta. Każdy klient widzi jej zasoby i może wykonywać operacje API. Zachowano
schemat, filtrowanie przestrzeni danych i rozliczenia; nie łączono ani nie usuwano danych.
Produkcja może wskazać dotychczasowego właściciela, np. `acme`, z wpisem w quota.
Stare `batches.auth` jest błędem nieznanego pola; instrukcja aktualizacji jest w docs/BATCHES.md.
Usunięcie parametrów konstruktorów SDK jest zamierzoną zmianą niezgodną wstecznie.
Nowe SDK wymagają serwera 0.1.13+ dla Batch/Files, ponieważ starszy wymaga tokenu.

## Weryfikacja lokalna

Linux x86_64. Nie użyto modelu, GPU, produkcji ani upstream. Doraźny operator
uruchomił rzeczywisty BatchService/router Axum z SQLite i artefaktami FS,
syntetycznym katalogiem i `processing:false`. Prywatny hook został usunięty przed
kontrolami produktu; końcową binarkę serwera zbudowano ponownie bez niego.
Pierwsza kompilacja prywatnego hooka nie powiodła się przez położenie modułu przed
komentarzami wewnętrznymi Rust; poprawiono instrumentację, bez zmiany produktu.

- PASS: Python sync/async, JS ESM/CJS, C# i Rust — obie ścieżki create oraz dziewięć
  operacji zasobów bez tokenu, dokładna treść JSONL i nazwa Unicode, timeout bez
  anulowania na serwerze, not-ready, jawne anulowanie i puste wyniki terminalne.
- PASS: brak nagłówka oraz nieprawidłowy Authorization dają te same zasoby;
  nagłówki ownera nie zmieniają przestrzeni. Brak zasobu daje 404, błędne wejście
  błąd walidacji, bez WWW-Authenticate. Stary nieprawidłowy plik tokenu jest ignorowany
  i pozostaje nietknięty.
- PASS: na kopii lokalnej bazy i artefaktów z kwalifikacji 0.1.12 odczytano bez
  tokenu metadane wszystkich 14 zachowanych plików i 14 batchy. Oryginału nie zmieniono.
  To obserwacja lokalnej aktualizacji, nie kwalifikacja każdej bazy i topologii.
- PASS: lokalna przestrzeń `acme` z domyślną quota; odrzucenie starego auth,
  niepoprawnej nazwy i nazwy bez quota przed utworzeniem bazy; usunięte opcje w CLI.
- PASS: fmt, Clippy całego workspace/all-targets (domyślnie i Intel), Rustdoc
  z prywatnymi elementami i ostrzeżeniami jako błędami, cargo deny/all-features,
  polityka paste, OSV dziewięciu przypiętych pakietów Python/npm i check-versions.
- PASS: Python Flit wheel/sdist i instalacja, npm ESM/CJS z deklaracjami i pack,
  C# locked restore/build bez ostrzeżeń/pack, cargo package z kompilacją i deny paczki.

C# kompilowano dla net8.0, operator działał na lokalnym runtime .NET 10 przez
RollForward. Brak obserwacji runtime Windows, macOS, przeglądarki i .NET 8.
Nie dodano ani nie uruchomiono testów projektu. Nie zmieniono kodu obliczeń modelu;
nie przeprowadzono nowego porównania bitowego, numeryki ani pomiaru wydajności.
CUDA i Metal wymagają nowego CI; lokalne kontrole nie zastępują go ani runtime GPU.

## Dane

Archiwum lokalne, bez publikacji HF i bez publicznego URI:
`.cache/measurement-artifacts/remove-api-key-20261021/measurement-data.tar.gz`.
SHA-256: `3b9093c1bcb5358fcfcfcc7b20b970949270c51bd71aa363991929c4df09d8cb`.
Rozmiar 240 962 B; świeże rozpakowanie potwierdziło 62/62 sumy plików.

Logiczna ścieżka `reports/remove-api-key-2026-10-21/` zawiera snapshot 43 zmienionych
plików, manifest z usuniętym auth.rs, obserwacje HTTP/aktualizacji/konfiguracji,
logi kompilacji i METHOD.md. Dane listenera, baz, tokenów i binarki pozostają prywatne.
Logi zanonimizowano. Do odtworzenia kontroli nałóż `source/` na punkt bazowy,
usuń pliki wskazane w manifeście i wykonaj polecenia docs/CI.md oraz build/pack SDK.
Obserwacje runtime wymagają własnego tymczasowego stanu i prywatnej instrumentacji
opisanej w METHOD.md; archiwum nie jest automatycznym runnerem ani zestawem testów.
