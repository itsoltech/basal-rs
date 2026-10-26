# Cache tylko do odczytu: dwie regresje startu

## Wynik

**FAIL zgodności startu w dwóch konfiguracjach.** Po refaktorze nie startuje
serwer z jawną tabelą i pustym cache tylko do odczytu oraz serwer `auto`
ze starym cache wymagającym importu. Obie konfiguracje działają przed
refaktorem. To potwierdzone regresje kompatybilności, nie tylko ryzyko.

Weryfikacja została zlecona lokalnie przez użytkownika. Nie zmieniano kodu
Rust ani zwykłego cache użytkownika; błędy pozostają do poprawienia.
Wcześniejszy [raport A/B](../model-profiles-rtx4060-mini-2026-10-25/README.md)
dotyczy zgodności wyników modelu i nie obejmował tej macierzy praw dostępu.

| Konfiguracja cache tylko do odczytu | Przed refaktorem | Po refaktorze | Ocena |
|---|---|---|---|
| Jawna ścieżka tabeli, pusty cache | start i odpowiedzi OK | brak startu, `Permission denied` | REGRESSION |
| Jawna ścieżka tabeli, istniejący artefakt CAS | start i odpowiedzi OK | start i odpowiedzi OK | PASS |
| `auto`, stary plik `<model>-f16.json`, bez CAS/powiązania | start i odpowiedzi OK | brak startu przy imporcie, `Permission denied` | REGRESSION |
| `auto`, istniejące powiązanie profilu i artefakt CAS, zachowany stary plik | start i odpowiedzi OK | start i odpowiedzi OK | PASS |
| `auto`, pusty cache | brak startu, `Permission denied` | brak startu, `Permission denied` | wcześniejsze ograniczenie |

Każdy z sześciu serwerów, które wystartowały, obsłużył cztery żądania HTTP
przy C4. Odpowiedzi, dane wejściowe i usage były identyczne po pominięciu
wyłącznie `usage.latency_ms` i kolejności kluczy JSON. Nie oceniano wydajności.

## Metoda i wersje

RTX 4060, 8188 MiB, compute capability 8.9, sterownik 615.78.08,
cuBLASLt 120901. Model:
`Remek/basal-2.0-mini@cb81b32ce89b0b87e62fcdad28f0435e25c953f3`.
Tabela repozytorium: RTX 4060 / basal-2.0-mini / f16, search v2,
100 wpisów, digest
`2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96`.

| Wersja | Tożsamość |
|---|---|
| Baseline źródeł | `94c97f8fa92f24c20c8f592f13480b7e33a9173a` |
| SHA-256 binarki przed | `711e49dfb5d42427033d34e30093f660a28fe1c9468d30734dc4db5d14070db4` |
| SHA-256 binarki po | `e600140776f03e6cf7a52dd8a740b09ed03d06fc03499951acaa242664a9cd67` |
| Archiwum zawierające patch źródeł kandydata | SHA-256 `9791845b1eba8766a1f78b36427713ad2ebf1c36587eeeaf9afc557f43c92260`, opis w poprzednim raporcie |

Przygotowano pięć osobnych cache. Katalogi miały prawa `0555`, pliki `0444`;
procesy pracowały jako zwykły użytkownik, bez root. Próba `touch` w każdym
cache zwróciła błąd uprawnień. Brak zapisu wymuszono prawami systemu plików,
nie montowaniem woluminu read-only.

Każda próba używała tego samego YAML: model lokalny, FP16, batch 1024,
long lane 256, maksymalnie 64 żądania. Różniły się `gemm_cache` i
`gemm_table`. Wejścia to cztery żądania routing/tickets z wcześniejszej
kampanii, każde z jawnym modelem i pytaniami choice/noul.

Start potwierdzano jednocześnie przez `/health` i linię ready właściwego
procesu. Każde uruchomienie miało osobny port. Po próbie SIGINT kierowano
do PID tego serwera i sprawdzano exit 0. Wszystkie sześć uruchomionych
serwerów zakończyło się poprawnie; nie pozostały listenery prób. Końcowa
zajętość GPU: 2 MiB. Sumy wszystkich pięciu plików cache oraz prawa
wszystkich katalogów i plików były identyczne przed i po próbach.

Pierwszą serię unieważniono: ponowne użycie portu pozwoliło zapytaniu health
trafić do wcześniejszego procesu, a część kolejnych startów miała kolizję
adresu. Surowe logi zachowano prywatnie. Wyniki w tabeli i archiwum pochodzą
wyłącznie z powtórzonej serii z osobnymi portami i kontrolą PID.

## Przyczyna

Dla jawnej ścieżki `gemm_artifact::resolve` odczytuje poprawną migawkę,
a następnie obowiązkowo wywołuje `store` przed użyciem jej przez backend
(`crates/basal-cli/src/gemm_artifact/mod.rs:395–400`). `store` tworzy
`artifacts/`, a przy braku artefaktu plik tymczasowy. Brak praw zapisu
przerywa resolve, choć oryginalna tabela jest czytelna i poprawna.

Przy `auto` stary cache jest importowany przez `local_table`, które
obowiązkowo zapisuje artefakt i powiązanie profilu
(`gemm_artifact/mod.rs:516–518`). Samo pozostawienie pliku źródłowego bez
zmian nie wystarcza do obsługi cache tylko do odczytu.

Jeżeli artefakt już istnieje, jawna ścieżka może użyć go bez zapisu.
Jeżeli istnieje również poprawne powiązanie profilu, `auto` omija import
starego pliku. Takie przypadki przeszły. Całkowicie pusty cache `auto`
wymagał zapisu także przed refaktorem, więc jego odmowa startu nie jest
nową regresją.

Osobno sprawdzono uprawnienia domyślnego lokalnego cache: katalog istnieje,
ma `0755`, a `test -w` zwrócił sukces. Nie istnieje w nim podkatalog CAS.
Zmienne zmieniające domyślną lokalizację były nieustawione. Nie wykonywano
zapisu do tego katalogu ani zmiany jego praw. To kontrola uprawnień dla
bieżącego użytkownika, nie audyt innego konta serwisowego lub kontenera.

## Dane i ograniczenia

Nowe dane są poza Gitem w
`.cache/measurement-artifacts/cache-readonly-rtx4060-mini-2026-10-26/`.

| Pole | Wartość |
|---|---|
| Lokalne archiwum | `measurement-data.tar.gz` w katalogu kampanii |
| SHA-256 | `30b31a34d8e3dffca497482f573bb98d51434b74d47577b29a63c36bddb50574` |
| Rozmiar | 9 762 B |
| Weryfikacja | rozpakowanie do osobnego katalogu, 28/28 wpisów `SHA256SUMS` zgodnych |
| Ścieżka danych w archiwum | `reports/cache-readonly-rtx4060-mini-2026-10-26/` |
| Publikacja / URI | nie publikowano w HF; URI publiczne nie istnieje |

Archiwum zawiera `results.json`, instrukcję `REPRODUCE.md`, pięć cache,
ich sumy i prawa przed/po, wejścia, sześć plików odpowiedzi HTTP i logi
dziesięciu prób z usuniętymi lokalnymi adresami i prefiksem kampanii.
Po rozpakowaniu uruchom `sha256sum -c SHA256SUMS`; szczegóły odtworzenia
YAML, praw dostępu, startu i porównania są w `REPRODUCE.md`.

Nie wykonywano poprawki ani nowego buildu. Nie sprawdzano fizycznego
mounta read-only, kontenerów, roota, innych modeli ani innych kart.
Wynik dotyczy zgodności startu i odpowiedzi w opisanych konfiguracjach,
nie pełnej kwalifikacji pozostałych ścieżek cache.
