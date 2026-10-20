# Artefakty S3: błędy trwałe i zwalnianie slotów readera — 2026-10-20

Status: **PASS w opisanych scenariuszach aktualnego klienta i lokalnego SeaweedFS 4.48**.
Potwierdzono typed `NotFound` przy otwarciu i po usunięciu obiektu podczas odczytu zakresowego,
`Integrity/Permanent` dla złej długości/hash oraz zwolnienie slotu nieodpytywanego readera S3.
Nie jest to kwalifikacja AWS, HA ani gotowości produkcyjnej. Pierwszą nieudaną próbę gotowości
helpera TLS i wymuszone zatrzymanie dostawcy zachowano jako osobne ograniczenia.

## Źródła, wejścia i metoda

Użyto bieżącej roboczej implementacji `ArtifactStore`, `S3Store` i `ArtifactReader`, z
`object_store 0.14.2`, na Linux x86_64 CPU. Nie uruchamiano modeli, inferencji, GPU ani DB.
Walidacja konfiguracji zawierała nieużywaną konfigurację DB; nie otwierano połączenia do niej.
To ręczna diagnostyka implementacyjna, nie kolejny pełny review, framework testów lub bramka CI.

Źródła artefaktów i konfiguracji są bitowo identyczne z produkcyjną kopią roboczą. Prywatna
kopia dodaje tylko moduł operatora i wejście w `main`; zmiany ujawnia
`private-entrypoint-patch.diff`. Pełne źródła kompilacji przypina `full-source-SHA256SUMS`, a
porównanie z kopią roboczą — `source-provenance.json`. Sam bazowy commit nie identyfikuje
wszystkich niezacommitowanych zmian. Kompilacja: Rust/Cargo 1.95.0, `--locked --release`,
domyślne features Linux; build zakończył się poprawnie.

| Źródło | SHA-256 |
|---|---|
| `crates/basal-cli/src/serve/batches/artifacts/mod.rs` | `824e45debf46c88925ccb4eac34ad6978a1568e2626150afc8df601340ec8318` |
| `crates/basal-cli/src/serve/batches/artifacts/s3.rs` | `970c6d4c36a507292005928d269130ccff041903ea892b80a9d2a213486f61bb` |
| Zbudowany prywatny operator | `0b19d8f3d3a6d2c9bf4a700537a2a59ef96a123e4de57e582e2652a2c6d5c846` |

Świeża instancja SeaweedFS miała własne dane i poświadczenia 0600. Wszystkie osiem listenerów
było lokalnych loopback. Wyłączono telemetry oraz dodatkowe serwisy Lance/Iceberg; cache readera
S3 ograniczono do 8 MiB, liczbę woluminów do 2 po 64 MiB. Użyto przygotowanej prywatnej CA i
certyfikatu serwera ważnego w czasie próby. Aktualny produktowy klient wymagał HTTPS, sprawdzał
certyfikat/host i podpisywał żądania; nie wyłączano weryfikacji, podpisów ani nie obniżano TLS.
`request_timeout_ms=1000`, `max_retries=0` dla klienta produktu.

Dane były deterministycznymi fixture'ami 16 KiB i 9 MiB. Większy obiekt zapisano rzeczywistym
writerem multipart, z częściami po 5 MiB; odczyt pobiera zakresy po 4 MiB. Surowy podpisany
klient celowo nadpisywał/usuwał wyłącznie świeże własne klucze, aby wywołać utratę lub korupcję.
To kontrolowane obejście immutable-key writera, bez zmiany implementacji produktu.

## Obserwacje aktualnego klienta — próba R2

Natywny store probe HTTPS zakończył się poprawnie. Pełny odczyt 16 384 B i 9 437 184 B oraz
mały zakres były dokładnie zgodne z fixture'em.

| Scenariusz | Rzeczywisty wynik |
|---|---|
| Nieistniejący obiekt przy `read().await` | `NotFound`, `Permanent` już podczas HEAD/open |
| Zła zapisana długość | `Integrity`, `Permanent` podczas open |
| Zły zapisany digest całego obiektu | `Integrity`, `Permanent` |
| Zły digest zakresu | `Integrity`, `Permanent` |
| Nadpisanie obiektu innymi bajtami tej samej długości | `Integrity`, `Permanent` |
| Skrócenie obiektu przed open | `Integrity`, `Permanent` |
| Usunięcie po odczytaniu dokładnie pierwszych 4 194 304 B | kolejny GET: `io::NotFound` → typed `NotFound`, `Permanent` |
| Następne otwarcie usuniętego obiektu | `NotFound`, `Permanent` |
| Skrócenie obiektu po pierwszych 4 194 304 B | kolejny GET: `io::InvalidData` → `Integrity`, `Permanent` |

Dla wszystkich opisanych błędów trwałych `is_unavailable()` zwróciło false. Typowanie przez
`ArtifactError::from_read` zachowało znaczenie `NotFound` i `InvalidData`; nie zastąpiło ich
ogólnym błędem I/O. Nie uruchamiano tutaj workera modelu ani nie badano końcowego statusu
rekordu batcha po takiej awarii.

## Reader S3 bez dalszego pollingu

Obiekt otwarto przez rzeczywisty klient S3 i do readera przypięto jeden permit. Reader
pozostawał posiadany przez operatora, bez odpytywania jego body.

| Scenariusz | Slot przed | Slot po | Kolejny odczyt |
|---|---:|---:|---|
| Deadline 200 ms, odczekano 350 ms | 0 | 1 | `DeadlineExceeded`, `Transient` |
| Token stop, odczekano 30 ms | 0 | 1 | `DeadlineExceeded`, `Transient` |

Zwolnienie slotu nastąpiło, gdy reader nadal był przechowywany. To potwierdza revocation
niezależne od dalszego pollingu również dla źródła S3, nie tylko filesystem lub fixture'u
readerów. Nie jest to pomiar backpressure całego HTTP ani limitów pamięci produkcyjnej.

## TLS, cleanup i zachowane sygnały

Probe z niewłaściwą CA został odrzucony jako `Remote/Transient`; probe z błędnym podpisem
jako `Remote/Configuration`. Poprawny natywny klient z właściwą prywatną CA wcześniej
odczytał i zapisał dane. Nie użyto AWS, zewnętrznego bucketa lub default credential chain.

Próba R1 zakończyła się **przed wywołaniem natywnego operatora**: helper gotowości
Python/OpenSSL ze sprawdzaniem CA napotykał `bad record MAC`. Przyczyny nie ustalano.
Nie jest to wynik klasyfikacji błędów `ArtifactStore`. Zachowano logi prywatne i wynik
setupu; fresh R2 użyła gotowości TCP, a następnie rzeczywistego, niezmienionego, ścisłego
probe rustls/S3 jako sprawdzenia TLS. Samo otwarcie TCP nie stanowiło potwierdzenia TLS.

Po idempotentnym usunięciu własnych kluczy listowanie prefixu wykazało **0 obiektów i 0
błędów listowania**. Nie badano odzyskiwania po crashu multipart ani fizycznego vacuum danych.

SeaweedFS nie zakończył się w 10 s po SIGTERM; obie własne instancje zatrzymano przez
SIGKILL po tym limicie. To ograniczenie obserwacji graceful shutdown dostawcy, nie status
PASS dla jego łagodnego zamknięcia. Początkowy check przez bind po R2 zgłosił jeden zajęty
port; niezależny check PID/LISTEN potwierdził brak procesu i 0 listenerów, a późniejszy
check wszystkich stanów socketów na własnych portach także 0. Nie ustalano przyczyny
chwilowego wyniku bind. Wszystkie własne usługi są zatrzymane; stare dane/harnessy nie
zostały nadpisane. Dane i logi providerów pozostały prywatne.

## Ograniczenia

- Wyłącznie lokalny SeaweedFS 4.48, statyczne świeże credentials i prywatna CA. Brak AWS,
  HA, multi-host, provider failover, default credential chain i reguł lifecycle multipart.
- Skracano rzeczywisty obiekt i zmieniano jego metadane/dane; nie wstrzykiwano odrębnie
  przedwcześnie zakończonego HTTP body przy nadal deklarowanej poprawnej długości.
- Brak GPU/inferencji/numeryki, DB runtime, E2E workera, RSS, throughput i kwalifikacji
  produkcyjnej. Nie rozszerzamy wyniku na inne storage, modele ani topologie.
- Nie potwierdzono prawidłowego graceful shutdown SeaweedFS; zatrzymanie było ograniczone
  czasem i następnie wymuszone. Weryfikacja braku własnych usług jest oddzielna.
- Nie zmieniano źródeł produktu i nie dodano testów projektu/frameworków/bramek CI.

## Dane, integralność i odtworzenie

Lokalne nieopublikowane archiwum:
`.cache/measurement-artifacts/batches-s3-r2-20261020/measurement-data.tar.gz`, 24 286 B.
SHA-256:

```text
68b41a7e833d4babc34a67504ed1bd5227204d1846063f02e81dfca1f3c15159
```

**Nie wysłano do HF.** Przewidziany docelowy URI po ewentualnej późniejszej publikacji i
osobnej weryfikacji:
`hf://buckets/itsoltech/basal-rs/measurements/batches-s3-r2-20261020/68b41a7e833d4babc34a67504ed1bd5227204d1846063f02e81dfca1f3c15159/measurement-data.tar.gz`.
URI nie potwierdza dostępności. Nie pobierano z HF ani anonimowo.

Autor wykonał świeże lokalne rozpakowanie i sprawdził **12/12** wpisów `SHA256SUMS`.
Osobna weryfikacja rodzica pozostaje oddzielna; zachowano lokalne archiwum.
W archiwum ścieżki mają prefix `reports/batches-s3-r2-2026-10-20/`:

- `operator-r2.log`: pełne natywne obserwacje; pusty `operator-r1.log` potwierdza brak uruchomienia w R1;
- `run-result-r1/r2.json`, `cleanup-verification*.json`: setup, zakończenia i check usług;
- `metadata.json`, manifest źródeł, source provenance, entrypoint patch: wersje i dokładny kod;
- `driver/trace-s3-r2.rs`, `METHOD.md`: operator i komendy uruchomienia prywatnej kopii.

Po rozpakowaniu do nowego katalogu sprawdź hash archiwum i wszystkie `SHA256SUMS`.
`METHOD.md` opisuje utworzenie świeżej prywatnej kopii przypiętych źródeł, zastosowanie
entrypoint patcha, build i uruchomienie przeciw własnemu bounded loopback S3 z prawidłowym
TLS i świeżymi credentials. Surowe komendy providerów, adresy, klucze, ścieżki i logi z
identyfikatorami pozostają wyłącznie lokalnie. Dalsza publikacja wymaga
[procedury artefaktów pomiarowych](../../docs/MEASUREMENT_ARTIFACTS.md).
