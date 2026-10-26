# Naprawa obsługi cache GEMM tylko do odczytu

Status: PASS w zmierzonym zakresie. Obie potwierdzone regresje startu z cache
tylko do odczytu usunięto. Wynik nie stanowi kwalifikacji innych modeli i kart.

## Zakres

Naprawa dwóch regresji opisanych w
[diagnozie cache](../cache-readonly-rtx4060-mini-2026-10-26/README.md):
startu z jawną poprawną tabelą przy pustym cache tylko do odczytu oraz startu
`auto` z historyczną tabelą w takim cache. Zweryfikowana migawka tabeli ma być
wykonywalna bez zapisu artefaktu lub powiązania z profilem. Odmowa zapisu ma
powodować ostrzeżenie. Błędy odczytu, niezgodność artefaktu i inne błędy I/O
pozostają błędami. Jawny import oraz przeszukanie wymagają zapisywalnego cache.

Sprawdzamy także zachowanie fingerprintów Batch API. Fingerprint v1 obejmuje
SHA-256 surowych bajtów tabeli; digest znormalizowanego artefaktu nie zastępuje
tej wartości. Istniejący plik CAS może mieć inne metadane niż źródło, mimo tej
samej zawartości wykonawczej. Odcisk bajtów musi zostać przechwycony przed
ładowaniem wag, a późniejsza zmiana pliku źródłowego nie może go zmienić.

## Metoda i ograniczenia

Lokalna karta RTX 4060 8 GiB, compute capability 8.9, 24 SM; CUDA 12.9.1,
cuBLASLt 120901. Model `Remek/basal-2.0-mini`, rewizja
`cb81b32ce89b0b87e62fcdad28f0435e25c953f3`, FP16.

Każdy start używa osobnego portu i własnego logu gotowości. Serwery są
zatrzymywane SIGINT i sprawdzane przed następnym uruchomieniem. Fixture cache
tylko do odczytu mają katalogi 0555 i pliki 0444; proces działa jako zwykły
użytkownik. Nie zmieniamy normalnego cache użytkownika.

Referencją odpowiedzi jest wersja sprzed refaktoru, commit
`94c97f8fa92f24c20c8f592f13480b7e33a9173a`. Referencją istniejących fingerprintów
CAS jest dodatkowo binarka refaktoru sprzed tej naprawy. Nie mierzymy wydajności
ani innych modeli, Gemmy, innych kart czy wielu GPU.

## Wersje i wynik przeglądu

Końcowa binarka CUDA: SHA-256
`7d6cc808cdfab6561679700509562938662a3733d85d147b1ed2fcec81e5189c`.
Źródła pozostają niecommitowane; archiwum zawiera patch plików śledzonych
względem wskazanego commitu i cztery nowe pliki źródłowe refaktoru.
Opus 5.5 wykonał implementację, Astra niezależny przegląd. W pierwszej rundzie
znaleziono dwie luki P2: brak prawa przejścia przez katalog był traktowany jak
odmowa zapisu, a uszkodzony artefakt wskazany przez powiązanie pozwalał przejść
do innej tabeli. Po poprawkach druga runda zakończyła się `approve`.

## Start i obsługa błędów

| Przypadek | Wynik po poprawce |
|---|---|
| Jawna tabela, pusty cache RO | PASS; tabela w pamięci, ostrzeżenie o odmowie zapisu |
| Jawna tabela, istniejący CAS RO | PASS; istniejący plik pozostaje bez zmian |
| `auto`, tylko historyczna tabela RO | PASS; migawka w pamięci, bez zapisu powiązania |
| `auto`, historyczna tabela, CAS i powiązanie RO | PASS |
| `auto`, pusty cache RO | PASS; tabela wbudowana w pamięci, bez search |
| Jawna tabela, pusty cache RW | PASS; artefakt zapisany |
| `auto`, historyczna tabela RW | PASS; artefakt i powiązanie zapisane |
| `auto`, historyczna tabela i CAS RO, brak powiązania | PASS; odmowa zapisu powiązania jest ostrzeżeniem |
| `auto`, zapisywalny cache, sam katalog powiązań RO | PASS; CAS zapisany, powiązanie pominięte z ostrzeżeniem |
| Jawna tabela, CAS z dodatkowymi metadanymi RO | PASS; zachowany fingerprint surowych bajtów istniejącego CAS |
| `pinned`, pusty cache RO | PASS; przypięta tabela wbudowana w pamięci |
| Jawna tabela, późniejsza zmiana pliku źródłowego | PASS; fingerprint i odpowiedzi pozostają zgodne z migawką |
| Jawna tabela, uszkodzony istniejący CAS | Oczekiwany błąd integralności przed wagami |
| Ścieżka cache jest plikiem | Oczekiwany `Not a directory` przed wagami |
| Brak żądanego artefaktu `sha256:` | Oczekiwany błąd przed wagami; bez search |
| Cache bez prawa przejścia (`x`) | Oczekiwany `Permission denied` podczas sprawdzania CAS, przed wagami |
| Poprawne powiązanie do uszkodzonego CAS, dostępna inna tabela legacy | Oczekiwany błąd przed wagami; innej tabeli nie użyto |

Łącznie 17 scenariuszy: 12 poprawnych startów i pięć oczekiwanych odrzuceń.
Dodatkowy `gemm-digest --import` do cache RO zakończył się oczekiwanym błędem
zapisu; jawne polecenie importu nie staje się operacją pamięciową.

Pusty cache RO w `auto` nie działał także przed refaktorem. Jego obecny PASS
jest świadomym rozszerzeniem: na tej karcie istnieje przypięta tabela
wbudowana, więc można użyć jej bez zapisu. Nie oznacza to, że `auto` bez
dostępnego artefaktu może wykonywać search bez zapisywalnego cache.

W negatywnym przypadku powiązanego CAS tabela A ma digest
`2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96`,
lecz plik zawiera inną nazwę GPU. Alternatywna tabela B zachowuje poprawne
algorytmy i dane karty, ma `gemm_search_version: 3` (akceptowany przez legacy
adaptive) i digest
`e59a55d606d9d345e13f1c073adc3f6235b620844e3cacc4a7ac5e00f7f9e8d9`.
Start zatrzymuje się na uszkodzeniu A, zanim wybierze B.

## Odpowiedzi i zgodność bitowa

Każdy z 12 poprawnych startów obsłużył te same cztery żądania HTTP przy C4.
Wszystkie 48 odpowiedzi jest zgodnych z referencją sprzed refaktoru:
porównano pełne rekordy po uporządkowaniu kluczy JSON i usunięciu jedynie
`response.usage.latency_ms`. Zachowano wartości liczbowe, odpowiedzi i usage.

| Eksport z cache RO | Pary bench | Pary SystemOne | Pełne pliki JSONL | `basal compare` |
|---|---:|---:|---|---|
| Jawna tabela, batching `tree` | 8 | 33 | Bajtowo identyczne | 0 różnic tokenów, max różnica logitów 0, TV 0 |
| `auto`, batching `single` | 8 | 33 | Bajtowo identyczne | 0 różnic tokenów, max różnica logitów 0, TV 0 |

To 82 porównane pary rekordów, wraz z błędami i pustymi wynikami obecnymi
w referencji. Nie oznacza to 82 poprawnych odpowiedzi modelu. Nie porównywano
wewnętrznych aktywacji każdej warstwy ani nowych danych względem FP32 upstream;
naprawa nie zmienia arytmetyki.

## Fingerprinty Batch API

| Źródło surowych bajtów | Fingerprint przed i po |
|---|---|
| Jawny plik, pusty CAS | `fp1-02aca26f24dfbbad9a75dbeaedc4811e7ea267ad674437521cc8d66c2cca2b2c` |
| Istniejący CAS albo historyczna tabela legacy | `fp1-822717b909432cb26feb8c6cd24905b3abcf6cf8f13b688f10775f50e41e33b3` |
| Istniejący CAS z dodatkowymi metadanymi | `fp1-b5c90fcf41dd248b2d2b10dcf50a3fb3b8f65b562550b93f5969680e79b48283` |

Cztery uruchomienia referencyjne i cztery odpowiadające im uruchomienia po
naprawie dały te same fingerprinty. W dodatkowym przypadku plik źródłowy
zmieniono po zalogowaniu zamrożonego planu, przed gotowością serwera; zachowany
log z tej chwili zawiera wyłącznie konfigurację, ostrzeżenie cache i plan.
Dodanie metadanych zmieniło SHA-256 pliku z `0ec5907f…` na `6b0fccea…`,
ale fingerprint pozostał `fp1-822717…`, zgodny z pierwotną migawką.

Sprawdzono gotowość profili przez health Batch API. Nie tworzono ani nie
wykonywano zadań Batch; kwalifikacja wykonania background była wyłączona.

## Kontrole i ograniczenia

Opus po końcowych edycjach wykonał `cargo fmt`, Clippy workspace release
all-targets w wariantach all-features, domyślnym i Intel oraz rustdoc workspace
release all-features z `-D warnings`: PASS. Integrator zbudował końcową binarkę
CUDA i wykonał powyższe pomiary; źródła po buildzie zachowały te same sumy.
`git diff --check`: PASS.

`cargo deny check`: FAIL — wcześniej potwierdzone także na baseline
`RUSTSEC-2026-0385` (`ctutils 0.4.3`) i `RUSTSEC-2026-0386` (`tokio 1.53.2`).
Ta naprawa nie zmienia zależności. Bans, licenses i sources: OK.

Sumy i uprawnienia wszystkich wcześniejszych fixture RO pozostały identyczne.
W nowych fixture istniejące pliki i ich uprawnienia pozostały bez zmian;
nowe pliki pojawiły się tylko w zapisywalnych kontrolach. Wszystkie serwery
zatrzymano, końcowa zajętość GPU wyniosła 2 MiB. Jedną próbę z ciepłym cache
unieważniono: integrator potraktował nadal działające polecenie oczekiwania
na gotowość jako niepowodzenie i zatrzymał serwer przed gotowością. Próba jest
zachowana prywatnie i nie wchodzi do wyników.

Zmierzono `PermissionDenied` na uprawnieniach Unix, bez montowania systemu
plików RO (`EROFS`). Nie wykonywano autotuningu, prób na Gemmie, Metal, Intel,
innych GPU, wielu GPU ani innych modelach. Akceptacja Astry jest przeglądem
statycznym, nie dodatkowym pomiarem runtime.

## Artefakty

Kampania lokalna: `gemm-readonly-cache-fix-rtx4060-mini-2026-10-26`.
Publikacja: brak. URI: brak. Archiwum lokalne:
`.cache/measurement-artifacts/gemm-readonly-cache-fix-rtx4060-mini-2026-10-26/measurement-data.tar.gz`.
Rozmiar: 162 461 B. SHA-256:
`55736c0b5c89129d37f88078d70c4a15d08ade4c5c061c3fbf69ad94ee441836`.
Rozpakowanie do osobnego katalogu potwierdziło wszystkie 45/45 sum
`SHA256SUMS`. Nie wykonywano publikacji ani weryfikacji pobrania z HF.

Logiczna ścieżka danych w archiwum:
`reports/gemm-readonly-cache-fix-rtx4060-mini-2026-10-26/`.
`results.json` zawiera macierz przypadków i metadane, `http/` odpowiedzi po
normalizacji, `batch-profiles/` profile przed i po naprawie, `mutation.json`
wynik zmiany źródła, `baseline/` oraz `exports/` pełne porównywane JSONL,
`inputs/` wejścia i tabele fixture, a `source/` źródła umożliwiające
odtworzenie. Polecenia i warunki pomiarów są w `REPRODUCE.md`.

Prywatne logi, konfiguracje, binarki i dane Batch pozostają lokalnie. Nie
zmieniano normalnego cache użytkownika. Przed odtworzeniem sprawdź SHA-256
archiwum, rozpakuj je do nowego katalogu, następnie wykonaj w nim
`sha256sum -c SHA256SUMS` i zastosuj instrukcje z `REPRODUCE.md`.
