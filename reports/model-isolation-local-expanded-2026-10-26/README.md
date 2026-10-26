# Rozszerzona lokalna weryfikacja izolacji modeli

Status: PASS w opisanym zakresie. Znaleziona regresja publicznych konstruktorów
GPU została naprawiona, odtworzona przed poprawką i zweryfikowana po niej.
Nie jest to deklaracja pokrycia wszystkich modeli i platform.

## Zakres i środowisko

Kontynuacja zaakceptowanego refaktoru profili i
[naprawy cache RO](../gemm-readonly-cache-fix-rtx4060-mini-2026-10-26/README.md).
Użytkownik zlecił wszystkie lokalnie wykonalne sprawdzenia i naprawę znalezionych
regresji. Kampania używa istniejących poleceń `export`, `compare`, `client`,
`gemm-search`, HTTP, manualnych runnerów SDK i loadtestu; bez dodawania
frameworków testowych i bramek CI.

Sprzęt: RTX 4060, 8188 MiB VRAM, compute capability 8.9, 24 SM; 15 GiB RAM.
CUDA 12.9.1, cuBLASLt 120901, Rust 1.95.0. Lokalnie dostępny jest wyłącznie
checkpoint `Remek/basal-2.0-mini`, rewizja
`cb81b32ce89b0b87e62fcdad28f0435e25c953f3`. Nie pobierano dodatkowych wag,
nie wynajmowano GPU i nie publikowano wyników.

Baseline: kod `94c97f8fa92f24c20c8f592f13480b7e33a9173a`, binarka SHA-256
`711e49dfb5d42427033d34e30093f660a28fe1c9468d30734dc4db5d14070db4`.
Pierwsza część kampanii używa poprawionej wersji cache, SHA-256
`7d6cc808cdfab6561679700509562938662a3733d85d147b1ed2fcec81e5189c`.
Końcowa binarka po naprawie konstruktorów: SHA-256
`9d82d28c0dc2d4f23815838b11e93d01c5618bb83d4181bb160850049444b3b0`.
Wszystkie cache i magazyny Batch są odseparowane od normalnych danych użytkownika.

## Eksporty A/B

Wejścia: osiem pozycji bench oraz 33 rekordy SystemOne z wcześniejszej kampanii.
Porównanie obejmuje pełne pliki `bench.jsonl` i `systemone.jsonl`, również
błędy i puste wyniki obecne w baseline.

| Wariant | Pełne pliki przed–po |
|---|---|
| Q8_0, FP16, tree | Bajtowo identyczne |
| Q4_K, FP16, tree | Bajtowo identyczne |
| Candle, FP16, single | Bajtowo identyczne |
| Readout zaokrąglany do BF16 | Bajtowo identyczne |
| Dwa porządki opcji | Bajtowo identyczne |
| Wyłączony prefix cache | Bajtowo identyczne |
| State cache 32 MiB, single | Bajtowo identyczne |
| Limit tree 128 tokenów | Bajtowo identyczne |
| BF16 z kontrolowaną tabelą per M | Bajtowo identyczne |

Dziewięć par po 41 rekordów daje 369 zgodnych par. Nie oznacza to 369 poprawnych
odpowiedzi modelu: błędy referencji również są zachowane. Wariantów domyślnego
FP16, FP32, accurate/balanced i wcześniejszych nadpisań środowiska, sprawdzonych
w poprzedniej kampanii, nie powtarzano bez nowej przyczyny.

### BF16 bez tabeli — zmienność obecna przed refaktorem

Pierwsza para bez przypiętego GEMM ma identyczne bench i primitive logits,
ale różni się natywnymi odpowiedziami dwóch rekordów multi. Drugi niezależny
eksport samej starej wersji także różni się od pierwszego, w innych dwóch
rekordach. To reprodukcja zmienności baseline, bez dowodu nowej regresji.
`basal compare` podaje max różnicę primitive logitów 0, więc samo jego krótkie
podsumowanie nie zastępuje porównania kompletnych odpowiedzi SystemOne.

GEMM bez tabeli mierzy heurystyki w runtime. Dla kontrolowanego A/B wygenerowano
rzeczywistą tabelę BF16 dla klas M=16,128,512. Następnie jako jawną diagnostyczną
tabelę per M powielono jeden jej algorytm na kształt na klasy do 8192 tokenów.
Tabela nadal ma `invariant: false`; nie jest kwalifikowanym tuningiem ani
deklaracją niezależności od składu partii. Ten sam artefakt w obu wersjach
dał pełną zgodność bajtową. Nie zmieniano polityki adaptive ani arytmetyki,
aby ukryć zmienność. Invariant search z samą klasą M=16 jest odrzucany
identycznie przez obie wersje: brak grupy poprawnej dla wszystkich M.

Pierwszą próbę baseline omyłkowo wywołano z `--gemm-table none`, które stara
wersja traktuje jako nazwę pliku. Zachowano jej błąd, a właściwy baseline
uruchomiono bez tego argumentu. Nowa wersja ma jawne słowo kluczowe `none`.

## Współbieżność i izolacja profili

Loadtest obu wersji: warmup 4, sekwencyjnie 16, następnie po 64 żądania przy
C1, C4, C16 i C64. Każda wersja obsłużyła 276 żądań, w tym 272 mierzone,
bez błędów i bez różnic odpowiedzi względem pierwszego wykonania tego wejścia.
Pełne obiekty answers pomiędzy wersjami są identyczne.

To krótkie próby closed-loop, bez wniosków o wydajności przy dowolnym ruchu.
Stosunek throughput after/before wyniósł od 0,9939 do 1,0117, a p99 przy C4
wzrósł z 82,92 do 85,19 ms. Nie ustalono progu SLO ani istotności statystycznej;
tych obserwacji nie opisujemy jako przyspieszenia lub dowodu zerowej regresji
wydajności. Przebiegi nie były równoległe z runnerami SDK.

Próba wielomodelowa używa kopii metadanych tych samych wag, z nazwą
`local-generic-mini`. Nie jest to drugi checkpoint ani kwalifikacja innego
modelu produktowego. Standardowy profil działa w FP16, generic w Q8 lub Q4;
oba korzystają z tej samej jawnej tabeli algorytmów, ale z odrębnych backendów.
Sprawdzono osobne uruchomienia generic Q8/Q4, wspólny proces w obu kolejnościach
inicjalizacji oraz zmianę wyłącznie generic z Q8 na Q4. Odpowiedzi standardowego
profilu pozostały bez zmian, a generic zgadza się z odpowiednią referencją solo.
Dodatkowo mieszane osiem żądań przy C8, naprzemiennie do obu profili, zachowało
pełne odpowiedzi względem solo dla Q8 i Q4. Próba nie izoluje zużycia VRAM ani
czasu wykonania współdzielonego GPU.

## Walidacja GEMM

Cztery jawne artefakty `sha256:` odrzucono przed wagami i przed zamrożeniem
planu: nagłówek F16 z wpisami BF16, przyszła wersja search 3 wobec obsługiwanej
2, brak kształtu 17920×1536 oraz nieakceptowany algorytm cuBLASLt.
Nie wykonywano fallbacku ani nieprzypiętego search.

## SDK i Batch API

Python sync/async/schema, JavaScript ESM/CJS, C# i Rust map oraz named APIs wykonały
rzeczywiste żądania podstawowych decyzji, tool/multi i classify/verify/compress/
extract. Kontrole obejmują także discovery, HTTP422 dla nieznanego modelu oraz
lokalne odrzucenie pustego media. Zbiorczy audyt: PASS, 27 odpowiedzi,
7 odczytów discovery i 14 kontroli błędów, bez rozbieżności. Porównanie decoded
answers oraz usage wykorzystuje bity binary64 według istniejącego auditora.
Python i npm zbudowano z lokalnych źródeł SDK; zależności Pythona instalowano
z przypiętymi hashami. C# zbudowano SDK .NET 8.0.427, bez ostrzeżeń.
Pierwszy install .NET w domyślnym katalogu tymczasowym zatrzymał się na quota;
instalacja z prywatnym katalogiem tymczasowym na dysku zakończyła się poprawnie.
Pierwszą kompilację runnera TypeScript poprawiono przez jawne podanie typów
Node; kod SDK nie wymagał zmiany.

Przy wyłączonej kwalifikacji background plik został poprawnie przyjęty,
batch przeszedł do oczekiwania `in_progress`, replay idempotency zwrócił ten
sam identyfikator. Aktywny input nie daje się usunąć (409); po anulowaniu
batcha usunięcie działa (200). Pobranie inputu odtworzyło dokładne bajty.

Osobna konfiguracja laboratoryjna włączyła solo GPU: limit pracy 2 GiB,
zapas online 512 MiB, granularity 2 MiB, koszt 120000 ns/token. To bootstrap
eksperymentu na krótkim korpusie, na podstawie obserwacji online i wcześniejszego
kontraktu mini, a nie kwalifikacja produkcyjna ani ścisły pomiar każdej alokacji.
API raportuje deklarację operatora jako `qualified`; ten napis sam nie dowodzi
zakresu pomiarów. Nie zmieniano konfiguracji produkcyjnej.

GPU wykonało 8/8, a następnie 32/32 rekordy bez błędów. Wszystkie 40 pełnych
body są zgodne z online po usunięciu latency. Restart zachował ukończony batch
32/32, katalog revision 1 i ten sam fingerprint. Doctor po pierwszym batchu:
zero quota drift, zero nierozliczonych settlement i zarezerwowanych bajtów,
bez problemów. Aktywne anulowanie osobnego batcha 128 rekordów nastąpiło po
zaobserwowaniu 2 ukończeń; terminalny stan `cancelled` zachował trzy wyniki.
Osobny batch 128 rekordów otrzymał SIGINT po zaobserwowaniu 6 ukończeń;
serwer zakończył się poprawnie, zachowując 7 wyników. Końcowa binarka wznowiła
pozostałe 121 i doprowadziła batch do 128/128, bez błędów i duplikatów
`custom_id`. Katalog revision 1 i fingerprint pozostały bez zmian.
Wszystkie 128 body oraz trzy zachowane po aktywnym cancel są zgodne z online:
łącznie porównano 171 zapisanych wyników Batch. Końcowy doctor zgłosił zero
problemów, quota drift, nierozliczonych settlement, otwartych sesji i
zarezerwowanych bajtów. Był to graceful stop/restart, bez SIGKILL lub utraty GPU.

## Wykryta regresja i naprawa

Publiczne `GpuBackend::load` i `load_split` przed poprawką zostawiały pustą
listę domyślnych kerneli attention. CLI stosuje plan, ale bezpośredni użytkownik
biblioteki trafiał w `no planned attention kernel for stage 0` podczas warmupu
lub pierwszego forwardu. Opus 5.5 zachował publiczne sygnatury i przed wagami
ustala wersjonowaną regułę domyślną dla każdego etapu. Wewnętrzne loadery
`unplanned` służą wyłącznie `RuntimeBackend::load_planned`, które stosuje plan
raz, przed K/V i lane'ami. Nie dodano polityki nazw modeli ani dynamicznego
fallbacku. Astra zakończyła przegląd `APPROVE`; hashe końcowych źródeł są zgodne
z przeglądanymi.

Istniejący CLI nie wywołuje bezpośrednio tych konstruktorów bez planu. Dodano
więc manualny przykład `crates/basal-gpu/examples/public_load.rs`, wykonujący
publiczny load, `Engine::new`, warmup i niepusty forward. To narzędzie pomiarowe,
bez frameworka i bramki CI. Logity odczytuje z dwóch końcowych tokenów promptu;
nie są odpowiedzią na pytanie ani miarą jakości modelu.

| Próba API | Przed naprawą | Po naprawie |
|---|---|---|
| Fused FP16 | Odtworzony błąd brakującego kernela etapu 0 | Warmup i finite logits, TC |
| Fused BF16 | Ten sam odtworzony błąd | Warmup i finite logits, TC |
| Fused FP32 | Ten sam odtworzony błąd | Warmup i finite logits, SIMT F32 |
| Candle FP16 — kontrola | PASS | PASS, bity logitów identyczne |
| Publiczny load Q8, Q4, readout BF16 | Nie powtarzano | PASS, warmup i finite logits |
| Nieistniejący GPU 99, split z duplikatem GPU 0 | Nie powtarzano | Oczekiwane błędy przed wagami |

Snapshot przed naprawą odtworzono z poprzedniego archiwum źródeł; uruchomiono
dokładnie ten sam przykład. Hash binarki sprzed naprawy:
`70534ebaf73673a698a59da349aa9c57c94b9bcdc13cd92cf66e7b494828acad`,
po naprawie:
`960d5a3173218aad51302a8f2432bddbb89668f9e6543cb476d69a66896b630b`.
Przy pierwszym buildzie starszego snapshotu wspólny target użył nowszego
artefaktu ze względu na mtime źródeł. Wykryto identyczny hash przed próbą;
wymuszono ponowną kompilację, otrzymano inną binarkę i dopiero jej użyto do
reprodukcji. Przed końcowym buildem przywrócono kompilację bieżących źródeł.

Ta naprawa zachowuje sygnatury dwóch konstruktorów. Nie wycofuje zamierzonych
zmian bibliotecznego API z całego refaktoru, m.in. tabel przyjmowanych jako
zwalidowane dane zamiast ścieżki i stosowania `load_planned`. Nie zmieniono
kontraktu HTTP ani SDK.

## Końcowe sprawdzenia i sprzątnięcie procesów

Po naprawie wykonano jeszcze pełny eksport FP16 tree na końcowej binarce:
osiem bench i 33 SystemOne są bajtowo identyczne z wersją sprzed refaktoru.
`basal compare`: zero różnic tokenów, logitów i prawdopodobieństw. Wraz
z dziewięcioma parami wariantów daje to 410 zgodnych par rekordów tej kampanii.

Autor wykonał fmt, Clippy default/CUDA/Intel i rustdoc dla tych konfiguracji.
Jedyny późniejszy wyjątek lintów dotyczy dead code w buildzie Intel bez CUDA;
po nim powtórzono właściwy wariant Intel. Integrator zbudował końcową binarkę
CUDA i ponownie wykonał workspace release all-targets all-features Clippy
z `-D warnings`: PASS. Końcowe sumy źródeł odpowiadają przeglądowi Astry;
`git diff --check`: PASS. Nie zmieniano zależności, więc nie powtarzano
wcześniejszego nieudanego cargo-deny bez nowej przyczyny: nadal znane są
`RUSTSEC-2026-0385` (`ctutils 0.4.3`) i `RUSTSEC-2026-0386` (`tokio 1.53.2`),
obecne również na baseline.

Wszystkie własne serwery zatrzymano SIGINT i potwierdzono ich zakończenie.
Końcowa zajętość GPU wyniosła 2 MiB. Nie usuwano stanów pomiarowych Batch,
nie zmieniano normalnego cache użytkownika i nie tworzono commita.

## Granice i artefakty

Metal nie jest lokalnie dostępny; nie ma drugiej karty ani urządzenia Intel.
Brak lokalnych wag Gemmy i innych modeli. Testy metadanych generic nie
zastępują tych braków. Nie wykonywano jakościowego porównania z FP32 upstream,
wynajmu GPU, zmian zależności ani publikacji. Wcześniejsze dwa FAIL cargo-deny
pozostają poza dowodem regresji refaktoru.

Kampania: `model-isolation-local-expanded-2026-10-26`. Dane surowe i narzędzia
lokalne są poza Gitem, w odpowiadającym katalogu measurement-artifacts.
URI: brak (niepublikowane). Lokalne archiwum:
`.cache/measurement-artifacts/model-isolation-local-expanded-2026-10-26/measurement-data.tar.gz`.
Rozmiar: 602 865 B; SHA-256:
`2678cad50054f76375ece0eed9bc6fcc7eec0f6c647ea82afef31f17626e6a31`.
Rozpakowanie do osobnego katalogu potwierdziło 128/128 wpisów SHA256SUMS;
niezależny replay auditora SDK z rozpakowanych danych także dał PASS.
Nie weryfikowano pobrania z HF, ponieważ niczego nie wysłano.

Wszystkie logiczne ścieżki zaczynają się od
`reports/model-isolation-local-expanded-2026-10-26/`. `results.json` zawiera
zestawienie i wersje, `exports/` pełne pary JSONL i niezgodne próby BF16,
`inputs/` korpusy i tabele, `sdk/` referencje i obserwacje czterech klientów,
`isolation/` odpowiedzi obu profili, `batches/` wyniki i sprawdzenia 171 body,
`public-api/` pomiary konstruktorów i odwracalne patche ich naprawy, a
`source/` patch śledzonych plików oraz nowe pliki Rust.

Po sprawdzeniu SHA archiwum rozpakuj je do nowego katalogu, wykonaj
`sha256sum -c SHA256SUMS` i zastosuj `REPRODUCE.md`. Prywatne logi, konfiguracje,
stany baz, binarki, narzędzia i wagi nie należą do tego zestawu danych.
