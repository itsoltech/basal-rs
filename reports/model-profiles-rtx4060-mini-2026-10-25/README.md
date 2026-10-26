# Izolacja profili modeli: RTX 4060 / basal-2.0-mini

## Cel i zakres

Oddzielenie katalogu i polityk modeli w CLI od typowanego planu wykonania
w GPU, bez zmian transportu HTTP, SDK ani arytmetyki kerneli. Implementację
wykonał Opus 5.5; niezależny przegląd wykonała Astra. Poniższa weryfikacja
dotyczy lokalnej RTX 4060 i jednego checkpointu; nie kwalifikuje innych modeli.

Wynik: **PASS bitowej zgodności dla zmierzonej konfiguracji**. Dziesięć
wariantów przed/po oraz dwa dodatkowe warianty GEMM dały identyczne bajty
obu plików wyników. Kontrole cache, migawki tabeli i HTTP również przeszły.
Kontrole jakości Rust przeszły z wyjątkiem **istniejącego wcześniej FAIL
cargo-deny** opisanego niżej. Kampania rozpoczęta 25 października została
zakończona 26 października 2026 r.

Katalog i wersjonowane profile są w `basal-cli/src/models.rs`, niezmienny
plan w `basal-gpu/src/plan.rs`, a pozyskiwanie i walidacja tabel w
`basal-cli/src/gemm_artifact/`. Rodzinne implementacje forward pozostały
wspólne. Dodanie profilu lub tabeli nowego modelu nie rozszerza automatycznie
list artefaktów dotychczasowych profili. Zmiana wspólnej implementacji
nadal wymaga przeglądu i kwalifikacji profili, które z niej korzystają.

## Wersje i wejścia

| Element | Wartość |
|---|---|
| Baseline źródeł | `94c97f8fa92f24c20c8f592f13480b7e33a9173a` |
| SHA-256 lokalnej binarki baseline | `711e49dfb5d42427033d34e30093f660a28fe1c9468d30734dc4db5d14070db4` |
| SHA-256 końcowej binarki kandydata | `e600140776f03e6cf7a52dd8a740b09ed03d06fc03499951acaa242664a9cd67` |
| SHA-256 patcha plików śledzonych | `619e27d95d24ff848cdc84c5520f5353c236469508d3f100beb2923160f1b186` |
| Model | `Remek/basal-2.0-mini@cb81b32ce89b0b87e62fcdad28f0435e25c953f3` |
| GPU | NVIDIA GeForce RTX 4060, 8188 MiB, compute capability 8.9 |
| Sterownik / CUDA / cuBLASLt | 615.78.08 / 12.9.1 (nvcc 12.9.86) / 120901 |
| Toolchain | Rust 1.95.0, release; nvcc host GCC 14.2.1, nagłówki glibc 2.40; własne PTX 80, 89 i 90 |
| Tabela GEMM | zasób repozytorium RTX 4060 / basal-2.0-mini / f16, search v2, invariant; 100 wpisów |
| Tożsamość znormalizowanej tabeli | `2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96` |

Baseline powstał z archiwum Git sprzed refaktoru, tym samym lokalnym
toolchainem co kandydat. Eksport tree był bajtowo identyczny z wcześniej
zainstalowaną wersją `07585380` dla obu plików wyników. To ogranicza ryzyko,
że samo odtworzenie toolchainu zmieniło referencję.

Wejścia to osiem pytań bench z przykładów routing/tickets oraz 33 żądania
SystemOne: cztery z wcześniejszej lokalnej kampanii i przypadki z
`tools/reference/systemone_cases.jsonl` i `systemone_cases_1.5.jsonl`,
z jawnym modelem `basal-2.0-mini`. Obejmują choice, noul, score, multi,
act, facts oraz błędy capabilities/walidacji. W baseline 25 żądań ma
niepuste `items`, siedem zawiera `error`, a jedno ma puste `items` bez
tego pola. Nie traktujemy wszystkich 33 jako udanych inferencji.

## Macierz decyzji przenoszonych z dotychczasowego kodu

To zapis reguł z kodu, a nie wynik pomiarów wszystkich wymienionych kart.

| Warunek | Zachowane rozstrzygnięcie |
|---|---|
| Llama, `dtype: auto` | FP16 |
| Llama, `accurate` | przenośny attention hi/lo |
| Llama, `balanced` | QK1 tylko CUDA FP16 + fused + readout FP32; inaczej błąd |
| Basal 2.0 mini/4.5B/max, `auto`, bez `BASAL_ATT`, wszystkie etapy na RTX 6000 Ada | QK1 przy powyższych opcjach |
| Pozostałe Llama / inne GPU | dotychczasowa wersjonowana reguła urządzenia, z wyjątkami dtype/shape i diagnostyką |
| Gemma, `dtype: auto` | FP16 na karcie pierwszego etapu z listy profilu; inaczej BF16 |
| Omni 1.0 mini/max na RTX 6000 Ada, FP16, `auto` | iloczyny attention FP16, osobno dla każdego etapu |
| Omni 1.1 max na RTX 6000 Ada | dekoder FP16, iloczyny attention FP32 |
| Omni Ultra na H100 PCIe, `auto`, FP16/readout FP32 | fuzje FP16; iloczyny attention FP32 |
| Gemma `balanced` | błąd |
| Jawne CUDA `accurate` / `balanced` z `BASAL_ATT=simt` | jawny tryb wygrywa; `auto` respektuje zmienną |
| BF16 Hopper z `wg`/`wgp`/`wgp64` | zachowany dalszy wyjątek dispatchu w `fused.rs` |
| Omni 1.0 max, attention FP16 na pierwszym etapie | dotychczasowe zbieranie wracających żądań przez 10 ms |
| GEMM `auto`, ciepły zgodny cache v1 | zachowanie faktycznych algorytmów i import bez nadpisania źródła |
| GEMM `auto`, zimny cache | jawna lista artefaktów profilu; przy braku Llama search, Gemma fixed per-prompt |
| GEMM `pinned` / `sha256:` | jawny tryb odtwarzalny, bez zastępowania brakującego artefaktu wyszukiwaniem |

## Metoda A/B

Eksporty przed/po używają identycznych wejść, tabeli i opcji. Warianty:
tree, single, budget, accurate, balanced, auto z `BASAL_ATT=simt`, auto
z `BASAL_NORM=register BASAL_SILU_BLOCK=128`, FP32/single bez tabeli,
accurate z `BASAL_ATT=simt` oraz balanced z tą samą zmienną.

Kryterium refaktoru to bajtowa zgodność całych `bench.jsonl` i
`systemone.jsonl` w każdej parze (`cmp`), uzupełniona `basal compare`.
Pliki zawierają tokeny, logity, rozkłady i odpowiedzi wraz z błędami;
nie zawierają czasów pomiaru. Manifesty z czasami i lokalnymi ścieżkami
pozostają prywatne. Samo zgodne argmax nie wystarcza.

## Wyniki A/B i granic GEMM

| Wariant | Pełne bench/SystemOne przed–po | `basal compare` |
|---|---|---|
| tree / single / budget | 3/3 pary bajtowo identyczne | brak różnic tokenów, logitów i rozkładów |
| accurate / balanced | 2/2 pary bajtowo identyczne | jak wyżej |
| auto + simt / norm + SiLU | 2/2 pary bajtowo identyczne | jak wyżej |
| FP32 / single | bajtowo identyczne | jak wyżej |
| accurate + simt / balanced + simt | 2/2 pary bajtowo identyczne | jak wyżej |
| `pinned` / `sha256:` względem baseline tree | 2/2 pary bajtowo identyczne | jak wyżej |

Każda para obejmuje osiem rekordów bench i 33 rekordy SystemOne. Maksymalna
różnica logitów i TV w odpowiednich sekcjach porównania wynosi 0,0;
zgodność argmax bench to 8/8 w każdej parze. Zachowane zostały też błędy
capabilities/walidacji i odpowiedzi obu dialektów. Raporty `compare`
zawierają dodatkowo istniejące porównania między dialektami endpointów;
ich różnice nie oznaczają regresji przed/po. O wyniku decyduje również
`cmp` całych JSONL.

| Kontrola | Wynik |
|---|---|
| Zimny cache `auto` | przypięty artefakt `bundled:2e9ba4f7…`, bez search |
| Import starego cache | zachowane 100 wpisów i SHA-256 pliku źródłowego `0ec5907f19e333643e99775f7d5f7fbbe6ed3030518bcfda45f8fd3b54611c44` |
| Własny artefakt lokalny | import 99 wpisów jako `adaptive/local:aab794b5…`, bez zastąpienia tabelą buildu i bez zmiany źródła |
| Migawka jawnej tabeli | podmiana pliku źródłowego po freeze, przed końcem ładowania wag, nie zmieniła eksportu |
| Przyszła wersja search, błędny dtype wpisów, brak kształtu, brak digestu, inne GPU, inna biblioteka | wszystkie sześć prób zakończonych błędem przed ładowaniem wag |

Próba lokalnego artefaktu używa syntetycznej kopii tabeli: usunięto wpis
M=16 dla 1536x1536; następny wpis M=32 ma ten sam algorytm. Nie wykonano
nowego autotuningu i nie kwalifikowano tej kopii jako profilu odtwarzalnego.
Jej digest to `aab794b568140388f571ae7294b6d8cabd5cd312cfb28c19f0ab82307d0e01ff`,
a SHA-256 niezmienionego źródła
`dd0559514b9860544f6482790fc1db5cdcda39bec027ff54ec214d675abca776`.

HTTP C4: po cztery żądania dla zimnego cache, importu starej tabeli oraz
syntetycznej tabeli lokalnej. Wszystkie zakończyły się powodzeniem.
Odpowiedzi i usage były dokładnie równe baseline po pominięciu wyłącznie
`usage.latency_ms` i kolejności kluczy obiektów. `/health` był identyczny;
`/v1/models` różnił się tylko kolejnością kluczy, po kanonizacji był równy.
Uruchomione serwery zostały zamknięte; końcowy odczyt zajętości GPU: 2 MiB.

Mała próba `benchmark` (cztery żądania sekwencyjne, bez warmupu) zakończyła
się statusem `completed` i zapisała rzeczywisty plan:
`basal-execution-plan.2`, `basal-2.0-mini.v1`, `basal-kernels.1`,
`attention-default.1 → attn_tree_tc`, GPU 8.9 / 24 SM, przypięta tabela
i fallback `refused`. Służyła sprawdzeniu serializacji planu;
nie wyciągamy wniosków o przepustowości ani opóźnieniach.

## Przegląd i kontrole jakości

Pierwszy przegląd wskazał cztery problemy: nieprzypięty
autotuning w strict przy złym dtype wpisów, mutowalną jawną ścieżkę tabeli,
niepełną tożsamość GPU/reguły domyślnej oraz akceptację przyszłej wersji
search w strict. Opus poprawił wszystkie; Astra potwierdziła zamknięcie
w drugiej rundzie. Pozostałą uwagę o preflight jawnej tabeli Gemmy
poprawił Opus i sprawdził integrator: invariant, niepuste wpisy właściwego
dtype i kontrola algorytmów przed wagami, bez zmiany częściowego pokrycia
i fallbacku Gemmy. Nie przeprowadzono trzeciej rundy delegowanego przeglądu.

Końcowy Clippy wykrył jeszcze `manual_map` w tym walidatorze; Opus poprawił
sam zapis. Dziesięć eksportów oraz `pinned` i `sha256:` ponowiono końcową
binarką o SHA-256 podanym wyżej. HTTP, migawkę i przypadki negatywne
sprawdzono przed tą stylistyczną poprawką Gemmy, binarką
`d4e2d3f84804ef7ed12834029f5942769d19070b100ab2a8daa2afa5507e9a2d`.

| Kontrola końcowa | Stan |
|---|---|
| `cargo fmt --all --check`, `git diff --check` | PASS |
| Clippy: workspace, release, locked, all-targets, all-features, `-D warnings` | PASS |
| Rustdoc: workspace, release, locked, all-features, private items, `-D warnings` | PASS |
| Rustdoc: workspace, release, wariant Intel bez CUDA | PASS |
| Kontrola wyjątku `paste` | PASS |

`cargo deny --locked --all-features check` zwraca **FAIL** dla
`ctutils 0.4.3` (RUSTSEC-2026-0385) oraz `tokio 1.53.2`
(RUSTSEC-2026-0386). Taki sam wynik potwierdzono dla archiwalnego baseline.
`Cargo.lock` nie został zmieniony; refaktor nie dodaje wyjątków advisory.
Nie zmieniano polityki zależności, CI ani SDK. Autor dodatkowo sprawdził
Clippy zmienionych crate'ów bez GPU, z CUDA i z Intel; nie wykonano lokalnie
kontroli Metal wymagających macOS. Nie dodano frameworków ani testów do CI.

## Artefakty i ograniczenia

Surowe dane pozostają poza Gitem w
`.cache/measurement-artifacts/model-profiles-rtx4060-mini-2026-10-25/`.
Nie publikowano ich w HF; URI publiczne nie istnieje.

| Element | Wartość |
|---|---|
| Lokalne archiwum | `measurement-data.tar.gz` w katalogu kampanii |
| SHA-256 archiwum | `9791845b1eba8766a1f78b36427713ad2ebf1c36587eeeaf9afc557f43c92260` |
| Rozmiar | 612 538 B |
| Weryfikacja | rozpakowanie do osobnego katalogu, 81/81 wpisów `SHA256SUMS` zgodnych |
| Katalog w archiwum | `reports/model-profiles-rtx4060-mini-2026-10-25/` |

Archiwum zawiera wejścia, po dwa JSONL dla 10 wariantów baseline i 12
wariantów kandydata, 12 raportów compare, dane HTTP, wejścia cache,
rzeczywisty plan, `verification.json`, `source.patch`, cztery nowe pliki
źródłowe w `source-overlay/` i instrukcję `REPRODUCE.md`. Patch oraz overlay
odtwarzają źródła nad commitem baseline; nie są osobnym commitem.
Prywatne logi toolchainu, manifesty z lokalnymi ścieżkami i binarki
pozostają poza archiwum.

Po lokalnym rozpakowaniu sprawdź `sha256sum -c SHA256SUMS`. Szczegółowe
polecenia budowania, eksportu i porównania są w `REPRODUCE.md` archiwum;
używają przypiętego checkpointu, dostarczonych wejść i nowych katalogów
wynikowych. Nie wykonano wysyłania ani sprawdzenia pobrania z HF.

Nie wykonano runtime na innych GPU, Metal, Intel, Gemma ani innych modelach
Llama. Nie sprawdzano podziału warstw ani współistnienia różnych modeli;
nie wykonano kwalifikacji wydajności. Izolacja profili nie zapewnia izolacji
VRAM i czasu dostępu do GPU.

Jawna ścieżka tabeli trafia teraz do magazynu; pierwszy import wymaga
zapisywalnego cache. Błąd identyfikacji GPU podczas freeze kończy start,
zamiast zastępować kartę nieznaną. Plan ma nowy format diagnostyczny;
raport benchmarku zawiera `execution_plan`. Format profilu batchy pozostaje,
ale hash surowego pliku tabeli może się zmienić przy istniejącym artefakcie
z innymi metadanymi pomijanymi w normalizacji. Nie migrowano ani nie
kwalifikowano istniejących trwałych zadań Batch API.

FP32 w tej kampanii jest trybem poprzedniego i nowego runtime'u, a nie
nową referencją FP32 upstream. Nie zmieniano numeryki i nie deklarujemy
nowej oceny jakości względem upstream. Zmiany arytmetyki lub przenoszenie
profili na nowe warianty kerneli wymagają osobnej kwalifikacji.
