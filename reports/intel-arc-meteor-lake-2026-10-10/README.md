# Intel Arc Meteor Lake — 2026-10-10–11

Backend Intel Vulkan uruchamia Basal 1.5 mini, 4.5B i max na zintegrowanym Arc.
Forward używa FP16, z akumulacją GEMM, RoPE, attention i logitami liter w FP32.
Instrukcja budowania i uruchomienia: [Intel Arc](../../docs/INTEL.md).

## Poprawność

Porównanie z zapisanymi referencjami upstream FP32, na identycznych tokenach:

| Model | Zadania | Zgodne decyzje avg i cal | Max Δ logitu | Max Δ p_cal |
|---|---:|---:|---:|---:|
| mini | 44 | 44/44 | 0,021352 | 0,004326 |
| 4.5B | 44 | 44/44 | 0,039001 | 0,005328 |
| max | 44 | 44/44 | 0,261522 | 0,008733 |

Odpowiedzi zgodnego z upstream endpointu `/v1/basal` porównano osobno:

| Model | Przypadki / pytania | Zgodne porównane decyzje i pola struktury | Max Δ prawdopodobieństwa |
|---|---:|---|---:|
| mini | 34 / 55 | tak | 0,003562 |
| 4.5B | 34 / 55 | tak | 0,003064 |
| max | 4 / 10 | tak | 0,001425 |

Mini i 4.5B obejmują choice, multi, act i evidence. Max obejmuje
`m15-facts-amounts`, `m15-mixed`, `m15-evidence-choice-en` i `cx-01-ticket-fan-out`.
To zgodność w opisanych próbkach, nie gwarancja jakości dla dowolnych danych.
Starsze referencje mają różnice przygotowania surowych multi/facts/act;
dlatego ocenę tych funkcji oparto na rzeczywistych odpowiedziach endpointu.
W nowych wejściach mini i max zaktualizowano wyłącznie `request.model`.

Zmiany cache, pakowania, ładowania i synchronizacji sprawdzono bitowo między
wariantami Intel: po 44 zadania mini i max oraz 3 zadania 4.5B miały identyczne
logity i prawdopodobieństwa. Pełny eksport 4.5B pochodzi sprzed zmiany ładowania
i kolejki; końcowy wariant potwierdzono na trzech zadaniach i powtórzeniach HTTP.
Mini ma pełny końcowy eksport Tree; max pełny końcowy Single oraz cztery przypadki API.

## Wydajność i pamięć

Mediany sekwencyjnych żądań `/v1/systemone`, po dwie próbki na klasę:

| Model | Krótkie pytanie | Kilka pytań / wspólny stan | Dokument 50 zdań |
|---|---:|---:|---:|
| mini | 0,938 s | 4,825 s | 9,744 s |
| 4.5B | 2,925 s | 14,895 s | — |

Łącznie z fazami współbieżności 1 i 2 ukończono 18 żądań mini oraz 12 żądań 4.5B:
bez błędów HTTP i zmian odpowiedzi. Dwa klienty zwiększały opóźnienia;
próbka nie uzasadnia deklaracji wzrostu przepustowości ani produkcyjnego p95/p99.
Maksymalne zaobserwowane alokacje GPU: mini 3,03 GiB, 4.5B 8,91 GiB.
Minimalny zapas RAM z uwzględnieniem cgroup: odpowiednio 14,62 i 5,49 GiB;
globalny wzrost swapu: 0 i 35,86 MiB. Próbkowanie co 200 ms nie mierzy dokładnego szczytu.

Benchmark max **nie został uruchomiony**: kontrola wstępna wykazała 14,69 GiB
dostępnego RAM przy wymaganiu 20,80 GiB wag + 2 GiB rezerwy. Limitów nie zmieniano.
Osobny zwykły serwer max poprawnie obsłużył po dwa wywołania `m15-mixed` przez
oba endpointy. Cztery odpowiedzi były identyczne z poprawnym eksportem,
po pominięciu wyłącznie dynamicznego `usage.latency_ms`. Minimalne fizyczne
MemAvailable wyniosło 4,92 GiB. To weryfikacja funkcjonalna, nie zastępstwo
benchmarku ani dowód zapasu dla dowolnego promptu i współbieżności.

W podstawowym pomiarze próg długiego workera wynosił 128 tokenów; wszystkie
wejścia trafiały do niego. Dodatkowy pomiar mini z progiem 256 ukończył sześć
żądań bez błędów i zmian odpowiedzi. Następnie dwukrotnie wysłano dokument
(1142 spakowane tokeny), a po 750 ms krótkie pytanie (168 tokenów).
W obu próbach krótkie pytanie skończyło się przed dokumentem; odpowiedzi
obu wejść pozostały identyczne z rozgrzaniem.

## Sprawdzone optymalizacje i poprawki

- GEMM mini, trzy identyczne zadania, średnio 236,67 tokenu: kafel 32×32×32
  dał 1543 i 1561 ms/zadanie, 16×32×32 — 1931 i 1973 ms, 32×64×32 — 1750 ms.
  Zachowano 32×32×32. Końcowe pomiary po zmianie ładowania i synchronizacji:
  1706 i 1662 ms/zadanie. To profile sekcji, nie czas żądania HTTP;
  nie przeprowadzono osobnego strojenia kafelków dla wszystkich modeli.
- Cache stałych prefiksów pomija 53/66 tokenów. Embedding i `lm_head` pozostają
  w mapowanym checkpointcie; przesyłane są potrzebne wiersze. Ostatnia warstwa
  ogranicza o_proj i MLP do pozycji odczytu. Workery współdzielą niemutowalne wagi.
- Pierwsze ładowanie max wyczerpało RAM przez buforowanie drugiej kopii wag.
  Bezpośrednie mapowanie docelowych alokacji na zintegrowanym GPU usunęło ten problem.
- Wczesny max zwrócił błędne odpowiedzi Tree, mimo poprawnych wyników Single,
  a następnie i915 zgłosił GPU HANG i reset kontekstu. Synchronizacja każdej
  warstwy ograniczyła kolejkę; po poprawce przeszły pełne 44 zadania,
  cztery przypadki API i powtórzenia HTTP. Nie ustalono przyczyny w sterowniku.
  W okresie końcowych przebiegów nie odnotowano nowych wygasłych fence'ów,
  GPU HANG ani resetów kontekstu. Nie wymuszano resetu w celu sprawdzenia obsługi błędu.

## Warunki i zakres kontroli

Core Ultra 9 185H, Arc Meteor Lake (`8086:7d55`), 32 GB nominalnego RAM
(30,7 GiB fizycznej pamięci), limit cgroup 26,70 GiB, i915,
Mesa 26.2.2-arch1.1, kernel 7.2.5-3-omarchy, Rust 1.95.0, wgpu 30.0.1.
Bazowy commit: `e9444bb0a64a3fc90414a0c0bb95c44208d56f6b`.
Rewizje modeli i sumy SHA-256 zweryfikowano względem manifestów referencji FP32.
Jeden proces modelu naraz; pulpit aktywny, zegary i moc niekontrolowane.
Wczesne eksporty numeryczne nakładały się na pobieranie i kompilację;
ich czasy nie są miarą przyspieszenia. Pomiary HTTP odbyły się bez tych prac.
Współdzielonego RAM i budżetu Vulkan nie należy sumować jako osobnych pul pamięci.

Przeszły: formatowanie, Clippy i Rustdoc dla Intel oraz Linux bez GPU,
kompilacja release Intel, statyczna walidacja siedmiu shaderów WGSL,
cargo-deny, kontrola wyjątku `paste`, actionlint/ShellCheck i `git diff --check`.
Przejrzano granice ownership, FFI, błędów oraz współdzielenia GPU.
Nie dodano testów ani frameworków. CUDA/Metal i zdalne CI nie były tutaj uruchamiane.
Nie zmierzono innych kart Arc, maksymalnego kontekstu ani długotrwałego obciążenia.
Backend nie korzysta z XMX; pomiary nie dowodzą globalnie optymalnej wydajności.

Pełne lokalne dane i szczegóły odtwarzania zachowano w tym katalogu,
w tym `DETAILS.local.md`; są wyłączone z Gita zgodnie z polityką raportów.
[prepare-inputs.py](prepare-inputs.py) odtwarza użyte podzbiory z referencji FP32
do nowego katalogu. Polecenia eksportu, porównania i profilowania opisuje
[instrukcja backendu](../../docs/INTEL.md).
