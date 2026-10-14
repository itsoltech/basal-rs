# Basal v2 i Omni — porównania H100, 2026-10-14

Pięć podstawowych checkpointów BF16 działa natywnie na CUDA. Oba modele omni przeszły
porównania tekstu, obrazu default/high, audio i obu kolejności mediów mieszanych.
Na stałym korpusie orders=1 prompty, token IDs, kody i argmax każdego odczytu zgadzają
się z upstream FP32. Prawdopodobieństwa różnią się numerycznie; osobne porównanie BF16
z auto wykazało jeden zmieniony wybór w niejednoznacznym pytaniu audio.

H100 PCIe 80 GB, driver 570.195.03, limit mocy 350 W; CUDA 12.9.1, PyTorch 2.11.0+cu128,
Transformers 5.17.0. Referencja to eager z tagu Basal v2.0.0,
`af1ce43a1770dd4fdfc1611e3fe720b93d42fc73`; rewizje pięciu modeli w danych (`readouts.json`).
Wagi BF16 nie oznaczają BF16 wykonania: dotychczasowe auto Llamy to FP16, auto Gemmy to BF16.

| Model | Native dtype | Argmax 49 pytań | Argmax gałęzi żądań | Max \|ΔP calibrated\| wobec FP32 |
|---|---|---|---|---|
| basal-2.0-mini | f32 | 49/49 | 53/53 | 1.21593e-05 |
| basal-2.0-mini | f16 | 49/49 | 53/53 | 0.00390589 |
| basal-2.0-4.5B | f32 | 49/49 | 53/53 | 1.31428e-05 |
| basal-2.0-4.5B | f16 | 49/49 | 53/53 | 0.0080865 |
| basal-2.0-max | f32 | 49/49 | 53/53 | 2.83867e-06 |
| basal-2.0-max | f16 | 49/49 | 53/53 | 0.00221053 |
| basal-omni-1.0-mini | f32 | 49/49 | 63/63 | 0.000151992 |
| basal-omni-1.0-mini | bf16 | 49/49 | 63/63 | 0.0270085 |
| basal-omni-1.0-max | f32 | 49/49 | 63/63 | 0.000326633 |
| basal-omni-1.0-max | bf16 | 49/49 | 63/63 | 0.0503154 |

Korpus to 44 pytania basal-bench z seed=0 i pięć dodatkowych choice z 10/11/26/27/255
opcjami. Żądania obejmują istniejące System One, przykłady complex upstream, cztery
cele multi, tool single/multi, score 255 i odrzucenie choice 256. Omni ma dodatkowo
pięć kombinacji mediów, po dwa pytania. Liczby 53/63 oznaczają odczyty gałęzi,
nie liczbę niezależnych żądań. Dane rozkładów różnic: `readouts.json`.
Porównanie argmax nie oznacza identyczności oczekiwanej wartości score lub confidence.

Osobne referencje upstream BF16: `bf16.json`. Max różnica kalibrowanej
probability native BF16 / upstream BF16 to 0,06962 dla mini i 0,02935 dla max
(łącznie tekst i media). W fixed orders=1 nadal zgadza się każdy z 49 i 63 argmax.
Native Gemma liczy uwagę w FP32; nie odtwarza bitowo zaokrągleń wszystkich operacji
upstream BF16. Także upstream BF16 różni się od upstream FP32. Nie deklarujemy
ogólnej równości decyzji przy małych marginesach.

PL i pełne okno audio (`media-pl-and-window.json`): po cztery dodatkowe pytania na
każdym omni — obraz, audio, media mieszane po polsku i 30-sekundowy sinus po angielsku.
Wszystkie token IDs, usage i argmax są zgodne. Max |ΔP| FP32: mini 1,09e-4,
max 4,19e-3 (polski mixed); sam pełny sinus to 5,22e-6 / 1,40e-6.
Wektory procesora i dekodery formatów są porównane osobno w [raporcie CPU](../upstream-2.0-cpu-2026-10-14/README.md).

Polityki kolejności (`orders-policy.json`): trzy pytania 2/3/27 dla 2le3 na Llama mini
oraz score/audio/mixed z auto na omni mini, FP32. Wszystkie usage są dokładnie zgodne,
wybory zgodne; max |ΔP| 1,22e-4. Oddzielny eksport dwóch kolejności również porównał
te same 2/3/27 opcji. [Kalibracja](../upstream-2.0-boundaries-2026-10-14/README.md)
obejmuje rzeczywisty wybór pliku o2, brak/per-type/pooled certyfikatu i mismatch.

HTTP BF16 auto (`http.json`): metadane, tool i client --value, score 255, obraz/audio/mixed,
422 dla 256 opcji, błędnego JSON/base64, thinking i evidence. Wszystkie liczniki usage
pozytywnych żądań są zgodne. W media-audio native wybrał audio, upstream BF16 obraz
(przy zbliżonych prawdopodobieństwach około 0,39); FP32 auto na tym wejściu zachował
wybór. Ten wybór native BF16 również zgadza się z upstream FP32; pełne decyzje analizuje [osobny raport](../upstream-2.0-decisions-2026-10-14/README.md). Score 255 różnił się o -0,45297 w oczekiwanej wartości na skali 0–254.
Powtórzony obraz po innych mediach ma dokładnie te same native answers.
TypeSafe noul zachowuje swój prostszy obiekt odpowiedzi.

Partie (`batching.json`): wszystkie pięć modeli, 49 pytań w single/budget/tree,
bez jawnej tabeli GEMM. Argmax jest zgodny we wszystkich wariantach. Oba omni są
bitowo równe wektorowo; Llamy wykazują różnice do 0,01649 w logitach przy domyślnym
wyborze algorytmu matmul. Ten pomiar nie potwierdza niezmienności Llamy bez tabeli
ani cache stanów. Omni nie stosuje cache pomiędzy żądaniami.

Stara ścieżka (`legacy.json`): trzy basal-1.5, explicit bundled H100/cuBLASLt120901
FP16 table, wszystkie single/budget/tree. 44 pytania i 13 rekordów żądań na konfigurację.
Wektory logitów/probability oraz obiekty answers, usage i błędy są bitowo identyczne
w A/B z commitem `4f68eb216ec3570f4e44c8262af566a0b0173ef7`.
Sprawdzono reprezentacje binarne liczb, również signed zero. To dowód dla tych
konfiguracji; basal-1.0 nie był objęty nową sesją pomiarową.

Admission i lifecycle (`admission.json`): jeden serwer z basal-1.5-mini i omni-mini,
właściwe metadane i model domyślny z konfiguracji. Przy max_inflight=4 dwanaście
równoległych żądań dało 4×200 i 8×529 z Retry-After=1. Bezpośrednio po anulowaniu
prac kolejny request dostał 529, po ich ukończeniu 200. Stara odpowiedź powtórzona
po mediach pozostała dokładnie równa. SIGINT zakończył proces kodem 0.

Hostowe reguły zbiorów/narzędzi ponownie porównano po końcowej zmianie parsera:
280/280 (`readouts-final-host.json`), 17/17 act/certyfikaty (`certificates-final-host.json`),
4/4 błędne definicje tools (`negative-tools.json`). Żadne narzędzie nie jest wykonywane.

Fingerprint (`fingerprint.json`) wiąże 74 pliki źródeł/zależności i hashe 376 surowych
artefaktów zachowanych w ignorowanym cache. Końcowy binary hash dotyczy świeżego
sprawdzenia admission; główne eksporty powstawały w etapach portu. Po końcowym eksporcie
omni-mini nie zmieniono jego decoderów/encoderów ani procesora. Późniejsze poprawki
objęły CLI, zachowanie legacy export i limity serwera/parsera; ich powierzchnie
mają osobne świeże porównania. Surowe manifesty i logi nie są publikowane w Git.

Kontrole kompilatora i lintów (`verification.json`) przeszły lokalnie na Linux,
Intel i CUDA; nie uruchomiono CI w GitHub ani lokalnej kompilacji Metal.
Nowe modele tekstowe mają wspólny runtime Llama na Metal/Intel, lecz bez nowego
pomiaru na tych kartach. Omni jest obsługiwane wyłącznie na CUDA. Nie dodano testów,
snapshotów ani bramek CI; nie trenowano kalibracji.

Syntetyczne PNG i sinusy nie są oceną rozumienia realnych obrazów/mowy, WER ani
certyfikatem multimodalnej kalibracji. Kalibracja mediów to text; mismatch kolejności
to unvalidated. Nie mierzono szybkości portu, pełnego kontekstu ani wszystkich
kodeków przez forward GPU. H100 usunięto po pobraniu danych, co potwierdzono przez API;
sesja nie pozostała aktywna.

## Dane poza Gitem

- Bucket: [itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs).
- URI archiwum: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-2026-10-14/c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536/measurement-data.tar.gz`.
- SHA-256 archiwum: `c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536`.
- Dane tego raportu: `reports/upstream-2.0-h100-2026-10-14/` wewnątrz archiwum; zachowano też oryginalny README sprzed migracji.
- Pobranie i odtworzenie: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Obie weryfikacje pobrania, w tym anonimowa, potwierdziły wszystkie 34 sumy plików.
