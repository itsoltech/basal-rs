# Końcowa weryfikacja upstream 2.0 — 2026-10-14

Pięć nowych checkpointów BF16 działa na CUDA. W sprawdzonym korpusie wybory odpowiadają upstream FP32. Metal potwierdza trzy modele tekstowe, Intel Arc mini i 4.5B. Próby długiego kontekstu i pojemności mają opisane poniżej niepowodzenia; nie oznaczamy ich jako PASS.

## Źródła i metoda

Kod: `10324c59717b6c5ce310043d1ca6a4499ba01ed3`; upstream v2.0.0: `af1ce43a1770dd4fdfc1611e3fe720b93d42fc73`. `metadata/fingerprint.json` w archiwum zawiera SHA-256 103 plików źródłowych i pięciu binarek. Hash zestawu źródeł: `bc8c874bc73430246b485dd500550bb91ef9c6480e98b27b08c14383f3a5d889`. Te same źródła potwierdzono na Macu. Podstawowy korpus Metal/Intel wykonano przed poprawką klienta, na części numerycznej z `05d3be5`; późniejsze zmiany dotyczyły workflow oraz obsługi `dtype=auto` w CLI. Dodatkowe act, kontekst i klient korzystały z końcowych buildów. CUDA dodatkowo potwierdza bitową równość całego korpusu przed i po poprawce klienta. Część `runtime_version` zachowała bazowy hash z cache kompilacji; identyfikację buildu opieramy na fingerprintach i SHA binarek.

| Checkpoint | Przypięta rewizja | Wykonanie auto |
|---|---|---|
| basal-2.0-mini | `cb81b32ce89b0b87e62fcdad28f0435e25c953f3` | FP16 |
| basal-2.0-4.5B | `e5ba9149760d99b73b1963d8e0ccf948b58afacb` | FP16 |
| basal-2.0-max | `2fffc5cc41e5cf7a2f752f38f17e7b4d9a4a8252` | FP16 |
| basal-omni-1.0-mini | `73c993dc3d9ab3cb8887ac0b1ee1cf1c92783240` | BF16 |
| basal-omni-1.0-max | `e538228c4f3ffd96dde298a073ed10247da0d637` | BF16 |

Sprzęt: H100 PCIe 80 GB, sterownik 580.126.09, cuBLASLt 12.9.1; Apple M2 Max, 32 GiB pamięci współdzielonej, macOS 27.0.1; Intel Arc Meteor Lake, Core Ultra 9 185H, 32 GB RAM, Linux/i915/Mesa 26.2.2-arch1.1, wgpu 30.0.1. Rust 1.95.0. Referencja: PyTorch 2.11.0+cu128, Transformers 5.17.0, upstream eager. Modele uruchamiano pojedynczo, orders=1.

Porównujemy faktyczne odpowiedzi: choice, selected multi, tool/tools, call, action, answer oraz odroczenie/refused. Noul obejmuje pomocniczo stronę progu 0.5; score najbardziej prawdopodobny poziom, nie wartość oczekiwaną. Ranking tools porównuje kolejność kluczy. Równość confidence, probability i score nie jest kryterium akceptacji. Zgodność jest względem FP32 upstream, nie gwarantuje poprawnej odpowiedzi ani zachowania na innych wejściach.

## Wyniki wyborów

| Backend / modele | Benchmark argmax | Odpowiedzi żądań i klasyfikacje pomocnicze | Ranking tools | Rzeczywiste act |
|---|---|---|---|---|
| CUDA: trzy basal-2.0 | po 49/49 | po 41/41 | po 2/2 | po 20/20 |
| CUDA: dwa omni | po 49/49 | po 51/51 | po 2/2 | po 20/20 |
| Metal: trzy basal-2.0 | po 49/49 | po 41/41 | po 2/2 | po 20/20 |
| Intel: basal-2.0 mini / 4.5B | po 49/49 | po 41/41 | po 2/2 | po 20/20 |

Benchmark i odpowiedzi żądań mają osobne mianowniki. 41 pytań/model obejmuje 17 choice, 12 noul, 5 score, 5 multi i 2 tool; omni dodaje 10 pytań z mediami. Dwa odrzucone rekordy/model są poza tym mianownikiem. Nowe eksporty Metal/Intel nie mają różnic tokenów względem referencji. Dane: `metadata/final-decision-audit.json`, porównania i eksporty w katalogach backendów.

Act: cztery stany PL/EN, pięć polityk/stanu: koszt błędu/odroczenia, brak certyfikatu, tani defer oraz pełna macierz kosztów. 100/100 pytań CUDA, 60/60 Metal i 40/40 Intel zachowało pola decyzji. `max_error=0.1` nie ma certyfikatu w tych publikacjach. Osobny forward CUDA z istniejącym certyfikatem `0.05`, dwie polityki/stanu, dał dalsze 40/40 zgodnych pytań, po 8/model. Dane: `cuda/final-cert/`, `metadata/cert-coverage.json`. Nie trenowano kalibracji ani nie mierzono jej populacyjnego błędu.

## Rzeczywiste media

Oba omni: po 38/38 wyborów zgodnych z FP32 i z etykietą ustaloną przed forwardem: 16 obrazowych, 14 mowy i 8 mieszanych. To 34 żądania/model. Cztery obrazy COCO val2017, PL/EN, detail default/high; osiem unikalnych wypowiedzi FLEURS dev PL/EN, 4.44–12.84 s. WAV oraz konwersje pierwszej wypowiedzi każdego języka do FLAC, OGG/Vorbis i MP3. Cztery mieszane żądania obejmują oba porządki mediów i dwa pytania/żądanie.

FLEURS: `google/fleurs@70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC-BY-4.0; pierwsze cztery unikalne ID/language z dev, bez selekcji na podstawie wyniku. COCO ID: 39769, 139, 285, 632. Źródła, hashe i etykiety: `inputs/media-source-manifest.json`. Wybór jednej z czterech transkrypcji nie jest pomiarem ASR/WER. Cztery obrazy nie dowodzą ogólnej jakości rozumienia obrazu.

Dodatkowe dwa wejścia/model mają dokładnie 30 s: powtarzana i przycięta rzeczywista wypowiedź PL/EN, nie niezależne nagranie 30 s. Oba omni: po 2/2 zgodnych wyborów względem FP32 i etykiety. Dane: `cuda/final-post/*-window-*`, `metadata/media-gold-audit.json`.

## Kontekst i pamięć

Jeden syntetyczny wybór BLUE/RED, stan z powtarzanym słowem wypełniającym. `prepare` zaakceptował dokładny limit pozycji każdego modelu i odrzucił limit+1. To walidacja granicy, nie dowód wykonalności forwardu. `export` wykonuje trzy ścieżki odczytu na rekord oraz ładowanie modelu; podane limity czasu dotyczą całego procesu, nie pojedynczego forwardu.

| Model / backend | Zakończone wykonanie i zgodny wybór z FP32 | Niepowodzenie / niepotwierdzona zgodność |
|---|---|---|
| 2.0 mini, CUDA / Metal | 1024, 8192 tokeny | — |
| 2.0 4.5B, CUDA | 1024, 8192, 32768 | — |
| 2.0 max, CUDA | 1024, 8192 | Native 32768 zakończone; referencja FP32 przekroczyła 60 s |
| omni mini, CUDA | 1024, 8192 | Native 131072 przekroczyło 150 s |
| omni max, CUDA | 1024 | Native 8192 przekroczyło 90 s, 262144 przekroczyło 150 s; FP32 8192 zakończone |
| 2.0 4.5B, Metal | 1024, 8192 | Dodatkowy [benchmark 32768](../upstream-2.0-metal-context-2026-10-14/README.md) przerwany przez kontrolę pamięci, bez odebranej odpowiedzi |
| 2.0 max, Metal | 1024 | Próba 8192 przerwana bez wyniku/kodu zakończenia; przyczyna nieustalona |
| 2.0 mini / 4.5B, Intel | po 1024 | Po 8192 przekroczyło 240 s |

Na 8192 mini wybrało RED także w FP32, mimo BLUE w treści; ten przypadek potwierdza zgodność wyboru, nie skuteczność przy długim kontekście. Dane: `cuda/final-context/`, `cuda/final-post/`, `metadata/context-decision-audit.json` i dodatkowe eksporty Metal/Intel.

Benchmark max z domyślnymi rezerwami nie rozpoczął forwardu: M2 Max 20.27 GiB dostępnej RAM przy wymaganych 22.80 GiB; Intel 19.53 GiB przy wymaganych 22.31 GiB. Po przerwaniu długiej próby na Macu powtórzenie przez benchmark, z większą rezerwą RAM 4 GiB, odmówiło ładowania: dostępne 15.17 GiB, wymagane 24.80 GiB. Nie zmniejszano rezerw. Udane krótkie eksporty Metal max nie potwierdzają powtarzalnego startu serwera ani wydajności. Jakość Intel max pozostaje niepotwierdzona na tym sprzęcie.

## Bitowa równość i klient

Końcowy build względem `84d0c84`: wszystkie pięć modeli, po 49 rekordów benchmarku oraz 26 rekordów żądań/Llama lub 31/omni, bitowo równe. Ten wcześniejszy build porównano z binarką pierwszej kampanii: max i oba omni bez różnic; mini/4.5B bez ustalonej tabeli GEMM miały różne liczby. Kontrola z jedną tabelą H100 dla tych samych kształtów projekcji dała bitową równość po 49+26 rekordów. Porównanie liczb zachowuje też znak zera; pomija niedeterministyczne metadane manifestu.

Basal-1.0-4.5B@`b95288041cb1975bb930c6f0410819273fd2f54f` względem bazowego `4f68eb2`: single/budget/tree z ustaloną tabelą GEMM, po 44+18 rekordów, bitowo równe. Bez tabeli single/budget różniły się na jednym rekordzie żądania, tree był równy. Kontrola wskazuje wpływ wyboru algorytmu cuBLASLt; nie jest to dowód bitowej powtarzalności automatycznego strojenia. Dane: `cuda/final-fixed-ab/`, `cuda/final-build/`; wcześniejsze [A/B modeli 1.5](../upstream-2.0-h100-2026-10-14/README.md).

Wykryto i poprawiono odrzucanie `dtype=auto` przez lokalnego klienta przed odczytem metadanych. Po poprawce: pięć modeli CUDA przez klienta lokalnego, oba omni z obrazem; trzy Llamy Metal; mini i domyślny 4.5B Intel. Nieznana jawna precyzja nadal jest odrzucana przed I/O. Metal/Intel odrzucają omni z komunikatem wymagania CUDA. HTTP bez modelu i konfiguracji zwróciło `basal-2.0-4.5B` przez `/v1/basal`.

## Wydajność

H100, ten sam klient HTTP i dwa krótkie żądania PL/EN, po jednym pytaniu noul. Każdy silnik: 2 warmup, 8 żądań sekwencyjnych, po 16 closed-loop przy concurrency 1 i 4; jeden przebieg, bez przedziałów ufności. Native: cache stanu wyłączony, GEMM table none, long-tokens=0. Upstream eager; obie strony FP16/Llama lub BF16/omni. To osobna referencja wydajnościowa, nie FP32 używane do akceptacji wyborów. Wszystkie fazy miały zero błędów. Native zachowało dokładnie obiekty answers względem pierwszego odczytu. Upstream przy concurrency 4 miało odpowiednio 15/16, 15/16, 14/16, 15/16 i 15/16 różniących się obiektów answers (liczby również są porównywane). Runner zachował pierwsze odpowiedzi i maksymalną zmianę probability, nie wszystkie obiekty. Max |ΔP| ≤ 0.000127, mniejszy od odległości każdej pierwszej odpowiedzi od progu 0.5; ta granica wyklucza zmianę strony progu noul w zmierzonych powtórzeniach. Liczniki pełnych obiektów nie oznaczają liczników innych wyborów. Pochodna analiza progu jest w [archiwum dodatkowej kontroli](../upstream-2.0-metal-context-2026-10-14/README.md).

| Model | Mediana sekwencyjna native / upstream, ms | Concurrency 4, native / upstream, żądania/s | P95 concurrency 4 native / upstream, ms |
|---|---|---|---|
| 2.0 mini | 5.26 / 27.27 | 288.9 / 7.4 | 17.04 / 1983.21 |
| 2.0 4.5B | 11.41 / 55.25 | 128.9 / 10.2 | 33.71 / 1326.43 |
| 2.0 max | 18.04 / 41.51 | 77.5 / 7.0 | 53.66 / 1081.42 |
| omni mini | 89.14 / 91.14 | 11.8 / 24.9 | 340.70 / 264.54 |
| omni max | 210.10 / 82.53 | 6.2 / 30.2 | 653.74 / 187.33 |

Dane: `cuda/final-extra/`, `metadata/performance-summary.json`. Omni nie wykazało przewagi wydajnościowej; omni max było wolniejsze także sekwencyjnie. Małe N, krótki tekst, brak mediów w obciążeniu i upstream eager wykluczają tezę o ogólnym przyspieszeniu lub porównaniu z upstream compile/CUDA graphs. Telemetria jest próbkowana, nie stanowi pomiaru energii ani maksymalnej pojemności.

Metal/Intel: własny `basal benchmark`, dwa te same payloady, warmup 1/klasę, po 2 żądania/fazę, sequential i mixed concurrency 1/4. Mediana sequential mini/4.5B: M2 Max 39.00/106.25 ms; Intel 653.43/1985.89 ms. Zero błędów w zakończonych fazach. N=2/fazę nie opisuje stabilnych percentyli ani wydajności maksymalnej; nie uruchamiano porównawczej referencji upstream na tych backendach. Dwie próbki nie obciążają stale czterech jednoczesnych klientów. Max ma wynik `not_run`, nie wynik wydajności.

## Kontrole i dane

[GitHub CI dla kodu 10324c5](https://github.com/itsoltech/basal-rs/actions/runs/38583989461): wszystkie joby PASS, w tym kompilacja/lint Linux, Intel, CUDA i Metal. Lokalne fmt, Clippy i Rustdoc z ostrzeżeniami jako błędami: Linux, Intel, CUDA i Metal; kontrole zależności PASS. CI nie uruchamia modeli na GPU. Nie dodano frameworków testowych ani bramek testowych. H100 usunięta; brak maszyny potwierdzony przez API po pobraniu wyników.

Dane poza Gitem: [HF bucket itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs), archiwum `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-final-2026-10-14/3bf4dac1aab9a203d731cd67feae1372ea1bf7eaea46f9baf5b2bfc64c474f61/measurement-data.tar.gz`. SHA-256: `3bf4dac1aab9a203d731cd67feae1372ea1bf7eaea46f9baf5b2bfc64c474f61`, gzip 3 148 341 B, 639 plików danych i `SHA256SUMS`. Ścieżki opisane wyżej są względne do `reports/upstream-2.0-final-2026-10-14/` w archiwum. Ponowne pobranie z autoryzacją i anonimowo: SHA archiwum oraz 639/639 sum plików zgodne.

Publikacja zachowuje logity, tokeny, prawdopodobieństwa, odpowiedzi i niepowodzenia. Base64 mediów zastąpiono hashami i liczbą bajtów; usunięto identyfikatory/ścieżki maszyn. Prywatne pełne dane pozostają lokalnie. Przed ponownym forwardem należy odtworzyć media z manifestu; archiwum nie rozprowadza zdjęć ani nagrań. Procedura pobrania i kontroli sum: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Historycznych raportów i zakończonych prefiksów bucketu nie zmieniano.
