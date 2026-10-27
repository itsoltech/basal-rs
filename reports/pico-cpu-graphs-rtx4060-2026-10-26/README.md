# Pico: CPU, CUDA Graphs i rezydentne wagi FP32

Limity 1/2/4/8 wątków CPU nie dały wyraźnego zysku C1 dla 5080 tokenów.
Przyspieszenie FP32 dało przechowywanie projekcji FP32 na GPU zamiast
ich rozszerzania z BF16 przed każdym iloczynem. Wariant jest jawny,
domyślnie wyłączony: `BASAL_QWEN_F32_RESIDENT=1` z `--dtype f32`.
Profil v4 i `dtype: auto` pozostają bez zmian.

**Cel łączny pozostaje nieosiągnięty:** zgodne decyzje i różnica
skalibrowanego prawdopodobieństwa ≤0,0001, czas niższy od upstream fast
oraz brak regresji krótkich dokumentów względem upstream. Nowy wariant
spełnia próg jakości na badanym zestawie, ale jest znacznie wolniejszy od
historycznego fast BF16 około 269–270 ms. To nie jest nowa para ABBA z fast.

## Warunki i metoda

- Baza `db55af9d7cab5f12a18347b0bb27eb4dab4b1531` z niecommitowanym Pico v4;
  baseline SHA-256 `2d2bb90841958d4ad84f67581e922260e854dccbd8274cc7145769e2a2caba4b`.
  Końcowa binarka: `fb1799eb15f6a66737f6e3ff48924bb8c6fa08a68051245ebf2a96cd6496c0d2`.
- Model `Remek/basal-omni-pico`, rewizja
  `a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`, SHA-256 wag
  `ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9`.
- Upstream 2.4.0, commit `615fe321a818fdf5d2897f758b8d823f919d7f94`, bez zmian
  checkoutu. Referencje FP32 pochodzą z poprzednich kampanii i są w archiwum.
- RTX 4060, SM 8.9, 8188 MiB, sterownik 615.78.08, CUDA/cuBLASLt 12.9.1.
  Limit mocy 115 W, bez zmiany ustawień GPU i bez wynajmowania maszyny.
- HTTP: jedna kolejność, FIFO, budżet 8192 tokenów, bez cache stanu i prefiksów.
  Dwa punkty 255/5080 tokenów i cztery krótkie dokumenty. Pomiar obejmuje
  serializację żądania i odczyt całego HTTP body, bez parsowania odpowiedzi.

Budżety CPU ustawiają OMP/MKL/Rayon. Nie zmieniają liczby wątków Tokio
ani nie tworzą równoległej puli planowania. Współbieżne żądania nadal
dzielą GPU zgodnie z istniejącym schedulerem. Nie rezerwowano rdzeni CPU.

## Limity CPU i rozkład kosztów

Każdy screen CPU: 12 próbek na punkt długości po dwóch rozgrzewkach;
po 32 krótkie żądania przy C1 i C8. Audyty: zero błędów i zmian odpowiedzi
wewnątrz wariantu. Po usunięciu czasu usage odpowiedzi pięciu dokumentów
są jednakowe również między wszystkimi czterema budżetami CPU.

| Budżet CPU | 255 tokenów p50 [ms] | 5080 tokenów p50 [ms] | Krótkie C8 p50 [ms] |
|---|---:|---:|---:|
| 1 | 24,30 | 532,43 | 327,74 |
| 2 | 24,40 | 533,11 | 328,18 |
| 4 | 24,28 | 531,94 | 350,59 |
| 8 | 24,21 | 531,53 | 327,42 |

To **SIGNAL**, bez ABBA między budżetami. Rozrzut C8 nie uzasadnia wyboru
zwycięzcy. C1/5080 różni się o mniej niż 0,3%; nie potwierdza przyspieszenia
przez zwiększenie limitu wątków.

Osobny pomiar diagnostyczny: planowanie CPU, 68 żądań, mediana 0,964 ms,
zakres 0,342–11,689 ms. Nie ma identyfikatorów łączących każdy zapis czasu
z dokumentem, więc nie przypisujemy tej mediany wyłącznie do 5080 tokenów.
Prefill/enqueue dla 5080: 14 zapisów, mediana 407,96 ms,
zakres 405,94–411,59 ms. Odczyt z oczekiwaniem zajmuje około 119 ms dla
długiego forwardu. Enqueue zawiera blokowanie przez GPU, alokator i kolejkę;
**nie jest to 408 ms czystej pracy CPU**.

Nsight Systems 2026.3.2 objął trzy osobne forwardy eksportu 5080 tokenów,
bez ładowania i planowania. W trzecim, rozgrzanym zakresie: 2325 kerneli
na jednym strumieniu, 529,96 ms sumy czasu kerneli w przedziale 534,36 ms
(99,18%). Dwa wcześniejsze zakresy: 554,41/615,96 ms oraz 546,24/550,31 ms.
To pokrycie czasu strumienia, nie procent zajętości jednostek GPU.
Praca GPU dominuje nad lukami między kernelami; sam nowy worker CPU
nie usunie tego kosztu. Czasy API nakładają się z GPU i obejmują oczekiwanie.

Największe pozycje trzeciego profilu: rekurencja DeltaNet 135,60 ms,
`ucopy_bf16` 55,06 ms, maskowanie/softmax 21,78 ms, normalizacja 20,08 ms,
`copy2d_f32` 20,02 ms. Pozostałe znaczące pozycje to różne kernele GEMM.
Pełne agregaty zachowano; nie sumujemy czasu API i GPU.

## Sprawdzone sposoby wykonania

Poniższe wstępne serie są **SIGNAL**, nie potwierdzającym A/B.
FP32: cztery próbki na długość, osiem krótkich żądań; BF16 z grafami:
12 próbek i 32 krótkie żądania. Część wstępnych prób FP32 nakładała się
z kompilacją CPU; końcowe ABBA jest osobne, bez kompilacji w tle.

| Wariant | 255 tokenów p50 [ms] | 5080 tokenów p50 [ms] | Krótkie C1 p50 [ms] |
|---|---:|---:|---:|
| FP32 dotychczasowe | 108,75 | 2364,65 | 160,09 |
| FP32 rezydentne projekcje | 68,96 | 1559,83 | 110,52 |
| FP32 rezydentne + grafy MLP | 69,13 | 1566,76 | 101,66 |
| FP32 rezydentne + globalny fragment 512 | 50,30 | 1123,81 | 107,05 |
| BF16 v4 + grafy MLP | 24,40 | 533,24 | 35,04 |

**Grafy MLP odrzucono.** Prototyp naprawdę przechwytywał i odtwarzał CUDA
Graphs, z prywatnymi wejściami, kopiowaniem wyjść i cache na wątku workera.
Obejmował osobne MLP o 1–512 wierszach; nie obejmował całego forwardu,
rekurencji, attention, połączonych MLP wielu promptów ani bloków BF16 >512.
Dlatego wynik nie rozstrzyga potencjału grafu całego modelu.
BF16 dał zerowe różnice dla 44 pytań i 12 gałęzi API względem v4.
Wstępne odpowiedzi FP32 pięciu dokumentów były takie same jak bez grafów.
Pełny eksport FP32 zakończył się jednak **OOM**, zanim zapisał bench.
Limit 512 MiB dotyczył szacowanych zachowanych aktywacji; nie ograniczył
całej pamięci CUDA Graphs i ich alokacji wewnętrznych. Kod prototypu
pozostaje tylko w archiwum, wraz z błędem i diagnostyką capture.

**Globalny fragment FP32 512 odrzucono.** Zachował 44/44 decyzje benchmarku,
12/12 gałęzi API i 2/2 na granicach; maksymalna różnica skalibrowanych
prawdopodobieństw względem FP32 wyniosła 0,000004590. Zmienia kształty GEMM,
więc nie jest bitowo zgodny z poprzednim FP32. Przyspieszył długi dokument,
ale pogorszył krótki `pico-bank-six-types`: 168,33 → 206,57 ms,
oraz `pico-long-prefix`: 136,62 → 150,12 ms względem samych rezydentnych wag.
Agregat krótkich dokumentów ukrywał te regresje. Prototyp jest w archiwum.

## Końcowa zmiana i kwalifikacja

W kodzie pozostają wyłącznie rezydentne projekcje FP32 jako opt-in oraz
diagnostyka faz i zakresów profilera. Rozszerzenie BF16→FP32 jest dokładne,
wykonywane przy ładowaniu. Nie zmienia kształtów GEMM ani kolejności operacji.
Embeddings pozostają kompaktowe; rozszerzane są tylko wybrane wiersze.
Kosztem jest większe zużycie VRAM; nie zmierzono jego szczytu ani pojemności
serwera. Nie zmienia się domyślny profil, Metal, BF16 ani inne architektury.

Wstępna kwalifikacja rezydentnych wag: orders=1 oraz orders=2 po 44/44
decyzje benchmarku i 12/12 gałęzi numerycznych API; osobno 255 opcji,
odmowa 256 opcji i 5080 tokenów. Tokeny i pola decyzji endpointów są zgodne.
Maksymalna różnica skalibrowanego prawdopodobieństwa względem FP32:
0,000002041 (bench orders=1), 0,0000050515 (bench orders=2),
0,000000656 (gałęzie numeryczne API), 0,000001014 (odpowiedzi endpointu),
0,000004530 (granice). Wszystkie są poniżej 0,0001.
Bench orders=2 jest bajtowo równy wcześniejszemu v4 FP32; porównania
liczbowe i API z wcześniejszym FP32 dały zero różnic, również na granicach.
To dowód wyłącznie dla zbadanych wejść i tej karty.

Zachowane błędy procedury: pierwsza próba czasów faz nie przekazała flagi
przez sanitizer środowiska runnera. Dodano jawne, rejestrowane
`--candidate-diagnostic` i powtórzono diagnostykę. Pierwsze porównanie
orders=2 miało faktycznie eksport orders=1: tokens mismatch i null wyników.
Poprawiono polecenie przez `--orders 2`; nie uznano null za zgodność.

## Potwierdzające ABBA FP32

Kolejność: baseline, resident, resident, baseline, baseline, resident,
resident, baseline. Każdy start miał 10 s rozgrzewki i po dwie dodatkowe
próbki rozgrzewki na punkt długości. Pierwszy start każdego silnika wykluczono.
Zostają trzy serie po 12 próbek na punkt długości i 32 krótkie żądania:
36 próbek na punkt oraz 96 krótkich żądań na wariant. Tabela pokazuje
medianę metryk serii. Nie wyznaczamy przedziału ufności z trzech powtórzeń.
Nie było równoległej kompilacji ani innych forwardów na GPU.

| C1 | Baseline FP32 p50/p95 [ms] | Rezydentne FP32 p50/p95 [ms] |
|---|---:|---:|
| 5080 tokenów | 2374,18 / 2374,95 | **1571,23 / 1573,11** |
| 255 tokenów | 109,33 / 109,46 | **69,64 / 70,04** |
| Cztery krótkie dokumenty | 160,76 / 264,65 | **101,65 / 165,74** |

C1/5080: **33,82% mniej czasu, przyspieszenie 1,511×**. Mediany serii:
2373,98–2374,71 ms baseline i 1567,61–1571,65 ms resident.
Audyt: **AUDITED**, zero problemów, zero błędnych odpowiedzi i zero zmian
bitowych odpowiedzi wewnątrz każdego wariantu. Po wyłączeniu latency usage
odpowiedzi pięciu dokumentów są identyczne również między wariantami.

Każdy krótki dokument poprawił p50 i p95. Poniżej mediany p50 serii,
po 24 zachowane próbki na dokument i wariant:

| Dokument | Baseline FP32 [ms] | Rezydentne FP32 [ms] |
|---|---:|---:|
| `pico-27-options` | 109,37 | 69,36 |
| `pico-bank-six-types` | 264,49 | 165,17 |
| `pico-long-prefix` | 212,07 | 133,85 |
| `pico-unicode` | 102,10 | 63,68 |

To brak regresji względem **naszego poprzedniego FP32** w zbadanych czterech
przypadkach. Nie jest to brak regresji względem upstream fast: historyczna
seria fast miała 14,15 ms dla 255 tokenów i 21,66 ms dla mieszanki krótkich
dokumentów. Cel przewagi nad fast i bliskości FP32 pozostaje nieosiągnięty.

Końcowa binarka ponownie przeszła orders=2 oraz granice w FP32 resident:
44/44, 12/12 i 2/2 decyzje, maksymalna różnica prawdopodobieństw 0,0000050515
względem upstream. Bench jest bajtowo równy wcześniejszemu FP32.
Całe systemone JSONL po uporządkowaniu kluczy i usunięciu latency usage
jest równy wcześniejszemu FP32 orders=2. Końcowy domyślny BF16 orders=1
także zachował bajtowo bench i znormalizowane API v4, 44/44 i 12/12.

## Weryfikacja kodu i ograniczenia

Zastosowano Rust implementation, Rust review i self-review: ownership wag,
niemutowalność stanów, tożsamość planu, blokowanie schedulera i sprzątanie
zakresów profilera. Oryginalny błąd inferencji zachowuje pierwszeństwo,
gdy zatrzymanie profilera też się nie powiedzie. Odrzucony cache grafów
nie pozostaje w kodzie wykonawczym. Nie dodano testów ani frameworków.

Build release CUDA, formatowanie, Clippy Linux i CUDA/all-features,
Rustdoc prywatnych elementów z `-D warnings` w obu konfiguracjach,
kompilacja Pythona i kontrola wyjątku `paste` przeszły. Graf zależności
pozostaje niezmieniony. `cargo deny` ma **FAIL** dla istniejących
`zstd 0.13.3` i `zstd-safe 7.3.0`: RUSTSEC-2026-0391 i
RUSTSEC-2026-0395–0406. Nie wyciszano advisory ani nie rozszerzano
optymalizacji o aktualizację zależności.

Nie zmierzono Metal, innych GPU, energii, szczytowej pamięci ani pojemności
serwera. Screen C8 nie jest kwalifikacją wielu workerów planowania.
Kwalifikacja jakości ma zakres badanego korpusu, nie wszystkich dokumentów.
Pełny CUDA Graph pozostaje niezbadany; prototyp obejmował wyłącznie MLP.

## Artefakty

Paczka zawiera wejścia i referencje FP32, odpowiedzi wszystkich screenów
i ABBA, audyty, końcowe eksporty/porównania, trzy agregaty profilu,
czasy faz, błąd OOM, odrzucone źródła oraz źródła końcowego wariantu.
`REPRODUCE.md` opisuje odbudowę, nowe pomiary i audyt bez GPU.
Binarki, wagi, surowe profile Nsight/SQLite oraz prywatne logi pozostają lokalnie.

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/pico-cpu-graphs-rtx4060-2026-10-26/9ee7a73bde6ec87f6666848d10d0fc967545e27b861008206f3f4c07171a1332/measurement-data.tar.gz`.

SHA-256: `9ee7a73bde6ec87f6666848d10d0fc967545e27b861008206f3f4c07171a1332`,
rozmiar 801 723 B, 169 plików w `SHA256SUMS`. Ponowne pobranie z logowaniem
i anonimowo potwierdziło SHA-256 archiwum oraz **169/169 sum plików w obu
kopiach**. Lokalna kopia archiwum pozostaje zachowana. Dane są pod
`reports/pico-cpu-graphs-rtx4060-2026-10-26/`, a odtworzenie opisuje
`REPRODUCE.md` w archiwum.

```bash
hf buckets cp 'hf://buckets/itsoltech/basal-rs/measurements/pico-cpu-graphs-rtx4060-2026-10-26/9ee7a73bde6ec87f6666848d10d0fc967545e27b861008206f3f4c07171a1332/measurement-data.tar.gz' measurement-data.tar.gz
printf '%s  measurement-data.tar.gz\n' '9ee7a73bde6ec87f6666848d10d0fc967545e27b861008206f3f4c07171a1332' | sha256sum --check -
tar -xzf measurement-data.tar.gz
sha256sum --check SHA256SUMS
```

Audyt danych bez GPU: `tools/perf/summarize-omni.py
--measurement-artifacts <rozpakowane-dane>/confirm-f32-resident
--inputs <rozpakowane-dane>/inputs --out <nowy-plik.json>`.
`--measurement-artifacts` jest opcją runnera, nie CLI basal.
