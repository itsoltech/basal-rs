# basal-1.0-4.5B na RTX 6000 Ada: pierwsza wersja CUDA

Data: 2026-10-04. NVIDIA RTX 6000 Ada Generation 48 GB, sterownik 580.159.03
(CUDA 13.0), Linux. Budowanie i uruchamianie w kontenerze z
`tools/cuda/Dockerfile` (CUDA 12.9.1 devel, Rust 1.95.0, cuBLASLt 12.9.1).
Upstream `3fa2eeab` w tym samym kontenerze: torch 2.13.0+cu130, transformers
5.17.0. Model basal-1.0-4.5B w rewizji `b9528804…`, SHA-256 safetensors
`3cfe3672…`.

Warunki: karta pracuje pod obciążeniem na limicie mocy 300 W (telemetria:
średnio ~275 W, zegar SM ~1665 MHz zamiast 2775 MHz, maks. 67 °C), więc
kontrolny GEMM f16 480×2048×11008 daje 170 TFLOPS na zimnej karcie i 115–150
TFLOPS po serii. Telemetria co 2 s: `../baseline-cuda/telemetry/`. Upstream
`fast-nocompile` nie został zmierzony (przebieg przerwał reset serwera).

## Zgodność z FP32 upstream

Referencja: `tools/reference/export_reference.py --mode eager --dtype float32`
na CUDA ([../reference-cuda-fp32](../reference-cuda-fp32/)).

| Wariant względem FP32 upstream | Decyzje 44 | Maks. różnica logitu | Maks. różnica p_avg | p95 p_avg |
|---|---:|---:|---:|---:|
| Rust FP32 | 44/44 | 0,00007 | 0,00001 | 0,000003 |
| Rust FP16 (fused attention, tabela GEMM) | 44/44 | 0,041 | 0,012 | 0,0013 |
| Rust BF16 | 44/44 | 0,351 | 0,066 | 0,016 |
| Upstream BF16, forward bez kompilacji | 44/44 | 0,442 | 0,106 | 0,017 |
| Upstream `fast` (BF16, torch.compile + CUDA graphs) w `basal-bench` | 43/44 | | | |

Pytania System One (29 o identycznych tokenach): Rust FP16 maks. różnica
p_avg 0,0028. Prompty i token IDs zgodne. Rust FP32 zgadza się z PyTorch FP32
do 7·10⁻⁵ logitu.

## Pojedyncza decyzja

Metodyka `basal-bench` (44 przykłady, oba porządki, batch 1, 39 pomiarów po
odrzuceniu 5; przepustowość w grupach po 16).

| Wariant | lat2 mediana | lat2 p95 | lat1 | Decyzje/s | Trafność |
|---|---:|---:|---:|---:|---:|
| Upstream eager-fp32 | 185,7 ms | 224,2 ms | 106,0 ms | 5,2 | 0,795 |
| Upstream `fast` BF16 | 30,5 ms | 40,4 ms | 26,7 ms | 27,6 | 0,773 |
| Rust FP16: cuBLAS, attention z operacji candle | 24,3 ms | 29,5 ms | 20,3 ms | 50,4 | 0,795 |
| + softmax z maską i skalą w jednym kernelu | 23,7 ms | 28,3 ms | 20,2 ms | 50,8 | 0,795 |
| + cuBLASLt, czas 16 algorytmów heurystyki | 22,0 ms | 24,7 ms | 19,2 ms | 51,6 | 0,795 |
| + attention f32 jednym kernelem dla wszystkich segmentów | 21,9 ms | 25,8 ms | 18,6 ms | 51,8 | 0,795 |
| + tabela GEMM z pełnego przeszukania cuBLASLt | 20,8 ms | 26,5 ms | 18,8 ms | 54,0 | 0,795 |

Wersja końcowa względem upstream `fast`: mediana niższa o 32%, p95 o 34%,
lat1 o 30%, przepustowość 1,96×. Pojedyncze przebiegi w jednej sesji.

## Żądania System One

`tools/reference/bench_requests.py --mode fast` (upstream `Server.decide`) i
`basal bench-requests` (wiersze-drzewa), jeden klient, 10 powtórzeń, mediany:

| Żądanie | Pytania | Upstream `fast` | Rust FP16 |
|---|---:|---:|---:|
| cx-01 | 3 | 131,9 ms | 57,0 ms |
| cx-02 | 2 | 97,8 ms | 72,3 ms |
| cx-03 | 3 | 180,0 ms | 85,5 ms |
| cx-04 | 3 | 156,6 ms | 55,5 ms |
| cx-05 | 3 | 133,5 ms | 60,7 ms |
| 14 pytań o stan cx-01 | 14 | 690,5 ms | 414,1 ms |
| jedno pytanie cx-01 | 1 | 41,6 ms | 28,2 ms |

## Profil (nsys, pojedyncza decyzja)

GPU jest zajęte przez 93–95% czasu forwardu, więc narzut hosta (~1100 kerneli
na forward) nie jest głównym kosztem; CUDA graphs dałyby najwyżej kilka
procent. Przed tabelą GEMM: GEMM FP16 ~74% czasu GPU, attention ~13%, reszta
~10%. Pełne przeszukanie konfiguracji cuBLASLt (algorytm, kafelek, etapy,
split-K z redukcją f32, swizzle; `gemm/gemm-algos-f16.json`, 208 klas) daje w
mikrobenchmarku 1,1–1,6× względem pierwszego wyboru heurystyki.

Tabela GEMM jest ważna dla tej karty i cuBLASLt 12.9.1; przy innej wersji
runtime ją odrzuca.
