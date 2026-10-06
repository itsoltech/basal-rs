# Obrazy z GHCR na RTX 6000 Ada

Data: 2026-10-06, RTX 6000 Ada (compute capability 8.9), limit mocy 300 W.
Obrazy `ghcr.io/itsoltech/basal-rs` z commitu `d354311`, zbudowane przez
`.github/workflows/docker.yml` (skróty w `images.txt`). Skrypt: `run.sh`
(uruchamiany na hoście Dockera).

## Start „out of the box” (`oob.txt`, `oob.log`)

Obraz `latest`, pusty wolumen, `serve.yml` z `repo: Remek/basal-1.5-mini`:
pobranie modelu, wygenerowanie tabeli GEMM i gotowość API po 203 s.
`docker stop` kończy proces z kodem 0.

## `latest-sm90` na karcie 8.9 (`sm90-on-ada.log`)

Obraz nie działa (PTX dla 9.0 nie uruchamia się na starszej karcie):
`CUDA_ERROR_INVALID_PTX, "a PTX JIT compilation failed"`. Od commitu
`0154deb` runtime sprawdza to przy starcie i podaje, którego obrazu użyć.

## `latest` (8.9) a `latest-sm80` (8.0) na karcie 8.9

Zgodność (`compare-*.json`): oba obrazy dają te same decyzje co FP32 upstream
(44/44) dla basal-1.5-max i basal-1.5-mini, a ich wyniki są bitowo
identyczne (różnica 0,0).

Pojedyncza decyzja, metodyka basal-bench, ABBA po 2 rundy (`ab-bench-*`):

| Model | `latest`: lat2 mediana | `latest-sm80`: lat2 mediana | Iloraz sm80 / latest (mediana po pytaniach) |
|---|---:|---:|---:|
| basal-1.5-max | 46,1–47,4 ms | 47,1–47,2 ms | 1,007 |
| basal-1.5-mini | 7,70 ms | 7,71 ms | 1,003 |

HTTP, basal-1.5-max, jedno pytanie na żądanie, ABBA (`load-*.json`):

| Obraz | 1 klient: żądania/s, p50 | 32 klientów: żądania/s, p50 |
|---|---:|---:|
| `latest` | 17,6–17,7, 55–56 ms | 26,95–27,21, 1161–1173 ms |
| `latest-sm80` | 17,5–17,6, 56 ms | 27,00–27,01, 1169–1171 ms |

Różnice mieszczą się w rozrzucie przebiegów (do 1%). Na karcie 8.9 obraz
`-sm80` jest więc tak samo szybki jak `latest`; obrazów `-sm80` i `-sm90` na
kartach A100 i H100 nie testowano.
