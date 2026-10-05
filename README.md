# basal-rs

Runtime w Rust dla modeli decyzyjnych [basal](https://github.com/rkinas/basal)
(Remek Kinas). Serwuje API TypeSafe System One oraz endpoint zgodny z
serwerem upstream. Backend CUDA (NVIDIA, sprawdzony na RTX 6000 Ada) i Metal
(Apple Silicon, sprawdzony na M1 Pro).

Runtime liczy te same decyzje co upstream w FP32: ten sam prompt, te same
token IDs, oba porządki opcji, uśrednienie i kalibracja z `CALIBRATION.json`.
Różni się sposobem wykonania: pakowaniem promptów w drzewo wspólnych
prefiksów, własnymi kernelami i harmonogramem serwera.

## Modele

| Model | Status |
|---|---|
| `Remek/basal-1.5-max` (11B), rewizja `be1b5ee7` | główny cel, CUDA |
| `Remek/basal-1.0-4.5B`, rewizja `b9528804` | CUDA i Metal |

Typy pytań: `choice` (2–10 opcji oraz 11–255 strategią grupową), `noul`,
`score`, a z basal-1.5 także `multi`, `act`, `facts: "auto"` i `evidence`.

## Wyniki

basal-1.5-max na RTX 6000 Ada z limitem mocy 250 W; upstream v1.5.0
`basal-serve --mode fast` (BF16, torch.compile, CUDA graphs) na tej samej
karcie. Szczegóły i dane:
[reports/rust-cuda-1.5-max](reports/rust-cuda-1.5-max/README.md).

| Pomiar | Upstream | basal-rs |
|---|---:|---:|
| Pojedyncza decyzja (basal-bench, mediana) | 90,7 ms | 63,4–64,0 ms |
| Jedno pytanie przez HTTP, p50 | 118 ms | 73 ms |
| Jedno pytanie przez HTTP, 32 klientów | 6,8 żądania/s | 21,1–21,2 żądania/s |
| Dokument 16k tokenów, 5 pytań | 49,1 s | 6,4 s |
| Ruch mieszany (stany 0,1–16k tokenów, 1–14 pytań)¹ | 17 żądań/min | 104 żądania/min |

¹ Zmierzone przed zmianami tabeli GEMM i kernela attention z
[gemm-equiv](reports/rust-cuda-1.5-max/gemm-equiv/README.md) i
[attn-kernel](reports/rust-cuda-1.5-max/attn-kernel/README.md).

Zgodność z upstream FP32 na 44 przykładach basal-bench: te same decyzje
44/44; maksymalna różnica logitu 1,3·10⁻⁴ dla ścieżki FP32 i 0,27 dla
domyślnej FP16 (mediana 0,013, maks. różnica prawdopodobieństwa po
kalibracji 0,0008). `multi`, `act`, `facts` i `evidence` dają te same pola
odpowiedzi co upstream; `facts` jest zgodne co do bajtu na 432 stanach.
Wynik pytania nie zależy od tego, z czym trafi do partii (bitowo te same
logity pojedynczo, w partii i pod obciążeniem HTTP).

## Budowanie

Rust stable (sprawdzony na 1.95).

```sh
# Metal (macOS)
cargo build --release

# CUDA (Linux, sm_89; obraz z toolkitem: tools/cuda/Dockerfile)
docker build -t basal-dev:cuda tools/cuda
docker run -d --name basal-dev --gpus all --ipc=host -v "$PWD":/work -w /work basal-dev:cuda sleep infinity
docker exec basal-dev cargo build --release --features basal-cli/cuda
```

Kernele CUDA są kompilowane do PTX przez `nvcc` w `crates/basal-gpu/build.rs`
(`CUDA_COMPUTE_CAP`, domyślnie 89).

## Model

```sh
huggingface-cli download Remek/basal-1.5-max \
  --revision be1b5ee7e7a9755a931262fa7fab4f59be0fd03c --local-dir .models/basal-1.5-max
```

Runtime sprawdza przy ładowaniu `config.json`, `basal.json`, szablon czatu i
tokenizer; niezgodność z obsługiwanym kontraktem promptu kończy się błędem.

## Serwer

```sh
./target/release/basal serve --model .models/basal-1.5-max \
  --gemm-table reports/rust-cuda-1.5-max/gemm-equiv/gemm-algos-f16-invariant-groups.json \
  --addr 0.0.0.0:8000
```

| Endpoint | Opis |
|---|---|
| `POST /v1/systemone` | TypeSafe System One: ścisła walidacja, odpowiedzi i confidence TypeSafe, rozszerzenia basal-1.5 |
| `POST /v1/basal` | konwencje `Server.decide` upstream v1.5.0: walidacja `to_items`, `confidence = max(p)`, błędy jako 422 `{"error"}` |
| `GET /v1/models` | lista modeli TypeSafe z polami upstream `mode` i `early_exit` |
| `GET /health` | gotowość |

Odpowiedzi mają nagłówki `x-basal-queue-ms`, `x-basal-compute-ms` i
`x-basal-batch-requests`.

```sh
curl --fail-with-body localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "basal-1.5-max",
  "state": "Zgłoszenie: klient prosi o zwrot pieniędzy za uszkodzony telefon.",
  "questions": {
    "dept": {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
             "criteria": {"returns": "Zwroty", "tech": "Wsparcie techniczne", "sales": "Sprzedaż"}},
    "urgent": {"type": "noul", "instructions": "Czy sprawa jest pilna?"}
  }
}'
```

Najważniejsze opcje `basal serve`:

| Opcja | Domyślnie | Znaczenie |
|---|---|---|
| `--dtype` | `f16` | `f16`, `bf16` albo `f32` (referencja numeryczna, wolna) |
| `--gemm-table` | brak | tabela algorytmów cuBLASLt z `basal gemm-search`; wariant `--invariant` daje wyniki niezależne od partii |
| `--max-batch-tokens` | 8192 | tokeny jednej partii, liczone po współdzieleniu prefiksów |
| `--schedule` | `hrrn` | kolejność przyjmowania żądań: `hrrn` (krótkie przed długimi, bez zagłodzenia) albo `fifo` |
| `--long-tokens` | 4096 | żądania powyżej tej liczby tokenów idą do drugiego toru, który oddaje GPU krótkim partiom między warstwami; `0` wyłącza |
| `--long-slice-ms` | 100 | minimalny czas pracy toru długich żądań między oddaniami GPU |
| `--state-cache-mb` | 0 | cache K/V stanu między żądaniami |
| `--max-inflight` | 1024 | limit żądań w kolejce i w trakcie; nadmiar dostaje 503 |

Tabela GEMM w repozytorium (`gemm-equiv/gemm-algos-f16-invariant-groups.json`)
jest dla RTX 6000 Ada i cuBLASLt 12.9.1: dla każdej klasy M najszybszy
algorytm z grupy algorytmów dających bitowo te same wyniki
([pomiar](reports/rust-cuda-1.5-max/gemm-equiv/README.md)). Dla innej karty
lub wersji trzeba ją wygenerować (`basal gemm-search --model ... --invariant
--m-classes 16,32,48,64,96,128,160,192,224,256,320,384,448,512,640,768,1024,1536,2048,3072,4096,6144,8192,12288,16384
--out gemm.json`). Bez tabeli algorytmy są dobierane przy
pierwszym użyciu.

## Polecenia

`basal --help` opisuje wszystkie polecenia. Poza `serve` najczęściej:
`decide` (jedno żądanie z pliku), `export` i `compare` (porównanie z
referencją upstream), `bench` i `bench-requests` (pomiary), `gemm-search`.

## Struktura

| Katalog | Zawartość |
|---|---|
| `crates/basal-core` | kontrakt System One, prompt, tokenizer, pakowanie, decyzje, `facts`, `evidence`, silnik i trait `Backend` |
| `crates/basal-gpu` | forward Llama na candle z własnymi kernelami CUDA i Metal, cuBLASLt, attention po węzłach drzewa |
| `crates/basal-cli` | polecenie `basal` i serwer HTTP |
| `tools/reference` | eksport referencji z upstream, zestawy żądań |
| `tools/bench` | pomiary A/B i obciążeniowe, generatory ruchu |
| `contracts/typesafe` | migawka OpenAPI TypeSafe |
| `reports` | pomiary zgodności i wydajności z danymi |

## Dokumentacja

- [Architektura](docs/ARCHITECTURE.md): potok żądania, pakowanie, kernele, harmonogram serwera.
- [Zgodność API](docs/SYSTEM_ONE.md): TypeSafe System One, endpoint upstream, rozszerzenia basal-1.5.
- [Pomiary](docs/BENCHMARKS.md): metodyka, narzędzia, odtwarzanie wyników.
- [Raporty](reports/README.md).

## Licencja

Apache-2.0. Kod portuje zachowanie [rkinas/basal](https://github.com/rkinas/basal)
(Apache-2.0); atrybucja w [NOTICE](NOTICE). Wagi modeli mają własne licencje
na Hugging Face.
