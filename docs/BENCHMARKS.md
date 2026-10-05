# Pomiary

## Kryteria

Punktem odniesienia jest forward FP32 upstream (autor ewaluował modele w FP32,
`basal.json: eval_dtype`). Zmianę runtime'u przyjmujemy, gdy:

- prompty, token IDs, litery, permutacje i pakowanie są identyczne z upstream;
- decyzje są takie same jak w FP32 upstream dla każdego przypadku zestawu
  odniesienia, a różnice logitów i prawdopodobieństw są zapisane w raporcie
  (zgodny argmax nie dowodzi zgodnej kalibracji);
- dla optymalizacji, które nie zmieniają obliczeń (pakowanie, cache, harmonogram),
  wynik jest bitowo równy poprzedniej wersji.

Szybkość mierzymy decyzjami (pytanie po dwóch porządkach opcji i kalibracji)
oraz żądaniami, z rozkładem opóźnień. Raport zapisuje warunki: GPU, limit
mocy, zegary, wersje bibliotek, rewizje modelu i upstream.

## Referencja upstream

Repozytorium upstream w przypiętej rewizji i jego środowisko Pythona
umieszczamy w `.baseline/` (poza repozytorium). Eksport referencji:

```sh
# basal-1.5-max: FP32 na CPU (wagi FP32 11B nie mieszczą się w 48 GB GPU razem z aktywacjami)
BASAL_UPSTREAM=.baseline/upstream-1.5 .baseline/upstream-1.5/.venv/bin/python \
  tools/reference/export_reference.py --model .models/basal-1.5-max \
  --repo Remek/basal-1.5-max --revision be1b5ee7e7a9755a931262fa7fab4f59be0fd03c \
  --mode eager --device cpu --dtype float32 \
  --cases tools/reference/systemone_cases_1.5.jsonl --out reports/NOWA-REFERENCJA
```

Wynik: `bench.jsonl` (44 przykłady basal-bench: prompty, token IDs, logity
liter obu porządków, rozkłady), `systemone.jsonl` (żądania System One wraz z
odpowiedzią `Server.decide`) i `manifest.json` (rewizje, wersje, sumy SHA-256).

## Zgodność runtime'u

```sh
# prompty i tokeny bez GPU
./target/release/basal check-prompts --model .models/basal-1.5-max --reference reports/reference-1.5-max-fp32

# wyniki runtime'u w formacie referencji
./target/release/basal export --model .models/basal-1.5-max --dtype f32 \
  --inputs reports/reference-1.5-max-fp32 --out reports/X/export-f32
./target/release/basal export --model .models/basal-1.5-max \
  --gemm-table reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json \
  --batching tree --inputs reports/reference-1.5-max-fp32 --out reports/X/export-f16-tree

# porównanie: tokeny, decyzje, różnice logitów i prawdopodobieństw, pola odpowiedzi
./target/release/basal compare --a reports/reference-1.5-max-fp32 --b reports/X/export-f16-tree --out reports/X/compare.json
```

`--batching single|budget|tree` i `--no-prefix-cache` pozwalają sprawdzić,
że wynik nie zależy od sposobu pakowania (porównanie dwóch eksportów runtime'u
powinno dać różnicę 0,0). Polecenia zapisujące wyniki odmawiają nadpisania
istniejącej ścieżki.

## Wydajność

| Narzędzie | Co mierzy |
|---|---|
| `basal bench` | pojedyncza decyzja metodyką basal-bench: 44 przykłady, oba porządki, batch 1, 39 pomiarów po odrzuceniu 5; przepustowość w grupach po 16 pytań |
| `basal bench-requests` | całe żądania System One, jeden klient (odpowiednik `tools/reference/bench_requests.py` dla upstream) |
| `tools/bench/ab.py` | porównanie dwóch wyników `bench` w parach, kolejność ABBA |
| `tools/bench/loadtest.py` | serwer HTTP, ten sam klient dla runtime'u i upstream: fazy sekwencyjna, zamknięta pętla N klientów i napływ otwarty (Poisson, `--rates` lub `--rate-fractions`); opóźnienia według klas żądań; z `--gpu` moc, zegar SM i J na decyzję z `nvidia-smi` |
| `tools/bench/make_long_states.py` | żądania o dokumentach 1k–16k tokenów |
| `tools/bench/make_mixed.py` | ruch mieszany: krótkie stany, wiele pytań, rozszerzenia 1.5, dokumenty do 16k |
| `tools/bench/split_requests.py` | rozbicie żądań wielopytaniowych na pojedyncze pytania o ten sam stan |

Przykład obciążenia serwera:

```sh
python3 tools/bench/make_mixed.py --model basal-1.5-max --out tools/bench/mixed.jsonl
python3 tools/bench/loadtest.py --gpu --model basal-1.5-max --requests tools/bench/mixed.jsonl \
  --n-warm 44 --n-seq 400 --n-conc 400 --concurrency 8 32 --rates 0.9 1.3 1.6 \
  --url http://127.0.0.1:8000/v1/systemone --out wynik.json
```

Loadtest porównuje odpowiedzi na to samo żądanie między fazami; różnica inna
niż 0,0 przy tabeli `--invariant` oznacza zależność wyniku od partii.

## Warunki

Pomiary CUDA wykonano na RTX 6000 Ada z limitem mocy 250 W (domyślnie 300 W).
Model 11B pracuje przy tym limicie stale: zegar SM spada do ~750–1000 MHz, więc
bezwzględne czasy są wyższe niż przy pełnej mocy, a porównania dotyczą tych
samych warunków. Upstream mierzony jest w trybie `fast` (BF16, torch.compile,
CUDA graphs), czyli w trybie, w którym serwuje.

Zestawy odniesienia (44 przykłady autora, 34 żądania System One dla
basal-1.5) wystarczają do sprawdzenia zgodności z upstream, ale nie do oceny
jakości na danych konkretnej aplikacji.
