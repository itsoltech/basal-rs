# Zestaw decyzyjny 900 pytań wobec upstream FP32

Data: 2026-10-08. Rozszerzenie porównania z upstream (dotąd 44 przykłady
basal-bench i 34 żądania System One) o 900 pytań z publicznych zbiorów z
etykietami, z najwyżej 10 opcjami, czyli bez strategii grupowej. Kryterium
jak w [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md): decyzje basal-rs (f16,
precyzja domyślna) mają być takie jak w forwardzie upstream FP32.

## Zestaw

[tools/decision-sets/build.py](../../tools/decision-sets/build.py), ziarno 0,
po 100 pytań z każdego zbioru (tak/nie: po 50 z każdej etykiety). SHA-256
zbudowanego pliku: `5b701fe861d626b3983928d3e5e168e452bbf6ba4137d75c76597e71e7691291`.

| Zbiór | Język | Typ | Opcje |
|---|---|---|---:|
| polemo2-in (KLEJ) | pl | choice | 4 |
| allegro-reviews (KLEJ) | pl | score | 5 |
| cdsc-e (KLEJ) | pl | choice | 3 |
| dyk (KLEJ) | pl | noul | 2 |
| cbd (KLEJ) | pl | noul | 2 |
| ag-news | en | choice | 4 |
| emotion | en | choice | 6 |
| boolq | en | noul | 2 |
| sst5 | en | score | 5 |

Teksty pytań nie są w repozytorium: część zbiorów ma licencje niekomercyjne
lub badawcze. Zestaw odtwarza skrypt z tym samym ziarnem (sumę kontrolną
wyżej można sprawdzić po zbudowaniu); w repozytorium są identyfikatory,
etykiety i rozkłady.

## Pomiar

Wynajęta RTX 6000 Ada (sterownik 580, 300 W;
[tools/cloud](../../tools/cloud/basal-cloud.py)), skrypt
[tools/decision-sets/run-cloud.sh](../../tools/decision-sets/run-cloud.sh):

- referencja upstream v1.5.0 (`cd63c083`) FP32: `export_reference.py --mode eager
  --device cuda --dtype float32 --questions … --no-system-one`;
- ścieżka serwowana upstream BF16 (`--mode fast`) dla porównania;
- basal-rs 0.1.4: `basal export` f16 (drzewo, tabele GEMM z serwera pomiarów,
  ta sama karta) i f32, `basal compare` z referencją FP32.

Modele: basal-1.5-4.5B (`784a683b`) i basal-1.5-mini (`1978d070`). Dla
basal-1.5-max referencja FP32 wymaga ponad 48 GB pamięci GPU (wagi 11B w FP32
to ok. 44 GB) i nie jest jeszcze policzona.

Dane: `compare-*.json` (wynik `basal compare`), `decisions-*.jsonl`
(rozkład po kalibracji dla każdego pytania: upstream FP32, basal-rs f16 i
f32, upstream BF16), `accuracy.json`, `manifest-reference-*.json`.

## Wynik

| Model | Wariant | Decyzje jak FP32 | Maks. różnica logitu | Maks. TV rozkładu |
|---|---|---:|---:|---:|
| 4.5B | basal-rs f32 | 900/900 | 0,00007 | 0,00002 |
| 4.5B | basal-rs f16 | 899/900 | 0,055 | 0,0084 |
| 4.5B | upstream BF16 | 890/900 | 0,566 | 0,080 |
| mini | basal-rs f32 | 900/900 | 0,00011 | 0,00002 |
| mini | basal-rs f16 | 899/900 | 0,055 | 0,012 |
| mini | upstream BF16 | 896/900 | 0,519 | 0,082 |

Tokeny i prompty: 1800 porządków na model, 0 różnic.

Jedyne różnice basal-rs f16 to remisy w FP32:

| Model | Pytanie | FP32: dwie najwyższe p | basal-rs f16 |
|---|---|---|---|
| 4.5B | dyk-15 (noul) | 0,503 / 0,497 | 0,500 dla drugiej opcji |
| mini | allegro-reviews-37 (score) | 0,336 / 0,335 | 0,336 dla drugiej opcji |

Upstream BF16 zmienia decyzje w 10 (4.5B) i 4 (mini) pytaniach, przy
różnicach w FP32 od 0,001 do 0,05; jedno z nich (mini, allegro-reviews-37)
to to samo pytanie.

Trafność wobec etykiet (z 100 na zbiór) jest w f16 taka sama jak w FP32 w 16
z 18 par model–zbiór; różnica o jedno pytanie w dyk (4.5B) i
allegro-reviews (mini) to te dwa remisy (`accuracy.json`).

## Powtórzenie

```sh
python3 tools/decision-sets/build.py .cache/decision-sets/set.jsonl
python3 tools/cloud/basal-cloud.py up --prefetch "mini 4.5B"   # build, push jak w docs/BENCHMARKS.md
python3 tools/cloud/basal-cloud.py sync tools/reference tools/decision-sets .cache/decision-sets/set.jsonl .cache/gemm .baseline/upstream-1.5
python3 tools/cloud/basal-cloud.py run 'tools/decision-sets/run-cloud.sh'
python3 tools/cloud/basal-cloud.py pull out/decision-sets .cache/decision-sets/RUN
```

Referencja FP32 liczy się raz na rewizję modelu i upstream (4.5B: ok. 6
minut, mini: 2 minuty). Zawiera teksty pytań, więc zostaje poza
repozytorium (`.cache/decision-sets/`); przy zmianach basal-rs wystarczy
`basal export` i `basal compare` z zapisaną referencją.
