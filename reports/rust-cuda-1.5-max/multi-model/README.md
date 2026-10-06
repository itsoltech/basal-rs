# Kilka modeli w jednym procesie

Data: 2026-10-06, RTX 6000 Ada, limit mocy 300 W. `basal serve --config
config.yml`: basal-1.5-max (tabela GEMM z repozytorium), basal-1.5-4.5B
(rewizja `784a683b`) i basal-1.5-mini (rewizja `1978d070`) z `gemm_table:
auto`. Skrypt: `run.sh`, sprawdzenia: `smoke.py`.

## Start

- Pierwszy start z generowaniem tabel GEMM dla 4.5B (213 s) i mini (140 s):
  391 s łącznie. Pamięć GPU po starcie 34,4 GB.
- Ponowny start z tabelami z `gemm_cache`: 39 s.

## Sprawdzenia (`smoke-multi.json`, 13/13)

`/v1/models` wymienia trzy modele. Oba endpointy kierują żądanie do modelu z
pola `model`, a odpowiedź ma jego nazwę. `/v1/basal` bez tego pola trafia do
`default_model` i daje te same odpowiedzi co z jawną nazwą. Nieznana nazwa
daje 422: `unknown_model` z listą modeli w `/v1/systemone` albo `{"error"}`
w `/v1/basal`. `/v1/systemone` bez pola `model` daje błąd schematu.

## Zgodność

- basal-1.5-max: odpowiedzi serwera wielomodelowego bitowo równe odpowiedziom
  serwera z jednym modelem (`smoke-multi.json` wobec `smoke-single.json`).
- Wygenerowane tabele 4.5B i mini: eksport 44 przykładów pojedynczo i
  drzewem bitowo równy (`compare-*-single-vs-tree.json`: 0,0).
- Zgodności 4.5B i mini z FP32 upstream nie sprawdzano (brak referencji dla
  tych wag).

## Obciążenie trzech modeli naraz

Trzy procesy `loadtest.py` jednocześnie, każdy na swój model (44 przykłady
basal-bench), fazy po kolei: 1 klient, potem 16 klientów. Fazy różnych
modeli nakładają się tylko częściowo, więc liczby nie są przepustowością
pojedynczego modelu na wyłącznej karcie.

| Model | 1 klient: żądania/s, p50 / p95 | 16 klientów: żądania/s, p50 / p95 | Śr. partia |
|---|---:|---:|---:|
| basal-1.5-max | 7,2, 94 / 468 ms | 15,9, 1013 / 1259 ms | 13,8 |
| basal-1.5-4.5B | 9,9, 98 / 183 ms | 14,8, 1060 / 1787 ms | 13,8 |
| basal-1.5-mini | 10,3, 96 / 161 ms | 18,6, 786 / 1683 ms | 10,1 |

Bez błędów; odpowiedzi na to samo żądanie identyczne we wszystkich fazach
(różnica 0,0). GPU przez cały czas przy limicie (292–298 W).

Pierwsze wersje serwera kończyły się po Ctrl-C segfaultem (`run.out`: wątki
GPU trzymały zasoby CUDA w chwili wyjścia procesu). Serwer czeka teraz na
zakończenie wątków modeli; po poprawce wyjście z kodem 0 dla jednego i trzech
modeli.
