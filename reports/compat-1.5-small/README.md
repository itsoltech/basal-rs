# basal-1.5-4.5B i basal-1.5-mini: zgodność z upstream

Data: 2026-10-06, RTX 6000 Ada (300 W). Modele `Remek/basal-1.5-4.5B`
(rewizja `784a683b`) i `Remek/basal-1.5-mini` (`1978d070`), upstream v1.5.0
(`cd63c083`). Tabele GEMM wygenerowane przez `gemm_table: auto`
([multi-model](../rust-cuda-1.5-max/multi-model/README.md)). Skrypt: `run.sh`.

Referencje upstream (`../reference-basal-1.5-*-fp32`): eksport
`--mode eager --device cuda --dtype float32`, 44 przykłady basal-bench i 34
żądania System One (w tym `multi`, `act`, `facts`, `evidence` z
`tools/reference/systemone_cases_1.5.jsonl`). Dla porównania ścieżka
serwowana upstream w BF16 (`../reference-basal-1.5-*-bf16`, `--mode fast`).

## 44 przykłady basal-bench, wobec upstream FP32

| Model | Wariant | Decyzje | Maks. różnica logitu | Maks. różnica p (kal.) |
|---|---|---:|---:|---:|
| 1.5-4.5B | basal-rs FP32 | 44/44 | 0,00007 | 0,00002 |
| 1.5-4.5B | basal-rs FP16, pojedynczo | 44/44 | 0,037 | 0,0027 |
| 1.5-4.5B | basal-rs FP16, drzewo | 44/44 | 0,037 | 0,0027 |
| 1.5-4.5B | upstream BF16 (`fast`) | 44/44 | 0,373 | 0,021 |
| 1.5-mini | basal-rs FP32 | 44/44 | 0,00007 | 0,00001 |
| 1.5-mini | basal-rs FP16, pojedynczo | 44/44 | 0,025 | 0,0060 |
| 1.5-mini | basal-rs FP16, drzewo | 44/44 | 0,025 | 0,0060 |
| 1.5-mini | upstream BF16 (`fast`) | 44/44 | 0,415 | 0,051 |

Trafność wobec etykiet autora taka sama jak upstream FP32: 35/44 (4.5B) i
31/44 (mini). FP16 pojedynczo i drzewem daje bitowo te same logity.

## Żądania System One

`/v1/basal` (`answer_upstream` wobec `Server.decide` upstream FP32), oba
modele, FP16 i FP32: na 34 żądaniach błędy w tych samych miejscach (2
odrzucone przez oba), 55/55 pytań z tymi samymi polami i decyzjami: wybór
`multi`, akcja, odmowa i kalibracja `act`, te same fragmenty `evidence`
(offsety i teksty), odpowiedzi przy `facts: "auto"`. Maks. różnica
prawdopodobieństwa w FP16: 0,0067 (4.5B), 0,0032 (mini); w FP32 0,00001.

Prompty i token IDs (`check-prompts-*.json`): 124 porządki, 8 różnic w
formacie eksportu, te same co dla basal-1.5-max: pytania `multi` w
eksporcie upstream nie mają własnych porządków (rozwijane w gałęzie), a przy
`facts` eksport upstream zapisuje prompt sprzed dopisania faktów. Odpowiedzi
tych pytań są zgodne (wyżej).

Odpowiedzi `/v1/systemone` różnią się od upstream formatem w sposób opisany
w [docs/SYSTEM_ONE.md](../../docs/SYSTEM_ONE.md) (np. noul bez
`probabilities` i `confidence`, confidence według TypeSafe); decyzje są te
same.
