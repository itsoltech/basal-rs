# basal-1.5-max (11B) na RTX 6000 Ada: zgodność i wydajność

Data: 2026-10-05. Model `Remek/basal-1.5-max`, rewizja `be1b5ee7`, SHA-256
wag `e11abe4a…`. Upstream `v1.5.0` (`cd63c083`), torch 2.13.0+cu130.
Środowisko jak w [rust-cuda-v1](../rust-cuda-v1/README.md); GPU bez innych
procesów, limit mocy obniżony do 250 W (`nvidia-smi -pl 250`, domyślnie 300 W).
Przy ciągłym obciążeniu modelem 11B karta pracuje na tym limicie: średnio
815 MHz zegara SM (maks. 2775), 76–81 °C. Bezwzględne czasy obu implementacji
są przez to wyższe niż przy pełnej mocy; porównania dotyczą tych samych
warunków.

## Model

Llama, 50 warstw, hidden 4096, intermediate 14336, 32/8 głowic, head_dim 128,
bez biasów, rms_eps 1e-5, 22,3 GB w bf16. Kontrakt promptu jest ten sam co w
basal-1.0 (tokeny 88 promptów identyczne). Maksymalna wartość bezwzględna
strumienia residualnego w FP16 na zestawie odniesienia: 669 (limit 65504).

## Zgodność z FP32 upstream 1.5

Referencja: `export_reference.py --mode eager --device cpu --dtype float32`
z upstream v1.5.0 (`../reference-1.5-max-fp32`, 923 s na CPU).

| Wariant | Decyzje 44 | Maks. różnica logitu | Maks. różnica p_avg | Trafność |
|---|---:|---:|---:|---:|
| Rust FP32 | 44/44 | 0,00013 | 0,00002 | 36/44 |
| Rust FP16 (cuBLASLt, heurystyka) | 44/44 | 0,180 | 0,0040 | 36/44 |
| Rust FP16, tabela niezależna od partii | 44/44 | 0,202 | 0,0063 | 36/44 |

`multi` i `act` (8 żądań z `tools/reference/systemone_cases_1.5.jsonl`
wobec `Server.decide` upstream 1.5 w FP32): wszystkie pola
(`selected`, `action`, `refused`, `answer`, `calibration`, `max_error`)
identyczne, maks. różnica liczbowa 4·10⁻⁶ (Rust FP32) i 4·10⁻⁴ (FP16).

Niezależność od partii (tabela `gemm/gemm-algos-f16-invariant.json`,
attention po węzłach drzewa): eksporty pojedynczo, partiami budżetowymi,
wierszami-drzewami i bez cache prefiksu dają bitowo te same logity (różnica
0,0); pod obciążeniem HTTP odpowiedź na to samo żądanie jest identyczna
niezależnie od partii (upstream: różnice do 0,062).

`facts: "auto"` i `evidence` (8 kolejnych żądań, 16 łącznie z `multi`/`act`,
`../reference-1.5-max-fp32-cases2`): na 22 pytaniach odpowiedzi i fragmenty
dowodów identyczne jak w upstream FP32 (te same offsety i teksty fragmentów;
różnica prawdopodobieństw fragmentów do 8·10⁻⁷ w Rust FP32, 3·10⁻⁴ w FP16;
maks. różnica prawdopodobieństw odpowiedzi w FP16 0,0016). Port `facts.py`:
432/432 stanów identycznych co do bajtu (`tools/reference/facts_parity.py`).

## Attention na tensor cores

Kernel `attn_tree_tc` (mma.sync m16n8k16, akumulacja f32, operandy f16 hi+lo,
czyli dokładność bliska f32; K/V dzielone raz na warstwę; P·V_lo pominięte,
gdy V jest dokładne w f16). Wobec FP32 upstream: 44/44, maks. różnica logitu
0,165, TV 0,002 (kernel f32 SIMT: 0,202 / 0,006). Niezależność od partii
zachowana (różnica 0,0 pojedynczo / drzewa / bez cache prefiksu). A/B na
krótkich decyzjach (`ab-attn-simt-vs-tc/`): szybciej o ~3%.

## Długie stany (tools/bench/make_long_states.py, mediany z 2 powtórzeń)

| Stan (tokeny) | Upstream, 1 pytanie | Rust, 1 pytanie | Upstream, 5 pytań | Rust, 5 pytań |
|---:|---:|---:|---:|---:|
| 1 078 | 391 ms | 270 ms | 3 009 ms | 431 ms |
| 2 146 | 1 045 ms | 552 ms | 5 101 ms | 725 ms |
| 4 169 | 1 591 ms | 1 144 ms | 7 947 ms | 1 370 ms |
| 8 221 | 3 722 ms | 2 671 ms | 18 464 ms | 3 030 ms |
| 16 527 | 9 806 ms | 7 064 ms | 49 112 ms | 7 866 ms |

Pierwsza wersja (`long-states-rust-v1.json`) potrzebowała przy 16k tokenach i
5 pytaniach 110,8 s: limit wiersza 3072 tokenów (jak w upstream) dzielił
pytania na osobne wiersze, więc stan był liczony 5 razy, a kernel attention
f32 SIMT skalował się kwadratowo z dużą stałą. Limit wiersza wynosi 32768
tokenów (przy attention po węzłach drzewa długość wiersza nie zmienia wyniku
pytania). Upstream przy wielu pytaniach o długi dokument dzieli wiersze po
3072 tokeny. Przy 16k tokenach attention
to ~60% czasu GPU (nsys `nsys-long16-tc2`).

## Pojedyncza decyzja (basal-bench, 44 przykłady)

| Wariant | lat2 mediana | lat2 p95 | lat1 | Decyzje/s |
|---|---:|---:|---:|---:|
| Upstream `fast` (bf16, compile + CUDA graphs) | 90,7 ms | 122,3 ms | 86,3 ms | 9,5 |
| Rust FP16, tabela GEMM strojona | 68,5–69,1 ms | 81,6–82,7 ms | 46,5 ms | 21,1–21,2 |
| Rust FP16, tabela niezależna od partii | 63,1 ms | 69,7–69,9 ms | 46,0 ms | 19,8 |
| Rust FP16, jw. + attention na tensor cores (wersja końcowa, 2 przebiegi) | **59,1–59,7 ms** | **65,2 ms** | **45,9 ms** | **20,6–20,8** |

Wersja końcowa wobec upstream: mediana niższa o 35%, p95 o 47%, lat1 o 47%,
przepustowość 2,2×. Narzut serwera HTTP: ~1,4 ms na żądanie (nagłówki
`x-basal-compute-ms` wobec czasu curl). Porównanie dwóch tabel Rust w parach (ABBA, 2
rundy, `ab-table-vs-invariant/`): wariant niezależny od partii szybszy na
36/39 pytań (mediana ilorazu 0,90), przepustowość partii niższa o ~6%.
Tabelę strojoną przeszukano przy zegarze karty spadającym do ~600 MHz, co
mogło zniekształcić wybór.

## Serwer HTTP (tools/bench/loadtest.py, ten sam klient)

Jedno pytanie na żądanie (44 przykłady w pętli):

| Klienci | Upstream p50 / p95 | Upstream żądania/s | Rust p50 / p95 | Rust żądania/s |
|---:|---:|---:|---:|---:|
| 1 (sekwencyjnie) | 118 / 136 ms | 9,15 | 66 / 92 ms | 14,7 |
| 4 | 493 / 583 ms | 8,0 | 280 / 317 ms | 14,2 |
| 8 | 900 / 1033 ms | 8,5 | 522 / 569 ms | 15,1 |
| 16 | 1876 / 1923 ms | 8,8 | 996 / 1031 ms | 15,9 |
| 32 | 4683 / 5109 ms | 6,8 | 1980 / 2015 ms | 16,1 |

Żądania wielopytaniowe (`tools/reference/requests_fanout.jsonl`): 1 klient
p50 205 ms (upstream) vs 160 ms (Rust), 14,2 vs 17,6 decyzji/s; 8
klientów 11,2 vs 16,2 decyzji/s.

## Współbieżność

Przegląd `sweep-concurrency/` (`run.sh`): tabela GEMM niezależna od partii
(`invariant`) i dostrojona per klasa M (`tuned`, split-K) × budżet partii
2048–16384 tokenów × 1–64 klientów, 44 jednopytaniowe żądania o różnych
stanach, próbkowanie `nvidia-smi` co 200 ms.

| Klienci | invariant 8192: p50 | dec/s | J/decyzję | śr. partia | tuned 4096: dec/s | J/decyzję |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67 ms | 14,5 | 17,2 | 1,0 | 13,1 | 19,0 |
| 8 | 499 ms | 15,8 | 15,8 | 6,2 | 15,6 | 16,0 |
| 32 | 1897 ms | 16,8 | 14,8 | 21,4 | 18,2 | 13,7 |
| 64 | 3509 ms | 17,1 | 14,6 | 28,7 | 18,0 | 13,9 |

Moc 249 W we wszystkich fazach, także przy jednym kliencie: prefill 11B na
~350 tokenach jest ograniczony limitem mocy, zegar SM spada z ~1000 MHz
(1 klient) do 740–840 MHz (partie). Łączenie żądań daje +16–25%
przepustowości; opóźnienie przy N klientach to prawie wyłącznie kolejka
(p50 ≈ N / przepustowość). Budżet partii 2048–16384 zmienia przepustowość
o kilka procent, w granicach rozrzutu przebiegów. Tabela `tuned` jest przy
partiach do 8% szybsza, ale wynik zależy od partii (do 0,012
prawdopodobieństwa); `invariant` daje 0,0 we wszystkich fazach. Przebieg
`tuned` 16384 przy 64 klientach (10,2 dec/s, 649 MHz) odstaje; w tym czasie
działały na hoście inne procesy CPU.

Współdzielenie stanu między żądaniami (`trie-load/`, `run.sh`, ABBA: wersja
z wierszem o trzech poziomach i wersja z wierszem-trie oraz budżetem liczonym
po współdzieleniu, tabela `invariant`, budżet 8192). Mediany p50, decyzje/s z
przebiegów 2–3 (nowa) i 4 (stara; przebieg 1 na chłodniejszej karcie był
szybszy o ~8% dla obu wersji):

| Ruch | Klienci | Stara p50 | Stara dec/s | Nowa p50 | Nowa dec/s |
|---|---:|---:|---:|---:|---:|
| 44 różne stany | 1 | 68 ms | 14,3 | 68 ms | 14,3 |
| 44 różne stany | 32 | 1878 ms | 17,1 | 1862–1869 ms | 17,1 |
| 14 pytań o 5 stanów, 1 pytanie na żądanie | 8 | 615 ms | 13,6 | 510 ms | 15,6 |
| jw. | 32 | 2238 ms | 14,1 | 1312–1344 ms | 23,5–24,2 |
| 5 pytań o dokument 2k i 4k tokenów, 1 pytanie na żądanie | 5 | 4022 ms | 1,14 | 2135 ms | 2,33 |
| jw. | 10 | 8786 ms | 1,12 | 2655 ms | 3,76 |

Stara wersja liczyła budżet partii bez współdzielenia (5 pytań o stan 4k
tokenów przekraczało 8192), więc partie miały średnio 1,55 żądania; nowa
łączy 8,2 żądania i liczy stan raz. Prawdopodobieństwa identyczne we
wszystkich fazach (różnica 0,0). Weryfikacja zmiany pakowania: eksport
pojedynczo, budżetowo, drzewem i bez cache prefiksu bitowo równy
wersji przed zmianą (`compare-tc2-single-vs-trie-*.json`: 0,0), przypadki 1.5
pojedynczo = drzewem (`compare-trie-cases-single-vs-tree.json`), FP32
drzewem względem FP32 upstream 2,7·10⁻⁴ logitu, 44/44
(`compare-upstream-fp32-vs-rust-f32-trie-tree.json`).

## Ruch mieszany

Różne stany, 1–14 pytań, wszystkie typy, dokumenty do 16k tokenów, oraz
harmonogram z drugim torem dla długich żądań: [mixed-load/](mixed-load/README.md).
Sekwencyjnie Rust 1,74 żądania/s, upstream 0,29; z dwoma torami żądania krótkie
i do ~2k tokenów mają p95 poniżej 1,8 s przy napływie do 1,6 żądania/s.

## Ograniczenia

- Pomiary przy limicie 250 W; przy 300 W oba runtime'y byłyby szybsze, czego
  nie mierzono.
- Pojedyncze przebiegi (benchmark A/B: 2 rundy).
- Upstream FP32 tylko na CPU (wagi FP32 nie mieszczą się w GPU z
  aktywacjami); Rust FP32 trzyma wagi bf16 i rozszerza je per warstwa.
- Długie stany to syntetyczny dokument (pomiar czasu, nie jakości).
