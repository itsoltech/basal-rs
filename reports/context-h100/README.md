# Długość kontekstu na H100: basal-rs wobec upstream

Data: 2026-10-08. Wynajęta H100 PCIe (80 GB, limit 350 W, sterownik 580,
compute capability 9.0, 20 vCPU), basal-rs 0.1.4 i upstream v1.5.0 `fast`
(BF16, torch.compile, CUDA graphs). Skrypt:
[tools/perf/run-context.sh](../../tools/perf/run-context.sh), zestaw
[tools/bench/make_context_ladder.py](../../tools/bench/make_context_ladder.py),
tabele [tools/perf/context_table.py](../../tools/perf/context_table.py)
(`tables.md`).

Pytanie: jak basal-rs wypada przy promptach o długości z tabeli przyspieszeń
autora basala (prompt 1792 tokenów, druga kolejność opcji jako 128 tokenów
nad policzonym stanem) i na całej drabinie długości stanu. Żądania System One
ze stanem ~512–16384 tokenów (syntetyczna umowa jak w
[make_long_states.py](../../tools/bench/make_long_states.py)) i 1 lub 5
pytaniami; mediana całego żądania (renderowanie, tokenizacja, forward obu
porządków, kalibracja), jeden klient, bez HTTP: `basal bench-requests` (3
powtórzenia po rozgrzewce) i `tools/reference/bench_requests.py` (2). basal-rs
pakuje oba porządki i wszystkie pytania w jedno drzewo prefiksów (kolumna
„tokeny drzewa”), upstream liczy stan i drugi porządek osobno.

H100 PCIe ma niższy limit mocy i zegar niż H100 SXM (700 W) z
[perf-gpus](../perf-gpus/README.md) i z tabeli autora; czasy bezwzględne nie
są porównywalne z SXM, porównanie basal-rs z upstream jest na tej samej karcie.

## basal-1.5-4.5B

| Stan | Pytania | Tokeny promptów (oba porządki) | Tokeny drzewa basal-rs | Upstream | basal-rs (tabela GEMM) | basal-rs (heurystyka cuBLASLt) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 1 030 | 553 | 23 ms | 31 ms | 23 ms |
| 512 | 5 | 7 178 | 949 | 48 ms | 65 ms | 47 ms |
| 1 024 | 1 | 1 962 | 1 019 | 40 ms | 56 ms | |
| 1 792 | 1 | 3 518 | 1 797 | 84 ms | 100 ms | 76 ms |
| 1 792 | 5 | 24 582 | 2 193 | 109 ms | 151 ms | 117 ms |
| 2 048 | 1 | 4 100 | 2 088 | 96 ms | 115 ms | 90 ms |
| 2 048 | 5 | 28 656 | 2 484 | 143 ms | 170 ms | 136 ms |
| 4 096 | 1 | 8 338 | 4 207 | 487 ms | 267 ms | 212 ms |
| 4 096 | 5 | 58 322 | 4 603 | 3 399 ms | 346 ms | 284 ms |
| 8 192 | 1 | 16 630 | 8 353 | 1 340 ms | 620 ms | |
| 8 192 | 5 | 116 366 | 8 749 | 9 376 ms | 754 ms | |
| 16 384 | 1 | 33 644 | 16 860 | 4 200 ms | 1 743 ms | |
| 16 384 | 5 | 235 452 | 17 256 | 29 458 ms | 1 991 ms | |

## basal-1.5-max

| Stan | Pytania | Upstream | basal-rs (tabela GEMM) | basal-rs (heurystyka) |
|---:|---:|---:|---:|---:|
| 512 | 1 | 41 ms | 56 ms | 40 ms |
| 512 | 5 | 89 ms | 130 ms | 86 ms |
| 1 792 | 1 | 152 ms | 194 ms | 142 ms |
| 1 792 | 5 | 206 ms | 286 ms | 213 ms |
| 2 048 | 1 | 180 ms | 225 ms | 167 ms |
| 2 048 | 5 | 258 ms | 323 ms | 248 ms |
| 4 096 | 1 | 892 ms | 510 ms | 387 ms |
| 4 096 | 5 | 6 235 ms | 631 ms | 504 ms |
| 8 192 | 1 | | 1 169 ms | |
| 16 384 | 5 | | 3 626 ms | |

Upstream na max mierzony do 4k tokenów (limit czasu sesji).

## Wnioski

- Od ~4k tokenów basal-rs jest szybszy 1,75–2,4 raza przy jednym pytaniu i
  9,8–15 razy przy pięciu: upstream liczy drugi porządek opcji i każde pytanie
  z gęstą maską nad całym stanem, czas rośnie z kwadratem długości.
- Do ~2k tokenów w konfiguracji domyślnej (tabela GEMM niezależna od partii)
  basal-rs jest na H100 wolniejszy o 16–32%. Z heurystyką cuBLASLt (`gemm_table:
  none`) jest porównywalny z upstream (od 7% wolniej do 10% szybciej; 4.5B,
  1792 tokenów, jedno pytanie: 76 wobec 84 ms), a cała drabina jest szybsza o
  18–34% niż z tabelą.
- Przyczyna: GEMM-y z tabelą osiągają na H100 ~160–230 TFLOP/s, z
  heurystyką ~260–360 TFLOP/s (4.5B: 4,62 mld parametrów w warstwach
  liniowych; sekcje GEMM z `basal profile`: 73,2 i 47,3 ms przy 1831
  tokenach). Przegląd konfiguracji cuBLASLt dla tabeli niezależnej od partii
  (`crates/basal-gpu/src/cublaslt.rs`, `invariant_search`) nie bierze
  algorytmów z heurystyki (dodaje je tylko przy dozwolonym split-K) i nie
  ustawia rozmiaru klastra, od którego zależą kernele Hoppera. Na RTX 6000 Ada
  tabela była szybsza od heurystyki o ~8%
  ([ab-table-vs-invariant](../rust-cuda-1.5-max/ab-table-vs-invariant/)).
- Udział sekcji forwardu basal-rs (`profile-*.json`, czasy z synchronizacją po
  każdej sekcji): GEMM 62–64% do 2k tokenów, attention rośnie z 13–16% przy
  512 do 54–55% przy 16k tokenach.

Heurystyka cuBLASLt nie gwarantuje wyniku niezależnego od partii (bez tabeli
odpowiedź może zależeć od tego, z czym pytanie trafi do partii); zgodność z
FP32 bez tabeli sprawdzona na A100 w
[decision-sets](../decision-sets/README.md) (899–900/900).

Koszt sesji: ~$2 (45 minut).
