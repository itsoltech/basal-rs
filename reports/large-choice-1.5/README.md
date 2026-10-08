# Choice 11–255 na modelach basal-1.5

Data: 2026-10-07. Ta sama ocena strategii grupowej co w
[large-choice](../large-choice/README.md) (basal-1.0-4.5B, CUDA), teraz na
basal-1.5-mini, basal-1.5-4.5B i basal-1.5-max (rewizje z
`basal init`: `1978d070`, `784a683b`, `be1b5ee7`). Metal f16 na Apple M2 Max
(zasilacz), basal-rs 0.1.3. Polecenie dla każdego modelu:

```sh
basal eval-large-choice --model .models/basal-1.5-MODEL \
  --reference reports/reference-basal-1.5-MODEL-fp32 \
  --out reports/large-choice-1.5/MODEL-metal --sizes 11,20,40,100
```

(dla max referencja `reports/reference-1.5-max-fp32`). Dane: `*-metal/items.jsonl`
i `summary.json`, logi `*-metal.log`.

Każdy z 44 przykładów basal-bench dostaje dystraktory z przykładów o innym
temacie do n opcji, w dwóch ziarnach (88 pytań na rozmiar). Pula ma 114
opcji, więc n=100 obejmuje 60 pytań, a n=255 nie da się tak zbudować.
„Bezpośrednio” to to samo pytanie z samymi oryginalnymi opcjami (2–10, jeden
odczyt), czyli najlepszy wynik, jakiego można oczekiwać od strategii.

| Model | n | Trafność | Bezpośrednio | Argmax oryginalnych = bezpośrednie | Mediana TV wobec bezpośredniego | Mediana masy na oryginalnych | Ten sam zwycięzca w obu ziarnach | Mediana czasu pytania | Prompty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mini | 11 | 58/88 | 62/88 | 82/88 | 0,016 | 0,89 | 38/44 | 374 ms | 10 |
| mini | 20 | 53/88 | 62/88 | 83/88 | 0,032 | 0,79 | 34/44 | 468 ms | 10 |
| mini | 40 | 53/88 | 62/88 | 82/88 | 0,032 | 0,56 | 33/44 | 748 ms | 18 |
| mini | 100 | 30/60 | 42/60 | 52/60 | 0,043 | 0,29 | 25/30 | 1636 ms | 42 |
| 4.5B | 11 | 71/88 | 70/88 | 85/88 | 0,008 | 0,93 | 43/44 | 1123 ms | 10 |
| 4.5B | 20 | 71/88 | 70/88 | 85/88 | 0,010 | 0,80 | 43/44 | 1402 ms | 10 |
| 4.5B | 40 | 66/88 | 70/88 | 85/88 | 0,009 | 0,60 | 40/44 | 2248 ms | 18 |
| 4.5B | 100 | 36/60 | 46/60 | 59/60 | 0,018 | 0,38 | 26/30 | 4906 ms | 42 |
| max | 11 | 73/88 | 72/88 | 85/88 | 0,005 | 0,95 | 43/44 | 2509 ms | 10 |
| max | 20 | 73/88 | 72/88 | 86/88 | 0,007 | 0,89 | 43/44 | 3124 ms | 10 |
| max | 40 | 70/88 | 72/88 | 86/88 | 0,011 | 0,69 | 42/44 | 5140 ms | 18 |
| max | 100 | 41/60 | 50/60 | 60/60 | 0,028 | 0,48 | 26/30 | 11154 ms | 42 |

Dla porównania basal-1.0-4.5B (CUDA): 66/88, 61/88, 56/88 i 36/60 przy
bezpośrednich 70/88; mediana TV 0,035–0,095.

- basal-1.5-4.5B i basal-1.5-max przy 11 i 20 opcjach mają trafność równą
  pytaniu bezpośredniemu (±1), przy 40 opcjach niższą o 2–4 pytania, przy
  100 o 9–10 z 60.
- Wśród oryginalnych opcji strategia wybiera to samo co pytanie bezpośrednie
  w 85–86/88 (n=100: 59–60/60), a rozkład oryginalnych opcji różni się
  medianowo o 0,005–0,028 TV, kilka razy mniej niż na basal-1.0. Spadek
  trafności przy dużym n pochodzi z dystraktorów, które przejmują coraz
  więcej masy, a nie z innego porządku oryginalnych opcji.
- basal-1.5-mini traci już przy 11 opcjach (58/88 wobec 62/88) i przy 100
  opcjach zostawia na oryginalnych opcjach medianowo 0,29 masy.
- Mapa tematów jest zgrubna, więc część dystraktorów może być sensowną
  odpowiedzią; nie ma referencji upstream (upstream obsługuje do 10 opcji).

Czasy dotyczą pojedynczego pytania wykonywanego osobno (bez partii z innymi
żądaniami) na M2 Max; liczba promptów rośnie z n (10 do 42), a czas z nią i z
rozmiarem modelu. Na M2 Max pytanie ze 100 opcjami trwa 1,6 s (mini), 4,9 s
(4.5B) i 11,2 s (max). Pomiar CUDA basal-1.5 nie był robiony: GPU serwera
testowego obsługiwał w tym czasie produkcyjny kontener.
