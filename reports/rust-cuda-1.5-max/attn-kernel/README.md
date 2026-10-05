# Kernel attention: kolejne iteracje

Data: 2026-10-05, basal-1.5-max, RTX 6000 Ada, limit 250 W, tabela GEMM
`gemm-retune/gemm-algos-f16-invariant-ladder.json`. Punkt wyjścia: kernel z
[attn-bm128](../attn-bm128/) (Q w rejestrach, 32 tokeny zapytań na blok).

Każda iteracja: eksport 44 przykładów i przypadków 1.5 porównany z
poprzednią wersją i z FP32 upstream, potem `bench-requests` na 10 żądaniach z
długimi stanami w kolejności ABBA (`ab.sh`: poprzedni commit kontra drzewo
robocze; `ab-env.sh`: dwa warianty `BASAL_ATT` jednej binarki). Kontrola z
identycznym kodem po obu stronach dała ilorazy 1,007–1,027 na niekorzyść
drugiej strony, więc różnice do ~2% to szum i skrzywienie kolejności (wyniki
kontroli nie zostały zachowane).

| Iteracja | Zmiana | Wynik bitowo równy | Wobec FP32 upstream | 16k, 1 / 5 pytań (iloraz) | 1k–8k |
|---|---|---|---:|---:|---:|
| [cpasync](cpasync/) | podwójne buforowanie K/V przez `cp.async` | tak | 0,171 | 1,038 / 1,030 | 1,03–1,04 |
| [bn64](bn64/) | kafelek 64 kluczy | nie (0,092 logitu) | 0,207 | 0,994 / 1,000 | 1,00–1,02 |
| [ex2b](ex2b/) | softmax przy podstawie 2 (`ex2.approx`), bez przeskalowania przez 1 | nie (0,132 logitu) | 0,164 | 0,976 / 0,982 | 0,99–1,02 |
| [ex2-mma](ex2-mma/) | jw. + `mma` bez `volatile` | tak względem ex2b | 0,164 | 0,969 / 0,983 | 0,99–1,02 |
| [pipe](pipe/), [pipe2](pipe2/), [pipe3](pipe3/) | `attn_tree_tc_pipe`: Q·K następnego kafelka obok softmaxu i P·V bieżącego, K/V asynchronicznie z wyprzedzeniem; w pipe3 adres kafelka raz na kafelek i maska tylko przy przekątnej | tak względem `tc` | 0,164 | 1,036 / 1,019 (pipe3) | 1,02–1,03 |

Do kodu weszło `ex2` (stan 16k ~3% szybciej po odjęciu skrzywienia, różnica
wobec FP32 0,164 jak dotąd 0,165–0,171, 44/44). `attn_tree_tc_pipe`
zostaje jako wariant `BASAL_ATT=tc-pipe`.

Nsight Compute, jedno wywołanie w forwardzie stanu ~4k tokenów, zegar
ustalony przez profiler:

| Metryka | `tc` | `tc-pipe` |
|---|---:|---:|
| Czas | 6,38 ms | 6,10 ms |
| Tensor cores aktywne | 63,6% | 67,0% |
| Wykonane instrukcje | 461 mln | 539 mln |
| Przestoje: bariera / długi scoreboard (cykle na warp) | 47,8k / 90,1k | 20,0k / 30,9k |

Przy stałym zegarze potok jest szybszy o 4,4%. Pod limitem mocy, gdzie
zegar zależy od poboru, dodatkowe 17% instrukcji zjada ten zysk. Na
kafelek i warp przypada ~220 instrukcji ALU i ~190 FMA wobec 160 MMA.

Pomiar graniczny (wyniki błędne, tylko czas; stan 16k, 1 pytanie, 6,03 s z
pełnym kernelem): bez mnożeń Q·K 4,82 s, bez P·V 5,41 s, bez eksponenty
5,74 s. Same mnożenia z rozbiciem hi/lo to około połowy czasu attention.
