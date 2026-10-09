# Wyniki inferencji: H100 i RTX 6000 Ada — 2026-10-09

CUDA 12.9.1, f16, basal-1.5-4.5B i max, przypięte modele i tabele GEMM.
H100 PCIe 350 W, RTX 6000 Ada 300 W. Kernele: cztery powtórzenia,
mediany 2–4; krótkie decyzje i konteksty 512–16384 tokenów.

## Co sprawdzono

| Zmiana | Wynik | Decyzja |
|---|---|---|
| SiLU 256, H100/4.5B | latency 12,574 → 12,327 ms (−1,97%); throughput 168,863 → 170,471 dec/s (+0,95%) | włączone dla Hopper f16 vec8 |
| SiLU 256, H100/max i HTTP | brak rozstrzygającego zysku | bez deklaracji poprawy |
| SiLU 256, Ada/4.5B | latency +2,1%; throughput −0,6% | pozostaje 1024 |
| SiLU 128/512, H100/4.5B | latency −1,6%/−1,0%; throughput +0,9%/+0,5% | wybrano 256 |
| SiLU 128/512, Ada/4.5B | latency +0,6%/+1,6%; throughput −0,4%/−0,5% | pozostaje 1024 |
| RMSNorm z rejestrów | throughput −1,7% H100, −1,1% Ada; latency bez poprawy | wyłączone |
| Batch 4096/2048 i kwant 10 ms | poprawa części opóźnień kosztem throughput lub innych klas żądań | domyślne wartości bez zmian |
| Istniejący cache 2048 MiB/tor, powtarzane stany | throughput ×1,30 H100, ×3,08 Ada; większe VRAM, możliwe pogorszenie krótkich żądań | pozostaje opcjonalny |

Wynik cache dotyczy rozgrzanego cache, dwóch wspólnych dokumentów i 32
klientów; nie jest zyskiem nowego kernela ani wynikiem dla unikalnych stanów.

## Co zweryfikowano

- Warianty kerneli i single/tree: zero różnic tokenów, logitów,
  prawdopodobieństw i zapisanych odpowiedzi drabiny.
- Końcowe eksporty 4.5B na obu kartach: identyczne bajtowo z bazą dla
  tree, budget, bez cache prefiksu i z cache stanu; 44 bench oraz 34
  System One (30 poprawnych, cztery oczekiwane błędy).
- HTTP: 129 faz, 29 882 odpowiedzi bez rozgrzewek, zero błędów i różnic
  wewnątrz przebiegów. Końcowe odpowiedzi 400 ID porównano również
  między konfiguracjami, osobno dla każdego GPU.
- Fmt, Clippy, Rustdoc na macOS i Linux/CUDA, cargo-deny, ShellCheck
  i py_compile przeszły.

Brak dowodu poprawy innych GPU/modeli ani pełnej zgodności z FP32 upstream:
osiem rozbieżności System One występowało już w bazie. Małe zyski i wyniki
syntetycznego HTTP nie gwarantują produkcyjnego SLO. Pełne dane zachowano
w lokalnym archiwum poza Gitem.
