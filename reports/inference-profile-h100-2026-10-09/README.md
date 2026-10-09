# Wyniki SW128: H100 — 2026-10-09

H100 PCIe 80 GB, 350 W, CUDA 12.9.1, f16, basal-1.5-4.5B i max,
przypięte modele i tabele GEMM. Porównanie z poprzednim układem attention
po pierwszej sesji, nie skumulowany zysk wszystkich zmian względem main.
Kernele: cztery powtórzenia A/B w odwracanej kolejności, mediany 2–4.
HTTP: ABBA, dwa przebiegi na wariant/model. Czasy bez profilera.

## Co sprawdzono

Same ciągłe odczyty K/V (coalesced) dały mniej niż 1% zysku dla 16k,
a krótką decyzję 4.5B spowolniły o 0,91%; nie zostały domyślnym wariantem.
Włączono SW128 dla WGP f16 na Hopperze:

| Pomiar | 4.5B: poprzedni → SW128 | max: poprzedni → SW128 |
|---|---|---|
| Krótka decyzja, ms | 12,298 → 12,270 (−0,23%) | 18,644 → 18,554 (−0,48%) |
| Throughput bench, dec/s | 169,063 → 171,053 (+1,18%) | 81,989 → 82,585 (+0,73%) |
| Forward 4k, jedno pytanie, ms | 181,1 → 165,8 (−8,5%) | 336,9 → 318,2 (−5,6%) |
| Forward 16k, jedno pytanie, ms | 1179,4 → 1016,1 (−13,8%) | 2151,2 → 1894,6 (−11,9%) |
| HTTP 32 klientów, req/s | 13,627 → 15,075 (+10,6%) | 7,066 → 7,762 (+9,9%) |
| HTTP jednakowy napływ, p95 ms | 1979,8 → 1278,9 (−35,4%, 11/s) | 2751,1 → 1993,8 (−27,5%, 5/s) |
| Energia GPU/decyzję, 32 klientów | −9,5% | −8,4% |

**Regresja:** p50 4.5B przy 16/32 klientach wzrosło o 7,6%/6,3%.
Przy jednakowym napływie p50 spadło o 20,7%/11,1% dla 4.5B/max;
p95 i p99 obu modeli również spadły. Krótkie decyzje zmieniają się zbyt
mało względem rozrzutu, by deklarować istotny zysk.

## Co zweryfikowano

- Drabina 512–16384 tokenów, 1 i 5 pytań: pełne zapisane odpowiedzi
  wszystkich 14 przypadków/model identyczne między ośmioma przebiegami A/B.
- Eksporty obu modeli: single/tree/budget, bez cache prefiksu i z cache
  stanu; po 44 bench i 34 System One (30 poprawnych, cztery oczekiwane
  błędy). Wyniki identyczne bajtowo z bazą, również dla końcowego buildu.
- HTTP ABBA: 9952 żądania w 32 fazach; końcowy build: dodatkowe 2488
  żądań w ośmiu fazach. Zero błędów, różnic pełnych odpowiedzi
  i prawdopodobieństw. Odpowiedzi 400 ID/model zgodne między przebiegami.
- Końcowy build: dwa dodatkowe przebiegi drabiny bez różnic odpowiedzi;
  Nsight potwierdził uruchomienie obu domyślnych kerneli SW128.
- Fmt, Clippy i Rustdoc na macOS i CUDA/all-features, CUDA release
  dla 80/89/90, cargo-deny i ShellCheck przeszły.

Końcowy SW128 zmierzono tylko na H100 PCIe, f16, dla tych dwóch modeli.
Nie potwierdzono poprawy ani braku regresji na innych GPU. Identyczność
z bazą nie oznacza pełnej zgodności z FP32 upstream. Dwa powtórzenia HTTP
nie gwarantują SLO; rzeczywiste zegary GPU zmieniały się mimo zadanej blokady.
Pełne dane i identyfikatory buildów zachowano w lokalnym archiwum poza Gitem.
