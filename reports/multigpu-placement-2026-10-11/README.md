# Modele na osobnych GPU — 2026-10-11

basal-1.5-4.5B i mini, f16, Linux, 2× RTX 6000 Ada 48 GB (300 W),
sterownik 580.126.09, CUDA 12.9.1 i cuBLASLt 120901. Użyto przypiętych
modeli oraz dołączonych tabel GEMM niezależnych od partii. Ręczny przebieg
`tools/perf/run-multigpu-cloud.py`: **167 PASS, 0 FAIL, 0 GAP**.

## Wyniki

| Sprawdzenie | Wynik |
|---|---|
| Eksporty wobec wersji sprzed zmiany | 24 eksporty: dwa modele × tree, single i tree bez cache prefiksu × cztery warianty urządzenia. Każdy zawierał 44 bench i 34 System One. Oba pliki wyników identyczne bajtowo w 12 porównaniach nowej wersji z bazą oraz sześciu kontrolnych porównaniach bazy między kartami; `basal compare`: zero różnic |
| 4.5B na GPU 0, mini na GPU 1 | Pamięć procesu po ładowaniu: odpowiednio 9726 i 3646 MiB. Oba modele dostępne przez jeden serwer; poprawny routing, lista modeli i odrzucenie nieznanej nazwy |
| Oba modele na GPU 0 | Pamięć procesu tylko na GPU 0 (12838 MiB po ładowaniu); odpowiedzi zgodne z osobnymi serwerami |
| `CUDA_VISIBLE_DEVICES=1,0` | Fizyczne przypisanie zamienione: 3646 MiB na GPU 0, 9726 MiB na GPU 1; odpowiedzi bez zmian |
| HTTP sekwencyjnie i przy 16 klientach | 1408 żądań workloadu oraz jedno podczas zamykania. Rodzic niezależnie porównał 1233 odpowiedzi z odpowiedziami dwóch osobnych serwerów: zero różnic statusu i treści po pominięciu `usage.latency_ms`; zero błędów transportu i 5xx |
| Harmonogramy | Oba modele użyły głównego i długiego toru, zarówno na osobnych GPU, jak i na wspólnej karcie |
| Zamykanie | Wszystkie pięć serwerów zakończyło pracę kodem 0 i zwolniło pamięć procesu. Długie żądanie trwało 2,56 s; SIGTERM wysłano około 0,5 s po jego rozpoczęciu, odpowiedź miała status 200 |
| Diagnostyka i błędna konfiguracja | Doctor poprawnie wyliczył widoczne urządzenia, także po zmianie kolejności i ukryciu jednej karty. Nieistniejące urządzenie, zduplikowany model i `--device` z `--config` zostały odrzucone przed uruchomieniem serwera |

W fazie równoległej na osobnych GPU obie karty miały niezerową aktywność
w 29 z 96 próbek telemetrii pobieranych co 200 ms. To obserwacja aktywności
obu kart, bez deklaracji poprawy przepustowości. Nie wykonano powtarzanego
pomiaru A/B/B/A; kolejność mieszanych żądań różniła się między przebiegami.

Odrzucenia HTTP były zgodne z serwerami referencyjnymi: błędne instrukcje
i brak typu pytania, a dla mini także cztery długie żądania przekraczające
8192 tokeny kontekstu. Te odpowiedzi 400/422 nie są udaną inferencją długiego
dokumentu. Łącznie uzyskano 1345 odpowiedzi 200 i 64 takie odrzucenia.

## Zakres weryfikacji

Baza: commit `87da7867657dd052ffa9aac5f7c40cbf348fa6f6`. Kandydat:
ten commit z niezatwierdzonymi zmianami przypisania urządzeń i harmonogramów.
Obie binarki zbudowano tym samym Rust 1.95.0, GCC 11 i CUDA 12.9.1,
z PTX 80/89/90. Suma SHA-256 diffu Rust (`git diff HEAD -- crates`):
`d7422df254bfa644484e61a8db6d9ac10b653de517faa9ada31b2f416b1eb899`.

SHA-256 binarek:

- baza: `6e45d9db4b5b7c82b4ceaf303b6e74aec1d351947f64829377da186ce1fd5e13`;
- kandydat: `f8528cc92314abddfed4269b93cf71c42d5833a0bf256f41481a921d60ab2006`.

Rewizje modeli: 4.5B `784a683bfadcc8865238fc0fc74a83b4000269bc`,
mini `1978d0705ce09cd2d7d8d3e87b304468121e7255`.
Wejścia: bench i System One z `reference-basal-1.5-4.5B-fp32` oraz
`tools/bench/long_states.jsonl`. Porównano wyniki Rust sprzed i po zmianie;
nie wykonano ponownego eksportu FP32 upstream.

Fmt, Clippy i Rustdoc przeszły na Linux bez GPU, z Intel oraz z CUDA
(wszystkie features, PTX 80/89/90). Kontrola zależności, wyjątku `paste`,
workflow i składni runnera przeszła. Nie uruchamiano `cargo test` ani
nie dodawano testów lub bramek CI.

Pomiar obejmuje dwa identyczne GPU i dwa modele w f16. Nie weryfikuje
różnych architektur kart w jednym procesie, modelu max, bf16/f32, runtime'u
Intel ani kompilacji/runtime'u Metal po zmianie. Repliki, podział modelu
i kwantyzacja nie należą do tego etapu. Surowe dane zachowano poza Gitem;
wynajętą stację usunięto po pobraniu danych.
