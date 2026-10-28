# Odtworzenie diagnostyk trained paths

Archiwum pozostaje lokalne i prywatne. Nie ma publicznego URI ani automatycznego
pobierania. Zawiera surowe ścieżki sesji i ślady Nsight; przed publikacją wymaga
osobnego przygotowania. Nie zawiera checkpointów, binarki ani środowisk Python.
Wersje źródeł, przypięte rewizje modeli i hash binarki podaje [README](README.md).

## Weryfikacja i audyt bez modelu

Z katalogu repozytorium wskaż lokalne archiwum i nowy katalog rozpakowania.
SHA-256 porównaj z końcową notą w README w repozytorium przed rozpakowaniem.
Kopia README wewnątrz archiwum poprzedza dopisanie tej noty.

```sh
BASAL_ARCHIVE=.cache/measurement-artifacts/upstream-api-241-trained-2026-10-28/measurement-data.tar.gz
BASAL_REPLAY=.cache/measurement-replay/upstream-api-241-trained-2026-10-28
sha256sum "$BASAL_ARCHIVE"
mkdir -p .cache/measurement-replay
mkdir "$BASAL_REPLAY" && tar -xzf "$BASAL_ARCHIVE" -C "$BASAL_REPLAY"
(cd "$BASAL_REPLAY" && sha256sum -c SHA256SUMS)
BASAL_DATA="$BASAL_REPLAY/reports/upstream-api-241-trained-2026-10-28"
python3 "$BASAL_DATA/source/audit.py" --measurement-artifacts "$BASAL_DATA"
```

Audyt korzysta tylko ze standardowej biblioteki Python. Nie uruchamia modelu,
HTTP ani sieci. Zapisuje ponownie `audit-summary.json`; po audycie można
porównać jego hash z `SHA256SUMS`. Próg wynosi 1e-4. W tabeli `native`
liczone są asercje runnera, a w `reference` porównane przypadki; tych dwóch
liczników nie należy sumować jako liczby żądań. Zapisane błędy porównań
pozostają wynikami negatywnymi, mimo że sam audyt kończy się kodem 0.

## Układ danych

Wszystkie poniższe ścieżki są względem `reports/upstream-api-241-trained-2026-10-28/`.

| Ścieżka | Zawartość |
|---|---|
| `exit-plain-f32-01/`, `exit-heads-f32-01/`, `upstream-exit-f32-02/` | Plain/off, polityki, współbieżność, odpowiedzi i ślady głębokości referencji |
| `think-plain-f32-01/`, `think-f32-256-01/`, `think-context-f32-01/` | P1/P2, generacja, walidacja, limit kontekstu i dwa FAIL bitowe |
| `upstream-think-torch-f32-02/`, `upstream-think-vllm-01/`, `think-bf16-256-01/` | Obie referencje, prompty, logity/logprobs i kontrola dtype |
| `think-plain-control-f32-01/`, `lifecycle-heads-01/`, `lifecycle-think-01/` | Kontrola ordinary batching, rozłączenie i SIGTERM |
| `native-exit-profile-01/`, `native-exit-profile-02/` | Oryginalne captures Nsight, SQLite, statystyki i kody zakończenia |
| `source/`, `inputs/`, `evidence/` | Dokładne runnery, archiwum upstream, wejścia, pakiety, próby nieudane i przywrócenie usługi |

Pierwsze nieudane próby referencji (`upstream-exit-f32-01/` i
`upstream-think-torch-f32-01/`) są zachowane; nie są liczone jako akceptacja.
`source/upstream-source.tar.gz` zawiera niezmienione pliki upstream z
`b20f0f9b159f71726459aa98ede1356ab65695fd`. Runnery są prywatną diagnostyką,
bez frameworka testowego i bez zmiany źródeł upstream.

## Metoda porównań

Early-exit plain/off i mieszane polityki porównują pełne JSON bodies po
usunięciu wyłącznie `usage.latency_ms`. Thinking usuwa dodatkowo
`answers.*.think.ms`. Przy porównaniu plain z serwerem zdolnym do thinking
usuwane są wyłącznie addytywne `answers.*.meta.think`, `usage.think_questions`
i puste `meta`. Prawdopodobieństwa zawsze pozostają w porównaniu.

Referencje porównują wszystkie liście `probabilities`, w tym `no_think`,
oraz zgodność ich ścieżek. Notatki porównują tekst i liczbę tokenów; surowe
native ID nie są dostępne. Kontrola native BF16 względem vLLM porównuje
tylko P2 oraz tekst/liczbę tokenów, bo bazowe P1 referencji pozostało FP32.
Argmax w tym audycie oznacza maksimum rozkładu, nie wszystkie pola decyzji
specyficzne dla act/tool.

W `native-exit-profile-02/*-cuda_gpu_kern_sum.csv` policz instancje
`usilu_f32` i `ampere_sgemm_128x64_tn`. Dają odpowiednio 60/300 dla off,
30/150 dla 0.98 oraz 50/250 dla 0.999. Druga próba korzysta z tego samego
udanego off capture co pierwsza; nie jest to drugi niezależny pomiar off.

## Ponowna inferencja

To osobna czynność wymagająca dostępnej pamięci i zgody na pomiary w docelowym
środowisku. Poniższe polecenia dokumentują wykonaną serię; nie są wykonywane
przez audyt. Nie należy zatrzymywać usług ani odtwarzać dawnych adresów sesji.

Przygotuj nowy prywatny katalog runnerów. Umieść w nim cztery pliki z `source/`,
zweryfikowaną binarkę jako `basal`, źródła upstream jako `upstream/`,
przypiętą parę modeli w `models/basal-2.0-4.5B/` i środowiska
`reference-venv/` oraz `vllm-venv/`. Model early-exit ma w oryginalnych runnerach
stałą ścieżkę kontenerową; dostosowanie do nowej maszyny wymaga zapisania
zmiany jako nowej próby. Nie kopiuj produkcyjnej konfiguracji usługi.

Pełne argumenty native i środowisko są w `command.json` każdego wariantu.
Manifesty i `evidence/remote-logs/*-packages.txt` zapisują zależności.
Proces referencji używa Torch 2.11.0 / Transformers 5.17.0; osobny sidecar
używa vLLM 0.26.0 / Torch 2.11.0. Pole `versions.vllm=null` w manifeście
procesu nadrzędnego jest prawidłowe: vLLM działa w osobnym środowisku.
Flagi TF32 zapisane dla rodzica nie dowodzą ustawień procesu sidecara.

Podstawowe wywołania z nowego katalogu runnerów:

```sh
reference-venv/bin/python native_probe.py plain-exit --tag exit-plain-f32-01
reference-venv/bin/python native_probe.py heads --tag exit-heads-f32-01
reference-venv/bin/python reference_probe.py exit --native exit-heads-f32-01 --tag upstream-exit-f32-02
reference-venv/bin/python native_probe.py think-off --tag think-plain-f32-01
reference-venv/bin/python native_probe.py think --tokens 256 --max-len 2048 --tag think-f32-256-01
reference-venv/bin/python reference_probe.py think-torch --native think-f32-256-01 --tokens 256 --max-len 2048 --tag upstream-think-torch-f32-02
reference-venv/bin/python reference_probe.py think-vllm --native think-f32-256-01 --tokens 256 --max-len 2048 --tag upstream-think-vllm-01
reference-venv/bin/python native_probe.py think-context --tokens 32 --max-len 64 --tag think-context-f32-01
reference-venv/bin/python native_probe.py think-control --tag think-plain-control-f32-01
reference-venv/bin/python native_probe.py think --dtype bf16 --tokens 256 --max-len 2048 --tag think-bf16-256-01
reference-venv/bin/python lifecycle_probe.py heads
reference-venv/bin/python lifecycle_probe.py think
```

Uruchamiaj warianty sekwencyjnie; nie łącz ich przez warunkowe `&&`, jeśli
celem jest zachowanie także kolejnych obserwacji po negatywnym porównaniu.
`native_probe.py` kończy się kodem 1 przy dwóch zachowanych FAIL FP32;
odpowiedzi HTTP są wtedy poprawne i proces serwera kończy się kodem 0.
Runnery odmawiają nadpisania istniejących katalogów.

`profile_exit.py` drugiej próby wymaga istniejącego udanego off capture
z pierwszej próby i oryginalnej ścieżki Nsight. Dokładne komendy capture
i stats są zapisane pod `native-exit-profile-02/`; do nowego pomiaru należy
wykonać wszystkie trzy capture od nowa i użyć nowych katalogów, zamiast
przedstawiać ponowne przetworzenie starego capture jako nową inferencję.
