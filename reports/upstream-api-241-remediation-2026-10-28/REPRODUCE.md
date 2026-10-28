# Odtworzenie i dowody

Nowe dane są w
`.cache/measurement-artifacts/upstream-api-241-remediation-2026-10-28/`.
Logiczny katalog danych w archiwum:
`reports/upstream-api-241-remediation-2026-10-28/`.
Archiwum `measurement-data.tar.gz`: **102 632 002 B**, SHA-256
`1d944f98ff35a7bc5636f3a299017c8e8308ddefd3ef70e9f7b72757fdeb7d1a`.
Manifest obejmuje **323 pliki**; źródło i świeża osobna kopia po rozpakowaniu
przeszły **323/323** kontroli SHA-256. SHA samego archiwum także jest zgodny
w obu kopiach. Nie ma opublikowanego URI HF; nie wykonano wysyłania ani
pobrania z bucketa. Lokalną kopię zachowano.
Surowe logi i konfiguracje zawierają lokalne ścieżki i nie są gotowe do publikacji.
Kopie README/REPRODUCE w archiwum pochodzą sprzed dopisania tej noty
z jego hashem. Puste `parent-checksums.log` pakietu CPU jest zachowaną
nieudaną pierwszą próbą przed utworzeniem jego manifestu; udana kontrola
111/111 jest w `evidence/d-parent-checksums.log`.

Wymagane wejścia z poprzedniego pakietu:
`reports/upstream-api-241-local-2026-10-28/inputs/` oraz
`exports/final-pico-{f32,qfirst-f32,state_twice-f32}-o1/`.
Archiwum poprzedniej kampanii i jego SHA podaje jej raport. Wagi i binarka
inferencji nie są redystrybuowane. Pakiet CPU zawiera własne programy
diagnostyczne i archiwa źródeł obu wersji. Podstaw lokalne ścieżki modeli i kompilacji; nie
modyfikuj checkoutu upstream ani ukończonych katalogów wyników.

## Porównania bez nowej inferencji

1. Zweryfikuj SHA archiwum, rozpakuj do nowego katalogu i uruchom
   `sha256sum -c SHA256SUMS` z jego korzenia.
2. Porównaj `exports/fixed-pico-f32-o1/{bench,systemone}.jsonl` przez `cmp`
   z poprzednim `exports/final-pico-f32-o1/`. Oba pliki muszą być identyczne.
3. Powtórz `basal compare --a REF --b NATIVE --out NOWY.json` dla
   canonical, qfirst i state_twice. Dokładne wywołania i JSON są w
   `evidence/private/` oraz `comparisons/`. Próg cal_prob_diff.max to 0,0001.
4. W `background/mini-batch-output.jsonl`, `mini-mixed-output.jsonl`
   i `mini-restart-output.jsonl` dopasuj `custom_id` do sześciu
   `background/online-*.json`; w powtórzeniach usuń końcowe `-N` z ID.
   Porównaj całą `response.body` po usunięciu wyłącznie `usage.latency_ms`
   z tak samo znormalizowaną odpowiedzią online. Jest 6 + 16 + 60 wyników.
5. Sprawdź końcowe statusy i liczniki w `evidence/private/mini-*.json`.
   Kolejność: normalny batch completed6, mixed cancelled16/60, restart
   in_progress11/60 przed shutdown i completed60/60 po wznowieniu.

`background/comparison-*.json` zachowuje sumaryczne porównania 16 + 60,
a `comparison-verdict.json` pierwszy batch i online w ruchu mieszanym.
Normalizacja nie usuwa answers, usage tokens, lane ani pól opcjonalnych.
JSON porównano po sortowaniu kluczy, więc zgodność dotyczy wartości,
nie kolejności kluczy czy surowych bajtów odpowiedzi HTTP.

## Ponowna inferencja

Użyj nowego pustego katalogu wyników i dokładnych źródeł z `source/`.
Referencja Qwen F32 wymaga oryginalnego środowiska z FLA; domyślna polityka
eksportera musi wybrać `torch`. Manifest powinien pokazać FLA jako default,
a funkcje Transformers jako used. `--qwen-delta auto` odtwarza poprzednią
ścieżkę, której wyniki nie były ścisłą referencją FP32.

Pico startuje z parametrami H1 zapisanymi w logu serwera i metodzie QA.
Oczekuj profilu gotowego, lecz `validates=false/background_supported=false`.
Nie traktuj oczekiwania batcha w `validating` jako ukończonej walidacji.

Mini bootstrap ma odtwarzalny `background/mini-workload.jsonl` i pełne
polecenie `basal benchmark` w logu. `mini-background-config.json` w dowodach
to eksperymentalna konfiguracja tej karty i tego korpusu, z wyjaśnionymi
marginesami. Nie jest szablonem do automatycznego wdrożenia na innym GPU.
Wskaźnik used z prób alokatora nie mierzy fizycznego narzutu zaokrągleń.

Surowa metoda, źródło próbnika CUDA, snapshoty health, statusy batchy,
komendy/wyjścia referencji, nieudane przygotowania i kontrole statyczne
są zachowane w `evidence/`. Pomiary nie wymagają nowej binarki dla samych
zmian Python/dokumentacji. Upstream, stare raporty i dane pozostają niezmienione.

## Budżety CPU

`evidence/gap-d-bounds/HANDOFF.md` podaje pełne polecenia odtworzenia,
wersje, ograniczenia i opis prywatnego programu. Najpierw sprawdź jego
111 sum w `SHA256SUMS`. Następnie porównaj bajtowo
`final-current-default.jsonl` i `final-current-refuse.jsonl`.
`final-comparison.json` obejmuje 3915 par prób z 243 kombinacji.
Przy porównaniu z bazą nie usuwaj różnic estymatorów z podsumowania:
retained +200 B, host +200/+400 B i task exec +200 B są rzeczywiste.
Źródła programu wywołują publiczne API każdej osobno przypiętej wersji;
nie odtwarzają równań estymatorów. Zestaw nie uruchamia GPU ani puli
wykonania serwera i nie jest kwalifikacją granic fizycznej pamięci.
