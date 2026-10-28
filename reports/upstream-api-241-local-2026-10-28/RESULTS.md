# Czasy i uzupełniające obserwacje

## Podstawowy pomiar ABBAABBA

Oba audyty: **AUDITED, 0 issues**. Każdy zachował trzy powtórzenia na wariant.
Wszystkie mierzone odpowiedzi były poprawne i miały niezmienione answers.
To kwalifikacja wykonania pomiaru; próg FP32 pozostaje osobnym wynikiem.

Poniżej mediana p50/p95 z trzech zachowanych powtórzeń, w ms. Każdy krótki
dokument ma 48 próbek na client/wariant; długi ma 30. Identyfikatory krótkich
przypadków zachowują prefiks `pico-` także przy Mini, ponieważ pochodzą ze
wspólnego zamrożonego korpusu. Nazwa modelu w body jest podstawiana przez runner.

| Pico, dokument | C | p50 baza → finał | p95 baza → finał | Δ p50 |
|---|---:|---:|---:|---:|
| Sześć typów | 1 | 107,790 → 107,789 | 108,167 → 108,270 | −0,001% |
| Wspólny prefiks | 1 | 74,435 → 74,446 | 74,746 → 74,733 | +0,015% |
| 27 opcji | 1 | 26,971 → 26,976 | 27,208 → 27,175 | +0,017% |
| Unicode | 1 | 30,176 → 30,181 | 30,272 → 30,251 | +0,018% |
| Sześć typów | 4 | 239,580 → 239,730 | 239,964 → 240,501 | +0,063% |
| Wspólny prefiks | 4 | 239,856 → 239,852 | 240,563 → 240,620 | −0,002% |
| 27 opcji | 4 | 239,498 → 239,651 | 240,300 → 240,292 | +0,064% |
| Unicode | 4 | 239,254 → 239,399 | 239,912 → 240,321 | +0,060% |
| 5080 tokenów | 1 | 613,293 → 613,749 | 614,082 → 614,247 | +0,074% |

Pico: QPS krótkiego korpusu C1 16,688 → 16,676; C4 16,683 → 16,671.
Najwyższy p95 pojedynczego zachowanego powtórzenia sześciu typów/C4 wzrósł
do 271,968 ms, przy medianie p95 240,501 ms. Ten ogon pozostaje w danych;
agregat go nie usuwa. Zakres p50 długiego dokumentu: baza 612,771–613,336 ms,
finał 613,374–613,841 ms.

| Mini, dokument | C | p50 baza → finał | p95 baza → finał | Δ p50 |
|---|---:|---:|---:|---:|
| Sześć typów | 1 | 53,156 → 53,414 | 54,975 → 55,312 | +0,486% |
| Wspólny prefiks | 1 | 39,407 → 39,562 | 40,499 → 41,024 | +0,392% |
| 27 opcji | 1 | 39,572 → 39,691 | 40,424 → 40,902 | +0,301% |
| Unicode | 1 | 23,420 → 23,459 | 24,306 → 24,315 | +0,170% |
| Sześć typów | 4 | 146,026 → 145,825 | 149,361 → 147,312 | −0,137% |
| Wspólny prefiks | 4 | 145,506 → 145,509 | 149,730 → 147,722 | +0,002% |
| 27 opcji | 4 | 145,437 → 145,334 | 150,141 → 147,978 | −0,071% |
| Unicode | 4 | 145,031 → 145,216 | 149,669 → 147,744 | +0,128% |
| 4130 tokenów | 1 | 664,091 → 664,497 | 666,545 → 667,558 | +0,061% |

Mini: QPS C1 25,647 → 25,567; C4 27,377 → 27,419. Zakres p50 długiego
dokumentu: baza 663,230–664,395 ms, finał 663,395–664,585 ms. W krótkim C1
finał jest wolniejszy o 0,04–0,26 ms. Wyniki zachowują te różnice; kampania
nie dowodzi przyspieszenia ani ich konkretnej przyczyny. Nie wyprowadzamy
wniosku o równości wydajności z samej zgodności answers.

Dowód: `performance/audit-pico.json`, `audit-mini.json` oraz odpowiadające
`method.json`, `progress.jsonl` i `*-observations.jsonl`.

## Osobny pomiar 255 opcji

Oba audyty: **AUDITED, 0 issues**, bez zmiany answers. Po 48 zachowanych próbek
na komórkę, trzy powtórzenia; wejście ma 1997 tokenów Pico i 2278 Mini.

| Model | C | p50 baza → finał [ms] | p95 baza → finał [ms] | Δ p50 |
|---|---:|---:|---:|---:|
| Pico | 1 | 259,194 → 258,577 | 262,612 → 261,343 | −0,238% |
| Pico | 4 | 1036,492 → 1036,247 | 1042,886 → 1041,300 | −0,024% |
| Mini | 1 | 324,661 → 324,195 | 327,276 → 326,953 | −0,144% |
| Mini | 4 | 650,853 → 650,700 | 654,062 → 654,400 | −0,024% |

Dowód: `performance/audit-{pico,mini}-options255.json` i osobne katalogi
`*-options255-abba-01`. Te liczby nie są czasami Groups: format 2 obsługuje
255 opcji bez podziału na rundy formatu 1.

Telemetria całej sekwencji, 3747 próbek co około 200 ms: temperatura 40–72°C,
SM clock 210–2805 MHz, utilization 0–100%, pamięć urządzenia 2–3686 MiB.
Zakres obejmuje również przerwy i starty, nie jest zakresem pojedynczej fazy.
Odczyt mocy zwracał N/A, więc **GAP dla mocy**. Źródło: `performance/telemetry.csv`,
poprawnie nazwane kolumny i null dla mocy: `telemetry-ranges-v2.json`.
Wcześniejszy nienazwany skrót `telemetry-ranges.json` pomijał pole N/A i nie jest
używany do interpretacji kolumn.

## Orders auto i efektywne ustawienia

Na obu oryginalnych modelach, przy wyłączonej kalibracji dla izolacji samego
gate, wykonano dwupytaniowe wejście Unicode. Auto z tau=1 i band=[0,1] daje
identyczne całe answers jak stałe orders=2; auto z tau=0 i band=[0,0] daje
identyczne answers jak orders=1. `second_order_branches` wynosi odpowiednio 2 i 0.
Pico używa 512/256 tokenów, Mini 578/289. Nie maskowano pól answers.
Dowód: `adaptive/{pico,mini}-comparison.json` i wejścia z jawnym modelem.
Pierwszy CLI request bez `model` odrzucono przed inferencją odpowiedzi;
zachowano go jako nieudaną próbę, bez przypisywania PASS.

Istniejący `basal benchmark` wykonał osobny pojedynczy request na model z
`layout=qfirst`, `orders=auto`, `tau=0.7`, `band=[0.1,0.9]`, `overflow=truncate`,
`truncate_head=0.25`. Oba raporty mają `complete=true`, status completed i dokładnie
te ustawienia. Emitowane hashe kalibracji zgadzają się z plikami:

| Model | SHA-256 CALIBRATION.json |
|---|---|
| Pico | `37271567a03f819ea9529cf5930880ece5b6f03b858847cc3eeb6dc38792544b` |
| Mini | `971cea10874d2961ce6aa5c78dfa3405ce49b609f5b71e9be82ec78bc73e323e` |

Dowód: `metadata/{pico,mini}-benchmark.json`, `effective-settings-check.json`
i `effective-settings-verdict.txt`. Jest to obserwacja metadanych, nie kolejny
porównywalny pomiar czasu — pojedynczy request nie miał warmupu.

## Fingerprint i ograniczenie Batch API

Mini z qfirst/auto/truncate: poczekano na `profiles_ready=true` i działającą
bazę, odczytano discovery oraz wykonano poprawne żądanie online. Discovery
podaje qfirst/auto i ostrzeżenie z nazwą `CALIBRATION.json`, bez lokalnej ścieżki.
Zmiana **wyłącznie tau 0,7 → 0,8** przy tych samych modelu, kalibracji, GEMM,
dtype i pozostałych ustawieniach zmienia fingerprint:

| Tau | Fingerprint |
|---|---|
| 0,7 | `fp1-3942021d6d4438ecf097b5295247f4f75d91db11bdd3b1e92fdac88b5056f591` |
| 0,8 | `fp1-674ff3da1ef82e67bb84fd094fa3918a24379f0f1f0deb61dd02c9b485a5ef56` |

Planner: `basal-planner.5`. Oba profile mają `qualified=false`, reason
`background disabled in batch_profile`, skonfigurowane lanes=0. **Ready i
fingerprint nie dowodzą kwalifikacji wykonania w tle.** Nie wytwarzano sztucznego
profilu budżetu urządzenia. Dowód: `metadata/mini-batches-tau{07,08}.json`,
`fingerprint-comparison.json`, `mini-models-tau07.json`, `mini-decision-tau07.json`.

Pico z `--batches` kończy start błędem offline planner:
`bos token  is not in the vocabulary`. Powtórzenie na binarce oryginalnej bazy
`729fe47` daje **ten sam błąd i exit 1**. Qwen Pico celowo ma pusty BOS,
a offline planner go sprawdza. Ograniczenie istniało przed inicjatywą;
nie naprawiano go przy kwalifikacji API 2.4.1. Dodatnie pokrycie Batch API
Pico pozostaje GAP. Logi obu startów są w `evidence/private/serve-pico-*-batches-01.log`
oraz `serve-pico-metadata-batches-01.log`.

Po zakończeniu wszystkie serwery i telemetria zostały zatrzymane. Nie pozostał
proces obliczeniowy GPU ani listener kampanii. Checkouty upstream i bazy są czyste.

## Integralność lokalnych danych

Archiwum: `.cache/measurement-artifacts/upstream-api-241-local-2026-10-28/measurement-data.tar.gz`.
Rozmiar: **2 934 990 B**. SHA-256:
`6c34fe763abbacd842452119108fbb08c8eae2ad2b5a09ec8b483b42cceebf98`.

Manifest `SHA256SUMS` obejmuje **1421 regularnych plików**. Wszystkie sumy
sprawdzono w źródle oraz po skopiowaniu archiwum do osobnego katalogu,
sprawdzeniu SHA-256 i rozpakowaniu: **1421/1421 PASS w obu kopiach**.
Symlinki fixture wskazują lokalne modele; wagi ani binarki aplikacji nie są
w archiwum. Dane mają logiczne ścieżki `reports/upstream-api-241-local-2026-10-28/`.
Wewnątrz są również nieudane próby, surowe logi, dokładny diff implementacji
i prywatny helper. Surowe logi zawierają lokalne ścieżki i nie są gotowe do publikacji.

Nie opublikowano URI HF ani nie wykonano transferu do bucketa. Lokalna
weryfikacja kopii nie jest potwierdzeniem pobrania z HF lub dostępu anonimowego.
