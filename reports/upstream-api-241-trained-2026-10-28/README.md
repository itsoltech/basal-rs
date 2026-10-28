# Basal 2.4.1: trained thinking i early-exit na RTX 6000 Ada

Pomiar z 2026-10-28 rozszerza [lokalną naprawę Pico i QA Mini](../upstream-api-241-remediation-2026-10-28/README.md)
o rzeczywiste opublikowane checkpointy 4.5B. Dane pozostają lokalne; nie
publikowano ich ani nie wdrażano nowej binarki do istniejącej usługi.

**Wynik:** early-exit FP32 przechodzi skończone porównanie z upstream,
a profil potwierdza pomijanie warstw. Thinking FP32 zgadza się z diagnostyczną
referencją PyTorch FP32, ale **nie zachowuje parytetu z oryginalnym sidecarem
vLLM BF16**. Pozostaje również FAIL ścisłej równości P1 przy zmianie rozmiaru
batcha w badanej konfiguracji. Nie zmieniano kodu ani numeryki aplikacji.

## Wersje i metoda

Kod aplikacji pochodzi z `bc5ad0aa3a8070d22e33ed6ab1db0084d97abe70`;
późniejsze zmiany przed pomiarem dotyczą eksportera Python i dokumentacji.
SHA-256 użytej binarki CUDA:
`e6d4acb3efbcd444c89756561f47716b4e71dad303368a469801455e2afd9331`.
Referencja to niezmienione źródła upstream
`b20f0f9b159f71726459aa98ede1356ab65695fd` (2.4.1).

Sprzęt: NVIDIA RTX 6000 Ada Generation, 49 140 MiB według `nvidia-smi`,
SM 8.9, sterownik 580.159.03, limit mocy 300 W. Natywny runtime CUDA 12.9.1.
Na czas obliczeń zatrzymywano istniejącą usługę za zgodą operatora.
W danej chwili działał jeden badany silnik; procesy sidecara vLLM i jego
bazowego silnika upstream stanowią razem jeden badany wariant.

Native: jawne `dtype=f32`, `readout=f32`, `inference-mode=accurate`, orders 1,
bez prefix/state cache, bez long lane, `gemm-table=none`. F32 przechowuje
wagi BF16 i rozszerza je dokładnie do F32 podczas użycia; nie jest to
niekwantyzowany model przechowywany stale w F32. Early-exit używa `kernels=candle`;
thinking używa zwykłego fused. Nie kwalifikuje to domyślnego F16 ani innych
konfiguracji. Diagnostyczne `gemm-table=none` nie jest domyślną tabelą
zapewniającą niezależność od rozmiaru partii.

Korpus obejmuje polskie choice/noul/score/act/tool, multi, nieznany typ
`rank`, trzy pytania w jednym żądaniu oraz wybór z Unicode. Surowe requesty,
odpowiedzi, nagłówki, komendy i dzienniki są zachowane. Próg porównania
prawdopodobieństw wynosi 0,0001; nie zmieniano go po pomiarze.

## Early-exit

Użyto `Remek/basal-1.0-4.5B@b95288041cb1975bb930c6f0410819273fd2f54f`.
Zweryfikowano SHA-256 wag, `heads.pt` i `thresholds.json` zgodnie z
[kontraktem loadera](../../docs/EARLY_EXIT.md). Rzeczywiste głowice załadował
natywny loader Candle; wszystkie cztery aktywne polityki były wykonywane.

Pierwsza seria daje 41/41 poprawnych kontroli: statusy, cztery polityki,
off/null, odmowę nieznanej polityki i osiem wyników współbieżnych równych
osobnym wywołaniom po usunięciu wyłącznie `usage.latency_ms`.
Wariant plain daje 9/9 odpowiedzi. Jego dziewięć pełnych ciał odpowiedzi
jest identycznych z head-capable/off po tej samej normalizacji czasu.
Oba natywne procesy zakończyły się kodem 0.

Porównanie z upstream: **32/32 przypadki** mają te same klucze rozkładów
i różnicę nie większą niż **0,000004112071362083469**. Referencja używa
Torch 2.11.0, Transformers 5.17.0, FP32 i wyłączonego TF32 w swoim procesie.
Prywatny adapter wyłącza kompilację i odracza inicjalne tworzenie wszystkich
CUDA graphs; oryginalne metody upstream wykonują faktyczne packing,
runtime CUDA graphs, głowice i gate dla użytych kształtów. Źródła upstream
nie są edytowane. To jawne ograniczenie zakresu wobec pełnego `fast-exit`.

Dla 20 przypadków z jawną polityką upstream odnotował: `0.98` — jeden
chunk na głębokości 30 i trzy na 50; `0.99` i `0.995` — po cztery na 50;
`0.999` — trzy na 50 i jeden na 60; `off` — cztery na 60.
Sam zgodny wynik i te liczniki referencji nie dowodzą jeszcze głębokości
natywnego wykonania. Potwierdza ją osobny profil Nsight Systems 2025.1.3
dla tego samego jednopytaniowego wejścia, z załadowanymi głowicami także
przy off, bez warmup forwardów i bez cache:

| Polityka | Wywołania `usilu_f32` | Wywołania `ampere_sgemm_128x64_tn` | Warstwy |
|---|---:|---:|---:|
| off | 60 | 300 | 60 |
| 0.98 | 30 | 150 | 30 |
| 0.999 | 50 | 250 | 50 |

Każdy decoder MLP wykonuje jedną SiLU; adapter głowicy nie używa SiLU.
Liczby zgadzają się z głębokościami upstream dla tego pytania i potwierdzają
brak wykonania tail w tych dwóch aktywnych przypadkach. Nie kwalifikują
każdego kształtu ani przyspieszenia; czasy instrumentowanego profilu nie są
benchmarkiem wydajności. Capture off z pierwszej próby został ponownie
użyty bez nowej inferencji w drugiej próbie przetwarzania statystyk.

## Thinking

Użyto pary `Remek/basal-2.0-4.5B@e5ba9149760d99b73b1963d8e0ccf948b58afacb`,
root i `think/`. Oba pliki po 9 514 629 496 B mają oczekiwane hashe:
root `481d8ea4972379a611f46784196e407bee239a37c3981c6e29ff1aaaeb0f19c2`,
think `1e295ab2bcc6e812bc9e22f710cd0a6fc4d91371eefffcb271d77cb6ad97c8a5`.
Użyto opublikowanej osobnej `CALIBRATION.think.json`, limitu 256 tokenów
i context cap 2048. Nie zmieniano kalibracji, wag ani EOS.

Wymuszone thinking działa dla choice/noul/score/act/tool i Unicode, a dla
grupy wykonuje trzy P2. Multi i nieznany typ nie generują. Pojedyncze
wykonania wygenerowały odpowiednio 47, 51, 73, 65, 53 i 53 tokeny;
grupa 171. Dwa powtórzenia choice są identyczne po pominięciu
`usage.latency_ms` oraz `answers.*.think.ms`. Walidacja null i błędnego
`think_text` zwraca 422, alias HTTP zachowuje tę samą odpowiedź.

Dziewięć odpowiedzi off zachowuje dane serwera bez thinking. W porównaniu
usunięto czas oraz wyłącznie dodatkowe metadane włączonej funkcji:
`answers.*.meta.think` i `usage.think_questions`, wraz z pustym obiektem
`meta`. Nie usuwano prawdopodobieństw, liczników wejścia ani innych danych.

### Zachowany FAIL porównania bitowego

Seria thinking ma 38/40 poprawnych kontroli. Dwie niezgodności dotyczą
pełnych ciał odpowiedzi w jednym batchu: off i force. Jedyną różnicą
poza czasem są dwa małe prawdopodobieństwa P1:

| Klasa | Osobno | W batchu | Różnica |
|---|---:|---:|---:|
| card | 0,007339949253946543 | 0,007339959964156151 | 0,000000010710209608078003 |
| loan | 0,007021189667284489 | 0,007021199446171522 | 0,000000009778887033462524 |

Dla force różnica występuje wyłącznie w `think.no_think.probabilities`.
Końcowy P2, tekst, liczba tokenów i decyzja są identyczne. Dla off są to
końcowe prawdopodobieństwa. Obie odpowiedzi mają 200, a nagłówek potwierdza
batch dwóch żądań. Zapis pozostaje **FAIL ścisłej równości**, mimo wielkości
znacznie poniżej progu FP32. Nie jest to dowód wycieku polityki ani
regresji wywołanej włączeniem thinking.

Kontrola z całkowicie wyłączonym thinking odtwarza dokładnie te same P1:
z 16 równoległych odpowiedzi jedna została wykonana w batchu 1 i zgadza się
z solo, a 15 w batchu 15 ma wartości z powyższej tabeli. Jej 17/17 kontroli
statusu przechodzi; nie oznacza to 17 kontroli równości. Współdzielenie
identycznych promptów pozwoliło spakować taki batch mimo cap 216 tokenów.
Różnica występuje więc również w zwykłej ścieżce badanego modelu i ustawień.

### Porównanie z PyTorch FP32

**19/19 przypadków PASS**, maksimum różnicy prawdopodobieństwa
**0,0000011026859283447266**. Klucze rozkładów są zgodne. We wszystkich
dziewięciu faktycznie wygenerowanych notatkach zgadzają się tekst i liczba
tokenów, także w trzyczęściowym żądaniu. Natywne surowe ID generacji nie są
eksportowane, więc nie jest to porównanie sekwencji ID.

Jest to jawny adapter diagnostyczny: niezmieniony upstream `ThinkServer`
korzysta z zastępczego sidecara PyTorch FP32, full-prefix greedy bez KV cache,
z opublikowanymi EOS i stop string. Adapter zapisuje prompty P2, logity oraz
ID referencji. Pozostałe routowanie i kalibracja należą do upstream.
To potwierdza badaną ścieżkę FP32; nie zastępuje porównania z vLLM.

Osobny context guard (`think_max_len=64`, `think_tokens=32`) daje 16/16
poprawnych kontroli. Wymuszone choice jest pominięte z `skipped=context`,
bez generacji, z zachowaniem P1; proces kończy się kodem 0.

### Zachowany FAIL parytetu z vLLM

Oryginalny upstream Sidecar uruchomiono z vLLM 0.26.0, Torch 2.11.0,
BF16, deterministycznym trybem upstream, limitem kontekstu 2048 i bez
prefix cache. Bazowy silnik upstream pozostał FP32. Żądania, prompty,
odpowiedzi sidecara i manifest pakietów są zapisane. Wybór BF16 sidecara
pochodzi z upstream; nie jest pełną referencją PyTorch FP32.

Porównanie z native FP32 ma **7/19 przypadków powyżej 0,0001**, z maksimum
**0,24679961800575256**. Nie ma błędów wykonania ani niezgodnych kluczy.
Wśród dziewięciu wygenerowanych notatek tekst i liczba tokenów zgadzają
się w **4/9**, a argmax końcowego rozkładu w **7/9**. Rozbieżności notatek
dotyczą przypadków group, noul, score i tool.

Kontrola native BF16 przechodzi własne 40/40 kontroli funkcjonalnych,
również oba ścisłe porównania mieszanych żądań. Jednak względem tych samych
dziewięciu odpowiedzi vLLM nadal ma **4/9** zgodnych tekstów/liczników,
**7/9** zgodnych argmax i maksymalną różnicę P2 **0,31981161236763**.
W tym porównaniu oceniano końcowe P2, nie P1/no_think, ponieważ bazowy
silnik referencji był FP32. Wymuszone prompty P2 nie zależą od routowania P1.

Samo dopasowanie deklarowanego dtype do BF16 nie usuwa rozbieżności.
Native nadal używa readoutu F32 i pełnego prefiksu; vLLM ma własne kernele
i KV cache. Ich wpływ nie został rozdzielony. Nie przypisujemy całego FAIL
jednej z tych przyczyn i nie zmieniamy poprawnego względem PyTorch toru FP32
bez osobnego rozstrzygnięcia kontraktu parytetu.

## Lifecycle i przywrócenie usługi

Po rozłączeniu klienta kolejne żądanie zwróciło 200 dla obu backendów.
W osobnym przebiegu SIGTERM podczas aktywnego żądania zakończył proces
z kodem 0 po 364,162 ms dla heads i 4924,997 ms dla thinking. Oba rozpoczęte
żądania otrzymały 200; thinking dokończył 47-tokenową notatkę. Zapisano
wykorzystanie GPU przy sygnale i po zatrzymaniu. To skończony pomiar drain
i odzyskania obsługi, bez dowodu preemption ani ogólnego boundu anulowania.

Osobne surowe żądania JSON z liczbami `0.990` i `9.9e-1` zwróciły 200
i wynik równy polityce `"0.99"` po usunięciu czasu. Wcześniejszy runner
serializował obie liczby jako `0.99`; tylko ta osobna próba sprawdza leksykę.

Istniejącą usługę przywrócono. Końcowe zapisane kontrole potwierdzają
stan `running healthy`, HTTP 200 na health, wyłącznie produkcyjny proces
Basal w zestawieniu procesów GPU i brak listenera diagnostycznego.
Nie zastępowano obrazu ani konfiguracji istniejącej usługi.

## Dowody i odtworzenie

[REPRODUCE.md](REPRODUCE.md) opisuje układ danych, audyt bez inferencji
i odtworzenie wariantów. Zachowano wszystkie nieudane próby: niedozwolony
argument referencji, omyłkę nazwy klasy, błąd odświeżania SQLite przez Nsight,
jak również rzeczywiste dwa negatywne porównania native i negatywną
referencję vLLM. Błędy runnerów poprawiono; nie są defektami Basal.
Ostrzeżenie vLLM o jednym pozostawionym semaforze też pozostaje w logu.

Dane i surowe ślady są prywatne, lokalne, poza Gitem. Archiwum nie jest
przygotowane do publikacji: zawiera ścieżki i informacje środowiska.
Nie ma publicznego URI HF i nie twierdzimy, że wykonano kontrolę anonimowego
pobrania. Wagi, środowiska Python i binarka nie należą do archiwum.

Lokalne archiwum: `.cache/measurement-artifacts/upstream-api-241-trained-2026-10-28/measurement-data.tar.gz`,
2 558 355 B, SHA-256
`508e438ae97079db9ead8dcbaa1734fac9accb4a8399a1fae992d5069b447cb7`.
Rozpakowanie do osobnego nowego katalogu potwierdziło **422/422** wpisy
`SHA256SUMS`. Ponowny audyt bez modelu odtworzył ten sam `audit-summary.json`;
po nim nadal zgadza się 422/422 sum. Kopia README w archiwum jest wersją
sprzed dopisania tej noty z hashem archiwum.

## Granice wyniku

Są to skończone diagnostyki na jednym korpusie, orders 1 i wskazanych
ustawieniach. Nie dowodzą jakości na danych użytkownika, przyspieszenia,
długotrwałego obciążenia, wszystkich ścieżek błędów ani boundu pamięci.
Metal, media, większe konteksty i rzeczywisty OOM/device-loss pozostają
poza tym pomiarem. Orders auto/2, kontynuacje z aktywną polityką,
pełny upstream fast-exit startup, domyślne F16, długotrwała współbieżność,
fizyczne granice alokatora i background z aktywnymi trained ścieżkami nie
zostały zakwalifikowane. Nie badano korpusu jakościowego ani poboru energii.
Starsze raporty i nieudane próby zachowano. Pełnego zadania zamknięcia
wszystkich FAIL/GAP nie uznajemy za zakończone.
