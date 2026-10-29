# Thinking: wybór algorytmów GEMM bez pomiaru czasu

**Diagnostyka działa, ale nie naprawia zgodności z upstream.** Jawne
`BASAL_THINK_GEMM=heuristic` dało identyczne zapisane wyniki w dwóch procesach.
Pełne HTTP osiągnęło jednak 13/19 odpowiedzi w progu 1e-4, wobec wcześniejszych
14/19 profilu z pomiarem czasu. Pozostają 4/9 zgodnych notatek i 7/9 zgodnych
argmax. Nie zmieniono ustawień domyślnych ani wdrożonej binarki produkcyjnej.

## Wersje i zakres

Pomiar 2026-10-29: NVIDIA RTX 6000 Ada Generation 48 GB, SM 8.9,
sterownik 580.159.03, CUDA 12.9.1, cuBLASLt 12.9.1.4, Rust 1.95.0.
Kod aplikacji i probe: `f9f915617482947950bb273807052c7b2c236e4c`.
Źródła w archiwum odpowiadają temu commitowi; późniejsza nota w dokumentacji
nie zmienia kodu aplikacji. SHA-256 binarki HTTP:
`9648d47279ba9c80ed98a04e01ed8d237a8a55703d639188f46b9a9f2187fac5`.
SHA-256 binarki probe:
`cba5a93b7e02dc804385347e520835688900c53e57e88bfd297b69ec04bd45ec`.

Model: `Remek/basal-2.0-4.5B@e5ba9149760d99b73b1963d8e0ccf948b58afacb`,
te same zainstalowane wagi bazowe i `think/` co w J2/J3; całych wag ponownie
nie hashowano. Sprawdzono konfigurację thinking, w tym `rms_norm_eps=1e-6`,
oraz SHA-256 kalibracji thinking:
`8938f99af2d23223e197cfbd52e3941627934cf33dd132fd310d1a84db9b217d`.

Referencją pozostają zachowane odpowiedzi rzeczywistego Basal 2.4.1
(`b20f0f9b159f71726459aa98ede1356ab65695fd`, vLLM 0.26.0,
Torch 2.11.0, Transformers 5.17.0). Nie uruchamiano nowego vLLM ani upstream.
Torch FP32 jest dodatkową diagnostyką. Korpus i dotychczasowe wyniki opisują
[J2](../upstream-api-241-thinking-rebuild-2026-10-28/README.md) oraz
[J3](../upstream-api-241-score-isolation-2026-10-28/README.md).
Dziewiętnaście przypadków HTTP korzysta z tego samego stanu bankowego i osobnego
przypadku Unicode; pytania powtórzone w grupie nie są niezależnymi próbkami.

## Zmiana diagnostyczna

Tylko jawny profil `upstream-bf16-pointwise` może zamiast pomiaru czasu użyć
pierwszego wyniku heurystyki cuBLASLt dla dokładnego `(M,N,K)`. Algorytm jest
zapamiętywany dla tego kształtu. Nie dodano paddingu ani implementacji vLLM;
nie dowiedziono, że jest to algorytm wybierany przez upstream.

`gemm_tuning()` zapisuje konfigurację oraz osiem surowych słów algorytmu.
Gettery używają wymaganych buforów 4/2 B i kontrolują status oraz rozmiar zapisu.
Probe odrzuca pusty lub częściowy odczyt dziewięciu atrybutów i zapisuje
algorytm także jako dokładny ciąg szesnastkowy. Wszystkie odczyty na badanej
wersji cuBLASLt przeszły tę kontrolę.

Bez zmiennej pozostaje dotychczasowy wybór przez pomiar czasu. Zmienna trafia
do tożsamości planów procesu, również bazowego, ale nie zmienia jego obliczeń.
Nie zmieniono kerneli ani zależności runtime. Kod przeszedł dwa przeglądy
poprawności Opus 5.5 i osobny etap upraszczania przed integracją.

## Siedem procesów probe

Użyto niezmienionych tokenów wejścia J3, SHA-256:
`09b22e1f6a625ae5f6347842b28ff926b4bbc1d0c5bc0a8a280de994a1783242`.
Każdy proces wykonuje cztery P2, 26 kroków teacher forcing i jeden odczyt
pełnego prefiksu po zamknięciu KV. Łącznie: 28 P2, 182 kroki i 7 odczytów
stateless. Zachowano top-8 i parę tokenów rozgałęzienia, nie pełny słownik.

| Próba | Wynik |
|---|---|
| D0: FP32 bez zmiennej i ze zmienną | Oba P2 i cała zapisana generacja dokładnie równe J3 |
| D1: trzy procesy z pomiarem czasu | P2 równe; zapisana generacja i wybrane algorytmy różnią się |
| D2: dwa procesy bez pomiaru czasu | P2, generacja oraz wszystkie 20 konfiguracji i surowych algorytmów równe |
| D3: P2 ustalonej notatki upstream | Oba procesy heurystyczne dokładnie równe upstream; kroki 0–24 zgodne 25/25 |

Wszystkie siedem procesów zakończyło się kodem 0. Wszystkie 14 par powtórzeń
P2 wewnątrz procesu jest równych. W D1 względem procesu 01 różnią się odczyty
24/26 kroków w procesach 02 i 03, przy tych samych top-1 i odczycie stateless.
Różnią się też zapisane algorytmy klas M=16, a w procesie 03 także M=96.
To jest zgodne z hipotezą wpływu wyboru GEMM, lecz nie dowodzi przyczyny:
rekord timed opisuje zwycięzcę pierwszego pomiaru klasy, a późniejszy fallback
dla dokładnego M nie jest rejestrowany. D1 nie potwierdza ani nie obala przyczyny.

Heurystyczne P2 notatki upstream daje effective logits
`[20.875, 22.25, 27.75]` i dokładnie jej skalibrowane prawdopodobieństwa
`[0.15201598405838013, 0.20296710729599, 0.6450169086456299]`.
Mimo tego w kroku 25 KV nadal wybiera `konta` 2695 zamiast upstream `danych`
1847: effective logits wynoszą 19.25 i 19.125. Upstream miał remis logprobów.
Nie porównujemy absolutnych logitów native z logprobami upstream.

## Pełne HTTP w dwóch świeżych procesach

Spełnienie D3 uruchomiło przewidziane pełne porównanie. Użyto tego samego
runnera J2, accurate/fused, bazowego FP32, orders 1, bez cache prefix/state
i tabel GEMM, `think_tokens=256`, `think_max_len=2048`, z jawnym profilem
pointwise i `BASAL_THINK_GEMM=heuristic`. Każdy proces wykonał też po dwa
powtórzenia dziewięciu żądań force i ruch mieszany off/force.

| Wariant i referencja | Odpowiedzi ≤1e-4 | Notatki | Argmax thinking | Maks. różnica prawdopodobieństw |
|---|---:|---:|---:|---:|
| J2 timed → rzeczywisty upstream, historycznie | 14/19 | 4/9 | 7/9 | 0.2533319890 |
| J4 heuristic → rzeczywisty upstream, oba procesy | 13/19 | 4/9 | 7/9 | 0.2436882854 |
| J4 heuristic → diagnostyka Torch FP32, oba procesy | 12/19 | 7/9 | 9/9 | 0.0220165700 |

Próg liczony jest po wszystkich odpowiadających sobie polach probabilities
odpowiedzi. Notatki i argmax liczone są dla dziewięciu odpowiedzi thinking.
Wobec rzeczywistego upstream `force-act` i `force-choice` tracą wcześniejszy
PASS, a `force-tool` go zyskuje. Dla act zmieniła się też notatka; tool teraz
ma zgodną notatkę. Równa liczba 4/9 nie oznacza tego samego zestawu dopasowań.
Największa różnica nadal dotyczy score, także powtórzonego w grupie.

Oba serwery zakończyły się kodem 0; wszystkie 144 kontrole runnera przeszły.
Między procesami 42/42 pełne zapisane body odpowiedzi są równe po usunięciu
wyłącznie `usage.latency_ms` i `answers.*.think.ms`. Dziewięć odpowiedzi off
jest tak samo równych wcześniejszemu J2 unset. To kwalifikacja skończonej
serii, nie gwarancja powtarzalności dla wszystkich kształtów i urządzeń.

## Kto ma poprawne wyniki i decyzja

Względem wymaganego kontraktu odpowiedzi upstream niezgodność pozostaje po
stronie naszego silnika: **FAIL**. Nie jest to dowód błędu matematycznego
vLLM. Różne algorytmy, harmonogramy i zaokrąglenia BF16 mogą zmieniać bliskie
logity oraz całą dalszą notatkę. Nie wykazano też konkretnego błędnego
operatora native. Historyczna zgodność domyślnego native FP32 z Torch FP32
(19/19, maks. około 1.1e-6) jest dowodem zgodności tych implementacji na korpusie,
nie dokładnej matematyki modelu. Trafności semantycznej nie rozstrzygnięto:
brakuje oznaczonych prawidłowych odpowiedzi.

Tryb heurystyczny pozostaje jawną diagnostyką. Nie spełnił warunku braku
pogorszenia wobec J2 i nie jest nowym domyślnym profilem. Skończona próba
J4 jest zakończona; nie wykonano dalszych zmian arytmetyki ani wdrożenia.

## Weryfikacja, ograniczenia i dane

Formatowanie, Linux/CUDA Clippy z `-D warnings`, prywatny CUDA Rustdoc oraz
formatowanie, Clippy i build probe przeszły u autora końcowej poprawki;
rodzic zweryfikował logi i hashe, nie powtarzał tych sprawdzeń. Rodzic
zbudował zintegrowane CLI CUDA offline, `--locked`, exit 0. Zależności i
lockfile nie zmieniły się; cargo-deny nie uruchamiano ponownie.
Nie kompilowano tu Metal/Intel i nie mierzono wydajności, fizycznych peaków
pamięci, EOS, limitu generacji, błędów ani split. Bufor getterów ma alignment 1;
review uznał dodatkowe wyrównanie za opcjonalne, nie blokadę tej próby.

Po obu sesjach przywrócono oryginalną usługę. Świeży dowód po końcowej sesji
potwierdza healthy, HTTP 200, wyłącznie oryginalny proces GPU i brak listenera
diagnostycznego. Nie wdrożono żadnej z nowych binarek do produkcji.

Surowe dane, źródła, porównania, dokładne polecenia i logi są prywatne i lokalne.
Nie ma publicznego URI; nie wysyłano ich do HF ani nie wykonywano push.
[REPRODUCE.md](REPRODUCE.md) opisuje strukturę oraz odtworzenie.

Lokalne prywatne `measurement-data.tar.gz`: **14 695 834 B**, SHA-256
`77ea1b8401ca92cd32385804340eeb2ce4add7058268a08c368981c0c95db751`.
Osobna kopia ma ten sam SHA; świeże rozpakowanie potwierdziło **331/331** sum
plików. Dane są pod
`reports/upstream-api-241-gemm-diagnostic-2026-10-29/evidence/private/`.
README w archiwum poprzedza dopisanie tej noty o integralności.
