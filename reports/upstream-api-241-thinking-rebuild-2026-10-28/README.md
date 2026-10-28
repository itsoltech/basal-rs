# Thinking J1/J2: natywna generacja i jawna numeryka BF16

**Zgodność z rzeczywistym upstream nadal ma status FAIL.** Nowy profil
BF16 daje 14/19 odpowiedzi w progu 1e-4 zamiast 12/19, lecz nie poprawia
zgodności notatek (4/9) ani argmax (7/9). Maksymalna różnica rośnie
z 0,246800 do 0,253332. Domyślne `inherit` zachowuje wcześniejsze wyniki.
Nie zmieniono domyślnej konfiguracji ani binarki usługi produkcyjnej.

Użytkownik wybrał przebudowę wyłącznie w Rust/CUDA bez gwarancji identycznych
wyników. Implementacja i skończona seria pomiarów są zakończone. Zgodnie
z warunkiem zatrzymania brak poprawy notatek i decyzji kończy tę serię zmian
arytmetyki. Profil pozostaje opcjonalny; nie jest trybem parytetu ani nową
referencją akceptacyjną.

## Wersje i wejścia

Pomiar: 2026-10-28, NVIDIA RTX 6000 Ada Generation 48 GB, SM 8.9,
sterownik 580.159.03. Native: Rust 1.95.0, CUDA 12.9.1.
Upstream: niezmienione Basal 2.4.1,
`b20f0f9b159f71726459aa98ede1356ab65695fd`, Torch 2.11.0,
Transformers 5.17.0 i sidecar vLLM 0.26.0.

| Wariant | Commit | SHA-256 binarki |
|---|---|---|
| J1, incremental KV | `1c8625189cbe30cea8553bdcfba7277bf23dc2b1` | `c64e5863112c993a8039e5a535457aff646ee9fa2f9e1017689e5ed435b6beb2` |
| J2, także default/inherit | `8f135a3cc06617019af0b9adbb600d696263cdd1` | `f2a20aaae029e576a7264e8f23132b43e63abe5c06be08ab8566ff355e01773c` |

Para wag to
`Remek/basal-2.0-4.5B@e5ba9149760d99b73b1963d8e0ccf948b58afacb`,
root i `think/`. SHA-256 bazowych wag:
`481d8ea4972379a611f46784196e407bee239a37c3981c6e29ff1aaaeb0f19c2`;
generatora: `1e295ab2bcc6e812bc9e22f710cd0a6fc4d91371eefffcb271d77cb6ad97c8a5`.
Sumy wag pochodzą z wcześniejszej kontroli tej instalacji; w J2 ponownie
sprawdzono konfigurację generatora, nie hashowano całych wag. Nie zmieniano
promptów, tokenizera, kalibracji P2 ani progu 1e-4.

Referencje są kopiami zachowanych odpowiedzi rzeczywistego upstream i
dodatkowej diagnostyki Torch FP32. Wcześniejsza kontrola oryginalnego
`python -m basal.server` potwierdziła 19/19 odpowiedzi względem zapisanej
referencji sidecara; szczegóły opisuje
[raport diagnozy](../upstream-api-241-thinking-diagnosis-2026-10-28/README.md).
W tej serii upstream nie uruchamiano ponownie. Wpis `vllm: null` w manifeście
sterownika nie opisuje sidecara; jego wersję potwierdza zachowany `vllm.log`.

## Zmiana implementacji

J1 wykonuje jeden prefill i kolejne kroki tylko dla nowych tokenów.
Pamięć KV należy do żądania i jest zamykana przed niezależnym P2.
Bounds generatora korzystają z jednej sekwencji o efektywnym `think_max_len`,
nie z większego kształtu batcha modelu bazowego. Stary wynik FP32 z pełnym
przeliczaniem prefiksu nie został przeniesiony bez pomiaru na nowy harmonogram.

J2 dodaje `think_numerics: upstream-bf16-pointwise`, domyślnie `inherit`.
Osobny natywny backend thinking używa BF16 oraz zaokrąglonego readoutu BF16,
zmienionych punktów zapisu RoPE/Q/K, biasu w epilogu cuBLASLt, SiLU i
wewnętrznego residualu FP32. Używa istniejącego uchwytu Lt, bez dodatkowego
procesu, zależności runtime ani workspace Lt. Split-K może zaokrąglać sumy
częściowe do BF16; nie ma obietnicy pojedynczego zaokrąglenia całego GEMM.
Dokładny kontrakt i odmowy są w [THINKING.md](../../docs/THINKING.md).

Profil nie obsługuje tabel GEMM i **wyłącza cały background executor silnika,
także dla żądań `think: off`**. Pozostaje ograniczony do wspieranej pary Llama,
CUDA i jednego etapu. Nie zmieniono ustawienia domyślnego.

## Metoda i kontrola zachowania

Sześć osobnych procesów, kolejno: J1 thinking, J2 bez thinking, J2 z opcją
pominiętą, J2 z jawnym `inherit`, dwa uruchomienia profilu BF16. Model bazowy
zawsze FP32, readout FP32, orders 1, bez prefix/state cache i bez tabel GEMM.
Thinking: 256 tokenów, kontekst 2048. Każdy proces thinking obsłużył bazowe
żądania, force/auto, nieprawidłowe wejścia, alias endpointu, dwa powtórzenia
wszystkich dziewięciu przypadków oraz mieszane off/force.

Łącznie **375/375 kontroli runnera PASS**, sześć kodów zakończenia serwera 0.
To kontrole tej skończonej serii HTTP, nie zestaw testów projektu ani dowód
parytetu numerycznego. Dwa nieprawidłowe wejścia w każdej fazie thinking
prawidłowo zwracają 422.

| Porównanie | Dokładnie równe odpowiedzi po usunięciu czasu |
|---|---:|
| Nowy J1 względem wcześniejszego J1 | 19/19 |
| J2 bez thinking względem J1 bez thinking | 9/9 |
| J2 default względem nowego J1 | 19/19 |
| Jawne `inherit` względem default, także błędy i powtórzenia | 42/42 |
| Off przy profilu BF16 względem off przy default | 9/9 |
| Profil BF16 po restarcie względem pierwszego procesu | 42/42 |

Usunięto wyłącznie `usage.latency_ms` i `answers.*.think.ms`. Mowa o
dokładnej równości pozostałych wartości JSON; nie mierzono wszystkich
wewnętrznych tensorów. Powtórzenia w każdym z pięciu procesów thinking
były identyczne: po 18/18 odpowiedzi. Kontrola restartu obejmuje dwa
procesy na tej jednej karcie i konfiguracji, nie dowolny wybór algorytmu Lt.

## Wyniki względem referencji

Każde porównanie obejmuje 19 sparowanych żądań, takie same statusy 200
i te same ścieżki prawdopodobieństw. Kolumna progu wymaga maksymalnej
bezwzględnej różnicy ≤1e-4 w całej odpowiedzi, także `no_think`.
Tekst/tokeny/argmax dotyczą dziewięciu odpowiedzi z notatką w żądaniach
force. Grupa powtarza część pytań; nie są to niezależne losowe próbki.

| Native | Referencja | Odpowiedzi ≤1e-4 | Maks. różnica | Tekst | Liczba tokenów | Argmax |
|---|---|---:|---:|---:|---:|---:|
| J2 default FP32 | Rzeczywisty upstream | 12/19 | 0,2467996180 | 4/9 | 4/9 | 7/9 |
| J2 BF16, oba procesy | Rzeczywisty upstream | 14/19 | 0,2533319890 | 4/9 | 4/9 | 7/9 |
| J2 default FP32 | Upstream Torch FP32 | 19/19 | 1,1026859283e-6 | 9/9 | 9/9 | 9/9 |
| J2 BF16, oba procesy | Upstream Torch FP32 | 12/19 | 0,0098609626 | 8/9 | 9/9 | 9/9 |

Przejście progu poprawiło się w `force-act` (0,00112970 → 6,26e-7)
i `force-choice` (0,00023893 → 3,17e-8). Pięć odpowiedzi profilu BF16
nadal nie przechodzi progu rzeczywistego upstream:

| Żądanie | Maks. różnica |
|---|---:|
| `force-group` | 0,2533319890 |
| `force-score` | 0,2533319890 |
| `force-noul` | 0,0459786057 |
| `force-tool` | 0,0021823719 |
| `force-unicode` | 0,0021705627 |

Argmax różni się w score i jego powtórzeniu w grupie. Względem FP32
profil BF16 zmienia też notatkę tool: pierwszy różny znak Unicode ma indeks
98, mimo tej samej liczby 53 tokenów. Wcześniejszego PASS diagnostyki FP32
nie można przypisać nowemu profilowi BF16.

## Zasoby i czas: SIGNAL

Próbkowanie `nvidia-smi` co około 100 ms obejmowało cały cykl życia każdego
serwera. Maksimum próbek dotyczy całej karty i jest dolnym ograniczeniem
rzeczywistego maksimum. Nie jest licznikiem high-water alokatora ani dowodem
fizycznej granicy pamięci. Produkcja podczas faz była zatrzymana.

| Faza | Maks. próbka VRAM, MiB | Suma `think.ms` dla pierwszych 9 odpowiedzi | Suma ms / wygenerowane tokeny |
|---|---:|---:|---:|
| J1 | 19114 | 27086,206 | 52,800 |
| J2 bez thinking | 10018 | — | — |
| J2 default | 19114 | 27468,940 | 53,546 |
| J2 inherit | 19114 | 27551,145 | 53,706 |
| J2 BF16, proces 1 | 19152 | 7968,459 | 15,533 |
| J2 BF16, proces 2 | 19152 | 7949,447 | 15,496 |

Każdy zestaw dziewięciu odpowiedzi zawiera łącznie 513 wygenerowanych
tokenów. `think.ms` obejmuje prefill, generację i P2; iloraz nie jest
czasem pojedynczego kroku decode. Krótszy czas profilu jest obserwacją
w tej serii ze zmienioną numeryką i wynikami. Brak ABBA, długiego obciążenia
i porównania na identycznych wymuszonych tokenach nie pozwala deklarować
ogólnego przyspieszenia. Próbkowane VRAM profilu jest o 38 MiB większe.

## Weryfikacja, ograniczenia i decyzja

Implementacje miały niezależne przeglądy poprawności i osobne upraszczanie
Opus 5.5. Po ostatniej korekcie lifetime biasu przeszły fmt, Clippy i prywatny
Rustdoc dla default/Intel/CUDA. Release CUDA build końcowego commitu ma kod 0;
suma przesłanej binarki zgadza się ze źródłową. Nie dodawano testów.
263/263 istniejących ciał entry PTX pozostaje identycznych w czterech
kompilacjach architektur; jest to statyczny dowód zachowania tych entry,
nie wszystkich ścieżek runtime. Konfiguracja przypiętego generatora spełnia
kontrolę geometrii i epsilon 1e-6. Dowody zależności odziedziczono, bo ich
wejścia nie zmieniły się; nie deklarujemy nowego audytu zależności.

Pozostają GAP: operatorowy replay rzeczywiście skompilowanych kerneli upstream,
odczyty logitów na identycznych wymuszonych tokenach, fizyczne maksimum
alokatora i zgodność z bounds, Metal/macOS, split oraz zachowanie przy EOS,
limicie tokenów, błędach, przerwaniu, utracie urządzenia i długim obciążeniu.
Przejście zwykłego zamknięcia sześciu serwerów nie kwalifikuje tych ścieżek.
Profil wyłącza background; tej konfiguracji nie kwalifikowano do pracy w tle.
Nie uruchamiano kolejnych prób arytmetyki po braku poprawy tekstu i decyzji.

Oryginalną usługę przywrócono. Świeża końcowa kontrola zapisała
`running healthy`, HTTP 200, jeden produkcyjny proces GPU i brak listenera
diagnostycznego. Wcześniejszy odczyt podczas startu pokazywał `starting`
i resetowane połączenia; o przywróceniu świadczy późniejsza kontrola,
nie sam kod zakończenia sesji. Nie wdrożono nowej binarki.

Dane, źródła, dokładne komendy, wyniki per przypadek i logi są zachowane
lokalnie. Odtworzenie opisuje [REPRODUCE.md](REPRODUCE.md). Publicznego URI
nie ma; nie wykonywano publikacji ani push. Historyczne raporty i statusy
FAIL pozostają bez zmian.

Lokalne prywatne archiwum `measurement-data.tar.gz`: **35 969 713 B**,
SHA-256 `3acd2b7fe16c8f25c189608841869a472bdb48e4e0a4e1f23f30b6cfb3b69511`.
Po rozpakowaniu do osobnego katalogu potwierdzono **1597/1597** sum plików
oraz sumę archiwum. Dane są pod
`reports/upstream-api-241-thinking-rebuild-2026-10-28/evidence/private/`.
Kopia README wewnątrz archiwum poprzedza dopisanie tej noty.
