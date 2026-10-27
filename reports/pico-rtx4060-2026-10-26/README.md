# Pico: zgodność z upstream FP32 i HTTP na RTX 4060

**PASS w zbadanym zakresie.** Model działa przez alias `pico`. Natywne FP32
oraz domyślne BF16 zachowały wszystkie sprawdzone decyzje upstream FP32.
Eksporty `single` i `tree` są identyczne bajtowo. Oba endpointy zachowały
pełne odpowiedzi przy C1/C4 oraz przy pytaniu zadanym samodzielnie i razem
z innymi pytaniami. [Użycie i ograniczenia](../../docs/PICO.md).

Nie jest to potwierdzenie jakości na nowych danych ani wydajności zbliżonej
do zoptymalizowanego upstream. Metal i inne GPU pozostają niepotwierdzone.

## Wersje, sprzęt i metoda

Kod basal-rs: baza `db55af9d7cab5f12a18347b0bb27eb4dab4b1531` z patchem
`source/tracked.patch` i nowymi plikami `source/crates/` w archiwum poniżej.
Nie tworzono commita. Pomiar nie wymagał dalszych poprawek arytmetyki backendu.

Model: [`Remek/basal-omni-pico`](https://huggingface.co/Remek/basal-omni-pico),
rewizja `a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`, tekstowy Qwen3.5 0,8B.
SHA-256 checkpointu:
`ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9`.
Sprawdzono sumy wag, tokenizerów, konfiguracji i obu kalibracji.

Referencja: upstream **2.4.0**, commit
`615fe321a818fdf5d2897f758b8d823f919d7f94`, eager, rezydentne FP32 na CUDA.
Python 3.12.15, PyTorch 2.11.0+cu128, Transformers 5.17.0, tokenizers 0.23.3.
Nie instalowano FLA ani `causal_conv1d`: upstream używał referencyjnych
operacji PyTorch. Checkout upstream pozostał bez zmian.

Sprzęt: NVIDIA GeForce RTX 4060, 8188 MiB VRAM, compute capability 8.9,
sterownik 615.78.08, limit mocy 115 W. Rust 1.95.0, toolkit CUDA 12.9.1;
PyTorch używał swoich bibliotek CUDA 12.8. W sesji korzystano z jednego
procesu wykonującego inferencję na GPU naraz.

Zestaw podstawowy: 44 domyślne pytania basal-bench z przypiętego upstream
oraz 4 własne żądania z 12 gałęziami odczytu. Obejmuje PL/EN, `choice`,
`noul`, `score`, `multi`, `act`, `tool`, Unicode NFC i obiektowy stan.
Obie kolejności mają osobne referencje i kalibracje. Zestaw zawiera końcowe
fragmenty 0, 1, 2 i 3 tokenów po granicy 64 tokenów. To 56 odczytów dla
orders 1 i 112 forwardów opcji dla orders 2; nie są to niezależne zestawy
do sumowania trafności.

## Wyniki numeryczne

W porównaniach podstawowych: zero brakujących rekordów, niezgodnych promptów,
token IDs, kodów opcji, permutacji i pakowania. Bez zmian argmax przed ani
po kalibracji. Poniżej maksima po 44 pytaniach i 12 gałęziach API.

| Wariant wobec upstream FP32 | Decyzje bench / API | Maks. różnica logitu | Maks. różnica P po kalibracji | Maks. TV po kalibracji |
|---|---|---:|---:|---:|
| basal-rs F32, orders 2 | 44/44, 12/12 | 0,0000195143 | 0,0000050515 | 0,0000050291 |
| basal-rs domyślny BF16, orders 1 | 44/44, 12/12 | 0,0671544 | 0,0186341 | 0,0186486 |
| basal-rs BF16, orders 2 | 44/44, 12/12 | 0,0747900 | 0,0187020 | 0,0187020 |
| upstream BF16, orders 2, kontrola | 44/44, 12/12 | 0,1332843 | 0,0296126 | 0,0296126 |

Wspólny argmax nie oznacza identycznych prawdopodobieństw ani potwierdzenia
kalibracji. Natywny BF16 i upstream BF16 także nie są bitowo równe: maksymalna
TV średniej rozkładów przed kalibracją na bench wyniosła 0,0486403, bez zmian
decyzji w tej próbce. Nie wyciągamy z tego wniosku o przewadze jakościowej.

Dodatkowe dwa poprawne wejścia, orders 1: stan 5080 tokenów oraz 255 opcji
z promptem 1997 tokenów. F32 i BF16 zachowały decyzje FP32 upstream w **2/2**
przypadkach, w tym wybór `label-254`. Maksymalna różnica logitu zestawu:
F32 `0,0000228578`, BF16 `0,0514017`; maksymalna TV po kalibracji:
odpowiednio `0,0000045317` i `0,0043920`. 256 opcji odrzuciły oba silniki.

Sprawdzono typy i pola odpowiedzi, decyzje `act`, selekcję `multi` oraz
osobno nazwę narzędzia i `call`. Eksporter raportuje do `6,71·10⁻⁸` w polu
`answer_vs_items_max_abs_prob_diff`: rekonstruuje drugą składową `noul`
jako `1 - P(true)`. To pole nie jest pomiarem bitowej równości logitów.
Właściwy eksport `single`/`tree` opisany poniżej jest identyczny bajtowo.

## Pakowanie i HTTP

`cmp` potwierdził identyczność `bench.jsonl` oraz `systemone.jsonl` eksportów
domyślnego BF16 orders 1 dla `single` i `tree`. Dodatkowy `basal compare`
ma maksymalną różnicę logitów i TV równą 0.

Serwer wystartował przez `--model pico` z przypiętym snapshotem. Ustawienia:
auto/BF16, orders 1, FIFO, 4096 tokenów partii, limit 64 żądań, drugi tor
powyżej 256 tokenów, slice 10 ms. Dla każdego endpointu wykonano 4 żądania
rozgrzewkowe, 4 sekwencyjne, 8 przy C1 i 8 przy C4.

- `/v1/basal`: 24 poprawne POST; w 20 żądaniach faz mierzonych 0 błędów,
  0 zmienionych obiektów `answers`, maksymalna różnica P względem pierwszej
  odpowiedzi 0. Cztery reprezentatywne odpowiedzi są równe eksportowi.
- `/v1/systemone`: ten sam zakres i wynik. Przy C4 obserwowano średnio 2
  żądania na partię; dla `/v1/basal` 1,75. Próba obejmuje rzeczywiste grupowanie.
- Dwa dodatkowe poprawne POST z samodzielnym `dept`: identyczny obiekt
  odpowiedzi jak w żądaniu z sześcioma pytaniami, dla obu endpointów.
- Dwa niepoprawne POST z 256 opcjami: `/v1/basal` zwrócił 422, a TypeSafe
  `/v1/systemone` 400, z komunikatami właściwymi dla tych kontraktów.
- `/health` po obciążeniu: `ok`. SIGINT zakończył własny serwer z kodem 0;
  po zakończeniu GPU wskazywało 2 MiB zajętej pamięci.

Łącznie: **50 poprawnych POST i 2 zamierzone odmowy**. Równość HTTP dotyczy
pełnych obiektów `answers` po odczycie JSON. Bajtową równość sprawdzono
oddzielnie na plikach eksportu. Czasy i próbki GPU zachowano, lecz ta krótka
kontrola nie kwalifikuje wydajności ani nie porównuje jej z upstream z FLA.

## Kontrole i ograniczenia

Build CUDA, formatowanie, Clippy release workspace all-targets dla konfiguracji
default, Intel i CUDA+Intel all-features oraz Rustdoc z `-D warnings`: PASS.
Kontrola zależności i wyjątku `paste`: PASS; pozostały istniejące ostrzeżenia
o duplikatach, bez zmiany grafu zależności. Nie dodawano frameworka testów.

CLI odrzuciło `q8_0`, `balanced`, media i dodatni cache stanu, każde z kodem 1.
Cache stanu jest odrzucany po załadowaniu wag; próba nie dowodzi odmowy tej
opcji przed alokacją GPU. Nie badano odmów na sprzęcie Metal.

Zachowano niepowodzenia przygotowania środowiska: trzy pełne buildy CUDA
nie przeszły przez konflikt z nowym glibc lub kolejnością nagłówków C++.
Końcowo użyto lokalnego toolkitu, GCC 14.2.1, nagłówków glibc 2.40 i
`-nostdinc++` z jawną kolejnością include. Nie zmieniano nagłówków systemu,
zależności ani kodu upstream. Rozdzielono próby PTX i kompilacji obiektowej.

**FAIL sumy README wydawcy:** plik w przypiętym modelu ma SHA-256
`e54a440e8427951530b96f1759648eb338bfdd387028fb1a3bba351e0d3b6c33`,
a publikowane `SHA256SUMS` podaje
`80124e99c6f1e3a0bf13136e0436785bde1db7153133e3798c1eb9217f983d49`.
Pozostałe 10 wpisów, w tym checkpoint, tokenizer, config i kalibracje, przeszły.
Nie zmieniano plików modelu; rozbieżność README pozostaje w danych kampanii.

Zakres kończy się na jednym RTX 4060 i małym korpusie zgodności. Nie badano
Metal, innych GPU, FP16, GGUF, kwantyzacji, wielu GPU, pracy w tle, kontekstów
powyżej 5080 tokenów ani jakości na Werdykcie. Ścieżka pozostaje referencyjna,
z sekwencyjną rekurencją i ponownym przeliczaniem wspólnego stanu.

## Artefakty i odtworzenie

URI:
`hf://buckets/itsoltech/basal-rs/measurements/pico-rtx4060-2026-10-26/fa1f9d2e43e0f66e5e38d9a3d8e3311220a13ebaa15124d986ffa9d283a5f994/measurement-data.tar.gz`

SHA-256 archiwum:
`fa1f9d2e43e0f66e5e38d9a3d8e3311220a13ebaa15124d986ffa9d283a5f994`.
Rozmiar: **363 686 B**. Ponowne pobranie z autoryzacją i anonimowo
potwierdziło SHA archiwum i **75/75** sum plików w obu kopiach.
Lokalne archiwum zostało zachowane.

Dane są pod `reports/pico-rtx4060-2026-10-26/` wewnątrz archiwum:
referencje i eksporty, porównania, audyty, odpowiedzi HTTP, wejścia, metadane
modelu, wersje pakietów, dokładny patch i nowe źródła. `REPRODUCE.md` podaje
odtworzenie; procedura pobrania i weryfikacji jest w
[MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).
Po pobraniu do nowego katalogu:

```sh
sha256sum measurement-data.tar.gz
tar -xzf measurement-data.tar.gz
sha256sum -c SHA256SUMS
```

Przed rozpakowaniem porównaj sumę z podaną wyżej. Publiczne metadane używają
logicznych nazw zbiorów; lokalne ścieżki, adres serwera i prywatne logi
zostały poza archiwum. Tokenów, logitów, prawdopodobieństw i odpowiedzi nie zmieniano.
