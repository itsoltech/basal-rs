# Pico: próg FP32 i szybkość na RTX 4060

**Cel nie został osiągnięty.** Prototyp z korekcją reszt spełnił na zbadanych
wejściach próg maksymalnej różnicy skalibrowanego prawdopodobieństwa
**≤0,0001**, zachowując decyzje upstream FP32. Pozostał znacznie wolniejszy
od upstream `fast`, również dla krótkiego dokumentu. Odrzucono zmianę
domyślnego profilu: kod wykonania pozostaje w wersji v4. Eksperymentalne
źródła zachowano w archiwum, poza kodem produkcyjnym.

Wymagania użytkownika są łączne: wyprzedzić `fast` dla C1/około 5k tokenów,
nie powodować regresji krótkich dokumentów względem upstream oraz zachować
zgodne decyzje i maksymalną różnicę skalibrowanego prawdopodobieństwa
≤0,0001 względem upstream FP32. Próg dotyczy prawdopodobieństw, nie logitów.
0,0001 to 0,01 punktu procentowego. Zgodność samych decyzji nie wystarcza.

## Wersje, sprzęt i metoda

- Baza źródeł: `db55af9d7cab5f12a18347b0bb27eb4dab4b1531` oraz niecommitowane
  zmiany Pico z poprzednich kampanii. Profil początkowy/końcowy:
  `basal-omni-pico.v4`, `basal-qwen35.4`, `qwen35-dispatch.4`.
- Model `Remek/basal-omni-pico`, rewizja
  `a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`, SHA-256 wag
  `ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9`.
- Upstream 2.4.0, commit `615fe321a818fdf5d2897f758b8d823f919d7f94`,
  bez zmian checkoutu. Python 3.12.15, Torch 2.11.0+cu128,
  Transformers 5.17.0, Triton 3.6.0, FLA/fla-core 0.6.0.
- RTX 4060, SM 8.9, 8188 MiB, sterownik 615.78.08; natywny CUDA/cuBLASLt
  12.9.1. Limit mocy 115 W; bez zmian ustawień GPU, bez wynajmu maszyny.
- Jakość: `basal export` i `basal compare` z zamrożonymi referencjami FP32
  z poprzednich kampanii. Szybkość: HTTP C1, jedna kolejność, bez cache stanu
  i prefiksów, FIFO, natywny budżet 8192 tokenów, 4 wątki CPU.

Wejścia obejmują 44 pytania benchmarku, 12 gałęzi API, osobno 255 opcji,
odmowę 256 opcji i dokument 5080 tokenów. Korpus wydajności zawiera cztery
krótkie dokumenty oraz punkty 255/5080 tokenów. Dane i ich sumy są w archiwum.
To mały zestaw, nie dowód zgodności dla każdego dokumentu lub całego limitu
kontekstu modelu. Nie dopasowywano kalibracji ani wag do tego zestawu.

## Jakość prototypów

| Wariant / wejścia | Decyzje benchmarku | Decyzje API | Maks. różnica skalibrowanego prawdopodobieństwa |
|---|---|---|---|
| Stabilny blokowy DeltaNet, 1 kolejność | 44/44 | 12/12 | 0,000019312 benchmark; 0,000013113 API |
| Stabilny blokowy DeltaNet, 5080 tokenów | — | 1/1 | 0,000002802 |
| Wstępne pobieranie rekurencji, 5080 tokenów | — | 1/1 | 0,000004530 |
| Ostatni prototyp, 1 kolejność | 44/44 | 12/12 | **0,000020713 benchmark; 0,000004113 API** |
| Ostatni prototyp, 255 opcji i 5080 tokenów | — | 2/2 | **0,000016511** |

Ostatni prototyp zachował tokeny i pola decyzji endpointów, w tym wybory,
zbiory `multi`, akcje i schematy odpowiedzi. Odmowa 256 opcji jest zgodna.
Maksymalna różnica logitów benchmarku wyniosła około 0,000185952; nie jest
to naruszenie uzgodnionego progu prawdopodobieństw. Wyniki dwóch kolejności,
większego korpusu, współbieżności i innych GPU nie zostały dla tego prototypu
zakwalifikowane. Wynik jakości ma zakres podany w tabeli.

**Zachowany FAIL:** pierwsza wersja odwracania macierzy trójkątnej przez sumę
potęg była niestabilna dla skorelowanych kluczy. Nie spełniła zgodności decyzji
ani progu prawdopodobieństw: 42/44 i 11/12 decyzji, maksymalne różnice
0,344971 i 0,664555. Diagnostyka rekurencji pokazała błąd rzędu
10^35. Zastąpiono ją blokowym podstawianiem w przód; izolowany pomiar
obejmował prawie stałe klucze, beta bliską 1, powolny zanik i niezerowy stan.
Pliki `first-*` i `compare-first-o1.json` zachowują ujemny wynik.

## Wydajność i odrzucone zmiany

Wstępne serie HTTP po 6 mierzonych żądań na punkt, po 2 odrzuconych
rozgrzewkach, dały następujące mediany. Wszystkie mają status **SIGNAL**,
nie są potwierdzającym ABBA i nie uzasadniają deklaracji przyspieszenia.

| Prototyp | 255 tokenów [ms] | 5080 tokenów [ms] |
|---|---|---|
| Stabilny blokowy DeltaNet i oddzielne akumulatory projekcji | 32,70 | 961,02 |
| Dokładna rekurencja ze wstępnym pobieraniem, fragment 8192 | 35,43 | 1081,08 |
| Korekcja projekcji przez cuBLASLt, pozostałe jak wyżej | 37,39 | 988,84 |
| cuBLASLt, blokowy DeltaNet i zmieniony kafelek attention | 33,30 | 769,92 |

Końcowa kontrola na czterech kolejno uruchomionych silnikach obejmowała
po 12 zmierzonych żądań na punkt długości i 24 krótkie żądania na silnik.
To również **SIGNAL**, nie ABBA. Audyt przeliczył wszystkie obserwacje:
12/12 faz, 192 poprawne zmierzone odpowiedzi, zero problemów i zero zmian
bitowych odpowiedzi natywnych między powtórzeniami.

| C1, p50 / p95 [ms] | upstream fast BF16 | upstream eager BF16 | dotychczasowy v4 BF16 | prototyp bliski FP32 |
|---|---|---|---|---|
| 255 tokenów | **14,15 / 14,64** | 35,97 / 37,81 | 24,55 / 24,93 | 33,93 / 34,14 |
| 5080 tokenów | **269,06 / 272,42** | 554,50 / 556,56 | 535,49 / 537,84 | 775,97 / 777,41 |
| Cztery krótkie dokumenty | **21,66 / 59,22** | 44,95 / 88,25 | 35,09 / 55,18 | 69,07 / 143,21 |

Nawet bez interpretowania drobnych różnic seria jednoznacznie odrzuca
prototyp: jego mediany są większe niż w v4 i fast we wszystkich trzech
wierszach. Próg jakości i cel wydajności nie zostały osiągnięte łącznie.
Nie ogłaszamy poprawy p95 lub przewagi eager na całym krótkim korpusie.

Audyt pojedynczych krótkich przypadków (mediany, po 6 żądań na przypadek)
potwierdza regresję prototypu także poza agregatem:

| Przypadek | fast [ms] | v4 [ms] | prototyp [ms] |
|---|---|---|---|
| `pico-27-options` | 13,96 | 24,43 | 33,83 |
| `pico-bank-six-types` | 59,14 | 55,14 | 142,62 |
| `pico-long-prefix` | 26,00 | 44,75 | 95,03 |
| `pico-unicode` | 17,43 | 25,23 | 43,04 |

`pico-bank-six-types` ma kilka pytań API. Wynik nie jest czasem jednego
pytania i nie należy go mieszać z punktem 255 tokenów.

Historyczne punkty odniesienia v4 to 536,42 ms dla C1/5080 i 24,67 ms dla
C1/255 wobec 553,58/33,50 ms eager w ABBA. Osobny pomiar `fast` dał
270,19 ms dla 5080 tokenów. Nie zestawiamy tych osobnych kampanii jako
nowego potwierdzającego A/B; pełne dane i ograniczenia są w
[raporcie v4](../pico-parallel-rtx4060-2026-10-26/README.md).

Projekcje prototypu rozdzielają aktywację FP32 na część główną i resztę FP16;
wagi checkpointu BF16 są reprezentowane w FP16 po skalowaniu potęgą dwójki.
Dwa iloczyny cuBLASLt mają akumulację i wynik FP32, bez końcowego
zaokrąglenia do BF16. Attention liczy trzy iloczyny części FP16, pomijając
iloczyn obu reszt; softmax i wynik pozostają FP32. Blokowy DeltaNet stosuje
postać WY, stan FP32, blok 32 tokenów i stabilne rozwiązanie trójkątne.
Algorytm WY opisuje również projekt flash-linear-attention (MIT); generatory
w archiwum podają autorów i źródło równań. To przybliżenie FP32 wymagające
porównania wyników, a nie bitowa implementacja upstream FP32.

Eksperymentalne manifesty opisują `attention_products` jako `F32`; ten zapis
nie rozróżniał typu wyjścia od wejść iloczynów. Nie jest potwierdzeniem pełnej
arytmetyki FP32. Rzeczywiste warianty podano powyżej i zachowano w źródłach;
manifestów nie poprawiano poza usunięciem lokalnych ścieżek.

Osobna optymalizacja pobierania rekurencji zachowała bitowo wszystkie
elementy wyjścia i końcowego stanu względem starego kernela wide na
rzeczywistych operandach dokumentu 5080 tokenów, ale była wolniejsza.
Nie jest używana w końcowym prototypie ani w przywróconym v4.

Profil Nsight dla trzech forwardów wariantu z pobieraniem wykazał 1312,77 ms
projekcji, 854,20 ms attention i 496,22 ms rekurencji; to sumy czasów kerneli,
nie opóźnienia jednego żądania. W wariancie cuBLASLt suma dwóch głównych
rodzajów GEMM wyniosła około 958,17 ms, a rozdzielenie aktywacji 101,33 ms.
Zamiana pojedynczej operacji nie usuwa kosztu pozostałych bloków i przesłań.
Profile obejmują również ładowanie modelu; jego redukcji/castów nie należy
przypisywać do steady state.

## Decyzja i ograniczenia

Nie włączono profilu v5. Całość kodu wykonania przywrócono do wcześniejszego
v4, zachowując wcześniejszą obsługę Pico i jej pomiary. Nie ma nowej
domyślnej ścieżki pogarszającej czas krótkich dokumentów. **Nie oznacza to,
że v4 spełnia nowe warunki**: BF16 v4 ma większe różnice względem FP32
i pozostaje wolniejsze od jawnego `fast` w zbadanym C1.

Wątpliwe założenie tej serii brzmiało: samo użycie Tensor Cores z korekcją
reszty wystarczy, aby jednocześnie wyprzedzić BF16 `fast` i zachować bliskość
FP32 na RTX 4060. Pomiary tego nie potwierdziły. Przyszłe prace nad grafami
CUDA i fuzją operacji wymagają osobnej implementacji i pomiarów; nie
deklarujemy ich przyszłego wyniku.

Zmieniono tylko narzędzia pomiarowe: `omni-optimize.py --short-clients 1`
pozwala wydzielić C1 krótkich dokumentów; domyślnie pozostaje 1/8/32.
`summarize-omni.py` audytuje macierz zapisaną w `method.json`, zachowując
obsługę starszych archiwów bez tego pola. Raportuje też `short_case_summary`
dla każdego dokumentu i liczby klientów, aby agregat nie ukrywał regresji
pojedynczego wejścia. Brak nowych testów lub frameworków.

Eksploracje izolowanych kerneli nie potwierdzają jakości modelu. Część
pokrywała się z pracą kompilatora CPU; zachowujemy je jako SIGNAL, bez
wniosków o końcowym czasie HTTP. Brak pomiarów Metal, innych kart,
energii i pojemności serwera. Ustawienie 4 wątków nie rezerwuje fizycznie
rdzeni CPU przed innymi procesami środowiska.

## Artefakty i odtworzenie

Formatowanie, Clippy Linux oraz CUDA/all-features, rustdoc prywatnych
elementów, kompilacja Pythona, sprawdzenie wyjątku `paste` i build release
przeszły. Odbudowana binarka jest identyczna SHA-256 z zamrożonym v4:
`2d2bb90841958d4ad84f67581e922260e854dccbd8274cc7145769e2a2caba4b`.
Kontrola `cargo deny` ma **FAIL** dla istniejących, niezmienionych zależności
`zstd 0.13.3` i `zstd-safe 7.3.0`: RUSTSEC-2026-0391 oraz
RUSTSEC-2026-0395–0406. Nie rozszerzano tej optymalizacji o zmianę zależności
ani nie wyciszano advisory. `checks.json` zachowuje listę identyfikatorów.

Publiczna paczka obejmuje `control/`, `audit-control-per-case.json`,
`screen-*`, audyty, `quality-gate.json`, eksporty i porównania FP32,
`inputs/`, `diagnostics/`, `profiles/`, eksploracje kerneli oraz dokładne
źródła ostatniego odrzuconego prototypu w `source/tuned/`.
`REPRODUCE.md` w archiwum opisuje odtworzenie źródeł, eksportu i audytu.
Wagi, binarki, prywatne ścieżki, surowe profile i logi pozostają lokalnie.

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/pico-f32-fast-rtx4060-2026-10-26/dc59ed449a91742c55cdbcc44bd311250de88fcbe7d7b2c7568539f0c3bdfa90/measurement-data.tar.gz`.

SHA-256: `dc59ed449a91742c55cdbcc44bd311250de88fcbe7d7b2c7568539f0c3bdfa90`, rozmiar 602 232 B, 118 plików
w `SHA256SUMS`. Ponowne pobranie z autoryzacją i anonimowo potwierdziło
SHA-256 archiwum oraz **118/118 sum plików w obu kopiach**. Lokalna kopia
archiwum pozostaje zachowana. Dane są pod logiczną ścieżką
`reports/pico-f32-fast-rtx4060-2026-10-26/`.

Pobranie do nowego katalogu:

```bash
hf buckets cp 'hf://buckets/itsoltech/basal-rs/measurements/pico-f32-fast-rtx4060-2026-10-26/dc59ed449a91742c55cdbcc44bd311250de88fcbe7d7b2c7568539f0c3bdfa90/measurement-data.tar.gz' measurement-data.tar.gz
printf '%s  measurement-data.tar.gz\n' 'dc59ed449a91742c55cdbcc44bd311250de88fcbe7d7b2c7568539f0c3bdfa90' | sha256sum --check -
tar -xzf measurement-data.tar.gz
sha256sum --check SHA256SUMS
```

Audyt odtwarza się według archiwalnego `REPRODUCE.md`, wskazując katalog
wejść przez `--inputs`, a katalog wyniku przez `--measurement-artifacts`.
To opcja runnera, nie CLI `basal`. Końcowy audyt obsługuje też starszy
zestaw bez `short_clients`: ponowne przeliczenie starej kampanii krótkich
1/8/32 dało SIGNAL, 3/3 fazy, zero problemów.
