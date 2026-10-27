# Pico: precyzja pojedynczego FP16 i usunięcie kopii DeltaNet

## Cel i zakres

Sprawdzić propozycje z analizy Astra i Opus: najpierw budżet precyzji
projekcji, następnie usunąć ponowne składanie gotowego bufora DeltaNet.
Wymagania: zgodne decyzje i maksymalna bezwzględna różnica skalibrowanego
prawdopodobieństwa ≤0,0001 względem upstream FP32, przewaga nad upstream
fast również dla krótkich dokumentów. Cel łączny pozostaje nieosiągnięty.

RTX 4060 8 GiB, SM89, CUDA 12.9.1, bez zmiany zegarów i limitu mocy.
Checkpoint `Remek/basal-omni-pico`, rewizja
`a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`;
SHA-256 wag `ac52f85bf4af603b2d2408d314e9c1bb59e351884f7cacd72464c49dca9149e9`.
Upstream 2.4.0, commit `615fe321a818fdf5d2897f758b8d823f919d7f94`,
Torch 2.11, Transformers 5.17, FLA 0.6, Triton 3.6.
Kod lokalny bazuje na `db55af9d7cab5f12a18347b0bb27eb4dab4b1531`
z niezatwierdzoną implementacją Pico. Źródła wariantów są w archiwum;
sam commit bazowy ich nie odtwarza.

## Eksperyment 1: budżet precyzji — FAIL

Wariant diagnostyczny zaokrągla aktywacje przed projekcjami do FP16/BF16,
a następnie rozszerza je do FP32. GEMM, wynik i pozostałe operacje pozostają
FP32; rezydentne wagi FP32 są identyczne z poprzednią ścieżką.
Klasyfikacja projekcji używa tożsamości tensorów, a nie ich wymiarów.
To emulacja utraty precyzji aktywacji, **nie implementacja Tensor Cores ani
pomiar ich szybkości**. Nie odtwarza innej kolejności akumulacji GEMM,
skalowania aktywacji ani korekcji resztowej.

`basal export`, jedna kolejność, 44 pytania i 12 gałęzi liczbowych API;
porównanie z zamrożonym upstream FP32:

| Zaokrąglane projekcje | Maks. różnica benchmark | Maks. różnica API | Próg |
|---|---:|---:|---|
| wszystkie, FP16 | 0,00240674615 | 0,00013184547 | FAIL |
| tylko MLP, FP16 | 0,00063198805 | 0,00014185905 | FAIL |
| tylko attention, FP16 | 0,00098754466 | 0,00005355477 | FAIL |
| tylko DeltaNet, FP16 | 0,00129556656 | 0,00021570921 | FAIL |
| wszystkie, BF16, kontrola | 0,01622098684 | 0,00333356857 | FAIL |

W każdej próbie argmax benchmarku 44/44 i liczbowych gałęzi API 12/12
pozostał zgodny. To nie wystarcza do spełnienia progu prawdopodobieństw.
Nie uruchamiano pomiarów wydajności tych odrzuconych wariantów ani orders=2
i długich granic: dyskwalifikacja nastąpiła już na podstawowym zestawie.
Kod emulacji zachowano w artefaktach i usunięto z bieżącej implementacji.
Wyniki nie odrzucają wszystkich możliwych strategii mieszanej precyzji.

## Eksperyment 2: gotowy bufor rekurencji

`qwen35_prepare` tworzy już ciągłe płaszczyzny Q/K/V i decay/beta.
Wariant `BASAL_QWEN_PACKED_RECURRENCE=1` przekazuje ich niemutowalne widoki
bezpośrednio do istniejącego kernela rekurencji, zamiast wykonywać dwa
`Tensor::stack`. Typy, kolejność operacji, stan i checkpointy pozostają
takie same. Widoki posiadają współdzieloną alokację przez czas wywołania;
kernel sprawdza kształty, dtype, ciągłość i urządzenie.

Wariant jest jawnie włączany i zapisywany w tożsamości planu. Profil v4,
domyślny BF16 i dotychczasowy FP32 pozostają bez automatycznej zmiany.
Nie jest to blokowy algorytm WY ani CUDA Graphs.

### BF16: potwierdzenie ABBA

HTTP C1, FIFO, brak cache, budżet 8192 tokeny, OMP/MKL/Rayon=4.
Osiem uruchomień w kolejności A/B/B/A/A/B/B/A; pierwsze uruchomienie
każdego wariantu odrzucono. Każde uruchomienie ma ≥10 s rozgrzewki,
dwa żądania rozgrzewki przed punktem drabiny, 12 próbek 255 i 5080
tokenów oraz 32 krótkie żądania, po osiem na dokument.
Tabela podaje medianę median trzech zachowanych uruchomień.
Audyt wejść, odpowiedzi i liczby próbek: `AUDITED`, zero problemów.

| Dokument | Poprzedni v4 BF16, ms | Bez ponownego stack, ms |
|---|---:|---:|
| 5080 tokenów, osobny punkt C1 | 536,67 | 518,77 |
| 27 opcji, krótki korpus | 24,42 | 23,86 |
| bank, sześć typów pytań | 54,97 | 53,12 |
| wspólny prefiks | 44,65 | 43,48 |
| Unicode | 25,14 | 24,38 |

5080 tokenów: **3,33% mniej czasu**. Zakresy median uruchomień:
A 536,14–536,75 ms, B 517,22–518,86 ms.
Osobny punkt 255 tokenów: 24,53 → 23,95 ms.
Każdy krótki dokument przyspieszył wobec poprzedniego natywnego BF16;
nie jest to potwierdzenie przewagi wobec upstream fast ani zgodności BF16
z wymaganym progiem FP32. Wcześniejszy fast około 270 ms pochodzi
z innej kampanii, bez nowego bezpośredniego ABBA w tej próbie.

### FP32: potwierdzenie ABBA

Ta sama metoda, obie strony z `BASAL_QWEN_F32_RESIDENT=1` i bez tabeli
GEMM. Jedynym aktywnym dodatkiem kandydata jest gotowy bufor rekurencji.
Audyt: `AUDITED`, zero problemów.

| Dokument | Rezydentne FP32, ms | Bez ponownego stack, ms |
|---|---:|---:|
| 5080 tokenów, osobny punkt C1 | 1570,53 | 1547,62 |
| 27 opcji, krótki korpus | 69,48 | 68,16 |
| bank, sześć typów pytań | 165,52 | 162,67 |
| wspólny prefiks | 133,88 | 131,74 |
| Unicode | 63,70 | 62,64 |

5080 tokenów: **1,46% mniej czasu**. Zakresy median:
A 1570,43–1571,13 ms, B 1546,99–1547,92 ms.
Zysk dotyczy obu precyzji, bez nowej regresji czterech krótkich dokumentów
wobec własnej poprzedniej wersji. Dla FP32 nadal nie osiągnięto czasu fast.

Wariant pomiarowy zawierał wyłączoną diagnostykę emulacji; końcowy build
usuwa tę diagnostykę. Sumy binarek w metodzie ABBA dotyczą właśnie
wariantu mierzonego, nie końcowego builda. Eksporty końcowego builda są
osobno porównywane z eksportami wariantu mierzonego i poprzedniej wersji.

### Kwalifikacja końcowego kodu

- FP32 orders=1 i orders=2: benchmark 44/44 i API 12/12 zgodnych decyzji;
  największa różnica skalibrowanego prawdopodobieństwa wobec upstream FP32
  wynosi **0,000005051494**, poniżej 0,0001.
- FP32 granice: 5080 tokenów i 255 opcji, 2/2 zgodne decyzje,
  maksymalna różnica 0,000004529953. Odmowa 256 opcji zachowana.
- `cmp` potwierdził bitowo identyczne pliki `bench.jsonl` i `systemone.jsonl`
  dla FP32 orders=2 i granic oraz BF16 orders=1/2 i granic względem
  poprzednich wariantów. Końcowe FP32 orders=2 jest też bitowo identyczne
  z wariantem użytym w ABBA. Domyślne FP32 i BF16 bez nowej flagi zachowują
  poprzednie eksporty bitowo. Dotyczy to badanych wejść, nie wszystkich
  możliwych dokumentów.
- Build CUDA, formatowanie, Clippy Linux i CUDA all-features, Rustdoc
  Linux i CUDA all-features z prywatnymi elementami i `-D warnings`,
  kompilacja składni runnerów Python oraz kontrola wyjątku `paste`: PASS.
  Przejrzano ownership widoków, offsety, urządzenia, walidację kształtów,
  checkpointy i powierzchnię API według `rust-review` i `itsol-self-review`.
- `cargo deny`: **FAIL**, wcześniej obecne advisory dla `zstd 0.13.3`
  i `zstd-safe 7.3.0`: RUSTSEC-2026-0391 i RUSTSEC-2026-0395–0406.
  `Cargo.lock` i zależności nie zostały zmienione. Duplikaty z `cargo tree`
  są istniejącymi zależnościami upstream. Nie dodano wyciszeń advisory.

## Profil rzeczywistych żądań HTTP

Osobne, rozgrzane żądania C1: 5080 tokenów, 255 tokenów, bank z wieloma
pytaniami i dokument ze wspólnym prefiksem. Natywny BF16 v4, FIFO,
8192 tokeny budżetu, bez cache; dwa żądania rozgrzewki przed każdym profilem.
Nsight Systems 2026.3.2, CUDA/NVTX, bez próbkowania CPU.
To pojedyncze profile diagnostyczne z narzutem narzędzia; ich czasy
nie zastępują serii ABBA.

Aktywność GPU jest sumą długości **unii** przedziałów kerneli, memcpy
i memset; nakładających się operacji nie liczono dwukrotnie.
Surowe profile i SQLite pozostają prywatne; publiczne CSV zawierają sumy
według nazw kerneli i aktywność GPU bez danych maszyny.

| Żądanie | Aktywność GPU, ms | Przedział GPU, ms | Udział |
|---|---:|---:|---:|
| 5080 | 548,37 | 551,71 | 99,4% |
| 255 | 25,75 | 26,87 | 95,8% |
| bank | 48,84 | 55,75 | 87,6% |
| wspólny prefiks | 41,75 | 44,74 | 93,3% |

Nie jest to occupancy ani wykorzystanie jednostek obliczeniowych.
W długim żądaniu rekurencja zajęła 141,22 ms, `copy2d_f32` 20,14 ms,
a `ucopy_bf16` 57,04 ms. Profile dotyczą poprzedniego wariantu,
nie ścieżki bez ponownego stack. Przypisanie wszystkich kopii pojedynczemu
miejscu w kodzie byłoby nieuprawnione. Krótkie żądania również mają wysoki
udział aktywności GPU: samo zakończenie enqueue wcześniej nie dowodzi
dominacji narzutu CPU. Sam replay grafu nie usuwa kosztu kerneli.

Próby uruchomienia upstream fast pod interaktywnym profilerem nie udostępniły
endpointu health w czasie obserwacji; zakończono procesy. **GAP:** brak
profilu fast; nie przypisujemy jego przewagi zmierzonym kategoriom kerneli.
Nie ustalono przyczyny zatrzymania startupu pod tym profilerem.

## Odtworzenie i artefakty

Dane mają logiczny katalog `reports/pico-singlepass-rtx4060-2026-10-27/`:
`quality/` zawiera eksporty i porównania, `references/` zamrożone referencje,
`http/` obserwacje i audyty obu serii ABBA, `profiles/` agregaty CSV,
`inputs/` korpus i tabelę GEMM, `source/` źródła, a `validation.json`
statusy kontroli i sumy binarek. `REPRODUCE.md` opisuje odtworzenie
eksperymentów i audytu przez `--measurement-artifacts`.

Końcowy build: SHA-256
`514f08cde1874fa13c4676d659c5e49a7201d1286dee894e10fd102ad2970a81`.
Binarka pomiarowa kandydata:
`07be1d9a5b453f6f44ae57ba552231e0429590b63f3ad5cff4f139255bc5ea64`;
baza:
`fb1799eb15f6a66737f6e3ff48924bb8c6fa08a68051245ebf2a96cd6496c0d2`.

Publiczne archiwum:
`hf://buckets/itsoltech/basal-rs/measurements/pico-singlepass-rtx4060-2026-10-27/7ba3c8e317810fbfa02914b549d7460e07ef43a40ebeff0ce3281c65ecee31e4/measurement-data.tar.gz`.
SHA-256 `7ba3c8e317810fbfa02914b549d7460e07ef43a40ebeff0ce3281c65ecee31e4`,
1 023 267 B, 128 plików. Ponowne pobranie z logowaniem i anonimowo
potwierdziło SHA archiwum i 128/128 wpisów `SHA256SUMS` w obu kopiach.
Anonimowe środowisko potwierdziło brak logowania przed weryfikacją.
Lokalna kopia archiwum została zachowana; publicznego prefiksu nie nadpisano.

Pobierz przez `hf buckets cp` do nowego katalogu, sprawdź powyższą sumę
archiwum, rozpakuj i uruchom `sha256sum -c SHA256SUMS`. Następnie stosuj
`REPRODUCE.md`; audyt HTTP używa `--measurement-artifacts` dla odpowiedniego
katalogu `http/`. Archiwalne `SUMMARY.md` jest kopią raportu sprzed
dopisania tej noty publikacyjnej.
