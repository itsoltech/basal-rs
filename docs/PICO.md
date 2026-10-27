# Basal Pico (Qwen3.5)

```sh
basal serve --model pico
basal init --model pico
# Referencja obliczeń FP32; ten sam checkpoint i protokół:
basal serve --model pico --dtype f32
```

Aliasy `pico` i `omni-pico` wskazują
[`Remek/basal-omni-pico`](https://huggingface.co/Remek/basal-omni-pico),
rewizję `a959cef1ea65fb4824e86df3d9d6b38a0db0f4a9`. Wagi zajmują około
1,50 GB na dysku. Nazwa `basal-2.0-pico` z wcześniejszej
[zapowiedzi](https://basal.si5.pl/blog/basal-pico/) nie jest nazwą opublikowanego
repozytorium. Pico ma 0,8B parametrów i przyjmuje wyłącznie tekst, mimo
`omni` w nazwie. Domyślny model serwera pozostaje bez zmian.

## Kontrakt i wykonanie

Pico używa `Qwen3_5ForCausalLM` (`qwen3_5_text`): przeplata pełne attention
z Gated DeltaNet. Natywny backend Candle obsługuje CUDA oraz Metal; nie
korzysta z Pythona. `dtype: auto` wybiera BF16 na CUDA i F32 na Metal.
Jawne `f16` i `f32` są dostępne na obu backendach; BF16 na Metal jest odrzucane.
Na RTX 4060 FP32 i BF16 zachowały decyzje
upstream FP32 w zbadanych zestawach: 44 pytania, 12 gałęzi API, osobno 255
opcji i kontekst 5080 tokenów. Jedna i dwie kolejności mają osobne porównania;
liczby, różnice prawdopodobieństw i ograniczenia zawiera
[pierwszy raport](../reports/pico-rtx4060-2026-10-26/README.md).
Profil v2 ma osobny [pomiar wydajności i numeryki](../reports/pico-speed-rtx4060-2026-10-26/README.md).
Profil v3 ma osobne [porównanie C1 dla 5080 tokenów](../reports/pico-long-c1-rtx4060-2026-10-26/README.md).
Profil v4 ma osobne [porównanie z upstream eager i fast](../reports/pico-parallel-rtx4060-2026-10-26/README.md):
C1/5080 w ABBA wyniósł 536,42 ms wobec 553,58 ms eager (3,10% mniej czasu).
Eksperyment [bliskości FP32 i szybkości fast](../reports/pico-f32-fast-rtx4060-2026-10-26/README.md)
spełnił na zbadanych wejściach próg różnicy prawdopodobieństwa ≤0,0001,
ale pogorszył czas krótkich i długich dokumentów. Prototyp odrzucono;
profil v4 pozostaje domyślny i nie spełnia tych nowych wymagań łącznie.
Osobny [audyt upstream fast względem FP32](../reports/pico-fast-fp32-audit-rtx4060-2026-10-26/README.md)
potwierdził, że około 270 ms dotyczy BF16 i przekracza próg ≤0,0001,
mimo zgodnych decyzji na zbadanych wejściach. Nie jest to czas pełnego FP32.
Próby [budżetów CPU, CUDA Graphs i rezydentnych wag FP32](../reports/pico-cpu-graphs-rtx4060-2026-10-26/README.md)
mają osobny raport. Limity 1/2/4/8 wątków nie dały wyraźnej poprawy C1/5080;
prototypy grafów MLP i globalnego fragmentu FP32 512 nie są częścią profilu.
Zgodność decyzji nie oznacza identycznych prawdopodobieństw; v2 zwiększa część
różnic BF16 względem FP32. Metal i pozostałe GPU
nie mają potwierdzenia runtime'u. Limit pozycji z konfiguracji nie jest
potwierdzeniem jakości ani pojemności takiego kontekstu.

Stan konwolucji i rekurencji DeltaNet oraz K/V attention istnieją tylko podczas
bieżącego wywołania backendu. Wspólny prefiks może być policzony raz;
każda gałąź kontynuuje jego niemutowalny snapshot i zapisuje własny nowy stan.
Stan po wykonaniu rodzeństwa nie przechodzi do kolejnej gałęzi ani żądania.
CUDA BF16/F16 w v3/v4 używa fragmentów do 512 tokenów dla pełnego promptu do
2048 tokenów i do 2048 dla dłuższych promptów; FP32 i Metal zachowują 64.
Rekurencja, normalizacja i akumulacja attention używają FP32.
CUDA łączy konwolucję, przygotowanie DeltaNet, jego rekurencję, normy i SwiGLU
w dedykowane kernele. Attention dla głów 256 i kontekstu do 2048 tokenów ma
stałą kolejność obliczeń dla każdej pozycji i softmax online w FP32.
Dłuższy pełny prompt w v3 oraz v4 `accurate` od początku używa iloczynów Candle w FP32, kafelków
wyników do 64 MiB oraz wspólnego kernela maskowania przyczynowego i softmax
na GPU. Nie tworzy gęstej maski na CPU. Wybór ścieżki i granic fragmentów
dotyczy pełnego promptu: pozostaje taki sam w partii i przy kontynuacji prefiksu.
Profil v4 `auto` na CUDA używa dla długiego promptu Q/K w typie modelu,
iloczynu cuBLAS z akumulacją i wynikiem FP32 oraz softmaksu FP32.
Prawdopodobieństwa są zaokrąglane do typu modelu przed iloczynem P/V,
akumulowanym w FP32 z wynikiem w typie modelu. Zapis softmaksu łączy to
zaokrąglenie z kernelem redukcji. Jest to zmiana numeryki względem v3.
Rekurencja v4 przetwarza dwie niezależne kolumny wartości w jednym warpie,
zachowując kolejność operacji FP32 w każdej kolumnie i niemutowalne snapshoty.

Tabela GEMM zakwalifikowana jako bitowo zgodna między rozmiarami partii pozwala
łączyć projekcje do ośmiu promptów i współdzielić dowolny wspólny prefiks do
2048 tokenów. Bez pokrycia tabelą projekcje pozostają osobne, a współdzielenie
obejmuje pełne fragmenty. BF16 na RTX 4060 z cuBLASLt 12.9.1 automatycznie
otrzymuje tabelę 48 wpisów z profilu; bez pasującej tabeli nie ma autotuningu.
`basal export` wymaga `--gemm-table auto`, aby odtworzyć domyślny wybór serwera.
Odczyt liczy wybrane wiersze współdzielonych embeddings w FP32.
`A_log` i normy DeltaNet pozostają FP32 zgodnie z checkpointem.

Tokenizer zachowuje NFC i byte-level BPE. Prompt używa ChatML bez BOS,
z zamkniętym pustym blokiem `<think>` przed prefiksem odpowiedzi.
Loader sprawdza hash obsługiwanego szablonu, architekturę, kształty i typy wag.
Publikacja Pico zachowuje prefiks wag `model.language_model.`; loader rozpoznaje
go oraz standardowy tekstowy prefiks `model.`, bez mieszania obu układów.
Format 2, 2–255 opcji, typy `choice`, `noul`, `score`, `multi`, `act`, `tool`
i wybór `CALIBRATION.json` / `CALIBRATION.o2.json` korzystają ze wspólnego
[protokołu](UPSTREAM_2.0.md).

Profil `basal-omni-pico.v4` wybiera `qwen35-dispatch.4` i osobny zestaw
`basal-qwen35.4`. Nie dziedziczy tabel ani zmierzonych reguł Gemma lub Llama.
Zestawy v2/v3 pozostają zaimplementowane; ich historyczne pomiary nie opisują
nowej ścieżki długiego kontekstu. Krótkie prompty zachowują obliczenia v2.
Nieznany model tej architektury zachowuje referencyjny `qwen35-generic.v1`
z `qwen35-dispatch.1` i `basal-qwen35-reference.1`.

## Diagnostyka i rezydentne projekcje FP32

```sh
BASAL_QWEN_F32_RESIDENT=1 basal serve --model pico --dtype f32
```

To jawny wariant CUDA FP32: projekcje BF16 są rozszerzane raz przy ładowaniu,
zamiast przed każdym iloczynem. Embeddings pozostają kompaktowe; rozszerzane
są wybrane wiersze. Rośnie zużycie VRAM, a kształty i operacje FP32 pozostają
takie same. Wariant jest domyślnie wyłączony, bez zmiany `dtype: auto`,
profili BF16, Metal i innych modeli. Kwalifikację i pomiary zawiera powyższy
raport; nie jest to potwierdzenie przewagi nad upstream fast.

Kod backendu jest rozdzielony na prywatne moduły `qwen35/`: ładowanie,
attention, DeltaNet, prefill i prefiksy. Operacje CUDA Pico są w
`fused/qwen35.rs`. Refaktor zachowuje profile i arytmetykę; jego kwalifikację
przed/po opisuje [raport refaktoru](../reports/pico-refactor-rtx4060-2026-10-27/README.md).

`BASAL_QWEN_PACKED_RECURRENCE=1` włącza osobny eksperyment CUDA:
rekurencja DeltaNet czyta gotowe płaszczyzny Q/K/V i bramek bez ponownego
`Tensor::stack`. Arytmetyka kernela i checkpointy pozostają takie same;
flaga jest zapisywana w tożsamości planu. W FP32 można ją łączyć z
`BASAL_QWEN_F32_RESIDENT=1`. Kwalifikację bitową, pomiary każdego krótkiego
dokumentu oraz odrzuconą emulację FP16 opisuje
[raport z 27 października](../reports/pico-singlepass-rtx4060-2026-10-27/README.md).

`BASAL_PHASE_TIMING=1` zapisuje na stderr czasy planowania CPU, enqueue Pico
i odczytu z oczekiwaniem. Enqueue obejmuje również blokowanie przez GPU;
tych czasów nie należy traktować jako rozłącznych kosztów CPU/GPU.
`BASAL_CUDA_PROFILE=1` otwiera zakresy cuProfilerApi wokół forwardów Pico,
poza ładowaniem i planowaniem. Profilowanie nie służy do pomiarów HTTP.
Flagi są przechwytywane raz na proces i uwzględniane w tożsamości planu.
Runner `tools/perf/omni-optimize.py` przyjmuje jawne, zapisywane w metodzie
`--candidate-diagnostic phase-timing`, `qwen-f32-resident` lub
`qwen-packed-recurrence`; opcja `--baseline-diagnostic qwen-f32-resident`
lub `qwen-packed-recurrence` pozwala porównać wersje kodu z takimi samymi
flagami optymalizacji po obu stronach. Odziedziczone
zmienne `BASAL_*` usuwa, aby ograniczyć niejawne różnice między pomiarami.

## Ograniczenia

- Bez CPU, Intel Vulkan, GGUF, FP8 i kwantyzacji projekcji: `quantization: none`.
- Bez mediów, evidence, podziału warstw między GPU i wykonania batchy w tle.
  Repliki na osobnych urządzeniach używają własnych stanów i wspólnych zasad serwera.
- Bez cache prefiksów i stanu między żądaniami; `state_cache_mb` musi wynosić `0`.
- `inference_mode: accurate` zachowuje iloczyny attention FP32; `auto` w v4
  używa wyżej opisanego wariantu Tensor Cores dla długiego promptu CUDA BF16/F16;
  `balanced` jest odrzucane. W profilach v2–v4 można wskazać kwalifikowaną tabelę
  GEMM; profil ogólny v1 przyjmuje tylko `auto` lub `none`.
- Rekurencja CUDA pozostaje sekwencyjna po tokenach. Porównanie z upstream
  zależy od długości dokumentu, liczby pytań i trybu upstream; wyniki i regresje
  zawiera raport wydajności. Nie ma pomiarów na innych GPU ani w F16.

Zgodność należy oceniać według [BENCHMARKS.md](BENCHMARKS.md), w granicach
zbadanych wejść i sprzętu. Raport RTX 4060 obejmuje kilka pytań, obie
kolejności oraz końcowe fragmenty 0–3 tokenów po granicy 64 tokenów.
Nowe pomiary wymagają nowego raportu i osobnych artefaktów.
Implementacja odwzorowuje
[Transformers 5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py);
różnice redukcji i zaokrągleń wymagają pomiaru, także w BF16.
