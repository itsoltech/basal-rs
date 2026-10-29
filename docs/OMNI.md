# Basal Omni: tekst, obraz i audio

Pico (`pico`, `Remek/basal-omni-pico`) jest modelem wyłącznie tekstowym,
z osobnym backendem Qwen3.5. Opis Gemma poniżej go nie dotyczy;
[użycie i ograniczenia Pico](PICO.md).

Katalog zawiera `omni-mini` (`Remek/basal-omni-1.0-mini`, Gemma 4) oraz `omni-max`
(`Remek/basal-omni-1.0-max`, Gemma 4 Unified). Rewizje są przypięte w konfiguracji;
[pierwotny audyt](../reports/upstream-2.0-discovery-2026-10-14/README.md) opisuje metadane.
Podstawowe checkpointy są BF16. Sam mini ma około 15,9 GB wag, max około 23,9 GB;
encoder, aktywacje i KV wymagają dodatkowej pamięci.

Omni wymaga CUDA. Dla obu wymienionych modeli na RTX 6000 Ada `dtype: auto`
wybiera dekoder FP16; pozostałe urządzenia zachowują BF16. Jawny `dtype` ma
pierwszeństwo. Przy dekoderze FP16 wieże mediów zachowują wagi, obliczenia i granice
clippingu BF16. `--dtype f32` wybiera obliczenia FP32 do porównań numerycznych.
Metal i Intel Vulkan nie uruchamiają omni.

### Tabela PLE w RAM hosta (opt-in)

`basal serve --model omni-mini --ple-placement host` trzyma tabelę tokenów
per-layer input embeddings na CPU i kopiuje wybrane wiersze na CUDA. W YAML
ustaw `models[].ple_placement: host`. Brak opcji lub `device` zachowuje
dotychczasowy loader, JSON planu i wejścia fingerprintu. Przy pliku konfiguracji
CLI odmawia `--ple-placement`; ustawienie należy do konkretnego modelu.
Nie zmienia to nazwy źródła, profilu ani aliasu serwowanego modelu.

Host wymaga CUDA i nie-unified Gemma z PLE. Llama, Pico/Qwen, Gemma bez PLE,
Unified, Metal, Intel i `layer_split` są odrzucane. Dostępna lokalnie lub w cache
architektura jest sprawdzana przed odkrywaniem GPU i wyborem GEMM. Plan zamraża
placement oraz limit `8192 × (4W + 4)` bajtów na wywołanie, gdzie
`W = liczba warstw × per_layer_input`. Limit obejmuje do 8192 padded wierszy
krótkiego batcha oraz chunki tekstu/mediów do 2048 wierszy. Backend sprawdza go
przed alokacją ID PLE. Dwie kopie storage i ID są konserwatywnym boundem;
wybrany tensor identity nadal zajmuje pamięć GPU.

Tabela zajmuje `2 × per_layer_vocab × W` bajtów na osobno ładowaną replikę.
Forki współdzielą tabelę, ale każdy worker main/long/zakwalifikowanego tła ma
osobny limit stagingu. Startup sprawdza ich sumę względem dostępnej pamięci
hosta/cgroup, ponownie po ładowaniu. Pool Batch pozostaje dodatkowym budżetem
planów i wyników; modele Host nadal podlegają istniejącemu HostStaging/admission.
Kwalifikacja tła wymaga własnych pomiarów i jawnego
`batch_profile.ple_placement: host`; samo włączenie Host nie kwalifikuje Batch.

Startup, `doctor --config` i `benchmark --ple-placement host` rozdzielają
rozmiar checkpointu, szacunek źródłowych wag na GPU, tabelę w RAM, mapowania,
scratch ładowania i staging workerów. Źródłowy szacunek GPU odejmuje tylko
sprawdzony zakres tabeli PLE od rozmiaru checkpointu; nie używa estymatora split.
Scratch i mapowania nie są stałą rezydencją tabeli. Są to szacunki źródłowe,
bez gwarancji alokatora/OOM całego procesu ani zmierzonej oszczędności VRAM.
Zgodność bitowa Device/Host, jakość, transfery, rzeczywiste szczyty pamięci
i wydajność pozostają niezweryfikowane runtime'owo.
Szacunek Host w `doctor` wymaga uruchomienia binarki z włączoną obsługą CUDA;
doctor bez CUDA odmawia tej opcji także przy obecnej osobnej binarce CUDA.

Na wielu GPU omni działa na wskazanej karcie (`device`) albo jako niezależne repliki
(`replica_devices`, karty jednego typu). Przy replikach bez pasującej tabeli GEMM każda
replika używa stałych GEMM per prompt. Podział warstw (`layer_split`) obsługuje Gemma bez
współdzielonego KV między warstwami i bez per-layer input embeddings, w tym Ultra.
Działa dla tekstu i obrazów: enkodery mediów na pierwszej karcie, lokalne maski i KV
na kartach etapów, końcowa norma i powiązany embedding odczytu na ostatniej.
Wymaga CUDA, kart jednego typu, jawnego `dtype: f16`, `quantization: none`
i pokrycia wszystkich warstw. Dla Ultra: `--media --layer-split 0:30,1:30`.
Inne konfiguracje Gemma oraz kwantyzacja inna niż `none` kończą start jawnym błędem
przed wagami. Samo wsparcie w kodzie nie potwierdza numeryki ani wydajności;
referencją pozostają raporty z konkretnych kart i wersji.
[Pomiar Ultra na dwóch GPU](../reports/ultra-multigpu-2026-10-18/README.md)
obejmuje tekst/obrazy na dwóch Ada 48 GB, kontrolę umieszczenia na H100 i repliki.
Zachowuje jedną różną decyzję względem FP32 na Ada oraz różnice liczbowe C128.

`inference_mode: auto` wybiera iloczyny attention FP16 tylko dla tych dwóch modeli,
dekodera FP16 i rozpoznanej RTX 6000 Ada. Softmax i akumulacja pozostają FP32.
`accurate`, jawne BF16/FP32 i inne urządzenia używają iloczynów FP32. Sam `accurate`
nie zmienia precyzji całego dekodera. Zakres pomiarów, odległość od FP32 i ograniczenia
opisuje [kampania Ada](../reports/omni-performance-ada-2026-10-16/README.md).

```bash
basal serve --model omni-mini --media --long-slice-ms 10
basal client --local --model omni-max --media --request request.json --max-input-bytes 67108864
basal export --upstream --model omni-mini --media --dtype bf16 --inputs .cache/reference-omni --out .cache/native-omni
```

W konfiguracji YAML opcja należy do modelu:

```yaml
models:
  - repo: Remek/basal-omni-1.0-mini
    revision: 73c993dc3d9ab3cb8887ac0b1ee1cf1c92783240
    dtype: auto
    gemm_table: auto
    media: true
    orders: "1"
    state_cache_mb: 0
```

Żądanie do `/v1/basal` lub `/v1/systemone` zawiera media przed tekstem stanu,
w podanej kolejności. `data` przyjmuje base64 albo data URL; runtime nie pobiera URL.

```json
{
  "model": "basal-omni-1.0-mini",
  "state": "Sprawdź dostarczone materiały.",
  "media": [
    {"type": "image", "data": "<base64 obrazu>"},
    {"type": "audio", "data": "<base64 audio>"}
  ],
  "detail": "default",
  "questions": {"q": {"type": "noul", "instructions": "Czy materiały opisują to samo?"}}
}
```

Obrazy są konwertowane do RGB i skalowane z zachowaniem proporcji. `default` ma
budżet 280 soft tokenów, `high` 1120; faktyczna liczba zależy od proporcji obrazu.
Obsługiwane dekodery: PNG, JPEG, GIF (pierwsza klatka), WebP, BMP, TIFF, ICO, PNM i TGA.
Audio WAV, FLAC, OGG/Vorbis i MP3 jest mieszane do mono i resamplowane do 16 kHz.
Akceptowane częstotliwości wejścia to 1–384000 Hz, najwyżej 64 kanały.
Mini używa cech mel i wieży Conformer; max ramek po 640 próbek i projekcji do dekodera.
Pomiary zgodności poszczególnych formatów należy sprawdzać w raportach; wspólny
format wejścia nie gwarantuje bitowej identyczności różnych dekoderów stratnych.

Limity: osiem elementów, 20 MiB na element, 50 mln pikseli, do 30 sekund audio,
256 MiB cech procesora na żądanie oraz kontekst z `config.json`. Serwer z włączonymi
mediami przyjmuje body do 256 MiB. Najwyżej cztery duże/chunkowane body i cztery
żądania z mediami pozostają w toku; przeciążenie daje 529 z `Retry-After: 1`.
Globalny limit `max_inflight` obejmuje także zadania w kolejce i podczas forwardu
po anulowaniu HTTP. Permity pozostają przy dekoderze/kolejce do końca tego zadania. Duże JSON-y
są parsowane poza executorem Tokio; dekodowanie mediów odbywa się w workerze.
Domyślny limit wejścia klienta pozostaje 2 MiB; `--max-input-bytes` pozwala jawnie
podnieść go do 64 MiB. Limit body HTTP serwera wynosi 256 MiB.

Encodery wykonują się raz na żądanie. Ich wyniki są współdzielone przez pytania
i kolejności tego żądania, usuwane także po błędzie; repliki workerów mają osobne
sloty. Dane mediów nie trafiają do logów ani stałego cache. Cache prefiksu/stanu
i profil sekcji Llamy są dla omni niedostępne.

Projekcje dekodera mogą korzystać z tabel cuBLASLt zapewniających zgodny bitowo
wynik między rozmiarami batcha. Build zawiera tabele FP16 obu modeli dla RTX 6000
Ada i cuBLASLt 120901. `gemm_table: auto` sprawdza urządzenie, bibliotekę, precyzję
i kształty projekcji. Bez pasującej tabeli Gemma używa osobnych GEMM dla promptów;
nie uruchamia automatycznego poszukiwania algorytmów. Tabelę można przygotować
przez `basal gemm-search --invariant` i przekazać jawnie. Attention zachowuje osobne
iloczyny każdego promptu, aby jego odpowiedź nie zależała od pozostałych żądań.

W profilu FP16 max na Ada serwer po partii równoległej zbiera powracające żądania
przez najwyżej 10 ms. Po co najmniej 100 ms bezczynności i przy gotowej kolejce
startuje od razu. Mini nie dodaje takiego oczekiwania. Dla ruchu mieszanego
`--long-slice-ms 10` skraca czas oczekiwania małych żądań na GPU; jest to jawne
ustawienie procesu, którego domyślna wartość nadal wynosi 100 ms. W YAML należy
do poziomu głównego (`long_slice_ms`), a nie do pojedynczego modelu. Rozkłady czasów
krótkich i długich żądań oraz wpływ tego ustawienia trzeba czytać osobno.

W kampanii Ada większy QPS tekstowego miksu i niższe globalne p95 dało FIFO bez
osobnego toru długich. Ten profil wybiera się jawnie:

```bash
basal serve --model omni-mini --dtype auto --gemm-table auto --schedule fifo --long-tokens 0
basal serve --model omni-max --dtype auto --gemm-table auto --schedule fifo --long-tokens 0
```

Przy N=8/32 osiągnął 1,72–1,80× QPS sparowanego upstream eager BF16. Krótkie
żądania tracą część priorytetu względem HRRN; p95 samych długich przy N=32 nadal
jest wyższe od upstream o około 3,4% mini i 2,8% max. Pomiar używa dwóch
powtarzanych długich kontekstów oraz 128 różnych krótkich dokumentów. Nie wyznacza
maksimum QPS ani produkcyjnego SLO; pełne wyniki, zakresy i kontrola bitowej
zgodności są w [raporcie Ada](../reports/omni-performance-ada-2026-10-16/README.md).

W Gemma 4 Unified uwaga lokalna jest dwukierunkowa wewnątrz jednego bloku obrazu;
uwaga globalna pozostaje przyczynowa. Prefill nie rozcina takiego bloku obrazu.
Mini używa osobnych encoderów obrazu i audio oraz embeddings per-layer w dekoderze.
Obie architektury stosują RMSNorm z bezpośrednimi wagami i softcap logitów 30.

Odpowiedź z mediami ma `calibration: "text"`; niedopasowana kalibracja kolejności
pozostaje `unvalidated`. Kalibracja tekstowa nie stanowi pomiaru jakości mediów.
`usage.media` zawiera liczbę obrazów/audio, detail i soft tokeny wszystkich wykonanych
promptów. Brak głowicy evidence i nieobsługiwane opcje kończą się jawnym błędem.
Thinking, early exit, MCP server oraz trenowanie kalibracji nie należą do tego portu.

Zaobserwowaną zgodność i ograniczenia numeryczne opisuje [raport H100](../reports/upstream-2.0-h100-2026-10-14/README.md).

[Końcowa kampania](../reports/upstream-2.0-final-2026-10-14/README.md) obejmuje po 38/38 zgodnych wyborów każdego omni na rzeczywistych obrazach, mowie WAV/FLAC/OGG/MP3 i wejściach mieszanych oraz po 2/2 na skonstruowanym oknie 30 s. Są to wybory względem FP32 i ustalonych etykiet, nie pomiar ASR/WER ani walidacja kalibracji mediów. Długi kontekst nie jest potwierdzony w całym limicie config: mini 131072 i max 8192/262144 przekroczyły limit czasu procesu eksportu. Native omni max było też wolniejsze od upstream eager w opisanym krótkim obciążeniu tekstowym.

Późniejsza [kampania Ada](../reports/omni-performance-ada-2026-10-16/README.md)
potwierdziła przewagę wybranego Auto FP16 nad działającym upstream eager BF16:
niższą medianę w 36/36 punktach tekstowych 128–32768 i 68/68 przypadkach mediów
łącznie dla obu modeli oraz wyższe mediany QPS. Wszystkie 398 decyzji kontrolnych
zgadzają się z FP32; średnia TV jest mniejsza niż upstream BF16. Zakres jakości
FP32, pojedyncze regresje i ograniczenia kolejki pozostają opisane osobno.
