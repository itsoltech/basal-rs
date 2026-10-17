# Intel Arc na Linux

Backend `intel` wykonuje inferencję Basal 1.5 mini, 4.5B i max przez Vulkan,
z własnymi kernelami WGSL i `wgpu 30.0.1`. Forward używa wag i aktywacji FP16,
akumulacji GEMM w FP32, attention/RoPE w FP32 i odczytu logitów liter w FP32.
Nie wymaga Pythona, CUDA, OpenCL ani oneAPI.

Pomiary na zintegrowanym Arc Meteor Lake i ograniczenia ich zakresu opisuje
[raport](../reports/intel-arc-meteor-lake-2026-10-10/README.md).
Nowsze [porównanie GEMM i HTTP przed/po](../reports/intel-arc-meteor-lake-gemm-2026-10-11/README.md)
obejmuje optymalizację kernela z zachowaniem kolejności akumulacji FP32.
Inne generacje i osobne karty Arc wymagają własnej weryfikacji. Ten backend
nie korzysta z XMX; nie jest dowodem maksymalnej wydajności wszystkich kart Intel.

## Sprawdzony sprzęt i zakres weryfikacji

**Na 2026-10-11 jedynym GPU z pomiarami tego backendu jest zintegrowany
Intel Arc Meteor Lake.** Sprawdzona konfiguracja:

- GPU: Intel Arc Meteor Lake, identyfikator PCI `8086:7d55`.
- CPU: Intel Core Ultra 9 185H; RAM: 32 GB nominalnie, współdzielony z GPU.
- System i sterownik: Linux, i915, Mesa 26.2.2-arch1.1; wgpu 30.0.1.
- Modele: Basal 1.5 mini, 4.5B i max, uruchamiane pojedynczo, z wagami i aktywacjami FP16.

Potwierdzenie dotyczy wyłącznie przypadków, rewizji kodu i warunków opisanych
w raportach: [zgodność z FP32](../reports/intel-arc-meteor-lake-2026-10-10/README.md),
[GEMM i HTTP](../reports/intel-arc-meteor-lake-gemm-2026-10-11/README.md) oraz
[końcowa weryfikacja pamięci i wyników](../reports/intel-arc-meteor-lake-memory-2026-10-11/README.md).
Nie jest gwarancją poprawności wszystkich wejść ani stabilności przy dowolnym obciążeniu.
W szczególności dwie końcowe próby benchmarku max z domyślnymi rezerwami
zatrzymał przyrost globalnego swapu po ładowaniu modelu; udane eksporty
i wcześniejszy przebieg diagnostyczny nie potwierdzają powtarzalnego startu
z domyślnymi ustawieniami.

**Nie zweryfikowano osobnych kart Arc A-series, B-series ani Arc Pro,
innych zintegrowanych GPU Intela ani wyboru między kilkoma GPU.**
Nie sprawdzono też maksymalnego kontekstu ani długotrwałego obciążenia.
Spełnienie wymagań Vulkan i poprawne wykrycie przez `basal doctor` nie
potwierdzają zgodności wyników modelu. Każda nowa konfiguracja sprzętu
i sterownika wymaga własnych eksportów, porównań z upstream FP32 oraz pomiarów;
wyniki należy zapisać w nowym raporcie. CI sprawdza kompilację i lint,
ale nie uruchamia modeli na GPU.

Nowa [kampania basal-2.0 z 2026-10-14](../reports/upstream-2.0-final-2026-10-14/README.md): mini i 4.5B zachowały po 49/49 wyborów benchmarku oraz po 41/41 odpowiedzi i klasyfikacji pomocniczych względem FP32, a także po 20/20 decyzji act. Kontekst 1024 zakończył się zgodnym wyborem; 8192 przekroczyło 240 s. Max odmówił benchmarku przed ładowaniem z powodu dostępnej RAM; jego jakość pozostaje niepotwierdzona. Omni wymaga CUDA.

## Uruchomienie

Potrzebny jest loader Vulkan, sterownik Intel z `shaderFloat16` i subgroupami
compute oraz dostęp do urządzenia renderowania. Na Arch sterownikiem jest
pakiet `vulkan-intel`. Diagnostyka pamięci dodatkowo wymaga
`VK_EXT_memory_budget`.

Backend wybiera spośród fizycznych GPU Intel spełniających te wymagania
kartę osobną przed zintegrowaną; przy kilku równorzędnych pierwszą
w kolejności zgłoszonej przez Vulkan. Renderery programowe i GPU innych
producentów nie są używane, a brak zgodnego urządzenia kończy się błędem
z powodem odrzucenia każdego GPU Intel. `basal doctor` pokazuje wybrane
urządzenie i budżet jego pamięci. Wybór urządzenia nie ma opcji CLI.

```sh
cargo build --locked --release --features basal-cli/intel
./target/release/basal doctor --model .models/basal-1.5-mini --json
./target/release/basal client --local --model .models/basal-1.5-mini --request request.json
```

Wartości `mini`, `4.5B` i `max` wskazują oficjalne repozytoria modeli;
można też podać lokalny katalog. Brakujące pliki pobiera resolver CLI.
Do serwera HTTP użyj tej samej binarki, wybierając jeden model:

```sh
./target/release/basal serve --model .models/basal-1.5-mini
```

Feature `intel` wybiera backend CLI na Linux, gdy nie jest włączone `cuda`.
Na Linux przy `cuda` + `intel` pierwszeństwo ma CUDA. Na macOS samo `intel`
nie zmienia wyboru Metal; backend Intel jest tam wyłączony.
Samo `cargo build --release` na Linux nadal buduje wariant bez GPU.
Intel nie przełącza się na zainstalowaną obok binarkę CUDA i `setup` nie
pobiera bibliotek CUDA. Oficjalne paczki wydania nie zawierają jeszcze tego wariantu.
`basal update` odmawia zastąpienia go paczką CUDA; aktualizuj repozytorium
i ponownie buduj z `--features basal-cli/intel`.

## Zakres i pamięć

- Wszystkie trzy architektury Basal 1.5, w tym bias w mini/4.5B i brak bias w max.
- Pakowanie `single`, `budget`, `tree`, cache stałych prefiksów promptu,
  selektywny odczyt ostatniej warstwy oraz współdzielenie wag między workerami serwera.
- Funkcje System One, `facts` i `evidence`; mała projekcja evidence wykonuje się na CPU
  na stanach ukrytych wyliczonych na GPU.
- Obsługiwane opcje to `--dtype f16` i `--kernels fused`. `bf16`, `f32`,
  kernele Candle, tablice GEMM CUDA i `--state-cache-mb` większe od zera kończą się błędem.
  Cache stałych szablonów działa domyślnie; cache stanów między requestami nie jest zaimplementowany.
- GEMM wykonuje kolejne spakowane wiersze osobno. Wspólne prefiksy wewnątrz wiersza
  są liczone raz; zwiększenie współbieżności samo w sobie nie gwarantuje większej przepustowości.

Warstwy dekodera pozostają na GPU. Macierze embeddingu i `lm_head` pozostają
w pliku checkpointu, bez mapowania do pamięci; backend odczytuje (`pread`)
i przesyła tylko potrzebne wiersze. Nie modyfikuj plików wag podczas pracy modelu. Przy każdej iteracji alokowany
jest scratch zależny od liczby tokenów, używany ponownie przez wszystkie warstwy.
Backend czeka na zakończenie każdej warstwy, aby ograniczyć kolejkę poleceń GPU
i umożliwić przekazanie karty krótkim requestom. Po zgłoszeniu utraty urządzenia
kolejne operacje zwracają błąd; model wymaga ponownego uruchomienia.
`GET /health` serwera odpowiada wtedy 503 `{"status":"device_lost"}` z nazwami
modeli w `lost_models`, więc supervisor może zrestartować proces. Błędy
pojedynczych requestów, alokacji i timeouty nie zmieniają odpowiedzi `/health`.

Loader konwertuje BF16 na FP16 porcjami po 8 MiB prosto do zmapowanego bufora,
bez kopii całego tensora w RAM.
Na zintegrowanej karcie z `MAPPABLE_PRIMARY_BUFFERS` loader zapisuje wagi
bezpośrednio do docelowych alokacji we współdzielonej pamięci. Pozostała ścieżka
kończy transfer po każdej warstwie, aby nie zatrzymywać drugiej kopii wszystkich wag
w buforach transferowych. Wrapper sprawdza błąd alokacji przed mapowaniem
i propaguje błędy dostępu do mapowanego zakresu przez `Result`.

Zintegrowany Arc współdzieli RAM z systemem i aplikacjami. Budżet Vulkan
pochodzi od sterownika, a użycie pamięci jest szacunkiem procesu, nie osobną
pulą VRAM. `basal benchmark` sprawdza zarówno GPU, jak i dostępną pamięć hosta.
Nie odejmuj ponownie tych samych wag od obu pomiarów. Max wymaga szczególnej
uwagi przy długich promptach i kilku modelach w jednym procesie.
Repozytoryjny `serve.yml` ładuje wszystkie trzy modele naraz; ich wagi
nie mieszczą się razem w 32 GB RAM. Na takiej maszynie uruchamiaj je osobno.
Kontrola przed ładowaniem uwzględnia odzyskiwalny nieaktywny cache plików
w cgroup oraz wagi rzeczywiście przechowywane na GPU. Na iGPU zapas przed
ładowaniem pochodzi ze wspólnego RAM, a pełna rezerwa GPU jest sprawdzana
ponownie po ładowaniu i podczas inferencji. Budżet Vulkan może się zmieniać;
nie traktujemy odczytu sprzed ładowania jako stałej pojemności VRAM.
[Pomiary pamięci](../reports/intel-arc-meteor-lake-memory-2026-10-11/README.md)
obejmują również odmowy z powodu faktycznego przyrostu swapu podczas ładowania
max. Udana inferencja nie oznacza zapasu na dowolny prompt lub współbieżność.

## Weryfikacja i profilowanie

```sh
python3 reports/intel-arc-meteor-lake-2026-10-10/prepare-inputs.py .cache/intel-repeat
./target/release/basal export --model .models/basal-1.5-mini \
  --inputs .cache/intel-repeat/mini-normalized --out reports/NOWY-EKSPORT
./target/release/basal compare --a reports/reference-basal-1.5-mini-fp32 \
  --b reports/NOWY-EKSPORT --out reports/NOWE-POROWNANIE.json
./target/release/basal profile --model .models/basal-1.5-mini \
  --reference reports/reference-basal-1.5-mini-fp32 --n 10
```

Katalogi wynikowe muszą być nowe. Skrypt przygotowuje kopie referencji z aktualną
nazwą modelu w żądaniach; nie zmienia oryginalnych referencji FP32.
Profil synchronizuje GPU po sekcjach i służy do lokalizacji kosztów, nie
zastępuje pomiaru opóźnienia całego requestu. Wszystkie zmiany numeryki
wymagają porównania z FP32 upstream, a zmiany cache/pakowania zgodności bitowej.
CI kompiluje Rust i waliduje WGSL, ale nie uruchamia GPU.
