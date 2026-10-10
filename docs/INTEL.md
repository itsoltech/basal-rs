# Intel Arc na Linux

Backend `intel` wykonuje inferencję Basal 1.5 mini, 4.5B i max przez Vulkan,
z własnymi kernelami WGSL i `wgpu 30.0.1`. Forward używa wag i aktywacji FP16,
akumulacji GEMM w FP32, attention/RoPE w FP32 i odczytu logitów liter w FP32.
Nie wymaga Pythona, CUDA, OpenCL ani oneAPI.

Pomiary na zintegrowanym Arc Meteor Lake i ograniczenia ich zakresu opisuje
[raport](../reports/intel-arc-meteor-lake-2026-10-10/README.md).
Inne generacje i osobne karty Arc wymagają własnej weryfikacji. Ten backend
nie korzysta z XMX; nie jest dowodem maksymalnej wydajności wszystkich kart Intel.

## Uruchomienie

Potrzebny jest loader Vulkan, sterownik Intel z `shaderFloat16` i subgroupami
compute oraz dostęp do urządzenia renderowania. Na Arch sterownikiem jest
pakiet `vulkan-intel`. Diagnostyka pamięci dodatkowo wymaga
`VK_EXT_memory_budget`.

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
W opisanej sesji benchmark max zatrzymał się na kontroli rezerwy RAM przed
ładowaniem. Udana inferencja max nie oznacza, że zostaje zapas na dowolny
prompt lub współbieżność; zakres sprawdzonych przebiegów podaje raport.

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
