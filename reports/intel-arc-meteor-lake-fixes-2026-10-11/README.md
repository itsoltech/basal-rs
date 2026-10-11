# Intel Arc Meteor Lake: poprawki backendu, 2026-10-11

Poprawki backendu Intel Vulkan sprawdzone na tym samym sprzęcie i sterowniku co
[refaktor](../intel-arc-meteor-lake-refactor-2026-10-11/README.md): zintegrowany Arc Meteor Lake,
Mesa 26.2.2, wgpu 30.0.1, Rust 1.95.0. Punkt odniesienia to binarka zbudowana z commita
`1b7880536ab9490a123b0fa1c3244e36a40237b1`. Po zmianach zbudowano ją z tego samego commita
z niezacommitowanymi poprawkami.

## Zmiany

- **Wybór urządzenia.** Backend rozważa tylko fizyczne GPU Intel spełniające wymagania
  (`shaderFloat16`, subgroupy 8–64). Karta osobna ma pierwszeństwo przed zintegrowaną.
  Przy równorzędnych kartach decyduje kolejność enumeracji Vulkan.
  Gdy żadne GPU Intel nie spełnia wymagań, błąd podaje powód odrzucenia każdego z nich.
- **`/health`.** `Backend::device_health` udostępnia flagę utraty urządzenia, którą ustawia
  callback wgpu. Długi tor (fork) współdzieli tę flagę. `/health` odczytuje atomiki
  bez blokowania i bez odpytywania sterownika:
  - po utracie urządzenia któregoś modelu: 503 `{"status":"device_lost", …, "lost_models":[…]}`;
  - poza tym: 200 z dotychczasowym payloadem.

  Backendy, które nie zgłaszają stanu urządzenia (CUDA, Metal), zachowują się jak dotąd.
- **Loader.** Każda porcja BF16 o rozmiarze 8 MiB jest konwertowana w miejscu i kopiowana
  do zmapowanego bufora docelowego. Dzięki temu nie powstaje pośredni `Vec<u16>` całego tensora
  (dla `gate_up` w max to 234,9 MB). Zakres błędów alokacji jest zamykany przed mapowaniem.
  Widok zakresu jest zwalniany, a bufor odmapowany także po błędzie I/O.
  Ścieżki MAP_WRITE dla iGPU i stagingu z flush po każdej warstwie pozostały bez zmian.
- **Decyzje zależne od buildu.** Decyzje dotyczące Intela w `update`, `paths` i `setup`
  korzystają ze stałej `basal_gpu::INTEL_BACKEND` wyliczanej z `cfg`, a `doctor` wybiera
  ścieżkę Intela przez `cfg`. Nie zależą od tekstowej etykiety `BUILD_BACKEND`.
- **`doctor`.** Nazwę, sterownik i budżet pamięci odczytuje z jednego urządzenia. Brak
  telemetrii pozostaje ostrzeżeniem, a błąd utworzenia urządzenia — błędem.

Kernele, arytmetyka, synchronizacja warstw i timeouty nie zmieniły się.

## Zgodność bitowa

Obie binarki uruchomiono sekwencyjnie na tych samych wejściach. Rekordy JSONL porównano bajt po bajcie,
według indeksu lub id:

| Model | Wejścia | Pakowanie | Bitowo identyczne |
|---|---|---|---:|
| mini | 44 bench + 34 przypadki API | Tree | 78/78 |
| 4.5B | 3 bench | Single | 3/3 |
| max | 3 bench + 4 przypadki API, w tym `m15-mixed` | Single (bench), Tree (API) | 7/7 |

Pliki wynikowe mają identyczne SHA-256. Porównanie mini z FP32 upstream daje 44/44 zgodnych decyzji
(argmax uśrednionych i skalibrowanych prawdopodobieństw), max |Δlogit| 0,0214 i max TV 0,0050.

## Inne kontrole

- **`basal doctor`** (z `VK_LOADER_DEBUG`): przed zmianą 2 wywołania `vkCreateDevice`,
  po zmianie 1. Wpisy `gpu` są identyczne.
- **`basal serve`** dla mini, z długim torem: `/health` zwraca 200 z niezmienionym payloadem
  przed żądaniem `/v1/systemone` (200) i po nim.
- **`basal update`** w buildzie Intel nadal odmawia zastąpienia binarki paczką.
- Przed eksportami max MemAvailable wynosiło około 23 GiB. Podczas ładowania max starą binarką
  jądro zapisało nieudaną alokację strony w innym procesie. Przy nowej binarce takiego wpisu
  nie było. Jedna obserwacja nie pozwala tego interpretować.
- W żadnym przebiegu nie wystąpił GPU HANG, reset GPU ani wygasły fence.
- Przeszły: `cargo fmt --check`, Clippy i Rustdoc (`-D warnings`, `--locked`) dla Intel i Linux
  bez GPU, `cargo deny` 0.19.8, kontrola wyjątku `paste`, `git diff --check`.

## Ograniczenia

- **Utrata urządzenia.** Ścieżki 503 nie wywołano. Na współdzielonym desktopie nie odtwarzano
  utraty urządzenia. Propagację flagi potwierdzono tylko przeglądem kodu. Ustawia ją wyłącznie
  callback wgpu przy utracie urządzenia. Odrzucona alokacja bufora z powodu OOM nie ustawia flagi.
- **Wybór urządzenia.** Maszyna ma jedno GPU Intel, więc kolejności dGPU/iGPU i odrzucania
  niezgodnych kart nie sprawdzono w runtime.
- **Loader.** Nie mierzono szczytowego RSS ani czasu ładowania. Usunięcie alokacji wynika
  z kodu. Pojedyncze czasy `load_s` z eksportów nie są porównaniem wydajności.
- **Zakres eksportów.** Nie powtarzano pełnego eksportu 4.5B i max, Tree w bench max ani pomiarów
  HTTP pod obciążeniem.
- **Inne platformy.** Nie uruchamiano kompilacji CUDA i Metal ani zdalnego CI.

Surowe eksporty, logi i porównania pozostają lokalnie, poza Gitem.
