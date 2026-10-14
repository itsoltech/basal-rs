# Protokół upstream v2

Źródłem kontraktu jest Basal v2.0.0 (`af1ce43a1770dd4fdfc1611e3fe720b93d42fc73`).
Nowe modele używają `prompt: 2`: krótszych promptów PL/EN i 255 kodów opcji.
Kody po `Z` mają dwa znaki, lecz są pojedynczym tokenem w pozycji odczytu.
Choice i score z 2–255 opcjami korzystają z natywnego odczytu; granice 10/11
oraz 26/27 nie włączają strategii grupowej. Więcej niż 255 opcji jest odrzucane.
Wszystkie pięć przypiętych publikacji ma pojedynczy plik `model.safetensors`.
Stare modele format 1 zachowują dziesięć liter i istniejącą strategię dużych choice.

| Alias | Repozytorium | Architektura |
|---|---|---|
| `2.0-mini` | `Remek/basal-2.0-mini` | Llama |
| `2.0-4.5B` | `Remek/basal-2.0-4.5B` | Llama |
| `2.0-max` | `Remek/basal-2.0-max` | Llama |
| `omni-mini` | `Remek/basal-omni-1.0-mini` | Gemma 4 |
| `omni-max` | `Remek/basal-omni-1.0-max` | Gemma 4 Unified |

Suma samych pięciu checkpointów to około 75 GB; jednoczesne serwowanie wymaga
dodatkowej pamięci na aktywacje/KV i należy dobrać listę do dostępnej VRAM.

Bez konfiguracji i `--model` serwer wybiera przypięty `basal-2.0-4.5B`.
Ten sam model wybierają domyślnie `basal init` i `basal setup`.
Pomiary runtime obejmują CUDA H100, trzy Llamy na Metal M2 Max oraz mini/4.5B na Intel Arc Meteor Lake. Intel max nie uruchomił benchmarku z powodu dostępnej pamięci; jego jakość na tym sprzęcie pozostaje niepotwierdzona. [Końcowy raport](../reports/upstream-2.0-final-2026-10-14/README.md) podaje korpusy, kontekst i niepowodzenia.
Aliasy `mini`, `4.5B` i `max` nadal wskazują modele basal-1.5.
Jawna konfiguracja zachowuje swoją listę modeli oraz wybór `default_model`;
bez `default_model` żądanie trafia do pierwszego modelu z tej listy.
Wagi są z przypiętych podstawowych checkpointów BF16. Dla Llamy `auto` nadal
wybiera FP16 wykonania; dla omni BF16. FP8, GGUF i MLX 8-bit są poza zakresem. Dla nieznanych repozytoriów `--model owner/name` precyzja auto jest rozwiązywana z architektury po odczycie metadanych (Llama FP16, Gemma BF16), tak jak dla lokalnego katalogu; jawne dtype w konfiguracji pozostaje nadrzędne.

```bash
basal init --model 2.0-mini --model 2.0-4.5B --model 2.0-max --model omni-mini --model omni-max
basal serve --model 2.0-4.5B --orders 1
basal client --local --model 2.0-mini --ask 'Czy zgłoszenie jest pilne?' --state ticket.txt
basal prepare --model 2.0-max --upstream --orders 2 --request request.json
```

`--orders` albo YAML `orders: "..."`: `1`, `2`, `auto`, `2leN`. Format 2 domyślnie
używa jednej kolejności, format 1 dwóch. `auto` dodaje kolejność odwróconą dla
noul/gałęzi multi, gdy P(true) jest w [0,2; 0,8], dla pozostałych gdy maksimum
jest poniżej 0,6. W trybie auto sprawdzamy kontekst obu potencjalnych kolejności
przed forwardem. `usage.second_order_branches` rozlicza wykonane powtórzenia.

Format 2 liczy softmax, uśrednianie i kalibrację w FP32. Temperatura bucketowa
ma pierwszeństwo przed temperaturą typu. Dla dwóch kolejności loader wybiera
`CALIBRATION.o2.json`, jeżeli istnieje. Przy niedopasowaniu zachowuje temperatury,
usuwa progi certyfikatów i wyuczone progi fallback oraz oznacza odpowiedzi
`unvalidated`. `max_error` potrzebuje certyfikatu danego typu; sam próg pooled
nie wystarcza. W formacie 2 brak certyfikatu powoduje odroczenie. Format 1 zachowuje dotychczasowe
przyjmowanie starszego certyfikatu pooled oraz odpowiedź act bez wymuszonego
odroczenia przy jego braku. Loader nie trenuje kalibracji.

`lang: "pl" | "en"` wymusza język całego żądania. `multi` obsługuje cele
`threshold`, `exact`, `iou`, `f1`, granice `min`/`max`, `exclusive` i `fallback`.
Progi fallback z kalibracji są powiązane z konkretnym zestawem kluczy/opisów.
IoU/F1 dopuszczają do 256 etykiet, z osobną gałęzią noul dla każdej.

`tool` przyjmuje definicje OpenAI Chat/Responses, MCP i Anthropic. Opcje `extra`
obejmują `answer_directly`, `ask_user`, `none`; `rank`, `multi` i `max_error`
sterują rankingiem, zbiorem oraz odroczeniem. Runtime zwraca decyzję i nie wywołuje
narzędzia ani serwera MCP. Przykład:

```json
{"state":"Sprawdź pogodę w Warszawie.","questions":{"q":{"type":"tool",
 "tools":[{"type":"function","function":{"name":"weather","description":"Sprawdź pogodę",
 "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
 "extra":["answer_directly","none"],"rank":2}}}
```

`/v1/basal` odtwarza konwencje upstream. `/v1/systemone` zachowuje confidence i
legendę TypeSafe dla pierwotnych typów; nowe rozszerzenia odpowiadają jak upstream.
Metadane modeli podają format promptu, kolejności, temperatury i włączone media.
Nowe odpowiedzi rozliczają również najdłuższy wykonany prompt. Przed powieleniem
stanu runtime sprawdza budżet 4096 gałęzi i 32 MiB materiału promptów dla dwóch
potencjalnych kolejności. Żądania rozszerzające ponad 128 KiB współdzielą limit
czterech dużych body; przekroczenie budżetu daje 422, przeciążenie 529. Szczegóły mediów:
[OMNI.md](OMNI.md); różnice istniejącego API: [SYSTEM_ONE.md](SYSTEM_ONE.md).

Eksporter [export_reference_v2.py](../tools/reference/export_reference_v2.py)
wywołuje przypięty upstream eager i zapisuje wybrane logity, prompty/tokeny,
kalibrację, odpowiedzi oraz kształty/hashe cech procesora. Opcja `--cases-only --cases ...` wybiera własny korpus. Obsługuje stałe
kolejności 1/2; adaptacyjne wykonanie sprawdza się osobno przez serwer. Natywny
`basal export --upstream` wybiera ten sam dialekt; bez flagi zachowuje TypeSafe. Kompilacja
nie stanowi dowodu zgodności numerycznej ani wydajności. Wyniki porównań należą
do nowych raportów; zgodność na każdym backendzie wymaga pomiarów runtime'u.

`basal bench` nadal mierzy jawnie warianty lat1/lat2; jakość i throughput tej
komendy używają dwóch kolejności niezależnie od domyślnej polityki serwowania.

Zaobserwowaną zgodność i ograniczenia numeryczne opisuje [raport H100](../reports/upstream-2.0-h100-2026-10-14/README.md).

W [analizie decyzji względem upstream FP32](../reports/upstream-2.0-decisions-2026-10-14/README.md)
450 porównań odpowiedzi orders=1 zachowało wybory i pomocnicze klasyfikacje;
osobne pięć BF16 auto zachowało wybór. Porównania obejmują opcje, narzędzia, etykiety
i odroczenie narzędzi; dla noul stronę progu, a dla score najbardziej prawdopodobny poziom. `act` sprawdzono osobno:
17/17 porównań reguł hosta dla tych samych prawdopodobieństw, bez forwardu modeli. Confidence i score mogą różnić się
numerycznie. Zgodność dotyczy opisanego korpusu, nie wszystkich możliwych wejść.

[Końcowa kampania](../reports/upstream-2.0-final-2026-10-14/README.md) potwierdziła dodatkowo 100/100 rzeczywistych pytań act na pięciu modelach CUDA oraz 40/40 pytań z istniejącym certyfikatem 0.05. Metal: 60/60 pytań act, Intel mini/4.5B: 40/40. Domyślny HTTP i lokalny klient wskazały basal-2.0-4.5B. Próby pojemności oraz długiego kontekstu zawierają opisane ograniczenia.
