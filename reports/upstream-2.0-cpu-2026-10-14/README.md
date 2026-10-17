# Upstream v2 — porównania CPU, 2026-10-14

Źródła: Basal v2.0.0 `af1ce43a1770dd4fdfc1611e3fe720b93d42fc73`, Transformers 5.17.0,
Torch 2.11.0 CPU, tokenizers 0.23.3 (Rust 0.23.2), NumPy 2.5.4, SciPy 1.18.1,
Pillow 12.3.0. Rewizje modeli: discovery (`reports/upstream-2.0-discovery-2026-10-14/models.json`).
Jednorazowe wywołania rzeczywistego core/CLI i upstream; bez frameworków/bramek testowych.

| Zakres | Wynik | Dane |
|---|---|---|
| 43 oficjalne wektory na każdym z trzech tokenizerów Llama | 129/129 dokładnych promptów, token IDs, kodów i permutacji | wektory (`llama-vectors.json`) |
| PL/EN, 2/10/11/26/27/255 opcji, dwie kolejności, oba tokenizery Gemma | 24/24 dokładnych promptów i tokenów | chat (`gemma-chat.json`) |
| Obraz default/high, audio, oba porządki mieszanych mediów, dwa pytania, oba modele | 20/20 dokładnych tokenizacji | media (`media-prompts.json`) |
| Reguły zbiorów i definicje/ranking/zbiory narzędzi z tymi samymi prawdopodobieństwami | 280/280 zgodnych odpowiedzi (liczby do 1e-10) | odczyty (`readouts.json`) |
| Kalibracja pięciu modeli dla 1/2/auto/2le3 | 20/20: temperatury, bucket 27, certyfikat choice, status mismatch | kalibracja (`calibration.json`) |
| Błędne media/URL/base64/detail/31 sekund/9 elementów | 8/8 odrzuconych także przez upstream | błędy (`negative-media.json`) |
| YAML stary/nowy parser: konfiguracja i anchory; duplikaty i obce pola | zgodne przyjęcie/odrzucenie | YAML (`yaml.json`) |

52 porównania procesorów na deterministycznych obrazach i stereo sinusach o różnych
częstotliwościach próbkowania. Wszystkie kształty, pozycje, maski i soft tokeny zgodne.
20 obrazów PNG jest bitowo równych (w tym skrajne proporcje, default/high, oba modele).
WAV/FLAC: mini mel max |Δ| 4.77e-7; unified waveform bitowo równy w tych przypadkach.
Dekodery stratne różnią się: OGG mel do 8.11e-4, MP3 do 1.56e-3; waveform odpowiednio
do 5.96e-8 i 2.68e-7. Dane: procesory (`processors.json`).

Porównania wykryły i doprowadziły do naprawy uint8 bicubic (stałoprzecinkowe wagi,
poziomy/pionowy przebieg i zaokrąglanie jak PyTorch), pustego znacznika kanału Unified
przy wyłączonym thinking oraz maskowania ostatniej niepełnej ramki mel.
Żadne z tych porównań nie wykonuje forwardu GPU ani nie dowodzi jakości modelu,
zgodności wszystkich kodeków/obrazów czy przyspieszenia. PNG i syntetyczne sinusy nie
zastępują korpusu rzeczywistych obrazów/mowy. Nie trenowano progów kalibracji.

## Dane poza Gitem

- Bucket: [itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs).
- URI archiwum: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-2026-10-14/c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536/measurement-data.tar.gz`.
- SHA-256 archiwum: `c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536`.
- Dane tego raportu: `reports/upstream-2.0-cpu-2026-10-14/` wewnątrz archiwum; zachowano też oryginalny README sprzed migracji.
- Pobranie i odtworzenie: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Obie weryfikacje pobrania, w tym anonimowa, potwierdziły wszystkie 34 sumy plików.
