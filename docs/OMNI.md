# Basal Omni: tekst, obraz i audio

Katalog zawiera `omni-mini` (`Remek/basal-omni-1.0-mini`, Gemma 4) oraz `omni-max`
(`Remek/basal-omni-1.0-max`, Gemma 4 Unified). Rewizje są przypięte w konfiguracji;
[pierwotny audyt](../reports/upstream-2.0-discovery-2026-10-14/README.md) opisuje metadane.
Podstawowe checkpointy są BF16. Sam mini ma około 15,9 GB wag, max około 23,9 GB;
encoder, aktywacje i KV wymagają dodatkowej pamięci.

Pierwsza ścieżka omni wymaga CUDA. `auto` wybiera BF16 dla Gemmy, FP16 dla Llamy.
`--dtype f32` służy porównaniom numerycznym. Media wymagają BF16 lub FP32;
FP16 z mediami jest odrzucane. Metal i Intel Vulkan nie uruchamiają omni.

```bash
basal serve --model omni-mini --media --dtype bf16
basal client --local --model omni-max --media --request request.json --max-input-bytes 67108864
basal export --upstream --model omni-mini --media --dtype bf16 --inputs .cache/reference-omni --out .cache/native-omni
```

W konfiguracji YAML opcja należy do modelu:

```yaml
models:
  - repo: Remek/basal-omni-1.0-mini
    revision: 73c993dc3d9ab3cb8887ac0b1ee1cf1c92783240
    dtype: bf16
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
sloty. Dane mediów nie trafiają do logów ani stałego cache. Cache prefiksu/stanu,
tabele własnego GEMM i profil sekcji Llamy są dla omni niedostępne.

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
