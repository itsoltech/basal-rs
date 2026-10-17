# Discovery zgodności z Basal 2.0 — 2026-10-14

To raport analizy źródeł i metadanych, bez inferencji ani pomiarów wydajności. Nie potwierdza obsługi nowych modeli przez basal-rs.

## Źródła i zakres

- [Strona wydania](https://basal.si5.pl/): pięć modeli w dwóch rodzinach.
- [Upstream v2.0.0](https://github.com/rkinas/basal/tree/af1ce43a1770dd4fdfc1611e3fe720b93d42fc73): analizowany commit `af1ce43a1770dd4fdfc1611e3fe720b93d42fc73`.
- [FORMAT2.md](https://github.com/rkinas/basal/blob/af1ce43a1770dd4fdfc1611e3fe720b93d42fc73/docs/FORMAT2.md), [OMNI.md](https://github.com/rkinas/basal/blob/af1ce43a1770dd4fdfc1611e3fe720b93d42fc73/docs/OMNI.md), `prompt.py`, `server.py`, `sets.py`, `tools.py`, `media.py`, `engine.py` i `evidence.py`.
- Podstawowe repozytoria Hugging Face `Remek/basal-2.0-mini`, `Remek/basal-2.0-4.5B`, `Remek/basal-2.0-max`, `Remek/basal-omni-1.0-mini`, `Remek/basal-omni-1.0-max`; przypięcia i SHA-256 odczytanych plików w `models.json`.
- Repozytorium basal-rs przy commicie `4f68eb216ec3570f4e44c8262af566a0b0173ef7`; stan roboczy przed discovery był czysty.

## Ustalenia

1. Wszystkie modele deklarują `prompt: 2`, `readout.kind: "code logits"`, 255 kodów i ten sam prefill. Obecny manifest wymaga `letter logits` i A–J, więc je odrzuci.
2. Trzy modele tekstowe zachowują Llamę: mini ma kontekst 8192, 4.5B i max 32768. Nie trzeba zakładać zmiany samej architektury Llama; potrzebne są nowy protokół, katalog i zaobserwowana walidacja wyników.
3. Format 2 ma krótsze prompty PL/EN i domyślnie jedną kolejność opcji, zamiast dwóch. Opcje 11–255 są natywne; obecna strategia grupowa nie powinna obsługiwać ich zamiast formatu 2.
4. `temperature_per_bucket` ma pierwszeństwo przed `temperature_per_prim`. Plik `CALIBRATION.o2.json` odpowiada dwóm kolejnościom. Niedopasowanie kolejności pozbawia wynik certyfikatów i dopasowanych fallbacków, zamiast stosowania ich do innego odczytu.
5. `max_error` w formacie 2 używa `thresholds[target].per_type[type].confidence`. Brak/null oznacza brak certyfikatu i odroczenie. Próg zbiorczy `confidence` nie jest certyfikatem typu. Jest to istotna różnica wobec aktualnego `engine.rs`, który odczytuje próg zbiorczy.
6. `tool` buduje opcje z definicji narzędzi, wspiera extra/rank/multi/max_error. Nie wykonuje narzędzi. `multi` dodaje cele exact/iou/f1, exclusive/fallback i fallback kalibracji identyfikowane skrótem zbioru etykiet.
7. Omni mini to `Gemma4ForConditionalGeneration` / `gemma4`; omni max to `Gemma4UnifiedForConditionalGeneration` / `gemma4_unified`. Oba mają zagnieżdżone `text_config`, związane embeddingi/głowicę, GELU, lokalne/globalne attention i końcowy softcap 30. Mini dodatkowo ma wejścia per warstwa i współdzielenie KV; max ma odmienne K=V i konfigurację globalnych głów. Wymiary głów 256/512 oraz częściowe RoPE nie pasują do założeń Llama. Deklarowany kontekst mini/max to 131072/262144; nie jest to wynik pomiaru pojemności naszego runtime'u.
8. Upstream omni używa BF16 eager CUDA, poprawnego kernelu math attention i readoutu kodów w FP32 z softcap. Nie deklaruje dla tego wydania backendów fast, FP8, MLX ani GGUF. Nie wolno uznać naszego attention Llama za poprawne dla Gemma bez portu i porównania.
9. Pełna obsługa mediów wymaga encoderów i processorów, a nie tylko dekodowania base64. Do ośmiu elementów, 20 MiB/element, 50 mln pikseli; audio mono 16 kHz, WAV/FLAC/OGG/MP3 i okno procesora 30 s; obraz 280/1120 soft tokens. API przyjmuje media przed tekstem stanu i oznacza `calibration: "text"`.
10. Żaden z pięciu odczytanych listingów plików nie ma `evidence_head.pt`. Jawna niedostępność evidence jest zgodna z upstream. Między v1.5.0 a v2.0.0 `facts.py` nie zmienił się; `evidence.py` dodaje wybór formatu promptu.
11. Obecne pobieranie ma zamkniętą listę ośmiu nazw plików, bez `CALIBRATION.o2.json` i processorów. Dodanie wyłącznie pięciu rekordów `KNOWN_MODELS` nie zapewni obsługi.
12. Basal-rs ma własny istniejący dialekt TypeSafe i obliczenia hosta w f64, upstream odczyt/temperaturę w FP32. Aktualizacja wymaga świadomego wersjonowania tych granic; nie wolno deklarować bitowej równości nowej numeryki bez pomiaru ani zmieniać starej ścieżki bez bitowego A/B.

## Wykonane czynności i ograniczenia

Odczytano stronę wydania, pobrano źródła tagu v2.0.0, odczytano metadane Hub i pliki konfiguracji pięciu repozytoriów oraz porównano istotne pliki upstream v1.5.0/v2.0.0 przez `git diff`. Nie pobierano wag, nie uruchamiano modeli i nie korzystano z wynajętego GPU. Nie modyfikowano `.baseline/` ani poprzednich raportów.

Ten raport zachowuje stan discovery sprzed implementacji. Aktualny zakres opisuje [UPSTREAM_2.0.md](../../docs/UPSTREAM_2.0.md), a pomiary runtime’u [raport H100](../upstream-2.0-h100-2026-10-14/README.md). Plany agentów pozostają lokalne.

## Dane poza Gitem

- Bucket: [itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs).
- URI archiwum: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-2026-10-14/c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536/measurement-data.tar.gz`.
- SHA-256 archiwum: `c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536`.
- Dane tego raportu: `reports/upstream-2.0-discovery-2026-10-14/` wewnątrz archiwum; zachowano też oryginalny README sprzed migracji.
- Pobranie i odtworzenie: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Obie weryfikacje pobrania, w tym anonimowa, potwierdziły wszystkie 34 sumy plików.
