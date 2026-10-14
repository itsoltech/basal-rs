# Upstream v2 — granice kontraktu, 2026-10-14

Jednorazowe porównania rzeczywistego core i przypiętego upstream v2.0.0, bez frameworku testowego.

- Certyfikaty i act (`certificates.json`): 17/17 pełnych odpowiedzi dokładnie zgodnych. Brak certyfikatu, sam pooled i certyfikat innego typu odraczają; certyfikat choice działa po obu stronach progu. Porównano również całe odpowiedzi z kosztami 1, 1e-6, 1e290 oraz max_error null/0.1/1, w tym zaokrąglanie dużych liczb i klucz certyfikatu.
- Plik dla kolejności (`orders-calibration.json`): 4/4 zgodnych wyborów temperatur, certyfikatu i statusu dla 1/2/auto/2le3. Osobny fixture o2 ma inną temperaturę; loader rzeczywiście wybiera go przy dwóch kolejnościach. Pozostałe niedopasowane polityki zachowują temperaturę bazową i usuwają certyfikat.
- Budżet promptów (`prompt-budget.json`): małe wejście i 256 etykiet ze stanem 8 KiB przyjęte; 5000 gałęzi i 256 powieleń stanu 128 KiB odrzucone przed klonowaniem stanu. Limit dotyczy 4096 gałęzi i 32 MiB tekstu obu potencjalnych kolejności; nie jest pomiarem maksymalnej pamięci serwera.

Fixture z kalibracją i pomocniczy probe były jednorazowe, poza repozytorium. Nie zmieniono opublikowanej kalibracji ani checkoutów upstream. Nie dodano testów ani bramek CI. Pozostałe porównania CPU: [raport](../upstream-2.0-cpu-2026-10-14/README.md).

## Dane poza Gitem

- Bucket: [itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs).
- URI archiwum: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-2026-10-14/c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536/measurement-data.tar.gz`.
- SHA-256 archiwum: `c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536`.
- Dane tego raportu: `reports/upstream-2.0-boundaries-2026-10-14/` wewnątrz archiwum; zachowano też oryginalny README sprzed migracji.
- Pobranie i odtworzenie: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Obie weryfikacje pobrania, w tym anonimowa, potwierdziły wszystkie 34 sumy plików.
