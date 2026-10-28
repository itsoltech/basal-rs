# Odtworzenie pomiaru thinking J1/J2

Dane tej kampanii są prywatne i lokalne. Nie opublikowano ich w Hugging Face;
nie ma publicznego URI. Archiwum zawiera surowe ścieżki i logi środowiska.
Przed ewentualną publikacją wymaga osobnego przygotowania.

## Kontrola danych

Po otrzymaniu lokalnego `measurement-data.tar.gz` sprawdź SHA-256 podane
w README, rozpakuj do nowego katalogu i wykonaj w nim:

```bash
sha256sum -c SHA256SUMS
```

Logiczny katalog danych to
`reports/upstream-api-241-thinking-rebuild-2026-10-28/evidence/private/`.
Zmienna `EVIDENCE` w dalszych poleceniach wskazuje ten katalog.
Archiwum nie zawiera checkpointów, binarek ani katalogów kompilacji.

## Źródła i wejścia

`j1-source.tar` i `j2-source.tar` zawierają dokładne drzewa źródeł;
`j1-source-commit.txt` i `j2-source-commit.txt` identyfikują commity.
Rozpakowuj każde drzewo osobno. Polecenia kompilacji i środowisko są
w `j1-build-01.log` i `j2-build-01.log`. J2 budowano przez
`cargo build --locked --release -p basal-cli --features cuda`
z `CARGO_NET_OFFLINE=true`, Rust 1.95.0, CUDA 12.9.1.
Zastąp prywatne ścieżki lokalnymi katalogami narzędzi i wyjść.

Oba modele pochodzą z
`Remek/basal-2.0-4.5B@e5ba9149760d99b73b1963d8e0ccf948b58afacb`:
checkpoint bazowy z root i generator z `think/`, wraz z metadanymi,
tokenizerem oraz kalibracją. Nie używaj innej rewizji ani kalibracji.

`reference-inputs/` zawiera zapisane odpowiedzi rzeczywistego upstream
i dodatkowej referencji Torch FP32. `reference-copy-proof-01.log`
oraz `reference-inputs.sha256` dokumentują kopię wcześniejszych danych.
Nowa kampania nie uruchamiała upstream ponownie.
`reference-producer/` zachowuje wcześniejszy runner referencji i archiwum
niezmienionych źródeł upstream. Plik `manifest.json` pochodzi ze środowiska
sterownika eksperymentu; wersję rzeczywistego sidecara potwierdza `vllm.log`.

## Macierz HTTP

Prywatny runner `j2-runner/native_probe.py` zachowuje korpus w `cases.json`,
każde żądanie i odpowiedź, polecenie serwera, health, kody zakończenia,
kontrole powtórzeń i próbki `nvidia-smi`. Wymaga przygotowanych lokalnych
modeli w swoim `models/basal-2.0-4.5B`. Korzysta z jednego lokalnego portu;
fazy uruchamiaj kolejno. Nie uruchamiaj poleceń zatrzymujących usługę
z surowego logu w innym środowisku.

```bash
python3 native_probe.py think --binary "$J1_BINARY" \
  --tokens 256 --max-len 2048 --repeat-all --gpu-sample-ms 100 --tag baseline-new
python3 native_probe.py think-off --binary "$J2_BINARY" \
  --tokens 256 --max-len 2048 --gpu-sample-ms 100 --tag off-new
python3 native_probe.py think --binary "$J2_BINARY" \
  --tokens 256 --max-len 2048 --repeat-all --gpu-sample-ms 100 --tag unset-new
python3 native_probe.py think --binary "$J2_BINARY" \
  --tokens 256 --max-len 2048 --think-numerics inherit \
  --repeat-all --gpu-sample-ms 100 --tag inherit-new
python3 native_probe.py think --binary "$J2_BINARY" \
  --tokens 256 --max-len 2048 --think-numerics upstream-bf16-pointwise \
  --repeat-all --gpu-sample-ms 100 --tag pointwise-new-1
python3 native_probe.py think --binary "$J2_BINARY" \
  --tokens 256 --max-len 2048 --think-numerics upstream-bf16-pointwise \
  --repeat-all --gpu-sample-ms 100 --tag pointwise-new-2
```

Każde wywołanie startuje i kończy osobny proces. Ostatnie dwa służą też
porównaniu po restarcie. Używaj nowych tagów; runner odmawia nadpisania.
Argumenty rzeczywistej serii są w `j2-remote-out/*/arguments.json`,
a pełne polecenia i status całej sesji w `j2-remote-session-01.log`.

## Porównanie

Pliki `j2-compare-*.log` zapisują dokładne polecenia jq i kody zakończenia;
odpowiadające im JSON zawierają wyniki dla każdego przypadku.
Po przeniesieniu archiwum zastąp w poleceniach dawny prefiks katalogiem
`EVIDENCE`. Nie trzeba uruchamiać modelu do ponownego obliczenia porównań.

Kontrola zachowania usuwa wyłącznie `usage.latency_ms` i `answers.*.think.ms`.
Pozostałe wartości JSON muszą być dokładnie równe. Jest to równość wartości
odpowiedzi HTTP, bez dowodu bitowej równości wszystkich tensorów wewnętrznych.
Porównanie referencji paruje 19 nazw plików: `off-*.json`, `force-*.json`
i `auto-choice.json`. Najpierw sprawdza zgodność żądań, statusów i ścieżek
prawdopodobieństw. Następnie liczy maksymalną bezwzględną różnicę,
liczbę odpowiedzi spełniających próg 1e-4 oraz zgodność tekstu, liczby
tokenów i argmax dziewięciu odpowiedzi z notatką.

Próbki `gpu-samples.csv` opisują całą kartę co około 100 ms, od startu
do zamknięcia procesu. Maksimum próbek nie jest rzeczywistym maksimum
alokatora. Czasy `think.ms` obejmują pracę thinking i nie izolują decode.
Seria nie jest kontrolowanym benchmarkiem wydajności.
