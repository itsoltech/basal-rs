# Brakujące tabele GEMM: 2026-10-10

Dodano 12 tabel f16/cuBLASLt 12.9.1, invariant search v2, wygenerowanych
wydaniem basal 0.1.7 (`85ca8751`). Przed sesją potwierdzono, że kod
wyszukiwania i runtime'u tego wydania jest taki sam jak w checkoutcie.
Każda nowa tabela ma 25 klas M, 4 kształty wag i 100 wpisów algorytmów.

Nowe tabele i raporty:

1. [RTX 6000 Ada Generation](nvidia-rtx-6000-ada-generation/README.md): max i mini.
2. [L40S](nvidia-l40s/README.md): mini, 4.5B i max.
3. [RTX A6000](nvidia-rtx-a6000/README.md): mini, 4.5B i max.
4. [A100-SXM4-80GB](nvidia-a100-sxm4-80gb/README.md): mini, 4.5B i max.
5. [H100 PCIe](nvidia-h100-pcie/README.md): mini.

Z wcześniejszymi tabelami H100 PCIe dla max/4.5B i RTX 6000 Ada dla 4.5B
repozytorium zawiera teraz 15 tabel: wszystkie trzy modele dla pięciu
powyższych nazw GPU. Tabele są w
[crates/basal-cli/gemm-tables](../../crates/basal-cli/gemm-tables/);
`build.rs` osadza je automatycznie w kolejnych buildach CUDA. Ta sesja
nie publikuje nowego wydania ani obrazu Docker.

## Weryfikacja modeli

[Runner](../../tools/perf/run-gemm-tables-cloud.sh) wykonał dla każdej nowej tabeli:

1. Wyszukiwanie na tej samej karcie, precyzji i wersji cuBLASLt.
2. Eksport 44 przykładów oraz przypadków System One w trybach single,
   tree i budget; `bench.jsonl` i `systemone.jsonl` porównano przez `cmp`.
   Pliki są bajtowo identyczne między tymi trybami dla wszystkich 12 tabel.
3. Porównanie z upstream FP32 na 44 przykładach i zestawie 900 pytań,
   z logitami, prawdopodobieństwami, kalibracją i odpowiedziami System One.
4. Offline `basal bench` oraz drabinę około 512/1792/4096 tokenów,
   z 1/5 pytań i trzema powtórzeniami `bench-requests`.

Wynik względem FP32: wszystkie nowe tabele mają 44/44 zgodnych decyzji
na małym zestawie; max ma 900/900, a mini i 4.5B 899/900 na dużym zestawie.
To zgodność decyzji, nie bitowa zgodność z FP32. Raporty zachowują także
różnice logitów, prawdopodobieństw, confidence i konkretne `flips_cal`.
Różnice formatu eksportu pytań `multi` w referencji System One są widoczne
w porównaniach; nie deklarujemy dla nich zgodności tokenów z upstream.

Czasy nie dowodzą przyspieszenia względem wcześniejszej tabeli ani różnic
wydajności między kartami. Zegary nie były blokowane; zakres rejestracji
zegarów jest opisany osobno dla każdej karty. Surowe eksporty i pełne logi
pozostają w ignorowanym `.cache/gemm-tables-2026-10-10/`.

## Kontrole lokalne i zakres

Przeszły formatowanie, Clippy i Rustdoc na macOS, cargo-deny i kontrola
wyjątku `paste`, a dla runnera `bash -n` i ShellCheck. Kontrola tabel używa
rzeczywistego skryptu `build.rs` z włączoną gałęzią CUDA: sprawdza nazwy,
wersję wyszukiwania, kształty, algorytmy, unikalność i limity rozmiaru.
Wygenerowany moduł tabel jest kompilowany osobno. Nie wykonano pełnego
buildu ani Clippy CUDA; na lokalnym hoście nie ma toolkitu CUDA, więc
zgodności nagłówka `cublas_api.h` nie sprawdzono tym skryptem. Wszystkie
nowe tabele powstały i zostały użyte z runtime cuBLASLt 12.9.1, wersją
przypiętą w obrazie budowania projektu.

Dodanie `model` i `source` sprawdzono względem surowych tablic algorytmów,
bez zmiany ich słów konfiguracyjnych. Do repozytorium nie trafiają dane
identyfikujące maszyny ani koszty sesji. Wszystkie wynajęte maszyny usunięto
po pobraniu wyników.

## Karty konsumenckie

Sprawdzono katalog Shadeform także bez ograniczenia do jednej karty.
RTX 5090 nie wystartował w dwóch próbach u ExcessSupply. Jedyna dostępna
oferta RTX 4090 miała osiem kart; próba jej uruchomienia również zakończyła
się błędem dostawcy. Błędne instancje usunięto. Pozostałych modeli z zakresów
RTX 5090–5060, 4090–4060 i 3090–3060 nie było w dostępnych ofertach podczas
tej sesji. Nie generowano dla nich tabel i nie deklarujemy ich weryfikacji.
