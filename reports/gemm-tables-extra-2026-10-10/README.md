# Rozszerzenie tabel GEMM: 2026-10-10

Dodano 12 tabel f16/cuBLASLt 12.9.1, invariant search v2, wygenerowanych
wydaniem basal 0.1.7 (`85ca8751`). Kod wyszukiwania i runtime'u jest taki
sam jak w checkoutcie. Każda tabela ma 25 klas M, 4 kształty wag i 100 wpisów.

Karty objęte rozszerzeniem:

1. [L4](nvidia-l4/README.md): mini i 4.5B.
2. [A10](nvidia-a10/README.md): mini i 4.5B.
3. [L40](nvidia-l40/README.md): mini, 4.5B i max.
4. [RTX PRO 6000 Blackwell Server Edition](nvidia-rtx-pro-6000-blackwell-server-edition/README.md): mini, 4.5B i max.
5. [RTX A5000](nvidia-rtx-a5000/README.md): mini i 4.5B.

Z [poprzednią częścią kampanii](../gemm-tables-2026-10-10/README.md)
repozytorium zawiera 27 tabel dla dziesięciu konkretnych nazw GPU.
Max pominięto na kartach 24 GB z powodu zapasu pamięci potrzebnego na
aktywacje i cache. L40 ma własne tabele: nazwa GPU jest częścią klucza,
więc tabele L40S nie zastępują tabel L40. Wyniki dla Blackwell dotyczą
wariantu Server Edition; nie potwierdzają wariantów Workstation i Max-Q.

## Zakres weryfikacji

[Runner](../../tools/perf/run-gemm-tables-cloud.sh) wykonał dla każdej tabeli:

1. Wyszukiwanie na tej samej karcie, precyzji i wersji cuBLASLt.
2. Eksport 44 przykładów i przypadków System One w trybach single, tree
   i budget. `bench.jsonl` oraz `systemone.jsonl` porównano przez `cmp`:
   są bajtowo identyczne między trybami dla wszystkich nowych tabel.
3. Porównania z upstream FP32 na 44 przykładach i 900 pytaniach,
   obejmujące logity, prawdopodobieństwa, kalibrację i odpowiedzi System One.
4. Offline `basal bench` oraz drabinę około 512/1792/4096 tokenów,
   z 1/5 pytań i trzema powtórzeniami `bench-requests`.

Wynik decyzji względem FP32: 44/44 dla każdego modelu; na dużym zestawie
mini i 4.5B mają 899/900, a max 900/900. Nie jest to bitowa zgodność f16
z FP32. Raporty zachowują różnice logitów, prawdopodobieństw, confidence
i konkretne `flips_cal`. Różnice formatu eksportu pytań `multi` w referencji
System One są widoczne w porównaniach; nie deklarujemy dla nich zgodności
tokenów z upstream.

Zegary nie były blokowane. Runner rejestrował zegary SM, moc i temperaturę
przez całe przebiegi. Dane czasu nie są porównaniem wydajności między
kartami ani dowodem przyspieszenia wobec wcześniejszej tabeli.
Surowe eksporty i logi pozostają w ignorowanym
`.cache/gemm-tables-extra-2026-10-10/`. Pole `source` nowych tabel kieruje
do tego rozszerzenia raportu; nie zmieniono konfiguracji algorytmów.

## Osadzanie i ograniczenia

Tabele z [gemm-tables](../../crates/basal-cli/gemm-tables/) są automatycznie
osadzane przez `build.rs` w następnych buildach CUDA. Nie opublikowano
nowego wydania ani obrazu Docker.

Rzeczywisty skrypt `build.rs`, uruchomiony z włączoną gałęzią CUDA,
zaakceptował wszystkie 27 tabel: nazwy, wersję wyszukiwania, kształty,
algorytmy, unikalność i limity rozmiaru. Wygenerowany moduł tabel
(392 836 bajtów) skompilowano osobno przez `rustc`.
Porównano również zapisane słowa algorytmów wszystkich 12 nowych tabel
z surowymi wynikami wyszukiwania: są identyczne. Zbiorcza kontrola raportów
potwierdziła liczby przykładów, brak brakujących wyników i zerowe różnice
single/tree/budget. Przeszła kontrola `git diff --check`.

Nie wykonano pełnego buildu ani Clippy CUDA. Lokalnie nie ma nagłówka
`cublas_api.h`, więc `build.rs` nie sprawdził wersji wobec tego nagłówka.
Wszystkie nowe tabele wygenerowano i wykorzystano z runtime cuBLASLt
12.9.1, przypiętym w obrazie budowania projektu. Nie zmieniano kodu Rust
ani runnera względem poprzedniej części kampanii.

V100 pominięto bez wynajmu: compute capability 7.0 jest niższe niż minimum
8.0 obecnego backendu. Nowe próby uruchomienia RTX 4090 i RTX 5090 ponownie
zakończyły się błędami dostawcy; nie dodano dla nich tabel.
A5000 wystartował w drugiej próbie po przekroczeniu limitu uruchomienia
pierwszej instancji. Wszystkie wynajęte maszyny usunięto po pobraniu
wyników albo po nieudanym uruchomieniu. Raport nie zawiera danych
identyfikujących maszyny ani kosztów kampanii.
