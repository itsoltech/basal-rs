# Wbudowana tabela GEMM: RTX 4060, basal-2.0-mini

## Zmiana

Dodano zasób
`crates/basal-cli/gemm-tables/nvidia-geforce-rtx-4060--basal-2.0-mini--f16--cublaslt120901.json`.
Tabela pochodzi z lokalnego doboru opisanego w
[próbie uruchomienia](../rtx4060-basal2-mini-2026-10-22/README.md).
Zachowuje wszystkie konfiguracje algorytmów i metadane wygenerowane przez
basal 0.1.13 (`07585380`); dodano jedynie pole `source` z odsyłaczem do tego raportu.
Nie zmieniano kodu Rust, API ani zależności.

RTX 4060, compute capability 8.9, cuBLASLt 120901, FP16, wersja wyszukiwania 2.
Model: `Remek/basal-2.0-mini@cb81b32ce89b0b87e62fcdad28f0435e25c953f3`.
Tabela obejmuje 25 klas M i cztery kształty projekcji, razem 100 wpisów.
Po redukcji do pól runtime zajmuje 13 108 bajtów, poniżej limitu 32 KiB.
Plik nie zawiera ścieżek lokalnych, adresów ani identyfikatorów maszyny.

Istniejący `build.rs` automatycznie wbudowuje pliki JSON w build CUDA.
Przy braku pasującego cache runtime wybiera zasób według GPU, cuBLASLt,
precyzji, wersji wyszukiwania i kształtów, a następnie sprawdza algorytmy
na urządzeniu. Dzięki temu nowy build z tym zasobem może pominąć dobór GEMM
na pasującym sprzęcie. Przy zmianie biblioteki albo parametrów tabela może
zostać odrzucona. Metal i build bez CUDA jej nie zawierają.

Obecna lokalna binarka pozostaje bez zmian i korzysta z wcześniej utworzonego
cache. Samo dodanie pliku nie aktualizuje już zainstalowanych binarek.

## Weryfikacja runtime

Wykorzystano gotową binarkę CUDA 0.1.13 zgodną z commitem checkoutu.
Wejścia wyprowadzono z `examples/client/routing.json` i
`examples/client/tickets.jsonl`: osiem pytań bench oraz cztery pełne żądania,
każde z choice i noul, z jawnym polem modelu.

Wykonano pięć eksportów: oryginalna tabela cache w trybie single, zasób
repozytorium w trybach single/tree/budget oraz kopia po usunięciu pól
diagnostycznych zgodnie z reduktorem builda w trybie single.
W każdym eksporcie tabela została jawnie wskazana przez `--gemm-table`.

- Wszystkie cztery porównania `basal compare` z eksportem cache:
  zero różnic tokenów, 8/8 zgodnych argmax, maksymalna różnica logitów 0,0,
  maksymalna TV 0,0.
- Osobne porównanie logitów po zakodowaniu jako IEEE-754 FP32 potwierdziło
  równość bitową wektorów dla pytań bench i pytań pełnych żądań.
- Rozkłady `p_cal` oraz pola `answers` pełnych żądań były dokładnie równe.
- Sprawdzono pokrycie wszystkich klas M z `config.rs`, brak duplikatów,
  kształty, precyzję, osiem słów każdej konfiguracji i rozmiar po redukcji.
- `git diff --check` bez błędów.

Pierwsza próba eksportu pełnych żądań nie miała pola `model` i zwróciła
błędy walidacji. Zachowano ją lokalnie, ale powyższe wnioski dotyczą
ponowionych eksportów z poprawnymi wejściami (`*-v2`).

## Ograniczenia / GAP

Nie ma tutaj Rust/Cargo ani dostępu do demona Docker. Próby uruchomienia
formatowania, Clippy, rustdoc i cargo-deny zakończyły się brakiem Cargo.
Nie skompilowano nowej binarki i nie potwierdzono jej startu z pustym cache.
Sprawdzenie zredukowanej tabeli odbyło się przez istniejącą binarkę,
nie przez wykonanie `build.rs`.

Nie wykonano pełnej procedury maintainerów z `CONTRIBUTING.md`: brakuje
korpusu basal-bench i zbioru decyzyjnego względem upstream FP32, benchmarku
`basal bench` oraz drabiny kontekstu. Mały lokalny korpus potwierdza równość
opisanego wykonania, nie ogólną zgodność modelu ani całego zakresu tabeli.
CI musi jeszcze sprawdzić build CUDA z nowym zasobem.

Nie mierzono przyspieszenia inferencji ani ponownego startu. Historyczne
415 s doboru tabeli to koszt pierwszego przygotowania, którego pominięcie
umożliwia istniejący mechanizm tabel wbudowanych; nie jest to pomiar A/B
nowej binarki. Działający lokalnie serwer pozostał uruchomiony.

## Dane

Dane poza Gitem, niepublikowane; brak publicznego URI HF.
Archiwum lokalne:
`.cache/measurement-artifacts/gemm-table-rtx4060-basal2-mini-2026-10-22/measurement-data.tar.gz`.
SHA-256: `b9d79ea0d6d4d85a454abc04e2fa585fd5b68fcec4e6e5fef69238f238a65d46`.
Sprawdzono 12/12 sum plików archiwum.

Pod `reports/gemm-table-rtx4060-basal2-mini-2026-10-22/` archiwum zawiera
pary `bench.jsonl` i `systemone.jsonl` w katalogach `inputs`, `cached-single`,
`repo-single`, `repo-tree`, `repo-budget`, `reduced-single`.
Logi, manifesty z lokalnymi ścieżkami i pierwsze niepoprawne eksporty
pozostają poza archiwum. Po rozpakowaniu do nowego katalogu integralność
sprawdza `sha256sum -c SHA256SUMS`.

Odtworzenie po rozpakowaniu: uruchom `basal export --model 2.0-mini
--inputs KATALOG/inputs --out NOWY_KATALOG --gemm-table PLIK_TABELI
--batching single` (osobno z `tree` i `budget`), a następnie
`basal compare --a EKSPORT_CACHE --b NOWY_KATALOG --out NOWY_PLIK.json`.
Użyj tej samej wersji biblioteki cuBLASLt i GPU.
