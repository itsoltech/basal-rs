# Udział w rozwoju basal-rs

[English](CONTRIBUTING.md)

basal-rs uruchamia modele decyzyjne Basal z odpowiedziami modelu upstream FP32, szybciej. Zmianę przyjmujemy, gdy
zachowuje te odpowiedzi, a jej efekt jest zmierzony. Zgłoszenia i pull requesty można pisać po polsku albo po angielsku.

## Zanim zaczniesz

- Błędy, rozbieżności z upstream, wydajność i pomysły: [formularze zgłoszeń](https://github.com/itsoltech/basal-rs/issues/new/choose).
  Pytania: [Discussions](https://github.com/itsoltech/basal-rs/discussions).
- Luki bezpieczeństwa: prywatnie, zobacz [SECURITY.md](SECURITY.md).
- Przy większej zmianie (nowy kernel, harmonogram, API) najpierw issue, żeby przed pracą uzgodnić podejście i sposób
  pomiaru.

## Budowanie

Rust stable (sprawdzony na 1.95).

```sh
# Metal (macOS, Apple Silicon)
cargo build --release

# CUDA (Linux; obraz z toolkitem: tools/cuda/Dockerfile)
docker build -t basal-dev:cuda tools/cuda
docker run -d --name basal-dev --gpus all --ipc=host -v "$PWD":/work -w /work basal-dev:cuda sleep infinity
docker exec basal-dev cargo build --release --features basal-cli/cuda
```

Budowa kodu: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Jak sprawdzamy zmiany

Projekt nie ma testów jednostkowych, integracyjnych ani E2E, a pull requesty nie dodają frameworków testowych,
snapshotów ani bramek testowych w CI. Poprawność sprawdzamy porównaniem wyników runtime'u z referencją upstream, a
szybkość pomiarami ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

- Zmiana, która nie powinna zmieniać obliczeń (pakowanie, cache, harmonogram, refaktoryzacja), daje wynik bitowo
  równy `main`: `basal export` obiema wersjami, `basal compare` pokazuje 0,0.
- Zmianę numeryki (kernele, precyzja, redukcje) porównujemy z referencją FP32 upstream: decyzje się nie zmieniają, a
  różnice logitów i prawdopodobieństw są zapisane.
- Teza o wydajności wymaga pomiaru przed i po na tym samym sprzęcie i w tej samej precyzji. Porównanie z upstream
  także w tej samej precyzji (np. basal-rs f16 i MLX f16). Laptop mierzymy na zasilaczu; warunki (GPU, limit mocy,
  rewizja modelu) trafiają do raportu.

Pomiary trafiają do nowego katalogu w [reports/](reports/README.md) z `README.md` i danymi, z których powstały liczby;
istniejących raportów nie nadpisujemy. Raporty i kod nie zawierają danych pozwalających zidentyfikować maszyny (nazwy
hostów, adresy IP, ścieżki domowe). Checkoutów upstream w `.baseline/` nie modyfikujemy.

`cargo fmt --all --check` i `cargo clippy --locked --workspace --release --all-targets -- -D warnings` przechodzą.
CI uruchamia je na Linux bez CUDA i na macOS z Metal, a na Linux z CUDA sprawdza także wszystkie features.

Każdy pull request uruchamia kontrole formatowania, Clippy, Rustdoc, zależności i workflow. Zakres: [docs/CI.md](docs/CI.md).
CI nie uruchamia testów, dopóki projekt ich nie ma. Obrazy Docker budujemy i publikujemy tylko dla wydań lub ręcznie
ze wskazanym tagiem wydania.

## Tabele GEMM

`gemm_table: auto` wymaga tabeli cuBLASLt niezależnej od partii dla każdej pary GPU i wersji cuBLASLt, modelu i
dtype; jej generowanie przy pierwszym starcie trwa od kilku do kilkudziesięciu minut. Tabele z
[crates/basal-cli/gemm-tables/](crates/basal-cli/gemm-tables) są wkompilowane w binarkę, a `basal serve` używa
pasującej, gdy cuBLASLt przyjmie jej algorytmy na tej karcie. Użytkownik, który wygenerował tabelę dla GPU lub
modelu, których build nie ma, może ją wysłać przez `basal gemm-share` (issue z etykietą `gemm-table`, po
potwierdzeniu; bez `gh` przez formularz „Tabela GEMM / GEMM table”).

Tabele trafiają tylko do buildu CUDA, a `build.rs` ogranicza ich zbiór: f16, wersja cuBLASLt z toolkitu CUDA
(`cublas_api.h`) i bieżąca wersja wyszukiwania, jedna tabela na GPU i kształty wag, nazwa pliku
`<gpu>--<model>--f16--cublaslt<wersja>.json`. Zostawia tylko pola czytane przez serwer (13–15 KB na tabelę zamiast
~34 KB) i przerywa build, gdy tabela łamie regułę albo wszystkie razem przekraczają 1 MiB; CI (Clippy z CUDA)
sprawdza to w każdym pull requeście. Nowy toolkit CUDA lub nowa wersja wyszukiwania oznacza więc wymianę tabel,
a nie dokładanie kolejnych.

Przed dodaniem tabeli maintainer sprawdza ją na tym samym GPU i wersji cuBLASLt, z tabelą jako `gemm_table`:
eksporty przykładów basal-bench jako pojedyncze żądania, w drzewie i w partii z budżetem porównane ze sobą (0,0),
zestaw decyzyjny wobec FP32 upstream oraz czas wobec tabeli wygenerowanej na miejscu (`basal bench`, drabina
kontekstu). Nazwa pliku: `<gpu>--<model>--<dtype>--cublaslt<wersja>.json`, JSON z issue z dodanym polem `source`
wskazującym raport lub issue.

## Jakość kodu Rust

Zasady implementacji i przeglądu są w [AGENTS.md](AGENTS.md#jakość-kodu-rust), na podstawie
`itsolpowers:rust-implementation`. Sprawdzamy ownership i zbędne kopie, invariants publicznego API, typowane błędy
i ich przyczyny, walidację wejścia, lifecycle workerów async oraz udokumentowane bezpieczeństwo FFI i `unsafe`.
Najpierw poprawność i czytelność; optymalizacje uzasadniamy wymaganiami lub pomiarami.

Każdy crate dziedziczy linty workspace, które odrzucają `dbg!`, `todo!`, `unimplemented!`, nieudokumentowane `unsafe`
oraz niejawne operacje unsafe wewnątrz unsafe fn. Wyjątki od lintów ograniczamy do najmniejszego zakresu
i uzasadniamy przez `reason`; preferujemy `#[expect(..., reason = "...")]`.
Formatowanie i Clippy uzupełniają przegląd według `itsolpowers:rust-review` i `itsolpowers:itsol-self-review`.
CI kompiluje i lintuje kod Metal i CUDA; nie potwierdza poprawności runtime'u GPU ani zgodności modelu.

## Pull requesty

- Jedna spójna zmiana na pull request; [szablon](.github/pull_request_template.md) wymienia, co podać.
- Tytuł w konwencji Angular (`feat(cli): …`, `fix(gpu): …`, `perf(metal): …`, `docs: …`): pull requesty łączymy
  przez squash, a tytuł staje się commitem.
- Dokumentacja (README, `docs/`, `serve.example.yml`) zmienia się razem z zachowaniem lub opcjami, które opisuje.
- Pull request wymaga jednej akceptacji opiekuna (`@itsoltech/basal`) i przechodzącego checku `lint`.

## Licencja

Wkład przyjmujemy na licencji projektu, [Apache License 2.0](LICENSE). Obowiązuje
[kodeks postępowania](CODE_OF_CONDUCT.md).
