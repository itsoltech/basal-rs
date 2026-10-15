# SDK: przegląd supply chain, 2026-10-13

Niezależni recenzenci: GPT 6 Astra (XHigh) oraz Claude Opus 5.5 (XHigh).
Obie recenzje zakończone; rodzic zweryfikował poniższe ustalenia.
Zakres: cztery nieopublikowane SDK `0.1.0`, ich grafy konsumenckie,
budowanie i workflow wydawania. Przegląd był odczytowy. Ten raport nie
oznacza wdrożenia poprawek ani zatwierdzenia wydania.

**Werdykt: wymagane poprawki przed pierwszą publikacją.**

## Ustalenia wymagające zmian

1. **Blocker / High — Rust kompiluje zależności z uprawnieniami publikacji.**
   `.github/workflows/sdk-release.yml:173–190`: job ma `id-token: write`,
   bootstrap token znajduje się w środowisku całego joba, a zwykłe
   `cargo publish` ponownie buduje pakiet. Złośliwy build script lub
   procedural macro przyjęty z zależnością może odczytać token albo sam
   uzyskać token OIDC. Wcześniejszy build bez sekretów tego nie usuwa.
   Kompilację i weryfikację należy wykonać w osobnym jobie bez uprawnień
   publikacji; publisher nie może wykonywać kodu zależności. Przykładowy
   kierunek: `cargo publish --locked --no-verify`, kontrola tożsamości
   zawartości względem wcześniej zbudowanego `.crate` oraz bootstrap token
   tylko w kroku publikacji. Samo przesunięcie auth na koniec tego samego
   joba nie odbiera wcześniejszym krokom uprawnienia OIDC.
   [Semantyka Cargo publish](https://doc.rust-lang.org/cargo/commands/cargo-publish.html).

2. **High — niezamrożone narzędzia Pythona i wspólny builder wszystkich SDK.**
   `tools/sdk/requirements-build.txt:1–2`, `sdk/python/pyproject.toml:2`,
   `.github/workflows/sdk-release.yml:26–96`: przypięto tylko bezpośrednie
   `build` i `twine`; zależności pośrednie i backend Hatchling są rozwiązywane
   podczas wydania. Przejęta nowa wersja narzędzia może zmienić artefakty
   lub źródła pozostałych SDK w tym samym jobie. Potrzebne są osobne joby
   i artefakty dla czterech paczek, kompletny lock narzędzi/backendu z hashami
   oraz build bez ponownego rozwiązywania zależności. `--require-hashes`
   i `--only-binary=:all:` ograniczają niekontrolowane instalacje narzędzi.
   Zmiany locka nadal wymagają oceny: hash nie dowodzi, że kod jest uczciwy.
   [Bezpieczna instalacja pip](https://pip.pypa.io/en/stable/topics/secure-installs/).

3. **High / zgodność API — feature `arbitrary_precision` wycieka do konsumentów Rust.**
   `sdk/rust/Cargo.toml:21` dziedziczy oba feature'y JSON z workspace.
   Spakowany manifest potwierdza `preserve_order` i `arbitrary_precision`.
   Parent odtworzył na serde_json `1.0.151`: `{"x":0.5}` nie deserializuje
   się do enumu `untagged` z polem `f64` ani do struktury z takim polem
   poprzez `flatten`. Bez `arbitrary_precision` oba przypadki przechodzą.
   Dodanie SDK może więc zmienić zachowanie niezwiązanego kodu aplikacji.
   SDK powinno deklarować własną zależność serde_json z `preserve_order`,
   bez wymuszania `arbitrary_precision`. To problem zgodności, nie dowód RCE.
   Ograniczenia reprezentacji dużych liczb JSON trzeba opisać; duże
   identyfikatory najbezpieczniej przekazywać jako stringi. Pythonowe `int`
   zachowuje dowolną precyzję — nie należy utożsamiać go tutaj z JS Number.

4. **Should / High — obraz publikujący PyPI jest wskazany mutowalnym tagiem.**
   `.github/workflows/sdk-release.yml:135`: kod akcji jest przypięty SHA,
   ale dokładnie ta wersja generuje referencję `docker://ghcr.io/{repo}:{ref}`.
   Tag nazwany SHA nie jest digestem obrazu. Przejęcie uprawnień zapisu do
   obrazu upstream może zmienić wykonawcę publikacji bez zmiany workflow.
   Potrzebny jest zweryfikowany digest obrazu albo rozwiązanie gwarantujące
   jego niezmienność. To nie zarzut braku attestations: akcja PyPI je obsługuje.
   [Kod przypiętej akcji](https://github.com/pypa/gh-action-pypi-publish/blob/dc37677b2e1c63e2034f94d8a5b11f265b73ba33/create-docker-action.py).

5. **Should / Medium — Debug błędu Rust wypisuje body.**
   `sdk/rust/src/lib.rs:202–225`: `Display` jest bezpieczny, ale wygenerowany
   `Debug` zawiera `Error::Http.body`. Przykład z `main() -> Result` może
   wypisać to body do stderr. Jeśli serwer/proxy odbije w błędzie treść
   dokumentu lub inne dane, trafią one do logów. Zalecany ręczny redagujący
   Debug z zachowaniem jawnego dostępu do body i łańcucha przyczyn.

## Zależności i ich redukcja

Liczby nie obejmują samego SDK, aplikacji konsumenta ani platformowego
runtime'u Node/.NET. Dla Rust `normal` obejmuje również makra proceduralne,
więc nie należy całej liczby nazywać zależnościami runtime.

- **TS/JS:** zero zewnętrznych zależności runtime. Lock buildera zawiera
  TypeScript i 20 opcjonalnych binarek platformowych; Linux instaluje
  TypeScript i jedną binarkę. Opcjonalny kierunek redukcji buildera:
  kompilator TypeScript napisany w JS, o odpowiedniej zgodności i utrzymaniu.
- **C#:** zero zewnętrznych zależności NuGet. `HttpClient` i `System.Text.Json`
  pochodzą z frameworka. Zalecane jawne źródła restore i ustawienia builda CI.
- **Python:** siedem paczek w grafie obecnego konsumenta CPython 3.11:
  httpx, httpcore, anyio, certifi, h11, idna, typing-extensions. Markery zależą
  od wersji Pythona. Obaj recenzenci uznali HTTPX za uzasadniony dla pooling,
  TLS oraz prawdziwego async. Sync ze stdlib może mieć zero zależności;
  `asyncio.to_thread` nie zapewnia równoważnego anulowania i lifecycle.
  Builder można uprościć np. przez `flit_core`, bez jego własnych zależności,
  zamiast Hatchling. Nie oznacza to zera zależności całego procesu publikacji.
- **Rust:** świeży konsument Linux ma 99 unikalnych paczek `normal,build`:
  80 w `normal,no-proc-macro`, 11 wyłącznie w gałęziach makr, osiem dodatkowo
  w gałęziach build. macOS ARM64 daje 101, Windows MSVC 99, licząc rozwiązany
  graf targetu na hoście Linux. Spakowany lock SDK daje odpowiednio 98/100/98;
  nie ogranicza on przyszłego grafu rozwiązywanego przez aplikację użytkownika.

Parent potwierdził odczytowym rozwiązaniem alternatywnego manifestu na Linux:
**Hyper HTTP/1 + hyper-util + Tokio + Rustls/ring + platform verifier: 58 paczek**.
W tym grafie nie ma AWS-LC ani CMake; nadal występuje natywny kod `ring`.
To graf proponowanego rozwiązania, nie wdrożona lub uruchomiona implementacja.
Zmiana wymaga świadomego określenia obsługi proxy i IDN; nie wolno deklarować
pełnej zgodności transportów tylko na podstawie mniejszej liczby paczek.
Sama zamiana feature'a Reqwest na `rustls-no-provider` bez konfiguracji
providera jest błędna. Usunięcie `thiserror` daje tylko dwa crate'y mniej;
usunięcie bezpośredniego indexmap nie usuwa go z serde_json/preserve_order.

## Pozostałe zalecenia i granice

- Audytować graf runtime Python i buildery Python/npm osobno; `twine check`
  sprawdza metadane, nie advisory. Zakres audytu Rust rozszerzyć o Windows,
  jeśli jest wspieranym systemem SDK.
- Konfiguracja kont: chronione tagi i środowiska wydawania, dokładne powiązanie
  trusted publishera ze środowiskiem, kontrola dostępu do scope i nazw paczek.
  Nie sprawdzano zdalnych ustawień; nie stwierdzono ich niepoprawności.
- Rozważyć konfigurowalny limit odpowiedzi HTTP. To ograniczenie DoS,
  nie dowód możliwości przejęcia komputera.
- W aplikacjach stosować locki/constraints i przegląd aktualizacji, z możliwością
  przyjmowania poprawek bezpieczeństwa. Exact pin bezpośredniej zależności
  biblioteki nie zamraża całego grafu konsumenta.
- OIDC/provenance nie dowodzi, że źródła buildera były bezpieczne. Mały graf
  także nie daje gwarancji bezpieczeństwa; platformowe runtime'y i CA są
  częścią zaufanego stosu poza tym licznikiem.

Nie wykonywano publikacji, odczytu sekretów, pełnych rebuildów ani inferencji.
Nie wdrożono alternatywnego transportu i nie sprawdzono HTTPS na wszystkich
systemach. Stary advisory `unsafe-libyaml` dotyczy silnika, nie grafu SDK.
Nie dodano testów ani frameworków; diagnostyka odtwarzająca problem Serde
i rozwiązywanie alternatywnego grafu odbyły się poza repozytorium.
