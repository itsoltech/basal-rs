# Integracja multi-GPU z aktualnym main — 17 października 2026

## Cel i zakres

Połączono gałąź multi-GPU z `main` na commicie
`6f38056f103b020fbcf9e5a28e7eeabc119ebec1`. Punktem wyjścia PR był
`0af3f4e471a044fbbf9bad2f976d4aa8060bd895`.

Zachowano Basal 2.0 mini, 4.5B i max, Omni mini i max, nowe tablice GEMM dla Ada,
wybór precyzji i profilu inferencji, nowe SDK oraz poprawkę zależności Candle.
SDK, kontrakt API, tablice GEMM, kod kerneli CUDA i vendored Candle są identyczne
z powyższym main. Nie dodano nowych modeli Omni poza modelami obecnymi w main.

## Połączone zachowanie

| Architektura | Przypisanie GPU | Repliki | Podział warstw | Kwantyzacja projekcji |
|---|---|---|---|---|
| Llama: Basal 1.5 i Basal 2.0 | Tak | Tak | CUDA, f16, fused | Opcjonalne q8_0 lub eksperymentalne q4_k; q4_k wymaga zgodnych wymiarów K |
| Gemma: Omni mini i max | Tak | Tak | Odrzucony | Odrzucona; wymagane none |

Tabela opisuje zaimplementowane ścieżki, nie wynik nowych pomiarów sprzętowych.
Domyślnym trybem kwantyzacji pozostaje `none`. Nie przenosimy wyników jakości
Basal 1.5 na nowe modele Basal 2.0.

Każde urządzenie zachowuje własną bramkę wykonywania. Serwer łączy routing replik
z limitami żądań, mediów i dużych wejść z main. Precyzja modelu jest dobierana dla
wybranego urządzenia. Profil `auto` stosuje optymalizację przypisaną do RTX 6000
Ada tylko w jej dotychczasowym zakresie; przy podziale Llama warunek musi być
spełniony na wszystkich kartach.

Omni może używać dopasowanej tablicy GEMM. Bez niej zachowuje stałe operacje GEMM
na pojedynczym prompcie, zgodnie z main. Nie oznacza to potwierdzonego wzrostu
przepustowości replik. Dla kilku replik jawne `gemm_table: none` pozostaje
odrzucone przez walidację konfiguracji; brak dopasowanej tabeli dla Omni
w trybie `auto` jest osobnym, obsługiwanym przypadkiem.

Nieobsługiwany podział lub kwantyzacja Omni kończą się jawnym błędem po odczycie
manifestu i architektury, przed ładowaniem wag oraz tablicy GEMM. Przy modelu
pobieranym z Hugging Face samo pobieranie może nastąpić wcześniej.

## Zmiany i ograniczenia konfiguracji

- Zachowano ścisły parser YAML z PR. W przeciwieństwie do main odrzuca on skróty
  `yes/no/on/off` jako wartości logiczne i klucze scalania `<<`. Należy używać
  jawnych `true/false` i zwykłych map.
- `doctor` nie rozpoznaje architektury Omni przed pobraniem modelu. Ograniczenia
  architektury są egzekwowane podczas uruchomienia serwera i ładowania modelu.
- Omni zachowuje brak raportowania utraty urządzenia z main. Stan replik wykrywa
  zamkniętego workera, ale nie potwierdza sprawności jego GPU.

## Weryfikacja

Lokalnie zakończono z kodem 0:

- `cargo fmt --all --check`;
- `cargo clippy --locked --workspace --release --all-targets -- -D warnings`;
- Clippy i Rustdoc dla Intel (weryfikacja implementacji);
- `cargo clippy --locked --workspace --release --all-features --all-targets -- -D warnings`
  oraz Rustdoc z `-D warnings`, `--all-features`, `--release`,
  `--no-deps --document-private-items` w środowisku CUDA 12.9.1, Rust 1.95.0;
- `cargo deny --locked --all-features check --hide-inclusion-graph`;
- kontrolę wyjątku `paste`, wersji SDK i białych znaków względem aktualnego main.

Końcowe, niezależne przeglądy Opus 5.5 HIGH i GPT-6 Astra XHIGH zatwierdziły
integrację bez uwag blokujących. Przeglądy obejmowały ładowanie modeli,
przypisanie urządzeń, precyzję, podział warstw, bramki urządzeń, media,
rezerwacje replik i zwalnianie limitów żądań po anulowaniu.
Status wymaganych kontroli CI jest dostępny w [PR #7](https://github.com/itsoltech/basal-rs/pull/7/checks).

Pomiary na GPU zostały jawnie odłożone przez użytkownika do etapu po mergu.
W tej integracji nie uruchomiono inferencji na GPU i nie zebrano nowych
danych modelu ani wydajności.

Historyczne pomiary multi-GPU są opisane w
[raporcie zbiorczym](../multigpu-results-2026-10-14/README.md). Ich wersje kodu,
modele, regresje i ograniczenia pozostają aktualnymi granicami tych dowodów;
nie stanowią walidacji nowych kombinacji po tej integracji.

## Niezmierzone połączenia funkcji

Do sprawdzenia pozostają: Omni na urządzeniu innym niż 0 i w replikach,
izolacja rzeczywistych mediów, nowe modele Llama z kwantyzacją lub podziałem
warstw, połączenie profilu `balanced` z kwantyzacją oraz zgodność pojedynczego
GPU z aktualnym main. `auto` może łączyć kwantyzację Basal 2.0 z profilem
`balanced` na zgodnym GPU; skutków jakościowych i wydajnościowych tego
połączenia tutaj nie zmierzono. Kompilacja ani zatwierdzone review nie
zastępują tych pomiarów.
