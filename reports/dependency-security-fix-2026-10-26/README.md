# Aktualizacja zależności po audycie cargo deny

Status: PASS w opisanym zakresie. Zaktualizowano dwie biblioteki w Cargo.lock:

| Biblioteka | Przed | Po | Usunięty alert |
|---|---|---|---|
| ctutils | 0.4.3 | 0.4.4 | RUSTSEC-2026-0385 |
| tokio | 1.53.2 | 1.53.3 | RUSTSEC-2026-0386 |

Zmiana obejmuje tylko te wersje i ich sumy kontrolne. Cargo podczas update
przepięło także siedem istniejących zależności windows-sys; przywrócono
poprzednie powiązania. Końcowy lockfile przechodzi kontrole z --locked.
Nie zmieniano manifestów, kodu projektu ani wyjątków w deny.toml.
Oba alerty istniały przed refaktorem profili modeli.

Wydanie [ctutils 0.4.4](https://github.com/RustCrypto/utils/pull/1569)
poprawia stałoczasowe maskowanie i porównania u128 na platformach 32-bitowych.
Poprawka [Tokio AsyncFd](https://github.com/tokio-rs/tokio/pull/8587)
wprowadza jawny kontrakt rejestracji deskryptorów i deprecjonuje poprzednie
konstruktory. Przejrzano changelogi i manifesty obu wydań; wymagania wersji
Rust i ich zależności nie zmieniły się. Kod workspace nie wywołuje AsyncFd
bezpośrednio. Zielony audyt nie jest dowodem braku wszystkich podatności.

## Weryfikacja statyczna

- cargo deny --locked --all-features: advisories, bans, licenses, sources OK.
  Istniejący zatwierdzony wyjątek dla paste pozostaje bez zmian; kontrola
  jego zakresu i terminu także przeszła.
- Workspace release all-targets Clippy z -D warnings: default, CUDA
  all-features i Intel PASS.
- Workspace release all-features Rustdoc z prywatnymi elementami,
  bez zależności, z RUSTDOCFLAGS=-D warnings: PASS.
- cargo fmt --all --check, git diff --check oraz release build CLI CUDA: PASS.

Nie dodano testów, frameworków ani bramek CI.

## Porównanie lokalne

Sprzęt: RTX 4060, 8188 MiB VRAM, compute capability 8.9; CUDA 12.9.1,
cuBLASLt 120901, Rust 1.95.0. Checkpoint Remek/basal-2.0-mini, rewizja
cb81b32ce89b0b87e62fcdad28f0435e25c953f3. Obie binarki używają FP16,
tego samego snapshotu wag i jawnej tabeli GEMM o digeście
2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96.

Punktem odniesienia jest końcowa binarka z
[poprzedniej kampanii](../model-isolation-local-expanded-2026-10-26/README.md),
po naprawie konstruktorów, przed aktualizacją zależności. SHA-256 binarek:

- przed: 9d82d28c0dc2d4f23815838b11e93d01c5618bb83d4181bb160850049444b3b0;
- po: 3ac0a1155fe6fee310a8cb11ecdb07df0dcc291a1fca94d3e566b2c6b1122b5c.

Eksport tree: osiem rekordów bench i 33 SystemOne. Pełne JSONL są bajtowo
identyczne przed i po aktualizacji. Porównanie obejmuje także istniejące
błędy i puste wyniki korpusu; nie oznacza 41 poprawnych odpowiedzi.
basal compare potwierdza brak różnic tokenów, logitów i prawdopodobieństw.

HTTP: każda binarka obsłużyła 4 żądania rozgrzewkowe oraz 272 mierzone
(16 sekwencyjnych i po 64 przy C1, C4, C16 i C64). Wszystkie raportowane
fazy mają zero błędów, zero różnic prawdopodobieństw i pełnych answers
względem pierwszej odpowiedzi. Pełne answers dla czterech różnych wejść są
także identyczne pomiędzy binarkami po kanonizacji JSON.
Oba serwery zakończono SIGINT, z exit status 0.

Nie jest to kwalifikacja wydajności: to pojedyncze krótkie próby,
a baseline HTTP częściowo pokrywał się z kompilacją Rustdoc na CPU.
Statystyki czasowe zachowano bez wniosków o przyspieszeniu lub regresji.
Nie sprawdzano innych checkpointów, Metal ani rzeczywistego urządzenia Intel.
Nie deklarujemy braku regresji poza zmierzonym zakresem.

## Dane i odtworzenie

Nowa kampania: dependency-security-fix-2026-10-26. Nie nadpisywano poprzednich
raportów ani pomiarów. Normalny cache użytkownika pozostawiono bez zmian.
Nie tworzono commita i niczego nie publikowano. URI: brak.

Lokalne archiwum:
`.cache/measurement-artifacts/dependency-security-fix-2026-10-26/measurement-data.tar.gz`.
Rozmiar: 99 849 B; SHA-256:
`ae6c92231f45dfca1a1947b704787e72cd2730de4853bd49c5611c66e89a1a79`.
Rozpakowanie do osobnego katalogu potwierdziło 18/18 wpisów SHA256SUMS.

Logiczne dane pod reports/dependency-security-fix-2026-10-26/ obejmują
wyniki i wejścia, porównanie eksportów, odpowiedzi HTTP, podsumowanie audytu,
Cargo.lock i patch aktualizacji. REPRODUCE.md opisuje odtworzenie na snapshotcie
z poprzedniej kampanii. Wagi, binarki, prywatne logi i konfiguracje pozostają
poza archiwum. Przed odtworzeniem sprawdź SHA archiwum i SHA256SUMS.
