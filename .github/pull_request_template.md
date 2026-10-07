<!--
Tytuł PR w konwencji Angular, np. `feat(cli): …`, `fix(gpu): …`; po squash merge staje się commitem.
PR title in the Angular convention, e.g. `feat(cli): …`, `fix(gpu): …`; it becomes the commit after the squash merge.
Zasady / Rules: CONTRIBUTING.md, CONTRIBUTING.pl.md
-->

## Co i dlaczego / What and why

<!-- Closes #… -->

## Numeryka / Numerics

- [ ] Bez zmiany obliczeń: wynik bitowo równy `main` / No change in computation: output bit-identical to `main` (`basal compare` → 0.0): <!-- model, backend -->
- [ ] Zmiana numeryki: porównanie z FP32 upstream, wyniki w nowym katalogu `reports/` / Numerics change: compared with upstream FP32, results in a new `reports/` directory
- [ ] Nie dotyczy (dokumentacja, CI, samo CLI) / Not applicable (docs, CI, CLI only)

## Wydajność / Performance

- [ ] Bez tezy o wydajności / No performance claim
- [ ] Pomiar przed i po, ten sam sprzęt i precyzja, w nowym katalogu `reports/` / Measured before and after, same hardware and precision, in a new `reports/` directory

## Sprawdzenie / Checks

- [ ] `cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`
- [ ] Kod CUDA: build i uruchomienie z `--features basal-cli/cuda` / CUDA code: built and run with `--features basal-cli/cuda` on: <!-- GPU, compute capability -->
- [ ] Kod Metal: build i uruchomienie / Metal code: built and run on: <!-- chip -->
- [ ] Dokumentacja zaktualizowana przy zmianie zachowania lub opcji / Docs updated when behaviour or options change
- [ ] Bez nazw hostów, adresów IP i ścieżek domowych; `.baseline/` nietknięte; istniejące raporty nienadpisane / No hostnames, IP addresses or home paths; `.baseline/` untouched; existing reports not overwritten
