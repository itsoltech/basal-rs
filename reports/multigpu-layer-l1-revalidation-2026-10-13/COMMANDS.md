# Rewalidacja L1 — poprawka poleceń ilustracyjnych

> **Dane JSON w archiwum.** Pliki `facts.json`, `provenance.json` i `validation.json` z tego katalogu są poza
> Gitem, w archiwum pomiarów `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod ścieżką
> `reports/multigpu-layer-l1-revalidation-2026-10-13/<plik>`. Odwołania do nich w tekście dotyczą plików z
> archiwum. Pobieranie i sumy kontrolne: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).
> Archiwum zawiera też wersję tego pliku sprzed dopisania tej noty. Przeniesienie danych nie jest nowym
> pomiarem: liczby, statusy i rozstrzygnięcia raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

To erratum dotyczy tylko przykładu w sekcji „Polecenia (ilustracyjne)” w
[README.md](README.md) tego katalogu. Nie jest nowym pomiarem. README,
`facts.json`, `provenance.json` i `validation.json` pozostają bez zmian.
Wyniki, statusy i liczby raportu się nie zmieniają.

Linia `compare` w tej sekcji ma błędną składnię:

```sh
<candidate> compare <out>/base <out>/split
```

`basal compare` nie przyjmuje argumentów pozycyjnych. `Cmd::Compare` w
`crates/basal-cli/src/main.rs` ma trzy wymagane nazwane argumenty:

```rust
Compare {
    #[arg(long)]
    a: PathBuf,
    #[arg(long)]
    b: PathBuf,
    #[arg(long)]
    out: PathBuf,
},
```

Poprawna postać:

```sh
<candidate> compare --a <out>/base --b <out>/split --out <out>/compare.json
```

Runner tej sesji wywołał `compare` 44 razy, zawsze w tej postaci: `--a`
z eksportem BASE, `--b` z eksportem CAND i `--out` z plikiem JSON. Wszystkie
wywołania zakończyły się kodem 0. Było to 20 par C1 i 24 pary C2, czyli
dokładnie 44 pary z `facts.json`. Kroki C5 nie wywoływały `compare`.
Linie `export` i `profile` w README mają te same flagi co odpowiadające im
rzeczywiste polecenia runnera, więc nie wymagają poprawki.

Pełny blok C2 po poprawce. `<candidate>`, `<base>`, `<models>`, `<out>` i
`<table>` to symbole zastępcze:

```sh
# C2: eksport z podziałem (CAND) i referencja jednego GPU (BASE)
<candidate> export --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --batching tree --layer-split 0:30,1:30 --inputs reports/reference-basal-1.5-4.5B-fp32 --out <out>/split
<base> export --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --batching tree --device 0 --inputs reports/reference-basal-1.5-4.5B-fp32 --out <out>/base
cmp <out>/split/bench.jsonl <out>/base/bench.jsonl
<candidate> compare --a <out>/base --b <out>/split --out <out>/compare.json
```

`cmp` ilustruje porównanie bajtowe. Runner nie uruchamiał osobnego procesu
`cmp`, tylko sam liczył równość bajtową, a raport policzył ją ponownie.
