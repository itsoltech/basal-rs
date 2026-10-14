# Sesja L1 — poprawka poleceń ilustracyjnych

Dotyczy sekcji „Polecenia (ilustracyjne)” w [README.md](README.md) tego
katalogu. README, `facts.json`, `provenance.json` i `validation.json`
pozostają bez zmian.

Linia `compare` w tej sekcji ma błędną składnię:

```sh
<candidate> compare <out>/base <out>/split
```

`basal compare` przyjmuje tylko nazwane argumenty `--a`, `--b` i `--out`
(`Cmd::Compare` w `crates/basal-cli/src/main.rs`). Poprawna postać:

```sh
<candidate> compare --a <out>/base --b <out>/split --out <out>/compare.json
```

Runner sesji L1 wywołał `compare` w tej poprawnej postaci (`--a`, `--b`,
`--out`) we wszystkich 44 parach C1 i C2. Błąd dotyczy tylko ilustracji w
README. Wyniki, statusy i liczby raportu L1 się nie zmieniają.

Pełny blok C2 po poprawce (`<candidate>`, `<base>`, `<models>`, `<out>` i
`<table>` to symbole zastępcze):

```sh
# C2: eksport z podziałem (CAND) i referencja jednego GPU (BASE)
<candidate> export --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --batching tree --layer-split 0:30,1:30 --inputs reports/reference-basal-1.5-4.5B-fp32 --out <out>/split
<base> export --model <models>/basal-1.5-4.5B --gemm-table crates/basal-cli/gemm-tables/<table> \
  --batching tree --device 0 --inputs reports/reference-basal-1.5-4.5B-fp32 --out <out>/base
cmp <out>/split/bench.jsonl <out>/base/bench.jsonl
<candidate> compare --a <out>/base --b <out>/split --out <out>/compare.json
```
