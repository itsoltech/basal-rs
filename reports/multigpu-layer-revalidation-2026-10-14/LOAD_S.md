# Podział warstw — erratum zakresu `load_s`, 2026-10-14

Dotyczy czterech miejsc, które opisują koniec zegara `load_s` w wersji
bazowej `d42a400` (BASE) i w kandydacie (CAND, źródła `831e68e0…`):

- [zestawienie](README.md), sekcja „Zakres twierdzeń”, punkt `load_s`
  (wiersze 170–173),
- [raport L3](../multigpu-layer-l3-revalidation-2026-10-14/README.md), punkt
  „Poza oknem `dec_s`” (wiersze 475–479),
- [rozstrzygnięcie L3](../multigpu-layer-l3-revalidation-2026-10-14/DECISION.md),
  punkt o `load_s` A i B (wiersze 95–98),
- [fakty L3](../multigpu-layer-l3-revalidation-2026-10-14/facts.json), pole
  `memory_sources.load_s`.

Te pliki pozostają bez zmian. To erratum zastępuje tylko opisane niżej
twierdzenie o zakresie `load_s`.

## Błędne twierdzenie

Wszystkie cztery miejsca podają, że w CAND zegar zatrzymuje się przed
wczytaniem głowicy evidence, utworzeniem uchwytu cuBLASLt i synchronizacją
publikacji, a w `d42a400` po nich, czyli że `load_s` BASE obejmował te kroki.
Część o `d42a400` jest błędna. Część o CAND jest poprawna.

## Źródła

Plik `crates/basal-gpu/src/lib.rs` obu zmierzonych binarek:

| Wersja | SHA-256 pliku | Start zegara | Synchronizacja po wysłaniu wag | Odczyt `load_s` | Głowica evidence | Uchwyt cuBLASLt | Synchronizacja publikacji |
|---|---|---|---|---|---|---|---|
| BASE `d42a400` | `0652d6489766725e0dc5f3339a2ada461245d5bddd2b6e51364ae67b2e36322a` | 657 | 857 | 881 | 896 | 904 | brak |
| CAND `831e68e0…` | `1478871fa1672dd6a8766f0e3ef6cb42187902b9e236d01283e336a7d9e8327e` | 1357 | 1625 | 1642 | 1660 | 1677 | 1682 |

Liczby to wiersze pliku. Skrót pliku BASE jest równy wpisowi tego pliku w
rejestrze źródeł pary BASE/CAND, z której zbudowano zmierzone binarki.

- BASE: `GpuBackend::load` zwraca wyrażenie struktury `Ok(Self { … })`.
  Pole `load_s: t0.elapsed().as_secs_f64()` (881) stoi w nim przed polem
  `evidence` z `EvidenceHead::load` (896) i polem `lt` z
  `cublaslt::Lt::new` (904). Pola wyrażenia struktury są obliczane w
  kolejności zapisu, więc zegar jest odczytany przed oboma krokami. Po nich
  BASE nie ma osobnej synchronizacji publikacji.
- CAND: `GpuBackend::load_stages` odczytuje `load_s` (1642) przed
  wczytaniem głowicy evidence (1660), utworzeniem uchwytu cuBLASLt (1677) i
  synchronizacją publikacji (1682).

## Poprawny opis

- W obu wersjach `load_s` kończy się przed wczytaniem głowicy evidence i
  utworzeniem uchwytu cuBLASLt. Żadna z nich nie wlicza tych kroków.
- Obie wersje odczytują zegar po synchronizacji po wysłaniu wag.
- Synchronizację publikacji po głowicy i uchwycie ma tylko CAND. Ona także
  jest poza `load_s`.
- Żaden z tych zegarów nie jest pełnym czasem uruchomienia.

To erratum nie ustala, że BASE i CAND mierzą w `load_s` dokładnie tę samą
pracę, i nie wyjaśnia różnic `load_s` zapisanych w L3. Ich przyczyna
pozostaje nieustalona, jak podaje raport L3. Zdania, że różnice `load_s` A
i B nie są porównaniem czasu uruchomienia, pozostają w mocy z powodów
podanych wyżej, a nie z powodu innego końca zegara.

## Co się nie zmienia

Zapisane wartości `load_s`, `dec_s` i pozostałe liczby, statusy PASS, FAIL,
GAP i SIGNAL, reguła bez progu, decyzja użytkownika dla L2 i rozstrzygnięcie
koordynatora dla `q8_0` i `q4_k` w L3 są bez zmian. `load_s` nie jest polem
reguły. Erratum nie dodaje metryki, progu ani nowej decyzji.
