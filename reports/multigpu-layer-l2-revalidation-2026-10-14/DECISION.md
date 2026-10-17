# Decyzja użytkownika o wyniku L2, 2026-10-14

> **Dane JSON w archiwum.** Pliki `facts.json`, `provenance.json` i `validation.json` z tego katalogu są poza
> Gitem, w archiwum pomiarów `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod ścieżką
> `reports/multigpu-layer-l2-revalidation-2026-10-14/<plik>`. Odwołania do nich w tekście dotyczą plików z
> archiwum. Pobieranie i sumy kontrolne: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md).
> Archiwum zawiera też wersję tego pliku sprzed dopisania tej noty. Przeniesienie danych nie jest nowym
> pomiarem: liczby, statusy i rozstrzygnięcia raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

Dotyczy [raportu L2](README.md) i `facts.json` (w archiwum). Raport i jego
pliki pozostają bez zmian. Zdanie raportu „Decyzja nie jest podjęta” opisuje
stan sprzed tej decyzji; aktualny stan opisuje ten plik.

**Pytanie:** czy przyjąć utrzymany SIGNAL `basal benchmark`
`mixed-c1 queue_ms.p99` dla CAND `169aedfc…` (źródła `831e68e0…`) wobec
BASE `44c8e399…` i kontynuować L3. Pary A → B: +1,237 ms (+1,4565%),
+2,706 ms (+3,2066%), +11,330 ms (+12,7872%) i +3,896 ms (+4,1763%).
Nie znaleziono możliwej do wskazania przyczyny w kodzie. Analiza nie
wyklucza wpływu CPU hosta i szeregowania wątków, których telemetria nie
obejmuje. Latencja end-to-end i przepustowość `mixed-c1` nie mają
utrzymanego SIGNAL według ustalonej reguły.

**Odpowiedź użytkownika:** „Przyjąć wynik i przejść do L3 (rekomendowane)”.

**Zakres:** użytkownik przyjął ten konkretny zmierzony wynik dla tej pary
binarek, z ograniczeniami opisanymi w raporcie. Decyzja:

- nie zmienia statusu runnera: SIGNAL pozostaje SIGNAL-em;
- nie ustanawia ogólnego progu (np. 1–2%) i nie stwierdza, że różnica jest
  szumem ani że regresji nie ma;
- nie akceptuje etapu 05 i nie zastępuje końcowego review.

Etap przechodzi do zaplanowanej sesji L3 dla tej samej pary binarek. Na
dzień tej decyzji L3 jest w przygotowaniu i nie ma jej wyniku.
