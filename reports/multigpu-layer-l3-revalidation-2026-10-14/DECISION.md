# Rozstrzygnięcie koordynatora o wyniku L3 `q4_k`, 2026-10-14

> **Dane JSON w archiwum.** Pliki `facts.json`, `provenance.json` i `validation.json` z tego katalogu są poza
> Gitem, w archiwum pomiarów `multigpu-2026-10-14` (SHA-256 `71acff72…`), pod ścieżką
> `reports/multigpu-layer-l3-revalidation-2026-10-14/<plik>`. Odwołania do nich w tekście dotyczą plików z
> archiwum. Opis końca zegara `load_s` w tym pliku poprawia
> [erratum `load_s`](../multigpu-layer-revalidation-2026-10-14/LOAD_S.md). Pobieranie i sumy kontrolne:
> [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Archiwum zawiera też wersję tego pliku
> sprzed dopisania tej noty. Przeniesienie danych nie jest nowym pomiarem: liczby, statusy i rozstrzygnięcia
> raportu są bez zmian. Bieżący stan funkcji i wyników opisuje
> [zestawienie z 2026-10-14](../multigpu-results-2026-10-14/README.md).

Dotyczy [raportu L3](README.md) i `facts.json` (w archiwum). Raport i jego
pliki pozostają bez zmian. Zdania raportu „Ta decyzja jeszcze nie zapadła”
i sekcja „Decyzja użytkownika (oczekująca)” opisują stan sprzed tego
rozstrzygnięcia i są historyczną migawką. Aktualny stan opisuje ten plik.

## Kto rozstrzygnął i na jakiej podstawie

Rozstrzygnął koordynator programu (ROOT), 2026-10-14 (06:39 UTC). Podstawą
jest wcześniejsza instrukcja użytkownika. Użytkownik nie oceniał tego
raportu ani tego wyniku, więc ten plik nie zapisuje jego nowej zgody.

Instrukcję użytkownik wydał przy decyzji o wyniku S3 W4A16
([decyzja](../../.itsol/initiatives/basal-multigpu/phases/04-w4a16/performance-decision.md),
[raport S3](../w4a16-s3-2026-10-12/README.md)). Brzmi dosłownie:

> Może zapiszmy stan na chwilę obecną, a spadek do 1-2% jest w granicy
> normy, czyli te 0.11% regresji jest w normie

Późniejsza instrukcja użytkownika: „kontynuuj”. Koordynator zastosował
podaną tam granicę spadku 1–2% do jednego, konkretnego wyniku opisanego
niżej. Raport L3 zgodnie z planem przewidział osobną decyzję użytkownika.
Koordynator uznał, że istniejąca instrukcja użytkownika ma pierwszeństwo
przed ponownym pytaniem o wynik mieszczący się w podanej granicy.

## Rozstrzygany wynik

Utrzymany SIGNAL runnera `perf L3 q4_k: persistent regression signal`, pole
`bench: dec_s` (`basal bench`, batch z budżetem tokenów, 44 pytania):

- B = CAND `169aedfc…`: `d42a400` + zmiany etapu 05 + poprawka po review R1
  + scalenie `main` `4f68eb2`, źródła `831e68e0…` (66 plików);
- A = BASE `44c8e399…`: przebudowane `d42a400`, źródła `94ef2131…`
  (65 plików);
- obie na GPU 0, `q4_k`, ta sama sesja L3 (2026-10-14 UTC), te same pary co
  w raporcie.

| Runda | Para | `dec_s` A | `dec_s` B | B/A | zmiana | przebieg 44 pytań B−A |
|---|---:|---:|---:|---:|---:|---:|
| główna | 0 | 56,50732 | 56,41085 | 0,998293 | −0,1707% | +1,332 ms |
| główna | 1 | 56,42699 | 56,42107 | 0,999895 | −0,0105% | +0,082 ms |
| główna | 2 | 56,36410 | 56,05061 | 0,994438 | −0,5562% | +4,366 ms |
| główna | 3 | 56,37587 | 56,28809 | 0,998443 | −0,1557% | +1,217 ms |
| potwierdzenie | 0 | 56,50918 | 56,47005 | 0,999307 | −0,0693% | +0,540 ms |
| potwierdzenie | 1 | 56,48337 | 56,37773 | 0,998130 | −0,1870% | +1,460 ms |
| potwierdzenie | 2 | 56,50446 | 56,49680 | 0,999864 | −0,0136% | +0,106 ms |
| potwierdzenie | 3 | 56,46153 | 56,42929 | 0,999429 | −0,0571% | +0,445 ms |

B ma niższe `dec_s` we wszystkich 8 parach. Spadek wynosi od 0,0104938% do
0,5561912%. Przebieg 44 pytań, z którego pochodzi mediana `dec_s`, trwa w B
dłużej o +0,082 do +4,366 ms (mediana par +0,878 ms) przy 778,6–785,0 ms na
przebieg. To czas przebiegu `basal bench`, a nie latencja żądania HTTP.
Największy spadek, 0,5562%, jest mniejszy niż granica 1–2% z instrukcji
użytkownika.

Podstawa rozstrzygnięcia:

- sesja L3 jest kompletna: 172 PASS, 0 FAIL, 2 GAP, 1 SIGNAL, kod wyjścia 4;
- surowe dane, binarki i źródła są powiązane z raportem; koordynator
  przeliczył dane raportu z surowych plików (4033 kontrole, 0 błędów);
- analiza danych i przegląd kodu tylko do odczytu nie wskazały przyczyny w
  kodzie, więc nie wprowadzono spekulacyjnej optymalizacji.

## Zakres

Koordynator przyjmuje ten zmierzony wynik tej pary binarek, z
ograniczeniami opisanymi w raporcie. Rozstrzygnięcie:

- nie zmienia statusów runnera: SIGNAL pozostaje SIGNAL-em, oba GAP-y
  pozostają (w tym niezmierzony HTTP jednego GPU w `q8_0` i `q4_k`);
- nie dodaje progu do runnera ani reguły dla przyszłych wyników;
- nie stwierdza, że różnica jest szumem ani że regresji nie ma; nie
  dotyczy też nieutrzymanych pól `q8_0` `dec_s` i `q4_k` `lat2_ms` (B gorsze
  w 7 z 8 par), które nie dowodzą braku narzutu;
- nie przenosi [decyzji użytkownika o L2](../multigpu-layer-l2-revalidation-2026-10-14/DECISION.md),
  która dotyczy tylko tamtego wyniku;
- nie jest akceptacją etapu 05 ani końcowym review; realny defekt znaleziony
  w niezależnym review nadal wymaga poprawki;
- nie zmienia zaleceń trybów: domyślny i zalecany jest `none`, `q8_0` jest
  opcjonalny, a `q4_k` opcjonalny i eksperymentalny; oba mają jawne różnice
  jakości wobec FP32 opisane w decyzjach jakościowych.

## Doprecyzowanie dwóch sformułowań raportu

Raport pozostaje bez zmian. Te dwa sformułowania należy czytać tak:

- „CAND nie dodaje synchronizacji GPU, alokacji, kopii ani kernela” i
  „Nie doszło […] alokacja” dotyczą **alokacji pamięci GPU**. Po stronie
  hosta każde wywołanie `letter_logits` buduje właścicieli bramek
  (`gate_owners` w `crates/basal-gpu/src/lib.rs`): dwa wektory (bramki i
  urządzenia etapów, przy jednym GPU po jednym elemencie) oraz klony
  uchwytów urządzenia i stanu zdrowia. Raport wymienia te kopie, ale sesja
  nie mierzyła ich czasu.
- `load_s` A i B mierzy inny zakres. W CAND zegar zatrzymuje się przed
  wczytaniem głowicy evidence, utworzeniem uchwytu cuBLASLt każdego etapu i
  synchronizacją publikacji, a w `d42a400` po nich. Różnice `load_s` w
  raporcie nie są porównaniem czasu uruchomienia.

Zestawienie bieżących sesji L1, L2 i L3:
[multigpu-layer-revalidation-2026-10-14](../multigpu-layer-revalidation-2026-10-14/README.md).
