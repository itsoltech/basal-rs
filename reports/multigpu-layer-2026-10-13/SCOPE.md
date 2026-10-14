# Podział warstw — poprawka zakresu zestawienia, 2026-10-13

Dotyczy [zestawienia](README.md) w tym katalogu. Zestawienie i raporty sesji
([L1](../multigpu-layer-l1-2026-10-13/README.md) z
[poprawką poleceń](../multigpu-layer-l1-2026-10-13/COMMANDS.md),
[L2](../multigpu-layer-l2-2026-10-13/README.md),
[L3](../multigpu-layer-l3-2026-10-13/README.md)) oraz ich `facts.json`,
`provenance.json` i `validation.json` pozostają bez zmian. Liczby, statusy i
wnioski poszczególnych sesji się nie zmieniają.

## Sprzęt między sesjami

Zestawienie podaje, że trzy sesje wykonano „na jednej maszynie” (wstęp) i
opisuje je jako „Jedna maszyna, dwie identyczne karty RTX 6000 Ada 48 GB, po
jednej sesji” (Ograniczenia). Dane tego nie ustalają. Poprawny opis:

- L1, L2 i L3 to trzy osobne wynajęte sesje. Każda działała na maszynie z
  dwiema NVIDIA RTX 6000 Ada Generation 48 GB.
- We wszystkich trzech sesjach zgadzają się zapisane parametry: model kart,
  pamięć, sterownik 580.126.09, limit mocy i układ kart.
- Dane nie zawierają identyfikatorów fizycznych kart ani maszyn. Nie wiadomo,
  czy sesje działały na tej samej maszynie i tych samych kartach.

Porównania w obrębie jednej sesji, czyli pary A/B toru jednego GPU i podział
wobec jednego GPU tego samego kandydata, nie zależą od tej poprawki.

Zestawienia wyników różnych sesji są tylko opisowe. Nie są porównaniem na
tym samym sprzęcie. Dotyczy to zdania z sekcji „Wydajność”: „w trybach
kwantyzowanych różnica w `basal bench` jest większa niż w `none`”. Wartość
`none` pochodzi z L2, a `q8_0` i `q4_k` z L3. Dotyczy to także porównań
warunków L1 z L2 i L3.

## Wcześniejszy przebieg L1

Raport L1 nie omawia wcześniejszego przebiegu L1. Wykonano go na poprzednim
kandydacie (źródła `f13d9a6a…`) i zakończył się wynikiem 599 PASS, 1 FAIL i
41 GAP.

Jedyny FAIL dotyczył kontroli odmów C3. `basal benchmark` z niepoprawnym
pokryciem warstw zwracał właściwy błąd, ale przed odmową zapytanie o pamięć
GPU tworzyło konteksty i zajmowało pamięć obu kart. Poprawka w `benchmark.rs`
sprawdza pokrycie warstw przed pierwszym zapytaniem CUDA.

Raport L1 to pełne powtórzenie na poprawionym kandydacie (źródła
`77b629ed…`). Ta kontrola ma tam PASS. Brak alokacji nie został zmierzony i
pozostaje GAP-em. Dane wcześniejszego przebiegu dotyczą innej binarki i nie
są dowodem tego raportu.

## Zakres

Liczby L1, L2 i L3 dotyczą kandydata `dfd13b7e…` (źródła `77b629ed…`) i bazy
`d8a6a968…`, opisanych w zestawieniu. Nie są pomiarem późniejszych wersji
kodu.

| Plik | SHA-256 |
|---|---|
| `README.md` (zestawienie) | `339b01edad875f0f7ef7179d053ec2772de42417a5164bbee9653f1197cfb5d3` |
| `../multigpu-layer-l1-2026-10-13/README.md` | `3526a7b569ff45d1650ac838f4cc45c618a6724588b4810c43f7e1ede873b99d` |
| `../multigpu-layer-l2-2026-10-13/README.md` | `284364738f8775e30e452d12a4c15b117be3ed1f840d795bbb2e4a30d9d0012e` |
| `../multigpu-layer-l3-2026-10-13/README.md` | `e9e1b360e50e10de85cab4a7ea9eddefa35633b7fd8b5fe6520a37a6cf80447b` |
