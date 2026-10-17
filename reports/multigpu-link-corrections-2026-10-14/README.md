# Publiczne źródła zamiast historycznych odsyłaczy do `.itsol`

Katalog `.itsol/` zawiera lokalny stan pracy i nie jest częścią
repozytorium. Istniejące raporty pozostają bez zmian, także ich dane,
sumy kontrolne i rozstrzygnięcia. Poniższa tabela wskazuje publiczne
źródła dla odsyłaczy, które w tych raportach prowadziły do `.itsol/`.
Nie jest to nowy pomiar ani nowa decyzja o jakości lub wydajności.

| Historyczny odsyłacz | Publiczne źródło |
|---|---|
| `phases/03-w8a16/quality-decision.md` | [W8A16: jakość i przyjęty zakres opcjonalnego Q8](../w8a16-2026-10-11/README.md#poprawność-none-i-jakość) |
| `phases/04-w4a16/quality-decision.md` | [W4A16 S3: opcjonalne eksperymentalne Q4, domyślnie none](../w4a16-s3-2026-10-12/README.md#wynik-w-skrócie) |
| `phases/04-w4a16/performance-decision.md` | [W4A16 S3: decyzja użytkownika o wydajności](../w4a16-s3-2026-10-12/README.md#decyzja-użytkownika) |
| `progress.md` | [PR #7: wynik realizacji i kontrole CI](https://github.com/itsoltech/basal-rs/pull/7) |

Dotyczy odsyłaczy w raportach:

- [W8A16](../w8a16-2026-10-11/README.md);
- [pierwotne dane W8](../multigpu-w8-original-2026-10-11/README.md);
- [historyczne L3](../multigpu-layer-l3-2026-10-13/README.md);
- [bieżące L3](../multigpu-layer-l3-revalidation-2026-10-14/README.md)
  i jego [rozstrzygnięcie](../multigpu-layer-l3-revalidation-2026-10-14/DECISION.md).

Pomiary zachowują etykiety commitów z chwili ich wykonania. Oczyszczenie
historii gałęzi z `.itsol/` zmienia identyfikatory commitów, ale nie zmienia
źródeł silnika ani plików pomiarowych. Tożsamość zmierzonych źródeł i
binarek nadal określają zapisane w raportach sumy SHA-256.
