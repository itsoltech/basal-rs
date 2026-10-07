# CUDA: wynik bez zmian

Data: 2026-10-07, RTX 6000 Ada, limit mocy 250 W na czas testu (przy starcie
obciążenia przy 300 W serwer zrestartował się, jak 2026-10-04). Binarka z
`main` (`7c9da42`) i wersja z nowym loaderem, obie z `--features cuda`, te
same tabele GEMM. `run.sh` w kontenerze CUDA (`/work/main`, `/work/new`:
źródła; `/base`: modele, tabele GEMM, referencje).

Eksporty pojedynczo i w drzewie, porównanie `main` z nową wersją
(`cmp-*.json`): dla basal-1.5-mini, 1.5-4.5B i 1.5-max wszystkie logity
bitowo równe (różnica 0,0), 44/44. Eksport `main` 4.5B pojedynczo przerwany
restartem serwera został powtórzony.

Wczytywanie (`*.log`): max 24,8–30,0 s z odczytem przez mmap (`main`),
21,0–21,2 s z nowym loaderem; 4.5B 11,2–13,3 i 10,4–10,5 s; mini bez
różnicy (3,6–4,3 s).
