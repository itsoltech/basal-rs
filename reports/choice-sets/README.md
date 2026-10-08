# Choice 11–255 na zbiorach z etykietami

Data: 2026-10-08. Ocena strategii grupowej dla pytań choice z więcej niż 10
opcjami na danych, w których każda odpowiedź ma etykietę, oraz porównanie z
API TypeSafe na tych samych pytaniach. Poprzednia ocena
([large-choice-1.5](../large-choice-1.5/README.md)) używała dystraktorów z
innych pytań basal-bench; część z nich była poprawną odpowiedzią, więc
zawyżała stratę przy dużej liczbie opcji.

## Zestaw

`set.jsonl` ([tools/choice-sets/build.py](../../tools/choice-sets/build.py),
ziarno 0): po 200 losowych pytań ze zbiorów testowych.

| Zbiór | Opcje | Język | Źródło |
|---|---:|---|---|
| banking77 | 77 | angielski | `legacy-datasets/banking77`, CC BY 4.0 |
| clinc150 | 150 | angielski | `clinc/clinc_oos` (plus), bez przykładów i etykiety spoza zakresu, CC BY 3.0 |
| massive-pl | 59 | polski | `mteb/amazon_massive_intent` (pl) z AmazonScience/massive, CC BY 4.0; 59 z 60 intencji występuje w zbiorze testowym |

Stan to wypowiedź, pytanie jedno na zbiór („What is the customer asking
about?”, „What does the user want?”, „Czego chce użytkownik?”), opcje to
nazwy intencji bez podkreśleń, bez opisów.

## Pomiar

- basal-rs: `basal eval-choice-set` (CUDA, RTX 6000 Ada, limit mocy 200 W,
  f16, tabele GEMM), basal-1.5-mini, 4.5B i max w rewizjach z `basal init`,
  każda strategia na wszystkich 600 pytaniach. Dane w `cuda/MODEL/STRATEGIA.jsonl`
  (bez pełnych rozkładów: prawdopodobieństwo etykiety i zwycięzcy, finaliści,
  prompty, czas).
- TypeSafe: `jev-latest` (odpowiada `jev-1.13.0`),
  [tools/choice-sets/typesafe.py](../../tools/choice-sets/typesafe.py), wynik
  `typesafe-jev.jsonl`.
- Tabele: [tools/choice-sets/report.py](../../tools/choice-sets/report.py).

Strategie (`basal_core::large_choice::Strategy`): `luce2` to strategia 0.1.x
(dwa podziały na grupy po ≤10, model Luce, finał z 10 najmocniejszych opcji);
`luce1-final` ma jeden podział i finał decyduje; `knockoutK` przepuszcza K
najlepszych opcji z każdej grupy do kolejnej rundy; przyrostek `-1o` to jedna
kolejność opcji w rundach przed finałem (finał zawsze w obu kolejnościach).

## Trafność (na 200 pytań)

| | banking77 | clinc150 | massive-pl | prompty na pytanie | NLL etykiety |
|---|---:|---:|---:|---|---|
| TypeSafe jev-1.13.0 | 159 | 185 | 159 | | 1,28 / 0,27 / 1,32 |
| mini `luce2` | 146 | 169 | 146 | 34 / 62 / 26 | 1,01 / 0,76 / 1,04 |
| mini `luce2-1o` | 146 | 169 | 142 | 18 / 32 / 14 | 1,04 / 0,74 / 1,05 |
| mini `luce1-final` | 150 | 165 | 146 | 18 / 32 / 14 | 1,11 / 0,93 / 1,08 |
| mini `luce1-final-1o` | 148 | 168 | 143 | 10 / 17 / 8 | 1,14 / 0,87 / 1,12 |
| mini `knockout1` | 147 | 164 | 139 | 22 / 36 / 18 | 1,09 / 0,90 / 1,08 |
| mini `knockout2` | 147 | 163 | 139 | 22 / 42 / 18 | 1,09 / 0,80 / 1,08 |
| mini `knockout1-1o` | 147 | 166 | 142 | 12 / 19 / 10 | 1,06 / 0,76 / 1,10 |
| 4.5B `luce2` | 147 | 174 | 153 | 34 / 62 / 26 | 0,85 / 0,50 / 0,96 |
| 4.5B `luce2-1o` | 146 | 174 | 153 | 18 / 32 / 14 | 0,81 / 0,49 / 1,00 |
| 4.5B `luce1-final` | 150 | 175 | 146 | 18 / 32 / 14 | 0,87 / 0,65 / 1,05 |
| 4.5B `luce1-final-1o` | 150 | 172 | 148 | 10 / 17 / 8 | 0,87 / 0,61 / 1,15 |
| 4.5B `knockout1` | 151 | 173 | 147 | 22 / 36 / 18 | 0,87 / 0,70 / 1,04 |
| 4.5B `knockout2` | 151 | 175 | 147 | 22 / 42 / 18 | 0,87 / 0,62 / 1,04 |
| 4.5B `knockout1-1o` | 154 | 169 | 144 | 12 / 19 / 10 | 0,87 / 0,91 / 1,16 |
| max `luce2` | 156 | 182 | 157 | 34 / 62 / 26 | 0,81 / 0,44 / 0,79 |
| max `luce2-1o` | 156 | 182 | 157 | 18 / 32 / 14 | 0,83 / 0,44 / 0,81 |
| max `luce1-final` | 156 | 182 | 151 | 18 / 32 / 14 | 0,81 / 0,58 / 0,89 |
| max `luce1-final-1o` | 155 | 183 | 156 | 10 / 17 / 8 | 0,92 / 0,59 / 0,84 |
| max `knockout1` | 154 | 175 | 155 | 22 / 36 / 18 | 0,88 / 0,86 / 0,86 |
| max `knockout2` | 154 | 179 | 155 | 22 / 42 / 18 | 0,88 / 0,66 / 0,86 |
| max `knockout1-1o` | 156 | 170 | 152 | 12 / 19 / 10 | 0,85 / 1,09 / 0,86 |

| | etykieta w finale | zgodność z jev |
|---|---|---|
| mini `luce2` | 194 / 196 / 191 | 168 / 176 / 160 |
| mini `luce2-1o` | 194 / 194 / 191 | 167 / 176 / 154 |
| mini `luce1-final-1o` | 182 / 191 / 175 | 173 / 175 / 152 |
| 4.5B `luce2` | 196 / 197 / 190 | 168 / 181 / 158 |
| 4.5B `luce2-1o` | 198 / 197 / 186 | 170 / 179 / 158 |
| 4.5B `luce1-final-1o` | 186 / 197 / 174 | 173 / 181 / 156 |
| max `luce2` | 196 / 198 / 195 | 178 / 183 / 170 |
| max `luce2-1o` | 196 / 198 / 191 | 183 / 183 / 167 |
| max `luce1-final-1o` | 183 / 196 / 183 | 180 / 183 / 168 |

Wnioski:

- `luce2-1o` ma trafność `luce2` (różnice 0–1 pytania na 200, poza mini na
  massive-pl: 142 wobec 146) przy połowie promptów. Jest strategią domyślną
  od tej wersji.
- Strategie z jednym podziałem (`luce1-final`, `knockout`) częściej gubią
  etykietę przed finałem (massive-pl: 174–183 wobec 186–195 z 200); przy dwóch
  podziałach każda opcja jest oceniana dwa razy.
- Etykieta dochodzi do finału w 93–99% pytań; większość błędów to wybór w
  finale między podobnymi intencjami. Z ok. 105 błędów max (`luce2`) w 79
  myli się też jev, w 54 z tą samą odpowiedzią: w tych zbiorach część
  etykiet jest niejednoznaczna.
- basal-1.5-max (`luce2-1o`) jest 2–3 pytania na 200 poniżej jev, 4.5B 6–13,
  mini 13–17.
  NLL etykiety basal-rs jest niższa niż jev na banking77 i massive-pl (jev
  częściej daje wysokie prawdopodobieństwo błędnej odpowiedzi), wyższa na
  clinc150.

## Czas

Czasy pytań z tabel zbiorczych (`ms_per_question` w `summary.json`)
porównywać tylko w obrębie jednego przebiegu: ta sama strategia na 4.5B
(`luce1-final-1o`, 200 pytań banking77) trwała 246 ms w pierwszym przebiegu i
138 ms w późniejszym, przy tym samym limicie mocy. Liczba promptów na
pytanie jest miarą kosztu niezależną od warunków. Pomiar czasu wszystkich
kandydatów w jednym przebiegu (250 W) przerwał twardy reset serwera po 3
minutach.

## Czekanie na GPU (CUDA)

`BASAL_CUDA_SYNC` (auto, spin, yield, blocking) ustawia sposób, w jaki wątek
czeka na GPU. Domyślnie (`auto`) sterownik czeka aktywnie, co zajmuje cały
rdzeń CPU w czasie pracy GPU. 4.5B, `luce1-final-1o`, 200 pytań banking77, ten
sam proces i warunki (`cuda-sync/*.log`; czas procesu z wczytaniem modelu):

| Tryb | czas pytania p50 | CPU user + sys procesu |
|---|---:|---:|
| auto | 138 ms | 60,6 + 10,5 s |
| yield | 138 ms | 49,3 + 21,5 s |
| blocking | 142 ms | 42,5 + 9,8 s |

Odpowiedzi są bitowo równe we wszystkich trybach. `blocking` oszczędza ok.
0,7 rdzenia w czasie pracy GPU kosztem 2,5% czasu pytania; domyślny tryb
zostaje bez zmian.

## Etykiety, które nie weszły do finału

Pytanie: czy strategia (grupy przed finałem) traci poprawne odpowiedzi.
`cuda-semifinal/` (2026-10-08, ten sam zestaw, 200 W) zapisuje dla każdego
pytania pozycję etykiety we wspólnym rozkładzie po pierwszej rundzie
(`gold_rank_first_round`). Z `luce2-1o` do finału nie weszło 21 (mini), 19
(4.5B) i 15 (max) etykiet na 600; ich pozycje po pierwszej rundzie to od 12.
do 58. (połowa poniżej 30.), a jev daje im zwykle prawdopodobieństwo bliskie
zera (np. „nowy jork” → `transport query`, „system cisco” →
`recommendation movies`).

Sprawdzenie (`label-in-final/`, Metal): finałowa dziesiątka każdego z tych
pytań z jedną opcją zastąpioną etykietą, zadana jako zwykłe pytanie z 10
opcjami. Etykieta wygrywa w 0/21 (mini), 1/19 (4.5B: „nowy jork” →
`transport query`, p = 0,19) i 0/15 (max). Wejście etykiety do finału
zmieniłoby więc najwyżej jedną odpowiedź na 600; pozostałe odrzuca sam model.

Półfinał (`-sN`: N najmocniejszych opcji po pierwszej rundzie w dodatkowej
rundzie przed finałem, 2–4 prompty więcej) wprowadza do finału 1–3 etykiety
więcej, a trafność zmienia nieregularnie (mini 457 → 457–464, 4.5B 473 →
468–470, max 495 → 493–498 na 600), przez inny skład finału. Domyślna
strategia zostaje bez półfinału.
