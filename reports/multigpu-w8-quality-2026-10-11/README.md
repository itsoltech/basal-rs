# W8: dane do decyzji o jakości, pomiar częściowy

Q8_0 ma nowe różnice publicznych pól evidence względem FP32: mini zmienia
3 pola w 2 pytaniach, 4.5B zmienia 2 pola w 1 pytaniu. Kontrolne F16
z tej samej sesji zachowuje wszystkie granice evidence w tych pytaniach.
To rzeczywiste różnice numeryki; błąd referencji HTTP opisany niżej jest
odrębnym problemem runnera. Nie uznajemy etapu W8 za zaakceptowany.

## Zakres i pochodzenie

Pomiar z 2026-10-11, 2 × NVIDIA RTX 6000 Ada Generation 48 GB,
sterownik 580.126.09. Kandydat: SHA-256
`21bdb702b2cb4c4bb59908c772eca8ce6599cafb7851a0b731a02b3de3cb8bce`.
Oryginalny runner R4: SHA-256
`cd57da8dd1977522dfb8b4af4493aae43152e99e010c7972a0b1fe3c16da2a3a`.
Konfiguracja Q8_0 dotyczy wag projekcji, obliczenia CUDA F16/fused;
referencja FP32 upstream pochodzi z przypiętych raportów repozytorium.

Ten dokument obejmuje zamknięte pliki diagnostyki 900 pytań oraz
55 pytań endpointu na model z eksportów System One. Porównujemy
`answer_upstream` (dialekt zgodny z upstream) kandydata z `upstream_answer`
referencji FP32; to główne porównanie jakości, odrębne od referencji
pełnych natywnych odpowiedzi HTTP. W tych eksportach natywne items
16 przypadków wskazujących inny model są nieobecne (Unknown model);
to wcześniejsze zachowanie, które nie usuwa ich z 55 pytań porównania
upstream. Wszystkie 44 przykłady bench oraz natywne items pozostają
poza tym częściowym pakietem. Skróty plików zostały
porównane z zamkniętymi plikami z maszyny. Pełne pobranie kampanii,
jej końcowy manifest, wydajność, VRAM i poprawiona referencja HTTP są
jeszcze otwarte. Nie jest to końcowy raport ani akceptacja całego 44+34.
`facts.json` zawiera ścieżki względne i skróty źródeł, manifestu
zamkniętych plików, referencji 900, skal i kontekstu BF16 oraz przeliczone
liczby; pliki `quality-900-*`
zachowują wszystkie rozkłady 900 pytań; `endpoint-answers-*` wszystkie
porównywane odpowiedzi w tym wycinku.

## Publiczne granice evidence

| Model i pytanie | FP32 = F16 | Q8_0 | Fałszywe pola porównania |
|---|---|---|---|
| mini, m15-evidence-noul/form | [53,245], [245,328] | [53,328] | spans, first |
| mini, m15-evidence-choice-en/cause | [123,261], [1,68], [68,104] | [123,261], [1,68] | spans |
| 4.5B, m15-evidence-choice-en/cause | [86,243], [0,16] | [68,243], [0,16] | spans, first |

Porównanie obejmuje również tekst fragmentów, a nie tylko współrzędne.
Prawdopodobieństwa mogą się zmieniać i nie mają tutaj nowego progu.
Statyczna diagnoza nie znalazła błędu dekodera ani skalowania. Ten sam
rodzaj granic cause występuje w zapisanym porównaniu upstream BF16
z FP32; to kontekst wrażliwości, nie automatyczna akceptacja Q8.
Nie uchwycono odrzuconych kandydatów dekodera ani marginesów ich rankingu.

## Diagnostyka 900 pytań

| Model | Nowe zmiany względem FP32 | Naprawione różnice F16 | Wspólne różnice F16/Q8 | Zmiany Q8 względem F16 | Poprawne FP32 / F16 / Q8 |
|---|---:|---:|---:|---:|---|
| mini | 3 | 0 | 1 | 3 | 664 / 663 / 664 |
| 4.5B | 5 | 1 | 0 | 6 | 703 / 702 / 704 |

Każdy model ma 900 unikalnych ID. Kontrola F32 tej sesji daje 900/900
te same decyzje co zapisany upstream FP32; rozkłady są identyczne
z zapisaną referencją Basal F32. Względem upstream pozostają małe,
niezerowe różnice prawdopodobieństw. Zestaw obejmuje 400 choice,
200 score i 300 noul. Nie ustalono wymogu zerowej liczby zmian na 900. Pole
`known_close_ties_of_this_model_and_card` jest stałą informacyjną runnera,
nie wyliczoną klasyfikacją i nie usprawiedliwia żadnej zmiany. Kopie
plików źródłowych pozostają bajtowo identyczne.

Mini: wśród 3 nowych zmian 2 poprawiają odpowiedź względem etykiety,
1 pogarsza. Dwie nowe zmiany noul mają marginesy FP32 około 7,39 i
8,45 punktu procentowego. Nie można opisać wszystkich zmian jako remisy.
4.5B: wśród 5 nowych zmian 3 poprawiają odpowiedź, 2 pogarszają;
naprawienie wcześniejszej różnicy F16 daje dodatkową poprawną odpowiedź.
Wzrost liczby poprawnych odpowiedzi w tym zestawie nie dowodzi zachowania
jakości ogólnie i nie uchyla różnic publicznych pól w głównym porównaniu.

## Marginesy FP32 na tle zmian Q8 względem F16

Dla każdego pytania a i b są dwoma najwyższymi indeksami FP32.
Przesunięcie oznacza `(Q8[a] − Q8[b]) − (F16[a] − F16[b])`.
TV to połowa sumy bezwzględnych różnic wszystkich prawdopodobieństw.
Rangi TV liczymy malejąco w całych 900 pytaniach, rangi marginesów FP32
rosnąco. W tych ośmiu nowych zmianach zwycięzcą Q8 jest drugi indeks FP32.
Ujemna luka Q8 potwierdza zmianę kolejności tych dwóch odpowiedzi.
To opis wyniku numerycznego, bez progu akceptacji i bez dowodu przyczyny.

| Model / nowe ID | Margines FP32 | Luka F16 | Luka Q8 | Przesunięcie | Ranga TV / 900 | Ranga marginesu FP32 / 900 |
|---|---:|---:|---:|---:|---:|---:|
| mini / allegro-reviews-84 | 0.013511 | 0.011780 | -0.000296 | -0.012076 | 115 | 6 |
| mini / cbd-11 | 0.073859 | 0.069997 | -0.016379 | -0.086376 | 4 | 53 |
| mini / boolq-10 | 0.084521 | 0.080357 | -0.007045 | -0.087402 | 3 | 62 |
| 4.5B / polemo2-in-85 | 0.002058 | 0.003594 | -0.024741 | -0.028336 | 48 | 1 |
| 4.5B / ag-news-63 | 0.002851 | 0.006038 | -0.005577 | -0.011615 | 240 | 2 |
| 4.5B / emotion-27 | 0.013111 | 0.006974 | -0.081654 | -0.088627 | 1 | 12 |
| 4.5B / boolq-96 | 0.010339 | 0.010998 | -0.005789 | -0.016788 | 160 | 9 |
| 4.5B / sst5-61 | 0.009407 | 0.004729 | -0.029630 | -0.034359 | 12 | 8 |

Tło stanowią wszystkie pytania, w których F16 i Q8 zachowują decyzję
FP32: 896 mini i 894 4.5B. Ich liczby, rozkłady przesunięć i porównanie
z każdym nowym marginesem są zapisane w `facts.json`; główne liczenie
jakości nadal obejmuje wszystkie 900 pytań.

Dwie nowe zmiany mini noul znajdują się w ogonie zmierzonych zmian:
TV ma rangę 4 dla cbd-11 i 3 dla boolq-10; tylko 3 i 2 z 896 pytań tła
mają bezwzględne przesunięcie co najmniej równe ich marginesowi FP32.
W 4.5B wszystkie pięć nowych zmian mieści się w 12 najmniejszych
marginesach FP32; emotion-27 ma największe TV, a sst5-61 rangę TV 12.
Nie wykazano poprawki implementacyjnej; opcjonalne porównanie bajtów
dekwantyzacji GPU z hostem z punktu 6 planu nie zostało wykonane.
Nie ustalono źródła tych zmian w rozumieniu planu; pozostają w decyzji M2.

## Diagnoza i pozostałe rozstrzygnięcia

Nie wykazano defektu implementacji. Diagnoza skal sprawdziła pełne
macierze: brak zerowych bloków i skal zerowanych przez konwersję,
ale występują podnormalne skale F16: mini 542 z 46 792 704 bloków,
4.5B 3 495 796 z 144 506 880 (około 2,42%). Ich mniejsza precyzja
jest właściwością formatu skal F16 i motywacją dla opcji A6. Wkład
tego zjawiska w zaobserwowane różnice nie został zmierzony ani
wykluczony. To właściwość formatu, nie wykazany defekt. PTX wykonuje ich
konwersję do F32 przed mnożeniem, bez FTZ; nie wykazano gubienia tych
skal przez GPU. To analiza źródeł i binarki, nie osobny pomiar bajtów
zdekwantyzowanych wag na GPU.

Oryginalny runner porównywał 18 przepisanych na mini wejść HTTP z
nieprzepisanym eksportem. Naprawa R5 została zaakceptowana przez Opusa
i Astrę, lecz pełna zgodność 34/34 oraz dwa rzeczywiste wyniki CLI
wymagają nowego pomiaru. Nie zmienia ona silnika ani powyższych danych.

Plan etapu, punkt M2, oddaje użytkownikowi decyzję po pomiarze, jeśli
nowej różnicy jakości nie da się naprawić w obecnym zakresie: zawęzić
kwantyzację, rozważyć własny format i kernel ze skalami F32 lub jawnie
zmienić kryterium. Koordynator nie może zaakceptować tych różnic sam.
Potwierdzenie wydajności toru none nadal trwa; jego wynik może wymagać
osobnego rozstrzygnięcia. Brak pomiarów max, innych kart i innych modeli.
