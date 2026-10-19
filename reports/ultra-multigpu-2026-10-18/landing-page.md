# Ultra multi-GPU — materiał dla landing page

Źródło: [szczegółowy raport i metoda](README.md), pomiar 2026-10-18.
To uzupełnienie wcześniejszych porównań z upstream. Nowe QPS poniżej porównuje
**jedną i dwie karty basal-rs**, bez nowego benchmarku upstream.

## Tekst do wykorzystania

> Ultra na dwóch kartach 48 GB. Podział warstw obsługuje tekst i obrazy,
> z lokalnymi wagami i cache na każdej karcie.

Przypis: Ultra, FP16, 2 × RTX 6000 Ada 48 GB, podział 30/30; 234 odpowiedzi
C1/C2/C8, badane dokumenty do 8192 tokenów oraz cztery zdjęcia i para obrazów
high detail. Maksymalnie cztery jednoczesne żądania obrazów. To potwierdzenie
pojemności badanego zakresu, nie pełnego kontekstu ani maksymalnego QPS.

> Dwie repliki, 1,98× więcej odpowiedzi na sekundę.

Przypis: Ultra, 2 × H100 PCIe vs 1 × H100 PCIe, 128 klientów, krótkie dokumenty,
sześć zadań na żądanie, FIFO, bez prefix/state cache. ABBA, dwa starty na wariant.
**4,30 vs 2,16 odpowiedzi/s**. Przy C32: **4,12 vs 2,11**, 1,95×.
Dla ruchu z długimi dokumentami przy C128: **2,70 vs 1,41**, 1,91×.

## Karty metryk i obowiązkowe warunki

| Metryka | Wartość | Warunek |
|---|---|---|
| skalowanie replik | 1,98× | C128 krótkie zadania; dwie H100 vs jedna |
| QPS replik | 4,30 vs 2,16 | poprawne HTTP/s, sześć zadań w żądaniu |
| RAM API, dwie repliki | 1,38 GiB PSS | C128 krótkie zadania; jedna replika 1,02 GiB |
| CPU, dwie repliki | 1,98 rdzenia | jedna replika 1,00; około 0,462 CPU s/odpowiedź w obu |
| VRAM, dwie repliki | 126,07 GiB łącznie | średnia fazowa; jedna około 63,05 GiB; pełne wagi na każdej karcie |
| VRAM podziału Ada | maks. 36,38 GiB/kartę | maksimum próbkowane dla tekstu i obrazów; suma do 71,22 GiB |
| RAM podziału Ada | maks. 1,68 GiB PSS | praca API; nie szczyt ładowania ani RAM hosta |

Dwa tryby mają inne zastosowanie. **Repliki** zwiększają obsługiwany ruch i
powielają wagi. **Podział warstw** rozkłada jeden model między karty, aby go
zmieścić. Nie nazywać podziału tensor parallelism ani obiecywać dwukrotnego
przyspieszenia pojedynczego pytania.

## Jakość — zachować ograniczenia

Nowa kontrola Ada: **160/161 zgodnych argmaxów głowic oraz 64/65 odpowiedzi
kategorycznych względem FP32**. Jedna etykieta verify jest różna; upstream BF16
zmienia tę samą etykietę. Średnia TV względem FP32 wynosi 0,000871 native i
0,001914 BF16: natywna odległość jest **2,20× mniejsza**. To porównanie
rozkładów w małej próbie, nie deklaracja accuracy ani identycznych decyzji FP32.
**Ścisły status jakości pozostaje FAIL.** Nie używać hasła „2,20× lepsza jakość”.

W kontroli H100 wszystkie 156 porównań nowej binarki i podziałów ze starą
jednokartową są bitowo zgodne. W replikach pod C128 wystąpiło 226 różnic
liczbowych, bez zmian decyzji: status pełnej niezmienności bitowej FAIL.
Nie publikować „100% identyczności”, „zero różnic numerycznych”, „2× szybciej
od upstream”, „2× krótsza inferencja C1” ani ekstrapolacji na 3+ GPU.
Nowe runtime pomiary obejmują Ultra; nie całą rodzinę Omni, audio ani pełny kontekst.

## Dane do weryfikacji

Archiwum: `hf://buckets/itsoltech/basal-rs/measurements/ultra-multigpu-2026-10-18/2411d4f8eb1761fdb0a7caf54b57831f8ce566b75651c6ea465afd4a76d5e982/measurement-data.tar.gz`.

SHA-256 `2411d4f8eb1761fdb0a7caf54b57831f8ce566b75651c6ea465afd4a76d5e982`.

Potwierdzone pobranie z autoryzacją i anonimowo, 950/950 sum plików. Pełna
metoda, źródła, nieudane próby, różnice i polecenia odtworzenia są w raporcie
oraz archiwum. Liczby zaokrąglono wyłącznie na potrzeby prezentacji.
