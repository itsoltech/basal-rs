# Attention po jednostkach drzewa na Metal, Apple M2 Max

Data: 2026-10-07. Apple M2 Max (GPU 30 rdzeni, 32 GB pamięci wspólnej),
macOS 27.0.1, zasilanie sieciowe. Modele `Remek/basal-1.5-max` (rewizja
`be1b5ee7`), `Remek/basal-1.5-4.5B` (`784a683b`), `Remek/basal-1.5-mini`
(`1978d070`), f16. Skrypt: `run.sh`; wczytywanie: `load/`.

## Zmiana

Na Metal attention liczyło SDPA z MLX po całym wierszu-drzewie z maską
addytywną. Kolejność sumowania po kluczach zależała więc od tego, gdzie w
wierszu stoi pytanie i co jeszcze dzieli z nim wiersz: to samo pytanie
liczone pojedynczo i w drzewie z innymi dawało inne logity
([metal-m1-pro-1.5](../metal-m1-pro-1.5/README.md)).

`attn_tree_f32` w `kernels.metal` liczy attention po tych samych jednostkach
co kernel CUDA: jednostka to blok wiersza (zapytania jednego węzła drzewa), a
jej klucze to prefiks z cache, bloki przodków od korzenia i własny blok
(przyczynowo), kafelkowane po 16 według kolejności w tej sekwencji. Wiersz
zapytania liczy się tylko ze swojego Q i tych kafelków, więc wynik nie zależy
od pakowania. Iloczyny i sumy w f32 na macierzach `simdgroup_float8x8`;
softmax i przeskalowanie akumulatora na elementach fragmentów w rejestrach.
`BASAL_ATT=sdpa` przywraca poprzednią ścieżkę.

## Zgodność i zależność od pakowania

Eksporty: pytanie w osobnym forwardzie (`single`), pytania w drzewach
wspólnych prefiksów (`tree`), kilka wierszy w jednym forwardzie (`budget`).
Wobec FP32 upstream: 44 przykłady basal-bench i `/v1/basal` wobec
`Server.decide` (pytania z tymi samymi polami).

| Model | Attention | Pojedynczo a drzewo: maks. różnica logitu / p | Pojedynczo a partia | Wobec FP32: decyzje | Maks. różnica p (kal.) | `/v1/basal` |
|---|---|---:|---:|---:|---:|---:|
| max | SDPA | 0,174 / 0,0024 | 0,0 | 44/44 | 0,0039 | 33/33 |
| max | drzewo | 0,0 / 0,0 | 0,0 | 44/44 | 0,0009 | 33/33 |
| 4.5B | SDPA | 0,034 / 0,0081 | 0,0 | 44/44 | 0,0045 | 55/55 |
| 4.5B | drzewo | 0,0 / 0,0 | 0,0 | 44/44 | 0,0021 | 55/55 |
| mini | SDPA | 0,021 / 0,0053 | 0,0 | 44/44 | 0,0073 | 55/55 |
| mini | drzewo | 0,0 / 0,0 | 0,0 | 44/44 | 0,0040 | 55/55 |

Ścieżka f32 z nowym kernelem: 44/44, maks. różnica logitu 0,00021 (4.5B)
i 0,00012 (mini). Wyniki SDPA na M2 Max są bitowo równe wynikom z M1 Pro
z poprzedniego raportu. Maks. różnica logitu nowego kernela na max wobec
FP32 wynosi 0,306 (SDPA 0,207) przy mniejszej różnicy prawdopodobieństw:
przesunięcie logitów wszystkich liter o podobną wartość.

## Pojedyncza decyzja

Metodyka basal-bench, SDPA (A) i nowy kernel (B) w kolejności ABBA, 3 rundy
(`ab-bench-*`). Iloraz B/A mediany lat2 po przykładach.

| Model | lat2 mediana A / B | Iloraz B/A (mediana, p25–p75) | Przykłady szybsze w B | lat1 A / B | Decyzje/s A / B |
|---|---:|---:|---:|---:|---:|
| max | 526–529 / 526 ms | 0,999 (0,997–1,001) | 27/39 | 362–371 / 366–367 ms | 2,20 / 2,33 |
| 4.5B | 235,1–235,4 / 234,8–235,0 ms | 0,999 (0,997–1,001) | 27/39 | 165,0–165,4 / 166,7–167,3 ms | 4,93 / 5,26 |
| mini | 79,6–79,7 / 79,1–79,8 ms | 1,002 (1,000–1,004) | 9/39 | 59,9–60,1 / 61,5–62,0 ms | 14,5 / 15,8 |

Decyzja z obu porządków opcji (tak liczy serwer) trwa tyle samo, jeden
porządek (lat1) do 3% dłużej na mini i 1% na 4.5B, przepustowość w grupach
po 16 pytań jest o 6–9% wyższa. Wcześniejsze wersje kernela (SIMT oraz
macierze simdgroup z mniejszą liczbą wątków lub przeskalowaniem przez
mnożenie macierzy) miały na M1 Pro lat2 dłuższe o 2,8–4,2% na mini i nie
weszły.

## Wczytywanie basal-1.5-max

Loader czytał wagi z mapowanego pliku (mmap). Przy 22 GB checkpointu i
21,6 GB wag f16 na GPU w 32 GB pamięci wolna pamięć spadała do zera po
~18 s i system przez minutę kompresował i zrzucał strony. Teraz dane są
czytane zwykłym odczytem, konwersja bf16 → f16 idzie na wszystkich rdzeniach
CPU bez pośredniej kopii (ta sama arytmetyka). Eksporty przed zmianą i po
niej są bitowo równe (max, mini).

| Odczyt | Czas wczytania max | Źródło |
|---|---:|---|
| mmap (poprzedni loader, konwersja równoległa) | 83–123 s | `ab-bench-basal-1.5-max/*.json` (`load_s`) |
| odczyt z pominięciem cache stron (`F_NOCACHE`) | 59–65 s | `load/` |
| odczyt przez cache stron (wersja końcowa) | 37–39 s | `load/` |

basal-1.5-4.5B wczytuje się w 1,5 s z cache stron i w 11–14 s z dysku,
mini w 0,5–4 s.

## CUDA

Loader jest wspólny z CUDA. Wynik na RTX 6000 Ada: sekcja
[cuda-unchanged](cuda-unchanged/README.md).
