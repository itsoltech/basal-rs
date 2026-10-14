# Decyzje względem upstream FP32 — 2026-10-14

Ponowna analiza zachowanych wyników H100 potwierdza zgodność wyborów wszystkich
pięciu modeli na sprawdzonych wejściach. Celem jest upstream FP32. Nie wymagamy
bitowej równości confidence, probability ani oczekiwanej wartości score.
W tej analizie nie zmieniano arytmetyki runtime’u ani domyślnych precyzji.
Nie uruchamiano nowej sesji GPU.

Decyzje (`decisions.json`) porównują rzeczywiste `answer_upstream` serwera, nie tylko
argmax pojedynczych gałęzi: choice, tool/tools, call i odroczenie/refused narzędzi,
selected multi oraz kolejność zwróconego rankingu tools.
Dla noul diagnostycznie porównano stronę progu P(true)>=0.5, a dla score najbardziej
prawdopodobny poziom; sam score jest wyłączony z kryterium akceptacji.

| Model | Precyzja runtime’u | Pytania z odpowiedziami / zgodne decyzje, orders=1 |
|---|---|---|
| basal-2.0 mini / 4.5B / max | FP32 | po 41/41 |
| basal-2.0 mini / 4.5B / max | FP16 z wag BF16 | po 41/41 |
| basal-omni mini / max | FP32 | po 51/51 |
| basal-omni mini / max | BF16 | po 51/51 |

Mianowniki 41/51 liczą pytania z odpowiedziami w eksportach żądań (Llama/omni);
nie odnoszą się do osobnego zestawu 49 pytań benchmarku. Dwa odrzucone rekordy
na konfigurację są poza mianownikiem.
To 450 porównań odpowiedzi w dwóch precyzjach, bez rozbieżności wyborów
i pomocniczych klasyfikacji. Dla noul klasyfikacją jest strona progu 0.5,
a dla score najbardziej prawdopodobny poziom; nie są to porównania wartości liczbowych.
Obejmują choice, noul, score, multi i tool; dla omni także pytania z mediami. Niezależnie
20/20 rankingów tools zachowało kolejność kluczy. Błędne wejścia mają odrębne rekordy
refused_cases; ich treść komunikatu nie jest kryterium decyzji.
Te 450 odpowiedzi nie obejmuje `act`. Osobne [17/17 porównań act i certyfikatów](../upstream-2.0-boundaries-2026-10-14/README.md)
sprawdza reguły hosta dla tych samych zadanych prawdopodobieństw, bez forwardu modeli.

Dodatkowe pięć odpowiedzi omni-mini z BF16 auto porównano z zachowanym upstream
FP32 auto: 5/5 zgodnych wyborów. Obejmuje to omawiany przypadek media-audio:

- native BF16: image 0.38610, audio 0.39697, mixed 0.21693 — wybór audio;
- upstream FP32: image 0.38202, audio 0.40105, mixed 0.21693 — wybór audio;
- native FP32: image 0.38190, audio 0.40117, mixed 0.21694 — wybór audio.

Poprzednio opisana różnica dotyczyła porównania z upstream BF16, który na tym wejściu
wybrał image. Nie jest to odmienna decyzja względem wybranego teraz celu FP32.

Arytmetyka hosta (`host-math.json`): 541 ponownych odczytów rzeczywistego basal-core z tymi
samymi logitami co referencja FP32 (także PL/full-window i dwie kolejności). Wybory
541/541 zgodne; max różnica kalibrowanej probability 1.02e-6. To izoluje softmax/
uśrednianie/kalibrację od forwardu GPU. Nie jest to ponowny forward modelu.

Alternatywy sumowania (`summation-candidates.json`): 4770 odtworzeń odczytu dla
zapisanych logitów reference/native FP32/native half, z sumą liniową, parami i FP64.
Nie dały spójnej poprawy końcowej odległości do upstream FP32 we wszystkich modelach.
Żaden wariant nie zmienił argmax w tych przypadkach. Pozostały jednorazowym programem diagnostycznym poza
repozytorium; nie zmieniono produkcyjnego softmax ani arytmetyki starej ścieżki.

Przegląd źródeł potwierdził, że Candle nie włącza domyślnie TF32 dla F32 GEMM,
a projekt nie włącza tej flagi. Zidentyfikowane miejsca różnych kolejności obliczeń
obejmują redukcje/LayerNorm, RoPE liczone przez CPU oraz GPU, kształty GEMM, kafelkowanie
uwagi i podział prefill. To kandydaci do przyszłego porównania warstwa po warstwie,
nie dowód przyczyny każdej pozostałej różnicy ani dowód, że nie da się jej zmniejszyć.

Kryterium akceptacji to zgodny wybór względem upstream FP32; confidence i score
nie muszą być równe. Obecne wybory spełniają je na objętych pomiarem przypadkach.
Arytmetyka runtime’u pozostaje bez zmian. Nie gwarantujemy
identycznej decyzji dla każdego niezmierzonego wejścia lub marginesu bliskiego progowi.
Runtime FP32 jest najbliższy zmierzonej referencji numerycznie, lecz sama zgodność
wyborów na tym korpusie nie wymaga zmiany dotychczasowych ustawień auto.

Źródła i surowe dane są związane z niezmienionym fingerprintem (`reports/upstream-2.0-h100-2026-10-14/fingerprint.json`).
[Raport źródłowej sesji H100](../upstream-2.0-h100-2026-10-14/README.md) zachowuje
wyniki tej sesji. Nowa analiza nie dowodzi jakości na innym korpusie, skuteczności
kalibracji ani przyspieszenia; nie dodano frameworków testowych lub bramek CI.

## Dane poza Gitem

- Bucket: [itsoltech/basal-rs](https://huggingface.co/buckets/itsoltech/basal-rs).
- URI archiwum: `hf://buckets/itsoltech/basal-rs/measurements/upstream-2.0-2026-10-14/c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536/measurement-data.tar.gz`.
- SHA-256 archiwum: `c762b90c590e5db3d308df714faef9545e727eb7fc97af2edb403456db725536`.
- Dane tego raportu: `reports/upstream-2.0-decisions-2026-10-14/` wewnątrz archiwum; zachowano też oryginalny README sprzed migracji.
- Pobranie i odtworzenie: [MEASUREMENT_ARTIFACTS.md](../../docs/MEASUREMENT_ARTIFACTS.md). Obie weryfikacje pobrania, w tym anonimowa, potwierdziły wszystkie 34 sumy plików.
