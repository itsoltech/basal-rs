# Dodatkowy przegląd zgodności YAML

Przegląd GPT 6 Astra XHigh: brak uwag blokujących lub wymaganych poprawek.
Rodzic sprawdził źródło diagnostyki, wyniki i walidację źródła modelu.

Izolowana diagnostyka 52 wariantów deserializacji wyodrębnionych struktur dała
42 zgodne wyniki i 10 różnic. Zgodność oznacza identyczne wartości albo
odrzucenie przez oba parsery; nie oznacza identycznego tekstu błędu.
Program nie uruchamiał serwera, modelu ani pełnej walidacji konfiguracji.
Nie dodano programu ani bramki testowej do repozytorium.

Serde-saphyr dodatkowo akceptuje yes/no/on/off i cytowane "true" jako bool,
liczby z separatorem 1_024, merge << oraz models: [null]. Ten ostatni zapis
nie jest prawidłową konfiguracją serwera: model_dir nadal odrzuca wpis bez
path i repo komunikatem "model entry: needs path or repo".

Dwa wcześniej akceptowane zapisy mają inną semantykę:

- release_date: null wcześniej dawało tekst "null", teraz jest odrzucane;
  tekstową wartość null należy ująć w cudzysłowy.
- !!binary eA== w polu tekstowym wcześniej dawało "eA==", teraz daje "x".

Nie deklarujemy pełnej równoważności parserów. Dokumentowany przykład,
wartości domyślne, wszystkie pola, zwykłe aliasy i Unicode zachowują wartości
w sprawdzonym zakresie. Nie wprowadzano dodatkowej emulacji starego parsera.
YAML pozostaje zależnością wyłącznie CLI; nie trafia do grafu Rust SDK.

Dowody: [wejścia](extended-config-inputs.json),
[wyniki](extended-config-comparison.jsonl). Wyniki rozszerzają ograniczony
zestaw opisany w [pierwszym sprawdzeniu](README.md).
