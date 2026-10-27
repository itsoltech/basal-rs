# Uzupełnienie przeglądu Opus 5.5

Wznowiony po limicie API przegląd Claude Opus 5.5 zakończył się bez uwag
blokujących. Był tylko odczytem kodu: nie zmieniono plików aplikacji,
nie uruchamiano kompilacji ani inferencji.

## Zakres i wynik

Porównano zamrożone `qwen35.rs` i `fused.rs` sprzed refaktoru z aktualnym
backendem oraz jego prywatnymi modułami. Przegląd objął ładowanie i typy
wag, wybór ścieżek attention i DeltaNet, kolejność obliczeń, pokrycie
tabelą GEMM, prefiksy, walidację wejść, checkpointy i czas życia buforów.
Nie znaleziono regresji zachowania, numeryki ani obsługi błędów w kodzie.
`plan.rs` i `kernels.cu` pozostają identyczne z bazą.

## Korekta opisu `is_cuda()`

Wcześniejsze określenie głównego agenta „przywrócono warunek `is_cuda()`”
było błędne. Pierwsza krótka ścieżka attention w zamrożonej bazie nie
miała tego warunku; miał go następujący po niej wariant kafelkowy v3/v4.
W refaktorze warunek został **dodany**, a nie przywrócony.

Nie zmienia to działania obsługiwanych konstruktorów: przy feature `cuda`
funkcja `gpu_device` tworzy urządzenie CUDA, więc dodany warunek jest
prawdziwy. Bez tego feature cała krótka ścieżka CUDA jest wyłączona
podczas kompilacji. Nie wykryto innego pominiętego warunku urządzenia.
Wersja z tym jawnym sprawdzeniem była już objęta opublikowaną kwalifikacją.

## Weryfikacja i ograniczenia

Przegląd Opus nie zastępuje pomiarów. Główny agent wcześniej potwierdził
bitową zgodność 13 konfiguracji, kompilację, lint i ABBA opisane w
[raporcie](README.md). Wznowienie przeglądu nie zmieniło kodu, dlatego nie
powtarzano tych sprawdzeń. Metal i inne GPU nadal nie mają potwierdzenia
runtime'u w tej kampanii.

Pierwotnego archiwum HF nie zmieniono. Status przerwanego przeglądu
w jego `validation.json` opisuje chwilę publikacji; niniejsza notatka
uzupełnia go o otrzymany później wynik. Opis „restored explicit CUDA device
guard” w tym pliku należy czytać z powyższą korektą. Wyniki liczbowe,
sumy binarek i integralność archiwum pozostają aktualne.
