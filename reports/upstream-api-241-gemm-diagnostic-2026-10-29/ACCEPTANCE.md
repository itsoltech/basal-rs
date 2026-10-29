# Akceptacja natywnego thinking — 2026-10-29

**Obecne thinking Rust/CUDA zostało zaakceptowane z udokumentowanymi różnicami
względem upstream.** Użytkownik podjął decyzję po omówieniu wyników J4 oraz
jakościowej ocenie konkretnych odpowiedzi: „no dobra czyli akceptujemy to”.

Akceptacja dotyczy stanu aplikacji opisanego w [raporcie J4](README.md):
kod `f9f915617482947950bb273807052c7b2c236e4c`, raport w commicie
`beac4f7b9ee41adc0d7c6a640de0802258270cea`. Dokładne odtworzenie tekstu
notatek i prawdopodobieństw upstream nie jest warunkiem zamknięcia tej próby.
Kończymy dopasowywanie natywnej arytmetyki w ramach J1–J4.

Pozostaje wyłącznie silnik Rust/CUDA. Domyślne `think_numerics: inherit`
zachowuje dotychczasowe zachowanie. Profil `upstream-bf16-pointwise` oraz
`BASAL_THINK_GEMM=heuristic` pozostają opcjonalną diagnostyką; ta decyzja
nie przełącza ustawień ani nie wdraża nowej binarki produkcyjnej.

## Znaczenie wyników

Status pomiaru zgodności J4 pozostaje **FAIL**: 13/19 odpowiedzi w progu 1e-4,
4/9 identycznych notatek i 7/9 zgodnych argmax wobec rzeczywistego upstream.
Status decyzji użytkownika to **zaakceptowane odstępstwo**. Historyczne raporty,
dane i archiwa pozostają niezmienione.

Przegląd treści nie wykazał ogólnej przewagi trafności upstream. Rozbieżny wybór
pilności dotyczy jednego pytania powtórzonego w grupie; poziomy nie mają
zdefiniowanej reguły biznesowej. Oba silniki mają też słabo uzasadnione
wykluczenie narzędzia do odzyskania dostępu. Jest to jakościowa ocena małego
korpusu, a nie wynik oznaczonego benchmarku trafności.

Akceptacja nie zmienia wyników pomiarów w dowód błędu vLLM, dokładnej matematyki
ani ogólnej jakości modelu. Pozostałe luki pomiarowe z raportu, w tym inne
platformy, pamięć i ścieżki błędów, zachowują swój status. Kryteria pozostałych
funkcji, w szczególności zwykłej inferencji bez thinking, pozostają bez zmian.

Ten zapis powstał po zakończeniu pomiarów i nie jest częścią ich wcześniejszego
archiwum danych. Nie wykonano nowych pomiarów ani zmian kodu aplikacji.
