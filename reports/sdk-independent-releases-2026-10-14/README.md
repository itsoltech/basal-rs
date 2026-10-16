# Niezależne wydania SDK

Data: 2026-10-14. Przygotowanie CI/CD bez publikacji. Każde SDK ma własną wersję
w manifeście i osobny changelog. Usunięto sdk/VERSION. Wersja opisu kontraktu
oraz wersja silnika pozostają niezależne. Historyczne raporty nie są zmieniane.

Tag sdk-<python|typescript|csharp|rust>-vX.Y.Z wybiera jeden builder i jeden
publisher. Ręczny przebieg wybiera SDK i domyślnie tylko buduje. Publikacja
wymaga zgodnego tagu, SDK, manifestu i właściwego locka. Skrypty pakowania
wybierają dokładne nazwy archiwów z wersji danej paczki. Proces obsługuje
stabilne X.Y.Z; prerelease są odrzucane.

Zachowano izolację OIDC, uprawnienia tylko do odczytu podczas budowania,
przypięte akcje/obraz PyPA, artefakty po niezmiennych ID i SHA-256 oraz
Cargo repack/porównanie bez wykonywania kodu zależności w jobie publikacji.
Audyt OSV działa oddzielnie dla Python (8 pakietów) i npm (1 pakiet).
Zwykły CI PR nadal sprawdza wszystkie SDK, akceptując różne wersje.

## Dowody

- W izolowanej kopii metadanych zaakceptowano Python 2.3.4, npm 0.1.1,
  C# 4.0.2 i Rust 0.6.0 z wersją kontraktu 9.8.7.
- Potwierdzono wybór własnych archiwów i wyjść joba dla każdego tagu,
  odrzucenie ośmiu błędnych tagów/argumentów i rozbieżnego npm locka.
  Błędny niewybrany manifest nie blokuje wybranego wydania.
- Wykonano rzeczywisty shell metadata lokalnie dla czterech tagów i czterech
  ręcznych build-only. Cztery błędne tryby publikacji zakończyły się przed
  zapisaniem wyjść. Nie wykonywano kroków publikacji.
- Actionlint z ShellCheck przeszedł dla wszystkich workflow; helpery działają
  na Python 3.11. Przeszły sprawdzenie manifestów i git diff --check.
- Ponownie zbudowano pięć dystrybucji: sdist, wheel, npm, NuGet i crate.
  Crate pakowano z --no-verify; zmiana nie dotyczy źródeł Rust SDK.
  Sprawdzono kolekcję archiwów i hashe wszystkich artefaktów.

[Wersje i artefakty](versions-and-artifacts.json), [routing workflow](workflow-routing.json),
[podsumowanie](verification.json), [hashe plików i dystrybucji](SHA256SUMS)
względem katalogu repozytorium. Nie dodano trwałych testów ani frameworków.

## Review i ograniczenia

GPT-6 Astra XHigh zakończył review bez materialnych Blocker/Should. Opus 5.5
XHigh nie znalazł problemów blokujących ani związanych z bezpieczeństwem.
Jego uwagi operacyjne rozstrzygnięto w dokumentacji: każdy tag wysyła się
osobno i sprawdza rozpoczęcie runu, przed tagowaniem wykonuje się build-only
tego samego commitu, a lokalne ścieżki archiwów odczytują własną wersję SDK.
Limit wysyłania więcej niż trzech tagów potwierdzono w
[dokumentacji GitHub](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#push).
Poprawki dokumentacji zweryfikował rodzic; druga runda review nie była potrzebna.

Nie uruchamiano workflow w GitHub ani publikacji. Konta, środowiska,
reguły ochrony tagów i trusted publishers wymagają konfiguracji poza repo.
Nie potwierdzono wymiany tokenów OIDC ani retry prawdziwego joba. Ponowienie
samego nieudanego publishera w istniejącym runie zachowuje wybór artefaktu;
nowy workflow_dispatch buduje nowy artefakt. Częściowy upload wheel/sdist do
PyPI wymaga obsługi operatora i weryfikacji hashów przed uzupełnieniem plików;
workflow nie pomija automatycznie istniejących plików. Bramka całego workspace
ma wcześniej opisany advisory unsafe-libyaml w silniku, poza grafem SDK.
