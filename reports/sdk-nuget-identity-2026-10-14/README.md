# NuGet: ITSOL.Basal.Client

Data: 2026-10-14. Identyfikator NuGet, nazwa projektu i assembly zostały zmienione
na ITSOL.Basal.Client. Zaktualizowano ProjectReference, project lock, ścieżki CI,
helper wersji, selekcję artefaktu i dokumentację, w tym nazwę paczki dla polityki
Trusted Publishing. Namespace pozostaje ITSOL.Basal. Pierwsza wersja 0.1.0 nie
była publikowana; historyczne raporty pozostają bez zmian.

Przeszły restore w locked mode, build przykładu bez ostrzeżeń i błędów, pack,
walidacja tagu sdk-csharp-v0.1.0, actionlint, git diff --check i weryfikacja hashów.
Nupkg zawiera właściwy PackageId, assembly i aktualny README oraz zero
zewnętrznych zależności. Świeży konsument z PackageReference do nowej nazwy
zainstalował lokalną paczkę i uruchomił typowane przygotowanie pytań.
Nie wykonywano inferencji ani publikacji; politykę NuGet konfiguruje użytkownik.
Usunięto przestarzałe lokalne archiwum wygenerowane pod wcześniejszą nazwą.
