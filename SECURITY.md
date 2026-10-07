# Security policy / Bezpieczeństwo

## Reporting a vulnerability / Zgłaszanie luk

Do not report vulnerabilities in public issues. Use a private report:
[Security Advisories](https://github.com/itsoltech/basal-rs/security/advisories/new), or e-mail dev@itsol.tech.
Include the version (`basal --version`), how basal is run (Homebrew, install.sh, Docker, source) and the steps to
reproduce.

Nie zgłaszaj luk w publicznych issue. Zgłoszenie prywatne:
[Security Advisories](https://github.com/itsoltech/basal-rs/security/advisories/new) albo e-mail dev@itsol.tech.
Podaj wersję (`basal --version`), sposób uruchomienia (Homebrew, install.sh, Docker, źródła) i kroki odtworzenia.

## Supported versions / Wspierane wersje

Fixes are released in the newest release only (0.1.x). / Poprawki trafiają tylko do najnowszego wydania (0.1.x).

## Scope / Zakres

The HTTP server (`basal serve`) has no authentication; by default it listens on 127.0.0.1. Exposing it to a network
(`0.0.0.0`, the container image) is meant for a trusted network or behind a proxy that authenticates requests. Also
in scope: `install.sh`, `basal update` and `basal setup` (downloads and their checksums) and the container image.

Serwer HTTP (`basal serve`) nie ma uwierzytelniania; domyślnie słucha na 127.0.0.1. Wystawienie go do sieci
(`0.0.0.0`, obraz kontenera) jest przeznaczone dla zaufanej sieci albo za proxy uwierzytelniającym żądania. W zakresie
są też `install.sh`, `basal update` i `basal setup` (pobierane pliki i ich sumy kontrolne) oraz obraz kontenera.
