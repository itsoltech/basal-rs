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

By default the HTTP server (`basal serve`) has no authentication and listens on 127.0.0.1. Optional API keys stored
in the configured database (`auth.enabled`, docs/AUTH.md) protect every route except health checks. Without them,
exposing the server to a network (`0.0.0.0`, the container image) is meant for a trusted network or behind a proxy
that authenticates requests; with them, use TLS in front of the server so tokens are not sent in plain text. The
optional browser UI (`ui.enabled`, docs/UI.md) logs in with passwords and session cookies of those users. Also
in scope: `install.sh`, `basal update` and `basal setup` (downloads and their checksums) and the container image.

Domyślnie serwer HTTP (`basal serve`) nie ma uwierzytelniania i słucha na 127.0.0.1. Opcjonalne klucze API
przechowywane w skonfigurowanej bazie (`auth.enabled`, docs/AUTH.md) chronią wszystkie trasy poza health checkami.
Bez nich wystawienie serwera do sieci (`0.0.0.0`, obraz kontenera) jest przeznaczone dla zaufanej sieci albo za proxy
uwierzytelniającym żądania; z nimi użyj TLS przed serwerem, aby tokeny nie były przesyłane jawnym tekstem.
Opcjonalny interfejs w przeglądarce (`ui.enabled`, docs/UI.md) loguje tych użytkowników hasłem i ciasteczkiem sesji. W zakresie
są też `install.sh`, `basal update` i `basal setup` (pobierane pliki i ich sumy kontrolne) oraz obraz kontenera.
