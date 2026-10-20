# Niezależna weryfikacja pakietu — 2026-10-20

Rodzic sprawdził SHA-256 archiwum i rozpakował je do nowego, osobnego katalogu.
Wszystkie **354/354** sumy są zgodne; manifest obejmuje każdy zwykły plik poza
własnym głównym `SHA256SUMS`. Zweryfikowano również 269/269 dokładnych źródeł.

Z rozpakowanych danych ponownie wykonano analizę przekazania bramki i pamięci.
JSON przekazań oraz JSON/TXT pamięci są bitowo identyczne z wynikami w pakiecie.
Niezależnie potwierdzono osiem sekwencji oddania/wznowienia, zgodność ośmiu
wyników z wcześniejszą referencją i 17 przyrostów puli poniżej estymat kodu.
Kontrola danych poza dokładnymi źródłami nie znalazła identyfikatorów własnego
hosta ani jego fizycznych ścieżek. Ograniczenia raportu pozostają w mocy.

Archiwum lokalne:
`.cache/measurement-artifacts/batches-gpu-followup-packet-20261020/measurement-data-qualified.tar.gz`.
Rozmiar: 1 744 180 B. SHA-256:

```text
b8dbff8d8caa2e882aee151d742f90086d30f9c0f593fb4a5018ecdb1e0f4a61
```

Dane mają logiczny prefiks `reports/batches-mini-followup-2026-10-20/`.
Sposób odtworzenia analizy opisują [raport](README.md) i `METHOD.md` w archiwum.

Nie publikowano w HF ani nie sprawdzano pobrania zdalnego. Docelowy URI przy
ewentualnej późniejszej publikacji, wymagającej osobnej weryfikacji, to:
`hf://buckets/itsoltech/basal-rs/measurements/batches-gpu-followup-20261020/b8dbff8d8caa2e882aee151d742f90086d30f9c0f593fb4a5018ecdb1e0f4a61/measurement-data-qualified.tar.gz`.
Ten URI nie potwierdza dostępności. Pierwotne raporty i archiwa zachowano.
