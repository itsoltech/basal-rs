# Intel Arc Meteor Lake — refaktor backendu, 2026-10-11

Przegląd i refaktor backendu Intel Vulkan z
[poprzedniego raportu](../intel-arc-meteor-lake-2026-10-10/README.md), na tym samym
sprzęcie i sterowniku (Arc Meteor Lake `8086:7d55`, Mesa 26.2.2, wgpu 30.0.1, Rust 1.95.0).
Bazowy commit: `e9444bb0a64a3fc90414a0c0bb95c44208d56f6b` z niezacommitowanym backendem.

## Zmiany

- Checkpoint nie jest mapowany do pamięci: wagi warstw są czytane pozycyjnie
  (`pread`) porcjami po 8 MiB, a embedding i `lm_head` tylko w potrzebnych wierszach.
  Usuwa to `unsafe` mapowania bez kopiowania całych macierzy do RAM.
- Telemetria `VK_EXT_memory_budget` sprawdza obsługę Vulkan 1.1 przed wywołaniem funkcji core;
  trzy pozostałe bloki `unsafe` mają opisane warunki bezpieczeństwa.
- Końcowy przegląd poprawił sprawdzanie rozmiaru checkpointu: suma rozmiaru nagłówka
  i danych z uszkodzonego pliku mogła przepełnić `u64`. Walidacja używa teraz różnicy
  już sprawdzonych rozmiarów; taki plik zwraca błąd także przy włączonej kontroli przepełnień.
- Modele bez biasów współdzielą jeden zerowy bufor zamiast czterech na warstwę;
  forward bez cache prefiksu nie alokuje zastępczego bufora.
- `basal doctor` wypisuje nazwę urządzenia i sterownika bez cudzysłowów JSON.

Arytmetyka, kernele, cache i synchronizacja warstw pozostały bez zmian.

## Zgodność

Porównanie bajtów rekordów eksportu z wynikami sprzed refaktoru, na tych samych wejściach:

| Model | Wejścia | Bitowo identyczne |
|---|---|---:|
| mini | 44 bench Tree + 34 przypadki API | 78/78 |
| mini, po odbudowie binarki | 44 bench Tree | 44/44 |
| 4.5B | 3 bench | 3/3 |
| max | 3 bench + 4 przypadki API, w tym Tree `m15-mixed` | 7/7 |

Porównania z referencjami FP32 upstream są identyczne z raportami sprzed refaktoru
(mini 44/44 decyzji, max 3/3 oraz cztery przypadki API).
Po końcowej korekcie walidacji rozmiaru sprawdzono odrzucenie uszkodzonego nagłówka
oraz akceptację nagłówków trzech rzeczywistych modeli, bez ponownego eksportu.
Ta korekta nie zmienia danych wag ani arytmetyki forwardu.

## Wydajność

- Czas ładowania 4.5B, naprzemiennie przy ciepłym page cache: przed 11,74 i 9,91 s,
  po 10,63 i 10,28 s. Ta mała próbka nie uzasadnia deklaracji przyspieszenia; pierwszy przebieg po zmianie
  (23,5 s, chłodniejszy cache) nie jest porównywalny.
- Pełny eksport mini zużył 6,5 s CPU użytkownika i 3,3 s systemowego na 361 s czasu
  ściennego. Nie zmierzono osobno kosztu tworzenia parametrów dispatchu;
  tej optymalizacji nie wprowadzono.
- Odrzucony GEMM: kafle FP32 w pamięci lokalnej i wektorowe odczyty, przy tej samej
  kolejności FMA. Profil mini, 3 zadania, 3 pary naprzemienne: obecny kernel
  1644, 1513, 1543 ms/zadanie; wariant 1900, 1879, 1910 ms/zadanie.

## Zakres i ograniczenia

Przeszły: formatowanie, Clippy i Rustdoc dla Intel oraz Linux bez GPU, cargo-deny,
kontrola wyjątku `paste`, actionlint/ShellCheck, `git diff --check`, `basal doctor`.
Podczas eksportu max minimalne MemAvailable wyniosło 3,25 GiB, a jądro zapisało
ostrzeżenie o nieudanej alokacji strony w innym procesie (podobne występowały
w dzienniku także podczas kampanii z 2026-10-10); bez zdarzeń GPU HANG, resetu ani
wygasłych fence'ów. Zapasu RAM tego eksportu nie mierzono przed refaktorem.
Nie powtarzano pełnego eksportu 4.5B i max, pomiarów HTTP ani benchmarku max. CUDA, Metal i zdalne CI nie były tutaj uruchamiane.
Surowe dane i porównania są zachowane lokalnie w tym katalogu, poza Gitem.
