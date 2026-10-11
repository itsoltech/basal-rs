# Intel Arc Meteor Lake: szacunek pamięci benchmarku

Zmiana względem `aaafdfd`: benchmark uwzględnia nieaktywny cache plików
w cgroup i rozmiar wag przechowywanych przez backend Intel. Bez zmian
w kernelach, ładowaniu tensorów ani numeryce; bez nowego `unsafe` i zależności.
Pomiar na zintegrowanym Arc Meteor Lake, Mesa 26.2.2, wgpu 30.0.1, FP16.
Modele uruchamiano pojedynczo. Systemowe `MemTotal` wynosiło 30,70 GiB,
a limit cgroup sesji wynosił 26,70 GiB.

## Co liczy kontrola

- RAM: minimum `MemAvailable` i zapasu pod każdym limitem cgroup, z ulgą
  tylko za nieaktywny cache plików po odjęciu stron brudnych i zapisywanych.
  Shmem, pamięć wymagająca swapu i aktywny cache nie zwiększają szacowanego zapasu.
  Cgroup v1 używa liczników `total_*`; brak liczników oznacza brak ulgi.
- Wagi Intel: rozmiary tensorów ze zwalidowanego nagłówka safetensors,
  bez pełnych embeddingów i `lm_head`, z wyrównaniem buforów i zerowym biasem.
  Cache prefiksów, scratch i narzut sterownika wymagają dodatkowej pamięci.
- Zintegrowany Intel: przed ładowaniem RAM musi pomieścić wagi i większą
  z rezerw RAM/GPU. Bieżący budżet GPU musi pomieścić wagi oraz, osobno,
  rezerwę. Po ładowaniu i podczas inferencji pełna rezerwa GPU jest sprawdzana
  względem rzeczywistego użycia. Metal, CUDA i osobny Intel zachowują sumę
  wag i rezerwy GPU w kontroli wstępnej.
- Domyślne rezerwy pozostają 2 GiB RAM i 1 GiB GPU. Limit 64 MiB przyrostu
  globalnego swapu/zapisów do swapu również pozostaje bez zmian.

Podstawa interpretacji liczników: [cgroup memory.stat](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files).
[Budżet Vulkan](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryBudgetPropertiesEXT.html)
jest zmienną wskazówką sterownika. Dla max odczyt przed ładowaniem wyniósł
20,73 GiB, po ładowaniu 23,03 GiB; same alokacje procesu zajmowały 20,58 GiB.

W jednym odczycie przed ładowaniem stare `limit − użycie` dawało 16,73 GiB.
Nowy szacunek, ograniczony `MemAvailable`, dawał 23,29 GiB. Liczniki cgroup
obejmowały 9,18 GiB szacowanego odzyskiwalnego cache. To oszacowanie, nie
gwarancja natychmiastowej alokacji bez swapu.

## Wagi i rzeczywiste alokacje

| Model | Cały checkpoint, GiB | Szacunek wag GPU, GiB | Alokacje GPU po ładowaniu i prefiksach, GiB |
|---|---:|---:|---:|
| mini | 2,9738 | 2,7906 | 2,9294 |
| 4.5B | 8,8612 | 8,6169 | 8,8489 |
| max | 20,8016 | 20,3133 | 20,5762 |

Różnica między wagami a alokacjami pokazuje koszt dodatkowych buforów i
sterownika. Pomiar po ładowaniu nie jest szczytem dowolnego forwardu.

## Przebiegi HTTP i odmowy

Krótki request choice, jedno pytanie i dwie opcje. Jedna rozgrzewka przed
każdą fazą, sześć żądań sekwencyjnych i sześć przy dwóch klientach.
Kompilacja nie odbywała się podczas pomiarów HTTP. To sprawdzenie działania
kontroli zasobów i krótka próbka, bez tezy o przyspieszeniu przez tę zmianę.

| Końcowy kod, domyślne rezerwy | Zmierzone żądania | Błędy / zmiany answers | Min. RAM, GiB | Min. GPU, GiB | Przyrost swapu, MiB |
|---|---:|---:|---:|---:|---:|
| mini | 12 | 0 / 0 | 21,52 | 19,37 | 0 |
| 4.5B | 12 | 0 / 0 | 15,68 | 14,11 | 0 |
| max, próba 1 | 0 | kontrola po ładowaniu | 3,61 | 2,45 | 323,12 |
| max, próba 2 | 0 | kontrola po ładowaniu | 3,94 | 2,45 | 4517,38 |

Przyrost swapu to większa z dodatnich zmian zajętości swapu i licznika bajtów
zapisanych do swapu; nie oznacza jednoczesnego zajęcia podanej ilości pamięci.

Stary kod odrzucił max przed ładowaniem z powodu szacunku RAM.
Końcowy kod załadował max i prefiksy przy domyślnych rezerwach. Obie próby
zatrzymał dopiero rzeczywisty przyrost globalnego swapu. Nie są to udane
benchmarki inferencji; nie wyłączano ani nie zerowano kontroli swapu.
Nie przypisujemy globalnego swapu wyłącznie procesowi Basal.

Wcześniejsza próba diagnostyczna, po poprawieniu cache i rozmiaru wag, ale
przed zmianą kontroli dynamicznego budżetu iGPU, używała jawnego
`--gpu-reserve-gib 0.1` i domyślnej rezerwy RAM 2 GiB. Jedno ładowanie
zatrzymał swap (2952,05 MiB). Powtórny przebieg zakończył 12/12 żądań,
bez błędów i zmian answers: minimum RAM 3,01 GiB, GPU 2,45 GiB,
przyrost swapu 1,23 MiB. Mediana HTTP: 6,495 s przy jednym kliencie,
12,553 s przy dwóch; przepustowość odpowiednio 0,1533 i 0,1573 żądania/s.
Ten przebieg nie dowodzi powtarzalnego startu z domyślnymi ustawieniami.

Jawne rezerwy 100 GiB RAM i 100 GiB GPU zostały osobno odrzucone przed
ładowaniem, z liczbowym wyjaśnieniem. Dla max z rezerwami RAM 0,1 GiB / GPU
10 GiB kontrola wymagała 30,31 GiB wspólnego RAM i również odmówiła przed
ładowaniem: używa większej rezerwy. Rezerwy nie zostały wyłączone.

## Zgodność i kontrole

Mini: 44 zadania bench Tree i 34 przypadki API, **78/78 bitowo identycznych**
z eksportem sprzed zmiany. 4.5B: **44/44 bitowo identyczne** zadania bench
Single. Dla obu ponowne porównanie bench z FP32 daje 44/44 zgodnych decyzji
avg/cal, bez dodatkowego błędu liczbowego. API porównujemy bitowo z poprzednim
eksportem; nie deklarujemy tu nowej zgodności całego API z FP32, którego
referencje zawierają opisane wcześniej różnice w przetwarzaniu wejścia.

Max: trzy zadania bench Single i cztery przypadki API (w tym `facts`,
`evidence` i wiele pytań), **7/7 bitowo identycznych** z zachowanym eksportem
`87da786`, sprzed optymalizacji GEMM. Te trzy zadania bench dały też 3/3
zgodnych decyzji avg/cal z FP32. Nie powtarzano pełnych 44 zadań max ani API
4.5B. Eksport jest osobną weryfikacją numeryki, nie benchmarkiem z kontrolą
rezerw; nie zastępuje nieudanych przebiegów HTTP max z domyślnymi rezerwami.

Przeszły: build release Intel, formatowanie, Clippy i Rustdoc dla Intel oraz
Linux bez GPU, cargo-deny, kontrola wyjątku `paste`, przegląd duplikatów
zależności, actionlint/ShellCheck i `git diff --check`. Przegląd Rust obejmował
ograniczenie odzysku do limitu/użycia cgroup, brakujące liczniki, arytmetykę
z kontrolą przepełnienia, wyrównanie tensorów i zachowanie rezerw po ładowaniu.

Pełne logi i eksporty pozostają poza Gitem. Nie zmieniano limitów cgroup,
ustawień swapu ani cache systemu. Desktop był aktywny; pomiary nie obejmują
długiego obciążenia, maksymalnego kontekstu, osobnego GPU, runtime Metal/CUDA
ani cgroup v1. Próbki pamięci nie gwarantują uchwycenia chwilowego szczytu.
