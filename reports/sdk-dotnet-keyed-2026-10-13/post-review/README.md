# .NET SDK: zakończenie review Opusa

Data: 2026-10-13. SDK 0.1.0, przed publikacją. Uzupełnienie
[raportu odpowiedzi po kluczach](../README.md). Wcześniejsze raporty i hashe
pozostają bez zmian. Zakres dotyczy istniejącego API z `q.Add`; proponowane
uproszczenie do fabryk `q.Choice` nie zostało wdrożone w tej iteracji.

Opus 5.5 XHigh zakończył review bez ustaleń blokujących. Zgłosił trzy uwagi
niskiej wagi oraz dwie drobne uwagi o diagnostyce/dokumentacji. Rodzic sprawdził
je względem kodu i gotowej paczki oraz wprowadził poniższe poprawki.

## Ustalenia i rozstrzygnięcia

- **Asynchroniczne fabryki:** wcześniej `async q => ...` mogło wywnioskować
  TAnswers jako Task i wysłać część pytań przed zakończeniem fabryki. Dodano
  odrzucane przez kompilator przeciążenia dla Task/ValueTask, również odmian
  generycznych. Metody te nigdy nie wykonują I/O. Dokumentacja wymaga zakończenia
  przygotowania asynchronicznego przed wywołaniem synchronicznej fabryki.
- **Własne typy pytań:** `QuestionSpec<TAnswer>.Parse` jest teraz sealed override
  i deleguje do typowanego Read. Podklasa nie może już nadpisać Parse tak, żeby
  zwrócić inny rodzaj wyniku i spowodować InvalidCastException po publikacji.
- **Pola z wcześniejszego żądania:** zachowano warunek tworzenia świeżych pól
  w każdej fabryce. README i główna dokumentacja opisują konkretnie, aby nie
  cache'ować pól w static ani przez `??=`. Można współdzielić definicje pytań;
  pole odpowiedzi pozostaje związane ze swoim pierwotnym żądaniem. To nadal
  udokumentowany warunek użycia, bez introspekcji dowolnego obiektu użytkownika.
- **Pusta fabryka:** lokalny ArgumentException wskazuje teraz parametr
  `createAnswers`, a nie `questions` z wewnętrznego przeciążenia.
- **Opcje:** doprecyzowano, że DecisionOptions i CancellationToken dotyczą
  zarówno słownika, jak i fabryki obiektu.

## Weryfikacja rodzica

Po diagnostyce recenzenta odtworzono assets przez `dotnet restore` w trybie
locked z repozytoryjnym nuget.config. Bibliotekę net8.0, przykład i NuGet
zbudowano ponownie; build przykładu: zero ostrzeżeń i błędów. Gotową paczkę
zainstalowano do nowego, oddzielnego cache konsumenta.

Konsument z językiem C# 12 na runtime .NET 10 sprawdza poprawne typowanie
anonimowego obiektu i krotki, własny generyczny typ pytania oraz diagnostykę
pustej fabryki przed HTTP. Wszystkie te przypadki przechodzą.

[Sześć negatywnych przypadków kompilacji](compiler-rejections.json):
Task<T>, ValueTask<T>, Task i ValueTask są odrzucane przez CS0619. Nietypowana
lambda async jest odrzucana przez CS0121, ponieważ pasuje do zabronionych
przeciążeń Task i ValueTask; tę postać diagnostyki opisano w README.
Nadpisanie sealed Parse jest odrzucane przez CS0239. Nie wystąpiły dodatkowe
nieoczekiwane błędy w tych kontrolach.

Ponownie uruchomiono lokalną diagnostykę z zainstalowaną paczką: anonimowy
obiekt, własna klasa, słownik, pięć rodzajów odpowiedzi, różne wyniki pod ID
tej samej definicji, osiem współbieżnych wywołań, jednokrotna fabryka,
zamknięcie buildera, dziesięć przypadków brakujących pól, brak częściowo
dostępnych wyników i anulowanie. Wynik: PASS.

Collector/verifier potwierdziły hash paczki. Wersje pozostają zgodne z 0.1.0;
git diff --check przeszedł. Metadane NuGet nadal mają zero zależności i nie
zawierają tools/build/buildTransitive. README w paczce odpowiada źródłu.

## Granice

Nie dodano zależności, reflection, generatora ani trwałych testów i bramek CI.
Diagnostyki były tymczasowe, poza repozytorium. Nie zmieniano transportu ani
obliczeń/serializacji odpowiedzi; nie powtarzano pomiaru live, opisanego w
raporcie nadrzędnym. Nie uruchamiano oddzielnego runtime .NET 8.

Opus nie wykonał ponownego review poprawek. Ich akceptacja opiera się na
weryfikacji rodzica. Nie pozostały otwarte materialne ustalenia review;
warunek świeżych pól w fabryce pozostaje jawnie udokumentowany.
Oba zamówione przeglądy tego zakresu — Astra i Opus — są zakończone.

[Aktualny SHA-256 NuGet](distribution-SHA256SUMS), ścieżka względem sdk/.
Nie publikowano paczek.
