# Eksperymentalne zadania Basal 2.1

Warstwa zadań odtwarza `basal/tasks.py` z upstream v2.1.0
(`62542d31e8eb69000839c01e8f9b45da1fb80e49`). Działa przez `/v1/basal`,
`/v1/systemone` i natywne polecenia CLI. Korzysta z tych samych decyzji modelu;
nie generuje nowego tekstu. Interfejsy upstream są oznaczone `experimental: true`.

| Typ | Wejście pytania | Odpowiedź |
|---|---|---|
| `classify` | `labels`: mapa rubryk, lista nazw lub obiektów `name`/`rubric`; `multi_label`, `none`, `max_error` | `scores` i `label`/`top_label`/`confidence`/`automated` albo `labels`/`set_confidence` |
| `verify` | `claims`: mapa albo lista; `binary`, `max_error` | `claims` z werdyktami, prawdopodobieństwami i `automated`; `all_supported`, `not_supported` |
| `compress` | `unit`: sentence/line/paragraph; instrukcja, `keep_ratio` albo `threshold`, `min_keep` | `compressed`, `kept`, `units`, liczby znaków i `ratio` |
| `extract` | `schema`: obiekt z zagnieżdżonymi polami | `data` oraz `fields` z pochodzeniem, prawdopodobieństwami i wskazaniem fragmentu źródła |

Puste lub pominięte instrukcje korzystają z domyślnej treści PL/EN. `lang` może
być jawne. Klasyfikacja multi przyjmuje `threshold`, `min`, `max`, `objective`.
`none: true` dodaje kategorię „żadna z powyższych” tylko w trybie pojedynczym.
Weryfikacja domyślnie rozróżnia supported/contradicted/not_stated; `binary: true`
zwraca supported/not_supported.

```json
{
  "state": "Sprzedawca: ACME\nKwota: 123,45 PLN\nTermin: 20 października.",
  "lang": "pl",
  "questions": {
    "kategoria": {"type": "classify", "labels": {"faktura": "Dokument sprzedaży", "inne": "Inny tekst"}},
    "kontrola": {"type": "verify", "claims": {"kwota": "Kwota wynosi 123,45 PLN."}},
    "skrot": {"type": "compress", "instructions": "Jaka jest kwota?", "unit": "line"},
    "dane": {"type": "extract", "schema": {"type": "object", "properties": {"kwota": {"type": "number"}}}}
  }
}
```

## Interpretacja wyników

`automated` ma klucze żądanych błędów, domyślnie `0.01` i `0.05`. Wartość
`null` oznacza brak certyfikowanego progu dla tego zadania. Nie zastępuj jej
`true` na podstawie zwykłej kalibracji `choice`/`noul`. basal-rs używa osobnego
`task_thresholds` z aktywnego pliku kalibracji. Przy mediach zwraca `null`,
ponieważ progi tekstowe nie potwierdzają jakości decyzji multimodalnych.

Kompresja wybiera całe jednostki w kolejności źródłowej. `keep_ratio` jest
budżetem znaków, który ostatnia jednostka może przekroczyć; `min_keep` może
zwiększyć zachowany tekst. `compressed` jest składane z oryginalnych fragmentów.

Ekstrakcja obsługuje enum/const (`oneOf`/`anyOf`), boolean, string, number,
integer, zagnieżdżone obiekty i tablice wartości enum. Wartości brakujące są
`null`; nieudana konwersja liczby zachowuje fragment i pole `error`.
Dla tekstu model najpierw wybiera linię, potem kandydat fragmentu. Wszystkie
rundy wykonuje ta sama replika przy zachowaniu slotu żądania.
`fields.*.confidence` dla fragmentu jest iloczynem prawdopodobieństw linii
i fragmentu, z oznaczeniem `calibration: "per decision"`. Nie jest certyfikatem
poprawności całego pola. Enum i boolean mogą korzystać z mediów; wybór fragmentu
wymaga tekstowego `state`.

Offsety `start`/`end` oraz liczby znaków liczymy w **punktach kodowych Unicode**:
w Pythonie są to indeksy `str`; w JavaScript użyj `Array.from(text)`, w .NET
`text.EnumerateRunes()`, w Rust `text.chars()`. Nie są to bajty UTF-8 ani indeksy
UTF-16. Koniec zakresu jest wyłączny.

## Ograniczenia i różnice walidacji

Obowiązuje budżet 4096 gałęzi i 32 MiB rozwiniętych promptów dla obu porządków,
wliczając ostrożne oszacowanie kontynuacji. Długość promptu sprawdza też model.
Kompresja ma limit 600 jednostek; schemat ekstrakcji 512 pól i 16 poziomów,
kandydaci są ograniczeni do 254 plus opcji braku. Nazwy zadań nie mogą zawierać
`::`; kolizje z nazwami ich wewnętrznych pytań są odrzucane. Nazwy właściwości
schematu są niepuste i nie zawierają kropki; `__missing__` jest zarezerwowane.
Te ograniczenia zapobiegają niejednoznacznemu składaniu odpowiedzi i nadmiernej
alokacji, zamiast cichego nadpisania danych.

`usage.questions` liczy oryginalne pytania. `input_tokens` i `branches` obejmują
obie rundy; basal-rs sumuje też diagnostyczne `second_order_branches` i zachowuje
maksymalne `longest_prompt_tokens`. Są to rozróżnienia względem prostszego
zliczania upstream. Eksport CLI pokazuje surowe readouty pierwszej rundy
oraz gotową odpowiedź z obu rund; `task_export_scope` wskazuje ten zakres.

## Modele

`omni-1.1-max` wskazuje `Remek/basal-omni-1.1-max@c8ab50cd973c116b3ca6bd9e4d574f713bab1f68`.
Ma ten sam decoder i projekcje obrazu/audio Unified co dotychczasowy max.
Domyślny model i alias `omni-max` pozostają bez zmiany. Certyfikat klasyfikacji
w opublikowanej kalibracji jednego porządku obejmuje tylko cel błędu 5%.

`omni-ultra` wskazuje `Remek/basal-omni-1.0-ultra@b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4`.
Checkpoint BF16 ma dwa shardy i około 62,55 GB. Encoder obrazu ma 27 warstw,
1152 kanały oraz standaryzację w FP32 przed końcowym rzutowaniem dtype.
Model obsługuje tekst i obraz; obecność konfiguracji procesora audio nie
ustanawia obsługi audio bez wag audio. Gemma korzysta z placement/replik,
nie z podziału warstw ani kwantyzacji Llama. 48 GiB VRAM nie pomieści pełnego Ultra.
Wariant ModelOpt FP8 wymaga odrębnej implementacji i jest jawnie odrzucany.

Opis implementacji nie stanowi potwierdzenia poprawności ani przewagi GPU.
Wyniki porównań i ograniczenia kampanii są zapisywane osobno w `reports/`.
