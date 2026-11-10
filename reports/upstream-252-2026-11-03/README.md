# Nieznany typ pytania: zgodność z upstream 2.5.2

## Wynik i zakres

Ręczne wywołanie `/v1/basal` z `type: "nuol"` zmieniło wynik z HTTP 200
na HTTP 422 z body:

```json
{"error":"question \"q\": unknown type \"nuol\"","code":"INVALID_REQUEST"}
```

Brak pola `type` nadal oznacza `choice`. Odrzucane są także jawne `null` i typ
liczbowy. Niepoprawny JSON oraz body inne niż obiekt mają ten sam kod błędu.
Natywny `/v1/systemone`, z wymaganym polem `model`, nadal zwraca 400
„Unknown question type”; body przed i po jest bajtowo identyczne.

To realizacja [upstream #84](https://github.com/rkinas/basal/pull/84),
z [wydania 2.5.2](https://github.com/rkinas/basal/releases/tag/v2.5.2),
oraz niewielkie uzupełnienie metadanych z #83. Nie oznacza pełnej zgodności
wszystkich funkcji z upstream 2.5.2. Nie dodano testów, frameworków ani bramek CI.

## Wersje, wejścia i metoda

Baza: basal-rs `c7ed6974c6fa311a22d7ee168d3412f04536c8b9`.
Końcowe zmiany kodu i dokumentacji zawiera archiwalny `source.patch`, SHA-256
`9f1263e7323fe6ee2eca0cac010e3bfc4c4407c4562bcff0f063343ba9a9dd75`.
Model: `Remek/basal-2.0-mini`, rewizja
`cb81b32ce89b0b87e62fcdad28f0435e25c953f3`.
Sprzęt: NVIDIA GeForce RTX 4060. Rust 1.95.0, CUDA/cuBLASLt 12.9.1.

Na binarce bazowej i końcowej wykonano te same wywołania curl. Jeden request
`valid.json` zawiera siedem pytań: choice, noul, score, multi, act, tool
oraz pytanie bez pola typu. Oba warianty używały jednej kolejności, trybu
`accurate` i `--no-prefix-cache`. Porównywano całe odpowiedzi po wyłącznie
`jq -c 'del(.usage.latency_ms)'`, następnie `cmp`.

| Konfiguracja | HTTP przed/po | Porównanie siedmiu odpowiedzi |
|---|---|---|
| F32, Candle, `gemm_table: none` | 200 / 200 | PASS, bajtowo identyczne po usunięciu czasu |
| F16, fused, przypięta tabela RTX 4060 mini | 200 / 200 | PASS, bajtowo identyczne po usunięciu czasu |
| Początkowa próba F16, fused, `gemm_table: none` | 200 / 200 | FAIL, różnice numeryczne między startami |

SHA-256 porównywanych odpowiedzi F32:
`dbf1a673317927c98f37149cb137a57a674a8c840345a92cfa5ddc84e0d93d83`.
SHA-256 odpowiedzi F16 z przypiętą tabelą:
`161c1dd8bb590873fd1bafaf7795fe090e82e5cb4ad6cb3ffb49d64223d54b99`.
Tabela to `nvidia-geforce-rtx-4060--basal-2.0-mini--f16--cublaslt120901.json`,
digest artefaktu `2e9ba4f7c06a1e26ff22e0f46558f242723a2d15767d2d966a21d0833fb83c96`.

Nieudana pierwsza para F16 miała plan `runtime_timing`; nie wykonano kontroli
dwóch startów tej samej binarki, która ustaliłaby przyczynę różnic. Nie zaliczamy
tej pary jako potwierdzenia zgodności bitowej. Dane pozostają w archiwum.
Nie wykonano nowego porównania numeryki z upstream FP32 ani pomiaru wydajności.

## Wynik audytu dodatkowego zakresu

1. **`/v1/models`: potrzebne uzupełnienie, wdrożone.** Runtime stosował już
   temperatury bucketowe, ale ich nie ujawniał. Dodano `temperature_per_bucket`
   w kształcie `{typ: {"min-max": temperatura}}`. Odpowiedź modelu mini jest
   równa polu z jego kalibracji. Przy `--no-calibration` oba pola temperatur są
   `{}`. Duplikaty zakresów po normalizacji zachowują pierwszą temperaturę,
   zgodnie z odczytem runtime'u. Obliczenia i loader pozostają bez zmian.
2. **Layout `CALIBRATION.oN.json`: już sprawdzany.** Przegląd
   `crates/basal-core/src/calibration.rs` potwierdził ponowne `layout_mismatch`
   po wybraniu wariantu kolejności. Nie zmieniano tego kodu ani nie wykonywano
   nowego sprawdzenia runtime z błędnym plikiem.
3. **Startowe 503 + Retry-After: luka, niewdrożona.** Modele są ładowane przed
   otwarciem listenera. Podczas ładowania GET, POST, HEAD, PUT, DELETE, OPTIONS,
   PATCH, TRACE, CONNECT i PROPFIND zakończyły się curl exit 7 / HTTP `000`:
   połączenie odrzucone, bez odpowiedzi HTTP i nagłówka Retry-After. Upstream
   odpowiada 503, `Retry-After: 5`, z body `{"status":"starting"}` poza HEAD.
   Wczesny listener oraz jego lifecycle wymagają osobnej zmiany; nie jest to
   trywialna korekta routingu. Późniejsze 503 przy OOM i 529 przy przeciążeniu
   dotyczą innych sytuacji.
4. **`type: "rank"`: dodatkowa istniejąca luka.** Upstream rozwija go przed
   walidacją prymitywów. basal-rs ma ekspansję classify/verify/compress/extract,
   ale nie rank. Teraz rank daje jawne 422 zamiast błędnego fallbacku choice.
   Wdrożenie rank pozostaje poza zakresem tej poprawki.

Pełne Pythonowe formatowanie komunikatów oraz `detail` błędów upstream nie są
częścią tej zmiany. Zachowano istniejący sposób cytowania Rust/JSON.

## Weryfikacja lokalna

Przeszły build CUDA (`cargo build --locked --release -p basal-cli --features cuda`,
compute capability 89), `cargo fmt --all --check`, Clippy całego workspace
z `--release --all-targets -- -D warnings`, Rustdoc z prywatnymi elementami
i `-D warnings`, cargo-deny ze wszystkimi features, kontrola wyjątku paste,
przegląd duplikatów zależności, kontrola wersji SDK oraz składni OpenAPI.
Actionlint 1.7.12 z ShellCheck 0.11.0 przeszedł; CI przypina actionlint 1.7.11.
Wykonano przegląd poprawności oraz osobny przegląd uproszczeń.

Lokalne Clippy/Rustdoc obejmowały Linux bez GPU. Nie potwierdzają Metal ani
Intel Vulkan; pozostałe warianty kompilacji i lintów wykonuje istniejący CI PR.
Pierwsze próby builda wymagały lokalnego GCC 14, nagłówków glibc 2.40 oraz
uzupełnienia ścieżki `libgcc_s`; nie zmieniano repozytorium dla toolchainu.
Lokalne serwery po ręcznej weryfikacji zatrzymano.

## Dane i odtworzenie

Archiwum:
`hf://buckets/itsoltech/basal-rs/measurements/upstream-252-2026-11-03/769125008e5ce7293702d318322eaa218ebef757dfe1933bcecb1784a189887b/measurement-data.tar.gz`.
SHA-256: `769125008e5ce7293702d318322eaa218ebef757dfe1933bcecb1784a189887b`.
Potwierdzono pobranie z autoryzacją i anonimowe, SHA-256 archiwum i 46/46
wpisów `SHA256SUMS` w obu kopiach. Lokalne archiwum zachowano.

Dane znajdują się pod `reports/upstream-252-2026-11-03/` w archiwum.
`OBSERVATIONS.md` opisuje wejścia i wyniki; `source.patch` odtwarza końcowy kod
na podanej bazie. `before-*` to baza, `after-*` to obserwacje pośrednie,
`final-*` to końcowy patch. `before-native.json` jest wstępną próbą bez wymaganego
pola modelu i nie wchodzi do porównania. Właściwa para to
`before-native-complete.json` i `final-native.json`.

Pobierz archiwum globalnym `hf buckets cp`, sprawdź podany SHA-256, rozpakuj
`tar -xzf measurement-data.tar.gz`, następnie uruchom `sha256sum -c SHA256SUMS`.
Po uruchomieniu lokalnego serwera ustaw `BASAL_URL` na jego adres i wykonaj:

```sh
curl -i -H 'Content-Type: application/json' \
  --data-binary @reports/upstream-252-2026-11-03/nuol.json "$BASAL_URL/v1/basal"
```

Dla porównania poprawnych wejść użyj `valid.json` z tymi samymi opcjami
modelu przed i po. Pliki `before-stable-*.json` oraz `final-stable-*.json`
zawierają odpowiedzi po opisanym usunięciu czasu; można je porównać przez `cmp`.
