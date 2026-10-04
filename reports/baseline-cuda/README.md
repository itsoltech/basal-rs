# Baseline upstream na RTX 6000 Ada

Data: 2026-10-04. Upstream `3fa2eeab`, torch 2.13.0+cu130, transformers 5.17.0,
kontener `tools/cuda/Dockerfile`. Warunki opisane w
[rust-cuda-v1](../rust-cuda-v1/README.md).

- `bench-upstream-partial.jsonl`: `basal-bench --modes eager-fp32 fast
  fast-nocompile` na 44 przykładach. Zapisane wiersze eager-fp32 i `fast`;
  serwer zresetował się w trakcie `fast-nocompile`, więc
  `bench-upstream.json` nie powstał (pełny log: `bench-upstream.log`).
- `requests-upstream-fast.json`: `tools/reference/bench_requests.py --mode
  fast`, żądania `tools/reference/requests_fanout.jsonl`.
- `gpu-before.csv`: stan GPU przed pierwszym pomiarem.
- `telemetry/`: moc, temperatura, zegary, pamięć i flagi throttle co 2 s.
  Pierwszy plik zaczyna się po resecie serwera.

Eksporty referencji: `../reference-cuda-fp32` (`--mode eager --dtype
float32`) i `../reference-cuda-bf16` (`--mode fast --dtype bfloat16`;
logity z niekompilowanego forwardu, `probs_run_shared` i `upstream_answer` ze
ścieżki serwowanej).
