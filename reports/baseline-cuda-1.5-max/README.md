# Baseline upstream v1.5.0 z basal-1.5-max na RTX 6000 Ada

Data: 2026-10-05, limit mocy GPU 250 W, warunki jak w
[rust-cuda-1.5-max](../rust-cuda-1.5-max/README.md).

- `bench-upstream-fast.json` / `.log`: `basal-bench --modes fast` (44
  przykłady). eager-fp32 nie mierzony: wagi FP32 (44,6 GB) nie mieszczą
  się w 48 GB z aktywacjami.
- `loadtest-upstream.json`, `loadtest-upstream-requests.json`:
  `tools/bench/loadtest.py` wobec `basal-serve --mode fast` (start z
  kompilacją 319 s).
- `serve-upstream.log`, `gpu-before.csv`.
- `long-states-upstream.json`: `tools/reference/bench_requests.py --mode fast`
  na `tools/bench/long_states.jsonl` (stany 1k–16k tokenów, 1 i 5 pytań).
