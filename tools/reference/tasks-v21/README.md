# Manual host comparison with upstream v2.1

This runner compares task expansion, dependent extraction requests and response assembly.
It supplies explicit synthetic primitive answers; it does **not** run a model or measure
accuracy/performance. It is not a CI test framework or gate. Use a read-only upstream
checkout at `62542d31e8eb69000839c01e8f9b45da1fb80e49` and a new artifact directory.

```sh
python3 tools/reference/tasks-v21/generate.py --upstream UPSTREAM --measurement-artifacts ARTIFACTS
python3 tools/reference/tasks-v21/host-reference.py --upstream UPSTREAM --measurement-artifacts ARTIFACTS > ARTIFACTS/upstream.jsonl
cargo run --locked --release -p basal-core --example tasks_reference -- --measurement-artifacts ARTIFACTS > ARTIFACTS/basal-rs-r1.jsonl
python3 tools/reference/tasks-v21/audit.py --measurement-artifacts ARTIFACTS --candidate basal-rs-r1.jsonl
```

The auditor requires exact IDs and compares numeric values as binary64, including signed zero.
Keep failed attempts. Live SDK comparisons use `tools/sdk/live-basal2-omni/` with task inputs.
Publish sanitized observations using `docs/MEASUREMENT_ARTIFACTS.md`.

Pass `--max-options 10` to the generator to exercise the format-1 task candidate limit;
its limit is saved in each input row and used by both exporters. The default is 255.

`run-gpu.py` runs finite baseline/candidate exports and current upstream FP32/BF16 exports
inside a separately supervised session. It never rents hardware, downloads models or
changes cooling. Supply `--measurement-artifacts` with `primitives/` and `tasks/` inputs,
`--model-dir`, explicit binaries, a new `--run-name`, and optionally the pinned upstream
checkout and its Python environment. Native and reference stages can run separately.
`method.json` reports execution status; inspect A/B and numerical/task comparisons before
making a quality or performance claim. Retain failed exports and private process logs.

Use `--fp32-device cpu --reference-dtypes float32 --reference-cpu-threads N` for a
checkpoint whose full upstream FP32 weights and activations exceed device memory.
This changes the quality reference's device; HTTP timing still compares GPU BF16.
`--fp32-device cpu-offload` instead retains the FP32 master weights in RAM and
streams layers through CUDA with Accelerate; the readout head remains resident.
It disables TF32 through the highest float32 matmul precision and records placement
in `manifest.json`. Validate the exported probabilities against a CPU FP32 subset
before using this reference. Neither CPU nor streamed FP32 export time is an HTTP
performance result.
`--native-library-path` affects only the native process. `--half-gemm-table` freezes
f16 GEMMs in auto/f16 A/B exports; without a fixed Llama table, autotuning can pick
different reductions. The reference exporter uses exactly the frozen API cases and
retains the supplied benchmark rows, without adding its default corpora.

`prepare-http.py` freezes PL/EN text ladders using each model's pinned tokenizer,
32 distinct task documents and checksum-verified real media. Pass the resulting
catalogue to `tools/perf/omni-optimize.py --model-names FULL_NAME ...`; use
`tools/perf/summarize-omni.py` to audit complete runs. Task extraction may perform
different continuation work when engines choose different source regions; all
response token/branch counts are retained. Finite corpus timing is separate from
FP32 answer quality and cannot establish a general performance advantage.
Pass `--ladder-choice blue` to the performance summarizer for this generator's
known-answer ladder, in addition to checking status, token lengths and invariance.
If a combined run stops after one surface completes, the summarizer's explicit
`--surfaces text` or `--surfaces media` audits only that surface, including all
scheduled phases and server stops. It preserves the incomplete parent status;
it does not mark missing phases on other surfaces as completed.
For finite GPU budgets, `omni-optimize.py` accepts explicit
`--latency-warmup-samples` and `--load-warmup-samples`. They change discarded
warmup work only and are recorded in `method.json`; defaults retain the older
protocol. Report these settings together with retained sample counts.

`restore-inputs.py` reconstructs one archived frozen input directory using separately
downloaded media. It accepts the campaign through `--measurement-artifacts`, a
relative `--input-subdir`, supplied `--assets` and a new `--out`. Public and restored
original byte checksums are verified. Media sources and hashes remain in the archive;
the runner does not download anything.

`probe-http.py` records task answers on both endpoints, intentional admission
rejections at `max_inflight=4`, and replica reservation drain after disconnected
clients. It needs a separately supervised server with `--metrics`, and a new output
directory. Its intentional 529 series is separate from capacity measurements.

`summarize-quality.py` takes complete exports through `--reference` and repeated
`--candidate LABEL=RELATIVE_DIRECTORY`, invokes a supplied CPU CLI `--basal`, and
retains categorical answers, extraction spans/data, automation changes and mean TV.
It audits observations; it does not turn FP32 agreement into semantic accuracy.
