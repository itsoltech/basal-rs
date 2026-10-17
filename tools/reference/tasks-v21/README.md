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
