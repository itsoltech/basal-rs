# Manual SDK comparison with a live GPU server

These finite runners exercise the real Python, TypeScript, .NET and Rust SDKs.
They are measurement tools, never CI gates or a test framework. They do not
provision, start or stop a server. Use a separately supervised, loopback-only
server with one model loaded, order 1 (plus an explicit auto-order series), prefix/state caches disabled and no GEMM
autotuning. Run one client at a time. This is contract and decoding verification,
not a latency benchmark or a new model-quality evaluation.

Download the archive referenced by the SDK campaign report, verify its SHA-256
and every `SHA256SUMS` entry, and point `--measurement-artifacts` at its
`reports/sdk-basal2-omni-2026-10-17/` directory. `audit.py` can check archived
observations without a server:

```bash
python tools/sdk/live-basal2-omni/audit.py --measurement-artifacts "$BASAL_DATA/runs/basal2-mini-r1"
```

For new inference, first reconstruct the earlier Omni campaign's real media
using its report, then restore the two assets (the destination must be new):

```bash
python tools/sdk/live-basal2-omni/restore-media.py \
  --measurement-artifacts "$BASAL_DATA" --source-cases "$BASAL_RESTORED/cases-real-media.jsonl"
```

Build and install the Python wheel with the hash-locked tooling under `tools/sdk`.
Build and pack the npm SDK, install that local tarball in a separate private npm
project with `@types/node`, copy `typescript.mts` there, and compile it with
`tsc --strict --target ES2022 --module NodeNext --moduleResolution NodeNext`.
Both its ESM import and CJS require use the installed tarball. Build the .NET
runner with `dotnet build tools/sdk/live-basal2-omni/Live.csproj -c Release` and
the Rust example with `cargo build --locked --release -p basal-client --example
live_basal2_omni`. The .NET runner targets .NET 8; `run.py` permits a newer
installed runtime through `DOTNET_ROLL_FORWARD=Major`.

```bash
python tools/sdk/live-basal2-omni/run.py \
  --measurement-artifacts "$BASAL_DATA" --model basal-2.0-mini \
  --run-name basal2-mini-new --url http://127.0.0.1:19439 \
  --python "$BASAL_PYTHON" --typescript "$BASAL_NPM/build/typescript.mjs" \
  --dotnet tools/sdk/live-basal2-omni/bin/Release/net8.0/Live.dll \
  --rust target/release/examples/live_basal2_omni
```

For Omni, select its exact discovery name and append `--media`. New run names
must be unused; retain failed attempts. The auditor requires all SDK modes,
named APIs, discovery metadata, HTTP 422 handling, empty-media rejection and native case references. It compares decoded
answer fields and usage numerically by IEEE-754 binary64 bits, including signed
zero. All adapters forward `facts`, `think`, `think_text` and `early_exit` when supplied,
including the named API modes. Missing/null early-exit are semantically identical;
ergonomic clients omit null. Corpus policies are strings/null and thinking options
use valid strings/booleans, matching the ergonomic API. No orders/layout/tau/overflow
request overrides are added.

The auditor observes presence and values of optional usage `lane`, `truncated`,
`think_questions`, discovery `layout`, `calibration_warning`, `think`, policy inventory/
default, and raw answer `think`/`meta`, including their nested unknown keys. It compares
only the presence of `think.ms`, not its nondeterministic value; latency remains excluded.
Existing probability comparisons and required answer projections are unchanged.
It does not compare raw JSON text, other unknown response fields or transport framing.
Rust records typed projections, usage and raw think/meta rather than the complete response.
Static compilation is not a live QA result. Pico/Mini cannot provide positive trained
thinking/head observations; those remain a separate 4.5B runtime GAP.
