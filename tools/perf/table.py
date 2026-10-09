"""Tables of tools/perf/run-cloud.sh runs, one directory per GPU:

    python3 tools/perf/table.py DIR [DIR ...]

Prints markdown tables: GPUs (machine.json), single decision (basal bench / upstream basal-bench), HTTP with one
question per request (1, 8, 32 clients), mixed workload (basal-rs) and energy per decision.
"""
import json
import os
import sys

MODELS = ["basal-1.5-mini", "basal-1.5-4.5B", "basal-1.5-max"]
SHORT = {"basal-1.5-mini": "mini", "basal-1.5-4.5B": "4.5B", "basal-1.5-max": "max"}


def load(d, f):
    p = os.path.join(d, f)
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def gpu_name(d):
    m = load(d, "machine.json")
    if not m:
        return os.path.basename(d.rstrip("/"))
    return m["gpu"].split(",")[0].replace("NVIDIA ", "").replace(" Generation", "")


def num(x, fmt="{:.1f}"):
    return fmt.format(x).replace(".", ",") if x is not None else "–"


def phase(doc, name, conc=None):
    if not doc:
        return None
    for p in doc["phases"]:
        if p["phase"] == name and (conc is None or p["concurrency"] == conc):
            return p
    return None


def main(dirs):
    print("| GPU | Sterownik | Compute capability | Pamięć | Limit mocy | CPU (vCPU) |")
    print("|---|---|---|---|---|---|")
    for d in dirs:
        m = load(d, "machine.json")
        if m:
            g = [x.strip() for x in m["gpu"].split(",")]
            print(f"| {gpu_name(d)} | {g[1]} | {g[2]} | {g[3]} | {g[4]} | {m['cpu']} ({m['cpus']}) |")
    print()
    print("Pojedyncza decyzja (mediana / p95 lat2, ms; w nawiasie decyzje/s):")
    print()
    print("| GPU | " + " | ".join(f"{SHORT[m]} upstream | {SHORT[m]} basal-rs" for m in MODELS) + " |")
    print("|---|" + "---:|" * (2 * len(MODELS)))
    for d in dirs:
        cells = []
        for m in MODELS:
            u = load(d, f"bench-upstream-{m}.json")
            u = u[0] if isinstance(u, list) and u else None
            r = load(d, f"bench-rust-{m}.json")
            cells.append(f"{num(u['lat2_ms'])} / {num(u['lat2_p95_ms'])} ({num(u['dec_s'], '{:.0f}')})" if u else "–")
            cells.append(f"{num(r['lat2_ms'])} / {num(r['lat2_p95_ms'])} ({num(r['dec_s'], '{:.0f}')})" if r else "–")
        print(f"| {gpu_name(d)} | " + " | ".join(cells) + " |")
    for conc, label in [(1, "1 klient"), (8, "8 klientów"), (32, "32 klientów")]:
        print()
        print(f"HTTP, jedno pytanie na żądanie, {label} (żądania/s; p50 / p99 ms):")
        print()
        print("| GPU | " + " | ".join(f"{SHORT[m]} upstream | {SHORT[m]} basal-rs" for m in MODELS) + " |")
        print("|---|" + "---:|" * (2 * len(MODELS)))
        for d in dirs:
            cells = []
            for m in MODELS:
                for who in ("upstream", "rust"):
                    doc = load(d, f"short-{who}-{m}.json")
                    p = phase(doc, "sequential") if conc == 1 else phase(doc, "concurrent", conc)
                    cells.append(f"{num(p['requests_per_s'])}; {num(p['p50_ms'], '{:.0f}')} / {num(p['p99_ms'], '{:.0f}')}"
                                 if p else "–")
            print(f"| {gpu_name(d)} | " + " | ".join(cells) + " |")
    print()
    print("Ruch mieszany, basal-rs (żądania/min; p50 / p95 s):")
    print()
    print("| GPU | " + " | ".join(f"{SHORT[m]} sekwencyjnie | {SHORT[m]} 32 klientów" for m in MODELS) + " |")
    print("|---|" + "---:|" * (2 * len(MODELS)))
    for d in dirs:
        cells = []
        for m in MODELS:
            doc = load(d, f"mixed-rust-{m}.json")
            for p in (phase(doc, "sequential"), phase(doc, "concurrent", 32)):
                cells.append(f"{num(p['requests_per_s'] * 60, '{:.0f}')}; {num(p['p50_ms'] / 1e3, '{:.2f}')} / "
                             f"{num(p['p95_ms'] / 1e3, '{:.2f}')}" if p else "–")
        print(f"| {gpu_name(d)} | " + " | ".join(cells) + " |")
    print()
    print("Energia na decyzję przy 32 klientach, jedno pytanie (J; upstream / basal-rs):")
    print()
    print("| GPU | " + " | ".join(SHORT[m] for m in MODELS) + " |")
    print("|---|" + "---:|" * len(MODELS))
    for d in dirs:
        cells = []
        for m in MODELS:
            u = phase(load(d, f"short-upstream-{m}.json"), "concurrent", 32)
            r = phase(load(d, f"short-rust-{m}.json"), "concurrent", 32)
            cells.append(f"{num(u.get('joules_per_decision') if u else None)} / "
                         f"{num(r.get('joules_per_decision') if r else None)}")
        print(f"| {gpu_name(d)} | " + " | ".join(cells) + " |")


main(sys.argv[1:])
