"""Host side of basal forwards in an Nsight Systems trace (export -t sqlite):

  python3 tools/perf/nsys_host.py TRACE.sqlite [--gap-ms 1.0] [--skip 1]

GPU bursts as in nsys_forward.py; per burst, the CUDA API calls of the host thread from the end of the previous burst
to the end of this one: launches (count, time spent in them), copies, synchronisations, and the times that decide
whether the host or the GPU is the limit: host lead (first launch -> first kernel start), the time the GPU waited for
launches inside the burst (sum of gaps between kernels), the time from the last launch to the end of the burst. Prints
medians per kernel count.
"""
import argparse
import sqlite3
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-ms", type=float, default=1.0)
    ap.add_argument("--skip", type=int, default=1)
    a = ap.parse_args()
    db = sqlite3.connect(a.trace)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    acts = sorted(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    api = sorted((s, e, names.get(n, "?")) for s, e, n in
                 db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    gap = a.gap_ms * 1e6
    bursts, cur = [], []
    for k in acts:
        if cur and k[0] - max(x[1] for x in cur[-8:]) > gap:
            bursts.append(cur)
            cur = []
        cur.append(k)
    if cur:
        bursts.append(cur)
    rows = defaultdict(list)
    prev_end = 0
    ai = 0
    for b in bursts:
        b0, b1 = b[0][0], max(x[1] for x in b)
        calls = []
        while ai < len(api) and api[ai][0] < b1:
            if api[ai][0] >= prev_end:
                calls.append(api[ai])
            ai += 1
        prev_end = b1
        launches = [c for c in calls if "Launch" in c[2]]
        if not launches:
            continue
        idle, last = 0, b[0][1]
        for s, e in b[1:]:
            if s > last:
                idle += s - last
            last = max(last, e)
        r = {
            "kernels": len(b),
            "span": (b1 - b0) / 1e6,
            "busy": sum(e - s for s, e in b) / 1e6,
            "idle": idle / 1e6,
            "launches": len(launches),
            "launch_ms": sum(e - s for s, e, _ in launches) / 1e6,
            "enqueue": (launches[-1][1] - launches[0][0]) / 1e6,
            "lead": (b0 - launches[0][0]) / 1e6,
            "tail": (b1 - launches[-1][1]) / 1e6,
            "copy_ms": sum(e - s for s, e, n in calls if "Memcpy" in n) / 1e6,
            "sync_ms": sum(e - s for s, e, n in calls if "Synchronize" in n) / 1e6,
            "alloc_ms": sum(e - s for s, e, n in calls if "Alloc" in n or "Free" in n) / 1e6,
        }
        rows[r["kernels"]].append(r)
    keys = ["span", "busy", "idle", "launches", "launch_ms", "enqueue", "lead", "tail", "copy_ms", "sync_ms", "alloc_ms"]
    print("| kernels | n | " + " | ".join(keys) + " |")
    print("|---:|---:|" + "---:|" * len(keys))
    for k in sorted(rows):
        rs = rows[k][a.skip:] or rows[k]
        med = {key: statistics.median(r[key] for r in rs) for key in keys}
        print(f"| {k} | {len(rs)} | " + " | ".join(f"{med[key]:.2f}" for key in keys) + " |")


if __name__ == "__main__":
    main()
