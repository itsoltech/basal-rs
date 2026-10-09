"""Where the GPU time of basal forwards goes, from an Nsight Systems trace:

  nsys profile -t cuda -o TRACE basal ...; nsys export -t sqlite TRACE.nsys-rep
  python3 tools/perf/nsys_forward.py TRACE.sqlite [--gap-ms 1.0] [--skip 1]

GPU activities (kernels, copies, memsets) are split into bursts at idle gaps longer than --gap-ms (a forward, or the
forwards of one request, run back to back); the first --skip bursts (warm-up) are left out. Per burst: wall span from
the first start to the last end, GPU busy time (union of the activities), idle share, number of kernels, and the
kernel time per category. Prints the median burst of each distinct kernel count (one line per kind of forward).
"""
import argparse
import sqlite3
import statistics
from collections import defaultdict

CATEGORIES = [
    ("attention", ("attn_tree", "attention", "masked_softmax")),
    ("split_hilo", ("split_hilo",)),
    ("merge_heads", ("merge_heads",)),
    ("qkv_rope", ("qkv_rope",)),
    ("norm+residual", ("rmsnorm", "residual")),
    ("silu", ("silu",)),
    ("readout", ("letter_logits",)),
    ("gemm", ("gemm", "xmma", "cutlass", "splitk", "gemv", "kernel2", "nvjet")),  # Kernel2, nvjet: cuBLASLt
]


def category(name):
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-ms", type=float, default=1.0)
    ap.add_argument("--skip", type=int, default=1)
    a = ap.parse_args()
    db = sqlite3.connect(a.trace)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    acts = []  # (start, end, kind, name)
    for s, e, n in db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        acts.append((s, e, "kernel", names.get(n, "?")))
    for t, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if t in tables:
            acts += [(s, e, kind, kind) for s, e in db.execute(f"SELECT start, end FROM {t}")]
    acts.sort()
    gap = a.gap_ms * 1e6
    bursts, cur, last_end = [], [], None
    for act in acts:
        if cur and act[0] - last_end > gap:
            bursts.append(cur)
            cur = []
        cur.append(act)
        last_end = act[1] if last_end is None or not cur[:-1] else max(last_end, act[1])
    if cur:
        bursts.append(cur)
    rows = []
    for b in bursts[a.skip:]:
        span = max(x[1] for x in b) - b[0][0]
        busy, s0, e0 = 0, None, None
        for s, e, _, _ in b:
            if e0 is None or s > e0:
                if e0 is not None:
                    busy += e0 - s0
                s0, e0 = s, e
            else:
                e0 = max(e0, e)
        busy += e0 - s0
        cats = defaultdict(float)
        for s, e, kind, name in b:
            cats[category(name) if kind == "kernel" else kind] += e - s
        rows.append((sum(1 for x in b if x[2] == "kernel"), span, busy, cats))
    by_n = defaultdict(list)
    for r in rows:
        by_n[r[0]].append(r)
    cols = [c for c, _ in CATEGORIES] + ["other", "memcpy", "memset"]
    print("| kernels | bursts | span ms | busy ms | idle % | " + " | ".join(cols) + " |")
    print("|---:|---:|---:|---:|---:|" + "---:|" * len(cols))
    for n in sorted(by_n):
        rs = sorted(by_n[n], key=lambda r: r[1])
        span, busy, cats = rs[len(rs) // 2][1:]
        shares = [f"{100 * cats.get(c, 0) / busy:.1f}%" for c in cols]
        print(f"| {n} | {len(rs)} | {span / 1e6:.2f} | {busy / 1e6:.2f} | {100 * (span - busy) / span:.1f} | "
              + " | ".join(shares) + " |")
    if rows:
        med = statistics.median(r[1] for r in rows) / 1e6
        print(f"\n{len(rows)} bursts, median span {med:.2f} ms")


if __name__ == "__main__":
    main()
