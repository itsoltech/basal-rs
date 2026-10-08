"""Export a per-item reference from the pinned upstream basal runtime (MLX backend on Apple Silicon, PyTorch on CUDA).

Run with the upstream environment, from the project root:

  .baseline/upstream/.venv/bin/python tools/reference/export_reference.py \
      --model .models/basal-1.0-4.5B --out reports/reference-mlx-bf16
  # CUDA: upstream FP32 reference (EagerBackend) and the served bf16 path (GraphBackend, compile + CUDA graphs)
  ... export_reference.py --mode eager --dtype float32 --out reports/reference-cuda-fp32
  ... export_reference.py --mode fast --dtype bfloat16 --out reports/reference-cuda-bf16
  # basal-1.5 models: upstream v1.5.0 checkout and its environment; FP32 of the 11B model on the CPU
  BASAL_UPSTREAM=.baseline/upstream-1.5 .baseline/upstream-1.5/.venv/bin/python tools/reference/export_reference.py \
      --model .models/basal-1.5-max --repo Remek/basal-1.5-max --revision be1b5ee7e7a9755a931262fa7fab4f59be0fd03c \
      --mode eager --device cpu --dtype float32 --out reports/reference-1.5-max-fp32

Writes (into --out, which must not exist yet):
  bench.jsonl      the 44 default basal-bench items in basal-bench order (seed 0): prompts, token ids, letter ids,
                   letter logits of both option orders (batch 1), per-order softmax, the order average and the
                   calibrated distribution (CALIBRATION.json temperature of the item type)
  systemone.jsonl  System One requests (tools/reference/systemone_cases.jsonl + upstream examples/complex.jsonl):
                   upstream to_items() view, prompts, token ids, letter ids and letter logits per order (batch 1),
                   plus the full answer of upstream Server.decide (batched as served) or its error
  manifest.json    revisions, versions, precision and parameters

The script only reads upstream; it does not modify it. Logits are taken from the same functions the backend uses:
MLX (_pack, _host_rows, _forward_mlx, lm_head; 1.5: _prefix, _forward, _head) or PyTorch (_pack, _mask_from_seg,
_forward_masked, lm_head; the uncompiled forward, as GraphBackend._eager_shared). run_shared() of the backend is called as well and its probabilities
are stored next to ours so a mismatch of this exporter with the served path would be visible.
"""
import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# BASAL_UPSTREAM selects another pinned checkout, e.g. .baseline/upstream-1.5 for the basal-1.5 models
UPSTREAM = Path(os.environ.get("BASAL_UPSTREAM", ROOT / ".baseline" / "upstream")).resolve()
sys.path.insert(0, str(UPSTREAM))

from basal.bench import DEFAULT_QUESTIONS  # noqa: E402
from basal.engine import GraphBackend  # noqa: E402
from basal.prompt import LETTERS, lang_of, letter_ids, render  # noqa: E402
from basal.server import Server, parser, to_items  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def calibrate(p, T):
    """Upstream Server.decide: softmax(log(max(p, 1e-12)) / T) when T != 1 (computed here in float64)."""
    p = np.asarray(p, dtype=np.float64)
    if T == 1.0:
        return p
    return softmax(np.log(np.maximum(p, 1e-12)) / T)


def bench_items(n, questions=None):
    """Same items and order as basal.bench.load_questions(DEFAULT_QUESTIONS, n), with source file and line; with
    `questions` (a JSONL of basal-bench questions, e.g. tools/decision-sets/build.py) its items in file order."""
    if questions:
        rows = [(Path(questions).name, i, json.loads(line))
                for i, line in enumerate(Path(questions).read_text().splitlines(), 1) if line.strip()]
        return rows[:n] if n else rows
    rows = []
    for path in DEFAULT_QUESTIONS:
        for i, line in enumerate(Path(path).read_text().splitlines(), 1):
            if line.strip():
                rows.append((Path(path).name, i, json.loads(line)))
    rows = [r for r in rows if "options" in r[2]]
    random.Random(0).shuffle(rows)  # the permutation depends only on the length, as in load_questions
    return rows[:n]


def readout(be, prompts, lids):
    """One packed group at batch 1 -> (pack, letter logits per order, logsumexp of the full vocabulary per order)."""
    toks = [be.tok(p, add_special_tokens=False).input_ids for p in prompts]
    pack = GraphBackend._pack(toks)  # 1.0: (ids, pos, seg, last); 1.5: + block parents (prefix trie)
    if hasattr(be, "mx") and not hasattr(be, "_host_rows"):  # 1.5 MLX: shared prefix in a KV cache, as MLXBackend._group
        mx = be.mx

        def run():
            P = min(len(os.path.commonprefix(toks)), min(map(len, toks)) - 1)
            prefix = be._prefix(toks[0][:P]) if P > 0 else None
            return np.array(be._head(be._forward([t[P:] for t in toks], prefix)).astype(mx.float32))
        lg = be.pool.submit(run).result()
    elif hasattr(be, "mx"):
        mx = be.mx
        ids, pos, seg, rows, cols = be._host_rows([pack], [0], 1, len(pack[0]))
        h = be._forward_mlx(mx.array(ids.numpy(), dtype=mx.int32), mx.array(pos.numpy(), dtype=mx.int32),
                            mx.array(seg.numpy(), dtype=mx.int32))
        lg = np.array(be.model.lm_head(h[mx.array(rows), mx.array(cols)]).astype(mx.float32))
    else:  # PyTorch backends (EagerBackend has no packing helpers; GraphBackend's only use model, dev and dtype)
        import torch
        t, pp, sg, last = pack[:4]
        with torch.no_grad():
            seg = torch.tensor([sg], device=be.dev)
            mask = (GraphBackend._mask_from_seg(be, seg, [pack[4]]) if len(pack) > 4
                    else GraphBackend._mask_from_seg(be, seg))
            h = GraphBackend._forward_masked(be, torch.tensor([t], device=be.dev), mask,
                                             torch.tensor([pp], device=be.dev))[0, last]
            lg = be.model.lm_head(h).float().cpu().numpy()
    lse = [float(np.log(np.exp(r.astype(np.float64) - r.max()).sum()) + r.max()) for r in lg]
    return toks, pack, [lg[m, x].astype(np.float64).tolist() for m, x in enumerate(lids)], lse


def orders_record(be, prompts, perms, lids):
    toks, pack, logits, lse = readout(be, prompts, lids)
    served = be.run_shared([(prompts, lids)])[0]
    orders = []
    for perm, prompt, t, ids, lg, s, srv in zip(perms, prompts, toks, lids, logits, lse, served):
        orders.append(dict(perm=perm, prompt=prompt, input_ids=t, letter_ids=ids, letter_logits=lg,
                           logsumexp_vocab=s, probs=softmax(lg).tolist(), probs_run_shared=srv))
    canon = np.zeros(len(perms[0]))
    for o in orders:
        for k, j in enumerate(o["perm"]):
            canon[j] += o["probs"][k] / len(orders)
    return orders, dict(prefix_len=_prefix_len(toks), packed_len=len(pack[0]), readout_cols=pack[3]), canon


def _prefix_len(toks):
    """Shared prefix length exactly as GraphBackend._pack computes it."""
    if len(toks) == 1:
        return 0
    P = min(len(t) for t in toks) - 1
    for k in range(P):
        if any(t[k] != toks[0][k] for t in toks[1:]):
            return k
    return P


def export_bench(be, temps, n, out, questions=None):
    items = bench_items(n, questions)
    n = len(items)
    with open(out / "bench.jsonl", "w") as f:
        for idx, (src, line, q) in enumerate(items):
            k = len(q["options"])
            perms = [list(range(k)), list(range(k))[::-1]]
            prompts = [render(be.tok, q["state"], q["question"], [q["options"][c] for c in p]) for p in perms]
            ids = letter_ids(be.tok, prompts[0], k)
            if not ids:
                raise SystemExit(f"item {idx}: letters are not single tokens")
            orders, pack, p_avg = orders_record(be, prompts, perms, [ids, ids])
            qtype = q.get("type", "choice")
            T = float(temps.get(qtype, 1.0))
            p_cal = calibrate(p_avg, T)
            rec = dict(index=idx, source=src, line=line, id=q.get("id"), dataset=q.get("dataset"), type=qtype, state=q["state"],
                       question=q["question"], options=q["options"], gold=q.get("gold"),
                       lang=lang_of(q["state"] + q["question"]), orders=orders, pack=pack,
                       p_avg=p_avg.tolist(), argmax_avg=int(np.argmax(p_avg)), temperature=T,
                       p_cal=p_cal.tolist(), argmax_cal=int(np.argmax(p_cal)))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"bench {idx + 1}/{n} {src}:{line} argmax={rec['argmax_avg']} gold={rec['gold']}", flush=True)


def system_one_cases(extra=None, model_name=None):
    cases = [json.loads(x) for x in (ROOT / "tools/reference/systemone_cases.jsonl").read_text().splitlines() if x.strip()]
    if extra:  # e.g. tools/reference/systemone_cases_1.5.jsonl (multi / act); "MODEL" -> the served model name
        for x in Path(extra).read_text().splitlines():
            if x.strip():
                c = json.loads(x.replace('"MODEL"', json.dumps(model_name)))
                c["source"] = str(extra)
                cases.append(c)
    for x in (UPSTREAM / "basal/examples/complex.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            req = {"model": "basal-1.0-4.5B", "state": r["state"], "questions": r["questions"]}
            cases.append({"id": r.get("id"), "request": req, "source": "upstream examples/complex.jsonl"})
    return cases


async def export_system_one(srv, out, extra=None):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    with open(out / "systemone.jsonl", "w") as f:
        for case in system_one_cases(extra, srv.name):
            req = case["request"]
            rec = dict(id=case["id"], source=case.get("source", "tools/reference/systemone_cases.jsonl"), request=req)
            try:
                items = to_items(req["state"], req["questions"])
            except Exception as e:  # noqa: BLE001
                rec["to_items_error"] = f"{type(e).__name__}: {e}"
                items = []
            rec_items = []
            for q in items:
                if len(q.get("branches", [None])) > 1:  # upstream >= 1.5 multi: one prompt per label
                    rec_items.append(dict(name=q["name"], type=q["type"], keys=q["keys"], lang=q["lang"],
                                          branches=q["branches"], note="multi: see upstream_answer"))
                    continue
                jobs = srv.jobs_for(q)
                perms = [perm for perm, _, _ in jobs]
                prompts = [p for _, p, _ in jobs]
                lids = [ids for _, _, ids in jobs]
                item = dict(name=q["name"], type=q["type"], keys=q["keys"], options=q["options"], lang=q["lang"],
                            state_text=q["state"], question_text=q["question"])
                if any(x is None for x in lids):
                    item["letter_error"] = "letters are not single tokens"
                else:
                    orders, pack, p_avg = orders_record(srv.backend, prompts, perms, lids)
                    T = float(srv.temps.get(q["type"], 1.0))
                    item.update(orders=orders, pack=pack, p_avg=p_avg.tolist(), temperature=T,
                                p_cal=calibrate(p_avg, T).tolist())
                rec_items.append(item)
            rec["items"] = rec_items
            try:
                rec["upstream_answer"] = await srv.decide(req)
            except Exception as e:  # noqa: BLE001
                rec["upstream_error"] = f"{type(e).__name__}: {e}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"systemone {case['id']}: {'error' if 'upstream_error' in rec else 'ok'}", flush=True)
    worker.cancel()


def versions():
    import tokenizers
    import torch
    import transformers
    v = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
             tokenizers=tokenizers.__version__)
    if torch.cuda.is_available():
        v.update(torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    try:
        import mlx.core as mx
        import mlx_lm
        v.update(mlx=mx.__version__, mlx_lm=mlx_lm.__version__)
    except ImportError:
        pass
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / ".models/basal-1.0-4.5B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None, help="bench items (default: 44, or all of --questions)")
    ap.add_argument("--questions", default=None,
                    help="JSONL of basal-bench questions instead of the default ones, e.g. tools/decision-sets/build.py")
    ap.add_argument("--no-system-one", action="store_true", help="bench items only (no systemone.jsonl)")
    ap.add_argument("--mode", default="mlx", help="upstream server mode (mlx; on CUDA: eager, fast, fast-nocompile)")
    ap.add_argument("--device", default=None, help="upstream >= 1.5, mode eager: cuda, mps or cpu")
    ap.add_argument("--cases", default=None, help="extra System One requests (JSONL), e.g. tools/reference/systemone_cases_1.5.jsonl")
    ap.add_argument("--repo", default="Remek/basal-1.0-4.5B", help="Hugging Face repository of --model (manifest)")
    ap.add_argument("--revision", default="b95288041cb1975bb930c6f0410819273fd2f54f", help="its revision (manifest)")
    ap.add_argument("--dtype", default="bfloat16", help="dtype of the upstream backend (bfloat16, float16, float32)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    md = Path(a.model)
    t0 = time.time()
    name = json.loads((md / "basal.json").read_text())["name"]
    args = ["--model", str(md), "--mode", a.mode, "--name", name, "--dtype", a.dtype]
    if a.device:
        args += ["--device", a.device]
    srv = Server(parser().parse_args(args))
    be = srv.backend
    if hasattr(be, "mx") and not hasattr(be, "_host_rows") and a.dtype != "bfloat16":
        # 1.5 MLX loads the checkpoint dtype (bf16); cast the whole model (as the 1.0 MLXBackend dtype argument)
        mx = be.mx
        be.pool.submit(lambda: (be.model.set_dtype(getattr(mx, a.dtype)), mx.eval(be.model.parameters()))).result()
    load_s = time.time() - t0
    t1 = time.time()
    if a.n is None and not a.questions:
        a.n = 44
    export_bench(srv.backend, srv.temps, a.n, out, a.questions)
    if a.no_system_one:
        (out / "systemone.jsonl").write_text("")  # basal export / compare read the file
    else:
        asyncio.run(export_system_one(srv, out, a.cases))
    rev = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout
    manifest = dict(
        date=time.strftime("%Y-%m-%d %H:%M:%S %z"), tool="tools/reference/export_reference.py",
        upstream=dict(repository="https://github.com/rkinas/basal", revision=rev, worktree_clean=not dirty.strip()),
        model=dict(path=str(md), repository=a.repo, revision=a.revision,
                   safetensors_sha256=sha256(md / "model.safetensors"),
                   tokenizer_json_sha256=sha256(md / "tokenizer.json"), config_json_sha256=sha256(md / "config.json"),
                   calibration_json_sha256=sha256(md / "CALIBRATION.json")),
        backend=dict(mode=a.mode, dtype=a.dtype, batch=1, letters=LETTERS,
                     note=f"letter logits computed by the {a.mode} backend forward ({a.dtype}) and lm_head, cast to float32; "
                          "softmax, average and calibration in float64 here; upstream_answer comes from "
                          "Server.decide (float32 torch, batched as served, confidence = max(p))"),
        calibration=dict(temperatures=srv.temps), n_bench=len(bench_items(a.n, a.questions)), questions=a.questions, load_s=round(load_s, 2),
        export_s=round(time.time() - t1, 2), platform=dict(machine=platform.machine(), mac_ver=platform.mac_ver()[0]),
        versions=versions())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
