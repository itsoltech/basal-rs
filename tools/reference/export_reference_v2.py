"""Export pinned upstream v2 eager readouts, including Gemma image/audio processors.

BASAL_UPSTREAM must point to the immutable v2 checkout. --out must be new.
Uses the upstream backend and server for forwards, calibration and answers; no training.
Bench input is the upstream default decision corpus, or --questions. --cases adds requests (including media).
Large raw exports belong in .cache, with a summary and their hashes in a new reports directory.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

UPSTREAM = Path(os.environ["BASAL_UPSTREAM"]).resolve()
sys.path.insert(0, str(UPSTREAM))
import numpy as np
import torch
from basal.prompt import media_messages, lang_of, PREFILL
from basal.server import Server, parser, to_items, temperature
from export_reference import bench_items, system_one_cases, sha256, versions, _prefix_len


def qwen_delta_reference(model_dir, dtype, requested):
    """Pin Qwen3.5 quality references: FLA may use TF32 dots with FP32 inputs."""
    config = json.loads((model_dir / "config.json").read_text())
    text_config = config.get("text_config") or config
    if text_config.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
        if requested is not None:
            raise ValueError("--qwen-delta requires a Qwen3.5 model")
        return None
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
    policy = requested or ("torch" if dtype == "float32" else "auto")
    selected = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        wrapper = getattr(q35, name)
        default = inspect.getclosurevars(wrapper).nonlocals.get("implementation")
        if default is None:
            raise RuntimeError(f"cannot identify the Transformers implementation of {name}")
        used = default
        if policy == "torch":
            used = inspect.unwrap(wrapper)
            if (used.__module__ != q35.__name__ or used.__name__ != name
                    or inspect.getsourcefile(used) != q35.__file__):
                raise RuntimeError(f"cannot pin the Transformers PyTorch reference for {name}")
            setattr(q35, name, used)
        selected[name] = {"default": f"{default.__module__}.{default.__qualname__}",
                          "used": f"{used.__module__}.{used.__qualname__}"}
    packages = {}
    for package in ("transformers", "flash-linear-attention", "fla-core", "causal-conv1d", "kernels", "triton"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return dict(policy=policy, functions=selected, packages=packages,
                transformers_source_sha256=sha256(q35.__file__))


def pack_meta(ids):
    prefix = _prefix_len(ids)
    lengths = [len(x) - prefix for x in ids]
    return dict(prefix_len=prefix, packed_len=prefix + sum(lengths),
                readout_cols=[prefix + sum(lengths[:i + 1]) - 1 for i in range(len(ids))])


def capture_codes(backend):
    original = backend._codes
    backend.export_logits = []

    def captured(h, ids_list):
        for i, codes in enumerate(ids_list):
            indices = torch.tensor(codes, device=h.device)
            z = h[i] @ backend.head.weight.index_select(0, indices).float().T
            if backend.head.bias is not None:
                z = z + backend.head.bias.index_select(0, indices).float()
            if backend.softcap:
                z = torch.tanh(z / backend.softcap) * backend.softcap
            backend.export_logits.append(z.detach().float().cpu().tolist())
        return original(h, ids_list)
    backend._codes = captured


def readout(srv, q, branch=0, media=None):
    b = srv.branch(0, branch, q)
    _, _, perms, prompts, codes = b
    be = srv.backend
    be.export_logits.clear()
    processor_meta = []
    if media is None:
        ids = be.encode(prompts)
        probs = be.run(prompts, codes)
    else:
        question, options = q["branches"][branch]
        messages = [media_messages(q["state"], question, [options[i] for i in perm],
                                  media["kinds"], q["lang"], srv.fmt) for perm in perms]
        prompts, ids = [], []
        for msgs in messages:
            text = srv.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False) + PREFILL
            kw = dict(text=[text], return_tensors="pt")
            if media["images"]:
                kw.update(images=[media["images"]], images_kwargs=dict(max_soft_tokens=media["max_soft_tokens"]))
            if media["audio"]:
                kw.update(audio=media["audio"])
            enc = be.proc(**kw)
            ids.append(enc["input_ids"][0].tolist())
            # Decode the expanded special tokens for exact comparison with the native prepared prompt.
            prompts.append(srv.tok.decode(ids[-1], skip_special_tokens=False, clean_up_tokenization_spaces=False))
            processor_meta.append({k: dict(shape=list(v.shape), dtype=str(v.dtype),
                                           sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest())
                                   for k, v in enc.items() if isinstance(v, torch.Tensor)})
        probs = [p for p, _, _ in be.run_media(messages, codes, media["images"], media["audio"],
                                              media["max_soft_tokens"])]
    orders = [dict(perm=perm, prompt=prompt, input_ids=tok, letter_ids=code, letter_logits=z, probs=p)
              for perm, prompt, tok, code, z, p in zip(perms, prompts, ids, codes, be.export_logits, probs)]
    canonical = np.zeros(len(perms[0]), dtype=np.float64)
    for perm, p in zip(perms, probs):
        canonical[perm] += np.asarray(p) / len(perms)
    avg = canonical.astype(np.float32).tolist()
    kind = "noul" if q["type"] == "multi" or q.get("multi") else q.get("base", q["type"])
    return dict(orders=orders, pack=pack_meta(ids), p_avg=avg, temperature=temperature(srv.cal, kind, len(avg)),
                p_cal=srv.canonical(kind, perms, probs), processor=processor_meta)


async def export(srv, args, out):
    worker = asyncio.create_task(srv.worker())
    try:
        with (out / "bench.jsonl").open("w") as f:
            for index, (source, line, item) in enumerate([] if args.cases_only else bench_items(args.n, args.questions)):
                k = len(item["options"])
                q = dict(name=str(index), type=item.get("type", "choice"), state=item["state"], question=item["question"],
                         options=item["options"], lang=item.get("lang") or lang_of(item["state"] + item["question"]),
                         branches=[(item["question"], item["options"])])
                data = dict(item)
                data.setdefault("type", "choice")
                data.setdefault("id", None)
                data.setdefault("gold", None)
                rec = {**data, "index": index, "source": source, "line": line, **readout(srv, q)}
                rec.update(argmax_avg=int(np.argmax(rec["p_avg"])), argmax_cal=int(np.argmax(rec["p_cal"])))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"bench {index}: {k} options", flush=True)
        with (out / "systemone.jsonl").open("w") as f:
            cases = ([json.loads(line.replace('"MODEL"', json.dumps(srv.name)))
                      for line in Path(args.cases).read_text().splitlines() if line.strip()]
                     if args.cases_only or args.exact_cases else system_one_cases(args.cases, srv.name))
            for case in cases:
                req = case["request"]
                req["model"] = srv.name
                rec = dict(id=case["id"], source=case.get("source"), request=req)
                try:
                    state = req.get("state")
                    if req.get("facts") == "auto":
                        from basal.facts import inject
                        state = inject(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
                    primitive = req
                    if any(q.get("type") in ("classify", "verify", "compress", "extract") for q in req["questions"].values()):
                        from basal import tasks
                        primitive, _ = tasks.expand(req, srv.max_options)
                        rec["task_export_scope"] = "items contain first-round readouts; upstream_answer contains the served task result including dependent extraction"
                    qs = to_items(state, primitive["questions"], srv.max_options, force_lang=req.get("lang"))
                    media = srv.media_of(primitive, None)
                    items = []
                    for q in qs:
                        for bi, (question, options) in enumerate(q["branches"]):
                            multi = q["type"] == "multi" or q.get("multi")
                            items.append(dict(name=f"{q['name']}#{bi}" if multi else q["name"],
                                              type="noul" if multi else q["type"], keys=["true", "false"] if multi else q["keys"],
                                              options=options, lang=q["lang"], state_text=q["state"], question_text=question,
                                              **readout(srv, q, bi, media)))
                    rec["items"] = items
                    rec["upstream_answer"] = await srv.decide(req)
                except ValueError as e:
                    rec.update(items=[], upstream_error=f"ValueError: {e}")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"systemone {case['id']}", flush=True)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--qwen-delta", choices=["torch", "auto"],
                    help="Qwen3.5 DeltaNet/convolution: default torch for float32, package auto for bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32-cpu-offload", action="store_true", help="Keep FP32 master weights in CPU RAM and stream layers to CUDA; quality reference only")
    ap.add_argument("--orders", default="1", choices=["1", "2"])
    ap.add_argument("--media", action="store_true")
    ap.add_argument("--questions")
    ap.add_argument("--cases")
    ap.add_argument("--exact-cases", action="store_true", help="Use exactly --cases without adding built-in API cases; retain bench inputs")
    ap.add_argument("--cases-only", action="store_true", help="Export only --cases, without the default corpora")
    ap.add_argument("--n", type=int, default=44)
    a = ap.parse_args()
    if (a.cases_only or a.exact_cases) and not a.cases:
        ap.error("--cases-only/--exact-cases requires --cases")
    if a.fp32_cpu_offload and (a.dtype != "float32" or a.device != "cpu"):
        ap.error("--fp32-cpu-offload requires --dtype float32 --device cpu")
    md = Path(a.model)
    qwen_delta = qwen_delta_reference(md, a.dtype, a.qwen_delta)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    argv = ["--model", str(md), "--mode", "eager", "--device", a.device, "--dtype", a.dtype, "--orders", a.orders]
    if a.media:
        argv += ["--media", "on"]
    srv = Server(parser().parse_args(argv))
    reference_execution = {"forward_device": a.device, "weights": "resident"}
    if a.fp32_cpu_offload:
        from accelerate import cpu_offload
        from accelerate.hooks import remove_hook_from_module
        be = srv.backend
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in be.model.parameters()):
            raise ValueError("FP32 reference contains non-FP32 parameters")
        torch.set_float32_matmul_precision("highest")
        # Only placement changes. Original FP32 weights remain in RAM and every forward operation runs on CUDA.
        # The upstream backend reads code rows after the model call, so the readout needs a resident head.
        # A separate allocation preserves the values of the originally tied embedding/readout weights.
        head = be.head
        weight = head.weight.detach()
        bias = head.bias.detach() if head.bias is not None else None
        cpu_offload(be.model, execution_device=torch.device("cuda:0"), offload_buffers=False)
        remove_hook_from_module(head)
        head.weight = torch.nn.Parameter(weight.to("cuda:0"), requires_grad=False)
        if bias is not None:
            head.bias = torch.nn.Parameter(bias.to("cuda:0"), requires_grad=False)
        be.dev = "cuda:0"
        reference_execution = {"forward_device": "cuda:0", "master_weights": "CPU FP32", "layer_streaming": True,
                               "resident_head_bytes": head.weight.numel() * head.weight.element_size(),
                               "scope": "Quality reference only; excluded from all HTTP performance comparisons"}
    reference_execution.update(matmul_precision=torch.get_float32_matmul_precision(),
                               cuda_tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
                               cudnn_tf32_allowed=torch.backends.cudnn.allow_tf32)
    if qwen_delta is not None:
        reference_execution["qwen35_linear_attention"] = qwen_delta
    capture_codes(srv.backend)
    asyncio.run(export(srv, a, out))
    import subprocess
    manifest = dict(tool=Path(__file__).name, exporter_sha256=sha256(__file__), upstream_revision=subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        model=srv.name, dtype=a.dtype, reference_execution=reference_execution, prompt_format=srv.fmt, orders=srv.orders, media=a.media,
        calibration=srv.cal, versions=versions(), elapsed_s=time.monotonic() - start,
        files={p.name: sha256(p) for p in md.iterdir() if p.name in (
            "config.json", "basal.json", "tokenizer.json", "CALIBRATION.json", "CALIBRATION.o2.json", "processor_config.json")})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
