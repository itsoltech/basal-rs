#!/usr/bin/env python3
"""Manual SDK/model comparison; requires a live GPU server, never runs in CI."""
import argparse
import asyncio
import base64
from dataclasses import asdict
import json
import os
from pathlib import Path

from basal_client import (AsyncBasalClient, BasalClient, Questions, choice, noul, score,
                          multi, act, tool, image, audio, BasalHTTPError)


def definitions(case):
    result = {}
    for key, wire in case["questions"].items():
        args = dict(wire)
        kind, instructions = args.pop("type"), args.pop("instructions")
        if kind == "choice": result[key] = choice(instructions, args.pop("criteria"), **args)
        elif kind == "score": result[key] = score(instructions, args.pop("criteria"), **args)
        elif kind == "noul": result[key] = noul(instructions, **args)
        elif kind == "multi": result[key] = multi(instructions, args.pop("criteria"), **args)
        elif kind == "act": result[key] = act(instructions, args.pop("costs"), **args)
        elif kind == "tool": result[key] = tool(instructions, args.pop("tools"), **args)
        else: raise ValueError("unsupported diagnostic case")
    return result


def options(case, root):
    result = {key: case[key] for key in ("lang", "detail") if key in case}
    result["model"] = os.environ["BASAL_MODEL"]
    if "media" in case:
        result["media"] = []
        for item in case["media"]:
            data = (root / item["file"]).read_bytes()
            if item["encoding"] != "bytes":
                data = base64.b64encode(data).decode("ascii")
                if item["encoding"] == "data-url": data = f'data:{item["mime"]};base64,{data}'
            result["media"].append((image if item["type"] == "image" else audio)(data))
    return result


def project(answer):
    result = asdict(answer)
    result["type"] = type(answer).__name__.removesuffix("Answer").lower()
    if result.get("selection") is not None: result["selection"]["type"] = "multi"
    return result


def emit(case, response, mode):
    answers = response.answers
    projected = {key: project(response.answer(key) if mode != "schema" else getattr(answers, key)) for key in case["questions"]}
    print(json.dumps({"sdk": "python-" + mode, "case": case["id"], "answers": projected,
                      "raw": response.raw}, ensure_ascii=False, allow_nan=False), flush=True)


async def asynchronous(cases, root, url):
    async with AsyncBasalClient(url, timeout=300) as client:
        print(json.dumps({"sdk": "python-async", "models": await client.models()}), flush=True)
        fields={"n":noul("Is this a diagnostic?")}
        try: await client.decide("diagnostic",fields,model="__sdk_unknown_model__")
        except BasalHTTPError as error:
            assert error.status_code==422
            print(json.dumps({"sdk":"python-async","check":"unknown-model","status":error.status_code}),flush=True)
        else: raise ValueError("unknown model accepted")
        try: await client.decide("diagnostic",fields,media=[])
        except ValueError: print(json.dumps({"sdk":"python-async","check":"empty-media","status":"local-rejection"}),flush=True)
        else: raise ValueError("empty media accepted")
        for case in cases: emit(case, await client.decide(case["state"], definitions(case), **options(case, root)), "async")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-artifacts", type=Path, required=True)
    parser.add_argument("--mode", choices=["sync", "async", "schema"], default="sync")
    args = parser.parse_args()
    root = args.measurement_artifacts
    cases = [json.loads(line) for line in (root / "inputs/cases.jsonl").read_text().splitlines()]
    if os.environ.get("BASAL_MEDIA", "0") != "1": cases = [c for c in cases if "media" not in c]
    url = os.environ.get("BASAL_URL", "http://127.0.0.1:19439")
    if args.mode == "async": asyncio.run(asynchronous(cases, root, url)); return
    with BasalClient(url, timeout=300) as client:
        print(json.dumps({"sdk": "python-" + args.mode, "models": client.models()}), flush=True)
        fields={"n":noul("Is this a diagnostic?")}
        try: client.decide("diagnostic",fields,model="__sdk_unknown_model__")
        except BasalHTTPError as error:
            assert error.status_code==422
            print(json.dumps({"sdk":"python-"+args.mode,"check":"unknown-model","status":error.status_code}),flush=True)
        else: raise ValueError("unknown model accepted")
        try: client.decide("diagnostic",fields,media=[])
        except ValueError: print(json.dumps({"sdk":"python-"+args.mode,"check":"empty-media","status":"local-rejection"}),flush=True)
        else: raise ValueError("empty media accepted")
        for case in cases:
            fields = definitions(case)
            schema = type("LiveQuestions", (Questions,), fields) if args.mode == "schema" else fields
            emit(case, client.decide(case["state"], schema, **options(case, root)), args.mode)


if __name__ == "__main__": main()
