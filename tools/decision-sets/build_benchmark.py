"""Freeze the public-900 and mixed HTTP workloads of `basal benchmark` (maintainer tool).

    python3 tools/decision-sets/build_benchmark.py --set .cache/decision-sets/set.jsonl \
        --out crates/basal-cli/benchmark-data/v2

No network, tokenizer or GPU. Inputs must match the published public-900 SHA-256 and the lock of the
profile. Outputs are model-independent UTF-8 JSONL with LF, plus a manifest; an existing directory is
refused. Users do not run this: the binary embeds the v2 output from crates/basal-cli/benchmark-data/v2.

--profile v2 (default) keeps the key order and number literals of the sources, so that each payload
parses to the request the source produces. `model` is removed from the name but its position is kept
as null; benchmark sets the selected model in place. Sequential requests have no `model` key.
--profile v1 reproduces the historical v1 files byte for byte. v1 sorts keys at every level, which
reorders object-valued states, criteria and questions of some mixed.jsonl requests (docs/BENCHMARKS.md).

The lock pins the inputs and the generated files; a mismatch is an error, not a new set under the
same name. A new profile starts with a lock that pins only the inputs; the generator then prints the
output hashes to add to it.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "v1": {
        "name": "public_900_and_synthetic_mixed_v1",
        "lock": "tools/decision-sets/benchmark-v1.lock.json",
        "schema_version": 1,
        "sort_keys": True,
        "model_slot": False,
    },
    "v2": {
        "name": "public_900_and_synthetic_mixed_v2",
        "lock": "tools/decision-sets/benchmark-v2.lock.json",
        "schema_version": 2,
        "sort_keys": False,
        "model_slot": True,
    },
}
DATASETS = {
    "polemo2-in",
    "allegro-reviews",
    "cdsc-e",
    "dyk",
    "cbd",
    "ag-news",
    "emotion",
    "boolq",
    "sst5",
}
FEATURE_IDS = {"m15-multi-pl", "m15-mixed", "m15-evidence-choice-en", "m15-facts-dates"}
ORDER_PREFIX = "mixed-v1/7/"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encode(value, sort_keys):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def number(kind):
    # serde_json arbitrary_precision keeps the literal; refuse a literal that json.dumps would rewrite.
    def parse(literal):
        value = kind(literal)
        if json.dumps(value) != literal:
            raise ValueError(
                f"number {literal} would be written as {json.dumps(value)}"
            )
        return value

    return parse


def unique_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate object key among {keys}")
    return dict(pairs)


def no_constant(name):
    raise ValueError(f"{name} is not JSON")


def records(data):
    # LF only, like Rust str::lines(); splitlines() also splits on U+2028 and similar characters.
    return [
        json.loads(
            line,
            object_pairs_hook=unique_keys,
            parse_float=number(float),
            parse_int=number(int),
            parse_constant=no_constant,
        )
        for line in data.decode("utf-8").split("\n")
        if line.strip()
    ]


def public_request(row):
    options = row["options"]
    kind = row["type"]
    if kind == "choice":
        criteria = {f"option_{i}": value for i, value in enumerate(options)}
    elif kind == "score":
        criteria = options
    elif kind == "noul" and len(options) == 2:
        criteria = {"true": options[0], "false": options[1]}
    else:
        raise ValueError(f"invalid public question {row['id']}: {kind}")
    return {
        "state": row["state"],
        "questions": {
            "q": {
                "type": kind,
                "instructions": row["question"],
                "criteria": criteria,
                "option_keys": "hide",
            }
        },
    }


def summary(rows, data, sort_keys):
    payloads = [encode(row["request"], sort_keys) for row in rows]
    return {
        "sha256": sha256(data),
        "requests": len(rows),
        "questions": sum(len(row["request"]["questions"]) for row in rows),
        "classes": dict(
            sorted(
                collections.Counter(
                    row.get("class", "auto-payload-size") for row in rows
                ).items()
            )
        ),
        "unique_payloads": len(set(payloads)),
        "ordered_model_free_payloads_sha256": sha256(
            b"".join(hashlib.sha256(body).digest() for body in payloads)
        ),
    }


def generate(set_path, profile):
    lock = json.loads((ROOT / profile["lock"]).read_text(encoding="utf-8"))
    if lock["profile"] != profile["name"]:
        raise ValueError(f"{profile['lock']} pins {lock['profile']}")
    sort_keys = profile["sort_keys"]
    data = set_path.read_bytes()
    digest = sha256(data)
    if digest != lock["public_set_sha256"]:
        raise ValueError(
            f"public set SHA-256 {digest}; expected {lock['public_set_sha256']}"
        )
    public = records(data)
    datasets = collections.Counter(row["dataset"] for row in public)
    if len(public) != 900 or datasets != {name: 100 for name in DATASETS}:
        raise ValueError(
            "expected exactly 100 questions from each of the nine public datasets"
        )
    if len({row["id"] for row in public}) != 900:
        raise ValueError("public question IDs must be unique")
    sequential = [{"id": row["id"], "request": public_request(row)} for row in public]

    inputs = {}

    def fixture(path):
        raw = (ROOT / path).read_bytes()
        inputs[path] = sha256(raw)
        if inputs[path] != lock["fixture_sha256"][path]:
            raise ValueError(
                f"{path}: content differs from {profile['lock']}; use a new profile version"
            )
        return records(raw)

    fanout = fixture("tools/reference/requests_fanout.jsonl")
    features = [
        row
        for row in fixture("tools/reference/systemone_cases_1.5.jsonl")
        if row["id"] in FEATURE_IDS
    ]
    if {row["id"] for row in features} != FEATURE_IDS or len(features) != len(
        FEATURE_IDS
    ):
        raise ValueError(
            "the mixed profile needs each of the four selected feature cases exactly once"
        )
    docs = fixture("tools/bench/long_states.jsonl")
    mixed = [
        {
            "id": f"short-1q/{i}",
            "class": "short-1q",
            "request": sequential[(i * 541 + 7) % 900]["request"],
        }
        for i in range(360)
    ]
    pools = [
        ("short-multi", 180, fanout),
        ("features", 90, features),
        (
            "doc-1k-2k",
            180,
            [row for row in docs if row["id"].startswith(("long-1000-", "long-2000-"))],
        ),
        (
            "doc-4k-8k",
            72,
            [row for row in docs if row["id"].startswith(("long-4000-", "long-8000-"))],
        ),
        ("doc-16k", 18, [row for row in docs if row["id"].startswith("long-16000-")]),
    ]
    for name, count, pool in pools:
        if not pool:
            raise ValueError(f"empty mixed pool: {name}")
        for i in range(count):
            source = pool[i % len(pool)]["request"]
            if profile["model_slot"]:
                request = {
                    key: None if key == "model" else value
                    for key, value in source.items()
                }
            else:
                request = {
                    key: value for key, value in source.items() if key != "model"
                }
            mixed.append({"id": f"{name}/{i}", "class": name, "request": request})
    mixed.sort(
        key=lambda row: hashlib.sha256(
            (ORDER_PREFIX + row["id"]).encode("utf-8")
        ).digest()
    )

    files = {}
    workloads = {}
    for name, rows in [("sequential", sequential), ("mixed", mixed)]:
        encoded = b"".join(encode(row, sort_keys) + b"\n" for row in rows)
        files[f"{name}.jsonl"] = encoded
        workloads[f"{name}.jsonl"] = summary(rows, encoded, sort_keys)
    if profile["model_slot"]:
        conditions = [
            "Payloads contain no model name; benchmark sets the selected model, in place where model is null.",
        ]
    else:
        conditions = [
            "Payloads contain no model name; benchmark inserts the selected model.",
        ]
    conditions += [
        "Sequential keeps public-set order; mixed uses the same ID-based order at every concurrency.",
        "Use identical concurrency, warmup, precision, scheduler and cache settings for comparisons.",
        "Repeated payloads are intentional; unique_payloads documents cache opportunities.",
        "Concurrent completion/batching order and measured times are not deterministic.",
    ]
    manifest = {
        "schema_version": profile["schema_version"],
        "profile": profile["name"],
        "public_set_sha256": digest,
        "public_sampling": {
            "seed": 0,
            "per_dataset": 100,
            "datasets": dict(sorted(datasets.items())),
        },
        "fixture_sha256": inputs,
    }
    if profile["schema_version"] >= 2:
        manifest["encoding"] = (
            "UTF-8 JSONL, LF, compact separators; source key order and number literals kept"
        )
        manifest["mixed_recipe"] = [
            {
                "class": "short-1q",
                "requests": 360,
                "source": "sequential request (i * 541 + 7) % 900",
            },
        ] + [
            {
                "class": name,
                "requests": count,
                "source": "cycled: " + ", ".join(row["id"] for row in pool),
            }
            for name, count, pool in pools
        ]
    manifest["mixed_order"] = {
        "seed": 7,
        "method": 'ascending SHA-256("mixed-v1/7/" + request_id)',
    }
    manifest["workloads"] = workloads
    manifest["conditions"] = conditions
    files["manifest.json"] = encode(manifest, sort_keys) + b"\n"

    locked = lock.get("workload_sha256", {})
    for name, encoded in files.items():
        if name in locked and sha256(encoded) != locked[name]:
            raise ValueError(
                f"{name}: generated content differs from {profile['lock']}; use a new profile version"
            )
    return files, locked


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--set", type=Path, default=Path(".cache/decision-sets/set.jsonl")
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v2")
    parser.add_argument(
        "--out", type=Path, required=True, help="new directory; refuses overwriting"
    )
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise ValueError(f"output directory already exists: {args.out}")
        files, locked = generate(args.set, PROFILES[args.profile])
        args.out.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            with (args.out / name).open("xb") as output:
                output.write(data)
            pinned = "pinned" if name in locked else "NOT pinned by the lock"
            print(f"{name}: {len(data)} bytes; SHA-256 {sha256(data)}; {pinned}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"benchmark workload generation failed: {error}\n")


if __name__ == "__main__":
    main()
