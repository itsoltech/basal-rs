"""Short GPU sessions on Shadeform for tests of the working tree: rent, build, copy, run, fetch, delete.

    python3 tools/cloud/basal-cloud.py COMMAND [options]

    types   [--gpu RTX6000Ada] [--num-gpus 1]    on-demand machines available now, cheapest first
    template                                     save or update the Shadeform template basal-rs-test
    up      [--gpu RTX6000Ada] [--max-price 1.2] [--hours 3] [--spend 4] [--prefetch "4.5B"]
                                                 cheapest machine with the template; waits until SSH works
    build   [--host USER@HOST] [--caps 80,89,90] CUDA binary of the working tree (Docker on HOST, no GPU used)
    push                                         the binary to the machine (/opt/basal-dev, command basal-dev)
    sync    PATH...                              repository paths to /work on the machine (e.g. reports/choice-sets)
    run     'COMMAND'                            in /work on the machine, with basal-dev in PATH
    pull    REMOTE LOCAL                         a file or directory from /work back to the repository
    ssh                                          the ssh command of the machine
    ls                                           machines tagged basal-rs
    down    [--all | ID]                         delete the session's machine (or all tagged basal-rs)

Needs the `shade` CLI logged in (`shade auth login`), ssh and rsync. Every machine is created with Shadeform's
auto-delete after --hours and after --spend dollars, so a forgotten one stops costing money. The session (machine id,
address) is kept in .cache/cloud/session.json. Defaults: the RTX 6000 Ada (the GPU of the measurements in reports/),
other GPUs only with --gpu. The build host defaults to BASAL_BUILD_HOST.

The template's start script installs the latest basal release (install.sh), downloads the CUDA 12.9 libraries
(`basal setup`) and the models of BASAL_PREFETCH into /data, then writes /data/ready.
"""
import base64
import datetime
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(ROOT, ".cache", "cloud")
SESSION = os.path.join(STATE_DIR, "session.json")
KNOWN_HOSTS = os.path.join(STATE_DIR, "known_hosts")
TEMPLATE = "basal-rs-test"
TAG = "basal-rs"
SSH_KEY_NAME = os.environ.get("BASAL_CLOUD_SSH_KEY", "macbook")
CUDA_LIBS = "/data/basal/.local/cuda/12.9.1/lib"

START_SCRIPT = r"""#!/bin/bash
# basal-rs test machine (tools/cloud/basal-cloud.py): release install, CUDA libraries, models; then /data/ready
set -eux
mkdir -p /data && chmod 1777 /data
cat >> /etc/environment <<'ENV'
HF_HOME=/data/hf
BASAL_HOME=/data/basal
BASAL_NO_UPDATE_CHECK=1
ENV
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
curl --fail -sSL -o /tmp/install.sh https://raw.githubusercontent.com/itsoltech/basal-rs/main/install.sh
sh /tmp/install.sh --prefix /opt/basal-release --no-setup
/opt/basal-release/bin/basal --color never setup
for m in ${BASAL_PREFETCH:-4.5B}; do /opt/basal-release/bin/basal --color never setup --prefetch --model "$m"; done
mkdir -p /opt/basal-dev /work && chmod 1777 /opt/basal-dev /work
cat > /usr/local/bin/basal-dev <<'SH'
#!/bin/sh
# the CUDA build of the working tree (basal-cloud.py push) with the CUDA libraries of `basal setup`
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
export LD_LIBRARY_PATH=@LIBS@${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /opt/basal-dev/basal-cuda "$@"
SH
chmod 755 /usr/local/bin/basal-dev
chmod -R a+rwX /data
touch /data/ready
""".replace("@LIBS@", CUDA_LIBS)


def shade(*args, body=None):
    cmd = ["shade", *args, "-o", "json", "--agent-mode=false", "--no-interactive"]
    if body is not None:
        cmd += ["--body", "@-"]
    r = subprocess.run(cmd, input=None if body is None else json.dumps(body), capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"shade {' '.join(args)}: {r.stdout.strip()} {r.stderr.strip()}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def load_session():
    if not os.path.exists(SESSION):
        sys.exit("no session: `basal-cloud.py up` first")
    return json.load(open(SESSION))


def ssh_base(s):
    return ["ssh", "-p", str(s["port"]), "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", f"{s['user']}@{s['ip']}"]


def ssh(s, command, check=True):
    return subprocess.run(ssh_base(s) + [command], check=check).returncode


def rsync(s, *paths, dest):
    e = f"ssh -p {s['port']} -o UserKnownHostsFile={KNOWN_HOSTS} -o StrictHostKeyChecking=accept-new"
    subprocess.run(["rsync", "-az", "--relative", "-e", e, *paths, f"{s['user']}@{s['ip']}:{dest}"], check=True,
                   cwd=ROOT)


def types(args):
    gpu = opt(args, "--gpu", "RTX6000Ada")
    data = shade("instances", "list-types", "--gpu-type", gpu, "--num-gpus", opt(args, "--num-gpus", "1"), "--available",
                 "--sort", "price")
    rows = []
    for t in data.get("instance_types", []):
        for a in t["availability"]:
            if a["available"] and a.get("rental_type", "on_demand") == "on_demand":
                rows.append((t["hourly_price"], t, a["region"]))
    rows.sort(key=lambda r: r[0])
    return rows


def cmd_types(args):
    for price, t, region in types(args):
        c = t["configuration"]
        print(f"${price / 100:.2f}/h  {t['cloud']:14s} {t['shade_instance_type']:16s} {region:18s} {c['num_gpus']}x "
              f"{c['gpu_type']} {c['vram_per_gpu_in_gb']} GB, {c['vcpus']} vCPU, {c['memory_in_gb']} GB RAM, "
              f"{c['storage_in_gb']} GB disk, {','.join(c['os_options'])}")


def cmd_template(args):
    body = {
        "name": TEMPLATE,
        "description": "basal-rs tests: latest release, CUDA 12.9 libraries and models in /data (tools/cloud)",
        "launch_configuration": {"type": "script", "script_configuration": {
            "base64_script": base64.b64encode(START_SCRIPT.encode()).decode()}},
        "envs": [{"name": "BASAL_PREFETCH", "value": "4.5B"}],
        "tags": [TAG],
    }
    existing = [t for t in shade("templates", "list").get("templates", []) if t.get("name") == TEMPLATE]
    if existing:
        shade("templates", "update", existing[0]["id"], body=body)
        print(f"updated template {existing[0]['id']}")
    else:
        print(f"saved template {shade('templates', 'save', body=body).get('id')}")


def template_id():
    for t in shade("templates", "list").get("templates", []):
        if t.get("name") == TEMPLATE:
            return t["id"]
    sys.exit("no template: `basal-cloud.py template` first")


def cmd_up(args):
    if os.path.exists(SESSION):
        sys.exit(f"a session exists ({SESSION}): `down` it first")
    max_price = float(opt(args, "--max-price", "1.20"))
    hours = float(opt(args, "--hours", "3"))
    spend = opt(args, "--spend", "4")
    rows = [r for r in types(args) if r[0] / 100 <= max_price]
    if not rows:
        sys.exit(f"no machine with {opt(args, '--gpu', 'RTX6000Ada')} at most ${max_price}/h available now")
    price, t, region = rows[0]
    oses = t["configuration"]["os_options"]
    os_name = next((o for o in oses if "cuda13" in o), next((o for o in oses if "cuda12.8" in o), oses[0]))
    key = next((k["id"] for k in shade("ssh-keys", "list").get("ssh_keys", []) if k["name"] == SSH_KEY_NAME), None)
    if key is None:
        sys.exit(f"no SSH key named {SSH_KEY_NAME} in Shadeform (BASAL_CLOUD_SSH_KEY)")
    until = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours)).isoformat(timespec="seconds")
    name = "basal-test-" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    body = {
        "cloud": t["cloud"], "region": region, "shade_instance_type": t["shade_instance_type"], "shade_cloud": True,
        "name": name, "os": os_name, "template_id": template_id(), "ssh_key_id": key, "tags": [TAG],
        "auto_delete": {"date_threshold": until, "spend_threshold": spend},
        "alert": {"spend_threshold": str(round(float(spend) * 0.75, 2))},
        "envs": [{"name": "BASAL_PREFETCH", "value": opt(args, "--prefetch", "4.5B")}],
    }
    iid = shade("instances", "create", body=body)["id"]
    os.makedirs(STATE_DIR, exist_ok=True)
    s = {"id": iid, "name": name, "cloud": t["cloud"], "type": t["shade_instance_type"], "price": price / 100,
         "auto_delete": until, "spend": spend}
    json.dump(s, open(SESSION, "w"), indent=1)
    print(f"created {name} ({iid}): {t['cloud']} {t['shade_instance_type']} {region}, ${price / 100:.2f}/h, {os_name}; "
          f"deleted at {until} or after ${spend}")
    while True:
        info = shade("instances", "get", iid)
        if info.get("status") == "active" and info.get("ip"):
            break
        if info.get("status") in ("error", "deleted"):
            sys.exit(f"instance {info.get('status')}: {info.get('status_details')}")
        time.sleep(20)
    s.update({"ip": info["ip"], "user": info.get("ssh_user", "shadeform"), "port": info.get("ssh_port", 22)})
    json.dump(s, open(SESSION, "w"), indent=1)
    print(f"active: {' '.join(ssh_base(s))}")
    while ssh(s, "test -f /data/ready", check=False) != 0:
        time.sleep(20)
    print("ready (/data/ready): release, CUDA libraries and models in place")


def cmd_build(args):
    host = opt(args, "--host", os.environ.get("BASAL_BUILD_HOST"))
    if not host:
        sys.exit("--host USER@HOST or BASAL_BUILD_HOST: a Docker host with buildx (no GPU needed)")
    caps = opt(args, "--caps", "80,89,90")
    sha = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    sha = (sha or "unknown") + ("-dirty" if dirty else "")
    remote = "/tmp/basal-cloud-build"
    subprocess.run(["rsync", "-az", "--delete", "--exclude", "target", "Cargo.toml", "Cargo.lock", "rustfmt.toml",
                    "crates", "tools", "LICENSE", "NOTICE", f"{host}:{remote}/"], check=True, cwd=ROOT)
    build = (f"cd {remote} && nice docker buildx build -f tools/release/linux.Dockerfile --target dev "
             f"--build-arg CUDA_COMPUTE_CAPS={caps} --build-arg BASAL_GIT_SHA={sha} "
             f"--output type=local,dest={remote}/out .")
    subprocess.run(["ssh", host, build], check=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    subprocess.run(["rsync", "-az", f"{host}:{remote}/out/basal-cuda", os.path.join(STATE_DIR, "basal-cuda")],
                   check=True)
    print(f"built {os.path.join(STATE_DIR, 'basal-cuda')} ({sha}, compute capability {caps})")


def cmd_push(args):
    s = load_session()
    e = f"ssh -p {s['port']} -o UserKnownHostsFile={KNOWN_HOSTS} -o StrictHostKeyChecking=accept-new"
    subprocess.run(["rsync", "-az", "-e", e, os.path.join(STATE_DIR, "basal-cuda"),
                    f"{s['user']}@{s['ip']}:/opt/basal-dev/basal-cuda"], check=True)
    ssh(s, "basal-dev --version && nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader")


def cmd_sync(args):
    rsync(load_session(), *args, dest="/work/")


def cmd_run(args):
    s = load_session()
    sys.exit(ssh(s, "cd /work && " + " ".join(args), check=False))


def cmd_pull(args):
    s = load_session()
    remote, local = args[0], args[1]
    e = f"ssh -p {s['port']} -o UserKnownHostsFile={KNOWN_HOSTS} -o StrictHostKeyChecking=accept-new"
    subprocess.run(["rsync", "-az", "-e", e, f"{s['user']}@{s['ip']}:/work/{remote}", os.path.join(ROOT, local)],
                   check=True)


def cmd_ssh(args):
    print(" ".join(shlex.quote(a) for a in ssh_base(load_session())))


def cmd_ls(args):
    for i in shade("instances", "list").get("instances", []):
        if TAG in (i.get("tags") or []) or i.get("name", "").startswith("basal-test-"):
            print(i["id"], i.get("name"), i.get("status"), i.get("cloud"), i.get("shade_instance_type"), i.get("ip"),
                  f"${i.get('hourly_price', 0) / 100:.2f}/h", i.get("created_at"))


def cmd_down(args):
    ids = []
    if "--all" in args:
        ids = [i["id"] for i in shade("instances", "list").get("instances", [])
               if TAG in (i.get("tags") or []) or i.get("name", "").startswith("basal-test-")]
    elif args:
        ids = [args[0]]
    else:
        ids = [load_session()["id"]]
    for i in ids:
        shade("instances", "delete", i)
        print(f"deleted {i}")
    if os.path.exists(SESSION) and (not args or "--all" in args or load_session()["id"] in ids):
        os.remove(SESSION)


COMMANDS = {"types": cmd_types, "template": cmd_template, "up": cmd_up, "build": cmd_build, "push": cmd_push,
            "sync": cmd_sync, "run": cmd_run, "pull": cmd_pull, "ssh": cmd_ssh, "ls": cmd_ls, "down": cmd_down}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]](sys.argv[2:])
