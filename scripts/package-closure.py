#!/usr/bin/env python3
"""Local-only relative-path research handoff. Never pushes or uploads.

The Git bundle preserves baseline history; a separately hashed working-tree
overlay supplies the authoritative uncommitted closure source and evidence.
Large runtime is split into three archives. A relocated audit can restore
the entire packet without consulting original absolute artifact locations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final-closure"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def files_under(path):
    return [p for p in path.rglob("*") if p.is_file() and
            not any(x in p.parts for x in ("__pycache__", "pdf-review", "paper-review", ".pytest_cache")) and
            not p.name.endswith((".tar.gz", ".zip", ".bundle", ".tmp", ".pyc")) and
            p.name != ".generation.lock"]


def archive(job):
    target, paths = job
    target = Path(target)
    tick = time.perf_counter()
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT, dereference=True) as tar:
                for name in paths:
                    path = ROOT / name
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as f:
                        tar.addfile(info, f)
    return dict(path=str(target), file=target.name, sha256=sha(target), bytes=target.stat().st_size,
                files=len(paths), seconds=time.perf_counter()-tick)


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=True)
    plan = destination / "PACKET.json"
    if plan.exists():
        raise SystemExit("Packet already exists; use a fresh output directory rather than overwrite it.")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    bundle = destination / "baseline-source.bundle"
    subprocess.run(["git", "-c", "pack.threads=2", "bundle", "create", str(bundle), branch], cwd=ROOT, check=True)
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=ROOT, check=True, capture_output=True)
    print("Baseline Git history bundled and verified", flush=True)
    degree = files_under(ROOT / "results/multiteacher-degree-generalization")
    cf = files_under(ROOT / "results/multiteacher-counterfactual")
    native = files_under(ROOT / "results/sample-complexity") + files_under(ROOT / "data/sample-complexity-v1")
    cert = json.loads((OUT / "mathematical-certificate-audit.json").read_text())
    native += [ROOT / r["checkpoint"] for r in cert["historical"]]
    native += [ROOT / r["certificate"] for r in cert["historical"]]
    source = []
    for directory in ("src", "scripts", "experiments", "tests", "configs", "docs", "paper", "requirements"):
        if (ROOT / directory).exists():
            source += files_under(ROOT / directory)
    source += [ROOT / name for name in ("README.md", "pyproject.toml", ".gitignore")]
    source += [p for p in files_under(OUT) if p.name not in ("delivery.json", "artifact-manifest.json", "relocated-recovery.json")]
    groups = {"degree-runtime.tar.gz": degree, "counterfactual-runtime.tar.gz": cf,
              "native-runtime.tar.gz": native, "closure-source-overlay.tar.gz": source}
    paths_by_group = {name: sorted({str(p.relative_to(ROOT)) for p in paths}) for name, paths in groups.items()}
    manifest = {}
    for names in paths_by_group.values():
        for name in names:
            p = ROOT / name
            manifest[name] = dict(sha256=sha(p), bytes=p.stat().st_size)
    write(destination / "INPUTS.json", manifest)
    print("Packet inputs authenticated:", len(manifest), "files", sum(r["bytes"] for r in manifest.values()), "bytes", flush=True)
    jobs = [(str(destination / name), paths) for name, paths in paths_by_group.items()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        artifacts = list(pool.map(archive, jobs))
    versions = subprocess.check_output([str(ROOT / ".venv/bin/python"), "-m", "pip", "freeze"], cwd=ROOT, text=True)
    (destination / "environment.txt").write_text("\n".join(line for line in versions.splitlines() if not line.startswith("-e ")) + "\n")
    packet = dict(date="2026-10-26", source_baseline_commit=head, branch=branch,
                  source_state="Baseline Git bundle plus authenticated current working-tree overlay; closure edits are not represented as a fabricated baseline commit.",
                  bundle=dict(file=bundle.name, sha256=sha(bundle), bytes=bundle.stat().st_size),
                  inputs=dict(file="INPUTS.json", sha256=sha(destination / "INPUTS.json"), files=len(manifest)),
                  archives=artifacts, environment_sha256=sha(destination / "environment.txt"),
                  script_sha256=sha(__file__), published=False, pushed=False, new_training_updates=0)
    write(plan, packet)
    lines = [f"{packet['bundle']['sha256']}  {bundle.name}", f"{packet['inputs']['sha256']}  INPUTS.json",
             f"{sha(plan)}  PACKET.json", f"{sha(destination/'environment.txt')}  environment.txt"]
    lines += [f"{r['sha256']}  {r['file']}" for r in artifacts]
    (destination / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    (destination / "README.md").write_text(f"""# Local scientific closure packet

This packet is local only. No push, upload or release was made.
The Git bundle authenticates baseline history at {head}; the working-tree
overlay contains the edited closure source and evidence. All member paths
are relative. INPUTS.json authenticates the exact pre-recovery snapshot.

```bash
sha256sum -c SHA256SUMS
git clone --branch {branch} baseline-source.bundle recovered
tar -xzf degree-runtime.tar.gz -C recovered
tar -xzf counterfactual-runtime.tar.gz -C recovered
tar -xzf native-runtime.tar.gz -C recovered
tar -xzf closure-source-overlay.tar.gz -C recovered
cd recovered
python -m venv .venv
# Reproduce the tested environment using environment.txt, then install this
# package with: .venv/bin/pip install -e '.[dev,study,paper]'
PYTHONPATH=src .venv/bin/python scripts/final-evidence-audit.py
PYTHONPATH=src .venv/bin/python scripts/final-native-audit.py
PYTHONPATH=src .venv/bin/python scripts/final-counterfactual-audit.py
PYTHONPATH=src .venv/bin/python scripts/final-certificate-audit.py
PYTHONPATH=src .venv/bin/python scripts/verify-multicf.py
PYTHONPATH=src .venv/bin/python scripts/verify-sample-complexity.py --runtime
```

Original absolute source paths remain as provenance inside historical records;
the closure audits use the restored relative runtime. No missing Pedro weights
are imputed. Byte-for-byte training on arbitrary hardware is not promised.
The supplied relocated smoke uses an unchanged shared Python environment;
it verifies relocation/source/runtime, not an independent dependency install.
The packet is larger than typical venue supplements and requires a separate
approved artifact channel for external distribution. Current code, weights,
raw rows and negative candidates are present, not just summaries.
""")
    write(OUT / "delivery.json", dict(local_directory=str(destination), packet_sha256=sha(plan),
          packet=packet, checksum_file=str(destination / "SHA256SUMS"),
          availability="Local packet, no upload or publication", recovery_pending=True))
    print(json.dumps(dict(directory=str(destination), files=len(manifest), archives=artifacts)), flush=True)


def restore_and_verify(destination):
    packet = json.loads((destination / "PACKET.json").read_text())
    assert sha(destination / packet["inputs"]["file"]) == packet["inputs"]["sha256"]
    assert sha(destination / packet["bundle"]["file"]) == packet["bundle"]["sha256"]
    restored = destination / "recovered"
    if restored.exists():
        raise SystemExit("Recovery destination already exists; refusing to replace it.")
    subprocess.run(["git", "clone", "--quiet", "--branch", packet["branch"], str(destination / packet["bundle"]["file"]), str(restored)], check=True)
    for item in packet["archives"]:
        archive_path = destination / item["file"]
        assert sha(archive_path) == item["sha256"]
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                p = Path(member.name)
                assert not p.is_absolute() and ".." not in p.parts
                assert member.isfile(), "Packet contains only regular files"
            tar.extractall(restored, members=members)
    manifest = json.loads((destination / "INPUTS.json").read_text())
    for name, meta in manifest.items():
        assert sha(restored / name) == meta["sha256"], name
    print("Restored every packet input in a separate directory:", len(manifest), flush=True)
    env = dict(os.environ, PYTHONPATH="src", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    python = str(ROOT / ".venv/bin/python")
    commands = [
        ["scripts/final-evidence-audit.py"], ["scripts/final-native-audit.py"],
        ["scripts/final-counterfactual-audit.py"], ["docs/reviews/REVIEW_2_BE_STRATA.py"],
        ["scripts/final-certificate-audit.py"], ["scripts/verify-multicf.py"],
        ["scripts/verify-sample-complexity.py", "--runtime"],
    ]
    runs = []
    for i, command in enumerate(commands):
        tick = time.perf_counter()
        result = subprocess.run([python] + command, cwd=restored, env=env, text=True, capture_output=True)
        log = destination / f"recovery-{i+1:02d}.txt"
        log.write_text(result.stdout + result.stderr)
        runs.append(dict(command=[python]+command, returncode=result.returncode, log=str(log),
                         log_sha256=sha(log), seconds=time.perf_counter()-tick))
        print(command, "returncode", result.returncode, "seconds", round(runs[-1]["seconds"], 2), flush=True)
        if result.returncode:
            write(destination / "RECOVERY.json", dict(passed=False, runs=runs, restored=str(restored)))
            raise SystemExit(f"Recovery failed; inspect {log}")
    g = json.loads((restored / "results/final-closure/generalization-audit.json").read_text())
    n = json.loads((restored / "results/final-closure/native-certifications.json").read_text())
    c = json.loads((restored / "results/final-closure/counterfactual-audit.json").read_text())
    assert g["stages"]["confirmatory"]["successes"] == 808413
    assert n["successes"] == 105000
    assert sum(r["correct"] for r in c["rows"] if r["method"] == "counterfactual") == 499166
    receipt = dict(passed=True, restored=str(restored), packet_sha256=sha(destination / "PACKET.json"),
                   authenticated_files=len(manifest), runs=runs, new_training_updates=0,
                   environment_scope="Separate source/runtime directory; same Python environment, not an independent clean dependency install.",
                   original_absolute_runtime_paths_not_required=True)
    write(destination / "RECOVERY.json", receipt)
    write(OUT / "relocated-recovery.json", receipt)
    delivery = json.loads((OUT / "delivery.json").read_text())
    delivery["recovery_pending"] = False
    delivery["recovery"] = dict(passed=True, receipt_sha256=sha(destination / "RECOVERY.json"), receipt=str(destination / "RECOVERY.json"))
    write(OUT / "delivery.json", delivery)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "recover"))
    parser.add_argument("--destination", type=Path, default=ROOT.parent / "closure-delivery-20261026")
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else restore_and_verify)(args.destination.resolve())
