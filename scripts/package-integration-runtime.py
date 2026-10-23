#!/usr/bin/env python3
"""Package the complete new runtime without historical/private directories."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "ucgnn-scientific-integration-runtime.tar.gz"
MANIFEST = "INTEGRATION_RUNTIME_MANIFEST.json"
REPO = "mrioswilson/distributed-coloring-learning"
TAG = "scientific-integration-2026-10-23"


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT.parent / "integration-delivery")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "configs", "scripts"], check=True)
    paths = sorted(x for x in Path("data/integration-balanced").rglob("*") if x.is_file())
    assert len([x for x in paths if x.name == "final.pt"]) == 70
    assert len([x for x in paths if x.suffix == ".npz" and "banks" in x.parts]) == 385
    assert paths and all(not x.is_symlink() for x in paths)
    records = [dict(path=str(x), bytes=x.stat().st_size, sha256=digest(x)) for x in paths]
    manifest = dict(schema="integration-runtime-v1", source_commit=commit, repository=REPO,
                    protocol="integration-teacher-palette-v1", models=70, raw_banks=385,
                    scope="Entire new integration runtime: final weights, teacher pools, histories, and all output/status banks.",
                    files=records)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (args.output / MANIFEST).write_bytes(payload)
    archive = args.output / ARCHIVE
    with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            info = tarfile.TarInfo(MANIFEST); info.size = len(payload); info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
            for record, path in zip(records, paths):
                assert digest(path) == record["sha256"]
                info = tar.gettarinfo(str(path), arcname=str(path))
                info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0; info.mode = 0o644
                with path.open("rb") as f:
                    tar.addfile(info, f)
                assert digest(path) == record["sha256"]
    release = dict(repository=REPO, tag=TAG, scientific_artifact_commit=commit,
                   release_url=f"https://github.com/{REPO}/releases/tag/{TAG}",
                   runtime=dict(name=ARCHIVE, bytes=archive.stat().st_size, sha256=digest(archive),
                                url=f"https://github.com/{REPO}/releases/download/{TAG}/{ARCHIVE}",
                                manifest=MANIFEST, manifest_sha256=hashlib.sha256(payload).hexdigest()),
                   remote_verified=False)
    Path("docs/integration/INTEGRATION_RELEASE.json").write_text(json.dumps(release, indent=2)+"\n")
    print(json.dumps(dict(archive=str(archive), files=len(paths), bytes=archive.stat().st_size,
                          sha256=release["runtime"]["sha256"], source_commit=commit)))


if __name__ == "__main__":
    main()
