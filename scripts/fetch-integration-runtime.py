#!/usr/bin/env python3
"""Verify a pinned runtime release and safely restore missing identical files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def restore(archive, destination, release):
    pin = release["runtime"]
    if archive.stat().st_size != pin["bytes"] or digest(archive) != pin["sha256"]:
        raise ValueError("Runtime archive does not match the pinned size/hash.")
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len({m.name for m in members}) != len(members) or not all(m.isfile() for m in members):
            raise ValueError("Duplicate or nonregular archive member.")
        manifest_member = tar.getmember(pin["manifest"])
        payload = tar.extractfile(manifest_member).read()
        if hashlib.sha256(payload).hexdigest() != pin["manifest_sha256"]:
            raise ValueError("Runtime manifest does not match its pin.")
        manifest = json.loads(payload)
        expected = {r["path"]: r for r in manifest["files"]}
        if len(expected) != len(manifest["files"]) or {m.name for m in members} != {*expected, pin["manifest"]}:
            raise ValueError("Archive member set differs from the runtime manifest.")
        # Validate every byte and all destinations before writing anything.
        missing = []
        for name, record in expected.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("data", "integration-balanced"):
                raise ValueError("Unsafe runtime path.")
            target = destination / name
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Destination escapes the requested root.")
            if target.is_symlink() or any(x.is_symlink() for x in target.parents if x != destination.parent):
                raise ValueError("Symlink in runtime destination.")
            member = tar.getmember(name)
            with tar.extractfile(member) as source:
                sha = hashlib.file_digest(source, "sha256").hexdigest()
            if member.size != record["bytes"] or sha != record["sha256"]:
                raise ValueError("Runtime member checksum mismatch.")
            if target.exists():
                if not target.is_file() or digest(target) != sha:
                    raise ValueError(f"Existing file differs; refusing to replace {name}.")
            else:
                missing.append((member, target))
        for member, target in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest)
    return dict(verified_files=len(expected), restored_files=len(missing), source_commit=manifest["source_commit"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path)
    p.add_argument("--destination", type=Path, default=ROOT)
    args = p.parse_args()
    release = json.loads((ROOT / "docs/integration/INTEGRATION_RELEASE.json").read_text())
    if args.archive:
        result = restore(args.archive.resolve(), args.destination, release)
    else:
        with tempfile.TemporaryDirectory(prefix="integration-runtime-") as tmp:
            pin = release["runtime"]; archive = Path(tmp) / pin["name"]
            try:
                request = urllib.request.Request(pin["url"], headers={"User-Agent": "scientific-integration-reproduction"})
                with urllib.request.urlopen(request, timeout=60) as src, archive.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            except Exception:
                if not shutil.which("gh"):
                    raise
                archive.unlink(missing_ok=True)
                subprocess.run(["gh", "release", "download", release["tag"], "--repo", release["repository"],
                                "--pattern", pin["name"], "--dir", tmp], check=True)
            result = restore(archive, args.destination, release)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
