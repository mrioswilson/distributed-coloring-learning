#!/usr/bin/env python3
"""Install the pinned official Linux compiler after archive/binary verification."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TAG = "tectonic@0.17.0"
ASSET = "tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz"
ARCHIVE_SHA = "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7"
BINARY_SHA = "a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da"


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / ".tools/tectonic")
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This pin is Linux x86-64 only; supply an appropriate compiler to the paper build.")
    target = args.output.resolve()
    if target.exists():
        if digest(target) != BINARY_SHA:
            raise SystemExit("Existing compiler differs; refusing to replace it.")
    else:
        with tempfile.TemporaryDirectory(prefix="integration-tectonic-") as tmp:
            archive = Path(tmp) / ASSET
            url = f"https://github.com/tectonic-typesetting/tectonic/releases/download/{TAG}/{ASSET}"
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "scientific-integration-reproduction"})
                with urllib.request.urlopen(request, timeout=60) as src, archive.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            except Exception:
                if not shutil.which("gh"):
                    raise
                archive.unlink(missing_ok=True)
                subprocess.run(["gh", "release", "download", TAG, "--repo", "tectonic-typesetting/tectonic",
                                "--pattern", ASSET, "--dir", tmp], check=True)
            if digest(archive) != ARCHIVE_SHA:
                raise SystemExit("Official archive checksum mismatch.")
            binary = Path(tmp) / "tectonic"
            with tarfile.open(archive, "r:gz") as tar:
                matches = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "tectonic"]
                if len(matches) != 1:
                    raise SystemExit("Expected one compiler in the official archive.")
                with tar.extractfile(matches[0]) as src, binary.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if digest(binary) != BINARY_SHA:
                raise SystemExit("Compiler checksum mismatch.")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(binary, target)
            target.chmod(0o755)
    version = subprocess.check_output([str(target), "--version"], text=True).strip()
    print(json.dumps(dict(compiler=str(target), sha256=digest(target), version=version)))


if __name__ == "__main__":
    main()
