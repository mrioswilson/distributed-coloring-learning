#!/usr/bin/env python3
"""Download, verify, and safely restore the released query-budgeted runtime."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

out=Path("results/sample-complexity")
release=json.loads((out/"runtime-release.json").read_text())
manifest=json.loads((out/"runtime-manifest.json").read_text())
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
expected={e["path"]:e for e in manifest["files"]}
with tempfile.TemporaryDirectory(prefix="sample-complexity-") as temporary:
    temporary=Path(temporary);archive=temporary/release["archive"]
    urllib.request.urlretrieve(release["url"],archive)
    if archive.stat().st_size!=release["bytes"] or sha(archive)!=release["sha256"]:
        raise RuntimeError("Runtime archive digest/length mismatch")
    stage=temporary/"extracted";stage.mkdir()
    with tarfile.open(archive,"r:gz") as tar:
        members=tar.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected):
            raise RuntimeError("Archive members differ from manifest")
        for m in members:
            path=PurePosixPath(m.name)
            if not m.isfile() or path.is_absolute() or ".." in path.parts or not m.name.startswith("data/sample-complexity-v1/"):
                raise RuntimeError("Unsafe archive member")
            if m.size!=expected[m.name]["bytes"]: raise RuntimeError("Member length mismatch")
            destination=stage/m.name;destination.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as source,destination.open("wb") as target: shutil.copyfileobj(source,target)
            if sha(destination)!=expected[m.name]["sha256"]: raise RuntimeError("Member digest mismatch")
    for name,e in expected.items():
        destination=Path(name)
        if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
            raise RuntimeError("Refusing symlink destination")
        if destination.exists() and sha(destination)!=e["sha256"]: raise RuntimeError(f"Refusing to overwrite different file: {destination}")
    for name in expected:
        destination=Path(name);destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists(): shutil.copyfile(stage/name,destination)
print(f"Verified and restored {len(expected)} runtime files")
