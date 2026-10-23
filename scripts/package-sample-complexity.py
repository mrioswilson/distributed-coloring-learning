#!/usr/bin/env python3
"""Hash and archive the complete new runtime without historical checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

root=Path("data/sample-complexity-v1")
out=Path("results/sample-complexity")
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,default=Path("sample-complexity-native-v1-runtime.tar.gz"))
target=parser.parse_args().output
if not (out/"certification.json").exists(): raise SystemExit("Complete certification outcomes required")
files=sorted(p for p in root.rglob("*") if p.is_file())
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
entries=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
manifest=dict(root=str(root),files=entries,bytes=sum(e["bytes"] for e in entries),
              checkpoints=sum(p.suffix==".pt" for p in files),
              preregistration_sha256=sha(out/"preregistration.json"),
              note="All new teacher caches, final/negative checkpoints, optimizer/RNG states and development rows. Historical 70 checkpoints excluded.")
(out/"runtime-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
target.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(target,"w:gz",compresslevel=1) as archive:
    for p in files: archive.add(p,arcname=str(p),recursive=False)
release=dict(archive=target.name,bytes=target.stat().st_size,sha256=sha(target),
             files=len(files),checkpoints=manifest["checkpoints"],
             url="https://github.com/mrioswilson/distributed-coloring-learning/releases/download/sample-complexity-2026-10-23/"+target.name)
(out/"runtime-release.json").write_text(json.dumps(release,indent=2)+"\n")
print(json.dumps(release,indent=2))
