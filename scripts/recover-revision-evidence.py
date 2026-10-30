#!/usr/bin/env python3
"""Restore public evidence by pinned SHA256; preserve all existing metadata."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

SPECS = (
    ("cf", "multiteacher-counterfactual-2026-10-24", "Evidencia_reproducible.tar.gz",
     "a00452f5802b0bc67f254e86e7220a38e60e83094264d2b45cdc5a6772c61b3c", 656423144),
    ("native", "sample-complexity-2026-10-23", "sample-complexity-native-v1-runtime.tar.gz",
     "53e7755035910e28dc16f1448a5b697e88fa188a0e18ed4db74368b1b3d92ce7", 1145842922),
    ("selected", "research-recovery-2026-10-23", "ucgnn-selected-checkpoints.tar.gz",
     "c7e78cdeda6c36a5db32d6b62d34429c502b631e40e6f9bdf8f3b30cb10bffe6", 25981372),
)


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args();args.cache.mkdir(parents=True,exist_ok=True)
    index={};reports=[]
    for kind,tag,name,digest,size in SPECS:
        archive=args.cache/name
        if not archive.exists():
            url=f"https://github.com/mrioswilson/distributed-coloring-learning/releases/download/{tag}/{name}"
            partial=archive.with_suffix(archive.suffix+".partial")
            with urllib.request.urlopen(url,timeout=60) as src,partial.open("wb") as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            if partial.stat().st_size!=size or sha(partial)!=digest:raise RuntimeError("Download identity mismatch")
            partial.rename(archive)
        if archive.stat().st_size!=size or sha(archive)!=digest:raise RuntimeError(f"Archive mismatch: {archive}")
        extracted=args.cache/(kind+"-verified")
        with tarfile.open(archive) as tar:
            members=[]
            for member in tar:
                path=PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not member.isfile():
                    raise RuntimeError("Unsafe/unexpected archive entry")
                if kind=="cf" and member.name.startswith("results/multiteacher-counterfactual/"):
                    members.append(member)
                elif kind=="native" and member.name.endswith(".pt"):
                    members.append(member)
                elif kind=="selected" and member.name.endswith(".pt"):
                    members.append(member)
            # Re-extract from the authenticated archive, rather than trusting a stale cache.
            tar.extractall(extracted,members=members,filter="data")
        linked=0;preserved=[]
        if kind=="cf":
            manifest=json.loads((extracted/"results/multiteacher-counterfactual/RUNTIME_MANIFEST.json").read_text())
            for entry in manifest["files"]:
                src=extracted/entry["path"]
                if sha(src)!=entry["sha256"]:raise RuntimeError("Runtime member mismatch")
                dst=Path(entry["path"])
                if dst.exists():
                    if sha(dst)!=entry["sha256"]:
                        published=dst.read_bytes();normalization=hashlib.sha256(published.replace(b"\n",b"\r\n")).hexdigest()==entry["sha256"]
                        preserved.append(dict(path=str(dst),published_sha256=sha(dst),release_sha256=entry["sha256"],lf_to_crlf_only=normalization))
                else:
                    dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src.resolve());linked+=1
        for cp in extracted.rglob("*.pt"):index.setdefault(sha(cp),[]).append(str(cp.resolve()))
        reports.append(dict(kind=kind,archive_sha256=digest,bytes=size,linked=linked,preserved_differences=preserved))
    # Historical palette metadata retains original paths; materialize only exact matches.
    for rec in json.loads(Path("results/ucgnn-local-palettes/frozen-checkpoints.json").read_text()):
        digest=rec["sha256"]
        if digest not in index:raise RuntimeError("Selected historical checkpoint missing")
        parts=Path(rec["checkpoint"]).parts
        path=Path(*parts[parts.index("data"):])
        if path.exists():
            if sha(path)!=digest:raise RuntimeError("Refusing to overwrite historical weight")
        else:path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(index[digest][0])
    out=Path("results/paper-revision-20261030");out.mkdir(exist_ok=True)
    (out/"recovered-checkpoints.json").write_text(json.dumps(index,indent=2)+"\n")
    (out/"public-recovery.json").write_text(json.dumps(reports,indent=2)+"\n")
    print("Verified",len(index),"distinct checkpoint identities; original result metadata preserved")


if __name__=="__main__":main()
