#!/usr/bin/env python3
"""Recover the three author-supplied archives by immutable SHA256 pins.

Downloads only. Extraction/validation is a separate inspectable step.
No training, result replacement, Git mutation or external publication.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/martin-artifacts-20261031"
BASE="https://github.com/mrioswilson/distributed-coloring-learning/releases/download/pedrorev-experimental-data-2026-10-31/"
PINS={"pedrorev-cf":"8353b7135cefad7159832ad0b4df901a6e0cd6168e57b0872a49d041d84fe2cc",
      "pedrorev-cf-followup":"6a009a4abd5853b9e7259e82d3ee458bdbbff7a6720c99b1eada5559a67773bb",
      "confirmatory-rows":"ed1039ac91f572eeb4c0e5e5b06e630d08c6d407a24aceb28b14fd57233ac489"}


def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def fetch(name):
    directory=OUT/"downloads";directory.mkdir(parents=True,exist_ok=True)
    dest=directory/name
    if dest.exists():return dest
    req=urllib.request.Request(BASE+name,headers={"User-Agent":"DCL-scientific-artifact-audit"})
    tmp=dest.with_suffix(dest.suffix+".partial")
    with urllib.request.urlopen(req,timeout=120) as source,tmp.open("wb") as target:
        while block:=source.read(1024*1024):target.write(block)
    tmp.replace(dest)
    return dest


def main():
    rows=[]
    for name,digest in PINS.items():
        for ext in (".README.md",".SHA256SUMS",".tar.gz"):
            p=fetch(name+ext)
            if ext==".tar.gz":assert sha(p)==digest,(name,sha(p))
            rows.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),url=BASE+p.name))
            print(json.dumps(rows[-1]),flush=True)
        assert digest in (OUT/"downloads"/(name+".SHA256SUMS")).read_text()
    receipt=OUT/"download-receipt.json"
    if receipt.exists():
        assert json.loads(receipt.read_text())["files"]==rows,"Existing receipt differs; preserve it for investigation"
    else:
        receipt.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
            scope="Three published archive hashes independently verified against the user's pins",files=rows),indent=2)+"\n")


if __name__=="__main__":main()
