#!/usr/bin/env python3
"""Durable portable runtime plus a manifest of actual delivered bytes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

OUT=Path("results/paper-revision-20261030")
DEST=Path("data/paper-revision-20261030")


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runtime",type=Path,default=DEST)
    parser.add_argument("--manifest-only",action="store_true",help="Refresh delivered file hashes without rewriting runtime or scientific receipts")
    args=parser.parse_args()
    assert len(list((OUT/"prospective").rglob("*-test.json")))==40
    assert json.loads((OUT/"verification.json").read_text())["verified"]
    if args.manifest_only:
        index=json.loads((OUT/"runtime-index.json").read_text())
        assert sha(Path(index["archive"]["path"]))==index["archive"]["sha256"]
        return write_manifest(index)
    if not DEST.exists():
        shutil.copytree(args.runtime,DEST)
    files=[]
    for path in sorted(args.runtime.rglob("*")):
        if not path.is_file():continue
        relative=path.relative_to(args.runtime);destination=DEST/relative;digest=sha(path)
        if not destination.is_file() or sha(destination)!=digest:raise RuntimeError("Durable runtime differs")
        files.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=digest))
    records=[json.loads(p.read_text()) for p in sorted(DEST.glob("models/*/s*/*/record.json"))]
    assert len(records)==40
    (OUT/"training-records.json").write_text(json.dumps(records,indent=2)+"\n")
    archive=OUT/"paper-revision-runtime.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=6) as tar:
        for entry in files:tar.add(DEST/entry["path"],arcname=str(DEST/entry["path"]),recursive=False)
    index=dict(runtime_directory=str(DEST),files=files,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))
    (OUT/"runtime-index.json").write_text(json.dumps(index,indent=2)+"\n")
    write_manifest(index)


def write_manifest(index):
    paths={Path("README.md"),Path(".gitignore"),Path("src/dcl/artifacts.py"),
           Path("src/dcl/localpalettes/config.py"),Path("src/dcl/sizedegree/common.py"),
           Path("tests/test_globaldist_integration.py"),Path("tests/test_artifacts.py"),Path("tests/test_paperrevision.py"),
           Path("configs/paper-revision-20261030.json")}
    paths.update(Path("src/dcl/paperrevision").glob("*.py"))
    paths.update(Path("docs").glob("PAPER_REVISION_*.md"))
    for name in ("recover-revision-evidence.py","reanalyse-paper-be.py","probe-paper-rescues.py",
                 "summarize-paper-revision.py","audit-revision-continuations.py","probe-revision-be-failures.py","build-revision-assets.py","build-revision-transfer-figure.py","build-revised-paper.py",
                 "build-closure-paper.py",
                 "build-revision-mechanism-figure.py",
                 "verify-paper-revision.py","replay-revision-test.py","package-paper-revision.py"):
        paths.add(Path("scripts")/name)
    paper=Path("paper/ucgnn");paths.update(paper.glob("*.tex"));paths.update(paper.glob("*.bib"));paths.add(paper/"main.pdf")
    paths.add(paper/"README.md");paths.add(paper/"figures/closure-generalization.pdf")
    # Preserve removed plots as historical assets, without inserting them in the article.
    paths.update(paper/"figures"/name for name in ("revision-prospective.pdf", "closure-generalization-compact.pdf"))
    mechanism_inputs=json.loads((OUT/"failure-mechanisms.json").read_text())["inputs"]
    paths.update(Path(p) for p in mechanism_inputs)
    paths.update((paper/"archive").glob("pre-revision-c3affd16.*"));paths.update((paper/"tables").glob("*.tex"))
    for p in list(paper.glob("*.tex")):
        for figure in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",p.read_text()):paths.add(paper/figure)
    for p in OUT.rglob("*"):
        if p.is_file() and "pdf-review" not in p.parts and p.name not in ("manifest.json","recovered-checkpoints.json"):
            paths.add(p)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),base_commit="c3affd16ac5654c6e2a700d2312710753adb9518",
        branch="codex/paper-revision-20261030",files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)],
        byte_policy="SHA256 of actual delivered bytes, with no hidden newline conversion. Historical manifests are preserved as earlier receipts.",
        remaining_limits=["Independent human scientific review pending","810000 original closure raw rows unavailable; original totals remain attributed"])
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(manifest_files=len(manifest["files"]),runtime_files=len(index["files"]),archive=index["archive"]),indent=2))


if __name__=="__main__":main()
