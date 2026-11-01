#!/usr/bin/env python3
"""Package seven original transfer weights and hash the integrated delivery."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/integrated-paper-20261031"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime():
    freeze = json.loads((OUT / "transfer-replay-freeze.json").read_text())
    files = [ROOT/p for p in freeze["inputs"] if p.startswith("data/coauthor-integration/checkpoints/")]
    records = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    index = OUT / "integration-runtime-index.json"
    index.write_text(json.dumps(dict(scope="Seven original weights for the15 failed historical cells; no new weights",
        files=records),indent=2)+"\n")
    archive = OUT / "integration-replay-runtime.tar.gz"
    if not archive.exists():
        with archive.open("wb") as raw:
            with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
                with tarfile.open(mode="w",fileobj=gz,format=tarfile.PAX_FORMAT) as tf:
                    for path in sorted(files):
                        data = path.read_bytes()
                        info = tarfile.TarInfo(str(path.relative_to(ROOT)))
                        info.size, info.mtime, info.mode = len(data), 0, 0o644
                        tf.addfile(info,io.BytesIO(data))
    with tarfile.open(archive) as tf:
        assert len(tf.getmembers()) == len(records)
        for row in records:
            assert hashlib.sha256(tf.extractfile(row["path"]).read()).hexdigest() == row["sha256"]
    (OUT / "integration-runtime-receipt.json").write_text(json.dumps(dict(path=str(archive.relative_to(ROOT)),
        bytes=archive.stat().st_size,sha256=sha(archive),index_sha256=sha(index),members=len(records)),indent=2)+"\n")


def manifest():
    sources = json.loads((OUT/"sources.json").read_text())
    build = json.loads((OUT/"paper-build.json").read_text())
    old = json.loads((ROOT/"results/paper-revision-20261030/manifest.json").read_text())
    selected = {ROOT/p for p in sources["imported_sha256"]}
    selected |= {ROOT/p for p in build["actual_inputs"]}
    selected |= {ROOT/r["path"] for r in old["files"]}
    selected |= {ROOT/p for p in ("README.md","paper/ucgnn/README.md","paper/ucgnn/main.pdf",
        "paper/ucgnn/archive/pre-integration-5993a5e4.tex","paper/ucgnn/archive/pre-integration-5993a5e4.pdf",
        "results/paper-revision-20261030/manifest.json","results/final-closure/native-certifications.json",
        "results/final-closure/generalization-cells.csv","results/multiteacher-degree-generalization/confirmatory/design.json",
        "results/multiteacher-degree-generalization/checkpoint-inventory.json",
        "results/multiteacher-degree-generalization/supplemental-checkpoint-inventory.json",
        "requirements/integrated-paper-tested.txt")}
    selected |= set((ROOT/"src/dcl").rglob("*.py"))
    selected |= set((ROOT/"scripts").glob("*coauthor*.py")) | set((ROOT/"scripts").glob("integrated-*.py"))
    selected |= set((ROOT/"tests").glob("test_coauthor*.py")) | set((ROOT/"tests").glob("test_integration_support*.py"))
    selected |= set((ROOT/"docs").glob("*INTEGRAT*.md")) | {ROOT/"docs/COAUTHOR_RECONCILIATION.md"}
    selected |= set((ROOT/"docs").glob("CONTINUATION_*RECONCILIATION.md")) | {ROOT/"docs/CONTINUATION_PROGRESS_THEORY_AUDIT.md"}
    selected |= set((ROOT/"docs").glob("MARTIN_ARTIFACT*.md"))
    selected.add(ROOT/"scripts/replay-release-artifacts.py")
    selected |= {p for p in OUT.rglob("*") if p.is_file() and p.name not in ("manifest.json","verification.json")
                 and "fresh-replay" not in str(p.relative_to(OUT))}
    release=ROOT/"results/martin-artifacts-20261031"
    selected |= {p for p in release.rglob("*") if p.is_file()
                 and "fresh-replay" not in str(p.relative_to(release))
                 and not (p.parent==release/"downloads" and p.name.endswith(".tar.gz"))}
    selected |= {p for p in (ROOT/"results/paper-editorial-cleanup-20261101").rglob("*") if p.is_file()}
    rows=[]
    for p in sorted(selected):
        if not p.is_file():
            raise FileNotFoundError(p)
        rows.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),version="integrated-paper-20261031",
        pedro_commit=sources["pedro_commit"],martin_commit=sources["martin_commit"],common_base=sources["common_base"],
        scope="Local coauthor-review delivery; no publication or shared-branch merge",files=rows,
        deliberate_import_edits={"paper/ucgnn/closure-references.bib":"Retain verified arXiv version; unconfirmed ICML oral metadata removed"},
        external_runtime="Three published archives pinned in results/martin-artifacts-20261031/download-receipt.json; member-index.json authenticates all1988 restored members. Large downloaded archives and restored scientific payloads are available via the release and are not duplicated in this compact manifest.",
        exclusions="Fresh optional replay directories; verification.json authenticates this manifest separately. The separate240000-run contrast's240 original raw files remain undelivered and outside all three archives. Original integrated delivery is preserved in the release audit's prior-integration.tar.gz.")
    (OUT/"manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(dict(files=len(rows),manifest_sha256=sha(OUT/"manifest.json"))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-only",action="store_true")
    args=p.parse_args()
    if not args.manifest_only: runtime()
    manifest()


if __name__ == "__main__":
    main()
