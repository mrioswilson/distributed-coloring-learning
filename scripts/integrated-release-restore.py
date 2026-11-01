#!/usr/bin/env python3
"""Verify every archive member and restore only scientific payloads safely.

Package README/provenance files are kept in a namespaced metadata directory,
so extracting several archives never overwrites the project's README or a
different package's provenance. Existing different files are preserved.
"""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from importlib.util import spec_from_file_location,module_from_spec

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/martin-artifacts-20261031"


def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def preserve_prior():
    target=OUT/"prior-integration.tar.gz"
    if target.exists():return
    manifest=ROOT/"results/integrated-paper-20261031/manifest.json"
    original=json.loads(manifest.read_text())
    paths={r["path"]:r for r in original["files"]}
    for name in ("results/integrated-paper-20261031/manifest.json","results/integrated-paper-20261031/verification.json"):
        p=ROOT/name;paths[name]=dict(path=name,bytes=p.stat().st_size,sha256=sha(p))
    tmp=target.with_suffix(".tmp")
    with tmp.open("wb") as f,gzip.GzipFile(fileobj=f,filename="",mode="wb",mtime=0) as gz,tarfile.open(fileobj=gz,mode="w") as tf:
        for name,record in sorted(paths.items()):
            p=ROOT/name
            assert p.stat().st_size==record["bytes"] and sha(p)==record["sha256"],name
            data=p.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644
            tf.addfile(info,io.BytesIO(data))
    tmp.replace(target)
    (OUT/"prior-integration.json").write_text(json.dumps(dict(path=str(target.relative_to(ROOT)),sha256=sha(target),
        files=len(paths),prior_manifest_sha256=sha(manifest),scope="Complete byte-verified integrated delivery before receiving the three new runtimes"),indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserve-prior",action="store_true",
        help="Archive an existing verified integration before updating it; unnecessary in a clean clone")
    args=parser.parse_args()
    spec=spec_from_file_location("release_download",ROOT/"scripts/integrated-release-download.py")
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    if args.preserve_prior:preserve_prior()
    records=[]
    for package,digest in mod.PINS.items():
        archive=OUT/"downloads"/(package+".tar.gz")
        assert sha(archive)==digest
        expected,seen={},set()
        with tarfile.open(archive,mode="r|gz") as tf:
            for member in tf:
                name=member.name;rel=Path(name)
                assert not rel.is_absolute() and ".." not in rel.parts and member.isfile(),name
                data=tf.extractfile(member).read();actual=hashlib.sha256(data).hexdigest()
                if name=="SHA256SUMS":
                    for line in data.decode().splitlines():
                        h,p=line.split(maxsplit=1);p=p.lstrip("*")
                        assert len(h)==64 and p not in expected
                        expected[p]=h
                else:
                    seen.add(name)
                scientific=name.startswith(("data/pedrorev-cf/","data/pedrorev-cf-followup/","results/multiteacher-degree-generalization/"))
                dest=ROOT/rel if scientific else OUT/"package-metadata"/package/rel
                # The incoming cell inventory is metadata: preserve the existing Git copy.
                if dest.exists():assert sha(dest)==actual,(str(dest),"existing different bytes preserved")
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(data)
                if expected and name!="SHA256SUMS":assert actual==expected[name],name
                records.append(dict(package=package,member=name,path=str(dest.relative_to(ROOT)),bytes=len(data),sha256=actual,
                    scientific_payload=scientific))
        assert seen==set(expected),(package,"manifest coverage")
        for r in records:
            if r["package"]==package and r["member"]!="SHA256SUMS":assert r["sha256"]==expected[r["member"]]
        print(json.dumps(dict(package=package,files=sum(r["package"]==package for r in records),payload=sum(r["package"]==package and r["scientific_payload"] for r in records))),flush=True)
    # Published inventory copies in the package must equal the audited source records.
    for package,source in (("pedrorev-cf","cf-experiment"),("pedrorev-cf-followup","cf-followup")):
        for p in (OUT/"package-metadata"/package/"provenance").glob("*.json"):
            q=ROOT/"results/pedro-revision"/source/p.name
            assert sha(p)==sha(q),(p,q)
    index=OUT/"member-index.json"
    if index.exists():
        assert json.loads(index.read_text())["files"]==records,"Existing member index differs; preserve it for investigation"
    else:
        index.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
            files=records,verified_all_archive_members=True,existing_scientific_files_preserved=True),indent=2)+"\n")


if __name__=="__main__":main()
