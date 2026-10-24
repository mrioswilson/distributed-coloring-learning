#!/usr/bin/env python3
"""Download, verify, and safely restore the released query-budgeted runtime."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def download_archive(url, archive, size, *, timeout=30, chunk_bytes=16*1024*1024, attempts=6):
    """Resume bounded HTTP ranges; extraction still requires the complete SHA-256."""
    archive=Path(archive)
    if archive.is_symlink(): raise RuntimeError("Refusing symlink archive cache")
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists() and archive.stat().st_size>size:
        raise RuntimeError("Archive cache exceeds the expected length")
    failures=0
    while (archive.stat().st_size if archive.exists() else 0)<size:
        start=archive.stat().st_size if archive.exists() else 0
        end=min(size-1,start+chunk_bytes-1)
        request=urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request,timeout=timeout) as response:
                if response.status==206:
                    if response.headers.get("Content-Range")!=f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Unexpected HTTP range; refusing to append")
                    remaining=end-start+1;mode="ab"
                elif response.status==200:
                    # A server may ignore Range. Restart safely rather than append
                    # a full response to an existing prefix.
                    if int(response.headers.get("Content-Length","-1"))!=size:
                        raise RuntimeError("Unexpected full-response length")
                    remaining=size;mode="wb"
                else: raise RuntimeError("Unexpected download response")
                with archive.open(mode) as target:
                    while remaining:
                        block=response.read(min(1024*1024,remaining))
                        if not block: raise EOFError("Truncated HTTP response")
                        target.write(block);remaining-=len(block)
            failures=0
            print(f"Downloaded {archive.stat().st_size:,}/{size:,} bytes",flush=True)
        except (urllib.error.URLError, http.client.HTTPException, OSError, EOFError) as error:
            failures+=1
            if failures>=attempts: raise RuntimeError("Download retries exhausted; retain the archive cache to resume") from error
            print(f"Download interrupted ({type(error).__name__}); resuming cached bytes",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,help="Persistent archive cache; partial downloads resume automatically")
    args=parser.parse_args()
    out=Path("results/sample-complexity")
    release=json.loads((out/"runtime-release.json").read_text())
    manifest=json.loads((out/"runtime-manifest.json").read_text())
    expected={e["path"]:e for e in manifest["files"]}
    with tempfile.TemporaryDirectory(prefix="sample-complexity-") as temporary:
        temporary=Path(temporary);archive=args.archive or temporary/release["archive"]
        download_archive(release["url"],archive,release["bytes"])
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


if __name__=="__main__": main()
