#!/usr/bin/env python3
"""Restore byte-identical historical snapshots from Git-sized transport parts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "results/integrated-paper-20261031/git-snapshot-parts.json"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(destination):
    records = json.loads(INDEX.read_text())["archives"]
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe target: {relative}")
        target = destination / relative
        combined = hashlib.sha256()
        size = 0
        for part in record["parts"]:
            name = Path(part["path"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe part: {name}")
            path = ROOT / name
            if path.stat().st_size != part["bytes"] or sha(path) != part["sha256"]:
                raise ValueError(f"Part differs: {name}")
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    combined.update(block)
                    size += len(block)
        if size != record["bytes"] or combined.hexdigest() != record["sha256"]:
            raise ValueError(f"Combined snapshot differs: {relative}")
        existed = target.exists()
        if existed:
            if target.stat().st_size != size or sha(target) != record["sha256"]:
                raise FileExistsError(f"Refusing to replace different bytes: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves existing files, including a concurrent target.
            with target.open("xb") as output:
                for part in record["parts"]:
                    with (ROOT / part["path"]).open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(block)
            if sha(target) != record["sha256"]:
                raise ValueError(f"Written snapshot differs: {target}")
        print(json.dumps(dict(path=str(target), bytes=size, sha256=record["sha256"],
                              parts=len(record["parts"]), existing_preserved=existed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT,
                        help="Repository root (or a separate directory for a restoration check)")
    restore(parser.parse_args().destination.resolve())


if __name__ == "__main__":
    main()
