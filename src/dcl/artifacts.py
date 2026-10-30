"""Relocate historical artifact paths while preserving their frozen identities."""
import hashlib
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def resolve_artifact(path, sha256=None):
    original = Path(path)
    candidates = [original]
    root = Path(os.environ.get("DCL_ARTIFACT_ROOT", REPOSITORY))
    parts = original.parts
    for marker in ("data", "results"):
        if marker in parts:
            candidates.append(root.joinpath(*parts[parts.index(marker):]))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            if sha256 is not None:
                with candidate.open("rb") as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                if actual != sha256:
                    raise RuntimeError(f"Recovered artifact identity differs: {candidate}")
            return candidate
    raise FileNotFoundError(f"Artifact not restored: {original}. Restore the verified runtime under {root}, or set DCL_ARTIFACT_ROOT to its repository root.")
