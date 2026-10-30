from pathlib import Path
import hashlib
import pytest
from dcl.artifacts import resolve_artifact


def test_relocates_frozen_identity_and_rejects_substitution(tmp_path,monkeypatch):
    monkeypatch.setenv("DCL_ARTIFACT_ROOT",str(tmp_path))
    target=tmp_path/"data"/"study"/"checkpoint.pt"
    target.parent.mkdir(parents=True);target.write_bytes(b"authentic fixture")
    historical="/historical/absent/worktree/data/study/checkpoint.pt"
    assert resolve_artifact(historical,hashlib.sha256(target.read_bytes()).hexdigest())==target
    with pytest.raises(RuntimeError,match="identity differs"):
        resolve_artifact(historical,"0"*64)


def test_missing_artifact_is_not_guessed(tmp_path,monkeypatch):
    monkeypatch.setenv("DCL_ARTIFACT_ROOT",str(tmp_path))
    with pytest.raises(FileNotFoundError,match="not restored"):
        resolve_artifact("/historical/data/study/missing.pt")
