import importlib.util
from contextlib import nullcontext
import json
from dataclasses import replace
from pathlib import Path
import types

import pandas as pd
import pytest

from dcl.study.config import Config
from dcl.study.storage import atomic_json, file_hash


spec = importlib.util.spec_from_file_location("public_export", Path(__file__).resolve().parents[1] / "experiments/export_m6study_results.py")
public_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public_export)


def test_public_pairs_keep_denominators_and_reject_unmatched_cases():
    a = {"a": (True, 10, 0), "b": (True, 12, 0), "c": (False, 9, 2)}
    b = {"a": (False, 10, 1), "b": (True, 11, 0), "c": (True, 9, 0)}
    metadata = {"a": ("REGULAR", 40), "b": ("REGULAR", 40), "c": ("IRREGULAR", 160)}
    rows = public_export.compare_cases(a, b, metadata)
    regular = next(row for row in rows if row["distribution"] == "REGULAR")
    assert regular["paired_graphs"] == 2
    assert regular["a_only"] == 1 and regular["b_only"] == 0
    assert regular["a_successes"] == 2 and regular["b_successes"] == 1
    assert regular["rounds_difference"] == .5
    with pytest.raises(ValueError, match="identical"):
        public_export.compare_cases(a, {"a": b["a"]}, metadata)


def test_public_snapshot_exports_completed_views_and_preserves_protocol(tmp_path, monkeypatch):
    from dcl.study.cli import evaluate, freeze
    from dcl.study.train import train
    # The isolated technical evaluation uses CPU; leave the live GPU lock alone.
    monkeypatch.setattr("dcl.study.cli.training_lock", lambda root: nullcontext())
    config = Config(seed=998877, graphs=2, tapes=1, epochs=1, warmup=1, val_graphs=2,
                    val_rollouts=1, val_sizes=(40,), val_every=1, max_rounds=3, checkpoint_epochs=(1,))
    configs = [config, replace(config, method="OWN_ONLY")]
    root = tmp_path / "runtime"
    for item in configs:
        train(item, root, "cpu")
    plan, frozen_path = tmp_path / "plan.json", root / "frozen-final-test.json"
    atomic_json(plan, {"protocol": "m6study-v1", "experiments": [item.to_dict() for item in configs]})
    freeze(types.SimpleNamespace(plan=plan, root=root, output=frozen_path, checkpoint_epochs=[]))
    frozen = json.loads(frozen_path.read_text())
    frozen.update(test_seed_namespace=998877, counts={"40": 1}, max_rounds=3)
    atomic_json(frozen_path, frozen)
    evaluate(types.SimpleNamespace(freeze=frozen_path, root=root, device="cpu"))
    before = file_hash(frozen_path)
    monkeypatch.setattr(public_export, "make_figures", lambda *args: None)
    output = tmp_path / "published"
    public_export.export(root, output)
    metadata = json.loads((output / "snapshot.json").read_text())
    assert metadata["complete_checkpoint_views_exported"] == 2
    assert metadata["graph_evaluations_exported"] == 4
    assert len(pd.read_csv(output / "paired-by-seed.csv")) == 2
    assert sum(pd.read_csv(output / "paired-by-seed.csv").paired_graphs) == 2
    assert file_hash(frozen_path) == before
    assert str(root) not in (output / "snapshot.json").read_text()
    with pytest.raises(RuntimeError, match="overwrite"):
        public_export.export(root, output)
    next((root / "evaluations").glob("*/*/*.json")).unlink()
    partial = tmp_path / "partial-published"
    public_export.export(root, partial)
    metadata = json.loads((partial / "snapshot.json").read_text())
    assert metadata["complete_checkpoint_views_exported"] == 1
    assert metadata["graph_evaluations_exported"] == 2
    assert metadata["graph_evaluations_stored_at_capture"] == 3
