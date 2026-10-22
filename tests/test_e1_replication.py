"""Canonical configuration and authentication of new, distinct E1 replicas."""
import json
from pathlib import Path

import pytest
import torch

from dcl import e1_behavior as b
from dcl import e1_replication as r
from dcl.universal.config import UConfig
from dcl.universal.model import UniversalColorGNN


def test_all_five_configs_exactly_match_original_e1_plan():
    canonical = r.configs()
    original = json.loads(b.INDEX.read_text())
    assert [c.seed for c in canonical] == list(range(5))
    assert [c.id for c in canonical] == [item["run"] for item in original["checkpoints"]]
    assert all((c.epochs, c.graphs_per_degree, c.tapes, c.batch_size, c.warmup, c.method) ==
               (120, (800,), 3, 128, 10, "COUNTERFACTUAL") for c in canonical)


def test_training_plan_snapshot_is_idempotent_and_freeze_requires_complete_weights(tmp_path):
    first = r.prepare_training(tmp_path)
    assert first == r.prepare_training(tmp_path)
    assert first["cohort_origin"] == "newly-trained-replicas"
    assert first["sources"]["src/dcl/universal/train.py"] == b.file_hash(b.REPO / "src/dcl/universal/train.py")
    with pytest.raises(FileNotFoundError):
        r.freeze(tmp_path)
    assert not (tmp_path / "frozen.json").exists()


@pytest.fixture
def technical_index(tmp_path):
    # Untrained random state_dicts are strictly confined to test temporaries.
    # This fixture tests the index interface; production freeze checks actual
    # optimizer history, source provenance, data hashes and the full budget.
    plan = r.prepare_training(tmp_path)
    index = {"protocol": r.PROTOCOL, "cohort_origin": "newly-trained-replicas", "upstream_commit": b.UPSTREAM,
             "canonical_config_sha256": plan["canonical_config_sha256"], "sources": plan["sources"],
             "frozen_before_comparison": True, "architecture": "UniversalColorGNN NORMALIZED",
             "method": "COUNTERFACTUAL", "parameters": 16035, "checkpoints": []}
    for config in r.configs():
        path = tmp_path / "runs" / config.id / "epoch-120.pt"
        path.parent.mkdir(parents=True)
        torch.save({"config": config.to_dict(), "model_state_dict": UniversalColorGNN().state_dict(),
                    "epoch": 120, "parameters": 16035, "complete": True}, path)
        index["checkpoints"].append({"seed": config.seed, "run": config.id,
                                     "checkpoint": str(path.relative_to(tmp_path)), "bytes": path.stat().st_size,
                                     "sha256": b.file_hash(path)})
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(index))
    return tmp_path, path, index


def test_new_index_accepts_own_hashes_without_claiming_historical_identity(technical_index):
    root, path, index = technical_index
    inputs = b.inspect_inputs(root, path)
    assert inputs["ready"] and inputs["cohort_origin"] == "newly-trained-replicas"
    models, actual = b.load_models(root, path)
    assert set(models) == set(range(5))
    assert actual == inputs
    assert not b.inspect_inputs(root)["ready"]


def test_changed_new_weights_are_rejected_before_loading(technical_index):
    root, path, index = technical_index
    checkpoint = root / index["checkpoints"][2]["checkpoint"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="s2=mismatch"):
        b.load_models(root, path)


@pytest.mark.parametrize("field,value", [("cohort_origin", "historical-original"), ("frozen_before_comparison", False)])
def test_invalid_reproduction_index_rejected(technical_index, field, value):
    root, path, index = technical_index
    index[field] = value
    path.write_text(json.dumps(index))
    with pytest.raises(RuntimeError):
        b.inspect_inputs(root, path)
