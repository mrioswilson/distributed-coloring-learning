"""Delivery checks in addition to the scalar replay audit; no training changes."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "data/e1-retrained-20261021"
OUT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    baseline = json.loads(Path("/workspace/coordination/tasks/e1-retrain-20261021-baseline.json").read_text())
    changed = {p for p, expected in baseline.items() if not (REPO / p).is_file() or sha(REPO / p) != expected}
    assert changed == {"src/dcl/e1_behavior.py", "experiments/compare_e1_behavior.py"}, changed
    original = json.loads((REPO / "results/e1-behavior/2026-10-21/upstream-source.json").read_text())
    assert all(sha(REPO / p) == expected for p, expected in original["files"].items())
    train = json.loads((ROOT / "training-plan.json").read_text())
    assert all(sha(REPO / p) == expected for p, expected in train["sources"].items())
    planned = json.loads((ROOT / "planned-cohort.json").read_text())
    manifest = json.loads((ROOT / "comparison/manifest.json").read_text())
    frozen = json.loads((ROOT / "frozen.json").read_text())
    assert planned["graphs"] == manifest["graphs"] and planned["plan"] == manifest["plan"]
    assert (ROOT / "frozen.json").stat().st_mtime_ns < (ROOT / "comparison/manifest.json").stat().st_mtime_ns
    assert (ROOT / "freeze-time.json").stat().st_mtime_ns < (ROOT / "comparison/manifest.json").stat().st_mtime_ns
    graphs = {r["fingerprint"] for r in manifest["graphs"]}
    training_fingerprints, validation_fingerprints, rollout_fingerprints, budget = set(), set(), set(), []
    for item in frozen["checkpoints"]:
        path = ROOT / item["checkpoint"]
        assert sha(path) == item["sha256"]
        assert path.stat().st_mtime_ns <= (ROOT / "frozen.json").stat().st_mtime_ns
        data = json.loads((path.parent / "data_manifest.json").read_text())
        history = json.loads((path.parent / "history.json").read_text())
        assert [r["epoch"] for r in history] == list(range(1, 121))
        assert history[-1]["updates"] == 120 * ((data["training_rounds"] + 127) // 128)
        assert len(data["train"]) == 800 and len(data["validation"]) == 200
        assert sum(len(r["trajectories"]) for r in data["train"]) == 2400
        assert sum(len(r["trajectories"]) for r in data["validation"]) == 200
        training_fingerprints.update(r["fingerprint"] for r in data["train"])
        validation_fingerprints.update(r["fingerprint"] for r in data["validation"])
        rollout_fingerprints.update(r["fingerprint"] for r in history[-1]["validation_rollouts"]["per_graph"])
        budget.append({"seed": item["seed"], "epochs": 120, "updates": item["updates"],
                       "round_examples": data["training_rounds"], "active_node_presentations": history[-1]["nodes_seen"],
                       "training_seconds": history[-1]["elapsed_seconds"], "sha256": item["sha256"]})
    assert not graphs & (training_fingerprints | validation_fingerprints | rollout_fingerprints)
    audit = json.loads((OUT / "audit.json").read_text())
    assert audit["complete"] and audit["colorings_checked"] == 148736
    record = {"complete": True, "prior_files_preserved": len(baseline) - len(changed),
              "authorized_prior_files_changed": sorted(changed), "original_source_files_preserved": len(original["files"]),
              "training_sources_unchanged": True, "test_graphs_equal_prefixed_cohort": True,
              "all_models_frozen_before_test": True, "training_graphs": len(training_fingerprints),
              "local_validation_graphs": len(validation_fingerprints), "rollout_validation_graphs": len(rollout_fingerprints),
              "evaluation_graph_overlap": 0, "budget": budget, "scalar_replay_audit": audit,
              "independent_scientific_review": "pending"}
    (OUT / "delivery-audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
