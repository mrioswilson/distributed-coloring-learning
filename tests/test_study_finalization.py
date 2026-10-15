import importlib
import json
from dataclasses import replace
from pathlib import Path
import sys
import types

import pandas as pd
import pytest
import torch

from dcl.study.config import Config
from dcl.study.storage import atomic_json, file_hash


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
review = importlib.import_module("review_m6study")
finish = importlib.import_module("finish_m6study")
reports = importlib.import_module("report_m6study_cohorts")


def test_budget_extension_uses_only_discovery_validation_and_keeps_sparse_grid():
    from dcl.study.cli import plans
    configurations = plans()["phase-c-epochs"] + plans()["phase-c-updates"]
    records = []
    winners = {(200, 3, None), (800, 3, 256), (2400, 1, None)}
    for config in configurations:
        key = (config.graphs, config.tapes, config.updates_per_epoch)
        rounds = 8 if key in winners else 10
        if key == (200, 3, None):
            rounds = 7
        records.append({"config": config.to_dict(), "id": config.id,
                        "selected_score": [1, 0, -rounds, .99, -.1], "training_seconds": 100,
                        "final_test_success_rate": 0 if key in winners else 1})
    summaries, selected, extensions = review.select_extensions(records)
    assert len(summaries) == 18
    assert {(row["graphs"], row["tapes"], row["updates_per_epoch"]) for row in selected.values()} == winners
    assert len(extensions) == 240 and len({config.id for config in extensions}) == 240
    assert {config.degree for config in extensions} == {3, 5, 8, 10}
    assert {config.seed for config in extensions} == set(range(20))
    assert {(config.graphs, config.tapes, config.updates_per_epoch) for config in extensions} == winners
    with pytest.raises(RuntimeError, match="five paired"):
        review.select_extensions(records[1:])


def fake_pipeline(tmp_path, mode):
    repo, root = tmp_path / "repo", tmp_path / "runs"
    (repo / "configs/m6study").mkdir(parents=True)
    (repo / "experiments").mkdir()
    root.mkdir()
    plans = {}
    for name in ("phase-c-extension", "phase-e-cohort"):
        path = repo / "configs/m6study" / f"{name}.json"
        atomic_json(path, {"protocol": "m6study-v1", "experiments": [Config().to_dict()]})
        plans[name] = {"path": str(path), "sha256": file_hash(path)}
    script = repo / "experiments/m6study"
    script.write_text("#!/usr/bin/env python3\nimport pathlib,sys\n"
                      "root=pathlib.Path(sys.argv[sys.argv.index('--root')+1])\n"
                      "with (root/'calls').open('a') as f:f.write(sys.argv[1]+'\\n')\n"
                      + ("sys.exit(7)\n" if mode == "failure" else "")
                      + ("(root/'STOP').touch()\n" if mode == "stop" else ""))
    script.chmod(0o755)
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    for name, label in (("review_m6study", "verify"), ("report_m6study_cohorts", "reports")):
        (repo / "experiments" / f"{name}.py").write_text(
            "import pathlib,sys\nroot=pathlib.Path(sys.argv[sys.argv.index('--root')+1])\n"
            f"with (root/'calls').open('a') as f:f.write('{label}\\n')\n"
            + ("sys.exit(9)\n" if mode == "verification_failure" and label == "verify" else ""))
    review_path = root / "review.json"
    atomic_json(review_path, {"review_complete": True, "final_test_observations_used": 0,
                            "plans": plans, "additional_unique_training": 1})
    return repo, root, review_path


def test_failed_extension_blocks_freeze_and_final_testing(tmp_path):
    repo, root, review_path = fake_pipeline(tmp_path, "failure")
    assert finish.run_pipeline(repo, root, review_path) == 7
    state = json.loads((root / "finalization-state.json").read_text())
    assert state["status"] == "failed" and state["stage"] == "extensions"
    assert (root / "calls").read_text() == "run\n"
    assert not (root / "frozen-final-test.json").exists()


def test_finalization_stop_blocks_next_stage_and_requires_explicit_resume(tmp_path):
    repo, root, review_path = fake_pipeline(tmp_path, "stop")
    assert finish.run_pipeline(repo, root, review_path) == 0
    assert json.loads((root / "finalization-state.json").read_text())["status"] == "paused"
    assert finish.run_pipeline(repo, root, review_path) == 0
    assert (root / "calls").read_text() == "run\n"
    assert finish.run_pipeline(repo, root, review_path, resume=True) == 0
    assert (root / "calls").read_text() == "run\nrun\n"
    assert not (root / "frozen-final-test.json").exists()


def test_finalization_rejects_changed_reviewed_plan(tmp_path):
    repo, root, review_path = fake_pipeline(tmp_path, "failure")
    (repo / "configs/m6study/phase-e-cohort.json").write_text("{}")
    with pytest.raises(RuntimeError, match="reviewed plan changed"):
        finish.run_pipeline(repo, root, review_path)
    assert not (root / "calls").exists()


def test_failed_verification_blocks_freeze_and_final_testing(tmp_path):
    repo, root, review_path = fake_pipeline(tmp_path, "verification_failure")
    assert finish.run_pipeline(repo, root, review_path) == 9
    assert (root / "calls").read_text().splitlines() == ["run", "verify"]
    assert not (root / "frozen-final-test.json").exists()
    assert json.loads((root / "finalization-state.json").read_text())["stage"] == "verification"


def test_finalization_runs_all_stages_and_does_not_repeat_completed_testing(tmp_path, monkeypatch):
    repo, root, review_path = fake_pipeline(tmp_path, "success")
    def technical_freeze(root, plan, review_path):
        with (root / "calls").open("a") as handle:
            handle.write("freeze\n")
        atomic_json(root / "frozen-final-test.json", {"technical": True})
    monkeypatch.setattr(finish, "freeze_cohort", technical_freeze)
    assert finish.run_pipeline(repo, root, review_path) == 0
    assert (root / "calls").read_text().splitlines() == ["run", "verify", "freeze", "evaluate", "reports"]
    assert json.loads((root / "finalization-state.json").read_text())["status"] == "complete"
    before = (root / "calls").read_bytes()
    assert finish.run_pipeline(repo, root, review_path) == 0
    assert (root / "calls").read_bytes() == before


def test_final_freeze_requires_completion_and_preserves_checkpoint_policies(tmp_path):
    config = Config()
    directory = tmp_path / "runs" / config.id
    directory.mkdir(parents=True)
    plan, review_path = tmp_path / "plan.json", tmp_path / "review.json"
    atomic_json(plan, {"protocol": "m6study-v1", "experiments": [config.to_dict()]})
    atomic_json(review_path, {"review_complete": True})
    value = {"complete": False, "config": config.to_dict(), "epoch": 1, "best_score": (1, 0, -8, .99, -.1)}
    torch.save(value, directory / "last.pt")
    with pytest.raises(RuntimeError, match="Incomplete"):
        finish.freeze_cohort(tmp_path, plan, review_path)
    assert not (tmp_path / "frozen-final-test.json").exists()
    torch.save({**value, "epoch": 120, "complete": True}, directory / "last.pt")
    torch.save({**value, "epoch": 60}, directory / "best.pt")
    for epoch in (20, 40, 60, 80, 120):
        torch.save({**value, "epoch": epoch}, directory / f"epoch-{epoch:03d}.pt")
    path = finish.freeze_cohort(tmp_path, plan, review_path)
    frozen = json.loads(path.read_text())
    assert len(frozen["models"]) == 6
    assert {entry["checkpoint_policy"] for entry in frozen["models"]} == {"best", "epoch-020", "epoch-040", "epoch-060", "epoch-080", "epoch-120"}
    assert frozen["selection_uses_final_test"] is False
    assert frozen["counts"] == finish.COUNTS
    assert finish.freeze_cohort(tmp_path, plan, review_path) == path
    atomic_json(review_path, {"review_complete": True, "changed": True})
    with pytest.raises(RuntimeError, match="another review"):
        finish.freeze_cohort(tmp_path, plan, review_path)


def test_report_groups_keep_controls_and_training_families_paired():
    treatment = Config()
    entries = [{"config": config.to_dict(), "checkpoint_policy": "best"} for config in
               (treatment, replace(treatment, method="OWN_ONLY"), replace(treatment, family="MIXED"))]
    assert len({reports.cohort_key(entry) for entry in entries}) == 1
    assert reports.cohort_key(entries[0]) != reports.cohort_key({**entries[0], "checkpoint_policy": "epoch-120"})


def test_cohort_reporting_preserves_paired_denominators_and_checks_completion(tmp_path, monkeypatch):
    from dcl.study.cli import evaluate, freeze
    from dcl.study import report
    from dcl.study.train import train
    config = Config(seed=998877, graphs=2, tapes=1, epochs=1, warmup=1, val_graphs=2,
                    val_rollouts=1, val_sizes=(40,), val_every=1, max_rounds=3, checkpoint_epochs=(1,))
    configs = [config, replace(config, method="OWN_ONLY")]
    for item in configs:
        train(item, tmp_path, "cpu")
    plan, frozen_path = tmp_path / "plan.json", tmp_path / "technical.json"
    atomic_json(plan, {"protocol": "m6study-v1", "experiments": [item.to_dict() for item in configs]})
    freeze(types.SimpleNamespace(plan=plan, root=tmp_path, output=frozen_path, checkpoint_epochs=[]))
    frozen = json.loads(frozen_path.read_text())
    frozen.update(test_seed_namespace=998877, counts={"40": 1}, max_rounds=3)
    atomic_json(frozen_path, frozen)
    evaluate(types.SimpleNamespace(freeze=frozen_path, root=tmp_path, device="cpu"))
    monkeypatch.setattr(report, "_plot", lambda *args, **kwargs: None)
    reports.aggregate_cohorts(tmp_path, frozen_path)
    summary = pd.read_csv(tmp_path / "reports/success-by-seed.csv")
    paired = pd.read_csv(tmp_path / "reports/paired-comparisons.csv")
    assert len(summary) == 4 and set(summary.graphs) == {1}
    assert len(paired) == 2 and set(paired.paired_graphs) == {1}
    assert len(pd.read_csv(tmp_path / "reports/final-per-graph.csv")) == 4
    before = (tmp_path / "reports/success-by-seed.csv").read_bytes()
    reports.aggregate_cohorts(tmp_path, frozen_path)
    assert (tmp_path / "reports/success-by-seed.csv").read_bytes() == before
    next((tmp_path / "evaluations").glob("*/*/*.json")).unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        reports.aggregate_cohorts(tmp_path, frozen_path)
