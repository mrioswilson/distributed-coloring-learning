import importlib.util
import json
from pathlib import Path


module_path = Path(__file__).resolve().parents[1] / "experiments/continue_m6study.py"
spec = importlib.util.spec_from_file_location("study_continuation", module_path)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def fake_repo(tmp_path, mode):
    repo = tmp_path / "repo"
    (repo / "configs/m6study").mkdir(parents=True)
    (repo / "experiments").mkdir()
    for name in ("first", "second", "third"):
        (repo / f"configs/m6study/{name}.json").write_text(json.dumps({"protocol": "m6study-v1", "experiments": [{}]}))
    script = repo / "experiments/m6study"
    script.write_text("#!/usr/bin/env python3\n"
                      "import pathlib,sys\n"
                      "root=pathlib.Path(sys.argv[sys.argv.index('--root')+1])\n"
                      "stage=pathlib.Path(sys.argv[sys.argv.index('--plan')+1]).stem\n"
                      "with (root/'calls.txt').open('a') as f:f.write(stage+'\\n')\n"
                      + ("if stage=='second':sys.exit(7)\n" if mode == "failure" else "")
                      + ("if stage=='first':(root/'STOP').touch()\n" if mode == "stop" else ""))
    script.chmod(0o755)
    return repo


def test_continuation_stops_after_failure(tmp_path):
    repo = fake_repo(tmp_path, "failure")
    root = tmp_path / "runs"
    assert continuation.run_pipeline(repo, root, stages=("first", "second", "third")) == 7
    assert (root / "calls.txt").read_text().splitlines() == ["first", "second"]
    state = json.loads((root / "continuation-state.json").read_text())
    assert state["status"] == "failed" and state["stage"] == "second"
    assert state["completed_stages"] == ["first"]


def test_continuation_respects_stop_between_stages(tmp_path):
    repo = fake_repo(tmp_path, "stop")
    root = tmp_path / "runs"
    assert continuation.run_pipeline(repo, root, stages=("first", "second")) == 0
    assert (root / "calls.txt").read_text().splitlines() == ["first"]
    assert json.loads((root / "continuation-state.json").read_text())["status"] == "paused"
    assert continuation.run_pipeline(repo, root, stages=("second",)) == 0
    assert (root / "calls.txt").read_text().splitlines() == ["first"]
    assert continuation.run_pipeline(repo, root, stages=("second",), resume=True) == 0
    assert (root / "calls.txt").read_text().splitlines() == ["first", "second"]


def test_continuation_finishes_before_final_test(tmp_path):
    repo = fake_repo(tmp_path, "success")
    root = tmp_path / "runs"
    assert continuation.run_pipeline(repo, root, stages=("first", "second")) == 0
    state = json.loads((root / "continuation-state.json").read_text())
    assert state["status"] == "awaiting_validation_review"
    assert state["final_test_started_by_pipeline"] is False
    assert state["completed_stages"] == ["first", "second"]
