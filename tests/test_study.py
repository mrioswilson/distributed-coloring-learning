from dataclasses import replace
import json
from pathlib import Path
import subprocess
import types

import networkx as nx
import numpy as np
import pytest
import torch

from dcl.counterfactual import exact_commit_labels, make_counterfactual_proposals, commit_logits_for_proposals
from dcl.m4_data import encode_m4_state
from dcl.m4_models import M4B
from dcl.oracle import compute_commit_mask, run_oracle
from dcl.study.config import Config, seed_for
from dcl.study.data import collect_examples, make_batch
from dcl.study.evaluation import available_matrix, run_rollout, evaluation_cases
from dcl.study.graphs import generate_graph, fingerprint
from dcl.study.storage import training_lock, Registry, atomic_json
from dcl.study.train import initialize, train, training_step, validate_local
from dcl.study.cli import plans, freeze
from dcl.study.report import binomial_intervals, paired_statistics


@pytest.mark.parametrize("degree,parameters", [(3, 8967), (5, 9737), (8, 10892), (10, 11662)])
def test_palette_architecture_and_oracle(degree, parameters):
    graph = generate_graph(40, degree, "REGULAR", 40)
    q = degree + 1
    result = run_oracle(graph, 124, num_colors=q)
    assert result.solved and np.all((result.colors >= 0) & (result.colors < q))
    state = encode_m4_state(graph, result.trace[0].colors_before, result.trace[0].activation_draw, result.trace[0].candidate_draw, q)
    model = M4B(num_colors=q)
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert state.x.shape == (40, q + 3)
    p, c, k = model(torch.from_numpy(state.x), torch.from_numpy(state.edge_index), hard_messages=True)
    assert p.shape == k.shape == (40,) and c.shape == (40, q + 1)
    forced = commit_logits_for_proposals(model, torch.from_numpy(state.x), torch.from_numpy(state.edge_index), p > 0, c.argmax(1))
    torch.testing.assert_close(forced, k, rtol=0, atol=0)


@pytest.mark.parametrize("degree", [3, 5, 8, 10])
@pytest.mark.parametrize("family", ["REGULAR", "IRREGULAR", "MIXED", "PATH", "CYCLE"])
def test_graph_generation(degree, family):
    a, b = generate_graph(40, degree, family, 913), generate_graph(40, degree, family, 913)
    assert fingerprint(a) == fingerprint(b)
    assert nx.is_connected(a) and set(a) == set(range(40))
    assert max(dict(a.degree()).values()) <= degree
    if family == "REGULAR":
        assert set(dict(a.degree()).values()) == {degree}
    if family == "IRREGULAR":
        assert len(set(dict(a.degree()).values())) > 1


@pytest.mark.parametrize("degree", [3, 5, 8, 10])
def test_exact_labels_and_counterfactuals_match_oracle(degree):
    q, rng = degree + 1, np.random.default_rng(541)
    graph = generate_graph(40, degree, "IRREGULAR", 945)
    for _ in range(30):
        colors = rng.integers(-1, q, 40)
        state = encode_m4_state(graph, colors, rng.random(40), rng.random(40), q)
        part, candidates = rng.random(40) < 0.5, rng.integers(0, q + 1, 40)
        for p, c in [(part, candidates), make_counterfactual_proposals(state.x, state.edge_index, part, candidates, rng)]:
            expected = compute_commit_mask(graph, colors, p, c - 1)
            np.testing.assert_array_equal(exact_commit_labels(state.x, state.edge_index, p, c), expected)
            assert np.all((c >= 0) & (c <= q))


@pytest.mark.parametrize("colors,part,candidates,expected", [
    ([0, -1], [0, 1], [0, 1], [0, 0]),  # fixed-color conflict
    ([-1, -1], [1, 1], [1, 1], [0, 0]),  # simultaneous collision
    ([-1, -1], [1, 1], [1, 2], [1, 1]),  # valid simultaneous offers
    ([-1, -1], [0, 1], [1, 2], [0, 1]),  # nonparticipant
    ([-1, -1], [1, 1], [0, 2], [0, 1]),  # no candidate
])
def test_commit_edge_cases(colors, part, candidates, expected):
    state = encode_m4_state(nx.path_graph(2), np.array(colors), np.zeros(2), np.zeros(2))
    np.testing.assert_array_equal(exact_commit_labels(state.x, state.edge_index, part, candidates), expected)


@pytest.mark.parametrize("degree", [3, 5, 8, 10])
def test_single_available_color(degree):
    graph = nx.star_graph(degree)
    colors = np.array([-1] + list(range(degree)))
    state = encode_m4_state(graph, colors, np.zeros(degree + 1), np.zeros(degree + 1), degree + 1)
    available = available_matrix(colors, state.edge_index, degree + 1)
    assert available[0].sum() == 1 and available[0, degree]
    part = np.zeros(degree + 1, bool)
    part[0] = True
    candidates = np.zeros(degree + 1, int)
    candidates[0] = degree + 1
    assert exact_commit_labels(state.x, state.edge_index, part, candidates)[0] == 1
    candidates[0] = 1
    assert exact_commit_labels(state.x, state.edge_index, part, candidates)[0] == 0


def test_historical_validation_label_mismatch_reproduction():
    graph, colors = nx.path_graph(2), np.array([0, -1])
    state = encode_m4_state(graph, colors, np.zeros(2), np.zeros(2))
    part = np.array([False, True])
    teacher_candidates, own_candidates = np.array([0, 2]), np.array([0, 1])
    teacher_labels = exact_commit_labels(state.x, state.edge_index, part, teacher_candidates)
    own_labels = exact_commit_labels(state.x, state.edge_index, part, own_candidates)
    assert teacher_labels[1] == 1 and own_labels[1] == 0


def historical_module(path):
    module = types.ModuleType("historical")
    source = subprocess.check_output(["git", "show", f"8df8a58:{path}"], text=True)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def test_delta_three_exact_historical_compatibility():
    historical = historical_module("src/dcl/m4_models.py")
    old_cf = historical_module("src/dcl/counterfactual.py")
    torch.manual_seed(910)
    old = historical.M4B()
    torch.manual_seed(910)
    new = M4B()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    graph = generate_graph(40, 3, "REGULAR", 540)
    colors = np.array([-1] * 20 + [0, 1, 2, 3] * 5)
    rng = np.random.default_rng(540)
    state = encode_m4_state(graph, colors, rng.random(40), rng.random(40))
    x, edges = torch.from_numpy(state.x), torch.from_numpy(state.edge_index)
    for hard in (False, True):
        for a, b in zip(old(x, edges, hard_messages=hard), new(x, edges, hard_messages=hard)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    p, c = rng.random(40) < 0.5, rng.integers(0, 5, 40)
    for a, b in zip(old_cf.make_counterfactual_proposals(state.x, state.edge_index, p, c, np.random.default_rng(76)),
                    make_counterfactual_proposals(state.x, state.edge_index, p, c, np.random.default_rng(76))):
        np.testing.assert_array_equal(a, b)


def small_config(**kwargs):
    return Config(graphs=2, tapes=1, epochs=3, warmup=1, val_graphs=2, batch_size=8,
                  val_rollouts=1, val_sizes=(40,), val_every=1, max_rounds=3, checkpoint_epochs=(1, 2, 3), **kwargs)


def test_seed_independence_pairing_and_dataset_separation():
    config = small_config()
    _, train_records = collect_examples(config, "train")
    _, val_records = collect_examples(config, "validation")
    _, other = collect_examples(replace(config, seed=1), "train")
    _, paired = collect_examples(replace(config, method="OWN_ONLY"), "train")
    assert train_records == paired and train_records != other
    assert not {r["fingerprint"] for r in train_records} & {r["fingerprint"] for r in val_records}
    assert len({seed_for(0, 3, name) for name in ("initialization", "minibatches", "counterfactual", "train/graph", "train/tape", "validation/graph", "validation/tape", "final_test/graph")}) == 8
    a, _, _, _ = initialize(config, "cpu")
    b, _, _, _ = initialize(replace(config, method="OWN_ONLY", family="MIXED"), "cpu")
    assert all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items())
    with pytest.raises(ValueError):
        collect_examples(config, "final_test")


@pytest.mark.parametrize("degree", [3, 5, 8, 10])
@pytest.mark.parametrize("epoch", [1, 2])
def test_all_parameters_receive_gradients(degree, epoch):
    config = small_config(degree=degree)
    examples, _ = collect_examples(config, "train")
    model, optimizer, _, rng = initialize(config, "cpu")
    training_step(model, optimizer, make_batch(examples[:8], "cpu"), config, epoch, rng)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            assert_nested_equal(av, bv)
    else:
        assert a == b


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_exact_optimizer_and_rng_resume(tmp_path, device):
    config = small_config()
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resumed"
    assert train(config, uninterrupted, device) == "complete"
    assert train(config, resumed, device, stop_after_epoch=1) == "paused"
    assert train(config, resumed, device) == "complete"
    a = torch.load(uninterrupted / "runs" / config.id / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(resumed / "runs" / config.id / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "rng", "updates", "rounds_seen", "nodes_seen", "data_hash", "initial_state_hash", "epoch", "best_score"):
        assert_nested_equal(a[key], b[key])
    for epoch in (1, 2, 3):
        assert (resumed / "runs" / config.id / f"epoch-{epoch:03d}.pt").exists()
    assert train(config, resumed, device) == "complete"


def test_no_shield_diagnostics_cannot_change_actions():
    # Deliberately unsafe model: both neighbors commit color zero. A shield
    # would make this test fail by preventing the monochromatic edge.
    class Unsafe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
        def forward(self, x, edges, hard_messages):
            n = len(x)
            logits = x.new_zeros((n, 5))
            logits[:, 1] = 10
            return x.new_full((n,), 10), logits, x.new_full((n,), 10)
    graph = nx.path_graph(2)
    a = run_rollout(Unsafe(), graph, 17, 4, diagnostics=True)
    b = run_rollout(Unsafe(), graph, 17, 4, diagnostics=False)
    assert a["finished"] and not a["success"] and a["monochromatic_edges"] == 1
    assert a["unsafe_commits"] == a["simultaneous_conflicts"] == 2
    np.testing.assert_array_equal(a["colors"], b["colors"])


def test_single_node_and_empty_edges():
    graph = nx.empty_graph(1)
    state = encode_m4_state(graph, np.array([-1]), np.zeros(1), np.zeros(1), 11)
    assert state.edge_index.shape == (2, 0)
    assert exact_commit_labels(state.x, state.edge_index, [True], [11])[0] == 1


def test_successful_rollout_is_json_serializable(tmp_path):
    class Valid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
        def forward(self, x, edges, hard_messages):
            logits = torch.nn.functional.one_hot(torch.tensor([1, 2], device=x.device), 5).float() * 10
            return x.new_full((2,), 10), logits, x.new_full((2,), 10)
    result = run_rollout(Valid(), nx.path_graph(2), 123, 4)
    result.pop("colors")
    assert result["finished"] is True and result["success"] is True
    atomic_json(tmp_path / "success.json", result)
    assert json.loads((tmp_path / "success.json").read_text())["success"] is True


def test_global_training_lock_and_registry(tmp_path):
    with training_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another study"):
            with training_lock(tmp_path / "other"):
                pass
    config = small_config()
    registry = Registry(tmp_path)
    registry.add(config)
    registry.set(config, "paused", 1, 12)
    assert registry.rows()[0]["epoch"] == 1 and registry.rows()[0]["updates"] == 12


def test_plan_scope_and_fixed_budget():
    p = plans()
    assert len(p["phase-a"]) == 80 and len(p["phase-b"]) == 20
    assert [(c.degree, c.seed) for c in p["phase-a"][:2]] == [(3, 0), (10, 0)]
    assert {c.seed for c in p["phase-a"]} == set(range(20))
    fixed = {(c.graphs, c.tapes) for c in p["phase-c-epochs"] if c.graphs * c.tapes == 2400}
    assert fixed == {(200, 12), (400, 6), (800, 3), (1200, 2), (2400, 1)}
    assert all(c.updates_per_epoch == 256 for c in p["phase-c-updates"])


def test_final_cases_are_paired_and_independent():
    config = small_config()
    a = list(evaluation_cases(config, "final_test", (40,), {40: 1}, test_seed_namespace=998877))
    b = list(evaluation_cases(replace(config, method="OWN_ONLY", seed=1), "final_test", (40,), {40: 1}, test_seed_namespace=998877))
    assert [m for _, m in a] == [m for _, m in b]
    _, train_records = collect_examples(config, "train")
    assert not {m["fingerprint"] for _, m in a} & {r["fingerprint"] for r in train_records}


def test_protocol_probabilities_locked():
    with pytest.raises(ValueError, match="frozen"):
        Config(fixed_neighbor_probability=0.5)


def test_counterfactual_and_ownonly_have_identical_ingredients(tmp_path):
    treatment = small_config()
    control = replace(treatment, method="OWN_ONLY")
    train(treatment, tmp_path, "cpu")
    train(control, tmp_path, "cpu")
    a = torch.load(tmp_path / "runs" / treatment.id / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "runs" / control.id / "last.pt", weights_only=False)
    assert a["initial_state_hash"] == b["initial_state_hash"] and a["data_hash"] == b["data_hash"]
    assert a["updates"] == b["updates"] and a["rounds_seen"] == b["rounds_seen"]
    for key in a["model_state_dict"]:
        if key.startswith(("participation_head", "candidate_head", "message1")):
            assert torch.equal(a["model_state_dict"][key], b["model_state_dict"][key])


def test_checkpoint_crash_after_atomic_epoch_persistence(tmp_path, monkeypatch):
    import importlib
    training = importlib.import_module("dcl.study.train")
    config = small_config()
    train(config, tmp_path / "full", "cpu")
    original = training.save_checkpoint
    def interrupted(path, value):
        original(path, value)
        if Path(path).name == "last.pt" and value["epoch"] == 2:
            raise RuntimeError("simulated sudden interruption")
    with monkeypatch.context() as context:
        context.setattr(training, "save_checkpoint", interrupted)
        with pytest.raises(RuntimeError, match="simulated"):
            train(config, tmp_path / "crash", "cpu")
    assert train(config, tmp_path / "crash", "cpu") == "complete"
    a = torch.load(tmp_path / "full" / "runs" / config.id / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "crash" / "runs" / config.id / "last.pt", weights_only=False)
    assert_nested_equal(a["model_state_dict"], b["model_state_dict"])
    assert_nested_equal(a["optimizer_state_dict"], b["optimizer_state_dict"])
    assert (tmp_path / "crash" / "runs" / config.id / "epoch-002.pt").exists()


def test_scientific_denominators_and_paired_statistics():
    interval = binomial_intervals(10, 10)
    assert 0 < interval["exact_low"] < 1 and interval["exact_high"] == 1
    paired = paired_statistics([1, 1, 1, 0], [1, 0, 0, 1])
    assert paired["paired_graphs"] == 4 and paired["a_only"] == 2 and paired["b_only"] == 1
    assert paired["success_difference"] == 0.25


def test_frozen_final_test_requires_completed_training(tmp_path):
    config = small_config()
    train(config, tmp_path, "cpu", stop_after_epoch=1)
    plan = tmp_path / "plan.json"
    atomic_json(plan, {"protocol": "m6study-v1", "experiments": [config.to_dict()]})
    with pytest.raises(RuntimeError, match="Incomplete"):
        freeze(types.SimpleNamespace(plan=plan, root=tmp_path, output=tmp_path / "frozen.json"))


def test_fixed_update_count_and_counterfactual_class_bounds(tmp_path):
    config = replace(small_config(), updates_per_epoch=2)
    train(config, tmp_path, "cpu")
    saved = torch.load(tmp_path / "runs" / config.id / "last.pt", weights_only=False)
    assert saved["updates"] == 6
    state = encode_m4_state(nx.path_graph(2), np.array([-1, -1]), np.zeros(2), np.zeros(2), 11)
    with pytest.raises(ValueError, match="classes"):
        exact_commit_labels(state.x, state.edge_index, [1, 1], [12, 1])


def test_evaluation_resume_and_scientific_report(tmp_path):
    from dcl.study.cli import evaluate
    from dcl.study.report import aggregate
    from dcl.study.config import digest
    config = small_config()
    train(config, tmp_path, "cpu")
    plan = tmp_path / "plan.json"
    frozen_path = tmp_path / "frozen.json"
    atomic_json(plan, {"protocol": "m6study-v1", "experiments": [config.to_dict()]})
    freeze(types.SimpleNamespace(plan=plan, root=tmp_path, output=frozen_path, checkpoint_epochs=[2]))
    # Dedicated technical cohort: the actual final-test random tapes are never
    # run while developing/validating this infrastructure.
    technical = json.loads(frozen_path.read_text())
    technical.update(test_seed_namespace=998877, counts={"40": 1}, max_rounds=3)
    atomic_json(frozen_path, technical)
    args = types.SimpleNamespace(freeze=frozen_path, root=tmp_path, device="cpu")
    evaluate(args)
    first = {p: p.read_bytes() for p in (tmp_path / "evaluations").glob("*/*/*.json")}
    evaluate(args)
    assert len(first) == 4 and first == {p: p.read_bytes() for p in first}
    aggregate(tmp_path)
    import pandas as pd
    summary = pd.read_csv(tmp_path / "reports/success-by-seed.csv")
    assert len(summary) == 4 and set(summary.checkpoint_policy) == {"best", "epoch-002"}
    assert (tmp_path / "reports/final-epochs.pdf").exists()
    assert (tmp_path / "reports/k-summary.csv").exists()
    assert "Observaciones del test final persistidas: 4" in (tmp_path / "reports/REPORT.md").read_text()
