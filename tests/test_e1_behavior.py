"""Technical validation only: fixtures/random weights are never E1 evidence."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from dcl import behavior_statistics as stats
from dcl import e1_behavior as b
from dcl.m4_data import _edge_index
from dcl.oracle import run_oracle
from dcl.universal.evaluation import run_rollout
from dcl.universal.model import UniversalColorGNN, count_parameters
from dcl.universal.rules import exact_commit_labels, teacher_offer


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(1304)


def plan(**overrides):
    original = json.loads(Path("configs/e1_behavior/comparison.json").read_text())
    return {**original, "max_rounds": 20, "permutations": 99, **overrides}


def test_preflight_identifies_original_e1_and_refuses_missing_weights(tmp_path, monkeypatch):
    result = b.inspect_inputs(tmp_path)
    assert result["source_verified"] and result["parameters"] == 16035
    assert [r["seed"] for r in result["checkpoints"]] == list(range(5))
    assert all(r["state"] == "missing" for r in result["checkpoints"])
    assert not result["ready"]
    monkeypatch.setattr(b, "load_model", lambda *a: pytest.fail("Must not load or substitute missing weights"))
    with pytest.raises(RuntimeError, match="no comparison was run"):
        b.load_models(tmp_path)


def test_preflight_rejects_same_size_but_wrong_weight_bytes(tmp_path):
    first = b.inspect_inputs(tmp_path)["checkpoints"][0]
    path = Path(first["resolved_path"])
    path.parent.mkdir(parents=True)
    path.write_bytes(bytes(first["bytes"]))
    result = b.inspect_inputs(tmp_path)
    assert result["checkpoints"][0]["state"] == "mismatch"
    assert not result["ready"]
    with pytest.raises(RuntimeError, match="s0=mismatch"):
        b.load_models(tmp_path)


def test_preflight_rejects_changed_upstream_source(monkeypatch, tmp_path):
    original = b.file_hash
    monkeypatch.setattr(b, "file_hash", lambda path: "0" * 64 if str(path).endswith("universal/model.py") else original(path))
    with pytest.raises(RuntimeError, match="Original E1 source changed"):
        b.inspect_inputs(tmp_path)


def test_actual_e1_factory_has_16035_parameters_and_palette_independent_weights():
    model = UniversalColorGNN(64, 32, "NORMALIZED")
    assert count_parameters(model) == 16035
    for palette in (4, 11):
        edges = torch.from_numpy(_edge_index(nx.path_graph(8)))
        out = model(torch.full((8,), -1), torch.rand(8), torch.rand(8), edges, torch.full((8,), palette))
        assert out.candidate_logits.shape == (8, palette)
        assert count_parameters(model) == 16035


@pytest.mark.parametrize("seeds", [[11, 12], [218, 219], [9172, 9173]])
def test_exact_teacher_queries_and_trajectories_equal_original_oracle(seeds):
    graph = nx.cycle_graph(12)
    edges = _edge_index(graph)
    actual = b.teacher_batch(graph, seeds, plan(max_rounds=128))
    for i, seed in enumerate(seeds):
        expected = run_oracle(graph, seed, num_colors=4, max_rounds=128)
        for trace in expected.trace:
            _, p, c, commit, _, _ = b.truth_from_draws(trace.colors_before[None], trace.activation_draw[None],
                                                       trace.candidate_draw[None], edges)
            np.testing.assert_array_equal(p[0], trace.participate)
            np.testing.assert_array_equal(c[0], trace.candidate_offer)
            np.testing.assert_array_equal(commit[0], trace.commit)
        np.testing.assert_array_equal(actual["colors"][i], expected.colors)
        assert actual["rounds"][i] == expected.rounds and actual["status"][i] == b.SUCCESS


@pytest.mark.parametrize("graph", [nx.path_graph(8), nx.cycle_graph(12), nx.complete_graph(4)])
def test_batched_e1_execution_matches_original_scalar_rollout(graph):
    # Untrained weights exercise the adapter, never the scientific comparison.
    model, seeds, p = UniversalColorGNN().eval(), [101, 102, 103], plan()
    reference, learned, counts, _ = b.coupled_batch(graph, seeds, {0: model}, p)
    for i, seed in enumerate(seeds):
        expected, colors = run_rollout(model, len(graph), _edge_index(graph), 4, seed,
                                       max_rounds=p["max_rounds"], diagnostics=False)
        np.testing.assert_array_equal(learned[0]["colors"][i], colors)
        assert learned[0]["rounds"][i] == expected["rounds"]
        assert (learned[0]["status"][i] == b.SUCCESS) == expected["success"]
        assert (learned[0]["status"][i] == b.COLLISION) == (expected["monochromatic_edges"] > 0)
    assert counts[(0, "teacher_states", -1)]["states"] > 0
    assert counts[(0, "gnn_states", -1)]["states"] > 0
    assert np.all(reference["status"] == b.SUCCESS)


class ExactFixture(torch.nn.Module):
    """Unit-test fixture only, structurally rejected by the real E1 loader."""
    def forward(self, colors, rp, rc, edges, palette, edge_chunk=None):
        cn, en = colors.numpy(), edges.numpy()
        cand, k = teacher_offer(cn, rc.numpy(), en, palette.numpy())
        part = (cn < 0) & (k > 0) & (rp.numpy() < .5)
        commit = exact_commit_labels(cn, en, part, cand)
        return SimpleNamespace(participate=torch.from_numpy(part), candidate=torch.from_numpy(cand),
                               commit_logit=torch.from_numpy(np.where(commit, 1., -1.)))

    def commit_logits(self, colors, edges, palette, participate, candidate, edge_chunk=None):
        truth = exact_commit_labels(colors.numpy(), edges.numpy(), participate.numpy(), candidate.numpy())
        return torch.from_numpy(np.where(truth, 1., -1.))


def test_same_state_and_draws_give_zero_errors_for_exact_fixture():
    reference, learned, counts, examples = b.coupled_batch(nx.cycle_graph(12), [78, 79, 80], {1: ExactFixture()}, plan(max_rounds=128))
    np.testing.assert_array_equal(reference["colors"], learned[1]["colors"])
    np.testing.assert_array_equal(reference["rounds"], learned[1]["rounds"])
    assert not learned[1]["first_action_round"].any()
    assert not learned[1]["first_decision_round"].any()
    assert not examples
    for row in counts.values():
        for name in ("participation_errors", "candidate_errors", "action_errors", "shared_commit_errors", "unsafe_commits"):
            assert row[name] == 0


class UnsafeFixture(UniversalColorGNN):
    def forward(self, colors, rp, rc, edges, palette, edge_chunk=None):
        out = super().forward(colors, rp, rc, edges, palette, edge_chunk)
        # On a 3-path, color the two adjacent vertices first, then the last.
        index = torch.arange(len(colors)) % 3
        first_vertex = torch.arange(len(colors)) // 3 * 3
        part = (colors < 0) & ((index < 2) | (colors[first_vertex] >= 0))
        return replace(out, participate=part, candidate=torch.where(colors < 0, 0, -1),
                       commit_logit=torch.ones_like(out.commit_logit))


def test_unsafe_e1_actions_are_unfiltered_and_collision_does_not_stop_rollout():
    graph, model, seeds = nx.path_graph(3), UnsafeFixture().eval(), [101, 214]
    reference, learned, counts, _ = b.coupled_batch(graph, seeds, {0: model}, plan(max_rounds=128))
    for i, seed in enumerate(seeds):
        original, colors = run_rollout(model, len(graph), _edge_index(graph), 4, seed, 128, diagnostics=False)
        assert original["rounds"] > 1
        np.testing.assert_array_equal(learned[0]["colors"][i], colors)
        assert learned[0]["rounds"][i] == original["rounds"]
        assert learned[0]["status"][i] == b.COLLISION
        assert np.all(colors == 0)
    assert counts[(0, "gnn_states", -1)]["unsafe_commits"] > 0
    assert np.all(reference["status"] == b.SUCCESS)


def test_shared_commit_receives_exact_common_proposals_and_queries_do_not_change_actions():
    class Recorder(UniversalColorGNN):
        def __init__(self):
            super().__init__()
            self.queries = []

        def commit_logits(self, colors, edges, palette, part, cand, edge_chunk=None):
            self.queries.append((colors.clone(), part.clone(), cand.clone()))
            return super().commit_logits(colors, edges, palette, part, cand, edge_chunk)

    model = Recorder().eval()
    graph, seeds = nx.cycle_graph(8), [71, 72]
    _, learned, _, _ = b.coupled_batch(graph, seeds, {0: model}, plan(max_rounds=1))
    # One model forward evaluates its own proposals, then the explicit shared query.
    assert len(model.queries) == 2
    colors, part, cand = model.queries[1]
    rngs = [np.random.default_rng(s) for s in seeds]
    rp, rc = b.draws(rngs, 8)
    initial = np.full((2, 8), -1, np.int64)
    _, expected_p, expected_c, _, _, _ = b.truth_from_draws(initial, rp, rc, _edge_index(graph))
    np.testing.assert_array_equal(part.numpy().reshape(4, 8), np.concatenate((expected_p, expected_p)))
    np.testing.assert_array_equal(cand.numpy().reshape(4, 8), np.concatenate((expected_c, expected_c)))
    for i, seed in enumerate(seeds):
        _, original_colors = run_rollout(model, 8, _edge_index(graph), 4, seed, 1, diagnostics=False)
        np.testing.assert_array_equal(learned[0]["colors"][i], original_colors)


def test_saved_local_counterexamples_replay_with_original_float64_tapes():
    graph, model = nx.cycle_graph(8), UniversalColorGNN().eval()
    _, _, _, examples = b.coupled_batch(graph, [12, 13], {0: model}, plan(max_rounds=8))
    assert examples
    for example in examples.values():
        colors = np.array(example["colors_before"], np.int64)
        rp, rc = np.array(example["activation_draw"]), np.array(example["candidate_draw"])
        _, p, c, commit, _, _ = b.truth_from_draws(colors[None], rp[None], rc[None], _edge_index(graph))
        np.testing.assert_array_equal(p[0], example["truth_p"])
        np.testing.assert_array_equal(c[0], example["truth_c"])
        np.testing.assert_array_equal(commit[0], example["truth_commit"])
        with torch.inference_mode():
            out = model(torch.from_numpy(colors), torch.from_numpy(rp.astype(np.float32)),
                        torch.from_numpy(rc.astype(np.float32)), torch.from_numpy(_edge_index(graph)), torch.full((8,), 4))
        np.testing.assert_array_equal(out.participate.numpy(), example["pred_p"])
        np.testing.assert_array_equal(out.candidate.numpy(), example["pred_c"])
        np.testing.assert_array_equal((out.commit_logit > 0).numpy(), example["pred_commit"])


def test_partial_coloring_and_conflicts_are_preserved_in_terminal_status():
    edges = _edge_index(nx.path_graph(3))
    colors = np.array([[0, 1, 0], [0, 0, -1], [0, -1, 1]])
    np.testing.assert_array_equal(b.terminal_status(colors, edges), [b.SUCCESS, b.COLLISION, b.TIMEOUT])


def test_tv_bound_has_exact_zero_mismatch_binomial_value_and_no_false_certificate():
    assert stats.coupling_bound(0, 100, .05) == pytest.approx(1 - .05 ** (1 / 100))
    assert stats.coupling_bound(256, 256) == 1
    bound = stats.coupling_bound(0, 256, .05 / (83 * 5 * 2))
    assert bound < .05


def test_mmd_statistic_equals_direct_unbiased_formula_and_detects_terminal_failures():
    a = np.zeros((8, 3), np.int64)
    bb = np.ones((8, 3), np.int64)
    p = plan(permutations=199)
    value = stats.kernel_test(a, bb, np.zeros(8), np.zeros(8), p, np.random.default_rng(12))
    assert value["mmd2"] == pytest.approx(2 - 2 * np.mean(np.exp(-np.array(p["kernel_scales"]))))
    failures = stats.kernel_test(a, a, np.zeros(8), np.ones(8), p, np.random.default_rng(12))
    assert failures["mmd2"] == pytest.approx(2)
    assert failures["p_value"] <= .01


def test_color_quotient_keeps_vertex_labels_and_uncolored_status():
    colors = np.array([-1, 2, 1, 2, 0])
    np.testing.assert_array_equal(stats.canonical_colors(colors), [-1, 0, 1, 0, 2])
    assert not np.array_equal(stats.canonical_colors(colors), stats.canonical_colors(colors[::-1]))


def test_no_partial_execution_can_be_summarized(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"protocol": b.PROTOCOL}))
    (tmp_path / "execution.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(RuntimeError, match="Complete authenticated E1"):
        b.summarize(tmp_path, tmp_path / "output")


def test_technical_pipeline_freezes_resumes_and_summarizes_without_scientific_weights(tmp_path, monkeypatch):
    # All fixture data stay under pytest's temporary directory. This explicitly
    # bypasses the loader for a technical test and cannot produce E1 evidence.
    inputs = {**b.inspect_inputs(tmp_path), "ready": True, "technical_fixture": True}
    models = {s: ExactFixture() for s in range(5)}
    graphs = []
    for family, graph in (("REGULAR", nx.cycle_graph(8)), ("IRREGULAR", nx.path_graph(8))):
        graphs.append((graph, {"id": family, "family": family, "n": 8, "role": "random"}))
    monkeypatch.setattr(b, "load_models", lambda *args: (models, inputs))
    monkeypatch.setattr(b, "inspect_inputs", lambda *args: inputs)
    monkeypatch.setattr(b, "build_graphs", lambda *args: (graphs, {"technical_fixture": True}))
    path, root, output = tmp_path / "plan.json", tmp_path / "raw", tmp_path / "output"
    p = plan(samples=4, max_rounds=64, batch_tapes=2, permutations=19)
    path.write_text(json.dumps(p))
    b.run(path, root, tmp_path)
    original_hashes = {str(f): b.file_hash(f) for f in root.glob("graphs/*")}
    b.run(path, root, tmp_path)
    assert original_hashes == {str(f): b.file_hash(f) for f in root.glob("graphs/*")}
    b.summarize(root, output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["graphs"] == 2 and summary["rollouts"] == 56
    assert summary["primary_tests"] == 20
    assert (output / "local-summary.csv").exists()
    # Resume must reject a changed protocol before any results are overwritten.
    path.write_text(json.dumps({**p, "namespace": p["namespace"] + 1}))
    with pytest.raises(RuntimeError, match="Existing manifest differs"):
        b.run(path, root, tmp_path)
    assert original_hashes == {str(f): b.file_hash(f) for f in root.glob("graphs/*")}
