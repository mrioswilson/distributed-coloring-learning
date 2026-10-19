"""Tests for the Universal Color GNN (protocol ucgnn-v1)."""
from dataclasses import replace
import inspect
import json

import networkx as nx
import numpy as np
import pytest
import torch

from dcl.counterfactual import exact_commit_labels as m6_exact_labels
from dcl.m4_data import _edge_index, encode_m4_state
from dcl.oracle import compute_commit_mask, run_oracle
from dcl.study.config import Config, seed_for
from dcl.study.evaluation import evaluation_cases as m6_evaluation_cases
from dcl.study.graphs import fingerprint, generate_graph
from dcl.universal.config import UConfig
from dcl.universal.data import collect_examples, examples_from_trace, make_batch
from dcl.universal.evaluation import build_case, cohort_cases, run_rollout, teacher_rollout
from dcl.universal.model import NEG, UniversalColorGNN, count_parameters, neighbor_max
from dcl.universal.rules import (available_matrix, conflict_flags, exact_commit_labels,
                                 make_counterfactual_proposals, teacher_offer)
from dcl.universal.train import initialize, load_model, train, training_step

TINY = dict(graphs_per_degree=(3,), tapes=1, epochs=3, warmup=1, batch_size=8, val_graphs_per_degree=2,
            val_rollouts=1, val_sizes=(40,), val_every=1, max_rounds=12, checkpoint_epochs=(1, 3))


def torch_inputs(colors, rp, rc, edges, palette):
    n = len(colors)
    return (torch.as_tensor(colors, dtype=torch.long), torch.as_tensor(rp, dtype=torch.float32),
            torch.as_tensor(rc, dtype=torch.float32), torch.as_tensor(edges, dtype=torch.long),
            torch.full((n,), palette, dtype=torch.long))


def random_state(graph, palette, rng, colored_fraction=0.5):
    """A proper partial coloring (greedy over a random order) plus a fresh tape."""
    n = len(graph)
    colors = np.full(n, -1, np.int64)
    for v in rng.permutation(n):
        if rng.random() < colored_fraction:
            used = {colors[u] for u in graph.neighbors(v)}
            free = [c for c in range(palette) if c not in used]
            colors[v] = rng.choice(free)
    return colors, rng.random(n), rng.random(n)


def trained_tiny(tmp_path, **changes):
    config = UConfig(**{**TINY, **changes})
    train(config, tmp_path, "cpu", check_resources=False)
    return config, tmp_path / "runs" / config.id


# 1, 2. Variable palettes and variable candidate cardinality ---------------------------------

@pytest.mark.parametrize("degree", [1, 3, 5, 8, 10, 15])
def test_variable_palettes_single_model(degree):
    torch.manual_seed(0)
    model = UniversalColorGNN()
    graph = generate_graph(40, degree, "REGULAR", 11) if degree > 1 else nx.path_graph(2)
    colors, rp, rc = random_state(graph, degree + 1, np.random.default_rng(degree))
    out = model(*torch_inputs(colors, rp, rc, _edge_index(graph), degree + 1))
    assert out.candidate_logits.shape == (len(graph), degree + 1)
    assert out.participation_logit.shape == out.commit_logit.shape == (len(graph),)
    uncolored = torch.as_tensor(colors < 0)
    assert bool(((out.candidate >= 0) & (out.candidate <= degree))[uncolored].all())
    assert bool((out.candidate[~uncolored] == -1).all()) and not bool(out.participate[~uncolored].any())


def test_mixed_palette_batch_equals_separate_evaluation():
    torch.manual_seed(1)
    model = UniversalColorGNN()
    rng = np.random.default_rng(3)
    examples, separate = [], []
    for degree in (3, 10, 5):
        graph = generate_graph(40, degree, "IRREGULAR", 100 + degree)
        result = run_oracle(graph, 7 + degree, num_colors=degree + 1)
        example = examples_from_trace(graph, result.trace, degree)[1]
        examples.append(example)
        batch = make_batch([example], "cpu")
        separate.append(model(*batch.inputs()))
    joint = model(*make_batch(examples, "cpu").inputs())
    assert joint.candidate_logits.shape[1] == 11
    offset = 0
    for example, out in zip(examples, separate):
        n, k = len(example.colors), example.palette
        rows = slice(offset, offset + n)
        torch.testing.assert_close(joint.candidate_logits[rows, :k], out.candidate_logits, rtol=1e-5, atol=1e-5)
        assert bool((joint.candidate_logits[rows, k:] == NEG).all())
        assert torch.equal(joint.candidate[rows], out.candidate)
        assert torch.equal(joint.participate[rows], out.participate)
        torch.testing.assert_close(joint.commit_logit[rows], out.commit_logit, rtol=1e-5, atol=1e-5)
        offset += n


# 3. Graphs and cohorts -----------------------------------------------------------------------

@pytest.mark.parametrize("degree", [3, 4, 5, 7, 8, 10, 12, 15])
@pytest.mark.parametrize("family", ["REGULAR", "IRREGULAR"])
def test_graph_connectivity_and_degree(degree, family):
    graph = generate_graph(160, degree, family, 913)
    assert nx.is_connected(graph) and set(graph) == set(range(160))
    degrees = set(dict(graph.degree()).values())
    assert max(degrees) <= degree
    assert degrees == {degree} if family == "REGULAR" else len(degrees) > 1
    edges = _edge_index(graph)
    assert edges.shape == (2, 2 * graph.number_of_edges())


@pytest.mark.parametrize("degree", [3, 10])
def test_final_cohort_is_the_frozen_m6b_cohort(degree):
    counts = {40: 3, 160: 2}
    ours = list(cohort_cases(degree, (40, 160), counts))
    theirs = [meta for _, meta in m6_evaluation_cases(Config(degree=degree), "final_test", (40, 160), counts)]
    assert [(a["graph_id"], a["graph_seed"], a["tape_seed"]) for a in ours] == \
           [(b["graph_id"], b["graph_seed"], b["tape_seed"]) for b in theirs]
    for a, b in zip(ours, theirs):
        assert build_case(a)[1]["fingerprint"] == b["fingerprint"]


def test_training_data_reuses_m6b_streams():
    config = UConfig(degrees=(3, 10), graphs_per_degree=(2, 2), tapes=2, val_graphs_per_degree=2)
    _, records = collect_examples(config, "train")
    for record in records:
        seed = seed_for(0, record["degree"], "train/graph", record["graph_i"])
        assert record["graph_seed"] == seed
        assert record["fingerprint"] == fingerprint(generate_graph(40, record["degree"], "REGULAR", seed))
        assert [t["tape_seed"] for t in record["trajectories"]] == \
               [seed_for(0, record["degree"], "train/tape", record["graph_i"], j) for j in range(2)]


# 4. Teacher labels and encoded states ---------------------------------------------------------

@pytest.mark.parametrize("degree", [3, 5, 8, 10])
def test_teacher_labels_match_encoded_states(degree):
    config = UConfig(degrees=(degree,), graphs_per_degree=(3,), tapes=2, val_graphs_per_degree=2)
    examples, _ = collect_examples(config, "train")
    for e in examples:
        offer, k = teacher_offer(e.colors, e.rc.astype(np.float64), e.edge_index, e.palette)
        uncolored = e.colors < 0
        assert np.all(k[uncolored] >= 1)
        # The float32 copy of r_c given to the model could only differ from the
        # teacher's float64 draw within ~2^-24 of a boundary j/k.
        assert np.array_equal(offer[uncolored], e.y_offer[uncolored])
        assert np.array_equal(e.y_participate > 0.5, uncolored & (e.rp < 0.5))
        labels = exact_commit_labels(e.colors, e.edge_index, e.y_participate > 0.5, e.y_offer)
        assert np.array_equal(labels, e.y_commit > 0.5)
        assert np.all(e.y_offer[~uncolored] == -1) and np.all(e.y_offer[uncolored] < e.palette)


@pytest.mark.parametrize("degree", [3, 8])
@pytest.mark.parametrize("family", ["REGULAR", "IRREGULAR"])
def test_vectorized_teacher_equals_oracle(degree, family):
    for seed in range(4):
        graph = generate_graph(160, degree, family, 50 + seed)
        reference = run_oracle(graph, 900 + seed, num_colors=degree + 1, save_trace=False)
        colors, rounds = teacher_rollout(160, _edge_index(graph), degree + 1, 900 + seed)
        assert rounds == reference.rounds and np.array_equal(colors, reference.colors)


# 5-9. Exact COMMIT labels, counterfactuals and conflict cases ---------------------------------

@pytest.mark.parametrize("degree", [3, 5, 10])
def test_exact_labels_match_oracle_and_m6b(degree):
    q, rng = degree + 1, np.random.default_rng(541)
    graph = generate_graph(40, degree, "IRREGULAR", 945)
    edges = _edge_index(graph)
    for _ in range(40):
        colors, rp, rc = random_state(graph, q, rng)
        part = (colors < 0) & (rng.random(40) < 0.6)
        cand = np.where(colors < 0, rng.integers(0, q, 40), -1)
        ours = exact_commit_labels(colors, edges, part, cand)
        assert np.array_equal(ours, compute_commit_mask(graph, colors, part, cand))
        x = encode_m4_state(graph, colors, rp, rc, q).x
        assert np.array_equal(ours, m6_exact_labels(x, edges, part, cand + 1) > 0.5)


def test_counterfactuals_cover_cases_and_labels_are_exact():
    rng = np.random.default_rng(5)
    seen = dict(fixed=0, simultaneous=0, valid=0, k1=0)
    for degree in (3, 10):
        q = degree + 1
        graph = generate_graph(40, degree, "REGULAR", 17 + degree)
        edges = _edge_index(graph)
        result = run_oracle(graph, 3, num_colors=q)
        for e in examples_from_trace(graph, result.trace, degree):
            part, cand = make_counterfactual_proposals(e.colors, edges, q, e.y_participate > 0.5, e.y_offer, rng)
            labels = exact_commit_labels(e.colors, edges, part, cand)
            assert np.array_equal(labels, compute_commit_mask(graph, e.colors, part, cand))
            assert np.all(cand[e.colors >= 0] == -1) and np.all(cand < q) and not part[e.colors >= 0].any()
            fixed, simultaneous = conflict_flags(e.colors, edges, part, cand)
            seen["fixed"] += int((part & fixed).sum())
            seen["simultaneous"] += int((part & simultaneous).sum())
            seen["valid"] += int(labels.sum())
            _, k = teacher_offer(e.colors, e.rc, edges, q)
            seen["k1"] += int(((k == 1) & (e.colors < 0)).sum())
    assert all(v > 0 for v in seen.values()), seen


def test_counterfactual_does_not_modify_teacher_inputs():
    graph = generate_graph(40, 3, "REGULAR", 4)
    e = examples_from_trace(graph, run_oracle(graph, 4).trace, 3)[2]
    before = (e.colors.copy(), e.y_participate.copy(), e.y_offer.copy())
    make_counterfactual_proposals(e.colors, e.edge_index, 4, e.y_participate > 0.5, e.y_offer, np.random.default_rng(0))
    assert all(np.array_equal(a, b) for a, b in zip(before, (e.colors, e.y_participate, e.y_offer)))


def path_case(colors, part, cand):
    edges = _edge_index(nx.path_graph(len(colors)))
    return exact_commit_labels(np.array(colors), edges, np.array(part, bool), np.array(cand))


def test_conflict_with_colored_neighbor():
    assert path_case([0, -1], [False, True], [-1, 0]).tolist() == [False, False]
    assert path_case([0, -1], [False, True], [-1, 1]).tolist() == [False, True]


def test_simultaneous_conflicts():
    assert path_case([-1, -1], [True, True], [1, 1]).tolist() == [False, False]
    # A non-participating neighbor offering the same color does not block.
    assert path_case([-1, -1], [True, False], [1, 1]).tolist() == [True, False]
    # Conflicts are only between neighbors: 0-1-2 with 0 and 2 proposing 2.
    assert path_case([-1, -1, -1], [True, False, True], [2, 2, 2]).tolist() == [True, False, True]


def test_valid_proposals_and_no_candidate():
    assert path_case([-1, -1, 3], [True, True, False], [0, 1, -1]).tolist() == [True, True, False]
    assert path_case([-1, -1], [True, True], [-1, 2]).tolist() == [False, True]


def test_single_available_color_k1():
    star = nx.star_graph(4)  # center 0 with neighbors colored 0, 1, 2, 4 -> only color 3 free
    colors = np.array([-1, 0, 1, 2, 4])
    for rc in (0.0, 0.3, 0.999):
        offer, k = teacher_offer(colors, np.full(5, rc), _edge_index(star), 5)
        assert k[0] == 1 and offer[0] == 3
    assert exact_commit_labels(colors, _edge_index(star), [True] + [False] * 4, [3, -1, -1, -1, -1])[0]
    assert not exact_commit_labels(colors, _edge_index(star), [True] + [False] * 4, [4, -1, -1, -1, -1])[0]


# 10. Gradients and updates ---------------------------------------------------------------------

@pytest.mark.parametrize("epoch", [1, 5])
@pytest.mark.parametrize("variant", ["NORMALIZED", "RAW_PREFIX"])
def test_all_parameters_receive_gradients_and_update(epoch, variant):
    config = UConfig(degrees=(3, 10), graphs_per_degree=(2, 2), tapes=1, warmup=2, variant=variant)
    model, optimizer, _, cf_rng = initialize(config, "cpu")
    examples, _ = collect_examples(config, "train")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    metrics = training_step(model, optimizer, make_batch(examples[:32], "cpu"), config, epoch, cf_rng)
    assert np.isfinite(metrics["loss"])
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name
        assert not torch.equal(before[name], parameter.detach()), name


# 11. No shield ------------------------------------------------------------------------------------

class GreedyBadModel(torch.nn.Module):
    """Everyone always participates with color 0 and commits: unsafe by design."""

    def __init__(self):
        super().__init__()
        self.message1 = torch.nn.Sequential(torch.nn.Linear(2, 32))
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, colors, rp, rc, edges, palette, edge_chunk=None):
        n, k = len(colors), int(palette.max())
        uncolored = colors < 0
        from dcl.universal.model import UOutput
        zeros = torch.zeros(n, k)
        return UOutput(torch.ones(n), zeros, torch.ones(n), uncolored, torch.where(uncolored, 0, -1),
                       zeros, torch.zeros(n), zeros, zeros)


def test_rollout_executes_unsafe_actions_without_shield():
    graph = generate_graph(40, 3, "REGULAR", 2)
    result, colors = run_rollout(GreedyBadModel(), 40, _edge_index(graph), 4, 9, max_rounds=5)
    assert result["finished"] and not result["success"]
    assert np.all(colors == 0) and result["monochromatic_edges"] == graph.number_of_edges()
    assert result["unsafe_commits"] == 40 and result["simultaneous_conflicts"] == 40


def test_rollout_never_calls_teacher_without_diagnostics(monkeypatch):
    import dcl.universal.evaluation as evaluation
    model = UniversalColorGNN()
    graph = generate_graph(40, 5, "IRREGULAR", 8)
    reference, ref_colors = run_rollout(model, 40, _edge_index(graph), 6, 21, max_rounds=20, diagnostics=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("teacher rule used by the learned action")
    for name in ("available_matrix", "conflict_flags", "exact_commit_labels", "teacher_offer"):
        monkeypatch.setattr(evaluation, name, forbidden)
    monkeypatch.setattr("dcl.oracle.compute_commit_mask", forbidden)
    monkeypatch.setattr("dcl.oracle.sample_random_trial_details", forbidden)
    result, colors = run_rollout(model, 40, _edge_index(graph), 6, 21, max_rounds=20, diagnostics=False)
    assert np.array_equal(colors, ref_colors) and result["rounds"] == reference["rounds"]


# 12. Node permutation ------------------------------------------------------------------------------

def test_node_permutation_equivariance():
    torch.manual_seed(4)
    model = UniversalColorGNN()
    rng = np.random.default_rng(6)
    graph = generate_graph(80, 8, "IRREGULAR", 33)
    colors, rp, rc = random_state(graph, 9, rng)
    edges = _edge_index(graph)
    perm = rng.permutation(80)          # node v becomes perm[v]
    inverse = np.argsort(perm)
    out = model(*torch_inputs(colors, rp, rc, edges, 9))
    permuted = model(*torch_inputs(colors[inverse], rp[inverse], rc[inverse], perm[edges], 9))
    torch.testing.assert_close(permuted.candidate_logits[perm], out.candidate_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(permuted.commit_logit[perm], out.commit_logit, rtol=1e-5, atol=1e-5)
    assert torch.equal(permuted.candidate[perm], out.candidate)
    # Closed loop: tape draws attached to the original nodes.
    result, final = run_rollout(model, 80, edges, 9, 77, max_rounds=30)
    result_p, final_p = run_rollout(model, 80, perm[edges], 9, 77, max_rounds=30, permutation=perm)
    assert np.array_equal(final_p[perm], final) and result_p["rounds"] == result["rounds"]


def test_commit_is_equivariant_to_color_permutations():
    torch.manual_seed(2)
    model = UniversalColorGNN()
    rng = np.random.default_rng(8)
    graph = generate_graph(40, 5, "REGULAR", 3)
    colors, rp, rc = random_state(graph, 6, rng)
    part = torch.as_tensor((colors < 0) & (rng.random(40) < 0.7))
    cand = torch.as_tensor(np.where(colors < 0, rng.integers(0, 6, 40), -1))
    sigma = torch.as_tensor(rng.permutation(6))
    relabel = lambda c: torch.where(c >= 0, sigma[c.clamp_min(0)], c)
    c_t, _, _, e_t, p_t = torch_inputs(colors, rp, rc, _edge_index(graph), 6)
    base = model.commit_logits(c_t, e_t, p_t, part, cand)
    moved = model.commit_logits(relabel(c_t), e_t, p_t, part, relabel(cand))
    torch.testing.assert_close(base, moved, rtol=1e-5, atol=1e-6)


# 13. Degree-independent parameters ----------------------------------------------------------------

def test_parameters_do_not_depend_on_palette():
    assert "num_colors" not in inspect.signature(UniversalColorGNN.__init__).parameters
    model = UniversalColorGNN()
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    count = count_parameters(model)
    for degree in (3, 5, 8, 10, 20):
        graph = generate_graph(60, degree, "REGULAR", 1)
        model(*torch_inputs(np.full(60, -1), np.random.rand(60), np.random.rand(60), _edge_index(graph), degree + 1))
        assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == shapes
        assert count_parameters(model) == count == 16035


# 14. Reproducibility and exact resume -----------------------------------------------------------

def test_reproducible_training_and_exact_resume(tmp_path):
    config, run_a = trained_tiny(tmp_path / "a", degrees=(3,))
    _, run_b = trained_tiny(tmp_path / "b", degrees=(3,))
    a = torch.load(run_a / "last.pt", weights_only=False)
    b = torch.load(run_b / "last.pt", weights_only=False)
    for key in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][key], b["model_state_dict"][key])
    assert json.loads((run_a / "history.json").read_text())[-1]["train"] == json.loads((run_b / "history.json").read_text())[-1]["train"]
    assert train(config, tmp_path / "c", "cpu", stop_after_epoch=2, check_resources=False) == "paused"
    assert train(config, tmp_path / "c", "cpu", check_resources=False) == "complete"
    c = torch.load(tmp_path / "c" / "runs" / config.id / "last.pt", weights_only=False)
    for key in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][key], c["model_state_dict"][key])
    assert (run_a / "epoch-001.pt").exists() and (run_a / "epoch-003.pt").exists() and (run_a / "best.pt").exists()


# 15. Same checkpoint on Delta=3 and Delta=10 -------------------------------------------------------

def test_same_checkpoint_runs_on_degree_3_and_10(tmp_path):
    _, run = trained_tiny(tmp_path, degrees=(3,))
    model, saved = load_model(run / "last.pt")
    for degree in (3, 10):
        metadata = next(cohort_cases(degree, (40,), {40: 1}, split="unit", namespace=3))
        edges, _ = build_case(metadata)
        result, colors = run_rollout(model, 40, edges, degree + 1, metadata["tape_seed"], 30)
        assert result["states"] > 0 and colors.max() <= degree
    assert count_parameters(model) == saved["parameters"]


# Leakage and locality ----------------------------------------------------------------------------

def test_model_inputs_exclude_teacher_labels():
    params = list(inspect.signature(UniversalColorGNN.forward).parameters)
    assert params == ["self", "colors", "rp", "rc", "edge_index", "palette", "edge_chunk"]
    graph = generate_graph(40, 3, "REGULAR", 4)
    examples = examples_from_trace(graph, run_oracle(graph, 4).trace, 3)
    batch = make_batch(examples[:3], "cpu")
    model = UniversalColorGNN()
    reference = model(*batch.inputs())
    for name in ("y_participate", "y_offer", "y_commit"):
        value = getattr(batch, name)
        setattr(batch, name, torch.flip(value, (0,)))
    altered = model(*batch.inputs())
    assert torch.equal(reference.candidate_logits, altered.candidate_logits)
    assert torch.equal(reference.commit_logit, altered.commit_logit)


def test_rollout_model_inputs_are_state_and_tape_only():
    graph = generate_graph(40, 3, "IRREGULAR", 9)
    model = UniversalColorGNN()
    calls = []
    original = model.forward

    def spy(colors, rp, rc, edges, palette, edge_chunk=None):
        calls.append((colors.clone(), rp.clone(), rc.clone()))
        return original(colors, rp, rc, edges, palette, edge_chunk)
    model.forward = spy
    result, _ = run_rollout(model, 40, _edge_index(graph), 4, 55, max_rounds=10)
    rng = np.random.default_rng(55)
    for colors, rp, rc in calls:
        assert torch.equal(rp, torch.from_numpy(rng.random(40).astype(np.float32)))
        assert torch.equal(rc, torch.from_numpy(rng.random(40).astype(np.float32)))
        assert set(colors.tolist()) <= set(range(-1, 4))


def test_messages_are_local():
    torch.manual_seed(5)
    model = UniversalColorGNN()
    graph = nx.path_graph(7)   # 0-1-2-3-4-5-6, inspect node 0
    edges = _edge_index(graph)
    rng = np.random.default_rng(0)
    colors, rp, rc = np.full(7, -1), rng.random(7), rng.random(7)
    base = model(*torch_inputs(colors, rp, rc, edges, 4))
    # Distance 3 and beyond: nothing about node 0 changes.
    far = colors.copy(); far[3:] = [1, 0, 1, 0]
    rp_far, rc_far = rp.copy(), rc.copy(); rp_far[3:] = 1 - rp[3:]; rc_far[3:] = 1 - rc[3:]
    out = model(*torch_inputs(far, rp_far, rc_far, edges, 4))
    assert torch.equal(out.candidate_logits[0], base.candidate_logits[0])
    assert torch.equal(out.commit_logit[0], base.commit_logit[0])
    # Distance 2: candidate of node 0 (one communication) is unchanged.
    two = colors.copy(); two[2] = 3
    out = model(*torch_inputs(two, rp, rc, edges, 4))
    assert torch.equal(out.candidate_logits[0], base.candidate_logits[0])
    assert torch.equal(out.participation_logit[0], base.participation_logit[0])
    # neighbor_max only reads neighbors.
    z = torch.rand(7, 5)
    changed = z.clone(); changed[2:] = 10.0
    assert torch.equal(neighbor_max(z, torch.as_tensor(edges))[0], neighbor_max(changed, torch.as_tensor(edges))[0])


def test_chunked_aggregation_is_identical():
    graph = generate_graph(160, 10, "IRREGULAR", 2)
    edges = torch.as_tensor(_edge_index(graph))
    z = torch.relu(torch.randn(160, 11 * 32))
    assert torch.equal(neighbor_max(z, edges), neighbor_max(z, edges, chunk=97))
    model = UniversalColorGNN()
    colors, rp, rc = random_state(graph, 11, np.random.default_rng(1))
    a = model(*torch_inputs(colors, rp, rc, edges.numpy(), 11))
    b = model(*torch_inputs(colors, rp, rc, edges.numpy(), 11), edge_chunk=101)
    assert torch.equal(a.candidate_logits, b.candidate_logits) and torch.equal(a.commit_logit, b.commit_logit)


def test_availability_matrix_and_palette_mask():
    star = nx.star_graph(3)
    available = available_matrix(np.array([-1, 0, 2, -1]), _edge_index(star), 4)
    assert available[0].tolist() == [False, True, False, True]
    model = UniversalColorGNN()
    out = model(*torch_inputs(np.array([-1, 0, 2, -1]), np.zeros(4), np.zeros(4), _edge_index(star), 4))
    assert bool((out.availability <= 1).all()) and out.availability.shape == (4, 4)
