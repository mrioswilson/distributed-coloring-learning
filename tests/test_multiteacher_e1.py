"""Scientific-interface and reproducibility checks for the E1 follow-up."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dcl.m4_data import _edge_index
from dcl.multiteacher.data import build_pool
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import (Choices, ColoringProgram, DETECT, KW, LINIAL, Snapshot, TEACHERS,
                                      teacher_choices)
from dcl.multiteacher_e1.experiment import (initialize, metadata, predictions, rollout_many,
                                           train_one)
from dcl.multiteacher_e1.model import RootedE1, native_prediction, parameters
from dcl.multiteacher_e1.supervision import (commit_targets, labels_from_inputs, loss_one,
                                            phase_groups, tensor_arrays)
from dcl.universal.rules import exact_commit_labels

torch.set_num_threads(1)


def roots_state(graph, teacher="LUBY", seed=17, steps=1):
    program = ColoringProgram(graph, teacher, seed)
    for _ in range(steps):
        state = program.snapshot()
        program.apply(state, teacher_choices(state, program.neighbors, teacher))
        if program.finished:
            break
    if program.finished:
        program = ColoringProgram(graph, teacher, seed)
    state = program.snapshot()
    roots = np.flatnonzero(state.permission)
    arrays = encode_roots(state, program.neighbors, roots)
    choices = teacher_choices(state, program.neighbors, teacher)
    labels = np.column_stack((choices.color[roots], choices.auxiliary[roots], choices.mark[roots],
                              np.full(len(roots), state.phase))).astype(np.int64)
    return program, state, roots, arrays, labels


@pytest.mark.parametrize("teacher", ("LUBY", "JOHANSSON"))
@pytest.mark.parametrize("graph", (nx.cycle_graph(12), nx.path_graph(12), nx.complete_graph(4)))
def test_rooted_core_matches_original_e1_logits_and_gradients(teacher, graph):
    torch.manual_seed(73)
    model = RootedE1(teacher)
    original = copy.deepcopy(model.core)
    program, state, roots, arrays, labels = roots_state(graph, teacher)
    assert parameters(model) == {"core": 16035, "total": 16035}
    colors = torch.from_numpy(state.colors)
    edges = torch.from_numpy(_edge_index(graph))
    out = original(colors, torch.from_numpy(state.draws[:, 0]), torch.from_numpy(state.draws[:, 1]),
                   edges, torch.full_like(colors, 4))
    rooted = model(*tensor_arrays(arrays), phase=state.phase)
    torch.testing.assert_close(rooted.state.p_logits[:, 0], out.participation_logit[roots], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(rooted.scores, out.candidate_logits[roots], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(rooted.commit, out.commit_logit[roots], atol=2e-6, rtol=2e-6)
    assert np.array_equal(rooted.candidates[:, 0].detach().numpy(), out.candidate[roots].detach().numpy())
    # Matching root losses must also yield the original graph's parameter gradients.
    loss = (rooted.scores.square().mean() + rooted.commit.square().mean()
            + rooted.state.p_logits[:, 0].square().mean())
    full_loss = (out.candidate_logits[roots].square().mean() + out.commit_logit[roots].square().mean()
                 + out.participation_logit[roots].square().mean())
    loss.backward(); full_loss.backward()
    for (_, actual), (_, expected) in zip(model.core.named_parameters(), original.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("teacher", ("LUBY", "JOHANSSON", "GREEDY"))
def test_canonical_e1_cannot_read_native_auxiliary_features(teacher):
    model = RootedE1(teacher)
    _, state, _, arrays, _ = roots_state(nx.cycle_graph(12), teacher)
    original = model(*tensor_arrays(arrays), phase=state.phase)
    changed = [a.copy() for a in arrays]
    for index in (0, 1):
        for selection in (slice(5, 7), slice(8, 32), slice(34, 42)):
            changed[index][..., selection] = 7.
    changed[2][:] = 1.
    altered = model(*tensor_arrays(changed), phase=state.phase)
    torch.testing.assert_close(original.scores, altered.scores, rtol=0, atol=0)
    torch.testing.assert_close(original.commit, altered.commit, rtol=0, atol=0)


@pytest.mark.parametrize("teacher", TEACHERS)
@pytest.mark.parametrize("graph", (nx.path_graph(8), nx.complete_graph(4), nx.random_regular_graph(3, 12, seed=7)))
def test_e1_supervision_reproduces_every_native_teacher_step(teacher, graph):
    program = ColoringProgram(graph, teacher, 431)
    seen = set()
    while not program.finished:
        state = program.snapshot()
        roots = np.flatnonzero(state.permission)
        choices = teacher_choices(state, program.neighbors, teacher)
        arrays = encode_roots(state, program.neighbors, roots)
        labels = np.column_stack((choices.color[roots], choices.auxiliary[roots], choices.mark[roots],
                                  np.full(len(roots), state.phase))).astype(np.int64)
        labels_from_inputs(arrays, labels, teacher)
        model = RootedE1(teacher)
        out = model(*tensor_arrays(arrays), phase=state.phase)
        if state.phase == LINIAL:
            prediction = native_prediction(out)[1].detach().numpy()
            assert np.all(prediction // 13 < state.field)
        seen.add(state.phase)
        program.apply(state, choices)
    assert program.failure is None


@pytest.mark.parametrize("teacher", ("LUBY", "JOHANSSON", "JP"))
def test_own_and_counterfactual_root_labels_equal_full_graph_rule(teacher):
    graph = nx.random_regular_graph(3, 20, seed=3)
    program, state, roots, arrays, labels = roots_state(graph, teacher)
    target = labels_from_inputs(arrays, labels, teacher)
    rng = np.random.default_rng(948)
    for _ in range(10):
        p = rng.random(len(graph)) < .7
        p &= state.colors < 0
        c = rng.integers(4, size=len(graph))
        full = exact_commit_labels(state.colors, _edge_index(graph), p, c)
        centers = np.column_stack((roots, program.neighbors[roots]))
        star_p = np.r_[p, False][centers]
        star_c = np.r_[c, 0][centers]
        assert np.array_equal(commit_targets(target, star_p, star_c), full[roots])


@pytest.mark.parametrize("teacher", TEACHERS)
def test_autonomous_inference_never_calls_teacher(monkeypatch, teacher):
    import dcl.multiteacher_e1.experiment as experiment
    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher used during learned inference")
    monkeypatch.setattr(experiment, "teacher_choices", forbidden)
    torch.manual_seed(31)
    rows = rollout_many(RootedE1(teacher), teacher,
                        [(nx.path_graph(8), {"tape_seed": 19}), (nx.complete_graph(4), {"tape_seed": 29})], 6)
    assert len(rows) == 2
    for graph, row in zip((nx.path_graph(8), nx.complete_graph(4)), rows):
        colors = row["colors"]
        good = all(0 <= c < 4 for c in colors) and all(colors[u] != colors[v] for u, v in graph.edges())
        assert good == row["success"]


def tiny_config():
    plan = json.loads(open("configs/multiteacher_e1/comparison.json").read())
    plan.update(n=12, graphs=3, tapes=1, val_graphs=2, updates=6, batch_decisions=18,
                warmup_updates=2, checkpoints=[2, 6], log_every=2, seeds=[0, 1], test_counts=[[12, 2]])
    return plan


@pytest.mark.parametrize("teacher", ("LUBY", "COLOR_TRIALS", "BE_BASE_D3", "SPECULATIVE_GM"))
def test_optimizer_and_private_rng_resume_exactly(tmp_path, teacher):
    config = tiny_config()
    data_config = {**config, "protocol": config["data_protocol"]}
    pool = build_pool(data_config, tmp_path / "data", teacher, "train")
    val = build_pool(data_config, tmp_path / "data", teacher, "validation")
    info = metadata()
    continuous = train_one(config, tmp_path / "continuous", teacher, 0, pool, val, info)
    train_one(config, tmp_path / "resumed", teacher, 0, pool, val, info, stop_after=2)
    resumed = train_one(config, tmp_path / "resumed", teacher, 0, pool, val, info)
    a, b = [torch.load(p, weights_only=False) for p in (continuous, resumed)]
    def same(x, y):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for key in x: same(x[key], y[key])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for xx, yy in zip(x, y): same(xx, yy)
        else:
            assert x == y
    for key in ("model_state_dict", "optimizer_state_dict", "order_rng", "counterfactual_rng"):
        same(a[key], b[key])


def test_threaded_initialization_preserves_paired_original_e1_cores():
    config = tiny_config()
    def init(teacher):
        return initialize(config, teacher, 2)[0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        models = list(executor.map(init, TEACHERS))
    for model in models[1:]:
        for key, tensor in models[0].core.state_dict().items():
            torch.testing.assert_close(tensor, model.core.state_dict()[key], atol=0, rtol=0)


def test_parallel_training_matches_serial_weights_optimizer_and_rng(tmp_path):
    config = tiny_config()
    config.update(updates=4, checkpoints=[4], log_every=4)
    teacher = "COLOR_TRIALS"
    data_config = {**config, "protocol": config["data_protocol"]}
    pool = build_pool(data_config, tmp_path / "data", teacher, "train")
    val = build_pool(data_config, tmp_path / "data", teacher, "validation")
    info = metadata()
    serial = [train_one(config, tmp_path / "serial", teacher, seed, pool, val, info) for seed in (0, 1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel = list(executor.map(lambda seed: train_one(config, tmp_path / "parallel", teacher,
                                                           seed, pool, val, info), (0, 1)))
    for left, right in zip(serial, parallel):
        a, b = [torch.load(p, weights_only=False) for p in (left, right)]
        for name, tensor in a["model_state_dict"].items():
            torch.testing.assert_close(tensor, b["model_state_dict"][name], rtol=0, atol=0)
        for parameter, state in a["optimizer_state_dict"]["state"].items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor, b["optimizer_state_dict"]["state"][parameter][name], rtol=0, atol=0)
        assert a["order_rng"] == b["order_rng"]
        assert a["counterfactual_rng"] == b["counterfactual_rng"]


def test_kw_commit_can_distinguish_blocked_and_free_counterfactual_options():
    # Two proper auxiliary colorings. The same proposed option is blocked in
    # one and free in the other; a COMMIT classifier must observe that fact.
    model = RootedE1("BE_BASE_D3")
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
        model.core.message2[0].weight[0, 1] = 1
        model.core.message2[2].weight[0, 0] = 1
        model.core.state2[0].weight[0, 4] = 1
        model.core.state2[2].weight[0, 0] = 1
        model.core.commit_head[0].weight[0, 0] = 1
        model.core.commit_head[2].weight[0, 0] = -1
        model.core.commit_head[2].bias[0] = .5
    program = ColoringProgram(nx.path_graph(2), "BE_BASE_D3", 1)
    logits, targets = [], []
    p = torch.tensor([[True, False, False, False]])
    c = torch.zeros((1, 4), dtype=torch.long)
    for adjacent_auxiliary in (0, 1):
        state = Snapshot(colors=np.full(2, -1), aux=np.array([4, adjacent_auxiliary]),
                         marks=np.zeros(2, bool), ids=np.array([4, adjacent_auxiliary]),
                         slots=np.zeros(2, np.int64), permission=np.ones(2, bool),
                         draws=np.zeros((2, 2), np.float32), preferences=np.zeros((2, 4), np.float32),
                         phase=KW, bound=5, target=4)
        arrays = encode_roots(state, program.neighbors, np.array([0]))
        choices = teacher_choices(state, program.neighbors, "BE_BASE_D3")
        labels = np.array([[choices.color[0], choices.auxiliary[0], choices.mark[0], KW]], dtype=np.int64)
        truth = labels_from_inputs(arrays, labels, "BE_BASE_D3")
        targets.append(bool(commit_targets(truth, p.numpy(), c.numpy())[0]))
        encoded = model.first(tensor_arrays(arrays), KW)
        logits.append(float(model.commit_for(encoded, p, c).detach()[0]))
    assert targets == [False, True]
    assert [value > 0 for value in logits] == targets
