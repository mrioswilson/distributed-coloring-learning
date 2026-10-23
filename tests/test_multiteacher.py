from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from dcl.multiteacher.programs import (ColoringProgram, Choices, TEACHERS, LINIAL, KW,
    DETECT, ID_SPACE, K, teacher_choices, run_teacher, linial_plan)
from dcl.multiteacher.model import (LocalGNN, encode_snapshot, encode_roots, rooted_arrays,
    tensor_batch)
from dcl.multiteacher.data import build_pool
from dcl.oracle import compute_commit_mask
from dcl.study.graphs import generate_graph


@pytest.mark.parametrize("name", TEACHERS)
@pytest.mark.parametrize("family", ("REGULAR", "IRREGULAR", "PATH", "CYCLE", "CLIQUE"))
def test_exact_native_programs_color_degree_three_graphs(name, family):
    graph = nx.complete_graph(4) if family == "CLIQUE" else generate_graph(40, 3, family, 573)
    result = run_teacher(graph, name, 894, save_trace=True)
    assert result["success"] and np.all((result["colors"] >= 0) & (result["colors"] < 4))
    assert all(result["colors"][u] != result["colors"][v] for u, v in graph.edges())
    if name == "BE_BASE_D3":
        for state, choice in result["trace"]:
            assert all(state.aux[u] != state.aux[v] for u, v in graph.edges())
            if state.phase == LINIAL:
                assert np.all(choice.auxiliary // 13 < state.field)
            if state.phase == KW:
                assert np.all(choice.color[state.aux < state.target] == 0)


def test_luby_and_johansson_labels_match_historical_commit_rule():
    graph = generate_graph(40, 3, "REGULAR", 789)
    for name, probability in (("LUBY", .5), ("JOHANSSON", 1.)):
        program = ColoringProgram(graph, name, 937)
        while not program.finished:
            state = program.snapshot()
            choices = teacher_choices(state, program.neighbors, name)
            part, candidate = np.zeros(40, bool), np.full(40, -1)
            for v in np.flatnonzero(state.permission):
                free = [c for c in range(4) if c not in state.colors[list(graph.neighbors(v))]]
                candidate[v] = free[int(float(state.draws[v, 1]) * len(free))]
                part[v] = state.draws[v, 0] < probability
            expected = compute_commit_mask(graph, state.colors, part, candidate)
            np.testing.assert_array_equal(choices.color > 0, expected)
            np.testing.assert_array_equal(choices.color[expected], candidate[expected] + 1)
            program.apply(state, choices)


def test_jp_equals_greedy_with_the_same_descending_priorities():
    graph = generate_graph(40, 3, "REGULAR", 518)
    program = ColoringProgram(graph, "JP", 789)
    expected = np.full(40, -1)
    for v in np.argsort(program.ids)[::-1]:
        expected[v] = next(c for c in range(4) if c not in expected[list(graph.neighbors(v))])
    result = run_teacher(graph, "JP", 789)
    np.testing.assert_array_equal(result["colors"], expected)


def test_color_trials_ties_and_multiple_color_wins():
    program = ColoringProgram(nx.path_graph(2), "COLOR_TRIALS", 57)
    state = program.snapshot()
    state.preferences[:] = 1
    assert not teacher_choices(state, program.neighbors, "COLOR_TRIALS").color.any()
    state.preferences[0] = [3, 0, 3, 0]
    state.preferences[1] = [0, 3, 0, 3]
    choices = teacher_choices(state, program.neighbors, "COLOR_TRIALS")
    np.testing.assert_array_equal(choices.color, [1, 2])


def test_speculative_algorithm_really_creates_and_repairs_conflicts():
    graph = nx.complete_graph(4)
    result = run_teacher(graph, "SPECULATIVE_GM", 37, True)
    assert result["success"]
    detection = [s for s, _ in result["trace"] if s.phase == DETECT]
    assert detection and any(np.all(s.colors == 0) for s in detection)
    assert len(set(result["colors"])) == 4


@pytest.mark.parametrize("name", TEACHERS)
def test_rooted_encoding_matches_full_cached_encoding(name):
    program = ColoringProgram(generate_graph(40, 3, "IRREGULAR", 51), name, 46)
    state = program.snapshot()
    roots = np.array([0, 4, 16])
    x, r = encode_snapshot(state, program.neighbors)
    expected = rooted_arrays(x, r, program.neighbors, roots)
    actual = encode_roots(state, program.neighbors, roots)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_two_hop_gnn_equals_full_graph_message_passing():
    torch.set_num_threads(1)
    torch.manual_seed(45)
    program = ColoringProgram(generate_graph(40, 3, "IRREGULAR", 53), "JP", 465)
    state = program.snapshot()
    x, r = encode_snapshot(state, program.neighbors)
    model = LocalGNN()
    root_outputs = model(*tensor_batch(encode_roots(state, program.neighbors, np.arange(40))))
    x, r = torch.from_numpy(x), torch.from_numpy(r.astype(np.float32))
    ns = torch.from_numpy(program.neighbors)
    mask = ns < 40
    diff = x[ns, 5:6] - x[:, None, 5:6]
    m1 = model.message1(torch.cat((x[ns], x[:, None].expand_as(x[ns]), r, diff), -1))
    h = model.node1(torch.cat((x, model.aggregate(m1, mask)), -1))
    m2 = model.message2(torch.cat((h[ns], h[:, None].expand_as(h[ns]), r, diff), -1))
    h2 = model.node2(torch.cat((h, model.aggregate(m2, mask)), -1))
    full = model.color_head(h2), model.auxiliary_head(h2), model.mark_head(h2).squeeze(-1)
    for a, b in zip(root_outputs, full):
        torch.testing.assert_close(a, b[:40], rtol=1e-5, atol=1e-6)


def test_persistent_pool_batch_matches_program_and_is_reusable(tmp_path):
    config = {"protocol": "native-policy-v1", "namespace": 998877, "n": 8, "graphs": 2, "tapes": 1, "val_graphs": 1}
    pool = build_pool(config, tmp_path, "JP", "train")
    arrays, labels = pool.batch(np.arange(min(4, len(pool.roots))))
    assert arrays[0].shape[1:] == (4, 42) and labels.shape[1] == 4
    same = build_pool(config, tmp_path, "JP", "train")
    assert pool.manifest == same.manifest
    validation = build_pool(config, tmp_path, "JP", "validation")
    assert not {r["fingerprint"] for r in pool.manifest["records"]} & {r["fingerprint"] for r in validation.manifest["records"]}
    with pytest.raises(ValueError):
        build_pool(config, tmp_path, "JP", "test")


def test_degree_and_identity_scope_is_explicit():
    assert linial_plan() == (((65536, 13), (169, 7)), 49)
    with pytest.raises(ValueError, match="degree <=3"):
        ColoringProgram(nx.complete_graph(5), "BE_BASE_D3", 4)


@pytest.mark.parametrize("teacher", TEACHERS)
def test_global_partition_position_is_available_only_to_speculative_schedule(teacher):
    program = ColoringProgram(generate_graph(40, 3, "REGULAR", 573), teacher, 894)
    state = program.snapshot()
    x, relations = encode_snapshot(state, program.neighbors)
    if teacher == "SPECULATIVE_GM":
        assert np.any(state.slots) and np.any(x[:, 6])
    else:
        assert not np.any(state.slots) and not np.any(x[:, 6])
        assert not np.any(relations[:, :, 1])


def tiny_config():
    import json
    config = json.loads(Path("configs/multiteacher/comparison.json").read_text())
    config.update(namespace=99887711, seeds=[0], n=8, graphs=2, tapes=1, val_graphs=1,
                  updates=2, batch_decisions=4, checkpoints=[1, 2], log_every=1,
                  test_counts=[[8, 1]], test_tapes=1, cohort_batch=4, workers=1, threads=1)
    return config


@pytest.mark.parametrize("name", TEACHERS)
def test_learned_execution_does_not_filter_invalid_actions(name, monkeypatch):
    from dcl.multiteacher import experiment as e
    class Unsafe(torch.nn.Module):
        def forward(self, centers, *rest):
            n = len(centers)
            c = centers.new_zeros((n, 5)); c[:, 1] = 10
            a = centers.new_zeros((n, 169)); a[:, 0] = 10
            return c, a, centers.new_full((n,), -10)
    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher must not be invoked in learned execution")
    monkeypatch.setattr(e, "teacher_choices", forbidden)
    result = e.rollout_many(Unsafe(), name, [(nx.complete_graph(4), {"tape_seed": 35})], 8)[0]
    assert not result["success"]
    assert result["failure"] in ("palette", "collision")


def test_complete_comparison_freezes_all_models_and_resumes(tmp_path):
    from dcl.multiteacher import experiment as e
    config = tiny_config()
    result = e.run(config, tmp_path)
    import json
    frozen = json.loads((tmp_path / "frozen.json").read_text())
    assert len(frozen["models"]) == 7 and result["cases"] == 8
    assert result["evaluation_rows"] == 112
    first = {p: p.read_bytes() for p in (tmp_path / "evaluations").rglob("*.json")}
    assert e.run(config, tmp_path) == result
    assert first == {p: p.read_bytes() for p in first}
    assert (tmp_path / "learning-curves.png").exists()


def test_optimizer_and_sampling_resume_exactly_after_interruption(tmp_path, monkeypatch):
    from dcl.multiteacher import experiment as e
    config = tiny_config()
    config.update(updates=4, checkpoints=[1, 2, 4])
    pool = build_pool(config, tmp_path, "JP", "train")
    validation = build_pool(config, tmp_path, "JP", "validation")
    metadata = e.source_metadata()
    reference = e.train_one(config, tmp_path / "reference", "JP", 0, pool, validation, metadata)
    original_save = e.save_checkpoint
    def interrupt(path, saved):
        original_save(path, saved)
        if path.name == "last.pt" and saved["updates"] == 2:
            raise InterruptedError("simulated stop after authoritative checkpoint")
    monkeypatch.setattr(e, "save_checkpoint", interrupt)
    with pytest.raises(InterruptedError):
        e.train_one(config, tmp_path / "resumed", "JP", 0, pool, validation, metadata)
    monkeypatch.setattr(e, "save_checkpoint", original_save)
    resumed = e.train_one(config, tmp_path / "resumed", "JP", 0, pool, validation, metadata)
    a, b = [torch.load(p, weights_only=False) for p in (reference, resumed)]
    for key in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
    assert a["order_rng"] == b["order_rng"]
    assert (resumed.parent / "step-000002.pt").exists()


def test_process_worker_matches_serial_training(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from dcl.multiteacher import experiment as e
    config = tiny_config()
    pool = build_pool(config, tmp_path, "JP", "train")
    validation = build_pool(config, tmp_path, "JP", "validation")
    metadata = e.source_metadata()
    reference = e.train_worker(config, tmp_path, "JP", 0, pool.manifest, validation.manifest, metadata)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        # A second seed in a subprocess is compared with its serial counterpart.
        parallel = executor.submit(e.train_worker, config, tmp_path, "JP", 1,
                                   pool.manifest, validation.manifest, metadata).result()
    a = torch.load(parallel, weights_only=False)
    serial = e.train_one(config, tmp_path / "serial", "JP", 1, pool, validation, metadata)
    b = torch.load(serial, weights_only=False)
    for key in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)


def test_shared_memory_schedule_preserves_fresh_and_resumed_weights(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    from experiments.run_multiteacher_threads import install_schedule
    from dcl.multiteacher import experiment as e
    config = tiny_config()
    config.update(updates=4, checkpoints=[2, 4], log_every=2)
    pool = build_pool(config, tmp_path, "JP", "train")
    validation = build_pool(config, tmp_path, "JP", "validation")
    metadata = e.source_metadata()
    reference = [e.train_one(config, tmp_path / "reference", "JP", s, pool, validation, metadata) for s in (0, 1)]
    resumed = tmp_path / "threaded" / "runs" / "jp" / "s1"
    resumed.mkdir(parents=True)
    shutil.copyfile(reference[1].parent / "step-000002.pt", resumed / "last.pt")
    with monkeypatch.context() as context:
        context.setattr(e, "initialize", e.initialize)
        context.setattr(torch, "set_rng_state", torch.set_rng_state)
        context.setattr(e, "ProcessPoolExecutor", e.ProcessPoolExecutor)
        install_schedule(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [executor.submit(e.train_one, config, tmp_path / "threaded", "JP", s,
                                    pool, validation, metadata) for s in (0, 1)]
            results = [job.result() for job in jobs]
    for x, y in zip(reference, results):
        a, b = [torch.load(p, weights_only=False) for p in (x, y)]
        assert a["initial_state_hash"] == b["initial_state_hash"] and a["order_rng"] == b["order_rng"]
        for key in a["model_state_dict"]:
            torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
