import copy

import networkx as nx
import numpy as np
import pytest

from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import TEACHERS, DETECT, LINIAL, KW, available
from dcl.multicf.oracles import CONTRACTS, confirmation, proposal_bank, stage_targets
from dcl.multicf.states import trajectory, replay, validate_choice


@pytest.mark.parametrize("teacher", TEACHERS)
@pytest.mark.parametrize("alternate", [False, True])
def test_native_differential_and_prefix_replay(teacher, alternate):
    g = nx.disjoint_union(nx.complete_graph(4), nx.path_graph(4))
    for p, s, exact, witness in trajectory(g, teacher, 71021, 2021 if alternate else None):
        roots = np.flatnonzero(s.permission)
        arrays = encode_roots(s, p.neighbors, roots)
        native = np.column_stack((exact.color[roots], exact.auxiliary[roots], exact.mark[roots], np.full(len(roots), s.phase)))
        checked, state = replay(witness)
        assert np.array_equal(state.colors, s.colors)
        assert np.array_equal(state.aux, s.aux)
        if s.phase in (LINIAL, DETECT):
            with pytest.raises(ValueError):
                confirmation(arrays, teacher, np.zeros((len(roots), 4), bool), np.zeros((len(roots), 4), int))
            continue
        targets = stage_targets(arrays, native, teacher)
        assert np.array_equal(confirmation(arrays, teacher, targets.participate, targets.candidate) > .5, exact.color[roots] > 0)
        ps, cs, ys = proposal_bank(arrays, native, teacher, np.random.default_rng(61))
        for v in range(4):
            assert np.array_equal(confirmation(arrays, teacher, ps[:, v], cs[:, v]), ys[:, v])
        assert set(CONTRACTS[teacher].components)


def slow_reference(p, s, part, cand):
    free = available(s, p.neighbors)
    values = []
    for v in range(p.n):
        allowed = bool(s.permission[v])
        if p.teacher == "JP":
            allowed &= all(s.colors[u] >= 0 or s.ids[u] < s.ids[v] for u in p.graph[v])
        if s.phase == KW:
            allowed &= s.aux[v] >= s.target
            option = (s.bound - 1 - s.aux[v]) * 4 + cand[v]
            clear = all(s.aux[u] != option for u in p.graph[v])
        else:
            clear = bool(free[v, cand[v]])
        yes = allowed and part[v] and clear
        if p.teacher == "COLOR_TRIALS":
            yes &= all(not s.permission[u] or not free[u, cand[v]] or
                       s.preferences[v, cand[v]] > s.preferences[u, cand[v]] for u in p.graph[v])
        if p.teacher in ("LUBY", "JOHANSSON", "JP"):
            yes &= all(not part[u] or cand[u] != cand[v] for u in p.graph[v])
        values.append(yes)
    return np.asarray(values)


@pytest.mark.parametrize("teacher", TEACHERS)
def test_intervened_messages_against_independent_slow_rule(teacher):
    rng = np.random.default_rng(71022)
    g = nx.disjoint_union(nx.cycle_graph(8), nx.empty_graph(4))
    for p, s, _, _ in trajectory(g, teacher, 71023, 218):
        if s.phase in (LINIAL, DETECT):
            continue
        roots = np.flatnonzero(s.permission)
        arrays = encode_roots(s, p.neighbors, roots)
        centers = np.c_[roots, p.neighbors[roots]]
        for _ in range(3):
            parts = s.permission & (rng.random(p.n) < .7)
            colors = rng.integers(4, size=p.n)
            rp = np.r_[parts, False][centers]
            rc = np.r_[colors, 0][centers]
            assert np.array_equal(confirmation(arrays, teacher, rp, rc) > .5,
                                  slow_reference(p, s, parts, colors)[roots])


def test_stale_label_changes_when_actual_neighbor_offer_changes():
    p, s, _, _ = next(trajectory(nx.path_graph(2), "LUBY", 71))
    arrays = encode_roots(s, p.neighbors, [0])
    part = np.array([[True, True, False, False]])
    candidates = np.array([[0, 1, 0, 0]])
    assert confirmation(arrays, "LUBY", part, candidates)[0] == 1
    candidates[0, 1] = 0
    assert confirmation(arrays, "LUBY", part, candidates)[0] == 0


def test_reject_arbitrary_speculative_history_and_stale_hash():
    gen = trajectory(nx.complete_graph(4), "SPECULATIVE_GM", 51, 15)
    first = next(gen)
    p, s, _, witness = next(gen)
    bad = copy.deepcopy(witness)
    bad["prefix"][0]["choices"]["color"] = [0] * 4
    with pytest.raises(ValueError):
        replay(bad)
    bad = copy.deepcopy(witness); bad["state_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        replay(bad)


def test_reject_stale_linial_arithmetic():
    p, s, exact, _ = next(trajectory(nx.complete_graph(4), "BE_BASE_D3", 17))
    bad = copy.deepcopy(exact); bad.auxiliary[0] = (bad.auxiliary[0] + 1) % 169
    with pytest.raises(ValueError):
        validate_choice(p, s, bad)


def test_kw_ineligible_root_cannot_confirm_even_after_forced_participation():
    for p, s, exact, _ in trajectory(nx.path_graph(8), "BE_BASE_D3", 19):
        low = np.flatnonzero(s.aux < s.target) if s.phase == KW else []
        if len(low):
            arrays = encode_roots(s, p.neighbors, low[:1])
            parts = arrays[0][..., 7] > .5
            for color in range(4):
                assert confirmation(arrays, "BE_BASE_D3", parts, np.full(parts.shape, color))[0] == 0
            return
    pytest.fail("Missing low KW example")


def small_config():
    import json
    from pathlib import Path
    c = json.loads(Path("configs/multiteacher-counterfactual.json").read_text())
    c.update(namespace="unit-only-multicf", training_sizes=[8], updates=6, checkpoints=[6],
             batch_size=4, warmup_updates=1, save_every=2, log_every=2)
    return c


@pytest.mark.parametrize("teacher", TEACHERS)
def test_banks_count_actual_counterfactual_acquisitions(teacher, tmp_path):
    from dcl.multicf.data import acquire, load
    c = small_config()
    m = acquire(c, teacher, 541, 90, tmp_path, alternate=True)
    packed, _ = load(tmp_path)
    native = packed["native"][:, 3]
    assert m["counterfactual_labels"] == int(((native != LINIAL) & (native != DETECT)).sum()) * 4
    assert m["total_acquired_component_labels"] == m["native_labels"] + m["counterfactual_labels"]
    assert 88 <= m["native_labels"] <= 90


@pytest.mark.parametrize("method", ["passive", "matched", "counterfactual", "owncf"])
def test_exact_resume_and_loss_query_accounting(method, tmp_path):
    import torch
    from dcl.multicf.data import acquire
    from dcl.multicf.training import train
    c = small_config()
    dataset = tmp_path/"data"
    acquire(c, "LUBY", 14, 90, dataset)
    train(c, "LUBY", "normalized", 14, dataset, tmp_path/"full", method, development=False)
    train(c, "LUBY", "normalized", 14, dataset, tmp_path/"resume", method, stop_after=3, development=False)
    train(c, "LUBY", "normalized", 14, dataset, tmp_path/"resume", method, development=False)
    full = torch.load(tmp_path/"full"/"last.pt", weights_only=False)
    resumed = torch.load(tmp_path/"resume"/"last.pt", weights_only=False)
    for k in full["model"]:
        assert torch.equal(full["model"][k], resumed["model"][k])
    assert full["record"]["total_acquired_component_labels"] == resumed["record"]["total_acquired_component_labels"]
    assert (full["record"]["own_confirmation_labels"] > 0) == (method == "owncf")
    # Completed training is idempotent and does not overwrite frozen outputs.
    train(c, "LUBY", "normalized", 14, dataset, tmp_path/"resume", method, development=False)
