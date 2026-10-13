import numpy as np

from dcl.dagger import oracle_from_tape
from dcl.graphs import connected_random_regular_graph
from dcl.oracle import UNCOLORED, oracle_round


def test_oracle_from_tape():
    G = connected_random_regular_graph(40, seed=2026)
    colors = np.full(40, UNCOLORED, dtype=int)

    rng = np.random.default_rng(42)

    for t in range(16):
        before = colors.copy()

        after, original = oracle_round(
            G, before, 4, rng, round_idx=t
        )

        reconstructed = oracle_from_tape(
            G,
            before,
            original.activation_draw,
            original.candidate_draw,
            round_idx=t,
        )

        for field in (
            "participate",
            "candidate_offer",
            "candidate",
            "commit",
            "colors_after",
        ):
            np.testing.assert_array_equal(
                getattr(original, field),
                getattr(reconstructed, field),
            )

        np.testing.assert_array_equal(before, colors)

        colors = after

        if np.all(colors != UNCOLORED):
            break


import networkx as nx
import pytest

from dcl.dagger import safety_shield
from dcl.oracle import apply_commit


def test_shield_blocks_fixed_conflict():
    G = nx.path_graph(3)

    colors = np.array([2, -1, -1])
    participate = np.array([False, True, True])
    candidates = np.array([0, 3, 2])
    commit = np.array([False, True, True])

    executed, stats = safety_shield(
        G, colors, participate, candidates, commit
    )

    np.testing.assert_array_equal(
        executed, [False, False, True]
    )

    assert stats["blocked_fixed"] == 1
    assert stats["executed"] == 1

    np.testing.assert_array_equal(
        apply_commit(colors, candidates - 1, executed),
        [2, -1, 1],
    )


def test_shield_blocks_simultaneous_conflicts():
    G = nx.path_graph(3)

    colors = np.array([-1, -1, -1])
    participate = np.ones(3, dtype=bool)
    candidates = np.array([2, 2, 4])
    commit = np.ones(3, dtype=bool)

    executed, stats = safety_shield(
        G, colors, participate, candidates, commit
    )

    np.testing.assert_array_equal(
        executed, [False, False, True]
    )

    assert stats["blocked_simultaneous"] == 2


def test_shield_preserves_safe_commits():
    G = nx.path_graph(3)

    colors = np.array([-1, -1, -1])
    participate = np.ones(3, dtype=bool)
    candidates = np.array([1, 2, 1])
    commit = np.ones(3, dtype=bool)

    executed, stats = safety_shield(
        G, colors, participate, candidates, commit
    )

    assert executed.all()
    assert stats["blocked_total"] == 0


def test_shield_does_not_overblock():
    G = nx.path_graph(3)

    colors = np.array([2, -1, -1])
    participate = np.array([False, True, True])
    candidates = np.array([0, 3, 3])
    commit = np.array([False, True, True])

    executed, stats = safety_shield(
        G, colors, participate, candidates, commit
    )

    np.testing.assert_array_equal(
        executed, [False, False, True]
    )

    assert stats["blocked_fixed"] == 1
    assert stats["blocked_simultaneous"] == 0


def test_shield_rejects_invalid_input_state():
    G = nx.path_graph(2)

    with pytest.raises(
        ValueError, match="invalid partial coloring"
    ):
        safety_shield(
            G,
            np.array([2, 2]),
            np.ones(2, dtype=bool),
            np.ones(2, dtype=int),
            np.ones(2, dtype=bool),
        )


from dcl.dagger import collect_dagger_rollout


def test_dagger_collects_own_states_and_preserves_safety():
    graph = nx.path_graph(3)

    # Predicciones deliberadamente imperfectas.
    scripted = [
        ([True, True, True], [1, 1, 2], [True, True, True]),
        ([True, True, False], [1, 2, 0], [True, True, False]),
        ([False, True, False], [0, 3, 0], [False, True, False]),
    ]

    calls = iter(scripted)

    def predictor(_model, _state):
        return tuple(
            np.array(part) for part in next(calls)
        )

    result = collect_dagger_rollout(
        graph,
        None,
        seed=42,
        max_rounds=3,
        predict_fn=predictor,
    )

    assert result.rounds == 3
    assert result.finished
    assert result.final_colors.tolist() == [0, 2, 1]

    assert result.blocked_fixed == 1
    assert result.blocked_simultaneous == 2
    assert result.action_disagreements > 0

    # La ronda 1 comienza desde el estado generado
    # por la GNN en la ronda 0, no por Luby.
    np.testing.assert_array_equal(
        result.examples[1].x[:, :5].argmax(axis=1),
        [0, 0, 2],
    )

    for u, v in graph.edges():
        assert result.final_colors[u] != result.final_colors[v]


def test_dagger_accepts_real_m4b_forward():
    from dcl.m4_models import make_m4_model

    model = make_m4_model(
        "B", hidden_dim=8, message_dim=4
    )

    graph = nx.cycle_graph(6)

    result = collect_dagger_rollout(
        graph, model, seed=9, max_rounds=2
    )

    assert 1 <= result.rounds <= 2
    assert len(result.examples) == result.rounds

    for u, v in graph.edges():
        assert (
            result.final_colors[u] == -1
            or result.final_colors[v] == -1
            or result.final_colors[u] != result.final_colors[v]
        )
