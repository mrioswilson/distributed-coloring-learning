import networkx as nx
import numpy as np

from dcl.oracle import (
    UNCOLORED,
    RoundTrace,
    available_colors,
    compute_commit_mask,
    apply_commit,
)


def oracle_from_tape(
    G: nx.Graph,
    colors: np.ndarray,
    activation_draw: np.ndarray,
    candidate_draw: np.ndarray,
    round_idx: int = 0,
) -> RoundTrace:
    """
    Evaluate the oracle on an externally supplied state
    and random tape, without consuming RNG or mutating colors.
    """

    n = G.number_of_nodes()

    if any(len(a) != n for a in (
        colors, activation_draw, candidate_draw
    )):
        raise ValueError("Array lengths do not match graph")

    for u, v in G.edges():
        if (
            colors[u] != UNCOLORED
            and colors[u] == colors[v]
        ):
            raise ValueError("Invalid partial coloring")

    if not (
        np.all((activation_draw >= 0) & (activation_draw < 1))
        and np.all((candidate_draw >= 0) & (candidate_draw < 1))
    ):
        raise ValueError("Random draws must be in [0, 1)")

    participate = np.zeros(n, dtype=bool)
    candidate_offer = np.full(n, UNCOLORED, dtype=int)
    candidate = np.full(n, UNCOLORED, dtype=int)

    for v in range(n):
        if colors[v] != UNCOLORED:
            continue

        palette = available_colors(G, colors, 4, v)

        if not palette:
            raise ValueError(f"Empty palette at vertex {v}")

        j = int(candidate_draw[v] * len(palette))
        candidate_offer[v] = palette[j]

        if activation_draw[v] < 0.5:
            participate[v] = True
            candidate[v] = palette[j]

    commit = compute_commit_mask(
        G, colors, participate, candidate
    )

    return RoundTrace(
        round_idx=round_idx,
        colors_before=colors.copy(),
        participate=participate,
        candidate=candidate,
        commit=commit,
        colors_after=apply_commit(colors, candidate, commit),
        activation_draw=activation_draw.copy(),
        candidate_draw=candidate_draw.copy(),
        candidate_offer=candidate_offer,
    )


def safety_shield(
    G: nx.Graph,
    colors: np.ndarray,
    pred_participate: np.ndarray,
    pred_candidate_class: np.ndarray,
    pred_commit: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Block unsafe GNN commits without using oracle decisions.

    M4 candidate encoding:
      0    = NO_CANDIDATE
      1..4 = actual colors 0..3

    Returns:
      executed: boolean mask of commits allowed to execute
      stats: intervention counts

    Does not modify the input arrays or graph state.
    """
    n = G.number_of_nodes()

    arrays = (
        colors,
        pred_participate,
        pred_candidate_class,
        pred_commit,
    )
    if any(np.asarray(a).shape != (n,) for a in arrays):
        raise ValueError("Expected one entry per vertex")

    if np.any((colors < UNCOLORED) | (colors > 3)):
        raise ValueError("Invalid current color")

    if np.any(
        (pred_candidate_class < 0)
        | (pred_candidate_class > 4)
    ):
        raise ValueError("Invalid M4 candidate class")

    if any(
        colors[u] != UNCOLORED and colors[u] == colors[v]
        for u, v in G.edges()
    ):
        raise ValueError(
            "Shield received an invalid partial coloring"
        )

    # M4 action semantics:
    # A vertex commits only when all three conditions hold.
    proposals = (
        (colors == UNCOLORED)
        & np.asarray(pred_participate, dtype=bool)
        & np.asarray(pred_commit, dtype=bool)
        & (pred_candidate_class > 0)
    )

    executed = proposals.copy()

    # 1. Reject conflicts with permanently colored neighbors.
    blocked_fixed = np.zeros(n, dtype=bool)

    for v in np.flatnonzero(proposals):
        selected_color = int(pred_candidate_class[v]) - 1

        if any(
            colors[u] == selected_color
            for u in G.neighbors(int(v))
        ):
            blocked_fixed[v] = True

    executed[blocked_fixed] = False

    # 2. Reject simultaneous conflicting commits.
    # Only surviving proposals are considered.
    blocked_simultaneous = np.zeros(n, dtype=bool)

    for u, v in G.edges():
        if (
            executed[u]
            and executed[v]
            and pred_candidate_class[u] == pred_candidate_class[v]
        ):
            blocked_simultaneous[u] = True
            blocked_simultaneous[v] = True

    executed[blocked_simultaneous] = False

    stats = {
        "proposed": int(proposals.sum()),
        "blocked_fixed": int(blocked_fixed.sum()),
        "blocked_simultaneous": int(blocked_simultaneous.sum()),
        "blocked_total": int(
            (proposals & ~executed).sum()
        ),
        "executed": int(executed.sum()),
    }

    return executed, stats


# --- M6: recolectar estados visitados por la GNN ---
from dataclasses import dataclass

from dcl.m4_data import M4Example, encode_m4_round, encode_m4_state


@dataclass
class DaggerRollout:
    examples: list[M4Example]
    final_colors: np.ndarray
    finished: bool
    rounds: int
    blocked_fixed: int
    blocked_simultaneous: int
    action_disagreements: int


def predict_m4(model, state, device="cpu"):
    """Predicciones de M4B, sin correcciones simbólicas."""
    import torch

    model.eval()

    with torch.inference_mode():
        x = torch.as_tensor(
            state.x, dtype=torch.float32, device=device
        )
        edges = torch.as_tensor(
            state.edge_index, dtype=torch.long, device=device
        )

        p, candidate, commit = model(
            x, edges, hard_messages=True
        )

    return (
        (p > 0).cpu().numpy(),
        candidate.argmax(dim=1).cpu().numpy(),
        (commit > 0).cpu().numpy(),
    )


def collect_dagger_rollout(
    G, model, seed, max_rounds=64,
    device="cpu", predict_fn=None,
):
    """
    La GNN genera la trayectoria.
    Luby etiqueta los estados visitados por ella.
    El shield bloquea los commits inseguros.
    """
    n = G.number_of_nodes()
    colors = np.full(n, UNCOLORED, dtype=np.int64)
    rng = np.random.default_rng(seed)

    examples = []
    fixed = simultaneous = disagreements = 0

    for t in range(max_rounds):
        if np.all(colors != UNCOLORED):
            break

        # Exactamente 2N números aleatorios por ronda.
        rp = rng.random(n)
        rc = rng.random(n)

        # 1. Consultamos Luby sobre el estado de la GNN.
        teacher = oracle_from_tape(
            G, colors, rp, rc, round_idx=t
        )

        # Guardamos un ejemplo con el formato original M4.
        examples.append(encode_m4_round(G, teacher))

        # 2. La GNN predice sus acciones.
        state = encode_m4_state(G, colors, rp, rc)

        if predict_fn is None:
            p, candidate_class, commit = predict_m4(
                model, state, device
            )
        else:
            p, candidate_class, commit = predict_fn(
                model, state
            )

        p = np.asarray(p, dtype=bool)
        candidate_class = np.asarray(
            candidate_class, dtype=np.int64
        )
        commit = np.asarray(commit, dtype=bool)

        # 3. El shield decide qué commits ejecutamos.
        executed, stats = safety_shield(
            G, colors, p, candidate_class, commit
        )

        proposed = (
            (colors == UNCOLORED)
            & p
            & commit
            & (candidate_class > 0)
        )

        proposed_action = np.where(
            proposed, candidate_class, 0
        )

        teacher_action = np.where(
            teacher.commit, teacher.candidate + 1, 0
        )

        disagreements += int(np.sum(
            (proposed_action != teacher_action)
            & (colors == UNCOLORED)
        ))

        fixed += stats["blocked_fixed"]
        simultaneous += stats["blocked_simultaneous"]

        # 4. Avanzamos con las acciones de la GNN,
        #    filtradas por el shield. NO con Luby.
        candidate_color = np.where(
            candidate_class > 0,
            candidate_class - 1,
            UNCOLORED,
        )

        colors = apply_commit(
            colors, candidate_color, executed
        )

        # Invariante de seguridad.
        if any(
            colors[u] != UNCOLORED
            and colors[u] == colors[v]
            for u, v in G.edges()
        ):
            raise AssertionError(
                "Shield allowed an invalid coloring"
            )

    return DaggerRollout(
        examples=examples,
        final_colors=colors,
        finished=bool(np.all(colors != UNCOLORED)),
        rounds=len(examples),
        blocked_fixed=fixed,
        blocked_simultaneous=simultaneous,
        action_disagreements=disagreements,
    )
