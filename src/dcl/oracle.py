from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


UNCOLORED = -1


@dataclass
class TrialSample:
    activation_draw: np.ndarray
    candidate_draw: np.ndarray
    candidate_offer: np.ndarray
    participate: np.ndarray
    candidate: np.ndarray


@dataclass
class RoundTrace:
    round_idx: int
    colors_before: np.ndarray
    participate: np.ndarray
    candidate: np.ndarray
    commit: np.ndarray
    colors_after: np.ndarray

    # Random tape / counterfactual proposal.
    # Optional so old manually-built traces remain valid.
    activation_draw: np.ndarray | None = None
    candidate_draw: np.ndarray | None = None
    candidate_offer: np.ndarray | None = None


@dataclass
class RunResult:
    solved: bool
    rounds: int
    colors: np.ndarray
    trace: list[RoundTrace]


def available_colors(
    G: nx.Graph,
    colors: np.ndarray,
    num_colors: int,
    v: int,
) -> list[int]:
    used = {
        int(colors[u])
        for u in G.neighbors(v)
        if colors[u] != UNCOLORED
    }

    return [
        c
        for c in range(num_colors)
        if c not in used
    ]


def sample_random_trial_details(
    G: nx.Graph,
    colors: np.ndarray,
    num_colors: int,
    rng: np.random.Generator,
    activation_probability: float = 0.5,
) -> TrialSample:
    n = G.number_of_nodes()

    # Exactly 2N random numbers per round.
    activation_draw = rng.random(n)
    candidate_draw = rng.random(n)

    participate = np.zeros(
        n,
        dtype=bool,
    )

    candidate_offer = np.full(
        n,
        UNCOLORED,
        dtype=np.int64,
    )

    candidate = np.full(
        n,
        UNCOLORED,
        dtype=np.int64,
    )

    for v in range(n):
        if colors[v] != UNCOLORED:
            continue

        palette = available_colors(
            G=G,
            colors=colors,
            num_colors=num_colors,
            v=v,
        )

        if not palette:
            continue

        # What color WOULD this node propose
        # if it participated?
        j = int(
            candidate_draw[v]
            * len(palette)
        )

        offer = palette[j]

        candidate_offer[v] = offer

        if (
            activation_draw[v]
            < activation_probability
        ):
            participate[v] = True
            candidate[v] = offer

    return TrialSample(
        activation_draw=activation_draw,
        candidate_draw=candidate_draw,
        candidate_offer=candidate_offer,
        participate=participate,
        candidate=candidate,
    )


def sample_random_trial(
    G: nx.Graph,
    colors: np.ndarray,
    num_colors: int,
    rng: np.random.Generator,
    activation_probability: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    sample = sample_random_trial_details(
        G=G,
        colors=colors,
        num_colors=num_colors,
        rng=rng,
        activation_probability=activation_probability,
    )

    return (
        sample.participate,
        sample.candidate,
    )


def compute_commit_mask(
    G: nx.Graph,
    colors: np.ndarray,
    participate: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    n = G.number_of_nodes()

    commit = np.zeros(
        n,
        dtype=bool,
    )

    for v in range(n):
        if colors[v] != UNCOLORED:
            continue

        if not participate[v]:
            continue

        if candidate[v] == UNCOLORED:
            continue

        blocked = False

        for u in G.neighbors(v):
            # Defensive check against an already
            # fixed neighbor using the same color.
            if (
                colors[u] != UNCOLORED
                and
                colors[u] == candidate[v]
            ):
                blocked = True
                break

            # Simultaneous candidate conflict.
            if (
                colors[u] == UNCOLORED
                and
                participate[u]
                and
                candidate[u] == candidate[v]
            ):
                blocked = True
                break

        if not blocked:
            commit[v] = True

    return commit


def apply_commit(
    colors: np.ndarray,
    candidate: np.ndarray,
    commit: np.ndarray,
) -> np.ndarray:
    result = colors.copy()

    result[commit] = candidate[commit]

    return result


def oracle_round(
    G: nx.Graph,
    colors: np.ndarray,
    num_colors: int,
    rng: np.random.Generator,
    round_idx: int = 0,
    activation_probability: float = 0.5,
) -> tuple[np.ndarray, RoundTrace]:
    sample = sample_random_trial_details(
        G=G,
        colors=colors,
        num_colors=num_colors,
        rng=rng,
        activation_probability=activation_probability,
    )

    commit = compute_commit_mask(
        G=G,
        colors=colors,
        participate=sample.participate,
        candidate=sample.candidate,
    )

    colors_after = apply_commit(
        colors=colors,
        candidate=sample.candidate,
        commit=commit,
    )

    trace = RoundTrace(
        round_idx=round_idx,
        colors_before=colors.copy(),
        participate=sample.participate.copy(),
        candidate=sample.candidate.copy(),
        commit=commit.copy(),
        colors_after=colors_after.copy(),
        activation_draw=sample.activation_draw.copy(),
        candidate_draw=sample.candidate_draw.copy(),
        candidate_offer=sample.candidate_offer.copy(),
    )

    return colors_after, trace


def is_proper_coloring(
    G: nx.Graph,
    colors: np.ndarray,
) -> bool:
    if np.any(colors == UNCOLORED):
        return False

    for u, v in G.edges():
        if colors[u] == colors[v]:
            return False

    return True


def run_oracle(
    G: nx.Graph,
    seed: int,
    num_colors: int = 4,
    max_rounds: int = 1000,
    save_trace: bool = True,
    activation_probability: float = 0.5,
) -> RunResult:
    n = G.number_of_nodes()

    colors = np.full(
        n,
        UNCOLORED,
        dtype=np.int64,
    )

    rng = np.random.default_rng(seed)

    traces: list[RoundTrace] = []

    for t in range(max_rounds):
        colors, trace = oracle_round(
            G=G,
            colors=colors,
            num_colors=num_colors,
            rng=rng,
            round_idx=t,
            activation_probability=activation_probability,
        )

        if save_trace:
            traces.append(trace)

        if np.all(colors != UNCOLORED):
            return RunResult(
                solved=is_proper_coloring(
                    G,
                    colors,
                ),
                rounds=t + 1,
                colors=colors,
                trace=traces,
            )

    return RunResult(
        solved=False,
        rounds=max_rounds,
        colors=colors,
        trace=traces,
    )
