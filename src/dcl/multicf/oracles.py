"""Training-only oracles. No policy/inference module imports this module.

Native decisions and proposal-continuation queries are different objects. The
latter extends the native confirmation rule to typed, permission-respecting
messages; it does not assert that every such message is an exact-teacher draw.
"""
from dataclasses import dataclass

import numpy as np

from dcl.multiteacher.programs import ASSIGN, COLOR, DETECT, KW, LINIAL, REPAIR, TEACHERS
from dcl.multiteacher_e1.supervision import labels_from_inputs


@dataclass(frozen=True)
class Contract:
    teacher: str
    state_domain: str
    interventions: tuple[str, ...]
    recompute: tuple[str, ...]
    components: tuple[str, ...]
    constraints: str
    hard_cases: tuple[str, ...]


CONTRACTS = {
    "LUBY": Contract("LUBY", "proper partial K4 coloring; exact or validated safe prefix",
        ("fresh private draws", "participation messages", "candidate messages"),
        ("availability", "rank proposal", "received offers", "confirmation"),
        ("participation", "candidate", "confirmation"), "anonymous two-stage exchange",
        ("fixed-color conflict", "simultaneous equal offers", "empty neighborhood")),
    "JOHANSSON": Contract("JOHANSSON", "proper partial K4 coloring; exact or validated safe prefix",
        ("fresh candidate draws", "typed proposal messages"),
        ("availability", "rank proposal", "received offers", "confirmation"),
        ("participation", "candidate", "confirmation"), "native participation is always true; silent messages are off-policy continuation queries",
        ("simultaneous equal offers", "one free color", "empty neighborhood")),
    "COLOR_TRIALS": Contract("COLOR_TRIALS", "proper partial K4 coloring; exact or validated safe prefix",
        ("private discrete preferences", "candidate query", "safe-prefix color choices"),
        ("own and neighbor availability", "per-color preference messages", "strict winners", "confirmation"),
        ("participation", "candidate", "confirmation"), "two exchanges; no independent edits of derived messages",
        ("ties", "neighbor lacks queried color", "multiple winning colors", "no winner")),
    "JP": Contract("JP", "proper partial K4 coloring; distinct fixed IDs; priority-compatible prefix",
        ("new distinct ID assignment before replay", "safe-prefix color choices", "typed proposal messages"),
        ("every edge order", "eligibility", "availability", "confirmation"),
        ("participation", "candidate", "confirmation"), "IDs fixed during each replay; ineligible roots never confirm",
        ("priority chain", "close IDs", "disconnected maxima")),
    "BE_BASE_D3": Contract("BE_BASE_D3", "proper bounded auxiliary coloring obtained by legal Linial/KW replay",
        ("new distinct initial IDs", "unblocked Linial point", "free KW option", "candidate query"),
        ("polynomial values and equality", "KW receiver-relative options", "phase bounds", "confirmation"),
        ("linial_point", "participation", "candidate", "confirmation"), "no independent edits of evaluations, phase or auxiliary equality; low KW colors cannot recolor",
        ("late KW", "three occupied options", "field 7", "field 13")),
    "SPECULATIVE_GM": Contract("SPECULATIVE_GM", "replay of four-block schedule with free-color assignments, same-slot detection and scheduled repair",
        ("new distinct IDs before schedule construction", "free ASSIGN/REPAIR color", "candidate query"),
        ("block slots and permissions", "actual speculative colors", "marks", "repair queue", "confirmation"),
        ("participation", "candidate", "confirmation", "conflict_mark"), "no oracle on an unverified speculative history; legal temporary conflicts only",
        ("isolates", "disconnected components", "same-slot conflict", "empty repair queue")),
    "GREEDY": Contract("GREEDY", "proper partial coloring produced in supplied increasing-ID order",
        ("new distinct IDs before replay", "free-color prefix writes", "candidate query"),
        ("schedule", "availability", "first free target", "confirmation"),
        ("participation", "candidate", "confirmation"), "only scheduled root may write",
        ("isolates", "last free color", "disconnected components")),
}
assert set(CONTRACTS) == set(TEACHERS)


def phase_of(arrays):
    x = arrays[0]
    phases = x[:, 0, 8:14].argmax(-1)
    if len(phases) == 0 or not np.all(phases == phases[0]):
        raise ValueError("A nonempty single native phase is required")
    return int(phases[0])


def stage_targets(arrays, native, teacher):
    """Acquire native P/C targets and verify their decomposition differentially."""
    return labels_from_inputs(arrays, native, teacher)


def confirmation(arrays, teacher, participate, candidate):
    """O_T(S,a,xi), restricted to a rooted proposal-continuation query.

    Arrays are always regenerated from a validated global snapshot. Input
    randomness/preferences live in S/xi. Interventions replace only the
    supplied P/C messages, after first exchange; state and availability stay
    fixed. ColorTrials' second exchange contains preferences, not offers.
    """
    if teacher not in CONTRACTS:
        raise ValueError(teacher)
    phase = phase_of(arrays)
    if phase in (LINIAL, DETECT):
        raise ValueError("Linial points and detection marks have their own native oracle")
    x, y, rel, masks = arrays
    p, c = np.asarray(participate, bool), np.asarray(candidate)
    if p.shape != x.shape[:2] or c.shape != p.shape or c.dtype.kind not in "iu":
        raise ValueError("One typed P/C message per root and first neighbor is required")
    if np.any((c < 0) | (c >= 4)):
        raise ValueError("Candidate must be in K4")
    permission = x[..., 7] > .5
    if np.any(p & ~permission):
        raise ValueError("Messages may not activate a non-writable or padding vertex")
    occupied = rel[..., -4:] > .5 if phase == KW else y[..., 1:5] > .5
    free = ~(occupied & masks[..., None]).any(2)
    eligible = permission[:, 0].copy()
    if teacher == "JP":
        larger_pending = (y[:, 0, :, 0] > .5) & (y[:, 0, :, 5] > x[:, :1, 5]) & masks[:, 0]
        eligible &= ~larger_pending.any(1)
    if phase == KW:
        eligible &= x[:, 0, 18] >= 0
    index = np.arange(len(x))
    result = eligible & p[:, 0] & free[index, 0, c[:, 0]]
    if teacher == "COLOR_TRIALS":
        prefs = np.where(free & permission[..., None], x[..., 34:38], -1.)
        received = np.where(masks[:, 0, :, None], prefs[:, 1:], -1.).max(1)
        winning = free[:, 0] & (prefs[:, 0] > received)
        result &= winning[index, c[:, 0]]
    if teacher in ("LUBY", "JOHANSSON", "JP"):
        same_offer = p[:, 1:] & masks[:, 0] & (c[:, 1:] == c[:, :1])
        result &= ~same_offer.any(1)
    return result.astype(np.float32)


def proposal_bank(arrays, native, teacher, rng, variants=4):
    """Finite, acquired-once interventions shared across architectures/updates."""
    targets = stage_targets(arrays, native, teacher)
    b, centers = targets.participate.shape
    ps, cs, ys = [], [], []
    allowed = arrays[0][..., 7] > .5
    for variant in range(variants):
        p, c = targets.participate.copy(), targets.candidate.copy()
        # Cycling each candidate covers all four root queries exactly once.
        c[:, 0] = (c[:, 0] + variant) % 4
        if teacher in ("LUBY", "JOHANSSON", "JP"):
            # Change received messages as well as the root's query. This is a
            # separate rooted context per example, never a claimed joint graph.
            change = rng.random((b, centers - 1)) < .25
            c[:, 1:] = np.where(change, rng.integers(4, size=(b, centers - 1)), c[:, 1:])
            p[:, 1:] ^= (rng.random((b, centers - 1)) < .15) & allowed[:, 1:]
            if variant == 1:
                c[:, 1:] = c[:, :1]
                p[:, 1:] = allowed[:, 1:]
        p[:, 0] = allowed[:, 0]
        if variant == 3:
            # Explicitly teach silence in 1/4 of typed continuation queries.
            p[rng.random(b) < .25, 0] = False
        p &= allowed
        ps.append(p); cs.append(c); ys.append(confirmation(arrays, teacher, p, c))
    return np.stack(ps, 1), np.stack(cs, 1), np.stack(ys, 1)
