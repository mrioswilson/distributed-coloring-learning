"""Fresh confirmatory continuation battery and per-query model responses.

The battery uses a new namespace, graph sizes 32/48/64 (training 8--24,
historical development 28/36) and is shared by every seed and arm, so arms are
compared on identical queries. No optimizer, selection rule or bank in
`banks.py` reads it.
"""
import json
from pathlib import Path

import numpy as np
import torch

from dcl.multicf.oracles import confirmation, proposal_bank
from dcl.multicf.protocol import cases
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, DETECT, KW, LINIAL, named_seed, teacher_choices
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.protocol import horizon
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash
from .banks import be_query_flags

FAMILIES = ("standard", "nuisance")


def battery_config(config):
    return dict(config, namespace=config["evaluation_namespace"], final_runs=config["evaluation_runs"],
                final_sizes=config["evaluation_sizes"])


def state_bank(config, teacher, root):
    """All permitted roots of exact-teacher trajectories on the fresh battery."""
    directory = Path(root) / "evaluation-banks" / teacher.lower()
    path, manifest_path = directory / "states.npz", directory / "manifest.json"
    if path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != file_hash(path):
            raise RuntimeError("Evaluation bank changed")
        with np.load(path, allow_pickle=False) as raw:
            return {k: raw[k] for k in raw.files}, manifest
    buffers = {k: [] for k in ("a0", "a1", "a2", "a3", "native", "case", "step")}
    suite = cases(battery_config(config), "final", 0)
    for i, (g, meta) in enumerate(suite):
        p = ColoringProgram(g, teacher, meta["tape_seed"], max_rounds=horizon(teacher, len(g)))
        while not p.finished:
            s = p.snapshot()
            exact = teacher_choices(s, p.neighbors, teacher)
            roots = np.flatnonzero(s.permission)
            if len(roots):
                for j, a in enumerate(encode_roots(s, p.neighbors, roots)):
                    buffers[f"a{j}"].append(a)
                buffers["native"].append(np.c_[exact.color[roots], exact.auxiliary[roots], exact.mark[roots],
                                               np.full(len(roots), s.phase)])
                buffers["case"].append(np.full(len(roots), i)); buffers["step"].append(np.full(len(roots), p.steps))
            p.apply(s, exact)
        if p.failure:
            raise RuntimeError("Exact teacher failed on the evaluation battery")
    packed = {k: np.concatenate(v) for k, v in buffers.items()}
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **packed)
    manifest = dict(teacher=teacher, namespace=config["evaluation_namespace"], runs=len(suite),
                    sizes=config["evaluation_sizes"], families=sorted({m["family"] for _, m in suite}),
                    case_fingerprints=[m["fingerprint"] for _, m in suite], roots=int(len(packed["native"])),
                    sha256=file_hash(path), scope="Confirmatory continuation battery; never used for training or selection")
    atomic_json(manifest_path, manifest)
    return packed, manifest


def nuisance(arrays, p, c, rng):
    """Intervene only on received neighbor messages (never in any training bank).

    For BE, GM and ColorTrials the declared confirmation rule does not read
    neighbor participation/offers, so the contract label is unchanged; the
    family tests invariance to a never-supervised message change.
    """
    allowed = arrays[0][..., 7] > .5
    p, c = p.copy(), c.copy()
    b, centers = p.shape
    p[:, 1:] = (rng.random((b, centers - 1)) < .5) & allowed[:, 1:]
    c[:, 1:] = rng.integers(4, size=(b, centers - 1))
    same = rng.random(b) < .5
    c[same, 1:] = c[same, :1]
    return p, c


def queries(config, teacher, root):
    """Teacher labels and annotations for every battery query (cached)."""
    directory = Path(root) / "evaluation-banks" / teacher.lower()
    path = directory / "queries.npz"
    bank, manifest = state_bank(config, teacher, root)
    if path.exists():
        with np.load(path, allow_pickle=False) as raw:
            return bank, {k: raw[k] for k in raw.files}
    rng = np.random.default_rng(named_seed(config["evaluation_namespace"], "queries", teacher))
    out = {k: [] for k in ("row", "variant", "family", "p", "c", "y", "flags", "bound", "be_key")}
    for phase in sorted(np.unique(bank["native"][:, 3])):
        if phase in (LINIAL, DETECT):
            continue
        positions = np.flatnonzero(bank["native"][:, 3] == phase)
        for start in range(0, len(positions), 256):
            use = positions[start:start + 256]
            arrays = tuple(bank[f"a{i}"][use] for i in range(4))
            ps, cs, ys = proposal_bank(arrays, bank["native"][use], teacher, rng)
            for v in range(4):
                for fam in FAMILIES:
                    p, c = ps[:, v], cs[:, v]
                    if fam == "nuisance":
                        p, c = nuisance(arrays, p, c, rng)
                    y = confirmation(arrays, teacher, p, c)
                    if fam == "nuisance" and teacher in ("BE_BASE_D3", "SPECULATIVE_GM", "COLOR_TRIALS"):
                        if not np.array_equal(y, ys[:, v]):
                            raise RuntimeError("Nuisance family changed a contract label")
                    flags, bound, key = annotate(arrays, teacher, p, c)
                    for k, val in (("row", use), ("variant", np.full(len(use), v)),
                                   ("family", np.full(len(use), FAMILIES.index(fam))), ("p", p), ("c", c),
                                   ("y", y), ("flags", flags), ("bound", bound), ("be_key", key)):
                        out[k].append(val)
    packed = {k: np.concatenate(v) for k, v in out.items()}
    np.savez_compressed(path, **packed)
    atomic_json(directory / "queries-manifest.json", dict(sha256=file_hash(path), queries=int(len(packed["y"])),
        positives=int(packed["y"].sum()), bank_sha256=manifest["sha256"],
        families={f: int((packed["family"] == i).sum()) for i, f in enumerate(FAMILIES)}))
    return bank, packed


def annotate(arrays, teacher, p, c):
    """Bit flags: 1 eligible, 2 participates, 4 queried option free,
    8 winning (ColorTrials), 16 query equals native teacher query."""
    x, y, rel, masks = arrays
    phase = int(np.argmax(x[0, 0, 8:14]))
    b = len(p)
    bound = np.zeros(b, np.int64)
    key = np.full(b, -1, np.int64)
    if teacher == "BE_BASE_D3" and phase == KW:
        e, part, free, bound = be_query_flags(x, rel, masks, p, c)
        key = be_commit_key(x, rel, masks, p, c)
    else:
        e = x[:, 0, 7] > .5
        part = p[:, 0].astype(bool)
        occupied = ((y[..., 1:5] > .5) & masks[..., None]).any(2)[:, 0]
        free = ~occupied[np.arange(b), c[:, 0]]
    winning = np.ones(b, bool)
    if teacher == "COLOR_TRIALS":
        permission = x[..., 7] > .5
        fr = ~((y[..., 1:5] > .5) & masks[..., None]).any(2)
        prefs = np.where(fr & permission[..., None], x[..., 34:38], -1.)
        received = np.where(masks[:, 0, :, None], prefs[:, 1:], -1.).max(1)
        winning = (fr[:, 0] & (prefs[:, 0] > received))[np.arange(b), c[:, 0]]
    flags = e.astype(np.int64) + 2 * part + 4 * free + 8 * winning
    return flags, bound, key


def be_commit_key(x, rel, masks, p, c):
    """Integer code of the sufficient input type of BE's queried-slot COMMIT.

    (sign of KW offset, root participation, MAX-set of neighbor types
    (occupies queried option, participates, offers the queried slot index)).
    Equivalent to `final-counterfactual-audit.be_commit_key` for KW queries.
    """
    sign = np.sign(x[:, 0, 18]).astype(np.int64) + 1
    q = c[:, 0]
    occ = rel[:, 0, :, -4:][np.arange(len(q)), :, q] > .5
    part = p[:, 1:].astype(bool)
    offer = c[:, 1:] == q[:, None]
    types = occ.astype(np.int64) * 4 + part * 2 + offer
    present = np.zeros(len(q), np.int64)
    for j in range(masks.shape[2]):
        present |= np.where(masks[:, 0, j], 1 << types[:, j], 0)
    return (sign * 2 + p[:, 0].astype(np.int64)) * 256 + present


@torch.inference_mode()
def responses(teacher, checkpoint, seed, bank, qs, architecture="normalized"):
    """COMMIT logits of one frozen checkpoint on every battery query."""
    torch.set_num_threads(1)
    model = initialize(teacher, architecture, seed)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()
    logits = np.zeros(len(qs["y"]), np.float32)
    pointer = 0
    # Replays the exact block order of `queries`: phase, chunk, variant, family.
    for phase in sorted(np.unique(bank["native"][:, 3])):
        if phase in (LINIAL, DETECT):
            continue
        positions = np.flatnonzero(bank["native"][:, 3] == phase)
        for start in range(0, len(positions), 256):
            use = positions[start:start + 256]
            arrays = tuple(bank[f"a{i}"][use] for i in range(4))
            state = model.first(tensor_arrays(arrays), int(phase))
            for v in range(4):
                for fam in range(len(FAMILIES)):
                    idx = np.arange(pointer, pointer + len(use))
                    if not (np.array_equal(qs["row"][idx], use) and np.all(qs["variant"][idx] == v)
                            and np.all(qs["family"][idx] == fam)):
                        raise RuntimeError("Query alignment failed")
                    out = model.commit_for(state, torch.from_numpy(qs["p"][idx]), torch.from_numpy(qs["c"][idx]))
                    logits[idx] = out.numpy()
                    pointer += len(use)
    if pointer != len(logits):
        raise RuntimeError("Not every query was answered")
    return logits
