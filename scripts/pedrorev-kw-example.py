#!/usr/bin/env python3
"""Worked BE/KW continuation-contract example from a real replayed state.

Replays an exact BE trajectory of the historical development split (seed
94101), selects a degree-three KW root with one occupied option and one
ineligible neighbor, and records every quantity of the contract: auxiliary
colors, bound/target, option block, receiver-relative occupancy, eligibility,
native decision, intervened queries, oracle labels and the decisions of the
three paired 4,000-update E1 checkpoints. Checks the oracle against the
teacher's own rule. Read-only with respect to all results.
"""
import json
from pathlib import Path

import numpy as np
import torch

from dcl.multicf.oracles import confirmation, stage_targets
from dcl.multicf.protocol import cases
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import ColoringProgram, KW, K, teacher_choices
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.protocol import horizon
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash

OUT = Path("results/pedro-revision/kw-example.json")
BASE = Path("results/multiteacher-counterfactual")


def main():
    torch.set_num_threads(1)
    config = json.loads(Path("configs/multiteacher-counterfactual.json").read_text())
    seed = 94101
    curves = json.loads((BASE / "training-curves.json").read_text())
    models = {}
    for r in curves:
        if (r["teacher"], r["architecture"], r["seed"], r["updates"], r["native_budget"]) == ("BE_BASE_D3", "normalized", seed, 4000, 3000):
            if r["method"] in ("passive", "matched", "counterfactual"):
                assert file_hash(r["checkpoint"]) == r["checkpoint_sha256"]
                m = initialize("BE_BASE_D3", "normalized", seed).eval()
                m.load_state_dict(torch.load(r["checkpoint"], weights_only=False)["model"])
                models[r["method"]] = (m, r["checkpoint_sha256"])
    for g, meta in cases(config, "development", seed):
        p = ColoringProgram(g, "BE_BASE_D3", meta["tape_seed"], max_rounds=horizon("BE_BASE_D3", len(g)))
        while not p.finished:
            s = p.snapshot()
            exact = teacher_choices(s, p.neighbors, "BE_BASE_D3")
            if s.phase == KW and 12 <= s.bound <= 26:
                offsets = (s.bound - 1 - s.aux) * K
                for v in range(p.n):
                    ns = [int(u) for u in p.neighbors[v] if u < p.n]
                    if len(ns) != 3 or s.aux[v] < s.target:
                        continue
                    options = offsets[v] + np.arange(K)
                    occ = [[int(s.aux[u] == o) for o in options] for u in ns]
                    occupied = np.any(occ, axis=0)
                    inel = [u for u in ns if s.aux[u] < s.target]
                    if occupied.sum() == 1 and 1 <= len(inel) <= 2 and not occupied[0]:
                        continue
                    if occupied.sum() == 1 and occupied[0] and len(inel) >= 1 and any(s.aux[u] >= s.target for u in ns):
                        return record(p, s, exact, v, ns, options, occ, models, meta, seed)
            p.apply(s, exact)
    raise RuntimeError("No example found")


@torch.inference_mode()
def record(p, s, exact, v, ns, options, occ, models, meta, seed):
    roots = np.array([v] + [u for u in ns if s.aux[u] < s.target][:1])
    arrays = encode_roots(s, p.neighbors, roots)
    native = np.column_stack((exact.color[roots], exact.auxiliary[roots], exact.mark[roots], np.full(len(roots), s.phase)))
    t = stage_targets(arrays, native, "BE_BASE_D3")
    queries = []
    # Query family: (root index in `roots`, description, participation of root, option slot)
    plan = [(0, "native query (teacher message)", bool(t.participate[0, 0]), int(t.candidate[0, 0])),
            (0, "alternative option, occupied", True, 0),
            (0, "alternative option, free", True, int([k for k in range(4) if k != t.candidate[0, 0] and not any(o[k] for o in occ)][-1])),
            (0, "silence (eligible root does not participate)", False, int(t.candidate[0, 0])),
            (1, "forced participation of an ineligible root", True, 0)]
    for r, name, part, slot in plan:
        a = tuple(x[r:r + 1] for x in arrays)
        pp, cc = t.participate[r:r + 1].copy(), t.candidate[r:r + 1].copy()
        pp[0, 0], cc[0, 0] = part, slot
        label = float(confirmation(a, "BE_BASE_D3", pp, cc)[0])
        decisions = {}
        for method, (m, _) in models.items():
            state = m.first(tensor_arrays(a), KW)
            decisions[method] = float(m.commit_for(state, torch.from_numpy(pp), torch.from_numpy(cc))[0])
        queries.append(dict(root=int(roots[r]), query=name, participates=part, option_slot=slot,
                            option_color=int((s.bound - 1 - s.aux[roots[r]]) * 4 + slot), label=label,
                            logits=decisions, accepts={k: d > 0 for k, d in decisions.items()}))
    # Teacher's own rule on the root: first free option, write iff aux >= target.
    free = [not any(o[k] for o in occ) for k in range(4)]
    assert exact.color[v] == (1 + free.index(True))
    neighbor_rows = [dict(vertex=u, aux=int(s.aux[u]), eligible=bool(s.aux[u] >= s.target),
                          options_equal=[o for o in occ_u], native_write=int(exact.color[u]))
                     for u, occ_u in zip(ns, occ)]
    result = dict(case_id=meta["case_id"], n=p.n, native_step=int(p.steps), bound=int(s.bound), target=int(s.target),
                  root=int(v), root_aux=int(s.aux[v]), root_eligible=bool(s.aux[v] >= s.target),
                  offset=int((s.bound - 1 - s.aux[v]) * 4), options=[int(o) for o in options], free=free,
                  native_write_option_slot=int(exact.color[v]) - 1, neighbors=neighbor_rows,
                  native_participation=bool(t.participate[0, 0]), native_candidate_slot=int(t.candidate[0, 0]),
                  queries=queries, checkpoints={k: h for k, (_, h) in models.items()}, seed=seed,
                  encoded_root_offset_feature=float(arrays[0][0, 0, 18]),
                  encoded_option_equalities=arrays[2][0, 0, :, -4:].tolist())
    atomic_json(OUT, result)
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
