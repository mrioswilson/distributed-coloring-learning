#!/usr/bin/env python3
"""Audit BE diagnostic denominators, pairing and exact input overlap.

Reconstructs the existing queries with their original RNG; no new queries,
training, model selection or new evaluation distribution is introduced.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dcl.multicf.oracles import proposal_bank
from dcl.multiteacher.programs import LINIAL, DETECT, KW
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash

BASE = Path("results/multiteacher-counterfactual")
OUT = Path("results/final-closure")


def input_key(arrays, i, p=None, c=None):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a[i]).tobytes())
    if p is not None:
        h.update(np.ascontiguousarray(p[i], dtype=bool).tobytes())
        h.update(np.ascontiguousarray(c[i], dtype=np.int64).tobytes())
    return h.digest()


def be_commit_key(arrays, i, p, c):
    """Sufficient actual inputs to BE's queried-slot COMMIT, after MAX.

    BE sets all real centers' u=1 and own=0, zeroes ID/slot residuals,
    and uses receiver-relative option occupancy in KW. QueryPolicy replaces
    the KW bound offset by its sign. The queried root slot always has its
    offer bit=1. Other slots are eliminated by gather; duplicate neighbor
    types are eliminated by MAX. This key does not characterize candidates.
    """
    x, _, rel, masks = arrays
    phase = int(np.argmax(x[i, 0, 8:14]))
    assert phase == KW
    query = c[i, 0]
    types = set()
    for j in np.flatnonzero(masks[i, 0]):
        types.add((int(rel[i, 0, j, -4 + query]), bool(p[i, j+1]), bool(c[i, j+1] == query)))
    return phase, int(x[i, 0, 7]), int(np.sign(x[i, 0, 18])), bool(p[i, 0]), tuple(sorted(types))


def main():
    torch.set_num_threads(1)
    training = json.loads((BASE / "training-curves.json").read_text())
    results, pairing = [], []
    for seed in range(94101, 94106):
        directory = BASE / "datasets/be_base_d3" / f"s{seed}" / "b3000-coverage1-alternate0"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert file_hash(directory / "dataset.npz") == manifest["sha256"]
        with np.load(directory / "dataset.npz", allow_pickle=False) as f:
            train = {k: f[k] for k in f.files}
        a = tuple(train[f"a{i}"] for i in range(4))
        train_states, train_queries, train_semantic = set(), set(), set()
        for i in np.flatnonzero(train["native"][:, 3] == KW):
            train_states.add(input_key(a, i))
            train_queries.add(input_key(a, i, train["p"], train["c"]))
            train_semantic.add(be_commit_key(a, i, train["p"], train["c"]))
            for v in range(4):
                p, c = train["cf_p"][:, v], train["cf_c"][:, v]
                train_queries.add(input_key(a, i, p, c))
                train_semantic.add(be_commit_key(a, i, p, c))
        bankdir = BASE / "diagnostics/common-state-banks/be_base_d3" / f"s{seed}"
        bank_manifest = json.loads((bankdir / "manifest.json").read_text())
        assert file_hash(bankdir / "states.npz") == bank_manifest["sha256"]
        with np.load(bankdir / "states.npz", allow_pickle=False) as f:
            bank = {k: f[k] for k in f.files}
        records = [r for r in training if r["teacher"] == "BE_BASE_D3" and r["architecture"] == "normalized" and
                   r["seed"] == seed and r["updates"] == 4000 and r["native_budget"] == 3000]
        assert {r["method"] for r in records} == {"passive", "matched", "counterfactual"}
        assert len({r["initial_hash"] for r in records}) == len({r["data_sha256"] for r in records}) == 1
        models, counts = {}, {}
        for rec in records:
            m = initialize("BE_BASE_D3", "normalized", seed).eval()
            assert file_hash(rec["checkpoint"]) == rec["checkpoint_sha256"]
            payload = torch.load(rec["checkpoint"], map_location="cpu", weights_only=False)
            m.load_state_dict(payload["model"])
            models[rec["method"]] = m
            counts[rec["method"]] = Counter()
        payloads = [torch.load(r["checkpoint"], map_location="cpu", weights_only=False) for r in records]
        assert all(np.array_equal(payloads[0]["visits"], p["visits"]) for p in payloads[1:])
        assert all(payloads[0]["batch_rng"] == p["batch_rng"] for p in payloads[1:])
        pairing.append(dict(seed=seed, initial_hash=records[0]["initial_hash"], data_sha256=records[0]["data_sha256"],
                            identical_native_visits=True, identical_final_minibatch_rng_state=True,
                            labels={r["method"]: r["total_acquired_component_labels"] for r in records},
                            training_graph_sizes=sorted({t["n"] for t in manifest["trajectories"]}),
                            training_manifest_sha256=file_hash(directory / "manifest.json"),
                            evaluation_manifest_sha256=file_hash(bankdir / "manifest.json"),
                            evaluation_split=bank_manifest["split"]))
        rng = np.random.default_rng(17001)
        overlap = Counter()
        semantic_counts = Counter()
        for phase in sorted(np.unique(bank["native"][:, 3])):
            if phase in (LINIAL, DETECT):
                continue
            positions = np.flatnonzero(bank["native"][:, 3] == phase)
            for start in range(0, len(positions), 256):
                use = positions[start:start+256]
                arrays = tuple(bank[f"a{i}"][use] for i in range(4))
                ps, cs, ys = proposal_bank(arrays, bank["native"][use], "BE_BASE_D3", rng)
                states = {method: model.first(tensor_arrays(arrays), int(phase)) for method, model in models.items()}
                for v in range(4):
                    p, c, truth = ps[:, v], cs[:, v], ys[:, v] > .5
                    raw_seen = np.array([input_key(arrays, i, p, c) in train_queries for i in range(len(use))])
                    semantic_seen = np.array([be_commit_key(arrays, i, p, c) in train_semantic for i in range(len(use))])
                    overlap["queries"] += len(use)
                    overlap["identical_complete_input_queries_in_training"] += int(raw_seen.sum())
                    overlap["sufficient_commit_input_types_in_training"] += int(semantic_seen.sum())
                    for i in range(len(use)):
                        semantic_counts[str(be_commit_key(arrays, i, p, c))] += 1
                    for method, model in models.items():
                        with torch.inference_mode():
                            pred = model.commit_for(states[method], torch.from_numpy(p), torch.from_numpy(c)).numpy() > 0
                        ctr = counts[method]
                        ctr["correct"] += int((pred == truth).sum())
                        ctr["total"] += len(use)
                        ctr["unseen_complete_input_correct"] += int(((pred == truth) & ~raw_seen).sum())
                        ctr["unseen_complete_input_total"] += int((~raw_seen).sum())
                        ctr["unseen_commit_type_correct"] += int(((pred == truth) & ~semantic_seen).sum())
                        ctr["unseen_commit_type_total"] += int((~semantic_seen).sum())
        for rec in records:
            old = json.loads((BASE / f"diagnostics/paired-u004000-{rec['method']}" / "be_base_d3/normalized" / f"s{seed}.json").read_text())
            ctr = counts[rec["method"]]
            assert (ctr["correct"], ctr["total"]) == (old["components"]["counterfactual_correct"], old["components"]["counterfactual_total"])
            results.append(dict(seed=seed, method=rec["method"], checkpoint=rec["checkpoint"],
                                checkpoint_sha256=rec["checkpoint_sha256"], labels=rec["total_acquired_component_labels"],
                                native_action_correct=old["action_correct"], native_action_total=old["action_total"],
                                **dict(ctr), overlap=dict(overlap), distinct_test_commit_types=len(semantic_counts),
                                distinct_train_commit_types=len(train_semantic),
                                evaluation_bank_sha256=bank_manifest["sha256"]))
        print(seed, dict(overlap), {m: dict(c) for m, c in counts.items()}, flush=True)
    atomic_json(OUT / "counterfactual-audit.json", dict(rows=results, pairing=pairing, script_sha256=file_hash(__file__),
                new_training_updates=0, new_random_queries=0,
                interpretation="Original development diagnostic queries reconstructed exactly. Disjoint graph sizes/PRNG namespaces do not imply unseen local COMMIT types. Complete-input novelty and sufficient queried-slot input-type novelty are reported separately. The same four-color query design is used in training and diagnostics; no held-out intervention family is identified. Native visitation/RNG receipts plus training code support identical minibatch sequences, not statistical independence across arms."))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
