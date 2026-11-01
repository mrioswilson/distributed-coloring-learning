#!/usr/bin/env python3
"""Posthoc mechanism check for BE continuation errors (pedrorev-cf-v1).

For every BE arm/seed: does the training input contain a *positive* COMMIT
type with a participating neighbor that does not occupy the queried option
(simultaneous KW recoloring in disjoint blocks, safe by contract)? And does
the frozen model accept such positive types on the confirmatory battery?
Reads stored banks/logits only.
"""
import json
from pathlib import Path

import numpy as np

from dcl.multiteacher.programs import KW
from dcl.pedrorev.evaluation import be_commit_key
from dcl.study.storage import atomic_json, file_hash

DATA = Path("data/pedrorev-cf")
OUT = Path("results/pedro-revision/cf-experiment/be-neighbor-activity.json")
ACTIVE_FREE = (1 << 2) | (1 << 3)  # neighbor types (occ=0, part=1, offer=0/1)


def has_active_free_neighbor(keys):
    return (keys % 256 & ACTIVE_FREE) > 0


def training_positive_keys(bank, use_cf):
    with np.load(Path(bank) / "dataset.npz", allow_pickle=False) as f:
        t = {k: f[k] for k in f.files}
    rows = np.flatnonzero(t["native"][:, 3] == KW)
    x, rel, masks = t["a0"][rows], t["a2"][rows], t["a3"][rows]
    pairs = [(t["p"][rows], t["c"][rows], t["native"][rows, 0] > 0)]
    if use_cf:
        pairs += [(t["cf_p"][rows, v], t["cf_c"][rows, v], t["cf_y"][rows, v] > .5) for v in range(4)]
    keys = set()
    for p, c, y in pairs:
        keys |= set(be_commit_key(x, rel, masks, p, c)[y].tolist())
    return keys


def main():
    with np.load(DATA / "evaluation-banks/be_base_d3/queries.npz", allow_pickle=False) as raw:
        q = {k: raw[k] for k in raw.files}
    banks = {(b["seed"], b["arm"]): b["bank"] for b in json.loads(Path("results/pedro-revision/cf-experiment/banks.json").read_text())
             if b["teacher"] == "BE_BASE_D3"}
    rows = []
    for (seed, arm), bank in sorted(banks.items()):
        pos_keys = np.array(sorted(training_positive_keys(bank, arm.startswith("cf"))), dtype=np.int64)
        trained = int(has_active_free_neighbor(pos_keys).sum()) if len(pos_keys) else 0
        pred = np.load(DATA / "responses/be_base_d3" / f"s{seed}-{arm}-u004000.npy") > 0
        for fam in (0, 1):
            sel = (q["family"] == fam) & (q["y"] > .5) & (q["be_key"] >= 0)
            act = sel & has_active_free_neighbor(q["be_key"])
            rows.append(dict(seed=seed, arm=arm, family=["standard", "nuisance"][fam],
                             training_positive_types_with_active_free_neighbor=trained,
                             training_positive_types=int(len(pos_keys)),
                             battery_positive_occurrences_with_active_free_neighbor=int(act.sum()),
                             accepted=int(pred[act].sum()),
                             tpr_with_active_free_neighbor=float(pred[act].mean()) if act.any() else None,
                             tpr_without=float(pred[sel & ~act].mean()) if (sel & ~act).any() else None))
            print(seed, f"{arm:13s}", rows[-1]["family"], "trained", trained, "TPR(active nbr)",
                  round(rows[-1]["tpr_with_active_free_neighbor"], 4), "TPR(other)", round(rows[-1]["tpr_without"], 4))
    atomic_json(OUT, dict(rows=rows, script_sha256=file_hash(__file__), scope="Posthoc mechanism analysis; exploratory."))


if __name__ == "__main__":
    main()
