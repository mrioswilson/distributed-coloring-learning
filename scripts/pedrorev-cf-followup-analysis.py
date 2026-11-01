#!/usr/bin/env python3
"""Experiment F analysis on battery v2 (protocol section 8). Reads logits only."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from dcl.study.storage import atomic_json, file_hash

spec = importlib.util.spec_from_file_location("an", "scripts/pedrorev-cf-analysis.py")
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
FDATA, OUT = Path("data/pedrorev-cf-followup"), Path("results/pedro-revision/cf-followup")
ACTIVE_FREE = (1 << 2) | (1 << 3)


def main():
    with np.load(FDATA / "evaluation-banks/be_base_d3/queries.npz", allow_pickle=False) as raw:
        q = {k: raw[k] for k in raw.files}
    truth, flags, bound = q["y"] > .5, q["flags"], q["bound"]
    active = (q["be_key"] >= 0) & ((q["be_key"] % 256 & ACTIVE_FREE) > 0)
    st = an.strata("BE_BASE_D3", flags, bound)
    st["positive_with_active_free_neighbor"] = truth & active
    st["positive_without_active_free_neighbor"] = truth & ~active & (q["be_key"] >= 0)
    rows = []
    for r in json.loads((OUT / "freeze.json").read_text())["checkpoints"]:
        pred = np.load(FDATA / "responses" / f"s{r['seed']}-{r['arm']}.npy") > 0
        for i, fam in enumerate(("standard", "nuisance")):
            sel = q["family"] == i
            rows.append(dict(seed=r["seed"], arm=r["arm"], family=fam, labels=r["total_acquired_component_labels"],
                             overall=an.confusion(pred[sel], truth[sel]),
                             strata={k: an.confusion(pred[sel & s], truth[sel & s]) for k, s in st.items()}))
            s = rows[-1]["strata"]
            print(r["seed"], f"{r['arm']:13s}", fam[:8], "BA", round(100 * rows[-1]["overall"]["balanced_accuracy"], 3),
                  "TPR(active)", None if s["positive_with_active_free_neighbor"]["tpr"] is None else round(s["positive_with_active_free_neighbor"]["tpr"], 4),
                  f"n={s['positive_with_active_free_neighbor']['n']}",
                  "TNR forced", round(s["forced_ineligible"]["tnr"], 4), "TNR occ", round(s["eligible_active_occupied"]["tnr"], 4))
    atomic_json(OUT / "analysis.json", dict(rows=rows, script_sha256=file_hash(__file__),
                                            battery_sha256=json.loads((OUT / "battery.json").read_text())["query_sha256"]))


if __name__ == "__main__":
    main()
