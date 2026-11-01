#!/usr/bin/env python3
"""Experiment E: which scale variable drives the ColorTrials COMMIT witness?

Fixed local state (as in scripts/probe-ct-normalization.py): an edge whose
endpoints have degree 1 and local list {0,1}, plus a disjoint star that sets
the global degree Delta. The root holds the strict winning preference w for
color 0 and its neighbor c < w; the queried COMMIT should accept. Delta enters
the network only through the preference normalization w/Delta, c/Delta and
the number of masked palette slots (K = Delta+1).

Panel G (gap): Delta = 10 fixed, (w, c) varied so the normalized gap
(w-c)/Delta and level w/Delta change. Panel D (degree): normalized (w, c)/Delta
held at (1, 2/3) and (1, 1/3) while Delta varies, so only the masked-slot
count changes. Read-only probe of frozen checkpoints (exploratory, registered
in docs/pedro-revision/PROTOCOL_PEDROREV_CF.md section 7 before running).
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from dcl.degree.adapter import FrozenAdapter
from dcl.degree.inventory import ROOT
from dcl.degree.teachers import DegreeProgram, exact_choices
from dcl.samplecomplexity.model import QueryPolicy
from dcl.study.storage import atomic_json, file_hash

OUT = Path("results/pedro-revision/ct-scale-probe.json")
PANEL_G = [(10, w, c) for w, c in ((10, 9), (10, 8), (10, 7), (10, 5), (10, 0), (9, 8), (7, 6), (4, 3), (1, 0),
                                   (7, 4), (4, 1), (9, 6))]
PANEL_D = [(d, d, round(2 * d / 3)) for d in (3, 6, 12, 30, 60)] + [(d, d, round(d / 3)) for d in (3, 6, 12, 30, 60)]


def checkpoints():
    primary = [dict(id=r["id"], architecture=r["architecture"], seed=r["seed"], path=r["path"], sha256=r["sha256"],
                    cohort="historical-primary-cf")
               for r in json.loads((ROOT / "checkpoint-inventory.json").read_text())
               if r["primary"] and r["teacher"] == "COLOR_TRIALS"]
    freeze = json.loads(Path("results/pedro-revision/cf-experiment/freeze.json").read_text())
    new = [dict(id=f"pedrorev-{r['arm']}-s{r['seed']}", architecture="normalized", seed=r["seed"], path=r["checkpoint"],
                sha256=r["sha256"], cohort=f"pedrorev-{r['arm']}")
           for r in freeze["checkpoints"] if r["teacher"] == "COLOR_TRIALS" and r["update"] == 4000]
    return primary + new


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    rows = []
    for cp in checkpoints():
        if file_hash(cp["path"]) != cp["sha256"]:
            raise RuntimeError("Checkpoint changed")
        model = QueryPolicy("COLOR_TRIALS", cp["architecture"]).eval()
        model.load_state_dict(torch.load(cp["path"], weights_only=False)["model"])
        adapter = FrozenAdapter(model)
        for panel, grid in (("gap", PANEL_G), ("degree", PANEL_D)):
            for delta, w, c in grid:
                g = nx.disjoint_union(nx.path_graph(2), nx.star_graph(delta))
                p = DegreeProgram(g, "COLOR_TRIALS", 9321, "local")
                s = p.snapshot()
                s.preferences[:] = 0
                s.preferences[0, 0], s.preferences[1, 0] = w, c
                assert int(exact_choices(p, s).color[0]) == 1
                out = adapter(p, s, overrides=dict(participation=s.permission, candidates=np.zeros(p.n, np.int64),
                                                   query=np.zeros(p.n, np.int64)))
                i = int(np.where(out.roots == 0)[0][0])
                rows.append(dict(checkpoint=cp["id"], cohort=cp["cohort"], architecture=cp["architecture"], seed=cp["seed"],
                                 panel=panel, delta=delta, winner=w, competitor=c, normalized_winner=w / delta,
                                 normalized_gap=(w - c) / delta, accepts=bool(out.commit[i]), logit=float(out.commit_logits[i])))
    atomic_json(OUT, dict(rows=rows, script_sha256=file_hash(__file__),
        scope="Exploratory read-only probe of one constructed local state; decision-level, not rollout success rates."))
    for panel in ("gap", "degree"):
        for cohort in sorted({r["cohort"] for r in rows}):
            rr = [r for r in rows if r["panel"] == panel and r["cohort"] == cohort]
            cells = sorted({(r["delta"], r["winner"], r["competitor"]) for r in rr})
            print(panel, cohort, [(d, w, c, sum(r["accepts"] for r in rr if (r["delta"], r["winner"], r["competitor"]) == (d, w, c)),
                                   sum(1 for r in rr if (r["delta"], r["winner"], r["competitor"]) == (d, w, c))) for d, w, c in cells])


if __name__ == "__main__":
    main()
