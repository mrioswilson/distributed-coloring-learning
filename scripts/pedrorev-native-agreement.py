#!/usr/bin/env python3
"""Native action agreement of frozen pedrorev checkpoints on battery v1 states.

For every permitted root of the exact-teacher battery trajectories: does the
model's own decision (participation, candidate, COMMIT on its own messages,
Linial point, GM mark) reproduce the teacher's native decision? Read-only.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from dcl.multicf.oracles import stage_targets
from dcl.multiteacher.programs import DETECT, LINIAL, MAX_FIELD
from dcl.multiteacher_e1.model import native_prediction
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.pedrorev.evaluation import state_bank
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash

OUT = Path("results/pedro-revision/cf-experiment/native-agreement.json")


@torch.inference_mode()
def one(job):
    cfg, row = job
    torch.set_num_threads(1)
    bank, _ = state_bank(cfg, row["teacher"], Path("data/pedrorev-cf"))
    model = initialize(row["teacher"], "normalized", row["seed"]).eval()
    model.load_state_dict(torch.load(row["checkpoint"], weights_only=False)["model"])
    c = Counter()
    for phase in sorted(np.unique(bank["native"][:, 3])):
        pos = np.flatnonzero(bank["native"][:, 3] == phase)
        for s in range(0, len(pos), 512):
            use = pos[s:s + 512]
            arrays = tuple(bank[f"a{i}"][use] for i in range(4))
            native = bank["native"][use]
            out = model(*tensor_arrays(arrays), phase=int(phase))
            color, aux, mark = (t.numpy() for t in native_prediction(out))
            key = "linial" if phase == LINIAL else ("detect" if phase == DETECT else "color")
            pred = aux if phase == LINIAL else (mark if phase == DETECT else color)
            truth = native[:, 1] if phase == LINIAL else (native[:, 2] if phase == DETECT else native[:, 0])
            c[f"{key}_total"] += len(use); c[f"{key}_action_correct"] += int((pred == truth).sum())
            if phase not in (LINIAL, DETECT):
                t = stage_targets(arrays, native, row["teacher"])
                c["participation_correct"] += int((out.p[:, 0].numpy() == t.participate[:, 0]).sum())
                active = t.participate[:, 0]
                c["candidate_total_active"] += int(active.sum())
                c["candidate_correct_active"] += int((out.candidates[:, 0].numpy() == t.candidate[:, 0])[active].sum())
                own_ok = (out.commit.numpy() > 0) == (truth > 0)
                c["commit_own_correct"] += int(own_ok.sum())
    return dict(teacher=row["teacher"], seed=row["seed"], arm=row["arm"], checkpoint_sha256=row["sha256"], **c)


def main():
    freeze = json.loads(Path("results/pedro-revision/cf-experiment/freeze.json").read_text())
    rows = [r for r in freeze["checkpoints"] if r["update"] == 4000]
    with ProcessPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(one, [(freeze["config"], r) for r in rows]))
    atomic_json(OUT, dict(rows=out, script_sha256=file_hash(__file__),
        scope="Native decisions on all permitted roots of the 90 exact-teacher battery-v1 trajectories"))
    for r in out:
        tot = r.get("color_total", 0)
        print(r["teacher"][:6], r["seed"], f"{r['arm']:13s}", "action", f"{r.get('color_action_correct',0)}/{tot}",
              "linial", f"{r.get('linial_action_correct',0)}/{r.get('linial_total',0)}",
              "P", r.get("participation_correct"), "C", f"{r.get('candidate_correct_active')}/{r.get('candidate_total_active')}")


if __name__ == "__main__":
    main()
