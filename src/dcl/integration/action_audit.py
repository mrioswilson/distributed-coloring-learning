"""Descriptive common-state action marginals, separate from autonomous laws.

Every validation root and its original random values are reused for each
architecture. Empirical histogram TV is descriptive, not a population bound.
This post-training audit never selects a checkpoint or changes training.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from dcl.integration.study import model_for
from dcl.multiteacher.data import load_pool
from dcl.multiteacher.programs import DETECT, LINIAL
from dcl.multiteacher_e1.model import native_prediction
from dcl.multiteacher_e1.supervision import (commit_targets, labels_from_inputs,
                                          phase_groups, tensor_arrays)
from dcl.study.storage import atomic_json, file_hash

OUT = Path("results/integration/multiteacher")


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    training = json.loads((OUT / "training.json").read_text())
    pools = {r["teacher"]: r["manifest"] for r in json.loads((OUT / "pool-manifests.json").read_text())
             if r["split"] == "validation"}
    rows = []
    for teacher, manifest in pools.items():
        pool = load_pool(Path("data/integration-balanced/pools") / teacher.lower() / "validation", manifest)
        for record in (r for r in training if r["teacher"] == teacher):
            assert file_hash(record["checkpoint"]) == record["checkpoint_sha256"]
            model = model_for(teacher, record["architecture"], record["seed"])
            model.load_state_dict(torch.load(record["checkpoint"], map_location="cpu", weights_only=False)["model"])
            model.eval()
            hist = defaultdict(lambda: [Counter(), Counter(), 0, 0])

            def add(phase, name, truth, prediction):
                truth = np.asarray(truth).astype(int); prediction = np.asarray(prediction).astype(int)
                counts = hist[phase, name]
                counts[0].update(map(int, truth)); counts[1].update(map(int, prediction))
                counts[2] += int((truth != prediction).sum()); counts[3] += len(truth)

            for start in range(0, len(pool.roots), 128):
                arrays, labels = pool.batch(np.arange(start, min(start + 128, len(pool.roots))))
                for phase, part, truth in phase_groups(arrays, labels):
                    out = model(*tensor_arrays(part), phase=phase)
                    native = native_prediction(out)
                    field = 2 if phase == DETECT else (1 if phase == LINIAL else 0)
                    add(phase, "native_action", truth[:, field], native[field].numpy())
                    if phase in (DETECT, LINIAL):
                        continue
                    target = labels_from_inputs(part, truth, teacher)
                    add(phase, "participation", target.participate[:, 0], out.p[:, 0].numpy())
                    add(phase, "candidate", target.candidate[:, 0], out.candidates[:, 0].numpy())
                    teacher_commit = commit_targets(target, target.participate, target.candidate)
                    logits = model.commit_for(out.state, torch.from_numpy(target.participate),
                                              torch.from_numpy(target.candidate))
                    add(phase, "commit_given_teacher_proposals", teacher_commit, (logits > 0).numpy())
                    own_target = commit_targets(target, out.p.numpy(), out.candidates.numpy())
                    add(phase, "commit_given_own_proposals", own_target, (out.commit > 0).numpy())
            for (phase, component), (reference, learned, mismatches, total) in sorted(hist.items()):
                keys = sorted(reference.keys() | learned.keys())
                tv = sum(abs(reference[k] - learned[k]) for k in keys) / (2 * total)
                rows.append(dict(teacher=teacher, architecture=record["architecture"], seed=record["seed"],
                                 checkpoint_sha256=record["checkpoint_sha256"], phase=phase, component=component,
                                 roots=total, paired_mismatches=mismatches,
                                 paired_mismatch_rate=mismatches / total, empirical_marginal_tv=tv,
                                 teacher_histogram=dict(sorted(reference.items())),
                                 model_histogram=dict(sorted(learned.items()))))
            assert sum(r["paired_mismatches"] for r in rows if r["teacher"] == teacher and
                       r["architecture"] == record["architecture"] and r["seed"] == record["seed"] and
                       r["component"] == "native_action") == record["local"]["decisions"] - record["local"]["correct"]
    output = dict(protocol="post-training-descriptive-common-validation-roots-v1", models=70,
                  scope="All original validation roots and random values. Histograms are dependent finite observations, not population TV estimates with certified uncertainty. Own-COMMIT reference is recomputed for model proposals.",
                  source_hashes={str(p): file_hash(p) for p in [Path(__file__), Path("src/dcl/multiteacher_e1/supervision.py")]},
                  rows=rows)
    # Keep the source path portable in the exported manifest.
    output["source_hashes"] = {str(Path(k).relative_to(Path.cwd())) if Path(k).is_absolute() else k: v
                               for k, v in output["source_hashes"].items()}
    atomic_json(OUT / "local-action-marginals.json", output)
    print(json.dumps(dict(models=70, component_phase_records=len(rows), validation_replay_passed=True)))


if __name__ == "__main__":
    run()
