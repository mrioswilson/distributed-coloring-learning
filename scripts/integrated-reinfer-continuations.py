#!/usr/bin/env python3
"""Re-infer all 160,000 frozen final continuation decisions of the 40 models."""
import hashlib
import json
from pathlib import Path

import torch

from dcl.paperrevision.experiment import diagnostics
from dcl.samplecomplexity.training import initialize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/integrated-paper-20261031"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/paper-revision-20261030.json").read_text())
    rows, inputs = [], {"scripts/integrated-reinfer-continuations.py": sha(__file__)}
    for teacher in config["teachers"]:
        for seed in config["seeds"]:
            keys = ROOT / f"data/paper-revision-20261030/datasets/{teacher.lower()}/s{seed}/type-keys.json"
            data = ROOT / f"data/paper-revision-20261030/test/{teacher.lower()}/s{seed}"
            for arm in ("native", "cf"):
                cp = ROOT / f"data/paper-revision-20261030/models/{teacher.lower()}/s{seed}/{arm}/final.pt"
                model = initialize(teacher, "normalized", seed).eval()
                model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False)["model"])
                result = diagnostics(model, teacher, data, set(json.loads(keys.read_text())))
                old = ROOT / f"results/paper-revision-20261030/prospective/{teacher.lower()}/s{seed}/{arm}-test.json"
                assert result == json.loads(old.read_text())["diagnostic"], (teacher, seed, arm)
                for p in (cp, old, keys, data / "aux.npz", data / "type-keys.json"):
                    inputs[str(p.relative_to(ROOT))] = sha(p)
                row = dict(teacher=teacher, seed=seed, arm=arm, original_metrics_equal=True, diagnostic=result)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k != "diagnostic"}), flush=True)
    (OUT / "continuation-reinference.json").write_text(json.dumps(dict(scope=__doc__, models=len(rows),
        decisions=sum(r["diagnostic"]["n"] for r in rows), all_original_metrics_equal=True, rows=rows, inputs=inputs,
        context="Original phase grouping, inference batches256, CPU float32, torch_threads1"), indent=2)+"\n")


if __name__ == "__main__":
    main()
