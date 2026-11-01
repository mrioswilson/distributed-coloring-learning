#!/usr/bin/env python3
"""Posthoc observer: distinguish a rejected COMMIT bit from a state change.

Motivated by the original frozen component diagnostic's unsafe_accepts count.
That historical field means contract-rejected active acceptance, not collision.
No intervention, retraining, case selection within a seed, or result rewriting.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dcl.coauthorintegration.functional import BEComponentPolicy
from dcl.multicf.oracles import confirmation
from dcl.multicf.protocol import cases
from dcl.multiteacher.programs import KW
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.training import initialize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/integrated-paper-20261031"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Observer(BEComponentPolicy):
    def __init__(self, model):
        super().__init__(model, "baseline")
        self.transitions = Counter()
        self.contexts = Counter()

    @torch.inference_mode()
    def forward(self, *arrays, phase=0):
        out = super().forward(*arrays, phase=phase)
        if phase == KW:
            raw = tuple(a.numpy() for a in arrays)
            p, c = out.p.numpy(), out.candidates.numpy()
            y = confirmation(raw, "BE_BASE_D3", p, c) > .5
            select = p[:, 0] & (out.commit.numpy() > 0) & ~y
            x = raw[0][:, 0]
            bound = np.rint(x[:, 17] * 65536).astype(int)
            old = np.rint(x[:, 14] * bound).astype(int)
            target = (4 * bound + 4) // 5
            decoded = (bound - 1 - old) * 4 + c[:, 0]
            for i in np.flatnonzero(select):
                self.transitions["contract_rejected_active_acceptances"] += 1
                self.transitions["identity_decodes" if decoded[i] == old[i] else "changed_decodes"] += 1
                self.transitions["next_bound_valid" if 0 <= decoded[i] < target[i] else "next_bound_invalid"] += 1
                self.contexts[(int(bound[i]), int(old[i]), int(c[i, 0]), int(decoded[i]))] += 1
        return out


def main():
    torch.set_num_threads(1)
    freeze = OUT / "redundant-write-observer-freeze.json"
    inputs = {"scripts/integrated-observe-redundant-writes.py": sha(__file__),
              "results/integrated-paper-20261031/functional-summary.json": sha(OUT / "functional-summary.json")}
    if freeze.exists():
        assert json.loads(freeze.read_text())["inputs"] == inputs
    else:
        freeze.write_text(json.dumps(dict(scope=__doc__, seeds=[105001, 105010], arm="native", cases_per_seed=1000,
            inputs=inputs), indent=2) + "\n")
    config = json.loads((ROOT / "configs/paper-revision-20261030.json").read_text())
    rows = []
    for seed in (105001, 105010):
        weight = ROOT / f"data/paper-revision-20261030/models/be_base_d3/s{seed}/native/final.pt"
        model = initialize("BE_BASE_D3", "normalized", seed).eval()
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=False)["model"])
        observer = Observer(model).eval()
        actual, _ = evaluate(observer, "BE_BASE_D3", cases(config, "final", seed), batch_size=1024, cohort=32)
        original = json.loads((ROOT / f"results/paper-revision-20261030/prospective/be_base_d3/s{seed}/native-test.json").read_text())["rows"]
        fields = ("case_id", "colors", "native_steps", "failure", "invariants", "terminal_correct", "technical_error")
        assert all(all(a[k] == b[k] for k in fields) for a, b in zip(actual, original))
        assert len(actual) == len(original) == 1000
        row = dict(seed=seed, weight_sha256=sha(weight), baseline_parity=True, counts=dict(observer.transitions),
                   contexts=[dict(bound=k[0], old=k[1], option=k[2], decoded=k[3], count=n)
                             for k, n in sorted(observer.contexts.items())])
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "redundant-write-observer.json").write_text(json.dumps(dict(scope=__doc__, rows=rows,
        freeze_sha256=sha(freeze), interpretation="A contract error can be dynamically inert even for an active root when decoding reproduces its old state."), indent=2) + "\n")


if __name__ == "__main__":
    main()
