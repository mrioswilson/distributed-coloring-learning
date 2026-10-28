#!/usr/bin/env python3
"""Reproduce preserved interval certificates, without changing historical files."""
import json
from pathlib import Path
import time
import torch
from dcl.samplecomplexity.model import QueryPolicy
from dcl.paperstudy.certification import commit_certificate
from dcl.universal.train import load_model
from dcl.study.storage import atomic_json, file_hash


def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    base = Path("results/multiteacher-degree-generalization")
    inv = {r["id"]: r for r in json.loads((base / "checkpoint-inventory.json").read_text())}
    expected = json.loads((base / "commit-certificates/index.json").read_text())
    rows, historical = [], []
    for r in expected:
        cp = inv[r["checkpoint"]]
        assert file_hash(cp["path"]) == cp["sha256"]
        m = QueryPolicy(cp["teacher"], cp["architecture"])
        m.load_state_dict(torch.load(cp["path"], weights_only=False, map_location="cpu")["model"])
        for mode, name in ((False, "real"), (True, "float32_model")):
            assert commit_certificate(m.core, mode) == r[name]
        rows.append(dict(checkpoint=cp["id"], sha256=cp["sha256"], exact_rule=r["float32_model"]["certified_exact_rule"], no_false_accept=r["float32_model"]["certified_no_false_accept"]))
    for p in sorted(Path("results/ucgnn-paper-study/certificates").glob("e1-*.json")):
        r = json.loads(p.read_text())
        old = Path(r["checkpoint"])
        # Original receipts retain absolute provenance; portable recovery uses
        # their data/ucgnn suffix under the new checkout.
        marker = "/data/ucgnn/"
        assert marker in str(old)
        cp = Path("data/ucgnn") / str(old).split(marker, 1)[1]
        assert file_hash(cp) == r["checkpoint_sha256"]
        m, _ = load_model(cp, "cpu")
        for mode, name in ((False, "real"), (True, "float32_model")):
            assert commit_certificate(m, mode) == r[name]
        historical.append(dict(checkpoint=str(cp), sha256=file_hash(cp), certificate=str(p), certificate_sha256=file_hash(p), exact_rule=r["float32_model"]["certified_exact_rule"]))
    assert len(rows) == 45 and len(historical) == 5
    atomic_json(Path("results/final-closure/mathematical-certificate-audit.json"),
                dict(selected=rows, historical=historical, all_stored_intervals_reproduced=True,
                     seconds=time.perf_counter()-start, source_sha256=file_hash("src/dcl/paperstudy/certification.py"),
                     script_sha256=file_hash(__file__), new_training_updates=0))
    print("All stored intervals reproduced: 45 selected canonical cores and five historical E1 weights.")


if __name__ == "__main__":
    main()
