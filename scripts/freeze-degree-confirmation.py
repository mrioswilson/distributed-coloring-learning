#!/usr/bin/env python3
"""Freeze the confirmatory design only after the completed exploratory pilot."""
from datetime import datetime, timezone
import json
from pathlib import Path
from dcl.degree.inventory import ROOT
from dcl.degree.graphs import FAMILIES
from dcl.degree.run import source_hashes, freeze
from dcl.study.storage import file_hash


if __name__ == "__main__":
    assert (ROOT/"pilot/index.json").exists(), "Pilot must finish first"
    inventory=json.loads((ROOT/"checkpoint-inventory.json").read_text())
    inventory+=json.loads((ROOT/"supplemental-checkpoint-inventory.json").read_text())
    inventory=[r for r in inventory if r["teacher"]!="BE_BASE_D3"]
    conditions=[]
    for delta in (3,4,10):
        conditions.append(dict(id=f"global-n64-d{delta}-mixture1000",n=64,delta=delta,palette="global",
            families=list(FAMILIES),runs=1000,family_sampling="iid_uniform"))
    conditions.append(dict(id="local-n64-d4-mixture1000",n=64,delta=4,palette="local",families=list(FAMILIES),
                           runs=1000,family_sampling="iid_uniform"))
    conditions.append(dict(id="global-n256-d4-mixture1000",n=256,delta=4,palette="global",families=list(FAMILIES),
                           runs=1000,family_sampling="iid_uniform",primary_only=True))
    freeze(ROOT/"confirmatory/design.json",dict(namespace="degree-20261025-confirmatory-v1",batch=16,
        conditions=conditions,checkpoints=[r["id"] for r in inventory],
        checkpoint_hashes={r["id"]:r["sha256"] for r in inventory},sources=source_hashes(),
        created_utc=datetime.now(timezone.utc).isoformat(),exploratory=False,
        pilot_index_sha256=file_hash(ROOT/"pilot/index.json"),pilot_cost_sha256=file_hash(ROOT/"pilot/cost.json"),
        rationale="Delta3 internal control; delta4 immediate transfer; delta10 substantive transfer; local4 list heterogeneity; n256/d4 size separated. Dense CT delta>=50 exploratory resource cutoffs motivate separate mechanism probes, not exclusion from main low-degree comparisons.",
        statistics="Pointwise two-sided 95% Clopper-Pearson under independent IID uniform-family mixture sampling; report family denominators separately; no simultaneous or universal claim.",
        resource_rule="No wall cutoff in confirmation; predeclared mathematical horizons. No replacement tapes or outcome-based checkpoint selection. Primary-only n256 cell controls cost.",
        training="Zero new updates; all weights already existed before this mission.",
        control_limit="Equal-update CT controls have matched initializations/native labels, but CF uses more total labels. Not an isolated causal label-efficiency comparison."))
    print("Frozen",len(inventory),"checkpoints, 810000 student runs: four conditions x180 plus n256 x90")
