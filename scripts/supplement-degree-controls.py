#!/usr/bin/env python3
"""Use pre-existing CT weights at 4000 updates to match primary CF updates."""
import json
import shutil
import time
from pathlib import Path
import torch
from dcl.degree.inventory import ROOT, locate
from dcl.study.storage import atomic_json, file_hash


if __name__ == "__main__":
    inventory = json.loads((ROOT / "checkpoint-inventory.json").read_text())
    rows = []
    for cp in inventory:
        if cp["teacher"] != "COLOR_TRIALS" or cp["cohort"] != "new-certified-control":
            continue
        source = Path(cp["source_absolute"]).with_name("u004000.pt")
        sha = file_hash(source)
        target = ROOT / "checkpoints" / f"{sha}.pt"
        if not target.exists():
            shutil.copyfile(source, target)
        assert file_hash(target) == sha
        obj = torch.load(target, weights_only=False)
        row = dict(cp, cohort="equal-update-control", update=4000, sha256=sha, path=str(target),
            source_relative=str(Path(cp["source_relative"]).with_name("u004000.pt")), source_absolute=str(source),
            evidence="Pre-existing 4000-update control; no historical native final certification claimed",
            training_record=obj["record"], training_config=obj.get("config", {}))
        row["id"] = f"{row['teacher']}-{row['architecture']}-{row['cohort']}-{row['method']}-s{row['seed']}"
        primary = next(r for r in inventory if r["primary"] and r["teacher"] == row["teacher"] and
                       r["architecture"] == row["architecture"] and r["seed"] == row["seed"])
        assert primary["training_record"]["initial_hash"] == row["training_record"]["initial_hash"]
        assert primary["training_record"]["native_labels"] == row["training_record"]["native_labels"]
        rows.append(row)
    assert len(rows) == 30
    atomic_json(ROOT / "supplemental-checkpoint-inventory.json", rows)
    atomic_json(ROOT / "equal-update-control-design.json", dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        rule="All CT matched/passive 4000-update weights, no outcome-based selection; same initializations and native labels as CF.",
        limitation="CF adds labels and changes their distribution; this does not isolate label efficiency at equal total budget.",
        checkpoints=[dict(id=r['id'],sha256=r['sha256']) for r in rows]))
    print(json.dumps(dict(existing_weights_added=len(rows), new_training_updates=0)))
