"""Authenticate and copy exact selected weights; never train or modify them."""
import json
import shutil
from pathlib import Path
import torch
from dcl.study.storage import atomic_json, file_hash

ROOT = Path("results/multiteacher-degree-generalization")
SOURCES = [Path("."), Path("/workspaces/AlgoLearning/ucgnn-sample-complexity"),
           Path("/workspaces/AlgoLearning/ucgnn-counterfactual-multiteacher")]


def locate(path, sha):
    for base in SOURCES:
        p = base / path
        if p.is_file() and file_hash(p) == sha:
            return p.resolve()
    raise FileNotFoundError(f"Authenticated checkpoint missing: {path}")


def build():
    selected = []
    old = json.loads(Path("results/sample-complexity/certification.json").read_text())
    frozen = json.loads(Path("results/sample-complexity/frozen-candidates.json").read_text())["candidates"]
    for row in old:
        if row["status"] != "PASS":
            continue
        match = next(r for r in frozen if r["teacher"] == row["teacher"] and r["architecture"] == row["architecture"])
        selected.append(dict(teacher=row["teacher"], architecture=row["architecture"], method="passive",
                             cohort="historical-primary", primary=True, update=500,
                             budget=row["validated_budget"], checkpoints=match["checkpoints"],
                             evidence="results/sample-complexity/certification.json"))
    assert len(selected) == 12
    new = json.loads(Path("results/multiteacher-counterfactual/successful-fixed-points.json").read_text())
    for row in new:
        selected.append(dict(teacher=row["teacher"], architecture=row["architecture"], method="counterfactual",
                             cohort="counterfactual-primary", primary=True, update=row["update"],
                             budget=row["native_labels"][0], checkpoints=row["checkpoint_manifest"],
                             evidence="results/multiteacher-counterfactual/successful-fixed-points.json"))
    for freeze in sorted(Path("results/multiteacher-counterfactual/confirmation").glob("*/freeze.json")):
        data = json.loads(freeze.read_text())
        row = data.get("selected", data)
        if row.get("method") not in ("matched", "passive"):
            continue
        result = freeze.with_name("result.json")
        if not result.exists() or json.loads(result.read_text())["status"] != "PASS":
            continue
        selected.append(dict(teacher=row["teacher"], architecture=row["architecture"], method=row["method"],
                             cohort="new-certified-control", primary=False, update=row["update"],
                             budget=row["budget"], checkpoints=row["checkpoints"], evidence=str(result)))
    output = []
    for package in selected:
        assert len(package["checkpoints"]) == 5
        for cp in package["checkpoints"]:
            source = locate(cp["path"], cp["sha256"])
            target = ROOT / "checkpoints" / f"{cp['sha256']}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            assert file_hash(target) == cp["sha256"]
            record = torch.load(target, weights_only=False, map_location="cpu")
            meta = record.get("record", {})
            row = {k: v for k, v in package.items() if k != "checkpoints"}
            row.update(seed=cp["seed"], source_relative=cp["path"], source_absolute=str(source),
                       path=str(target), sha256=cp["sha256"], parameters=sum(v.numel() for v in record["model"].values()),
                       training_record=meta, training_config=record.get("config", {}))
            row["id"] = f"{row['teacher']}-{row['architecture']}-{row['cohort']}-{row['method']}-s{row['seed']}"
            output.append(row)
    atomic_json(ROOT / "checkpoint-inventory.json", output)
    atomic_json(ROOT / "cohort-packages.json", selected)
    print(json.dumps(dict(packages=len(selected), checkpoints=len(output), primary=sum(r["primary"] for r in output))))


if __name__ == "__main__":
    build()
