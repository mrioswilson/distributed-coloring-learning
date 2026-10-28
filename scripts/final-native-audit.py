#!/usr/bin/env python3
"""Verify all 105 selected native checkpoints and recompute terminal validity."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from dcl.study.storage import atomic_json, file_hash

BASE = Path("results/multiteacher-degree-generalization")
OUT = Path("results/final-closure")


def main():
    inventory = [c for c in json.loads((BASE / "checkpoint-inventory.json").read_text()) if c["primary"]]
    old = json.loads(Path("results/sample-complexity/certification.json").read_text())
    new = json.loads(Path("results/multiteacher-counterfactual/plans/successful-policies.json").read_text())["selections"]
    assert len(inventory) == 105
    results, inputs = [], {}
    for cp in inventory:
        assert file_hash(cp["path"]) == cp["sha256"]
        if cp["cohort"] == "historical-primary":
            result_path = Path("results/sample-complexity/certification.json")
            result = next(r for r in old if (r["teacher"], r["architecture"]) == (cp["teacher"], cp["architecture"]))
            rows_path = Path("results/sample-complexity/final") / cp["teacher"].lower() / cp["architecture"] / f"s{cp['seed']}.json"
            protocol = "configs/sample-complexity.json"
        else:
            selected = next(r for r in new if (r["teacher"], r["architecture"]) == (cp["teacher"], cp["architecture"]))
            assert any(c["sha256"] == cp["sha256"] and c["seed"] == cp["seed"] for c in selected["checkpoints"])
            result_path = Path(selected["functional_result"])
            assert file_hash(result_path) == selected["functional_result_sha256"]
            result = json.loads(result_path.read_text())
            rows_path = result_path.parent / "final" / f"s{cp['seed']}.json"
            protocol = str(result_path.parent / "freeze.json")
        assert result["status"] == "PASS" and result["seeds_pass"] == 5
        expected = result["per_seed"][str(cp["seed"])]["rows_sha256"]
        assert file_hash(rows_path) == expected
        rows = json.loads(rows_path.read_text())
        assert len(rows) == len({r["case_id"] for r in rows}) == 1000
        families = Counter()
        for row in rows:
            colors = np.array(row["colors"])
            assert len(colors) == row["n"] and np.all((0 <= colors) & (colors < 4))
            edges = np.array(row["edges"], int).reshape(-1, 2)
            assert all(colors[a] != colors[b] for a, b in edges)
            assert row["invariants"] and row["terminal_correct"] and row["failure"] is None
            assert row["no_cutoff"] and row["no_technical_error"]
            families[row["family"]] += 1
        rec = cp["training_record"]
        labels = rec.get("total_acquired_component_labels", rec.get("accounting", {}).get("queries"))
        results.append(dict(checkpoint_id=cp["id"], checkpoint_path=cp["path"], sha256=cp["sha256"],
                            teacher=cp["teacher"], architecture=cp["architecture"], seed=cp["seed"],
                            cohort=cp["cohort"], method=cp["method"], updates=cp["update"], acquired_labels=labels,
                            parameters=cp["parameters"], training_cpu_seconds=rec["cpu_seconds"],
                            training_wall_seconds=rec["training_seconds"], training_domain="n=8,12,16,24; degree<=3; K=4",
                            final_sizes=sorted({r["n"] for r in rows}), families=dict(families),
                            executions=1000, successes=1000, rows_path=str(rows_path), rows_sha256=expected,
                            protocol=protocol, result_path=str(result_path), result_sha256=file_hash(result_path)))
        for p in (rows_path, result_path, Path(protocol), Path(cp["path"])):
            inputs[str(p)] = dict(sha256=file_hash(p), bytes=p.stat().st_size)
    assert len({(r["teacher"], r["architecture"]) for r in results}) == 21
    assert Counter(r["cohort"] for r in results) == {"historical-primary": 60, "counterfactual-primary": 45}
    atomic_json(OUT / "native-certifications.json", dict(checkpoints=results, packages=21, selected_policies=105,
                selected_final_executions=105000, successes=105000, input_files=inputs, script_sha256=file_hash(__file__),
                scope="Twelve historical and nine new packages. Terminal colors independently recomputed; stored intermediate invariant flags authenticated, not reconstructed for all runs."))
    print("21 packages; 105 selected policies; 105000/105000 terminal-valid native rows; hashes verified.")


if __name__ == "__main__":
    main()
