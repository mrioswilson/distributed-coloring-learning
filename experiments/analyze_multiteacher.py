"""Describe the completed, frozen comparison without selecting/retraining models."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dcl.multiteacher.programs import TEACHERS
from dcl.multiteacher.data import load_pool
from dcl.study.config import digest
from dcl.study.storage import atomic_json, file_hash


def analyze(root):
    root = Path(root)
    frozen = json.loads((root / "frozen.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    rows = json.loads((root / "evaluation-rows.json").read_text())
    config = frozen["config"]
    assert len(frozen["models"]) == len(TEACHERS) * len(config["seeds"])
    for relative, expected in frozen["provenance"]["source_hashes"].items():
        assert file_hash(root / "sources" / relative) == expected, relative
        assert file_hash(relative) == expected, relative
    for item in frozen["models"]:
        assert file_hash(root / item["path"]) == item["sha256"]
        assert item["updates"] == config["updates"]
        assert item["decisions"] == config["updates"] * config["batch_decisions"]
    for path in sorted(root.glob("execution*.json")):
        execution = json.loads(path.read_text())
        assert file_hash(execution["driver"]) == execution["driver_sha256"]
        assert file_hash(root / "sources" / execution["driver"]) == execution["driver_sha256"]
        for dependency, expected in execution.get("dependencies", {}).items():
            assert file_hash(dependency) == expected
            assert file_hash(root / "sources" / dependency) == expected
    cache_path = root / "caches" / "greedy-roots" / "manifest.json"
    resident_cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
    if resident_cache:
        for name, expected in resident_cache["hashes"].items():
            assert file_hash(cache_path.parent / resident_cache["attempt"] / name) == expected
    case_ids = {row["id"] for row in rows}
    table, phases, costs, baselines = [], [], [], []
    paired_data = {}
    for teacher in TEACHERS:
        manifests = {}
        for split in ("train", "validation"):
            folder = root / "pools" / teacher.lower() / split
            manifest = json.loads((folder / "manifest.json").read_text())
            for name, expected in manifest["hashes"].items():
                assert file_hash(folder / manifest["attempt"] / name) == expected
            signature = [(r["fingerprint"], r["graph_seed"], tuple(t["tape_seed"] for t in r["trajectories"]))
                         for r in manifest["records"]]
            if split in paired_data:
                assert paired_data[split] == signature, "Unpaired teacher graph/tape pool"
            else:
                paired_data[split] = signature
            manifests[split] = manifest
        if teacher == "GREEDY" and resident_cache:
            assert resident_cache["source_manifest_digest"] == digest(manifests["train"])
            assert resident_cache["roots"] == manifests["train"]["decisions"]
        folder = root / "pools" / teacher.lower() / "validation"
        pool = load_pool(folder, manifests["validation"])
        flat, width = pool.roots.astype(np.int64), pool.x.shape[1]
        labels = pool.targets[flat // width, flat % width]
        majority_by_phase = {}
        for phase, name in enumerate(("COLOR", "ASSIGN", "DETECT", "REPAIR", "LINIAL", "KW")):
            chosen = labels[labels[:, 3] == phase, 2 if phase == 2 else (1 if phase == 4 else 0)]
            if len(chosen):
                counts = np.bincount(chosen)
                baseline = int(max(counts)) / len(chosen)
                majority_by_phase[name] = baseline
                baselines.append({"teacher": teacher, "phase": name, "decisions": len(chosen),
                                  "majority_label": int(counts.argmax()), "phase_constant_accuracy": baseline})
        actors = [(None, None)] + [(seed, None) for seed in config["seeds"]]
        for seed, _ in actors:
            actor = [r for r in rows if r["teacher"] == teacher and r["seed"] == seed]
            assert len(actor) == len(case_ids) and {r["id"] for r in actor} == case_ids
        local = []
        for seed in config["seeds"]:
            history = json.loads((root / "runs" / teacher.lower() / f"s{seed}" / "history.json").read_text())
            final = history[-1]
            assert final["updates"] == config["updates"]
            val = final["validation_local"]
            local.append(val["accuracy"])
            for phase, name in enumerate(("COLOR", "ASSIGN", "DETECT", "REPAIR", "LINIAL", "KW")):
                count = val.get(f"phase_{phase}_decisions", 0)
                if count:
                    phases.append({"teacher": teacher, "seed": seed, "phase": name,
                                   "decisions": count, "accuracy": val[f"phase_{phase}_correct"] / count,
                                   "phase_constant_baseline": majority_by_phase[name]})
        learned = [r for r in rows if r["teacher"] == teacher and r["seed"] is not None]
        exact = [r for r in rows if r["teacher"] == teacher and r["seed"] is None]
        assert all(r["success"] for r in exact)
        exact_by_id = {r["id"]: r for r in exact}
        for group in summary["per_seed"]:
            if group["teacher"] != teacher or group["seed"] is None:
                continue
            selected = [r for r in learned if r["seed"] == group["seed"] and r["family"] == group["family"]
                        and r["n"] == group["n"] and r["success"]]
            record = {"teacher": teacher, "seed": group["seed"], "family": group["family"], "n": group["n"],
                      "paired_successes": len(selected),
                      "gnn_native_steps": float(np.mean([r["native_steps"] for r in selected])) if selected else None,
                      "exact_native_steps_same_cases": float(np.mean([exact_by_id[r["id"]]["native_steps"] for r in selected])) if selected else None}
            if teacher == "SPECULATIVE_GM":
                base = 2 * ((group["n"] + 3) // 4)
                record["gnn_repair_nodes"] = record["gnn_native_steps"] - base if selected else None
                record["exact_repair_nodes_same_cases"] = record["exact_native_steps_same_cases"] - base if selected else None
            costs.append(record)
        item = {"teacher": teacher, "local_validation_mean": float(np.mean(local)),
                "local_validation_min": min(local), "local_validation_max": max(local),
                "gnn_successes": sum(r["success"] for r in learned), "gnn_trials": len(learned),
                "exact_successes": len(exact), "failures": dict(Counter(r["failure"] for r in learned if not r["success"]))}
        for family in ("REGULAR", "IRREGULAR"):
            for n, _ in config["test_counts"]:
                rates = next(c for c in summary["cells"] if c["teacher"] == teacher and c["n"] == n and c["family"] == family)
                prefix = f"{family.lower()}_n{n}"
                item[f"{prefix}_mean"] = rates["mean"]
                item[f"{prefix}_min"] = rates["min"]
                item[f"{prefix}_max"] = rates["max"]
        table.append(item)
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k in table[0] if k != "failures"])
        writer.writeheader(); writer.writerows({k: v for k, v in item.items() if k != "failures"} for item in table)
    with (root / "validation-by-phase.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phases[0])); writer.writeheader(); writer.writerows(phases)
    with (root / "paired-cost.csv").open("w", newline="") as handle:
        names = list(dict.fromkeys(key for row in costs for key in row))
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader(); writer.writerows(costs)
    with (root / "teacher-task-baselines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baselines[0])); writer.writeheader(); writer.writerows(baselines)
    train_fps = {r[0] for r in paired_data["train"]}
    validation_fps = {r[0] for r in paired_data["validation"]}
    test_fps = {r["fingerprint"] for r in rows}
    assert not train_fps & validation_fps
    assert not test_fps & (train_fps | validation_fps)
    report = {"models": len(frozen["models"]), "cases": len(case_ids), "evaluations": len(rows),
              "unique_graphs": {"train_shared": len(train_fps), "validation_local_shared": len(validation_fps),
                                "test_including_fixed_controls": len(test_fps)},
              "table": table, "phase_accuracy": phases, "paired_native_costs": costs, "phase_constant_baselines": baselines,
              "interpretation": ["Descriptive means and initialization ranges, not confidence intervals.",
                  "Local accuracy is teacher imitation; final success checks the complete coloring.",
                  "Common data pool, paired initializations; five seeds do not establish universal correctness.",
                  "BE base for degree <=3 and repeated ColorTrials have the explicit scope in README."]}
    atomic_json(root / "comparison.json", report)
    success_plot(root, summary)
    phase_learning(root, config)
    return report


def success_plot(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"LUBY": "Luby", "JOHANSSON": "Johansson", "JP": "Jones–Plassmann",
              "COLOR_TRIALS": "ColorTrials repetido", "BE_BASE_D3": "BE, caso base Δ≤3",
              "SPECULATIVE_GM": "Especulativo GM", "GREEDY": "Greedy secuencial"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, family in zip(axes, ("REGULAR", "IRREGULAR")):
        for teacher in TEACHERS:
            cells = sorted((r for r in summary["cells"] if r["teacher"] == teacher and r["family"] == family), key=lambda r: r["n"])
            x = [r["n"] for r in cells]
            line, = ax.plot(x, [100*r["mean"] for r in cells], marker="o", markersize=3, label=labels[teacher])
            ax.fill_between(x, [100*r["min"] for r in cells], [100*r["max"] for r in cells], color=line.get_color(), alpha=.1)
        ax.set_xscale("log"); ax.set_ylim(-2, 102); ax.grid(alpha=.2)
        ax.set_title(family); ax.set_xlabel("Número de vértices (escala log)")
    axes[0].set_ylabel("Coloraciones completas correctas (%)")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle(f"Test congelado; media y rango de {len(summary['config']['seeds'])} inicializaciones")
    fig.tight_layout()
    fig.savefig(root / "success-by-size.png", dpi=170)
    fig.savefig(root / "success-by-size.pdf")
    plt.close(fig)


def phase_learning(root, config):
    """Keep easy KEEP/negative-mark labels from hiding the difficult heads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phases = ("COLOR", "ASSIGN", "DETECT", "REPAIR", "LINIAL", "KW")
    rows = []
    for teacher in TEACHERS:
        for seed in config["seeds"]:
            history = json.loads((root / "runs" / teacher.lower() / f"s{seed}" / "history.json").read_text())
            for checkpoint in history:
                val = checkpoint["validation_local"]
                if val is None:
                    continue
                for phase, name in enumerate(phases):
                    count = val.get(f"phase_{phase}_decisions", 0)
                    if not count:
                        continue
                    denominator = 2 * val.get("mark_tp", 0) + val.get("mark_fp", 0) + val.get("mark_fn", 0)
                    rows.append({"teacher": teacher, "seed": seed, "phase": name,
                                 "updates": checkpoint["updates"], "decision_nodes_seen": checkpoint["decision_nodes_seen"],
                                 "validation_decisions": count, "accuracy": val[f"phase_{phase}_correct"] / count,
                                 "mark_positive_f1": (2 * val["mark_tp"] / denominator
                                      if name == "DETECT" and denominator else None)})
    with (root / "learning-by-phase.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    definitions = (("BE_BASE_D3", (("LINIAL", "accuracy", "Auxiliar (Linial)"), ("KW", "accuracy", "Reducción KW"))),
                   ("SPECULATIVE_GM", (("ASSIGN", "accuracy", "Asignación"), ("DETECT", "mark_positive_f1", "Detección: F1 positivos"),
                                        ("REPAIR", "accuracy", "Reparación"))))
    for ax, (teacher, entries) in zip(axes, definitions):
        for phase, metric, label in entries:
            x, y, low, high = [], [], [], []
            for checkpoint in config["checkpoints"]:
                values = [r[metric] for r in rows if r["teacher"] == teacher and r["phase"] == phase
                          and r["updates"] == checkpoint and r[metric] is not None]
                if values:
                    x.append(checkpoint * config["batch_decisions"] / 1e6)
                    y.append(np.mean(values)); low.append(min(values)); high.append(max(values))
            if x:
                line, = ax.plot(x, y, marker="o", markersize=3, label=label)
                ax.fill_between(x, low, high, color=line.get_color(), alpha=.1)
        ax.set_title("BE, caso base Δ≤3" if teacher == "BE_BASE_D3" else "Especulativo GM")
        ax.set_xscale("log"); ax.set_ylim(-.02, 1.02); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.set_xlabel("Decisiones supervisadas (millones; escala log)")
    axes[0].set_ylabel("Exactitud (validación nativa)")
    axes[1].set_ylabel("Exactitud / F1 de marcas positivas")
    fig.suptitle(f"Aprendizaje por fase; media y rango de {len(config['seeds'])} inicializaciones")
    fig.tight_layout()
    fig.savefig(root / "learning-by-phase.png", dpi=170)
    fig.savefig(root / "learning-by-phase.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps({"models": result["models"], "cases": result["cases"],
                      "evaluations": result["evaluations"], "comparison": str(args.root / "comparison.json")}))
