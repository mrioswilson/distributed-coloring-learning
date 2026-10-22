"""Audit and describe the E1 seven-teacher follow-up against its old baseline."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dcl.multiteacher.programs import TEACHERS
from dcl.study.storage import atomic_json, file_hash, now
from analyze_multiteacher import phase_learning, success_plot


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def analyze(root, baseline):
    root, baseline = Path(root), Path(baseline)
    summary = json.loads((root / "summary.json").read_text())
    config = summary["config"]
    frozen = json.loads((root / "frozen.json").read_text())
    old_frozen = json.loads((baseline / "frozen.json").read_text())
    current = json.loads((root / "evaluation-rows.json").read_text())
    previous = json.loads((baseline / "evaluation-rows.json").read_text())
    if len(frozen["models"]) != len(TEACHERS) * len(config["seeds"]):
        raise RuntimeError("Incomplete E1 matrix")
    for folder, frozen_runs in ((root, frozen), (baseline, old_frozen)):
        for item in frozen_runs["models"]:
            if file_hash(folder / item["path"]) != item["sha256"]:
                raise RuntimeError("Frozen checkpoint changed")
    keys = ("teacher", "seed", "id")
    old_by_id = {tuple(row[k] for k in keys): row for row in previous}
    new_by_id = {tuple(row[k] for k in keys): row for row in current}
    if len(old_by_id) != len(previous) or len(new_by_id) != len(current) or old_by_id.keys() != new_by_id.keys():
        raise RuntimeError("Actor/case matrices differ")
    paired, exact_colors_checked = defaultdict(list), 0
    for key, row in new_by_id.items():
        old = old_by_id[key]
        for field in ("fingerprint", "tape_seed", "n", "family", "graph_seed"):
            if row[field] != old[field]:
                raise RuntimeError("Cases are not paired")
        if row["seed"] is None:
            if not row["success"] or not old["success"] or row["native_steps"] != old["native_steps"]:
                raise RuntimeError("Exact reference changed")
            relative = Path("evaluations") / row["teacher"].lower() / "exact" / f"{row['id']}.json"
            a, b = [json.loads((folder / relative).read_text())["result"]["colors"] for folder in (root, baseline)]
            if a != b:
                raise RuntimeError("Exact teacher colorings changed")
            exact_colors_checked += 1
        else:
            paired[row["teacher"], row["seed"], row["family"], row["n"]].append((old, row))
    changes = []
    for (teacher, seed, family, n), pairs in paired.items():
        changes.append({"teacher": teacher, "seed": seed, "family": family, "n": n, "cases": len(pairs),
                        "both_success": sum(a["success"] and b["success"] for a, b in pairs),
                        "new_only_success": sum(not a["success"] and b["success"] for a, b in pairs),
                        "old_only_success": sum(a["success"] and not b["success"] for a, b in pairs),
                        "both_fail": sum(not a["success"] and not b["success"] for a, b in pairs)})
    table, components = [], []
    for teacher in TEACHERS:
        new = [r for r in current if r["teacher"] == teacher and r["seed"] is not None]
        old = [r for r in previous if r["teacher"] == teacher and r["seed"] is not None]
        final_local = []
        for seed in config["seeds"]:
            history = json.loads((root / "runs" / teacher.lower() / f"s{seed}" / "history.json").read_text())
            if history[-1]["updates"] != config["updates"]:
                raise RuntimeError("A final model is unfinished")
            val = history[-1]["validation_local"]
            final_local.append(val["accuracy"])
            for point in history:
                val = point["validation_local"]
                if val is None:
                    continue
                n = val.get("component_decisions", 0)
                components.append({"teacher": teacher, "seed": seed, "updates": point["updates"],
                    "action_accuracy": val["accuracy"], "component_decisions": n,
                    **{name: val.get(name, 0) / n if n else None for name in
                        ("participation_correct", "candidate_correct", "own_commit_correct", "unsafe_own_commits")}})
        item = {"teacher": teacher, "old_parameters": 36719,
                "e1_core_parameters": 16035,
                "new_parameters": next(m["parameters"]["total"] for m in frozen["models"] if m["teacher"] == teacher),
                "old_successes": sum(r["success"] for r in old), "new_successes": sum(r["success"] for r in new),
                "trials": len(new), "new_local_action_accuracy": float(np.mean(final_local))}
        item["change_percentage_points"] = 100 * (item["new_successes"] - item["old_successes"]) / item["trials"]
        for family in ("REGULAR", "IRREGULAR"):
            for n, _ in config["test_counts"]:
                for tag, data in (("old", old), ("e1", new)):
                    values = [r["success"] for r in data if r["family"] == family and r["n"] == n]
                    item[f"{tag}_{family.lower()}_n{n}"] = float(np.mean(values))
        table.append(item)
    write_csv(root / "architecture-comparison.csv", table)
    write_csv(root / "paired-changes.csv", changes)
    write_csv(root / "component-learning.csv", components)
    phase_learning(root, config)
    success_plot(root, summary)
    architecture_plot(root, current, previous, config)
    report = {"created_utc": now(), "models": len(frozen["models"]), "cases": summary["cases"],
              "evaluations": len(current), "exact_colorings_match_baseline": exact_colors_checked,
              "table": table, "paired_changes": changes,
              "baseline_frozen_sha256": file_hash(baseline / "frozen.json"),
              "new_frozen_sha256": file_hash(root / "frozen.json"),
              "interpretation": ["Exploratory follow-up on previously published common test graphs.",
                  "Architecture, factorized training and some native representations change together.",
                  "E1 canonical for Luby/Johansson/greedy; explicit adapters for JP/CT/BE/GM.",
                  "Initialization seeds share the same source graph/tape pool.",
                  "Zero observed failures do not establish universal correctness.",
                  "Independent scientific review pending."]}
    atomic_json(root / "architecture-comparison.json", report)
    return report


def architecture_plot(root, current, previous, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), sharex=True, sharey=True)
    names = ("Luby", "Johansson", "Jones–Plassmann", "ColorTrials", "BE base Δ≤3", "Especulativo GM", "Greedy")
    for ax, teacher, name in zip(axes.flat, TEACHERS, names):
        for data, prefix, style in ((previous, "GNN anterior", "--"), (current, "E1 / E1 adaptada", "-")):
            for family, color in (("REGULAR", "#2166ac"), ("IRREGULAR", "#b2182b")):
                xs, center, low, high = [], [], [], []
                for n, _ in config["test_counts"]:
                    values = [100 * np.mean([r["success"] for r in data if r["teacher"] == teacher and
                              r["seed"] == seed and r["family"] == family and r["n"] == n]) for seed in config["seeds"]]
                    xs.append(n); center.append(np.mean(values)); low.append(min(values)); high.append(max(values))
                ax.plot(xs, center, style, color=color, marker="o", markersize=2,
                        label=f"{prefix}: {family}")
                if data is current:
                    ax.fill_between(xs, low, high, color=color, alpha=.1)
        ax.set_title(name); ax.set_xscale("log"); ax.set_ylim(-2, 102); ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=8)
    fig.supxlabel("Vértices; mismos casos y presupuesto de decisiones")
    fig.supylabel("Coloraciones completas correctas (%)")
    fig.suptitle("Seguimiento E1: media de cinco inicializaciones; banda E1 = rango")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(root / f"architecture-comparison.{extension}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.baseline)
    print(json.dumps({"models": report["models"], "cases": report["cases"], "evaluations": report["evaluations"]}))
