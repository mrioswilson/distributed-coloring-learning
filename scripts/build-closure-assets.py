#!/usr/bin/env python3
"""Build manuscript tables/figures from closure audits, without simulation."""
from collections import defaultdict
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("results/final-closure")
PAPER = Path("paper/ucgnn")
TEACHERS = ["LUBY", "JOHANSSON", "COLOR_TRIALS", "JP", "BE_BASE_D3", "SPECULATIVE_GM", "GREEDY"]
ARCH = ["normalized", "deepsets", "transformer"]
SHORT = dict(LUBY="Luby", JOHANSSON="Johansson", COLOR_TRIALS="ColorTrials", JP="JP", BE_BASE_D3="BE base", SPECULATIVE_GM="GM", GREEDY="Greedy", normalized="E1", deepsets="DeepSets", transformer="Transformer")


def main():
    figdir = PAPER / "figures"
    tabdir = PAPER / "tables"
    figdir.mkdir(exist_ok=True)
    tabdir.mkdir(exist_ok=True)
    cells = list(csv.DictReader((BASE / "generalization-cells.csv").open()))
    main = [c for c in cells if c["stage"] == "confirmatory" and c["primary"] == "True"]
    conds = ["global-n64-d3-mixture1000", "global-n64-d4-mixture1000", "global-n64-d10-mixture1000", "local-n64-d4-mixture1000", "global-n256-d4-mixture1000"]
    labels = ["n=64, Δ=3\nglobal K=4", "n=64, Δ=4\nglobal K=5", "n=64, Δ=10\nglobal K=11", "n=64, Δ=4\nlocal Kᵥ=d(v)+1", "n=256, Δ=4\nglobal K=5"]
    groups = [(t, a) for t in TEACHERS if t != "BE_BASE_D3" for a in ARCH]
    failures = np.zeros((len(groups), 25), int)
    table = []
    for i, (t, a) in enumerate(groups):
        entry = []
        for j, condition in enumerate(conds):
            rr = sorted([c for c in main if (c["teacher"], c["architecture"], c["condition"]) == (t, a, condition)], key=lambda c: c["seed"])
            assert len(rr) == 5 and all(int(c["executions"]) == 1000 for c in rr)
            failures[i, j*5:j*5+5] = [int(c["failures"]) for c in rr]
            perfect = sum(int(c["successes"]) == 1000 for c in rr)
            lo, hi = min(int(c["successes"]) for c in rr), max(int(c["successes"]) for c in rr)
            entry.append("5/5" if perfect == 5 else f"{perfect}/5 ({lo}--{hi})")
        table.append(f"{SHORT[t]} & {SHORT[a]} & " + " & ".join(entry) + r" \\")
    (tabdir / "closure-generalization.tex").write_text("\n".join(table) + "\n")
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.imshow(np.log10(1+failures), cmap="Blues", vmin=0, vmax=np.log10(182), aspect="auto")
    for i, j in zip(*np.nonzero(failures)):
        ax.text(j, i, str(failures[i, j]), ha="center", va="center", color="white" if failures[i, j] > 20 else "black", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(groups)), [SHORT[t] + " / " + SHORT[a] for t, a in groups], fontsize=9)
    ax.set_xticks([5*j+2 for j in range(5)], labels, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-.5, 25, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(groups), 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=.35)
    ax.tick_params(which="minor", length=0)
    for j in range(1, 5):
        ax.axvline(5*j-.5, color="black", lw=.8)
    for i in range(3, len(groups), 3):
        ax.axhline(i-.5, color="#999999", lw=.6)
    ax.set_title("Each square: one frozen checkpoint, 1,000 executions. White: zero failures.\nReplicas 1–5 run left to right inside every condition.", fontsize=10, pad=42)
    ax.set_xlabel("No averaging over training replicas; ten graph families per condition.", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / ("closure-generalization." + ext), dpi=180)
    plt.close(fig)

    audit = json.loads((BASE / "counterfactual-audit.json").read_text())["rows"]
    strata = json.loads((BASE / "review-ml-be-strata.json").read_text())["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    methods = ["passive", "matched", "counterfactual"]
    colors = plt.get_cmap("tab10")
    for j, ax in enumerate(axes):
        group = "all" if j == 0 else "unseen_types_all"
        for i, seed in enumerate(range(94101, 94106)):
            ys = [100 * next(r["balanced_accuracy"] for r in strata if r["seed"] == seed and r["method"] == m and r["group"] == group) for m in methods]
            ax.plot(range(3), ys, "o-", color=colors(i), alpha=.85, lw=1, ms=4, label=str(seed))
        baseline = [100*r["balanced_accuracy"] for r in strata if r["method"] == "eligibility_only" and r["group"] == group]
        ax.axhspan(min(baseline), max(baseline), color="gray", alpha=.16)
        ax.axhline(statistics.mean(baseline), color="gray", ls=":", lw=1, label="Eligibility-only (range/mean)")
        ax.axhline(50, color="black", ls="--", lw=.8, label="Always reject")
        ax.set_xticks(range(3), ["Passive", "Matched", "CF"], fontsize=10)
        ax.set_ylim(0, 104)
        ax.grid(axis="y", alpha=.2)
        ax.set_title("All existing queries" if j == 0 else "Unseen sufficient COMMIT types (posthoc)", fontsize=10)
        ax.set_ylabel("Balanced continuation accuracy (%)" if j == 0 else "")
    axes[1].legend(fontsize=6, loc="lower right", title="Paired seed / diagnostic baseline", title_fontsize=7)
    fig.suptitle("BE / E1, 4,000 updates; 3,000 / 3,000 / 6,776 acquired labels", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / ("closure-counterfactual." + ext), dpi=180)
    plt.close(fig)
    lines = []
    for seed in range(94101, 94106):
        rr = [next(r for r in audit if r["seed"] == seed and r["method"] == m) for m in methods]
        lines.append(f"{seed} & " + " & ".join(f"{r['correct']:,}" for r in rr) + f" & {rr[0]['unseen_commit_type_total']} & " + " & ".join(str(r["unseen_commit_type_correct"]) for r in rr) + r" \\")
    (tabdir / "closure-be-seeds.tex").write_text("\n".join(lines) + "\n")
    lines = []
    label_counts = {"passive": "3,000", "matched": "3,000", "counterfactual": "6,776", "always_reject": "--", "eligibility_only": "--"}
    names = {"passive": "Passive", "matched": "Matched", "counterfactual": "Counterfactual", "always_reject": "Always reject", "eligibility_only": "Eligibility only"}
    for method in names:
        rr = [r for r in strata if r["group"] == "all" and r["method"] == method]
        acc = [100*r["accuracy"] for r in rr]
        ba = [100*r["balanced_accuracy"] for r in rr]
        lines.append(f"{names[method]} & {label_counts[method]} & {statistics.mean(acc):.4f} & {min(acc):.4f}--{max(acc):.4f} & {statistics.mean(ba):.4f}" + r" \\")
    (tabdir / "closure-be-summary.tex").write_text("\n".join(lines) + "\n")
    costs = []
    cost_rows = []
    for teacher, budget in [("BE_BASE_D3",3000),("COLOR_TRIALS",30000),("SPECULATIVE_GM",3000)]:
        rr = []
        for seed in range(94101, 94106):
            path = Path("results/multiteacher-counterfactual/datasets") / teacher.lower() / f"s{seed}" / f"b{budget}-coverage1-alternate0/manifest.json"
            m = json.loads(path.read_text())
            row = dict(teacher=teacher, seed=seed, native_labels=m["native_labels"], added_cf_labels=m["counterfactual_labels"], total_labels=m["total_acquired_component_labels"], cpu_seconds=m["cpu_seconds"], wall_seconds=m["generation_seconds"], stored_bytes=m["stored_bytes"], simulator_steps=m["teacher_simulator_steps"], manifest=str(path))
            costs.append(row); rr.append(row)
        added = [r["added_cf_labels"] for r in rr]
        cpu = [r["cpu_seconds"] for r in rr]
        sizes = [r["stored_bytes"]/1e6 for r in rr]
        add_text = str(min(added)) if min(added) == max(added) else f"{min(added)}--{max(added)}"
        cost_rows.append(f"{SHORT[teacher]} & {budget} & {add_text} & {min(cpu):.3f}--{max(cpu):.3f} & {min(sizes):.3f}--{max(sizes):.3f}" + r" \\")
    (tabdir / "closure-label-cost.tex").write_text("\n".join(cost_rows) + "\n")
    (BASE / "label-generation-costs.json").write_text(json.dumps(costs, indent=2) + "\n")
    controls = json.loads(Path("results/sample-complexity/controls.json").read_text())
    training = json.loads(Path("results/sample-complexity/training.json").read_text())
    lines = []
    for teacher in ("LUBY", "JP", "BE_BASE_D3"):
        entries = []
        for arch in ("normalized", "small", "no_communication"):
            rr = [r for r in training if r["identity"]["teacher"] == teacher and r["identity"]["architecture"] == arch and r["identity"]["budget"] == 3000] if arch == "normalized" else [c["record"] for c in controls if c["kind"] == "capacity_or_communication" and c["record"]["identity"]["teacher"] == teacher and c["record"]["identity"]["architecture"] == arch]
            rr = sorted(rr, key=lambda r: r["identity"]["seed"])
            assert len(rr) == 5
            entries.append(", ".join(str(r["development"]["successes"]) for r in rr))
        lines.append(SHORT[teacher] + " & " + " & ".join(entries) + r" \\")
    (tabdir / "closure-baselines.tex").write_text("\n".join(lines) + "\n")
    native = list(csv.DictReader(Path("results/multiteacher-degree-generalization/native-21-consolidation.csv").open()))
    lines = []
    for t in TEACHERS:
        for a in ARCH:
            r = next(c for c in native if c["teacher"] == t and c["architecture"] == a)
            vals = list(map(int, r["labels_by_seed"].split(",")))
            label = str(min(vals)) if len(set(vals)) == 1 else f"{min(vals)}--{max(vals)}"
            lines.append(f"{SHORT[t]} & {SHORT[a]} & " + ("Historical" if r["cohort"] == "historical-primary" else "New CF") + f" & {label} & {r['updates']} & {r['parameters']} & 5/5" + r" \\")
    (tabdir / "closure-native21.tex").write_text("\n".join(lines) + "\n")
    information = json.loads(Path("results/multiteacher-counterfactual/information/results.json").read_text())
    lines = []
    for mode in ["without_ids", "numeric_ids", "numeric_differences", "relative_comparisons"]:
        rr = sorted([r for r in information if r["mode"] == mode], key=lambda r: r["seed"])
        assert len(rr) == 5
        lines.append(mode.replace("_", " ").capitalize() + " & " + " & ".join(str(r["development"]["successes"]) + "/60" for r in rr) + r" \\")
    (tabdir / "closure-jp.tex").write_text("\n".join(lines) + "\n")
    print("Generated closure tables and two figures from audited records.")


if __name__ == "__main__":
    main()
