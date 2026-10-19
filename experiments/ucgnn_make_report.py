"""Build UCGNN result tables and figures from the original per-graph records.

Usage (repository root, inside the study container):
    .venv/bin/python experiments/ucgnn_make_report.py --output results/ucgnn/<cut>
Reads data/ucgnn/evaluations, data/ucgnn/sweeps, data/ucgnn/permcheck,
data/ucgnn/benchmark, data/ucgnn/runs/*/history.json and the published M6B
final-test tables. Selects nothing: every view found is tabulated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dcl.universal.report import by_k, load_records, specialist_by_k, specialists, summarize

# Reference palette (light mode), categorical slots in fixed order.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
MODEL_COLOR = {"E1": SLOTS[0], "E2": SLOTS[1], "E3": SLOTS[2], "M6B": SLOTS[3]}
TRAINING = {"E1": (3,), "E2": (3, 10), "E3": (3, 5, 8, 10), "A1": (3,)}


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def pooled(frame, keys):
    sums = ["graphs", "success", "finished", "unsafe_commits", "fixed_neighbor_conflicts", "simultaneous_conflicts",
            "valid_commits_rejected", "states", "participation_correct", "candidate_correct", "commit_correct_own",
            "action_agree", "forbidden_offers", "rounds", "teacher_rounds", "monochromatic_edges", "uncolored_nodes",
            "teacher_identical_trajectory"]
    frame = frame.assign(replicas=frame.seed)
    table = frame.groupby(keys).agg({**{k: "sum" for k in sums}, "replicas": "nunique",
                                     "gpu_peak_mb": "max", "inference_seconds_max": "max"}).reset_index()
    table["success_rate"] = table.success / table.graphs
    for name, numerator in (("participation_accuracy", "participation_correct"), ("candidate_accuracy", "candidate_correct"),
                            ("commit_accuracy", "commit_correct_own"), ("action_agreement", "action_agree")):
        table[name] = table[numerator] / table.states
    table["mean_rounds"] = table.rounds / table.graphs
    table["mean_teacher_rounds"] = table.teacher_rounds / table.graphs
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ucgnn")
    parser.add_argument("--m6", default="results/m6study/2026-10-17-final")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Per-view tables ------------------------------------------------------------
    frames, k_frames = [], []
    for split in ("final_test", "ucgnn_extra"):
        records = load_records(root, split)
        if not records:
            continue
        table = summarize(records, keys=["view", "experiment", "seed", "variant", "checkpoint_epoch", "degree",
                                         "seen_degree", "n", "distribution"])
        table.insert(0, "split", split)
        frames.append(table)
        kt = by_k(records, keys=("view", "experiment", "seed", "checkpoint_epoch", "degree", "n"))
        kt.insert(0, "split", split)
        k_frames.append(kt)
    detail = pd.concat(frames, ignore_index=True)
    detail["policy"] = np.where(detail.view.str.endswith("-best"), "best", "epoch-" + detail.checkpoint_epoch.astype(str))
    detail.to_csv(out / "ucgnn-by-view-degree-size-family.csv", index=False)
    pd.concat(k_frames, ignore_index=True).to_csv(out / "ucgnn-by-k.csv", index=False)

    # Pooled over families and sizes, per replica and over replicas.
    per_replica = pooled(detail, ["split", "experiment", "variant", "policy", "seed", "degree", "regime"])
    per_replica.to_csv(out / "ucgnn-by-replica-degree.csv", index=False)
    per_size = pooled(detail, ["split", "experiment", "variant", "policy", "degree", "regime", "n"])
    per_size.to_csv(out / "ucgnn-by-degree-size.csv", index=False)
    overall = pooled(detail, ["split", "experiment", "variant", "policy", "degree", "regime"])
    spread = per_replica.groupby(["split", "experiment", "variant", "policy", "degree"]).agg(
        min_replica_success=("success_rate", "min"), min_replica_candidate=("candidate_accuracy", "min"),
        max_replica_candidate=("candidate_accuracy", "max")).reset_index()
    overall = overall.merge(spread, on=["split", "experiment", "variant", "policy", "degree"])
    overall.to_csv(out / "ucgnn-by-degree.csv", index=False)

    # M6B specialists on the same cohort.
    spec = specialists(args.m6)
    spec.to_csv(out / "m6b-specialists-by-degree-size-family.csv", index=False)
    spec_k = specialist_by_k(args.m6)
    spec_k.to_csv(out / "m6b-specialists-by-k.csv", index=False)
    kseed = pd.read_csv(Path(args.m6) / "k-by-seed.csv")
    main_k = kseed[(kseed.method == "COUNTERFACTUAL") & (kseed.training_family == "REGULAR") & (kseed.training_graphs == 800)
                   & (kseed.training_tapes == 3) & (kseed.updates_per_epoch == 0) & (kseed.training_epochs == 120)]
    spec_deg = main_k.groupby(["checkpoint_policy", "degree"])[["states", "candidate_correct", "unsafe", "forbidden"]].sum().reset_index()
    spec_deg["candidate_accuracy"] = spec_deg.candidate_correct / spec_deg.states
    succ = spec.groupby(["checkpoint_policy", "degree"]).agg(evaluations=("evaluations", "sum"), successes=("successes", "sum"),
                                                              rounds_sum=("rounds_sum", "sum")).reset_index()
    spec_deg = spec_deg.merge(succ, on=["checkpoint_policy", "degree"])
    spec_deg["mean_rounds"] = spec_deg.rounds_sum / spec_deg.evaluations
    per_seed = main_k.groupby(["checkpoint_policy", "degree", "seed"])[["states", "candidate_correct"]].sum().reset_index()
    per_seed["acc"] = per_seed.candidate_correct / per_seed.states
    spec_deg = spec_deg.merge(per_seed.groupby(["checkpoint_policy", "degree"]).acc.agg(["min", "max"]).reset_index()
                              .rename(columns={"min": "min_replica_candidate", "max": "max_replica_candidate"}),
                              on=["checkpoint_policy", "degree"])
    spec_deg.to_csv(out / "m6b-specialists-by-degree.csv", index=False)

    # Sweeps, permutation checks, benchmark, training cost -----------------------------------
    sweeps = sorted((root / "sweeps").glob("*-sweep-by-k.csv"))
    sweep = pd.concat([pd.read_csv(p) for p in sweeps], ignore_index=True) if sweeps else pd.DataFrame()
    if len(sweep):
        sweep.to_csv(out / "ucgnn-partition-sweep-by-k.csv", index=False)
        pd.concat([pd.read_csv(p) for p in sorted((root / "sweeps").glob("*-sweep.csv"))]).to_csv(out / "ucgnn-partition-sweep.csv", index=False)
    perms = []
    for path in sorted((root / "permcheck").glob("*.json")):
        value = json.loads(path.read_text())
        perms.append({"view": path.stem, "graphs": value["graphs"], "identical": value["identical"]})
    if perms:
        pd.DataFrame(perms).to_csv(out / "ucgnn-permutation-consistency.csv", index=False)
    bench = root / "benchmark" / "forward-cost.csv"
    if bench.exists():
        pd.read_csv(bench).to_csv(out / "ucgnn-forward-cost.csv", index=False)
    training = []
    for history in sorted((root / "runs").glob("*/history.json")):
        rows = json.loads(history.read_text())
        config = json.loads((history.parent / "config.json").read_text())
        manifest = json.loads((history.parent / "data_manifest.json").read_text())
        training.append({"run": history.parent.name, "experiment": config["name"], "seed": config["seed"],
                         "variant": config["variant"], "degrees": "+".join(map(str, config["degrees"])),
                         "graphs_per_degree": "+".join(map(str, config["graphs_per_degree"])), "epochs": len(rows),
                         "training_rounds": manifest["training_rounds"], "trajectories": manifest["training_trajectories"],
                         "updates": rows[-1]["updates"], "elapsed_seconds": rows[-1]["elapsed_seconds"],
                         "data_generation_seconds": rows[-1]["data_generation_seconds"],
                         "gpu_peak_mb": max(r["gpu_peak_bytes"] for r in rows) / 2**20,
                         "final_val_candidate": rows[-1]["validation_local"]["c_correct_rate"],
                         "final_val_action": rows[-1]["validation_local"]["action_correct_teacher_rate"]})
    pd.DataFrame(training).to_csv(out / "ucgnn-training.csv", index=False)

    # Figure 1: candidate agreement by degree ------------------------------------------------
    primary = overall[(overall.policy == "epoch-120") & (overall.variant == "NORMALIZED")]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name in ("E1", "E2", "E3"):
        rows = primary[primary.experiment == name].sort_values("degree")
        if not len(rows):
            continue
        color = MODEL_COLOR[name]
        ax.plot(rows.degree, 100 * rows.candidate_accuracy, color=color, linewidth=2, zorder=2)
        seen = rows.degree.isin(TRAINING[name])
        ax.scatter(rows.degree[seen], 100 * rows.candidate_accuracy[seen], s=40, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.scatter(rows.degree[~seen], 100 * rows.candidate_accuracy[~seen], s=40, facecolor=SURFACE, edgecolor=color,
                   linewidth=1.8, zorder=3)
        last = rows.iloc[-1]
        ax.annotate(f"{name}", (last.degree, 100 * last.candidate_accuracy), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    sd = spec_deg[spec_deg.checkpoint_policy == "epoch-120"].sort_values("degree")
    ax.plot(sd.degree, 100 * sd.candidate_accuracy, color=MODEL_COLOR["M6B"], linewidth=2, linestyle=(0, (4, 2)), zorder=2)
    ax.scatter(sd.degree, 100 * sd.candidate_accuracy, s=40, marker="s", color=MODEL_COLOR["M6B"], zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.annotate("M6B especialistas", (sd.degree.iloc[-1], 100 * sd.candidate_accuracy.iloc[-1]), xytext=(6, -8),
                textcoords="offset points", color=INK2, fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MODEL_COLOR[m], linewidth=2, label=label) for m, label in
               (("E1", "E1: entrenado en Δ=3"), ("E2", "E2: Δ=3 y 10"), ("E3", "E3: Δ=3,5,8,10"))]
    handles += [Line2D([], [], color=MODEL_COLOR["M6B"], linewidth=2, linestyle=(0, (4, 2)), marker="s",
                       label="M6B: un especialista por Δ (media de 20)"),
                Line2D([], [], color=MUTED, marker="o", linestyle="", label="grado entrenado"),
                Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, linestyle="", label="grado no observado")]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
    style(ax, "Concordancia del candidato con el profesor (época 120, réplicas agrupadas)", "Δ (grado máximo; paleta Δ+1)",
          "candidatos iguales al profesor (%)")
    ax.set_xticks(sorted(set(primary.degree)))
    save(fig, out / "fig-candidate-by-degree")

    # Figure 2: learned threshold shift by k --------------------------------------------------
    if len(sweep):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
        for ax, degree in zip(axes, (10, 15)):
            for name, color in (("e1", MODEL_COLOR["E1"]), ("e2", MODEL_COLOR["E2"]), ("e3", MODEL_COLOR["E3"])):
                rows = sweep[(sweep.view == f"{name}-s00-epoch-120") & (sweep.degree == degree)].sort_values("k")
                if len(rows):
                    ax.plot(rows.k, 1000 * rows.signed_threshold_error, color=color, linewidth=2, marker="o", markersize=4,
                            label=name.upper())
            ax.axhline(0, color=AXIS, linewidth=1)
            style(ax, f"estados de Δ={degree}", "k = colores disponibles", "desplazamiento medio del umbral (×10⁻³)" if degree == 10 else None)
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
        fig.suptitle("Umbrales aprendidos de selección frente a j/k exactos", color=INK, fontsize=10, x=0.02, ha="left")
        save(fig, out / "fig-threshold-shift")

    # Figure 3: measured forward cost ------------------------------------------------------------
    if bench.exists():
        b = pd.read_csv(bench)
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
        degrees = [d for d in (3, 10, 20) if d in set(b.degree)]
        for i, degree in enumerate(degrees):
            rows = b[b.degree == degree].sort_values("n")
            color = ORDINAL[min(2 * i, len(ORDINAL) - 1)]
            axes[0].plot(rows.n, rows.ucgnn_forward_peak_mb, color=color, linewidth=2, marker="o", markersize=4, label=f"UCGNN Δ={degree}")
            axes[0].plot(rows.n, rows.m4b_forward_peak_mb, color=color, linewidth=1.5, linestyle=(0, (4, 2)), marker="s", markersize=4,
                         label=f"M4B Δ={degree}")
            axes[1].plot(rows.n, 1000 * rows.ucgnn_forward_seconds, color=color, linewidth=2, marker="o", markersize=4)
            axes[1].plot(rows.n, 1000 * rows.m4b_forward_seconds, color=color, linewidth=1.5, linestyle=(0, (4, 2)), marker="s", markersize=4)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        style(axes[0], "memoria GPU de un forward", "N (nodos)", "MiB sobre la línea base")
        style(axes[1], "tiempo de un forward", "N (nodos)", "ms (mediana)")
        axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2)
        save(fig, out / "fig-forward-cost")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
