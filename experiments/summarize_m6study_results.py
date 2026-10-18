"""Summarize a complete public export, keeping replicas as statistical units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dcl.study.report import seed_interval


GROUPS = ["method", "degree", "training_family", "training_graphs", "training_tapes",
          "updates_per_epoch", "training_epochs", "checkpoint_policy", "n"]


def summarize(directory):
    metadata = json.loads((directory / "snapshot.json").read_text())
    if metadata["status"] != "complete":
        raise RuntimeError("A complete evaluation export is required")
    raw = pd.read_csv(directory / "success-by-seed.csv")
    assert raw.graphs.sum() == metadata["graph_evaluations_planned"]
    # Combine the two families within each replica before estimating dispersion.
    replicas = raw.groupby(GROUPS + ["seed"], as_index=False).agg(
        successes=("successes", "sum"), graphs=("graphs", "sum"),
        rounds_sum=("rounds_sum", "sum"), unsafe_commits=("unsafe_commits", "sum"))
    replicas["success_rate"] = replicas.successes / replicas.graphs
    replicas["mean_rounds"] = replicas.rounds_sum / replicas.graphs
    rows = []
    for key, group in replicas.groupby(GROUPS):
        low, high = seed_interval(group.success_rate)
        rows.append({**dict(zip(GROUPS, key)), "seeds": len(group),
                     "successes": int(group.successes.sum()), "evaluations": int(group.graphs.sum()),
                     "mean_success_rate": group.success_rate.mean(),
                     "median_success_rate": group.success_rate.median(),
                     "std_success_rate": group.success_rate.std(),
                     "min_success_rate": group.success_rate.min(),
                     "max_success_rate": group.success_rate.max(),
                     "bootstrap_seed_low": low, "bootstrap_seed_high": high,
                     "mean_rounds": group.mean_rounds.mean(),
                     "unsafe_commits": int(group.unsafe_commits.sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(directory / "seed-summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = summary[(summary.training_family == "REGULAR") & (summary.training_graphs == 800)
                   & (summary.training_tapes == 3) & (summary.updates_per_epoch == 0)
                   & (summary.checkpoint_policy == "epoch-120") & (summary.n == 40960)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, color in (("COUNTERFACTUAL", "#2468a2"), ("OWN_ONLY", "#cd7420")):
        group = main[main.method == method].sort_values("degree")
        assert len(group) == 4 and (group.seeds == 20).all()
        axes[0].errorbar(group.degree, 100 * group.mean_success_rate,
                         yerr=100 * group.std_success_rate, marker="o", capsize=3,
                         color=color, label=method)
        axes[1].plot(group.degree, group.mean_rounds, marker="o", color=color, label=method)
    for ax in axes:
        ax.set_xlabel("Grado máximo Δ")
        ax.set_xticks([3, 5, 8, 10])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Éxito medio entre réplicas (%)")
    axes[0].set_ylim(0, 115)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_title("Media ± desviación estándar, 20 réplicas")
    axes[1].set_ylabel("Rondas medias")
    axes[1].set_title("Ambas familias de prueba combinadas")
    fig.suptitle("Entrenamiento N=40; prueba N=40.960; época 120")
    fig.tight_layout()
    save_figure(fig, directory / "main-comparison")
    plt.close(fig)

    budgets = [(800, 3, 0), (200, 3, 0), (2400, 1, 0), (800, 3, 256)]
    grid = []
    for degree in (3, 5, 8, 10):
        values = []
        for graphs, tapes, updates in budgets:
            group = summary[(summary.method == "COUNTERFACTUAL") & (summary.training_family == "REGULAR")
                            & (summary.degree == degree) & (summary.training_graphs == graphs)
                            & (summary.training_tapes == tapes) & (summary.updates_per_epoch == updates)
                            & (summary.checkpoint_policy == "epoch-120") & (summary.n == 40960)]
            assert len(group) == 1 and group.iloc[0].seeds == 20
            values.append(100 * group.iloc[0].mean_success_rate)
        grid.append(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    heat = ax.imshow(grid, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(4), ["800 × 3\n120 épocas", "200 × 3\n120 épocas",
                            "2.400 × 1\n120 épocas", "800 × 3\n30.720 actualizaciones"])
    ax.set_yticks(range(4), ["Δ=3", "Δ=5", "Δ=8", "Δ=10"])
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                    color="white" if value > 65 else "black")
    ax.set_title("Counterfactual: grafos × tapes; prueba N=40.960; época 120\n"
                 "20 réplicas por celda, ambas familias de prueba")
    fig.colorbar(heat, ax=ax, label="Éxito medio (%)")
    fig.tight_layout()
    save_figure(fig, directory / "budget-comparison")
    plt.close(fig)


def save_figure(fig, destination):
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
