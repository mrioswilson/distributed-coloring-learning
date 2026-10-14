"""CSV tables and scientific plots computed solely from persistent raw data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, norm


def binomial_intervals(successes, total, confidence=0.95):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial denominator")
    alpha, z = 1 - confidence, norm.ppf((1 + confidence) / 2)
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    exact_low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    exact_high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return {"wilson_low": max(0.0, center - radius), "wilson_high": min(1.0, center + radius),
            "exact_low": exact_low, "exact_high": exact_high}


def seed_interval(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(20261014)
    means = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_statistics(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) != len(b) or not len(a):
        raise ValueError("Paired observations require equal nonzero denominators")
    a_only, b_only = int((a & ~b).sum()), int((b & ~a).sum())
    discordant = a_only + b_only
    return {"paired_graphs": len(a), "a_only": a_only, "b_only": b_only,
            "success_difference": float(a.mean() - b.mean()),
            "mcnemar_exact_p": float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0}


def _plot(frame, x, y, destination, title, groups, ylabel=None):
    if frame.empty or x not in frame or y not in frame:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.3))
    groups = [key for key in groups if frame[key].nunique(dropna=False) > 1]
    grouped = frame.groupby(groups, dropna=False) if groups else [("cohort", frame)]
    for label, group in grouped:
        summary = group.groupby(x)[y].agg(["mean", "std"]).sort_index()
        label = "/".join(map(str, label)) if isinstance(label, tuple) else str(label)
        if x == "training_seconds":
            ax.scatter(summary.index, summary["mean"], s=12, label=label)
        else:
            ax.plot(summary.index, summary["mean"], marker="o", markersize=3, label=label)
        spread = summary["std"].fillna(0)
        ax.fill_between(summary.index, summary["mean"] - spread, summary["mean"] + spread, alpha=0.12)
    if x in ("n", "training_graphs", "training_tapes"):
        ax.set_xscale("log", base=2)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    plt.close(fig)


def aggregate(root):
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    learning, rollout_validation, run_rows = [], [], []
    for config_file in sorted((root / "runs").glob("*/config.json")):
        config = json.loads(config_file.read_text())
        history_path = config_file.parent / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        base = {"experiment_id": config_file.parent.name, "method": config["method"], "degree": config["degree"],
                "training_family": config["family"], "seed": config["seed"], "training_graphs": config["graphs"],
                "training_tapes": config["tapes"], "updates_per_epoch": config["updates_per_epoch"]}
        base["updates_per_epoch"] = config["updates_per_epoch"] or 0
        base["training_epochs"] = config["epochs"]
        for record in history:
            learning.append({**base, "epoch": record["epoch"], "updates": record["updates"],
                             "rounds_seen": record["rounds_seen"], "nodes_seen": record["nodes_seen"],
                             "elapsed_seconds": record["elapsed_seconds"], **record["train"],
                             **{"val_" + k: v for k, v in record["validation_local"].items()}})
            if record["validation_rollouts"]:
                for graph in record["validation_rollouts"]["per_graph"]:
                    rollout_validation.append({**base, "epoch": record["epoch"], **{k: v for k, v in graph.items() if k != "by_k"}})
        if history:
            run_rows.append({**base, "epoch": history[-1]["epoch"], "complete": history[-1]["epoch"] == config["epochs"],
                             "updates": history[-1]["updates"], "rounds_seen": history[-1]["rounds_seen"],
                             "training_seconds": history[-1]["elapsed_seconds"] + history[-1]["data_generation_seconds"]})
    learning = pd.DataFrame(learning)
    learning.to_csv(output / "learning.csv", index=False)
    pd.DataFrame(rollout_validation).to_csv(output / "validation-per-graph.csv", index=False)
    pd.DataFrame(run_rows).to_csv(output / "training-cost.csv", index=False)
    for metric in ("loss", "participation_loss", "candidate_loss", "own_commit_loss", "counterfactual_commit_loss",
                   "val_p_correct_rate", "val_c_correct_rate", "val_commit_correct_own_rate", "val_action_correct_teacher_rate"):
        _plot(learning, "epoch", metric, output / metric, "Validación y aprendizaje por época", ["method", "degree", "training_family", "training_graphs", "training_tapes", "training_epochs", "updates_per_epoch"])

    raw = [json.loads(path.read_text()) for path in sorted((root / "evaluations").glob("*/*/*.json"))]
    final = pd.DataFrame([{k: v for k, v in row.items() if k != "by_k"} for row in raw])
    final.to_csv(output / "final-per-graph.csv", index=False)
    k_rows = [{"experiment_id": row["experiment_id"], "method": row["method"], "degree": row["degree"],
               "training_family": row["training_family"], "training_graphs": row["training_graphs"],
               "training_tapes": row["training_tapes"], "training_epochs": row["training_epochs"],
               "updates_per_epoch": row["updates_per_epoch"], "checkpoint_policy": row["checkpoint_policy"],
               "distribution": row["distribution"], "n": row["n"], "seed": row["seed"], "graph_id": row["graph_id"],
               "k": int(k), **counts} for row in raw for k, counts in row["by_k"].items()]
    k_frame = pd.DataFrame(k_rows)
    k_frame.to_csv(output / "k-per-graph.csv", index=False)
    summary, seed_summary, paired, structural = [], [], [], []
    group_keys = ["method", "degree", "distribution", "n", "training_family", "training_graphs", "training_tapes", "training_epochs", "updates_per_epoch", "checkpoint_policy"]
    if not final.empty:
        for label, group in final.groupby(group_keys + ["seed"]):
            successes, total = int(group["success"].sum()), len(group)
            summary.append({**dict(zip(group_keys + ["seed"], label)), "successes": successes, "graphs": total,
                            "success_rate": successes / total, **binomial_intervals(successes, total),
                            "rounds_mean": group["rounds"].mean(), "rounds_median": group["rounds"].median(),
                            "rounds_std": group["rounds"].std(), "rounds_min": group["rounds"].min(), "rounds_max": group["rounds"].max(),
                            "unsafe_commits": group["unsafe_commits"].sum(), "inference_seconds_mean": group["inference_seconds"].mean()})
        summary_frame = pd.DataFrame(summary)
        for label, group in summary_frame.groupby(group_keys):
            values = group["success_rate"]
            low, high = seed_interval(values)
            seed_summary.append({**dict(zip(group_keys, label)), "seeds": len(group), "mean": values.mean(),
                                 "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max(),
                                 "seed_bootstrap_low": low, "seed_bootstrap_high": high,
                                 "graphs_per_seed_min": group["graphs"].min(), "graphs_per_seed_max": group["graphs"].max()})
        for label, group in final.groupby(["pair_id", "degree", "distribution", "n"]):
            for a, b in [("COUNTERFACTUAL", "OWN_ONLY"), ("OWN_ONLY", "ORACLE_MATCHED"), ("COUNTERFACTUAL", "ORACLE_MATCHED")]:
                ga, gb = group[group.method == a], group[group.method == b]
                shared = ga.merge(gb, on=["graph_id", "graph_seed", "tape_seed", "fingerprint"], suffixes=("_a", "_b"))
                if shared.empty:
                    continue
                paired.append({"pair_id": label[0], "degree": label[1], "distribution": label[2], "n": label[3],
                               **{key: shared[key + "_a"].iloc[0] for key in ("training_family", "training_graphs", "training_tapes", "training_epochs", "updates_per_epoch", "checkpoint_policy")},
                               "a": a, "b": b, "seed": int(shared.seed_a.iloc[0]),
                               **paired_statistics(shared.success_a, shared.success_b),
                               "rounds_difference": float((shared.rounds_a - shared.rounds_b).mean())})
        structural_keys = ["method", "seed", "degree", "distribution", "n", "training_graphs", "training_tapes", "training_epochs", "updates_per_epoch", "checkpoint_policy"]
        for label, group in final.groupby(structural_keys):
            ga = group[group.training_family == "REGULAR"]
            gb = group[group.training_family == "MIXED"]
            shared = ga.merge(gb, on=["graph_id", "graph_seed", "tape_seed", "fingerprint"], suffixes=("_a", "_b"))
            if not shared.empty:
                structural.append({**dict(zip(structural_keys, label)), "a": "REGULAR", "b": "MIXED",
                                   **paired_statistics(shared.success_a, shared.success_b),
                                   "rounds_difference": float((shared.rounds_a - shared.rounds_b).mean())})
        for x, y, title in [("n", "success_rate", "Éxito frente al tamaño"), ("degree", "success_rate", "Éxito frente al grado"),
                             ("training_graphs", "success_rate", "Diversidad de grafos"), ("training_tapes", "success_rate", "Random tapes por grafo"),
                             ("seed", "success_rate", "Variación entre semillas"), ("n", "rounds_mean", "Rondas de ejecución")]:
            _plot(summary_frame, x, y, output / f"final-{x}-{y}", title, [key for key in group_keys if key != x])
        by_epoch = summary_frame[summary_frame.checkpoint_policy.str.startswith("epoch-")].copy()
        if not by_epoch.empty:
            by_epoch["checkpoint_epoch"] = by_epoch.checkpoint_policy.str.removeprefix("epoch-").astype(int)
            _plot(by_epoch, "checkpoint_epoch", "success_rate", output / "final-epochs", "Éxito frente a épocas", [key for key in group_keys if key != "checkpoint_policy"])
        if run_rows:
            quality_cost = summary_frame.merge(pd.DataFrame(run_rows), on=["method", "degree", "training_family", "seed", "training_graphs", "training_tapes", "training_epochs", "updates_per_epoch"])
            quality_cost.to_csv(output / "cost-quality.csv", index=False)
            _plot(quality_cost, "training_seconds", "success_rate", output / "cost-quality", "Coste y fiabilidad global", group_keys)
    pd.DataFrame(summary).to_csv(output / "success-by-seed.csv", index=False)
    pd.DataFrame(seed_summary).to_csv(output / "success-between-seeds.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired-comparisons.csv", index=False)
    pd.DataFrame(structural).to_csv(output / "paired-structural.csv", index=False)
    paired_seed_rows = []
    if paired:
        keys = ["a", "b", "degree", "distribution", "n", "training_family", "training_graphs", "training_tapes", "training_epochs", "updates_per_epoch", "checkpoint_policy"]
        for label, group in pd.DataFrame(paired).groupby(keys):
            low, high = seed_interval(group.success_difference)
            paired_seed_rows.append({**dict(zip(keys, label)), "seeds": len(group),
                                     "mean_success_difference": group.success_difference.mean(),
                                     "median_success_difference": group.success_difference.median(),
                                     "bootstrap_low": low, "bootstrap_high": high})
    pd.DataFrame(paired_seed_rows).to_csv(output / "paired-between-seeds.csv", index=False)
    if not k_frame.empty:
        keys = group_keys + ["k"]
        totals = k_frame.groupby(keys)[["states", "candidate_correct", "forbidden", "offers", "executed", "unsafe", "valid", "unnecessary_wait"]].sum().reset_index()
        totals["candidate_accuracy"] = totals.candidate_correct / totals.states.replace(0, np.nan)
        totals["state_frequency"] = totals.states / totals.groupby([key for key in keys if key != "k"]).states.transform("sum").replace(0, np.nan)
        totals["forbidden_per_state"] = totals.forbidden / totals.states.replace(0, np.nan)
        totals["forbidden_per_offer"] = totals.forbidden / totals.offers.replace(0, np.nan)
        totals["unsafe_per_commit"] = totals.unsafe / totals.executed.replace(0, np.nan)
        totals["unsafe_per_state"] = totals.unsafe / totals.states.replace(0, np.nan)
        totals["unnecessary_wait_per_valid"] = totals.unnecessary_wait / totals.valid.replace(0, np.nan)
        totals.to_csv(output / "k-summary.csv", index=False)
        for metric in ("candidate_accuracy", "forbidden_per_state", "unsafe_per_commit", "unnecessary_wait_per_valid"):
            _plot(totals, "k", metric, output / f"k-{metric}", "Diagnóstico por colores disponibles", group_keys)
    report = ("# Resultados generados desde datos originales\n\n"
              f"Ejecuciones con épocas persistidas: {len(run_rows)}. Completas: {sum(r['complete'] for r in run_rows)}.\n\n"
              f"Observaciones del test final persistidas: {len(raw)}.\n\n"
              "La validación sirve para seleccionar checkpoints. Las tablas final-* contienen exclusivamente prueba final congelada.\n\n"
              "Los intervalos por semilla son Wilson y Clopper–Pearson (95%). El bootstrap entre semillas condiciona en el conjunto de prueba observado. "
              "Los grafos compartidos no son observaciones independientes entre modelos. McNemar se calcula sobre pares del mismo grafo y tape; "
              "los valores p son descriptivos, sin corrección por comparaciones múltiples. Los estados locales están correlacionados, por lo que sus tasas "
              "se acompañan de denominadores y no se tratan como ensayos binomiales independientes.\n\n"
              "Las bandas de los gráficos muestran desviación estándar descriptiva. Una tasa observada de 100% no demuestra corrección universal.\n")
    (output / "REPORT.md").write_text(report)
    print(f"Generated CSV, PDF, PNG and report: {output}")
