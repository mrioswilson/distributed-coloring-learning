"""Export a portable snapshot of completed final evaluations for GitHub."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from dcl.study.config import Config, digest
from dcl.study.report import binomial_intervals, paired_statistics
from dcl.study.storage import file_hash


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def compare_cases(a, b, metadata):
    if set(a) != set(b):
        raise ValueError("Paired models must have identical graph/tape cases")
    grouped = defaultdict(list)
    for graph_id in sorted(a):
        grouped[metadata[graph_id]].append((a[graph_id], b[graph_id]))
    rows = []
    for (family, n), values in sorted(grouped.items()):
        left, right = [v[0] for v in values], [v[1] for v in values]
        rows.append({"distribution": family, "n": n,
                     **paired_statistics([r[0] for r in left], [r[0] for r in right]),
                     "a_successes": sum(r[0] for r in left), "b_successes": sum(r[0] for r in right),
                     "rounds_difference": statistics.mean(x[1] - y[1] for x, y in values),
                     "a_unsafe_commits": sum(r[2] for r in left), "b_unsafe_commits": sum(r[2] for r in right)})
    return rows


def export(root, output):
    if output.exists():
        raise RuntimeError("Snapshot overwrite refused; choose a new output directory")
    frozen_path = root / "frozen-final-test.json"
    frozen = json.loads(frozen_path.read_text())
    frozen_hash, evaluation_id = file_hash(frozen_path), digest(frozen)
    expected = len(frozen["families"]) * sum(frozen["counts"].values())
    complete, stored = [], 0
    for entry in frozen["models"]:
        files = sorted((root / "evaluations" / evaluation_id / entry["id"]).glob("*.json"))
        stored += len(files)
        if len(files) == expected:
            complete.append((entry, files))
    captured = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    seeds, failures, local, published, pairs = [], [], [], [], []
    identities, case_metadata, pairing = {}, {}, defaultdict(dict)
    for entry, files in complete:
        config = Config.from_dict(entry["config"])
        base = {"source_experiment_id": config.id, "method": config.method, "degree": config.degree,
                "training_family": config.family, "training_graphs": config.graphs, "training_tapes": config.tapes,
                "updates_per_epoch": config.updates_per_epoch or 0, "training_epochs": config.epochs,
                "seed": config.seed, "checkpoint_policy": entry["checkpoint_policy"],
                "checkpoint_epoch": entry["selected_epoch"]}
        totals, by_k, cases, counts = defaultdict(Counter), defaultdict(Counter), {}, Counter()
        for path in files:
            row = json.loads(path.read_text())
            if row["evaluation_id"] != evaluation_id or row["checkpoint_sha256"] != entry["checkpoint_sha256"]:
                raise RuntimeError("An observation does not belong to the frozen checkpoint")
            graph_id = row["graph_id"]
            identity = row["graph_seed"], row["tape_seed"], row["fingerprint"]
            key = config.degree, graph_id
            if key in identities and identities[key] != identity:
                raise RuntimeError("Paired test graphs or random tapes differ")
            identities[key] = identity
            case_metadata[graph_id] = row["distribution"], row["n"]
            if graph_id in cases:
                raise RuntimeError("Duplicate graph observation")
            cases[graph_id] = bool(row["success"]), row["rounds"], row["unsafe_commits"]
            group = row["distribution"], row["n"]
            counts[group] += 1
            total = totals[group]
            total["graphs"] += 1
            total["successes"] += int(row["success"])
            total["finished"] += int(row["finished"])
            total["rounds_sum"] += row["rounds"]
            for name in ("unsafe_commits", "monochromatic_edges", "fixed_neighbor_conflicts", "simultaneous_conflicts",
                         "valid_commits_rejected", "states", "uncolored_nodes"):
                total[name] += row[name]
            for k, value in row["by_k"].items():
                by_k[group + (int(k),)].update(value)
            if not row["success"]:
                failures.append({**base, "distribution": group[0], "n": group[1], "graph_id": graph_id,
                                 "finished": row["finished"], "monochromatic_edges": row["monochromatic_edges"],
                                 "uncolored_nodes": row["uncolored_nodes"], "unsafe_commits": row["unsafe_commits"]})
        expected_counts = Counter({(family, int(n)): count for family in frozen["families"] for n, count in frozen["counts"].items()})
        if counts != expected_counts:
            raise RuntimeError("A completed checkpoint has incorrect family/size denominators")
        for (family, n), total in sorted(totals.items()):
            seeds.append({**base, "distribution": family, "n": n, **dict(total),
                          "success_rate": total["successes"] / total["graphs"],
                          "mean_rounds": total["rounds_sum"] / total["graphs"],
                          **binomial_intervals(total["successes"], total["graphs"])})
        for (family, n, k), total in sorted(by_k.items()):
            local.append({**base, "distribution": family, "n": n, "k": k, **dict(total)})
        pairing[config.pair_id, entry["checkpoint_policy"]][config.method] = base, cases
        published.append({"id": entry["id"], "source_experiment_id": config.id,
                          "checkpoint_policy": entry["checkpoint_policy"], "selected_epoch": entry["selected_epoch"],
                          "checkpoint_sha256": entry["checkpoint_sha256"], "graphs": expected})
    for (pair_id, policy), methods in pairing.items():
        for a, b in (("COUNTERFACTUAL", "OWN_ONLY"), ("COUNTERFACTUAL", "ORACLE_MATCHED"),
                     ("OWN_ONLY", "ORACLE_MATCHED"), ("COUNTERFACTUAL", "TRADITIONAL"), ("OWN_ONLY", "TRADITIONAL")):
            if a not in methods or b not in methods:
                continue
            base, left = methods[a]
            _, right = methods[b]
            for result in compare_cases(left, right, case_metadata):
                pairs.append({"pair_id": pair_id, "a": a, "b": b,
                              **{key: value for key, value in base.items() if key not in ("source_experiment_id", "method", "checkpoint_epoch")},
                              **result})
    summary_groups = defaultdict(list)
    group_keys = ("method", "degree", "training_family", "training_graphs", "training_tapes", "updates_per_epoch",
                  "training_epochs", "checkpoint_policy", "distribution", "n")
    for row in seeds:
        summary_groups[tuple(row[key] for key in group_keys)].append(row)
    summary = []
    for key, rows in sorted(summary_groups.items()):
        rates = [row["success_rate"] for row in rows]
        summary.append({**dict(zip(group_keys, key)), "seeds": len(rows),
                        "successes": sum(row["successes"] for row in rows), "model_graph_evaluations": sum(row["graphs"] for row in rows),
                        "distinct_test_graphs": rows[0]["graphs"], "mean_seed_success_rate": statistics.mean(rates),
                        "std_seed_success_rate": statistics.stdev(rates) if len(rates) > 1 else "",
                        "min_seed_success_rate": min(rates), "max_seed_success_rate": max(rates),
                        "mean_rounds": statistics.mean(row["mean_rounds"] for row in rows),
                        "unsafe_commits": sum(row["unsafe_commits"] for row in rows)})
    training = []
    for path in sorted((root / "runs").glob("*/config.json")):
        config = Config.from_dict(json.loads(path.read_text()))
        history = json.loads(path.with_name("history.json").read_text())
        last = history[-1]
        training.append({"experiment_id": config.id, "method": config.method, "degree": config.degree,
                         "training_family": config.family, "graphs": config.graphs, "tapes": config.tapes,
                         "seed": config.seed, "epochs": last["epoch"], "updates_per_epoch": config.updates_per_epoch or 0,
                         "optimizer_updates": last["updates"], "rounds_seen": last["rounds_seen"], "nodes_seen": last["nodes_seen"],
                         "training_seconds": last["elapsed_seconds"] + last["data_generation_seconds"]})
    write_csv(output / "success-by-seed.csv", seeds)
    write_csv(output / "success-summary.csv", summary)
    write_csv(output / "paired-by-seed.csv", pairs)
    write_csv(output / "errors.csv", failures)
    write_csv(output / "k-by-seed.csv", local)
    write_csv(output / "training.csv", training)
    metadata = {"status": "partial", "captured_utc": captured, "protocol": frozen["protocol"],
                "frozen_protocol_sha256": frozen_hash, "evaluation_id": evaluation_id,
                "export_script_sha256": file_hash(Path(__file__)), "selection_uses_final_test": False,
                "training_runs": len(training), "complete_checkpoint_views_exported": len(complete),
                "planned_checkpoint_views": len(frozen["models"]), "graph_evaluations_exported": len(complete) * expected,
                "graph_evaluations_stored_at_capture": stored, "graph_evaluations_planned": len(frozen["models"]) * expected,
                "test_seed_namespace": frozen["test_seed_namespace"], "families": frozen["families"], "counts": frozen["counts"],
                "max_rounds": frozen["max_rounds"], "incomplete_checkpoint_views_excluded": True, "published_checkpoints": published}
    if file_hash(frozen_path) != frozen_hash:
        raise RuntimeError("The frozen protocol changed during export")
    (output / "snapshot.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    make_figures(output, summary, seeds)
    write_readme(output, metadata, summary, failures)
    print(json.dumps({key: value for key, value in metadata.items() if key != "published_checkpoints"}, indent=2))


def make_figures(output, summary, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, marker in (("COUNTERFACTUAL", "o"), ("OWN_ONLY", "s")):
        values = [row for row in summary if row["method"] == method and row["degree"] == 3 and row["training_family"] == "REGULAR"
                  and row["training_graphs"] == 800 and row["training_tapes"] == 3 and row["updates_per_epoch"] == 0 and row["checkpoint_policy"] == "epoch-120"]
        for family, style in (("REGULAR", "-"), ("IRREGULAR", "--")):
            rows = sorted((row for row in values if row["distribution"] == family), key=lambda row: row["n"])
            if rows:
                ax.plot([row["n"] for row in rows], [100 * row["mean_seed_success_rate"] for row in rows],
                        linestyle=style, marker=marker, label=f"{method} / {family} (n={rows[0]['seeds']} seeds)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Graph nodes")
    ax.set_ylabel("Mean success across seeds (%)")
    ax.set_title("Partial independent test: degree 3, epoch 120, no shield")
    ax.set_ylim(0, 102)
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "degree3-generalization.png", dpi=180)
    fig.savefig(output / "degree3-generalization.pdf")
    plt.close(fig)


def write_readme(output, metadata, summary, failures):
    lines = ["# M6B: resultados parciales independientes", "", f"Corte UTC: {metadata['captured_utc']}.", "",
             f"Estado: **evaluación final en curso**. {metadata['training_runs']} entrenamientos terminados; "
             f"{metadata['complete_checkpoint_views_exported']}/{metadata['planned_checkpoint_views']} vistas de checkpoints completas publicadas, "
             f"con {metadata['graph_evaluations_exported']:,} evaluaciones por grafo. Se excluyen las vistas incompletas.", "",
             "Los modelos de referencia se entrenaron en 800 grafos regulares de 40 nodos, tres tapes y 120 épocas. "
             "La inferencia es autónoma, sin shield ni correcciones externas. La vista de época 120 y la selección best "
             "por validación se fijaron antes de abrir la prueba final.", "", "## Pruebas de 40.960 nodos, época 120", "",
             "| Método | Grado | Réplicas | Coloraciones correctas | Commits inseguros |", "|---|---:|---:|---:|---:|"]
    for method in ("COUNTERFACTUAL", "OWN_ONLY"):
        for degree in (3, 5, 8, 10):
            rows = [r for r in summary if r["method"] == method and r["degree"] == degree and r["training_family"] == "REGULAR"
                    and r["training_graphs"] == 800 and r["training_tapes"] == 3 and r["updates_per_epoch"] == 0
                    and r["checkpoint_policy"] == "epoch-120" and r["n"] == 40960]
            if len(rows) == 2:
                lines.append(f"| {method} | {degree} | {rows[0]['seeds']}/20 | "
                             f"{sum(r['successes'] for r in rows)}/{sum(r['model_graph_evaluations'] for r in rows)} | "
                             f"{sum(r['unsafe_commits'] for r in rows)} |")
    lines += ["", "Cada grado utiliza los mismos 20 grafos grandes (10 REGULAR y 10 IRREGULAR) en todas sus réplicas. "
              "Las 400 evaluaciones corresponden a 20 modelos × 20 grafos compartidos. Los grados con menos réplicas "
              "evaluadas son resultados parciales y no deben compararse como cohortes completas.", "",
              "![Generalización emparejada de grado 3](degree3-generalization.png)", "",
              "## Interpretación y límites", "",
              "- La comparación principal de mecanismo usa época 120, donde P/C/message1 están emparejados entre "
              "Counterfactual y Own-only. La selección best puede escoger épocas diferentes.",
              "- Las decisiones locales contrarias al profesor y los errores de coloración global son métricas distintas. "
              "El desglose por k se conserva en k-by-seed.csv, con sus denominadores.",
              "- Los intervalos Wilson y Clopper–Pearson son por modelo/familia/tamaño. No se trata el total de "
              "evaluaciones de modelos distintos como ensayos binomiales independientes. McNemar es por par de modelos "
              "sobre los mismos grafos/tapes, exploratorio y sin corrección por multiplicidad.",
              "- Estos resultados no establecen corrección universal. Faltan partes de los controles, MIXED y presupuestos. "
              "No se modifica el protocolo ni el entrenamiento a partir del test."]
    negative = Counter((r["degree"], r["seed"], r["checkpoint_epoch"]) for r in failures
                       if r["method"] == "COUNTERFACTUAL" and r["training_family"] == "REGULAR"
                       and r["training_graphs"] == 800 and r["training_tapes"] == 3 and r["updates_per_epoch"] == 0
                       and r["checkpoint_policy"] == "best")
    if negative:
        lines += ["", "## Resultados negativos de la selección best", ""]
        for (degree, seed, epoch), count in sorted(negative.items()):
            warmup = " (dentro del warmup)" if epoch <= 10 else ""
            lines.append(f"- Counterfactual Δ={degree}, semilla {seed}, época seleccionada {epoch}{warmup}: "
                         f"{count} coloraciones incorrectas. Se conserva esa vista y su detalle en errors.csv.")
    lines += ["", "## Archivos", "",
              "- [Resumen entre semillas](success-summary.csv)", "- [Resultados e intervalos por semilla](success-by-seed.csv)",
              "- [Comparaciones emparejadas por semilla](paired-by-seed.csv)", "- [Coloraciones incorrectas](errors.csv)",
              "- [Diagnóstico por colores disponibles](k-by-seed.csv)", "- [Presupuesto y coste de entrenamiento](training.csv)",
              "- [Corte, protocolo y hashes de checkpoints](snapshot.json)", "", "## Reproducción", "", "```bash",
              ".venv/bin/python experiments/export_m6study_results.py --root data/m6study-release --output results/m6study/NUEVO_CORTE",
              "```", "", "Este comando requiere los resultados originales del servidor. GitHub contiene tablas, figuras, "
              "código y hashes; los checkpoints y JSON originales por grafo permanecen fuera de Git por su volumen. "
              "La cola de evaluación y el análisis final están documentados en [M6VALIDATION.md](../../../docs/M6VALIDATION.md).", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/m6study-release"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
