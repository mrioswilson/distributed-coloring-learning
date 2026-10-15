"""Generate final reports in bounded cohorts, then concatenate their tables."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from dcl.study.config import Config, digest
from dcl.study.storage import atomic_json, file_hash, now


def cohort_key(entry):
    config = Config.from_dict(entry["config"])
    return f"d{config.degree}-g{config.graphs}-t{config.tapes}-u{config.updates_per_epoch or 0}-{entry['checkpoint_policy']}"


def link_directory(target, link):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected cohort input link: {link}")
    elif link.exists():
        raise RuntimeError(f"Refusing to replace cohort input: {link}")
    else:
        link.symlink_to(target.resolve(), target_is_directory=True)


def combine_csv(paths, destination):
    """Copy compatible CSV streams without loading the observations into RAM."""
    temporary = destination.with_name(destination.name + ".partial")
    header = None
    with temporary.open("w") as output:
        for path in paths:
            if not path.exists():
                continue
            with path.open() as source:
                candidate = source.readline()
                if not candidate.strip():
                    continue
                if header is None:
                    header = candidate
                    output.write(header)
                elif candidate != header:
                    raise RuntimeError(f"Incompatible CSV columns: {path}")
                shutil.copyfileobj(source, output)
    temporary.replace(destination)


def aggregate_cohorts(root, frozen_path):
    from dcl.study import report
    import pandas as pd
    frozen = json.loads(frozen_path.read_text())
    evaluation_id = digest(frozen)
    expected_per_model = len(frozen["families"]) * sum(frozen["counts"].values())
    cohorts, configs = defaultdict(list), {}
    for entry in frozen["models"]:
        configs[Config.from_dict(entry["config"]).id] = Config.from_dict(entry["config"])
        directory = root / "evaluations" / evaluation_id / entry["id"]
        if len(list(directory.glob("*.json"))) != expected_per_model:
            raise RuntimeError(f"Final evaluation is incomplete: {entry['id']}")
        cohorts[cohort_key(entry)].append(entry)
    output = root / "reports"
    output.mkdir(exist_ok=True)
    training = output / "training-input"
    for config in configs.values():
        link_directory(root / "runs" / config.id, training / "runs" / config.id)
    training_identity = {"freeze_sha256": file_hash(frozen_path), "stage": "training"}
    marker = training / "complete.json"
    if not marker.exists() or json.loads(marker.read_text()) != training_identity:
        report.aggregate(training)
        atomic_json(marker, training_identity)
    training_reports = training / "reports"
    for path in training_reports.iterdir():
        if path.name in ("learning.csv", "training-cost.csv", "validation-per-graph.csv") or path.suffix in (".png", ".pdf"):
            shutil.copy2(path, output / path.name)

    destinations = []
    original_plot = report._plot
    # Learning curves are already generated once for the entire training cohort.
    def final_plot(frame, x, *args, **kwargs):
        if x != "epoch":
            return original_plot(frame, x, *args, **kwargs)
    report._plot = final_plot
    try:
        for key, entries in sorted(cohorts.items()):
            workspace = output / "cohorts" / key
            identity = {"freeze_sha256": file_hash(frozen_path), "entries": [entry["id"] for entry in entries]}
            marker = workspace / "complete.json"
            for entry in entries:
                config = Config.from_dict(entry["config"])
                link_directory(root / "runs" / config.id, workspace / "runs" / config.id)
                link_directory(root / "evaluations" / evaluation_id / entry["id"],
                               workspace / "evaluations" / evaluation_id / entry["id"])
            if not marker.exists() or json.loads(marker.read_text()) != identity:
                report.aggregate(workspace)
                atomic_json(marker, identity)
            destinations.append(workspace / "reports")
            print(f"Completed scientific report {key} ({len(entries)} checkpoint views)", flush=True)
    finally:
        report._plot = original_plot
    for filename in ("final-per-graph.csv", "k-per-graph.csv", "k-summary.csv", "success-by-seed.csv",
                     "success-between-seeds.csv", "paired-comparisons.csv", "paired-structural.csv",
                     "paired-between-seeds.csv", "cost-quality.csv"):
        combine_csv([directory / filename for directory in destinations], output / filename)
    summary = pd.read_csv(output / "success-by-seed.csv")
    keys = ["method", "degree", "distribution", "n", "training_family", "training_graphs", "training_tapes",
            "training_epochs", "updates_per_epoch", "checkpoint_policy"]
    for x, y in (("n", "success_rate"), ("degree", "success_rate"), ("training_graphs", "success_rate"),
                 ("training_tapes", "success_rate"), ("seed", "success_rate"), ("n", "rounds_mean")):
        report._plot(summary, x, y, output / f"final-{x}-{y}", "Prueba final independiente", [key for key in keys if key != x])
    epochs = summary[summary.checkpoint_policy.str.startswith("epoch-")].copy()
    epochs["checkpoint_epoch"] = epochs.checkpoint_policy.str.removeprefix("epoch-").astype(int)
    report._plot(epochs, "checkpoint_epoch", "success_rate", output / "final-epochs", "Éxito frente a épocas",
                 [key for key in keys if key != "checkpoint_policy"])
    links = "\n".join(f"- [{path.parent.name}](cohorts/{path.parent.name}/reports/REPORT.md)" for path in destinations)
    (output / "REPORT.md").write_text(
        "# Resultados finales del estudio M6B\n\n"
        f"Generados: {now()}. Entrenamientos: {len(configs)}. Vistas de checkpoints: {len(frozen['models'])}. "
        f"Evaluaciones por grafo: {len(frozen['models']) * expected_per_model}.\n\n"
        "Las tablas combinadas se obtuvieron de resultados originales, procesados por grado, presupuesto y política "
        "de checkpoint para limitar la memoria. Cada grupo conserva juntos los métodos y las familias de entrenamiento "
        "que se comparan de forma emparejada. Los mismos grafos se reutilizan entre modelos, por lo que no son ensayos "
        "independientes entre modelos. La validación eligió las ampliaciones antes de abrir la prueba final. "
        "Una tasa observada de 100% no demuestra corrección universal.\n\n" + links + "\n")
    atomic_json(output / "final-report-state.json", {"status": "complete", "created_utc": now(),
                "freeze_sha256": file_hash(frozen_path), "checkpoint_views": len(frozen["models"]),
                "graph_evaluations": len(frozen["models"]) * expected_per_model, "cohorts": len(cohorts)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/m6study-release"))
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    aggregate_cohorts(args.root.resolve(), args.freeze.resolve())


if __name__ == "__main__":
    main()
