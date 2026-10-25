"""Evidence-derived tables/figures; no training or inference in reporting."""
from collections import defaultdict, Counter
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from dcl.study.storage import atomic_json, file_hash
from .inventory import ROOT


def interval(s, n):
    return [0. if s == 0 else float(beta.ppf(.025, s, n-s+1)),
            1. if s == n else float(beta.ppf(.975, s+1, n-s))]


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                       *["| " + " | ".join(map(str, r)) + " |" for r in rows]])


def consolidate(inventory):
    groups = defaultdict(list)
    for r in inventory:
        if r["primary"]:
            groups[r["teacher"], r["architecture"]].append(r)
    rows = []
    for (t, a), cps in sorted(groups.items()):
        labels = []
        for cp in cps:
            meta = cp["training_record"]
            labels.append(meta.get("total_acquired_component_labels", meta.get("accounting", {}).get("queries")))
        r = cps[0]
        rows.append(dict(teacher=t, architecture=a, cohort=r["cohort"],
            interface="anonymous two-exchange" if t in ("LUBY", "JOHANSSON", "COLOR_TRIALS") else "declared extended native",
            seeds=",".join(str(c["seed"]) for c in cps), method=r["method"],
            labels_by_seed=",".join(map(str, labels)), updates=r["update"], parameters=r["parameters"],
            training_domain="n8/12/16/24; maxdegree3; globalK4; six families plus declared coverage in new banks",
            final_domain="n32/48/64; maxdegree3; K4; 7 strata historical or 10 strata new",
            criterion="5 independent checkpoints; 1000/1000 proper complete invariant-valid runs each",
            native_result="5x1000/1000", evidence=r["evidence"],
            new_reproduction="six native cases per checkpoint; BE unextended",
            limitations="finite native battery; heterogeneous label/update/supervision protocols; no universal certification"))
    assert len(rows) == 21
    write_csv(ROOT / "native-21-consolidation.csv", rows)
    atomic_json(ROOT / "native-21-consolidation.json", rows)
    return rows


def main():
    inventory = json.loads((ROOT / "checkpoint-inventory.json").read_text())
    consolidate(inventory)
    summaries, failures, detail = [], [], []
    for stage in ("pilot", "confirmatory"):
        index = ROOT / stage / "index.json"
        if not index.exists():
            raise RuntimeError(f"Incomplete stage: {stage}")
        for summary in json.loads(index.read_text()):
            path = Path(summary["rows_path"])
            assert file_hash(path) == summary["rows_sha256"]
            rows = json.loads(path.read_text())
            assert len(rows) == summary["executions"]
            assert sum(r["success"] for r in rows) == summary["successes"]
            c = summary["condition"]
            s, n = summary["successes"], summary["executions"]
            ci = interval(s, n)
            d = dict(stage=stage, checkpoint=summary["checkpoint_id"], teacher=summary["teacher"],
                architecture=summary["architecture"], method=summary["method"], cohort=summary["cohort"],
                primary=summary["primary"], seed=summary["seed"], condition=c["id"], delta=c["delta"],
                n=c["n"], palette=c["palette"], successes=s, executions=n, success_rate=s/n,
                cp95_lower=ci[0], cp95_upper=ci[1], conflicts=sum(r["conflicts"] for r in rows),
                palette_violations=sum(r["palette_violations"] for r in rows),
                invariant_failures=sum(not r["invariants"] for r in rows),
                incomplete_runs=sum(r["incomplete"] > 0 for r in rows), cutoffs=sum(r["cutoff"] for r in rows),
                mean_steps=float(np.mean([r["steps"] for r in rows])), max_steps=max(r["steps"] for r in rows),
                action_correct=sum(r["action_correct"] for r in rows), action_total=sum(r["action_total"] for r in rows),
                same_transitions=sum(r["same_transitions"] for r in rows), transitions=sum(r["transitions"] for r in rows),
                first_discrepancy_components=json.dumps(Counter(r["first_divergence"]["component"] for r in rows if r["first_divergence"])),
                cpu_seconds=summary["cost"]["cpu_seconds"], rows_sha256=summary["rows_sha256"])
            detail.append(d)
            summaries.append(dict(**summary, stage=stage))
            for r in rows:
                if not r["success"]:
                    failures.append(dict(stage=stage, checkpoint=summary["checkpoint_id"], teacher=r["teacher"],
                        architecture=summary["architecture"], method=summary["method"], primary=summary["primary"],
                        seed=summary["seed"], condition=c["id"], case_id=r["case_id"], n=r["n"], delta=r["delta"],
                        family=r["family"], failure=r["failure"], conflicts=r["conflicts"],
                        incomplete=r["incomplete"], first_divergence=r["first_divergence"],
                        rows_path=str(path), trace_path=str(path.with_name("first-failure.json"))))
    write_csv(ROOT / "checkpoint-condition.csv", detail)
    atomic_json(ROOT / "failure-index.json", failures)
    atomic_json(ROOT / "summary.json", dict(stages={stage:dict(executions=sum(d["executions"] for d in detail if d["stage"] == stage),
        successes=sum(d["successes"] for d in detail if d["stage"] == stage)) for stage in ("pilot", "confirmatory")},
        checkpoint_conditions=len(detail), distinct_checkpoints=len({d["checkpoint"] for d in detail}),
        failed_executions=len(failures), new_training_updates=0))
    plots(detail)
    report_results(detail, failures)


def plots(detail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(normalized="#0072B2", deepsets="#D55E00", transformer="#009E73")
    teachers = ("LUBY", "JOHANSSON", "COLOR_TRIALS", "JP", "SPECULATIVE_GM", "GREEDY")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True)
    for ax, teacher in zip(axes.flat, teachers):
        for arch, color in colors.items():
            rows = [d for d in detail if d["stage"] == "confirmatory" and d["primary"] and
                    d["teacher"] == teacher and d["architecture"] == arch and d["n"] == 64 and d["palette"] == "global"]
            for seed in sorted({d["seed"] for d in rows}):
                r = sorted([d for d in rows if d["seed"] == seed], key=lambda r:r["delta"])
                ax.plot([d["delta"] for d in r], [d["success_rate"] for d in r], color=color, alpha=.6,
                        marker="o", markersize=3, label=arch if seed == min(x["seed"] for x in rows) else None)
        ax.set_title(teacher, fontsize=10)
        ax.set_xlabel("Maximum degree (n=64, global palette)")
        ax.set_ylim(-.02, 1.02)
        ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Success, 1000 runs/checkpoint/condition")
    axes[1,0].set_ylabel("Success, 1000 runs/checkpoint/condition")
    axes[0,0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Frozen transfer: each line is one checkpoint; no averaged policy", fontsize=11)
    fig.tight_layout()
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"confirmatory-degree.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True)
    for ax, teacher in zip(axes.flat, teachers):
        for arch, color in colors.items():
            for seed in sorted({d["seed"] for d in detail if d["primary"] and d["teacher"] == teacher}):
                r = sorted([d for d in detail if d["stage"] == "pilot" and d["primary"] and d["teacher"] == teacher
                            and d["architecture"] == arch and d["seed"] == seed and d["n"] == 256 and d["palette"] == "global"], key=lambda r:r["delta"])
                ax.plot([d["delta"] for d in r], [d["success_rate"] for d in r], color=color, alpha=.5, marker=".")
        ax.set_xscale("log")
        ax.set_ylim(-.03, 1.03)
        ax.set_xlabel("Maximum degree, regular n=256")
        ax.set_title(teacher, fontsize=10)
        ax.grid(alpha=.2)
    fig.suptitle("Exploratory pilot: one run per checkpoint per point; not certification", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"pilot-degree.{ext}", dpi=160)
    plt.close(fig)


def report_results(detail, failures):
    main = [d for d in detail if d["stage"] == "confirmatory" and d["primary"]]
    conditions = sorted({d["condition"] for d in main})
    groups = sorted({(d["teacher"], d["architecture"]) for d in main})
    rows = []
    for teacher, arch in groups:
        values = []
        for c in conditions:
            ds = [d for d in main if d["teacher"] == teacher and d["architecture"] == arch and d["condition"] == c]
            values.append(f"{sum(d['successes'] == d['executions'] for d in ds)}/5; " +
                          f"{min(d['successes'] for d in ds)}–{max(d['successes'] for d in ds)}/1000")
        rows.append([teacher, arch, *values])
    body = "# Resultados de generalización sin reentrenamiento\n\n"
    body += "Resultados derivados de las filas crudas y sus hashes; cero actualizaciones nuevas. "
    body += "Cada celda muestra checkpoints sin fallos /5 y rango de éxitos por checkpoint. "
    body += "Nunca representa una política promedio. BE queda fuera del contraste de grado.\n\n"
    body += table(["Profesor", "Arquitectura", *conditions], rows) + "\n\n"
    body += "Los 1000 casos de cada condición corresponden a su mezcla de familias, no a cada familia por separado. "
    body += "Los intervalos y denominadores completos por checkpoint están en `checkpoint-condition.csv`. "
    body += "Son intervalos puntuales; 1000/1000 da CP bilateral95 [0.996318,1], sin garantía universal ni simultánea.\n\n"
    for stage in ("pilot", "confirmatory"):
        ds = [d for d in detail if d["stage"] == stage]
        body += f"**{stage}**: {sum(d['successes'] for d in ds):,}/{sum(d['executions'] for d in ds):,} ejecuciones correctas, "
        body += f"{sum(d['successes'] < d['executions'] for d in ds)} celdas checkpoint/condición con algún fallo. "
        body += "Este total es contabilidad, no estimación de una política combinada.\n\n"
    body += "## Grado y tamaño independientes\n\n"
    p_rows = []
    for teacher, arch in groups:
        pilot = [d for d in detail if d["stage"] == "pilot" and d["primary"] and d["teacher"] == teacher and d["architecture"] == arch]
        regular = [d for d in pilot if d["n"] == 256 and d["palette"] == "global"]
        passed = [d for d in regular if d["successes"] == d["executions"]]
        bad = [d for d in regular if d["successes"] < d["executions"]]
        large = [d for d in pilot if d["n"] == 1024 and d["delta"] == 10]
        p_rows.append([teacher, arch, max(d["delta"] for d in passed) if passed else "—",
                       min(d["delta"] for d in bad) if bad else "ninguno observado",
                       ",".join(str(d["successes"]) for d in sorted(large,key=lambda x:x["seed"])) + " (cada uno /1)"])
    body += table(["Profesor", "Arquitectura", "Mayor Δ con algún éxito piloto", "Primer Δ con algún fallo piloto", "n1024/Δ10, cinco checkpoints"], p_rows)
    body += "\n\nEl máximo con algún éxito no define un intervalo garantizado de grados: deben consultarse todas las semillas y celdas. "
    body += "El barrido n256 cambia Δ manteniendo n; el barrido Δ3/10 cambia n manteniendo grado y familia. "
    body += "La paleta global Δ+1 también cambia al cambiar grado: no se identifica causalmente grado frente a K.\n\n"
    body += "## Evidencia adicional y límites\n\n"
    body += "`native-21-consolidation.csv` reúne las 21 certificaciones nativas con métodos, cinco semillas, etiquetas, updates, parámetros, dominio, criterio y fuente. "
    body += "`failure-index.json` conserva todos los fallos; `first-failure.json` conserva testigos. "
    body += "Los costes medidos están en los `cost.json` de cada etapa. "
    body += "La comparación contrafactual y la interpretación causal se describen en el análisis de fallos y cierre.\n"
    Path("docs/DEGREE_GENERALIZATION_RESULTS.md").write_text(body)


if __name__ == "__main__":
    main()
