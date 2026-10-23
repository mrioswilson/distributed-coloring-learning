"""Build the final report only from completed, authenticated E1 measurements."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "data/e1-retrained-20261021"
OUT = Path(__file__).resolve().parent


def csv_rows(name):
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_figure(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=170)
    plt.close(fig)


def build():
    summary = json.loads((OUT / "summary.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    frozen = json.loads((ROOT / "frozen.json").read_text())
    assert summary["cohort_origin"] == "newly-trained-replicas" and audit["complete"]
    assert len(frozen["checkpoints"]) == 5
    tests = csv_rows("distribution-tests.csv")
    graphs = csv_rows("by-graph.csv")
    local = csv_rows("local-summary.csv")
    primary = [r for r in tests if r["role"] == "random" and r["replica"] != ""]
    baselines = [r for r in tests if r["role"] == "random" and r["replica"] == ""]
    sizes = [n for n, _ in summary["plan"]["graphs_per_family"]]
    train = {}
    for item in frozen["checkpoints"]:
        path = ROOT / item["checkpoint"]
        assert digest_file(path) == item["sha256"]
        rows = json.loads((path.parent / "history.json").read_text())
        assert len(rows) == 120 and rows[-1]["epoch"] == 120
        train[item["seed"]] = rows

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for s, rows in train.items():
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["train"]["loss"] for r in rows], color=colors[s], label=f"s{s}")
        axes[1].plot(epochs, [100 * r["validation_local"]["action_correct_teacher_rate"] for r in rows], color=colors[s])
    axes[0].set_ylabel("Pérdida de entrenamiento")
    axes[0].set_yscale("log")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].set_ylabel("Acuerdo de acción en validación (%)")
    axes[1].set_ylim(70, 100.2)
    for ax in axes:
        ax.set_xlabel("Época")
        ax.axvline(10, color="gray", ls=":", lw=1)
        ax.grid(alpha=.2)
    save_figure(fig, "training")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    cells = [(family, n) for family in ("REGULAR", "IRREGULAR") for n in sizes]
    matrices = []
    for rep in ("labelled", "color_permutation"):
        matrix = np.zeros((5, len(cells)))
        significant = np.zeros_like(matrix, bool)
        for s in range(5):
            for j, (family, n) in enumerate(cells):
                row = next(r for r in primary if int(r["replica"]) == s and r["representation"] == rep
                           and r["family"] == family and int(r["n"]) == n)
                matrix[s, j] = float(row["mmd2"])
                significant[s, j] = row["different_after_holm"] == "True"
        matrices.append((matrix, significant))
    limit = max(np.max(np.abs(m)) for m, _ in matrices) or .001
    for ax, title, (m, significant) in zip(axes, ("Con nombres de colores", "Módulo permutación de colores"), matrices):
        im = ax.imshow(m, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        for s, j in np.argwhere(significant):
            ax.text(j, s, "*", ha="center", va="center", fontsize=14)
        ax.set_xticks(range(len(cells)), [f"{'R' if f == 'REGULAR' else 'I'}{n}" for f, n in cells], rotation=60)
        ax.set_yticks(range(5), [f"s{s}" for s in range(5)])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="MMD² insesgada (*: diferencia tras Holm)", shrink=.85)
    save_figure(fig, "distribution-tests")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    keys = ("participation_agreement", "candidate_agreement", "action_agreement", "shared_commit_agreement")
    matrices = []
    for source in ("teacher_states", "gnn_states"):
        matrix = np.zeros((5, 4))
        for s in range(5):
            row = next(r for r in local if int(r["replica"]) == s and r["source"] == source and r["k"] == "-1")
            matrix[s] = [100 * (1 - float(row[k])) for k in keys]
        matrices.append(matrix)
    top = max(float(m.max()) for m in matrices) or .01
    for ax, title, m in zip(axes, ("Estados de Luby", "Estados propios de E1"), matrices):
        im = ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=top, aspect="auto")
        for s in range(5):
            for j in range(4):
                ax.text(j, s, f"{m[s,j]:.3f}", ha="center", va="center", fontsize=9)
        ax.set_yticks(range(5), [f"s{s}" for s in range(5)])
        ax.set_xticks(range(4), ["Participar", "Candidato", "Acción", "COMMIT\ncomún"])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="Desacuerdo local (%)", shrink=.85)
    save_figure(fig, "local-decisions")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for s in range(5):
        y, q = [], []
        for n in sizes:
            rows = [r for r in graphs if r["role"] == "random" and r["replica"] == str(s)
                    and int(r["n"]) == n and r["representation"] == "labelled"]
            y.append(100 * np.mean([float(r["success_rate"]) for r in rows]))
            q.append(100 * np.mean([float(r["coupled_mismatch_rate"]) for r in rows]))
        axes[0].plot(sizes, y, "o-", color=colors[s], label=f"s{s}")
        axes[1].plot(sizes, q, "o-", color=colors[s])
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes, labels=sizes)
        ax.set_xlabel("Vértices")
        ax.set_ylim(-2, 102)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Coloraciones correctas (%)")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].set_ylabel("Final distinto con la misma tape (%)")
    axes[1].set_title("Desacuerdo acoplado; no es distancia TV", fontsize=10)
    save_figure(fig, "rollouts")

    lines = ["# Cinco E1 nuevas frente a Luby", "",
             "Se entrenaron cinco réplicas nuevas con la configuración E1 original de Martín: "
             "UniversalColorGNN NORMALIZED, COUNTERFACTUAL, 16.035 parámetros, semillas 0–4, "
             "800 grafos REGULAR N40/Δ3/K4 y tres tapes por grafo. Las cinco completaron 120 épocas "
             "con Adam, lr 0,001, batch 128 y warmup 10. Entrenador y modelo originales sin modificar, "
             "ejecutados en CPU. Sus hashes son propios; no se presentan como los checkpoints históricos.", "",
             "Los cinco checkpoints de época 120 quedaron congelados antes de abrir el test. "
             "La comparación usa 80 grafos nuevos REGULAR/IRREGULAR de 8/12/40/160/640 nodos, "
             "más tres controles, y 256 tapes por muestra independiente A/B.", "",
             "## Distribuciones finales", "",
             "| Semilla | Éxitos /20480 casos aleatorios | Diferencias con etiquetas /10 | "
             "Diferencias módulo colores /10 | TV≤0,05 etiquetas /80 | TV≤0,05 particiones /80 |", "|---|---:|---:|---:|---:|---:|"]
    numerical = []
    for s in range(5):
        rows = [r for r in graphs if r["role"] == "random" and r["replica"] == str(s)]
        labelled = [r for r in rows if r["representation"] == "labelled"]
        successes = sum(round(float(r["success_rate"]) * int(r["samples"])) for r in labelled)
        differences = [sum(r["different_after_holm"] == "True" for r in primary
                           if r["replica"] == str(s) and r["representation"] == rep)
                       for rep in ("labelled", "color_permutation")]
        certificates = {rep: sum(r["tv_within_tolerance_certified"] == "True" for r in rows if r["representation"] == rep)
                        for rep in ("labelled", "color_permutation")}
        numerical.append({"seed": s, "successes": successes, "differences": differences, "certificates": certificates})
        lines.append(f"|{s}|{successes}|{differences[0]}|{differences[1]}|{certificates['labelled']}|{certificates['color_permutation']}|")
    lines.extend(["", f"MMD con 3.999 permutaciones dentro de cada grafo y ajuste de Holm sobre 100 tests "
                  f"principales: {summary['different_after_holm']} diferencias detectadas. "
                  f"En los 20 controles independientes Luby/Luby hubo {sum(float(r['p_value']) < .05 for r in baselines)} "
                  "p-valores menores que0,05 sin ajuste. Controles estructurales separados en los CSV.", "",
                  "No rechazar igualdad no prueba cercanía. La TV empírica es descriptiva. "
                  "Las 830 cotas binomiales simultáneas del desacuerdo final bajo tapes compartidas son cotas "
                  "superiores de TV; un desacuerdo alto no demuestra separación distribucional. "
                  "Los certificados se limitan a cada grafo y modelo fijos evaluados.", "",
                  "![Tests distribucionales](distribution-tests.png)", "",
                  "![Éxito y acoplamiento](rollouts.png)", "", "## Decisiones locales", "",
                  "| Semilla | Estado | Participación | Candidato | Acción WAIT/COMMIT(color) | COMMIT común | Inseguros ejecutados |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for s in range(5):
        for source in ("teacher_states", "gnn_states"):
            row = next(r for r in local if int(r["replica"]) == s and r["source"] == source and r["k"] == "-1")
            vals = "|".join(f"{100*float(row[k]):.4f}%" for k in keys)
            lines.append(f"|{s}|{'Luby' if source == 'teacher_states' else 'E1'}|{vals}|{row['unsafe_commits']}|")
    lines.extend(["", "Todos los acuerdos usan el mismo estado y los mismos sorteos por nodo. "
                  "COMMIT común se consulta con las propuestas de Luby y su denominador son los participantes "
                  "de Luby; los otros acuerdos usan todos los nodos activos. Los conteos por nodo/ronda son "
                  "correlacionados y descriptivos. Los diagnósticos exactos no filtran las acciones aprendidas.", "",
                  "![Decisiones locales](local-decisions.png)", "", "## Entrenamiento y auditoría", "",
                  "![Curvas de entrenamiento](training.png)", "",
                  f"Auditoría completa: {audit['colorings_checked']} coloraciones comprobadas, "
                  f"{audit['teacher_scalar_replays']} ejecuciones Luby escalares, "
                  f"{audit['e1_scalar_replays']} ejecuciones E1 escalares originales y "
                  f"{audit['local_examples_replayed']} ejemplos locales reproducidos. "
                  "Fuentes originales y los cinco hashes de pesos verificados. Pruebas dirigidas: 88 pasan. "
                  "Revisión científica independiente pendiente.", "",
                  "Datos originales, semillas, fuentes y modelos en data/e1-retrained-20261021. "
                  "frozen.json fija la cohorte; comparison/manifest.json fija los datos del test. "
                  "Esta evidencia experimental finita no prueba corrección universal ni equivalencia para todos los grafos.", ""])
    (OUT / "INTERPRETACION.md").write_text("\n".join(lines))
    (OUT / "key-results.json").write_text(json.dumps({"replicas": numerical, "summary": summary, "audit": audit}, indent=2) + "\n")
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name != "artifact-manifest.json"]
    (OUT / "artifact-manifest.json").write_text(json.dumps({"files": {p.name: digest_file(p) for p in sorted(paths)}}, indent=2) + "\n")


if __name__ == "__main__":
    build()
