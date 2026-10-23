"""Run, compare, archive and package the authorized E1 follow-up, without Git writes."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import traceback

from dcl.multiteacher_e1.experiment import run
from dcl.study.storage import atomic_json, file_hash, now
from analyze_multiteacher_e1 import analyze
from rooted_multiteacher_cache import resident_roots


def write_report(report, target):
    names = {"LUBY": "Luby", "JOHANSSON": "Johansson", "JP": "Jones–Plassmann",
             "COLOR_TRIALS": "ColorTrials repetido", "BE_BASE_D3": "BE, caso base Δ≤3",
             "SPECULATIVE_GM": "Especulativo GM", "GREEDY": "Greedy secuencial"}
    rows = []
    for r in report["table"]:
        rows.append(f"| {names[r['teacher']]} | {r['old_successes']}/{r['trials']} | "
                    f"{r['new_successes']}/{r['trials']} | {r['change_percentage_points']:+.2f} | "
                    f"{100*r['e1_regular_n10240']:.2f}% | {100*r['e1_irregular_n10240']:.2f}% |")
    # Official plan has N10240. Tiny end-to-end tests use the general JSON/CSV.
    text = """# Siete maestros con E1: seguimiento experimental

Se repite la comparación anterior usando el núcleo UniversalColorGNN NORMALIZED
y entrenamiento contrafactual. Luby/Johansson/greedy:16035 parámetros; los otros
cuatro incluyen adaptadores nativos explícitos y suman19396. Los fuentes
canónicos E1 siguen intactos. Protocolo y límites de cada adaptación en
[MULTITEACHER_E1.md](../../../docs/MULTITEACHER_E1.md).

| Maestro | GNN anterior: éxitos | E1 / adaptada: éxitos | Cambio (puntos %) | E1 REG N10240 | E1 IRREG N10240 |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + f"""

Se completaron **{report['models']} modelos**, **{report['cases']} casos comunes por
actor** y **{report['evaluations']} evaluaciones**. Las
{report['exact_colorings_match_baseline']} coloraciones de referencia exactas
coinciden con las de la comparación original. Todos los checkpoints corresponden
al último paso prefijado y quedaron congelados antes de evaluar.

Cada modelo recibió800 grafos N40/Δ3, tres tapes y17760 actualizaciones de1680
decisiones:29836800 decisiones muestreadas. Cinco inicializaciones comparten
ese pool; sus resultados no son cinco conjuntos de grafos independientes.
El test reutiliza casos ya publicados: seguimiento exploratorio, no test nuevo
ciego. En N10240 hay tres grafos por familia y dos tapes; las medias incluyen
las cinco inicializaciones sobre esos mismos casos.

![Comparación por tamaño](architecture-comparison.png)

![Curvas de aprendizaje](learning-curves.png)

![Aprendizaje por fases](learning-by-phase.png)

La tabla agrupa todos los tamaños y controles; para estudiar transferencia se
deben consultar las familias/tamaños y semillas por separado. Una precisión
local alta no garantiza que la coloración completa sea correcta.
Se conserva cada fallo; no se elige la mejor réplica.

La comparación antes/después cambia arquitectura, factorización de la pérdida
y algunas representaciones nativas simultáneamente; no aísla causalmente solo
la arquitectura. En BE se aprende el punto Linial y se calcula localmente su
valor polinomial, una restricción adicional respecto de la cabeza antigua de
169 pares. ColorTrials transmite disponibilidad aprendida y preferencias en
el segundo intercambio. Las fases y calendarios se suministran; los costes
secuenciales/paralelos no son rondas LOCAL equivalentes. Las predicciones
aprendidas se aplican sin llamadas al maestro ni filtros de validez.

Evidencia experimental finita; no prueba corrección universal. Revisión
científica independiente pendiente. Esta entrega no modifica los resultados
antiguos ni los cinco E1/Luby publicados aparte.

Artefactos: [tabla antes/después](architecture-comparison.csv),
[cambios sobre casos emparejados](paired-changes.csv),
[resultados por semilla](success-by-seed.csv),
[componentes](component-learning.csv), [fases](learning-by-phase.csv),
[congelación de pesos](frozen.json) y [manifiesto](manifest.json).
Los datos completos/checkpoints quedan en `data/multiteacher-e1-20261022`;
los pools fuente inmutables permanecen en `data/multiteacher-20261019`.
"""
    target.write_text(text)


def package(root, archive, output, report):
    root, archive, output = Path(root), Path(archive), Path(output)
    controls = {"run.log", "supervisor-status.json", "artifact-manifest.json"}
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in controls]
    manifest = {"created_utc": now(), "files": {str(p.relative_to(root)): file_hash(p) for p in files}}
    atomic_json(root / "artifact-manifest.json", manifest)
    for p in files + [root / "artifact-manifest.json"]:
        target = archive / p.relative_to(root)
        if target.exists() and file_hash(target) != file_hash(p):
            raise RuntimeError("Archive contains a different artifact; overwrite refused")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
        if file_hash(target) != file_hash(p):
            raise RuntimeError("Archive copy differs")
    output.mkdir(parents=True, exist_ok=True)
    selected = ["plan.json", "frozen.json", "provenance.json", "source-pools.json", "complete.json", "summary.json",
                "success-by-seed.csv", "learning-curves.csv", "learning-curves.png", "learning-curves.pdf",
                "learning-by-phase.csv", "learning-by-phase.png", "learning-by-phase.pdf",
                "success-by-size.png", "success-by-size.pdf", "architecture-comparison.csv",
                "architecture-comparison.json", "architecture-comparison.png", "architecture-comparison.pdf",
                "paired-changes.csv", "component-learning.csv", "artifact-manifest.json"]
    selected += [str(p.relative_to(root)) for p in sorted((root / "runs").rglob("history.json"))]
    entries = []
    for relative in selected:
        source, target = root / relative, output / relative
        raw = source.read_bytes()
        data = raw
        if source.suffix == ".csv":
            parsed = list(csv.reader(io.StringIO(raw.decode())))
            stream = io.StringIO(newline=""); csv.writer(stream, lineterminator="\n").writerows(parsed)
            data = stream.getvalue().encode()
            if list(csv.reader(io.StringIO(data.decode()))) != parsed:
                raise RuntimeError("CSV newline normalization changed values")
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError("Compact output already differs; overwrite refused")
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        entries.append({"path": relative, "sha256": file_hash(target), "source_sha256": file_hash(source), "bytes": len(data)})
    if any(r["teacher"] == "LUBY" and "e1_regular_n10240" in r for r in report["table"]):
        write_report(report, output / "README.md")
        entries.append({"path": "README.md", "sha256": file_hash(output / "README.md"), "bytes": (output / "README.md").stat().st_size})
    atomic_json(output / "manifest.json", {"created_utc": now(), "artifacts": entries,
                "runtime_archive": str(archive), "source_archive_files": len(files),
                "runtime_archive_manifest_sha256": file_hash(archive / "artifact-manifest.json")})
    return {"archive_files": len(files), "compact_artifacts": len(entries), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/multiteacher_e1/comparison.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "stage": "train/evaluate", "started_utc": now()}
    path = args.root / "supervisor-status.json"
    atomic_json(path, status)
    try:
        config = json.loads(args.plan.read_text())
        run(config, args.root, args.source, args.trainers, "all", resident_roots)
        status.update(stage="analysis"); atomic_json(path, status)
        report = analyze(args.root, args.source)
        status.update(stage="archive"); atomic_json(path, status)
        packed = package(args.root, args.archive, args.output, report)
        status.update(status="complete", stage="complete", finished_utc=now(), **packed)
        atomic_json(path, status)
        print(json.dumps(status), flush=True)
    except BaseException:
        status.update(status="failed", failed_utc=now(), traceback=traceback.format_exc())
        atomic_json(path, status)
        raise


if __name__ == "__main__":
    main()
