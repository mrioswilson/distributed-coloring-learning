#!/usr/bin/env python3
"""Build the seven-teacher E1 report from the frozen CSVs and vector figures.

No training, evaluation, plotting, or raw experimental artifact is regenerated.
Requires Python's standard library, pdflatex (TeX Live), and pdfinfo (Poppler).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
DELIVERY = REPO / "results/multiteacher-e1/2026-10-22"
NAMES = {
    "LUBY": "Luby",
    "JOHANSSON": "Johansson",
    "JP": "Jones--Plassmann",
    "COLOR_TRIALS": "ColorTrials repetido",
    "BE_BASE_D3": r"BE (base $\Delta\leq3$)",
    "SPECULATIVE_GM": "Especulativo GM",
    "GREEDY": "Greedy secuencial",
}
FROZEN_INPUTS = (
    "architecture-comparison.csv", "learning-milestones.csv",
    "summary.json", "verification.json", "audit.json", "frozen.json", "plan.json",
    "learning-curves.csv", "learning-by-phase.csv",
    "success-by-size.pdf", "architecture-comparison.pdf",
    "learning-curves.pdf", "learning-by-phase.pdf",
)
SOURCE_INPUTS = (
    "E1.md", "docs/MULTITEACHER_E1.md", "docs/MULTITEACHER_E1_METRICAS.md",
    "src/dcl/universal/model.py", "src/dcl/multiteacher_e1/model.py",
    "configs/multiteacher_e1/comparison.json",
    "experiments/build_multiteacher_e1_pdf.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {"path": path.relative_to(REPO).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha(path)}


def read_csv(name: str) -> list[dict]:
    with (DELIVERY / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def integer(value: int) -> str:
    return f"{value:,}".replace(",", r"\,")


def decimal(value: float, digits: int = 2, signed: bool = False) -> str:
    return format(value, f"{'+' if signed else ''}.{digits}f").replace(".", "{,}")


def macro(name: str, value: str) -> str:
    return "\\newcommand{\\" + name + "}{" + value + "}\n"


def verify_frozen_inputs() -> None:
    manifest = json.loads((DELIVERY / "manifest.json").read_text())
    indexed = {row["path"]: row for row in manifest["artifacts"]}
    for name in FROZEN_INPUTS:
        if sha(DELIVERY / name) != indexed[name]["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")


def generate_tables() -> str:
    comparison = read_csv("architecture-comparison.csv")
    milestones = read_csv("learning-milestones.csv")
    summary = json.loads((DELIVERY / "summary.json").read_text())
    verification = json.loads((DELIVERY / "verification.json").read_text())
    assert [row["teacher"] for row in comparison] == list(NAMES)
    assert len(comparison) == len(milestones) == 7
    assert verification["models"] == 35
    assert summary["cases"] == 396
    assert all(int(row["trials"]) == 1980 for row in comparison)
    learned = [row for row in summary["per_seed"] if row["seed"] is not None]
    totals = {name: sum(row[name] for row in learned) for name in
              ("trials", "successes", "collision_failures", "palette_failures", "incomplete_failures")}
    assert totals["trials"] == 13860
    assert totals["successes"] == sum(int(row["new_successes"]) for row in comparison)
    assert totals["collision_failures"] == totals["palette_failures"] == 0
    assert totals["incomplete_failures"] == totals["trials"] - totals["successes"] == 378
    assert sum(row["incomplete_failures"] for row in learned if row["teacher"] == "JP") == 349
    assert sum(row["incomplete_failures"] for row in learned if row["teacher"] == "SPECULATIVE_GM") == 29

    text = "% Generated from the frozen CSVs; do not edit manually.\n"
    text += macro("LearnedRuns", integer(totals["trials"]))
    text += macro("LearnedSuccesses", integer(totals["successes"]))
    text += macro("TotalSuccessPercent", decimal(100 * totals["successes"] / totals["trials"]))
    text += macro("LearnedFailures", integer(totals["incomplete_failures"]))
    result_rows = []
    size_rows = []
    for row in comparison:
        teacher = row["teacher"]
        result_rows.append(" & ".join((NAMES[teacher], integer(int(row["old_successes"])),
                           integer(int(row["new_successes"])),
                           decimal(100 * int(row["new_successes"]) / int(row["trials"])),
                           decimal(float(row["change_percentage_points"]), signed=True))) + r" \\")
        counts = []
        for family, prefix in (("REGULAR", "regular"), ("IRREGULAR", "irregular")):
            group = [item for item in learned if item["teacher"] == teacher
                     and item["family"] == family and item["n"] == 10240]
            denominator = sum(item["trials"] for item in group)
            assert denominator == 30
            new = sum(item["successes"] for item in group)
            old = round(float(row[f"old_{prefix}_n10240"]) * denominator)
            assert abs(new / denominator - float(row[f"e1_{prefix}_n10240"])) < 1e-12
            counts.extend((str(old), str(new)))
        size_rows.append(" & ".join((NAMES[teacher], *counts)) + r" \\")
    text += macro("ResultRows", "\n" + "\n".join(result_rows) + "\n")
    text += macro("LargeSizeRows", "\n" + "\n".join(size_rows) + "\n")
    milestone_rows = []
    for row in milestones:
        old = row["old_first_sustained_20_of_20_checkpoint"]
        new = row["e1_first_sustained_20_of_20_checkpoint"]
        milestone_rows.append(" & ".join((NAMES[row["teacher"]],
            integer(int(old)) if old else "No alcanzado",
            integer(int(new)) if new else "No alcanzado",
            decimal(100 * float(row["e1_final_local_action_accuracy"]), digits=4))) + r" \\")
    text += macro("MilestoneRows", "\n" + "\n".join(milestone_rows) + "\n")
    return text


def check() -> None:
    verify_frozen_inputs()
    manifest = json.loads((DELIVERY / "pdf-manifest.json").read_text())
    for row in manifest["inputs"] + manifest["artifacts"]:
        path = REPO / row["path"]
        if not path.is_file() or sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Document hash mismatch: {row['path']}")
    if (DELIVERY / "report-tables.tex").read_text() != generate_tables():
        raise ValueError("Report tables disagree with the frozen CSVs")
    print(json.dumps({"status": "verified", "pages": manifest["pages"],
                      "inputs": len(manifest["inputs"]), "artifacts": len(manifest["artifacts"])}))


def build(build_dir: Path) -> None:
    verify_frozen_inputs()
    (DELIVERY / "report-tables.tex").write_text(generate_tables())
    build_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = DELIVERY / "pdf-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    documented_utc = previous.get("documented_utc") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    epoch = str(int(datetime.fromisoformat(documented_utc).timestamp()))
    env = dict(os.environ, SOURCE_DATE_EPOCH=epoch, FORCE_SOURCE_DATE="1")
    for _ in range(2):
        process = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                  "-file-line-error", f"-output-directory={build_dir}", "INFORME.tex"],
                                 cwd=DELIVERY, env=env, text=True, encoding="utf-8",
                                 errors="replace", capture_output=True)
        if process.returncode:
            raise RuntimeError(process.stdout[-7000:] + process.stderr[-1000:])
    log = (build_dir / "INFORME.log").read_text(errors="replace")
    if "Overfull" in log or "undefined references" in log:
        raise ValueError(f"Layout/reference warnings; inspect {build_dir / 'INFORME.log'}")
    temporary_pdf = build_dir / "INFORME.pdf"
    info = subprocess.check_output(["pdfinfo", str(temporary_pdf)], text=True)
    pages = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")))
    if pages != 6:
        raise ValueError(f"Expected 6 pages, got {pages}; inspect {temporary_pdf}")
    shutil.copyfile(temporary_pdf, DELIVERY / "INFORME.pdf")
    inputs = [record(DELIVERY / name) for name in FROZEN_INPUTS]
    inputs.extend(record(REPO / name) for name in SOURCE_INPUTS)
    inputs.extend(record(DELIVERY / name) for name in ("INFORME.tex", "report-tables.tex"))
    manifest = {"kind": "additive PDF report, no numerical experiment changes",
                "protocol": "e1-native-v1", "experiment_date": "2026-10-22", "pages": pages,
                "documented_utc": documented_utc,
                "metadata_date": "first document build; preserved for reproducible metadata",
                "build": "python3 experiments/build_multiteacher_e1_pdf.py",
                "check": "python3 experiments/build_multiteacher_e1_pdf.py --check",
                "inputs": inputs, "artifacts": [record(DELIVERY / "INFORME.pdf")]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    check()
    print(f"PDF: {DELIVERY / 'INFORME.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the existing report without writing")
    parser.add_argument("--build-dir", type=Path, help="keep TeX logs/auxiliary files in this directory")
    args = parser.parse_args()
    if args.check:
        check()
    elif args.build_dir:
        build(args.build_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="multiteacher-e1-pdf-") as temporary:
            build(Path(temporary))


if __name__ == "__main__":
    main()
