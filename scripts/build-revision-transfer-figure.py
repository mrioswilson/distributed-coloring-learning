"""Build compact primary-transfer figure from the preserved closure cell table."""
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper-adversarial-20261030-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results/final-closure/generalization-cells.csv"
PREFIX = REPO / "results/paper-revision-20261030/figure4"
PAPER_FIGURE = REPO / "paper/ucgnn/figures/closure-generalization-compact.pdf"

with SOURCE.open(newline="") as stream:
    rows = [r for r in csv.DictReader(stream)
            if r["stage"] == "confirmatory" and r["primary"] == "True"]
assert len(rows) == 450
assert all(int(r["executions"]) == 1000 for r in rows)
failed = [r for r in rows if int(r["failures"]) > 0]
assert len(failed) == 2
failed.sort(key=lambda r: int(r["failures"]))
assert [(r["teacher"], int(r["failures"])) for r in failed] == [
    ("JOHANSSON", 28), ("SPECULATIVE_GM", 181)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(8.9, 2.9))
fig.subplots_adjust(left=.405, right=.985, bottom=.38, top=.85)
positions = [1, 0]
counts = [int(r["failures"]) for r in failed]
ax.barh(positions, counts, height=.39, color=["#B56A28", "#A74247"], zorder=3)
labels = [
    "Johansson / Transformer / s83101\nGlobal palette · n=64, Δ=10 · cutoff",
    "GM / Transformer / s94102\nLocal palettes · n=64, Δ=4 · collision",
]
ax.set_yticks(positions, labels=labels)
ax.tick_params(axis="y", length=0, pad=10, labelsize=9)
for y, count in zip(positions, counts):
    ax.text(count + 4, y, f"{count:,} ({count / 10:.1f}%)", va="center", fontsize=10)
ax.set_xlim(0, 240)
ax.set_ylim(-.55, 1.55)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xlabel("Failures per 1,000 runs", fontsize=10, labelpad=5)
ax.grid(axis="x", color="#E1E4E8", linewidth=.65, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#AEB4BC")
ax.tick_params(axis="x", length=3, color="#AEB4BC", labelsize=9)
fig.text(.025, .95, "Only two primary cells have observed failures", fontsize=12,
         weight="bold", va="top", color="#24282E")
fig.text(.025, .045,
         "448 of 450 primary checkpoint–condition cells had zero failures (1,000 runs each).\n"
         "Conditions differ between the two cells; these rates do not rank the teachers.",
         fontsize=8.5, va="bottom", color="#4D535B", linespacing=1.45)
for extension in ("png", "pdf"):
    fig.savefig(PREFIX.with_suffix("." + extension), dpi=200, facecolor="white")
plt.close(fig)
shutil.copyfile(PREFIX.with_suffix(".pdf"), PAPER_FIGURE)

record = {
    "status": "compact editorial replacement of Figure 4; experimental results unchanged",
    "source": str(SOURCE.relative_to(REPO)),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "filter": {"stage": "confirmatory", "primary": "True"},
    "cells": len(rows), "zero_failure_cells": len(rows) - len(failed),
    "executions": sum(int(r["executions"]) for r in rows),
    "failures": sum(counts), "nonzero_rows": failed,
    "outputs": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (PREFIX.with_suffix(".png"), PREFIX.with_suffix(".pdf"), PAPER_FIGURE)},
}
PREFIX.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: record[k] for k in
                  ("cells", "zero_failure_cells", "executions", "failures", "outputs")}))
