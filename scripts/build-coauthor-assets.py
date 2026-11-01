#!/usr/bin/env python3
"""Deterministic scientific figures and tables from integrated raw summaries."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/integrated-paper-20261031"
PAPER = ROOT / "paper/ucgnn"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source = OUT / "functional-summary.json"
    data = json.loads(source.read_text())
    assert data["executions"] == 30000
    rows = {(r["seed"], r["arm"], r["mode"]): r for r in data["rows"]}
    modes = ("baseline", "commit", "participation", "candidate", "kw_all")
    labels = ("Learned", "+ exact\nCOMMIT", "+ exact\nPARTICIPATE", "+ exact\nCANDIDATE", "+ all exact\nKW modules")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42,
                        "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=True)
    colors = ("#5b6f87", "#207c7b", "#ce8738", "#81729e", "#437747")
    for ax, seed in zip(axes, (105001, 105008, 105010)):
        values = [rows[seed,"cf",mode]["successes"] for mode in modes]
        bars = ax.bar(np.arange(5), values, color=colors, width=.7)
        ax.set_xticks(np.arange(5), labels, rotation=42, ha="right", fontsize=7.6)
        ax.set_ylim(0, 1160)
        ax.set_title(f"CF seed {seed}", fontsize=10)
        ax.axhline(1000, ls=":", color=".55", lw=.8, zorder=0)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+20, str(value), ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Successful original cases / 1,000")
    fig.subplots_adjust(left=.075, right=.99, bottom=.36, top=.88, wspace=.16)
    figure = PAPER / "figures/integration-functional-components.pdf"
    fig.savefig(figure, metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(OUT / "functional-components.png", dpi=170)
    plt.close(fig)
    outcomes = r"""All six unchanged baselines reproduce the original 6,000 terminal vectors,
statuses, step counts and invariant flags exactly. Across the 30,000
executions there are no technical errors and no newly failed cases under
an intervention. In CF seeds 105001 and 105008, exact COMMIT alone raises
successes from100 and101 to1,000 each, rescuing all900 and899 failures.
Changing only participation or candidates rescues none. All1,312 and1,322
missed valid writes reached by their unchanged policies occur at
$\kappa=0$; unlike the181 teacher-prefix false negatives, some have
participating neighbors (358 and406). These are visit counts, not
independent errors or new training replicas.

In seed105010, both arms complete204/1,000. Exact COMMIT or candidates
alone leave all796 failures unchanged. Exact participation alone rescues
all796 in each arm; the unchanged policy is silent on983 eligible reached
queries. All-exact KW modules yield1,000/1,000 for each of the six models,
while learned Linial modules remain fixed. Thus COMMIT-only rescues1,799
of the2,595 CF failures on this original battery, and participation-only
rescues the remaining796. Counts across arms share cases and first-stage
tensors; they are not independent intervention replications.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/integration-functional-components.pdf}
\caption{Component interventions on the three selected CF seeds, using all
original cases. Each replacement changes one KW component and recomputes
dependent learned outputs on the ensuing trajectory. All-exact replaces
three KW components. These retrospective diagnostic successes do not
establish a learned repair or general intervention success rate. Native
arms and case-level outputs are retained in Appendix~\ref{app:integration}.}
\label{fig:components}
\end{figure}
"""
    # Preserve normal word spacing; no manual copying of a generated table.
    for a,b in (("from100","from 100"),("and101","and 101"),("to1,000","to 1,000"),("all900","all 900"),
                ("and899","and 899"),("All1,312","All 1,312"),("and1,322","and 1,322"),("the181","the 181"),
                ("and406","and 406"),("seed105010","seed 105010"),("complete204","complete 204"),
                ("all796","all 796"),("on983","on 983"),("yield1,000","yield 1,000"),("rescues1,799","rescues 1,799"),
                ("the2,595","the 2,595"),("remaining796","remaining 796")):
        outcomes=outcomes.replace(a,b)
    (PAPER / "integration-functional-outcomes.tex").write_text(outcomes)
    table = []
    for seed in (105001,105008,105010):
        for arm in ("native","cf"):
            values = [str(rows[seed,arm,mode]["successes"]) for mode in modes]
            table.append(f"{seed} & {arm.upper() if arm=='cf' else 'Native'} & " + " & ".join(values) + r" \\")
    (PAPER / "tables/integration-components.tex").write_text("\n".join(table)+"\n")
    support = json.loads((ROOT / "results/paper-revision-20261030/continuation-audit.json").read_text())
    lines=[]
    for teacher, label in (("BE_BASE_D3","BE"),("SPECULATIVE_GM","GM")):
        subset=[r for r in support["rows"] if r["teacher"]==teacher]
        ranges=[]
        for arm in ("native","cf"):
            vals=[r["logical_test_occurrence_overlap"][arm]*100 for r in subset]
            ranges.append(f"{min(vals):.3f}--{max(vals):.3f}")
        lines.append(label+" & "+" & ".join(ranges)+r" \\")
    (PAPER / "tables/integration-logical-overlap.tex").write_text("\n".join(lines)+"\n")
    assets = [figure,PAPER/"integration-functional-outcomes.tex",PAPER/"tables/integration-components.tex",PAPER/"tables/integration-logical-overlap.tex"]
    (OUT / "assets.json").write_text(json.dumps(dict(source_sha256=sha(__file__), inputs={str(source.relative_to(ROOT)):sha(source)},
        outputs={str(p.relative_to(ROOT)):sha(p) for p in assets}), indent=2)+"\n")
    print(json.dumps(dict(assets=len(assets),observations=data["executions"])))


if __name__ == "__main__":
    main()
