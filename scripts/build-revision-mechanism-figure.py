#!/usr/bin/env python3
"""Draw two selected frozen witness fragments; no simulation or inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper-revision-20261030"
FIGURE = ROOT / "paper/ucgnn/figures/revision-failure-mechanisms.pdf"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evidence():
    source = OUT / "component-rescues.json"
    rows = json.loads(source.read_text())["rows"]
    inputs = {str(source.relative_to(ROOT)): sha(source)}
    panels = []
    for teacher in ("JOHANSSON", "SPECULATIVE_GM"):
        selected = {r["mode"]: r for r in rows if r["checkpoint"].startswith(teacher + "-")}
        baseline, changed, exact = (selected[k] for k in ("baseline", "candidate_once", "exact_commit_once"))
        original = ROOT / "results/final-closure/failure-replays" / baseline["checkpoint"] / (baseline["condition"]["id"] + ".json")
        assert sha(original) == baseline["original_witness_sha256"]
        inputs[str(original.relative_to(ROOT))] = sha(original)
        witness = json.loads(original.read_text())
        intervention = changed["intervention"]
        step, root = intervention["step"], intervention["vertex"]
        before = next(f for f in witness["frames"] if f["step"] == step)
        assert before["teacher_context_valid"] and witness["replay_matches_original"]
        assert baseline["baseline_terminal_matches"] and exact["baseline_terminal_matches"]
        assert changed["preintervention_matches"] and intervention["only_one_candidate_changed_with_identical_first_stage"]
        assert all(r["preserved_batch_size"] == 16 for r in selected.values())
        assert exact["intervention"]["oracle_confirm"] == baseline["intervention"]["baseline_commit"]
        assert baseline["frames"] == exact["frames"] and not baseline["success"] and changed["success"]
        after_base = next(f for f in baseline["frames"] if f["step"] == step + 1)
        after_changed = next(f for f in changed["frames"] if f["step"] == step + 1)
        if teacher == "JOHANSSON":
            vertices = [v for v, color in enumerate(before["colors"]) if color < 0]
            assert vertices == before["roots"] and len(vertices) == 2 and root == vertices[0]
            assert sorted(vertices) in [sorted(edge) for edge in witness["case_edges"]]
            assert intervention["baseline_candidates"] == [2, 2]
            assert intervention["result_candidates"] == [0, 2]
            assert intervention["result_commit"] == [True, True]
            assert intervention["available"] == [0, 2]
            for vertex in vertices:
                neighbors = [v if u == vertex else u for u, v in witness["case_edges"] if vertex in (u, v)]
                assert [c for c in range(11) if c not in [before["colors"][v] for v in neighbors]] == [0, 2]
        else:
            neighbors = [v if u == root else u for u, v in witness["case_edges"] if root in (u, v)]
            assert len(neighbors) == 1 and before["roots"] == [root]
            vertices = [root, neighbors[0]]
            assert intervention["baseline_candidates"] == [0] and intervention["result_candidates"] == [1]
            assert intervention["result_commit"] == [True] and intervention["available"] == [1]
            assert before["colors"][root] == before["colors"][neighbors[0]] == 0
        panels.append(dict(
            teacher=teacher, checkpoint=baseline["checkpoint"], case_id=baseline["case_id"],
            condition=baseline["condition"]["id"], step_before=step, step_after=step + 1,
            vertices=vertices, before_colors=[before["colors"][v] for v in vertices],
            baseline_after_colors=[after_base["colors"][v] for v in vertices],
            candidate_after_colors=[after_changed["colors"][v] for v in vertices],
            original_candidates=intervention["baseline_candidates"],
            changed_candidates=intervention["result_candidates"],
            original_commit=intervention["baseline_commit"], changed_commit=intervention["result_commit"],
            available=intervention["available"], changed_vertex=root,
            original_terminal=dict(steps=baseline["steps"], failure=baseline["failure"],
                                   bad_edges=baseline["terminal_bad_edges"]),
            changed_terminal=dict(steps=changed["steps"], success=changed["success"],
                                  bad_edges=changed["terminal_bad_edges"]),
            exact_commit_control_identical=True))
    return dict(scope="Two post-test selected 64-node witness fragments, not population rates or a general repair policy",
                inputs=inputs, panels=panels)


def draw(record):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.6))
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.055, hspace=.18)
    muted, ink, red, blue = "#58616B", "#20262D", "#A2363A", "#146B7A"
    for ax, panel in zip(axes, record["panels"]):
        ax.set(xlim=(0, 10), ylim=(0, 2.65), aspect="equal")
        ax.axis("off")
        johansson = panel["teacher"] == "JOHANSSON"
        heading = "A  Johansson: a correct veto leaves both vertices pending" if johansson else "B  GM: a correct veto skips a required repair"
        context = "Global palette; n=64, max degree 10" if johansson else "Local lists; n=64, max degree 4"
        ax.text(.08, 2.48, heading, fontsize=13, weight="bold", color=ink)
        ax.text(.08, 2.18, context, fontsize=10.5, color=muted)
        root = panel["changed_vertex"]
        old = panel["original_candidates"][0]
        new = panel["changed_candidates"][0]
        for center, colors, changed in ((2.45, panel["baseline_after_colors"], False),
                                        (7.5, panel["candidate_after_colors"], True)):
            label = f"Change only c{root}: {old} → {new}" if changed else "Original proposals"
            ax.text(center, 1.83, label, ha="center", fontsize=12, color=blue if changed else ink)
            xs = [center - .65, center + .65]
            conflict = not johansson and not changed
            ax.plot(xs, [1.12, 1.12], color=red if conflict else muted,
                    ls="--" if conflict else "-", lw=1.6, zorder=1)
            for x, vertex, color in zip(xs, panel["vertices"], colors):
                ax.add_patch(Circle((x, 1.12), .255, facecolor="white", edgecolor=ink, lw=1.25, zorder=2))
                ax.text(x, 1.12, "–" if color < 0 else str(color), ha="center", va="center", fontsize=14, zorder=3)
                ax.text(x, .70, f"v{vertex}", ha="center", fontsize=10.5, color=muted)
            if johansson:
                detail = "Proposals (0, 2) → COMMIT (1, 1)" if changed else "Proposals (2, 2) → COMMIT (0, 0)"
                outcome = f"Complete at step {panel['changed_terminal']['steps']}" if changed else f"2 pending at cutoff {panel['original_terminal']['steps']}"
            else:
                detail = "Propose 1 → COMMIT 1 → write 1" if changed else "Propose 0 → COMMIT 0 → keep 0"
                outcome = f"Proper after repair queue, step {panel['changed_terminal']['steps']}" if changed else f"1 conflict after queue, step {panel['original_terminal']['steps']}"
            ax.text(center, .35, detail, ha="center", fontsize=11.3, color=ink)
            ax.text(center, .02, outcome, ha="center", fontsize=11.3, color=blue if changed else red)
        ax.annotate("", xy=(5.4, 1.13), xytext=(4.7, 1.13),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color=muted))
    fig.text(.5, .01, "Circle = current color after the queried step; – = pending. Other vertices omitted.",
             fontsize=10.5, ha="center", color=muted)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / "failure-mechanisms.png", dpi=180)
    plt.close(fig)


def main():
    record = evidence()
    draw(record)
    record["script_sha256"] = sha(Path(__file__))
    record["outputs"] = {str(p.relative_to(ROOT)): sha(p) for p in
                         (FIGURE, OUT / "failure-mechanisms.png")}
    (OUT / "failure-mechanisms.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(panels=len(record["panels"]), source_checks="pass", figure=str(FIGURE))))


if __name__ == "__main__":
    main()
