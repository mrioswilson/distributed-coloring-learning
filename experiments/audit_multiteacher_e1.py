"""Post-training numerical audit against the unchanged E1 core and saved rollouts."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from dcl.m4_data import _edge_index
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import Choices, ColoringProgram
from dcl.multiteacher_e1.experiment import rollout_many, verify_metadata
from dcl.multiteacher_e1.model import CANONICAL, RootedE1, native_prediction
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.study.graphs import generate_graph
from dcl.study.storage import atomic_json, file_hash, now


def build_case(case):
    if case["family"] == "CLIQUE":
        return nx.complete_graph(4)
    return generate_graph(case["n"], 3, case["family"], case["graph_seed"])


@torch.inference_mode()
def audit_core(model, graph, case):
    """Run full original E1 modules on native states; compare rooted proposals.

    The schedule mask is applied between original stage1 and commit_logits;
    for Luby/Johansson it is exactly the original uncolored-node mask.
    """
    program = ColoringProgram(graph, model.teacher, case["tape_seed"])
    edges = torch.from_numpy(_edge_index(graph))
    palette = torch.full((len(graph),), 4, dtype=torch.long)
    differences = dict(participation=0., candidate=0., common_commit=0.)
    decision_nodes, action_mismatches = 0, 0
    while not program.finished:
        state = program.snapshot()
        roots = np.flatnonzero(state.permission)
        colors = torch.from_numpy(state.colors)
        p_logit, scores, *_ = model.core.stage1(colors, torch.from_numpy(state.draws[:, 0]),
                                               torch.from_numpy(state.draws[:, 1]), edges, palette)
        p = (p_logit > 0) & torch.from_numpy(state.permission)
        c = torch.where(colors < 0, scores.argmax(1), -1)
        commit = model.core.commit_logits(colors, edges, palette, p, c)
        full_action = torch.where(p & (commit > 0), c + 1, 0).numpy()
        out = model(*tensor_arrays(encode_roots(state, program.neighbors, roots)), phase=state.phase)
        centers = np.column_stack((roots, program.neighbors[roots]))
        star_p = torch.cat((p, torch.tensor([False])))[centers]
        star_c = torch.cat((c, torch.tensor([-1])))[centers]
        common = model.commit_for(out.state, star_p, star_c)
        for name, actual, expected in (("participation", out.state.p_logits[:, 0], p_logit[roots]),
                                        ("candidate", out.scores, scores[roots]),
                                        ("common_commit", common, commit[roots])):
            differences[name] = max(differences[name], float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        rooted_action = native_prediction(out)[0].numpy()
        action_mismatches += int((rooted_action != full_action[roots]).sum())
        decision_nodes += len(roots)
        program.apply(state, Choices(full_action, np.zeros(len(graph), np.int64), np.zeros(len(graph), bool)))
    return {"case": case["id"], "steps": program.steps, "decision_nodes": decision_nodes,
            "max_abs_logit_difference": differences, "hard_action_mismatches": action_mismatches,
            "full_core_colors": program.colors.tolist()}


def audit(root, output):
    root = Path(root)
    frozen = json.loads((root / "frozen.json").read_text())
    config = frozen["config"]
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    verify_metadata(frozen["provenance"])
    rows = json.loads((root / "evaluation-rows.json").read_text())
    cases = {r["id"]: {k: v for k, v in r.items() if k in
             ("id", "n", "family", "graph_i", "tape_i", "graph_seed", "fingerprint", "tape_seed")} for r in rows}
    first_n = config["test_counts"][0][0]
    ids = [f"REGULAR-n{first_n}-g0-t0", f"IRREGULAR-n{first_n}-g0-t0", "CLIQUE-n4-g0-t0"]
    selected = [(build_case(cases[i]), cases[i]) for i in ids]
    replays, core_checks = [], []
    for item in frozen["models"]:
        path = root / item["path"]
        if file_hash(path) != item["sha256"]:
            raise RuntimeError("Frozen checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = RootedE1(item["teacher"], config["hidden"], config["message"])
        model.load_state_dict(saved["model_state_dict"]); model.eval()
        results = rollout_many(model, item["teacher"], selected, config["batch_decisions"])
        for (graph, case), result in zip(selected, results):
            relative = Path("evaluations") / item["teacher"].lower() / f"s{item['seed']}" / f"{case['id']}.json"
            old = json.loads((root / relative).read_text())["result"]
            same = all(result[k] == old[k] for k in ("colors", "success", "failure", "native_steps"))
            replays.append({"teacher": item["teacher"], "seed": item["seed"], "case": case["id"], "identical": same})
        if item["teacher"] in CANONICAL:
            for graph, case in selected:
                checked = audit_core(model, graph, case)
                expected = json.loads((root / "evaluations" / item["teacher"].lower() /
                                       f"s{item['seed']}" / f"{case['id']}.json").read_text())["result"]["colors"]
                checked["full_core_matches_saved_coloring"] = checked.pop("full_core_colors") == expected
                core_checks.append({"teacher": item["teacher"], "seed": item["seed"], **checked})
    passed = (all(r["identical"] for r in replays) and
              all(r["hard_action_mismatches"] == 0 and r["full_core_matches_saved_coloring"] for r in core_checks))
    value = {"status": "passed" if passed else "review_required", "created_utc": now(),
             "frozen_sha256": file_hash(root / "frozen.json"), "models_checked": len(frozen["models"]),
             "native_replays": replays, "original_e1_core_checks": core_checks,
             "audit_source_sha256": file_hash(Path(__file__)),
             "scope": "Finite implementation audit on all seeds; scientific independent review still pending."}
    atomic_json(output, value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.output)
    print(json.dumps({"status": result["status"], "models": result["models_checked"],
                      "native_replays": len(result["native_replays"]), "core_checks": len(result["original_e1_core_checks"])}))
    if result["status"] != "passed":
        raise SystemExit(1)
