#!/usr/bin/env python3
"""New native controls and real-weight adapter parity, no checkpoint selection."""
from concurrent.futures import ProcessPoolExecutor
import json
import networkx as nx
import numpy as np
from pathlib import Path
import torch
from dcl.degree.inventory import ROOT
from dcl.degree.adapter import FrozenAdapter
from dcl.degree.teachers import DegreeProgram, exact_choices
from dcl.degree.evaluation import evaluate
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher_e1.model import native_prediction
from dcl.samplecomplexity.model import QueryPolicy
from dcl.samplecomplexity.evaluation import evaluate as native_evaluate
from dcl.study.storage import atomic_json, file_hash
from dcl.multiteacher.programs import named_seed


def task(cp):
    torch.set_num_threads(1)
    model = QueryPolicy(cp["teacher"], cp["architecture"])
    model.load_state_dict(torch.load(cp["path"], weights_only=False)["model"])
    model.eval()
    suite = []
    for i, g in enumerate((nx.path_graph(16), nx.star_graph(3), nx.complete_graph(4), nx.empty_graph(8),
                           nx.random_regular_graph(3, 32, seed=31), nx.disjoint_union(nx.complete_graph(4), nx.empty_graph(12)))):
        suite.append((g, dict(case_id=f"new-native-{i}", n=len(g), family=str(i),
                              tape_seed=named_seed("native-degree-20261025", cp["seed"] % 100, i))))
    rows, seconds = native_evaluate(model, cp["teacher"], suite)
    result = dict(checkpoint=cp["id"], native_rows=rows, native_seconds=seconds, parity_roots=0,
                  max_abs_logit_error=0., hard_choice_differences=0)
    if cp["teacher"] != "BE_BASE_D3":
        adapter = FrozenAdapter(model)
        with torch.inference_mode():
            for g, meta in suite:
                p = DegreeProgram(g, cp["teacher"], meta["tape_seed"], "native4")
                while not p.finished:
                    s = p.snapshot()
                    roots = np.flatnonzero(s.permission)
                    arrays = tuple(torch.from_numpy(a) for a in encode_roots(s, p.neighbors, roots))
                    original = model(*arrays, phase=s.phase)
                    out = adapter(p, s)
                    expected = native_prediction(original)
                    for a, b in zip(expected, (out.choices.color[roots], out.choices.auxiliary[roots], out.choices.mark[roots])):
                        result["hard_choice_differences"] += int(np.sum(a.numpy() != b))
                    result["parity_roots"] += len(roots)
                    result["max_abs_logit_error"] = max(result["max_abs_logit_error"],
                        float(np.max(np.abs(original.commit.numpy() - out.commit_logits))),
                        float(np.max(np.abs(original.scores.numpy() - out.scores))))
                    p.apply(s, exact_choices(p, s))
        new_rows, _, _ = evaluate(adapter, cp["teacher"], suite, "native4")
        result["adapter_native_successes"] = sum(r["success"] for r in new_rows)
        result["terminal_color_differences"] = sum(a["colors"] != b["colors"] for a, b in zip(rows, new_rows))
    atomic_json(ROOT / "validation" / "native" / f"{cp['id']}.json", result)
    return result


if __name__ == "__main__":
    cps = json.loads((ROOT / "checkpoint-inventory.json").read_text())
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(task, cps))
    atomic_json(ROOT / "validation" / "native-controls.json", results)
    print(json.dumps(dict(checkpoints=len(results), native_executions=sum(len(r["native_rows"]) for r in results),
        native_failures=sum(not r["terminal_correct"] or not r["invariants"] for x in results for r in x["native_rows"]),
        parity_roots=sum(r["parity_roots"] for r in results),
        hard_choice_differences=sum(r["hard_choice_differences"] for r in results),
        terminal_color_differences=sum(r.get("terminal_color_differences", 0) for r in results),
        max_abs_logit_error=max(r["max_abs_logit_error"] for r in results))))
