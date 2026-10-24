#!/usr/bin/env python3
"""Known seed-83105 failures are regression fixtures, never final evidence."""
import json
from pathlib import Path

import networkx as nx
import torch

from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.training import initialize
from dcl.samplecomplexity.protocol import passed_run
from dcl.study.storage import atomic_json,file_hash

OUT=Path("results/multiteacher-counterfactual")
OLD=Path("/workspaces/AlgoLearning/ucgnn-sample-complexity")


if __name__=="__main__":
    torch.set_num_threads(1)
    certification=json.loads((OLD/"results/sample-complexity/certification.json").read_text())
    records=[]
    for arch in ("normalized","deepsets","transformer"):
        c=next(r for r in certification if r["teacher"]=="SPECULATIVE_GM" and r["architecture"]==arch)
        budget=c["candidate_budget"]
        original=OLD/"data/sample-complexity-v1/runs/speculative_gm"/arch/"s83105"/f"b{budget}"/"final.pt"
        rows=json.loads((OLD/"results/sample-complexity/final/speculative_gm"/arch/"s83105.json").read_text())
        failed=next(r for r in rows if not passed_run(r))
        g=nx.Graph();g.add_nodes_from(range(failed["n"]));g.add_edges_from(failed["edges"])
        suite=[(nx.empty_graph(1),dict(case_id="known-isolated-singleton",tape_seed=1,n=1,family="regression-singleton")),
               (g,dict(case_id="historical-"+failed["case_id"],tape_seed=failed["tape_seed"],n=len(g),family="historical-regression"))]
        model=initialize("SPECULATIVE_GM",arch,83105)
        model.load_state_dict(torch.load(original,weights_only=False)["model"])
        actual,_=evaluate(model,"SPECULATIVE_GM",suite)
        records.append(dict(cohort="original-83105",architecture=arch,seed=83105,checkpoint_sha256=file_hash(original),
                            rows=actual,scope="Authenticated known-failure reproduction"))
        for seed in range(94101,94106):
            checkpoint=OUT/"runs/speculative_gm"/arch/f"s{seed}"/"b3000-coverage1-alternate0/counterfactual/u004000.pt"
            if not checkpoint.exists():raise RuntimeError("New regression cohort incomplete")
            model=initialize("SPECULATIVE_GM",arch,seed)
            model.load_state_dict(torch.load(checkpoint,weights_only=False)["model"])
            actual,_=evaluate(model,"SPECULATIVE_GM",suite)
            records.append(dict(cohort="new-coverage-counterfactual",architecture=arch,seed=seed,
                checkpoint_sha256=file_hash(checkpoint),rows=actual,
                scope="Historical regression only; not an independent certification or causal coverage ablation"))
    atomic_json(OUT/"gm-83105-regression.json",records)
    print(json.dumps(dict(original_failures=sum(not passed_run(row) for r in records if r["cohort"]=="original-83105" for row in r["rows"]),
        new_successes=sum(passed_run(row) for r in records if r["cohort"]!="original-83105" for row in r["rows"]))))
