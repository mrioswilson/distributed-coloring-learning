#!/usr/bin/env python3
"""Export hard private-preference, priority, algebraic and schedule witnesses."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np

from dcl.causal_interventions import contexts,replay_context
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import TEACHERS,LINIAL,DETECT,teacher_choices
from dcl.samplecomplexity.data import array_hash,components
from dcl.study.storage import atomic_json,file_hash
from dcl.multicf.oracles import CONTRACTS,stage_targets,confirmation,proposal_bank

OUT=Path("results/multiteacher-counterfactual/verified-interventions")


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True);summary=[]
    graphs=[nx.disjoint_union(nx.complete_graph(4),nx.empty_graph(4)),nx.path_graph(8)]
    rng=np.random.default_rng(77101)
    for teacher in TEACHERS:
        examples=[];counts=Counter()
        for graph in graphs:
            for p,s,exact,proof in contexts(graph,teacher):
                q,check=replay_context(proof)
                roots=np.flatnonzero(s.permission)
                a=encode_roots(s,p.neighbors,roots);b=encode_roots(check,q.neighbors,roots)
                if array_hash(*a)!=array_hash(*b):raise RuntimeError("Re-encoding changed a causal witness")
                native=np.c_[exact.color[roots],exact.auxiliary[roots],exact.mark[roots],np.full(len(roots),s.phase)]
                fields={}
                for c in components(s.phase):counts[c]+=len(roots)
                if s.phase not in (LINIAL,DETECT):
                    t=stage_targets(a,native,teacher)
                    if not np.array_equal(confirmation(a,teacher,t.participate,t.candidate)>0,exact.color[roots]>0):
                        raise RuntimeError("Intervened-state native differential failed")
                    ps,cs,ys=proposal_bank(a,native,teacher,rng)
                    fields=dict(p=ps.tolist(),c=cs.tolist(),confirmation=ys.tolist());counts["counterfactual_confirmation"]+=4*len(roots)
                examples.append(dict(proof=proof,roots=roots.tolist(),native=native.tolist(),
                    encoded_sha256=array_hash(*a),counterfactual=fields))
        target=OUT/f"{teacher.lower()}.json";atomic_json(target,examples)
        summary.append(dict(teacher=teacher,contract=asdict(CONTRACTS[teacher]),contexts=len(examples),
            acquired_diagnostic_labels=dict(counts),sha256=file_hash(target),
            scope="Verified, replayable supplementary cases; not used to fit or select the frozen main policies"))
    atomic_json(OUT/"manifest.json",dict(rows=summary,script_sha256=file_hash(__file__),
        causal_library_sha256=file_hash("src/dcl/causal_interventions.py")))
    print({r["teacher"]:r["contexts"] for r in summary})
