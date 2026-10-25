#!/usr/bin/env python3
"""Posthoc sensitivity to two declared preference encodings, without repairs.

Primary: raw preference / global degree (the frozen template). Diagnostic:
legacy units raw preference / 3, via an observation-only common rescaling.
Teacher inputs/targets, graph, weights and decisions remain separate.
This is not a replacement adapter selected for a better final score.
"""
import copy
import json
import networkx as nx
import numpy as np
import torch
from dcl.degree.inventory import ROOT
from dcl.degree.adapter import FrozenAdapter
from dcl.degree.teachers import DegreeProgram,exact_choices
from dcl.samplecomplexity.model import QueryPolicy
from dcl.study.storage import atomic_json,file_hash


if __name__=='__main__':
    torch.set_num_threads(1)
    cps=[r for r in json.loads((ROOT/'checkpoint-inventory.json').read_text()) if r['primary'] and r['teacher']=='COLOR_TRIALS']
    rows=[]
    for cp in cps:
        model=QueryPolicy('COLOR_TRIALS',cp['architecture']).eval()
        model.load_state_dict(torch.load(cp['path'],weights_only=False)['model'])
        adapter=FrozenAdapter(model)
        for delta in (3,5,10,50,200):
            g=nx.disjoint_union(nx.path_graph(2),nx.star_graph(delta))
            p=DegreeProgram(g,'COLOR_TRIALS',9321,'local');s=p.snapshot()
            s.preferences[:]=0;s.preferences[0,0]=delta;s.preferences[1,0]=delta-1
            truth=int(exact_choices(p,s).color[0]);assert truth==1
            for encoding in ('global_bound_fraction','legacy_units_fixed3'):
                observation=s
                if encoding=='legacy_units_fixed3':
                    observation=copy.copy(s)
                    observation.preferences=s.preferences*np.float32(delta/3)
                out=adapter(p,observation,overrides=dict(participation=s.permission,candidates=np.zeros(p.n,np.int64),query=np.zeros(p.n,np.int64)))
                index=int(np.where(out.roots==0)[0][0])
                rows.append(dict(checkpoint=cp['id'],architecture=cp['architecture'],seed=cp['seed'],delta=delta,
                    encoding=encoding,root_degree=1,root_palette=2,exact_color=truth,
                    queried_commit=bool(out.commit[index]),logit=float(out.commit_logits[index])))
    atomic_json(ROOT/'ct-normalization-sensitivity.json',dict(rows=rows,
        interpretation='Posthoc observation-encoding probe only. Primary frozen transfer results are unchanged. Neither encoding has information loss for strict preference order; no teacher action or availability is inserted.',
        caution='Legacy units change the embedding outside the native domain; these query responses do not establish autonomous high-degree success under that adapter.',
        script_sha256=file_hash(__file__)))
    print({(e,d):sum(r['queried_commit'] for r in rows if r['encoding']==e and r['delta']==d) for e in ('global_bound_fraction','legacy_units_fixed3') for d in (3,5,10,50,200)})
