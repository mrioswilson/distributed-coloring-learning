#!/usr/bin/env python3
"""Hold degree at three and compare global4/local lists on exact GM prefixes."""
import json
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from dcl.degree.inventory import ROOT
from dcl.degree.adapter import FrozenAdapter
from dcl.degree.teachers import DegreeProgram,exact_choices
from dcl.multiteacher.programs import REPAIR
from dcl.samplecomplexity.model import QueryPolicy
from dcl.study.storage import atomic_json,file_hash


if __name__=='__main__':
    torch.set_num_threads(1)
    # A legal adversarial schedule: the two edge endpoints are in slot zero
    # of distinct balanced blocks. IDs are supplied before any teacher step.
    g=nx.empty_graph(8);g.add_edge(0,1);g.add_edges_from((3,v) for v in (4,5,6))
    target=0;component={0,1}
    order=np.array([0,2,1,7,3,4,5,6])
    ids=np.zeros(8,np.int64);ids[order]=(np.arange(8)+1)*1024
    assert max(dict(g.degree()).values())==3 and g.degree(target)==1
    states=[]
    for palette in ('global','local'):
        p=DegreeProgram(g,'SPECULATIVE_GM',9332,palette)
        p.ids=ids.copy();p.order=np.argsort(p.ids);p.blocks=np.array_split(p.order,4)
        p.slots=np.zeros(p.n,np.int64)
        for block in p.blocks:p.slots[block]=np.arange(len(block))
        found=None
        while not p.finished:
            s=p.snapshot();truth=exact_choices(p,s)
            if s.phase==REPAIR and s.permission[target]:
                found=(p,s,truth);break
            p.apply(s,truth)
        assert found is not None,'Selected endpoint must belong to exact repair queue'
        states.append(found)
    for attr in ('colors','ids','slots'):
        np.testing.assert_array_equal(getattr(states[0][1],attr)[sorted(component)],getattr(states[1][1],attr)[sorted(component)])
    assert states[0][2].color[target]==states[1][2].color[target]==2
    cps=[r for r in json.loads((ROOT/'checkpoint-inventory.json').read_text()) if r['teacher']=='SPECULATIVE_GM']
    rows=[]
    for cp in cps:
        model=QueryPolicy(cp['teacher'],cp['architecture']).eval()
        model.load_state_dict(torch.load(cp['path'],weights_only=False)['model'])
        adapter=FrozenAdapter(model)
        for p,s,truth in states:
            out=adapter(p,s);index=int(np.where(out.roots==target)[0][0])
            rows.append(dict(checkpoint=cp['id'],architecture=cp['architecture'],method=cp['method'],seed=cp['seed'],
                delta=3,root=int(target),root_degree=1,palette_regime=p.palette_regime,root_palette=int(p.palette[target]),
                exact_color=int(truth.color[target]),proposal=int(out.candidates[index]),
                commit=bool(out.commit[index]),commit_logit=float(out.commit_logits[index]),
                predicted_color=int(out.choices.color[target]),scores=out.scores[index].tolist()))
    atomic_json(ROOT/'palette-degree-mechanism.json',dict(graph_edges=list(g.edges()),n=len(g),delta=3,
        selected_component=sorted(component),ids=ids.tolist(),order=order.tolist(),tape_seed=9332,rows=rows,
        interpretation='Posthoc diagnostic on exact legal GM repair states. Same degree, ID order and endpoint colors; teacher target unchanged. No student rollout is repaired or replaced.',
        script_sha256=file_hash(__file__)))
    print([(r['method'],r['palette_regime'],r['proposal'],r['commit']) for r in rows if r['architecture']=='transformer' and r['seed']==94102])
