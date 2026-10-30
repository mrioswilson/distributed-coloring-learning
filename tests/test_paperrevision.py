"""Scientific checks: sufficient-input split and budget/paired gradients."""
from collections import defaultdict
import json
import networkx as nx
import numpy as np
import pytest
import torch
from dcl.multicf.oracles import stage_targets
from dcl.multicf.states import trajectory
from dcl.multiteacher.model import encode_roots
from dcl.multiteacher.programs import LINIAL, DETECT
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.training import initialize
from dcl.paperrevision.keys import commit_key, held_out
from dcl.paperrevision.experiment import acquire, load_npz, objective


def test_manuscript_kw_example_is_a_replayed_legal_state():
    from dcl.multicf.states import replay
    from dcl.multicf.oracles import confirmation
    witness=json.loads(open("results/paper-revision-20261030/kw-example-witness.json").read())
    program,state=replay(witness)
    root=witness["root"]
    assert (state.bound,state.target,int(state.aux[root]))==(7,6,6)
    assert sorted(int(state.aux[u]) for u in program.graph[root])==[0,1,3]
    arrays=encode_roots(state,program.neighbors,np.array([root]))
    p=np.array([witness["p"]],bool);c=np.array([witness["c"]],int)
    assert c[0,0]==2
    assert confirmation(arrays,"BE_BASE_D3",p,c).tolist()==[1.]
    c[0,0]=1
    assert confirmation(arrays,"BE_BASE_D3",p,c).tolist()==[0.]
    c[0,0]=2;p[0,0]=False
    assert confirmation(arrays,"BE_BASE_D3",p,c).tolist()==[0.]


@pytest.mark.parametrize("teacher", ["BE_BASE_D3", "SPECULATIVE_GM"])
def test_sufficient_key_agrees_with_actual_logits(teacher):
    model = initialize(teacher, "normalized", 991).eval()
    seen = defaultdict(list)
    with torch.inference_mode():
        for graph in (nx.path_graph(8), nx.cycle_graph(8), nx.complete_graph(4)):
            for program, state, exact, _ in trajectory(graph, teacher, 881):
                if state.phase in (LINIAL, DETECT): continue
                roots = np.flatnonzero(state.permission)
                if not len(roots): continue
                arrays = encode_roots(state, program.neighbors, roots)
                native = np.column_stack((exact.color[roots],exact.auxiliary[roots],exact.mark[roots],np.full(len(roots),state.phase)))
                target = stage_targets(arrays, native, teacher)
                first = model.first(tensor_arrays(arrays), state.phase)
                for offset in range(4):
                    p, c = target.participate.copy(), target.candidate.copy()
                    c[:,0] = (c[:,0]+offset)%4
                    logits=model.commit_for(first,torch.from_numpy(p),torch.from_numpy(c)).numpy()
                    for i, logit in enumerate(logits): seen[commit_key(arrays,i,p,c,teacher)].append(logit)
    assert any(len(v)>1 for v in seen.values())
    assert all(np.ptp(v)<2e-6 for v in seen.values())


@pytest.mark.parametrize("teacher", ["BE_BASE_D3", "SPECULATIVE_GM"])
def test_budget_split_and_first_stage_gradients(tmp_path,teacher):
    config=json.loads(open("configs/paper-revision-20261030.json").read())
    config.update(native_budget=120,auxiliary_budget=48)
    manifest=acquire(config,teacher,990,tmp_path)
    assert manifest["optimization_targets_per_model"] == 168
    keys=json.loads((tmp_path/"type-keys.json").read_text())
    assert all(not held_out(json.loads(k)) for k in keys)
    base,aux=load_npz(tmp_path/"base.npz"),load_npz(tmp_path/"aux.npz")
    phase=int(aux["native"][0,3]); bi=np.flatnonzero(base["native"][:,3]==phase)[:8]
    ai=np.flatnonzero(aux["native"][:,3]==phase)[:8]
    assert len(bi) and len(ai)
    models=[initialize(teacher,"normalized",456) for _ in range(2)]
    for arm,model in zip(("native","cf"),models):
        objective(model,base,aux,bi,ai,arm,101,100).backward()
    for prefix in ("core.message1.","core.state1.","core.scorer.","core.participation.","context_message1.","context_state1.","context_participation."):
        left={k:v.grad for k,v in models[0].named_parameters() if k.startswith(prefix)}
        right={k:v.grad for k,v in models[1].named_parameters() if k.startswith(prefix)}
        for k in left:
            assert (left[k] is None and right[k] is None) or torch.equal(left[k],right[k]), k
