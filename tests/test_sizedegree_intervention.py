import numpy as np
import torch
from dcl.sizedegree.graphs import regular
from dcl.sizedegree.intervention import coupled
from dcl.globaldist.sampling import sample

def test_teacher_coupling_identical_and_frozen_sampler_marginal():
    g=regular(8,3,0,'test');stage='test-candidate-intervention';label='test-label'
    a=coupled(g,None,label,stage,64);b=sample(g,None,label,stage,64)
    assert not a['coarsened_mismatch'].any()
    assert np.array_equal(a['colors_Luby'],b['colors'])
    assert np.array_equal(a['status_Luby'],b['status'])
    assert np.array_equal(a['seeds'],b['seeds'])

def test_intervention_does_not_introduce_a_commit_shield(monkeypatch):
    import dcl.sizedegree.intervention as m
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1))
        def participation(self,x):return torch.ones((len(x),1))
    class UnsafeAdapter:
        def __init__(self,model):pass
        def commit_logits(self,colors,edges,pal,part,candidate):return torch.ones(len(colors))
    monkeypatch.setattr(m,'LocalPaletteAdapter',UnsafeAdapter)
    g=dict(id='edge',n=2,degrees=[1,1],delta=1,edges=[[0,1],[1,0]])
    a=m.coupled(g,Model(),'unsafe','test-no-new-shield',100)
    assert a['unsafe_commits'].sum()>0
    assert (a['status_intervened']==1).any()
    assert not (a['status_Luby']==1).any()
