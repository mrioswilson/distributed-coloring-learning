import json
from pathlib import Path
import numpy as np
import pytest
import torch

from dcl.paperstudy.models import PRIMARY,make_model
from dcl.paperstudy.mechanisms import center_state,choices,patterns,progress_states,candidate_metrics
from dcl.universal.rules import available_matrix


@pytest.mark.parametrize("treatment",PRIMARY)
def test_cached_learned_center_matches_actual_star_forward(treatment):
    torch.set_num_threads(1)
    torch.manual_seed(77)
    model=make_model(treatment).eval()
    colors=np.array([-1,0,3,-1,5,7,-1])
    k=11
    n=len(colors)
    leaves=np.arange(1,n)
    edges=np.vstack((np.r_[np.zeros(n-1,int),leaves],np.r_[leaves,np.zeros(n-1,int)]))
    grid=np.array([.01,.12,.31,.67,.88,.99],np.float32)
    cached=center_state(model,colors,k)
    selected=choices(model,cached,grid)
    for i,r in enumerate(grid):
        with torch.inference_mode():
            out=model.stage1(torch.tensor(colors),torch.full((n,),.25),torch.full((n,),float(r)),torch.tensor(edges),torch.full((n,),k))
        torch.testing.assert_close(cached[1],out[2][0],rtol=1e-5,atol=1e-6)
        assert int(out[1][0].argmax())==selected[i]


def test_controlled_progress_states_are_proper_and_have_declared_availability():
    states=list(progress_states())
    assert len(states)==20
    for meta,colors,edges in states:
        src,dst=edges
        assert not ((colors[src]>=0)&(colors[src]==colors[dst])).any()
        assert int(np.bincount(src,minlength=len(colors)).max())<=meta["declared_degree_bound"]
        free=available_matrix(colors,edges,meta["palette"])
        core=meta["pending_core_nodes"]
        assert np.all(free[:core].sum(1)==meta["available_count"])
        assert meta["available_count"]>=meta["core_degree"]+1


def test_distribution_metrics_separate_forbidden_mass_and_valid_rank_errors():
    selected=np.array([0,0,1,3])
    available=np.array([0,3])
    metrics,q=candidate_metrics(selected,available,4,np.array([.1,.3,.6,.9],np.float32))
    assert metrics["forbidden_rate"]==.25
    assert metrics["allowed_draws"]==3
    assert metrics["tv_uniform"]==.25
    assert metrics["pair_collision"]==.375
    assert q.sum()==1


def test_pattern_constructors_have_no_duplicates_and_endpoints():
    for k in (4,17,63,251,997):
        for count in (1,2,k//2,k):
            for pattern in ("LOW","HIGH","SCATTERED","ALTERNATING","EXTREMES"):
                values=patterns(k,count,pattern)
                assert len(np.unique(values))==count
                assert np.all(np.diff(values)>0) and values.min()>=0 and values.max()<k


def test_unsafe_commits_are_executed_and_rollout_continues():
    from dcl.paperstudy.evaluation import rollout
    from dcl.universal.model import UOutput
    class Unsafe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor=torch.nn.Parameter(torch.zeros(1))
        def forward(self,colors,rp,rc,edges,palette,edge_chunk=None):
            n,k=len(colors),int(palette.max())
            zeros=torch.zeros((n,k))
            p=(rp<.5)&(colors<0)
            candidate=torch.where(colors<0,torch.zeros(n,dtype=torch.long),torch.full((n,),-1,dtype=torch.long))
            return UOutput(.5-rp,zeros,torch.ones(n),p,candidate,zeros,zeros.sum(1),zeros,zeros)
    n=10
    src,dst=np.where(~np.eye(n,dtype=bool))
    result,colors=rollout(Unsafe(),n,np.array([src,dst]),n,11,max_rounds=16)
    assert result["first_unsafe_round"]==1
    assert result["rounds"]>1
    assert result["monochromatic_edges"]>0
    assert result["unsafe_commits"]>0 and not result["success"]
    assert (colors>=0).all()
