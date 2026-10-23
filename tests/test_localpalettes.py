import numpy as np
import pytest
import torch

from dcl.localpalettes.adapter import LocalPaletteAdapter, converted_uniform
from dcl.localpalettes.cohort import edge_array, make_graph, palette_for, state_specs
from dcl.localpalettes.statistics import binary_product_tv, law_metrics, likelihood_curve, product_bounds
from dcl.paperstudy.models import PRIMARY, make_model
from dcl.universal.rules import teacher_offer, exact_commit_labels

torch.set_num_threads(1)


@pytest.mark.parametrize('treatment',PRIMARY)
@pytest.mark.parametrize('mixed',[False,True])
def test_historical_equivalence(treatment,mixed):
    torch.manual_seed(123)
    model=make_model(treatment).eval()
    colors=torch.tensor([-1,5,-1,0,-1,1,-1])
    palettes=torch.tensor([3,6,2,4,7,2,1]) if mixed else torch.full((7,),7)
    edges=torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]])
    rp,rc=torch.rand(7),torch.rand(7)
    with torch.inference_mode():
        old=model(colors,rp,rc,edges,palettes)
        new=LocalPaletteAdapter(model)(colors,rp,rc,edges,palettes)
    for field in ('participation_logit','candidate_logits','availability','k_hat','lo','hi','commit_logit'):
        x,y=getattr(old,field),getattr(new,field)
        if field in ('lo','hi'):
            valid=torch.arange(x.shape[1])[None,:]<palettes[:,None]
            x,y=x[valid],y[valid]  # nonexistent diagnostic endpoints have no action path
        torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-5)
    assert torch.equal(old.candidate,new.candidate)
    assert torch.equal(old.participate,new.participate)


@pytest.mark.parametrize('treatment',PRIMARY)
def test_padding_permutation_batch_and_randomness(treatment):
    torch.manual_seed(156)
    model=make_model(treatment).eval()
    adapter=LocalPaletteAdapter(model)
    colors=torch.tensor([-1,5,-1,0,-1,1,-1])
    palettes=torch.tensor([3,6,2,4,7,2,1])
    edges=torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]])
    rp,rc=torch.rand(7),torch.rand(7)
    base=adapter(colors,rp,rc,edges,palettes)
    padded=adapter(colors,rp,rc,edges,palettes,output_width=51)
    torch.testing.assert_close(base.candidate_logits,padded.candidate_logits[:,:7],rtol=0,atol=0)
    assert torch.all(padded.candidate_logits[:,7:]==-1e9)
    order=torch.randperm(7);inverse=order.argsort()
    perm=adapter(colors[order],rp[order],rc[order],inverse[edges],palettes[order])
    torch.testing.assert_close(base.candidate_logits,perm.candidate_logits[inverse],atol=3e-5,rtol=3e-5)
    joint=adapter(torch.cat((colors,torch.tensor([-1]))),torch.cat((rp,torch.tensor([.3]))),
                  torch.cat((rc,torch.tensor([.8]))),edges,torch.cat((palettes,torch.tensor([201]))))
    torch.testing.assert_close(base.candidate_logits,joint.candidate_logits[:7,:7],atol=3e-5,rtol=3e-5)
    assert torch.equal(base.candidate,joint.candidate[:7])
    doubled=adapter(torch.cat((colors,colors)),torch.cat((rp,rp)),torch.cat((rc,rc)),
                    torch.cat((edges,edges+7),1),torch.cat((palettes,palettes)))
    torch.testing.assert_close(base.candidate_logits,doubled.candidate_logits[:7],atol=3e-5,rtol=3e-5)
    assert torch.equal(base.candidate,doubled.candidate[:7])
    swapped=adapter(colors,1-rp,rc,edges,palettes)
    assert torch.equal(base.candidate_logits,swapped.candidate_logits)
    swapped=adapter(colors,rp,1-rc,edges,palettes)
    assert torch.equal(base.participation_logit,swapped.participation_logit)
    # A fixed color outside the receiver's list is legal at the sender.
    assert int(colors[1])>=int(palettes[0]) and int(colors[1])<int(palettes[1])
    assert int(base.candidate[6])==0  # isolated singleton palette


def test_invalid_lists_and_converted_endpoints():
    model=LocalPaletteAdapter(make_model('normalized').eval())
    with pytest.raises(ValueError):model(torch.tensor([3]),torch.tensor([.2]),torch.tensor([.2]),torch.empty((2,0),dtype=torch.long),torch.tensor([3]))
    assert converted_uniform([np.nextafter(1.,0.)])[0]<1


def test_local_teacher_safety_and_availability():
    g,_=make_graph('HETEROGENEOUS',15,0,pilot=True)
    edges=edge_array(g);degrees=np.array([g.degree(i) for i in g])
    palette=palette_for('LOCAL',degrees);colors=np.full(len(g),-1)
    rng=np.random.default_rng(85)
    for _ in range(100):
        from dcl.universal.rules import available_matrix
        available=available_matrix(colors,edges,palette)
        pending=np.zeros(len(g),dtype=int)
        np.add.at(pending,edges[1],(colors[edges[0]]<0).astype(int))
        assert np.all(available.sum(1)[colors<0]>=pending[colors<0]+1)
        c,_=teacher_offer(colors,rng.random(len(g)),edges,palette)
        p=(rng.random(len(g))<.5)&(colors<0)
        commits=exact_commit_labels(colors,edges,p,c)
        old=colors.copy();colors[commits]=c[commits]
        assert np.all(colors[old>=0]==old[old>=0])
        assert not np.any((colors[edges[0]]>=0)&(colors[edges[0]]==colors[edges[1]]))
        assert np.all(colors<palette)
        if np.all(colors>=0):break
    assert np.all(colors>=0)


def test_state_realizability_witness():
    import networkx as nx
    for s in state_specs():
        assert len(s['colors'])==s['k']
        assert np.all(np.array(s['colors'])<np.array(s['palettes']))
        if s['k']!=11:continue
        g=nx.star_graph(s['k']-1)
        q=s['colors'].copy()
        for neighbor,extra in enumerate(s['witness_private_pending_leaves'],1):
            for _ in range(extra):
                g.add_edge(neighbor,len(q));q.append(-1)
        assert all(c<g.degree(v)+1 for v,c in enumerate(q))
        assert all(q[a]<0 or q[b]<0 or q[a]!=q[b] for a,b in g.edges())


def test_distribution_controls_and_lrt():
    from scipy.stats import binom
    assert binary_product_tv(.5,.5,2000)<1e-12
    assert binary_product_tv(.5,.8,100)>.99
    stats=law_metrics([100,100,100,100],np.arange(4))
    assert stats['tv']==0 and stats['effective_collision']==4
    assert product_bounds(0,1,100)==dict(lower=0,upper=0)
    rng=np.random.default_rng(857)
    qmodel=rng.choice(2,size=(256,2000),p=[.1,.9])
    qteacher=rng.integers(0,2,size=(256,2000))
    rows=likelihood_curve([1000,9000],np.arange(2),qmodel,qteacher,rng,256)
    assert rows[-1]['balanced_accuracy']>.99
    # A permuted quantile has zero shared-r fidelity and exactly equal law.
    r=(np.arange(400)+.5)/400
    teacher=(4*r).astype(int);shift=(teacher+1)%4
    assert np.mean(shift==teacher)==0
    assert np.array_equal(np.bincount(shift),np.bincount(teacher))


def test_conditional_formula_by_action_enumeration():
    import itertools
    from dcl.localpalettes.progress import expected_confirmations
    edges=np.array([[0,1,1,2],[1,0,2,1]])
    colors=np.full(3,-1);palette=np.array([2,3,2]);q=np.array([[.5,.5,0],[1/3,1/3,1/3],[.5,.5,0]])
    p=np.full(3,.5)
    expected=np.zeros(3)
    for participation in itertools.product((False,True),repeat=3):
        for candidate in itertools.product(range(2),range(3),range(2)):
            weight=(.5**3)*q[0,candidate[0]]*q[1,candidate[1]]*q[2,candidate[2]]
            expected+=weight*exact_commit_labels(colors,edges,participation,candidate)
    formula=expected_confirmations(q,p,colors,edges,np.arange(3)[None,:]<palette[:,None])
    np.testing.assert_allclose(expected,formula,atol=1e-14)
    assert np.all(formula>=.25)


def test_local_type_witness_legality():
    from dcl.localpalettes.verification import local_type_witness
    for mask in range(64):
        colors,part,cand,edges,palette=local_type_witness(mask)
        colors=np.array(colors);cand=np.array(cand)
        assert np.all(colors<palette) and np.all(cand<palette)
        assert palette[0]==1+mask.bit_count()
        assert not np.any((colors[edges[0]]>=0)&(colors[edges[0]]==colors[edges[1]]))
        label=exact_commit_labels(colors,edges,part,cand)[0]
        assert label==not_conflicting(mask)


def not_conflicting(mask):
    return not(mask&(1<<1) or mask&(1<<5))


def test_resume_charges_interruption_and_retains_partial_cases(tmp_path,monkeypatch):
    from dcl.localpalettes import runner
    from dcl.study.storage import atomic_json
    monkeypatch.setattr(runner,'ROOT',tmp_path/'runtime')
    monkeypatch.setattr(runner,'RESULTS',tmp_path/'results')
    partial=tmp_path/'runtime/graphs/m/LOCAL/case.json'
    atomic_json(partial,dict(status='success'))
    state=dict(gpu_seconds=0,tasks=[dict(id='work',status='running',started='2026-01-01T00:00:00+00:00')])
    result=runner.recover_and_extend(state)
    assert result['gpu_seconds']==1800 and result['tasks'][0]['status']=='pending'
    assert partial.exists()
    # Completed orphan worker receipt is credited and not rerun.
    atomic_json(tmp_path/'runtime/task-receipts/done.json',dict(seconds=23))
    state=dict(gpu_seconds=0,tasks=[dict(id='done',status='running')])
    result=runner.recover_and_extend(state)
    assert result['gpu_seconds']==23 and result['tasks'][0]['status']=='complete'


def test_adapter_never_calls_oracle(monkeypatch):
    import dcl.universal.rules as oracle
    def forbidden(*args,**kwargs):raise AssertionError('oracle entered inference')
    monkeypatch.setattr(oracle,'available_matrix',forbidden)
    monkeypatch.setattr(oracle,'teacher_offer',forbidden)
    for treatment in PRIMARY:
        model=LocalPaletteAdapter(make_model(treatment).eval())
        model(torch.tensor([-1,-1]),torch.tensor([.2,.8]),torch.tensor([.7,.3]),
              torch.tensor([[0,1],[1,0]]),torch.tensor([2,3]))


def test_worker_inherits_gpu_lock(tmp_path,monkeypatch):
    import subprocess,sys
    from pathlib import Path
    from dcl.localpalettes.runner import persistent_gpu_lock
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    with persistent_gpu_lock() as fd:
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(2)'],pass_fds=(fd,))
    try:
        with pytest.raises(RuntimeError):
            with persistent_gpu_lock():pass
    finally:child.terminate();child.wait()
    with persistent_gpu_lock():pass


def test_padding_cannot_win_against_extreme_valid_logits(monkeypatch):
    model=make_model('normalized').eval()
    monkeypatch.setattr(model,'palette_scores',lambda h,a,rc,p,m:torch.full_like(a,-1e12))
    output=LocalPaletteAdapter(model)(torch.tensor([-1,-1]),torch.tensor([.3,.7]),torch.tensor([.2,.8]),
                                    torch.tensor([[0,1],[1,0]]),torch.tensor([1,4]),output_width=50)
    assert output.candidate.tolist()==[0,0]


def test_exact_participation_integration():
    from dcl.localpalettes.controls import participation_measure
    head=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.ReLU(),torch.nn.Linear(2,1))
    with torch.no_grad():
        head[0].weight[:]=torch.tensor([[1.,0.],[-1.,0.]])
        head[0].bias[:]=torch.tensor([0.,0.])
        head[2].weight[:]=torch.tensor([[1.,0.]])
        head[2].bias[:]=-.25
    result=participation_measure(head)
    assert result['exact_rational']=='3/4' and result['probability_real']==.75


def test_calibration_rank_is_finite_sample_conservative():
    from dcl.localpalettes.statistics import conservative_threshold
    assert conservative_threshold(np.arange(1024))==973
    assert conservative_threshold(np.arange(19))==18
    assert np.isinf(conservative_threshold(np.arange(1)))
    # Rank of an independent continuous null amongst N+1 exchangeable scores.
    n=1024;k=974
    assert (n+1-k)/(n+1)<=.05


def test_singleton_product_law_and_sample_size():
    from dcl.localpalettes.singleton_products import product_tv,sample_size
    assert product_tv(0,2000)==0 and product_tv(1,1)==1
    assert product_tv(.5,5)==pytest.approx(1-1/32)
    assert sample_size(.5)==3 and sample_size(0) is None


def test_cyclic_candidate_law_with_gaps():
    from dcl.localpalettes.evaluation import cyclic_law
    q=cyclic_law([1,6,10],11)
    np.testing.assert_allclose(q[[1,6,10]],np.array([2,5,4])/11)
    assert q.sum()==pytest.approx(1.) and q[[0,2,3,4,5,7,8,9]].sum()==0
