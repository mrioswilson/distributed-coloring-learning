import numpy as np
from dcl.globaldist.core import *
from dcl.globaldist.sampling import sample

def test_exact_and_nonuniform():
    for g in small_graphs():
        x,p,r,states=exact_teacher(g)
        assert len(x)==len(p) and np.isclose(p.sum(),1) and np.all(p>0)
        if g['id']=='edge':assert np.allclose(p,[.5,.5])
        if g['id']=='path3':assert p.max()-p.min()>.01

def test_kernel_strict_pd_and_statistic():
    x=np.array(list(itertools.product(range(3),repeat=3)))
    K=kernel(x,x);assert np.linalg.eigvalsh(K).min()>0
    a=np.zeros((25,3),int);b=np.ones((25,3),int)
    r=mmd_test(a,b,np.random.default_rng(8),B=199)
    assert r['p']==.005 and r['mmd2_unbiased']>1
    assert holm([.01,.03,.9]).tolist()==[.03,.06,.9]

def test_reset_seeds_batching_and_teacher_mc():
    g=small_graphs()[1]
    a=sample(g,None,'Luby','unit',2000,batch_nodes=32)
    b=sample(g,None,'Luby','unit',2000,batch_nodes=128)
    assert np.array_equal(a['colors'],b['colors']) and not a['status'].any()
    assert len(set(a['seeds'].tolist()))==2000
    x,p,_,_=exact_teacher(g)
    q=np.array([np.mean((a['colors']==y).all(1)) for y in x])
    assert abs(q-p).max()<.04

def test_null_permutation_exchangeability():
    rng=np.random.default_rng(123);pv=[]
    for i in range(100):
        a=rng.integers(0,3,(30,4));b=rng.integers(0,3,(30,4))
        pv.append(mmd_test(a,b,rng,B=99)['p'])
    assert sum(p<=.05 for p in pv)<=12

def test_statistic_matches_direct_unequal_samples():
    rng=np.random.default_rng(93);x=rng.integers(0,4,(7,5));y=rng.integers(0,4,(9,5))
    kx=kernel(x,x);ky=kernel(y,y);kxy=kernel(x,y)
    expected=(kx.sum()-len(x))/(len(x)*(len(x)-1))+(ky.sum()-len(y))/(len(y)*(len(y)-1))-2*kxy.mean()
    result=mmd_test(x,y,rng,B=9)
    assert abs(result['mmd2_unbiased']-expected)<1e-12

def test_clique_symmetry_exact_teacher():
    for g in small_graphs():
        if g['id'] in ('edge','triangle'):
            x,p,_,_=exact_teacher(g)
            assert np.allclose(p,np.full(len(p),1/len(p)))

def test_technical_failure_preserves_already_completed_runs(monkeypatch):
    import dcl.globaldist.sampling as sampling
    original=sampling.teacher_offer;calls=[]
    def fail_after_first_round(*args,**kwargs):
        calls.append(1)
        if len(calls)>1:raise RuntimeError('injected technical error')
        return original(*args,**kwargs)
    monkeypatch.setattr(sampling,'teacher_offer',fail_after_first_round)
    a=sampling.sample(small_graphs()[0],None,'Luby','technical-unit',100)
    valid=(a['colors']>=0).all(1)&(a['colors'][:,0]!=a['colors'][:,1])
    assert valid.any() and (~valid).any()
    assert np.all(a['status'][valid]==0) and np.all(a['status'][~valid]==3)
    assert 'injected technical error' in str(a['errors'])

def test_homogeneous_cycle_teacher_is_not_uniform():
    g=next(g for g in small_graphs() if g['id']=='cycle4');x,p,r,_=exact_teacher(g)
    assert set(g['degrees'])=={2}
    assert set(r.values())=={'35051/859584','36071/573056'}
    assert np.isclose(np.abs(p-1/len(p)).sum()/2,38111/429792)
