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
