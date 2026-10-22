import itertools
import numpy as np
from dcl.robustdist.stats import *
from dcl.robustdist.controls import draw,CASES
from dcl.robustdist.common import rng,seed,valid

def test_controls_are_valid_and_independent():
    seeds=set()
    for case in CASES:
        for side in (0,1):
            s=seed('test',case,side);assert s not in seeds;seeds.add(s)
            z,g,meta=draw(case,30,np.random.default_rng(s),side)
            assert valid(z,g).all()
    assert not np.array_equal(draw('uniform_null',30,rng('x'))[0],draw('uniform_null',30,rng('y'))[0])

def test_unbiased_statistic_and_negative_values():
    x=np.array([[0,1],[1,0],[0,1],[1,0]])
    K=np.exp(-distance(x,x));np.fill_diagonal(K,0)
    A=np.array([[1,1,0,0.]])
    direct=K[:2,:2].sum()/2+K[2:,2:].sum()/2-2*K[:2,2:].sum()/4
    assert np.allclose(u_stat(K,A,2,2),direct)
    assert direct<0

def test_holm_reference():
    assert np.allclose(holm([.001,.04,.02,.7]),[.004,.08,.06,.7])

def test_degenerate_ties_and_reproducibility():
    x,g,_=draw('point_null',30,rng('point'))
    a=joint_test(x,x,g,rng('perm'),B=99)
    b=joint_test(x,x,g,rng('perm'),B=99)
    assert a['p']==1 and a['p_values']==b['p_values']
    assert multiscale_test(x,x,rng('perm'),B=99)['p_values']['old_hamming']==1
    assert multinomial_test(x,x,rng('perm'),B=99)['p']==1

def test_fixed_selection_ignores_evaluation_labels():
    x,g,_=draw('localized',60,rng('a'),0);y,_,_=draw('localized',60,rng('b'),1)
    d=fit_features(x[:30],y[:30],g)
    z=np.r_[x[30:],y[30:]]
    F=evaluate(d,z,g);order=rng('order').permutation(len(z))
    assert np.array_equal(F[order],evaluate(d,z[order],g))
    assert d==fit_features(x[:30][::-1],y[:30][::-1],g)

def test_exact_permutation_level_small_bank():
    # Enumerate all label assignments: finite permutation p-values are superuniform.
    values=np.array([0.,0.,1.,2.,3.,5.]);N=len(values);m=3
    assignments=list(itertools.combinations(range(N),m))
    T=[]
    for ids in assignments:
        mask=np.zeros(N,bool);mask[list(ids)]=True
        T.append(abs(values[mask].mean()-values[~mask].mean()))
    T=np.array(T);p=np.mean(T[None,:]>=T[:,None]-1e-12,axis=1)
    for alpha in (.05,.1,.2,.5):assert (p<=alpha).mean()<=alpha+1e-15

def test_labels_are_preserved_and_higher_order():
    x,g,_=draw('higher_order',1000,rng('parity'),1)
    assert np.all(x[:,3]==(x[:,1]^x[:,2]))
    # A named vertex event changes when only one sample's columns are swapped.
    y,_,_=draw('marginal',1000,rng('marginal'),1)
    assert abs(y[:,1].mean()-y[:,2].mean())>.2
    assert not np.array_equal(y,y[:,[0,2,1]+list(range(3,y.shape[1]))])

def test_positive_localized_control_and_minimum_p():
    x,g,_=draw('localized',250,rng('positive-a'),0);y,_,_=draw('localized',250,rng('positive-b'),1)
    a=joint_test(x,y,g,rng('positive-p'),B=199)
    assert a['p']<=.05 and a['p']>=1/200

def test_futility_is_conservative():
    x,g,_=draw('star_null',50,rng('f-a'));y,_,_=draw('star_null',50,rng('f-b'))
    a=joint_test(x,y,g,rng('f-p'),B=299,futility=1)
    b=joint_test(x,y,g,rng('f-p'),B=299)
    assert a['p']>=b['p']
