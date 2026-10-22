import numpy as np
from dcl.sizedegree.envelope import masses

def test_two_affine_winners():
    q=masses([[0],[1]],[[1],[-1]],[1])
    assert np.allclose(q['q_reference'],[.5,.5])

def test_nonmonotone_winner_regions_and_ties():
    q=masses([[-.25,.75,0],[0,0,.6]],[[1,-1,0],[0,0,0]],[1,1,1])
    assert np.allclose(q['q_reference'],[.3,.7])
    assert masses([[1],[1]],[[0],[0]],[1])['q_reference']==[1.,0.]

def test_reference_integral_matches_independent_dense_grid():
    r=np.random.default_rng(954);A=r.normal(size=(5,7));B=r.normal(size=(5,7));w=r.normal(size=7)
    q=masses(A,B,w)['q_reference'];grid=(np.arange(100001)+.5)/100001
    winners=np.concatenate([np.maximum(A[None]+B[None]*g[:,None,None],0)@w for g in np.array_split(grid,20)]).argmax(1)
    actual=np.bincount(winners,minlength=5)/len(grid)
    assert np.max(abs(actual-q))<5e-5
