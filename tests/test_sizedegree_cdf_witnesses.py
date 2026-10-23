import numpy as np
from scipy.stats import binom
from dcl.sizedegree.cdf_witnesses import witness,radius
from dcl.sizedegree.inference import measurements

def test_cdf_witness_respects_discrete_ties_and_row_order():
    x=np.array([0,0,1,2,2]);y=np.array([0,1,1,1,2])
    r=witness(x,y,.025)
    assert np.isclose(r['ks'],.2) and r['threshold'] in (0,1)
    assert (r['count_Luby'],r['count_model']) in ((2,1),(3,4))
    assert r==witness(x[::-1],np.roll(y,2),.025)
    assert witness([],y,.025)['tv_lower']==0

def test_dkw_point_null_and_valid_coloring_positive_control():
    g=dict(n=3,delta=2,degrees=[1,2,1],edges=[[0,1,1,2],[1,0,2,1]])
    x=np.tile([0,1,0],(200,1));y=np.tile([0,2,1],(200,1))
    for z in (x,y):
        assert (z<=np.array(g['degrees'])).all()
        assert (z[:,g['edges'][0]]!=z[:,g['edges'][1]]).all()
    a=measurements(x,g)['used_colors'];b=measurements(y,g)['used_colors']
    assert witness(a,a,.025)['tv_lower']==0
    r=witness(a,b,.025)
    assert r['ks']==1 and .75<r['tv_lower']<1

def test_exact_binomial_null_level_after_threshold_selection():
    n=100;k=np.arange(n+1);p=binom.pmf(k,n,.5)
    # For a binary variable the empirical supremum is exactly the gap in
    # zero frequencies. Enumerate every possible pair of independent counts.
    reject=abs(k[:,None]-k[None,:])/n>2*radius(n,.025)
    exact_rate=float(np.sum(p[:,None]*p[None,:]*reject))
    assert 0<exact_rate<=.05
