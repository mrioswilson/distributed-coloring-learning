import itertools
import numpy as np
from dcl.robustdist.common import rng
from dcl.robustdist.controls import draw
from dcl.robustdist.stats import joint_test,multinomial_test

def test_explanatory_event_never_uses_reserved_values_for_selection(monkeypatch):
    import dcl.robustdist.confirmation as c
    a,g,_=draw('localized',750,rng('explain-a'),0);b,_,_=draw('localized',750,rng('explain-b'),1)
    banks={'Luby':dict(colors=a,status=np.zeros(750,int)),'model':dict(colors=b,status=np.zeros(750,int))}
    monkeypatch.setattr(c,'loadbank',lambda label,graph:banks[label])
    desc=[dict(kind='threshold',family='local',ids=[0],threshold=101),dict(kind='threshold',family='local',ids=[1],threshold=0)]
    first=c.explanation('model',g,desc,a[:500],b[:500])
    assert first['event']['ids']==[0]
    assert first['count_Luby']==int((a[-250:,0]<=101).sum())
    assert first['count_model']==250
    assert first['ci_Luby'][0]<=100/199<=first['ci_Luby'][1]
    a[-250:,0]=200
    second=c.explanation('model',g,desc,a[:500],b[:500])
    assert second['event']==first['event'] and second['count_Luby']==0

def test_sparse_direct_test_uses_conditional_calibration():
    x=np.arange(10).reshape(-1,1);y=np.arange(10,20).reshape(-1,1)
    result=multinomial_test(x,y,rng('sparse-direct'),B=99)
    assert result['support_observed']==20 and result['p']==1

def test_actual_omnibus_exact_conditional_ranks():
    z,g,_=draw('uniform_null',20,rng('exact-pool'))
    pool=np.array([(0,1,0),(1,0,1),(0,2,1),(1,2,0)],np.int16)
    trainx=z[:2];trainy=z[2:4];T=[]
    for ids in itertools.combinations(range(4),2):
        ids=list(ids);other=[j for j in range(4) if j not in ids]
        x=np.r_[trainx,pool[ids]];y=np.r_[trainy,pool[other]]
        T.append(joint_test(x,y,g,rng('exact-no-monte-carlo'),B=0)['statistics']['omnibus'])
    T=np.array(T);p=(T[None,:]>=T[:,None]-1e-12).mean(1)
    for alpha in (.05,.2,.5,1.):assert (p<=alpha).mean()<=alpha+1e-14

def test_small_count_sampler_matches_enumerated_hypergeometric_tail():
    from dcl.robustdist.small import categorical_test
    x=np.zeros((3,1),int);y=np.ones((3,1),int)
    # Two extreme allocations among choose(6,3)=20: exact two-sided p=.1.
    r=categorical_test(x,y,rng('small-count-exact'),B=9999)
    assert abs(r['p']-.1)<.02
    assert abs(r['statistic']-12*np.log(2))<1e-12
    assert categorical_test(x,x,rng('small-count-point'),B=99)['p']==1
