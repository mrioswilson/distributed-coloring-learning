import numpy as np
import networkx as nx
from dcl.sizedegree.graphs import regular,irregular,validate
from dcl.sizedegree.common import seed,NS,DS
from dcl.sizedegree.inference import *
from dcl.globaldist.core import graph_record

def test_regular_labeled_reproducible_and_distinct():
    a=regular(128,3,0);b=regular(128,3,1)
    assert a==regular(128,3,0)
    assert a['sha256']!=b['sha256']
    assert validate([a,b])['passed']

def test_irregular_exact_max_connected_and_heterogeneous():
    for n,d in [(128,3),(128,100),(256,10)]:
        g=irregular(n,d)
        assert validate([g])['passed']
        assert min(g['degrees'])==2 and max(g['degrees'])==d
        assert g['degree_variance']>0
        assert sum(g['degree_histogram'].values())==n

def test_structural_definition_and_local_palettes():
    g=graph_record('path3',nx.path_graph(3),'path')
    z=np.array([[0,1,0],[1,2,0]])
    m=measurements(z,g)
    assert m['used_colors'].tolist()==[2,3]
    assert m['used_le_delta'].tolist()==[True,False]
    assert m['uses_top_color'].tolist()==[False,True]
    assert m['nonedge_reuse'].tolist()==[1.,0.]
    assert np.allclose(m['mean_color_normalized'],[1/6,2/3])

def test_shared_teacher_weights_are_combined():
    gs=[dict(id=f'g{i}',stage='main',n=128,delta=3) for i in range(3)]
    w=cell_weights(gs,['a','b','c','d','e'],{(128,3):1})
    assert len(w)==18
    assert np.isclose(w[('g0','Luby')],-1/3)
    assert np.isclose(w[('g0','a')],1/15)
    assert np.isclose(sum(w.values()),0.)
    assert np.isclose(sum(v*v for v in w.values()),.4)

def test_factorial_interaction_is_difference_of_differences():
    specs=contrast_specs(NS,DS)
    assert len(specs)==50
    assert sum(kind=='interaction' for kind,name,c in specs)==10
    for kind,name,c in specs:
        if kind=='interaction':
            assert sum(c.values())==0
            # Additive risk-difference surface has exactly zero interaction.
            assert np.isclose(sum(w*(np.log2(n)+d/100) for (n,d),w in c.items()),0.,atol=1e-14)

def test_simultaneous_bounds_degenerate_and_effect():
    a=event_difference(0,500,500,500,.00001)
    assert a['tv_lower']>.95 and a['ci'][1]==1
    rows={('g','a'):dict(count=500,total=500),('g','Luby'):dict(count=0,total=500)}
    w={('g','a'):1,('g','Luby'):-1}
    b=linear_interval(rows,w,.00001,.00001)
    assert b['estimate']==1 and b['significant'] and not b['bounds_conflict']
    assert b['ci'][0]>.95
    assert event_difference(0,0,1,1,.05)['tv_lower']==0

def test_new_namespace_separates_phases_and_generators():
    values=[seed(phase,label,g) for phase in ('pilot','confirmation','mechanism') for label in ('Luby','normalized-s41000') for g in ('a','b')]
    assert len(set(values))==len(values)

def test_budget_alarm_escapes_sampler_exception_handler():
    from dcl.sizedegree.run import BudgetExceeded
    assert not issubclass(BudgetExceeded,Exception)

def test_read_only_observer_preserves_frozen_sampler_actions():
    from dcl.globaldist.sampling import sample
    from dcl.sizedegree.mechanism import trace
    from dcl.sizedegree.common import specs
    from dcl.paperstudy.training import load_model
    g=regular(8,3,0,'test')
    model,_=load_model(specs()[0]['checkpoint'],'cpu')
    for network,label in [(None,'Luby'),(model,'normalized-s41000')]:
        a=sample(g,network,label,'factorial-v1/mechanism',8)
        b=trace(g,network,label,8)
        for key in ('colors','status','seeds'):assert np.array_equal(a[key],b[key])
        assert b['observer_errors'][:,:,1].sum()==0
        assert np.all(b['commit_counts'].sum((0,2))==g['n'])

def test_cutoffs_are_separate_and_not_coloring_observations(monkeypatch,tmp_path):
    import dcl.sizedegree.analysis as a
    from dcl.globaldist.sampling import sample
    g=regular(8,3,0,'test');x=sample(g,None,'Luby','test-conditional',30)
    y={k:v.copy() for k,v in x.items()};y['status'][0]=2;y['colors'][0,0]=-1
    s=dict(label='model',treatment='normalized',seed=41000,sha256='unchanged')
    monkeypatch.setattr(a,'OUT',tmp_path)
    monkeypatch.setattr(a,'verify_frozen',lambda:dict(test_runs_per_generator=20,permutations=99,total_comparisons=1))
    monkeypatch.setattr(a,'graphs',lambda:[g]);monkeypatch.setattr(a,'specs',lambda:[s])
    monkeypatch.setattr(a,'bank',lambda who,gid:x if who=='Luby' else y)
    r=a.comparison((g['id'],'model'))
    assert r['m_Luby']==20 and r['m_model']==19
    assert r['status_model']==[19,0,1,0]
    assert 'coarsened_unconditional' in r and 'status_test' in r
    assert r['witness']['n_model']==10

def test_probe_accepts_archived_int16_state_without_changing_state():
    from dcl.sizedegree.mechanism import quadrature
    from dcl.sizedegree.common import specs
    from dcl.paperstudy.training import load_model
    g=regular(8,3,0,'test');model,_=load_model(specs()[0]['checkpoint'],'cpu')
    state=np.full(8,-1,dtype=np.int16)
    r=quadrature(g,state,model,64)
    assert np.isclose(sum(r['q']),1) and np.all(state==-1)
