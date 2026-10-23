from dcl.sizedegree.intervention_witnesses import effect_bounds

def test_strong_independent_witness_and_close_coupling_imply_effect():
    r=effect_bounds(0,125,125,125,0,1000,1000,1000,.05/360)
    assert r['original_witness_tv_lower']>.8
    assert r['conditional_tv_upper']<.025
    assert r['intervention_tv_effect_lower']>.78

def test_loose_coupling_or_absent_original_holdout_does_not_prove_effect():
    r=effect_bounds(0,125,125,125,1000,1000,1000,1000,.05/360)
    assert r['conditional_tv_upper']==1 and r['intervention_tv_effect_lower']==0
    r=effect_bounds(0,0,125,125,0,1000,1000,1000,.05/360)
    assert r['original_witness_tv_lower']==0 and r['intervention_tv_effect_lower']==0

def test_fixed_event_can_detect_effect_despite_loose_path_coupling():
    r=effect_bounds(0,125,125,125,1000,1000,1000,1000,.05/360,ki=0,ni=1000)
    assert r['conditional_tv_upper']==1 and r['triangle_effect_lower']==0
    assert r['direct_effect_lower']>.9
    assert r['intervention_tv_effect_lower']==r['direct_effect_lower']
