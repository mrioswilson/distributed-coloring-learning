import numpy as np
from scipy.special import gammaln
from dcl.robustdist.numerics import classifier_probability

def test_classifier_underflow_gets_positive_valid_upper_bound():
    r=classifier_probability(dict(p=0.,balanced_accuracy=1.,evaluation_per_class=1000))
    exact_log_p=2*gammaln(1001)-gammaln(2001)
    assert 0<r['p']<.05 and np.log(r['p'])>=exact_log_p
    assert r['raw_float_p']==0 and 'upper_bound' in r['p_kind']

def test_nonextreme_classifier_probabilities_are_unchanged():
    assert classifier_probability(dict(p=.37))['p']==.37
    assert classifier_probability(dict(p=0.,balanced_accuracy=.5,evaluation_per_class=1000))['p']==1.
