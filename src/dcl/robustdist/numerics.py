"""Conservative reporting when an exact classifier tail underflows in float."""
import math

def classifier_probability(result):
    p=float(result['p'])
    if p>0:return dict(p=p,p_kind='exact_conditional_hypergeometric',raw_float_p=p)
    m=int(result['evaluation_per_class']);a=float(result['balanced_accuracy'])
    # For balanced classes, correct_count=m-K+2TP and TP~Hypergeom(2m,K,m).
    # Hoeffding without replacement gives P(TP-K/2>=m(a-.5))
    # <=exp(-2m(a-.5)^2), uniformly over the fixed prediction count K.
    gap=max(0.,a-.5-1e-12);log_bound=-2*m*gap*gap
    bound=math.exp(log_bound)
    if bound==0:bound=float.fromhex('0x1.0p-1022')
    return dict(p=bound,p_kind='conservative_hypergeometric_Hoeffding_upper_bound',raw_float_p=0.,log_p_upper_bound=log_bound)
