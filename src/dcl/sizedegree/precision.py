"""Conservative finite-sample guarantees for the registered event contrasts.

These are independent-run concentration calculations, not power estimates for
omnibus V2 or for unknown alternatives in complete-coloring TV.
"""
import numpy as np
import pandas as pd
from .common import *

def main():
    f=verify_frozen();alpha=.025/f['factorial_contrasts'];rows=[]
    norms={'architecture_cell':.4,'checkpoint_cell':2/3,
           'architecture_adjacent_change':.8,'architecture_interaction':1.6}
    for m in (100,250,500,1000,2000):
        for name,norm in norms.items():
            S=norm/m;R=np.sqrt(S*np.log(2/alpha)/2)
            for power in (.8,.9):
                delta=R+np.sqrt(S*np.log(1/(1-power))/2)
                rows.append(dict(m_valid_per_bank=m,contrast=name,confidence_radius=float(R),
                                 target_power=power,sufficient_absolute_effect=float(delta),
                                 squared_bank_weight_sum=norm,alpha_per_contrast=alpha,
                                 scope='Sufficient concentration guarantee for a fixed registered Bernoulli contrast; conditional on valid counts; not omnibus power'))
    pd.DataFrame(rows).to_csv(OUT/'event-precision-and-power-guarantees.csv',index=False)
    atomic_json(OUT/'event-precision-and-power-guarantees.json',rows)
    print(pd.DataFrame(rows)[lambda a:(a['m_valid_per_bank']==500)&(a['target_power']==.8)].to_string(index=False),flush=True)

if __name__=='__main__':main()
