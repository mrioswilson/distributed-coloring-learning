"""Derived analyses of completed CSV/array data; no policy selection or fitting."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .config import ROOT, RESULTS
from .progress import expected_confirmations
from .report import write_csv


def analyze(frame,law,b3):
    curve=pd.DataFrame(b3)
    curve['treatment']=curve.model.str.rsplit('-s',n=1).str[0]
    classified=curve.groupby(['treatment','state','k','m']).agg(networks=('model','nunique'),
        balanced_accuracy=('balanced_accuracy','mean'),accuracy_min=('balanced_accuracy','min'),accuracy_max=('balanced_accuracy','max'),
        power=('power','mean'),power_min=('power','min'),power_max=('power','max'),fpr=('fpr','mean')).reset_index()
    classified.to_csv(RESULTS/'classifier-summary.csv',index=False)
    wide=frame.pivot(index=['model','graph_id','tape'],columns='regime',values='status')
    rows=[]
    for treatment in frame.treatment.unique():
        prefix='exact-teacher' if treatment=='teacher' else treatment+'-'
        part=wide[wide.index.get_level_values('model').str.startswith(prefix)]
        rows.append(dict(treatment=treatment,global_only_success=int(((part.GLOBAL=='success')&(part.LOCAL!='success')).sum()),
                         local_only_success=int(((part.LOCAL=='success')&(part.GLOBAL!='success')).sum()),
                         both_success=int(((part.GLOBAL=='success')&(part.LOCAL=='success')).sum()),
                         both_failed=int(((part.GLOBAL!='success')&(part.LOCAL!='success')).sum())))
    write_csv(RESULTS/'paired-success-discordances.csv',rows)
    state_map={s['id']:s for name in ('state-manifest.json','autonomous-state-manifest.json','historical-state-manifest.json')
               for s in json.loads((RESULTS/name).read_text())}
    bad=[]
    for _,r in law[law.forbidden_mass>0].iterrows():
        bad.append(dict(model=r.model,state=r.state,k=int(r.k),pattern=r.pattern,forbidden_mass=float(r.forbidden_mass),
                        fidelity=float(r.fidelity),kind=r.kind))
        if r.model.startswith(('deepsets-','transformer-')):
            source=ROOT/'distributions'/r.model/f'{r.state}.json'
            dest=RESULTS/'candidate-counterexamples'/r.model/f'{r.state}.json';dest.parent.mkdir(parents=True,exist_ok=True)
            record=json.loads(source.read_text())
            dest.write_text(json.dumps(dict(observation=record,state=state_map[r.state],
                interpretation='Proper realizable state; forbidden proposals, not an observed unsafe COMMIT'),indent=2,sort_keys=True))
            shutil.copy2(source.with_suffix('.npz'),dest.with_suffix('.npz'))
    write_csv(RESULTS/'forbidden-candidate-states.csv',bad)
    safety=law.groupby('treatment').agg(model_states=('model','size'),networks=('model','nunique'),
                  observed_forbidden_states=('forbidden_mass',lambda x:int((x>0).sum())),max_forbidden_mass=('forbidden_mass','max')).reset_index()
    safety.to_csv(RESULTS/'candidate-safety-summary.csv',index=False)
    progress=[]
    for row in json.loads((RESULTS/'controlled-progress.json').read_text())['heterogeneous']:
        s=state_map[row['state']];colors=np.array(s['colors']);edges=np.array(s['edges'])
        x=np.load(ROOT/'controlled-progress'/f'{row["model"]}-{row["state"]}.npz')
        q,p,available=x['q'],x['p'],x['available'];pending=colors<0
        target=available/available.sum(1,keepdims=True)
        tv=.5*np.abs(q-target).sum(1)
        src,dst=edges;use=pending[src]&pending[dst]&(src<dst)
        collision=float(np.sum(p[src[use]]*p[dst[use]]*np.sum(q[src[use]]*q[dst[use]],axis=1)))
        progress.append(dict(**row,mean_candidate_tv=float(tv[pending].mean()),
                             mean_candidate_collision=float(np.sum(q[pending]**2,axis=1).mean()),
                             expected_adjacent_participating_equal_pairs=collision,
                             note='Fixed whole state; estimated node marginals, not autonomous-state-mixture histograms'))
    write_csv(RESULTS/'heterogeneous-distribution-progress.csv',progress)
    return classified
