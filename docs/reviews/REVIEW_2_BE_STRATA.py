#!/usr/bin/env python3
"""Reconstruct existing BE queries and report existing-model diagnostic strata.
No training and no new random queries: original rng=17001, phases, batch=256.
"""
from pathlib import Path
from collections import Counter
import json,hashlib,importlib.util
import numpy as np
import torch
from dcl.multicf.oracles import proposal_bank
from dcl.multiteacher.programs import LINIAL, DETECT
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.training import initialize

BASE=Path('results/multiteacher-counterfactual')
spec=importlib.util.spec_from_file_location('closure_cf_audit','scripts/final-counterfactual-audit.py')
key_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(key_module)
be_commit_key=key_module.be_commit_key
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 torch.set_num_threads(1)
 records=json.loads((BASE/'training-curves.json').read_text()); rows=[]
 for seed in range(94101,94106):
  p=BASE/'diagnostics/common-state-banks/be_base_d3'/f's{seed}'/'states.npz'
  with np.load(p,allow_pickle=False) as f: bank={k:f[k] for k in f.files}
  tp=BASE/'datasets/be_base_d3'/f's{seed}'/'b3000-coverage1-alternate0/dataset.npz'
  with np.load(tp,allow_pickle=False) as f:train={k:f[k] for k in f.files}
  ta=tuple(train[f'a{i}'] for i in range(4));keys=set()
  from dcl.multiteacher.programs import KW
  for i in np.flatnonzero(train['native'][:,3]==KW):
   keys.add(be_commit_key(ta,i,train['p'],train['c']))
   for v in range(4):keys.add(be_commit_key(ta,i,train['cf_p'][:,v],train['cf_c'][:,v]))
  rr=[r for r in records if r['teacher']=='BE_BASE_D3' and r['architecture']=='normalized' and r['updates']==4000 and r['seed']==seed and r['native_budget']==3000]
  models={};counts={}
  for r in rr:
   m=initialize('BE_BASE_D3','normalized',seed).eval();assert digest(r['checkpoint'])==r['checkpoint_sha256'];m.load_state_dict(torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model']);models[r['method']]=m
  rng=np.random.default_rng(17001)
  for phase in sorted(np.unique(bank['native'][:,3])):
   if phase in (LINIAL,DETECT):continue
   positions=np.flatnonzero(bank['native'][:,3]==phase)
   for start in range(0,len(positions),256):
    use=positions[start:start+256];arrays=tuple(bank[f'a{i}'][use] for i in range(4));ps,cs,ys=proposal_bank(arrays,bank['native'][use],'BE_BASE_D3',rng);states={method:m.first(tensor_arrays(arrays),int(phase)) for method,m in models.items()}
    for v in range(4):
     pp,cc,truth=ps[:,v],cs[:,v],ys[:,v]>.5;eligible=arrays[0][:,0,18]>=0;active=pp[:,0]
     unseen=np.array([be_commit_key(arrays,i,pp,cc) not in keys for i in range(len(use))])
     masks={'all':np.ones(len(use),bool),'ineligible':~eligible,'eligible_silent':eligible&~active,'eligible_active':eligible&active}
     masks.update(unseen_types_all=unseen,unseen_types_positive=unseen&truth,unseen_types_negative=unseen&~truth,unseen_types_eligible_active=unseen&eligible&active)
     predictions={method:m.commit_for(states[method],torch.from_numpy(pp),torch.from_numpy(cc)).numpy()>0 for method,m in models.items()};predictions['always_reject']=np.zeros(len(use),bool);predictions['eligibility_only']=eligible&active
     for method,pred in predictions.items():
      for group,at in masks.items():
       ctr=counts.setdefault((method,group),Counter());ctr['total']+=int(at.sum());ctr['positive_truth']+=int((truth&at).sum());ctr['true_positive']+=int((pred&truth&at).sum());ctr['true_negative']+=int((~pred&~truth&at).sum());ctr['false_positive']+=int((pred&~truth&at).sum());ctr['false_negative']+=int((~pred&truth&at).sum())
  for (method,group),c in counts.items():
   c=dict(c);c['accuracy']=(c['true_positive']+c['true_negative'])/c['total'] if c['total'] else None
   pos=c['positive_truth'];neg=c['total']-pos;c['tpr']=c['true_positive']/pos if pos else None;c['tnr']=c['true_negative']/neg if neg else None;c['balanced_accuracy']=(c['tpr']+c['tnr'])/2 if pos and neg else None
   rows.append(dict(seed=seed,method=method,group=group,**c))
  for method in models:
   old=json.loads((BASE/f'diagnostics/paired-u004000-{method}/be_base_d3/normalized'/f's{seed}.json').read_text())['components'];c=counts[method,'all'];assert c['true_positive']+c['true_negative']==old['counterfactual_correct'];assert c['false_positive']==old['counterfactual_false_positive'];assert c['false_negative']==old['counterfactual_false_negative']
  print(seed,[(r['method'],r['group'],r['accuracy'],r['total']) for r in rows if r['seed']==seed and r['group'] in ('all','eligible_active')],flush=True)
 out=Path('results/final-closure/review-ml-be-strata.json');out.write_text(json.dumps(dict(rows=rows,script_sha256=digest(__file__),scope='Existing diagnostics only, observational posthoc stratification; untrained baselines are declared descriptive rule controls. Eligibility-only tests root permission/bound and participation but ignores occupied options. No new scientific model or query was acquired.'),indent=2)+'\n')
if __name__=='__main__':
 with torch.inference_mode():main()
