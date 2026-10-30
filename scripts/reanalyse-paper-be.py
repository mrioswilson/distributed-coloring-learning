#!/usr/bin/env python3
"""Reconstruct original BE diagnostics, adding equal-type and stratified metrics.

Original seed/RNG/chunk order is preserved. The five original count pairs must
match before anything is exported. No new training or diagnostic distribution.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from dcl.multicf.oracles import proposal_bank
from dcl.multiteacher.programs import LINIAL, DETECT, KW
from dcl.multiteacher_e1.supervision import tensor_arrays
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json, file_hash
from dcl.paperrevision.keys import commit_key, key_text
from dcl.paperrevision.experiment import confusion

BASE=Path("results/multiteacher-counterfactual")
OUT=Path("results/paper-revision-20261030")


def main():
    torch.set_num_threads(1)
    records=json.loads((BASE/"training-curves.json").read_text())
    rows=[]; types=[]
    for seed in range(94101,94106):
        dataset=BASE/"datasets/be_base_d3"/f"s{seed}"/"b3000-coverage1-alternate0"
        manifest=json.loads((dataset/"manifest.json").read_text())
        assert file_hash(dataset/"dataset.npz")==manifest["sha256"]
        with np.load(dataset/"dataset.npz",allow_pickle=False) as f: train={k:f[k] for k in f.files}
        arrays=tuple(train[f"a{j}"] for j in range(4)); trainkeys=set()
        for i in np.flatnonzero(train["native"][:,3]==KW):
            trainkeys.add(key_text(commit_key(arrays,i,train["p"],train["c"],"BE_BASE_D3")))
            for v in range(4): trainkeys.add(key_text(commit_key(arrays,i,train["cf_p"][:,v],train["cf_c"][:,v],"BE_BASE_D3")))
        bank=BASE/"diagnostics/common-state-banks/be_base_d3"/f"s{seed}"
        bmanifest=json.loads((bank/"manifest.json").read_text())
        assert file_hash(bank/"states.npz")==bmanifest["sha256"]
        with np.load(bank/"states.npz",allow_pickle=False) as f: packed={k:f[k] for k in f.files}
        selected=[r for r in records if r["teacher"]=="BE_BASE_D3" and r["architecture"]=="normalized" and r["seed"]==seed and r["updates"]==4000 and r["native_budget"]==3000]
        models={}
        for r in selected:
            assert file_hash(r["checkpoint"])==r["checkpoint_sha256"]
            m=initialize("BE_BASE_D3","normalized",seed).eval()
            m.load_state_dict(torch.load(r["checkpoint"],weights_only=False)["model"])
            models[r["method"]]=m
        predictions=defaultdict(list); truths=[]; keys=[]; groups=defaultdict(list)
        rng=np.random.default_rng(17001)
        for phase in sorted(np.unique(packed["native"][:,3])):
            if phase in (LINIAL,DETECT): continue
            positions=np.flatnonzero(packed["native"][:,3]==phase)
            for start in range(0,len(positions),256):
                use=positions[start:start+256]; arrays=tuple(packed[f"a{j}"][use] for j in range(4))
                ps,cs,ys=proposal_bank(arrays,packed["native"][use],"BE_BASE_D3",rng)
                states={arm:m.first(tensor_arrays(arrays),int(phase)) for arm,m in models.items()}
                for v in range(4):
                    p,c,y=ps[:,v],cs[:,v],ys[:,v]>.5
                    k=[key_text(commit_key(arrays,i,p,c,"BE_BASE_D3")) for i in range(len(use))]
                    eligible=(arrays[0][:,0,7]>.5)&(arrays[0][:,0,18]>=0)
                    occupied=(arrays[2][:,0,:,-4:] & arrays[3][:,0,:,None]).any(1) if arrays[2].dtype==bool else ((arrays[2][:,0,:,-4:]>.5)&arrays[3][:,0,:,None]).any(1)
                    blocked=occupied[np.arange(len(use)),c[:,0]]
                    truths.extend(y); keys.extend(k)
                    groups["seen"].extend([a in trainkeys for a in k])
                    groups["eligible"].extend(eligible); groups["active"].extend(p[:,0]); groups["blocked"].extend(blocked)
                    for arm,m in models.items():
                        predictions[arm].extend(m.commit_for(states[arm],torch.from_numpy(p),torch.from_numpy(c)).numpy()>0)
                    predictions["always_reject"].extend(np.zeros(len(y),bool))
                    predictions["eligibility_only"].extend(eligible&p[:,0])
        truth=np.asarray(truths,bool); selections={k:np.asarray(v,bool) for k,v in groups.items()}
        bytype=defaultdict(list)
        for i,k in enumerate(keys): bytype[k].append(i)
        assert all(len(set(truth[idx]))==1 for idx in bytype.values())
        masks=dict(all=np.ones(len(truth),bool),seen=selections["seen"],unseen=~selections["seen"],
                   eligible_active=selections["eligible"]&selections["active"],
                   eligible_silent=selections["eligible"]&~selections["active"],ineligible=~selections["eligible"],
                   eligible_active_occupied=selections["eligible"]&selections["active"]&selections["blocked"],
                   eligible_active_free=selections["eligible"]&selections["active"]&~selections["blocked"])
        for arm,pr in predictions.items():
            pred=np.asarray(pr,bool)
            if arm in models:
                old=json.loads((BASE/f"diagnostics/paired-u004000-{arm}/be_base_d3/normalized/s{seed}.json").read_text())
                assert int((pred==truth).sum())==old["components"]["counterfactual_correct"]
                assert len(truth)==old["components"]["counterfactual_total"]
            for name,sel in masks.items():
                if not sel.any(): continue
                macro=[]
                for k,idx in bytype.items():
                    ix=np.asarray(idx); ix=ix[sel[ix]]
                    if len(ix): macro.append(float((pred[ix]==truth[ix]).mean()))
                rows.append(dict(seed=seed,method=arm,group=name,**confusion(pred[sel],truth[sel]),
                    equal_type_accuracy=float(np.mean(macro)),distinct_types=len(macro),
                    train_types=len(trainkeys),evaluation_bank_sha256=bmanifest["sha256"]))
            for k,idx in bytype.items():
                types.append(dict(seed=seed,method=arm,key=json.loads(k),seen=k in trainkeys,
                    occurrences=len(idx),correct=int((pred[idx]==truth[idx]).sum()),truth=bool(truth[idx[0]])))
        print(seed,{a:float(np.mean(np.asarray(v)==truth)) for a,v in predictions.items()},flush=True)
    atomic_json(OUT/"be-reanalysis.json",dict(rows=rows,script_sha256=file_hash(__file__),
        scope="Exactly reconstructed historical development queries; new equal-per-sufficient-type summaries, no final-test or causal reclassification.",
        count_reproduction_all_match=True,training_updates=0))
    atomic_json(OUT/"be-type-counts.json",types)


if __name__=="__main__":
    with torch.inference_mode():main()
