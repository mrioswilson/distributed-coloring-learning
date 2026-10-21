"""CPU-only invariance and legal local-message certificate checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dcl.paperstudy.training import load_model
from dcl.study.storage import atomic_json, file_hash
from dcl.universal.rules import exact_commit_labels, teacher_offer
from .adapter import LocalPaletteAdapter, converted_uniform
from .cohort import edge_array, make_graph
from .config import PREVIOUS, RESULTS, seed


def local_type_witness(mask):
    """A proper partial local-list graph realizing a selected-slot type set.

    The center proposes its LAST local color; low-degree neighbors need not
    possess that color. Private pending leaves legalize high-color neighbors.
    """
    ids=[i for i in range(6) if mask&(1<<i)]
    degree=len(ids);candidate=degree
    edges=[];colors=[-1];parts=[True];offers=[candidate]
    for typ in ids:
        other=0 if candidate!=0 else 1
        colors.append(other if typ==0 else candidate if typ==1 else -1)
        parts.append(typ in (4,5))
        offers.append(-1 if typ in (0,1) else candidate if typ in (3,5) else other)
        neighbor=len(colors)-1
        edges.extend(((0,neighbor),(neighbor,0)))
    for neighbor,typ in enumerate(ids,1):
        needed=candidate if typ in (1,3,5) else 1
        for _ in range(max(0,needed-1)):
            leaf=len(colors);colors.append(-1);parts.append(False);offers.append(0)
            edges.extend(((neighbor,leaf),(leaf,neighbor)))
    e=np.array(edges,dtype=np.int64).reshape(-1,2).T.copy()
    palettes=np.bincount(e[1],minlength=len(colors))+1
    return colors,parts,offers,e,palettes


def verify():
    torch.set_num_threads(1)
    specs=json.loads((RESULTS/'frozen-checkpoints.json').read_text())
    g,_=make_graph('HETEROGENEOUS',15,1,pilot=True)
    edges=edge_array(g);palette=np.array([g.degree(i)+1 for i in g]);n=len(g)
    colors=np.full(n,-1,dtype=np.int64)
    rng=np.random.default_rng(seed('verification'))
    for _ in range(2):
        c,_=teacher_offer(colors,rng.random(n),edges,palette)
        p=(rng.random(n)<.5)&(colors<0)
        fixed=exact_commit_labels(colors,edges,p,c);colors[fixed]=c[fixed]
    permutation=rng.permutation(n);inverse=np.argsort(permutation)
    rp=torch.tensor(converted_uniform(rng.random(n)));rc=torch.tensor(converted_uniform(rng.random(n)))
    ct=torch.tensor(colors);et=torch.tensor(edges);pt=torch.tensor(palette)
    rows=[];certificates=[];witnesses=[]
    for spec in specs:
        model,_=load_model(spec['checkpoint'],'cpu');adapter=LocalPaletteAdapter(model)
        with torch.inference_mode():
            base=adapter(ct,rp,rc,et,pt)
            historical=model(ct,rp,rc,et,pt)
            padded=adapter(ct,rp,rc,et,pt,output_width=501)
            perm=adapter(ct[permutation],rp[permutation],rc[permutation],torch.tensor(inverse[edges]),pt[permutation])
            joint=adapter(torch.cat((ct,torch.tensor([-1]))),torch.cat((rp,torch.tensor([.1]))),
                          torch.cat((rc,torch.tensor([.9]))),et,torch.cat((pt,torch.tensor([201]))))
        width=int(pt.max());valid=torch.arange(width)[None,:]<pt[:,None]
        delta=float((base.candidate_logits[valid]-historical.candidate_logits[valid]).abs().max())
        margin=base.candidate_logits.topk(min(2,width),1).values
        hard_diff=base.candidate!=historical.candidate
        row=dict(model=spec['label'],historical_score_error=delta,historical_commit_error=float((base.commit_logit-historical.commit_logit).abs().max()),
                 historical_candidate_disagreements=int(hard_diff.sum()),historical_commit_sign_disagreements=int(((base.commit_logit>0)!=(historical.commit_logit>0)).sum()),
                 padding_bitwise_equal=bool(torch.equal(base.candidate_logits,padded.candidate_logits[:,:width])),
                 batch_error=float((base.candidate_logits[valid]-joint.candidate_logits[:n,:width][valid]).abs().max()),
                 permutation_error=float((base.candidate_logits[valid]-perm.candidate_logits[inverse][valid]).abs().max()),
                 batch_actions_equal=bool(torch.equal(base.candidate,joint.candidate[:n])),
                 permuted_actions_equal=bool(torch.equal(base.candidate,perm.candidate[inverse])),
                 historical_scores_within_tolerance=bool(torch.allclose(base.candidate_logits[valid],historical.candidate_logits[valid],atol=3e-5,rtol=3e-5)),
                 permutation_scores_within_tolerance=bool(torch.allclose(base.candidate_logits[valid],perm.candidate_logits[inverse][valid],atol=3e-5,rtol=3e-5)),
                 batch_scores_within_tolerance=bool(torch.allclose(base.candidate_logits[valid],joint.candidate_logits[:n,:width][valid],atol=3e-5,rtol=3e-5)))
        rows.append(row)
        if getattr(model,'reduction','max')=='max':
            oldpath=PREVIOUS/f'results/ucgnn-paper-study/certificates/final/{spec["label"]}-epoch120.json'
            cert=json.loads(oldpath.read_text())
            assert cert['checkpoint_sha256']==spec['sha256']
            certificates.append(dict(model=spec['label'],checkpoint_sha256=spec['sha256'],certificate=str(oldpath),
                                     certificate_sha256=file_hash(oldpath),real_no_false_accept=cert['real']['certified_no_false_accept'],
                                     real_exact_rule=cert['real']['certified_exact_rule'],
                                     conditional_float32_exact_rule=cert['float32_model']['certified_exact_rule'],
                                     min_real_margin=cert['real']['min_signed_margin_lower'],
                                     min_float32_margin=cert['float32_model']['min_signed_margin_lower'],
                                     float32_assumptions=cert['float32_model']['assumptions']))
            for mask in range(64):
                q,p,c,e,k=local_type_witness(mask)
                q,p,c,e,k=torch.tensor(q),torch.tensor(p),torch.tensor(c),torch.tensor(e),torch.tensor(k)
                assert bool((q<k).all()) and bool((c<k).all())
                with torch.inference_mode():
                    old=model.commit_logits(q,e,k,p,c)[0]
                    new=adapter.commit_logits(q,e,k,p,c)[0]
                valid=not(mask&(1<<1) or mask&(1<<5))
                witnesses.append(dict(model=spec['label'],mask=mask,center_k=int(k[0]),n=len(q),
                                      logit=float(new),error=float(abs(new-old)),correct=bool((float(new)>0)==valid)))
    success=all(r['historical_candidate_disagreements']==0 and r['historical_commit_sign_disagreements']==0 and
                r['padding_bitwise_equal'] and r['batch_actions_equal'] and r['permuted_actions_equal'] and
                r['historical_scores_within_tolerance'] and r['permutation_scores_within_tolerance'] and
                r['batch_scores_within_tolerance'] for r in rows)
    result=dict(passed=success,trained_networks=len(rows),invariance=rows,local_message_witnesses=witnesses,
                transferred_certificates=certificates,legal_witnesses_correct=all(r['correct'] for r in witnesses),
                note='The preregistered tolerance is atol=rtol=3e-5. Initial absolute-only checker was incorrect; its unchanged-model output is retained separately. All hard actions agreed in both audits. No progress certificate.')
    atomic_json(RESULTS/'adapter-verification.json',result)
    print(f'Trained-network invariance: {success}; {len(rows)} networks; {len(witnesses)} local-type witnesses.')
    if not success:raise RuntimeError('Action or padding invariance discrepancy; retained in adapter-verification.json')
    return result


if __name__=='__main__':verify()
