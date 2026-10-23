"""Read-only mechanistic diagnostics for docs/UCGNN_MODELOS_EXPLICADOS.md.

Run inside codex-disco with the study Python. No optimizer steps, checkpoint
writes or cohort changes. Outputs are a separate explanatory diagnostic cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def inventory(repo, out, device):
    from dcl.universal.train import load_model, initialize
    from dcl.universal.config import UConfig
    from dcl.study.config import digest
    from dcl.extreme.common import weights_hash
    from dcl.m4_models import M4B
    models, saved, paths, rows, initial = {}, {}, {}, [], {}
    for name in ('E1', 'E2', 'E3', 'A1', 'C1', 'E2F', 'E3F'):
        path = next((repo/'data/ucgnn/runs').glob(f'{name.lower()}-*-s00-*/epoch-120.pt'))
        model, value = load_model(path, device)
        models[name], saved[name], paths[name] = model, value, path
        config = UConfig.from_dict(value['config'])
        init, *_ = initialize(config, 'cpu')
        initial[name] = init.state_dict()
        assert digest({k:v.detach().cpu().tolist() for k,v in initial[name].items()}) == value['initial_state_hash']
        manifest = json.loads(path.with_name('data_manifest.json').read_text())
        last = value['history'][-1]
        rows.append({'model': name, 'checkpoint': str(path), 'sha256': sha(path), 'epoch': value['epoch'],
                     'parameters': sum(p.numel() for p in model.parameters()), 'weights_sha256': weights_hash(model),
                     'initial_state_hash': value['initial_state_hash'], 'config': value['config'],
                     'training_rounds': manifest['training_rounds'], 'trajectories': manifest['training_trajectories'],
                     'updates': value['updates'], 'validation_rounds': manifest['validation_rounds'],
                     'final_nodes_seen': value['nodes_seen'],
                     'modules': {n: {'parameters': sum(p.numel() for p in m.parameters()),
                                     'tensors': {k: list(v.shape) for k,v in m.state_dict().items()}}
                                 for n,m in model.named_children()}})
    comparisons = []
    for other in ('C1', 'E2', 'E3', 'A1'):
        for module in models['E1']._modules:
            a = models['E1'].get_submodule(module).state_dict()
            b = models[other].get_submodule(module).state_dict()
            shape = all(a[k].shape == b[k].shape for k in a)
            ia = {k:v for k,v in initial['E1'].items() if k.startswith(module+'.')}
            ib = initial[other]
            comparisons.append({'reference':'E1','other':other,'module':module,'same_shapes':shape,
                                'initial_exact_equal': all(k in ib and v.shape == ib[k].shape and torch.equal(v,ib[k]) for k,v in ia.items()),
                                'trained_exact_equal': shape and all(torch.equal(a[k], b[k]) for k in a),
                                'trained_max_abs_difference': max(float((a[k]-b[k]).abs().max()) for k in a) if shape else None})
    pd.DataFrame(comparisons).to_csv(out/'checkpoint-comparisons.csv',index=False)
    specialists=[]
    for degree in (3,5,8,10):
        for directory in (repo/'data/m6study-release/runs').glob(f'd{degree}-regular-counterfactual-s00-*'):
            config=json.loads((directory/'config.json').read_text())
            if config['graphs']!=800 or config['tapes']!=3 or config['epochs']!=120 or config['updates_per_epoch']: continue
            path=directory/'epoch-120.pt'
            value=torch.load(path,map_location='cpu',weights_only=False)
            model=M4B(num_colors=degree+1)
            model.load_state_dict(value['model_state_dict'])
            specialists.append({'degree':degree,'checkpoint':str(path),'sha256':sha(path),
                                'parameters':sum(p.numel() for p in model.parameters()),'config':config,
                                'tensor_shapes':{k:list(v.shape) for k,v in model.state_dict().items()}})
    save_json(out/'model-inventory.json',{'universal':rows,'specialists':specialists})
    # Availability has at most eight neighbor-type sets at a proposed color.
    table=[]
    types=torch.tensor([[1,0],[0,0],[0,1]],dtype=torch.float32,device=device)
    for name in ('E1','A1'):
        model=models[name]
        with torch.inference_mode():
            messages=model.message1(types)
            for mask in range(8):
                ids=[j for j in range(3) if mask&(1<<j)]
                received=messages[ids].amax(0) if ids else torch.zeros(32,device=device)
                h=model.state1(torch.cat((torch.ones(1,device=device),received)))
                alpha=float(torch.sigmoid(model.availability(h)).squeeze())
                table.append({'model':name,'mask':mask,'has_uncolored':bool(mask&1),
                              'has_other_fixed_color':bool(mask&2),'has_same_fixed_color':bool(mask&4),
                              'available':not bool(mask&4),'alpha':alpha})
    pd.DataFrame(table).to_csv(out/'availability-type-sets.csv',index=False)
    # Demonstrate loss routing on one teacher trajectory without optimizer.step.
    from dcl.universal.data import collect_examples, make_batch
    from dcl.universal.rules import exact_commit_labels, make_counterfactual_proposals
    import torch.nn.functional as F
    model=models['E1']; config=UConfig.from_dict(saved['E1']['config'])
    examples,_=collect_examples(replace(config,graphs_per_degree=(1,),tapes=1), 'train')
    batch=make_batch(examples, device); prediction=model(*batch.inputs()); mask=batch.mask
    colors=batch.colors.cpu().numpy(); edges=batch.edge_index.cpu().numpy()
    own=exact_commit_labels(colors,edges,prediction.participate.cpu().numpy(),prediction.candidate.cpu().numpy())
    cf_p,cf_c=make_counterfactual_proposals(colors,edges,batch.palette.cpu().numpy(),batch.y_participate.cpu().numpy()>.5,
                                          batch.y_offer.cpu().numpy(),np.random.default_rng(671))
    target=exact_commit_labels(colors,edges,cf_p,cf_c)
    cf=model.commit_logits(batch.colors,batch.edge_index,batch.palette,torch.tensor(cf_p,device=device),torch.tensor(cf_c,device=device))
    losses={'PARTICIPATE':F.binary_cross_entropy_with_logits(prediction.participation_logit[mask],batch.y_participate[mask]),
            'CANDIDATE':F.cross_entropy(prediction.candidate_logits[mask],batch.y_offer[mask]),
            'COMMIT_OWN':F.binary_cross_entropy_with_logits(prediction.commit_logit[mask],torch.tensor(own,dtype=torch.float32,device=device)[mask]),
            'COMMIT_CF':F.binary_cross_entropy_with_logits(cf[mask],torch.tensor(target,dtype=torch.float32,device=device)[mask])}
    route=[]; params=list(model.named_parameters())
    for loss_name,loss in losses.items():
        grads=torch.autograd.grad(loss,[p for _,p in params],retain_graph=True,allow_unused=True)
        for module in model._modules:
            selected=[g for (name,_),g in zip(params,grads) if name.startswith(module+'.') and g is not None]
            route.append({'loss':loss_name,'module':module,'has_autograd_path':bool(selected),
                          'gradient_l1':sum(float(g.abs().sum()) for g in selected)})
    pd.DataFrame(route).to_csv(out/'gradient-routing.csv',index=False)
    from dcl.extreme.diagnostics import commit_presence_table
    presence=[]
    for name in ('E1','C1'):
        for row in commit_presence_table(models[name]):
            presence.append({'model':name,**row})
    pd.DataFrame(presence).to_csv(out/'commit-presence-reproduced.csv',index=False)
    for name,model in models.items():
        assert weights_hash(model)==next(r['weights_sha256'] for r in rows if r['model']==name)
    return models,paths


def selection(models, extreme, out, device):
    from dcl.extreme.diagnostics import learned_center, choices_from_center, rc_grid
    from dcl.extreme.common import weights_hash
    rows=[]; summary=[]; dense={}; continuous=[]
    for name, palette in [('E1',4),('E1',11),('E1',101),('E1',201),('E1',501),('A1',101)]:
        model=models[name]; colors=np.full(palette,-1,np.int64)
        state=learned_center(model,colors,palette)
        grid,_=rc_grid(palette); choices=choices_from_center(model,state,grid)
        target=np.minimum((grid.astype(np.float64)*palette).astype(int),palette-1)
        bulk=np.isin(grid,((np.arange(1001)+.5)/1001).astype(np.float32))
        original=extreme/'diagnostics/cuda'/next(p.name for p in (extreme/'diagnostics/cuda').glob(f'{name.lower()}-*-s00-*'))/f'selection-raw/K{palette}-k{palette}-LOW_AVAILABLE.npz'
        with np.load(original) as cached:
            assert np.array_equal(cached['rc'],grid)
            mismatches=int((cached['choices']!=choices).sum())
        for r,t,c,b in zip(grid,target,choices,bulk):
            rows.append({'model':name,'palette':palette,'rc':float(r),'teacher':int(t),'candidate':int(c),'bulk':bool(b),'available':True})
        dense_rc=((np.arange(20001)+.5)/20001).astype(np.float32)
        dc=choices_from_center(model,state,dense_rc); dt=np.floor(dense_rc.astype(np.float64)*palette).astype(int)
        hist=np.bincount(dc,minlength=palette); q=hist/len(dc); collision=float(q@q)
        dense[(name,palette)]=(dense_rc,dc,q)
        summary.append({'model':name,'palette':palette,'bulk_draws':int(bulk.sum()),'bulk_correct':int((choices[bulk]==target[bulk]).sum()),
                        'original_choices_mismatched':mismatches,'distinct_bulk_colors':len(np.unique(choices[bulk])),
                        'dense_draws':len(dc),'dense_correct':int((dc==dt).sum()),'dense_colors':int((hist>0).sum()),
                        'mean_rank_error':float(np.mean(dc-dt)),'max_absolute_rank_error':int(np.max(np.abs(dc-dt))),
                        'mean_normalized_rank_error':float(np.mean((dc-dt)/palette)),
                        'monotone':bool(np.all(np.diff(dc)>=0)),'pair_collision':collision,'uniform_pair_collision':1/palette,
                        'effective_colors':1/collision,'largest_color_probability':float(q.max()),
                        'alpha':float(state[1][0]),'k_hat_raw':float(state[2])})
        pd.DataFrame({'color':np.arange(palette),'probability':q,'uniform_probability':1/palette}).to_csv(out/f'proposal-law-{name}-K{palette}.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'selection-map.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'selection-summary.csv',index=False)
    # Continuous score on the same all-uncolored h and alpha. A one-hidden-
    # layer ReLU network is piecewise affine in x=lo. Its maxima on [0,1-w]
    # occur at the endpoints or one of 64 activation breakpoints.
    model=models['E1']; state=learned_center(model,np.full(201,-1,np.int64),201)
    linear=model.scorer[0]; W=linear.weight.detach().cpu().double().numpy(); b=linear.bias.detach().cpu().double().numpy()
    v=model.scorer[2].weight.detach().cpu().double().numpy().ravel()
    h=state[0][0].cpu().double().numpy(); a=float(state[1][0])
    B=W[:,:32]@h+W[:,32]*a+b; A=W[:,33]+W[:,34]; R=W[:,35]; C=W[:,34]
    rr=((np.arange(1001)+.5)/1001)
    for r in rr:
        row={'rc':float(r)}
        for label,k in [('limit',None),('K201',201),('K501',501)]:
            width=0 if k is None else 1/k
            roots=-(B+R*r+C*width)/np.where(A==0,np.nan,A)
            x=np.concatenate(([0,1-width],roots)); x=np.clip(np.nan_to_num(x),0,1-width)
            scores=np.maximum(B[None,:]+x[:,None]*A[None,:]+r*R[None,:]+width*C[None,:],0)@v
            j=int(scores.argmax()); q=float(x[j]); unit=j-2
            row.update({f'x_{label}':q,f'unit_{label}':unit})
            if j>=2:
                row.update({f'rc_slope_{label}':float(-R[unit]/A[unit]),f'intercept_{label}':float(-B[unit]/A[unit]),
                            f'width_slope_{label}':float(-C[unit]/A[unit])})
        continuous.append(row)
    pd.DataFrame(continuous).to_csv(out/'continuous-score-maxima.csv',index=False)
    # How often does the controlled K=k=201 state occur in actual E1 rollouts?
    counts=Counter(); by_seed=defaultdict(Counter)
    for path in sorted((extreme/'results').glob('*.json')):
        r=json.loads(path.read_text())
        if r['experiment']!='E1' or r['graph']['degree']!=200: continue
        destination=by_seed[r['training_seed']]
        counts.update(graph_executions=1,all_uncolored_decisions=r['states'])
        destination.update(graph_executions=1,all_uncolored_decisions=r['states'])
        by=r['by_k'].get('201',{})
        for key in ('states','candidate_correct','forbidden','executed','unsafe'):
            counts[key]+=by.get(key,0)
            destination[key]+=by.get(key,0)
    save_json(out/'autonomous-K201-frequency.json',{'all_seeds':dict(counts),
              'by_training_seed':{str(k):dict(v) for k,v in sorted(by_seed.items())}})
    return dense


def stats(colors, edges, p, cand, commit, palette):
    from dcl.universal.rules import exact_commit_labels, conflict_flags, available_matrix
    u=colors<0; exact=exact_commit_labels(colors,edges,p,cand); fixed, simultaneous=conflict_flags(colors,edges,p,cand)
    executed=u&p&(cand>=0)&commit
    available=available_matrix(colors,edges,palette)
    hist=np.bincount(cand[u&p],minlength=palette)
    q=hist/max(1,hist.sum())
    return {'pending_before':int(u.sum()),'participants':int((p&u).sum()),'distinct_colors':int((hist>0).sum()),
            'largest_proposal_count':int(hist.max()),'pair_collision':float(q@q),
            'fixed_conflicts':int((u&p&fixed).sum()),'simultaneous_conflicts':int((u&p&simultaneous).sum()),
            'available_candidates':int((u&available[np.arange(len(u)),np.maximum(cand,0)]).sum()),
            'exact_safe_proposals':int(exact.sum()),'commits':int(executed.sum()),
            'unsafe_commits':int((executed&~exact).sum()),'pending_after':int(u.sum()-executed.sum())},hist,executed


@torch.inference_mode()
def traces(models, repo, extreme, out, device, dense):
    evidence=json.loads((repo/'results/ucgnn/2026-10-20-final/failure-reproduction.json').read_text())
    rows=[]; comparisons=[]; checks=[]
    for entry in evidence['rows']:
        assert sha(entry['source_record']) == entry['source_record_sha256']
        graph=entry['graph']; label=f"{graph['family']}-n{graph['n']}"; n=graph['n']; palette=graph['palette']
        with np.load(extreme/'graphs'/f"{graph['graph_id']}.npz") as cache: edges=cache['edge_index']
        et=torch.tensor(edges,device=device); kt=torch.full((n,),palette,dtype=torch.long,device=device)
        for trajectory in ('A1','E1'):
            colors=np.full(n,-1,np.int64); rng=np.random.default_rng(graph['tape_seed']); arrays=defaultdict(list)
            for round_i in range(1,257):
                rp,rc=rng.random(n),rng.random(n)
                args=(torch.tensor(colors,device=device),torch.tensor(rp,dtype=torch.float32,device=device),
                      torch.tensor(rc,dtype=torch.float32,device=device),et,kt)
                policies=('A1','E1') if trajectory=='A1' else ('E1',)
                outputs={}
                for policy in policies:
                    prediction=models[policy](*args)
                    p=prediction.participate.cpu().numpy(); cand=prediction.candidate.cpu().numpy(); commit=(prediction.commit_logit>0).cpu().numpy()
                    values,hist,executed=stats(colors,edges,p,cand,commit,palette)
                    outputs[policy]=(p,cand,commit,executed,hist)
                    rows.append({'case':label,'trajectory':trajectory,'policy':policy,'round':round_i,**values})
                    if trajectory=='A1':
                        shared=(rp<.5)&(colors<0)
                        cv,_,_=stats(colors,edges,shared,cand,np.zeros(n,bool),palette)
                        comparisons.append({'case':label,'round':round_i,'policy':policy,'participation':'common_rp_lt_0.5',**cv})
                p,cand,commit,executed,hist=outputs[trajectory]
                for key,val in [('colors_before',colors.copy()),('rp',rp),('rc',rc),('participate',p),('candidate',cand),('commit',commit),('histogram',hist)]:
                    arrays[key].append(val)
                colors=colors.copy(); colors[executed]=cand[executed]
                if not (colors<0).any(): break
            np.savez_compressed(out/f'trace-{trajectory}-{label}.npz',**{k:np.asarray(v) for k,v in arrays.items()},colors_final=colors)
            if trajectory=='A1':
                same=hashlib.sha256(colors.tobytes()).hexdigest()==entry['final_colors_sha256']
                assert same
                checks.append({'case':label,'saved_colors_reproduced':same,'rounds':round_i,'pending':int((colors<0).sum()),
                               'graph':graph,'source_record':entry['source_record'],
                               'source_record_sha256':entry['source_record_sha256']})
            else:
                assert not (colors<0).any() and not np.any(colors[edges[0]]==colors[edges[1]])
            print(f'trace {trajectory} {label} rounds={round_i} pending={(colors<0).sum()}',flush=True)
        # Initial-round expectation: iid draws, common all-uncolored local state.
        for policy in ('A1','E1'):
            q=dense[(policy,101)][2]
            rp_grid=torch.linspace(0,1,100001,device=device)
            p=float((models[policy].participation(torch.stack((rp_grid,torch.ones_like(rp_grid)),-1)).squeeze(-1)>0).float().mean())
            degree=np.bincount(edges[1],minlength=n)
            expected=float(sum(p*np.sum(q*(1-p*q)**d) for d in degree))
            checks.append({'case':label,'policy':policy,'initial_expected_safe_commits':expected,'participation_probability_grid':p})
    pd.DataFrame(rows).to_csv(out/'round-traces.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'matched-state-candidate-comparison.csv',index=False)
    save_json(out/'trace-checks.json',checks)


@torch.inference_mode()
def examples(model, out, device):
    from dcl.universal.rules import exact_commit_labels, teacher_offer
    for palette in (4,11):
        colors=np.full(palette,-1,np.int64)
        if palette==4: colors[1:3]=[0,2]
        else: colors[1:4]=[0,2,4]
        n=len(colors); edges=np.array([(0,j) for j in range(1,n)]+[(j,0) for j in range(1,n)]).T
        rp=np.full(n,.9); rc=np.linspace(.05,.95,n); rp[0]=.2;rc[0]=.63;rp[-1]=.1;rc[-1]=.1
        captured={}; handles=[]
        for name,module in model.named_children():
            def capture(module,inputs,outputs,name=name):
                captured[name+'_input']=inputs[0].detach().cpu().numpy(); captured[name+'_output']=outputs.detach().cpu().numpy()
            handles.append(module.register_forward_hook(capture))
        pred=model(torch.tensor(colors,device=device),torch.tensor(rp,dtype=torch.float32,device=device),
                   torch.tensor(rc,dtype=torch.float32,device=device),torch.tensor(edges,device=device),torch.full((n,),palette,device=device))
        for handle in handles: handle.remove()
        p=pred.participate.cpu().numpy(); c=pred.candidate.cpu().numpy(); z=(pred.commit_logit>0).cpu().numpy()
        t,k=teacher_offer(colors,rc,edges,palette); exact=exact_commit_labels(colors,edges,p,c)
        updated=colors.copy(); executed=(colors<0)&p&z; updated[executed]=c[executed]
        slots=[]
        for color in range(palette):
            slots.append({'color':color,'alpha':float(pred.availability[0,color]),'prefix':float(pred.availability[0,:color].sum()),
                          'lo':float(pred.lo[0,color]),'hi':float(pred.hi[0,color]),'score':float(pred.candidate_logits[0,color]),
                          **{f'h{j}':float(captured['state1_output'][0,color,j]) for j in range(4)}})
        pd.DataFrame(slots).to_csv(out/f'example-K{palette}-slots.csv',index=False)
        save_json(out/f'example-K{palette}.json',{'colors_before':colors.tolist(),'rp':rp.tolist(),'rc':rc.tolist(),'palette':palette,
                  'available_center':[j for j in range(palette) if j not in colors[1:]],'k_hat_center':float(pred.k_hat[0]),
                  'teacher_candidate':t.tolist(),'available_count':k.tolist(),'participation_logit':pred.participation_logit.cpu().tolist(),
                  'participate':p.tolist(),'candidate':c.tolist(),'commit_logit':pred.commit_logit.cpu().tolist(),
                  'exact_safe':exact.tolist(),'colors_after':updated.tolist(),'tensor_shapes':{key:list(v.shape) for key,v in captured.items()}})
        np.savez_compressed(out/f'example-K{palette}-tensors.npz',**captured)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('all','inventory','selection','traces','examples'),default='all')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(); repo=Path(__file__).resolve().parents[1]
    extreme_repo=repo.parent/'ucgnn-extreme-degree-study'; extreme=extreme_repo/'data/ucgnn-extreme'
    sys.path.insert(0,str(extreme_repo/'src'))
    from dcl.study.storage import training_lock
    from dcl.extreme.common import weights_hash, now
    from dcl.universal.train import load_model
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    with training_lock(extreme):
        models,paths=inventory(repo,out,args.device)
        before={name:weights_hash(model) for name,model in models.items()}
        dense=None
        if args.stage in ('all','selection','traces'):
            dense=selection(models,extreme,out,args.device)
            print('selection diagnostics complete',flush=True)
        if args.stage in ('all','traces'): traces(models,repo,extreme,out,args.device,dense)
        if args.stage in ('all','examples'): examples(models['E1'],out,args.device)
        assert before=={name:weights_hash(model) for name,model in models.items()}
        assert all(sha(paths[name])==next(r['sha256'] for r in json.loads((out/'model-inventory.json').read_text())['universal'] if r['model']==name) for name in models)
    save_json(out/'diagnostic-run.json',{'utc':now(),'device':args.device,'stage':args.stage,'script_sha256':sha(__file__),
               'weights_unchanged':True,'checkpoints_unchanged':True,'checkpoint_weights_sha256':before,
               'source_hashes':{str(p.relative_to(repo)):sha(p) for p in sorted((repo/'src/dcl/universal').glob('*.py'))}})


if __name__=='__main__': main()
