# Independent recomputation from frozen bank arrays (runtime tarball) of: COMMIT truth labels,
# fine sufficient keys (per paper spec, Appendix "Prospective acquisition and sufficient key"),
# hash split, test-type counts, dominant mass, coarse logical overlap, programmed-gate metrics,
# training label counts by phase, base component counts.
# Feature indices (a0[...,7]=permission w, a0[...,18]=signed KW offset, a0[...,8:14]=phase one-hot,
# a2[...,-4:]=BE neighbor occupancy of queried option, a1[:,0,:,1:5]/a0[:,:,1:5]=GM own-color bits,
# a0[...,0]=GM pending, a0[...,5]=ID, a2[...,1]=GM same-slot) are taken from the encoder layout used by Pedro's code.
import numpy as np, json, sys, hashlib
from collections import Counter, defaultdict
X=sys.argv[1]; TEACH={'be_base_d3':'BE_BASE_D3','speculative_gm':'SPECULATIVE_GM'}
SALT="revision-20261030-commit-types-v1"; KW,ASSIGN,DETECT,REPAIR,LINIAL=5,1,2,3,4
def L(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def key(b,i,p,c,T):
    x,a1,rel,m=b['a0'],b['a1'],b['a2'],b['a3']
    ph=int(np.argmax(x[i,0,8:14])); q=int(c[i,0]); be=T=='BE_BASE_D3'
    u=np.ones(4) if be else np.maximum(x[i,:,0],x[i,:,7])
    own=np.zeros(4) if be else x[i,:,1+q]
    root=[int(u[0]),int(own[0]),int(p[i,0]),int(u[0]>.5)]
    nb=set()
    for j in np.flatnonzero(m[i,0]):
        v=j+1
        occ=rel[i,0,j,-4+q] if ph==KW else own[v]
        nb.add((int(u[v]),int(occ),int(p[i,v]),int(u[v]>.5 and c[i,v]==q),
                0 if be else int(np.sign(x[i,v,5]-x[i,0,5])), 0 if be else int(rel[i,0,j,1])))
    k=[T,ph,int(x[i,0,7]),int(np.sign(x[i,0,18])) if ph==KW else 0,root,[list(t) for t in sorted(nb)]]
    return json.dumps(k,separators=(',',':'))
def held(ks): return int.from_bytes(hashlib.sha256((SALT+ks).encode()).digest()[:8],'big')%5==0
def facts(b,T,pre=''):
    x,rel,m=b['a0'],b['a2'],b['a3']; p,c=b[pre+'p'],b[pre+'c']
    occ=(rel[:,0,:,-4:]>.5) if T=='BE_BASE_D3' else (b['a1'][:,0,:,1:5]>.5)
    occupied=(occ&m[:,0,:,None]).any(1)[np.arange(len(x)),c[:,0]]
    elig=x[:,0,7]>.5
    if T=='BE_BASE_D3': elig&=x[:,0,18]>=0
    act=p[:,0].astype(bool)
    return dict(elig=elig,act=act,occ=occupied,truth=elig&act&~occupied,
                logical=[(int(ph),bool(e),bool(a),bool(o)) for ph,e,a,o in zip(b['native'][:,3],elig,act,occupied)],
                fine=[key(b,i,p,c,T) for i in range(len(x))])
res={}
for t,T in TEACH.items():
    for s in range(105001,105011):
        D=f'{X}/datasets/{t}/s{s}'; E=f'{X}/test/{t}/s{s}'
        base,aux,test=L(D+'/base.npz'),L(D+'/aux.npz'),L(E+'/aux.npz')
        man=json.load(open(D+'/manifest.json'))
        use=~np.isin(base['native'][:,3],[LINIAL,DETECT]); 
        nbase=len(base['native']); ncolor=int(use.sum())
        lin=int((base['native'][:,3]==LINIAL).sum()); det=int((base['native'][:,3]==DETECT).sum())
        comps=3*ncolor+lin+det
        bb={k:v[use] for k,v in base.items()}
        bf,nf,cf,tf=facts(bb,T),facts(aux,T),facts(aux,T,'cf_'),facts(test,T,'cf_')
        chk=dict(base_truth=bool(np.array_equal(bf['truth'],bb['native'][:,0]>0)),
                 aux_native=bool(np.array_equal(nf['truth'],aux['native_y']>.5)),
                 aux_cf=bool(np.array_equal(cf['truth'],aux['cf_y']>.5)),
                 test_cf=bool(np.array_equal(tf['truth'],test['cf_y']>.5)))
        trainfine=set(bf['fine'])|set(nf['fine'])|set(cf['fine'])
        chk['train_keys_match_file']=trainfine==set(json.load(open(D+'/type-keys.json')))
        chk['test_keys_match_file']=set(tf['fine'])==set(json.load(open(E+'/type-keys.json')))
        chk['no_intersection']=not (trainfine & set(tf['fine']))
        chk['train_not_held']=all(not held(k) for k in trainfine)
        chk['test_held']=all(held(k) for k in set(tf['fine']))
        cnt=Counter(tf['fine']); truth=tf['truth']
        gate=tf['elig']&tf['act']
        tp=int((gate&truth).sum()); tn=int((~gate&~truth).sum()); fp=int((gate&~truth).sum()); fn=int((~gate&truth).sum())
        gba=.5*(tp/(tp+fn)+tn/(tn+fp))
        bt=defaultdict(list)
        for k,g in zip(tf['fine'],gate==truth): bt[k].append(g)
        gtype=float(np.mean([np.mean(v) for v in bt.values()]))
        # labels
        def lab(f): return int(f['truth'].sum()),int((~f['truth']).sum())
        trlab={arm:tuple(a+b for a,b in zip(lab(bf),lab(f))) for arm,f in (('native',nf),('cf',cf))}
        byphase={}
        for ph in sorted(set(aux['native'][:,3].tolist())):
            bsel=bb['native'][:,3]==ph; asel=aux['native'][:,3]==ph
            for arm,f in (('native',nf),('cf',cf)):
                byphase[(ph,arm)]=(int(bf['truth'][bsel].sum()+f['truth'][asel].sum()), int((~bf['truth'][bsel]).sum()+(~f['truth'][asel]).sum()))
        trlog={arm:set(bf['logical'])|set(f['logical']) for arm,f in (('native',nf),('cf',cf))}
        ov={arm:float(np.mean([k in trlog[arm] for k in tf['logical']])) for arm in trlog}
        # novelty within logical: test logical types unseen by CF
        unseen_cf=Counter(k for k in tf['logical'] if k not in trlog['cf'])
        r=dict(chk=chk,types=len(cnt),maxmass=max(cnt.values())/len(truth),test_pos=int(truth.sum()),test_neg=int((~truth).sum()),
               gate_ba=gba,gate_type=gtype,train_labels=trlab,byphase={f'{k[0]}|{k[1]}':v for k,v in byphase.items()},
               overlap=ov,logical_test_types=len(set(tf['logical'])),unseen_cf_logical={str(k):v for k,v in unseen_cf.items()},
               base_roots=nbase,base_color_roots=ncolor,linial=lin,detect=det,base_components=comps,
               man_acc=man['accounting'],man_targets=man['optimization_targets_per_model'],
               aux_n=len(aux['native']),test_n=len(truth),
               native_aux_silent=int((~aux['p'][:,0]).sum()),cf_aux_silent=int((~aux['cf_p'][:,0]).sum()),
               test_silent=int((~test['cf_p'][:,0]).sum()), test_phase=dict(Counter(test['native'][:,3].tolist())),
               aux_phase=dict(Counter(aux['native'][:,3].tolist())),
               aux_elig_active_cf=int((cf['elig']&cf['act']).sum()), aux_occ_cf=int((cf['elig']&cf['act']&cf['occ']).sum()),
               aux_inelig_active_cf=int((~cf['elig']&cf['act']).sum()), aux_inelig_active_native=int((~nf['elig']&nf['act']).sum()),
               test_inelig_active=int((~tf['elig']&tf['act']).sum()))
        res[f'{t}|{s}']={k:(v if not isinstance(v,dict) or k!='train_labels' else {a:list(b) for a,b in v.items()}) for k,v in r.items()}
        print(t,s,'checks',all(chk.values()),'types',r['types'],'max%.4f'%r['maxmass'],'pos/neg',r['test_pos'],r['test_neg'],
              'gateBA %.3f gateType %.3f'%(100*gba,100*gtype),'trainlab',trlab,'ovl cf %.5f nat %.5f'%(ov['cf'],ov['native']),
              'base comps',comps,'(color',ncolor,'lin',lin,'det',det,')', 'unseenCF',dict(unseen_cf), flush=True)
        if not all(chk.values()): print('   FAILED CHECKS',chk)
json.dump(res,open(sys.argv[2],'w'),indent=1,default=str)
