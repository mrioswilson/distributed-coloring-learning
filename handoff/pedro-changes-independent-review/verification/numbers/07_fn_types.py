# For BE CF seeds: list held-out test fine types by stratum with counts and label,
# then enumerate type-level prediction assignments consistent with stored confusion + equal-type accuracy
# (prediction is constant within a fine type: be-failure-diagnostics within_type_sign_conflicts=0, logit span 0).
import numpy as np, json, sys, itertools
from collections import Counter, defaultdict
sys.path.insert(0, sys.argv[3]); 
X=sys.argv[1]; T=json.load(open(sys.argv[2]))
exec(open(sys.argv[3]+'/05_array_audit.py').read().split('res={}')[0])  # reuse key()/facts()
for s in range(105001,105011):
    t='be_base_d3'; test=L(f'{X}/test/{t}/s{s}/aux.npz'); f=facts(test,'BE_BASE_D3','cf_')
    stratum=np.where(f['elig']&f['act'],'EA',np.where(f['elig'],'ES','IN'))
    info=defaultdict(lambda:[0,None,None])
    for k,y,st in zip(f['fine'],f['truth'],stratum):
        info[k][0]+=1; info[k][1]=bool(y); info[k][2]=str(st)
    types=sorted(info.items(), key=lambda kv:-kv[1][0])
    c=T[f'{t}|{s}|cf-test']; ntypes=len(types); target_correct=round(c['eqtype']*ntypes)
    if s in (105001,105008) or s==105002:
        print(f'== seed {s} CF tp={c["tp"]} fn={c["fn"]} fp={c["fp"]} tn={c["tn"]} eqtype={c["eqtype"]:.4f} ({target_correct}/{ntypes} types correct)')
        for k,(n,y,st) in types:
            kk=json.loads(k); nb=kk[5]
            act_nonoffer_nonocc=any(m[2]==1 and m[3]==0 and m[1]==0 for m in nb)
            any_active=any(m[2]==1 for m in nb)
            print(f'  {st} y={int(y)} n={n:4d} root={kk[4]} w={kk[2]} kappa={kk[3]} nbrs={nb}  activeNbr={int(any_active)} activeNonOfferFree={int(act_nonoffer_nonocc)}')
        # enumerate positive-type subsets matching TP
        pos=[(k,n) for k,(n,y,st) in types if y]
        sols=[]
        for r in range(len(pos)+1):
            for sub in itertools.combinations(range(len(pos)),r):
                if sum(pos[i][1] for i in sub)==c['tp']: sols.append(sub)
        print('  positive types',len(pos),'subsets of accepted positive types matching TP:',len(sols))
        for sub in sols[:10]:
            acc=[pos[i] for i in sub]; rej=[pos[i] for i in range(len(pos)) if i not in sub]
            def feat(k): nb=json.loads(k)[5]; return int(any(m[2]==1 and m[3]==0 and m[1]==0 for m in nb))
            print('   accepted:',[(n,feat(k)) for k,n in acc],' rejected(FN):',[(n,feat(k)) for k,n in rej])
