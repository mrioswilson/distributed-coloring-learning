# How much of the BE positive class sits on the two dominant held-out kappa=0 all-silent-neighbor types;
# and how often CF/native training saw kappa=0 active-root configurations.
import numpy as np, json, sys
from collections import Counter
X=sys.argv[1]
exec(open(sys.argv[2]+'/05_array_audit.py').read().split('res={}')[0])
K1='["BE_BASE_D3",5,1,0,[1,0,1,1],[[1,0,0,1,0,0]]]'
K2='["BE_BASE_D3",5,1,0,[1,0,1,1],[[1,0,0,0,0,0],[1,0,0,1,0,0]]]'
print('K1 held out',held(K1),'K2 held out',held(K2))
for s in range(105001,105011):
    t='be_base_d3'; D=f'{X}/datasets/{t}/s{s}'; E=f'{X}/test/{t}/s{s}'
    test=L(E+'/aux.npz'); tf=facts(test,'BE_BASE_D3','cf_')
    pos=Counter(k for k,y in zip(tf['fine'],tf['truth']) if y)
    base=L(D+'/base.npz'); use=~np.isin(base['native'][:,3],[4,2]); bb={k:v[use] for k,v in base.items()}
    aux=L(D+'/aux.npz'); bf=facts(bb,'BE_BASE_D3'); nf=facts(aux,'BE_BASE_D3'); cf=facts(aux,'BE_BASE_D3','cf_')
    def k0(f): # kappa=0, active root, eligible -> count and positives
        ks=[json.loads(k) for k in f['fine']]
        sel=[i for i,k in enumerate(ks) if k[3]==0 and k[4][2]==1]
        return len(sel), int(sum(f['truth'][i] for i in sel))
    print(s,'test positives',sum(pos.values()),'in K1+K2',pos[K1]+pos[K2],'(%.1f%%)'%(100*(pos[K1]+pos[K2])/sum(pos.values())),
          '| train kappa0-active (n,pos): base',k0(bf),'native-aux',k0(nf),'cf-aux',k0(cf))
