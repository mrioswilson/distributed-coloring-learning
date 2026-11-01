import json,sys
T=json.load(open(sys.argv[1])); seeds=range(105001,105011)
for t in ['be_base_d3','speculative_gm']:
    for arm in ['native-test','cf-test']:
        for st in ['eligible_active','eligible_silent','ineligible']:
            k='st_'+st
            if k not in T[f'{t}|105001|{arm}']: continue
            tot=[0,0,0,0]
            for s in seeds:
                v=T[f'{t}|{s}|{arm}'].get(k)
                if v is None: continue
                tot=[a+b for a,b in zip(tot,v)]
            print(t,arm,st,'TP TN FP FN',tot)
    # per seed eligible-active
    for s in seeds:
        n=T[f'{t}|{s}|native-test']; c=T[f'{t}|{s}|cf-test']
        print(' ',s,'EA native',n['st_eligible_active'],'cf',c['st_eligible_active'],'ES native',n.get('st_eligible_silent'),'cf',c.get('st_eligible_silent'),'INEL native',n.get('st_ineligible'),'cf',c.get('st_ineligible'))
    print(t,'negatives per seed',[T[f'{t}|{s}|native-test']['tn']+T[f'{t}|{s}|native-test']['fp'] for s in seeds])
    print(t,'positives per seed',[T[f'{t}|{s}|native-test']['tp']+T[f'{t}|{s}|native-test']['fn'] for s in seeds])
