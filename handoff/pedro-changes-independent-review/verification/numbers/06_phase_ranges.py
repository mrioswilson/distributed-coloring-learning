import json,sys
r=json.load(open(sys.argv[1]))
from collections import defaultdict
rng=defaultdict(list)
for k,v in r.items():
    t=k.split('|')[0]
    for pk,val in v['byphase'].items():
        rng[(t,pk,'pos')].append(val[0]); rng[(t,pk,'neg')].append(val[1])
    rng[(t,'types')].append(v['types']); rng[(t,'maxmass')].append(v['maxmass']); rng[(t,'ovl_cf')].append(v['overlap']['cf']); rng[(t,'ovl_native')].append(v['overlap']['native'])
    rng[(t,'test_neg')].append(v['test_neg']); rng[(t,'test_pos')].append(v['test_pos'])
    rng[(t,'gate_ba')].append(v['gate_ba']); rng[(t,'gate_type')].append(v['gate_type'])
    rng[(t,'base_color')].append(v['base_color_roots']); rng[(t,'detect')].append(v['detect'])
    rng[(t,'nat_lab_pos')].append(v['train_labels']['native'][0]); rng[(t,'nat_lab_neg')].append(v['train_labels']['native'][1])
    rng[(t,'cf_lab_pos')].append(v['train_labels']['cf'][0]); rng[(t,'cf_lab_neg')].append(v['train_labels']['cf'][1])
    rng[(t,'test_silent')].append(v['test_silent']); rng[(t,'test_inelig_active')].append(v['test_inelig_active'])
    rng[(t,'aux_inelig_active_cf')].append(v['aux_inelig_active_cf']); rng[(t,'aux_inelig_active_native')].append(v['aux_inelig_active_native'])
    rng[(t,'aux_occ_cf')].append(v['aux_occ_cf'])
for k,v in sorted(rng.items(), key=str):
    m=sum(v)/len(v)
    print(k,'min',min(v),'max',max(v),'mean',round(m,6))
