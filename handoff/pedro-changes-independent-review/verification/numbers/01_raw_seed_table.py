# Recompute per-seed BA, strata, successes, steps from raw prospective JSONs.
import json,sys,os,statistics as st
D=sys.argv[1]
out={}
for t in ['be_base_d3','speculative_gm']:
    for s in range(105001,105011):
        for arm in ['native-test','cf-test','teacher-reference']:
            p=f'{D}/{t}/s{s}/{arm}.json'
            d=json.load(open(p))
            r={}
            dg=d.get('diagnostic')
            if dg:
                tp,tn,fp,fn=(dg[k] for k in('tp','tn','fp','fn'))
                assert tp+tn+fp+fn==dg['n']==4000,(p,tp+tn+fp+fn)
                sens=tp/(tp+fn); spec=tn/(tn+fp)
                ba=(sens+spec)/2
                r.update(tp=tp,tn=tn,fp=fp,fn=fn,ba=ba,ba_stored=dg['balanced_accuracy'],
                         eqtype=dg.get('equal_type_accuracy'),types=dg.get('distinct_types'),
                         inter=dg.get('train_test_type_intersection'),
                         acc=dg['accuracy'])
                # strata sum check
                S={k:0 for k in('tp','tn','fp','fn','n')}
                for nm,sd in dg['strata'].items():
                    for k in S: S[k]+=sd[k]
                    r['st_'+nm]=(sd['tp'],sd['tn'],sd['fp'],sd['fn'])
                assert (S['tp'],S['tn'],S['fp'],S['fn'],S['n'])==(tp,tn,fp,fn,4000),(p,S)
            rows=d['rows']
            succ=sum(1 for x in rows if x['terminal_correct'])
            r['n_rows']=len(rows); r['succ']=succ
            rs=d.get('rollout_summary') or d['summary']; r['succ_summary']=rs['successes']
            r['mean_steps']=sum(x['native_steps'] for x in rows)/len(rows)
            r['mean_steps_summary']=rs.get('mean_native_steps')
            r['succ_steps']=sorted(set(x['native_steps'] for x in rows if x['terminal_correct']))
            fl=[x['native_steps'] for x in rows if not x['terminal_correct']]
            r['fail_steps_sum']=sum(fl); r['n_fail']=len(fl)
            r['fail_kinds']={}
            for x in rows:
                if not x['terminal_correct']:
                    k=str(x['failure'])[:40]; r['fail_kinds'][k]=r['fail_kinds'].get(k,0)+1
            r['fingerprints']=[x['fingerprint'] for x in rows]
            r['case_ids']=[x['case_id'] for x in rows]
            r['correct_vec']=[x['terminal_correct'] for x in rows]
            r['freeze']=d.get('freeze_sha256'); r['manifest']=d.get('test_manifest_sha256')
            out[f'{t}|{s}|{arm}']=r
json.dump(out,open(sys.argv[2],'w'))
for t in ['be_base_d3','speculative_gm']:
    print('==',t)
    print('seed  N_BA   CF_BA  diff  Nsucc CFsucc  Nsteps CFsteps  N_eqtype CF_eqtype types refsucc')
    for s in range(105001,105011):
        n=out[f'{t}|{s}|native-test']; c=out[f'{t}|{s}|cf-test']; ref=out[f'{t}|{s}|teacher-reference']
        print(s, f"{100*n['ba']:.3f} {100*c['ba']:.3f} {100*(c['ba']-n['ba']):+.3f} {n['succ']} {c['succ']} {n['mean_steps']:.3f} {c['mean_steps']:.3f} {100*n['eqtype']:.3f} {100*c['eqtype']:.3f} {n['types']}/{c['types']} ref={ref['succ']} inter={n['inter']},{c['inter']}",
              'BAstored-match', abs(n['ba']-n['ba_stored'])<1e-12 and abs(c['ba']-c['ba_stored'])<1e-12,
              'summ', n['succ']==n['succ_summary'], c['succ']==c['succ_summary'])
