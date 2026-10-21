"""Generate scientific summaries from recorded observations, never hand tables."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from dcl.study.storage import atomic_json, file_hash
from .config import ROOT, RESULTS
from .report import collect, report, write_csv


def aggregate_laws(laws,b3):
    law=pd.DataFrame(laws)
    if law.empty:return law,pd.DataFrame()
    law['treatment']=law.model.str.rsplit('-s',n=1).str[0]
    summary=law.groupby(['treatment','k','pattern','kind'],dropna=False).agg(
        networks=('model','nunique'),tv=('tv','mean'),tv_min=('tv','min'),tv_max=('tv','max'),
        fidelity=('fidelity','mean'),effective_colors=('effective_collision','mean'),
        max_mass=('max_mass','mean'),forbidden_mass=('forbidden_mass','mean')).reset_index()
    summary.to_csv(RESULTS/'law-summary.csv',index=False)
    rows=[]
    curve=pd.DataFrame(b3)
    if not curve.empty:
        for (model,state),part in curve.groupby(['model','state']):
            part=part.sort_values('m')
            adequate=part[(part.power>=.8)&(part.fpr<=.05)]
            rows.append(dict(model=model,state=state,k=int(part.k.iloc[0]),
                             first_observed_m_power08=int(adequate.m.iloc[0]) if len(adequate) else None,
                             max_m=int(part.m.max()),balanced_accuracy_m1=float(part.balanced_accuracy.iloc[0]),
                             balanced_accuracy_m2000=float(part.balanced_accuracy.iloc[-1]),
                             power_m2000=float(part.power.iloc[-1]),fpr_m2000=float(part.fpr.iloc[-1]),
                             note='Observed threshold only; consult pointwise intervals; not a guaranteed population sample size'))
        write_csv(RESULTS/'sample-requirements.csv',rows)
    return law,summary


def cluster_uncertainty(frame):
    rows=[]
    for (treatment,regime),part in frame.groupby(['treatment','regime']):
        groups=part.groupby('graph_cluster').agg(value=('success','mean'),n=('success','size'))
        w=groups.n/groups.n.sum()
        radius=float(np.sqrt(.5*np.sum(w*w)*np.log(40)))
        rate=float(part.success.mean())
        rows.append(dict(treatment=treatment,regime=regime,graph_clusters=len(groups),success_rate=rate,
                         hoeffding_lower=max(0.,rate-radius),hoeffding_upper=min(1.,rate+radius),
                         interpretation='Conservative pointwise interval for expected prescribed-mixture performance conditional on fixed networks and independent graph/tape clusters; no new-architecture guarantee'))
    write_csv(RESULTS/'cluster-uncertainty.csv',rows)


def extra_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=RESULTS/'controlled-progress.json'
    if not path.exists():return
    data=json.loads(path.read_text())
    c=pd.DataFrame(data['cliques']);h=pd.DataFrame(data['heterogeneous'])
    write_csv(RESULTS/'clique-progress.csv',data['cliques']);write_csv(RESULTS/'heterogeneous-progress.csv',data['heterogeneous'])
    directory=Path('paper/ucgnn/figures')
    if not c.empty:
        fig,ax=plt.subplots(figsize=(6.5,3.4))
        colors={'normalized':'#2166ac','deepsets':'#1b9e77','transformer':'#984ea3'}
        for treatment,color in colors.items():
            part=c[(c.intervention=='learned-estimated')&c.model.str.startswith(treatment+'-')]
            ax.scatter(part.effective_colors,part.exact_commit_probability_plugin,label=treatment,s=15,color=color)
        ax.set(xlabel='Effective colors (inverse collision probability)',ylabel='Exact-COMMIT probability (estimated law)')
        ax.legend();fig.tight_layout();fig.savefig(directory/'diversity-and-progress.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(6.5,3.4))
        part=c[(c.k==201)&c.model.str.startswith(('normalized-','deepsets-','transformer-'))].copy()
        part['treatment']=part.model.str.rsplit('-s',n=1).str[0]
        table=part.groupby(['treatment','intervention']).exact_commit_probability_plugin.mean().unstack()
        table.plot.bar(ax=ax);ax.set(ylabel='Per-node exact-COMMIT probability',xlabel='All-free clique K=201')
        ax.tick_params(axis='x',rotation=0);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(directory/'candidate-mixture-intervention.pdf');plt.close(fig)
    if not h.empty:
        fig,ax=plt.subplots(figsize=(6,3.4))
        for model,part in h.groupby('model'):
            ax.errorbar(part.expected_exact_commits_plugin,part.empirical_exact_commits,yerr=1.96*part.exact_standard_error,
                        fmt='o',markersize=4,label=model)
        maximum=max(h.expected_exact_commits_plugin.max(),h.empirical_exact_commits.max())*1.05
        ax.plot([0,maximum],[0,maximum],'--',color='black',alpha=.4)
        ax.set(xlabel='Predicted exact confirmations (estimated marginals)',ylabel='Observed one-round exact confirmations')
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(directory/'heterogeneous-state-progress.pdf');plt.close(fig)


def archive():
    target=RESULTS/'graph-records.jsonl.gz'
    with gzip.open(target.with_suffix('.tmp'),'wt',compresslevel=6) as handle:
        for p in sorted((ROOT/'graphs').rglob('*.json')):handle.write(json.dumps(json.loads(p.read_text()),separators=(',',':'))+'\n')
    target.with_suffix('.tmp').replace(target)
    hist={};index=[]
    for p in sorted((ROOT/'distributions').rglob('*.npz')):
        index.append(dict(path=str(p),sha256=file_hash(p),bytes=p.stat().st_size))
        if p.name.endswith('-b3.npz'):continue
        x=np.load(p)
        key=p.parent.name+'__'+p.stem
        for name in ('estimation_counts','test_counts','available','quadrature_choices','boundary_rc','boundary_choices'):
            hist[key+'__'+name]=x[name]
    np.savez_compressed(RESULTS/'candidate-counts.npz',**hist)
    atomic_json(RESULTS/'candidate-array-manifest.json',index)
    receipts=[]
    for p in sorted((ROOT/'task-receipts').glob('*.json')):
        x=json.loads(p.read_text())
        receipts.append({k:v for k,v in x.items() if k not in ('source','before')}|{'source':x.get('source'), 'before':x.get('before')})
    write_csv(RESULTS/'resources.csv',receipts)


def prose(frame,law,summary,b3):
    directory=Path('paper/ucgnn/tables')
    lines=[]
    local=frame[frame.regime=='LOCAL']
    for treatment,part in local.groupby('treatment'):
        if treatment=='teacher':continue
        escaped=treatment.replace('_','\\_')
        lines.append(f'{escaped} completed {int(part.success.sum())}/{len(part)} local-list evaluations, with '
                     f'{int(part.unsafe_commits.sum()):,} exact-rule-violating authorizations. ')
    lines.append('These are finite-cohort outcomes; shared graphs and five networks do not create independent graph samples. ')
    (directory/'local-results-prose.tex').write_text(''.join(lines)+'\n')
    lines=[]
    for treatment in ('normalized','deepsets','transformer','e1-original'):
        part=summary[(summary.treatment==treatment)&(summary.k==201)&(summary.pattern=='ALL_FREE')]
        if len(part):
            r=part.iloc[0]
            lines.append(f'For all-free $K=201$, {treatment} had mean estimated TV {r.tv:.3f}, '
                         f'shared-r fidelity {r.fidelity:.3f}, and {r.effective_colors:.1f} effective colors '
                         f'(inverse collision probability). ')
    lines.append('The original E1 networks are a separate historical cohort, not the new NORMALIZED replicas. ')
    lines.append('Estimation uncertainty and per-state sample-based classification curves are reported separately; '
                 'low power is not a proof of equality. ')
    (directory/'distribution-results-prose.tex').write_text(''.join(lines)+'\n')
    p=RESULTS/'controlled-progress.json'
    if p.exists():
        c=pd.DataFrame(json.loads(p.read_text())['cliques'])
        lines=[]
        for treatment in ('normalized','deepsets','transformer'):
            part=c[(c.k==201)&c.model.str.startswith(treatment+'-')]
            for intervention in ('learned-estimated','half-uniform-mixture'):
                values=part[part.intervention==intervention].exact_commit_probability_plugin
                if len(values):lines.append(f'{treatment}, {intervention}: mean exact-COMMIT probability {values.mean():.4f}. ')
        lines.append('These are controlled plug-in predictions from estimated laws, with fixed participation 1/2; '
                     'the heterogeneous-state experiment independently checks one-round confirmations. ')
        (directory/'progress-results-prose.tex').write_text(''.join(lines)+'\n')


def finalize():
    report()
    cases,laws,b3=collect();frame=pd.DataFrame(cases);frame['success']=(frame.status=='success').astype(int)
    law,summary=aggregate_laws(laws,b3)
    cluster_uncertainty(frame);extra_figures();prose(frame,law,summary,b3);archive()
    rows=[]
    for _,r in frame[frame.status!='success'].groupby(['model','regime']).head(1).iterrows():
        p=Path(r.output);target=RESULTS/'counterexamples'/p.parent.parent.name/p.parent.name/p.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        if p.with_suffix('.npz').exists():shutil.copy2(p.with_suffix('.npz'),target.with_suffix('.npz'))
        rows.append(dict(model=r.model,regime=r.regime,graph=r.graph_id,status=r.status,result=str(target)))
    write_csv(RESULTS/'counterexample-index.csv',rows)
    print(f'Final tables, figures, independent-state summaries and compressed archive generated for {len(cases)} graph evaluations.')


if __name__=='__main__':finalize()
