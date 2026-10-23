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
    if law.empty:return law,pd.DataFrame(columns=['treatment','k','pattern','kind','networks','tv','fidelity','effective_colors','max_mass','forbidden_mass'])
    law['treatment']=law.model.str.rsplit('-s',n=1).str[0]
    summary=law.groupby(['treatment','k','pattern','kind'],dropna=False).agg(
        networks=('model','nunique'),tv=('tv','mean'),tv_min=('tv','min'),tv_max=('tv','max'),
        fidelity=('fidelity','mean'),effective_colors=('effective_collision','mean'),entropy_effective_colors=('effective_entropy','mean'),
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
                             nonattainment_reason='' if len(adequate) else 'Joint criterion not met; inspect power and FPR separately',
                             note='Observed joint power/FPR threshold only; consult pointwise intervals; not guaranteed population sample complexity'))
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
    historical=RESULTS/'historical-e1-checkpoints.json'
    if historical.exists():
        for spec in json.loads(historical.read_text()):
            for k in (4,11,51,201):
                source=ROOT/'distributions'/spec['label']/f'k{k}-ALL_FREE.npz'
                if not source.exists():continue
                q=np.load(source)['estimation_counts'].astype(float);q/=q.sum()
                for name,distribution in (('learned-estimated',q),('uniform',np.full(k,1/k)),('half-uniform-mixture',.5*q+.5/k)):
                    probability=float(.5*np.sum(distribution*(1-.5*distribution)**(k-1)))
                    data['cliques'].append(dict(model=spec['label'],k=k,intervention=name,p=.5,
                        exact_commit_probability_plugin=probability,expected_commits=k*probability,
                        collision=float(np.sum(distribution**2)),effective_colors=float(1/np.sum(distribution**2)),
                        tv=float(.5*np.abs(distribution-1/k).sum()),interpretation='Secondary original-E1 law; exact formula for estimated categorical law'))
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
    queue=json.loads((ROOT/'queue.json').read_text())
    for extra in queue.get('posthoc_gpu_checks',[]):
        receipts.append(dict(task=extra['id'],seconds=extra['seconds'],kind='separate-posthoc-validation',
                             peak_gpu_bytes=None,note='Peak memory not recorded for this supplementary check; elapsed GPU-worker time charged'))
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
    examined=frame[frame.historical_graph_overlap];fresh=frame[frame.random_graph_spec & ~frame.historical_graph_overlap]
    lines.append(f'{examined.graph_id.nunique()} specifications ({examined.graph_cluster.nunique()} distinct structures) '
                 f'are previously examined controls; {fresh.graph_id.nunique()} fresh random specifications are reported separately. ')
    (directory/'local-results-prose.tex').write_text(''.join(lines).rstrip()+'\n')
    abstract=[]
    for treatment in ('normalized','deepsets','transformer'):
        part=local[local.treatment==treatment]
        abstract.append(f'{treatment} {int(part.success.sum())}/{len(part)}')
    (directory/'local-abstract.tex').write_text('A separate frozen-weight extension using only local-degree palettes recorded '+
        ', '.join(abstract)+' proper completions. Conditional-law tests distinguish pointwise fidelity from candidate-distribution similarity.\n')
    lines=[]
    for treatment in ('normalized','deepsets','transformer','e1-original'):
        part=summary[(summary.treatment==treatment)&(summary.k==201)&(summary.pattern=='ALL_FREE')]
        if len(part):
            r=part.iloc[0]
            lines.append(f'For all-free $K=201$, {treatment} had mean estimated TV {r.tv:.3f}, '
                         f'shared-r fidelity {r.fidelity:.3f}, and {r.effective_colors:.1f} effective colors '
                         f'(inverse collision probability). ')
    lines.append('The original E1 networks are a separate historical cohort, not the new NORMALIZED replicas. ')
    for treatment in ('deepsets','transformer'):
        part=law[law.treatment==treatment];worst=part.loc[part.forbidden_mass.idxmax()]
        escaped_state=str(worst.state).replace('_','\\_')
        lines.append(f'{int((part.forbidden_mass>0).sum())}/{len(part)} model--state estimates for {treatment} '
                     f'contain forbidden proposals; the largest observed mass is {worst.forbidden_mass:.3f} '
                     f'in {escaped_state}. These are proposal errors; '
                     'the MAX COMMIT certificate rejects the corresponding fixed-color conflicts. ')
    lines.append('Estimation uncertainty and per-state sample-based classification curves are reported separately; '
                 'low power is not a proof of equality. ')
    (directory/'distribution-results-prose.tex').write_text(''.join(lines).rstrip()+'\n')
    singleton=RESULTS/'singleton-product-laws.csv'
    if singleton.exists():
        single=pd.read_csv(singleton)
        point=single[(single.model=='transformer-s41002')&(single.state=='k201-SINGLE-last')&(single.m==5)].iloc[0]
        (directory/'singleton-results.tex').write_text(
            'An exploratory analytic readout of singleton states illustrates the effect. '
            f'Transformer seed 41002 proposed a forbidden color with estimated probability {point.forbidden_mass:.3f} '
            f'when only the last color at $K=201$ was free. For five IID candidates, the support test has '
            f'estimated power {point.analytic_power_estimate:.3f}, with simultaneous lower/upper bounds '
            f'[{point.analytic_power_lower:.3f},{point.analytic_power_upper:.3f}] and exact null FPR zero. '
            'The formula is exact in the unknown true mass; these numerical values are estimates. '
            'Bonferroni Clopper--Pearson intervals cover all 1,015 recorded singleton model--state laws; '
            'this readout does not add empirical five-sample test trials.\n')
    p=RESULTS/'controlled-progress.json'
    if p.exists():
        c=pd.read_csv(RESULTS/'clique-progress.csv')
        lines=[]
        for treatment in ('normalized','deepsets','transformer'):
            part=c[(c.k==201)&c.model.str.startswith(treatment+'-')]
            for intervention in ('learned-estimated','half-uniform-mixture'):
                values=part[part.intervention==intervention].exact_commit_probability_plugin
                if len(values):lines.append(f'{treatment}, {intervention}: mean exact-COMMIT probability {values.mean():.4f}. ')
        lines.append('These are controlled plug-in predictions from estimated laws, with fixed participation 1/2; '
                     'the heterogeneous-state experiment independently checks one-round confirmations. ')
        replication=RESULTS/'progress-replication.json'
        if replication.exists():
            rep=json.loads(replication.read_text())
            lines.append('The largest original heterogeneous-state residual triggered a post-hoc independent replication, '
                         f'which predicted {rep["prediction"]:.2f} versus {rep["observed_mean"]:.2f} confirmations '
                         f'(observation-only standard error {rep["observed_standard_error"]:.2f}, 1,024 trials). '
                         'Prediction uncertainty is additional; the original 256-trial results remain unchanged. ')
        (directory/'progress-results-prose.tex').write_text(''.join(lines).rstrip()+'\n')
    negative=RESULTS/'negative-commit-certificates.json'
    if negative.exists():
        records=pd.DataFrame(json.loads(negative.read_text()))
        real=records[records.arithmetic=='exact-real-coefficients']
        text=[]
        for case,degree in (('all-pending-other',8),('one-fixed-same',8),('one-pending-same',16)):
            part=real[(real['case']==case)&(real.degree==degree)]
            text.append(f'{int(part.certified_error.sum())}/{len(part)} frozen gates have a certified error '
                        f'in the degree-{degree} {case} local-star intervention. ')
        text.append('These are exploratory counterexamples selected from the recorded degree-probe grid, '
                    'with unchanged checkpoints. They establish errors of these learned gates in real arithmetic; '
                    'they do not establish an expressivity impossibility for SUM. ')
        (directory/'negative-commit-prose.tex').write_text(''.join(text).rstrip()+'\n')


def executive(frame,law,summary,b3):
    from .executive_report import write
    write(frame,law,summary,b3)


def finalize():
    report()
    frame=pd.read_csv(RESULTS/'graph-cases.csv');frame['success']=(frame.status=='success').astype(int)
    laws=pd.read_csv(RESULTS/'candidate-laws.csv').to_dict('records') if (RESULTS/'candidate-laws.csv').exists() else []
    b3=pd.read_csv(RESULTS/'distinguishability.csv').to_dict('records') if (RESULTS/'distinguishability.csv').exists() else []
    law,summary=aggregate_laws(laws,b3)
    from .results_analysis import analyze
    from .singleton_products import analyze as singleton_analysis
    analyze(frame,law,b3);singleton_analysis()
    cluster_uncertainty(frame);extra_figures();prose(frame,law,summary,b3);archive();executive(frame,law,summary,b3)
    rows=[]
    for _,r in frame[frame.status!='success'].groupby(['model','regime']).head(1).iterrows():
        p=Path(r.output);target=RESULTS/'counterexamples'/p.parent.parent.name/p.parent.name/p.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        if p.with_suffix('.npz').exists():shutil.copy2(p.with_suffix('.npz'),target.with_suffix('.npz'))
        rows.append(dict(model=r.model,regime=r.regime,graph=r.graph_id,status=r.status,result=str(target)))
    write_csv(RESULTS/'counterexample-index.csv',rows)
    print(f'Final tables, figures, independent-state summaries and compressed archive generated for {len(frame)} graph evaluations.')


if __name__=='__main__':finalize()
