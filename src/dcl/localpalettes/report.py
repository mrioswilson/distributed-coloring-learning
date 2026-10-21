from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dcl.study.storage import atomic_json
from .config import ROOT, RESULTS, seed


def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def collect():
    cases=[];laws=[];b3=[]
    audit=RESULTS/'historical-overlap-audit.json'
    fingerprints={r['graph']:r for r in json.loads(audit.read_text())['graphs']} if audit.exists() else {}
    for path in sorted((ROOT/'graphs').rglob('*.json')):
        row=json.loads(path.read_text());g=row['graph']
        fp=fingerprints.get(g['id'],{})
        cases.append({k:v for k,v in row.items() if k not in ('trace','graph','candidate_histogram_kind')}|
                     {k:g[k] for k in ('family','n','degree','target_degree','degree_cv')}|
                     {'graph_id':g['id'],'graph_cluster':fp.get('fingerprint',g['edges_sha256']),
                      'historical_graph_overlap':fp.get('historical_overlap',False),'output':str(path)})
    for path in sorted((ROOT/'distributions').rglob('*.json')):
        row=json.loads(path.read_text());laws.append(row['metrics']);b3.extend(row['b3'])
    return cases,laws,b3


def paired_bootstrap(frame):
    rows=[]
    if frame.empty:return rows
    for model,part in frame.groupby('model'):
        wide=part.pivot(index=['graph_id','graph_cluster','tape'],columns='regime',values=['success','rounds'])
        if not all(('success',r) in wide for r in ('GLOBAL','LOCAL')):continue
        wide=wide.dropna(subset=[('success','GLOBAL'),('success','LOCAL')])
        difference=pd.DataFrame(dict(success=wide['success']['LOCAL']-wide['success']['GLOBAL'],
                                     rounds=wide['rounds']['LOCAL']-wide['rounds']['GLOBAL']))
        pergraph=difference.groupby(level=1).mean()
        rng=np.random.default_rng(seed('bootstrap',model))
        draws=rng.integers(0,len(pergraph),size=(2000,len(pergraph)))
        bootstrap=pergraph.to_numpy()[draws].mean(1)
        rows.append(dict(model=model,paired_graphs=len(pergraph),success_difference=float(pergraph.success.mean()),
                         success_lower=float(np.quantile(bootstrap[:,0],.025)),success_upper=float(np.quantile(bootstrap[:,0],.975)),
                         capped_round_difference=float(pergraph.rounds.mean()),round_lower=float(np.quantile(bootstrap[:,1],.025)),
                         round_upper=float(np.quantile(bootstrap[:,1],.975)),
                         note='Graph-cluster paired bootstrap; deterministic families descriptive; rounds censored at cutoff'))
    return rows


def figures(frame,laws,b3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path('paper/ucgnn/figures');directory.mkdir(parents=True,exist_ok=True)
    colors={'normalized':'#2166ac','deepsets':'#1b9e77','transformer':'#984ea3','raw_endpoints':'#d95f02','sum':'#e41a1c','mean':'#777777'}
    if not frame.empty:
        summary=frame.groupby(['treatment','regime'],dropna=False).agg(success=('success','mean'),rounds=('rounds','mean')).reset_index()
        fig,axes=plt.subplots(1,2,figsize=(10,3.4))
        for treatment,group in summary.groupby('treatment'):
            if treatment=='teacher':continue
            for ax,metric in zip(axes,('success','rounds')):
                order=['GLOBAL','LOCAL','OVERSIZED-1.5','OVERSIZED-2'];part=group.set_index('regime').reindex(order)
                ax.plot(order,part[metric],'-o',label=treatment,color=colors[treatment],markersize=3)
        axes[0].set_ylabel('Proper complete coloring rate');axes[1].set_ylabel('Mean rounds (cutoff included)')
        for ax in axes:ax.tick_params(axis='x',rotation=20);ax.grid(alpha=.2)
        axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(directory/'local-palette-outcomes.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,3.5))
        local=frame[(frame.regime=='LOCAL')&frame.treatment.isin(['normalized','deepsets','transformer'])]
        for treatment,group in local.groupby('treatment'):
            bygraph=group.groupby('graph_id').agg(cv=('degree_cv','first'),rounds=('rounds','mean'))
            ax.scatter(bygraph.cv,bygraph.rounds,label=treatment,s=12,alpha=.6,color=colors[treatment])
        ax.set(xlabel='Degree coefficient of variation',ylabel='LOCAL mean rounds');ax.legend();fig.tight_layout()
        fig.savefig(directory/'local-degree-heterogeneity.pdf');plt.close(fig)
    if laws:
        law=pd.DataFrame(laws);law['treatment']=law.model.str.rsplit('-s',n=1).str[0]
        subset=law[law.pattern=='ALL_FREE']
        fig,axes=plt.subplots(1,2,figsize=(9,3.3))
        for treatment,group in subset.groupby('treatment'):
            if treatment not in colors:continue
            byk=group.groupby('k').agg(tv=('tv','mean'),fidelity=('fidelity','mean'))
            axes[0].plot(byk.index,byk.tv,'-o',label=treatment,color=colors[treatment],markersize=3)
        for treatment,group in law.groupby('treatment'):
            if treatment in ('normalized','deepsets','transformer'):
                axes[1].scatter(group.fidelity,group.tv,s=7,alpha=.35,label=treatment,color=colors[treatment])
        axes[0].set(xscale='log',xlabel='K (all free)',ylabel='Estimated TV from uniform teacher')
        axes[1].set(xlabel='Shared-r fidelity',ylabel='Estimated TV (all states)')
        axes[0].legend(fontsize=6);axes[1].legend(fontsize=6);fig.tight_layout()
        fig.savefig(directory/'candidate-fidelity-and-tv.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,3.3))
        for treatment in ('normalized','deepsets','transformer'):
            path=ROOT/'distributions'/f'{treatment}-s41000'/'k201-ALL_FREE.npz'
            if not path.exists():continue
            counts=np.load(path)['estimation_counts'];ax.plot(np.arange(201),counts/counts.sum(),label=treatment,alpha=.8,color=colors[treatment])
        ax.axhline(1/201,color='black',linestyle='--',label='exact teacher');ax.set(xlabel='Candidate color',ylabel='Estimated probability')
        ax.legend();fig.tight_layout();fig.savefig(directory/'candidate-laws-k201.pdf');plt.close(fig)
    if b3:
        data=pd.DataFrame(b3);data['treatment']=data.model.str.rsplit('-s',n=1).str[0]
        fig,axes=plt.subplots(1,2,figsize=(9,3.3))
        for treatment,group in data[data.state=='k201-ALL_FREE'].groupby('treatment'):
            if treatment not in colors:continue
            summary=group.groupby('m').mean(numeric_only=True)
            axes[0].plot(summary.index,summary.balanced_accuracy,'-o',label=treatment,color=colors[treatment],markersize=3)
            axes[1].plot(summary.index,summary.power,'-o',color=colors[treatment],markersize=3)
        axes[0].set(xscale='log',xlabel='Observed IID candidates m',ylabel='Balanced accuracy (K=201 all free)',ylim=(.4,1.02))
        axes[1].set(xscale='log',xlabel='Observed IID candidates m',ylabel='Power at calibrated 5% level',ylim=(-.02,1.02))
        axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(directory/'candidate-distinguishability.pdf');plt.close(fig)


def report():
    cases,laws,b3=collect()
    write_csv(RESULTS/'graph-cases.csv',cases);write_csv(RESULTS/'candidate-laws.csv',laws);write_csv(RESULTS/'distinguishability.csv',b3)
    frame=pd.DataFrame(cases)
    if not frame.empty:
        frame['success']=(frame.status=='success').astype(int)
        matrix=frame.groupby(['treatment','network_seed','family','degree','n','regime'],dropna=False).agg(
            successes=('success','sum'),cases=('success','size'),mean_rounds=('rounds','mean'),
            unsafe_commits=('unsafe_commits','sum'),conflicts=('conflicts','sum'),pending=('pending','sum')).reset_index()
        matrix.to_csv(RESULTS/'success-matrix.csv',index=False)
        aggregates=frame.groupby(['treatment','regime'],dropna=False).agg(successes=('success','sum'),cases=('success','size'),
                   mean_rounds=('rounds','mean'),unsafe_commits=('unsafe_commits','sum')).reset_index()
        aggregates.to_csv(RESULTS/'regime-summary.csv',index=False)
        write_csv(RESULTS/'paired-comparisons.csv',paired_bootstrap(frame))
        summary=aggregates.to_string(index=False)
        table=Path('paper/ucgnn/tables/local-palettes.tex')
        with table.open('w') as out:
            out.write('\\begin{tabular}{llrr}\\toprule\nTreatment & Regime & Success / cases & Mean rounds\\\\\\midrule\n')
            for _,r in aggregates.iterrows():
                name=str(r.treatment).replace('_','\\_')
                out.write(f'{name} & {r.regime} & {r.successes}/{r.cases} & {r.mean_rounds:.2f}\\\\\n')
            out.write('\\bottomrule\\end{tabular}\n')
    else:summary='No graph results yet.'
    figures(frame,laws,b3)
    queue=json.loads((ROOT/'queue.json').read_text()) if (ROOT/'queue.json').exists() else {}
    counts={s:sum(t['status']==s for t in queue.get('tasks',[])) for s in ('complete','running','pending','technical_failure')}
    text=f'''# Local palettes and distinguishability — generated report

Graph cases observed: {len(cases)} / 18848 planned. Conditional laws: {len(laws)}.
Finite-sample classifier rows: {len(b3)}. Task states: {counts}.
GPU-process hours: {queue.get('gpu_seconds',0)/3600:.4f} / 12.

```text
{summary}
```

Primary outcome is finite-cohort proper completion. Exact-rule-violating
authorizations, conflicts and cutoffs are separately recorded in CSV. Shared
graphs are not independent observations for each network. See the preregistered
protocol for pairing, intervals, RNG conversion and independent sample splits.
Reports may be regenerated during execution and do not imply full completion.
'''
    (RESULTS/'REPORT.md').write_text(text)
    print(f'Report: {len(cases)} graph cases, {len(laws)} laws, {len(b3)} classifier rows.')
