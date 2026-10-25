#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dcl.degree.inventory import ROOT


if __name__=='__main__':
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    gaps=list(csv.DictReader((ROOT/'counterfactual-probes/preference-gap.csv').open()))
    rows=[r for r in gaps if r['cohort']=='counterfactual-primary' and r['palette']=='local' and r['gap']=='1']
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    colors=dict(normalized='#0072B2',deepsets='#D55E00',transformer='#009E73')
    for a,color in colors.items():
        for seed in sorted({r['seed'] for r in rows}):
            rr=sorted([r for r in rows if r['architecture']==a and r['seed']==seed],key=lambda x:float(x['normalized_gap']))
            axes[0].plot([float(r['normalized_gap']) for r in rr],[float(r['forced_zero_logit']) for r in rr],
                         color=color,alpha=.65,marker='.',label=a if seed==min(r['seed'] for r in rows) else None)
    axes[0].axhline(0,color='black',linewidth=.7)
    axes[0].set_xscale('log');axes[0].set_xlabel('Normalized preference gap 1 / global degree')
    axes[0].set_ylabel('COMMIT logit for the exact winning color')
    axes[0].set_title('CT: one neighbor, two local colors, 15 CF weights',fontsize=9)
    axes[0].legend(fontsize=7)
    path=ROOT/'failure-replays/JOHANSSON-transformer-historical-primary-passive-s83101/global-n64-d10-mixture1000.json'
    replay=json.loads(path.read_text());sampler=replay['sampler'][0]
    available=sampler['available'];law=np.array(sampler['counts'])/sampler['samples']
    x=np.arange(len(available));width=.34
    axes[1].bar(x-width/2,law[available],width,label='Frozen Transformer (8192 draws)',color='#009E73')
    axes[1].bar(x+width/2,np.full(len(available),1/len(available)),width,label='Exact uniform proposal',color='#777777')
    axes[1].set_xticks(x,[str(c) for c in available]);axes[1].set_ylim(0,1.06)
    axes[1].set_xlabel('Available color');axes[1].set_ylabel('Proposal probability / empirical frequency')
    axes[1].set_title('Johansson: stalled adjacent pending vertices',fontsize=9)
    axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(folder/f'degree-mechanisms.{ext}',dpi=180)
    plt.close(fig)
    data=ROOT/'checkpoint-condition.csv'
    if data.exists():
        allrows=list(csv.DictReader(data.open()))
        rows=[r for r in allrows if r['stage']=='confirmatory' and r['primary']=='True'
              and r['palette']=='global' and r['delta']=='4']
        fig,axes=plt.subplots(2,3,figsize=(10,6),sharey=True)
        for ax,t in zip(axes.flat,('LUBY','JOHANSSON','COLOR_TRIALS','JP','SPECULATIVE_GM','GREEDY')):
            for a,color in colors.items():
                for seed in sorted({r['seed'] for r in rows if r['teacher']==t}):
                    rr=sorted([r for r in rows if r['teacher']==t and r['architecture']==a and r['seed']==seed],key=lambda r:int(r['n']))
                    ax.plot([int(r['n']) for r in rr],[float(r['success_rate']) for r in rr],marker='.',color=color,alpha=.6)
            ax.set_title(t,fontsize=10);ax.set_xticks([64,256]);ax.set_xlabel('n, fixed degree 4, global K5');ax.grid(alpha=.2)
            ax.set_ylim(-.02,1.02)
        fig.suptitle('Size transfer: each line is one frozen checkpoint, 1000 runs per point',fontsize=10)
        fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(folder/f'confirmatory-size.{ext}',dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(10,6),sharey=True)
        for ax,t in zip(axes.flat,('LUBY','JOHANSSON','COLOR_TRIALS','JP','SPECULATIVE_GM','GREEDY')):
            for a,color in colors.items():
                xs=[r for r in allrows if r['stage']=='confirmatory' and r['primary']=='True' and r['teacher']==t
                    and r['architecture']==a and r['n']=='64' and r['palette']=='global']
                for seed in sorted({r['seed'] for r in xs}):
                    rr=sorted([r for r in xs if r['seed']==seed],key=lambda r:int(r['delta']))
                    ax.plot([int(r['delta']) for r in rr],[float(r['success_rate']) for r in rr],marker='.',color=color,alpha=.6,
                            label=a if seed==min(r['seed'] for r in xs) else None)
            ax.set_title(t,fontsize=10);ax.set_xticks([3,4,10]);ax.set_xlabel('Maximum degree, fixed n=64, global palette')
            ax.set_ylim(.94,1.005);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7,loc='lower left')
        axes[0,0].set_ylabel('Success (axis starts at 0.94)')
        axes[1,0].set_ylabel('Success (axis starts at 0.94)')
        fig.suptitle('Frozen transfer: one line per checkpoint, 1000 runs per point',fontsize=10)
        fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(folder/f'confirmatory-degree.{ext}',dpi=180)
