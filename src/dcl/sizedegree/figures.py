"""All scientific figures are derived automatically from archived results."""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from .common import *

ARCHES=('normalized','deepsets','transformer')
NAMES=dict(normalized='E1 NORMALIZED',deepsets='DeepSets',transformer='LocalPaletteTransformer')
COLORS=dict(normalized='#176a9c',deepsets='#bb6818',transformer='#8270b5')
SEEDCOLORS=plt.get_cmap('tab10')(np.arange(5))

def save(fig,name):
    directory=OUT/'figures';directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(directory/f'{name}.png',bbox_inches='tight',dpi=180)
    target=Path('paper/ucgnn/figures');target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(directory/f'{name}.pdf',target/f'{name}.pdf')
    plt.close(fig)

def main():
    verify_frozen()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fc=read(OUT/'factorial-contrasts.json');events=read(OUT/'elementary-events.json')
    cell={(r['event'],r['group'],r['contrast']):r for r in fc if r['kind']=='cell'}
    def C(arch,n,d,event='used_le_delta'):return cell[(event,arch,f'n{n}-d{d}')]
    counts=pd.read_csv(OUT/'cell-test-counts.csv')
    diffs=pd.DataFrame(read(OUT/'event-differences.json'))
    lookup={(r['graph'],r['model'],r['event']):r for r in events}
    gs=graphs()
    def prob(labels,graphs_,event):
        a=[lookup[(g['id'],label,event)] for g in graphs_ for label in labels]
        return np.mean([r['probability'] for r in a]),np.mean([r['ci'][0] for r in a]),np.mean([r['ci'][1] for r in a])

    fig,axes=plt.subplots(1,3,figsize=(12.2,3.8),layout='constrained')
    for ax,arch in zip(axes,ARCHES):
        a=counts[(counts['stratum']=='main')&(counts['architecture']==arch)]
        values=np.zeros((4,4))
        for i,n in enumerate(NS):
            for j,d in enumerate(DS):
                r=a[(a['n']==n)&(a['delta']==d)].iloc[0]
                values[i,j]=r['reject']/r['comparisons']
                ax.text(j,i,f"V2 {r['reject']}/{r['comparisons']}\nMMD {r['old_reject']}/{r['comparisons']}",ha='center',va='center',fontsize=8,color='white' if values[i,j]>.65 else 'black')
        im=ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(xticks=range(4),xticklabels=DS,yticks=range(4),yticklabels=NS,xlabel='Degree / K - 1',ylabel='Vertices n',title=NAMES[arch])
    fig.colorbar(im,ax=axes,shrink=.85,label='Fraction rejected (not a distance)')
    fig.suptitle('1. Global tests: 3 graphs × 5 checkpoints per cell; Holm across 720 hypotheses')
    save(fig,'factorial-01-significance')

    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,arch in zip(axes,ARCHES):
        z=np.array([[C(arch,n,d)['estimate'] for d in DS] for n in NS])
        im=ax.imshow(z,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
        for i,n in enumerate(NS):
            for j,d in enumerate(DS):
                r=C(arch,n,d);lo,hi=r['ci']
                ax.text(j,i,f"{r['estimate']:+.2f}\n[{lo:+.2f}, {hi:+.2f}]",ha='center',va='center',fontsize=7,color='white' if abs(r['estimate'])>.65 else 'black')
        ax.set(xticks=range(4),xticklabels=DS,yticks=range(4),yticklabels=NS,xlabel='Degree / K - 1',ylabel='Vertices n',title=NAMES[arch])
    fig.colorbar(im,ax=axes,shrink=.85,label='P(model: U ≤ Δ) - P(Luby: U ≤ Δ)')
    fig.suptitle('2. Structural event discrepancy; simultaneous 95% fixed-cohort intervals')
    save(fig,'factorial-02-event-discrepancy')

    for by,size,name in [('degree',DS,'factorial-03-size-effect'),('size',NS,'factorial-04-degree-effect')]:
        fig,axes=plt.subplots(2,2,figsize=(9.5,6.5),layout='constrained')
        for ax,fixed in zip(axes.ravel(),size):
            xx=NS if by=='degree' else DS
            for arch in ARCHES:
                aa=[C(arch,x,fixed) if by=='degree' else C(arch,fixed,x) for x in xx]
                yy=np.array([a['estimate'] for a in aa]);ci=np.array([a['ci'] for a in aa])
                ax.plot(xx,yy,'o-',label=NAMES[arch],color=COLORS[arch],markersize=4)
                ax.fill_between(xx,ci[:,0],ci[:,1],alpha=.13,color=COLORS[arch])
            ax.axhline(0,color='.5',lw=.6);ax.set_ylim(-1.05,1.05)
            ax.set_xscale('log',base=2 if by=='degree' else 10);ax.set_xticks(xx,labels=xx)
            ax.set(title=f'Δ = {fixed}' if by=='degree' else f'n = {fixed}',xlabel='Vertices n' if by=='degree' else 'Degree Δ / K - 1',ylabel='P(model: U ≤ Δ) - P(Luby: U ≤ Δ)')
        axes[0,0].legend(fontsize=7)
        fig.suptitle('3. Size effects at fixed degree' if by=='degree' else '4. Degree/palette effects at fixed size')
        save(fig,name)

    fig,axes=plt.subplots(3,4,figsize=(12,8.5),layout='constrained',sharey=True)
    for i,arch in enumerate(ARCHES):
        labels=[s['label'] for s in specs() if s['treatment']==arch]
        for j,d in enumerate(DS):
            ax=axes[i,j]
            for names,color,label in [(labels,COLORS[arch],NAMES[arch]),(['Luby'],'black','Luby')]:
                z=np.array([prob(names,[g for g in gs if g['stage']=='main' and g['n']==n and g['delta']==d],'uses_top_color') for n in NS])
                ax.plot(NS,z[:,0],'o-',color=color,label=label,ms=3);ax.fill_between(NS,z[:,1],z[:,2],color=color,alpha=.14)
            ax.set_xscale('log',base=2);ax.set_xticks(NS,labels=NS,rotation=30);ax.set_ylim(-.03,1.03)
            ax.set(title=f'{NAMES[arch]} / Δ={d}',xlabel='n',ylabel='P(color Δ occurs)')
            if j==0:ax.legend(fontsize=6)
    fig.suptitle('5. Top-color appearance; simultaneous elementary-binomial CP envelopes')
    save(fig,'factorial-05-top-color')

    runs=pd.read_csv(OUT/'all-run-statistics.csv.gz')
    targets=[g for g in gs if g['n']==512 and g['delta']==100 and g['family']=='regular']
    targets.sort(key=lambda g:(g['stage']!='main',g['rep']))
    fig,axes=plt.subplots(3,len(targets),figsize=(12,8),layout='constrained',sharex=True)
    for i,arch in enumerate(ARCHES):
        for j,g in enumerate(targets):
            ax=axes[i,j];a=runs[runs['graph']==g['id']]
            labels=['Luby']+[s['label'] for s in specs() if s['treatment']==arch]
            values=a[a['model'].isin(labels)]['used_colors'];bins=np.arange(values.min()-.5,values.max()+1.5)
            for k,label in enumerate(labels):
                v=a[a['model']==label]['used_colors']
                ax.hist(v,bins=bins,density=True,histtype='step',lw=1.4 if label=='Luby' else .8,color='black' if label=='Luby' else SEEDCOLORS[k-1],label='Luby' if label=='Luby' else label[-5:])
            ax.set(title=f'{NAMES[arch]} / '+(f'new r{g["rep"]}' if g['stage']=='main' else 'historical graph, fresh data'),xlabel='Distinct colors U',ylabel='Probability mass')
            if j==0:ax.legend(fontsize=6,ncol=2)
    fig.suptitle('6. Complete-coloring occupancy at n=512, Δ=100 (500 attempts per generator)')
    save(fig,'factorial-06-used-colors')

    # Supplement contains the entire 4x4 matrix; priority-case plots do not
    # replace less favorable or null cells.
    with PdfPages(OUT/'figures/factorial-used-colors-all-cells.pdf') as pdf:
        for n in NS:
            for d in DS:
                fig,axes=plt.subplots(3,3,figsize=(11,8),layout='constrained')
                for i,arch in enumerate(ARCHES):
                    for rep in range(3):
                        ax=axes[i,rep];g=next(g for g in gs if g['stage']=='main' and g['n']==n and g['delta']==d and g['rep']==rep)
                        a=runs[runs['graph']==g['id']];labels=['Luby']+[s['label'] for s in specs() if s['treatment']==arch]
                        values=a[a['model'].isin(labels)]['used_colors'];bins=np.arange(values.min()-.5,values.max()+1.5)
                        for k,label in enumerate(labels):
                            ax.hist(a[a['model']==label]['used_colors'],bins=bins,density=True,histtype='step',color='black' if k==0 else SEEDCOLORS[k-1],lw=1.2 if k==0 else .7)
                        ax.set(title=f'{NAMES[arch]} / graph {rep}',xlabel='U',ylabel='Mass')
                fig.suptitle(f'Full cohort: n={n}, Δ={d}; black=Luby, five colored curves=checkpoints')
                pdf.savefig(fig);plt.close(fig)

    fig,axes=plt.subplots(3,4,figsize=(12,8.5),layout='constrained',sharey=True)
    for i,arch in enumerate(ARCHES):
        for j,d in enumerate(DS):
            ax=axes[i,j]
            for k,s in enumerate([s for s in specs() if s['treatment']==arch]):
                a=diffs[(diffs['stratum']=='main')&(diffs['model']==s['label'])&(diffs['event']=='used_le_delta')&(diffs['delta']==d)]
                ax.scatter(a['n'],a['difference'],color=SEEDCOLORS[k],alpha=.25,s=10)
                yy=[C(s['label'],n,d)['estimate'] for n in NS]
                ax.plot(NS,yy,'o-',color=SEEDCOLORS[k],ms=3,lw=1,label=str(s['seed']))
            ax.axhline(0,color='.5',lw=.6);ax.set_xscale('log',base=2);ax.set_xticks(NS,labels=NS,rotation=30);ax.set_ylim(-1.05,1.05)
            ax.set(title=f'{NAMES[arch]} / Δ={d}',xlabel='n',ylabel='Event probability difference')
            if j==0:ax.legend(fontsize=6,ncol=2)
    fig.suptitle('7. All five checkpoints; faint points are individual graphs, lines are graph means')
    save(fig,'factorial-07-checkpoints')

    fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout='constrained',sharey=True)
    for ax,g in zip(axes.ravel(),targets):
        labels=['Luby']+[s['label'] for s in specs() if s['treatment']=='normalized']
        for k,label in enumerate(labels):
            r=lookup[(g['id'],label,'used_le_delta')];y=r['probability'];lo,hi=r['ci']
            ax.errorbar(k,y,yerr=[[max(0.,y-lo)],[max(0.,hi-y)]],fmt='o',color='black' if k==0 else SEEDCOLORS[k-1],capsize=3)
        ax.set(xticks=range(6),xticklabels=['Luby']+[str(s['seed']) for s in specs() if s['treatment']=='normalized'],ylabel='P(U ≤ 100)',ylim=(-.03,1.03),title=f'New graph r{g["rep"]}' if g['stage']=='main' else 'Historical graph; independent new runs')
        ax.tick_params(axis='x',rotation=25)
    fig.suptitle('8. E1 replication at n=512, Δ=100; no historical outcomes pooled')
    save(fig,'factorial-08-target-replication')

    probes=pd.read_csv(OUT/'candidate-probe-summary.csv');initial=probes[(probes['round']==0)&probes['graph'].str.startswith('main-')]
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    degreecolors=plt.get_cmap('viridis')(np.linspace(.1,.9,4));markers=('o','s','^','D')
    for j,arch in enumerate(ARCHES):
        ax=axes[0,j];lower=axes[1,j]
        for k,s in enumerate([s for s in specs() if s['treatment']==arch]):
            a=initial[(initial['model']==s['label'])&(initial['n']==512)].sort_values('delta')
            ax.plot(a['delta']+1,a['q_top']*(a['delta']+1),'o-',color=SEEDCOLORS[k],ms=3,label=str(s['seed']))
            for ni,n in enumerate(NS):
                for di,d in enumerate(DS):
                    q=initial[(initial['model']==s['label'])&(initial['n']==n)&(initial['delta']==d)].iloc[0]
                    value=C(s['label'],n,d,'uses_top_color')['estimate']
                    lower.scatter(q['q_top']*(d+1),value,color=degreecolors[di],marker=markers[ni],s=20,alpha=.7)
        ax.axhline(1,color='black',ls='--',lw=.8,label='Uniform teacher');ax.set_xscale('log');ax.set_xticks(np.array(DS)+1,labels=np.array(DS)+1)
        ax.set(title=NAMES[arch],xlabel='Palette size K',ylabel='Initial q(top) / (1/K)')
        if j==0:ax.legend(fontsize=6,ncol=2)
        lower.axhline(0,color='.5',lw=.6);lower.set(xlabel='Initial q(top) / (1/K)',ylabel='Final top-appearance probability difference',ylim=(-1.05,1.05))
    handles=[Line2D([0],[0],marker='o',ls='',color=c,label=f'Δ={d}') for c,d in zip(degreecolors,DS)]
    handles += [Line2D([0],[0],marker=m,ls='',color='.3',label=f'n={n}') for m,n in zip(markers,NS)]
    fig.legend(handles=handles,loc='outside lower center',ncol=8,fontsize=7)
    fig.suptitle('9. Candidate quadrature and independent final samples: association, not causal identification')
    save(fig,'factorial-09-candidates-and-global-law')

    tr=pd.read_csv(OUT/'trajectory-accounting.csv');target=next(g for g in targets if g['stage']=='main' and g['rep']==0)
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
    metrics=['candidate_top_fraction','same_state_teacher_top_fraction','committed_top_fraction','final_top_fraction']
    for ax,arch in zip(axes,ARCHES):
        for k,s in enumerate([s for s in specs() if s['treatment']==arch]):
            a=tr[(tr['graph']==target['id'])&(tr['model']==s['label'])]
            ax.plot(range(4),a[metrics].mean(),'o-',color=SEEDCOLORS[k],lw=1,ms=3,label=str(s['seed']))
            for x,metric in enumerate(metrics):ax.scatter(np.full(len(a),x),a[metric],s=8,alpha=.2,color=SEEDCOLORS[k])
        ax.set(xticks=range(4),xticklabels=['Proposals','Teacher on\nsame states','COMMIT','Final class'],ylabel='Top-color fraction',title=NAMES[arch]);ax.tick_params(axis='x',labelsize=7)
    axes[0].legend(fontsize=6)
    fig.suptitle('Read-only trajectory accounting: n=512, Δ=100, eight independent runs per checkpoint')
    save(fig,'factorial-10-proposal-commit-chain')

    secondary=[g for g in gs if g['stage']=='secondary'];sns=sorted(set(g['n'] for g in secondary))
    from .capacity import bound as capacity_bound
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,arch in zip(axes,ARCHES):
        labels=[s['label'] for s in specs() if s['treatment']==arch]
        z=np.zeros((len(sns),4))
        for i,n in enumerate(sns):
            for j,d in enumerate(DS):
                a=[g for g in secondary if g['n']==n and g['delta']==d]
                if not a:continue
                q,ql,qh=prob(labels,a,'used_le_delta');p,pl,ph=prob(['Luby'],a,'used_le_delta');z[i,j]=q-p
                annotation='A ≡ 1\n(list bound)' if capacity_bound(a[0]['degrees'])['A_deterministically_true'] else f'{q-p:+.2f}\n[{ql-ph:+.2f}, {qh-pl:+.2f}]'
                ax.text(j,i,annotation,ha='center',va='center',fontsize=7,color='white' if abs(q-p)>.65 else 'black')
        im=ax.imshow(z,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');ax.set(xticks=range(4),xticklabels=DS,yticks=range(len(sns)),yticklabels=sns,xlabel='Maximum degree Δ',ylabel='n',title=NAMES[arch])
    fig.colorbar(im,ax=axes,shrink=.85,label='Event probability difference')
    fig.suptitle('Secondary irregular cohort: one graph × five checkpoints; forced events labeled explicitly')
    save(fig,'factorial-11-irregular-cohort')

    intervention_path=OUT/'candidate-intervention/results.json'
    if intervention_path.exists() and read(intervention_path).get('rows'):
        rows=read(intervention_path)['rows']
        supplement=OUT/'candidate-intervention/witness-effects.json'
        if supplement.exists() and read(supplement).get('rows'):rows=read(supplement)['rows']
        gids=sorted(set(r['graph'] for r in rows))
        graphcolors=plt.get_cmap('tab10')(np.arange(len(gids)))
        offsets=np.linspace(-.24,.24,len(gids))
        fig,axes=plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
        for j,arch in enumerate(ARCHES):
            labels=[s['label'] for s in specs() if s['treatment']==arch]
            for gi,gid in enumerate(gids):
                selected={r['model']:r for r in rows if r['graph']==gid and r['architecture']==arch}
                xx=[i+offsets[gi] for i,label in enumerate(labels) if label in selected]
                vals=[selected[label] for label in labels if label in selected]
                axes[0,j].scatter(xx,[r['conditional_tv_upper'] for r in vals],s=25,color=graphcolors[gi])
                axes[1,j].scatter(xx,[r['intervention_tv_effect_lower'] for r in vals],s=25,color=graphcolors[gi])
            for ax in axes[:,j]:
                ax.set(xticks=range(5),xticklabels=[s['seed'] for s in specs() if s['treatment']==arch],ylim=(-.025,1.025))
                ax.tick_params(axis='x',rotation=30,labelsize=7)
            axes[0,j].axhline(.05,color='black',ls='--',lw=.8)
            axes[0,j].set(title=NAMES[arch],ylabel='Upper TV bound: intervened vs Luby')
            axes[1,j].set(xlabel='Checkpoint seed',ylabel='Lower TV bound: original vs intervened')
        handles=[Line2D([0],[0],marker='o',ls='',color=graphcolors[i],label='Historical graph' if gid.startswith('historical-') else 'New graph r'+gid.rsplit('r',1)[-1]) for i,gid in enumerate(gids)]
        fig.legend(handles=handles,loc='outside lower center',ncol=len(gids),fontsize=8)
        fig.suptitle('12. Exploratory candidate replacement; simultaneous 95% witness/coupling bounds')
        save(fig,'factorial-12-candidate-intervention')
    atomic_json(OUT/'figure-manifest.json',dict(required_figures=9,figures=[p.stem for p in sorted((OUT/'figures').glob('*.png'))],
                                               supplementary_occupancy_pages=16,generated=now()))
    print('Required figure pairs and exploratory supplements generated',flush=True)

if __name__=='__main__':main()
