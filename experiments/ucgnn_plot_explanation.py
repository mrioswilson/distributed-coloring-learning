"""Static scientific figures and descriptive summaries of explanatory probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E1='#2367a3'; A1='#bd5025'; TEACHER='#555555'


def decorate(ax,title,xlabel,ylabel):
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)


def save(fig,root,name):
    fig.tight_layout()
    fig.savefig(root/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(root/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); root=p.parse_args().root
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig,ax=plt.subplots(figsize=(12,6)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    def box(x,y,w,label,learned=False):
        ax.add_patch(FancyBboxPatch((x,y),w,.12,boxstyle='round,pad=0.008',
                                   facecolor='#e9f1f8' if learned else '#f6f6f6',
                                   edgecolor=E1 if learned else '#666666',linewidth=1.1))
        ax.text(x+w/2,y+.06,label,ha='center',va='center',fontsize=9)
    def arrow(start,end):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,color='#555555',linewidth=1.1))
    box(.03,.81,.17,'Estado q y paleta K\npor nodo')
    box(.27,.81,.22,'φ1 → MAX vecinos\nComunicación 1 · [N,K,32]',True)
    box(.55,.81,.17,'ψ1 → h → α\npor nodo y color',True)
    box(.78,.81,.19,'Suma + prefijo / masa\nL,U locales al nodo')
    box(.03,.52,.17,'r_p propio y u')
    box(.27,.52,.22,'MLP participación\numbral → p',True)
    box(.57,.52,.40,'Scorer(h,α,L,U,r_c propio)\nargmax en la paleta → candidato',True)
    box(.06,.24,.27,'Estado y propuestas por color\n[u, own, p, offer]')
    box(.40,.24,.27,'φ2 → MAX vecinos → ψ2\nComunicación 2 · [N,K,32]',True)
    box(.75,.24,.22,'Leer ranura candidata\nMLP COMMIT → logit',True)
    box(.75,.025,.22,'Si participa y COMMIT>0:\nfijar color permanentemente')
    for start,end in [((.205,.87),(.26,.87)),((.495,.87),(.54,.87)),((.725,.87),(.77,.87)),
                      ((.205,.58),(.26,.58)),((.88,.80),(.88,.65)),((.64,.80),(.64,.65)),
                      ((.38,.51),(.25,.37)),((.70,.51),(.30,.37)),((.335,.30),(.39,.30)),
                      ((.675,.30),(.74,.30)),((.86,.23),(.86,.155))]: arrow(start,end)
    ax.text(.02,.07,'Azul: bloques con parámetros aprendidos; MAX es fijo. Gris: protocolo fijo.\nLas dos comunicaciones usan sólo aristas del grafo; las sumas/prefijos son locales.',fontsize=9,va='center')
    ax.set_title('UCGNN: una ronda con pesos compartidos entre nodos y colores',fontsize=13,pad=14)
    save(fig,root,'architecture')
    data=pd.read_csv(root/'selection-map.csv'); bulk=data[data.bulk & (data.model=='E1')]
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,k in zip(axes.flat[:3],(4,11,201)):
        r=bulk[bulk.palette==k]
        ax.step(r.rc,r.teacher,where='mid',color=TEACHER,ls='--',lw=1.4,label='Profesor')
        ax.step(r.rc,r.candidate,where='mid',color=E1,lw=1.1,label='E1-0')
        decorate(ax,f'K = k = {k}',r'$r_c$','Color propuesto (desde 0)'); ax.legend(fontsize=8)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax=axes.flat[3]
    for k,style in ((201,'-'),(501,'--')):
        r=bulk[bulk.palette==k]
        ax.plot(r.rc,r.candidate-r.teacher,style,lw=1,color=E1 if k==201 else A1,label=f'K={k}')
    decorate(ax,'Desacuerdo de rango, sin colores prohibidos',r'$r_c$','Candidato E1 − profesor'); ax.legend(fontsize=8)
    save(fig,root,'selection-mapping')

    fig,axes=plt.subplots(3,1,figsize=(10,8))
    for ax,name,k in zip(axes,('E1','E1','A1'),(201,101,101)):
        r=pd.read_csv(root/f'proposal-law-{name}-K{k}.csv')
        color=E1 if name=='E1' else A1
        ax.bar(r.color,100*r.probability,width=.9,color=color,label=f'{name}-0, malla de 20.001 valores')
        ax.axhline(100/k,color=TEACHER,ls='--',lw=1,label='Uniforme ideal')
        decorate(ax,f'{name}-0: K = k = {k}','Color','Propuestas (%)'); ax.legend(fontsize=8,loc='upper right')
    save(fig,root,'proposal-distribution')

    r=pd.read_csv(root/'round-traces.csv'); matched=pd.read_csv(root/'matched-state-candidate-comparison.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for col,n in enumerate((160,640)):
        case=f'REGULAR-n{n}'
        for policy,color,ls in (('A1',A1,'-'),('E1',E1,'--')):
            d=r[(r['case']==case)&(r.trajectory==policy)&(r.policy==policy)]
            axes[0,col].step(np.r_[0,d['round']],np.r_[n,d.pending_after],where='post',label=policy,color=color,ls=ls)
            d=matched[(matched['case']==case)&(matched.policy==policy)&(matched['round']<=20)]
            axes[1,col].plot(d['round'],d.exact_safe_proposals,label=policy,color=color,ls=ls,marker='.',ms=3)
        decorate(axes[0,col],f'Trayectorias propias, Δ=100, N={n}','Ronda','Nodos pendientes')
        decorate(axes[1,col],f'Mismo estado A1 y misma participación, N={n}','Ronda del trazado A1','Propuestas seguras (observador)')
        axes[0,col].legend(); axes[1,col].legend()
    save(fig,root,'collision-stall')

    curve=pd.read_csv(root/'continuous-score-maxima.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for label,caption,style in (('limit','Ancho → 0','-'),('K201','Ancho 1/201','--'),('K501','Ancho 1/501',':')):
        axes[0].plot(curve.rc,curve[f'x_{label}']-curve.rc,ls=style,label=caption,lw=1.5)
    decorate(axes[0],'Cuantil preferido por el scorer continuo',r'$r_c$',r'$x^*(r_c,w)-r_c$'); axes[0].legend(fontsize=8)
    predictions=[]
    for k,color in ((201,E1),(501,A1)):
        d=bulk[bulk.palette==k].reset_index(drop=True)
        pred=np.clip(np.floor(k*curve.x_limit.to_numpy()).astype(int),0,k-1)
        err=d.candidate.to_numpy()-pred
        axes[1].plot(d.rc,err,color=color,lw=.8,label=f'K={k}')
        predictions.append({'palette':k,'probe':'all-uncolored','rule':'floor(K*x_limit(rc)); clipped to public palette',
                            'exact_matches':int((err==0).sum()),'draws':len(err),'max_rank_error':int(np.max(np.abs(err))),
                            'mean_abs_rank_error':float(np.mean(np.abs(err)))})
    decorate(axes[1],'Prueba del mismo mapa sin ajuste por K',r'$r_c$',r'Candidato − floor(K·x*(r,0))'); axes[1].legend(fontsize=8)
    save(fig,root,'preferred-quantile')
    (root/'quantile-hypothesis-check.json').write_text(json.dumps(predictions,indent=2)+'\n')
    # Small, human-readable mapping table.
    samples=[]
    for k in (4,11,201,501):
        d=bulk[bulk.palette==k].reset_index(drop=True)
        for i in (0,50,100,250,500,750,900,950,1000): samples.append(d.iloc[i].to_dict())
    pd.DataFrame(samples).to_csv(root/'selection-examples.csv',index=False)
    # Descriptive first-round comparisons, avoiding cross-trajectory pooling.
    first=r[(r.trajectory=='A1')&(r['round']==1)]
    first.to_csv(root/'first-round-matched-state.csv',index=False)
    print('Architecture diagram, four figures and descriptive hypothesis checks written.')


if __name__=='__main__': main()
