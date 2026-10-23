"""Reproducible scientific figures and reports from archived result objects."""
from pathlib import Path
import shutil,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

NAMES={'normalized':'NORMALIZED (E1)','deepsets':'DeepSets','transformer':'LocalPaletteTransformer'}
COLORS={'old_hamming':'#777777','multi_max':'#bc7728','omnibus_v1':'#a76aaf','omnibus':'#1769aa','classifier':'#2b8c56'}
LABELS={'old_hamming':'Old Hamming MMD','multi_max':'Global multiscale max','omnibus_v1':'Honest omnibus V1','omnibus':'Honest omnibus V2','classifier':'Classifier'}

def save(fig,name):
    d=OUT/'figures';d.mkdir(exist_ok=True)
    fig.savefig(d/f'{name}.png',dpi=160,bbox_inches='tight');fig.savefig(d/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def curve(ax,frame,case,methods=('old_hamming','omnibus_v1','omnibus','classifier')):
    for method in methods:
        a=frame[(frame['case']==case)&(frame.method==method)].sort_values('m')
        if len(a):
            style='--' if method=='classifier' else (':' if method=='omnibus_v1' else '-')
            ax.plot(a.m,a.rate,marker='o',ls=style,ms=3,color=COLORS[method],label=LABELS[method]);ax.fill_between(a.m,a.ci_low,a.ci_high,color=COLORS[method],alpha=.09)
    ax.set(xscale='log',ylim=(-.03,1.03),xlabel='Total colorings per generator',ylabel='Detection probability',title=case)

def power_heatmap(v):
    cases=list(dict.fromkeys(v[~v['case'].str.endswith('_null')]['case']));fig,axes=plt.subplots(1,3,figsize=(14,5.2),layout='constrained')
    for ax,method in zip(axes,('old_hamming','omnibus_v1','omnibus')):
        a=v[(v.method==method)&v['case'].isin(cases)].pivot(index='case',columns='m',values='rate').reindex(cases)
        im=ax.imshow(a.values,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set(xticks=range(len(a.columns)),xticklabels=a.columns,yticks=range(len(cases)),yticklabels=cases if ax is axes[0] else [],xlabel='Total colorings per generator',title=LABELS[method])
        for (i,j),value in np.ndenumerate(a.values):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=7,color='black' if value>.6 else 'white')
    fig.colorbar(im,ax=axes,label='Detection probability; 30 independent trials');save(fig,'figure8-sample-size-power')

def representative_witnesses(w):
    chosen=w[w['reject']&w['witness_reject']].copy()
    chosen['abs_difference']=chosen.difference.abs()
    chosen['event_type']=chosen.event.map(lambda e:e.get('name') or ('vertices',tuple(e.get('ids',[]))))
    chosen['scope']=chosen.event.map(lambda e:len(e.get('ids',[])) or 'structural')
    rows=[]
    for arch in NAMES:
        a=chosen[chosen.architecture==arch].sort_values('abs_difference',ascending=False)
        first=a.drop_duplicates('scope').head(3)
        rows.append(pd.concat([first,a]).drop_duplicates('event_type').head(3))
    return pd.concat(rows,ignore_index=True)

def figures(p,w,s,v):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,c in zip(axes.flat,('localized','correlation_remote','higher_order','higher_order_remote','diffuse','rare')):curve(ax,v,c)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle('Independent validation 2: 30 trials per point; nominal alpha .05; pointwise Wilson 95% intervals')
    save(fig,'figure1-power-comparison')

    audit=read(OUT/'old-statistical-audit.json')
    nullU=[read(f)['global_statistics']['old_hamming'] for f in (OUT/'benchmark-trials').glob('validation2-star_null-D200-eNone-m250-t*.json')]
    altU=[read(f)['global_statistics']['old_hamming'] for f in (OUT/'benchmark-trials').glob('validation2-localized-D200-eNone-m250-t*.json')]
    pd.DataFrame([dict(law=label,trials=len(a),mean=float(np.mean(a)),sd=float(np.std(a,ddof=1)),q05=float(np.quantile(a,.05)),q50=float(np.median(a)),q95=float(np.quantile(a,.95))) for label,a in [('A versus A',nullU),('A versus B',altU)]]).to_csv(OUT/'old-star-statistic-distributions.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    bins=np.linspace(min(nullU+altU),max(nullU+altU),24)
    axes[0].hist(nullU,bins=bins,density=True,alpha=.5,color='#888888',label='Known null A/A (400 trials)');axes[0].hist(altU,bins=bins,density=True,alpha=.5,color='#1769aa',label='Known alternative A/B (30 trials)');axes[0].axvline(audit['star_exact_mmd2'],color='#bf3b3b',label='Population MMD squared')
    axes[0].set(title='Old statistic on fresh controls, m=250',xlabel='Unbiased MMD squared',ylabel='Density');axes[0].legend(fontsize=6)
    axes[0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    curve(axes[1],v,'localized',methods=('old_hamming','multi_max','omnibus'))
    axes[1].legend(fontsize=6,loc='center right')
    axes[2].bar(['Artificial A','Artificial B'],[100/199,1.],color=['#777777','#1769aa']);axes[2].set(ylim=(0,1.05),ylabel='Exact event probability',title='Event: center color <=101')
    fig.suptitle('Proper local-list star colorings; exact TV=99/199; irrelevant iid leaf noise')
    save(fig,'figure2-known-star')

    stress=pd.read_csv(OUT/'benchmark-sensitivity2.csv');fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for method in ('old_hamming','omnibus'):
        for m,style in [(250,'-'),(1000,'--')]:
            a=stress[(stress['case']=='localized')&(stress.method==method)&(stress.m==m)][['D','rate','ci_low','ci_high']]
            b=v[(v['case']=='localized')&(v.method==method)&(v.m==m)][['D','rate','ci_low','ci_high']]
            a=pd.concat([a,b]).sort_values('D')
            axes[0].plot(a.D,a.rate,'o'+style,color=COLORS[method],label=f'{LABELS[method]}, m={m}')
    axes[0].set(xscale='log',ylim=(-.03,1.03),xlabel='Noisy leaves / center degree D',ylabel='Detection probability',title='Noisy-star scaling (n=D+1)');axes[0].legend(fontsize=6)
    for ax,case in zip(axes[1:],('marginal','diffuse')):
        for method in ('old_hamming','omnibus'):
            a=stress[(stress['case']==case)&(stress.method==method)&(stress.m==1000)].sort_values('effect')
            b=v[(v['case']==case)&(v.method==method)&(v.m==1000)].copy();b['effect']=.85 if case=='marginal' else .55
            a=pd.concat([a,b]).sort_values('effect')
            ax.errorbar(a.effect,a.rate,yerr=np.maximum(0,np.array([a.rate-a.ci_low,a.ci_high-a.rate])),fmt='o-',color=COLORS[method],capsize=2,label=LABELS[method])
        ax.set(ylim=(-.03,1.03),xlabel='Bernoulli alternative probability',ylabel='Detection probability',title=case+'; m=1000')
    save(fig,'figure3-localization-stress')

    syn=v[v['case'].str.endswith('_null')].copy();syn['null_law']=syn['case']
    actual=pd.read_csv(OUT/'actual-null-validation2.csv');actual['null_law']='actual '+actual.label
    extra=[]
    if (OUT/'benchmark-null-replication2.csv').exists():
        rep=pd.read_csv(OUT/'benchmark-null-replication2.csv');rep['null_law']='star_null (independent N=4000)';extra=[rep]
    null=pd.concat([syn,actual,*extra],ignore_index=True);null.to_csv(OUT/'all-null-calibration.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    laws=list(dict.fromkeys(null.null_law))
    for ax,method in zip(axes,('old_hamming','omnibus','classifier')):
        a=null[null.method==method].set_index('null_law').reindex(laws);yy=np.arange(len(a))
        ax.errorbar(a.rate,yy,xerr=np.maximum(0,np.array([a.rate-a.ci_low,a.ci_high-a.rate])),fmt='o',capsize=2,color=COLORS[method]);ax.axvline(.05,color='black',ls=':')
        ax.set(yticks=yy,yticklabels=laws if ax is axes[0] else [],xlabel='False rejection rate',title=LABELS[method],xlim=(-.005,max(.12,float(a.ci_high.max())+.005)))
    fig.suptitle('Null calibration: 400 trials per original law; separate star replication 4000; pointwise Wilson 95% intervals')
    save(fig,'figure4-type-I-error')
    if p is None:
        power_heatmap(v);return

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');xx=np.arange(3);width=.25
    for i,(field,label,color) in enumerate([('historical_reject','Historical m=250','#aaaaaa'),('old_reject','Old test, fresh m=500','#666666'),('reject','Omnibus, fresh total m=500','#1769aa')]):
        axes[0].bar(xx+(i-1)*width,[int(p[p.architecture==a][field].sum()) for a in NAMES],width,label=label,color=color)
    axes[0].set(xticks=xx,xticklabels=['E1','DeepSets','Transformer'],ylabel='Holm rejections /265',title='Counts describe this fixed cohort');axes[0].legend(fontsize=7)
    e=p[p.architecture=='normalized'];M=np.array([[int(((e.old_reject==old)&(e.reject==new)).sum()) for new in (False,True)] for old in (False,True)])
    axes[1].imshow(M,cmap='Blues');axes[1].set(xticks=[0,1],xticklabels=['New: no rejection','New: rejection'],yticks=[0,1],yticklabels=['Old: no rejection','Old: rejection'],title='E1; matched fresh 500-row banks')
    for (i,j),value in np.ndenumerate(M):axes[1].text(j,i,str(value),ha='center',va='center',color='white' if value>M.max()*.6 else 'black')
    save(fig,'figure5-old-new-rejections')

    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');colors=['#1769aa','#d36c22','#6c43a2']
    for j,n in enumerate((256,512)):
        for arch,c in zip(NAMES,colors):
            a=p[(p.architecture==arch)&(p.family=='regular')&(p.n==n)]
            b=a.groupby('delta',as_index=False).agg(rejections=('reject','sum'))
            axes[0,j].plot(b.delta,b.rejections,'o-',label=NAMES[arch],color=c)
            for _,group in a.groupby('training_seed'):
                group=group.sort_values('delta');axes[1,j].plot(group.delta,group.old_mmd2_unbiased,'o-',ms=3,alpha=.5,color=c)
        axes[0,j].set(xscale='log',ylim=(-.2,5.2),title=f'Fixed regular graphs; n={n}',xlabel='Actual maximum degree',ylabel='New rejections among five policies')
        axes[1,j].set(xscale='log',xlabel='Actual maximum degree',ylabel='Graph-specific unbiased MMD squared')
        for ax in axes[:,j]:ax.set_xticks([2,3,5,10,20,50,100,200],labels=['2','3','5','10','20','50','100','200'])
    axes[0,0].legend(fontsize=7);fig.suptitle('Rejection frequency is not distance; kernel effects are not universally normalized across graph spaces')
    save(fig,'figure6-degree-architecture')

    chosen=representative_witnesses(w)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    if len(chosen):
        yy=np.arange(len(chosen));labels=[]
        for i,r in enumerate(chosen.itertuples()):
            for shift,value,ci,color in [(-.13,r.probability_Luby,r.ci_Luby,'#777777'),(.13,r.probability_model,r.ci_model,'#1769aa')]:
                ax.errorbar(value,i+shift,xerr=[[value-ci[0]],[ci[1]-value]],fmt='o',color=color,capsize=2)
            event=r.event;name='vertices '+','.join(map(str,event.get('ids',[]))) if event.get('ids') else event.get('name','structural event')
            labels.append(r.graph+' / '+r.model+' / '+name)
        ax.set(yticks=yy,yticklabels=labels,xlim=(-.02,1.02),xlabel='Independent reserved-bank event probability',title='Luby (gray) versus network (blue); simultaneous 95% CP bounds')
    else:ax.text(.5,.5,'No independent explanatory event passed its criterion',ha='center');ax.set_axis_off()
    save(fig,'figure7-independent-witnesses')

    power_heatmap(v)

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained');smallnames=[g['id'] for g in graphs() if g['stage']=='small'];ymax=max(.3,float(max(ci[1] for ci in s.simultaneous_tv_ci))+.02)
    for ax,arch in zip(axes,NAMES):
        a=s[s.architecture==arch]
        for k,training in enumerate(range(41000,41005)):
            vals=[];cis=[];pos=[]
            for j,gid in enumerate(smallnames):
                b=a[(a.graph==gid)&(a.model==f'{arch}-s{training}')]
                if len(b):
                    r=b.iloc[0];vals.append(r.tv_estimate_vs_exact_eventual_teacher);cis.append(r.simultaneous_tv_ci);pos.append(j+(k-2)*.12)
            vals=np.array(vals);ci=np.array(cis);ax.errorbar(pos,vals,yerr=[vals-ci[:,0],ci[:,1]-vals],fmt='o',ms=3,capsize=1,lw=.6,alpha=.7,label=f'seed {training}')
        ax.axhline(.05,color='black',ls=':');ax.set(xticks=range(len(smallnames)),xticklabels=smallnames,ylabel='TV estimate and simultaneous interval',title=NAMES[arch],ylim=(0,ymax))
        ax.tick_params(axis='x',rotation=40)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle('Small exact teacher laws; five checkpoints separately; prospective equivalence tolerance .05; M=20,000')
    save(fig,'figure9-small-exact-reference')

def event_text(d):
    if d['kind']=='threshold':return f"c[{d['ids'][0]}] <= {d['threshold']}"
    if d['kind']=='categorical':return 'tuple at '+str(d['ids'])+' belongs to '+str([k for k,s in zip(d['keys'],d['scores']) if s>0])
    threshold=d.get('event_threshold',d.get('threshold'))
    if d.get('name')=='used_colors':threshold=int(np.floor(threshold))
    return d.get('name','structural statistic')+' <= '+str(threshold)

def descriptive_summaries(p,w,s,v):
    """Report fixed analyses; never select or alter a confirmatory decision."""
    grid=[]
    for case in dict.fromkeys(v[~v['case'].str.endswith('_null')]['case']):
        r={'case':case}
        for method in ('old_hamming','omnibus','classifier'):
            a=v[(v['case']==case)&(v.method==method)&(v.ci_low>=.8)].sort_values('m')
            r[method]=int(a.m.iloc[0]) if len(a) else None
        grid.append(r)
    pd.DataFrame(grid).to_csv(OUT/'sample-budgets-for-80-percent-power.csv',index=False)
    power='| Alternativa | MMD antiguo | Omnibus V2 | Clasificador |\n|---|---:|---:|---:|\n'
    for r in grid:power+='| '+r['case']+' | '+' | '.join(str(r[k]) if r[k] is not None else '>2000 / no acreditado' for k in ('old_hamming','omnibus','classifier'))+' |\n'
    bench=pd.read_csv(OUT/'benchmark-validation2-trials.csv').drop_duplicates(['case','m','trial'])
    bench=bench[~bench['case'].str.endswith('_null')]
    cost=bench.groupby('m',as_index=False)[['omnibus_seconds','quadratic_seconds','seconds']].median()
    cost.to_csv(OUT/'benchmark-computation-cost.csv',index=False)
    timing='| m | Omnibus V2 (s) | Cuatro MMD globales juntos (s) | Pipeline comparativo completo (s) |\n|---:|---:|---:|---:|\n'
    for r in cost.itertuples():timing+=f'| {r.m} | {r.omnibus_seconds:.3f} | {r.quadratic_seconds:.3f} | {r.seconds:.3f} |\n'
    measured=p.assign(robust_seconds=p.robust.map(lambda r:r['seconds']),old_seconds=p.old.map(lambda r:r['seconds']))
    measured.groupby('architecture',as_index=False).agg(median_robust_seconds=('robust_seconds','median'),median_old_seconds=('old_seconds','median'),sum_robust_seconds=('robust_seconds','sum'),sum_old_seconds=('old_seconds','sum')).to_csv(OUT/'confirmatory-computation-cost.csv',index=False)
    def category(d):
        if d['kind'] in ('categorical','threshold'):
            return {1:'vértice individual',2:'par de vértices',3:'triple de vértices'}.get(len(d['ids']),'región local')
        return 'resumen estructural'
    explained=w[w.reject&w.witness_reject].copy();explained['category']=explained.event.map(category)
    rows=explained.groupby(['architecture','category'],as_index=False).agg(cells=('model','size'),median_absolute_probability_gap=('difference',lambda x:float(x.abs().median())))
    rows.to_csv(OUT/'witness-categories.csv',index=False)
    witnesses='| Arquitectura | Tipo de evento validado | Celdas | Mediana de diferencia absoluta de probabilidad |\n|---|---|---:|---:|\n'
    for r in rows.itertuples():witnesses+=f'| {NAMES[r.architecture]} | {r.category} | {r.cells} | {r.median_absolute_probability_gap:.3f} |\n'
    if not len(rows):witnesses+='| — | Ningún testigo validado | 0 | — |\n'
    small=s.groupby(['graph','architecture'],as_index=False).agg(approximately_equivalent=('approximate_equivalence','sum'),exact_equality_rejected=('reject','sum'),min_estimated_tv=('tv_estimate_vs_exact_eventual_teacher','min'),max_estimated_tv=('tv_estimate_vs_exact_eventual_teacher','max'))
    small.to_csv(OUT/'small-reference-summary.csv',index=False)
    smallarch=s.assign(upper_tv=s.simultaneous_tv_ci.map(lambda ci:ci[1])).groupby('architecture',as_index=False).agg(cells=('model','size'),approximately_equivalent=('approximate_equivalence','sum'),max_upper_tv=('upper_tv','max'))
    smallarch.to_csv(OUT/'small-reference-by-architecture.csv',index=False)
    smalltable='| Arquitectura | Casos pequeños con TV<=.05 respaldada | Mayor límite superior de TV entre sus 35 casos |\n|---|---:|---:|\n'
    for r in smallarch.itertuples():smalltable+=f'| {NAMES[r.architecture]} | {r.approximately_equivalent}/{r.cells} | {r.max_upper_tv:.4f} |\n'
    paired=[]
    for arch in NAMES:
        a=p[p.architecture==arch]
        paired.append(dict(architecture=arch,both=int((a.reject&a.old_reject).sum()),new_only=int((a.reject&~a.old_reject).sum()),old_only=int((~a.reject&a.old_reject).sum())))
    pd.DataFrame(paired).to_csv(OUT/'paired-primary-decisions.csv',index=False)
    pairtable='| Arquitectura | Ambos rechazan | Solo omnibus | Solo MMD antiguo |\n|---|---:|---:|---:|\n'
    for r in paired:pairtable+=f'| {NAMES[r["architecture"]]} | {r["both"]} | {r["new_only"]} | {r["old_only"]} |\n'
    joined=explained.merge(p[['graph','model','old_reject']],on=['graph','model'])
    localized=int(((joined.architecture=='normalized')&~joined.old_reject&joined.category.isin(['vértice individual','par de vértices','triple de vértices'])).sum())
    return dict(power=power,timing=timing,witnesses=witnesses,new_localized_e1=localized,smalltable=smalltable,pairtable=pairtable)

def report():
    primary=read(OUT/'primary.json');p=pd.DataFrame(primary);w=pd.DataFrame(read(OUT/'explanatory-witnesses.json'));s=pd.DataFrame(read(OUT/'small-reference.json'))
    secondary=pd.DataFrame(read(OUT/'secondary.json'))
    secondary_summary=secondary.groupby(['total_per_generator','architecture'],as_index=False).agg(cells=('p','size'),new_rejections=('reject','sum'),old_rejections=('old_reject','sum'))
    secondary_summary.to_csv(OUT/'secondary-by-size-architecture.csv',index=False)
    secondary.groupby(['total_per_generator','architecture','training_seed'],as_index=False).agg(cells=('p','size'),new_rejections=('reject','sum'),old_rejections=('old_reject','sum')).to_csv(OUT/'secondary-by-size-checkpoint.csv',index=False)
    secondary_table='| m previsto | Arquitectura | Celdas | MMD antiguo | Omnibus nuevo |\n|---:|---|---:|---:|---:|\n'
    for r in secondary_summary.itertuples():secondary_table+=f'| {r.total_per_generator} | {NAMES[r.architecture]} | {r.cells} | {r.old_rejections} | {r.new_rejections} |\n'
    w['tv_lower_bound_from_event']=[max(0.,ci[0],-ci[1]) for ci in w.difference_ci]
    w.to_csv(OUT/'witness-TV-lower-bounds.csv',index=False)
    v=pd.read_csv(OUT/'benchmark-validation2.csv');audit=read(OUT/'audit.json');accept=read(OUT/'instrument-acceptance.json');old=read(OUT/'old-statistical-audit.json')
    figures(p,w,s,v)
    summaries=descriptive_summaries(p,w,s,v)
    nulls=pd.read_csv(OUT/'all-null-calibration.csv');worst=nulls[nulls.method=='omnibus'].sort_values('rate',ascending=False).iloc[0]
    replication=read(OUT/'null-replication-diagnostic.json')['target']
    p.groupby(['architecture','training_seed'],as_index=False).agg(tests=('p','size'),new_rejections=('reject','sum'),matched_old_rejections=('old_reject','sum'),historical_rejections=('historical_reject','sum'),median_old_unbiased_mmd2=('old_mmd2_unbiased','median')).to_csv(OUT/'by-training-checkpoint.csv',index=False)
    p.groupby(['architecture','family','n','delta'],as_index=False).agg(tests=('p','size'),new_rejections=('reject','sum'),matched_old_rejections=('old_reject','sum'),median_old_unbiased_mmd2=('old_mmd2_unbiased','median')).to_csv(OUT/'by-degree-family-size.csv',index=False)
    p[p.reject].to_csv(OUT/'rejected.csv',index=False);p[~p.reject].to_csv(OUT/'not-rejected.csv',index=False)
    e=p[p.architecture=='normalized'];e.to_csv(OUT/'e1-transitions.csv',index=False)
    newe=int((e.reject&~e.historical_reject).sum());only=int((e.reject&~e.old_reject).sum())
    w[w.reject&w.witness_reject].to_csv(OUT/'independently-validated-witnesses.csv',index=False)
    w[w.witness_reject].to_csv(OUT/'all-independently-significant-events.csv',index=False)
    total=int(p.reject.sum());matched=int(p.old_reject.sum());equiv=int(s.approximate_equivalence.sum())
    confirm_counts=next(r['statuses'] for r in audit['phase_summaries'] if r['phase']=='confirmatory')
    regular_tf=p[(p.architecture=='transformer')&(p.family=='regular')&(p.n==512)].groupby('delta')['reject'].sum().to_dict()
    counts=[]
    for arch in NAMES:
        a=p[p.architecture==arch];counts.append((arch,int(a.historical_reject.sum()),int(a.old_reject.sum()),int(a.reject.sum())))
    table='| Arquitectura | Histórico m=250 | MMD antiguo, banco nuevo m=500 | Omnibus nuevo, total m=500 |\n|---|---:|---:|---:|\n'
    for arch,h,o,n in counts:table+=f'| {NAMES[arch]} | {h}/265 | {o}/265 | {n}/265 |\n'
    lines=['# Pruebas robustas de distribuciones de coloraciones completas','',
      f'La evaluación confirmatoria independiente completó **795 contrastes** sobre 53 entradas de grafos fijos (51 grafos etiquetados únicos) y quince políticas congeladas. El nuevo omnibus rechazó **{total}/795** tras Holm al 5%; el MMD anterior aplicado a los mismos bancos nuevos rechazó **{matched}/795**. Las observaciones son vectores completos, condicionadas a completar en un máximo de 128 rondas. Los recuentos de rechazos no son medidas de distancia.','',table,
      'La comparación histórica usó 250 ejecuciones; el presupuesto principal nuevo es de 500 ejecuciones por generador. En bancos sin fallos, el omnibus utiliza 250 coloraciones para selección y 250 para evaluación. Cuando hay cutoffs u otros fallos se utilizan solo las completadas de ese prefijo; actual_m_Luby, actual_m_model y selection_per_class/evaluation_per_class registran los tamaños efectivos. No se reemplazan fallos retrospectivamente. Por eso el cambio respecto del resultado histórico mezcla procedimiento, tamaño y muestras nuevas; la comparación entre procedimientos sobre el mismo banco separa el presupuesto muestral de esa comparación. Ningún checkpoint se mezcló con otro.','',
      '## P1. Por qué falló el control estrella','',
      f'Las dos leyes son coloraciones propias de una estrella con 200 hojas binarias iid. Solo difiere el centro: uniforme en 199 colores frente a uniforme en un subconjunto de 100. Su TV es 99/199, pero MMD²={old["star_exact_mmd2"]:.12g}. A m=250, su señal es apenas {old["star"][1]["signal_over_null_sd"]:.4f} desviaciones nulas; la desviación exacta es {old["star"][1]["exact_null_sd"]:.9g}. Las hojas irrelevantes aportan variabilidad, mientras la normalización por n y el soporte grande atenúan la señal del centro. Se reprodujeron exactamente los 100 valores-p originales. No fue un problema causado únicamente por Holm: el control ya fallaba al nivel nominal .05. Véanse OLD_TEST_STATISTICAL_AUDIT.md, la demostración y las figuras 1–3.','',
      '## P2. Calibración y Holm anteriores','',
      'El test anterior empleó el U-estadístico no sesgado, conservando valores negativos, kernel exp(-d_H/n), gamma=1 prefijado, permutaciones uniformes de etiquetas de vectores completos, empates inclusivos y corrección de una unidad. Los 19.999 sorteos permiten p mínimo .00005, inferior a .05/795=.000062893. Se recalcularon exactamente los 795 ajustes Holm. No se identificó inflación por ese procedimiento ni insuficiencia de resolución primaria; muchos resultados saturaron en el mínimo. Que el kernel sea característico no implica buena potencia con estas muestras.','',
      '## P3. Método seleccionado, nivel y coste','',
      'Se seleccionó el omnibus honesto V2 antes de su validación independiente. En una mitad aprende testigos locales, de pares/triples, estructurales, marginales densos y cuatro testigos de kernels globales. La búsqueda de pares por covarianzas utiliza solo esa mitad. En la otra mitad, un máximo conjunto incluye dichos testigos y el MMD característico de vectores completos. Cada permutación recalcula el máximo; no se ejecuta una colección de pruebas al 5% sin corrección. Condicional a la selección, las etiquetas de evaluación son intercambiables bajo H0, lo que justifica el nivel para cualquier ley del espacio finito.','',
      f'La auditoría de validación independiente incluyó {accept["calibration_cells"]} celdas método/ley nula, con 400 ensayos por ley. Superó el criterio binomial múltiple prefijado y las metas de potencia. Los intervalos de tasa son puntuales; una simulación finita no certifica universalmente el nivel. La demostración por permutaciones aporta la garantía. Los controles degenerados se conservaron. Se compararon MMD global multiescala por máximo y softmax, versiones locales/estructurales, clasificadores simples con entrenamiento/validación/evaluación separados y LRT categórico calibrado. Las adaptaciones no se presentan como implementaciones exactas de MMDAgg o MMD-FUSE ni heredan automáticamente sus garantías minimax.','',
      f'La mayor frecuencia nula observada para V2 fue {worst.rate:.3f} en {worst.null_law} ({int(worst.rejections)}/{int(worst.trials)}; Wilson95 [{worst.ci_low:.3f}, {worst.ci_high:.3f}]). Ese intervalo puntual queda por encima de .05. Su prueba binomial unilateral no superó el umbral múltiple prospectivo de .01/{accept["calibration_cells"]}; no se ocultó esa fluctuación ni se usó para reajustar el método. La simulación no demuestra que la tasa real sea exactamente .05; la garantía de nivel depende de las hipótesis de independencia e intercambiabilidad documentadas.','',
      f'Para investigar esa celda se preregistró después de congelar el método una réplica independiente de 4.000 ensayos CPU, antes de generar sus datos y sin consultar valores-p neuronales. Dio {replication["rejections"]}/4000 rechazos ({replication["rate"]:.4f}; Wilson95 [{replication["ci_low"]:.4f}, {replication["ci_high"]:.4f}]); probabilidad binomial unilateral de excedencia {replication["null_exceedance_p"]:.4g}. El criterio diagnóstico prefijado fue .01. Este resultado adicional no reemplaza el gate original, no cambia la elección del método y no certifica otras leyes. Plan y commit: ROBUST_NULL_REPLICATION_PLAN.md, 9bd3398; datos: null-replication-diagnostic.json.','',
      'Los CSV de validación retienen los once componentes/candidatos originales y la comparación con V1, potencia, incertidumbre y tiempos. omnibus_seconds mide el procedimiento completo; quadratic_seconds mide conjuntamente las cuatro escalas globales, no el coste aislado de una sola escala. La elección priorizó sensibilidad verificable y nivel correcto, con coste CPU compatible con la batería. No existe un ganador universal.','',
      'Medianas observadas en las alternativas de la validación independiente, con B=199 y un hilo BLAS por proceso. Los tiempos incluyen operaciones diferentes en cada columna y no deben extrapolarse linealmente a B=19.999; confirmatory-computation-cost.csv conserva los costes medidos de la comparación principal, incluido su cálculo conservador de futilidad.','',summaries['timing'],'',
      '## P4. Diferencias localizadas detectadas','',
      'La primera versión cumplía las metas iniciales, pero una prueba de desarrollo trasladada a vértices no prefijados reveló 0/10 detecciones de un par correlacionado a m=250. V2 añadió un cribado de todos los pares por cambios de covarianza en la muestra de selección, y se evaluó de nuevo con semillas y muestras independientes. Ese cribado no garantiza detectar toda diferencia entre leyes categóricas de pares: solo usa dos representaciones de covarianza. Se conservan ambas versiones, las pruebas previas y los fallos de triples remotos. No se modificó el test después de observar resultados neuronales confirmatorios.','',
      '## P5. Cuántas coloraciones hacen falta','',
      'Las tablas benchmark-validation2.csv y las figuras 1 y 8 presentan tasas a m=100,250,500,1000,2000; m cuenta también los datos usados para selección. Cada punto tiene treinta ensayos independientes y un intervalo Wilson. Las metas de 80% se refieren exclusivamente a clases y tamaños prefijados. El chequeo adicional con 19.999 permutaciones usa el umbral estricto .05/795; sus resultados son:','',
      '| Alternativa | m total/generador | Detecciones | IC95 de potencia |','|---|---:|---:|---|']
    for r in read(OUT/'tail-validation.json'):lines.append(f'| {r["case"]} | {r["m"]} | {r["detections"]}/{r["trials"]} | [{r["ci_low"]:.3f}, {r["ci_high"]:.3f}] |')
    lines+=['','El menor tamaño de la rejilla con límite inferior Wilson de potencia >=.80 al nivel nominal .05 fue el siguiente. Es un resumen empírico de estas leyes específicas y de treinta repeticiones por punto, no un tamaño mínimo teórico ni una garantía simultánea para todas las alternativas. «No acreditado» incluye tasas insuficientes y falta de precisión; no significa imposibilidad.','',summaries['power'],'',f'large_support repite deliberadamente la ley de marginal con semillas independientes para auditar soporte enorme; no constituye otra clase matemática de alternativas. En sus {audit["large_support_trials"]["count"]} bancos sintéticos, el máximo número de repeticiones de un vector completo en cada muestra combinada fue {audit["large_support_trials"]["maximum_full_vector_repeats"]}. large-support-observed-uniqueness.csv conserva el detalle. La detección no requiere colisiones de vectores completos. Las definiciones de todas las leyes se documentan en ROBUST_CONTROL_DISTRIBUTIONS.md.']
    lines+=['','En las nueve entradas neuronales extendidas, los contrastes adicionales dieron los recuentos siguientes, tras Holm sobre la familia auxiliar completa de 405 hipótesis por procedimiento. Son resultados de prefijos anidados de una sola muestra nueva por celda; **no son estimaciones de potencia** ni observaciones independientes entre tamaños. El análisis principal m=500 tiene su propia familia de 795. No se aumentaron muestras buscando significación.','',secondary_table,'','El omnibus no domina cada comparación: en esta cohorte auxiliar, por ejemplo, Transformer a m=2000 tuvo 20 rechazos frente a 24 del MMD antiguo. Estos recuentos de una muestra no estiman una diferencia de potencia, pero muestran resultados donde reservar datos para selección no mejora la decisión. Las pérdidas se conservan junto a las ganancias.']
    lines+=['','No se promete esa potencia para otras distancias, regiones no representadas, todas las semillas o la familia completa de 795 hipótesis. La probabilidad de un evento raro limita la información disponible. El triple remoto tiene TV=.5 pero un efecto Hamming MMD² de solo 9.3653e-9; el omnibus tampoco busca todas las regiones de tercer orden. Su evento de paridad conocido sí separa las leyes: este fallo es una limitación de representación y potencia del instrumento, no una imposibilidad estadística para ese caso. Los resultados negativos aparecen en todas las tablas, sin reinterpretarlos como igualdad.','',
      '## P6. Qué ocurre con los 255 no rechazos históricos de E1','',
      f'**{newe} de los 255** pasan a ser significativos con el nuevo procedimiento y muestras independientes. En los mismos bancos nuevos de 500, E1 tiene {int(e.old_reject.sum())} rechazos por MMD anterior y {int(e.reject.sum())} por omnibus; {only} son rechazos del omnibus sin rechazo por el test antiguo sobre ese mismo banco. e1-transitions.csv identifica exactamente cada celda. El cambio histórico no puede atribuirse solo al procedimiento porque también cambia m; la evidencia de mejora metodológica procede de controles independientes y de las comparaciones con presupuesto igualado.','',
      (f'De esos rechazos nuevos sin detección por el MMD emparejado, {summaries["new_localized_e1"]} cuentan con un evento validado de uno, dos o tres vértices en el banco reservado. Esa evidencia local solo corresponde a esas celdas.' if summaries['new_localized_e1'] else 'Ninguno de esos rechazos adicionales de E1 cuenta con un testigo de uno, dos o tres vértices validado en el banco reservado. La mejora frente a diferencias locales está demostrada en los controles; no se ha demostrado que las nuevas detecciones reales de E1 sean locales. Los testigos confirmados de E1 son resúmenes estructurales.'),'',
      '## P7. DeepSets y Transformer','',table,
      'La comparación emparejada de decisiones principales incluye también los casos perdidos por el nuevo método:','',summaries['pairtable'],'',
      'by-training-checkpoint.csv mantiene los cinco entrenamientos de cada arquitectura separados. Sus diferencias son variación entre políticas fijas, no cinco ejecuciones de una misma política. by-degree-family-size.csv y la figura 6 separan tamaño y familia; no se ajustó una regresión poblacional no planificada ni se infirió una tendencia monótona a partir del número de rechazos.','',
      f'Un ejemplo descriptivo con n y familia fijos: en los regulares n=512, Transformer tuvo {regular_tf[2]}/5 rechazos a grado 2, {regular_tf[3]}/5 a grado 3 y {regular_tf[5]}/5 a grado 5. Ni siquiera los recuentos de esta rejilla son monótonos. Las curvas del efecto del kernel también dependen de su geometría y potencia; no establecen una tendencia universal de TV con el grado. Cada grado representa un grafo fijo, no una muestra poblacional de grafos.','',
      '## P8. Qué información permite distinguirlos','',
      f'Se seleccionó un evento binario por celda usando exclusivamente el banco primario y se estimaron sus dos probabilidades en 250 ejecuciones reservadas independientes. **{int((w.reject&w.witness_reject).sum())}** rechazos globales cuentan además con un testigo significativo tras Holm en la familia explicativa. La figura 7 usa intervalos CP simultáneos sobre las 1.590 probabilidades, de modo que seleccionar ejemplos para la presentación conserva cobertura. De cada intervalo [l,u] se deriva TV>=max(0,l,-u), porque la diferencia de probabilidad de cualquier evento está acotada por TV. Hay {int((w.tv_lower_bound_from_event>0).sum())} cotas inferiores positivas en witness-TV-lower-bounds.csv. Son garantías de tamaño para eventos específicos, no estimaciones exactas de TV en espacios grandes. Los testigos son observables predictivos, no explicaciones causales.','',
      f'Además, {int((~w.reject&w.witness_reject).sum())} celdas sin rechazo primario tuvieron un evento significativo en el banco independiente. Son evidencia secundaria de la familia explicativa y se conservan en all-independently-significant-events.csv; no se suman a los {total} rechazos primarios. La separación evita ocultar casos que el omnibus tampoco distinguió en su muestra principal.','',
      summaries['witnesses'],'','La tabla clasifica el evento seleccionado y validado en cada celda. Su frecuencia depende de esa selección y no es una descomposición exhaustiva de la discrepancia de cada arquitectura.','',
      '| Grafo / checkpoint | Evento | P Luby | P red | Diferencia red−Luby e IC simultáneo |','|---|---|---:|---:|---|']
    shown=representative_witnesses(w)
    for r in shown.itertuples():
        name=event_text(r.event);name=name if len(name)<150 else name[:145]+'…'
        lines.append(f'| {r.graph} / {r.model} | {name} | {r.probability_Luby:.3f} | {r.probability_model:.3f} | {r.difference:+.3f} [{r.difference_ci[0]:+.3f}, {r.difference_ci[1]:+.3f}] |')
    if not len(shown):lines.append('| Ningún testigo superó su criterio simultáneo | — | — | — | — |')
    lines+=['','Para la exposición se muestran hasta tres tipos de evento por arquitectura, ordenados por brecha observada; esa selección utiliza las cotas simultáneas de toda la familia. Un rechazo global sin testigo validado no demuestra que la diferencia sea local: puede reflejar acumulación de diferencias débiles o una característica que el banco explicativo no permite estimar con precisión. Las definiciones completas, probabilidades, intervalos y decisiones están en explanatory-witnesses.json. Las comparaciones de coloraciones etiquetadas nunca fueron sustituidas por clases de color o particiones sin nombres.','',
      '## P9. Igualdad, diferencia demostrada y equivalencia aproximada','',
      'En cliques, un protocolo ideal equivarante por permutaciones de vértices y con aleatoriedad privada iid tiene ley uniforme sobre las n! coloraciones propias, condicionada al éxito. La demostración es por invariancia y transitividad, no por ausencia de rechazo. La ejecución en coma flotante se comprueba mediante pruebas finitas; no es una certificación bit a bit universal. Los controles artificiales de estrella, correlación y paridad tienen diferencias demostradas mediante eventos y TV conocida. Los rechazos neuronales calibrados aportan evidencia estadística sobre cada grafo/checkpoint, no una demostración algebraica de su ley completa.','',
      f'En siete grafos pequeños se recalcularon leyes exactas de Luby por recursión racional. Para esta referencia se reservaron 20.000 ejecuciones por generador: la cota simultánea de muestreo no supera .02244 antes de añadir los ajustes. Las 105 cotas de TV incluyen el desfase entre absorción eventual y cutoff y una cota de precisión float64. **{equiv}/105** celdas satisfacen el criterio prospectivo de equivalencia aproximada TV<=.05 entre las leyes condicionadas a completar por R=128. La figura 9 y small-reference.csv identifican cuáles. La tolerancia .05 no es igualdad exacta, y no se aplica a grafos grandes. No se convierte una MMD pequeña en una cota útil de TV sin una desigualdad adecuada.','',
      summaries['smalltable'],'','En particular, los 35 casos pequeños de E1 satisfacen la tolerancia fijada; eso respalda cercanía en esos siete grafos y esas cinco políticas. Las diferencias estructurales grandes observadas en otros grafos impiden extrapolar esa cercanía a toda la arquitectura.','',
      f'En {int((s.approximate_equivalence&s.reject).sum())} casos pequeños se rechaza igualdad exacta y a la vez se respalda TV<=.05: ambas conclusiones son compatibles porque una diferencia pequeña puede ser detectable. La integración exacta de todas las regiones de las redes sobre sus cintas privadas no resultó viable; la referencia neuronal es Monte Carlo con cotas explícitas, no una ley exacta inventada.','',
      '## P10. Qué conclusiones sobreviven','',
      'La calibración del estudio anterior y sus 183 rechazos no quedan invalidados por su falta de sensibilidad en ciertas alternativas. Lo que cambia es la interpretación de los no rechazos: nunca demostraron semejanza global fuerte. El instrumento nuevo mejora verificablemente la sensibilidad para las clases especificadas, sin elevar el nivel por seleccionar regiones o combinar componentes sin calibración. Los resultados confirmatorios solo sustentan sus grafos y políticas fijas. Los casos de simetría, las diferencias conocidas, las equivalencias aproximadas respaldadas por cotas y los casos de igualdad desconocida se distinguen expresamente.','',
      '## Ejecución, auditoría y reproducción','',
      f'El auditor verificó {audit["banks"]} bancos y {audit["independent_algorithm_run_seeds"]:,} semillas nuevas distintas, sin solapamiento con las {audit["no_overlap_with_old_run_seeds"]:,} semillas históricas. Estados [válida, inválida completa, cutoff, error]: {audit["statuses"]}. Las fases se desglosan en audit.json; no se agrupan controles como si fueran ejecuciones confirmatorias. Consumo GPU-proceso cargado: {audit["gpu_process_hours"]:.3f}/12 horas, con inicialización y sobrecostes conservadores. No se reentrenaron las redes ni se hizo push.','',
      'Los bancos con fallos se identifican en la tabla siguiente. Sus intervalos de éxito son binomiales puntuales, no afirmaciones de cobertura simultánea. failure-law-tests.json conserva los contrastes de estados y de la ley incondicional coarsened, con ajuste sobre las 795 hipótesis previstas por familia. Un no rechazo de esos contrastes no establece que las probabilidades de fallo coincidan. La igualdad condicional por simetría en cliques no implica igualdad de las frecuencias de terminación.','',
      '| Fase / grafo / modelo | Válidas | Inválidas | Cutoff | Error | IC95 de probabilidad de éxito |','|---|---:|---:|---:|---:|---|',
      *[f'| {r["phase"]} / {r["graph"]} / {r["model"]} | {r["statuses"][0]} | {r["statuses"][1]} | {r["statuses"][2]} | {r["statuses"][3]} | [{r["success_ci95"][0]:.5f}, {r["success_ci95"][1]:.5f}] |' for r in audit['bank_validation'] if sum(r['statuses'][1:])],
      *(['| — | Ningún fallo observado | — | — | — | — |'] if not any(sum(r['statuses'][1:]) for r in audit['bank_validation']) else []),'',
      'La familia primaria nueva, la línea base emparejada, las 405 comparaciones de tamaño, los 795 testigos y los 105 contrastes pequeños tienen ajustes explícitos por separado; no se afirma FWER conjunto .05 al unir todas esas familias. El refinamiento por cotas conserva exactamente el conjunto de rechazos del cálculo conceptual completo con 19.999 permutaciones. Los valores-p censurados no rechazados se marcan como conservadores y retienen su cota inferior; no son estimaciones precisas de una cola.','',
      f'En los controles extremos hubo {audit["numerical_exact_classifier_underflows"]["count"]} probabilidades hipergeométricas exactas subrepresentables como float que aparecen como cero numérico en los datos brutos de clasificación. No son ceros de Monte Carlo ni probabilidades matemáticamente nulas; sus decisiones al 5% no cambian. Los clasificadores confirmatorios usan tamaños menores y sus colas son representables. Todos los valores-p de permutaciones conservan la corrección de una unidad.','',
      'Véanse ROBUST_DISTRIBUTIONS.md, el protocolo versionado, las pruebas y los manifiestos. Los datos de confirmación se generaron después del commit que congeló el procedimiento. Los módulos V1 y V2 y los fallos de ambos permanecen disponibles. Los pesos y grafos se identifican por hashes; el archivo reproducible conserva los insumos requeridos.']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    executive=f'''# Resumen ejecutivo: pruebas robustas

El instrumento se validó antes de la nueva comparación neuronal. El fallo
histórico de la estrella se reprodujo exactamente: TV=99/199 coexistía con
una señal MMD² 16 veces menor que su desviación nula a m=250. La calibración
primaria antigua usó suficientes permutaciones y el ajuste Holm era correcto.

El nuevo omnibus separa selección y evaluación y calibra conjuntamente
testigos locales, de interacciones, estructurales y globales. Detectó la
estrella en 30/30 ensayos independientes incluso al umbral .05/795; IC95
de potencia [.886,1]. Una segunda versión amplió la búsqueda de pares tras
un fallo de localización de la primera. Se conservaron los fallos frente a
triples remotos y eventos raros: no se afirma potencia universal.

Una réplica nula independiente de 4.000 ensayos, preregistrada después de
congelar el método, produjo {replication['rejections']}/4000 rechazos
({100*replication['rate']:.2f}%; IC95 {100*replication['ci_low']:.2f}–{100*replication['ci_high']:.2f}%).
No se modificó el instrumento a partir de esa réplica.

{table}

De los 255 no rechazos históricos de E1, {newe} ahora son significativos.
El histórico usó m=250; la comparación emparejada nueva presupuesta 500
ejecuciones para ambos procedimientos. El omnibus reserva la mitad de
las completadas para selección y registra los tamaños efectivos.
Los recuentos de rechazo no son distancias distribucionales.

Fase confirmatoria: {confirm_counts[0]:,} coloraciones válidas,
{confirm_counts[1]} inválidas completas, {confirm_counts[2]} cutoffs y
{confirm_counts[3]} errores técnicos. Las leyes de coloraciones se contrastan
condicionadas a completar antes del límite; los fallos se conservan aparte.

Se respaldó equivalencia aproximada TV<=.05 en {equiv}/105 celdas pequeñas
mediante cotas simultáneas prefijadas. En cliques, la igualdad condicional
ideal procede de un teorema de simetría. En los demás casos, no rechazar no
demuestra igualdad. Hay {int((w.reject&w.witness_reject).sum())} rechazos con
testigos observables validados en bancos independientes.

Auditoría: {audit['independent_algorithm_run_seeds']:,} semillas nuevas,
{audit['banks']} bancos; {audit['gpu_process_hours']:.3f}/12 horas GPU-proceso.
No se reentrenaron redes ni se hizo push. El informe completo responde
P1–P10 y conserva todas las decisiones, incertidumbres y limitaciones.
'''
    (OUT/'RESUMEN_EJECUTIVO.md').write_text(executive)
    write_paper(counts,total,matched,newe,equiv,accept)
    print(executive,flush=True)

def write_paper(counts,total,matched,newe,equiv,accept):
    paper=Path('paper/ucgnn');v=pd.read_csv(OUT/'benchmark-validation2.csv');audit=read(OUT/'audit.json')
    primary=pd.DataFrame(read(OUT/'primary.json'));small=pd.DataFrame(read(OUT/'small-reference.json'));witnesses=pd.DataFrame(read(OUT/'explanatory-witnesses.json'))
    # Dedicated print-size figures retain readable fonts at manuscript width.
    fig,axes=plt.subplots(1,3,figsize=(6.5,3.0),layout='constrained')
    for ax,case,title in zip(axes,('localized','correlation_remote','higher_order_remote'),('Localized star','Remote pair','Remote triple')):
        curve(ax,v,case,methods=('old_hamming','omnibus_v1','omnibus'));ax.set(title=title,xlabel='Total m / generator',ylabel='Detection rate')
        ax.tick_params(labelsize=8)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=7)
    fig.savefig(paper/'figures/robust-power.pdf',bbox_inches='tight');plt.close(fig)
    nullU=[read(f)['global_statistics']['old_hamming'] for f in (OUT/'benchmark-trials').glob('validation2-star_null-D200-eNone-m250-t*.json')]
    altU=[read(f)['global_statistics']['old_hamming'] for f in (OUT/'benchmark-trials').glob('validation2-localized-D200-eNone-m250-t*.json')]
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.8),layout='constrained');bins=np.linspace(min(nullU+altU),max(nullU+altU),20)
    axes[0].hist(nullU,bins=bins,density=True,alpha=.5,color='#888888',label='Null A/A');axes[0].hist(altU,bins=bins,density=True,alpha=.5,color='#1769aa',label='Alternative A/B')
    axes[0].axvline(read(OUT/'old-statistical-audit.json')['star_exact_mmd2'],color='#bf3b3b',label='Population effect')
    axes[0].set(title='Old statistic, m=250',xlabel='Unbiased MMD squared',ylabel='Density');axes[0].legend(fontsize=7)
    axes[0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[1].bar(['Artificial A','Artificial B'],[100/199,1.],color=['#888888','#1769aa']);axes[1].set(ylim=(0,1.05),title='Event: center color <=101',ylabel='Exact event probability')
    fig.savefig(paper/'figures/robust-star.pdf',bbox_inches='tight');plt.close(fig)
    text=r'''\section{Robust Testing of Complete Coloring Laws}
\label{sec:robust-laws}
The previous 183 rejections concern a calibrated but potentially insensitive
instrument. We audited all 795 historical results and reproduced all 100
known-star control $p$-values exactly. The original 19,999 permutations give
$p_{\min}=0.00005<0.05/795$; Holm resolution was sufficient. Characteristicness
of the Hamming kernel does not imply useful finite-sample power.

On the 201-vertex star, let leaves be independent uniform binary colors and
let the independent center be uniform on 199 high colors under $P$, versus
a 100-color subset under $Q$. Every vector is a proper local-list coloring.
Writing $a=e^{-1/201}$ and $b=(1+a)/2$ gives
\[
 \operatorname{TV}(P,Q)=99/199,\qquad
 \operatorname{MMD}^2=b^{200}(1-a)(1/100-1/199)
 \simeq1.5021\,10^{-5}.
\]
Under $P=P$, the kernel row mean is constant. If
$\zeta_2=\mathbb E[k^2]-(\mathbb E k)^2$, then the equal-size unbiased
statistic has exact variance
$[4/(m(m-1))+4/m^2]\zeta_2$. At $m=250$, its null standard deviation is
$2.4129\,10^{-4}$: the population signal is only $0.0623$ null standard
deviations. The named event that the center is at most 101 has probability
gap $99/199$ and discards irrelevant leaves.

\paragraph{An honestly selected omnibus.}
We split each independent sample into selection and evaluation halves.
Selection learns local categorical/threshold witnesses, pair and triple
features, structural summaries, a dense marginal score, and empirical
kernel witnesses at $\gamma\in\{.25,1,4,16\}$. An all-pair covariance search
on selection data extends fixed interaction regions. Evaluation uses the
maximum of pooled-standardized mean differences and an always-retained
characteristic full-vector MMD component. Selection never sees evaluation
labels; vertex and color labels are preserved.

\begin{proposition}[Conditional level of the honest maximum]
Under equality of the two output laws, conditioning on selection data and
pooled evaluation observations makes source labels uniformly exchangeable.
Recomputing the entire maximum under fixed-size label permutations, using
inclusive ties and the plus-one rule, yields a superuniform $p$-value.
\end{proposition}
\begin{proof}
The selected functions and pooled scales are fixed under this conditioning.
Uniform allocations therefore give the statistic's conditional null law.
The inclusive upper rank of the observed allocation among uniformly drawn
allocations is superuniform; averaging over the conditioning variables
preserves this inequality. Maximization is included in every allocation,
so it introduces no additional uncorrected search.
\end{proof}
Full finite-space arguments, consistency and computational-futility bounds
are given in the accompanying theory. Our global multiscale max/softmax
benchmarks are explicit adaptations inspired by MMDAgg and MMD-FUSE
\citep{schrab2023mmdagg,biggs2023mmdfuse}; we do not transfer their continuous
Sobolev minimax claims to discrete coloring spaces. Conditional random-rank
calibration follows the finite permutation framework
\citep{hemerik2018exact,phipson2010permutation}.

\paragraph{Independent instrument validation.}
Proper-coloring controls cover localized marginal changes, equal-marginal
dependence, higher-order parity, structural shifts, diffuse bias, rare mass
and huge supports. Null laws include point masses, uniform/discrete laws,
noisy stars, clique symmetry, Luby/Luby and actual checkpoint/self draws.
Each null law has 400 independent trials. Alternative grids use 30 trials
at $m=100,250,500,1000,2000$, counting selection observations in $m$.
The first version missed relocated pairs; a training-only pair search was
then validated with fresh second-round data before neural confirmation.
The final instrument passed its prespecified calibration and power gate.
The known-star alternative was detected in 30/30 independent trials at
both $m=250$ and $500$, even at $0.05/795$ with 19,999 permutations;
the pointwise Wilson lower power bound is $0.886$. Unknown remote triples
and sufficiently rare events remain sensitivity limitations, preserved
in the reported curves. No universal finite-sample power is claimed.

\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{figures/robust-power.pdf}
\caption{Selected independent second-round controls. Each total sample budget
includes the selection half. Bands are pointwise Wilson 95\% intervals
over thirty independent trials. Remote higher-order alternatives
retain failures; the known TV alone does not determine power.}
\end{figure}

\paragraph{Fresh frozen-policy confirmation.}
The final protocol was committed after instrument validation and before
generating new comparison banks. It retains the same 53 historical graph
entries (51 unique labeled graphs), all fifteen recent epoch-120 policies,
local lists and independent execution streams. Primary tests use 500 runs
per generator: 250 selection and 250 evaluation when all runs complete.
Conditional tests use only proper completions in that fixed prefix and
record their effective selection/evaluation counts if a cutoff occurs.
The old MMD uses all proper completions in the same fresh prefix.
Holm controls each planned 795-test
family at 0.05. The historical 250-run result is preserved separately.
'''
    text+=f'\nThe robust primary family rejected {total}/795 hypotheses; the old test on the same fresh 500-run banks rejected {matched}/795. Of E1\'s 255 historical nonrejections, {newe} reject under the new procedure. Changes from history combine method, sample budget and fresh sampling; matched-budget comparisons and independent controls are needed to assess sensitivity.\n'
    text+=f'The matched primary comparison contains {int((primary.reject&~primary.old_reject).sum())} new-only and {int((~primary.reject&primary.old_reject).sum())} old-only rejections. The new instrument does not dominate every alternative: in the auxiliary Transformer cohort at total m=2000 it rejected 20/45 cells versus 24/45 for the old MMD. These are nested-sample decisions, not estimates of alternative-specific power.\n'
    confirm=next(r['statuses'] for r in audit['phase_summaries'] if r['phase']=='confirmatory')
    text+=f'The full fresh confirmation allocation recorded {confirm[0]:,} valid completions, {confirm[1]} invalid complete outputs, {confirm[2]} cutoffs and {confirm[3]} technical errors. Cutoffs are retained as operational outcomes; they are not replaced by additional draws. Conditional clique symmetry does not imply identical termination probabilities.\n'
    text+=r'''\begin{center}
\begin{tabular}{lrrr}\toprule
Architecture & Historical 250 & Old, fresh 500 & New, total 500\\\midrule
'''
    for arch,h,o,n in counts:text+=f'{NAMES[arch]} & {h} & {o} & {n}\\\\\n'
    text+=r'''\bottomrule\end{tabular}
\end{center}
All counts are rejections among 265 fixed graph--checkpoint cells, not
distributional distances. Complete negative results and checkpoint-specific
effects are archived. Auxiliary sample-size comparisons use a separate
405-hypothesis family on nine prespecified entries. Conditional success
frequencies, all invalid outputs, cutoffs and technical errors remain
separate outcomes.

\paragraph{Independent witnesses and small-space inference.}
After primary selection, the last 250 separately reserved runs per generator
estimate one binary event per cell. Simultaneous Clopper--Pearson intervals
cover all 1,590 event probabilities, and explanatory Fisher tests have their
own Holm family. They identify predictive observables rather than causal
mechanisms. Small graphs additionally use rational teacher laws,
permutation-calibrated categorical likelihood ratios and simultaneous TV
intervals from 20,000 runs per generator, including the teacher
cutoff/eventual-law discrepancy and a conservative float64 rank bound.
'''
    text+=f'The prospective TV tolerance 0.05 was supported in {equiv}/105 small-graph policy cells. This is approximate conditional equivalence at a stated tolerance, not exact equality or a claim for large graphs.\n'
    small_counts=small.groupby('architecture').approximate_equivalence.sum()
    e1upper=max(ci[1] for ci in small[small.architecture=='normalized'].simultaneous_tv_ci)
    text+=f'All {int(small_counts["normalized"])}/35 small E1 cells met this criterion (largest simultaneous upper TV bound {e1upper:.4f}), versus {int(small_counts["deepsets"])}/35 for DeepSets and {int(small_counts["transformer"])}/35 for Transformer.\n'
    example=witnesses[(witnesses.architecture=='normalized')&witnesses.reject&witnesses.witness_reject].copy()
    example['gap']=example.difference.abs();e=example.sort_values('gap',ascending=False).iloc[0]
    threshold=e.event.get('event_threshold',e.event.get('threshold'))
    if e.event.get('name')=='used_colors':threshold=int(np.floor(threshold))
    text+=f'For a contrasting large-graph E1 example, {e.model} on {e.graph} has a selected event that the number of used colors is at most {threshold:g}. Its independent probabilities are {e.probability_Luby:.3f} under Luby and {e.probability_model:.3f} under E1; the simultaneous difference interval is [{e.difference_ci[0]:.3f},{e.difference_ci[1]:.3f}]. Hence complete-vector conditional TV is at least {max(0.,e.difference_ci[0],-e.difference_ci[1]):.3f} at the simultaneous coverage level. These selected predictive witnesses are not causal explanations, and no localized E1 witness passed the independent explanatory criterion.\n'
    text+=r'''\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{figures/robust-star.pdf}
\caption{Why large TV can coexist with weak old-MMD power. Left: the old
statistic under independent known-null and known-alternative draws, and its
tiny population effect. Right: the exact localized event probability
gap that an honestly selected witness can preserve.}
\end{figure}

\paragraph{What remains unknown.}
Clique equality follows from ideal vertex symmetry conditional on success;
finite hardware probes do not prove bitwise equivariance. Other neural
laws generally remain algebraically unknown. Nonrejection does not establish
equality, and a small MMD does not imply a useful TV bound without an
appropriate quantitative inequality. Rare-event coupling and large-support
mixtures supply explicit finite-sample lower bounds in the theory document.
The new instrument mitigates verified power failures for named alternatives;
its uncovered interactions are retained as limitations. No universal degree
trend or population statement over graph seeds or training procedures follows.
'''
    rep=read(OUT/'null-replication-diagnostic.json')['target']
    replication_text=f'An independently prespecified post-freeze replication investigated the noisy-star null after its original 32/400 rejections. The unchanged omnibus rejected {rep["rejections"]}/4000 independent null trials (rate {rep["rate"]:.4f}; pointwise Wilson 95\\% interval [{rep["ci_low"]:.4f},{rep["ci_high"]:.4f}]). This diagnostic neither replaced the original gate nor selected a new procedure.\n'
    text=text.replace('\\paragraph{Fresh frozen-policy confirmation.}',replication_text+'\n\\paragraph{Fresh frozen-policy confirmation.}')
    (paper/'robust-distributions.tex').write_text(text)
    abstract=f'An independently validated honest omnibus then addressed localized power failures of Hamming MMD and rejected {total}/795 hypotheses on fresh complete-coloring banks, versus {matched}/795 for the old test at the same total sample budget. Its verified sensitivity has explicit higher-order and rare-event limits.\n'
    (paper/'tables/robust-abstract.tex').write_text(abstract)
    main=paper/'main.tex';source=main.read_text()
    if 'tables/robust-abstract.tex' not in source:source=source.replace('\\end{abstract}','\\IfFileExists{tables/robust-abstract.tex}{\\input{tables/robust-abstract.tex}}{}\n\\end{abstract}')
    if '\\input{robust-distributions}' not in source:source=source.replace('\\input{global-distributions}','\\input{global-distributions}\n\\input{robust-distributions}')
    main.write_text(source)
    bib=paper/'references.bib'
    if not bib.exists():bib=next(paper.glob('*.bib'))
    entries=r'''
@article{schrab2023mmdagg,
 author={Schrab, Antonin and Kim, Ilmun and Albert, Melisande and Laurent, Beatrice and Guedj, Benjamin and Gretton, Arthur},
 title={MMD Aggregated Two-Sample Test}, journal={Journal of Machine Learning Research}, year={2023}, volume={24}, number={194}, pages={1--81}}
@inproceedings{biggs2023mmdfuse,
 author={Biggs, Felix and Schrab, Antonin and Gretton, Arthur},
 title={MMD-FUSE: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting}, booktitle={Advances in Neural Information Processing Systems}, year={2023}, volume={36}, pages={75151--75188}}
@article{hemerik2018exact,
 author={Hemerik, Jesse and Goeman, Jelle}, title={Exact testing with random permutations}, journal={TEST}, year={2018}, volume={27}, pages={811--825}, doi={10.1007/s11749-017-0571-1}}
@article{phipson2010permutation,
 author={Phipson, Belinda and Smyth, Gordon K.}, title={Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn}, journal={Statistical Applications in Genetics and Molecular Biology}, year={2010}, volume={9}, number={1}, pages={Article 39}, doi={10.2202/1544-6115.1585}}
'''
    existing=bib.read_text()
    if 'schrab2023mmdagg' not in existing:bib.write_text(existing+'\n'+entries)

if __name__=='__main__':report()
