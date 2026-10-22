from __future__ import annotations
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import *
from .analyze import read

COLORS={'normalized':'#2463a6','deepsets':'#d36c22','transformer':'#6c43a2'}
NAMES={'normalized':'NORMALIZED (E1)','deepsets':'DeepSets','transformer':'LocalPaletteTransformer'}

def figures(primary,tv):
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    def save(fig,name):
        fig.savefig(directory/f'{name}.pdf',bbox_inches='tight');fig.savefig(directory/f'{name}.png',bbox_inches='tight');plt.close(fig)
    laws=read(OUT/'small-laws.json')
    fig,axes=plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
    for ax,g in zip(axes.flat,('path3','triangle','star3','cycle4')):
        rows=[r for r in laws if r['graph']==g];p=rows[0]['teacher_probabilities'];x=np.arange(len(p))
        for r in rows:
            if r['model']=='Luby':continue
            arch=r['model'].split('-s')[0]
            ax.plot(x,r['probabilities'],color=COLORS[arch],alpha=.35,lw=.8)
        ax.plot(x,p,'ko-',ms=3,lw=1.5,label='Exact Luby');ax.axhline(1/len(p),color='grey',ls=':',label='Uniform reference')
        ax.set(title=f'{g}: labeled valid vectors',xlabel='Lexicographic coloring index',ylabel='Probability')
    handles=[plt.Line2D([],[],color=c,label=NAMES[a]) for a,c in COLORS.items()]+[plt.Line2D([],[],color='k',label='Exact Luby')]
    fig.legend(handles=handles,loc='outside lower center',ncol=4);save(fig,'figure1-small-laws')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,n in zip(axes,(256,512)):
        for arch,c in COLORS.items():
            for s in range(41000,41005):
                a=primary[(primary.architecture==arch)&(primary.training_seed==s)&(primary.family=='regular')&(primary.n_vertices==n)].sort_values('delta')
                ax.plot(a.delta,a.mmd2_unbiased,color=c,alpha=.6,marker='o',ms=3,label=NAMES[arch] if s==41000 else None)
        ax.set(xscale='log',xlabel='Maximum degree',ylabel='Unbiased MMD squared estimate',title=f'Regular graphs, n={n}; five policies each')
        ax.legend(fontsize=7)
    save(fig,'figure2-degree')
    power=pd.read_csv(OUT/'power.csv');fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,g in zip(axes,('path3','star-d200','regular-n256-d100')):
        for arch,c in COLORS.items():
            a=power[(power.graph==g)&(power.architecture==arch)].sort_values('m')
            ax.plot(a.m,a.conditional_empirical_bank_power,'o-',color=c,label=NAMES[arch]);ax.fill_between(a.m,a.ci_low,a.ci_high,color=c,alpha=.12)
        ax.set(xscale='log',ylim=(-.03,1.03),title=g,xlabel='Complete observations per generator',ylabel='Conditional bank detection rate')
    axes[0].legend(fontsize=6);fig.suptitle('Seed 41000; nominal alpha=.05; 30 resamples; intervals omit bank uncertainty',fontsize=10)
    save(fig,'figure3-power')
    independent=pd.read_csv(OUT/'independent-power.csv')
    fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
    for arch,c in COLORS.items():
        a=independent[(independent.architecture==arch)&(independent.graph=='path3')].sort_values('m')
        ax.plot(a.m,a.independent_population_power,'o-',color=c,label=NAMES[arch])
        ax.fill_between(a.m,a.ci_low,a.ci_high,color=c,alpha=.15)
    ax.set(xscale='log',ylim=(-.03,1.03),xlabel='Complete observations per generator',ylabel='Independent-experiment detection rate',title='Path3: independent population-power supplement')
    ax.legend(fontsize=7);fig.suptitle('20 disjoint experiments per m; seed 41000; nominal alpha=.05',fontsize=9)
    save(fig,'figure3b-independent-power')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,g in zip(axes,('star-d200','regular-n256-d100')):
        for arch,c in COLORS.items():
            a=independent[(independent.architecture==arch)&(independent.graph==g)].sort_values('m')
            ax.plot(a.m,a.independent_population_power,'o-',color=c,label=NAMES[arch])
            ax.fill_between(a.m,a.ci_low,a.ci_high,color=c,alpha=.15)
        ax.set(ylim=(-.03,1.03),xlabel='Complete observations per generator',ylabel='Independent-experiment detection rate',title=g,xticks=[100,250])
    axes[0].legend(fontsize=7);fig.suptitle('20 disjoint experiments per m; seed 41000; nominal alpha=.05',fontsize=9)
    save(fig,'figure3c-independent-large-power')
    piv=primary.pivot(index='graph',columns='model',values='p_holm');fig,ax=plt.subplots(figsize=(10,13),layout='constrained')
    im=ax.imshow(-np.log10(np.maximum(piv.values,1e-6)),aspect='auto',vmin=0,vmax=max(1.5,float(-np.log10(max(piv.values.min(),1e-6)))),cmap='viridis')
    ax.set_xticks(range(len(piv.columns)),piv.columns,rotation=70,ha='right',fontsize=7);ax.set_yticks(range(len(piv.index)),[g+' [actual D=4]' if g=='bipartite-n256-d3' else g for g in piv.index],fontsize=6)
    iy,ix=np.where(piv.values<=.05);ax.scatter(ix,iy,s=6,color='white')
    ax.set_title('Primary family: -log10 Holm p; white dots: rejection; 795 tests');fig.colorbar(im,ax=ax,label='-log10 adjusted p')
    save(fig,'figure4-tests')
    st=pd.DataFrame(read(OUT/'structural.json'));gg=read(OUT/'graphs.json');meta={g['id']:g for g in gg}
    st['delta']=st.graph.map(lambda g:meta[g]['delta']);st['family']=st.graph.map(lambda g:meta[g]['family']);st['n']=st.graph.map(lambda g:meta[g]['n'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,('used_mean','largest_class_mean'),('Number of used colors','Largest color class / n')):
        for label in ['Luby']+[s['label'] for s in specs()]:
            a=st[(st.model==label)&(st.family=='regular')&(st.n==256)].sort_values('delta')
            arch=label.split('-s')[0];c='black' if label=='Luby' else COLORS[arch]
            ax.plot(a.delta,a[key],color=c,alpha=1 if label=='Luby' else .4,lw=1.5 if label=='Luby' else .8,
                    label=label if label=='Luby' else (NAMES[arch] if label.endswith('41000') else None))
        ax.set(xscale='log',xlabel='Maximum degree',ylabel=title,title='Regular n=256; per-policy run means')
    axes[0].legend(fontsize=7);save(fig,'figure5-structure')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,g in zip(axes,('path3','star-d200','regular-n256-d100')):
        for arch,c in COLORS.items():
            a=primary[(primary.graph==g)&(primary.architecture==arch)].sort_values('training_seed')
            ci=np.array(a.mmd_population_ci95.tolist())
            ax.errorbar(a.training_seed-41000,a.mmd_biased,yerr=np.vstack((a.mmd_biased.to_numpy()-ci[:,0],ci[:,1]-a.mmd_biased.to_numpy())),fmt='o-',color=c,label=NAMES[arch],alpha=.7,lw=.8,elinewidth=.5,capsize=2)
        ax.set(title=g,xlabel='Training replicate (41000 + index)',ylabel='Empirical MMD norm',xticks=range(5))
    axes[0].legend(fontsize=6);fig.suptitle('Five fixed policies; conservative pointwise 95% population-MMD intervals',fontsize=10);save(fig,'figure6-training-variation')
    return directory

def report():
    p=pd.DataFrame(read(OUT/'primary.json'));tv=pd.read_csv(OUT/'small-tv.csv');audit=read(OUT/'audit.json');ctrl=read(OUT/'controls.json')
    figures(p,tv)
    structure=[]
    for g in read(OUT/'graphs.json'):
        graph=nx.Graph();graph.add_nodes_from(range(g['n']));graph.add_edges_from(np.array(g['edges']).T.tolist())
        structure.append(dict(graph=g['id'],n=g['n'],actual_delta=g['delta'],minimum_degree=min(g['degrees']),mean_degree=g['mean_degree'],degree_cv=g['degree_cv'],components=nx.number_connected_components(graph),density=nx.density(graph),bipartite=nx.is_bipartite(graph),sha256=g['sha256']))
    pd.DataFrame(structure).to_csv(OUT/'graph-structure.csv',index=False)
    timing=[]
    for task in read(OUT/'sampling-manifest.json')['tasks']:
        timing.append(dict(stage=task['stage'],graph=task['graph'],model='Luby' if task['model'] is None else task['model']['label'],runs=task['m'],process_seconds=task['seconds'],seconds_per_run=task['seconds']/task['m']))
    pd.DataFrame(timing).to_csv(OUT/'resource-costs.csv',index=False)
    clique_ids=[g['id'] for g in read(OUT/'graphs.json') if len(g['edges'][0])==g['n']*(g['n']-1)]
    clique_tests=p[p.graph.isin(clique_ids)];clique_tests.to_csv(OUT/'clique-symmetry-check.csv',index=False)
    p.groupby(['architecture','family','delta','n_vertices'],as_index=False).agg(tests=('p','size'),rejections=('reject_holm','sum'),median_empirical_mmd=('mmd_biased','median'),median_unbiased_mmd2=('mmd2_unbiased','median')).to_csv(OUT/'by-degree-family-size.csv',index=False)
    p.groupby(['architecture','training_seed'],as_index=False).agg(tests=('p','size'),rejections=('reject_holm','sum'),median_empirical_mmd=('mmd_biased','median'),median_unbiased_mmd2=('mmd2_unbiased','median')).to_csv(OUT/'by-training-seed.csv',index=False)
    # Descriptive controlled adjacent-degree changes, one graph per grid cell.
    differences=[]
    for (arch,s,n),a in p[p.family=='regular'].groupby(['architecture','training_seed','n_vertices']):
        a=a.sort_values('delta')
        for i in range(1,len(a)):
            differences.append(dict(architecture=arch,training_seed=int(s),n=int(n),degree_from=int(a.iloc[i-1].delta),degree_to=int(a.iloc[i].delta),
                                    unbiased_mmd2_change=float(a.iloc[i].mmd2_unbiased-a.iloc[i-1].mmd2_unbiased)))
    pd.DataFrame(differences).to_csv(OUT/'controlled-degree-changes.csv',index=False)
    aggregate=[]
    for arch in COLORS:
        a=p[p.architecture==arch]
        aggregate.append(f"| {NAMES[arch]} | {len(a)} | {int(a.reject_holm.sum())} | {len(a)-int(a.reject_holm.sum())} | {a.mmd_biased.min():.4f}–{a.mmd_biased.max():.4f} |")
    example=p[p.reject_holm].sort_values(['p_holm','graph','model']).head(12)
    lines=['# Global complete-coloring distributions: final technical and statistical report','',
      'No new networks were trained. Historical files were read only. No remote push was performed.','',
      f"The prospective primary family contains **795 tests on 53 fixed graphs and 15 frozen policies**, with 250 fresh complete runs per generator per graph. **{int(p.reject_holm.sum())}/795** reject after Holm correction at familywise alpha 0.05. These are conditional-on-valid-completion comparisons. Nonrejection does not prove equality.",'',
      '| Architecture | Tests | Holm rejections | Nonrejections | Empirical MMD range across graphs |','|---|---:|---:|---:|---:|',*aggregate,'',
      'There are 53 planned graph entries but 51 unique labeled graphs: star3/star-d3 and wheel-d3/clique-d3 are duplicate graph hashes with fresh banks. They are not independent graph replicates. The bipartite-n256-d3 entry has actual maximum degree 4, correctly recorded in the frozen graph manifest; its suffix is the generation target, not its measured degree. The MMD ranges above describe this finite battery. They are not universally comparable normalized distances across different graphs. Every individual test, effect interval and adjusted p-value is in `primary.json` and `primary.csv`.','',
      '## Execution and success', '',f"The audit verified {audit['banks']} banks and **{audit['independent_run_seeds']:,} distinct execution seeds**. Status totals [valid, invalid-complete, cutoff, technical] = {audit['statuses']}. Charged GPU-process time, including pilot, teacher sampling and lock-held overhead: **{audit['gpu_hours']:.3f} / 12 hours**.",'',
      'Every final vector was checked against all edges and its own local list. The frozen model hashes match the recent paper-study cohort (seeds 41000–41004, epoch 120), not the historical E1 cohort. TF32 was disabled, float32 neural input conversion is explicit, and the teacher uses float64 PCG64 uniforms. Actual seed-41000 policies and all fifteen CPU integration tests reproduced hard actions across reset/interleaving and batching probes. This is numerical validation, not a proof that every boundary input is invariant.','',
      '## Calibration controls','', '| Control | Rejections / trials | 95% rate interval |','|---|---:|---|']
    for r in ctrl['summary']:lines.append(f"| {r['kind']} | {r['rejections']}/{r['trials']} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    lines += ['', 'Null control groups are compatible with their preregistered 99% binomial acceptance intervals. The two point-mass control vectors (0,1,0) and (1,2,1) have identical color counts, class sizes and unlabeled partitions, yet different labeled laws. A point-mass positive control is intentionally easy; it cannot establish sensitivity against every neural alternative. Teacher/teacher controls include a high-dimensional degree-100 graph. Same-checkpoint controls cover all fifteen policies on path3 only.','',
      '## Exact small graphs','',
      'All seven supports were enumerated and teacher laws computed using rational absorbing-chain recursion, independently of Monte Carlo estimation banks. The path3 probabilities are 343/1628 (two vectors), 205/1628 (two), and 133/814 (two), rather than 1/6. Uniform proposals therefore do not imply uniformly distributed final solutions. This also occurs with homogeneous lists: on cycle4, each two-color vector has mass 35051/859584 and each three-color vector 36071/573056, giving TV 38111/429792 from the uniform valid-coloring law. `exact-*.json` archives every rational mass and reachable-state count.','',
      'The exact reference is the teacher eventual-absorption law; the empirical banks are conditional on valid completion by 128 rounds. A conservative per-node progress argument bounds their teacher-reference TV difference by n(1-2^(-Delta-1))^128; each small-graph row reports this bound. This distinction is retained even with zero observed cutoffs. The 2000-run independent neural estimation banks give plug-in TV against exact teacher probabilities and finite-support concentration intervals in `small-tv.csv`. TV estimation has sampling bias; intervals, rather than raw point estimates alone, govern interpretation. Exact teacher versus uniform is a separate reference. Teacher Monte Carlo validation is reported in the same table; finite nonzero empirical TV is expected.','',
      '## Statistically supported differences','', '| Graph | Policy | U-statistic MMD² | Raw p | Holm p |','|---|---|---:|---:|---:|']
    for _,r in example.iterrows():lines.append(f"| {r.graph} | {r.model} | {r.mmd2_unbiased:.6g} | {r.p:.5g} | {r.p_holm:.5g} |")
    if not len(example):lines.append('| No primary rejection | — | — | — | — |')
    lines += ['', 'This table shows at most twelve examples; `primary.csv` contains every result, including all negative results. The complete list of rejected and nonrejected cells is also exported separately. A valid coloring in both groups is not evidence of equal output laws.','',
      '## Answers to the ten scientific questions','']
    for i,arch in enumerate(COLORS,1):
        a=p[p.architecture==arch];k=int(a.reject_holm.sum())
        lines.append(f"**P{i}. Luby versus {NAMES[arch]}:** "+(f"yes, in {k} of the 265 fixed graph/checkpoint contrasts after Holm correction; this conclusion concerns precisely the cells in `rejected.csv`." if k else 'no primary difference was detected in this battery. This does not establish equality.')+'\n')
    lines += ['**P4. Graphs, degrees and sizes:** `by-degree-family-size.csv` reports all covered strata; `primary.csv` retains exact graph/checkpoint identities. Regular graphs cover all eight requested degrees at both n=256 and n=512; heterogeneous families extend through degree 200. Stars and cliques have size dictated by degree. One graph per random grid cell limits graph-seed generalization. The derived graph-structure.csv reports connectivity and density; degree-2 random regular graphs can be disconnected. The initial resource plan used coarse pilot extrapolation rather than a separately archived per-cell forecast; resource-costs.csv retains all realized cell costs, and the hard budget was enforced independently of test outcomes.','',
      '**P5. Does distinguishability increase with degree at fixed size and family?** Figure 2 and `controlled-degree-changes.csv` describe the fixed-n regular grids separately for every policy. We do not claim a universal monotone relationship: kernel geometry changes with the graph, only one graph is sampled per cell, and degree is not randomized independently of graph construction. Figure 2 and the adjacent changes use unbiased MMD-squared estimates to avoid a degree-dependent positive finite-sample bias of the nonnegative empirical norm; negative estimates are permitted. Descriptive signs and variability are retained rather than replacing them with an unplanned pooled regression.','',
      '**P6. How many independent complete colorings?** Primary detections use 250 per generator. Figure 3 and `power.csv` examine 100, 250, 500, 1000 and 2000 for three prespecified graphs and seed 41000 of each architecture. These are conditional empirical-bank power estimates at nominal alpha .05; their 30-trial intervals omit finite-bank uncertainty and do not estimate Holm-adjusted population power. They cannot justify a universal required sample size.','',
      '**P7. Variation among independent trainings:** Figure 6 and `by-training-seed.csv` show each of the five fixed policies separately. These are five independently trained functions, never a single mixed output distribution. Variation between policies remains distinct from Monte Carlo intervals within each policy. Shared graphs are blocks; no independent graph-by-seed population standard error is claimed.','',
      '**P8. Different proposals but indistinguishable complete outputs?** Yes for the ideal equivariant protocols on cliques: conditional on proper completion, vertex-permutation symmetry forces the uniform law on all n! valid colorings for every fixed checkpoint and for Luby. This is a mathematical sufficient-condition result, not an inference from nonrejection. Initially all-free K=201 proposal discrepancies also apply to any node of clique K201 because its deterministic MAX first-stage input matches the archived star-center witness. Floating-point implementation equivariance has numerical probes rather than a universal bitwise certificate. Earlier proposal tests alone cannot answer other graphs. This extension archives a contextual, explicitly post-hoc match between the previous all-free center law at K=201 and the identical initial degree-200 star, if available. Any nonrejection there means failure to detect a global difference at m=250, not equality or cancellation of all proposal differences. See `proposal-context.csv`; conditional proposals along entire trajectories were not exhaustively measured.','',
      '**P9. Structural explanations:** Figure 5 and `structural.json` report color count, class concentration/entropy, degree-wise color masses, nonadjacent co-color probabilities and partition diversity. Differences provide interpretable correlates, not causal explanations or a replacement for the labeled-vector test. Equal color-count laws would not imply equal complete laws. Partition encodings deliberately discard color names and are secondary.','',
      '**P10. Paper conclusions:** The provisional manuscript section reports exact teacher nonuniformity, precisely enumerated Holm rejections and negative-result limits. Observed differences show that imitation can produce different solution laws on these finite graphs. No observation proves universal inequality, equality, termination, monotonic degree dependence or equivalence over all graphs.','',
      '## Uncertainty and limitations','',
      'Population MMD intervals use a conservative distribution-free RKHS mean bound. They are pointwise, may be wide, and are not exact TV distances on large spaces. Approximate equivalence tolerance MMD<=0.02 was specified prospectively; cells satisfying the upper-bound criterion: '+str(int(p.equivalent_mmd_002.sum()))+'. Failure of this criterion is not evidence of inequivalence. With only 19,999 permutations, p-values have finite resolution; strong alternatives can share the minimum raw p. Holm depends on the whole planned family, not only detected cases.','',
      'Success-frequency intervals and unconditional/coarsened failure tests are separate artifacts. If all runs succeed, this records zero observed failures rather than probability-zero failure. No training, calibration or classifier test split is reused. Power resampling repeats may coincidentally draw an identical vector, which is allowed by the empirical law and does not merge RNG streams.','',
      '## Files and reproduction','',
      'See `GLOBAL_DISTRIBUTIONS.md` at the worktree root for exact commands. `sampling-manifest.json` contains all raw-bank paths and SHA256 hashes; `audit.json` checks every archive and seed, and replays three primary tests. `freeze.json` records the immutable protocol and checkpoint/graph hashes. `docs/GLOBAL_COLORING_DISTRIBUTIONS_THEORY.md` gives full finite-space proofs and the verified primary MMD reference. All six PDF/PNG figures regenerate from archived data.']
    p[p.reject_holm].to_csv(OUT/'rejected.csv',index=False);p[~p.reject_holm].to_csv(OUT/'not-rejected.csv',index=False)
    # Post-hoc context: exact same all-free central star state, not a new test.
    old=Path('/workspaces/AlgoLearning/ucgnn-local-palette-study/results/ucgnn-local-palettes/candidate-laws.csv')
    if old.exists():
        prior=pd.read_csv(old);prior=prior[(prior.state=='k201-ALL_FREE')&prior.model.isin([s['label'] for s in specs()])]
        context=p[p.graph=='star-d200'].merge(prior[['model','tv','tv_lower','tv_upper']],on='model',how='inner')
        context['prior_candidate_csv_sha256']=file_hash(old)
        context.to_csv(OUT/'proposal-context.csv',index=False)
    lines += ['', f'Clique symmetry diagnostic: {int(clique_tests.reject_holm.sum())}/{len(clique_tests)} primary clique-entry/checkpoint tests reject after Holm. The mathematical symmetry result concerns ideal equivariant conditional laws; finite numerical probes and tests do not constitute a universal hardware-level symmetry certificate. All these cells are retained in clique-symmetry-check.csv.', '']
    lines += ['', '## Observed degree contrasts and structural diagnostics', '']
    dd=pd.DataFrame(differences)
    for (arch,n),a in dd.groupby(['architecture','n']):
        positive=int((a.unbiased_mmd2_change>0).sum())
        lines.append(f'{NAMES[arch]}, regular n={n}: {positive}/{len(a)} adjacent-degree unbiased MMD-squared estimate changes are positive across the five separately plotted policies. These signs are descriptive estimates, without a monotonic population claim.')
    structural_rows=read(OUT/'structural.json');st=pd.DataFrame(structural_rows)
    reference={r['graph']:r for r in structural_rows if r['model']=='Luby'}
    graph_meta={g['id']:g for g in read(OUT/'graphs.json')};comparisons=[]
    for row in structural_rows:
        if row['model']=='Luby':continue
        ref=reference[row['graph']];g=graph_meta[row['graph']]
        marginal=0.
        for degree,value in row['color_by_degree'].items():
            weight=g['degrees'].count(int(degree))/g['n']
            marginal+=weight*.5*np.abs(np.array(value['mean'])-np.array(ref['color_by_degree'][degree]['mean'])).sum()
        co=np.abs(np.array(row['cocolor_mean'])-np.array(ref['cocolor_mean']))
        comparisons.append(dict(graph=row['graph'],model=row['model'],actual_delta=g['delta'],n=g['n'],
           used_colors_mean_difference=row['used_mean']-ref['used_mean'],largest_class_mean_difference=row['largest_class_mean']-ref['largest_class_mean'],
           entropy_mean_difference=row['class_entropy_mean']-ref['class_entropy_mean'],empirical_degree_color_frequency_tv=float(marginal),
           used_colors_difference_mc_se=float(np.sqrt(np.var(row['used_values'],ddof=1)/len(row['used_values'])+np.var(ref['used_values'],ddof=1)/len(ref['used_values']))),
           n_nonadjacent_pairs=len(co),mean_nonadjacent_cocolor_difference=float(co.mean()) if len(co) else None,max_nonadjacent_cocolor_difference=float(co.max()) if len(co) else None))
    diagnostic=pd.DataFrame(comparisons).merge(p[['graph','model','reject_holm','p_holm','mmd2_unbiased']],on=['graph','model'])
    diagnostic.to_csv(OUT/'structural-comparisons.csv',index=False)
    lines += ['', 'Degree-wise color frequencies and nonadjacent co-color comparisons are in structural-comparisons.csv. Their point estimates aggregate full-run features; vertices are never used as independent units for uncertainty. The following rejected cells have the largest observed degree-weighted color-frequency TV projections (descriptive, not complete-vector TV or causal attribution).', '', '| Graph | Policy | Empirical degree-color projection TV | Mean used-color shift |', '|---|---|---:|---:|']
    for _,r in diagnostic[diagnostic.reject_holm].sort_values('empirical_degree_color_frequency_tv',ascending=False).head(8).iterrows():
        lines.append(f'| {r.graph} | {r.model} | {r.empirical_degree_color_frequency_tv:.4f} | {r.used_colors_mean_difference:.3f} |')
    baseline=st[st.model=='Luby'].set_index('graph').used_mean
    st=st[st.model!='Luby'].copy();st['used_colors_difference']=st.used_mean-st.graph.map(baseline)
    top=st.reindex(st.used_colors_difference.abs().sort_values(ascending=False).index).head(8)
    lines += ['', '| Graph | Policy | Mean used-colors difference (model minus Luby) |', '|---|---|---:|']
    for _,r in top.iterrows():lines.append(f'| {r.graph} | {r.model} | {r.used_colors_difference:.3f} |')
    lines += ['', 'These are observed run-level mean shifts, not separately calibrated structural discoveries or causal mechanisms.', '', '## Independent population-power supplement', '']
    ind=pd.read_csv(OUT/'independent-power.csv')
    lines += ['The prospective supplement in `docs/GLOBAL_COLORING_POWER_SUPPLEMENT.md` fixes a new 77,000-run bank per generator on path3, split into disjoint blocks for twenty independent experiments at each m. Its rates estimate population detection probability for the specified policies and graphs. Teacher blocks are shared across policies but never across trials or m. Figure 3b and `independent-power.csv` report broad binomial intervals; Independent population power at m=100 and 250 is also measured on star-d200 and regular-n256-d100; higher m on those two graphs remain conditionally estimated.', '', '| Graph | Policy | m | Rejections / 20 | 95% population-power interval |', '|---|---|---:|---:|---|']
    for _,r in ind.iterrows():lines.append(f'| {r.graph} | {r.model} | {r.m} | {r.rejections}/20 | [{r.ci_low:.3f}, {r.ci_high:.3f}] |')
    if old.exists() and len(context):
        matched=context[(context.tv_lower>0)&(~context.reject_holm)]
        lines += ['', f'P8 contextual result: {len(matched)} of fifteen star/checkpoint comparisons have a previous initial-center proposal-TV lower bound above zero and no Holm rejection of the complete-vector law here. This is a post-hoc match of the identical initial state, not evidence of equal global laws. The complete fifteen-row comparison and original source hash are in proposal-context.csv.']
    localized=pd.read_csv(OUT/'localized-positive-control.csv')
    lines += ['', '## Known localized alternative: a finite-power failure', '', 'A separately prospective supplementary control holds all 200 star leaves iid uniform on {0,1}; only the center law changes, from uniform on 199 high colors to uniform on 100. Every vector is a valid local-list coloring. Exact global TV is 99/199 (about 0.4975), whereas exact MMD squared is '+str(localized.exact_mmd2.iloc[0])+'. The primary kernel is unchanged. This control demonstrates why characteristicness does not imply useful power for every finite-sample alternative.', '', '| m | Detections / 20 independent experiments | 95% power interval |', '|---:|---:|---|']
    for _,r in localized.iterrows():lines.append(f'| {r.m} | {r.rejections}/20 | [{r.ci_low:.3f}, {r.ci_high:.3f}] |')
    lines += ['', 'For these known localized laws, the decision rule selecting B when the center lies in {2,...,101} has optimal balanced accuracy 149/199 (about .7487). Thus weak MMD power here is not an information-theoretic impossibility of distinguishing the two artificial generators. The easy point-mass control and this localized control answer different sensitivity questions. A nonrejected learned law may still differ substantially in TV; no equivalence conclusion follows from these primary tests.']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    # Finite-run success uncertainty separately, no claim of universal correctness.
    successes=[]
    for t in read(OUT/'sampling-manifest.json')['tasks']:
        if t['stage']=='confirmatory':
            k=t['statuses'][0];successes.append(dict(graph=t['graph'],model='Luby' if t['model'] is None else t['model']['label'],valid=k,total=t['m'],wilson95=__import__('dcl.globaldist.analyze',fromlist=['wilson']).wilson(k,t['m'])))
    atomic_json(OUT/'success.json',successes)
    # Manuscript section summarizes final results and retains all policy counts.
    section=r'''\section{Are Learned Distributed Coloring Algorithms Distributionally Equivalent to Their Teacher?}
\label{sec:global-laws}
Proposal imitation concerns $q_\theta(c\mid s)$, whereas the present question
concerns the complete labeled vector $X_G=(c_1,\ldots,c_n)$ obtained after all
rounds. On a fixed labeled graph with lists $C_v=\{0,\ldots,d(v)\}$, write
$P_L^G$ and $P_\theta^G$ for the teacher and frozen-policy output laws,
conditional on valid completion by the preregistered 128-round cutoff.
Different proposal laws need not imply different absorbing laws, and correct
coloring by two procedures does not establish equality of those laws.

We reused the fifteen recent epoch-120 checkpoints (five independent
trainings each of NORMALIZED, DeepSets and LocalPaletteTransformer), keeping
policies separate and using the verified receiver-local palette adapter.
No new training or safety shield was introduced. Private execution uniforms
were independent between complete runs and generators; no global degree
was supplied as an input feature. The teacher used float64 uniforms and
networks their explicit float32 conversion, with TF32 disabled.

\begin{proposition}[Clique symmetry]
On $K_n$ with local lists $\{0,\ldots,n-1\}$, any vertex-equivariant
randomized protocol with iid private node randomness has the uniform output
law on the $n!$ proper colorings, conditional on positive-probability valid
completion (including completion by a fixed cutoff).
\end{proposition}
\begin{proof}
The uncolored initial state and private-randomness law are invariant under
all vertex permutations. Equivariance preserves this invariance through
every round. Conditioning on the invariant event of proper completion
preserves invariance. Vertex permutations act transitively on proper
colorings, which are bijections to the $n$ colors, so their probabilities
are equal and sum to one.
\end{proof}
The selected shared-weight architectures are vertex-equivariant in real
arithmetic, even though their proposal heads can prefer some color labels.
Thus identical conditional clique laws can coexist with different proposal
laws and different success or termination-time laws. This is an ideal
symmetry argument; numerical equivariance is tested on finite probes, not
certified for every hardware input. It establishes no uniformity on general
graphs and no equality of unconditional laws when failure probabilities differ.

Seven small labeled graphs permit exact rational computation of the teacher's
absorbing law. For the three-vertex path with palettes $(2,3,2)$, its six
valid colorings have masses $343/1628$, $205/1628$ and $133/814$, each twice.
They are not uniform. Separate 2,000-run estimation banks quantify neural
TV distances to these exact laws with conservative finite-support intervals.

The confirmatory battery contains 53 fixed graph entries (51 unique labeled
graphs; two prespecified family aliases), including regular graphs
at $n=256,512$ and each $\Delta\in\{2,3,5,10,20,50,100,200\}$, plus paths,
cycles, stars, trees, random irregular graphs, irregular bipartite graphs,
wheels and cliques. Each graph--checkpoint contrast uses 250 complete runs
per generator, a fixed characteristic kernel
\[
 k_G(x,y)=\exp(-d_H(x,y)/|V|),
\]
an unbiased MMD statistic \citep{gretton2012kernel} and 19,999 fixed-size label permutations. Its
finite-product Gram matrix is strictly positive definite: each coordinate
matrix is $(1-a)I+a\mathbf1\mathbf1^\top$ with $a=e^{-1/|V|}\in(0,1)$;
Kronecker products and restriction to valid colorings preserve strict
positivity. Holm correction controls familywise error at $0.05$ across all
795 planned hypotheses, allowing dependence from shared teacher banks.
'''
    section+=f"\nThe primary family rejected {int(p.reject_holm.sum())} of 795 hypotheses. "
    section+='; '.join(f"{NAMES[arch]}: {int(p[p.architecture==arch].reject_holm.sum())}/265" for arch in COLORS)+'.\n'
    section+=f"The audited queued runs had status totals (valid, invalid, cutoff, technical) $({','.join(map(str,audit['statuses']))})$. Zero observed failures does not establish zero failure probability.\n"
    section+=r'''
\begin{center}
\begin{tabular}{lrr}
\toprule Architecture & Contrasts & Holm rejections\\\midrule
'''
    for arch in COLORS:section+=f"{NAMES[arch]} & 265 & {int(p[p.architecture==arch].reject_holm.sum())}\\\\\n"
    section+=r'''\bottomrule
\end{tabular}
\end{center}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{figures/global-degree.pdf}
\caption{Complete-vector unbiased MMD-squared estimates on the two fixed-size regular grids.
Each curve is one frozen training checkpoint; unbiased estimates may be
negative. These effects are graph-specific;
the normalized Hamming kernel does not make different graph spaces universally
comparable. One graph per grid cell limits graph-seed generalization.}
\end{figure}

All per-graph, checkpoint, raw and Holm-adjusted $p$-values are archived in
\texttt{results/ucgnn-global-distribution/primary.csv}; the accompanying
\texttt{rejected.csv} identifies exactly the cells supporting rejection.
Five-policy plots retain training variability separately from within-policy
Monte Carlo uncertainty. Structural diagnostics describe color usage,
class sizes, degree-conditioned color frequencies, co-color probabilities
and induced partitions; none substitutes for the labeled-vector hypothesis.
Secondary classifiers use independent training, validation and test banks.
A separately prospective supplement measures population power on the
three-vertex path using twenty disjoint experiments at each sample size and
seed 41000 of each architecture; an additional 100/250 subgrid covers the
star and regular power graphs. Binomial intervals retain finite-trial
uncertainty. A known-law supplementary star alternative has global TV $99/199$ while
its kernel effect is attenuated by $1-e^{-1/201}$; its independent detection
trials document finite-power limitations despite characteristicness.
Power curves over 100--2,000 observations on larger graphs use independent empirical banks
for three prespecified graphs and seed 41000. Their resampling intervals are
conditional on those banks and omit uncertainty in their population laws;
they describe nominal, not Holm-adjusted, detection rates.

Nonrejection means insufficient evidence at this sample size and correction,
not equality. Conservative pointwise population-MMD intervals do not establish
approximate equivalence at the preregistered tolerance $0.02$. Rejection
supports different solution laws only for its specified graph and frozen
policy. No universal monotone degree trend, population effect over training
seeds or graph families, or universal learned termination theorem follows.
The preregistration, exact-law proof, complete negative results and reproducible
figure sources accompany this provisional section.
'''
    Path('paper/ucgnn/global-distributions.tex').write_text(section)
    shutil.copy2(OUT/'figures/figure2-degree.pdf','paper/ucgnn/figures/global-degree.pdf')
    abstract=f'A subsequent complete-coloring-law study used fifteen frozen policies on 53 fixed graph entries and rejected {int(p.reject_holm.sum())}/795 graph--checkpoint equality hypotheses after Holm correction. Exact teacher laws are nonuniform on small paths and cycles, whereas vertex symmetry forces uniform conditional solution laws on cliques for ideal equivariant protocols. Nonrejection alone is not evidence of equivalence.\n'
    Path('paper/ucgnn/tables/global-abstract.tex').write_text(abstract)
    main=Path('paper/ucgnn/main.tex');text=main.read_text()
    if 'tables/global-abstract.tex' not in text:text=text.replace('\\end{abstract}', '\\IfFileExists{tables/global-abstract.tex}{\\input{tables/global-abstract.tex}}{}\n\\end{abstract}')
    if '\\input{global-distributions}' not in text:text=text.replace('\\input{local-palettes}','\\input{local-palettes}\n\\input{global-distributions}')
    main.write_text(text)
    executive=f'''# Resumen ejecutivo

Se completaron 795 contrastes preregistrados sobre distribuciones de coloraciones completas: {int(p.reject_holm.sum())} rechazaron igualdad con Holm al 5%. Se mantuvieron separados los quince checkpoints recientes y se usaron 250 ejecuciones independientes por algoritmo y grafo.

'''+ '\n'.join(f"- {NAMES[a]}: {int(p[p.architecture==a].reject_holm.sum())}/265 contrastes rechazados." for a in COLORS)+f'''

Una demostración por simetría establece igualdad de las leyes condicionadas al éxito en cliques para los protocolos ideales equivariantes, aun con propuestas diferentes.

Se verificaron {audit['independent_run_seeds']:,} semillas distintas y {audit['banks']} bancos; estados [válidas, inválidas, cutoff, error] = {audit['statuses']}. Tiempo de proceso cargado: {audit['gpu_hours']:.3f}/12 horas. No hubo entrenamiento nuevo ni push.

Luby no es uniforme sobre las soluciones del camino de tres vértices. La comparación principal usa vectores completos etiquetados, no colores individuales ni estadísticas marginales. Los resultados no rechazados no prueban igualdad. Los efectos entre grafos no son distancias universalmente normalizadas. Las curvas de grafos grandes son condicionales a bancos empíricos; el suplemento del camino usa ensayos independientes. Un control de estrella con TV conocida cercana a 0,5 documenta falta de potencia para diferencias localizadas. No se establece un tamaño muestral universal. No se afirma una tendencia monótona con el grado ni generalización a todas las semillas o grafos.

El informe completo responde P1–P10, conserva todos los resultados negativos y especifica las limitaciones. Véanse FINAL_REPORT.md, primary.csv, small-tv.csv y las seis figuras.
'''
    (OUT/'RESUMEN_EJECUTIVO.md').write_text(executive)
    print(executive)
if __name__=='__main__':report()
