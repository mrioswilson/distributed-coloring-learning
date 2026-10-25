#!/usr/bin/env python3
"""Add only completed degree results, compile, and audit new PDF content."""
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import pypdfium2 as pdfium
from dcl.degree.inventory import ROOT
from dcl.study.storage import atomic_json, file_hash

PAPER=Path('paper/ucgnn')
SHORT=dict(LUBY='Luby',JOHANSSON='Johansson',COLOR_TRIALS='ColorTrials',JP='JP',SPECULATIVE_GM='GM',GREEDY='Greedy')
ARCH=dict(normalized='E1',deepsets='DeepSets',transformer='Transformer')


if __name__=='__main__':
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['stages']['confirmatory']['executions']==810000
    rows=list(csv.DictReader((ROOT/'checkpoint-condition.csv').open()))
    main=[r for r in rows if r['stage']=='confirmatory' and r['primary']=='True']
    primary_total=sum(int(r['executions']) for r in main)
    primary_ok=sum(int(r['successes']) for r in main)
    native=[r for r in main if r['delta']=='3']
    native_ok=sum(int(r['successes']) for r in native)
    groups=sorted({(r['teacher'],r['architecture']) for r in main})
    conditions=['global-n64-d4-mixture1000','global-n64-d10-mixture1000','local-n64-d4-mixture1000','global-n256-d4-mixture1000']
    lines=[]
    for t,a in groups:
        cells=[]
        for condition in conditions:
            xs=[r for r in main if r['teacher']==t and r['architecture']==a and r['condition']==condition]
            assert len(xs)==5 and all(int(r['executions'])==1000 for r in xs)
            passed=sum(int(r['successes'])==1000 for r in xs)
            lo,hi=min(int(r['successes']) for r in xs),max(int(r['successes']) for r in xs)
            cells.append(f'{passed}/5' if passed==5 else f'{passed}/5 ({lo}--{hi})')
        lines.append(f"{SHORT[t]} & {ARCH[a]} & "+' & '.join(cells)+r' \\')
    tex=r'''\begin{table}[htbp]
\centering\small
\begin{tabular}{llcccc}
\toprule
Teacher & Core & $\Delta4$ global & $\Delta10$ global & $\Delta4$ local & $n256,\Delta4$ \\
\midrule
'''+ '\n'.join(lines)+r'''
\bottomrule
\end{tabular}
\caption{Frozen primary policies: checkpoints with 1000/1000 successes out
of five. Nonperfect cells additionally give the per-checkpoint success
range out of 1000. The first three columns use $n=64$; the last uses the
global palette at $n=256$. Each condition samples the declared uniform
mixture of ten exact-degree families. There is no averaged policy.}
\label{tab:frozen-degree}
\end{table}
'''
    (PAPER/'tables/degree-generalization.tex').write_text(tex)
    for name in ('confirmatory-degree.pdf','degree-mechanisms.pdf'):
        shutil.copyfile(ROOT/'figures'/name,PAPER/'figures'/f'degree-{name}')
    section=rf'''
% BEGIN COMPLETED DEGREE STUDY
\section{{Frozen transfer across degree and palettes}}\label{{frozen-degree}}

The native coverage statement remains 21/21 under heterogeneous historical
protocols, including five checkpoints and 1000 final runs per checkpoint.
It does not mean that all policies used the same supervision or budget.
We now extend the twelve passive sample-complexity packages and use the
certified counterfactual checkpoints as the primary cohort for the nine
new packages. Native matched/passive checkpoints remain secondary controls.
No weights are modified or retrained in the transfer study.

\paragraph{{Teacher and information audit.}}
The exact repeated Luby, always-active Johansson, repeated ColorTrials,
synchronous JP, four-block speculative GM and scheduled Greedy rules admit
degree/palette parameterization. Both global lists of size $\Delta+1$ and
local lists $\{{0,\ldots,d(v)\}}$ are justified. We preserve coherent ID orders,
the four GM blocks, detection and serial repair, and Greedy permissions.
The algebraic BE\_BASE\_D3 construction is excluded: its fixed fields and
four-option reductions do not justify increasing the degree. A different
Linial/KW or recursive BE variant would change the learning problem.

The historical rooted serializer fixes four colors and three neighbor
positions. The three neural cores instead share weights across colors and
neighbors. An explicit sparse structural adapter evaluates those same
weights with variable palettes and the same raw predicates, signed IDs,
phases and private inputs. It supplies no exact availability, winning flag,
COMMIT or repair rule. Palette masks describe declared lists only. The two
graph exchanges remain MAX; DeepSets and attention act within each palette.
Across the 195 original selected/control checkpoints, 1170 new native
executions succeed. For the 150 extensible checkpoints, 21630 teacher-state
decisions match the original wrapper, with no hard-action or final-color
differences in these controls; the largest observed absolute logit
discrepancy is $1.526\times10^{{-5}}$. These are finite parity checks.

\paragraph{{Conditional exact-rule guarantees.}}
With a proper partial coloring, $|A_v|\ge d_U(v)+1$ under either list regime.
Availability and exclusion of equal simultaneous offers prove random-trial
safety. For ideal independent uniform proposals, Luby has conditional
progress at least $1/4$. The implemented always-active Johansson variant
also admits a self-contained bound: pending neighbors have at least two
free colors, hence $q_u(c)\le1/2$; using
$\log(1-x)\ge-2\log(2)x$ and Jensen yields
$|A_v|^{{-1}}\sum_c\prod_u(1-q_u(c))\ge
2^{{-2d_U(v)/|A_v|}}\ge1/4$.
Discrete ColorTrials with preferences in $\{{0,\ldots,\Delta\}}$ has
progress at least $(\Delta+1)^{{-1}}(\Delta/(\Delta+1))^\Delta$ for
$\Delta>0$. JP completes in at most $n$ steps, Greedy in $n$, and GM in
at most $2\lceil n/\min(4,n)\rceil+n$ scheduled steps.
The random-tail bounds concern the ideal exact rules, not arbitrary
learned proposals, a formal PRNG proof, or universal learned termination.

\paragraph{{Frozen confirmatory experiment.}}
After an exploratory degree/size pilot, we freeze graph/tape namespaces,
checkpoint hashes, horizons, families and sample sizes before final
outcomes. We validate all 150,000 exact-reference runs first. Graphs have
maximum degree exactly requested. The 90 primary extensible policies receive
1000 runs at each of five conditions: global $n64/\Delta3,4,10$,
local $n64/\Delta4$, and global $n256/\Delta4$.
Sixty certified matched/passive controls and thirty pre-existing CT controls
at the same 4000-update point receive the four $n64$ conditions. The extra
controls are not called historically certified. All runs use frozen weights;
an additional CPU executor changes only scheduling, not the frozen design.

There are 810,000 confirmatory student executions, of which
{summary['stages']['confirmatory']['successes']:,} are correct across distinct
policies. The primary cohort contributes {primary_ok:,}/{primary_total:,},
including {native_ok:,}/90,000 on the common degree-three control.
These totals are accounting, not a pooled success probability of one policy.
Table~\ref{{tab:frozen-degree}} reports each teacher/core across all five
training replicas. The pointwise two-sided 95\% Clopper--Pearson interval
for 1000/1000 is approximately $[.996318,1]$, under the independent
uniform-family mixture model; no simultaneous or universal claim follows.

\input{{tables/degree-generalization.tex}}

Sixteen of eighteen extensible teacher/core combinations pass every primary
condition in all five replicas. Johansson/Transformer seed83101 obtains
972/1000 at global degree ten, with 28 clique cutoffs. GM/Transformer
seed94102 obtains 819/1000 on local degree-four lists; its global conditions
pass. E1 and DeepSets pass all selected primary conditions for all six
extensible teachers. Other Transformer replicas and conditions are retained
individually. Successful transfer therefore depends on both palette regime
and checkpoint, not only a single maximum-degree number.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\linewidth]{{figures/degree-confirmatory-degree.pdf}}
\caption{{Frozen degree transfer at fixed $n=64$ and global palettes.
Each line is one training checkpoint, with 1000 runs per point. Curves do
not average the five policies and do not separate degree from palette size.}}
\label{{fig:frozen-degree}}
\end{{figure}}

The exploratory pilot spans $\Delta=2,3,4,5,10,20,50,100,200$ at regular
$n256$, and sizes $64$--$1024$ at fixed degree3 or10, with additional
irregular, hub, tree, bipartite, disconnected, isolated and clique cases.
It gives 3526/3570 correct outcomes. The remainder are two matched CT
collisions and 42 dense CT resource cutoffs. Some degree50 runs completed
before a pilot resource amendment; unfinished dense CT cells then received
a ten-second cap. Those outcomes do not have a homogeneous resource horizon.
All five teachers other than CT have successful degree200 pilot cases in
each replica, but one run per checkpoint/cell does not certify that range.

\paragraph{{Why functionality does not establish the learned rule.}}
A typed CT preference intervention isolates a local limitation. Place an
edge in one component and a degree-$\Delta$ star in another. The edge
endpoints always have degree1 and local lists $\{{0,1\}}$. Give the first
endpoint preference $\Delta$ for color0, the second $\Delta-1$, and zero
otherwise. The first is the strict exact winner. The conservative input
normalization makes the gap $1/\Delta$. All fifteen primary CT CF policies
accept the queried winner at degrees3--5 but reject it from degree10 in
this probe. Neighbor count and receiver palette stay fixed; the wrong
decision is not caused by aggregating more neighbors. These are valid typed
private-input counterfactuals, not a claim that a recorded PCG64 seed emits
the entire constructed vector, nor a random-rollout failure estimate.

The primary encoder uses preference divided by global degree, agreeing
with the native preference/3 template. A separate posthoc observation
encoding keeps the old preference/3 units at all degrees. On this same
raw-gap1 query it accepts 13/15 winners at degree5, 5/15 at10, and0/15
at50 and200, versus15/15,0/15,0/15 under the primary encoder. Both preserve
preference order; neither resolves every measured case. The main frozen
results are not replaced by an outcome-selected normalization. These
witnesses establish sensitivity to the input scale/range, not information
loss or impossibility under every possible embedding.

The Johansson failure exposes another mechanism. A full-batch replay
reproduces 128 steps and the original cutoff. Two adjacent pending vertices
remain with available colors $\{{0,2\}}$; both propose color2 in the final
rounds and COMMIT correctly vetoes the collision. In a separate conditional
probe, each vertex proposes color2 for all 8192 new private draws. This
observed concentration does not prove exactly zero mass elsewhere, but
explains the lost progress in the recorded trace. Figure~\ref{{fig:degree-mechanisms}}
separates this proposal-diversity failure from CT's comparison failure.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\linewidth]{{figures/degree-degree-mechanisms.pdf}}
\caption{{Left: fifteen CT counterfactual checkpoints queried on the same
winning color as the normalized preference gap shrinks, with one neighbor
and two local colors. Right: one stalled Johansson/Transformer state;
the two pending receivers have identical observed histograms, shown for one
of them, versus the exact uniform proposal. These are component diagnostics,
not complete-coloring distribution tests.}}
\label{{fig:degree-mechanisms}}
\end{{figure}}

For GM, the same bad local-list repair proposal occurs in CF, matched and
passive seed94102. CF rejects that proposal while the other heads accept
it, but the external schedule does not revisit the missed repair; all
three retain a conflict in the diagnostic case. Thus improved conditional
COMMIT response need not improve final functionality. Learned alpha values
are latent features, not directly supervised calibrated availability
probabilities; threshold disagreement is not itself a decision error.

A degree-three control separates the GM palette effect. On an edge plus a
disconnected three-star and isolates, coherent IDs place both edge endpoints
in the same speculative slot. Exact prefixes require the same repair color1
under global K4 and local $\{{0,1\}}$. All three Transformer seed94102
procedures choose color1 globally but color0 locally; the other four seeds
choose correctly. Thus this failure can occur without increasing degree:
it is a local-palette transfer limitation, not evidence that degree4 alone
causes the error. These constructed states are diagnostic, not extra finals.

Exploratory COMMIT probes use 120 policies and 838,188 typed queries on
exact safe prefixes. At matched initializations, native minibatches and
update counts (CT4000, GM500), CF improves mean per-seed query fidelity
against matched, with effects depending on degree and architecture. CF
also uses more labels and changes the loss distribution. These controls
therefore do not isolate counterfactual supervision at equal total label
budget, nor prove a monotone functional benefit in degree. Full paired
denominators and all raw head errors are retained.

Finally, the historical outward-interval test of 64 finite MAX message
supports certifies neither exact COMMIT nor absence of false acceptance
for any of the 45 selected passive canonical Luby/Johansson/Greedy cores.
This concerns conditional head inputs, not necessarily states reachable
under their own proposals or Greedy's single-writer schedule. It does not
contradict their finite functional success, and cannot certify progress.
The interval/float32 arithmetic assumptions are recorded separately.

The completed extension supports a bounded addition to this article.
Multigrade retraining, degree/palette causal separation, label-budget-matched
CF, uniform learned progress, and a high-degree BE variant remain distinct
future experiments; none is run here.
% END COMPLETED DEGREE STUDY
'''
    section=section.replace(r'\paragraph{',r'\paragraph*{')
    path=PAPER/'main.tex';maintex=path.read_text()
    maintex=re.sub(r'\n% BEGIN COMPLETED DEGREE STUDY.*?% END COMPLETED DEGREE STUDY\n','\n',maintex,flags=re.S)
    marker=r'\section{Discussion}\label{discussion}'
    assert maintex.count(marker)==1
    maintex=maintex.replace(marker,section+'\n'+marker)
    large=ROOT/'larger-size-pilot/index.json'
    if large.exists():
        rr=json.loads(large.read_text())
        extra=rf'''
A separate post-confirmation exploratory size extension gives
{sum(r['successes'] for r in rr)}/{sum(r['executions'] for r in rr)} correct
executions at regular $n2048/4096$, degrees3 and10, for all sixty primary
Luby/Johansson/ColorTrials/JP checkpoints. It uses an independent namespace
and one run per checkpoint/point, after validating eighty exact references.
It is not included in the confirmatory denominator or a size-range certificate.
'''
        maintex=maintex.replace(r'\paragraph*{Why functionality does not establish the learned rule.}',extra+'\n'+r'\paragraph*{Why functionality does not establish the learned rule.}')
    maintex=maintex.replace('Curves do\nnot average the five policies and do not separate degree from palette size.',
        'The success axis starts at .94 to expose rare failures. Curves do\nnot average the five policies and do not separate degree from palette size.')
    maintex=maintex.replace(r'\Gscale@div\@tempa{\textheight}',r'\Gscale@div\@tempa{.68\textheight}')
    old='''Native teacher generalization above
degree 3 or to local palettes is unsupported by this interface. The
separate historical local-palette evidence must retain its own
checkpoints and protocols.'''
    new='''The historical rooted serializer by itself does not support native
teacher transfer above degree three or to local lists. The explicit frozen
structural extension in Section~\\ref{frozen-degree} addresses six teachers
on its stated checkpoints and finite conditions. Separate historical
local-palette evidence retains its own checkpoints and protocols.'''
    if old in maintex:maintex=maintex.replace(old,new)
    path.write_text(maintex)
    compiler=Path('/tmp/algolearning-verified-tectonic/tectonic')
    assert file_hash(compiler)=='a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da'
    result=subprocess.run([str(compiler),'--keep-logs','main.tex'],cwd=PAPER,capture_output=True,text=True)
    (ROOT/'paper-build.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Paper compilation failed: see paper-build.txt')
    doc=pdfium.PdfDocument(str(PAPER/'main.pdf'))
    texts=[p.get_textpage().get_text_range() for p in doc]
    alltext='\n'.join(texts)
    for needle in ('Frozen transfer across degree','449,791','972','819','838,188'):
        assert needle in alltext, f'PDF text missing {needle}'
    pages=[i for i,t in enumerate(texts) if any(s in t for s in ('Frozen transfer across degree','Frozen primary policies:','COMMIT logit for the exact','CT: one neighbor','Lost progress','Frozen degree transfer at'))]
    # Also render the entire contiguous new section, including floated tables.
    first=next(i for i,t in enumerate(texts) if 'Frozen transfer across degree' in t)
    last=next((i for i in range(first+1,len(texts)) if 'Discussion' in texts[i]),min(first+8,len(texts)-1))
    pages=sorted(set(pages+list(range(first,min(last+1,len(doc))))))
    review=ROOT/'paper-review';review.mkdir(exist_ok=True)
    for i in pages:doc[i].render(scale=1.25).to_pil().save(review/f'page-{i+1:02d}.png')
    unresolved=[line for line in (result.stdout+result.stderr).splitlines() if re.search(r'undefined|missing character|error:',line,re.I)]
    atomic_json(ROOT/'paper-build.json',dict(compiled=True,pdf_sha256=file_hash(PAPER/'main.pdf'),pages=len(doc),
        reviewed_pages=[i+1 for i in pages],compiler_sha256=file_hash(compiler),
        source_sha256=file_hash(path),table_sha256=file_hash(PAPER/'tables/degree-generalization.tex'),
        figures={str(p):file_hash(p) for p in (PAPER/'figures').glob('degree-*.pdf')},
        unresolved_diagnostics=unresolved,table_rows=len(lines),new_training_updates=0))
    assert not unresolved, unresolved
    print(json.dumps(dict(compiled=True,pages=len(doc),reviewed_pages=[i+1 for i in pages])))
