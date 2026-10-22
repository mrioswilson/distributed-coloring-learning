"""Append the verified factorial contribution while preserving earlier sections."""
from pathlib import Path
from .common import *
from .report import facts,ARCHES,NAMES

def make_paper():
    f,s,a,fc,events,tests,target=facts()
    counts=[next(r for r in s['counts'] if r['stratum']=='main' and r['architecture']==arch) for arch in ARCHES]
    table='\n'.join(f"{NAMES[r['architecture']]} & {r['robust']}/240 & {r['old']}/240 & {r['robust_only']} & {r['old_only']} \\\\" for r in counts)
    interactions=[]
    for arch in ARCHES:
        r=next(r for r in fc if r['group']==arch and r['event']=='used_le_delta' and r['contrast']=='full-corners')
        interactions.append(f"{NAMES[arch]} & {r['estimate']:+.3f} & [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] \\\\")
    tv=min(r['tv_lower'] for r in target)
    text=r'''\section{How Graph Size and Degree Affect Distributional Generalization}
\label{sec:factorial}
\paragraph{Motivation and controlled design.}
The preceding robust study found an occupancy-event TV lower bound of .894
on one 512-node degree-100 regular graph, while supporting approximate
conditional equivalence in seven small graphs. To separate observed size and
degree effects, we preregistered a new factorial study with all fifteen
historical epoch-120 checkpoints, without further training. The primary grid
crosses $n\in\{128,256,512,1024\}$ and $d\in\{3,10,50,100\}$, with three
independent connected labeled regular realizations in each cell. Exact vertex
counts, all degrees, simplicity, connectivity, seeds and distinct labeled
edge hashes were audited. The generator is not assumed exactly uniform over
all regular graphs, and three realizations cannot characterize that population.

Cost pilots fixed 500 fresh testing attempts per generator and a separate
125-attempt explanatory holdout. The secondary cohort has one heterogeneous
graph per cell of $n\in\{128,1024\}$ crossed with all four degrees, including
low-degree vertices, intermediate degrees and a small hub group. Degree and
palette histograms are archived. A third stratum replicates the exact
historical edge list using new independent seeds. Every local palette remains
$\{0,\ldots,d(v)\}$; no global-degree input or new action shield is introduced.

\paragraph{Complete-law tests and estimands.}
An observation is a complete labeled coloring vector, not a vertex. The
principal law conditions on proper completion by round 128; improper
completion, cutoff and technical error remain separate outcomes. We reuse
the frozen omnibus V2, its honest selection/evaluation split, local and
pair/triple witnesses, structural features and characteristic full-vector
kernel unchanged. Joint label permutations number 19,999, with inclusive
ties and plus-one p-values. Holm controls the 720 primary graph--checkpoint
hypotheses at .05. The 120 irregular and 15 historical-replication hypotheses
have separate families. The old Hamming MMD is evaluated on exactly the same
new bank in its own method family. Event, completion and independent witness
analyses also have separate families; no .05 FWER is asserted over their union.

The primary structural events are
\[
 A_{n,d}=\{U(X)\le d\},\qquad
 T_{n,d}=\{\text{color }d\text{ occurs}\}.
\]
The first means that at least one palette color is absent, which need not be
the top color. A saturated stratified binomial model has one event probability
per fixed graph/generator bank, conditional on its valid count. Architecture
summaries average the three graphs and five fixed checkpoints equally, counting
the shared teacher bank once. We preregistered 1,800 cell, adjacent-size,
adjacent-degree and probability-scale interaction contrasts. Elementary exact
binomial intervals and weighted independent-run Hoeffding intervals receive
.025 error budgets each; their intersections cover all contrasts simultaneously
with probability at least .95. These are fixed-cohort intervals, rather than
population intervals assuming many independent graphs or trained networks.

An event-difference interval $[l,u]$ yields a same-graph lower bound
$\operatorname{TV}(P_L,P_M)\ge\max(0,l,-u)$. Lower bounds do not order the
unknown true TV distances. Across sizes the vector spaces also differ, so
we compare consistently defined functions and events, rather than computing
cross-dimensional TV. Small MMD or a nonrejection supplies no useful TV upper
bound without an additional quantitative argument.
'''
    text+=f'''
\\paragraph{{Results across size and degree.}}
Table~\\ref{{tab:factorial-tests}} retains all gains and losses of V2 on the
same new samples. Counts are not distance estimates and cannot be directly
ranked against the earlier 265-comparison architecture cohorts.
\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule Architecture & V2 & Old MMD & V2 only & MMD only\\\\
\\midrule
{table}
\\bottomrule
\\end{{tabular}}
\\caption{{Regular factorial cohort: 240 comparisons per architecture;
Holm is applied across all 720 hypotheses separately for each method.}}
\\label{{tab:factorial-tests}}
\\end{{table}}

Figures~\\ref{{fig:factorial-discrepancy}}--\\ref{{fig:factorial-degree}}
give the full event-effect matrix and its slices with simultaneous intervals.
These are measured changes in occupancy probability, not merely changes in
test power. Top-color appearance is separately reported in
Figure~\\ref{{fig:factorial-top}}, including regimes where $U\\le d$ is nearly
certain under both generators. We impose no monotone full-law TV interpretation.

\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lrr}}
\\toprule Architecture & Four-corner interaction for $A$ & Simultaneous interval\\\\
\\midrule
{chr(10).join(interactions)}
\\bottomrule
\\end{{tabular}}
\\caption{{Extreme-level difference-in-differences of event risk differences.
All adjacent interactions and individual-checkpoint contrasts are archived.
This is a probability-scale fixed-cohort interaction, not causal identification.}}
\\label{{tab:factorial-interaction}}
\\end{{table}}

\\paragraph{{Independent replication and checkpoint variation.}}
Archived historical data exactly reproduce .008 versus 1.000 for $U\\le100$
in \\texttt{{normalized-s41000}} on the original graph. They are never pooled
with new outcomes. Across the three new $n=512,d=100$ graphs and five E1
checkpoints, the smallest simultaneous event-TV lower bound is {tv:.3f}.
Figures~\\ref{{fig:factorial-used}}--\\ref{{fig:factorial-replication}}
retain occupancy distributions, all five training replicas and the independent
replication on the original edge list. Leave-one-graph/checkpoint ranges are
sensitivity analyses rather than population confidence intervals.
'''
    text+=r'''
\paragraph{Mechanistic evidence.}
Read-only observers record proposals, participation, exact availability
references, accepted colors and terminal classes on eight fresh trajectories
per generator and selected graph. Every checkpoint is also probed on identical
teacher states with midpoint grids of 1,024 and 4,096 private uniforms. A grid
zero is not a proof of zero operational mass. Candidate biases and independent
terminal event estimates are related in Figure~\ref{fig:factorial-candidates};
the proposal/COMMIT accounting remains descriptive, not an intervention or
causal mediation estimate.

At an all-uncolored nonisolated vertex, MAX returns the same finite message
type regardless of neighbor multiplicity. Thus at fixed $K$, the initial
candidate function is independent of $n$ and the particular regular edge
realization. This does not imply invariance of subsequent states or terminal
laws. Ideal continuous teacher randomness also has global color-permutation
symmetry on regular graphs, implying a uniform single-vertex marginal and
$E_L[U]=K P_L(T)$ conditional on the invariant completion event. It does not
imply uniformity over proper colorings on arbitrary regular graphs. The ideal
argument is distinct from a bitwise floating-point certificate.

\paragraph{Exploration and limitations.}
Six binomial predictor forms---size, degree/palette, $n/K$, density, additive
and interaction---are compared with checkpoint intercepts and whole-graph
replicate holdouts. Their predictive ranking is not claimed as confirmatory
causal variable selection. Structural projections, per-color profiles and
MMD effects retain their stated scale and uncertainty limitations. Selected
events are validated on the independent explanation bank; prospectively
measuring every color permits simultaneous rather than naive selected-color
intervals. A small normalized occupancy-mean difference can coexist with a
large discrepancy of an occupancy-tail event.
'''
    text+=f'''
The final ledger charged {a['gpu_process_hours']:.3f} of 12 GPU-process hours,
including pilots and diagnostic trajectories. Statistical analyses use CPU.
The audit checked {a['new_algorithm_seeds']:,} distinct new algorithm seeds,
without historical overlap, unchanged checkpoint/source hashes, a complete
teacher-bank replay, two omnibus replays and multiplicity recalculation.
Confirmation statuses are {s['statuses_confirmation']} in the order proper,
improper, cutoff and technical error. The protocol was committed as
\\texttt{{{read(OUT/'protocol-commit.json')['commit'][:12]}}} before confirmation.
'''
    text+=r'''
\paragraph{What We Can Prove, What We Observe, and What Remains Open.}
We can prove ideal teacher color symmetry, initial MAX-state invariance,
conditional permutation validity and stated coverage under independent runs.
We observe finite-cohort effects, interactions, replications and diagnostic
proposal/acceptance chains. We cannot order all complete-law TV distances,
generalize three graphs and five trainings universally, or infer a causal
mechanism from association. On regular graphs, degree and palette size vary
together, leaving their separate causal effects unidentified. Previous
small-graph equivalence statements retain their original scope; large-graph
nonrejections establish no approximate equivalence.
'''
    captions=[
      ('01-significance','significance','Global tests; each cell contains three graphs and five checkpoints. Rejection fractions are not distances.'),
      ('02-event-discrepancy','discrepancy','Differences in the probability of using at most the degree many colors, with simultaneous fixed-cohort intervals.'),
      ('03-size-effect','size','Size slices at fixed degree, with simultaneous fixed-cohort uncertainty.'),
      ('04-degree-effect','degree','Degree/palette slices at fixed size, without a monotone TV assumption.'),
      ('05-top-color','top','Top-color appearance probabilities with simultaneous elementary CP envelopes.'),
      ('06-used-colors','used','Occupancy distributions on three new priority graphs and the historical graph with independent fresh data.'),
      ('07-checkpoints','checkpoints','All five checkpoints; faint points retain individual graph variation.'),
      ('08-target-replication','replication','Independent priority-case E1 replications, without pooling historical outcomes.'),
      ('09-candidates-and-global-law','candidates','Same-state candidate quadrature and independent final samples: association does not identify causality.')]
    for name,label,caption in captions:
        text+='\n\\begin{figure}[p]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/factorial-'+name+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:factorial-'+label+'}\n\\end{figure}\n'
    text+='\n\\clearpage\n'
    Path('paper/ucgnn/factorial-size-degree.tex').write_text(text)
    abstract=f'''A subsequent size--degree factorial study uses three new regular graphs in
each of sixteen cells, unchanged checkpoints and 500 testing attempts per
generator. The smallest priority-case E1 occupancy-event TV lower bound over
three new 512-node degree-100 graphs and five checkpoints is {tv:.3f}.
Simultaneous event contrasts separate structural changes from rejection rates,
while degree remains confounded with local palette size.
'''
    Path('paper/ucgnn/tables/factorial-abstract.tex').write_text(abstract)
    path=Path('paper/ucgnn/main.tex');source=path.read_text()
    if '\\input{factorial-size-degree}' not in source:
        source=source.replace('\\input{robust-distributions}','\\input{robust-distributions}\n\\input{factorial-size-degree}')
    marker='\\IfFileExists{tables/robust-abstract.tex}{\\input{tables/robust-abstract.tex}}{}'
    if 'tables/factorial-abstract.tex' not in source:
        source=source.replace(marker,marker+'\n\\IfFileExists{tables/factorial-abstract.tex}{\\input{tables/factorial-abstract.tex}}{}')
    path.write_text(source)

if __name__=='__main__':make_paper()
