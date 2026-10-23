"""Append the verified factorial contribution while preserving earlier sections."""
from pathlib import Path
from .common import *
from .report import facts,intervention_summary,ARCHES,NAMES

def make_paper():
    f,s,a,fc,events,tests,target=facts()
    counts=[next(r for r in s['counts'] if r['stratum']=='main' and r['architecture']==arch) for arch in ARCHES]
    table='\n'.join(f"{NAMES[r['architecture']]} & {r['robust']}/240 & {r['old']}/240 & {r['robust_only']} & {r['old_only']} \\\\" for r in counts)
    interactions=[]
    for arch in ARCHES:
        r=next(r for r in fc if r['group']==arch and r['event']=='used_le_delta' and r['contrast']=='full-corners')
        interactions.append(f"{NAMES[arch]} & {r['estimate']:+.3f} & [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] \\\\")
    tv=min(r['tv_lower'] for r in target)
    both=[r for r in events if r['stratum']=='main' and r['n']==512 and r['delta']==100 and r['architecture']=='normalized']
    tv_both=min(max(q['tv_lower'] for q in both if q['graph']==r['graph'] and q['model']==r['model']) for r in target)
    def effect(group,event,name):
        r=next(r for r in fc if r['group']==group and r['event']==event and r['contrast']==name)
        return f"{r['estimate']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
    ci50=[next(r['ci'] for r in fc if r['group']=='normalized' and r['event']=='uses_top_color' and r['contrast']==f'n{n}-d50') for n in (128,1024)]
    reduction50=max(0.,max(0.,ci50[0][0],-ci50[0][1])-max(map(abs,ci50[1])))
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

For E1 at degree 50, the top-appearance risk difference is
{effect('normalized','uses_top_color','n128-d50')} at 128 vertices and
{effect('normalized','uses_top_color','n1024-d50')} at 1,024 vertices.
Thus this observable discrepancy attenuates in that regime. At degree 100,
the adjacent 256-to-512 change in the missing-color risk difference is
{effect('normalized','used_le_delta','n512-n256|d100')}.
Already at $n=128,d=100$, the top-appearance difference is
{effect('normalized','uses_top_color','n128-d100')}, despite a missing-color
risk difference of {effect('normalized','used_le_delta','n128-d100')}.
The two observables therefore supply different information; an uninformative
missing-color event is not evidence of equal complete laws.
The simultaneous cell bands imply a reduction of at least {reduction50:.3f}
in the absolute top-appearance risk difference between 128 and 1,024 vertices
at degree 50. This conservative deduction orders that fixed-cohort observable,
not the unknown complete-law TV distances.

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
Using both preregistered events, every one of these fifteen pairs has TV at
least {tv_both:.3f}; maximizing two bounds from their same simultaneous family
preserves its coverage. The effect is heterogeneous across checkpoints.
Figures~\\ref{{fig:factorial-used}}--\\ref{{fig:factorial-replication}}
retain occupancy distributions, all five training replicas and the independent
replication on the original edge list. Leave-one-graph/checkpoint ranges are
sensitivity analyses rather than population confidence intervals.
'''
    text+='\nArchitecture-level missing-color differences at $n=512,d=100$ are '
    text+='; '.join(NAMES[arch]+': '+effect(arch,'used_le_delta','n512-d100') for arch in ARCHES)+'. '
    text+='These contrasts retain a common probability scale and are not rankings by rejection frequency.\n'
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
laws. With independent initial candidates of law $q$, independent participation
of probability $p$ and the exact conflict COMMIT rule, the unconditional first-round reference is
\[
 \Pr(v\text{ confirms }c)=p q_c(1-p q_c)^d.
\]
Every neighbor must avoid participating with candidate $c$. This nonlinear
relation does not make different vertices' confirmations independent, and
we do not assume the learned COMMIT head always equals that exact rule.
The ideal teacher has $p=1/2$, $q_c=1/(d+1)$.
Ideal continuous teacher randomness also has global color-permutation
symmetry on regular graphs, implying a uniform single-vertex marginal and
$E_L[U]=K P_L(T)$ conditional on the invariant completion event. It does not
imply uniformity over proper colorings on arbitrary regular graphs. The ideal
argument is distinct from a bitwise floating-point certificate.

The earlier clique result instead uses transitivity of vertex permutations
on the proper bijective colorings. Conditional uniformity there can coexist
with nonuniform candidate proposals. General regular graphs admit multiple
color-class profiles and lack that transitivity, so clique equality does
not establish general complete-law transfer beyond those symmetry conditions.

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
    target_graph=next(g for g in graphs() if g['stage']=='main' and g['n']==512 and g['delta']==100 and g['rep']==0)
    probe=read(OUT/'candidate-probes'/f"{target_graph['id']}__normalized-s41004.json")
    initial_s4=next(r['q_top'] for r in probe['probes'] if r['round']==0)
    final_s4=[r['probability_model'] for r in events if r['stratum']=='main' and r['n']==512 and r['delta']==100 and r['model']=='normalized-s41004' and r['event']=='uses_top_color']
    text+=f'''
The distinction between initial proposals and terminal laws is quantitative:
for E1 \\texttt{{normalized-s41004}}, the initial 4,096-point top-color mass
at $K=101$ is {initial_s4:.6f}, yet the mean final top-color appearance across
the three new priority graphs is {sum(final_s4)/len(final_s4):.3f}.
Availability changes along a trajectory; a zero initial grid mass cannot
certify absence in every later state.
'''
    representatives=read(OUT/'representative-heldout-witnesses.json')
    for r in representatives:
        e=r['event']
        if e['kind'] not in ('structural','structural_threshold'):continue
        name=e['name'].replace('_',r'\_');threshold=e.get('threshold',e.get('event_threshold'))
        text+=f'''
For an illustrative {NAMES[r['architecture']]} checkpoint
\\texttt{{{r['model'].replace('_', chr(92)+'_')}}} on $n={r['n']},d={r['delta']}$,
the held-out event \\texttt{{{name}}}$\\le {threshold:.8g}$ has probabilities
{r['probability_Luby']:.3f} versus {r['probability_model']:.3f}, risk difference
{r['difference']:+.3f} with simultaneous interval
[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}], giving TV at least {r['tv_lower']:.3f}.
'''
    text+=r'''
Here mean color uses the numeric labels and class entropy uses natural logs.
Each witness was selected only on the main bank and measured on the independent
125-run holdout. Simultaneous exact-binomial bounds over the entire 855-witness
family allow choosing these examples after measurement. They illustrate
different observable discrepancies; they are not universal explanatory or
causal rankings of features.

For heterogeneous prefix lists, every completed assignment obeys
$U\le\min_t[t+\#\{v:d(v)\ge t\}]$. Five of the eight secondary graphs
therefore force $U\le d$ under every generator (for example $U\le44$ on the
1,024-node degree-100 graph). Agreement on that event has no discriminating
content in these cells; the full test, top-color event and other profiles
remain informative targets. Irregular results are not pooled with regulars.
'''
    irregular_counts=[next(r for r in s['counts'] if r['stratum']=='secondary' and r['architecture']==arch) for arch in ARCHES]
    text+='\nIn the separate 120-hypothesis irregular family, the V2/old-MMD rejection counts are '
    text+='; '.join(f"{NAMES[r['architecture']]}: {r['robust']}/40 versus {r['old']}/40" for r in irregular_counts)+'. '
    text+=r'''Figure~\ref{fig:factorial-irregular} reports top-color appearance differences,
which retain information when missing-color occupancy is forced by the lists.
Its fixed-checkpoint mean intervals are linear images of the original
simultaneous elementary CP bounds; the shared teacher enters once. One graph
per irregular cell limits generalization, and maximum degree does not specify
its full palette-capacity profile.
'''
    iv=intervention_summary()
    if iv and iv['cells']:
        text+=f'''
\\paragraph{{Separate exploratory candidate intervention.}}
A prospectively recorded extra experiment replaces only the candidate
channel by exact local teacher proposals on the intervened policy's own
state. Learned participation, original COMMIT, checkpoints and cutoff are
unchanged; no safety mask is added. Across {iv['cells']} of 60 planned cells
(all checkpoints on three new priority graphs and the historical graph),
the status is \\texttt{{{iv['status'].replace('_', chr(92)+'_')}}}.
The sample count was selected exclusively from a separate timing pilot,
within the same total GPU budget. These observations do not enter the
original 720 hypotheses.

Common-random-number coupling, with independent pairs, gives simultaneous
upper bounds on conditional TV between the intervened policy $I$ and Luby;
paired outputs are never treated as independent two-sample observations.
The bound is at most .05 in {iv['equivalent']}/{iv['cells']} cells.
Original and intervention event intervals jointly support a positive
lower bound on $\\operatorname{{TV}}(P_M,P_I)$ in
{iv['effects']}/{iv['cells']} cells, with coverage at least .95.
We recorded {iv['unsafe']} unsafe COMMIT decisions and did not repair them.
A large coupling upper bound does not prove residual discrepancy, since
the coupling need not be optimal. This controlled computational intervention
addresses the candidate channel; it does not identify complete mediation
or separate degree from palette causality.
'''
        supplement=OUT/'candidate-intervention/witness-effects.json'
        if supplement.exists() and read(supplement)['rows']:
            sw=read(supplement)
            text+=f'''
A separate supplementary addendum was recorded after original witness
examples were inspected but before any intervention sample. It retains the
single honestly selected original event for each of all sixty cells and
uses its original independent holdout. Simultaneous CP bounds on 360 event,
mismatch and success probabilities, followed by direct fixed-event comparison
under the intervened policy and the triangle inequality,
give positive lower bounds on $\\operatorname{{TV}}(P_M,P_I)$ in
{sw['effect_positive']}/{sw['cells_completed']} cells. The simultaneous
coverage is at least .95 for this supplementary family; it is separate from
the preceding event-effect family, and no joint .95 guarantee is claimed
over their union. No new simulations or event reselection are involved.
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
proposal/acceptance chains. Any candidate intervention has the separate
scope stated above. We cannot order all complete-law TV distances, generalize
three graphs and five trainings universally, or infer complete mediation
from association. On regular graphs, degree and palette size vary
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
      ('09-candidates-and-global-law','candidates','Same-state candidate quadrature and independent final samples: association does not identify causality.'),
      ('11-irregular-cohort','irregular','Separate irregular cohort: top-color appearance risk differences and simultaneous CP envelopes, averaged over the five fixed checkpoints. A dagger marks list profiles that force the missing-color event A to be true under every completed generator. All eight graphs are retained.')]
    if iv and iv['cells']:
        captions.append(('12-candidate-intervention','intervention','Separate exploratory candidate-channel intervention on four priority graphs. The supplementary witness/coupling family gives simultaneous 95 percent upper bounds for intervened versus Luby and lower effect bounds for original versus intervened. All original-policy tests remain separate.'))
    for name,label,caption in captions:
        text+='\n\\begin{figure}[p]\n\\centering\n\\includegraphics[width=\\linewidth,height=.8\\textheight,keepaspectratio]{figures/factorial-'+name+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:factorial-'+label+'}\n\\end{figure}\n'
    text+='\n\\clearpage\n'
    Path('paper/ucgnn/factorial-size-degree.tex').write_text(text)
    abstract=f'''A subsequent size--degree factorial study uses three new regular graphs in
each of sixteen cells, unchanged checkpoints and 500 testing attempts per
generator. The smallest priority-case E1 TV lower bound from two preregistered
events over three new 512-node degree-100 graphs and five checkpoints is {tv_both:.3f}.
Simultaneous event contrasts separate structural changes from rejection rates,
including a reduction of at least {reduction50:.3f} in E1's absolute top-color
appearance difference from 128 to 1,024 vertices at degree 50. Size effects
therefore depend on the regime; degree remains confounded with palette size.
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
