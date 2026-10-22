"""Write the final protocol only after the independent instrument gate passes."""
import shutil,subprocess
from .common import *
from dcl.study.storage import now,provenance

EXTENDED=['edge','path3','path4','triangle','cycle4','star3','paw','star-d200','regular-n256-d100']

def main():
    assert read(OUT/'instrument-acceptance.json')['passed']
    if (OUT/'freeze.json').exists():raise RuntimeError('Final protocol already frozen; no overwrite')
    assert not (DATA/'queue-confirmatory.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('graphs.json','checkpoints.json'):shutil.copy2(OLD/'results/ucgnn-global-distribution'/name,OUT/name)
    for s in specs():assert file_hash(s['checkpoint'])==s['sha256']
    old=read(OLD/'data/ucgnn-global-distribution/queue.json');forecasts=[]
    for t in old['tasks']:
        if t['stage']!='confirmatory':continue
        runs=2250 if t['graph'] in EXTENDED else 750
        forecasts.append(dict(graph=t['graph'],model='Luby' if t['model'] is None else t['model']['label'],runs=runs,
            old_reference_runs=t['m'],old_reference_seconds=t['seconds'],predicted_seconds=runs*t['seconds']/t['m'],reserved_seconds=1.5*runs*t['seconds']/t['m']))
    atomic_json(OUT/'prospective-resource-costs.json',forecasts)
    import pandas as pd
    pd.DataFrame(forecasts).to_csv(OUT/'prospective-resource-costs.csv',index=False)
    budget=read(DATA/'budget.json')
    if not budget.get('legacy_initialization_reserve'):
        budget['seconds']+=60.;budget['legacy_initialization_reserve']=60.
        atomic_json(DATA/'budget.json',budget)
    predicted=sum(r['reserved_seconds'] for r in forecasts)+budget['seconds']
    assert predicted<43200,('Prospective resource matrix exceeds cap',predicted)
    text='''# Frozen robust complete-coloring confirmation protocol

## Status and separation

This protocol is written only after independent validation round two passed.
No new teacher-versus-neural confirmatory coloring bank exists at freezing.
DEVELOPMENT, validation rounds 1/2, CONFIRMATORY and EXPLANATION have separate
data and seed namespaces. The first-version instrument and its relocation
failures are retained. The acceptance target is sensitivity for named
alternative classes, never universal finite-sample power.

## Scientific laws and cohort

For each fixed labeled G test equality of the Luby and frozen-network laws
conditional on proper completion by 128 rounds. Each row is one complete
execution. Record invalid-complete, cutoff and technical categories, without
silently discarding failures. When failures occur, also compute status-law
and coarsened-unconditional kernel tests, in separate secondary families.

Exactly the fifteen recent epoch-120 paper checkpoints, five seeds
41000–41004 for NORMALIZED, DeepSets and LocalPaletteTransformer, with file
hashes in checkpoints.json. No historic E1 replacement, new neural training,
weight tuning or safety shield. Lists C_v={0,...,d(v)}; no global Delta input.
Use the validated historical local adapter and exact local teacher. Private
PCG64 streams are independent per run, vertex and round; converted float32
neural uniforms and float64 teacher uniforms are explicit. TF32 is disabled,
deterministic Torch algorithms enabled. Dynamic state resets across runs.

## Fixed graphs and hypotheses

All 53 historical graph entries, 51 unique labeled graphs, are retained,
with the same directed edges, graph seeds and SHA256 hashes. They include
regular n=256,512 at Delta=2,3,5,10,20,50,100,200; paths/cycles, stars,
irregular trees/random/bipartite, hubs, wheels and cliques. Every architecture
and checkpoint uses the same graph per contrast. star3/star-d3 and
wheel-d3/clique-d3 are family aliases, not independent graph realizations.
bipartite-n256-d3 has actual Delta=4; retain its frozen graph and report 4.
No graph-population or universal monotonic-degree claim is planned.

Primary family: 53*15=795 graph/checkpoint hypotheses, Holm FWER .05. Report
all raw valid p-values, adjusted values, rejections and nonrejections.
Five checkpoints remain separate policies; never pool their output vectors.

## Samples and independent explanation

Generate 750 new runs per graph/generator. For nine fixed entries generate
2250 instead: edge,path3,path4,triangle,cycle4,star3,paw,star-d200,
regular-n256-d100. Primary tests use the first 500 observations per generator:
250 for selection and 250 for evaluation, by preassigned run index. New
baseline MMD uses all the same first 500, allowing a matched sample-budget
comparison. Historical baseline used 250, so its change alone does not
isolate a method effect. Report that distinction explicitly.

For the nine extended entries, secondary tests at total m=250,1000,2000
form one separate 9*15*3=405-hypothesis Holm family, with correlated nested
prefixes allowed. No optional stopping or sample-size escalation occurs.

The final 250 observations of each bank are reserved for explanation,
disjoint from all primary/secondary test prefixes. Select one observable
binary event per cell using primary data only (local categorical or
threshold, joint pair/triple categorical, or structural threshold). Report
Luby/model probabilities, their difference and simultaneous CP intervals
with per-probability error .05/(2*795). Independent Fisher exact event tests
receive a separate 795-hypothesis Holm family. No naive selected-event CI
uses the observations that selected the event. A failed explanatory event
is retained and does not invalidate a global result or provide a mechanism.

## Frozen main statistic

Version two of the honest joint-permutation omnibus, exactly stats.py.
Fit on selection data only: four local, four interaction and four structural
witnesses, a dense marginal witness, and four global kernel witnesses at
gamma=.25,1,4,16. All single vertices are eligible. Fixed seeded pairs and
triples are augmented by the top eight selection-covariance changes for
raw and median-binary colors, with pooled selection standardization. This
all-pair search uses no evaluation labels. It does not search all triples.

On pooled evaluation data center/scale witness columns without source labels.
Use absolute mean differences divided by sqrt(1/t_X+1/t_Y). Retain the
unbiased characteristic full-vector exp(-d_H/n) MMD U, including negatives,
scaled by max(sqrt(2)(1/t_X+1/t_Y)||HKH||_F/N,1e-15), with zero diagonal K.
The statistic is the maximum across these components, coefficient one for
the characteristic term. No color or vertex canonicalization is performed.
Selection size and fixed region construction are not tuned using confirmation.

## Permutations, numerical ties and compute shortcuts

B=19,999 independent uniform fixed-size evaluation-label permutations.
Recompute the entire maximum each time. Inclusive exceedances use tolerance
1e-12 on the standardized omnibus, p=(1+exceedances)/(B+1). Minimum p=.00005
resolves .05/795 and .05/405. Independent seed namespaces include stage,
graph, checkpoint, m and purpose. Baseline uses the exact old unbiased MMD
statistic and its size-dependent tolerance 1e-14*min(1,500/(m+n)), with the
same B and full 500 observations. Pooled effect norm and conservative RKHS
intervals are separate from its unbiased test statistic.

Conservative futility: after >=20 exceedances for every returned component,
the helper may stop and return p=1 for all components. Its theoretical
complete-B p would be no larger, so type-I control is preserved. This is
recorded as a conservative valid p, not a precise full-tail estimate. Strong
results use all B permutations. Record permutations used and p-kind.
No fitted tail extrapolation, repeated significance peeking or early
rejection is allowed. Holm uses all planned hypotheses, including incomplete
cells assigned conservative p=1 and explicitly marked incomplete.

## Small-space reference and equivalence

Recompute exact rational eventual teacher laws and supports on seven small
graphs. Use first 2000 neural rows for finite-support TV estimates. The
105 graph/checkpoint intervals use simultaneous union-bound coverage at
95%; add the teacher eventual/cutoff tail bound. Tolerance TV<=.10 is fixed
here, before confirmation, for this family only. Declare approximate
equivalence only when its upper bound is <=.10. Direct categorical
likelihood-ratio tests use conditional permutation calibration, B=19,999,
and a separate 105-hypothesis Holm family. No sparse-cell chi-square
approximation is used. No TV equivalence claim is planned for large graphs.

## Calibration, power and limits

The passed independent gate and all its failures/limitations are archived.
The final procedure improves mandatory known-star detection and is checked
on local, pair, higher-order, global, rare and huge-support controls. Its
specified >=80% lower Wilson targets apply to named classes and m only.
Unknown relocated triples and rare events remain limitation controls.
Classifier results are complementary: logistic L2/.1,1,10 and nonlinear
GaussianNB, separate training/validation/evaluation, exact hypergeometric
accuracy significance and classwise CP intervals. They do not replace the
main omnibus; their nine planned seed-41000 power-graph tests form a
separate Holm family. No new neural classifier is trained.

Nonrejection is insufficient evidence, not equality or a useful TV bound.
Clique equality is an ideal vertex-equivariance theorem conditional on
success; numerical probes do not certify every floating-point input.
Rejection counts do not measure distance. Report effects, uncertainty and
class-specific sensitivity separately. Compare fixed-size regular grids
descriptively without an unplanned population trend regression.

## Cost, persistence and stopping

The fresh 12 GPU-process-hour cap includes cost pilots, both real-null
sampling rounds, initialization, model sampling and lock-held overhead.
The prospective per-cell matrix scales archived actual timings to this
fixed new allocation with a 50% reserve. Early cost/null initialization
receives an additional conservative 60-second reserve. CPU statistical
workers do not initialize a GPU. A shared nonblocking GPU lock protects
other services. Atomic queues and bank receipts resume completed work;
interrupted active elapsed time is charged conservatively. A BaseException
watchdog bypasses the sampler's ordinary error handler and fires 30 seconds
before remaining budget. No foreign process is stopped. No remote push.
Keep all incomplete cells and all errors if the cap prevents completion.
'''
    Path('docs/ROBUST_GLOBAL_DISTRIBUTION_PROTOCOL.md').write_text(text)
    source={str(p):file_hash(p) for p in Path('src/dcl/robustdist').glob('*.py')}
    atomic_json(OUT/'freeze.json',dict(created=now(),protocol_sha256=file_hash('docs/ROBUST_GLOBAL_DISTRIBUTION_PROTOCOL.md'),
        graphs_sha256=file_hash(OUT/'graphs.json'),checkpoints_sha256=file_hash(OUT/'checkpoints.json'),source_hashes=source,
        extended_graphs=EXTENDED,primary_hypotheses=795,primary_total_per_generator=500,selection_per_generator=250,evaluation_per_generator=250,
        explanation_per_generator=250,B=19999,alpha=.05,secondary_hypotheses=405,planned_banks=848,planned_runs=sum(r['runs'] for r in forecasts),
        forecast_with_reserve_hours=predicted/3600,budget_hours=12,method='honest omnibus version 2',pre_freeze_provenance=provenance()))
    print('Final protocol prepared; commit before new sampling',predicted/3600,flush=True)
if __name__=='__main__':main()
