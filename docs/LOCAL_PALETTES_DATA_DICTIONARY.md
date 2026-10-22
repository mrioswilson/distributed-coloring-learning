# Data dictionary and scope of the observations

## Graph evaluation

`graph-cases.csv` has 18,848 model–graph–regime–tape rows: 30 frozen main models
and an exact teacher, each on 608 cases. A treatment's 760 cases per regime are
five networks × 76 specifications × two tapes, not 760 independent graph draws.
There are 73 exact graph fingerprints; five specifications repeat four historically
examined clique structures. `fresh-random-summary.csv` is a separate design subset.

* `success`: all vertices fixed and no final monochromatic edge. This can coexist
  with a rule-violating authorization in a MEAN rollout; it is not a safety certificate.
* `unsafe_commits`: participating pending vertices accepted by learned COMMIT but
  rejected by the exact fixed/simultaneous conflict rule. This is different from
  final edge conflicts. A conflicting proposer may itself fail to commit.
* `progress_cutoff`: a normally completed forward reached 256 rounds with pending
  vertices. `unsafe_failure` records an improper result or unsafe incomplete case.
  Neither is an OOM or technical timeout. There were no technical failures here.
* `out_of_list`: implementation violation, separate from proposing an occupied
  color that is in the node's list. All main rows have zero out-of-list proposals.
* `exact_eligible_commits`, `false_rejects`: read-only intervention diagnostics
  for the same state/proposals, never an inference shield. Their additional
  observer was present for all SUM/MEAN/DeepSets/Transformer cases and 304 RAW
  cases. Earlier cases lack it; missing values must not be interpreted as zero.

Per-round traces are in `graph-records.jsonl.gz` and runtime JSONs. `pending` is
measured before the update. `simultaneous` counts participating pending vertices
with at least one adjacent same-color proposer. The historical field name
`pair_collisions` instead counts **all pending node pairs** with equal candidates,
including nonedges and nonparticipants: it is a global concentration summary,
not an edge-conflict count or a conditional candidate law. `distinct` and
`max_candidate_mass` are likewise autonomous round summaries. For conditional
distribution claims use the fixed-state arrays, not these state mixtures.

## Fixed-state candidate laws

There are 3,255 laws = 35 checkpoints × 93 states: 85 constructed local-list
neighborhoods, six new frozen autonomous LOCAL states, and two historical GLOBAL
states. The five original E1 checkpoints are separate from the five new
NORMALIZED checkpoints. Constructed neighborhoods include a witness extending
high-color fixed neighbors with private pending leaves; they are not invalid
local-list stars whose degree-one leaves illegally hold large colors.

`candidate-counts.npz` packs estimation/test counts, available colors, midpoint
choices and boundary probes. Keys are `MODEL__STATE__FIELD`. Raw per-context NPZs
and independent classifier banks have hashes in `candidate-array-manifest.json`.
The law-estimation stream has 32,768 IID inputs; the fidelity/test stream has
8,192 independent inputs. All six stream seeds per context are distinct and
their PCG64 initial states are saved. No model probabilities are fitted on B3
test banks. The same teacher/model draw is paired only for B1 fidelity.

`fidelity` compares the model and teacher using identical clipped float32 rc.
`fidelity_float64_teacher` separately uses the original float64 teacher input.
`teacher_float_precision_disagreement` isolates that conversion. Midpoint and
nextafter/boundary probes are deterministic and have no binomial intervals.

TV/JS/Hellinger compare the candidate law to uniform over the exact available set.
`effective_entropy` = exp(H); `effective_collision` = 1/Σq². They are not
interchangeable. TV uncertainty is a conservative pointwise multinomial bound;
equivalence flags mean its upper bound is ≤.05, not exact equality and not a
simultaneous claim about every state. `candidate-sample-audit.csv` separately
counts forbidden observations in estimation, independent test, quadrature,
boundaries and classifier test. Zero observed counts are not zero true mass.

## Candidate-only classification

`distinguishability.csv` has 1,260 empirical cells: 35 checkpoints × six fixed
contexts × six m values. Each uses 256 test trials per source. Nested prefixes
pair the m values within a curve. The observer receives candidates, not rc/rp,
logits, hidden activations, or independently evolving policy trajectories.

Model likelihoods have pseudocount .5 per palette slot. Teacher support is known;
an observation outside it has infinite likelihood ratio. The equal-prior accuracy
threshold is zero with independent randomized ties. The level-.05 test uses
1,024 independent teacher calibration trials and order statistic 974, rejecting
strictly above it. The resulting FPR guarantee is marginal over calibration;
a realized 256-trial FPR can exceed .05. All FPRs are reported, not adjusted.

The sample-size flag requires BOTH observed power ≥.8 and FPR ≤.05. A missing
flag may reflect either component; it does not necessarily mean low power.
Wilson intervals and true-law product bounds retain their stated scope.
Plug-in estimated-law bounds are not exact learned-law Bayes certificates.

`singleton-product-laws.csv` is a separate post-hoc analytic readout. Its 6,090
rows are not additional empirical classification experiments. For a singleton
null, TV(P^m,Q^m)=1−(1−ε)^m exactly in the unknown forbidden mass ε. Binomial
intervals cover all 1,015 singleton model–state laws simultaneously by Bonferroni.
Reported transformed estimates/bounds do not prove exact equality when no error
was observed. The support test requires no fitted threshold and has exact FPR zero.

## Controlled progress and mathematical claims

Clique tables use p=1/2 and exact COMMIT; inserting an estimated categorical law
produces a plug-in prediction, not a universal theorem about a checkpoint.
The eighteen heterogeneous experiments query three seed-41000 policies on the
same six whole states, with 2,048 candidate draws and 8,192 participation draws
per pending vertex and 256 fresh joint trials. Their NPZs separate candidate,
participating-candidate, exact-confirmed and learned-confirmed histograms per node.
Plot error bars cover observed trial variability, not prediction uncertainty.

The independent post-hoc progress replication preserves the original outlier.
Its fresh streams and increased counts are recorded separately; its GPU time is
charged to the same twelve-hour budget. It does not turn nonrejection into a proof.

Safety certificates concern exact stored real coefficients or an explicitly
conditional IEEE/gamma error model. They do not establish universal neural
termination. Negative COMMIT certificates are finite legal-state interventions;
they do not alone establish that the autonomous candidate policy visits them.
The professor's 1/4 bound is a theorem for independent uniform available-color
proposals and exact rejection, never attributed to the learned policies.

Initial absolute-only checker output and tests blocked by an actively held GPU
lock are preserved as diagnostics. They are superseded by the correct combined
absolute/relative check, final trained-network verification and final test XML.
No weights were changed to repair those verification issues.
