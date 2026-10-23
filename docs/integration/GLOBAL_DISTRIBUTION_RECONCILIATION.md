# Global distribution reconciliation

## 1. Audit conclusion and remaining boundary

The published findings do not currently constitute a mathematical
contradiction. Pedro's 0/100 rejections concern new E1 weights, degree≤3,
global K4, two output representations, cutoff 256 and family/size-stratified
mixture-MMD tests. Our high-degree results concern different checkpoints,
local palettes and graph-specific conditional full-vector tests.

On six identical archived banks, both statistical procedures agree: no
rejection in the three degree-three cases, and rejection in all three
degree 100 cases. Independently generated known-law controls show a real
finite-sample power difference. Neither observation identifies the cause
of every historical nonrejection. Pedro's exact weights and raw banks are
missing; his original numerical study has not been independently replayed.
This is an explicit unresolved boundary, not an inferred reconciliation.

## 2. Reconstruction of our methodology

Original evidence is preserved in:

* `results/ucgnn-global-distribution/`: initial full-law study,53 fixed graph
  entries,15 immutable checkpoints,250 attempts per principal bank,
 19,999 permutations and a 795-hypothesis Holm family.
* `results/ucgnn-robust-distribution/`: independent 500-attempt banks,
  prospective validation and V2 confirmation on the corresponding cohort.
* `results/ucgnn-size-degree/`:16 regular N/degree cells,48 new graphs,
  five checkpoints of each architecture,500 primary plus 125 independent
  explanation attempts per bank,720 principal hypotheses.

The observed row is a full labeled vector X_G=(c_1,...,c_n), with local
lists C_v={0,...,d(v)}. No vertex is treated as an independent statistical
replicate. Exact Luby/random-trial participation is 1/2, proposals are uniform
available ranks in the ideal rule, and confirmation rejects fixed and
simultaneous neighboring conflicts. The fifteen models are NORMALIZED,
DeepSets and LocalPaletteTransformer at epoch 120, seeds 41000–41004.

The primary law conditions on proper completion by 128 rounds. Improper
completion, cutoff and technical errors remain separate statuses and have
separate operational summaries. The historical numerical sampler and
checkpoint bytes are frozen. Our adapter uses receiver-local palette sizes;
the models do not receive a global maximum-degree input.

V2 selects bounded categorical, interaction, structural and multiscale
witnesses on half the observations, then evaluates a joint maximum with a
characteristic full-vector MMD component on the other half. Whole-row
permutations preserve selection and recompute the joint statistic. Inclusive
ties and p=(1+exceedances)/(B+1), B19,999, give minimum p.00005. Holm is
computed over each full declared family; separate families do not provide
a joint 5% guarantee over their union. Conservative early-stop p 1 values
are refined if they could affect the original Holm rejection set.

The original MMD study rejected 10/265 E1,46/265 DeepSets and 127/265
Transformer comparisons. In independent robust banks, old MMD rejected
21,71,147 and V2 rejected 52,107,181 of 265 respectively. In the factorial
principal cohort, V2 rejected 115,170,180 of 240, versus 49,121,174 for old
MMD on identical rows. These are cohort/test decisions, not TV distances,
population architecture rankings or universal detectability statements.

## 3. Reconstruction of Pedro's methodology and original results

`src/dcl/e1_behavior.py` and `behavior_statistics.py` define the relevant
experiment. `results/e1-retrained/2026-10-21/` contains its protocol,
manifests, training histories, per-graph metrics and distribution tests.
The full reconstruction is in `PEDRO_RESEARCH_RECONSTRUCTION.md`.

Five newly trained canonical E1s use seeds 0–4. These are neither our
seed 41000–41004 cohort nor the earlier historical seed 0–4 weights. His
80 random graph instances cover N8,12,40,160,640 at degree≤3, with a global
palette of four colors; path 8/cycle 8/K4 are separate controls. Each bank
has 256 runs and a 256-round cap. Learned execution continues after collisions
until completion/cutoff, preserving terminal colors and status.

Luby A tapes and model B tapes are independent. Luby B is additionally
coupled to the model on identical private draws. The two-sample test uses
independent A/B rows, whereas B/B disagreement supplies a coupling TV upper
bound. The latter is not a TV lower bound and need not be an optimal coupling.

The kernel averages exp(-gamma d_H/n), gamma 1,4,16, and multiplies by
terminal-status equality. An unbiased MMD statistic is computed separately
within each graph, with 3,999 whole-row label permutations. Independent
permutation statistics are averaged within graph-family/size strata. Holm
covers 100 random-stratum/model/representation tests, including full color
labels and global-color-permutation canonicalization.

Published tables report 0/100 rejections and 102,400/102,400 successful learned
random-graph runs. Ten labeled and eleven quotient graph/model pairs obtain
simultaneous TV≤.05 certificates from exact binomial coupling bounds. These
are affirmative finite-graph approximate-equivalence statements, unlike
nonrejections. They do not establish equality, other graphs, or high-degree
generalization. We checked their counts in the published tables; without
raw banks and weights we cannot authenticate the underlying reruns.

## 4. Explicit comparability matrix

| Dimension | Our principal studies | Pedro behavior experiment | Comparable? |
|---|---|---|---|
| Teacher | p 1/2 random-trial coloring | p 1/2 random-trial coloring | Same ideal rule only after matching lists/tapes |
| Palettes | Local d(v)+1 | Global 4, degree≤3 | Identical on cubic degree 3 graphs; different at lower degrees |
| Architectures | E1, DeepSets, Transformer | Canonical E1 | Only E1 overlaps |
| Checkpoints | Seeds 41000–41004, GPU paper-study cohort | New CPU seeds 0–4 | Different trained functions and data |
| Graphs | Broad degrees/families; factorial up to N1024,degree 100 | N8–640, degree≤3 | Limited overlapping regime, different instances |
| Randomness | Float 64 teacher tapes; declared float 32 model conversion | Float 64 tapes; original float 32 conversion | Explicit numerical tapes must be aligned |
| Observation | Complete labeled coloring | Labeled coloring/status and color-name quotient | Labeled observations can align; quotient loses information |
| Success conditioning | Proper completion by 128 | Joint terminal law by 256 | Different estimands when failures occur |
| Test | V2 honest split omnibus; old single-scale MMD secondary | Mixture MMD, stratified across graphs | Both calibrated, different statistics/null families |
| Sample size | 250 initially,500 principal newer attempts | 256 attempts per graph | Can align in a matched-bank experiment |
| Multiplicity | 795 or 720 graph/checkpoint hypotheses | 100 stratum/checkpoint/representation hypotheses | Rejection percentages cannot be directly ranked |
| Power | Independent controls, including documented blind spots | No equivalent broad positive-control study in published behavior artifacts | Independently crosschecked below |

## 5. Mathematical and statistical audit

For equal sample sizes m, let S be the sum of the pooled zero-diagonal
kernel matrix and D the signed quadratic form with signs-1/+1 for sample
membership. Pedro's statistic

    [S + (2m-1)D] / [2m²(m-1)]

equals the ordinary unbiased within/within/minus-two-cross MMD statistic.
An independent loop-based regression test verifies the identity, including
the terminal-status kernel. No sign, diagonal, or missing-plus-one defect
was found. Inclusive ties give conservative tests at degenerate null laws.
Independent within-graph permutations and a fixed mean across graphs test
the joint stratum null without assuming vertices are independent.

For V2, conditional on the independent selection data, evaluation labels
are exchangeable under equality of the two laws. The witness selection is
fixed; pooled evaluation scaling is label invariant. Consequently the joint
maximum has a valid permutation calibration. Monte Carlo plus-one p-values
are super-uniform with inclusive ties. Holm controls each declared family
under arbitrary dependence, including shared exact-reference banks. These
guarantees require independent full executions and correct data separation.

Color canonicalization is many-to-one. The proper point masses(0,1,0) and
(2,0,2) have disjoint labeled laws but the same canonical partition. Equality
of quotient laws does not imply equality of labeled laws. Similarly, equal
candidate marginals do not imply equal full-coloring laws, and candidate
disagreement does not force different terminal laws. On K_n with exactly n
colors, vertex-equivariant protocols with exchangeable private input have a
uniform conditional proper-completion law, by transitivity on the n! proper
vectors. This proves a scoped equality case without using nonrejection.

The joint terminal-law and conditional-success-law comparisons coincide on
observed banks with all successful rows; this does not prove population
failure probability zero. For an exact teacher with eventual termination,
TV between its eventual law and its law conditioned on completion by R is
at most P(T>R). For a learned policy without a termination theorem, no
unrestricted eventual law is assumed to exist.

## 6. Cross-implementation traces

Independent scalar implementations audited all seven native teachers over
103 small graph representatives and two tapes each. Eight LUBY cases on
path 8, star 3, K4 and cubic 40 used identical explicit float 32 arrays in both
the native and canonical teacher, with common K4. Every pre-state,
participation, available count, candidate, COMMIT and post-state matched.
The complete star tapes and native traces are under
`results/integration/teachers/`. No claim is based on merely giving two
different RNG consumers the same numerical seed.

This establishes agreement in the declared finite checks and supports the
common mathematical teacher definition. It does not equate local-list and
global-list protocols on irregular graphs or reproduce unavailable weights.

## 7. Matched historical-bank experiment

Plan: `STATISTICAL_CROSSCHECK_PROTOCOL.md`; results:
`results/integration/statistics/matched-archives.{csv,json}`. Six already
observed archival banks use 500 successful rows per generator, identical
graph, frozen checkpoint, local lists, cutoff and teacher outcomes. Both
methods use 19,999 permutations. V2 exactly reproduces every original
statistic/p-value, including its recorded conservative-stop convention.

| Fixed graph / checkpoint cohort | V2 | Pedro mixture | Scope |
|---|---|---|---|
| First N128/degree 3 regular graph; s 41000 of all three architectures | 0/3 reject | 0/3 reject | Method-specific six-test Holm families |
| First N512/degree 100 regular graph; same three architectures | 3/3 reject | 3/3 reject | All raw p.00005; six-test adjusted p.0003 |

The original high-degree V2 family-adjusted p-values remain.036, not.0003;
the latter belongs only to this six-case reanalysis. This is exploratory
method comparison on historical data, not independent confirmation of the
historical scientific hypotheses. It demonstrates that the two procedures
can agree when the compared data are actually aligned.

## 8. Fresh known-law controls and power

On a degree 200 star, both laws have iid binary leaves, while the center is
uniform on 199 colors versus a 100-color subset. All outputs are proper with
the local lists. The true TV is 99/199≈.497487. The mixture's population MMD²
is 9.569688e-6; its three component effects are archived in the JSON.

Twenty fresh independent experiments per law/size, both tests on the same
rows, used B999 at nominal.05. No procedure was retuned after these trials.

| Law / m | Mixture detections /20 | V2 detections /20 |
|---|---:|---:|
| Same-law star,256 | 0 | 1 |
| Same-law star,500 | 1 | 1 |
| Center-only alternative,256 | 0 | 20 |
| Center-only alternative,500 | 3 | 20 |

Pointwise exact 95% power intervals for the alternative are[0,.1684] and
[.0321,.3789] for the mixture, and[.8316,1] for V2 at both sizes. Twenty
null trials give wide uncertainty and cannot certify calibration universally.
These are nominal-level power results for these exact laws, not adjusted
power for 720 hypotheses or evidence that Pedro's neural outputs realize
this alternative. The original absent-data disagreement remains partly open.

## 9. Defects, repairs and invalidated results

The native graph-domain guard was missing; directed, multigraph and loop
inputs were accepted despite the declared simple-undirected scope. A
21-case failing regression was saved before the guard repair and passes
after it. Supported historical graphs and algorithms are unchanged, so no
published numerical result is invalidated by this repair. Pedro's earlier
KW adapter repair is separately documented in his original protocol.

No new statistical implementation defect was identified in either tested
procedure. The older Hamming test's known power failures and V2's rare/
remote-higher-order blind spots are limitations, not evidence of an invalid
level guarantee. No old p-value, negative result or checkpoint was erased.

The subsequent native integration exposed a separate compatibility error:
the original V2 histogram allocator assumes max(degree)+1 colors. Native
global-K4 path outputs can contain color 3 at maximum degree 2, so the first
call failed before a p-value. The declared-alphabet adapter preserves actual
degrees, feature definitions, selection and permutations, leaving the frozen
source untouched. `DECLARED_ALPHABET_ADDENDUM.md` records the failure, tests
and inferential scope. It invalidates no completed historical local-list
result and is not represented as a newly confirmed power guarantee.

## 10. Discrepancy classification

| Issue | Class | Established conclusion / unresolved part |
|---|---|---|
| Labeled versus color-quotient law | A: definitions | Different questions; quotient agreement cannot establish labeled agreement |
| Joint terminal versus conditional-success law | A/B | Different with failures/cutoffs; must match the estimand |
| Degree/palette/graph/checkpoint cohorts | B: protocols | Historical rejection totals are not direct contradictions |
| Mixture versus V2 sensitivity | C: statistics | A genuine power gap is independently measured for the specified star alternative |
| Unsupported graph acceptance | D: implementation | Guard repaired; no supported historical result invalidated |
| Our high-degree neural/teacher differences | E: scientific | Survive matched-procedure checks on representative frozen banks; not attributed to Pedro's missing models |
| What V2 would find in Pedro's original banks | F: unresolved | Requires his exact banks/weights; no speculative cause assigned |

## 11. Conclusions admissible for the paper

Functional coloring success does not determine one unique distribution of
solutions. Some frozen learned policies have detected full-law differences,
and observable-event confidence bounds independently quantify differences
on specified graphs. Other cells remain nonrejections; some graphs have
symmetry-based equality, and Pedro reports limited finite-graph approximate
equivalence certificates. These statements can coexist.

The completed balanced extension provides a new, separately reproducible
example under two teachers. Small LUBY seed 73001 and small ColorTrials
seed 73000 each complete 640/640 functional attempts, but reject their own
teacher law on three graphs each (raw p.00005, Holm.0175 across 350 planned
comparisons). Of those 350,227 are testable and 123 have too few valid outcomes;
unavailable comparisons are not equality evidence. All three large
architectures have no new rejections in this low-degree cohort, which remains
a nonrejection result. This extension supports functional/distributional
separation without imputing Pedro's missing models.

The paper may claim an integrated comparison framework, supported
teacher-correctness arguments, separate notions of fidelity, independently
measured method sensitivity, and exact provenance. It must not claim that
Pedro's historical conclusions were disproved, that every successful network
differs from its teacher, that V2 dominates all tests, that rejection counts
rank TV distances, or that missing historical numerical reproduction has
been completed. This explicit boundary is fixed before drafting final
global-distribution conclusions.
