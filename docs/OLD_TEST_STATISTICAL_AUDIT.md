# Statistical audit of the original complete-coloring test

Verified actual worktree `/workspaces/AlgoLearning/ucgnn-global-distribution-study`, branch research/ucgnn-global-distribution,
commit 1ba4546. All original files were read only. The original 795 results
remain the baseline; they are not confirmatory data for this extension.

## Statistic, randomness and calibration

The kernel was exp(-d_H/n), gamma=1 fixed prospectively. A complete labeled
vector was one observation. Each algorithm run had a distinct per-run PCG64
stream, reset dynamic coloring/cache state, and independently generated
private participation/proposal uniforms. Teacher banks were shared only
across comparisons on the same graph, which creates dependence between tests
but does not invalidate Holm. Neural uniforms were explicitly converted to
float32; teacher uniforms remained float64. The audit checks original sample
and protocol metadata rather than treating vertices as independent replicates.

Permutation calibration used the unbiased two-sample MMD squared U statistic,
including negative values without truncation. The nonnegative biased norm
was separately reported as an effect, never used to replace U in the test.
The fixed family was all 53 graph entries times all 15 policies: 795 tests,
including two duplicate graph aliases with fresh banks (51 unique graphs).
All adjusted values were recomputed exactly and match the stored Holm values.
There were 183 rejections, including exactly 10 E1 cases, all retained here.

B=19,999 uniform fixed-size label permutations; p=(1+#{T_perm >= T_obs-tol})/(B+1).
Inclusive ties with tolerance 1e-14 at the primary size were conservative.
The final reusable helper decreases tolerance above N=500; the historical
star supplement used constant 1e-14, which this audit reconstructs explicitly.
Minimum p=5e-05, smaller than .05/795=6.28930818e-05. Thus the resolution
was sufficient. Many p-values saturated at that minimum: finer ranking was
unavailable. No parametric tail extrapolation was used. The 795 stored
individual results have the expected independent samples, gamma and B.

Each coordinate kernel matrix is (1-a)I+a11^T with a=exp(-1/n), strictly
positive definite. Its finite product and restriction to proper colorings
remain strictly positive definite, hence characteristic. This proves
identifiability and asymptotic consistency, not finite-sample sensitivity.

## Exact reproduction and quantitative failure of the star control

On the 201-vertex star, leaves are iid uniform {0,1} under both laws.
The independent center is uniform {2,...,200} under A and {2,...,101}
under B. Both use valid local lists. TV=99/199. Let a=exp(-1/201), b=(1+a)/2.
The exact kernel effect is b^200(1-a)(1/100-1/199)=1.50213168648e-05.
The disjoint center supports and irrelevant leaves attenuate the signal.

Under A/A, the kernel row expectation is constant mu=b^200[a+(1-a)/199].
For the centered kernel, zeta2=E[k^2]-mu^2=0.000453932507842, with
E[k^2]=((1+a^2)/2)^200[a^2+(1-a^2)/199]. For equal sample sizes the exact
null variance of U is [4/(m(m-1))+4/m^2] zeta2. This provides the signal/noise
comparison below independently of Monte Carlo estimates.

| m | Rejections/20 | Exact null SD | Signal/null SD |
|---:|---:|---:|---:|
| 100 | 1/20 | 0.0006041358 | 0.0249 |
| 250 | 0/20 | 0.0002412883 | 0.0623 |
| 500 | 0/20 | 0.0001205836 | 0.1246 |
| 1000 | 0/20 | 6.027668e-05 | 0.2492 |
| 2000 | 4/20 | 3.013457e-05 | 0.4985 |

All 100 historical p-values were reproduced exactly from archived vectors
and named permutation seeds. Full observed statistics, first-trial null
distributions, quantiles and binomial intervals are archived. The control
used B=199, nominal alpha=.05, without external Holm. Thus its failure cannot
be blamed on the 795-test correction. Leaves contribute kernel variability
although their laws coincide; Hamming normalization and high center support
jointly make the mean effect small. Increasing bandwidth globally may also
suppress similarity between noisy leaf vectors; it does not isolate the hub.
For analogous growing stars the effect is O(D^-2), while null SD is
O(1/(m sqrt(D))). Characteristicness supplies no useful dimension-free TV
lower bound. A normal signal/SD planning approximation is recorded as a
diagnostic only, never used to calculate p-values or claim guaranteed power.

## E1 inventory and limits

old-e1-all-265.csv retains graph/family/actual degree/size/checkpoint, U,
raw and adjusted p, graph and checkpoint hashes, structural projections,
and available independent m=250 power (only seed 41000 on selected graphs).
Separate ten-rejection and 255-nonrejection files preserve both outcomes.
Missing power is missing, not zero. No nonrejection is interpreted as equality.
The previous conservative MMD intervals establish no TV bound or approximate
equivalence. The initial old resource forecast lacked a per-cell archived
table; realized costs and the original budget ledger remain available.
The permutation and Holm implementations show no identified inflation in
this audit. Their finite controls do not certify numerical behavior for all
inputs; the observed principal methodological defect is power for localized
alternatives, not insufficient primary permutation resolution.
