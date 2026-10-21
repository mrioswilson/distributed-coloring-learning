# Prospective addendum A1 (2026-10-21)

The original protocol at 023deaf and its SHA256 remain unchanged. This addendum
must be committed before examining any held-out candidate state or confirmatory
graph outcome. At registration all six full-budget pilots had passed the
original promotion gate; five-seed training was running. The confirmatory
graph directory and held-out calibration outputs did not yet exist.

## Execution and statistical clarifications

Controlled E1-0 candidate calibration and held-out prediction may run explicitly
on CPU in float32 while the single GPU worker trains the fixed replicas.
Calibration and held-out probes use the same CPU implementation. This is not
a resource-failure fallback, not a new training/selection policy, and not a
CUDA confirmation cohort. Record device, dtype and source/checkpoint hashes.
The later mechanism job reuses these durable measurements. CUDA graph rollouts
remain the confirmatory graph experiment. A device consistency check on the
fixed all-free K=4,101,201 probes can be reported separately, without refitting
the calibration or pooling CPU/GPU observations as independent tests.

Mechanistic jobs are split by fixed checkpoint, sequentially, with the original
1,800-second per-job limit. All original probe states, seeds, round limits and
promotion criteria remain unchanged. Epoch-120 confirmatory evaluations precede
the optional validation-best tier. Neither change depends on a test outcome.

Complete bipartite stress graphs when Delta=N/2 are deterministic up to node
permutation. Their labelings/tapes are not additional independent structural
families, just as clique tapes are not independent graph structures. Report
them as structural stress tests, without a random-structure binomial interval.

## Two additional frozen, exploratory quantile predictions

The primary HQ calibration (257 midpoint rank observations at all-free K=101,
monotone accumulation, linear interpolation) and its <=2-rank/0.001-forbidden
criterion remain unchanged. In particular a negative HQ result is retained.

Separately freeze a weight-derived limiting-quantile description BEFORE the
held-out states. For a free slot with fixed learned state h and availability
alpha, write the trained two-layer scorer as

    s(x,r,w) = sum_j v_j ReLU(B_j + A_j x + R_j r + C_j w),

where x=L, w=U-L, A=W_L+W_U, R=W_r, C=W_U, and B absorbs h, alpha and bias.
For fixed r,w, a continuous maximum on [0,1-w] occurs at an endpoint or ReLU
breakpoint. Define g(r) as the smallest maximizing x at w=0, evaluated in
float64 from frozen coefficients. This is an explanatory numerical prediction,
not a new inference policy, fitted teacher or certified candidate theorem.

Predict floor(k*g(r)), clipped to [0,k-1], mapped into sorted available colors.
The first predictor freezes h/alpha from the all-free K=101 star. The second
uses a finite catalog of four free-slot contexts, all calibrated at K=101:

| Context (pending neighbors, fixed neighbors) | Calibration state |
|---|---|
| yes, no | all-free degree-100 star |
| yes, yes | degree-100 star with lowest 51 colors available |
| no, yes | degree-100 star with color 0 alone available |
| no, no | isolated pending node with public K=101 |

The context lookup uses only presence of pending/fixed neighbors; the h/alpha
values and coefficient catalog are never recomputed from held-out choices.
The architectural justification is that every free slot in a MAX neighborhood
sees only the pending-neighbor and fixed-other-color types. Nonzero occupied
availability, finite interval width, numerical rounding and discrete maximization
can still invalidate these approximations. Monotonicity is measured, not
enforced for the analytic predictors.

Evaluate both predictors on exactly the original HQ held-out states and uniform
midpoint draws, reporting fidelity and rank errors separately from primary HQ.
There is no new success threshold, tuning or model-selection rule. No predictor
is refitted after viewing K=17,63,251,997 or their held-out availability patterns.
