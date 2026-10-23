# Prospective supplement, before conditional-law testing

21 October 2026. Main protocol f516530 is unchanged. Graph evaluation has begun;
no new conditional-law test results have been opened. This supplement does not
change its hypotheses, primary checkpoints, graph generation or metrics.

1. To prevent conflating the five new NORMALIZED networks (41000–41004) with
   the actual five historical E1 networks (0–4), apply the **same** conditional-law
   grid, sample sizes and B3 splits to historical E1 as a separately labeled
   secondary cohort. Freeze its epoch-120 paths/hashes from E1_INDEX.json.
   No pooling as ten independent NORMALIZED networks and no new graph-training
   comparison are authorized by this supplement.
2. Secondary historical autonomous states: take the first two lexically sorted
   saved failure NPZs for the previous SUM-s41000-epoch120 evaluation that contain
   a proper partial coloring with a pending node and nonempty available set.
   Select the lowest-index pending vertex. Preserve the whole state and exact
   original case metadata, checkpoint and graph provenance. These are **global**
   historical lists, not claimed to be local-list reachable. Evaluate policies
   on the same frozen states without refitting. These analyses are exploratory.
3. The graph-overlap audit finds deterministic controls recurring from historical
   evaluation (four cliques and the degree-three wheel, itself a clique). They
   are retained, reported as examined controls, and excluded from any claim of
   virgin random-graph testing. The independent seed namespaces have no overlap.
   Different specifications may encode the exact same graph (e.g. star and
   complete bipartite K1,3). Bootstrap clustering uses the **actual graph
   fingerprint**, carrying its different tapes/specifications and all networks,
   rather than pretending those are independent structures. Report counts both
   with the full prescribed matrix and for fresh random families separately.

The extra five law tasks start only after the 155-task main queue and share its
12-hour resource ceiling. Pending/failed cells remain explicit. No extra training.

Before starting conditional-law tests, the conservative calibration quantile is
made precise as the order statistic ceil((N_cal+1)×.95), with strict rejection.
For N_cal=1024 this is order 974 (one-based). A generic 95th empirical percentile
would not necessarily give finite-sample level ≤.05. Exchangeability of the
independent calibration/test null scores gives unconditional FPR at most .05
(ties can only make it smaller). This does not assert conditional FPR ≤.05 for
every realized calibration set; the independent test FPR and intervals remain
reported. No model probability estimates, sample sizes or hypotheses change.
