# Matched statistical crosschecks

The source implementations are preserved: Pedro's mixture of Hamming kernels
at scales 1,4,16 times terminal-status equality, and the audited V2 omnibus
with independent witness selection/evaluation and inclusive permutations.
The signed-form mixture statistic equals the ordinary unbiased two-sample
MMD U statistic, including possible negative estimates. Both use whole
executions as observations. Neither a nonrejection nor a small plug-in TV
establishes equality.

Before new crosscheck outcomes, fix these analyses:

1. Apply both implementations to the identical 500 successful-prefix rows
   from the archived first regular graph at N128/degree3 and N512/degree100,
   for checkpoint s41000 of each of the three architectures. Keep all six
   comparisons, including nonrejections. Replay V2 with its original seed
   and stopping convention and verify its statistic/p-value exactly. Use
   19,999 permutations for the mixture too. Report method-specific Holm
   families of six separately from the untouched original family of720.
   These are exploratory reanalyses of already observed historical banks,
   and use our checkpoint cohort. They do not reproduce Pedro's absent
   checkpoint cohort or his across-graph stratified test.
2. Draw twenty genuinely independent experiments per cell of the known
   local-list star null and center-only alternative, at m256 and m500.
   Compare both methods on each identical independent dataset. Use999
   permutations and nominal .05. Report all rates with pointwise exact
   binomial intervals. The alternative has TV99/199 by construction.
   These trials estimate finite-law nominal power, not strict Holm-adjusted
   power or calibration over every distribution. No method adaptation or
   selection follows these trials.
3. Apply unchanged V2 to all350 planned new teacher/model/graph comparisons,
   each against its own exact teacher. Use19,999 permutations and one Holm
   family. Condition on proper native completion and retain failure banks.
   Record insufficient valid samples as untestable (p1 only for planned-
   family bookkeeping). Do not replace unsuccessful attempts. All source
   dependencies of the historical V2 remain frozen byte for byte.

The first two analyses isolate procedure differences on matched rows. Only
access to Pedro's exact weights and raw banks can establish which procedure
would change the decisions of his original reported experiments. The new
balanced study is separately sourced and cannot silently fill that gap.

Implementation addendum, 23 October 2026: native global K=4 on paths exposed
the original local-list histogram allocator. The explicitly scoped repair
is documented in `DECLARED_ALPHABET_ADDENDUM.md`; the frozen original source
remains unchanged. The matched archival and known-law crosschecks above
continue to call that original implementation directly.

Executed matched-bank command (from the integration repository):

```bash
python -m dcl.integration.statistics_audit matched --archive-root /workspaces/AlgoLearning/ucgnn-size-degree-study
```

For another checkout, restore the historical factorial archive at a separate
root containing its `data/` and exported `results/` hierarchy, and supply that
root instead. It must contain the exact bank hashes recorded in the output.
