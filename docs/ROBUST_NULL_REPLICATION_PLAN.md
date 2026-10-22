# Independent post-freeze null replication

The final V2 gate passed its prespecified 104-cell calibration screen.
The noisy-star null had 32/400 omnibus rejections and a pointwise Wilson
interval above .05. This is a reason for a larger independent diagnostic;
it is neither hidden nor treated as proof of inflation or of correct level.

This plan is committed before these new diagnostic draws. No confirmatory
neural p-values have been used for this decision. The primary procedure,
protocol, graph cohort, sample sizes and all confirmatory inference remain
frozen at 46a8a4010ad8629b5133431634622e533a0498ef. No retuning is permitted.

- One law: the existing star_null at D=200, n=201.
- 4,000 independent trials, total m=250 per generator.
- Stage namespace null-replication2, with the existing distinct A/B and
  permutation seed derivations. No reuse of earlier vectors.
- Use the unchanged benchmark.trial pipeline, V2 routing and B=199.
  All candidates receive the same arrays; the omnibus is the diagnostic
  target and old Hamming MMD is the fixed reference.
- Report every candidate's count, Wilson interval and one-sided binomial
  exceedance probability under rejection probability .05.
- Flag the omnibus for further implementation investigation if that
  one-sided probability is below .01. This is a diagnostic, not a replacement
  for the original acceptance gate or an adjustment of neural inference.
- Save every array, seed, p-value and hash using the existing benchmark
  artifact format, so the final read-only audit replays these draws too.
- CPU only, four workers; no GPU use or additional neural runs.

The interval estimates the rejection probability for this specific null
law and finite permutation procedure. It does not certify other laws.
Choosing this law from the previous diagnostic does not reuse its samples:
the new trials and their uncertainty are independent of that selection.
