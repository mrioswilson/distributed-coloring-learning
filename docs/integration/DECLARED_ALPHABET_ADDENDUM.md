# Declared-alphabet compatibility repair

The first integration call of the historical V2 on a native global-K4 path
failed before producing a p-value: its structural histogram allocated
max(degree)+1=3 bins, while valid native samples contain color3. The saved
trace is `results/integration/statistics/global-alphabet-before.txt`.
All training and raw native banks are unchanged by this repair.

The historical instrument was written for local d(v)+1 lists, where that
allocation is valid. This is an integration-domain incompatibility, not a
discovered error in its published local-list analyses. Falsifying the graph
degree to force four bins would corrupt other features and is not done.

`src/dcl/integration/native_statistics.py` loads the frozen V2 source in an
isolated namespace and replaces only the structural histogram allocator with
an explicit declared alphabet. Used-color count, largest class, entropy,
mean color, co-color and actual-degree features retain their definitions.
Selection, full-vector kernel, maximum aggregation and permutations reuse
the original functions. No shared Python module or frozen source file is
modified. For an already matching alphabet it calls the original feature
function exactly; a regression reproduces the statistic and p-value.

An independent point-mass null using color3 on path3 verifies the previously
unsupported case. Exchangeability gives the same conditional permutation
level proof for these deterministic label-blind feature definitions. This
does not borrow a finite-power guarantee from the old local-list validation.
The native extension is named explicitly in the new results and paper.

No native path p-value existed before the repair. Previously cached cubic
graph tests use identical features and seeds and remain unchanged. The
original matched-bank and known-law controls continue to call the original
V2 directly. No new data are described as independent confirmation of an
outcome-tuned feature search; this repair follows a declared alphabet mismatch
and changes neither the model nor the target distribution.
