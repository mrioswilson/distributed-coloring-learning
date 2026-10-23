# Balanced native teacher / palette study

This prospective short-budget extension fills the seven-teacher by E1,
DeepSets and LocalPaletteTransformer interface, and adds two controls. It is
not a rerun of Pedro's unavailable 35 final checkpoints or a budget-matched
comparison against his 17,760-update study.

The exact plan is `configs/integration-balanced.json`: seven supplied native
interpreters, five architectures, two independent training seeds, 24 cubic
N40 training graphs with one teacher trajectory each, eight validation graphs,
Adam at .001, batch 128 root decisions. The same graph/tape pools and sampled
root positions are used across architectures within each teacher/seed. Every
seed is a new initialization; different rollout tapes are not training replicas.
The small network has hidden/message dimensions 8/4. The no-communication
control erases all received neighbor states, relations and masks; supplied
native phase/schedule permissions and own observations remain available.

Before generating scientific pools, a separate cycle8 timing pilot measures
each teacher/architecture. Its loss and accuracy are not used for selection.
Choose the first equal update budget in [500,250,100] whose twofold training
cost estimate plus a 600-second evaluation reserve fits 3,600 CPU wall seconds.
Ten percent of updates use teacher COMMIT proposals. All remaining steps use
the declared native own/counterfactual labels. The selected update count is
fixed before training. The final update is the sole checkpoint selection rule.
This small design tests implementability and finite-budget sufficiency; an
architecture that fails here is not proved incapable of learning the teacher.

DeepSets replaces the ordered-prefix scorer with a shared per-color function
and mean palette context. Transformer uses local palette self-attention. Both
retain learned availability and COMMIT, two neighbor communications, and the
same teacher-specific supplied context as the E1 adaptation. BE's variable
Linial fields use their own valid-slot masks; polynomial evaluation after
predicting a point remains supplied arithmetic. ColorTrials still predicts
its candidate after the second exchange. The native interpreters continue
to apply model actions without an oracle safety filter.

The five frozen evaluation graphs are a new regular graph at N16, an irregular
graph at N40, a regular graph at N160, path16 and K4. Exclude training and
validation labeled fingerprints. All lists are global K=4, maximum degree
three. Higher-degree or local-list support is not silently added to these
native teachers. The historical local-palette study supplies that separate
axis of evidence.

For each graph/teacher, draw one independent 128-run exact reference bank and
one independent bank per checkpoint. Node states within an execution are
dependent. Record complete vectors, operational status, native steps and
all seeds. Distinguish improper completion, pending-at-termination and valid
completion. A pending native sequential schedule and a parallel round cutoff
are not identical causes; original interpreter failure labels must also be
retained. Native steps are compared only within a teacher.

Freeze all final checkpoint hashes before generating evaluation banks. Global
law tests use the unchanged audited V2, 19,999 inclusive permutations, and Holm
over every planned model/graph comparison (350 if all 70 models finish). The
estimand conditions on proper completion. If either side has fewer than four
valid rows, record the test as unavailable with conservative p=1 for the
planned-family adjustment; do not interpret it as equality. Failures and their
frequencies remain separate. No resampling to obtain a desired success count,
post-hoc model promotion, favorable-cell deletion, or extra training is allowed.

The new results are a balanced finite cohort with two initialization replicas,
not a population ranking of teachers/architectures. Validation local action,
participation, candidate and native COMMIT accuracy, phase counts, GM marks,
functional failures, runtime and output-law comparisons are separate metrics.
No-communication and small controls inform necessity only within their precise
observation/model families. No universal architectural minimality is claimed.

Exact checkpoint-level reproduction of Pedro remains separately incomplete
until his original weights and raw banks are accessible. New successful
training cannot substitute for those missing artifacts.

Implementation addendum, 23 October 2026: the first native path call exposed
the historical V2 allocator's local-list assumption before any path p-value
was produced. `DECLARED_ALPHABET_ADDENDUM.md` records the explicit K=4 adapter,
saved failure, regression checks and unchanged frozen historical source.
This documented change qualifies the prospective phrase “unchanged V2” above;
the new native finite-power behavior is not inherited from old validation.
