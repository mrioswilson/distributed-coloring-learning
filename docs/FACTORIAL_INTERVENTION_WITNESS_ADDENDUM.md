# Separate exploratory witness-based intervention bounds

This addendum is written after inspecting examples of original-policy held-out
witnesses, and before any candidate-intervention bank exists. It is an
exploratory supplementary analysis, not part of the original confirmation.
It changes no action, checkpoint, original test, graph, sample count, event
selection rule or intervention worker. Its entire code is committed before
the new paired intervention data. All sixty previously selected priority
cells (fifteen checkpoints, three new graphs and the historical graph) remain
in the family regardless of the eventual result.

The original frozen pipeline selects exactly one event per graph/checkpoint
using its first 500 original-policy samples and evaluates that fixed event
on the separate 125-attempt holdout. Use that existing descriptor without
altering its threshold or tuple set. Do not search for an event using the
intervention data, or choose another event after seeing its effect. The
original held-out examples motivated this supplement because some large
DeepSets differences concern mean color, rather than palette occupancy.

Revision 2 adds direct evaluation of this same fixed event under the
intervened policy, still before any paired intervention observation. The
first version is preserved at commit e7291b9fd0b0641647671880a2ce4e5359a85472;
its 300-probability analysis is superseded and will not be reported as the
final supplementary family. No intervention samples, actions or event
selection rules have changed.

For each of the sixty cells bound six Bernoulli probabilities: the original
teacher and model holdout event probabilities, the same event under the
intervened policy, the new paired mismatch probability, and each member of
the pair's proper-completion probability. Use two-sided exact CP intervals
with alpha=.05/(60*6). A union bound gives simultaneous coverage at least .95
for this separate family of 360
probabilities. Conditional on the original main data, selected events are
fixed and their original holdout measurements remain independent executions.
The mismatch/success measurements come from independent new pairs; within
pair dependence is allowed. The intervention sample size is selected only
from an independent timing pilot, as in its existing preregistration.

Let l be the resulting TV lower bound between original M and Luby from the
held-out event. If d bounds the coarsened mismatch probability above and s
bounds both proper-completion probabilities below, the coupling argument
gives u=min(1,2d/s) as an upper bound for conditional TV(I,Luby). Therefore

    TV(M,I) >= max(0,l-u).

The direct difference between the original-model and intervened event
probabilities also supplies a TV(M,I) lower bound. Evaluate the original
frozen descriptor on the new intervened complete vectors, using every proper
completion of its fixed bank. No new event is fitted. These observations are
independent of the original model's reserved holdout. Take the maximum of
the direct and triangle lower bounds, since the same simultaneous family
covers both. Direct evaluation can show a proposal-channel effect even if
the particular coupling is too loose to prove closeness to Luby.

Here all laws condition on proper completion by round128. Report all sixty
cells, including zero or uninformative lower bounds. No independent
two-sample test is applied to the paired I/Luby outputs. A large u does not
prove residual discrepancy because the particular coupling can be loose.

This .95 family is separate from the original intervention event-effect
family and all original-policy test families. Do not claim joint .95 coverage
over their union or take the maximum of their differently allocated bounds
while retaining .95. A positive supplementary bound supports an effect of
the specified computational proposal-channel replacement for the fixed policy
and graph. It does not establish complete mediation, population generality,
or separate degree from local palette size. No extra simulations are added.
