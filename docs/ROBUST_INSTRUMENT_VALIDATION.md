# Independent instrument validation, round 1

The omnibus implementation is selected from the development benchmark before
validation data are consulted. The existing development grid gives strong
sensitivity for localized, correlation and parity controls. It cannot justify
claims about unseen neural alternatives. No confirmatory model comparison
has been generated.

Use the exact candidates in stats.py/classifier.py committed with this file.
The benchmark runs all candidates at every m in {100,250,500,1000,2000}:
the m=2000 timing pilot cost approximately 13 CPU seconds per trial for all
candidates together, versus .5 seconds at m=250. Four CPU workers fit the
memory and CPU budget. No neural GPU time is charged for CPU statistics.

Fresh namespaces validation and sensitivity are disjoint from pilot and
development. Five synthetic null laws, 400 repeats each at m=250; ten
alternative classes, 30 repeats at all five sizes. Add 400 Luby/Luby and 400
same-checkpoint trials for each architecture's seed-41000 checkpoint on path3,
using newly generated null banks (200,000 runs per generator, never teacher
versus model comparisons). Uniform clique nulls simulate the known law;
the theorem establishes ideal equality, and a failed rejection is not its
proof. The frozen-network controls are actual executions, not draws from a
fitted empirical law.

Gate: the prospective targets in ROBUST_TEST_DEVELOPMENT_PLAN.md must pass;
the omnibus must have Wilson lower bound >=.80 for localized m=250,
four biased leaves m=500, pair dependence/higher-order parity/diffuse .55
m=1000. Null calibration acceptance uses a one-sided binomial rejection
threshold .01 divided by every null/method cell, with all results retained.
This conservative multiple-control audit detects gross inflation and does
not prove exact nominal calibration; the conditional-permutation theorem
supplies the level guarantee. Baseline and omnibus trial outcomes are paired
on the same data; report paired improvements with intervals.

Stress grid: D=20,1000 for the single-center difference; leaf marginal bias
.6,.7; diffuse leaf bias .51,.53; rare-event probability .01, at m=250,1000,
30 trials each. Original rare .001 trials remain limitation controls. These
stress outcomes do not change the acceptance targets or method.

Repeated simulations have binomial uncertainty. An 80% target applies only to
this named finite collection and specified m, not arbitrary TV=.5 laws,
unknown interactions, graph populations, or externally Holm-adjusted power.
Any failed gate blocks new confirmation, requiring a separately documented
redesign and new validation namespace.
