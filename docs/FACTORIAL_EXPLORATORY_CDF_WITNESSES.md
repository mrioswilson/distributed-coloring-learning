# Exploratory simultaneous CDF witnesses

This supplementary analysis was introduced after observing original main-bank
results and some original explanation failures. It is not preregistered
confirmation and changes no original omnibus p-value, sample count, original
held-out witness or frozen candidate-intervention analysis. No new GPU runs
are generated. All original attempts were already fixed at625 per bank.

The seven scalar functions in the originally frozen inference.py are U,
maximum color, U/K, largest class fraction, normalized entropy, normalized
mean color and nonedge color reuse. No function is learned from these data
or added because it separated models. Their frozen source hash is retained.

For all57 graphs,16 original generators and7 functions, form6384 empirical
CDFs from every proper completion among the625 original attempts. Let
alpha_CDF=.05/6384. Conditional on a bank's valid count N, DKW gives

    Pr(sup_t |F_hat(t)-F(t)| > epsilon_N) <= alpha_CDF,
    epsilon_N = sqrt(log(2/alpha_CDF)/(2N)).

Clip epsilon at1; with no valid completions give an uninformative interval.
Uniformity over t includes discrete ties and requires no independent vertex
colors or repeated complete vectors. A union bound covers all6384 true CDFs
simultaneously with probability at least .95. The shared teacher CDF is
counted once per graph/function; dependence between functions is allowed.

For any threshold t, subtract the two clipped CDF bands. The resulting
interval covers P_M(phi(X)<=t)-P_L(phi(X)<=t). Since the coverage event holds
for every threshold and every predefined function, it also holds after
choosing the metric/threshold or graph/checkpoint using these observations.
Maximizing empirical CDF discrepancy therefore permits a valid explanatory
event with probabilities, uncertainty and a full-law TV lower bound.

Report all855 comparisons and all7 metrics, including zero lower bounds.
These bounds belong to a new separate95% family; do not take maxima across
it and earlier CP families while retaining joint95% coverage. The same data
can support exploratory valid bands without becoming independent held-out
confirmation. The original reserved125-run analysis remains separately
reported, including its failures. These noninjective structural projections
cannot prove equality or an upper bound on full-vector TV.

One observed failure motivates this supplement. A selected categorical
triple had500 unique fitted keys,250 of them assigned positive scores. It
covered250/250 teacher selection rows versus0/250 model rows, then1/250
versus0/250 evaluation rows and0/125 versus1/125 reserved rows. That is
resubstitution overfitting in a large support. The independent holdout
correctly refused to validate it; a global rejection is not evidence that
this particular triple explains the difference. The omnibus itself selects
on independent data and remains calibrated. Fixed structural CDFs provide
an alternative explanation route with explicit uniform selection control.

The DKW argument is the same distribution-free inequality already used for
the preregistered structural W1 diagnostics; here the additional union bound
and uniform-threshold interpretation are made explicit. At N=625 its
per-CDF radius is about .0998. A zero resulting TV lower bound is a limitation
of the band or projection, not evidence of equal complete laws.
