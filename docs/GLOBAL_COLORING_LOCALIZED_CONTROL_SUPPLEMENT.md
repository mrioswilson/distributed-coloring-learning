# Prospective supplementary localized positive control

This diagnostic is added after primary sampling began, motivated by the
analytic attenuation calculation in theory section F. It is not part of
the original confirmation or a basis for changing its kernel. It uses no
neural weights or primary observations. Keep all positive and negative
outcomes, including low power despite a large known TV.

On star-d200 (center vertex 0), draw each leaf independently uniform {0,1}.
Generator A draws center uniform {2,...,200}; B uniform {2,...,101}, all
independent. Both produce proper local-list colorings, with known TV=99/199
and exact MMD squared given in theory section F. For each m in
100,250,500,1000,2000 draw twenty independent pairs, use gamma=1 and B=199
permutations, nominal alpha=.05. Named seed namespace `localized-control`
contains m, trial and generator; permutation namespace differs. Save full
arrays and seeds, and report binomial Wilson intervals. These are independent
population-power trials for this analytic alternative, not empirical-bank
resampling. All tests are CPU only. No multiplicity-adjusted discovery
claim is drawn from these detection-rate diagnostics.
