# Prospective independent-power supplement

Declared during the ongoing frozen primary sampling queue, before analyzing
any power bank or independent-power data. It does not change the original
795 primary hypotheses, correction, kernel, graph list, primary n or decisions.
Its findings are supplementary and must not be presented as the original
confirmatory primary study. Reason: empirical-bank resampling can overstate
population power, especially on enormous supports with little bank overlap.

Use the already specified path3 only (small support allows affordable truly
independent repetitions), Luby and seed-41000 of each of the three architectures.
For each generator draw a new 77,000-run bank, stage `independent-power`.
Partition it sequentially in increasing m in [100,250,500,1000,2000]; for each
m assign twenty disjoint successive blocks of m rows. Total is
20*(100+250+500+1000+2000)=77,000. No run enters two blocks or any other stage.
At each matching generator/teacher block perform the same gamma=1 MMD test,
B=199 permutations, nominal alpha=0.05. Use a new `independent-power-perm`
seed namespace indexed by checkpoint, m and trial. Do not apply the original
Holm threshold to these power trials. Report twenty-trial binomial Wilson
intervals for the rejection probability at each m. These intervals quantify
independent repeated-experiment uncertainty, not only resampling conditional
on a single empirical law. Teacher blocks may be shared across models; each
model's repeated experiments remain independent across trials. Across-model
power estimates are dependent and cannot be combined as independent evidence.

This measures population power for the specified finite-precision generators
on path3, not for high-degree graphs, other checkpoints or all architectures.
Twenty repeats give broad intervals; no universal required m is claimed.
Retain failures and use conditional successes as in the principal protocol;
report actual counts. Charge all process time against the same 12-hour budget.
No additional primary rejection claim is introduced, no trained weights change,
and no prior power curve is consulted to select graph, checkpoint or m.
