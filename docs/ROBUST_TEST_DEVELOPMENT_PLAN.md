# Prospective instrument-development plan

Historical baseline: worktree verified using git worktree list, branch
research/ucgnn-global-distribution, commit 1ba4546. Its data remain read only.
This study is a separate worktree and has a fresh 12 GPU-process-hour cap.
No neural training, remote push, or changes to historical results are allowed.

DEVELOPMENT selects methods using synthetic proper local-list colorings and
historical exploratory data. VALIDATION uses different named RNG namespaces.
CONFIRMATORY data do not exist yet and are not generated until the instrument
passes its gate and a separate final protocol is committed.

Candidates: original unbiased Hamming MMD (gamma 1); jointly permuted maxima
and log-mean-exponential of four global scales (.25,1,4,16); honestly split
local, interaction, structural and global-kernel witness tests; a joint
permutation omnibus retaining gamma-1 full-vector characteristic MMD on its
evaluation half; categorical likelihood-ratio permutation tests for small
supports; independently trained simple classifiers. These are explicitly
defined adaptations, not unmodified MMDAgg or MMD-FUSE implementations.

Each total m is per generator, including the selection half; old tests use
all m and new split methods use floor(m/2) for selection and the remainder
for testing. Thus efficiency comparisons charge selection observations.
Selection includes categorical events and ordered thresholds at individual
vertices, fixed graph-based/random pairs/triples, structural counts and
parity, and dense marginal witnesses. Regions never use evaluation labels.
All selected components are jointly calibrated, not tested independently
at .05. No color-name or vertex-label canonicalization is used.

Grid m=100,250,500,1000,2000. Pilot timings precede the complete benchmark.
Initial development: ten independent trials per alternative at m=100,250.
Independent validation: 400 null trials per control at m=250 and 30 trials
per alternative and grid point (larger-m quadratic candidate comparisons
may use a prospectively cost-limited subgrid documented before validation).
Every trial uses B=199 permutations, plus-one inclusive-tail p-values.
Report Wilson 95% intervals, paired improvements and compute costs.
Null controls include point, small uniform/nonuniform, noisy-star equality,
exact Luby/Luby, same frozen checkpoint, and clique symmetry.

Power targets fixed before validation: omnibus lower 95% Wilson bound >=.80
for localized-star TV=99/199 at m=250, four biased leaves at m=500,
pair dependence and three-leaf parity at m=1000, and diffuse .55 binary leaf
bias at m=1000. Compare against old MMD on the same generated arrays.
Rare-event probability .001 is a limitation control, not an 80% target.
Null acceptance uses one-sided binomial exceedance p >= .01/(number of
null-control/method cells); a finite-sample permutation proof, rather than
passing a calibration simulation, supplies the level guarantee.
Global .55 bias at m=1000 must not lose more than .15 observed rejection
probability relative to old MMD. Record uncertainty rather than interpreting
this descriptive margin as a population noninferiority theorem.

Failure of a gate halts confirmation. Any redesign requires a new named
validation round, preserving earlier results. Frozen confirmation will
specify graphs, all fifteen checkpoint hashes, split sizes, B compatible with
external Holm, futility rule, success conditioning and independent witnesses.
