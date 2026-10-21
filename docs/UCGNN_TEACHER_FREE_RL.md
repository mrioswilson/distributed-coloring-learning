# Teacher-free distributed coloring: design for a separate study

Status: research specification, **no RL battery launched**. This document is not
a claim that teacher-free learning already works. It does not alter the
supervised paper protocol. A future preregistration must freeze its seeds,
budgets, validation and test matrix before training.

## Three distinct questions

| Regime | Initial policy | Training signal | Scientific question |
|---|---|---|---|
| RL from scratch | random shared weights | environmental rewards only | Can coordination and safety emerge without an algorithm teacher? |
| Supervision | random shared weights | teacher participation/candidate plus exact proposal COMMIT labels | Which explicit algorithmic mechanisms transfer? |
| Supervision then RL | a specified supervised checkpoint | rewards after supervised stage | Can fidelity be traded for faster or more robust valid coloring? |

Report both total compute and environment transitions; a pretrained policy
has already consumed labeled data and compute. Do not describe fine-tuning as
teacher-free discovery. Keep the same executable local interface in all arms.

## Environment and observations

Finite connected simple graphs, public K=Delta+1, synchronous rounds and two
neighbor communications. A node observes its own permanent color (or pending)
and fresh independent local randomness; learned shared messages convey only
the same color/participation/candidate information used by UCGNN. Parameters
do not depend on K; a token per color permits variable palettes. Inference
uses no availability oracle, exact safety shield, global state, IDs or learned
embeddings capped at training K. Public-palette padding may be masked; occupied
colors may not be removed by an external exact rule.

For training, graph generation and the environment's transition/reward can
inspect edges and colors. This is ordinary simulator access, not a policy
observation. The simulator never supplies Luby labels, available-color masks,
expert candidate ranks or supervised safety targets. Log all simulator access
to keep that boundary auditable.

## Actions and transition

First communication: each pending node samples participation B_v from a
Bernoulli and candidate C_v from a categorical distribution over its public
palette. Second communication sends the sampled discrete proposals, then each
participant samples authorization J_v from a second Bernoulli. Simultaneously
fix colors at all v with B_v=J_v=1 and a candidate. There is no rejection by an
exact rule. Existing permanent colors remain permanent.

The environment checks newly created monochromatic edges AFTER the transition.
In the simplest initial experiment any such violation ends the episode with
an explicit failure penalty, rather than undoing unsafe actions. This differs
from a shield, which would reject them before execution. Safe but unfinished
episodes stop at a prespecified round cap and are right-censored for timing.
If recoloring is later allowed it changes the problem and needs its own arm.

## Rewards, credit and waiting policies

One possible global round reward, with all coefficients frozen in advance:

    R_t = (# newly fixed nodes)/N
          - lambda * (# new monochromatic edges)/max(1,|E|)
          - eta * (# pending nodes)/N - xi/N,

plus a success bonus for a complete proper coloring and a fixed failure
penalty on unsafe termination. The first term cannot by itself distinguish
safe from unsafe progress; the explicit failure cost is essential. A positive
pending/time penalty makes waiting forever strictly worse than a policy that
safely progresses. Too large a penalty can instead encourage unsafe rushing.
Freeze a small coefficient grid and pilot selection on training-degree
validation, never on the degree-transfer test. Publish nonlearning pilots.

An alternative is a constrained objective: minimize expected rounds subject
to an upper bound on unsafe-episode probability, with a primal-dual multiplier.
Constraint satisfaction during optimization is not a deployment safety proof.
Local neighborhood reward components can reduce variance, but credit may
still conflict: a node's aggressive proposal affects its neighbors. Report
global success even if training optimizes local rewards.

## Optimizer and critic

Start with clipped PPO and generalized advantage estimation. The sampled joint
policy is a product across nodes conditional on the state and, within a round,
the second-stage policy is conditional on the sampled proposal messages:

    pi(a|s) = product_v pi_B(B_v|o_v)
                       pi_C(C_v|o_v)
                       pi_J(J_v|o_v,neighbor proposals).

Ignore masked colored-node actions and inactive COMMIT terms. Store the actual
sampled actions, masks, old log-probabilities, rewards, values and terminations;
reevaluate the SAME sampled proposals when computing second-stage ratios.
Do not recompute own hard argmax proposals during PPO replay. Sum node log
probabilities only with care: the joint importance ratio becomes unstable as
N grows. A shared per-node clipped objective with graph-level advantages is a
pragmatic multi-agent PPO variant, and its bias relative to joint PPO must be
documented. Compare node-mean and graph-mean weighting.

A centralized training-only graph critic with permutation-invariant pooling
may observe the full state and graph. It never enters the actor at inference.
Alternatively use a local two-hop critic to test whether centralized training
information matters. These critics have separate parameter/compute budgets.
PPO is motivated by stochastic mixed discrete actions and on-policy credit;
REINFORCE with a baseline is a simpler negative/control implementation. Value
decomposition or constrained actor-critic are later alternatives, not silent
substitutions after observing test failures.

Keep participation, candidate and COMMIT entropies separate. A joint entropy
bonus can hide a collapsed candidate policy behind random authorization.
Use private draws at every node; shared randomness changes the collision law.

## Initial future experiment and metrics

Proposed initial distribution: regular Delta=3, N=40, K=4; training/validation
graphs independent, random tapes held out. Freeze independent training seeds
and a fixed environment-step/optimizer budget before starting. First verify
learning and waiting-policy baselines. Only then run five independent networks
and a fresh degree/size test, with the same pairing/statistical distinctions
as the supervised study. Do not reuse the paper's observed test to tune RL.

Primary endpoint: complete proper coloring by the round cap. Also report
unsafe-episode probability, unsafe COMMIT per state, forbidden proposals,
participation rate, valid proposals, effective candidate colors, collision
rate, rounds, pending fraction, return, advantage variance, action entropy,
environment transitions, optimizer updates, wall/GPU time and memory. Compare
an always-wait policy, independent random proposals with learned authorization,
the supervised reference and fine-tuned reference. Exact random-trial coloring
is a separately labeled algorithm baseline, not a policy shield.

## Required environment tests before any large run

1. Simultaneous equal adjacent authorizations create a failure, with no hidden
   filtering. Fixed-neighbor conflicts are detected after transition.
2. Correct partial colors persist, masks and categorical log-probabilities are
   correct for mixed K, and private random tapes replay exactly.
3. Reward for always waiting is strictly below safe completion under the chosen
   finite-horizon objective; check unsafe rushing and sparse-reward traps.
4. PPO replay uses stored hard proposal messages; ratios equal one before an
   optimizer step. Colored/inactive actions cannot influence the objective.
5. Actor inference is invariant to the presence/absence of the critic and to
   hidden simulator oracle utilities. No Luby label appears in saved batches.
6. Resume restores optimizer, simulator, RNG and rollout buffer exactly.

These tests are requirements for the later project. No teacher-free
performance result is claimed by this specification.

Method reference: Schulman et al., *Proximal Policy Optimization Algorithms*,
arXiv:1707.06347 (metadata to be verified in the future RL preregistration).
