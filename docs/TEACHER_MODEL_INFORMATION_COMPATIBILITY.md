# Teacher–model information compatibility

Audited implementation: `src/dcl/multiteacher/programs.py`, preserved unchanged
from the scientific integration at `a54a18e3`. New adapters live exclusively in
`src/dcl/samplecomplexity/`. This audit concerns the implemented variants, not
all algorithms with the same bibliographic names.

All native programs accept finite simple undirected graphs, maximum degree
three, a common palette {0,1,2,3}, contiguous vertex labels, and at most 65,536
vertices. Labels used by the simulator are not automatically observations of
the policy. Persistent colors begin pending. Two complete rooted message
neighborhoods implement two learned graph exchanges; palette attention is a
separate within-node operation.

| Teacher | Anonymous | IDs | Clock/phases | External order | Auxiliary information | Required adaptation |
|---|---|---|---|---|---|---|
| LUBY | Yes (A) | No | Repeated color trials; no numerical round input | None | Two independent private float32 draws; own pending state, neighbor fixed colors and offers | Native K4; learn participation, proposal and confirmation |
| JOHANSSON | Yes (A) | No | Repeated color trials | None | Private candidate draw; participation target always true on pending nodes | Same anonymous interface; no imported triangle-free/high-degree theorem |
| COLOR_TRIALS | Yes (A) | No | Repeated preference subroutine | None | Four private discrete preferences in {0,1,2,3}; known global degree bound 3 | First exchange learns availability; second broadcasts preferences/learned availability; learn smallest winning color and confirmation |
| JP | No (B) | Distinct fixed IDs | Synchronous priority rounds | ID-induced partial order | Pending neighbors and relative priority | Explicit sign(ID_neighbor − ID_self) on every edge; first-fit, eligibility and confirmation learned |
| BE_BASE_D3 | No (B) | Distinct integers in [0,65535] initialize auxiliary colors | Two Linial phases then thirteen KW reductions | No sequential schedule | Field, bound, target, persistent auxiliary colors; local polynomial evaluations and per-edge polynomial/option equality | Supply finite-field arithmetic and phase interpreter; learn point selection and KW recolor/color decisions; sign(aux−target) supplied |
| SPECULATIVE_GM | No (B) | Distinct IDs break same-slot conflicts | ASSIGN, DETECT, REPAIR | Four balanced blocks, synchronized positions, ascending-ID marked repair queue | Write permission, equal-slot relation, persistent marks/colors | Supply block/repair scheduler; explicit edge order; learn first-fit, conflict marking and repair decisions |
| GREEDY | Only conditional on supplied schedule (B) | IDs construct random permutation | One scheduled write per native step | Ascending-ID sequential permutation | Own writable flag and current neighbors' colors | Supply writable-node scheduler; policy does not need numeric IDs once schedule is supplied; color choice remains learned |

## Actual information and computation

**LUBY.** Trial one exchanges pending/fixed-color observations and computes a
learned proposal from the private tape; trial two exchanges participation and
proposal messages for learned confirmation. Permanent colors and local pending
state persist; no global clock or unique ID is read by the canonical core.
Exact confirmation makes simultaneous equal colors impossible. The declared
ideal progress bound is at least 1/4 per pending vertex per trial.

**JOHANSSON.** The implemented always-active random-color variant has the same
two exchanges, persistent state and exact safety induction. A conservative
degree-three progress bound is 1/8. It is not a claim about the full scope of
every Johansson coloring theorem.

**COLOR_TRIALS.** Every available color receives an independent discrete
preference. A color wins only by strict comparison with competing pending
neighbors for which that color is available; ties wait. Availability must reach
neighbors before the second exchange. Permanent colors persist. The probability
of own preference 3 and all at most three competitors at most 2 is at least
27/256. This is the repeated subroutine, not RankingTrials. There are no IDs
or algebraic helpers in its decisions.

**JP.** A pending vertex is eligible precisely when no pending neighbor has a
larger ID. It selects the first free color. One exchange suffices for that
mathematical rule; the common neural decomposition retains two learned
exchanges and redundant participation/confirmation heads. Eligible nodes form
an independent set, and a global remaining maximum progresses every round.
Thus the exact uncapped program completes in at most n rounds. The sample
study overrides the historical cap using the constructor's existing
`max_rounds` argument, preserving the teacher's priority rule.

**BE_BASE_D3.** IDs are also initial proper auxiliary colors. Local integer
polynomials are evaluated over F13 and then F7. A learned point selects its own
polynomial value through a declared deterministic decoder; the decoder does
not search for an unblocked point. Edge equality bits compare the values at
each point. The subsequent KW bound sequence 49→40→32→26→21→17→14→12→10→8→7→6→5→4
uses one auxiliary-color exchange per mathematical reduction. Each high old
color owns a block of four candidates; raw neighbor equality against these
options is supplied, with no OR over neighbors or availability decision.
The interpreter supplies the clock, target/bound update and block decoding.
The point, recolor decision and color choice remain learned. Auxiliary state
must remain proper and below the new bound at every step. This degree-three
base case does not learn finite-field arithmetic or high-degree recursion.

**SPECULATIVE_GM.** The interpreter constructs blocks and serial repair order;
this global preprocessing is part of the evaluated system. Assignment needs
fixed colors; detection needs color equality, same slot and relative ID; repair
needs current colors. Detection marks are predicted, not computed for the
student by an exact conflict detector. The repair queue is constructed from
those predicted marks. Temporary conflicts during speculation are legal.
Unmarked/marked history and the phase persist. Exact detection covers every
conflicting edge, and sequential exact first-fit repair yields a proper final
coloring in at most 2 ceil(n/4)+n steps. Learned mistakes are not externally
repaired. Block construction/phase barriers are not free anonymous LOCAL rounds.

**GREEDY.** The external scheduler grants a single permission each step in the
ID-derived permutation. Fixed colors persist; one neighbor-color exchange
suffices for first-fit and the common model retains two. Four colors suffice
at degree three. Exact termination takes n writes. Supplying a writable flag
removes the need for the network to infer the global permutation; therefore
this is a scheduled extended-interface experiment.

Full conditional safety/progress arguments and original verified references
remain in `docs/integration/TEACHER_CORRECTNESS_AUDIT.md`. Neither those proofs
nor successful training prove a learned checkpoint correct universally.

## What is supplied to each architecture

E1 NORMALIZED uses shared node/color MLPs, MAX graph aggregation and normalized
ordered palette prefixes. DeepSets replaces the palette scorer with shared
color features and mean palette context; graph exchanges still use MAX.
LocalPaletteTransformer replaces that palette scorer with two-head attention
within each node's palette; its graph communication remains learned messages.
All three receive identical observations, native interpreters and losses.

The original canonical E1 was anonymous. However, the **previous integrated
70-model native cohort already had numeric ID differences for JP/GM, native
phase/schedule context, and BE polynomial/equality helpers** through RootedE1.
Those particular historical failures cannot accurately be described as
training an entirely ID-blind JP or algebra-blind BE. The present sign adapter
changes numerical representation/conditioning, and the present data/loss and
optimization protocol also change. Its role is not evidence of a previously
missing information bit in that integrated cohort.

For JP and GM, `relational_arrays` maps ID channels so the inherited message
code reads exactly −1/0/+1 comparisons. The transformation preserves every
center-to-neighbor comparison in the first exchange and the root's neighbor
comparisons in the second. No eligibility flag, winning flag, exact
availability, conflict mark or teacher action is supplied at inference.
Raw magnitudes alone in the historical model required resolving differences
as small as 1/65536. The new interface removes that dependence. It changes
information representation and is not a controlled architecture-only contrast
against the old study.

For BE, a general increasing ID transformation does **not** preserve polynomial
values or the exact coloring. Valid IDs must stay in the finite declared space.
The final tapes generate new valid ID assignments; JP/GM additionally have
exact invariance regression tests under increasing transforms within that
space. No transformation theorem is claimed for BE.

The small control uses hidden/message widths 8/4. The no-communication control
erases received state, relation and edge-mask channels, retaining local state
and external permission. It is intentionally information-deficient where
neighbor observations are necessary. Pedro's archived direct-action GNN has a
different objective and information representation, and no comparable
query-budgeted fresh cohort is asserted for it.

## Sufficiency arguments and indistinguishability

These are statements about interfaces, not universal fixed-width neural
representability. In a bounded native rooted state the exact teacher action
can be computed from the observations just listed. Availability is an OR of
neighbor color equalities; JP eligibility is an OR of pending/larger relations;
GM marking is an OR of equal-color/equal-slot/larger relations. ColorTrials
first obtains competitor availability and then compares preferences. BE point
feasibility is an OR of polynomial equality at each point, while KW uses
option equality and the local target comparison. The two exchanges and native
decoders therefore expose sufficient mathematical information. ReLU networks
can implement finite Boolean tables with sufficient capacity; this argument
does not establish that every specified hidden width realizes the whole rule,
or that optimization finds it. The finite precision random-rank sampler also
requires its prescribed boundary convention.

An anonymous two-node edge with both vertices pending, identical private
observations and four colors has identical rooted anonymous observations at
its endpoints. Swap IDs (1,2)→(2,1). JP's required eligible endpoint swaps, while
those observations do not. No deterministic function of that insufficient
interface can agree with both teachers on both assignments. The regression
test constructs this pair. Similarly, same-color/same-slot GM endpoints with
swapped priorities require opposite detection marks. Without BE auxiliary
colors/phase, the same permanent pending-color state can require entirely
different point or KW actions. Without greedy permission/order, symmetric
pending vertices cannot identify which single vertex the schedule selected.

Even after information is sufficient, failure may reflect optimization,
coverage, finite capacity or autonomous distribution shift. The fixed-label
optimization and interactive controls address parts of this ambiguity; none
isolates a cause universally.

Two concrete distinctions prevent overextending relational-ID arguments:

* On a two-node BE edge in the first F13 phase, IDs/auxiliary colors (1,2)
  are different constant polynomials, so point 0 is valid. With (1,14), the
  polynomials are 1 and 1+x; point 0 is blocked and the first valid point is 1.
  Both assignments have the same strict ID order and valid unique bounded
  IDs. Order comparisons alone therefore cannot determine BE's point action.
  The regression test checks these actual teacher outputs.
* For ColorTrials in the reset-state interface, consider a path v–u–w with
  v,u pending, preferences at v (2,0,0,0), and at u (3,3,3,3). Changing w's
  fixed color from 0 to 1 leaves v's one-hop pending/color/preference view
  unchanged but changes whether u competes for color 0. The required choice
  at v changes from 0 to 1. Availability from u's neighborhood must reach v;
  the two graph exchanges supply it. This is an information-radius argument,
  not a universal lower bound for protocols with different cached state.

The interpreter supplies pending-node gating, scheduled write permissions,
palette alphabets, phase/bound transitions, KW block decoding and termination
checks. BE's auxiliary-target **comparison** is coded by the sign adapter;
learning its arithmetic is not claimed. Learned heads still control whether
to write and which point/color to choose. Checking a collision or invariant
to record a failure never repairs or filters the predicted choice.
