# Protocol `pedrorev-cf-v1`: budget-controlled continuation supervision and held-out intervention families

Status: written and committed **before** any model of this protocol was trained
and before the confirmatory battery existed. Configuration:
`configs/pedrorev-cf.json`. Code: `src/dcl/pedrorev/`,
`scripts/pedrorev-cf-experiment.py`. Runtime artefacts (banks, checkpoints,
logits) live in `data/pedrorev-cf/` (not versioned); summaries in
`results/pedro-revision/cf-experiment/`. Local registration via Git commit;
this is not an external preregistration.

## 1. Problems addressed

Coauthor review, objection 3 (interpretation of the counterfactual result):

1. The existing BE contrast (passive 60.14%, matched 53.25%, CF 99.9965%
   seed-mean balanced accuracy) compares procedures that differ in total
   acquired labels (3,000 vs 6,776). It cannot separate *what* was labelled
   from *how many* labels were acquired.
2. The diagnostic states belong to a repeatedly used development split, and
   the four query colors and intervention generator are shared between training
   and diagnosis. No intervention family was held out.
3. It is unclear how much of the continuation behaviour is explained by simple
   predicates supplied by the interface.

## 2. Hypotheses and pre-declared success criteria

Unit of replication: training seed (five paired seeds 95101–95105). Queries are
correlated within a seed and are never treated as independent replicates.
Criteria are stated per seed; "all five seeds" is the pre-declared criterion.
Exact two-sided sign tests (minimum p = 0.0625 for 5/5) are descriptive only.

**Experiment A (label budget).** For each teacher in {BE, GM, ColorTrials}
and E1 (normalized) architecture, six arms at 4,000 updates:
passive/matched/CF at a low total `L_low` (BE, GM 3,000; CT 30,000) and at a
high total `L_high` (= native 3,000/30,000 plus that bank's continuation
labels; 6,104–6,776 for BE/GM, 70,000 for CT). Passive/matched controls at
`L_high` receive **additional native labels** (new teacher trajectories),
never continuation labels; CF at `L_low` receives fewer native labels plus
uniformly subsampled continuation labels so that its total equals `L_low`.

- A1. At equal total labels (`L_high`), CF has higher continuation balanced
  accuracy (BA) than both passive and matched in all five seeds, for BE and GM.
- A2. CF at `L_low` has higher BA than passive and matched at `L_high` (more
  than twice CF's labels) in all five seeds, for BE and GM.
- A3 (mechanism, predicted from native underdetermination): doubling native
  labels does not make passive/matched reject eligible-active occupied BE
  queries: their per-seed TNR on that stratum stays below 0.5 at both budgets.
- A4 (ColorTrials contrast, predicted): because native ColorTrials data contain
  non-winning negatives, the matched–CF BA gap for CT is smaller than for BE
  and GM in every seed. Reported descriptively.
- A5 (functional, descriptive): native functional success on a fresh
  1,000-run battery is reported for every arm; no superiority hypothesis.

**Experiment B (held-out intervention families; BE).** Arms at `L_high`
native bank: `cf-noforce` (all forced-participation-of-ineligible-root
queries removed from training, replaced by the root's native query) and
`cf-lateocc` (all eligible, participating, occupied-option queries at late KW
steps, bound ≤ 12, removed).

- B1. CF arms (`cf-high`, `cf-low`) reach BA ≥ 0.99 on query occurrences whose
  sufficient COMMIT type never occurred in that arm's training inputs, in every
  seed in which that subset contains both classes.
- B2. `cf-lateocc` rejects the held-out late-KW occupied queries with per-seed
  TNR ≥ 0.95 (genuine generalization to a new combination of an intervention
  type and phase values).
- B3. `cf-noforce` does **not** reject forced-ineligible queries (per-seed TNR
  on that family < 0.5) while keeping eligible-active BA ≥ 0.95: an
  off-support clause of the contract is not recovered from the other clauses.
  (Predicted negative result supporting semantic underdetermination.)
- B4. Nuisance family (received neighbor messages randomized; never
  intervened in any training bank; contract labels provably unchanged for BE,
  GM, CT): CF arms keep BA ≥ 0.99 per seed for BE and GM.

**Experiment C (semantic controls).** On the same battery, rule baselines
that read privileged decoded predicates: always-reject; accept-iff-P (root
participates); accept-iff-P∧E (KW eligibility); accept-iff-P∧F (queried option
free); and the oracle P∧E∧F. They are not trained competitors and use
predicates the network receives only implicitly.

## 3. Design details

- Training: `dcl.multicf.training.train` unchanged; Adam 1e-3, batch 64,
  warmup 100 (CF), checkpoints 1,500 and 4,000; **no selection**: 4,000 is the
  primary milestone, 1,500 secondary. Same initialization per seed in all arms.
- Training graphs: sizes 8/12/16/24, nine families, namespace
  `20261030-pedrorev-cf-v1` (disjoint from all earlier namespaces).
- Confirmatory continuation battery: namespace
  `20261030-pedrorev-cf-v1-confirmatory-battery`, 90 exact-teacher trajectories,
  sizes 32/48/64, ten families, shared by all seeds and arms. Every permitted
  root of every color phase is queried with the four standard continuation
  variants and four nuisance variants (fresh RNG).
- Functional battery: namespace `...-functional-battery`, 1,000 runs, sizes
  32/48/64, exact teachers validated first.
- Freeze: all checkpoint hashes are written to `freeze.json` before the
  battery is generated (the script refuses otherwise).

## 4. Metrics

Per seed, arm, family and stratum: accuracy, BA, TPR, TNR, precision,
confusion counts. BE strata: forced-ineligible (¬E∧P), silent (¬P),
eligible-active free (E∧P∧F), eligible-active occupied (E∧P∧¬F). Per sufficient
type (BE): each distinct type has one label and (at fixed weights) one
decision; type-level BA weights every distinct type equally; seen/unseen is
defined relative to (i) the arm's own training inputs and (ii) the shared bank.
Sufficiency of the type key is checked empirically (zero within-type decision
disagreement).

## 5. What would change the conclusions

If A1/A2 fail, the existing BE contrast is attributed to label budget and the
manuscript must say so. If B1/B2/B4 fail, continuation fidelity is reported as
not generalizing beyond trained types/combinations. If B3 fails (the clause is
recovered), the claim of semantic underdetermination is withdrawn. All outcomes
are reported.

## 6. Addendum (registered before Experiment D is run; after training, before any battery)

**Experiment D (continuation fidelity under shift; GM and ColorTrials).**
The question: does continuation supervision change *how* frozen policies fail
when their proposal/participation modules deviate under transfer? Existing
evidence is one replayed witness per failed cell (GM/Transformer: CF rejects a
bad repair proposal and leaves a conflict, matched/passive write a forbidden
color); it cannot separate training procedure from seed or budget.

Design: all 4,000-update GM and ColorTrials checkpoints of the six budget arms
(5 seeds, 60 checkpoints) are run, frozen and without selection, on four fresh
transfer conditions mirroring the earlier confirmatory n=64 cells (global
Δ=3, 4, 10; local Δ=4; 1,000 runs each; iid-uniform ten-family mixture;
namespace `20261030-pedrorev-transfer-v1`; one battery shared by every arm).
Exact teachers must pass every case first. At every color-phase step the
declared contract is evaluated on the network's own messages (measurement
only; never applied). Script `scripts/pedrorev-transfer-experiment.py`,
evaluator `src/dcl/pedrorev/transfer.py`.

- D1 (primary, GM): at equal total labels (`*-high`), CF has fewer unsafe
  accepted writes (writes the contract rejects) than matched and than passive,
  summed over the three shifted conditions (Δ=4, Δ=10, local Δ=4), in every
  seed in which the comparator has at least one unsafe accept.
- D2: in shifted conditions, CF arms contain at least 99% of their own bad
  proposals (contained / (contained + unsafe accepts)), per seed, GM and CT.
- D3 (descriptive): failures per arm and condition, split into safety
  (≥1 unsafe accept) and progress failures; CT is reported with the same
  measures without a directional hypothesis (its native data already contain
  non-winning negatives).
Interpretation limits fixed in advance: fewer unsafe accepts is not higher
functional success; a contained bad proposal can still cost progress (e.g. GM
repairs are not retried), which is exactly what D3 measures.

## 7. Addendum (registered before Experiment E is run)

**Experiment E (exploratory decision-level probe; ColorTrials scale).** In the
earlier posthoc probe all fifteen primary CF ColorTrials policies reject a
strict winner when the global degree reaches 10, with root degree and local list
fixed. In that probe Δ, the normalized preferences (Δ/Δ, (Δ−1)/Δ) and the
number of masked palette slots change together. Script
`scripts/pedrorev-ct-scale-probe.py` separates them on the same constructed
state: panel G fixes Δ=10 and varies (winner, competitor); panel D fixes the
normalized preferences at (1, 2/3) and (1, 1/3) and varies Δ ∈ {3,6,12,30,60}.
Applied to the 15 primary CF checkpoints and the 30 new pedrorev ColorTrials
checkpoints.

- E1: acceptance is (nearly) invariant in Δ at fixed normalized preferences
  (panel D), i.e. the masked-slot count is not the driver.
- E2: at Δ=10, acceptance fails for normalized gaps below the native minimum
  gap 1/3 and holds at gaps ≥ 1/3 (panel G).
This is decision-level evidence on one constructed state, not a rollout
failure rate and not a causal account of all ColorTrials transfer failures.

## 8. Follow-up registered after the Experiment A/B results (Experiment F)

**Observation that motivates it (confirmatory battery v1).** In five BE CF
runs (`cf-high`, `cf-lateocc`, `cf-noforce` and `cf-low` of seed 95105;
`cf-low` of seed 95103) the COMMIT rejects safe continuations in which a
neighbor also recolors in its own disjoint KW block. Posthoc
(`scripts/pedrorev-be-neighbor-activity.py`): this happens exactly in the
runs whose training bank contains no positive example with a participating,
non-occupying neighbor; the BE intervention generator never perturbs
neighbor messages.

**Experiment F.** Arm `cf-neighbor` (seeds 95101–95105) uses the `cf-high`
bank with identical roots, root queries and labels (6,776), but the neighbor
messages of every stored continuation query are perturbed (labels provably
unchanged for BE; checked). Training as in §3. Evaluation on a **new**
battery (namespace `20261030-pedrorev-cf-v2-confirmatory-battery`, 90
trajectories, sizes 32/48/64), together with the frozen `cf-high`,
`matched-high` and `passive-high` checkpoints, plus a fresh 1,000-run
functional battery. Config `configs/pedrorev-cf-followup.json`.

- F1: `cf-neighbor` accepts standard-family positive queries with a
  participating non-occupying neighbor (TPR ≥ 0.99) in all five seeds, while
  keeping TNR ≥ 0.99 on forced-ineligible and eligible-active-occupied queries.
- F2 (replication): on the new battery, `cf-high` seed 95105 again rejects
  most such positives (TPR < 0.5), and the other `cf-high` seeds do not.
This is a follow-up chosen after seeing battery v1; it is confirmatory only
with respect to battery v2, which did not exist when F was registered.

## 9. Outcome register (written after all batteries; nothing above was edited)

Sources: `results/pedro-revision/cf-experiment/analysis.json`,
`functional/`, `native-agreement.json`, `be-neighbor-activity.json`,
`results/pedro-revision/cf-followup/`, `results/pedro-revision/transfer-experiment/summary.json`,
`results/pedro-revision/ct-scale-probe.json`,
`results/pedro-revision/failed-cell-classification.json`.

| Hypothesis | Outcome |
|---|---|
| A1 equal labels, CF > passive and matched, all seeds (BE, GM) | **Supported** (BE CF-high 93.81–100% vs ≤71.66%; GM 100% vs 50%) |
| A2 CF at L_low > native-only at L_high, all seeds (BE, GM) | **Supported** (BE CF-low 94.61–100% vs ≤71.66%; GM 100% vs 50%) |
| A3 native-only TNR on eligible-active occupied < 0.5 at both budgets | **Supported** (≤0.59% in every BE run; 0% GM) |
| A4 CT gap smaller than BE/GM in every seed | **Supported** (CT native-only 99.34–99.39% vs CF 100%) |
| A5 functional (descriptive) | GM, CT: all 60 arms 1,000/1,000. BE: 34/40 arms 1,000/1,000; failures in seed 95105 (cf-high, cf-lateocc, cf-noforce 298; cf-low 133; matched-low 394; passive-low 985) and cf-low 95102 (989), 95103 (148) |
| B1 CF unseen-type BA ≥ 0.99 in every seed | **Not supported**: holds in 2/5 seeds for cf-high (52.3–100% type-level); unseen negative types robust, unseen positive types not |
| B2 cf-lateocc rejects held-out late occupied queries, TNR ≥ 0.95 | **Supported** (≥96.33%; 100% in four seeds) |
| B3 cf-noforce does not recover forced-ineligible clause; keeps eligible-active BA ≥ 0.95 | First part **supported** (TNR ≤10.57%); second part holds in 4/5 seeds (95105: 93.8%) |
| B4 nuisance family BA ≥ 0.99 (BE, GM) | **Not supported**: BE cf-high 4/5 seeds; GM CF 97.1–98.7% (errors only on execution-irrelevant silent queries) |
| D1 fewer unsafe accepts for CF under shift (GM) | **Untestable/null**: no arm of GM or CT produced any unsafe accept or failure in 240 cells × 1,000 runs; E1 GM proposals never deviated |
| D2 CF contains ≥99% of own bad proposals | Supported for CT (100%); vacuous for GM (no bad proposals) |
| D3 descriptive | all 240,000 runs succeed; CT completeness misses at Δ=10 in every arm (8,550–10,759 per 1,000 runs) |
| E1 Δ-invariance at fixed normalized preferences | **Supported** (all 45 checkpoints accept at Δ=3…60) |
| E2 gap threshold at Δ=10 | **Supported in the modified form**: reject at gap 0.1 (all 45), accept at gap ≥0.3 (all 45), gap 0.2 accepted by most; threshold below the native minimum gap 1/3 |
| F1 cf-neighbor TPR ≥ 0.99 on active-neighbor positives, all seeds | **Supported** (1.0 in all seeds, battery v2; 1,000/1,000 functional) |
| F2 cf-high 95105 defect replicates on battery v2 | **Supported** (TPR 0.0; 301/1,000 functional) |

Additional posthoc analysis (not pre-registered): every failed run of the 15
historical failed transfer cells re-executed and classified; all success
counts reproduce. GM/Transformer 94102: the CF, matched and passive
checkpoints share every Exchange-1 tensor; the same 181 runs fail in all
three, containing 274 bad scheduled repair proposals; CF rejects all 274
(181 progress failures, 0 unsafe writes); matched and passive accept all 274
(181 safety failures). Johansson/Transformer 83101: all 28 failures are
cutoffs on cliques ending with every pending vertex in a colliding proposal
group and every confirmation rejecting. All 11 ColorTrials control cells fail
only through unsafe accepted writes (collisions).
