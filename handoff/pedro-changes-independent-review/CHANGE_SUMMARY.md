# CHANGE_SUMMARY — Pedro's revision vs. baseline `c3affd16`

Compared versions

| Role | Ref | Commit | Manuscript | PDF SHA256 | Pages |
|---|---|---|---|---|---|
| Baseline ("our previous version") | `research/scientific-closure-20261026` | `c3affd16ac5654c6e2a700d2312710753adb9518` (Martín, 2026-10-28) | `paper/ucgnn/main.tex` (1,051 lines) | `3b05ee8d64e8d98de2bfaa79f2fbecf4d654d3cb2fe3409d14f7d2b252e8103f` | 18 |
| Pedro's revision | `origin/codex/paper-revision-20261030` | `e8241184bcf81505838b74c7acd2834f22676d78` (paper + evidence) + `5993a5e4782d44670926224263688943a1411ebf` (handoff), Pedro Montealegre, 2026-10-30 17:03/17:11 UTC | `paper/ucgnn/main.tex` (1,086 lines) + 13 new `paper/ucgnn/revision-*.tex` inputs | `196e2fccbd3401abebc42a995cf132b5c8a4a9406633172d97eb6623ddec5984` | 27 |

`paper/ucgnn/archive/pre-revision-c3affd16.tex/.pdf` in Pedro's branch are byte-identical to the baseline.

Scale of the commit: 157 files, ~17.2 M inserted lines, of which ~224 MB are 60 raw per-seed
JSON files under `results/paper-revision-20261030/prospective/`. Code: new package
`src/dcl/paperrevision/` (experiment.py, keys.py), 14 scripts, 2 test files; small edits to
`src/dcl/artifacts.py`, `src/dcl/localpalettes/config.py`, `src/dcl/sizedegree/common.py`,
`scripts/build-closure-paper.py`, `tests/test_globaldist_integration.py`.

Not Pedro's: local branch `research/pedro-review-manuscript-revision` (commits `9497d39d..6853dd7f`,
plus *uncommitted* edits to `paper/ucgnn/main.tex` and untracked `pedrorev-*` assets) is our own
parallel response to Pedro's objections. It was not pushed, is not an ancestor of Pedro's branch
and is not part of this comparison except where noted. Both lines edit `main.tex` from `c3affd16`.

## Section map (baseline → Pedro)

| Baseline section | Pedro's section | What happened |
|---|---|---|
| Title: *Learning Distributed Coloring Rules: Information, Counterfactual Fidelity, and Frozen Transfer* | *Learning Local Coloring Rules: Continuation Fidelity, Progress, and Failed Repair* | Thesis moves from a three-pillar study to an identification question about continuation behavior, progress and repair |
| Abstract (leads with "when does a learned policy execute an algorithm rather than solve its task"; 21/21 coverage; 22.64/9.27/99.99% continuation; 810,000 transfer runs) | Abstract leads with "can perfect agreement on observed actions conceal an incorrect local rule?"; adds the 40-model prospective trial (GM 50→100% BA; BE uncertain gain, fewer autonomous successes, eligibility gate 99.71% BA); coverage demoted to the last sentence | Rewritten |
| §1 Introduction (central question + 3 claims) | §1 Introduction opens with two concrete failure stories; 3 numbered contributions (contracts; diagnostic + prospective comparison; failure mechanisms ↔ progress hypotheses). Related-work paragraph unchanged | Rewritten opening/contributions |
| §2 Computational Model and Problem Definition | §2 Interfaces and Continuation Targets + **new** contracts block (`revision-contracts.tex`: eq. (1) oracle $O_T=E_v p_v F_v R_T$, Table of per-teacher contracts, KW worked example) | Renamed; text unchanged; contracts added |
| §3 Teachers and Neural Architectures | §3 Teachers and Learned Communication + **new** Figure 1 architecture and learned-vs-supplied paragraph (`revision-architecture.tex`) | Renamed; text unchanged; figure added |
| §4 Learning from Teachers | §4 Supervision and Experimental Comparisons | Renamed; text unchanged |
| §5 Functional Learnability (21/21 table, Clopper–Pearson, negative results) | **Moved to Appendix B** "Native Functional Coverage" (text preserved; "new study" → "pre-revision native-certification study") | Demoted |
| §6 Counterfactual Fidelity | §5 Native Agreement and Continuation Fidelity + **new** equal-type reweighting table (`revision-type-analysis.tex`) + **new** §5.x Prospective Equal-Budget Experiment (`revision-prospective.tex`, `-audit.tex`, `-mechanisms.tex`) | Expanded; core historical text unchanged |
| §7 Information and Identifiers (Prop. observation indistinguishability, JP table) | **Moved to Appendix C** (verbatim) | Demoted |
| §8 Generalization (with Figure: transfer grid `closure-generalization.pdf`) | §6 Frozen Transfer and Failure Mechanisms: provenance caveat added; **transfer grid figure removed**; **new** one-component interventions + Figure 3 (`revision-rescues.tex`); **new** subsection "What Confirmation Can and Cannot Guarantee" (`revision-theory.tex`, mostly relocated from old Discussion "Safety and progress") | Reframed around mechanisms |
| §9 Discussion (safety/progress; output laws & limitations; "we freeze scope", future work) | §7 Discussion and Limits (rewritten around the prospective audit; future-work paragraph removed; adds workspace-provenance statements) | Rewritten |
| §10 Conclusion | §8 Conclusion | Rewritten around agreement ≠ continuation; drops native coverage and transfer breadth |
| Appendices A–G | Appendix A **new** (prospective replicas, costs, acquisition key, post-test audit, BE failure diagnostics); B coverage; C IDs; D–J preserved with small edits (cost wording, reproduction block, rescue table) | Expanded |

## Figures

| Baseline | Pedro |
|---|---|
| Fig. 1 `closure-counterfactual.pdf` (BE BA per seed) | Fig. 1 `revision-architecture.pdf` (**new**) |
| Fig. 2 `closure-generalization.pdf` (transfer grid) | Fig. 2 `closure-counterfactual.pdf` (kept) |
| — | Fig. 3 `revision-failure-mechanisms.pdf` (**new**, Johansson 28/54 and GM 11/40 witnesses) |

Intermediate figures (paired prospective lines, two-failure bars, compact transfer grid) were produced and then
withdrawn by Pedro's third self-audit; files remain in `results/paper-revision-20261030/` and `paper/ucgnn/figures/`
but are not included. Note: `docs/PAPER_REVISION_REPORT.md:97` still says "La Figura 4 compacta se conserva";
the final `main.tex` includes no transfer figure (stale statement).

## New experiments and analyses (all in Pedro's commit `e8241184`)

1. **Prospective 40-model trial** (BE/E1 and speculative GM/E1, 19,396 params, 10 paired seeds × 2 arms
   [native vs CF auxiliary COMMIT target], 3,000 base + 3,600 auxiliary targets per model, 1,500 updates,
   20% SHA-bucket holdout of sufficient COMMIT types; final test 4,000 queries/seed/teacher + 1,000 autonomous runs/model).
2. **Post-test audits of that trial**: programmed eligibility/participation gate, equal-type accuracy,
   class/support counts, coarse-logic overlap (99.475–100%), confusion strata.
3. **Post-test BE failure diagnostics**: seeds 105001/105008 (fit vs held-out gap), 105010 (participation omission witness).
4. **Historical BE re-analysis**: equal-type accuracy and baselines on the 499,200 historical queries.
5. **One-component interventions** on two preserved historical transfer witnesses (Johansson/Transformer 83101, GM/Transformer 94102).
6. **Provenance work**: recovery of 225 closure checkpoints and 4,261 CF files by SHA256, re-verification of
   Pedro's 40 historical weights (local availability statement corrected).

## Unchanged

Teacher definitions and tables, training procedures text, historical numbers (BE 4,000-update diagnostic,
810,000 transfer total, 15 failed cells, 21/21 coverage), all appendix theorems/propositions (Availability lemma,
Diffuse proposals, Diffuse submass, Canonical finite-type MAX COMMIT, MEAN dilution/SUM construction, Symmetry and coupling,
Observation indistinguishability), related-work paragraph and bibliography files.
