# EVIDENCE_INDEX — sources for every major claim in the review

Repository: `github.com/mrioswilson/distributed-coloring-learning` (local clone `/srv/projects/AlgoLearning/distributed-coloring-learning`).

Commits:

- Baseline **B** = `c3affd16ac5654c6e2a700d2312710753adb9518`.
- Pedro **P** = `e8241184bcf81505838b74c7acd2834f22676d78` (paper + evidence) and `5993a5e4782d44670926224263688943a1411ebf` (handoff), on branch `codex/paper-revision-20261030`.

Repo paths are as of `5993a5e4` unless marked B.

Status legend:

- **IN-PKG**: copied into this package.
- **IN-GIT**: in git, not copied (size).
- **ABSENT**: not available on this machine.

## 1. Documents

| Item | Path | Status | SHA256 / note |
|---|---|---|---|
| Pedro's handoff | `docs/PAPER_REVISION_HANDOFF_MARTIN.md` → `pedro-original-docs/` | IN-PKG | `3a9103f5d1b3f84865e90a9992b0f7183990e479ed7d8ad72affeee0c879bb8c` |
| Claims matrix, audit 3, report, protocol, reproduce | `docs/PAPER_REVISION_{CLAIMS,AUDIT3,REPORT,PROTOCOL,REPRODUCE}.md` | IN-PKG | see `MANIFEST.sha256` |
| Revised manuscript PDF | `paper/ucgnn/main.pdf` → `manuscript-pedro/main.pdf` | IN-PKG | `196e2fccbd3401abebc42a995cf132b5c8a4a9406633172d97eb6623ddec5984`, 27 pp. |
| Revised LaTeX | `paper/ucgnn/main.tex`, `revision-*.tex`, 4 `.bib`, 7 `tables/closure-*.tex`, 3 figures → `manuscript-pedro/latex/` | IN-PKG | `main.tex` `bd303c9da5ff5068709b56a90f33a23c6d85807aa8d363122596d260a3032bb3` |
| Baseline manuscript | B `paper/ucgnn/main.{tex,pdf}` → `manuscript-baseline/` | IN-PKG | tex `5f197f7d97ec71beb351dc1b51f426a4e8fc8ceab794b40400a0ef8575538340`; pdf `3b05ee8d64e8d98de2bfaa79f2fbecf4d654d3cb2fe3409d14f7d2b252e8103f` |
| Unified diff of `main.tex` (B→P) | `manuscript-pedro/main.tex.diff-vs-c3affd16.patch` | IN-PKG | the 13 `revision-*.tex` are new files and appear only in `latex/` |

### PDF ↔ source check (this review)

- **Rebuild:** `latex/` rebuilt with the repo's pinned `.tools/tectonic`, which produced 28 pages because the engine differs from Pedro's pdfLaTeX.
- **Text comparison:** word multiset after ligature and hyphenation normalisation (`verification/pdf_text_normalize.py`). 95 of ~14,050 tokens differ, all line-break or URL wrapping artefacts.
- **Independent check (protocol subagent):** all 394 multi-digit numbers in the tex appear in the PDF. `paper-text.txt` equals the pypdfium2 extraction of the PDF. The build log has 0 undefined references.

## 2. Prospective 40-model trial

| Claim | Source | Status |
|---|---|---|
| Per-seed BA, successes, steps | `results/paper-revision-20261030/prospective/{be_base_d3,speculative_gm}/s105001…s105010/{native-test,cf-test,teacher-reference}.json` (60 files, ≈224 MB) | IN-GIT; recomputed by `verification/numbers/01_raw_seed_table.py` |
| Means, paired diffs, bootstrap CIs (+5.77 [−13.73, +25.26]; +50 [50, 50]) | `prospective-summary.json` (IN-PKG `evidence/pedro-results/`) | VERIFIED, `verification/numbers/02_bootstrap.{py,out}`, `03_boot_sens.*` |
| Gate 99.71 / 81.27; confusion strata 228/228, 814/995, 81 vs 42 silent FP | `continuation-audit.json` (IN-PKG) | VERIFIED, `04_pooled_strata.*`, `05_array_audit.*` (labels, keys and split recomputed from runtime arrays) |
| Native logical overlap 2.1–2.9 % (BE), 37.7–39.5 % (GM); CF 99.475–100 % | `continuation-audit.json` | VERIFIED; native figure not reported in the paper |
| 96.61 % of BE test queries ineligible | `continuation-audit.json` strata (38,644 / 40,000) | VERIFIED |
| Native COMMIT label = root participation (BE 45,440/45,440; GM 43,468/43,468) | runtime datasets in `paper-revision-runtime.tar.gz` | VERIFIED by protocol subagent (arrays) |
| 181 CF false negatives = 98 (105001) + 83 (105008), all-silent neighbours, κ = 0 types | runtime arrays + raw results | VERIFIED counts; type split inferred (unique given TP counts), `07_fn_types.*`, `10_kappa_boundary.*` |
| Label ranges by phase (GM native 0 negatives; CF 1,038–1,098) | runtime datasets | VERIFIED, `06_phase_ranges.*` |
| Budget 3,000 + 3,600 per model; identical init, minibatches and first stage | `training-records.json`, `weights-frozen.json` (IN-PKG) | VERIFIED |
| Chronology (protocol 09:54:44 → weights 10:11:58 → tests 10:12–10:25 UTC) | `protocol-frozen.json`, `weights-frozen.json`, tarball mtimes (`verification/protocol/tar-times.txt`) | VERIFIED as internally consistent; self-attested |
| Re-inference of all 40 models | `scripts/replay-revision-test.py`-equivalent run by protocol subagent | 0/40 confusion mismatches |
| Tests and verify | `tests/test_paperrevision.py`, `tests/test_artifacts.py`, `scripts/verify-paper-revision.py` | 7 passed; `verified: true` (identical to committed `verification.json`) |
| Seed 105001 / 105008 auxiliary-fit BA; 105010 participation witness | `be-failure-diagnostics.json` (IN-PKG) | RECEIPT |
| Runtime archive | `results/paper-revision-20261030/paper-revision-runtime.tar.gz` | IN-GIT, 26,328,083 B, `9a33d4bb8bd8b621fd35fd877e8c42f8afaa43f311c5ca8c4abda316551cec8a` (verified) |
| Delivery manifest | `manifest.json` (IN-PKG) | 235/235 entries verified |

## 3. Historical evidence (inherited from B)

| Claim | Source | Status |
|---|---|---|
| BE passive/matched/CF diagnostic, type table | `tables/closure-be-summary.tex`, `closure-be-seeds.tex` (byte-identical to B); `be-type-counts.json`, `be-reanalysis.json` | Tables unchanged; type table consistent (`08_historical_types.*`) |
| 21/21 native coverage | `tables/closure-native21.tex`; closure runtimes | Unchanged text; ABSENT raw |
| 810,000 transfer observations, 15 failed cells | `tables/closure-generalization.tex`; closure packet | Unchanged; raw rows ABSENT (Pedro also states this) |
| Johansson / GM witness interventions | `component-rescues.json`, `failure-mechanisms.json` (IN-PKG); historical witness frames | RECEIPT; generator checks witness hashes |
| KW worked example | `kw-example-witness.json` (IN-PKG) | Stage 2 + KW replayed (`verification/math/replay_kw_witness.py`); stage 1 needs IDs (ABSENT); text uses c = 1 vs stored active negatives c = 3, 0 |
| Pedro's 40 historical weights "available in this workspace" | `pedro-weight-availability.json` (paths under `/workspace/...`) | ABSENT on this machine |
| 225 recovered checkpoints | `recovery.json`, `public-archives.json` | RECEIPT; count not defined in paper |

## 4. Mathematics

| Object | B location | P location | Status |
|---|---|---|---|
| Appendix teacher rules, Availability, progress eq., bounds | B `main.tex`:608–800 | P `main.tex`:627–819 | byte-identical |
| Diffuse proposals, Diffuse submass, Counterexample, MAX COMMIT, MEAN/SUM | B :706–800 | P :725–819 | byte-identical |
| Symmetry and coupling | B :958–1000 | P :982–1024 | byte-identical |
| Observation indistinguishability | B :404 (body) | P :592 (Appendix C) | moved verbatim |
| Continuation oracle $O_T$, contract table, KW example | — | P `revision-contracts.tex`:1–69 | new (definitional); matches `src/dcl/multicf/oracles.py`:79–119 |
| Sufficient key $K$, hash split, acquisition | — | P `revision-prospective-methods.tex` | new; matches `src/dcl/paperrevision/keys.py`, `experiment.py`:120–156 |
| "What confirmation can and cannot guarantee" | B Discussion :557–575 | P `revision-theory.tex` | relocated prose; informal |

## 5. Our parallel local line (not Pedro's)

| Item | Location | Status |
|---|---|---|
| Commits `9497d39d`…`6853dd7f7e338821e074c145bb98dafe171b6e87` | branch `research/pedro-review-manuscript-revision` (local only) | Summarised, not audited |
| Uncommitted `paper/ucgnn/main.tex` (sha256 `5c142e4290056833997c06eed510055913281e4bd54e7f50547b2073c83a9bd3` at audit time) | working tree | Mid-edit; does not compile (per subagent) |
| Experiments A–F, literature review | `docs/pedro-revision/`, `results/pedro-revision/` | Numbers quoted in review §E.4 are as reported by that branch |

## 6. Data that are referenced but not included

- 60 raw prospective JSONs (≈224 MB) and the 26 MB runtime tarball: in git at `5993a5e4`.
- Closure raw rows (810,000) and closure runtimes: unpublished and not on this machine.
- `data/paper-revision-20261030/` (extracted runtime): reproducible from the tarball.
