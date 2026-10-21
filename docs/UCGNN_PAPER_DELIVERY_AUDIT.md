# Evidence for the research deliverables

The fixed-protocol battery is complete. This audit distinguishes delivery of
the requested finite study and initial paper from universal algorithmic claims
and future experiments. The executable coverage/provenance audit is
`python -m dcl.paperstudy.audit`; its output is
`results/ucgnn-paper-study/completion-audit.json`.

| Deliverable | Authoritative evidence | Outcome / scope |
|---|---|---|
| 1. Preregistered protocol | `docs/UCGNN_PAPER_PROTOCOL.md`, commit `023deaf`, hash checked by audit; prospective amendment `8c69168`; promotion records | Fixed before training/final cohort; original document unchanged |
| 2. Modular ablations and tests | `src/dcl/paperstudy/models.py`, `tests/test_paperstudy.py`, `causal-isolation-audit.csv` | Five normalization pairs match all tensors/data/tapes/updates; original UCGNN unchanged |
| 3. LocalPaletteTransformer | `models.py`, variable-palette and gradient/interface tests, training/resource CSVs | 16,995 parameters, within-node attention, no original prefixes or inference oracle |
| 4. Pilots and replications | 32 runtime run directories, `training.csv`, `final-promotions.json`, checkpoint hashes | Six full-budget pilots promoted; five seeds for each of six primary treatments; two auxiliary pilots |
| 5. Independent extrapolation | `final-confirmatory-cohort.json`, `final-historical-exclusions.json`, `cases.csv`, `cells.csv`, completion audit | 18,363 cases, 85 views, no missing/technical cells; compatible specialist subset included |
| 6. Selection/collision/progress/COMMIT | candidate CSVs/NPZ, progress CSVs/traces, commit certificates/multiplicities, per-round graph traces | 120 held-out E1 states, all-free laws for 39 checkpoints, patterned/progress battery for 11 fixed representatives; full histograms on local grids |
| 7. Mathematics | `docs/UCGNN_MATHEMATICAL_ANALYSIS.md`, manuscript propositions/corollary, interval certificates | Complete conditional local-safety and progress proofs; no claimed universal learned termination |
| 8. Alternative teacher | `teacher.py`, law/safety tests, fresh seed 41005, `cyclic-degree-summary.csv`, `cyclic-candidate-diagnostics.csv` | 304/304 correct graph outcomes; high-degree teacher fidelity declines; single-seed exploration |
| 9. Teacher-free RL specification | `docs/UCGNN_TEACHER_FREE_RL.md` | From-scratch RL and supervised/RL comparisons specified; no large RL run |
| 10. Initial LaTeX manuscript/figures | `paper/ucgnn/main.tex`, `references.bib`, generated tables/figures, `main.pdf`, finalization record | English draft with completed positive/negative results and verified bibliography; compiler exit 0 |
| 11. Results and limitations | `results/ucgnn-paper-study/FINAL_REPORT.md`, generated `REPORT.md`, raw CSVs | Confirms successful attention transfer; retains normalization/aggregation negatives and finite-data limitations |
| 12. Query/resume/reproduce commands | `FINAL_REPORT.md`, `paper/ucgnn/README.md`, `experiments/ucgnn-paper-study` | Exact current-state audit and fresh-worktree reproduction commands; no host-restart promise |

The full suite passed 259 tests; `tests-final.txt` records the output. The
machine audit additionally checks the actual checkpoint files, every paired
case identity/tape/fingerprint, exclusion of every current train/validation
manifest (including the later cyclic pilot), per-model coverage and numeric
certificate enclosures. These checks establish the stated audit scope; they
do not replace an independent implementation review.

The original historical source snapshot contains 57 files; all 57 hashes
remain unchanged. New training dependency hashes are stored in checkpoints.
Analysis/reporting extensions were made after observing results, without
changing architectures, training, checkpoint selection or confirmatory cases.
The diffuse-component corollary is explicitly a post-result mathematical
extension. The frozen primary candidate predictor was not replaced by the
better secondary context predictor.

Main limitations: five training seeds, small random graph strata, deterministic
structured stress graphs, finite round caps, quadrature estimates of candidate
laws, shared graph interfaces in the generic baselines, a one-seed secondary
teacher, and a conditional numerical arithmetic model. Both neighbor reductions
were changed together; shared node--color representations were not isolated.
Full per-color histograms are not stored for every graph round; per-round
support/mass/collision summaries and local-grid histograms are retained, with
failed final states and seeds available for replay. Universal candidate
anti-concentration and termination remain open research questions rather
than unfinished proofs presented as results.
