# Reproduce the revised article

Run from the repository root. The canonical revised source is
`paper/ucgnn/main.tex`; the previous source/PDF are preserved in
`paper/ucgnn/archive/pre-revision-c3affd16.*`. Original study results are not
overwritten. New outputs are in `results/paper-revision-20261030`.

Use the project's Python dependencies and `pypdfium2==5.14.0` for PDF review.
This workspace uses Python 3.12, PyTorch 2.5.1 and one CPU thread. For the PDF,
install `pdflatex`, `bibtex`, Latin Modern and the packages named in main.tex.
The revised build records the actual compiler and environment, rather than
pretending to execute the historical Linux-x86 Tectonic pin on ARM.

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python scripts/recover-revision-evidence.py --cache /path/to/archive-cache
python scripts/reanalyse-paper-be.py
python scripts/probe-paper-rescues.py
python scripts/summarize-paper-revision.py
python scripts/audit-revision-continuations.py
python scripts/probe-revision-be-failures.py
python scripts/build-revision-assets.py
python scripts/build-revised-paper.py
python scripts/verify-paper-revision.py
python scripts/package-paper-revision.py --manifest-only
```

Run heavy steps in series on memory-constrained hosts. The recovery command
verifies archive and file hashes, resolves checkpoint identity rather than
guessing by label, and preserves existing tracked metadata. Source-only
checkouts suffice to rebuild the paper from the compact committed analysis
outputs; checkpoint-based reanalysis needs the public runtime archives.
Historical CSV CRLF bytes in releases and LF bytes in Git are recorded as
separate file identities. The new manifest always hashes the actual delivered
bytes. The historical closure manifest remains a receipt of that delivery.

`audit-revision-continuations.py` reads the restored frozen base/auxiliary/test
banks and original prediction summaries. It recomputes truths, fine keys,
programmed diagnostics, class/phase coverage and coarser logical overlap;
it never trains, changes the split or rewrites the original outcomes.
Its output is `results/paper-revision-20261030/continuation-audit.json`.
The gate and coverage audit are post-test descriptive additions; the original
primary endpoint and all forty outcomes are retained. The asset builder
uses that compact JSON without requiring a restored runtime.

`probe-revision-be-failures.py` separately runs inference on twelve selected
frozen models and replays the first original 32-case BE105010/native cohort.
It checks acquired-label fit, final confusion counts and a local intervention
on one participation bit at the first violation of one selected case.
It does not train, continue an intervened rollout or estimate a population
rescue rate. Auxiliary P/C values condition COMMIT and are not direct
participation-loss targets. Its compact output and input hashes are in
`results/paper-revision-20261030/be-failure-diagnostics.json`; the asset builder
reads that JSON. To use a relocated revision runtime, pass
`--runtime /path/to/restored-runtime` to this probe. Both audits are explicitly
post-test and retain all original outcomes. The paper's reproduction block
includes both steps before generating assets.

The current Figure 3 explains the two copied failure witnesses rather than
repeating final percentages. `build-revision-assets.py` includes
`build-revision-mechanism-figure.py`, which reads `component-rescues.json`
and the two original, tracked first-failure JSON files. It checks their hashes,
actual edges/colors, single-candidate intervention and unchanged exact-COMMIT
control. No runtime restoration, neural inference or simulation is needed to
draw it. `failure-mechanisms.json` records the inputs and extracted fragments.
The complete six-row intervention table remains in the failure appendix.

The earlier paired-seed plot (`revision-prospective.pdf`), compact failure
bars (`closure-generalization-compact.pdf`) and original full grid
(`closure-generalization.pdf`) are preserved under `paper/ucgnn/figures`,
but are not inserted into the current article. All forty seed outcomes and
all 450 primary transfer cells remain in scope. The standalone historical
bar builder is retained; the current asset builder does not call it.

`build-revision-assets.py` also accounts for successful/failed BE step counts
from the unchanged twenty final result files, writing `be-step-accounting.json`.
The overall means include early failure exits; successful runs take 15 native
steps in both arms. This is descriptive accounting, not a new efficiency trial.

The 810,000 raw closure rows and later closure checkpoint copies remain in an
unpublished local delivery path recorded by Martín. All 225 checkpoint
identities can nevertheless be recovered from the earlier public archives;
the 810,000 raw rows cannot. The rescue analysis reconstructs the stored
first-failure cohorts using original checkpoint hashes and declared seeds.
Reported all-cohort closure totals remain attributed to the original study.

## New prospective trial

Read `PAPER_REVISION_PROTOCOL.md` before starting. Scientific stages are separate:

```bash
python -m dcl.paperrevision.experiment --stage train --runtime /path/to/new-runtime
python -m dcl.paperrevision.experiment --stage test --runtime /path/to/new-runtime
```

Training fixes the enumerated source/config/data identities and saves exact Adam/batch-RNG
state. It freezes all forty final weights before the test command can generate
diagnostic data or run rollouts. Only update 1,500 is analyzed; no development
selection is used. A source mismatch refuses resumption. Final tests are
idempotent and preserve completed results. Do not change the protocol after
opening its test. A new experiment requires a new namespace/output location.

The original source receipt is an enumerated subset, not a full transitive
dependency or environment lock: data/graph/storage helpers outside that list
are not authenticated by it. The receipt is preserved rather than expanded
retroactively. Independent human scientific review remains pending.

The delivered runtime index maps checkpoint and dataset hashes to relative
archive paths; set `DCL_ARTIFACT_ROOT` to a recovered repository root for legacy
absolute data/results paths. `DCL_PAPER_STUDY_ROOT` optionally points at the
historical palette study root. These compatibility paths do not alter frozen
checkpoint contents or replace missing weights with re-trained models.

The original forty-weight freeze is retained byte-for-byte, including its
original local path receipts. After relocating the revision runtime, replay
without rewriting those receipts:

```bash
python scripts/replay-revision-test.py --runtime /path/to/restored-runtime --teacher BE_BASE_D3 --seed 105002 --arm cf
```

This companion resolves the weight by teacher/seed/arm and checks the original
SHA256, data keys and source freeze. It writes a separate comparison of
diagnostics and all final colors/status/steps. It never edits the original
test results or uses them for checkpoint selection.

Historical frozen source hashes still refer to their original commit. Runtime
path compatibility changes are explicitly a revision; they must not be used
to claim byte-identical reproduction of an older source freeze. Restore that
commit when auditing the old source receipt.
