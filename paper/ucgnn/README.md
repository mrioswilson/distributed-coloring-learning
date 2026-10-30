# Revised scientific manuscript

`main.tex` is the English source revised on 30 October 2026, including the
scientific corrections after the adversarial audit. `main.pdf` is built from
that source. The current page count, source/PDF hashes,
compiler version, reference checks and render paths are recorded in
`results/paper-revision-20261030/paper-build.json`; do not infer the page count from an
older report. `paper/PAPER_DRAFT.md` and its template are historical drafts,
not the source of the closure manuscript.

From the repository root, with the restored runtime and `.[dev,study,paper]`:

```bash
PYTHONPATH=src python scripts/build-revision-assets.py
PYTHONPATH=src python scripts/build-revised-paper.py
```

The asset script reads existing analysis outputs and final rows, including the
post-test programmed controls, class/support audit and selected BE failure
diagnoses. The mechanism diagram also reads two tracked original witness JSONs;
it checks actual graph fragments and single-candidate interventions. It does
not train or simulate. The earlier paired-seed and failure-bar plots remain
preserved assets and are not included in the current article.
To recompute the scientific diagnostics
from the restored frozen banks, first run, in series:

```bash
PYTHONPATH=src python scripts/audit-revision-continuations.py
PYTHONPATH=src python scripts/probe-revision-be-failures.py
```

The second script runs frozen-model queries and observes an unchanged original
32-case cohort; its participation intervention is a local query, without
continuing an intervened rollout. These are selected post-test diagnoses.
The PDF uses native pdfLaTeX/BibTeX and records actual compiler hashes,
unresolved references, overfull boxes, embedded fonts and every page render.
Actual visual inspection is recorded separately, never implied by compilation.

The preceding closure used pinned Linux x86-64 Tectonic; its historical
receipt remains in `results/final-closure/paper-build.json`. Reproduce it in
a checkout of its original source. The historical builder refuses the
revised manuscript before writing to that receipt or compiling its PDF.
`main.tex` uses the four local bibliography files, including the new
`closure-references.bib`; no bibliography is fetched during manuscript writing.

## Evidence and reproduction

Start with `docs/PAPER_REVISION_CLAIMS.md`, `docs/PAPER_REVISION_REPORT.md` and
`docs/PAPER_REVISION_REPRODUCE.md`. The new delivery manifest uses actual bytes
and lives in `results/paper-revision-20261030/manifest.json`; it does not
replace original scientific freeze receipts. The retrospective gate,
coverage and BE failure diagnostics do not change the primary endpoint or original test.

For the historical closure, read `docs/FINAL_EVIDENCE_AUDIT.md`, `docs/EVIDENCE_REGISTER.md` and the
machine-readable `results/final-closure/evidence-register.json` for claim,
protocol, data and checkpoint provenance. `artifact-manifest.json` authenticates
the final source/result set, while `verified-inputs.json` lists checked raw
degree/size files and weights. `native-certifications.json` identifies all
105 selected native policies and their original final rows.

Large raw JSON, NPZ banks and weights require runtime restoration; they are
not made available merely by cloning source. The local closure delivery and
its checksums are described in `results/final-closure/delivery.json`. Historical
runtime recovery remains documented in `results/multiteacher-counterfactual/README.md`
and `scripts/fetch-sample-complexity.py`. The revised source, PDF and runtime
archive are delivered on `codex/paper-revision-20261030`; no separate GitHub
release is created. The delivery manifest checks their actual bytes.

## Historical builds

The immediately preceding closure is preserved in
`archive/pre-revision-c3affd16.{tex,pdf}`. The earlier integrated manuscript is preserved byte for byte in
`archive/pre-closure-71153993.{tex,pdf}`, along with earlier pre-integration,
pre-sample-complexity and pre-counterfactual copies. Reproduce those studies
in a separate checkout of their source commit. Their generators
`build-integrated-paper.py`, `build-multicf-paper.py` and `build-degree-paper.py`
belong to those historical versions and refuse to overwrite an authoritative
closure source in this checkout. They are not the commands for the current PDF.

## Editorial status

This is an externally reviewable scientific draft, not a formatted submission.
Author list/order and attribution require coauthor agreement; target-venue
style and anonymization remain editorial work. See
`docs/PAPER_CLOSURE_DECISION.md` for the explicit decision, all fifteen answers,
remaining items and the distinction between new results and reused history.
