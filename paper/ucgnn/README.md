# Authoritative closure manuscript

`main.tex` is the edited English closure source, dated 26 October 2026.
`main.pdf` is built from it. The current page count, source/PDF/figure hashes,
compiler version, reference checks and render paths are recorded in
`results/final-closure/paper-build.json`; do not infer the page count from an
older report. `paper/PAPER_DRAFT.md` and its template are historical drafts,
not the source of the closure manuscript.

From the repository root, with the restored runtime and `.[dev,study,paper]`:

```bash
python scripts/setup-paper-compiler.py
PYTHONPATH=src python scripts/build-closure-assets.py
PYTHONPATH=src python scripts/build-closure-paper.py
```

The asset script derives the native, JP, BE, baseline, label-cost and transfer
tables and both vector figures from authenticated existing records. It does
not train or simulate. The build checks the pinned Tectonic 0.17.0 binary,
unresolved citations/references, glyphs, overfull boxes, key PDF numerical
strings and off-page text, and renders every page. Actual visual inspection
is a separate `pdf-visual-review.json` receipt, never inferred from compilation.

The Linux x86-64 compiler SHA-256 is
`a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da`.
The setup script verifies the official archive and binary; first compilation
can download TeX packages. Compiler and rendered pages are local runtime.
`main.tex` uses the four local bibliography files, including the new
`closure-references.bib`; no bibliography is fetched during manuscript writing.

## Evidence and reproduction

Read `docs/FINAL_EVIDENCE_AUDIT.md`, `docs/EVIDENCE_REGISTER.md` and the
machine-readable `results/final-closure/evidence-register.json` for claim,
protocol, data and checkpoint provenance. `artifact-manifest.json` authenticates
the final source/result set, while `verified-inputs.json` lists checked raw
degree/size files and weights. `native-certifications.json` identifies all
105 selected native policies and their original final rows.

Large raw JSON, NPZ banks and weights require runtime restoration; they are
not made available merely by cloning source. The local closure delivery and
its checksums are described in `results/final-closure/delivery.json`. Historical
runtime recovery remains documented in `results/multiteacher-counterfactual/README.md`
and `scripts/fetch-sample-complexity.py`. No new release or push is performed.
The exported source/runtime can be inspected before any external publication.

## Historical builds

The previous integrated manuscript is preserved byte for byte in
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
