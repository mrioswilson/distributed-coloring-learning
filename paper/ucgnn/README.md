# Integrated manuscript and build

`../../paper/PAPER_DRAFT.md` is the complete English manuscript. Its source
template is `../PAPER_DRAFT.md.in`; `dcl.integration.report` inserts tables,
claim provenance and human-readable verified references from exported results.
Do not manually change generated numerical tables.

`main.tex` is generated from that Markdown with Pandoc 3.9/natbib.
`main.pdf` is a 36-page unpublished draft including the counterfactual extension.
The current evidence-derived build is:

```bash
PYTHONPATH=src python scripts/report-multicf.py
PYTHONPATH=src python scripts/plot-multicf.py
PYTHONPATH=src python scripts/build-multicf-paper.py --compiler /path/to/verified/tectonic
```

Its receipts and reviewed renders live in
`results/multiteacher-counterfactual/paper-build.json` and `paper-review/`.
`pre-counterfactual-main.{tex,pdf}` preserves the previous manuscript.
The commands below document the earlier integrated build.

`pre-integration-main.tex` and
`pre-integration-main.pdf` preserve the earlier manuscript byte for byte.
Its documentation remains in `docs/historical/PAPER_README_pre_integration.md`.
Earlier commits and recovery archives also retain the original study.

Run from the repository root after installing `.[dev,study,paper]`:

```bash
python -m dcl.integration.report
python scripts/setup-paper-compiler.py
python scripts/build-integrated-paper.py
```

The setup script pins Tectonic 0.17.0 from its
[official release](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0).
It validates the Linux x86-64 musl archive SHA256
`8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`
and compiler SHA256
`a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da`.
The binary remains in ignored `.tools/tectonic`, with upstream licensing.
On other systems supply an appropriate Tectonic binary with the build's
`--compiler` option. First use may download TeX packages/fonts.

Build outputs `results/integration/paper-build.{json,txt}` record source/PDF
hashes, bibliography keys, compiler version/hash, reference diagnostics,
missing glyphs and overfull boxes. Ignored `results/integration/paper-review/`
contains page renders. Compilation success does not imply visual review:
the script resets `visual_review_passed=false` for every new PDF. After
inspection, record review scope and exact reviewed hashes in that JSON.
The committed PDF passed both checks. Table 4 continues onto the next page.

Four vector PDF figures and PNG previews are generated from exported data.
Their hashes and input hashes are in `results/integration/figure-manifest.json`.
The integrated draft uses the vector versions. The old figure set remains
preserved for historical reporting.

`references.bib` retains earlier verified ML/statistical sources.
`integration-references.bib` adds teacher and Holm references. The two
bibliography-verification JSONs identify primary source URLs and the scope
of checks; this is a scoped review, not an exhaustive novelty claim.

See `docs/PAPER_SCIENTIFIC_AUDIT.md` for limitations and claim provenance,
especially the 40 unavailable historical collaborator checkpoints. The new
balanced experiment and matched statistical controls are independent work,
not an exact replay of those unavailable models. Author order and submission
venue remain undecided; no accepted publication is implied.
