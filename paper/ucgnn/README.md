# Integrated scientific manuscript

`main.tex` and `main.pdf` are the English coauthor-review version integrated
on 31 October 2026 and editorially revised on 1 November. Internal audit and
delivery notes stay in the working documents. Current page count, compiler, source/PDF hashes and reference
checks are in `results/integrated-paper-20261031/paper-build.json`.
Author list/order and target-venue formatting remain coauthor decisions.

From the repository root, in the Python environment documented in
`docs/INTEGRATED_PAPER_REPRODUCE.md`:

```bash
PYTHONPATH=src python scripts/build-coauthor-assets.py
PYTHONPATH=src python scripts/build-coauthor-paper.py
python scripts/verify-coauthor-integration.py
```

The figure builder reads compact verified results; it neither trains nor
simulates. The new component figure is based on all selected original cases,
including ineffective interventions. Original architecture, continuation and
Johansson/GM mechanism figures remain. The article uses five local bibliography
files; no bibliography is fetched during a build. The compiler receipt lists
actual figure/TeX/Bib inputs, references, overfull boxes and embedded fonts.
Renders and their visual inspection are documented separately.

The six handoff/audit documents are linked from the root README. The integrated
manifest authenticates current artifacts without changing historical manifests.
The portable runtimes contain the original 40-model trial plus seven original
transfer-weight files. Three additional hash-pinned packages now provide all
105 trainings' banks/weights and the full historical 810,000-row delivery.
The new [release audit](../../docs/MARTIN_ARTIFACT_DELIVERY_AUDIT.md) distinguishes
fresh inference, raw terminal reconstruction and trajectory replay. The
pre-release integrated delivery is preserved as a byte-verified archive.
The fourth package supplies the240 original files of the additional transfer
campaign; recovery and audit use `scripts/integrated-transfer-delivery.py`.
Its delivery is closed in the working documents; the article/PDF are unchanged.

## Historical versions

`archive/pre-integration-5993a5e4.{tex,pdf}` preserves Pedro's immediate previous
version. `archive/pre-revision-c3affd16.*` preserves its predecessor. Earlier
archives and scientific results remain untouched. To rebuild an old article,
use a separate checkout of its source commit and its matching runtime. Do not
run `build-revised-paper.py` or the historical closure generators against the
integrated article or use their manifests to validate this version.

The published Martín branch supplies additional experiments, code and a
literature review. Its completed rewritten manuscript was not present in the
published tree; integration therefore uses the actual artifacts, not an
assumed completed alternative text. See the provenance record for both hashes.
