# Learning Distributed Algorithms from Algorithmic Teachers

Unpublished integrated research draft, 23 October 2026. We study what a neural
policy learns from a graph-coloring teacher: local actions, transitions,
valid autonomous solutions, execution cost, and the probability law over
complete solutions. These are different learning objectives.

Start with the [complete paper](paper/PAPER_DRAFT.md), its
[compiled PDF](paper/ucgnn/main.pdf), the
[independent-review dossier](docs/PAPER_SCIENTIFIC_AUDIT.md), and the
[statistical reconciliation](docs/integration/GLOBAL_DISTRIBUTION_RECONCILIATION.md).
The [evidence ledger](results/integration/evidence-ledger.json) maps principal
claims to code, configurations, checkpoint identities and evidence boundaries.

The integrated historical baseline is `integration/scientific-audit-20261023`.
The query-budgeted study is on `research/sample-complexity-20261023`. Original
branches and negative results remain preserved; main is not overwritten. The
[preservation manifest](docs/integration/PRESERVATION_MANIFEST.md) and
[recovery verification](docs/integration/RELEASE_ASSETS.json) identify the
remote recovery point made before Pedro Montealegre's work was integrated.

## Scientific questions

1. When does local imitation produce valid autonomous solutions?
2. Which information, aggregation and palette representation suffice?
3. How do these answers change across teachers?
4. Which properties transfer across size, degree, palette and graph family?
5. Can a functionally successful policy change the complete solution law?
6. How many acquired teacher decisions allow all five independently trained
   policies to complete a frozen benchmark with zero failures?

The evidence supports the last separation on specified fixed cohorts. We do
not infer equality from nonrejection, universal termination from testing, or
a universally best or minimum-size architecture.

## Algorithmic teachers

The historical architecture/distribution studies use a Luby-type random-color
teacher with local lists `C_v = {0,...,d(v)}`. Native multi-teacher programs
instead use common K=4 on simple undirected graphs of maximum degree three,
contiguous vertex labels and at most 65,536 vertices, with distinct finite IDs
where used. High-degree native support is not implied by historical Luby data.

| Native teacher | Implemented decision | Supplied mechanism / scope |
|---|---|---|
| LUBY | Participate with probability 1/2; uniform available color; reject equal neighboring offers | Two exchanges per trial |
| JOHANSSON | Always-active random-color trial | Common four-color lists |
| JP | Pending strict ID maxima take first available color | Fixed ID order; historical cap 128, new supervision deadline n |
| COLOR_TRIALS | Strict per-color discrete preference winners | Repeated subroutine with known degree bound |
| BE_BASE_D3 | Two Linial reductions, then 13 KW reductions to four colors | Polynomial arithmetic and phase clock; degree-three base case |
| SPECULATIVE_GM | Four-block speculative first fit and serial conflict repair | Supplied schedules; temporary conflicts allowed |
| GREEDY | Ascending-ID sequential first fit | Supplied writable-node schedule |

These are scoped variants. Original-paper algorithms and complexity bounds
must not be inherited silently. The
[teacher audit](docs/integration/TEACHER_CORRECTNESS_AUDIT.md) gives definitions,
primary references, conditional safety/termination proofs and finite checks.
Almost-sure uncapped termination differs from completion by a fixed cutoff.
Exact-teacher guarantees do not automatically transfer to learned heads.

The [Pedro reconstruction](docs/integration/PEDRO_RESEARCH_RECONSTRUCTION.md)
covers his direct GNN, E1 adaptations and behavioral study. Five retrained and
35 native E1 weights, source pools and original raw banks remain unavailable;
their identities are in the [source manifest](docs/integration/PEDRO_SOURCE_MANIFEST.json).
Their numerical tables are archival evidence, not independently replayed
models. Newly trained checkpoints do not replace those missing artifacts.

## Architectures and baselines

| Architecture | Learned structure | Interpretation |
|---|---|---|
| E1 NORMALIZED | Shared node/color networks, MAX messages, normalized ordered palette prefixes | Color order can matter |
| DeepSets | Shared color scores and mean palette context | Prefix-free palette scorer; still uses graph messages |
| LocalPaletteTransformer | Attention within a node's palette | Palette attention plus learned graph messages |
| Small E1 | Hidden/message dimensions 8/4 instead of 64/32 | 551 parameters for Luby; 1,560 with native context |
| No communication | Large core with received states, relations and masks erased | Own observations and supplied schedules retained |
| Exact teachers | Explicit rules | Zero trainable parameters |

Pedro's 36,719-parameter direct-action GNN and historical raw-endpoint, SUM and
MEAN controls remain comparators. Representation/loss/budget differences are
not treated as a single controlled architecture factor.

Native models retain supplied palette-range masks, pending-node gating,
phase/schedule context and BE polynomial evaluation. Availability, candidate
and confirmation remain learned. Model actions are applied without an oracle
safety filter. All three palette architectures share the same native interface;
see the [balanced protocol](docs/integration/BALANCED_EXTENSION_PROTOCOL.md).

## Environment and installation

Tested integration environment: Debian 12, Python 3.11.2, Intel i7-10700,
one Torch/BLAS CPU thread. The new 70-model study used CPU. A local RTX 4060
8 GB was available and unused for it. Historical studies retain their own
hardware records. Major versions were Torch 2.11.0+cu128, NumPy 2.4.6,
SciPy 1.17.1, NetworkX 3.6.1, pandas 3.0.6, matplotlib 3.11.2 and pytest 9.1.1.
The installed CUDA-enabled Torch build also executes the CPU study.

The [environment record](results/integration/environment.json) and
[package snapshot](requirements/integration-tested.txt) record actual versions.
The snapshot includes Linux/CUDA-specific packages; it is not a portable lock.
Dependency ranges support installation. Cross-platform bitwise training
identity has not been established.

```bash
git clone --branch research/sample-complexity-20261023 https://github.com/mrioswilson/distributed-coloring-learning.git
cd distributed-coloring-learning
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,study,paper]'
```

Run subsequent commands from the repository root. The paper extra supplies
Pandoc 3.9 and PDFium bindings. Tectonic is installed separately below; its
first use may download TeX resources. Use recorded versions when examining
numerical differences.

## Sample complexity: strict five-seed benchmark

The [information audit](docs/TEACHER_MODEL_INFORMATION_COMPATIBILITY.md),
[frozen protocol](docs/SAMPLE_COMPLEXITY_PROTOCOL.md), and
[P1–P12 results](docs/SAMPLE_COMPLEXITY_RESULTS.md) define the new central study.
The [data dictionary](docs/SAMPLE_COMPLEXITY_DATA_DICTIONARY.md) and
[new evidence ledger](results/sample-complexity/evidence-ledger.json) map
query units, raw schemas, hashes and claim boundaries.
It uses seven teachers, three palette architectures, all seven supervision
budgets from 1,000 to 1,000,000 and five independent training seeds: 735 fresh
main trainings. A query is one acquired node/state/action-component label;
replay is counted separately. Anonymous and extended information regimes are
reported separately. A package is PASS only if **all five policies complete
all 1,000 final executions each without failure**. High average success never
substitutes for this gate. This finite empirical result is not universal
correctness or a theoretical sample-complexity bound.

To rebuild curves/tables and verify exported evidence without training:

```bash
PYTHONPATH=src python -m dcl.samplecomplexity.report
python scripts/verify-sample-complexity.py
python -m pytest -q tests/test_samplecomplexity.py
```

To restore the separate new caches, negative checkpoints, optimizer/RNG states
and raw development rows from the [study release](https://github.com/mrioswilson/distributed-coloring-learning/releases/tag/sample-complexity-2026-10-23):

```bash
python scripts/fetch-sample-complexity.py
python scripts/verify-sample-complexity.py --runtime
python scripts/replay-sample-complexity-cell.py --teacher LUBY --architecture normalized --seed 83101 --budget 1000
```

The fetcher verifies archive/member hashes and refuses conflicting files.
These checkpoints live under `data/sample-complexity-v1/`, separate from the
70 historical integration checkpoints. Primary raw final/exact execution rows,
frozen batteries, individual training/control summaries and law banks are
versioned under `results/sample-complexity/`.

For a new independent experiment, create an isolated checkout and namespace
with the same installed Python environment:

```bash
python scripts/replicate-sample-complexity.py \
  --directory /absolute/path/to/new-replication \
  --namespace independent-replication-01 --seed-offset 100000
```

This preserves the published cohort and runs the full sequence below in the
new checkout. Do not rerun acquisition/design over the published frozen
exports; use fetch/verify for that cohort. The stages, for a new experiment:

```bash
PYTHONPATH=src python -m dcl.samplecomplexity.runner pilot
PYTHONPATH=src python -m dcl.samplecomplexity.runner design
PYTHONPATH=src python -m dcl.samplecomplexity.runner validate
PYTHONPATH=src python -m dcl.samplecomplexity.runner train
PYTHONPATH=src python -m dcl.samplecomplexity.runner controls
PYTHONPATH=src python -m dcl.samplecomplexity.runner confirm
PYTHONPATH=src python -m dcl.samplecomplexity.optimization
PYTHONPATH=src python -m dcl.samplecomplexity.capacity
PYTHONPATH=src python -m dcl.samplecomplexity.law
PYTHONPATH=src python -m dcl.samplecomplexity.diagnostics
PYTHONPATH=src python -m dcl.samplecomplexity.full_fidelity
PYTHONPATH=src python -m dcl.samplecomplexity.failure_witnesses
python scripts/estimate-sample-complexity-flops.py
PYTHONPATH=src python -m dcl.samplecomplexity.report
python scripts/verify-sample-complexity.py --runtime
python scripts/build-integrated-paper.py
```

Completed frozen training/checkpoint results are reused, not renamed as fresh
trainings. A consumed final attempt cannot rerun automatically. For a new
scientific replication use an independent namespace/root and a new frozen
confirmation battery; editing the consumed protocol is rejected. Model controls
and the law supplement are separate exploratory analyses and never change the
functional gate or choose a second final candidate.

## Quick reproduction from Git evidence

This route needs exported Git results, no raw pools or GPU:

```bash
python -m pytest -q
python scripts/verify-integration.py
python -m dcl.integration.report
```

The verifier checks cell counts, graph separation, the full Holm family,
historical source guards, figures and manuscript hashes. The report rebuilds
four figures, summary tables, the claim ledger and complete Markdown draft
from persisted data; it does not rerun rollouts or permutations.

To obtain the pinned Linux x86-64 compiler and rebuild LaTeX/PDF:

```bash
python scripts/setup-paper-compiler.py
python scripts/build-integrated-paper.py
```

The build checks citations, unresolved references, missing characters and
overfull boxes and renders selected pages. A new build has
`visual_review_passed=false` until its renders are inspected. The committed
PDF has a review tied to exact hashes. See the
[paper instructions](paper/ucgnn/README.md). Earlier LaTeX/PDF are preserved as
`pre-integration-main.*` and in Git history.

## Checkpoints, datasets and graphs

The [recovery release](https://github.com/mrioswilson/distributed-coloring-learning/releases/tag/research-recovery-2026-10-23)
contains historical global/robust/factorial archives and all 85 final
paper-study checkpoints, including negative controls. Its eight assets total
1,620,537,692 bytes; local hashes and matching remote digests are in
`docs/integration/RELEASE_ASSETS.json`. The preservation inventory also records
local legacy files outside that selected release scope; it does not claim
every older evaluation file was remotely archived.

The [integration release](https://github.com/mrioswilson/distributed-coloring-learning/releases/tag/scientific-integration-2026-10-23)
contains all 70 new final checkpoints, 14 teacher pools, 385 raw output banks,
and the paper/audit. Its [asset manifest](docs/integration/INTEGRATION_RELEASE.json)
supports a verified download:

```bash
python scripts/fetch-integration-runtime.py
python scripts/verify-integration.py --runtime
```

The fetcher verifies the archive and every member, validates extraction paths,
and refuses to replace differing existing files. Data then live under ignored
`data/integration-balanced/`. The [data dictionary](docs/integration/DATA_DICTIONARY.md)
documents seeds, statuses, field units and artifact relationships.

[graphs.json](results/integration/multiteacher/graphs.json) includes edge lists,
degrees, generation seeds and labeled fingerprints. Pool graph identities and
hashes are in [pool-manifests.json](results/integration/multiteacher/pool-manifests.json).
All 70 final checkpoint hashes were
[frozen before evaluation](results/integration/multiteacher/frozen-checkpoints.json).
Two initialization seeds are two training replicas; 128 rollouts of one model
are not 128 replicas.

## Training and autonomous evaluation

The balanced cohort has seven teachers, five architectures and seeds
73000/73001. Every cell gets 500 Adam updates at .001, batch 128 root decisions,
24 cubic N40 graphs and one teacher trajectory per graph, plus eight validation
graphs. Architectures within teacher/seed share pools and sampled roots.
The last prespecified update is retained regardless of validation performance.
A separate cost-only pilot selected 500 updates before scientific data existed.

```bash
python -m dcl.integration.study cost
python -m dcl.integration.study run
```

Both commands were run for the archived study. `cost` writes a timing-dependent
design; retain the released record when reproducing its original provenance.
`run` uses the frozen design and caches pools/checkpoints/banks. It retrains
and evaluates on a clean runtime and resumes on restored artifacts. Archived
sampling/training took 1,988.3 CPU wall seconds, including 410.4 seconds of
parameter updates. Permutation tests are additional. Historical training
protocols differ; this budget is not Pedro's 17,760-update experiment.

The evaluator uses five held-out graphs and 128 independent runs per
graph/checkpoint, plus exact references: 49,280 attempts. All failures, native
step counts and seeds remain recorded. Steps mean different things across
teachers, including sequential writes, reductions and parallel trials.

Rerun the independent scalar and explicit-tape teacher audit with:

```bash
python -m dcl.integration.teacher_audit
```

It covers 103 unlabeled graphs on one through six vertices, degree at most
three, two tapes each, and eight common-array canonical/native Luby comparisons.
It is not exhaustive over all tapes, IDs or arbitrary partial states.

Common-state local action distributions and paired decisions can be replayed
from the released validation pools/checkpoints:

```bash
python -m dcl.integration.action_audit
```

This descriptive audit preserves every phase's native action, participation,
candidate and COMMIT histogram, including COMMIT queried on teacher versus
own proposals. It reuses the original state/random values for every policy.
Empirical histogram distances are not certified population TV bounds or
autonomous output-law tests.

## Statistical methodology and evaluation

Observations are whole output rows; coordinates within a coloring are
dependent. Historical and new primary tests compare labeled complete
colorings conditional on proper completion by the specified cutoff. Failures
remain separate. Pedro's behavior study includes terminal status and also a
color-name quotient, with different models, graphs and sample sizes.

V2 separates witness selection from evaluation, permutes whole rows, uses
inclusive plus-one Monte Carlo p-values and applies Holm to the entire planned
family. New native tests use 19,999 permutations and 350 planned hypotheses.
The [alphabet adapter](docs/integration/DECLARED_ALPHABET_ADDENDUM.md) allows
common K=4 on lower-degree graphs without changing actual degrees or the frozen
local-list implementation. Fewer than four proper rows on either side makes
a test unavailable; `p=1` is conservative bookkeeping, not equality evidence.

```bash
python -m dcl.integration.statistics_audit new-study
python -m dcl.integration.statistics_audit power
```

The first uses released banks and cached per-case tests, then adjusts all 350
together. The second regenerates known-law null/power controls with independent
seed namespaces, twenty experiments per law/size, identical data for both
tests, 999 permutations and nominal .05. It takes longer than regenerating
figures. Its power is specific to those laws, not a universal guarantee.

The six matched archival comparisons require the historical factorial runtime;
the tested command and required separate archive root are in the
[crosscheck protocol](docs/integration/STATISTICAL_CROSSCHECK_PROTOCOL.md).
Exported bank hashes/results remain accessible without that worktree. Historical
[factorial replay](scripts/replay-factorial-study.py) and the
[robust report](results/ucgnn-robust-distribution/FINAL_REPORT.md) preserve their
frozen methods and resource requirements.

The mixture statistic agrees with unbiased MMD; no calibration defect was
established. Both methods agree on the six matched archives. Finite power
differs on a specified localized alternative. MMD, permutation and Holm
references are verified in the manuscript bibliography.

## Principal results

| Evidence | Result | Scope |
|---|---|---|
| Historical architecture study | E1/DeepSets/Transformer each 1,520/1,520 proper completions | Five training seeds each; fixed cohort, degrees through 200 and sizes through 640 |
| Historical factorial laws | 115/170/180 V2 rejections, each out of 240 | 48 new regular graphs, 15 checkpoints; Holm family 720 |
| New balanced study | All three large models: 1,280/1,280 for each of Luby, Johansson, ColorTrials and greedy | Two seeds, five fixed graphs, K=4, short budget |
| Small models | Luby and greedy each 1,280/1,280; ColorTrials seeds 640/640 versus 0/640 | Finite sufficiency, not minimality |
| No communication | 38/8,960 proper completions | Received information matters under this interface/training |
| New output laws | Six Holm rejections; 350 planned, 227 testable | Small Luby and ColorTrials models each succeed 640/640 yet differ on three graphs |
| Known-law power | Mixture 0/20 and 3/20; V2 20/20 at both sizes | Center-only star alternative, TV=99/199; finite nominal-.05 controls |

JP, BE and speculative GM retain substantial/intermediate functional failures;
the [full matrix](results/integration/functional-summary.csv) keeps every cell.
[Training](results/integration/multiteacher/training.json),
[functional](results/integration/multiteacher/functional.csv) and
[global](results/integration/statistics/new-study-global.csv) records retain
replicas separately. Local accuracy, cost and the two fidelity notions must
not be collapsed into a universal model ranking.

## Repository map

| Path | Role |
|---|---|
| `src/dcl/multiteacher/`, `multiteacher_e1/` | Native teachers, direct GNN and E1 adaptations |
| `src/dcl/universal/`, `paperstudy/` | Canonical models and historical architecture study |
| `src/dcl/localpalettes/`, `globaldist/`, `robustdist/`, `sizedegree/` | Historical palette/global-law studies and frozen methods |
| `src/dcl/integration/` | Active interfaces, scalar audit, balanced study, statistics and reporting |
| `configs/`, `scripts/`, `tests/` | Protocols, reproduction tools and regressions |
| `results/integration/` | Exported new evidence, manifests and verification records |
| `results/ucgnn-*/`, `results/multiteacher*/`, `results/e1-behavior/` | Separate historical cohorts, including negative results |
| `docs/integration/`, `docs/historical/` | Audits/provenance and prior documents |
| `paper/` | Full English draft, LaTeX, PDF, figures and bibliography |

The established `dcl` layout remains for checkpoint/script compatibility.
The [pre-integration README](docs/historical/README_pre_integration.md) is
preserved. Historical partial snapshots are not the integrated final evidence.

## Limitations and next work

Recovering Pedro's 40 weights and raw banks is necessary for historical
model-level replay. New native training has two seeds and five fixed test
graphs, so it does not yield population rankings. Supplied native schedules
remain explicit. Earlier high-degree/local-list evidence does not transfer
automatically to every teacher or small model. Native status meanings differ
from historical ones; consult the data dictionary.

Nonrejection is not equality, rejection counts are not TV distances, and
event bounds do not establish monotone full-TV growth. Regular-graph degree
and palette effects are confounded. Synthetic power does not explain every
original comparison. Mathematical results require their stated hypotheses.
Before submission: recover missing artifacts, expand balanced full-budget
replicas, complete a broader novelty review and agree authorship/venue.

## Citation, license and acknowledgments

There is no accepted publication or assigned DOI. Cite the repository and
exact integration commit as an unpublished artifact, provisionally titled
*Learning Distributed Algorithms from Algorithmic Teachers* (2026).
Researchers must agree the author list/order before submission.

Git provenance attributes earlier degree/palette/mechanism studies to Martín
Ríos-Wilson and multi-teacher/retrained-E1 work to Pedro Montealegre. Both
histories and earlier repairs retain attribution. Algorithms and dependencies
retain their references/licenses. No repository-level LICENSE was present;
the integration invents no license or additional reuse permission. Tectonic
downloads retain their upstream licensing.
