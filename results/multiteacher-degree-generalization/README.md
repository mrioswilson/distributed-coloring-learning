# Frozen multiteacher degree study

Independent branch `research/multiteacher-degree-generalization`, based on
`291d4b82`. No push or release upload was performed.

The seven requested reports are in `docs/`; the closure review explicitly
answers P1–P15. `native-21-consolidation.csv` preserves the heterogeneous
21/21 native result, with training, seeds, labels, parameters and evidence.

The principal extension uses 90 original frozen policies for six justified
teachers, plus 90 secondary controls. BE remains native-only. The source
checkpoint inventory has195 entries; the supplemental equal-update CT
inventory adds30 existing weights. All225 original files remain unchanged.

The pilot contains3570 student runs; confirmation contains810000; the
separate larger-size pilot contains240. Only the selected confirmatory
conditions have1000 runs per checkpoint. The 450000 primary runs contain
449791 successes: 16/18 teacher/core combinations pass every selected
condition in all five replicas. Johansson/Transformer/s83101 has972/1000 at
global n64/degree10; GM/Transformer/s94102 has819/1000 at local n64/degree4.
These totals describe multiple distinct policies, not one averaged policy.

`checkpoint-condition.csv` gives all checkpoint/condition counts, confidence
intervals and costs. The raw rows, exact references, graphs/seeds and first
failure traces live under the three stage directories. `failure-index.json`
and `failure-replays/` retain all failed conditions and15 matching full-batch
diagnostic replays. `paired-functional-equal-update.csv` separates final
success from the838188 COMMIT diagnostic queries in `counterfactual-probes/`.
The CT preference-gap, Johansson diversity and GM palette/repair witnesses
are documented separately. Conditional 64-support COMMIT intervals for45
canonical passive policies are in `commit-certificates/`; none establishes
the sufficient universal head property for these weights.

The PDF is `paper/ucgnn/main.pdf`. Its build receipt records compiler,
source/figure hashes,40 pages and the reviewed new tables/figures. A layout
adjustment reserves caption space for older tall plots without changing
their data. The final regression result is388 tests. Some historical unit
tests optimize disposable temporary models to check gradients/resume; no
such weights are scientific models or inputs to this study. Campaign
training-update counters refer to scientific experiments, which use only
pre-existing frozen weights. The higher-degree training proposal was not run.

## Local runtime and recovery

`frozen-degree-runtime.tar.gz` contains all raw runtime files and225 weights;
it is a local artifact, intentionally not uploaded. `artifact-index.json`
records its SHA-256 and size; `runtime-manifest.json` gives file hashes.
The archive and large raw rows are ignored by Git, with their digests and
reviewable summaries retained. A clone alone does not contain those assets.

From this branch plus the runtime archive, with the recorded Python study
dependencies available:

```sh
tar -xzf results/multiteacher-degree-generalization/frozen-degree-runtime.tar.gz
export PYTHONPATH=src
python scripts/verify-degree-study.py --portable
python scripts/replay-degree-smoke.py
```

The portable audit needs no external historical worktree. The strict local
audit additionally checks225 source weight hashes and610 preserved
historical documents/figures. The smoke re-executes576 already frozen rows
from18 teacher/core representatives; it is reproduction, not new independent
test data. `clean-recovery.json` records extraction and execution in a fresh
temporary directory, using the current versioned source/manuscript and bundle.

To recompute final results from scratch, use a fresh checkout/runtime output
directory containing the two inventories, unchanged checkpoints and frozen
designs, then run `python -m dcl.degree.run confirmatory --workers 4`.
Existing completed outputs are verified and reused, never overwritten for
a better score. `scripts/degree-confirmatory-tail.py` is an optional recorded
scheduling optimization for an unstarted tail; it changes no final sample.
The pilot's historical resource amendment remains explicit and is not
silently converted into a homogeneous fixed-step experiment.

To regenerate analysis/manuscript from completed outputs:

```sh
python -m dcl.degree.report
python scripts/analyze-degree-probes.py
python scripts/close-degree-study.py
python scripts/compare-degree-functional-controls.py
python scripts/append-larger-size-results.py
python scripts/plot-degree-mechanisms.py
python scripts/build-degree-paper.py
```

The compiler path/hash is in the build script. No training command is part
of this reproduction workflow. Costs below refer to measured components,
not a fabricated complete machine-time total.

## Recursos medidos y límites contables

| Campaña | CPU de evaluaciones completas (s) | Wall de invocación final (s) | Pico RSS de un worker (MiB) |
|---|---|---|---|
| pilot | 2552.215 | 439.749 | 724.0 |
| confirmatory | 9079.829 | 1749.756 | 719.4 |
| larger-size-pilot | 88.706 | 24.770 | 538.6 |

CPU mide evaluate (inferencia/simulación y diagnósticos), no toda la carga de pesos, generación de grafos, I/O o auditoría. El wall piloto corresponde a la recuperación final que reutiliza resultados anteriores; no incluye todo el intento interrumpido. CPU de celdas interrumpidas sin resultado no se midió por separado y no se imputa como cero. La confirmación usó un executor principal de4 y otro de4 sobre60 jobs aún no iniciados: máximo8 workers simultáneos; la CPU de ese tail ya está en el total de evaluaciones y no debe sumarse otra vez. Las150000 referencias exactas confirmatorias consumieron159.297s wall con4 workers; su CPU no se guardó separadamente. Sondas, intervalos y auditorías tienen recibos propios o magnitudes no medidas; no se presenta la suma parcial anterior como coste total de la misión. Tiempo GPU de las campañas:0. Entrenamientos científicos nuevos:0. El RSS de la tabla es por proceso, no el pico agregado del sistema.
