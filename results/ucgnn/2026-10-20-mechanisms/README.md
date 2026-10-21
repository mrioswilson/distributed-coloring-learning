# Diagnósticos explicativos UCGNN — 20 de octubre de 2026

[Documento principal](../../../docs/UCGNN_MODELOS_EXPLICADOS.md) ·
[Resumen ejecutivo](../../../docs/UCGNN_PREGUNTAS_ABIERTAS.md).

Análisis post hoc de checkpoints existentes, principalmente réplica 0,
época 120. Los barridos son cuadraturas de funciones deterministas y las
trazas repiten grafos/cintas existentes; no aumentan el número de grafos
independientes de los experimentos. Se conservan los pesos y las cohortes.

| Archivo | Contenido |
|---|---|
| `model-inventory.json` | Configuraciones, formas, parámetros y hashes de siete UCGNN y cuatro M6B |
| `checkpoint-comparisons.csv` | Igualdad por módulo, inicial y tras entrenar, E1 frente a C1/E2/E3/A1 |
| `gradient-routing.csv` | Caminos autograd y norma L1 por pérdida/módulo, sin actualizar parámetros |
| `availability-type-sets.csv` | Los ocho conjuntos de tipos de primera comunicación, para E1/A1 |
| `commit-presence-reproduced.csv` | Los 64 conjuntos COMMIT de E1-0 y C1, reproducidos |
| `selection-map.csv` | r_c, profesor y candidato; separa malla principal de fronteras añadidas |
| `selection-summary.csv` | Coincidencias, reproducción original, monotonía, colisiones y colores efectivos |
| `proposal-law-*.csv` | Leyes de propuestas de mallas de 20.001 puntos medios |
| `continuous-score-maxima.csv` | Máximos del scorer continuo y fronteras ReLU que los explican |
| `quantile-hypothesis-check.json` | Contraste descriptivo de la aproximación por cuantil deformado |
| `autonomous-K201-frequency.json` | Frecuencia y comportamiento real k=201, por semilla y total |
| `round-traces.csv` | Todas las rondas: políticas propias y E1 evaluado en estados de A1 |
| `matched-state-candidate-comparison.csv` | Comparación con participación común rp<0,5 |
| `trace-*.npz` | Colores antes, cintas, propuestas, participantes, COMMIT, histogramas y color final |
| `trace-checks.json` | Grafos/semillas/procedencia, reproducción de negativos y expectativas de progreso inicial |
| `example-K*.json`, `example-K*-slots.csv` | Ejemplos pedagógicos de cuatro y once colores |
| `example-K*-tensors.npz` | Entradas/salidas completas de cada módulo en esos ejemplos |
| `diagnostic-run.json` | Dispositivo, hashes de código/pesos y preservación de checkpoints |
| `validation.json` | Comprobaciones cruzadas de los artefactos y referencias |
| `manifest.json` | Hashes de esta entrega y de sus scripts/documentos |

Figuras: `architecture`, `selection-mapping`, `proposal-distribution`,
`collision-stall` y `preferred-quantile`, cada una en PNG y SVG.

Reproducción desde la raíz del repositorio principal, dentro de `codex-disco`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python experiments/ucgnn_explain_models.py \
  --output results/ucgnn/2026-10-20-mechanisms
.venv/bin/python experiments/ucgnn_plot_explanation.py \
  --root results/ucgnn/2026-10-20-mechanisms
```

Las funciones de diagnóstico se importan del worktree extremo, cuyo código
Universal coincide con el principal. La ejecución toma el bloqueo GPU del
estudio. No hay pasos de optimización ni escritura de checkpoints.
