# Pruebas robustas de leyes de coloraciones completas

Este estudio conserva como referencia los 795 contrastes del worktree
`/workspaces/AlgoLearning/ucgnn-global-distribution-study`, commit `1ba4546`.
La nueva rama es `research/ucgnn-robust-distribution-tests`. No se reentrenan
redes ni se hace push. Solo se comparan Luby y los quince checkpoints
NORMALIZED, DeepSets y LocalPaletteTransformer del estudio reciente.

## Documentos y fases

- `docs/OLD_TEST_STATISTICAL_AUDIT.md`: reconstrucción y reproducción exacta
  del fallo histórico; inventario E1 de diez rechazos y 255 no rechazos.
- `docs/ROBUST_TEST_DEVELOPMENT_PLAN.md`: desarrollo prospectivo.
- `docs/ROBUST_INSTRUMENT_VALIDATION*.md`: rondas independientes y revisión
  de la búsqueda de pares tras fallar los controles trasladados.
- `results/ucgnn-robust-distribution/INSTRUMENT_VALIDATION_REPORT.md`:
  informe del instrumento, generado antes de la confirmación neuronal.
- `docs/ROBUST_GLOBAL_DISTRIBUTION_PROTOCOL.md`: protocolo final congelado
  después de validar y antes de generar las muestras confirmatorias.
- `docs/ROBUST_COLORING_DISTRIBUTION_THEORY.md`: pruebas y límites.
- `docs/ROBUST_CONTROL_DISTRIBUTIONS.md`: leyes conocidas y sus TV.
- `docs/ROBUST_NULL_REPLICATION_PLAN.md`: réplica diagnóstica independiente
  de 4.000 controles estrella nulos tras congelar el procedimiento, sin
  seleccionar métodos ni modificar la inferencia confirmatoria.
- `results/ucgnn-robust-distribution/FINAL_REPORT.md`: respuestas P1–P10.

DEVELOPMENT, validaciones 1/2, CONFIRMATORY y EXPLANATION tienen semillas y
funciones diferentes. La selección de características no utiliza etiquetas
de evaluación. Los vectores de los cinco checkpoints de una arquitectura
no se mezclan. Los grafos son bloques fijos; no son réplicas de una población.

## Reproducir los resultados archivados

Con el entorno original:

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python
$PY -m pytest -q tests/test_robustdist.py tests/test_robustdist_holdout.py tests/test_robustdist_numerics.py tests/test_globaldist.py tests/test_globaldist_integration.py tests/test_localpalettes.py tests/test_graphs.py tests/test_oracle.py
$PY -m dcl.robustdist.old_audit
$PY -m dcl.robustdist.audit
$PY -m dcl.robustdist.report
/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic -X compile --keep-logs paper/ucgnn/main.tex
```

Los análisis usan cachés de resultados por celda; los NPZ y sus hashes son
la fuente de datos. Para recalcular estadísticas en vez de leer cachés,
trabajar sobre una copia y retirar solo `confirmation-tests/` y
`small-tests/` del directorio de resultados, conservando sus originales.
Ejecutar entonces `python -m dcl.robustdist.confirmation --watch`. Las
semillas y el presupuesto muestral se mantienen fijos.

El archivo de entrega incluye una instantánea fuente, un bundle Git con
los commits, datos brutos nuevos e históricos y los quince checkpoints
congelados. Para restaurar la procedencia Git después de extraerlo en un
directorio vacío:

```bash
git init
git fetch source-history.bundle research/ucgnn-robust-distribution-tests
git reset --mixed FETCH_HEAD
mv source-history.bundle .git/source-history.bundle
export DCL_GLOBAL_BASELINE="$PWD"
export ROBUST_CHECKPOINT_ROOT="$PWD/frozen-checkpoints"
```

Estos overrides cambian rutas, nunca hashes o pesos. El entorno de paquetes
debe coincidir con `environment-freeze.txt`. Las pruebas estadísticas pueden
reproducirse en CPU; no necesitan inicializar una GPU. El compilador opcional
incluido está en `.tools/tectonic`; requiere su distribución TeX/caché.

## Reanudar la ejecución planificada

```bash
$PY scripts/robust-distribution-workflow.py
```

La cola atómica conserva bancos terminados y cobra conservadoramente las
interrupciones. El lock GPU es compartido con los estudios anteriores y no
detiene otros servicios. El watchdog propio termina antes del límite de
doce horas GPU-proceso. No se vuelve a ejecutar `freeze` en un estudio ya
congelado. Una réplica enteramente nueva requiere una copia sin bancos ni
colas y un protocolo prospectivo propio; no debe presentarse una muestra
histórica como una muestra nueva.

## Interpretación y limitaciones

El máximo omnibus incluye siempre información completa característica.
Los testigos explicativos usan las últimas 250 ejecuciones reservadas,
separadas de todos los prefijos que seleccionan eventos o prueban hipótesis.
En grafos pequeños se destinan 20.000 ejecuciones a TV y LRT exacto
condicional; el tamaño primario sigue siendo 500 por generador. La tolerancia
prospectiva de equivalencia es TV<=.05 únicamente en los 105 casos pequeños,
con intervalos simultáneos y ajustes de cutoff/precisión.

La versión 1 y sus fallos se conservan en `stats_v1.py`. La versión 2 mejora
la búsqueda de pares, pero no busca todos los triples. Un triple remoto
con evento de paridad conocido es detectable: si el instrumento lo omite,
es una limitación de sus regiones, no una imposibilidad estadística.
Los eventos raros y soportes desconocidos enormes tienen además límites
fundamentales. No se afirma un test universalmente potente.

Los p-valores no rechazados tras futility son conservadores y conservan una
cota inferior de su valor conceptual completo. El refinamiento por Holm
recalcula cualquier caso que aún pudiera rechazar, de modo que el conjunto
de rechazos coincide con el cálculo completo de las 19.999 permutaciones.
No se modelan paramétricamente las colas. Las familias primaria, auxiliar,
explicativa y de grafos pequeños se corrigen por separado; no se promete un
FWER conjunto de .05 al unir todas sus conclusiones.

En algunos controles extremos, el cálculo de una cola hipergeométrica exacta
de clasificación se subdesborda numéricamente. Los ceros brutos se conservan
identificados; `classifier-safe-pvalues.csv` proporciona cotas superiores
positivas válidas mediante la desigualdad finita de Hoeffding demostrada en
la teoría. El auditor comprueba que todas las decisiones al 5% coinciden.
No afecta a los valores-p Monte Carlo, al omnibus ni a sus decisiones.

Los tests históricos ajenos que ajustan redes durante pruebas de regresión
se excluyen para respetar la instrucción de no entrenar. La cobertura incluye
los módulos modificados, las quince políticas y los adaptadores reutilizados.
