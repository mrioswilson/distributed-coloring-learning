# Revisión de validación y continuación hasta E

**Cierre: 17 de octubre de 2026, hora de Chile.** Terminaron los 565 entrenamientos,
las 931.700 evaluaciones de 1.210 vistas y los 64 grupos de informes. La prueba
final cerró a las 22:43; el coordinador completó los informes a las 22:55. El
[análisis final](../results/m6study/2026-10-17-final/ANALISIS.md) y el
[manifiesto de cierre](../results/m6study/2026-10-17-final/final-report-state.json)
conservan los resultados y la identidad del protocolo congelado. El texto
siguiente documenta las decisiones anteriores a esa prueba.

El 15 de octubre de 2026 a las 14:15:32, hora de Santiago, terminaron los 340
entrenamientos únicos del bloque inicial A/B/referencias/C/D. No hubo fallos en
`data/m6study-release/`. El intento técnico antiguo permanece preservado en
`data/m6study/` y no cuenta entre las réplicas científicas.

La revisión reproducible está en
`data/m6study-release/reports/validation-review.json`. Contiene hashes de los
historiales, todos los resultados seleccionados y la auditoría de los 80 pares
principales. No se abrió ningún grafo del test final para tomar estas decisiones.

## Resultados de selección, no prueba final

Cada modelo se seleccionó con el criterio de validación ya definido: éxito,
menos commits inseguros por estado, menos rondas, accuracy de acciones y pérdida.
Cada candidato se observa en 16 grafos, REGULAR/IRREGULAR y N=40/160. Hay sesgo de
selección al elegir entre épocas; estas tasas no estiman la fiabilidad final.

En A, las 20 réplicas por grado alcanzaron cada una 16/16 éxitos y cero commits
inseguros. En los controles Own-only seleccionados, los totales descriptivos
por grado fueron 320/320, 320/320, 319/320 y 305/320 para Δ=3,5,8,10. Los dos
últimos grupos registraron uno y 186 commits inseguros, respectivamente. Los
denominadores reúnen evaluaciones de modelos distintos; no se presentan como
una prueba binomial independiente de un solo modelo. MIXED también alcanza
16/16 por réplica, sin que este conjunto pequeño distinga todavía su ventaja.

A época 120 se verificaron en los 80 pares A/B la igualdad exacta de datos,
inicialización, actualizaciones, exposiciones y pesos de P/C/message1. Seleccionar
épocas diferentes puede romper ese emparejamiento efectivo. Una réplica principal
de Δ=3 seleccionó época 10, antes de recibir contrafactuales; se conserva ese
resultado y se evalúa además la época 120 para todos los modelos.

## Ampliación de presupuestos prevista en la solicitud original

La solicitud pide ampliar configuraciones relevantes de cinco a veinte semillas
y otros grados. Se comparó la media entre las cinco semillas de descubrimiento
del mismo criterio lexicográfico de validación. Se eligió una configuración por
comparación: épocas fijas, actualizaciones fijas y reparto de 2.400 trayectorias.
Los empates exactos prefieren menos grafos y después menos tapes.

| Comparación | Grafos | Tapes | Actualizaciones por época | Semillas por grado |
|---|---:|---:|---:|---:|
| Épocas fijas | 200 | 3 | Las necesarias para recorrer los datos | 20 |
| Actualizaciones fijas | 800 | 3 | 256 | 20 |
| 2.400 trayectorias, épocas fijas | 2.400 | 1 | Las necesarias para recorrer los datos | 20 |

Cada fila se extiende a Δ=3,5,8,10. Son 240 configuraciones, 15 ya completas, por
lo que se agregan **225 entrenamientos únicos**. El total previsto pasa de 340
a **565**. El plan es `configs/m6study/phase-c-extension.json`. No se amplía el
producto completo de nueve presupuestos por cuatro grados. La selección es
exploratoria: los éxitos de validación están saturados, las diferencias de rondas
son pequeñas y no prueban una clasificación de robustez en grafos grandes.

## Congelación y ejecución automática de E

`experiments/finish_m6study.py` encadena:

1. La ampliación de presupuestos; salta las 15 configuraciones completas.
2. Verificación de todas las épocas, checkpoints, historiales revisados y pares.
3. Congelación inmutable de modelos y protocolo.
4. Evaluación final closed-loop, sin shield, con persistencia por grafo.
5. Tablas y figuras científicas desde los resultados originales.

La revisión inicial ya está realizada; este coordinador no requiere otra
confirmación para iniciar E. Un fallo o una detención explícita frena la cadena.
No se vuelve a seleccionar presupuestos a partir de las réplicas ampliadas o
del test final. Las constantes del profesor, pérdidas y probabilidades
contrafactuales conservan los valores del protocolo principal.

Los **105 tests** pasan, incluidos bloqueo de E ante entrenamientos o auditorías
fallidas, detención/reanudación, congelación, conservación de denominadores
emparejados y regeneración reanudable de informes por cohortes. Las comprobaciones
técnicas de evaluación usan el namespace 998877, separado del test científico.

El plan de modelos es `configs/m6study/phase-e-cohort.json`. Se congelan las vistas
`best` y `epoch-120` de los 565 modelos. La comparación de mecanismo entre
Counterfactual y Own-only usa la época 120 común; `best` conserva el análisis de
selección por validación. Para la curva de épocas se añaden las épocas 20/40/60/80
de las 20 réplicas principales Counterfactual Δ=3. Son **1.210 vistas de checkpoints**.

El test conserva los seis tamaños y cantidades ya establecidos: por familia,
100/100/100/50/25/10 grafos para N=40/160/640/2560/10240/40960. Incluye REGULAR e
IRREGULAR, máximo 256 rondas y namespace 20261014. Usa los mismos grafos/tapes
por grado para todos los modelos. Son **931.700 evaluaciones de grafos**, muchas
correlacionadas porque comparten grafos o pesos. No son 931.700 grafos
independientes ni garantizan corrección universal.

Los informes se procesan por grado/presupuesto/política para limitar RAM. Cada
grupo conserva juntos los métodos y familias de entrenamiento necesarios para
McNemar y las comparaciones REGULAR/MIXED. Las tablas se concatenan en streaming;
no se cargan todas las observaciones y desgloses por k simultáneamente.

## Operación

Desde `/workspaces/AlgoLearning/distributed-coloring-learning`:

```bash
.venv/bin/python experiments/finish_m6study.py --dry-run
tmux new-session -d -s m6study '.venv/bin/python -u experiments/finish_m6study.py > data/m6study-release/logs/finalization-first.log 2>&1'
cat data/m6study-release/finalization-state.json
experiments/m6study status --root data/m6study-release
experiments/m6study stop --root data/m6study-release
```

Después de que termine la sesión detenida:

```bash
tmux new-session -d -s m6study '.venv/bin/python -u experiments/finish_m6study.py --resume > data/m6study-release/logs/finalization-resume.log 2>&1'
```

`--resume` retira STOP explícitamente y verifica los hashes de revisión y planes.
La cola reutiliza los bloqueos existentes; no permite dos coordinadores ni dos
procesos de entrenamiento/evaluación GPU. Un entrenamiento se detiene al terminar
una época; una evaluación se detiene entre grafos. La prueba congelada se conserva
en `frozen-final-test.json` y los resultados en `evaluations/<id>/<modelo>/`.
`finalization-state.json` registra la etapa vigente; `continuation-state.json`
conserva el cierre del bloque inicial.
