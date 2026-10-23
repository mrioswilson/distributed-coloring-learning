# Cómo leer las métricas del seguimiento E1

El [protocolo](MULTITEACHER_E1.md) fija las arquitecturas, datos y presupuesto.
Esta nota aclara los nombres de los contadores; no cambia entrenamiento,
selección de checkpoints ni evaluación.

| Métrica | Qué cuenta |
|---|---|
| Éxito final | Todos los vértices tienen un color0–3 y no hay aristas monocromáticas, comprobado sobre la coloración completa |
| `action_accuracy` | Acuerdo de la decisión nativa completa en estados del maestro: WAIT/color, par auxiliar o marca según la fase |
| `participation_correct` | Acuerdo de PARTICIPATE, antes de combinarlo con CANDIDATE y COMMIT |
| `candidate_correct` | Acuerdo del candidato, incluso si el maestro no participa y esa propuesta no se utiliza |
| `own_commit_correct` | Acuerdo de COMMIT con la regla nativa aplicada a las propuestas efectivamente producidas por la red |
| `unsafe_own_commits` | COMMIT positivo cuando la etiqueta nativa de esa consulta es negativa; el nombre heredado requiere la precisión de abajo |
| F1 de marcas GM | Recuperación de las marcas positivas, para que los muchos negativos no oculten errores de reparación |
| Exactitud por fase | Evita mezclar los puntos de Linial, WAIT/opciones KW, asignación, detección y reparación |
| Rango entre semillas | Mínimo/máximo de cinco inicializaciones sobre los mismos datos; no es un intervalo de confianza |

En `component-learning.csv`, los contadores de componentes se entregan como
fracciones de `component_decisions`. Esa cantidad excluye Linial y DETECT,
que tienen otras decisiones supervisadas. El acuerdo de acción sí incluye
todas las fases correspondientes a cada maestro.

## Qué significa una confirmación incorrecta

`unsafe_own_commits` significa **violación de la regla de confirmación nativa**.
No tiene una interpretación idéntica entre los siete maestros:

- Luby/Johansson/JP: conflicto con un color fijado o con una propuesta vecina
  participante del mismo color, o confirmación sin participación.
- ColorTrials: además de los colores fijados, la consulta debe ganar su
  comparación estricta de preferencias con todos los competidores nativos.
  No ganar no implica necesariamente que esa elección causaría una colisión
  entre las propuestas concretas de la red.
- GM-asignación/reparación: se consultan los colores ya fijados. Las colisiones
  entre asignaciones simultáneas se permiten en la fase especulativa y se
  tratan con detección/reparación; no son una violación de esta etiqueta.
- Greedy: hay un único vértice escribiendo y la regla consulta sus vecinos
  fijados. KW: la consulta se refiere a una opción auxiliar y se compara con
  los auxiliares actuales de los vecinos.

En todos los casos se exige participación para una confirmación positiva.
Los estados/propuestas de los diagnósticos locales no se aplican como filtros
durante el rollout aprendido. Los choques, auxiliares fuera de paleta y falta
de terminación se conservan y se reflejan en el éxito final.

## Acuerdo alto y aprendizaje efectivo

Una decisión WAIT frecuente o una marca casi siempre negativa puede producir
exactitud alta sin aprender las decisiones difíciles. Los
[baselines constantes por fase del estudio original](../results/multiteacher/2026-10-20/teacher-task-baselines.csv)
siguen siendo los mismos: se reutiliza exactamente su pool de validación.
Conviene leer la exactitud con esos baselines, las fases/componentes y el
rollout autónomo completo.

Los cuatro rollouts de seguimiento usan N40/N160. Son útiles para seguir
entrenamiento; no sustituyen el test hasta N10240. Alcanzar20/20 seguimientos
entre cinco semillas no convierte esos casos compartidos en20 grafos nuevos.
La tabla global del test combina tamaños con distinta cantidad de casos;
para estudiar transferencia se leen familia, tamaño y semilla por separado.

BE-Linial incluye el cálculo local suministrado `p_v(x)` tras predecir `x`.
Por eso su exactitud de par auxiliar no es una ablación que cambie únicamente
la arquitectura de la antigua cabeza de169 clases. Las diferencias entre
GNN anterior y E1/adaptaciones se interpretan como diferencias del conjunto
de arquitectura, representación y entrenamiento.

`paired-cost.csv` compara pasos nativos de cada GNN exitosa con el profesor
exacto en esos mismos casos. Excluye fallos de ambos promedios para no atribuir
menor coste a una ejecución que terminó antes por error. Un grupo sin éxitos
deja el coste vacío. En GM se resta el calendario fijo de asignación/detección
para contar nodos enviados a reparación. Son comparaciones dentro de cada
maestro:un paso secuencial no equivale a una ronda LOCAL de otro algoritmo ni
a tiempo de pared del entrenamiento.
