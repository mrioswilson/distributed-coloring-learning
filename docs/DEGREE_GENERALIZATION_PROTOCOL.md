# Protocolo de transferencia congelada

## Alcance y orden

Worktree `ucgnn-degree-generalization`, rama
`research/multiteacher-degree-generalization`, base `291d4b82`.
No push, entrenamiento nuevo, selección por resultados finales, shield,
filtrado de acciones ni corrección docente en inferencia. Las fuentes y
resultados históricos permanecen intactos. Los archivos nuevos se guardan
en `results/multiteacher-degree-generalization/`.

Primero se auditan profesores e interfaces, se prueban las extensiones y se
autentican los checkpoints. Hay 78 tests nuevos aprobados y una reproducción
nativa adicional de los 195 checkpoints, seis casos por checkpoint. La
batería nativa no reemplaza las certificaciones históricas 5×1000.

## Inventario y comparabilidad

`checkpoint-inventory.json` conserva los 195 hashes, semillas, rutas, método,
cohorte, presupuesto, actualizaciones y registros originales. La cohorte
principal tiene 105 pesos: doce combinaciones históricas pasivas, semillas
83101–83105, y nueve nuevas contrafactuales, 94101–94105. Matched y pasivo
nuevos son controles secundarios. Los 45 pesos BE sólo tienen controles
nativos; ninguna celda BE de alto grado se cuenta como fallo estudiantil.

Las políticas nunca se promedian. Los resultados se conservan por checkpoint,
condición, grafo, cinta y ejecución. Una comparación CF/matched a distintos
hitos no identifica causalmente el efecto de supervisión. Aun con updates
iguales, el banco CF añade etiquetas: no equivale a igual presupuesto total.

## Piloto exploratorio

`pilot/design.json` se escribe antes de los resultados. Incluye:

* Δ=2,3,4,5,10,20,50,100,200 con n=256 y familia regular, manteniendo n fijo.
* n=64,128,256,512,1024 con Δ=3 o 10 regular, manteniendo Δ fijo.
* Diez familias con n64/Δ4/global: regular, aleatoria acotada, bipartita,
  árbol, irregular, hubs, desconectada, aislados, heterogénea y cliques.
* Listas locales heterogéneas con n64/Δ4 y estrellas/aislados con n256/Δ50,200.

Las noventa políticas principales extensibles recorren toda esa batería;
los sesenta controles nuevos extensibles usan las condiciones indicadas en
`degree/run.py`. Una ejecución por punto regular o dos/diez en los estratos
indicados es **exploratoria**, insuficiente para certificar transferencia.
No se llama 1000 ejecuciones a la suma de grados, tamaños o semillas.

## Generación y aleatoriedad

`degree/graphs.py` genera grafos con **grado máximo exactamente Δ** y verifica
n, simplicidad y máximo tras cada generación. No se rellenan celdas
imposibles Δ≥n. Los regulares requieren suma par. Bipartitos densos pueden
usar una estrella cuando Δ>n/2: se identifica la familia, sin llamarla
bipartita regular. La aleatoria acotada usa inserción con caps y un testigo
de grado, **no es G(n,p)**. Los árboles son conexos y los casos de aislados
contienen una estrella de grado exacto. Cliques fuerzan saturación de paleta;
IDs persistentes crean órdenes, slots y reparación nativos. Los casos
adversariales docentes y los extremos de información se amplían en sondas
separadas, sin contarlos como ejecuciones confirmatorias ordinarias.

La semilla de grafo depende de namespace, n, Δ, familia e índice, no de
arquitectura, checkpoint o profesor. Cada réplica usa otra cinta, derivada
además de su índice 0–4. Dentro de una comparación profesor/estudiante o de
tratamientos se acoplan las entradas privadas para localizar divergencias;
entre ejecuciones los identificadores de cinta son distintos. La PRNG de
preferencias y la asignación de IDs usan namespaces separados, como en la
implementación histórica. La misma cinta no implica independencia entre
políticas comparadas. Se conservan huellas y aristas para reproducir grafos.

## Validación exacta y ejecución

Cada cinta docente se ejecuta y valida **antes** de que el estudiante acceda
a su batería. Un fallo del profesor invalida la celda; no se reemplaza el
grafo ni se atribuye al estudiante. Los seis profesores tienen prueba de
seguridad para paletas globales y locales y tests diferenciales escalares.
BE no se redefine. Se mantienen cuatro bloques en GM, el orden JP coherente
y los permisos de Greedy. Los horizontes están en `degree/teachers.py` y
en cada fila. El coste del calendario serial se cuenta en pasos nativos;
no se equipara a rondas paralelas anónimas.

La inferencia usa float32 CPU, un hilo Torch por worker y hasta cuatro
procesos por campaña. Las sondas adicionales se ejecutan en otra campaña
de cuatro workers; durante el solapamiento se observaron hasta ocho workers
de inferencia/simulación en una máquina de dieciséis CPUs lógicas.
Agrupar componentes desconectadas para inferencia no crea aristas
entre grafos ni mezcla decisiones. Se conserva el orden de casos. Todos los
pesos se verifican por hash antes de cargar. Se guardan resultados aunque
haya conflicto, palette violation, cutoff o falta de progreso.

## Métricas y trazas

Por ejecución: éxito funcional, conflictos finales, número pendiente,
violaciones de paleta/invariantes, cutoff, pasos/horizonte, colores finales,
primera discrepancia de componente, fase, fidelidad de acciones y de
transiciones, aciertos/denominadores por disponibilidad, participación,
propuesta y COMMIT sobre las propuestas efectivas. La disponibilidad se
diagnostica mediante umbral alpha>.5; eso no se usa para decidir ni corregir
la política. Una propuesta diferente puede ser segura y no es por sí sola
un fracaso funcional. Fidelidad marginal no demuestra igualdad algorítmica.

Se guarda la primera traza de un fallo por checkpoint/condición, con estado,
grafo, ID, slots, preferencias, draws, decisiones, logits y comparación
docente. El primer componente discrepante es un diagnóstico temporal, no
una demostración de causa suficiente del fallo final. El profesor se
consulta sólo después de las decisiones para medirlas.

## Confirmación

La selección de condiciones y costes usa exclusivamente el piloto. Antes
de cualquier dato confirmatorio se escriben `confirmatory/design.json`,
hashes de fuentes, checkpoints, namespaces y criterio. Los detalles
efectivamente congelados están en ese archivo y en la enmienda fechada al
final de este documento. Cada condición principal elegida tendrá 1000
ejecuciones por checkpoint; el resto se etiqueta exploratorio. Ningún
resultado piloto se promueve retrospectivamente a confirmatorio.

Se informan intervalos Clopper–Pearson puntuales bilaterales del 95% por
checkpoint/condición con numerador y denominador. Con 1000/1000, el intervalo
bilateral tiene extremo inferior ≈.99632; la cota unilateral histórica
.99700875 es otra convención. No hay afirmación simultánea sobre todas las
celdas ni corrección universal. Una mezcla estratificada de familias tiene
por estimando la mezcla declarada; no da 1000 casos por familia.

El índice y los resúmenes son derivaciones de las filas crudas. Reanudar una
ejecución completada sólo verifica y reutiliza su hash; no la sustituye.
Una interrupción o cambio de código se registra como enmienda con la
evidencia previa preservada, sin cambiar silenciosamente el diseño.

## Reproducción

```sh
export PYTHONPATH=src
python -m dcl.degree.inventory
python -m pytest tests/test_degree.py -q
python scripts/validate-degree-native.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m dcl.degree.run pilot --workers 4
# Sólo después de escribir/congelar el diseño confirmatorio:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m dcl.degree.run confirmatory --workers 4
```

Las copias `.pt` son artefactos locales ignorados por Git. Sus rutas fuente,
SHA-256 y paquetes históricos de recuperación están en el inventario; no
se presenta un checkout sin esos pesos como reproducción autocontenida.

## Enmienda exploratoria y congelación confirmatoria

El piloto inicial conservó dos éxitos CT/Δ50 de más de un minuto antes de
interrumpir cuatro workers que seguían en celdas densas CT. No había finales
confirmatorios consumidos. `pilot/amendment-v2.json` conserva fuentes antes
y después, celdas completadas y motivo. Sólo las celdas CT/Δ≥50 aún no
escritas recibieron un techo de diez segundos por ejecución. Se conservaron
todos los resultados previos y sus hashes. Por ello el piloto denso CT
**no tiene un horizonte de recursos homogéneo entre todas las semillas**;
no permite ordenar sus tasas como si lo tuviera. Los 42 `resource_cutoff`
se distinguen de las dos colisiones reales de un control matched CT.
Una discrepancia de acciones al primer paso sí es evidencia finita de falta
de fidelidad, independientemente del tiempo disponible para terminar.

El diseño final se congeló, tras cerrar el piloto, en
`confirmatory/design.json`, mediante `scripts/freeze-degree-confirmation.py`:

* 180 checkpoints extensibles: noventa principales, sesenta controles
  certificados y treinta controles CT preexistentes a 4000 updates.
* n64/global con Δ3/4/10 y n64/local con Δ4: mil ejecuciones por cada uno.
* n256/global/Δ4: mil ejecuciones por cada checkpoint **principal**.
* Total previsto: 810000 ejecuciones estudiantiles y 150000 exactas previas.
* Familias elegidas de manera independiente y uniforme por índice, con un
  namespace de familia separado. Se conservan sus denominadores realizados.
* Ningún techo de tiempo en confirmación: se aplican los horizontes de pasos
  definidos antes de los datos. No hay sustitución de instancias ni selección
  de pesos posterior a los resultados.

Los treinta controles extra fueron elegidos por una regla previa explícita:
usar **todos** los pasivos/matched CT a 4000 updates disponibles de las cinco
semillas, con la misma inicialización, ejemplos nativos y minibatches del CF.
No se les atribuye certificación histórica. Su control nativo suplementario
conserva una colisión matched/E1/s94104 en K4; no se retira esa política.
La comparación de transferencia debe controlar ese desempeño basal.

Las 93 pruebas finales del módulo de grado y las 361 regresiones de
profesores/modelos/contrafactuales/paletas quedan archivadas por separado.
Las consultas contrafactuales de COMMIT fuerzan propuestas/query sólo dentro
de sus sondas, sobre prefijos docentes seguros: jamás dentro de los rollouts
autónomos. Se registran como diagnósticos exploratorios, no como finales
funcionales ni nuevos entrenamientos.

La ampliación final de tests termina con **388 comprobaciones** aprobadas,
incluido contraste escalar de los seis profesores hasta grado200 y los
invariantes de especulación/detección exactos. El registro de 119 tests de
grado es un subconjunto de esas388, no se suma para inflar su número.
Las regresiones históricas incluyen optimizaciones de prueba en directorios
temporales para verificar gradientes y reanudación. No generan políticas de
investigación ni se usan en resultados, selección o propuesta de entrenamiento;
ningún checkpoint evaluado se actualiza. Los contadores de updates cero de
las campañas se refieren a sus experimentos científicos de pesos congelados,
no a esos tests técnicos desechables.

## Extensión exploratoria separada de tamaño

`larger-size-pilot/design.json` se congela después de cerrar la confirmación,
antes de sus propias salidas. Incluye los sesenta checkpoints principales de
Luby/Johansson/ColorTrials/JP en regulares n2048/4096, Δ3/10, una ejecución por
punto. Sus ochenta referencias y240 estudiantes se guardan por separado.
No altera el diseño ni el denominador confirmatorio. GM/Greedy mantienen
el barrido hasta1024; no se confunden sus calendarios seriales con rondas
LOCAL paralelas ni se necesita ampliar esos tamaños para el contraste de grado.
