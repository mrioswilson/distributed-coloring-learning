# Análisis de fallos e información necesaria

La unidad funcional es una ejecución autónoma completa. La unidad de
replicación del aprendizaje es una semilla de entrenamiento. Las decisiones
locales correlacionadas de una trayectoria no se cuentan como réplicas
independientes. Los resultados cuantitativos nuevos y sus denominadores se
recogen en `MULTITEACHER_COUNTERFACTUAL_RESULTS.md`.

## Taxonomía operativa

| Error | Observación necesaria | Interpretación permitida |
|---|---|---|
| Participación incorrecta | P contra el objetivo nativo en el mismo estado | Error de componente; puede ser seguro o afectar progreso |
| Candidato incorrecto | C contra selección nativa, incluso al esperar si el componente está definido | Baja fidelidad; no implica colisión |
| COMMIT inseguro | Confirmación positiva de una consulta cuya continuación docente es negativa | Violación de regla en ese contexto; distinguir de una colisión física aplicada |
| COMMIT omitido | Negativo donde continuación permite confirmar | Puede causar retraso o no terminación; crítico en calendarios de una sola oportunidad |
| Marca GM incorrecta | DETECT sobre historia especulativa legal y colores efectivos | Puede omitir reparación o reparar innecesariamente |
| Linial incorrecto | Punto/valor, propiedad y cota auxiliar después de aplicar | Se termina en el primer fallo del invariante, sin corregir |
| KW incorrecto | Elegibilidad, opción y siguiente color auxiliar por bound | Un mal write y un WAIT pueden producir estados diferentes o la misma transición redundante |
| Error técnico | Excepción/NaN/archivo faltante/caso duplicado | Fallo de gate; nunca éxito ni caso omitido |
| Corte/no progreso | Horizonte prospectivo, vértices pendientes y permisos | Fallo funcional aunque los colores ya fijados sean propios |

El oráculo de continuación es explícito: no siempre equivale al predicado
«cualquier coloración resultante es propia». En ColorTrials exige preferencia
ganadora estricta. JP exige elegibilidad por ID. KW exige la fase y que el
color auxiliar sea alto. GM acepta los conflictos temporales que permite su
fase de asignación, y obtiene marcas y reparación de la historia real.

## Discordancia de etiquetas en la referencia pasiva

`samplecomplexity.training.passive_loss` ejecuta `out=model(...)`, con
propuestas duras aprendidas, y supervisa `out.commit` con la confirmación
almacenada del profesor. Cuando las propuestas difieren, la etiqueta ya no
responde a la consulta efectiva de continuación. Este hecho de código es
verificable; **no demuestra por sí solo** que explique los nueve paquetes no
certificados. La participación y el candidato mantienen objetivos docentes
bien definidos. La política completa todavía puede aprender una imitación
funcional con esa pérdida, como muestran los doce PASS originales.

Los controles nuevos separan `passive`, `matched` y `counterfactual` con la
misma inicialización y estados. La comparación debe especificar que cambiar
de `matched` a `counterfactual` adquiere etiquetas adicionales y modifica la
distribución de consultas y gradientes. No se presenta como efecto puro del
número de datos ni como eficiencia muestral a igual presupuesto total.

## BE y optimización

El resultado anterior a 1500 updates no se incorpora retroactivamente al
grid principal de 500. La nueva repetición numérica usa las mismas etiquetas,
inicialización y posiciones de minibatch, y verifica identidad de tensores
finales en las cinco semillas E1. Su extensión a DeepSets y Transformer no
hereda automáticamente ese resultado. Las curvas por fase y bound son
necesarias: un promedio dominado por fases fáciles puede ocultar un error KW
tardío que invalida una ejecución completa.

Los componentes algebraicos suministrados permanecen explícitos. Aprender a
seleccionar un punto a partir de evaluaciones/equalities no demuestra aprender
aritmética de cuerpos finitos desde IDs crudos. Un fallo con pocas actualizaciones
no prueba imposibilidad representacional. Tampoco garantiza monotonicidad de
la generalización al prolongar Adam.

## GM y vértices aislados

La evidencia original de semilla 83105 localiza omisiones COMMIT en ASSIGN
de aislados. Un nodo puede participar y proponer color 0, pero recibir un
logit COMMIT negativo. Como el calendario no le proporciona otra oportunidad
de asignación y la detección no marca ningún vecino inexistente, termina
pendiente. El caso singleton reproduce el mecanismo. No hay razón para
repararlo con una regla exacta nueva en inferencia.

La nueva cobertura de grafos sin aristas y componentes mixtas proporciona
ejemplos positivos y negativos de continuación en esos contextos. Las
regresiones históricas se conservan separadas del final nuevo. Que la
distribución ampliada resuelva una regresión no identifica por sí solo el
efecto causal de aislados: también deben controlarse actualizaciones,
tratamiento y el resto de la distribución. Las ablaciones de cobertura se
informan como cambios del procedimiento de muestreo realmente realizado.

## ColorTrials y los dos intercambios

Para cada c, la disponibilidad de un vecino depende de los colores de sus
propios vecinos; por tanto, la raíz puede requerir información de dos saltos.
El primer intercambio aprende disponibilidad y el segundo transmite
preferencias/disponibilidad aprendida. El candidato se lee después del
segundo intercambio. No se etiqueta un candidato mediante preferencias
originales si se han modificado las preferencias efectivamente transmitidas.

Diagnosticar por separado disponibilidad implícita, P, C, confirmación propia
y confirmación contrafactual. La disponibilidad es latente: un valor alpha
no debe llamarse probabilidad calibrada de estar libre sin comprobarlo. La
fidelidad nativa perfecta en un banco no garantiza respuesta correcta ante
otra propuesta ni una trayectoria autónoma universalmente correcta.

## Argumentos matemáticos de información (P12)

**Lema de indistinguibilidad.** Sean z(S) las observaciones suministradas a una
política determinista f. Si hay dos estados admisibles S y S' con z(S)=z(S')
pero decisión docente d(S) distinta de d(S'), no existe f que reproduzca
ambas. Si las etiquetas son binarias y las dos situaciones equiprobables,
la precisión máxima sobre ese par sin información adicional es 1/2. El lema
se refiere a la observación y tarea declaradas, no al tamaño de una red.

**JP sin orden.** Considérese una arista de dos vértices pendientes, con
idénticas observaciones anónimas y cintas nulas. Intercambiar dos IDs distintos
mantiene todas las observaciones sin IDs pero cambia cuál extremo es máximo.
El profesor decide COMMIT en extremos opuestos. La regla exacta de JP no es
función de ese estado anónimo. En cambio, en cualquier estado parcial,
`sign(ID_u-ID_v)` en cada arista basta para determinar si v tiene vecino
pendiente mayor; los valores absolutos y distancias numéricas no son
necesarios para esa decisión. Ello no prueba que el optimizador aprenda la
función ni que baste cualquier codificación anónima enriquecida.

**ColorTrials y radio.** Sean v-u-w un camino y un color c libre en v.
Mantener el estado propio de v y u y el vecindario observado por v, con u pendiente y preferencia mayor
que v para c. Cambiar sólo el color fijo de w entre c y otro color cambia
si c está disponible para u y, por tanto, si u compite contra v. La raíz no
puede decidir siempre ese ganador con una observación que omita la
disponibilidad del vecino y toda información equivalente de segundo salto.
Dos intercambios con disponibilidad y preferencias por color bastan para la
regla implementada. No se deduce una cota inferior para protocolos con otro
estado persistente o información inicial.

**Luby/Johansson.** Los colores fijos de vecinos determinan la paleta. Las
ofertas y participaciones conjuntas de cada vecino determinan conflictos.
Separar MAX de «participa» y MAX de «ofrece c» puede perder la conjunción
del mismo vecino. Transformar conjuntamente antes de agregar conserva la
información necesaria. La cinta privada es necesaria para imitar una
realización aleatoria dada, aunque no para cualquier política que coloree.

**BE.** Una coloración auxiliar propia y las igualdades polinómicas por punto
permiten identificar un punto no bloqueado. El plan de Linial cumple
q>3d para el grado polinómico d; cada vecino prohíbe a lo sumo d puntos,
dejando uno libre. KW asigna bloques disjuntos de cuatro opciones a colores
altos; tres vecinos bloquean a lo sumo tres opciones. Estos argumentos
justifican la existencia de elecciones legales en el dominio declarado;
no certifican las elecciones de una red. Sin ruptura de simetría inicial,
una red determinista anónima sobre una arista simétrica con entradas iguales
no puede fabricar colores finales distintos de manera simultánea.

**GM.** Dados colores reales, igualdad de slot y comparación de IDs, el mark
de detección es local. El particionado en bloques y la cola ordenada requieren
el preprocesamiento global suministrado. No se afirma que una red anónima
reconstruya ese calendario. La consulta docente sobre una historia inventada
sin esa relación no queda justificada por el mero hecho de procesar arrays.

**Greedy.** Dado el permiso exclusivo de escritura, el primer color libre es
función de los colores vecinos; el ID numérico no es necesario para la
elección local. Los IDs sí construyen el orden del intérprete. Quitar IDs
después de suministrar permisos no equivale a eliminar el orden global.

Estos resultados son suficiencias y obstrucciones para observaciones
especificadas. No son garantías universales de entrenamiento, límites
muestrales de las tres redes ni demostraciones de imposibilidad de aprender
los profesores que antes figuraban NOT CERTIFIED.

**Cuantización histórica.** En M6 primitivo el profesor genera uniforms
float64 y `encode_m4_state` los convierte a float32. Los valores
\(1/2-2^{-28}\) y \(1/2+2^{-28}\) se codifican ambos como float32 1/2,
pero su etiqueta de participación float64 difiere. Por indistinguibilidad,
ese encoder no permite fidelidad de participación universal a la ley
float64 original. El test constructivo queda en `test_multicf.py`. Ello no
invalida los éxitos funcionales históricos. En el protocolo nativo actual,
`snapshot` genera directamente float32 y el profesor consulta exactamente
esas entradas; no se hereda esa obstrucción numérica. La corrección universal
de una política entrenada sigue sin demostrarse.

## Generalización y leyes de salida

PASS sólo valida cinco políticas y la distribución finita estratificada de
sus baterías. La cota unilateral de cero fallos sobre 1000 ejecuciones por
política es .99700875 para el promedio ponderado por la asignación de estratos,
bajo independencia de ejecuciones; no es una garantía simultánea sobre todos
los paquetes ni sobre cualquier grafo. Los grafos de grado mayor y las paletas
locales del trabajo complementario no pertenecen a esta certificación.

Igualdad de acciones implica igualdad de transición en un estado común,
pero distintas acciones pueden producir la misma transición redundante.
Éxito funcional permite otras trayectorias, tiempos y leyes terminales. Las
comparaciones de leyes se declaran con su observable, referencia, tamaño de
banco y límites; no rechazo no se presenta como prueba de igualdad.
