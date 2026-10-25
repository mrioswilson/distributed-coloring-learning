# Propuesta posterior de entrenamiento a mayor grado

**Estado: propuesta; no ejecutada.** Esta campaña usa exclusivamente pesos
preexistentes. Las prioridades finales se ajustan a los resultados congelados,
sin convertir su batería final en conjunto de selección de nuevos modelos.

## Hipótesis y diseños que las distinguen

| Pregunta | Diseño futuro controlado | Resultado que distinguiría hipótesis |
|---|---|---|
| ¿Falta cobertura de nuevos estados? | Misma arquitectura/interfaz, entrenar hasta Δ10; validación intermedia disjunta; reservar Δ20/50/100/200 | Recuperación sin cambiar capacidad apoya cobertura/optimización; no prueba capacidad universal |
| ¿Importa K o el número de vecinos? | Factorial en (grado, tamaño de lista), con listas ≥d(v)+1; mantener K fijo al variar grado y viceversa donde sea válido | Separa densidad de mensajes de nuevas posiciones/longitudes de paleta |
| ¿Sirve CF para COMMIT fuera del grado entrenado? | Pasivo, matched y CF con mismas semillas, inicialización, bancos nativos, minibatches y updates; brazo CF con presupuesto total igualado | Cambios en COMMIT sobre intervenciones nuevas con iguales recursos permiten una atribución más precisa |
| ¿Sufre la selección de propuesta? | Supervisión desglosada P/C/COMMIT y trazas completas; medir disponibilidad, propuestas, colisiones y transición | Distingue muestreo/selección de un detector de conflictos defectuoso |
| ¿Es la representación el cuello? | Sólo tras el contraste sin cambios, comparar un contexto expresivo compatible y anchos controlados | Una mejora al cambiar ancho no separa sola capacidad de optimización; requiere repetición de presupuestos |
| ¿BE necesita otro problema? | Implementar y probar primero una variante parametrizada Linial+KW o BE recursivo; datasets nuevos identificados | No se llamará transferencia del profesor BE_BASE_D3 ni de sus checkpoints originales |

Entrenar hasta Δ10, luego una ampliación independiente hasta Δ20 y finalmente
hasta Δ50 sólo si el escalón anterior deja fallos relevantes. Reservar siempre
grados estrictamente mayores para extrapolación: 20/50/100/200 tras entrenar
hasta10; 50/100/200 tras entrenar hasta20; 100/200 tras entrenar hasta50.
Reservar tamaños y familias además de semillas de grafo. Cinco entrenamientos
por celda y selección prospectiva; ninguna política promedio sustituye réplicas.

## Información legítima y arquitectura

Mantener observaciones locales, paletas propias declaradas, draws/preferencias,
órdenes coherentes JP/GM y fases/calendarios explícitos. CT necesita comunicar
disponibilidad aprendida y preferencias antes de elegir. La información exacta
de disponibilidad, elegibilidad, COMMIT o reparación no puede entrar como una
mejora oculta. Un nuevo parámetro global Δ, si se suministra a la política,
se declara como cambio de interfaz y se abla, no se presenta como anonimato
sin información adicional. BE exige nuevas evaluaciones/decodificaciones
algebraicas y pruebas de cada fase, antes de aprender sus decisiones.

Las seis interfaces extensibles no requieren cambios de pesos para **ejecutar**
K mayor. Esto no garantiza capacidad o entrenamiento suficientes. Evitar
añadir capas por reflejo: primero medir si la política falla en propuesta,
COMMIT, o ambas. Para E1, investigar masas residuales alpha y resolución de
prefijos; para DeepSets posición/1/K y media; para Transformer longitud y
atención. Las reducciones de grafo de las tres siguen siendo MAX; no atribuir
a DeepSets una suma de vecinos que el código no implementa.

## Supervisión contrafactual propuesta

Añadir listas con un solo color disponible, vecinos fijos con colores fuera
de la lista del receptor, ofertas iguales de muchos pendientes, silencios,
preferencias CT casi empatadas, competidores que carecen del color consultado,
órdenes JP extensos y GM con conflictos del mismo slot y reparación tardía.
Recomputar todos los descendientes de cada intervención sobre estados
admisibles. Separar queries, componentes, presentaciones, updates, CPU/GPU y
simulación. Cambiar a la vez cantidad de etiquetas y tipo de supervisión no
demuestra eficiencia muestral causal de CF.

## Coste y decisiones de lanzamiento

No hay medición de coste de entrenamiento a Δ10/20/50 en esta campaña y no se
inventan horas. Usar como línea base los registros históricos por checkpoint
y los costes nuevos **de inferencia**, con magnitudes distintas. Un piloto
futuro explícitamente autorizado mediría adquisición y 100 updates por cada
arquitectura/profesor priorizado. Presupuesto de optimización de un hito:
5 semillas × 3 arquitecturas × profesores seleccionados × updates medidos;
añadir adquisición, development, selección y final 1000 por checkpoint/celda.

El estado por color cuesta O(nK) y los intercambios ingenuos O(mK); la
atención de paleta cuesta O(nK²). GM/Greedy añaden calendarios de O(n) pasos.
La compresión de mensajes MAX y la evaluación de receptores necesarios
reducen costes prácticos, pero no permiten convertir tiempo de inferencia en
tiempo de entrenamiento por un factor supuesto. Se debe publicar una tabla
de costes medidos antes de lanzar cada escalón. No es necesario entrenar
de nuevo profesores/arquitecturas que ya transfieren en las condiciones de
interés editorial.

## Prioridades que ya identifican los pesos congelados

**ColorTrials.** La sonda de arista con paleta local de dos colores mantiene
vecindad e información fijas y reduce el gap normalizado a 1/Δ. Los quince
CF examinados rechazan el ganador desde Δ10 en esa sonda. Priorizar gaps
discretos nuevos, empates, ganador único y competidor sin el color consultado.
Un entrenamiento nativo sólo hasta Δ10 deja sin cubrir gaps 1/20–1/200:
esos grados deben seguir siendo extrapolación real y no etiquetas finales
recicladas. Otra representación de comparaciones de preferencias sería una
ablación explícita de interfaz/arquitectura, no un adaptador retroactivo de
estos pesos. Su origen y la parte del cálculo suministrada deben declararse.

**Johansson/Transformer.** Un checkpoint se bloquea en cliques de grado diez
cuando quedan dos pendientes con disponibles {0,2}. Los dos eligen color2
en los últimos pasos; 8192 draws por receptor reproducen esa concentración.
Priorizar supervisión de la propuesta sobre listas disponibles con huecos,
distintos números de colores y estados tardíos de cliques, variando el draw
privado. Enseñar COMMIT por sí solo no restablece la diversidad que exige
la variante siempre activa. Comparar sin cambios el scorer existente y,
sólo si hace falta, una ablación con contexto de orden/prefijo declarado.

**GM y paletas locales.** Las celdas fallidas deben analizar por separado
propuesta, detección y reparación; el número de pasos programados no permite
esperar indefinidamente a que una mala propuesta se arregle. La escritura
omitida, el conflicto de otro slot y la marca errónea son targets diferentes.
Conservar controles CF/matched/pasivo del mismo hito e inicialización.

La sonda de n8/Δ3 reproduce el mismo error Transformer/s94102 al cambiar
globalK4 por la lista localK2, manteniendo objetivo de reparación, IDs y
vecindad. Por tanto, el primer entrenamiento GM propuesto debe variar listas
**dentro de grado tres**, antes de aumentar grado. Ese control puede distinguir
un problema de paleta de uno de agregación por alto grado con mucho menor
coste. No se ejecutó ningún entrenamiento de esta propuesta.

El frontend histórico de entrenamiento también tiene serialización K4 y
anchura tres. Su futura generalización requerirá una ruta diferenciable con
K/vecindarios variables y pérdidas por fases verificadas; el adaptador de
esta campaña es de inferencia congelada. Igualdad de la función MAX al
comprimir soportes no certifica automáticamente igualdad de todos los
subgradientes y minibatches de un entrenamiento nuevo.

Otra ablación legítima sería supervisar explícitamente disponibilidad como
objetivo auxiliar, con sus consultas por (raíz,color) cobradas y un control
del mismo presupuesto total. Hoy alpha es latente y no se le impone esa
calibración; no debe llamarse fallo de una etiqueta que nunca se entrenó.
La disponibilidad exacta seguiría restringida al entrenamiento/evaluación,
sin filtrado de propuestas durante inferencia.

Para CT también puede probarse una hipótesis más barata que aumentar grado:
entrenar en los mismos estados nativos de grado3 con **aumentos explícitos
de la codificación** de preferencias mediante un mismo mapa afín creciente
para todos los competidores. El conjunto de ganadores estrictos y su menor
color se conservan matemáticamente; se pueden reutilizar esas etiquetas,
contando por separado aumentos/presentaciones y updates. Comprimir y desplazar
los cuatro niveles permite consultar gaps de entrada pequeños sin afirmar
que el profesor nativo emita preferencias continuas. Comparar ese brazo
con aumentar Δ a igual presupuesto ayudaría a distinguir cobertura numérica
ordinal frente a cobertura de vecindarios. Es otra propuesta, no un resultado
ejecutado ni una nueva corrección aplicada al checkpoint congelado.
