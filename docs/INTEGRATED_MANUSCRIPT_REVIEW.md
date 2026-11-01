# Revisión adversarial de la versión integrada

Actualización del 1 de noviembre de 2026: el cuarto paquete ya está recibido
y verificado. Sus 240 JSON originales coinciden con `ROWS.sha256`; las
240.000 filas, los resúmenes y las identidades de 60 checkpoints/cuatro
condiciones coinciden. Se reconstruyeron las identidades de los casos sin
repetir inferencia neuronal. Este pendiente de datos queda cerrado. Recibos
en `results/martin-transfer-rows-20261101/`; el artículo y PDF no cambian.

Actualización editorial, 1 de noviembre de 2026: se retiraron del manuscrito
las notas internas de entrega, hashes de paquetes, conteos de archivos y
estado de auditoría personal. Estas comprobaciones y pendientes permanecen
en los documentos de trabajo. No se modificaron datos ni resultados y no
se ejecutó una nueva campaña experimental. La entrega previa completa está
preservada en `results/paper-editorial-cleanup-20261101/prior-integration.tar.gz`.

Fecha: 31 de octubre de 2026. Revisión crítica realizada por el agente que
integra, con cuatro perspectivas y comprobaciones independientes de los
resúmenes y ejecuciones con los pesos/arrays ahora entregados. **No es una evaluación independiente de
otro investigador o agente.** La auditoría de Claude importada evalúa la
versión anterior de Pedro; no se presenta como aprobación de esta integración.
La revisión externa y el acuerdo de ambos coautores siguen pendientes.

Versión examinada: `paper/ucgnn/main.tex` y PDF de 34 páginas;
SHA256 del PDF `22726ee0bd51cfe775d2eb8eef7ff40e2714b23957cc88774263a7b55018e45c`.
Las comprobaciones y hashes completos quedan en el manifiesto integrado.

## Algoritmos distribuidos

1. **Riesgo: atribuir a la red el algoritmo completo.** La interfaz suministra
   horarios, prioridades/slots, algebra y decodificación. Acción: tabla de
   módulos y contratos preservados, y definiciones F/C relativas al intérprete.
   Resultado: se evalúan módulos locales aprendidos dentro de variantes
   concretas; no se reclama aprender sus complejidades distribuidas generales.
2. **Riesgo: soporte nativo ambiguo o testigo no alcanzable.** La proposición
   separa N completo del banco observado y exige una clase H con dos reglas
   compatibles. Acción sustantiva: prueba explícita y nuevo test del prefijo
   completo con IDs legales (6,3,0,1), desde ambos pasos Linial hasta B7.
   Resultado: existe la consulta activa ocupada admisible donde difieren las
   extensiones. No se extrapola a Luby/Johansson, cuyos nativos tienen conflictos.
   La inclusión de las dos reglas en H es una hipótesis; no se prueba una
   imposibilidad para toda arquitectura neuronal o todo prior restrictivo.
3. **Riesgo: llamar inseguro a cualquier bit COMMIT distinto.** Acción: fórmula
   de transición con permisos, P y decodificador; observador posterior de 2.000
   ejecuciones. Resultado: las 84 aceptaciones contractualmente incorrectas
   observadas decodifican al mismo estado, sin conflictos ni violación de cota.
4. **Riesgo: una omisión local implica siempre fallo.** Acción: prueba KW para
   cota decreciente, condición de sufijo GM y contraejemplo ejecutado de tres
   vértices. Resultado: en KW un valor obligatorio sin escribir queda fuera
   de la siguiente cota; en GM otro extremo posterior puede reparar. Se corrige
   la supuesta necesidad de cada escritura de la cola.
   Se declara además el dominio diagnóstico de estados alcanzados: la misma
   regla KW se aplica con cotas/reloj y coloración auxiliar válidos, sin asumir
   que las decisiones previas fueron las del profesor. Las métricas del banco
   original de prefijos y las de estos rollouts permanecen separadas.
   La revisión final también corrigió una frase heredada del apéndice que
   todavía decía «every scheduled write/mark must be correct» y el cuantificador
   demasiado fuerte de un párrafo sobre raíces inelegibles activas. Las dos
   correcciones se apoyan en el contraejemplo y el observador, respectivamente.
5. **Riesgo: progreso difuso excluye Johansson con P=1.** Acción: desigualdad
   por probabilidad de oferta de cada vecino, demostrada con cuerda logarítmica
   y Jensen, y definición de U_v. Resultado: cubre Johansson uniforme; las redes
   no quedan certificadas y propuestas deterministas coincidentes pueden atascarse.

Pendiente de juicio externo: suficiencia del contrato como objetivo científico
frente a otras extensiones compatibles; no existe una extensión única inferida.

## Aprendizaje automático

1. **Riesgo: CF gana porque recibe negativos ausentes en el control.** Verificado
   en GM y en los controles de adquisición adicionales. Acción: se elimina la
   reivindicación de superioridad general, sin ocultar el resultado primario.
   No se declara resuelta la comparación con negativos informativos: no se hizo.
2. **Riesgo: novedad lógica del test sobreestimada.** Acción: reconstrucción de
   claves desde arrays y reporte del solapamiento de ambos brazos. Resultado:
   separación fina real, pero casi todo el soporte lógico está visto por CF.
   El holdout de cotas de otra rama tampoco prueba novedad del input efectivo.
3. **Riesgo: confundir correlación entre fidelidad y éxito con causalidad.**
   Acción sustantiva: 30.000 replays con sustituciones de componentes y paridad
   de las seis baselines. Resultado: COMMIT solo rescata 1.799 fallos CF y P sola
   los otros 796; C sola no rescata ninguno. Esto localiza causas suficientes
   bajo las intervenciones definidas, en casos consumidos, no una ley general.
4. **Riesgo: explicar las dos ramas por número de actualizaciones.** Acción:
   comparación de protocolos y snapshots a 1.500. Resultado: la ventaja estándar
   CF de Martín ya aparece entonces; la discrepancia no tiene una causa aislada.
5. **Riesgo: ocultar aprendizaje positivo.** Acción: 21/21 en el cuerpo, con
   cohortes heterogéneas y controles, más tabla de 18 combinaciones en transferencia.
   No se transforman pilotos de tamaño/grado extremo en certificaciones.

La limitación inicial de disponibilidad se cierra con los tres paquetes
verificados por SHA256. Los bancos de los 105 entrenamientos permiten repetir
las 220 evaluaciones de hitos/baterías: coinciden 50.173.920 decisiones
binarias y las 466 filas completas de análisis recomputado. Los logits varían
hasta `3.0517578125e-05`, sin cambios de signo. Esto es una réplica de
inferencia efectiva; sigue sin ser otra réplica de entrenamiento o test nuevo.

## Estadística experimental

1. **Riesgo: una media heterogénea es ininterpretable o prueba nulo.** La media
   sigue siendo un estimando definido. Acción: se preserva +5,77 pp y su
   intervalo [−13,73,+25,26], todos los pares y endpoints originales. Se describe
   la heterogeneidad y el intervalo amplio; no hay exclusiones de semillas.
2. **Riesgo: 40.000 consultas equivalen a 40.000 réplicas.** Acción: unidades de
   entrenamiento, tipos repetidos, casos/tapes pareados y tensores compartidos
   explícitos. El resultado por tipo mejora en diez pares; no se usa inferencia
   por consulta ni se inventa un tamaño muestral efectivo.
3. **Riesgo: escoger estratos después del test como principales.** Acción:
   quedan identificados como descripciones posteriores. El nuevo diagnóstico
   fija seis modelos, cinco modos y todos los casos antes de sus resultados,
   pero su selección depende de fallos conocidos: no es una confirmación nueva.
4. **Riesgo: denominadores mezclados.** Acción: campañas de 810.000, 240.000,
   40 modelos y 30.000 intervenciones separados. Los 15.000 replays verifican
   las trayectorias de las celdas con fallos. La auditoría nueva autentica todas
   las 810.000 filas y reconstruye sus coloraciones terminales, sin afirmar
   regeneración neuronal de todas las trayectorias ni de sus flags intermedios.
5. **Riesgo: corregir el conteo cambiando resultados.** Acción: 32/40 BE se
   deriva de los 40 registros; el protocolo original con 34/40 queda preservado.
   Se explican las diferencias de uno/dos objetivos entre presupuestos próximos.
6. **Riesgo nuevo: confundir un inventario completo con todos los datos del
   estudio.** La campaña separada de 240.000 ejecuciones conserva 240 rutas
   crudas ausentes de los tres paquetes. Acción: identificación de cada ruta
   y SHA y registro en los documentos de auditoría de la falta de filas originales.
   Se corrige además nuestro error de descripción: seis brazos y un hito a
   4.000, en vez de dos brazos/dos hitos. La integridad interna de los tres
   paquetes no queda cuestionada. La entrega adicional posterior ya está
   recibida:240 hashes coincidentes y todos sus resúmenes/casos comprobados.

No se identifica una tasa poblacional de rescate ni una ventaja universal
de CF. Sigue abierta la distribución sobre políticas que se querría estimar.

## Novedad y relevancia

Se verificaron precedentes primarios cercanos: CLRS, Hint-ReLIC, SALSA-CLRS,
IIT/DIITO, aprendizaje exacto de instrucciones LOCAL, Discrete NAR, teoría
de extrapolación y dos estudios de brechas/fallos mecánicos. No se reclama
prioridad para redes en coloración, trazas, contrafactuales, aprendizaje
distribuido o separación de fidelidad/éxito. La proposición de soporte no
es un gran teorema nuevo. El incremento defendible es específico:
contratos, soporte, alcance funcional de errores y mecanismos de progreso/
reparación en varios profesores, con replays de componentes verificables.

Valoración del integrador: contribución empírica potencialmente defendible
para TMLR si se mantiene este alcance. La disponibilidad del estudio adicional
ya se verifica en el release. No se predice aceptación. La longitud de 34 páginas
(18 de cuerpo, apéndices y referencias) merece decisión editorial de los
autores; no se amplía teoría para justificar extensión. El cuerpo conserva
la evidencia positiva y concentra los nuevos experimentos en una pregunta.

## Comprobaciones ejecutadas y límites

- 330 pruebas pertinentes aprobadas, más una del prefijo completo legal.
- 160.000 decisiones de continuación re-inferidas: métricas originales idénticas.
- 30.000 ejecuciones de componentes, seis baselines completas idénticas y
  cero errores técnicos/casos nuevos fallidos por las intervenciones.
- 15.000 replays históricos: conteos, casos fallidos, clases y contadores coinciden.
- 2.000 replays de observación: 84 aceptaciones redundantes verificadas.
- Tres paquetes y 1.988 miembros: hashes coincidentes; 220 referencias a
  205 rutas congeladas distintas de 105 entrenamientos, sin confundir controles
  reutilizados o nombres de snapshots con réplicas.
- 985.992 etiquetas crudas y 466 filas de análisis íntegro recomputadas;
  50.173.920 decisiones nuevas de inferencia coincidentes en todas las referencias.
- 810.000 coloraciones históricas reconstruidas contra 4.000 grafos:
  808.413 éxitos totales y 449.791 primarios, sin discrepancias terminales.
- 120.000 ejecuciones autónomas de los estudios inicial/follow-up:
  todos los resúmenes originales coinciden, incluidos 32/40 BE all-pass
  y 5.000/5.000 en los cinco modelos nuevos. Cero errores técnicos; filas
  nuevas completas guardadas. Las filas funcionales originales no se retenían.
- Prueba técnica del wrapper en 960 ejecuciones frescas: compara todas sus
  filas con los primeros cohortes de los resultados completos; no es otro test.
- Los helpers de recuperación se repiten manteniendo hashes de recibos.
  El wrapper nuevo repite efectivamente la reconstrucción terminal de las
  810.000 filas y un checkpoint con 449.280 consultas en destinos nuevos:
  resultados coincidentes, logits de ese chequeo idénticos a nuestro replay
  completo. Un primer lanzamiento sin PYTHONPATH no ejecutó datos; su nota
  técnica y directorio vacío se conservan. No se cuenta como resultado científico.
- pdfLaTeX/BibTeX: 34 páginas, cero citas/referencias indefinidas y cero cajas
  desbordadas; fuentes incrustadas. Se inspeccionaron los tres contactos con
  todas las páginas y a resolución completa las páginas 12, 13, 15, 18, 23
  y 31, incluidos resultados, figuras y las secciones editadas.
  No se detectan cortes o superposiciones. `pdf-visual-review.json` registra
  los hashes del PDF actual y sus imágenes en el directorio de esta corrección editorial.

La repetición técnica opcional adicional de transferencia se interrumpió por
pausas severas de memoria del host, después de un primer bloque coincidente.
Sus parciales y la nota de recursos se conservan; no forman parte del resultado
científico. La reproducción completa de 15.000 casos anterior sí terminó.
No se interpreta el tiempo de ese host como comparación de eficiencia.

La revisión independiente, autores/orden, estilo de envío, alcance final del
follow-up y eventual publicación requieren
acuerdo de los coautores. No se ha enviado, publicado, hecho push ni integrado
ninguna rama compartida.
La entrega original de las 240 celdas del contraste adicional queda cerrada
con el cuarto paquete: archivos originales, metadatos y contadores autenticados.
No es una nueva réplica de inferencia ni sustituye revisión científica externa.
