# Revisión de cierre científico del artículo

## Recomendación

**Alternativa 2, con alcance acotado:** incorporar una sección de transferencia
congelada y sus resultados negativos/mecanísticos, conservando el resultado
nativo heterogéneo 21/21. No ampliar esa frase a siete profesores en grado
arbitrario. Seis variantes admiten extensión; BE sigue siendo un caso base.
La pregunta sobre qué regla se aprendió une las pruebas informacionales,
supervisión CF, éxito funcional y contraejemplos de fidelidad. El estudio
no necesita implementar BE recursivo ni entrenar en alto grado para cerrar
esta contribución finita.

| Alternativa | Solidez y unidad | Coste pendiente y riesgo | Decisión |
|---|---|---|---|
| 1. Sólo dominio nativo | 21 combinaciones con evidencia trazable, sin protocolo homogéneo | Menor alcance; dejaría fuera transferencia ya terminada y un contraejemplo explicativo | Defendible si el espacio editorial obliga; conservar extensión como suplemento |
| 2. Añadir transferencia acotada | Pruebas de seis variantes, controles nuevos comunes, pesos congelados y diagnósticos de fidelidad | Explicar adaptador, paletas y cutoffs; no vender el piloto Δ200 como certificación | Recomendada para la versión actual |
| 3. Dos artículos | Otra pregunta más amplia puede separar efectos de Δ, K, familias, entrenamiento y certificación | Requiere grados entrenados nuevos, presupuestos CF igualados y quizá nueva variante BE | Reservar esa investigación, no frenar el artículo actual |

La novedad no se deduce de que el benchmark sea mayor ni se afirma prioridad
bibliográfica exhaustiva. La aportación concreta es separar validez del
profesor, compatibilidad de pesos, funcionalidad y fidelidad mediante
controles matemáticos y experimentales identificados.

## Contribuciones y nivel de evidencia

| Línea | Afirmación defendible | Nivel y límite |
|---|---|---|
| A. Heterogeneidad | 21/21 combinaciones funcionales en sus baterías nativas | Empírico histórico+nuevo; cinco pesos y mil ejecuciones por peso; protocolos diferentes |
| B. Contrafactual | Mejores respuestas a intervenciones en comparaciones apareadas; no igualdad universal del profesor | Empírico; presupuesto total de etiquetas difiere, y persisten errores |
| C. Información | Contraejemplos anónimos para prioridades, necesidad de auxiliares y segundo mensaje CT | Argumentos de indistinguibilidad y contratos de interfaz; no teorema de capacidad de todo ancho |
| D. Muestras/optimización | Consultas, presentaciones y updates se separan; más updates no mejora siempre | Evidencia de curvas y regresiones; no cota universal de complejidad muestral |
| E. Transferencia | Seis profesores parametrizados y ejecución de sus pesos originales fuera de grado tres | Pruebas condicionales docentes y resultados finitos; Δ cambia K en régimen global |
| F. Leyes globales | Éxito funcional y algunas fidelidades locales no igualan distribuciones de coloraciones completas | Se conserva evidencia estadística histórica; aquí no se rehace ni se amplía su objetivo principal |

## Consolidación única del 21/21

La tabla enlaza cada combinación con el inventario de cinco hashes y su
protocolo. Las doce históricas son pasivas del estudio de complejidad muestral;
las nueve nuevas principales son contrafactuales. Los matched certificados
siguen como controles secundarios. La evidencia original detallada de
familias, dominio, criterio, reproducción y limitaciones está en
`native-21-consolidation.csv/json`; no se repitieron entrenamientos.

| Profesor | Arquitectura | Cohorte | Método | Semillas | Etiquetas/semilla | Updates | Parámetros | Final nativo |
|---|---|---|---|---|---|---|---|---|
| BE_BASE_D3 | deepsets | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 6776 | 4000 | 21268 | 5x1000/1000 |
| BE_BASE_D3 | normalized | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 6776 | 4000 | 19396 | 5x1000/1000 |
| BE_BASE_D3 | transformer | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 6776 | 4000 | 20356 | 5x1000/1000 |
| COLOR_TRIALS | deepsets | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 70000 | 4000 | 21268 | 5x1000/1000 |
| COLOR_TRIALS | normalized | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 70000 | 4000 | 19396 | 5x1000/1000 |
| COLOR_TRIALS | transformer | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 70000 | 4000 | 20356 | 5x1000/1000 |
| GREEDY | deepsets | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 17907 | 5x1000/1000 |
| GREEDY | normalized | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 16035 | 5x1000/1000 |
| GREEDY | transformer | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 16995 | 5x1000/1000 |
| JOHANSSON | deepsets | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 17907 | 5x1000/1000 |
| JOHANSSON | normalized | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 16035 | 5x1000/1000 |
| JOHANSSON | transformer | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 16995 | 5x1000/1000 |
| JP | deepsets | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 21268 | 5x1000/1000 |
| JP | normalized | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 19396 | 5x1000/1000 |
| JP | transformer | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 20356 | 5x1000/1000 |
| LUBY | deepsets | historical-primary | passive | 83101,83102,83103,83104,83105 | 3000 | 500 | 17907 | 5x1000/1000 |
| LUBY | normalized | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 16035 | 5x1000/1000 |
| LUBY | transformer | historical-primary | passive | 83101,83102,83103,83104,83105 | 999 | 500 | 16995 | 5x1000/1000 |
| SPECULATIVE_GM | deepsets | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 6104–6120 | 500 | 21268 | 5x1000/1000 |
| SPECULATIVE_GM | normalized | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 6104–6120 | 500 | 19396 | 5x1000/1000 |
| SPECULATIVE_GM | transformer | counterfactual-primary | counterfactual | 94101,94102,94103,94104,94105 | 6104–6120 | 500 | 20356 | 5x1000/1000 |

Los cuatrocientos y miles de casos nuevos de cada peso no se utilizan para cambiar el peso escogido. El inventario también conserva treinta controles CT del mismo hito 4000, que no estaban certificados históricamente y cuyo resultado basal se publica. Pedro y M6B siguen reconstruidos por sus documentos específicos: no se atribuyen a esta cohorte los pesos de Pedro que no se recuperaron, ni la única semilla histórica M6B como cinco réplicas.

## Respuestas explícitas P1–P15

**P1.** Luby, Johansson siempre activo, ColorTrials repetido, JP síncrono, GM de cuatro bloques y Greedy programado: seis extensiones correctas mediante paleta/grado (B), con pruebas para global Δ+1 y local d(v)+1.

**P2.** BE_BASE_D3 (C). Nuevos campos/reducciones o BE recursivo constituyen otra variante; no se ejecuta ni bloquea la campaña. Los otros seis conservan regla y calendario al parametrizar.

**P3.** Los tres núcleos y 180 checkpoints extensibles inventariados, incluidos controles, funcionan técnicamente mediante adaptador estructural (2). El envoltorio histórico K4/vecinos3 requiere esa reorganización. BE no recibe una extensión algebraica; sus 45 pesos sólo se controlan nativamente.

**P4.** La confirmación completa contiene 808413/810000 ejecuciones correctas entre políticas distintas. El resultado por cada una de las tres arquitecturas, profesor y peso está en RESULTS y checkpoint-condition.csv; hubo 209 fallos principales, conservados. El total no describe una política promedio.

**P5.** Global n64: las dieciocho combinaciones pasan Δ4 en sus cinco semillas; diecisiete pasan Δ10, mientras Johansson/Transformer s83101 obtiene972/1000 (las otras cuatro1000). Local n64/Δ4: diecisiete pasan las cinco; GM/Transformer s94102 obtiene819/1000 (otras cuatro1000). Global n256/Δ4: todas pasan las cinco. Las otras dieciséis combinaciones pasan las cinco condiciones principales completas. Cinco profesores distintos de CT tienen éxitos exploratorios hasta Δ200, con una ejecución por peso; no se certifica ese rango. No se afirma monotonicidad ni que una lista local que falla a grado4 funcione a grado3: el control GM muestra precisamente un fallo local a grado3.

**P6.** Se comparan Δ3/4/10 a n64 con la misma mezcla y Δ4 a n64/n256. El piloto regular mantiene n256 al variar grado y Δ3/10 al variar n64–1024; otra extensión exploratoria separada llega a n2048/4096 para los cuatro profesores paralelos, con una ejecución por checkpoint/punto. Grado y paleta global Δ+1 cambian conjuntamente: su efecto causal separado sigue abierto.

**P7.** El índice de fallos y la tabla de primeros testigos identifican familia por peso; el piloto conserva dos colisiones matched CT y cutoffs CT en regulares densos. No se deduce de ausencia de fallos que una familia completa sea segura.

**P8.** Las trazas distinguen disponibilidad, participación, propuesta, COMMIT y marca. La sonda CT a paleta local constante aísla rechazo COMMIT ante preferencias casi empatadas; el primer desacuerdo estadístico de otra ejecución no prueba su causa suficiente. Véase el análisis de fallos por checkpoint.

**P9.** Sí mejora fidelidad de consultas COMMIT en los contrastes apareados a igual hito CT4000/GM500 frente a matched y pasivo, sin mejora universal o monótona en grado. La funcionalidad se evalúa por separado. Inicialización/batches/updates están controlados, pero las etiquetas totales no: no se aísla la naturaleza CF a presupuesto total igual.

**P10.** MAX es insensible a multiplicidad pero suficiente para predicados existenciales; las tres arquitecturas usan MAX en el grafo. Los prefijos, media/1-K y atención de paleta pueden cambiar fuera de distribución. El contraejemplo CT mantiene un vecino y dos colores locales, por lo que no se explica por aumentar la agregación. Los 45 certificados canónicos no establecen seguridad universal de COMMIT para estos pesos pasivos.

**P11.** JP/GM conservan IDs únicos y relaciones de orden coherentes; retirar IDs elimina información necesaria. GM y Greedy siguen recibiendo calendario/permiso, y BE aritmética y fases explícitas. No se presenta esa ayuda como protocolo anónimo aprendido. No hay ampliación del espacio de IDs más allá de 65536 vértices.

**P12.** Pruebas condicionales: disponibilidad, seguridad y progreso/terminación de seis profesores en ambas listas; JP≤n, Greedy=n y GM≤2ceil(n/4)+n; cotas geométricas ideales para aleatorios. Contraejemplos informacionales y soporte MAX son matemáticos. Los pesos sólo tienen evidencia finita y contraejemplos/certificados de cabeza condicionales; progreso aprendido uniforme y seguridad universal siguen abiertos.

**P13.** Priorizar CT con diferencias de preferencias nuevas, estados de lista heterogénea y controles a igual presupuesto total. Entrenar hasta10 y reservar20/50/100/200; ampliar a20 o50 sólo si la hipótesis lo exige. Otras arquitecturas/profesores se priorizan según fallos reales, no por obligación de llenar benchmarks. BE exige otro algoritmo. El plan no se ejecutó.

**P14.** Integrar 21/21 acotado, CF/funcionalidad separados, información, optimización, transferencia confirmada y el límite CT en el artículo actual. Reservar entrenamiento multigrado, separación causal Δ/K y garantías uniformes/BE general para otra investigación.

**P15.** La evidencia experimental principal queda cerrada al verificar denominadores, hashes, tests y PDF. Para defender el manuscrito se requiere revisión de coautores de alcance, pruebas y tablas y contraste bibliográfico editorial; no repetir entrenamiento ya certificado. Cualquier afirmación más amplia requerirá la confirmación/controles adicionales especificados, no puede obtenerse reinterpretando el piloto.

## Qué queda abierto y qué no bloquea el cierre

Sigue abierta la extrapolación confirmatoria densa a Δ20/50/100/200 y n1024, el aislamiento de grado frente a paleta, CF a igual número total de etiquetas, capacidad frente a optimización, y una variante BE general. Son ampliaciones de alcance, no resultados ya obtenidos. La compilación y el manifiesto final se verifican en los recibos locales; no se hace push ni publicación externa.
