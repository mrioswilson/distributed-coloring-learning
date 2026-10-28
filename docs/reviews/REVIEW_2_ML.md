# Revisión adversarial interna 2 — aprendizaje automático

Fecha: 26 de octubre de 2026. Revisión ejecutada por un agente distinto del editor principal sobre el borrador de cierre, el código y los artefactos conservados. Es una revisión interna independiente de ejecución, no un arbitraje humano externo ni una réplica independiente de los entrenamientos.

## Dictamen y alcance

La separación empírica entre ejecución nativa y respuestas al oráculo de continuación está respaldada. El 21/21 también es defendible como cobertura finita de sistemas con intérpretes e información declarados. La interpretación causal del contrafactual y la generalización amplia a intervenciones nuevas no están identificadas. El borrador ya retira esas dos interpretaciones fuertes.

Encontré un problema adicional importante de interpretación: el diagnóstico BE contiene aproximadamente un 97% de negativos. La exactitud absoluta por sí sola oculta que un COMMIT que siempre rechaza obtiene alrededor del 97%, y que buena parte de la diferencia entre procedimientos corresponde a rechazar raíces KW ineligibles. Esta objeción se puede resolver completamente con los datos existentes, sin entrenar. La estratificación reproducida independientemente confirma además un efecto más específico: en raíces elegibles activas CF identifica todas las opciones ocupadas consultadas, mientras passive y matched aceptan todas las consultas y tienen balanced accuracy del 50%.

No identifico necesidad de una nueva campaña científica para las afirmaciones delimitadas. Recomiendo revisión externa una vez incorporados el desequilibrio de clases, los controles de rechazo/elegibilidad, los estratos y las aclaraciones de representación señaladas abajo.

## Verificaciones realizadas

Se revisaron `src/dcl/multicf/{training,data,oracles}.py`, `src/dcl/multiteacher_e1/{model,supervision}.py`, `src/dcl/{samplecomplexity/model,integration/models,degree/adapter,paperstudy/models}.py`, los protocolos y los recibos de cierre. Se verificó la ruta de inferencia frente a la de adquisición de etiquetas; no se encontró una máscara exacta de disponibilidad o un veto docente oculto dentro del modelo.

Se reconstruyeron exactamente las consultas BE/E1 ya existentes, con RNG 17001, agrupación por fase y batches de 256. Cada numerador, falso positivo y falso negativo coincide con el diagnóstico histórico. El código de esta comprobación es `docs/reviews/REVIEW_2_BE_STRATA.py`; el recibo es `results/final-closure/review-ml-be-strata.json`, que registra el SHA-256 del script. No se produjeron modelos ni consultas aleatorias nuevas.

## Objeciones y resolución requerida

### ML-01 — Exactitud BE sin prevalencia o baseline de rechazo

**Categoría: evidencia insuficiente para la interpretación del mecanismo; problema editorial en la presentación de la exactitud.** Prioridad alta, resoluble ahora.

La sección Counterfactual Fidelity del borrador inicial informa 22,6404%, 9,2686% y 99,9932%, pero omite prevalencia, confusiones y controles triviales. En semillas 94101–94105 hay 3.070, 2.895, 2.891, 2.773 y 3.115 positivos, cada uno sobre 99.840 consultas. Un clasificador siempre-rechazar obtiene 96,8800–97,2226%. Todos los procedimientos tienen cero falsos negativos; la brecha corresponde a falsos positivos.

Medias descriptivas sobre cinco semillas emparejadas:

| Procedimiento/control | Exactitud (%) | Balanced accuracy (%) |
|---|---:|---:|
| Passive | 22,6404 | 60,1413 |
| Matched | 9,2686 | 53,2536 |
| CF | 99,9932 | 99,9965 |
| Siempre rechazar | 97,0465 | 50,0000 |
| Sólo elegibilidad y participación | 98,5990 | 99,2781 |

El último control predice `(offset_KW >= 0) and P_root`; ignora disponibilidad. No es un nuevo modelo entrenado y no pretende resolver el coloreado. Su balanced accuracy alta en la mezcla completa muestra cuánto pesan las condiciones de elegibilidad en ese estimando.

**Corrección concreta:** añadir prevalencia y confusiones o TPR/TNR, los dos controles y una estratificación por elegibilidad/participación. Mantener clara la diferencia entre respuesta COMMIT y escritura efectiva: rechazar todo puede tener buen score en esta mezcla y ningún progreso funcional.

En raíces elegibles activas, denominadores por semilla 4.450, 4.272, 4.268, 4.150 y 4.598, CF obtiene 100% en las cinco. Passive y matched obtienen 68,9888%, 67,7669%, 67,7366%, 66,8193%, 67,7468%; ambos aceptan todas y tienen balanced accuracy 50%. Por tanto la mejora no se reduce exclusivamente a copiar elegibilidad: CF también aprende a rechazar opciones ocupadas en este estrato. Los 34 errores CF totales pertenecen a raíces elegibles silenciadas, no a las activas ni a las ineligibles. Incorporar esta evidencia resuelve la objeción sin datos nuevos.

### ML-02 — Qué significa «intervención no observada»

**Categoría: evidencia insuficiente para generalización amplia; limitación ya reconocida en el borrador.**

Las intervenciones de entrenamiento y diagnóstico comparten `proposal_bank`, los cuatro colores raíz y las mismas familias de consultas. Los grafos y namespaces son nuevos, pero un nodo no observa el grafo entero. La clave suficiente de COMMIT agrupa más del 99% de las ocurrencias de diagnóstico en tipos presentes en el banco de entrenamiento. El contraste no es un holdout de familias de intervención.

La clave usada por `final-counterfactual-audit.py:35–51` es coherente con la segunda etapa BE: elimina IDs/slots y campos descartados, conserva fase/permiso/offset, participación y el conjunto de tipos de vecino en el slot consultado. El conjunto de comparación incluye consultas nativas y CF adquiridas; no representa tres bancos de datos distintos por tratamiento.

El subconjunto de tipos nuevos sí contiene positivos. Los denominadores son 549, 459, 717, 522, 722 y los positivos 52, 108, 90, 56, 60. CF conserva esos 366 positivos sin error, con 34 falsos positivos; no es únicamente una prueba de rechazo de negativos nuevos. El baseline siempre-rechazar consigue 497, 351, 627, 466, 662 aciertos, frente CF 549, 437, 717, 510, 722. El control sólo-elegibilidad consigue 366, 382, 508, 409, 599. CF mejora ambos en cada semilla.

**Corrección concreta:** conservar el resultado como diagnóstico posthoc de tipos locales ausentes en el banco, incluir las dos clases y sus baselines y no llamarlo confirmación independiente de intervención OOD. Un nuevo test de familias de intervención exige protocolo y estados nuevos, pero puede quedar en trabajo futuro si no se mantiene la afirmación fuerte.

### ML-03 — CF no está igualado en etiquetas, presentaciones ni coste

**Categoría: limitación reconocida.**

`multicf/training.py:38–69,127–154` mantiene el banco nativo y el muestreo emparejado, pero CF agrega pérdida y etiquetas. En BE usa 6.776 etiquetas adquiridas frente a 3.000. Matched y passive no tienen los mismos contextos de COMMIT. La intervención modifica simultáneamente cobertura, distribución de etiquetas y objetivo. Igualar updates no iguala todas las operaciones computacionales o presentaciones de etiquetas.

**Resolución:** el borrador ya lo dice explícitamente. Debe sostenerse en abstract, conclusiones y decisión editorial: se comparan procedimientos completos. El contraste causal más limpio disponible es el control de optimización con banco/inicialización/prefijo fijos; no identifica la contribución aislada de CF. Ningún nuevo experimento es necesario para esta narrativa limitada.

### ML-04 — Estados de desarrollo y duplicación de cabezas

**Categoría: limitación reconocida.**

El banco principal de fidelidad proviene de las 60 trayectorias de desarrollo, incluidas en el flujo de selección/diagnóstico, no de una batería final intocada. BE y GM pueden tener COMMIT exactamente idéntico entre arquitecturas emparejadas porque las pérdidas con mensajes fijados no pasan por los scorers de paleta. Hay cinco réplicas pertinentes de esas cabezas por hito, no quince independientes. Variantes, raíces, rondas y consultas de un mismo estado tampoco son réplicas de entrenamiento.

**Resolución:** las restricciones ya figuran correctamente. No asignar significación basada en 499.200 decisiones ni sumar arquitectura y consulta para elevar el n independiente.

### ML-05 — Congelar pesos no elimina la adaptación de cómputo

**Categoría: problema editorial y limitación de comparación fuera del dominio.**

El adaptador de `degree/adapter.py` generaliza la forma de tensores y suministra máscaras de lista declarada. En líneas 105–111 el contexto de participación promedia sólo slots válidos en listas locales, y en 137–138 cambia la normalización de preferencias según el bound global. Esto conserva pesos, pero también define explícitamente cómo se interpreta la red fuera de K4. La paridad nativa no determina una extensión única.

**Corrección concreta:** expresar que se evalúan pesos congelados bajo una extensión estructural declarada, e identificar promedio sobre slots válidos y normalización. No describirlo como extrapolación sin ninguna adaptación de inferencia. No es necesaria una nueva comparación de embeddings; los controles ya conservan sensibilidad a normalización.

### ML-06 — Baselines existentes desaparecen del manuscrito

**Categoría: problema editorial.**

El borrador inicial no conserva resultados pequeños/sin comunicación aunque `results/sample-complexity/controls.json` contiene controles al mismo B=3.000 y 500 updates. En desarrollo (28 ejecuciones por semilla), no-communication da 0/28 en las cinco semillas para Luby, JP y BE; small da Luby [0,0,28,1,0], JP [10,0,15,0,0], BE [0,0,0,0,0]. No son certificaciones ni prueban imposibilidad de otro modelo sin mensajes.

**Corrección concreta:** un párrafo o tabla de apéndice permite a un revisor identificar comparaciones ya existentes y sus límites. No hace falta crear nuevos baselines, ni transformar esos resultados en evidencia de superioridad uniforme de las tres arquitecturas.

### ML-07 — Fidelidad a un oráculo especificado, no contrafactual único de una traza

**Categoría: limitación conceptual reconocible; aclaración editorial.**

Fuera de los mensajes nativos, múltiples extensiones de una regla pueden coincidir en todas las trayectorias observadas. El código elige un oráculo conservador que conserva elegibilidad y comprobación de disponibilidad. En particular, una consulta puede forzar participación de una raíz KW que nativamente no se activa y cuyo oráculo devuelve falso. Esa es una pregunta bien definida por el contrato, pero no una continuación identificada únicamente por las trazas.

**Corrección concreta:** llamar al objeto «oráculo de continuación declarado» y evitar «se ha aprendido el algoritmo original completo». La distinción general entre solucionar, imitar y contestar intervenciones no es novedosa por sí misma; el incremento son las comparaciones/mecanismos concretos. No requiere entrenamiento adicional.

## Cierre de esta revisión

Las objeciones ML-01/02 requieren incluir la auditoría de estratos y baselines antes de cerrar la interpretación central. Las restantes se resuelven preservando las restricciones ya escritas o aclarando el texto. Las cifras adicionales confirman una ventaja descriptiva CF incluso entre consultas activas elegibles y tipos nuevos, pero no eliminan el presupuesto desigual ni la selección/reutilización de desarrollo. Una afirmación causal a igual etiqueta o un holdout amplio de intervenciones seguirían necesitando otro estudio; pueden retirarse legítimamente del alcance actual.

## Addendum — revisión de las correcciones incorporadas

Se volvió a leer el manuscrito corregido y las tablas/figura generadas, sin ejecutar experimentos adicionales. Instantánea de `paper/ucgnn/main.tex` revisada: SHA-256 `48db1d2dc46733e018489a8d18a4e00abba652c2fa113ec0b3d7671e6b35816c`. Se inspeccionó visualmente `paper/ucgnn/figures/closure-counterfactual.png` y se contrastaron `closure-be-summary.tex` y `closure-baselines.tex` con el recibo independiente.

| Objeción | Resolución observada | Estado |
|---|---|---|
| ML-01 | Tabla BE incluye exactitud, BA, siempre-rechazar y sólo-elegibilidad; texto informa prevalencia, todos los falsos positivos y estrato elegible activo | Resuelta con datos existentes |
| ML-02 | Se documentan positivos y negativos en tipos nuevos, BA y carácter posthoc/desarrollo; no se reclama holdout de familias | Resuelta por análisis y delimitación |
| ML-03 | Permanecen etiquetas 3.000/3.000/6.776 y retirada explícita de causalidad a igual presupuesto | Limitación reconocida correctamente |
| ML-04 | Se mantienen desarrollo reutilizado y cinco cabezas independientes pertinentes, sin pseudorreplicación arquitectónica | Limitación reconocida correctamente |
| ML-05 | Se explicita el promedio sobre slots válidos y que pesos fijos usan una adaptación declarada de cómputo | Resuelta editorialmente |
| ML-06 | Apéndice añade small/no-communication, tamaños, datos y denominadores de desarrollo por semilla | Resuelta sin nuevos baselines entrenados |
| ML-07 | Apéndice distingue oráculo elegido de semántica off-policy identificada por trazas; alcance de inferencia ya declara contratos | Resuelta conceptualmente |

Las cifras añadidas coinciden con `review-ml-be-strata.json`. La figura usa BA y muestra los baselines, evitando interpretar el 97% de rechazo como aprendizaje funcional. Sólo se notificó una corrección menor de leyenda: donde dice «Curves connect seeds» debe decir que cada curva conecta tratamientos dentro de una misma semilla emparejada.

**Dictamen actualizado:** no quedan objeciones sustantivas de aprendizaje automático sin respuesta para las afirmaciones delimitadas del manuscrito revisado. La eficiencia causal CF a igual etiqueta, un final de fidelidad totalmente nuevo y las familias de intervención no observadas permanecen trabajo futuro; no son resultados demostrados por esta revisión. La versión puede pasar a revisión externa desde esta perspectiva, condicionada a la comprobación editorial/PDF y al resto de revisiones internas.
