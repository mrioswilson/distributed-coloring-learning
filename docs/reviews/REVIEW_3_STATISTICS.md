# Revisor 3 — estadística experimental

Revisión independiente del borrador de cierre del 26 de octubre de 2026, base científica `71153993`. Alcance: contabilidad confirmatoria, unidades de replicación, intervalos, selección y diagnósticos BE. No se ha entrenado, simulado ni seleccionado ningún modelo. Las referencias a líneas corresponden al primer borrador de cierre y pueden desplazarse con las correcciones; se identifican también los pasajes y funciones.

## Comprobaciones independientes

He leído las **810.000 filas confirmatorias**, verificando el SHA-256 de cada archivo contra su índice, sin usar como entrada los totales del auditor principal. El recibo es `results/final-closure/reviewer3-statistics-check.json`.

| Unidad/cohorte | Ejecuciones | Correctas | Incorrectas | Celdas con fallo |
|---|---:|---:|---:|---:|
| Confirmación completa | 810.000 | 808.413 | 1.587 | 15/810 |
| Principal | 450.000 | 449.791 | 209 | 2/450 |
| Controles secundarios | 360.000 | 358.622 | 1.378 | 13/360 |

Hay 1.559 colisiones y 28 cutoffs. Los mismos datos contienen 4.000 grafos etiquetados distintos, 20.000 semillas de cinta y 25.000 pares caso–paleta. No existe una contradicción entre estas cifras: grafos, cintas, paletas y políticas se reutilizan explícitamente. `degree/run.py:suite_for` y `degree/graphs.py:make_case` explican esa reutilización. El total es una contabilidad de políticas fijas; convertirlo en probabilidad de éxito de un algoritmo promedio sería injustificado. El borrador ya evita esa conversión.

Los extremos para 1.000/1.000 se recalculan como 0,9963179161031344 (bilateral95) y 0,9970087504549047 (unilateral95). La justificación AM–GM del caso cero fallos con estratos independientes no idénticos es correcta para su media ponderada por número de ensayos. No justifica los mismos extremos para cada familia ni intervalos binomiales generales con fallos bajo cualquier estratificación.

La confirmación congela `family_sampling=iid_uniform`, 1.000 índices por condición. La familia depende de namespace/n/Δ/índice, no del profesor ni del checkpoint. El piloto usa selección round-robin de familias donde hay varias y tiene una enmienda de tiempos heterogénea. La extensión n2048/4096 tiene denominador uno por checkpoint/condición. Estas procedencias no pueden sumarse para certificar los extremos explorados.

## Objeciones

### R3-01 — la precisión bruta BE oculta un fuerte desequilibrio

**Clasificación:** evidencia insuficiente en la presentación inicial. **Prioridad:** alta, corregible con datos existentes.

El pasaje de fidelidad BE y su tabla inicialmente presentan 22,6404%, 9,2686% y 99,9932% sin un clasificador trivial ni métricas por clase. Los diagnósticos primarios `paired-u004000-{method}/be_base_d3/normalized/s*.json` contienen 14.744 positivos y 484.456 negativos entre 499.200 consultas por brazo. Rechazar siempre obtiene **97,046474%**. La exactitud bruta exageraría el progreso sobre una referencia útil si se leyera sola.

Todos los brazos tienen cero falsos negativos en este banco. El desglose descriptivo agregado es:

| Método | TP | TN | FP | FN | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Passive | 14.744 | 98.277 | 386.179 | 0 | 60,143026% |
| Matched | 14.744 | 31.525 | 452.931 | 0 | 53,253649% |
| Counterfactual | 14.744 | 484.422 | 34 | 0 | 99,996491% |

La diferencia sigue siendo real y grande, pero es principalmente aprendizaje de **rechazo de continuaciones negativas**, no mejora de las respuestas positivas. Estos agregados describen los bancos y no son ensayos independientes ni sustituyen los cinco resultados por semilla.

**Corrección requerida:** añadir prevalencia, always-reject, TP/TN/FP/FN y sensibilidad/especificidad o balanced accuracy por semilla; explicar qué comportamiento cambia. Desglosar también positivos/negativos y referencia trivial **dentro del subconjunto de tipos no vistos**. No atribuir generalización de respuestas positivas a un subconjunto que pudiera carecer de positivos. El responsable del cierre ha aceptado reconstruir estas métricas sobre las mismas consultas.

### R3-02 — intervalos de pilotos no tienen automáticamente cobertura binomial

**Clasificación:** evidencia insuficiente; problema de etiquetado del artefacto. **Prioridad:** media.

`scripts/final-evidence-audit.py` emite columnas `cp95_lower/upper` para todas las etapas, incluidas mezclas piloto round-robin y celdas con límites de recursos no homogéneos. Calcular cuantiles beta es numéricamente correcto, pero no demuestra cobertura para ese diseño. El manuscrito no necesita inferencias sobre esas tasas piloto.

**Corrección requerida:** omitir intervalos fuera de confirmación, o añadir `interval_scope` que los declare referencias binomiales sin garantía de cobertura del piloto. Mantener confirmatorios sólo bajo su modelo de independencia y mezcla especificado. No se requiere otro experimento. El responsable del cierre ha aceptado restringir/etiquetar este alcance.

### R3-03 — dependencia entre decisiones, arquitecturas y bancos

**Clasificación:** limitación reconocida. **Prioridad:** alta si se reintroducen tests de decisión; resuelta para la narrativa descriptiva actual.

499.200 consultas BE comparten estados, trayectorias y opciones, y los brazos comparten los mismos bancos. Las cabezas CF BE/GM idénticas entre tres arquitecturas no aportan quince réplicas independientes. Las cinco semillas son las unidades pareadas pertinentes para contrastes entre procedimientos. El manuscrito lo explica en “Other teachers and dependent architecture comparisons” y el apéndice de dependencia.

**Corrección requerida:** conservar esa explicación; toda incertidumbre de tratamiento debe respetar semillas/estados o usar un modelo jerárquico previamente definido. No calcular errores estándar dividiendo por la raíz de 499.200. No se necesita ampliar las semillas para reportar una comparación descriptiva delimitada. Incluso cinco diferencias del mismo signo dan p bilateral de signos 0,0625 bajo la hipótesis apropiada, antes de multiplicidad; no reclamar significación convencional con ese argumento.

### R3-04 — desarrollo reutilizado y subconjuntos posteriores

**Clasificación:** limitación reconocida / evidencia insuficiente para una afirmación más amplia. **Prioridad:** alta para generalización causal.

Los estados BE se generan a partir del split de desarrollo utilizado repetidamente durante el estudio. Separar tamaños, namespaces y grafos de entrenamiento evita una igualdad trivial de bancos, pero no convierte desarrollo en test final intacto. La definición de “no visto” se ha refinado después de observar resultados. El análisis de tipos no vistos es posthoc y comparte el mecanismo de consultas de entrenamiento.

**Corrección requerida:** mantener la etiqueta desarrollo/posthoc en texto, figura y conclusión; distinguir input serializado de tipo suficiente para COMMIT y no anunciar un holdout de familias de intervenciones. Los porcentajes 99,277–99,540 de tipos vistos concuerdan con el recibo actual. Una nueva evaluación independiente sólo sería necesaria para reclamar generalización prospectiva a nuevas distribuciones de intervención; esa afirmación puede retirarse y el experimento quedar fuera del alcance.

### R3-05 — no hay identificación causal a igual total de etiquetas

**Clasificación:** evidencia insuficiente para la afirmación fuerte, reconocida correctamente en el borrador. **Prioridad:** alta.

En BE se mantienen inicialización, datos nativos, visitas, RNG de minibatches y 4.000 updates, pero se comparan 3.000/3.000/6.776 etiquetas y se cambian contextos y distribución de la pérdida. El contraste identifica los resultados de procedimientos compuestos bajo bancos/semillas emparejados. No aísla la naturaleza contrafactual frente a más etiquetas ni demuestra eficiencia muestral.

**Corrección requerida:** conservar el lenguaje “procedures” y la ausencia de una conclusión causal de eficiencia. El control fijo de 500/1.500 updates permite una atribución más estrecha a optimización en sus propios bancos, no a CF. No usar sus resultados para suplir el control CF a recursos iguales. Si se quisiera recuperar esta afirmación fuerte, haría falta un protocolo nuevo; no es necesario para las conclusiones limitadas actuales.

### R3-06 — CP es condicional a independencia y neutralidad del batching

**Clasificación:** limitación reconocible adicional. **Prioridad:** media.

`degree/evaluation.py` ejecuta lotes de grafos desconectados con agrupación dinámica. No hay mezcla matemática de mensajes entre grafos, pero los replays preservan el lote completo precisamente porque la aritmética numérica puede depender de forma/tamaño del lote. Si el resultado de un ensayo cambiara con la duración o forma de otros, la independencia Bernoulli por ejecución no sería exacta para esa implementación numérica. Los índices distintos de PRNG no demuestran por sí solos neutralidad del kernel.

**Corrección requerida:** declarar que los intervalos por checkpoint son condicionales al modelo de ensayos independientes y al supuesto de neutralidad semántica del batching. Los conteos observados y replays siguen siendo válidos. No recomiendo una campaña nueva para certificar independencia a precisión finita; basta no vender el intervalo como garantía numérica formal. Si se requiriera una garantía robusta a dependencia dentro del lote, el lote sería una unidad de agrupación para otra inferencia.

### R3-07 — selección de condiciones, múltiples celdas y cobertura

**Clasificación:** limitación reconocida. **Prioridad:** media.

Las condiciones confirmatorias se escogieron con el piloto, pero usan namespaces nuevos y reglas congeladas; esto es compatible con confirmación de esas condiciones. No certifica grados/tamaños descartados. Los intervalos puntuales sobre 810 celdas no tienen cobertura simultánea95. La existencia de 21 paquetes seleccionados no estima probabilidad de éxito de un nuevo entrenamiento.

**Corrección requerida:** conservar la distinción entre contabilidad, validación de políticas congeladas y generalización a otra semilla. Preservar candidatos finales fallidos y no usar la confirmación para reintentar silenciosamente el mismo gate. La narrativa actual lo cumple. No es necesario corregir por comparaciones múltiples una tabla puramente descriptiva; sí lo sería para una nueva familia de decisiones inferenciales.

### R3-08 — evidencia histórica de leyes completas

**Clasificación:** limitación reconocida / nueva investigación fuera del alcance.

El apéndice evita convertir no rechazo en igualdad y exige observaciones de ejecuciones completas, permutaciones por filas y selección de testigos independiente o validada por permutación conjunta. Es correcto. No he repetido cada test histórico de distribuciones; esta revisión no equivale a autenticar todas sus implementaciones. No existe en el nuevo texto una nueva afirmación numérica global que yo deba validar como confirmación independiente.

**Corrección requerida:** cualquier cifra o afirmación fuerte que se reintroduzca de esos estudios debe conservar su objeto, banco, hipótesis, multiplicidad, potencia y procedencia. Si la contribución de un segundo artículo dependiera de identificar leyes completas, requeriría su propia revisión específica y no reutilizar el presente recibo como certificación de todos los contrastes.

## Dictamen estadístico

Los denominadores principales y confirmatorios están reconciliados independientemente. No encuentro un error numérico que destruya la cobertura nativa finita o la contabilidad de transferencia. Sí sería sustantivamente engañosa una narrativa de CF basada sólo en exactitud bruta, novedad del input completo o tamaños de muestra de decisiones. Las correcciones R3-01 y R3-02 pueden resolverse íntegramente con los datos existentes; R3-06 necesita una frase de alcance, no entrenamiento.

**Recomendación desde esta perspectiva: READY WITH MINOR REVISIONS**, condicionada a esas correcciones de presentación/inferencia. La comparación causal fuerte a igual presupuesto y el holdout prospectivo de familias de intervención siguen sin estar establecidos; no bloquean el artículo si se mantienen retirados. No recomiendo más campañas para intentar recuperar una conclusión más amplia antes del cierre.

## Addendum — verificación de las correcciones incorporadas

He revisado de nuevo el manuscrito cuyo SHA-256 es **`48db1d2dc46733e018489a8d18a4e00abba652c2fa113ec0b3d7671e6b35816c`**, las métricas existentes y los artefactos regenerados. Esta segunda pasada es una verificación puntual; no repite las 810.000 filas, consultas, simulaciones ni entrenamientos.

- **R3-01 resuelta.** La tabla `closure-be-summary.tex` incluye prevalencia, mean balanced accuracy, always-reject y eligibility-only. La balanced accuracy se calcula primero por semilla y luego se promedia; es correcto que difiera ligeramente del agregado descriptivo de la revisión inicial. Recalculé desde `review-ml-be-strata.json` BA media 60,141255% passive, 53,253641% matched y 99,996494% CF; always-reject es 50% y eligibility-only 99,278139%. El texto declara cero FN en todos los brazos y sitúa los 34 FP CF en raíces elegibles silenciosas. El estrato elegible/activo contiene ambas clases, entre 4.150 y 4.598 ocurrencias por semilla: CF tiene 100% y los otros dos procedimientos BA50%. Así se distingue aprender rechazo de opciones ocupadas de aprender sólo una puerta de elegibilidad.
- **R3-01, subconjunto no visto, resuelta.** Verifiqué positivos 52/108/90/56/60 y negativos 497/351/627/466/662. Hay ambas clases en cada semilla, ningún FN y 34 FP CF. La BA media del subconjunto es 99,115709% CF, 66,529900% matched y 50% always-reject, coherente con el texto. La ventaja se describe como especificidad; no se afirma mejora de nuevas respuestas positivas. Se conservan las etiquetas posthoc/desarrollo.
- **R3-02 resuelta en código y artefacto.** `final-evidence-audit.py`, SHA-256 `59706b86c2c220d1ba6ad2c644fe882089b5b4e740b4df2535c0e777d29135e3`, sólo emite CP para confirmación. La tabla regenerada contiene 2.070 celdas piloto y 240 de tamaño posterior con límites vacíos y `exploratory_no_confidence_interval`; las 810 confirmatorias llevan `pointwise_95_under_declared_iid_mixture`. El recibo de auditoría identifica el mismo SHA del script. No se han promovido pilotos a confirmación.
- **R3-06 resuelta como delimitación.** El pasaje de intervalos declara expresamente neutralidad semántica del batching y ausencia de una certificación universal de neutralidad numérica. Los intervalos no se anuncian como garantía formal del kernel de ejecución.
- **R3-03/04/05/07 siguen como limitaciones explícitas, no objeciones sin respuesta.** No hay pseudorreplicación inferencial añadida, y siguen retiradas la causalidad CF a igual presupuesto, la generalización prospectiva a familias de intervención nuevas y las garantías simultáneas/universales.
- **R3-08 conserva su límite de alcance.** Esta revisión no certifica todos los tests históricos de leyes completas ni un hipotético segundo artículo que dependa centralmente de ellos.

**Dictamen estadístico actualizado: READY FOR EXTERNAL REVIEW para las afirmaciones delimitadas del manuscrito identificado.** No queda un bloqueo estadístico sustantivo que exija nuevos experimentos para ese alcance. Esto no acredita la afirmación causal fuerte retirada, no sustituye revisión de autores, novedad o pruebas, y no verifica por sí solo que el PDF final incluya exactamente esta versión.
