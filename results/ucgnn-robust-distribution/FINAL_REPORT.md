# Pruebas robustas de distribuciones de coloraciones completas

La evaluación confirmatoria independiente completó **795 contrastes** sobre 53 entradas de grafos fijos (51 grafos etiquetados únicos) y quince políticas congeladas. El nuevo omnibus rechazó **340/795** tras Holm al 5%; el MMD anterior aplicado a los mismos bancos nuevos rechazó **239/795**. Las observaciones son vectores completos, condicionadas a completar en un máximo de 128 rondas. Los recuentos de rechazos no son medidas de distancia.

| Arquitectura | Histórico m=250 | MMD antiguo, banco nuevo m=500 | Omnibus nuevo, total m=500 |
|---|---:|---:|---:|
| NORMALIZED (E1) | 10/265 | 21/265 | 52/265 |
| DeepSets | 46/265 | 71/265 | 107/265 |
| LocalPaletteTransformer | 127/265 | 147/265 | 181/265 |

La comparación histórica usó 250 ejecuciones; el presupuesto principal nuevo es de 500 ejecuciones por generador. En bancos sin fallos, el omnibus utiliza 250 coloraciones para selección y 250 para evaluación. Cuando hay cutoffs u otros fallos se utilizan solo las completadas de ese prefijo; actual_m_Luby, actual_m_model y selection_per_class/evaluation_per_class registran los tamaños efectivos. No se reemplazan fallos retrospectivamente. Por eso el cambio respecto del resultado histórico mezcla procedimiento, tamaño y muestras nuevas; la comparación entre procedimientos sobre el mismo banco separa el presupuesto muestral de esa comparación. Ningún checkpoint se mezcló con otro.

## P1. Por qué falló el control estrella

Las dos leyes son coloraciones propias de una estrella con 200 hojas binarias iid. Solo difiere el centro: uniforme en 199 colores frente a uniforme en un subconjunto de 100. Su TV es 99/199, pero MMD²=1.50213168648e-05. A m=250, su señal es apenas 0.0623 desviaciones nulas; la desviación exacta es 0.000241288281. Las hojas irrelevantes aportan variabilidad, mientras la normalización por n y el soporte grande atenúan la señal del centro. Se reprodujeron exactamente los 100 valores-p originales. No fue un problema causado únicamente por Holm: el control ya fallaba al nivel nominal .05. Véanse OLD_TEST_STATISTICAL_AUDIT.md, la demostración y las figuras 1–3.

## P2. Calibración y Holm anteriores

El test anterior empleó el U-estadístico no sesgado, conservando valores negativos, kernel exp(-d_H/n), gamma=1 prefijado, permutaciones uniformes de etiquetas de vectores completos, empates inclusivos y corrección de una unidad. Los 19.999 sorteos permiten p mínimo .00005, inferior a .05/795=.000062893. Se recalcularon exactamente los 795 ajustes Holm. No se identificó inflación por ese procedimiento ni insuficiencia de resolución primaria; muchos resultados saturaron en el mínimo. Que el kernel sea característico no implica buena potencia con estas muestras.

## P3. Método seleccionado, nivel y coste

Se seleccionó el omnibus honesto V2 antes de su validación independiente. En una mitad aprende testigos locales, de pares/triples, estructurales, marginales densos y cuatro testigos de kernels globales. La búsqueda de pares por covarianzas utiliza solo esa mitad. En la otra mitad, un máximo conjunto incluye dichos testigos y el MMD característico de vectores completos. Cada permutación recalcula el máximo; no se ejecuta una colección de pruebas al 5% sin corrección. Condicional a la selección, las etiquetas de evaluación son intercambiables bajo H0, lo que justifica el nivel para cualquier ley del espacio finito.

La auditoría de validación independiente incluyó 104 celdas método/ley nula, con 400 ensayos por ley. Superó el criterio binomial múltiple prefijado y las metas de potencia. Los intervalos de tasa son puntuales; una simulación finita no certifica universalmente el nivel. La demostración por permutaciones aporta la garantía. Los controles degenerados se conservaron. Se compararon MMD global multiescala por máximo y softmax, versiones locales/estructurales, clasificadores simples con entrenamiento/validación/evaluación separados y LRT categórico calibrado. Las adaptaciones no se presentan como implementaciones exactas de MMDAgg o MMD-FUSE ni heredan automáticamente sus garantías minimax.

La mayor frecuencia nula observada para V2 fue 0.080 en star_null (32/400; Wilson95 [0.057, 0.111]). Ese intervalo puntual queda por encima de .05. Su prueba binomial unilateral no superó el umbral múltiple prospectivo de .01/104; no se ocultó esa fluctuación ni se usó para reajustar el método. La simulación no demuestra que la tasa real sea exactamente .05; la garantía de nivel depende de las hipótesis de independencia e intercambiabilidad documentadas.

Para investigar esa celda se preregistró después de congelar el método una réplica independiente de 4.000 ensayos CPU, antes de generar sus datos y sin consultar valores-p neuronales. Dio 202/4000 rechazos (0.0505; Wilson95 [0.0441, 0.0577]); probabilidad binomial unilateral de excedencia 0.4524. El criterio diagnóstico prefijado fue .01. Este resultado adicional no reemplaza el gate original, no cambia la elección del método y no certifica otras leyes. Plan y commit: ROBUST_NULL_REPLICATION_PLAN.md, 9bd3398; datos: null-replication-diagnostic.json.

Los CSV de validación retienen los once componentes/candidatos originales y la comparación con V1, potencia, incertidumbre y tiempos. omnibus_seconds mide el procedimiento completo; quadratic_seconds mide conjuntamente las cuatro escalas globales, no el coste aislado de una sola escala. La elección priorizó sensibilidad verificable y nivel correcto, con coste CPU compatible con la batería. No existe un ganador universal.

Medianas observadas en las alternativas de la validación independiente, con B=199 y un hilo BLAS por proceso. Los tiempos incluyen operaciones diferentes en cada columna y no deben extrapolarse linealmente a B=19.999; confirmatory-computation-cost.csv conserva los costes medidos de la comparación principal, incluido su cálculo conservador de futilidad.

| m | Omnibus V2 (s) | Cuatro MMD globales juntos (s) | Pipeline comparativo completo (s) |
|---:|---:|---:|---:|
| 100 | 0.124 | 0.015 | 0.422 |
| 250 | 0.183 | 0.075 | 0.657 |
| 500 | 0.337 | 0.396 | 1.393 |
| 1000 | 1.131 | 2.336 | 5.019 |
| 2000 | 4.795 | 11.480 | 21.887 |


## P4. Diferencias localizadas detectadas

La primera versión cumplía las metas iniciales, pero una prueba de desarrollo trasladada a vértices no prefijados reveló 0/10 detecciones de un par correlacionado a m=250. V2 añadió un cribado de todos los pares por cambios de covarianza en la muestra de selección, y se evaluó de nuevo con semillas y muestras independientes. Ese cribado no garantiza detectar toda diferencia entre leyes categóricas de pares: solo usa dos representaciones de covarianza. Se conservan ambas versiones, las pruebas previas y los fallos de triples remotos. No se modificó el test después de observar resultados neuronales confirmatorios.

## P5. Cuántas coloraciones hacen falta

Las tablas benchmark-validation2.csv y las figuras 1 y 8 presentan tasas a m=100,250,500,1000,2000; m cuenta también los datos usados para selección. Cada punto tiene treinta ensayos independientes y un intervalo Wilson. Las metas de 80% se refieren exclusivamente a clases y tamaños prefijados. El chequeo adicional con 19.999 permutaciones usa el umbral estricto .05/795; sus resultados son:

| Alternativa | m total/generador | Detecciones | IC95 de potencia |
|---|---:|---:|---|
| localized | 250 | 30/30 | [0.886, 1.000] |
| localized | 500 | 30/30 | [0.886, 1.000] |
| diffuse | 500 | 30/30 | [0.886, 1.000] |

El menor tamaño de la rejilla con límite inferior Wilson de potencia >=.80 al nivel nominal .05 fue el siguiente. Es un resumen empírico de estas leyes específicas y de treinta repeticiones por punto, no un tamaño mínimo teórico ni una garantía simultánea para todas las alternativas. «No acreditado» incluye tasas insuficientes y falta de precisión; no significa imposibilidad.

| Alternativa | MMD antiguo | Omnibus V2 | Clasificador |
|---|---:|---:|---:|
| correlation | >2000 / no acreditado | 100 | 250 |
| correlation_remote | >2000 / no acreditado | 100 | 250 |
| diffuse | 100 | 100 | 250 |
| higher_order | >2000 / no acreditado | 100 | 250 |
| higher_order_remote | >2000 / no acreditado | >2000 / no acreditado | >2000 / no acreditado |
| large_support | 500 | 250 | 500 |
| localized | >2000 / no acreditado | 100 | 250 |
| localized_4 | 250 | 100 | 250 |
| marginal | 500 | 250 | 250 |
| parity_global | >2000 / no acreditado | 100 | 250 |
| rare | >2000 / no acreditado | >2000 / no acreditado | >2000 / no acreditado |
| structural | 100 | 100 | 100 |


large_support repite deliberadamente la ley de marginal con semillas independientes para auditar soporte enorme; no constituye otra clase matemática de alternativas. En sus 340 bancos sintéticos, el máximo número de repeticiones de un vector completo en cada muestra combinada fue 0. large-support-observed-uniqueness.csv conserva el detalle. La detección no requiere colisiones de vectores completos. Las definiciones de todas las leyes se documentan en ROBUST_CONTROL_DISTRIBUTIONS.md.

En las nueve entradas neuronales extendidas, los contrastes adicionales dieron los recuentos siguientes, tras Holm sobre la familia auxiliar completa de 405 hipótesis por procedimiento. Son resultados de prefijos anidados de una sola muestra nueva por celda; **no son estimaciones de potencia** ni observaciones independientes entre tamaños. El análisis principal m=500 tiene su propia familia de 795. No se aumentaron muestras buscando significación.

| m previsto | Arquitectura | Celdas | MMD antiguo | Omnibus nuevo |
|---:|---|---:|---:|---:|
| 250 | DeepSets | 45 | 1 | 5 |
| 250 | NORMALIZED (E1) | 45 | 0 | 3 |
| 250 | LocalPaletteTransformer | 45 | 14 | 12 |
| 1000 | DeepSets | 45 | 6 | 9 |
| 1000 | NORMALIZED (E1) | 45 | 4 | 6 |
| 1000 | LocalPaletteTransformer | 45 | 19 | 17 |
| 2000 | DeepSets | 45 | 11 | 10 |
| 2000 | NORMALIZED (E1) | 45 | 5 | 8 |
| 2000 | LocalPaletteTransformer | 45 | 24 | 20 |


El omnibus no domina cada comparación: en esta cohorte auxiliar, por ejemplo, Transformer a m=2000 tuvo 20 rechazos frente a 24 del MMD antiguo. Estos recuentos de una muestra no estiman una diferencia de potencia, pero muestran resultados donde reservar datos para selección no mejora la decisión. Las pérdidas se conservan junto a las ganancias.

No se promete esa potencia para otras distancias, regiones no representadas, todas las semillas o la familia completa de 795 hipótesis. La probabilidad de un evento raro limita la información disponible. El triple remoto tiene TV=.5 pero un efecto Hamming MMD² de solo 9.3653e-9; el omnibus tampoco busca todas las regiones de tercer orden. Su evento de paridad conocido sí separa las leyes: este fallo es una limitación de representación y potencia del instrumento, no una imposibilidad estadística para ese caso. Los resultados negativos aparecen en todas las tablas, sin reinterpretarlos como igualdad.

## P6. Qué ocurre con los 255 no rechazos históricos de E1

**42 de los 255** pasan a ser significativos con el nuevo procedimiento y muestras independientes. En los mismos bancos nuevos de 500, E1 tiene 21 rechazos por MMD anterior y 52 por omnibus; 31 son rechazos del omnibus sin rechazo por el test antiguo sobre ese mismo banco. e1-transitions.csv identifica exactamente cada celda. El cambio histórico no puede atribuirse solo al procedimiento porque también cambia m; la evidencia de mejora metodológica procede de controles independientes y de las comparaciones con presupuesto igualado.

Ninguno de esos rechazos adicionales de E1 cuenta con un testigo de uno, dos o tres vértices validado en el banco reservado. La mejora frente a diferencias locales está demostrada en los controles; no se ha demostrado que las nuevas detecciones reales de E1 sean locales. Los testigos confirmados de E1 son resúmenes estructurales.

## P7. DeepSets y Transformer

| Arquitectura | Histórico m=250 | MMD antiguo, banco nuevo m=500 | Omnibus nuevo, total m=500 |
|---|---:|---:|---:|
| NORMALIZED (E1) | 10/265 | 21/265 | 52/265 |
| DeepSets | 46/265 | 71/265 | 107/265 |
| LocalPaletteTransformer | 127/265 | 147/265 | 181/265 |

La comparación emparejada de decisiones principales incluye también los casos perdidos por el nuevo método:

| Arquitectura | Ambos rechazan | Solo omnibus | Solo MMD antiguo |
|---|---:|---:|---:|
| NORMALIZED (E1) | 21 | 31 | 0 |
| DeepSets | 69 | 38 | 2 |
| LocalPaletteTransformer | 146 | 35 | 1 |


by-training-checkpoint.csv mantiene los cinco entrenamientos de cada arquitectura separados. Sus diferencias son variación entre políticas fijas, no cinco ejecuciones de una misma política. by-degree-family-size.csv y la figura 6 separan tamaño y familia; no se ajustó una regresión poblacional no planificada ni se infirió una tendencia monótona a partir del número de rechazos.

Un ejemplo descriptivo con n y familia fijos: en los regulares n=512, Transformer tuvo 5/5 rechazos a grado 2, 0/5 a grado 3 y 5/5 a grado 5. Ni siquiera los recuentos de esta rejilla son monótonos. Las curvas del efecto del kernel también dependen de su geometría y potencia; no establecen una tendencia universal de TV con el grado. Cada grado representa un grafo fijo, no una muestra poblacional de grafos.

## P8. Qué información permite distinguirlos

Se seleccionó un evento binario por celda usando exclusivamente el banco primario y se estimaron sus dos probabilidades en 250 ejecuciones reservadas independientes. **188** rechazos globales cuentan además con un testigo significativo tras Holm en la familia explicativa. La figura 7 usa intervalos CP simultáneos sobre las 1.590 probabilidades, de modo que seleccionar ejemplos para la presentación conserva cobertura. De cada intervalo [l,u] se deriva TV>=max(0,l,-u), porque la diferencia de probabilidad de cualquier evento está acotada por TV. Hay 168 cotas inferiores positivas en witness-TV-lower-bounds.csv. Son garantías de tamaño para eventos específicos, no estimaciones exactas de TV en espacios grandes. Los testigos son observables predictivos, no explicaciones causales.

Además, 4 celdas sin rechazo primario tuvieron un evento significativo en el banco independiente. Son evidencia secundaria de la familia explicativa y se conservan en all-independently-significant-events.csv; no se suman a los 340 rechazos primarios. La separación evita ocultar casos que el omnibus tampoco distinguió en su muestra principal.

| Arquitectura | Tipo de evento validado | Celdas | Mediana de diferencia absoluta de probabilidad |
|---|---|---:|---:|
| DeepSets | resumen estructural | 43 | 0.580 |
| NORMALIZED (E1) | resumen estructural | 9 | 0.608 |
| LocalPaletteTransformer | resumen estructural | 127 | 0.784 |
| LocalPaletteTransformer | triple de vértices | 8 | 0.208 |
| LocalPaletteTransformer | vértice individual | 1 | 0.252 |


La tabla clasifica el evento seleccionado y validado en cada celda. Su frecuencia depende de esa selección y no es una descomposición exhaustiva de la discrepancia de cada arquitectura.

| Grafo / checkpoint | Evento | P Luby | P red | Diferencia red−Luby e IC simultáneo |
|---|---|---:|---:|---|
| regular-n512-d100 / normalized-s41000 | used_colors <= 100 | 0.008 | 1.000 | +0.992 [+0.894, +1.000] |
| regular-n512-d50 / normalized-s41000 | class_entropy <= 3.9051062885346037 | 0.224 | 0.832 | +0.608 [+0.370, +0.788] |
| random-n256-d200 / normalized-s41000 | mean_color <= 71.458251953125 | 0.360 | 0.840 | +0.480 [+0.234, +0.681] |
| regular-n512-d200 / deepsets-s41004 | mean_color <= 97.701171875 | 0.072 | 0.920 | +0.848 [+0.664, +0.951] |
| bipartite-n256-d100 / deepsets-s41003 | largest_class_fraction <= 0.054931640625 | 0.948 | 0.152 | -0.796 [-0.915, -0.601] |
| bipartite-n256-d200 / deepsets-s41000 | class_entropy <= 3.5623716110652475 | 0.080 | 0.828 | +0.748 [+0.539, +0.886] |
| regular-n512-d20 / transformer-s41004 | class_entropy <= 3.023576872820173 | 0.000 | 1.000 | +1.000 [+0.913, +1.000] |
| star3 / transformer-s41000 | tuple at [1, 2, 3] belongs to [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1]] | 0.624 | 0.280 | -0.344 [-0.573, -0.082] |
| paw / transformer-s41000 | tuple at [3] belongs to [[0]] | 0.548 | 0.296 | -0.252 [-0.491, +0.011] |

Para la exposición se muestran hasta tres tipos de evento por arquitectura, ordenados por brecha observada; esa selección utiliza las cotas simultáneas de toda la familia. Un rechazo global sin testigo validado no demuestra que la diferencia sea local: puede reflejar acumulación de diferencias débiles o una característica que el banco explicativo no permite estimar con precisión. Las definiciones completas, probabilidades, intervalos y decisiones están en explanatory-witnesses.json. Las comparaciones de coloraciones etiquetadas nunca fueron sustituidas por clases de color o particiones sin nombres.

## P9. Igualdad, diferencia demostrada y equivalencia aproximada

En cliques, un protocolo ideal equivarante por permutaciones de vértices y con aleatoriedad privada iid tiene ley uniforme sobre las n! coloraciones propias, condicionada al éxito. La demostración es por invariancia y transitividad, no por ausencia de rechazo. La ejecución en coma flotante se comprueba mediante pruebas finitas; no es una certificación bit a bit universal. Los controles artificiales de estrella, correlación y paridad tienen diferencias demostradas mediante eventos y TV conocida. Los rechazos neuronales calibrados aportan evidencia estadística sobre cada grafo/checkpoint, no una demostración algebraica de su ley completa.

En siete grafos pequeños se recalcularon leyes exactas de Luby por recursión racional. Para esta referencia se reservaron 20.000 ejecuciones por generador: la cota simultánea de muestreo no supera .02244 antes de añadir los ajustes. Las 105 cotas de TV incluyen el desfase entre absorción eventual y cutoff y una cota de precisión float64. **78/105** celdas satisfacen el criterio prospectivo de equivalencia aproximada TV<=.05 entre las leyes condicionadas a completar por R=128. La figura 9 y small-reference.csv identifican cuáles. La tolerancia .05 no es igualdad exacta, y no se aplica a grafos grandes. No se convierte una MMD pequeña en una cota útil de TV sin una desigualdad adecuada.

| Arquitectura | Casos pequeños con TV<=.05 respaldada | Mayor límite superior de TV entre sus 35 casos |
|---|---:|---:|
| DeepSets | 31/35 | 0.1198 |
| NORMALIZED (E1) | 35/35 | 0.0437 |
| LocalPaletteTransformer | 12/35 | 0.3186 |


En particular, los 35 casos pequeños de E1 satisfacen la tolerancia fijada; eso respalda cercanía en esos siete grafos y esas cinco políticas. Las diferencias estructurales grandes observadas en otros grafos impiden extrapolar esa cercanía a toda la arquitectura.

En 5 casos pequeños se rechaza igualdad exacta y a la vez se respalda TV<=.05: ambas conclusiones son compatibles porque una diferencia pequeña puede ser detectable. La integración exacta de todas las regiones de las redes sobre sus cintas privadas no resultó viable; la referencia neuronal es Monte Carlo con cotas explícitas, no una ley exacta inventada.

## P10. Qué conclusiones sobreviven

La calibración del estudio anterior y sus 183 rechazos no quedan invalidados por su falta de sensibilidad en ciertas alternativas. Lo que cambia es la interpretación de los no rechazos: nunca demostraron semejanza global fuerte. El instrumento nuevo mejora verificablemente la sensibilidad para las clases especificadas, sin elevar el nivel por seleccionar regiones o combinar componentes sin calibración. Los resultados confirmatorios solo sustentan sus grafos y políticas fijas. Los casos de simetría, las diferencias conocidas, las equivalencias aproximadas respaldadas por cotas y los casos de igualdad desconocida se distinguen expresamente.

## Ejecución, auditoría y reproducción

El auditor verificó 888 bancos y 4,469,184 semillas nuevas distintas, sin solapamiento con las 937,600 semillas históricas. Estados [válida, inválida completa, cutoff, error]: [4469180, 0, 4, 0]. Las fases se desglosan en audit.json; no se agrupan controles como si fueran ejecuciones confirmatorias. Consumo GPU-proceso cargado: 6.268/12 horas, con inicialización y sobrecostes conservadores. No se reentrenaron las redes ni se hizo push.

Los bancos con fallos se identifican en la tabla siguiente. Sus intervalos de éxito son binomiales puntuales, no afirmaciones de cobertura simultánea. failure-law-tests.json conserva los contrastes de estados y de la ley incondicional coarsened, con ajuste sobre las 795 hipótesis previstas por familia. Un no rechazo de esos contrastes no establece que las probabilidades de fallo coincidan. La igualdad condicional por simetría en cliques no implica igualdad de las frecuencias de terminación.

| Fase / grafo / modelo | Válidas | Inválidas | Cutoff | Error | IC95 de probabilidad de éxito |
|---|---:|---:|---:|---:|---|
| confirmatory / clique-d200 / transformer-s41001 | 746 | 0 | 4 | 0 | [0.98640, 0.99854] |

La familia primaria nueva, la línea base emparejada, las 405 comparaciones de tamaño, los 795 testigos y los 105 contrastes pequeños tienen ajustes explícitos por separado; no se afirma FWER conjunto .05 al unir todas esas familias. El refinamiento por cotas conserva exactamente el conjunto de rechazos del cálculo conceptual completo con 19.999 permutaciones. Los valores-p censurados no rechazados se marcan como conservadores y retienen su cota inferior; no son estimaciones precisas de una cola.

En los controles extremos hubo 60 probabilidades hipergeométricas exactas subrepresentables como float que aparecen como cero numérico en los datos brutos de clasificación. No son ceros de Monte Carlo ni probabilidades matemáticamente nulas; sus decisiones al 5% no cambian. Los clasificadores confirmatorios usan tamaños menores y sus colas son representables. Todos los valores-p de permutaciones conservan la corrección de una unidad.

Véanse ROBUST_DISTRIBUTIONS.md, el protocolo versionado, las pruebas y los manifiestos. Los datos de confirmación se generaron después del commit que congeló el procedimiento. Los módulos V1 y V2 y los fallos de ambos permanecen disponibles. Los pesos y grafos se identifican por hashes; el archivo reproducible conserva los insumos requeridos.
