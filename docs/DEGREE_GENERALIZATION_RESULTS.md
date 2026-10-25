# Resultados de generalización sin reentrenamiento

Resultados derivados de las filas crudas y sus hashes; cero actualizaciones nuevas. Cada celda muestra checkpoints sin fallos /5 y rango de éxitos por checkpoint. Nunca representa una política promedio. BE queda fuera del contraste de grado.

| Profesor | Arquitectura | global-n256-d4-mixture1000 | global-n64-d10-mixture1000 | global-n64-d3-mixture1000 | global-n64-d4-mixture1000 | local-n64-d4-mixture1000 |
|---|---|---|---|---|---|---|
| COLOR_TRIALS | deepsets | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| COLOR_TRIALS | normalized | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| COLOR_TRIALS | transformer | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| GREEDY | deepsets | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| GREEDY | normalized | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| GREEDY | transformer | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| JOHANSSON | deepsets | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| JOHANSSON | normalized | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| JOHANSSON | transformer | 5/5; 1000–1000/1000 | 4/5; 972–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| JP | deepsets | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| JP | normalized | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| JP | transformer | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| LUBY | deepsets | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| LUBY | normalized | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| LUBY | transformer | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| SPECULATIVE_GM | deepsets | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| SPECULATIVE_GM | normalized | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 |
| SPECULATIVE_GM | transformer | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 5/5; 1000–1000/1000 | 4/5; 819–1000/1000 |

Los 1000 casos de cada condición corresponden a su mezcla de familias, no a cada familia por separado. Los intervalos y denominadores completos por checkpoint están en `checkpoint-condition.csv`. Son intervalos puntuales; 1000/1000 da CP bilateral95 [0.996318,1], sin garantía universal ni simultánea.

**pilot**: 3,526/3,570 ejecuciones correctas, 44 celdas checkpoint/condición con algún fallo. Este total es contabilidad, no estimación de una política combinada.

**confirmatory**: 808,413/810,000 ejecuciones correctas, 15 celdas checkpoint/condición con algún fallo. Este total es contabilidad, no estimación de una política combinada.

## Grado y tamaño independientes

| Profesor | Arquitectura | Mayor Δ con algún éxito piloto | Primer Δ con algún fallo piloto | n1024/Δ10, cinco checkpoints |
|---|---|---|---|---|
| COLOR_TRIALS | deepsets | 20 | 50 | 1,1,1,1,1 (cada uno /1) |
| COLOR_TRIALS | normalized | 50 | 50 | 1,1,1,1,1 (cada uno /1) |
| COLOR_TRIALS | transformer | 20 | 50 | 1,1,1,1,1 (cada uno /1) |
| GREEDY | deepsets | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| GREEDY | normalized | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| GREEDY | transformer | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| JOHANSSON | deepsets | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| JOHANSSON | normalized | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| JOHANSSON | transformer | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| JP | deepsets | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| JP | normalized | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| JP | transformer | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| LUBY | deepsets | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| LUBY | normalized | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| LUBY | transformer | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| SPECULATIVE_GM | deepsets | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| SPECULATIVE_GM | normalized | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |
| SPECULATIVE_GM | transformer | 200 | ninguno observado | 1,1,1,1,1 (cada uno /1) |

El máximo con algún éxito no define un intervalo garantizado de grados: deben consultarse todas las semillas y celdas. El barrido n256 cambia Δ manteniendo n; el barrido Δ3/10 cambia n manteniendo grado y familia. La paleta global Δ+1 también cambia al cambiar grado: no se identifica causalmente grado frente a K.

## Evidencia adicional y límites

`native-21-consolidation.csv` reúne las 21 certificaciones nativas con métodos, cinco semillas, etiquetas, updates, parámetros, dominio, criterio y fuente. `failure-index.json` conserva todos los fallos; `first-failure.json` conserva testigos. Los costes medidos están en los `cost.json` de cada etapa. La comparación contrafactual y la interpretación causal se describen en el análisis de fallos y cierre.

## Ampliación exploratoria de tamaño hasta 4096

Después de la confirmación, otra batería congelada en un namespace separado evalúa los sesenta checkpoints principales de Luby, Johansson, ColorTrials y JP. Son los cuatro profesores paralelos; GM y Greedy conservan su batería ya completada hasta1024 y BE permanece fuera del contraste. Se obtuvieron 240/240 ejecuciones correctas. Cada entrada siguiente es un vector de cinco resultados, uno por checkpoint; cada resultado tiene denominador uno. No es certificación ni parte de las 810000 ejecuciones confirmatorias.

| Profesor | Arquitectura | global-n2048-d10-regular | global-n2048-d3-regular | global-n4096-d10-regular | global-n4096-d3-regular |
|---|---|---|---|---|---|
| COLOR_TRIALS | deepsets | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| COLOR_TRIALS | normalized | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| COLOR_TRIALS | transformer | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| JOHANSSON | deepsets | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| JOHANSSON | normalized | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| JOHANSSON | transformer | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| JP | deepsets | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| JP | normalized | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| JP | transformer | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| LUBY | deepsets | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| LUBY | normalized | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
| LUBY | transformer | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) | 1,1,1,1,1 (cada uno /1) |
