# Fallos, límites y mecanismos de transferencia

> Informe histórico del estudio `71153993`. Para el cierre científico vigente,
> véase `GENERALIZATION_FAILURE_REVIEW.md`. Los testigos guardados aquí no son
> necesariamente la primera fila fallida de cada archivo: el cierre reproduce
> explícitamente esas quince primeras filas y distingue mecanismos CF/control.

## Separación de causas

**A, profesor.** Seis variantes tienen pruebas condicionales para ambas
paletas. Las referencias exactas validaron todas las cintas confirmatorias
antes de ejecutar estudiantes. BE no es aplicable fuera de grado tres y
no aporta fallos neuronales artificiales.

**B, interfaz.** Los seis profesores extensibles disponen de información
suficiente y evaluación estructural de pesos compartidos. La forma histórica
K4/vecinos3 se reemplaza explícitamente por una organización dispersa
semánticamente conservadora, contrastada con el envoltorio original.
La interfaz algebraica BE está especializada. No se demuestra capacidad
universal de ningún ancho fijo. Que no exista una pérdida informacional
identificada no descarta todas las limitaciones de representación.

**C, política congelada fuera del dominio aprendido.** Un error sobre un
estado legítimo, con referencia correcta y entradas conservadas, es evidencia
de fallo de esa política. No separa automáticamente cobertura, optimización
y capacidad. Las sondas de CT sí aíslan una dependencia concreta de la
comparación aprendida; los demás mecanismos requieren los contrastes
descritos, sin atribuciones causales por el nombre de la arquitectura.

## Fallos observados

| Etapa | Tipo | Ejecuciones |
|---|---|---|
| confirmatory | collision | 1559 |
| confirmatory | cutoff | 28 |
| pilot | collision | 2 |
| pilot | resource_cutoff | 42 |

En la cohorte principal confirmatoria hubo 0 fallos con Δ3 y 209 con Δ>3. Los controles se conservan por separado; un fallo basal K4 no se interpreta como provocado por aumentar grado.

El piloto incluye 42 cutoffs de recursos CT y dos colisiones reales del control matched/E1/s94105. El matched CT a 4000/s94104 también tiene una regresión nativa documentada. El techo de diez segundos sólo se aplicó a celdas exploratorias densas CT aún no completadas; los resultados anteriores largos permanecen, por lo que esos cutoffs no son comparables como una tasa homogénea. Ninguno demuestra no terminación matemática. Las discrepancias de acciones anteriores al cutoff sí son testigos de falta de fidelidad.

Todos los fallos están en `failure-index.json`, con checkpoint, semilla, grado, tamaño, familia y ruta de filas/traza. El primer componente discrepante es una observación temporal. La discordancia alpha>.5 es calibración de una variable latente, no una etiqueta supervisada ni un requisito demostrado de corrección: los scorers pueden usar h1 o compensar otra parametrización de alpha. La confirmación añade la primera divergencia de acción y guarda ambos estados cuando procede. Las métricas de regla sobre estados autónomos no equivalen a fidelidad de toda una trayectoria docente; después de un historial GM inválido no se usan como prueba de fidelidad nativa.

| Etapa | Profesor | Arq. | Método | Semilla | Δ | n | Familia | Resultado | Primer componente |
|---|---|---|---|---|---|---|---|---|---|
| confirmatory | COLOR_TRIALS | normalized | matched | 94104 | 3 | 64 | bipartite | collision | availability |
| confirmatory | SPECULATIVE_GM | transformer | counterfactual | 94102 | 4 | 64 | disconnected | collision | availability |
| confirmatory | SPECULATIVE_GM | transformer | matched | 94102 | 4 | 64 | disconnected | collision | availability |
| confirmatory | SPECULATIVE_GM | transformer | passive | 94102 | 4 | 64 | disconnected | collision | availability |
| confirmatory | JOHANSSON | transformer | passive | 83101 | 10 | 64 | cliques | cutoff | proposal |
| pilot | COLOR_TRIALS | normalized | matched | 94105 | 4 | 64 | regular | collision | proposal |
| pilot | COLOR_TRIALS | deepsets | counterfactual | 94101 | 50 | 256 | regular | resource_cutoff | proposal |
| pilot | COLOR_TRIALS | transformer | counterfactual | 94101 | 50 | 256 | regular | resource_cutoff | proposal |
| pilot | COLOR_TRIALS | normalized | counterfactual | 94104 | 50 | 256 | regular | resource_cutoff | proposal |

## Sonda que separa grado local, paleta y preferencias CT

Se construye una arista v–u y, en otra componente, una estrella de grado Δ. En régimen local, ambos extremos de la arista tienen siempre grado uno y paleta {0,1}. Con todos pendientes, pref_v(0)=Δ, pref_u(0)=Δ−1 y las restantes preferencias cero, v gana estrictamente color cero. Son intervenciones privadas discretas válidas para la regla extendida; no se afirma haber encontrado una semilla PCG64 que emita el vector completo construido. La normalización conservadora convierte la diferencia en 1/Δ.

| Δ | Paleta receptor | Rechazos del ganador exacto /15 | Rango logits COMMIT |
|---|---|---|---|
| 3 | 2 | 0 | [8.3122,12.9102] |
| 4 | 2 | 0 | [3.1333,8.1634] |
| 5 | 2 | 0 | [0.0259,3.9414] |
| 10 | 2 | 15 | [-10.0598,-3.4085] |
| 20 | 2 | 15 | [-13.5985,-3.5705] |
| 50 | 2 | 15 | [-15.0427,-3.6176] |
| 100 | 2 | 15 | [-15.3804,-3.6333] |
| 200 | 2 | 15 | [-15.5222,-3.6412] |

Los quince pesos CF rechazan el ganador desde Δ10 en esta sonda, aun forzando explícitamente la consulta de color cero. El error persiste con un único vecino y paleta propia constante: no lo explica perder multiplicidades en MAX ni recibir demasiados vecinos. Las preferencias normalizadas más próximas cambian la respuesta de COMMIT, y las magnitudes de los logits negativos son mucho mayores que las discrepancias numéricas observadas en los replays. No es una prueba universal de robustez numérica, pero tampoco evidencia de saturación o redondeo que borre el dato de entrada. Es un testigo de fidelidad de componentes, **no una estimación del fracaso funcional aleatorio**, pues futuras rondas pueden resolver el grafo.

La elección de normalización es explícita: el adaptador principal conserva el template pref/Δ (pref/3 en el dominio nativo). Una sonda posterior mantiene unidades antiguas pref/3 mediante una transformación común sólo de la observación, sin cambiar profesor ni pesos. Con gap crudo1, ese segundo embedding acepta 13/15 ganadores a Δ5, 5/15 a Δ10 y0/15 a Δ50/200, frente a15/15,0/15,0/15 en el principal. Ambos preservan el orden exacto; ninguno resuelve todos los casos medidos. No se escoge un embedding por resultado ni se reescriben las evaluaciones principales. La conclusión es sensibilidad a la escala/rango de las preferencias bajo estos embeddings, no una imposibilidad informacional ni una afirmación sobre todo adaptador posible. El contraste completo está en `ct-normalization-sensitivity.json`.

## Contrafactual frente a matched al mismo hito

Sondas exploratorias sobre prefijos docentes exactos: 120 checkpoints y 838188 consultas. CT se compara a 4000 updates usando los treinta controles adicionales; GM a 500. La siguiente tabla muestra la media descriptiva de las cinco diferencias apareadas en puntos porcentuales y su rango entre semillas, sin promediar políticas ni fingir independencia de consultas que comparten estado.

| Profesor | Arquitectura | Δ3 global | Δ10 global | Δ200 global |
|---|---|---|---|---|
| COLOR_TRIALS | normalized | +19.49 [+13.99,+25.00] | +13.19 [+10.47,+15.43] | +17.99 [+14.99,+22.03] |
| COLOR_TRIALS | deepsets | +17.94 [+16.67,+20.00] | +12.51 [+10.17,+14.58] | +16.38 [+15.16,+17.75] |
| COLOR_TRIALS | transformer | +17.28 [+15.48,+20.00] | +12.63 [+11.05,+13.99] | +16.17 [+11.26,+18.37] |
| SPECULATIVE_GM | normalized | +32.36 [+26.47,+39.47] | +29.12 [+25.00,+31.94] | +36.16 [+32.35,+40.62] |
| SPECULATIVE_GM | deepsets | +32.36 [+26.47,+39.47] | +29.12 [+25.00,+31.94] | +36.16 [+32.35,+40.62] |
| SPECULATIVE_GM | transformer | +32.36 [+26.47,+39.47] | +29.12 [+25.00,+31.94] | +36.16 [+32.35,+40.62] |

CF mejora estas respuestas frente a matched, pero la ventaja no crece monótonamente con Δ. Los denominadores positivos/negativos y diferencias por checkpoint están en `counterfactual-probes/checkpoint-metrics.csv` y `paired-equal-update.csv`. False accept en esa tabla significa discrepancia de la **cabeza COMMIT**; si participación es falsa no se ejecutaría una escritura insegura. Las propuestas/query se intervienen sólo aquí y jamás corrigen un rollout funcional. Se controlan inicialización, minibatches nativos, arquitectura y updates; CF añade consultas y cambia la pérdida/distribución. No se identifica una mejora causal de la naturaleza contrafactual a igual presupuesto total de etiquetas, ni una garantía de seguridad a cualquier grado.

## Johansson/Transformer: progreso y diversidad de propuestas

La política histórica s83101 conserva 28 cutoffs de 1000 en n64/Δ10/global, todos en la familia de cliques. El replay conserva **todo el batch original** y reproduce colores, fallo y 128 pasos, evitando cambiar el contexto numérico del testigo. Al final quedan dos vértices adyacentes con colores disponibles {0,2}. Ambos proponen color2 en las últimas rondas; COMMIT los rechaza correctamente porque Johansson está siempre activo. Una sonda con 8192 draws nuevos por vértice produce color2 en todos ellos, frente al reparto uniforme de la propuesta exacta entre dos colores. No se demuestra que la masa del otro color sea exactamente cero. El testigo sitúa la pérdida de progreso en la diversidad de propuestas de ese estado, sin convertir un COMMIT correcto en fallo de detección. No hay pérdida del conjunto disponible en la interfaz; sigue abierta la separación de cobertura/optimización frente a capacidad del scorer de atención. La traza y los histogramas están en `failure-replays/JOHANSSON-transformer-historical-primary-passive-s83101/`.

GM sólo exige permisos/paleta en el gate histórico y admite conflictos temporales. Los replays adicionales registran también conflictos entre slots diferentes, omisiones de escrituras programadas y cobertura de detección; no se presentan esas propiedades adicionales como comprobadas para todas las 810000 ejecuciones. Una reparación fortuita o una política final correcta no prueba que se hayan conservado todos los invariantes intermedios del algoritmo exacto.

En el primer testigo GM/Transformer/local/s94102, el paso32 es REPAIR de un vértice con lista {0,1}; la referencia exige escribir color1 y los tres procedimientos proponen color0. Sus scores y alphas son idénticos en ese estado. Matched/pasivo aceptan el color prohibido con logits14.2269/15.2735; CF lo rechaza con logit−13.6503. El rechazo CF evita esa escritura, pero conserva el conflicto previo y el turno de reparación ya se consume. Los tres fallan las mismas 181 instancias locales. Este replay distingue la propuesta incorrecta, la mejor respuesta CF y la insuficiencia de esa mejora para el calendario fijo.

Un segundo control construye una arista, una estrella de grado3 y dos aislados (n8), con IDs únicos y bloques coherentes que sitúan la arista en el mismo slot. Sobre prefijos exactos de GM, la reparación exige color1 bajo globalK4 y bajo la lista local {0,1}. Los tres checkpoints Transformer/s94102 proponen correctamente1 con globalK4 y erróneamente0 con localK2; CF lo rechaza y los otros lo aceptan. Las otras cuatro semillas Transformer y los quince pesos de cada E1/DeepSets eligen1 en ambos regímenes. Este caso válido a **grado máximo tres** separa el cambio de paleta del incremento de grado: los 181 fallos locales no prueban que el grado4 por sí solo sea la causa. Se conserva el vector de IDs y todas las respuestas en `palette-degree-mechanism.json`; es una sonda elegida después del fallo, no una nueva certificación.

## Certificados de COMMIT y garantías que siguen abiertas

Se reutiliza el certificador histórico de intervalos exteriores en 45 núcleos canónicos Luby/Johansson/Greedy. El canal de mensajes binarios tiene seis tipos legales y 64 soportes; MAX no depende de sus multiplicidades. Los certificados conciernen al receptor pendiente que participa y consulta un color, con gather de una sola posición. No se aplican ignorando residuales nativos en JP, CT, BE o GM.

| Profesor | Arquitectura | Regla exacta /5 | Sin falsa aceptación /5 | Nº soportes inseguros certificados |
|---|---|---|---|---|
| LUBY | normalized | 0 | 0 | 16–20 |
| LUBY | deepsets | 0 | 0 | 16 |
| LUBY | transformer | 0 | 0 | 16–24 |
| JOHANSSON | normalized | 0 | 0 | 8–16 |
| JOHANSSON | deepsets | 0 | 0 | 8–16 |
| JOHANSSON | transformer | 0 | 0 | 6–12 |
| GREEDY | normalized | 0 | 0 | 48 |
| GREEDY | deepsets | 0 | 0 | 48 |
| GREEDY | transformer | 0 | 0 | 48 |

Ninguno de estos 45 pesos pasivos satisface el criterio suficiente de COMMIT exacto o sin falsa aceptación en todos los soportes, incluso bajo el modelo de error float32 del certificador. Se conservan intervalos positivos en soportes que deberían rechazarse. Eso es un contraejemplo de la cabeza bajo intervenciones, no un contraejemplo automático de su ejecución autónoma: la política de propuestas y el calendario pueden evitar tales soportes. En Greedy sólo hay un permiso, por lo que soportes con otros participantes no son alcanzables en su calendario. Los soportes con vecino fijo del mismo color también prueban una limitación condicional, sin demostrar que el propio proponente lo elija.

La proposición de seguridad aprendida sigue siendo condicional: si las propuestas pertenecen a su lista y cada escritura cumple disponibilidad y exclusión de ofertas iguales, la coloración permanente es propia por inducción. Para terminación se requiere además una cota positiva de progreso uniforme en todos los estados alcanzables. Validar infinitamente muchos grados o demostrar esa cota no se sustituye por estos finales finitos. Los intervalos reales encierran operaciones por redondeo exterior float64; el certificado numérico float32 asume la cota gamma para affine, IEEE, ausencia de overflow/TF32 y MAX de mensajes no negativos. No es una verificación formal de todos los kernels del hardware.

## Funcionalidad apareada a igual hito

Vectores en orden s94101–s94105: diferencia CF menos matched de éxitos por 1000, con los mismos grafos y cintas. No se mezclan las cinco políticas. CT usa 4000 updates en ambos brazos; GM 500. Las etiquetas totales siguen siendo diferentes. Los controles certificados de hitos anteriores se conservan además en las filas crudas.

| Profesor | Arquitectura | Δ3 global | Δ4 global | Δ10 global | Δ4 local |
|---|---|---|---|---|---|
| COLOR_TRIALS | normalized | 0,0,0,136,0 | 0,0,0,145,0 | 0,0,0,453,0 | 0,0,0,165,0 |
| COLOR_TRIALS | deepsets | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 |
| COLOR_TRIALS | transformer | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 |
| SPECULATIVE_GM | normalized | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 |
| SPECULATIVE_GM | deepsets | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 |
| SPECULATIVE_GM | transformer | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 | 0,0,0,0,0 |

La ventaja funcional CT/E1 del hito4000 se concentra en s94104 y ya existe a grado tres: no debe presentarse toda la diferencia de grado diez como una mejora nueva causada por extrapolar grado. En GM/Transformer, los 181 fallos locales de s94102 son compartidos por CF, matched y pasivo, aunque sus COMMIT difieren en el replay. Mejor fidelidad condicional no implica una mejora funcional en todas las combinaciones.
