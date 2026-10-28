# Revisión final de fallos de generalización

Fuentes: filas confirmatorias congeladas, `checkpoint-inventory.json`,
`supplemental-checkpoint-inventory.json`, y replays de cierre en
`results/final-closure/failure-replays/`. No se cambia el conjunto evaluado.

## Resultado y método

Hay quince celdas checkpoint–condición con fallos: dos principales y trece
de controles. En cada una se tomó **la primera fila incorrecta del archivo
crudo**, se reconstruyeron grafo y cinta, se conservó todo el batch original
y se ejecutaron todos sus pasos. Los quince replays coinciden con colores,
fallo y pasos originales. Son reproducciones del mismo dato, no quince nuevas
observaciones independientes. `failure-replay-audit.json` contiene SHA-256
completos, semilla, familia, primera acción divergente, fase, escrituras
prohibidas y ruta de cada traza. `failed-cells.json` añade el denominador y
hash del archivo de mil filas. La tabla al final se genera de estos recibos.

## Mecanismos identificados

**Johansson–Transformer, 83101, global Δ10/n64.** 28 cutoffs en mil,
todos en cliques. En el primer caso quedan dos pendientes adyacentes, colores
libres {0,2}, propuesta reiterada 2 y COMMIT que rechaza correctamente la
colisión. Los 8.192 draws por vértice de la sonda conservada vuelven a producir
2. Evidencia fuerte de falta de diversidad observada y pérdida de progreso
en ese estado. No demuestra probabilidad exactamente cero del otro color,
imposibilidad de capacidad ni fallo del veto de conflictos. Primer desacuerdo
de propuestas puede aparecer antes sin ser suficiente para explicar el cutoff.

**GM–Transformer, 94102, local Δ4/n64.** 181 fallos en cada uno de tres
brazos. El primer caso crudo es de familia `random`; otro testigo histórico
era `disconnected`, y no deben confundirse. El primer CF mantiene un prefijo
docente válido hasta reparación, propone el color ya ocupado y COMMIT lo
rechaza correctamente: omite la escritura programada. Como el calendario
no reintenta la reparación, queda conflicto final. Los primeros matched y
passive aceptan una escritura prohibida. En general, no todos los fallos
comparten una única cadena local por tener el mismo total 181.

La sonda `palette-degree-mechanism.json` mantiene grado máximo tres, IDs,
colores de la componente y objetivo first-fit, cambia la lista propia de
cuatro a dos colores y altera el scorer Transformer. Es evidencia de
sensibilidad al contexto de paleta; demuestra que elevar Δ no es necesario
para ese testigo. No separa universalmente capacidad, cobertura y optimización.
Después de una escritura/especulación inválida, los oráculos sobre el estado
autónomo no certifican fidelidad de una trayectoria nativa admisible.

**ColorTrials E1 matched.** Once celdas fallidas pertenecen a 94101/94105
de controles nativos y 94104 de controles a igual hito. Hay errores a Δ3:
el fenómeno no puede atribuirse sólo a extrapolación de grado. En los primeros
casos se observan confirmaciones inconsistentes con la regla estricta y
colisiones simultáneas; algunas propuestas también divergen. El criterio
`availability>0.5` es una sonda de calibración latente, no una etiqueta
supervisada ni una prueba de pérdida informacional. No se asigna causalidad
automática a la primera columna de discrepancias.

**CT de alto grado.** Los 42 cutoffs de recursos del piloto tienen límite
de tiempo añadido sólo a celdas incompletas; otras de Δ50 habían terminado
tras más tiempo. No constituyen una batería homogénea ni no terminación.
La sonda con arista aislada y estrella separada mantiene grado local uno y
paleta {0,1}; el gap normalizado 1/Δ altera COMMIT aunque la comparación
estricta original se conserva. Los quince CF rechazan el ganador desde Δ10.
La codificación legacy pref/3 cambia respuestas sin resolver todas. Es un
testigo de sensibilidad a escala/rango de una política, no pérdida de orden,
fallo por multiplicidades MAX, causa de redondeo ni tasa de fracaso aleatorio.

## Taxonomía y atribución

| Componente | Evidencia | Conclusión permitida |
|---|---|---|
| Participación | comparada en todos los replays | no explica por sí sola los testigos principales |
| Propuesta | colapso Johansson; reparación GM; preferencias CT | fallos de regla/progreso específicos |
| Disponibilidad | alpha y color efectivo disponible | alpha no es máscara exacta ni causa suficiente |
| Comunicación | interfaz conserva predicados; paridad nativa finita | no hay pérdida de información demostrada, tampoco equivalencia universal de kernels |
| Confirmación | falso permiso CT; correcto rechazo GM CF/Johansson | separar falso permiso y falta de progreso |
| Recoloración/auxiliares | GM permite conflictos temporales | no declarar fallo por una colisión intermedia legal |
| Progreso/calendario | rechazo de reparación sin reintento; propuestas coincidentes | fallo de política dentro del calendario suministrado |
| Cutoff | 128 pasos o techo de recursos piloto | fracaso operacional, no no terminación demostrada |
| Precisión | replay mismo batch; márgenes CT grandes | no causa numérica aislada demostrada; neutralidad de otros contextos no probada |

Los seis profesores exactos validaron todas las cintas confirmatorias. Las
reglas tienen pruebas condicionales; los primeros fallos neuronales son de
políticas/interfaces declaradas fuera de entrenamiento, no errores docentes
detectados. Adaptación semántica correcta no garantiza buena normalización
para una función aprendida. BE no es aplicable a esta extensión y no se
contabiliza como fracaso neuronal.

## Inventario exhaustivo

La tabla generada siguiente usa hashes abreviados para lectura; los SHA-256
completos, grafos y cintas están en el JSON indicado. Todos los casos tienen
n64 y mil ejecuciones por celda. La primera familia corresponde a la primera
fila cruda fallida, no al primer diagnóstico guardado por otro proceso.

<!-- GENERATED_FAILURE_TABLE -->

| Checkpoint | SHA-256 abreviado | Condición | Éxitos/1000 | Primera familia | Primera acción (paso/fase) | Mecanismo observado |
|---|---|---|---:|---|---|---|
| `JOHANSSON-transformer-historical-primary-passive-s83101` | `231665deba8f1e73` | `global-n64-d10-mixture1000` | 972 | cliques | 0/0 | propuesta/diversidad; cutoff |
| `SPECULATIVE_GM-transformer-counterfactual-primary-counterfactual-s94102` | `e769226d5ac63f09` | `local-n64-d4-mixture1000` | 819 | random | 32/3 | propuesta ocupada; rechazo correcto; reparación omitida |
| `COLOR_TRIALS-normalized-new-certified-control-matched-s94101` | `353ffba083638410` | `global-n64-d3-mixture1000` | 999 | hubs | 1/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-new-certified-control-matched-s94101` | `353ffba083638410` | `global-n64-d4-mixture1000` | 999 | regular | 1/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-new-certified-control-matched-s94101` | `353ffba083638410` | `local-n64-d4-mixture1000` | 999 | regular | 1/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-new-certified-control-matched-s94105` | `5bd3673235249566` | `global-n64-d3-mixture1000` | 994 | hubs | 1/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-new-certified-control-matched-s94105` | `5bd3673235249566` | `global-n64-d4-mixture1000` | 987 | regular | 1/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-new-certified-control-matched-s94105` | `5bd3673235249566` | `global-n64-d10-mixture1000` | 912 | heterogeneous | 0/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-new-certified-control-matched-s94105` | `5bd3673235249566` | `local-n64-d4-mixture1000` | 993 | regular | 1/0 | confirmación no ganadora/colisión simultánea |
| `SPECULATIVE_GM-transformer-new-certified-control-matched-s94102` | `23c3ebcc7bb106e8` | `local-n64-d4-mixture1000` | 819 | random | 32/3 | escritura prohibida durante reparación |
| `SPECULATIVE_GM-transformer-new-certified-control-passive-s94102` | `2d801c1259c8bf29` | `local-n64-d4-mixture1000` | 819 | random | 32/3 | escritura prohibida durante reparación |
| `COLOR_TRIALS-normalized-equal-update-control-matched-s94104` | `e2c928fc018dc560` | `global-n64-d3-mixture1000` | 864 | bipartite | 1/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-equal-update-control-matched-s94104` | `e2c928fc018dc560` | `global-n64-d4-mixture1000` | 855 | regular | 1/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-equal-update-control-matched-s94104` | `e2c928fc018dc560` | `global-n64-d10-mixture1000` | 547 | irregular | 0/0 | confirmación no ganadora/colisión simultánea |
| `COLOR_TRIALS-normalized-equal-update-control-matched-s94104` | `e2c928fc018dc560` | `local-n64-d4-mixture1000` | 835 | regular | 1/0 | confirmación no ganadora/colisión simultánea |
