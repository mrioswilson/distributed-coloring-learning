# Aprendizaje multiteacher: reconstrucción, controles y resultados

## Resultado principal y alcance

Se cerraron las **nueve combinaciones pendientes** de ColorTrials, BE y GM en
E1 NORMALIZED, DeepSets y LocalPaletteTransformer: **cinco entrenamientos y
1000/1000 ejecuciones independientes por checkpoint** en cada punto exitoso.
Las doce certificaciones históricas se conservan. La cobertura es por tanto
**21/21 combinaciones**, combinando evidencia histórica y nuevas baterías;
no se presenta como una única cohorte homogénea ni se cambia el resultado
histórico primario de 12/21.

El tratamiento concordante (`matched`) pasó las nueve celdas nuevas con
como máximo 1500 updates en BE/ColorTrials y 500 en GM; ColorTrials/E1 ya
pasa a 500. La supervisión contrafactual también
certifica las nueve en los puntos de la tabla siguiente, tras controles de
optimización independientes. Se conservan todos los fallos de los puntos
anteriores. No hubo shields, reparación externa ni COMMIT del profesor en
inferencia. Los ocho archivos fuente del profesor/interfaz/inferencia
contrastados contra `61f39760` permanecen idénticos.

| Profesor | Arquitectura | Updates | Etiquetas nativas/semilla | Etiquetas totales/semilla | Final |
|---|---|---|---|---|---|
| BE_BASE_D3 | E1 NORMALIZED | 4000 | 3,000 | 6,776 | 5/5 × 1000/1000 |
| BE_BASE_D3 | DeepSets | 4000 | 3,000 | 6,776 | 5/5 × 1000/1000 |
| BE_BASE_D3 | LocalPaletteTransformer | 4000 | 3,000 | 6,776 | 5/5 × 1000/1000 |
| COLOR_TRIALS | E1 NORMALIZED | 4000 | 30,000 | 70,000 | 5/5 × 1000/1000 |
| COLOR_TRIALS | DeepSets | 4000 | 30,000 | 70,000 | 5/5 × 1000/1000 |
| COLOR_TRIALS | LocalPaletteTransformer | 4000 | 30,000 | 70,000 | 5/5 × 1000/1000 |
| SPECULATIVE_GM | E1 NORMALIZED | 500 | 3,000 | 6,104–6,120 | 5/5 × 1000/1000 |
| SPECULATIVE_GM | DeepSets | 500 | 3,000 | 6,104–6,120 | 5/5 × 1000/1000 |
| SPECULATIVE_GM | LocalPaletteTransformer | 500 | 3,000 | 6,104–6,120 | 5/5 × 1000/1000 |

Las cinco semillas nuevas son 94101–94105, con inicialización, datos y
minibatches reproducibles. El final nuevo usa grafos simples no dirigidos de
grado máximo 3, paleta global K4, tamaños 32/48/64 y diez estratos de 100 casos
por política. Training usa N8/12/16/24 y development N28/36. BE conserva su
aritmética y fases suministradas; GM su calendario nativo y reparación
aprendida. Cada política PASS tiene cota unilateral puntual de éxito
0.99700875 bajo el modelo de independencia del protocolo. No se afirma una
garantía simultánea sobre todas las políticas ni corrección universal.

## Cronología científica y resultados negativos

La infraestructura y el protocolo se comprometieron localmente en `3d9f6456`.
Una enmienda anterior al nuevo screen y a toda confirmación amplió development
de 60 a 300 casos por semilla. Se congelaron procedimientos y hashes antes de
cada final. La primera cohorte incluye los tres tratamientos y todas las
celdas; dos seguimientos exploratorios comparan los checkpoints ya existentes
de 1500/4000 updates sobre **nuevas baterías**, después de observar fallos
contrafactuales del primer final. No se entrenó sobre casos finales ni se
repitió una batería para mejorar su resultado.

La validación conserva 195,000 ejecuciones finales,
194,979 correctas y
170 checkpoints finales distintos;
algunos checkpoints se evalúan en dos namespaces y no se cuentan como nuevos
entrenamientos. Los fallos BE contrafactuales a 1500 se concentran en la quinta
semilla y comparten la cabeza COMMIT entre arquitecturas. Los testigos
reproducidos con el contexto numérico original localizan omisiones de una
recoloración KW obligatoria, que dejan el auxiliar fuera del bound nuevo.
ColorTrials conserva un fallo inicial de E1 a 500 y dos fallos del
Transformer a 1500 en el seguimiento independiente.
Los mismos bancos de etiquetas, con 4000 updates, pasan el seguimiento
independiente. Un brazo concordante de ColorTrials/E1 además regresa de 60/60
a 43/60 dev entre 1500 y 4000: más optimización no es uniformemente mejor.
Una errata conservada en `audit/ct-followup-rationale-erratum.json` corrige el
texto del campo `reason` del plan CT v2: decía 1500 para el fallo inicial E1,
cuando el resultado y checkpoint originales registran 500. El seguimiento
sí compara 1500/4000 y no vuelve a probar 500. El plan erróneo se conserva;
no se cambian pesos, baterías, filas ni gates.

| Profesor | Arquitectura | Tratamiento | Updates | Semillas PASS | Éxitos/5000 | Protocolo |
|---|---|---|---|---|---|---|
| BE_BASE_D3 | deepsets | counterfactual | 1500 | 4/5 | 4995 | BE v2 |
| BE_BASE_D3 | deepsets | counterfactual | 4000 | 5/5 | 5000 | BE v2 |
| BE_BASE_D3 | normalized | counterfactual | 1500 | 4/5 | 4995 | BE v2 |
| BE_BASE_D3 | normalized | counterfactual | 4000 | 5/5 | 5000 | BE v2 |
| BE_BASE_D3 | transformer | counterfactual | 1500 | 4/5 | 4995 | BE v2 |
| BE_BASE_D3 | transformer | counterfactual | 4000 | 5/5 | 5000 | BE v2 |
| BE_BASE_D3 | deepsets | counterfactual | 1500 | 4/5 | 4999 | inicial |
| BE_BASE_D3 | deepsets | matched | 1500 | 5/5 | 5000 | inicial |
| BE_BASE_D3 | deepsets | passive | 1500 | 5/5 | 5000 | inicial |
| BE_BASE_D3 | normalized | counterfactual | 1500 | 4/5 | 4999 | inicial |
| BE_BASE_D3 | normalized | matched | 1500 | 5/5 | 5000 | inicial |
| BE_BASE_D3 | normalized | passive | 1500 | 5/5 | 5000 | inicial |
| BE_BASE_D3 | transformer | counterfactual | 1500 | 4/5 | 4999 | inicial |
| BE_BASE_D3 | transformer | matched | 1500 | 5/5 | 5000 | inicial |
| BE_BASE_D3 | transformer | passive | 1500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | deepsets | counterfactual | 1500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | deepsets | matched | 1500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | deepsets | passive | 1500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | normalized | counterfactual | 500 | 4/5 | 4999 | inicial |
| COLOR_TRIALS | normalized | matched | 500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | normalized | passive | 1500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | transformer | counterfactual | 1500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | transformer | matched | 1500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | transformer | passive | 1500 | 5/5 | 5000 | inicial |
| COLOR_TRIALS | deepsets | counterfactual | 1500 | 5/5 | 5000 | CT v2 |
| COLOR_TRIALS | deepsets | counterfactual | 4000 | 5/5 | 5000 | CT v2 |
| COLOR_TRIALS | normalized | counterfactual | 1500 | 5/5 | 5000 | CT v2 |
| COLOR_TRIALS | normalized | counterfactual | 4000 | 5/5 | 5000 | CT v2 |
| COLOR_TRIALS | transformer | counterfactual | 1500 | 4/5 | 4998 | CT v2 |
| COLOR_TRIALS | transformer | counterfactual | 4000 | 5/5 | 5000 | CT v2 |
| SPECULATIVE_GM | deepsets | counterfactual | 500 | 5/5 | 5000 | inicial |
| SPECULATIVE_GM | deepsets | matched | 500 | 5/5 | 5000 | inicial |
| SPECULATIVE_GM | deepsets | passive | 500 | 5/5 | 5000 | inicial |
| SPECULATIVE_GM | normalized | counterfactual | 500 | 5/5 | 5000 | inicial |
| SPECULATIVE_GM | normalized | matched | 500 | 5/5 | 5000 | inicial |
| SPECULATIVE_GM | normalized | passive | 500 | 5/5 | 5000 | inicial |
| SPECULATIVE_GM | transformer | counterfactual | 500 | 5/5 | 5000 | inicial |
| SPECULATIVE_GM | transformer | matched | 500 | 5/5 | 5000 | inicial |
| SPECULATIVE_GM | transformer | passive | 500 | 5/5 | 5000 | inicial |

## Respuestas explícitas P1–P14

**P1. ¿Qué hacía M6B?** Usaba M4B (8967 parámetros a K4), no E1. Conservaba
grafo, colores y entradas privadas de la ronda; intervenía P/C y recalculaba
COMMIT sobre todas las ofertas efectivamente presentes. Tras diez épocas
docentes: P + C + .5 COMMIT propio + .5 COMMIT contrafactual. Las inyecciones
históricas eran .17 por arista fija→pendiente y .04 por arista entre
pendientes. Se autenticaron los 21 artefactos, el CSV de 500 casos y 30
ejecuciones con los pesos originales. 249/500, 476/500 y 500/500 corresponden
a una sola semilla 10001. El control propio y CF tienen P/mensaje1/C idénticos;
sólo difieren mensaje2/COMMIT. Se regeneraron 18877 rondas y 248122 decisiones
pendientes; época 53 implica 7844 updates, y el horizonte configurado 17760.
La pérdida pasiva del estudio posterior aplica la etiqueta COMMIT docente a
propuestas aprendidas que pueden diferir; `matched` corrige esa concordancia.

**P2. ¿Qué se transfiere?** Reconsulta de etiquetas tras intervenciones,
supervisión por fases, gradientes detenidos a través de decisiones duras,
contrastes propios/perturbados y trazas verificables. Cada profesor tiene un
contrato de estados y dependencias. Las ofertas Luby/Johansson, preferencias
y dos mensajes ColorTrials, órdenes JP, puntos Linial/opciones KW, historia
especulativa GM y orden Greedy se tratan individualmente. Hay 162 contextos
difíciles adicionales con replay de preferencias/draws/IDs y sus descendientes.
Se distingue una decisión nativa de una continuación defensiva consultada
sobre mensajes fuera de la emisión canónica; no se llama trayectoria docente
a cualquier array procesable. Los tres núcleos consumen la misma infraestructura.

**P3. ¿Mejora CF ColorTrials, BE y GM?** Mejora la respuesta en continuaciones
contrafactuales; no mejora uniformemente el gate funcional al mismo número
de updates. Los tres brazos cierran GM a 500. El concordante y el pasivo
cierran BE y ColorTrials a 500 o 1500 según la celda; algunos CF necesitan 4000 en sus nuevas
confirmaciones. Esto conserva evidencia positiva y negativa. CF no es
necesario en todas las celdas para resolver este benchmark, y sí aporta
supervisión de componentes que no está cubierta por la emisión nativa.

**P4. ¿Qué causa las mejoras?** Las curvas 500/1500/4000 dentro de cada brazo
usan los mismos datos adquiridos, inicialización y secuencia de minibatches.
Los seguimientos CF comparan 1500/4000 con idénticas etiquetas. El replay de
BE/E1 a 1500 reproduce exactamente los pesos originales de las cinco semillas.
No hubo cambios de arquitectura/representación en los controles principales.
`matched` frente a CF cambia distribución y cantidad de etiquetas; no identifica
una ventaja muestral a igual presupuesto total. La ablación GM sin ampliación
de cobertura también pasa sus cinco semillas en development, por lo que
estos controles no demuestran una ventaja causal aislada de añadir aislados.
El caso histórico 83105 y las 30 regresiones nuevas quedan separados del final.

**P5. ¿Qué explica Pedro y qué se reproduce?** Se verificaron sus 16 hashes
fuente. Sus cinco 1980/1980, JP 1631/1980 y GM 1951/1980 son resultados de
cinco inicializaciones por 396 casos publicados, con un pool de datos común
y tests reutilizados. Usó 17760 updates de 1680 raíces y confirmaciones
propias/CF; la GNN directa, los adaptadores, la optimización y evaluación
también difieren. No apareció ninguno de los 40 checkpoints esperados entre
6614 archivos locales ni en los manifiestos de las releases inspeccionadas.
Por ello sus políticas exactas siguen sin reevaluación independiente. Nuestros
resultados son nuevos entrenamientos y reproducen mecanismos y reglas
implementadas, no sus pesos ni su batería hasta N10240.

**P6. ¿Se incorporan sus mecanismos?** Sí: concordancia de COMMIT, bancos
contrastivos, calendario/arimética explícitos y supervisión por fases se usan
sin sustituir los núcleos. El piloto adicional `owncf` porta la pérdida propia
+ CF a los siete profesores con una semilla nueva y cobra cada consulta
propia distinta. Sus resultados no se llaman certificación; BE sigue siendo
difícil en ese piloto a 1500, mostrando que el calendario de aprendizaje
necesita adaptación por profesor.

| Profesor | Updates | Development /60 | Consultas propias nuevas | Etiquetas totales |
|---|---|---|---|---|
| BE_BASE_D3 | 1500 | 22 | 1036 | 7812 |
| COLOR_TRIALS | 1500 | 60 | 11403 | 81403 |
| GREEDY | 1500 | 60 | 1000 | 8000 |
| JOHANSSON | 1500 | 60 | 2141 | 9141 |
| JP | 1500 | 60 | 1000 | 8000 |
| LUBY | 1500 | 60 | 2717 | 9717 |
| SPECULATIVE_GM | 1500 | 60 | 780 | 6900 |

**P7. ¿Tres arquitecturas por siete profesores?** Sí, bajo los gates finitos
declarados: 12 celdas históricas preservadas y 9 nuevas con cinco políticas
congeladas y 1000 éxitos cada una. También existe un conjunto nuevo CF de
las nueve pendientes: BE/ColorTrials a 4000 y GM a 500. No se deduce que
cualquier nueva semilla, grafo o tamaño pasará, ni una certificación homogénea
de 21 celdas entrenadas todas en esta campaña.

**P8. ¿Qué sigue siendo difícil?** KW conserva errores de progreso a 1500
en CF: una recoloración omitida viola el bound aunque no haya colisión.
ColorTrials conserva contextos raros que pasan development pero fallan en el
final de algunos checkpoints, y una política exitosa difiere bajo cintas
compartidas en path8. Los pilotos propios y la transferencia no se convierten
en éxitos por agregar reparaciones. No queda una de las nueve celdas sin un
punto funcional certificado, pero sí quedan fallos de otros puntos y
fidelidad/generalización sin garantía universal.

**P9. ¿Etiquetas y recursos?** Se separan nativas, CF, propias nuevas,
presentaciones, actualizaciones y simulación. Los bancos principales se
comparten entre arquitecturas y tratamientos. Los prefijos de supervisión
son idénticos al banco mayor y no cobran etiquetas otra vez. Se realizaron
278 ejecuciones computacionales de entrenamiento,
incluidos pilotos, controles apareados y replays, con
683,504 updates y
2.580 horas CPU de optimización.
El entrenamiento usó cero tiempo GPU; el replay histórico M6B CUDA se
contabiliza aparte. El máximo RSS registrado de un worker fue
3158.3 MiB. Los tiempos por proceso
se solapan; no se suman como duración transcurrida. `compute-cost.json`
conserva el detalle y las magnitudes no medidas. No se afirma eficiencia
muestral frente al estudio previo al cambiar a la vez optimización y datos.

| Arquitectura CT | B nativo | Updates | Etiquetas totales | Semillas 60/60 dev |
|---|---|---|---|---|
| normalized | 1000 | 500 | 2,331 | 4/5 |
| normalized | 1000 | 1500 | 2,331 | 5/5 |
| normalized | 3000 | 500 | 7,000 | 3/5 |
| normalized | 3000 | 1500 | 7,000 | 5/5 |
| normalized | 10000 | 500 | 23,331 | 4/5 |
| normalized | 10000 | 1500 | 23,331 | 5/5 |
| normalized | 30000 | 500 | 70,000 | 5/5 |
| normalized | 30000 | 1500 | 70,000 | 5/5 |
| deepsets | 1000 | 500 | 2,331 | 3/5 |
| deepsets | 1000 | 1500 | 2,331 | 5/5 |
| deepsets | 3000 | 500 | 7,000 | 4/5 |
| deepsets | 3000 | 1500 | 7,000 | 5/5 |
| deepsets | 10000 | 500 | 23,331 | 4/5 |
| deepsets | 10000 | 1500 | 23,331 | 5/5 |
| deepsets | 30000 | 500 | 70,000 | 5/5 |
| deepsets | 30000 | 1500 | 70,000 | 5/5 |
| transformer | 1000 | 500 | 2,331 | 1/5 |
| transformer | 1000 | 1500 | 2,331 | 3/5 |
| transformer | 3000 | 500 | 7,000 | 0/5 |
| transformer | 3000 | 1500 | 7,000 | 5/5 |
| transformer | 10000 | 500 | 23,331 | 3/5 |
| transformer | 10000 | 1500 | 23,331 | 5/5 |
| transformer | 30000 | 500 | 70,000 | 3/5 |
| transformer | 30000 | 1500 | 70,000 | 5/5 |

**P10. ¿Interfaces e IDs?** El régimen A conserva anonimato en Luby,
Johansson y ColorTrials. JP/GM usan órdenes coherentes; BE fases y aritmética;
GM/Greedy calendarios declarados. El control JP mantiene arquitectura,
parámetros, inicialización, etiquetas, batches y 500 updates, variando sólo
la representación local de IDs. Las comparaciones relativas pasan 60/60 en
las cinco semillas; retirar IDs o usar representaciones numéricas no alcanza
ese punto. Esto identifica sensibilidad al presupuesto y representación,
no incapacidad matemática de procesar IDs numéricos.

| Representación JP | Éxitos development por semilla /60 |
|---|---|
| without_ids | 6, 6, 6, 6, 6 |
| numeric_ids | 13, 11, 6, 26, 15 |
| numeric_differences | 16, 15, 19, 31, 20 |
| relative_comparisons | 60, 60, 60, 60, 60 |

**P11. ¿Fidelidad, transiciones, rondas y leyes?** El diagnóstico apareado a
4000 cubre todas las raíces permitidas de 60 trayectorias docentes por semilla,
con denominadores separados para acciones, transiciones reales, P/C,
confirmaciones propias y cuatro consultas CF nuevas por raíz. Las tasas se
describen; las decisiones correlacionadas no inflan el número de réplicas.
Las 45 políticas certificadas tienen además sus auditorías de componentes
y transiciones en `certified-policy-fidelity.json`, usando sus updates reales
(incluido GM a 500). El cuadro siguiente mantiene 4000 para la comparación
controlada de los tres tratamientos. BE/GM tienen cabezas de confirmación idénticas entre arquitecturas con
ofertas fijas y semillas/batches apareados; ColorTrials acopla esa etapa al
selector. Las rondas nativas y las coloraciones finales se conservan por caso.

| Profesor | Arquitectura | Tratamiento a 4000 | Acciones correctas/total | CF correctos/total |
|---|---|---|---|---|
| BE_BASE_D3 | normalized | passive | 144000/144000 | 113021/499200 |
| BE_BASE_D3 | normalized | matched | 144000/144000 | 46269/499200 |
| BE_BASE_D3 | normalized | counterfactual | 144000/144000 | 499166/499200 |
| BE_BASE_D3 | deepsets | passive | 144000/144000 | 130030/499200 |
| BE_BASE_D3 | deepsets | matched | 144000/144000 | 46269/499200 |
| BE_BASE_D3 | deepsets | counterfactual | 144000/144000 | 499166/499200 |
| BE_BASE_D3 | transformer | passive | 144000/144000 | 68618/499200 |
| BE_BASE_D3 | transformer | matched | 144000/144000 | 46269/499200 |
| BE_BASE_D3 | transformer | counterfactual | 144000/144000 | 499166/499200 |
| SPECULATIVE_GM | normalized | passive | 19859/19859 | 30926/41036 |
| SPECULATIVE_GM | normalized | matched | 19859/19859 | 30926/41036 |
| SPECULATIVE_GM | normalized | counterfactual | 19859/19859 | 41036/41036 |
| SPECULATIVE_GM | deepsets | passive | 19859/19859 | 30926/41036 |
| SPECULATIVE_GM | deepsets | matched | 19859/19859 | 30926/41036 |
| SPECULATIVE_GM | deepsets | counterfactual | 19859/19859 | 41036/41036 |
| SPECULATIVE_GM | transformer | passive | 19858/19859 | 30926/41036 |
| SPECULATIVE_GM | transformer | matched | 19858/19859 | 30926/41036 |
| SPECULATIVE_GM | transformer | counterfactual | 19858/19859 | 41036/41036 |
| COLOR_TRIALS | normalized | passive | 14277/14394 | 55665/57576 |
| COLOR_TRIALS | normalized | matched | 14227/14394 | 55996/57576 |
| COLOR_TRIALS | normalized | counterfactual | 14277/14394 | 57429/57576 |
| COLOR_TRIALS | deepsets | passive | 14394/14394 | 54874/57576 |
| COLOR_TRIALS | deepsets | matched | 14394/14394 | 57056/57576 |
| COLOR_TRIALS | deepsets | counterfactual | 14394/14394 | 57576/57576 |
| COLOR_TRIALS | transformer | passive | 14394/14394 | 57052/57576 |
| COLOR_TRIALS | transformer | matched | 14394/14394 | 56468/57576 |
| COLOR_TRIALS | transformer | counterfactual | 14394/14394 | 57576/57576 |

En las baterías finales de los puntos CF exitosos también se compararon
pasos nativos y vectores completos con el profesor sobre la misma cinta.
Son métricas descriptivas de estados/tiempos, separadas del gate y de las
leyes sobre grafos fijos. No se convierten estos casos de familias diferentes
en una certificación de una única ley condicional.

| Profesor | Arquitectura | Vectores completos iguales /5000 | Pasos estudiante/profesor (media apareada) |
|---|---|---|---|
| BE_BASE_D3 | normalized | 5000 | 1.000000 |
| BE_BASE_D3 | deepsets | 5000 | 1.000000 |
| BE_BASE_D3 | transformer | 5000 | 1.000000 |
| COLOR_TRIALS | normalized | 4459 | 1.023217 |
| COLOR_TRIALS | deepsets | 4998 | 1.000000 |
| COLOR_TRIALS | transformer | 5000 | 1.000000 |
| SPECULATIVE_GM | normalized | 5000 | 1.000000 |
| SPECULATIVE_GM | deepsets | 5000 | 1.000000 |
| SPECULATIVE_GM | transformer | 4915 | 1.000026 |

En dos grafos pequeños fijados se compararon 256 cintas compartidas por
política y grafo, observando la coloración completa etiquetada y el estado
terminal sin filtrar fallos. Hay 89/90
certificados simultáneos de TV≤.05, mediante la desigualdad de acoplamiento
y cotas binomiales con Bonferroni .05/90. No se afirma igualdad exacta.
La excepción es COLOR_TRIALS/normalized, semilla
94104, path8: 36/256
desacuerdos, cota superior TV 0.223630,
2.191406 pasos frente a
2.097656. Desacuerdo bajo un acoplamiento
no es una cota inferior de TV ni prueba de leyes marginales diferentes.

**P12. ¿Qué se demuestra sobre información?** El lema de
indistinguibilidad impide imitar decisiones distintas con observaciones
iguales. Dos IDs intercambiados en una arista dan una obstrucción constructiva
para JP sin orden; comparaciones relativas bastan para su elegibilidad.
ColorTrials necesita información de disponibilidad del vecino o equivalente
de segundo salto en la observación especificada. La conjunción participación
y oferta del mismo vecino debe preservarse antes de agregar. Se explicitan
las suficiencias locales de puntos Linial y bloques KW y los permisos de
calendarios. Además, dos uniforms float64 a ambos lados de .5 se pueden
codificar como el mismo float32 en M6 histórico, impidiendo fidelidad
universal de esa etiqueta; el protocolo nativo actual consulta float32
directamente. Son resultados acotados a esas observaciones, no una teoría
universal de entrenabilidad o complejidad muestral.

**P13. ¿Generalización demostrada?** En transferencia exploratoria
N128/256/512, con diez familias del dominio y cuatro generadores nuevos
(árboles binarios, cadenas de diamantes/triángulos y K4 subdivididos), se
observaron 3780/3780 éxitos. Cada política tiene
84 casos; las proporciones de la tabla no reemplazan su gate final.
No se amplía el dominio a grado mayor que tres ni a paletas locales. La
línea de grado 200 se conserva como trabajo complementario.

| Profesor | Arquitectura | Éxitos agregados /420 | Cinco políticas perfectas |
|---|---|---|---|
| BE_BASE_D3 | normalized | 420 | 5 |
| BE_BASE_D3 | deepsets | 420 | 5 |
| BE_BASE_D3 | transformer | 420 | 5 |
| COLOR_TRIALS | normalized | 420 | 5 |
| COLOR_TRIALS | deepsets | 420 | 5 |
| COLOR_TRIALS | transformer | 420 | 5 |
| SPECULATIVE_GM | normalized | 420 | 5 |
| SPECULATIVE_GM | deepsets | 420 | 5 |
| SPECULATIVE_GM | transformer | 420 | 5 |

**P14. ¿Contribuciones centrales sólidas?** (i) Oráculos causales heterogéneos
con contratos de admisibilidad, etiquetas consistentes y testigos repetibles;
(ii) cierre reproducible de las nueve celdas con controles que separan
concordancia, CF y optimización; (iii) demostración de que PASS funcional,
fidelidad de continuación y ley terminal requieren evidencias diferentes;
(iv) reconciliación histórica autenticada y límites de información
constructivos. Los efectos de cobertura nulos en este control, fallos CF a
1500, piloto BE propio y límites de transferencia forman parte de la
contribución. CLRS y DAgger proporcionan antecedentes de trazas y adquisición
en estados del estudiante; la novedad propuesta aquí se concreta en estos
protocolos, adaptadores y controles de siete profesores, sin reclamar
prioridad bibliográfica universal.

## Reproducibilidad y límites

Los archivos obligatorios de reconstrucción, reconciliación, protocolo y
análisis de fallos están en `docs/`. `verification.json` recalcula todos los
gates y hashes y vuelve a comprobar los labels de los bancos; no hay etiquetas
almacenadas inconsistentes. Pasaron los 668 casos de la suite completa final.
Las pruebas incluyen ausencia de consultas docentes en inferencia,
reanudación exacta, diferenciales nativos, intervenciones y testigos de
información. Los pesos, Adam, RNG, visitas, datos, presupuestos, fallos y
manifiestos permanecen en el directorio de resultados nuevo. No se hizo push.

La revisión de antecedentes consulta [DAgger](https://proceedings.mlr.press/v15/ross11a.html)
y [CLRS](https://arxiv.org/abs/2205.15659). El trabajo reciente de
[coloración contrastiva](https://arxiv.org/abs/2606.03923) estudia un objetivo
de coloración aproximada. Estos antecedentes orientan la comparación;
nuestros resultados se atribuyen a las reglas nativas implementadas y a los
artefactos concretos auditados. Las fuentes de Pedro ausentes, el alcance
finito de los tests y las diferencias de protocolos permanecen explícitos.
