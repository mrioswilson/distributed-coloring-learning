# Reconciliación experimental de continuaciones y ejecución

Actualización del 1 de noviembre de 2026: el cuarto paquete ya está recibido
y verificado. Sus 240 JSON originales coinciden con `ROWS.sha256`; las
240.000 filas, los resúmenes y las identidades de 60 checkpoints/cuatro
condiciones coinciden. Se reconstruyeron las identidades de los casos sin
repetir inferencia neuronal. Este pendiente de datos queda cerrado. Recibos
en `results/martin-transfer-rows-20261101/`; el artículo y PDF no cambian.

Nota editorial (1 de noviembre de 2026): los estados de verificación y
entrega descritos aquí son documentación interna de trabajo. Se han retirado
del artículo, que conserva los protocolos, resultados y límites científicos.

Fecha: 31 de octubre de 2026. El análisis distingue inferencia reproducida,
aritmética comprobada sobre registros publicados, interpretación y preguntas
abiertas. Ningún resultado histórico se reemplaza por el de otra campaña.

## Campañas y preguntas que identifican

| Campaña | Réplicas y esfuerzo | Evaluación | Alcance |
|---|---|---|---|
| Diagnóstico histórico | Cinco semillas; presupuestos distintos por brazo; 4.000 actualizaciones en BE | 99.840 consultas BE por semilla, reutilizadas durante desarrollo | Diferencia de procedimientos en esas consultas; no comparación causal aislada de CF |
| Prospectivo de Pedro | BE/GM, diez pares por profesor, 40 modelos; base común de 3.000 objetivos y 3.600 auxiliares por brazo; 1.500 actualizaciones | 4.000 consultas por modelo, tipos finos reservados; 1.000 ejecuciones por modelo | Adquisición de respuestas en contextos reservados, con distinta información y cobertura lógica |
| Presupuesto de Martín | BE ocho brazos y GM/CT seis, cinco semillas; 100 modelos; 4.000 actualizaciones y snapshot secundario a 1.500 | Familias estándar, nuisance y unseen; baterías funcionales propias | Comparación de adquisiciones con presupuestos próximos; no controles alternativos de negativos informativos |
| Follow-up de vecinos | Cinco modelos BE nuevos, posteriores al diagnóstico anterior; controles congelados | Batería v2 fresca, perturbaciones vecinas | Intervención positiva específica sobre cobertura contextual; no réplica independiente del prospectivo de Pedro |
| Transferencia nueva de Martín | GM/CT, cinco semillas, seis brazos, checkpoint a 4.000, cuatro condiciones | 240.000 ejecuciones | Resúmenes de un contraste prospectivo funcional separado; su contraste direccional GM no encuentra propuestas inválidas. Sus 240 archivos crudos están autenticados en un cuarto paquete, sin nueva inferencia |
| Integración: componentes BE | Seis checkpoints, todas las semillas con algún fallo de la campaña de Pedro; cinco modos | 30.000 ejecuciones sobre casos originales consumidos | Diagnóstico retrospectivo por intervención; no entrenamiento ni test de superioridad |
| Integración: celdas históricas | Siete archivos de pesos originales por SHA, quince celdas | 15.000 ejecuciones, incluidos 1.587 fallos | Reproducción independiente de toda la clasificación de celdas con fallos; no regeneración de las otras 795.000 observaciones |

Las consultas y los casos pareados no son nuevas réplicas de entrenamiento.
Los estudios usan namespaces, adquisición, soportes y métricas diferentes.

## Prospectivo de 40 modelos: verificación completa de consultas

`scripts/integrated-reinfer-continuations.py` vuelve a inferir las 160.000
decisiones finales con los 40 pesos originales. Todas las métricas coinciden
exactamente con los registros de Pedro. `integrated-reconcile-evidence.py`
reproduce además las 20 filas del análisis de soporte desde los arrays,
incluidos etiquetas, claves finas disjuntas y configuraciones lógicas.
Recibos: `continuation-reinference.json` y `evidence-reconciliation.json`
en `results/integrated-paper-20261031/`.

BE: BA media 84,6428→90,4145%, diferencia +5,7717 puntos, IC bootstrap
pareado [−13,732,+25,258]. Tres semillas nativas están cerca de 50% BA;
dos CF están cerca de 52%. El intervalo amplio no prueba efecto nulo.
La exactitud con igual peso por tipo mejora en los diez pares:
51,6954→86,9386%. El baseline programado E·P alcanza 99,71% BA por
ocurrencia y 81,27% por tipo. CF gana al baseline por tipo en siete
semillas, empata en una y pierde en dos. Se conservan todos los pares.

| BE, diez bancos | Nativo | CF |
|---|---:|---:|
| Aceptaciones de opciones elegibles, activas y libres | 995/995 | 814/995 |
| Rechazos de opciones elegibles, activas y ocupadas | 0/228 | 228/228 |
| Falsos positivos elegibles y silenciosos | 42/133 | 81/133 |
| Ejecuciones autónomas exitosas | 9.204/10.000 | 7.405/10.000 |

Los 181 falsos negativos CF corresponden a 98 y 83 consultas de las
semillas 105001 y 105008: todos en κ=0 y sin vecinos participantes.
Son consultas de prefijos del profesor; no son 181 ejecuciones fallidas.
Las consultas inelegibles suman 38.644/40.000 (96,61%); 38.526 son
además activas (96,315%). No se convierte retrospectivamente un estrato
exploratorio en endpoint primario.

GM: 50→100% BA, pero ambos brazos completan 10.000/10.000 ejecuciones.
El nativo recibe cero negativos COMMIT en ASSIGN/REPAIR. Este dato explica
qué información falta; no fuerza matemáticamente ninguna predicción de la
red ni prueba superioridad sobre otra adquisición de negativos.

| Solapamiento lógico test–entrenamiento, por ocurrencia | Nativo | CF |
|---|---:|---:|
| BE | 2,125–2,900% | 99,475–100% |
| GM | 37,725–39,500% | 100% |

Las claves finas sí están reservadas. Las condiciones lógicas nuevas no
constituyen el mismo problema de extrapolación para ambos brazos. Igual
número de etiquetas no implica igual información ni igual soporte.

## Experimento nuevo de componentes: resultados completos

El protocolo `INTEGRATED_FUNCTIONAL_DIAGNOSTIC_PROTOCOL.md`, 199 fuentes
y los seis pesos se congelaron antes de sus resultados. Selección explícita:
las tres semillas con algún fallo, ambos brazos y todos sus 1.000 casos.
El protocolo es posterior al test original y no pretende confirmación nueva.
Los mensajes sustituidos se corrigen en todos los centros representados;
cuando cambian P o C se recomputa COMMIT aprendido. Después de divergir una
trayectoria se mantienen los pesos, no las salidas futuras del modelo.

| Semilla | Brazo | Original | Solo COMMIT exacto | Solo P exacta | Solo C exacta | Todos KW exactos |
|---|---|---:|---:|---:|---:|---:|
| 105001 | Nativo | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 105001 | CF | 100 | 1.000 | 100 | 100 | 1.000 |
| 105008 | Nativo | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 105008 | CF | 101 | 1.000 | 101 | 101 | 1.000 |
| 105010 | Nativo | 204 | 204 | 1.000 | 204 | 1.000 |
| 105010 | CF | 204 | 204 | 1.000 | 204 | 1.000 |

Hechos verificados: las seis baselines reproducen exactamente los 6.000
vectores terminales, pasos, estados de fallo y flags originales; no hay
errores técnicos en las 30.000 ejecuciones ni casos nuevos fallidos bajo
las intervenciones. Los modos inefectivos también se conservan.

Corregir solo COMMIT rescata 1.799 de los 2.595 fallos CF originales.
Corregir solo P rescata los otros 796; C sola no rescata ninguno. El modo
KW exacto mantiene Linial aprendido. El resultado demuestra suficiencia de
esas sustituciones para esos casos y modelos; no necesidad universal, una
política reparada aprendida ni un remedio que generalice.

En trayectorias CF originales, las semillas 105001/105008 visitan 1.312/1.322
rechazos de escrituras válidas, todos en κ=0. De ellos 358/406 tienen
algún vecino activo: la distribución visitada difiere del test de prefijos.
105010 visita 983 silencios elegibles por brazo. Estos son conteos de
visitas, correlacionados, no tasas sobre una población de políticas.

Un observador posterior resolvió otra aparente anomalía: las 64 y 20
aceptaciones activas rechazadas por el contrato en dos baselines nativas
decodifican siempre al mismo color anterior y satisfacen la nueva cota.
Sus 84 desacuerdos no producen una transición distinta. El nombre histórico
`learned_unsafe_accepts` de ese diagnóstico significa desacuerdo activo con
el contrato; no debe interpretarse como 84 escrituras que causan conflictos.
Recibo separado: `redundant-write-observer.json`, 2.000 replays con paridad.

## Rama paralela: incorporación y verificación tras el release

Inicialmente se validó la aritmética de 466 filas publicadas y de las 40
certificaciones BE. Tras recibir los datos se ejecutaron sus analizadores
originales sobre los arrays: los JSON completos, con las 466 filas, coinciden.
También coinciden las 985.992 etiquetas de consulta recomputadas. La inferencia
nueva reproduce 50.173.920 decisiones binarias de las 220 referencias
checkpoint–batería (205 rutas distintas, con controles reutilizados). Los
logits tienen diferencias de hasta `3.0517578125e-05`, sin cambios de signo.
Los recibos y la restauración por SHA se documentan en
[MARTIN_ARTIFACT_DELIVERY_AUDIT.md](MARTIN_ARTIFACT_DELIVERY_AUDIT.md).
Las 120 evaluaciones funcionales también se repitieron completas: 120.000
ejecuciones y cero errores técnicos; coinciden todos los resúmenes originales.
Se guardan ahora todas las filas por caso, que los programas originales
de esos dos estudios no habían retenido. No se afirma igualdad vector a
vector con esos originales inexistentes ni que los 15 controles reutilizados
del follow-up sean modelos entrenados nuevos.
El conteo correcto de checkpoints BE que pasan todos sus casos es 32/40,
no el 34/40 del protocolo original. El protocolo se conserva sin reescribir.
BE CF-high completa 4.298/5.000 frente a 5.000/5.000 matched-high; CF-low
completa 3.270/5.000. El criterio propio de ≥99% BA sobre tipos no vistos
solo se cumple en dos de cinco CF-high. Estos resultados adversos permanecen.

Los presupuestos son próximos, no exactamente iguales: BE high 6.776 CF
y 6.774 nativo; GM difiere en uno o dos objetivos según semilla;
CT 70.000/69.999. Grupos completos de componentes explican el redondeo.
Las variantes noforce/lateocc no consumen exactamente el mismo total.
Los controles nativos siguen sin negativos informativos equivalentes.

Los snapshots CF a 1.500 ya tienen BA estándar alta. Por tanto, atribuir la
discrepancia entre ramas solo a 4.000 versus 1.500 actualizaciones carece de
control. Reservar cotas KW tardías tampoco garantiza novedad del input
efectivo: el COMMIT consultado no lee la cota numérica B.

El follow-up de vecinos conserva raíces, consultas y etiquetas; cambia
mensajes vecinos irrelevantes para la regla BE con ocupación fija. Sus cinco
modelos completan 5.000/5.000 en la nueva batería. El control antiguo 95105
completa 301/1.000 en v2 (298/1.000 en v1), frente a 1.000 del nuevo.
Los 105 entrenamientos adicionales, bancos y respuestas ya están entregados
y su inferencia de continuación está repetida. No se trasladan sus conclusiones
a los 181 errores de frontera silenciosa del otro estudio.

## Transferencia y mecanismos: qué se reprodujo

Se reproducen todos los 15.000 casos de las quince celdas con fallos usando
siete archivos de pesos originales: coinciden éxitos, IDs fallidos, clases
y contadores. Johansson tiene 28 fallos de progreso sin aceptación insegura;
GM CF tiene 181 del mismo tipo. Los otros 1.378 fallos incluyen aceptaciones
inseguras en GM nativo y CT. Las intervenciones de testigo y las pruebas
matemáticas dan la explicación; la clasificación por sí sola no estima
cuántos casos puede rescatar una intervención distinta.

Los agregados históricos siguen siendo 449.791/450.000 primarios y
808.413/810.000 totales. Tras el release se verifican todas las 810 rutas
crudas y se reconstruyen los 4.000 grafos y todas las coloraciones terminales:
coinciden los agregados, conflictos, paletas y estados incompletos. Esto no
regenera las 810.000 trayectorias neuronales; sus flags intermedios se autentican
sin reconstruir cada transición. La nueva campaña de Martín de 240.000 ejecuciones es separada,
toda exitosa. GM no propone colores inválidos allí, por lo que el contraste
direccional de seguridad es no informativo; se conserva como resultado
negativo para esa hipótesis, sin sumarlo a 810.000.

La revisión de `design.json`, `summary.json`, código y `ROWS.sha256` detectó
un error de nuestra primera integración: esta campaña usa seis brazos
(pasivo/matched/CF × dos presupuestos) a 4.000 actualizaciones, no dos brazos
y dos hitos. Son 60 checkpoints × cuatro condiciones = 240 celdas. Se corrige
la descripción y se conservan los datos originales. La cuarta entrega ya
recupera todos los archivos bajo
`results/pedro-revision/transfer-experiment/rows/`, con sus 240 hashes originales
idénticos. Se verifican cobertura, cada resumen y todos los contadores:
240.000/240.000 éxitos, cero aceptaciones inseguras/cutoffs. GM tiene cero
propuestas inválidas; CT 3.743.920, todas contenidas, y 287.110 rechazos de
propuestas válidas. Son visitas de decisiones, no ejecuciones independientes.
Se reconstruyen3.000 identidades de casos compartidas y coinciden los hashes
de casos de los ocho controles exactos. Esta auditoría de filas no repite
las redes ni reconstruye colores completos, que no se guardaron aquí.
`separate-transfer-availability.json` conserva el hallazgo inicial; el cierre
está en `results/martin-transfer-rows-20261101/raw-transfer-audit.json`.

## Decisión sobre más entrenamiento

No es indispensable para la contribución integrada. Sí sería indispensable
un control alternativo con positivos y negativos informativos para defender
superioridad específica de CF, o familias completamente reservadas y
comprobadas a nivel de input efectivo para defender esa generalización.
La versión actual no hace ninguna de esas afirmaciones. La entrega de pesos,
bancos, splits y respuestas del follow-up ya se verifica efectivamente. Esa
verificación cierra una limitación de disponibilidad, sin convertir sus
conclusiones finitas en una garantía ni en una réplica de entrenamiento nueva.
