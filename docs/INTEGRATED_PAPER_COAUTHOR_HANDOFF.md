# Handoff de la versión integrada para Pedro y Martín

Entrega Git autorizada el 1 de noviembre de 2026: artículo, PDF y evidencia
de esta versión en la rama `codex/integrated-paper-20261031`, separada de
`main`. Las dos instantáneas históricas se transportan en partes menores de
50 MiB; `python scripts/restore-coauthor-snapshots.py` recompone sus bytes
originales antes de la verificación. No cambia el PDF ni el estado de la
revisión científica independiente. Esta actualización sustituye el estado
local sin commit/push indicado en los registros históricos siguientes.

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

Fecha: 31 de octubre de 2026. Borrador completo para revisión de coautores.
La integración, auditorías y experimentos descritos están hechos localmente;
la revisión independiente y la aprobación de ambos autores siguen pendientes.

**Actualización tras recibir el release de Martín:** los tres paquetes y sus
1.988 miembros están verificados. Ya no faltan pesos/bancos de los 105
entrenamientos ni las 810.000 filas crudas históricas. La inferencia nueva
reproduce 50.173.920 decisiones binarias; la reconstrucción de todas las
coloraciones terminales conserva los conteos históricos. El informe
[MARTIN_ARTIFACT_DELIVERY_AUDIT.md](MARTIN_ARTIFACT_DELIVERY_AUDIT.md) distingue
qué se repitió efectivamente y qué sigue limitado a datos consumidos.
Las 120.000 ejecuciones autónomas repetidas también coinciden en todos los
resúmenes. Los240 JSON originales de la campaña adicional de transferencia
también están recuperados y autenticados mediante un cuarto paquete.

## Versión y tesis

Artículo: [Learning Distributed Coloring Rules: Functional Success,
Continuations, and Transfer](../paper/ucgnn/main.pdf), 34 páginas,
[fuente canónico](../paper/ucgnn/main.tex). La narrativa parte de políticas
que funcionan y transfieren, define qué continuación se evalúa y explica
por qué ciertos desacuerdos afectan la ejecución mientras otros quedan
enmascarados por participación, decodificación o el calendario.

Tesis defendida: el éxito funcional dentro de un intérprete, la fidelidad
de módulos ante consultas definidas y su transferencia evalúan aspectos
diferentes. Las decisiones adquiridas fuera del soporte nativo y sus
consecuencias dependen de la interfaz y de su papel en progreso/reparación.
La transferencia se formula como evaluación de funcionalidad o fidelidad
bajo un cambio de dominio, no como otra variable de salida independiente.

No se afirma superioridad universal de CF, corrección universal, extrapolación
ilimitada ni que las tres nociones se separen de cualquier forma posible.

## Lo nuevo y verificable

**Experimento funcional prioritario completado.** Se congelaron protocolo,
fuentes y seis pesos antes de las intervenciones. Las tres semillas elegidas
incluyen todos los fallos BE del prospectivo; se conservan ambos brazos y
todos los casos originales. Los demás componentes mantienen sus pesos y
recalculan salidas después de que diverge la trayectoria.

| Casos CF | Aprendido | Solo COMMIT exacto | Solo P exacta | Solo C exacta |
|---|---:|---:|---:|---:|
| Semilla 105001, /1.000 | 100 | 1.000 | 100 | 100 |
| Semilla 105008, /1.000 | 101 | 1.000 | 101 | 101 |
| Semilla 105010, /1.000 | 204 | 204 | 1.000 | 204 |

Corregir COMMIT rescata los 1.799 fallos de las dos primeras semillas;
P rescata los otros 796. C sola no rescata ninguno. En 105010 el brazo
nativo tiene los mismos resultados por intervención. Los otros dos nativos
completan sus 1.000 casos en todos los modos. Todos los KW exactos completan
6.000/6.000, conservando Linial aprendido. Son 30.000 ejecuciones, seis
baselines completas idénticas al original y cero errores técnicos o casos
nuevos fallidos bajo las sustituciones.

Esto establece suficiencia de sustituciones concretas para casos consumidos.
No estima un remedio poblacional ni produce una política reparada aprendida.
Los estados de rollout pueden diferir de los prefijos exactos del profesor:
el dominio diagnóstico de la regla KW se declara por separado.

**Otra distinción funcional comprobada.** Un observador posterior reproduce
2.000 ejecuciones y verifica que las 84 aceptaciones activas contractualmente
incorrectas de dos nativos decodifican al mismo color anterior. No cambian
la transición. Esto sustenta el requisito «decodificación distinta del estado
anterior» de la proposición, además del enmascaramiento por P=0.

**Verificación ampliada.** Se re-infieren las 160.000 decisiones de continuación
de los 40 modelos originales: todas las métricas coinciden. Se reconstruyen
etiquetas/claves de los 20 pares desde arrays y se validan 466 filas de
resúmenes de la otra rama. Tras el release se recomputan esos análisis completos
desde los arrays, se verifican 985.992 etiquetas de consulta y se repite la
inferencia de las 220 referencias checkpoint–batería. Coinciden todas las
50.173.920 decisiones binarias, con diferencia máxima de logits
`3.0517578125e-05`; no se afirma identidad bit a bit. Se reproducen completas las quince celdas históricas
con fallos (15.000 ejecuciones): coinciden éxitos, casos fallidos, clases y
contadores. Los siete archivos de pesos necesarios se entregan por SHA256.

## Resultados positivos recuperados y contraste CF

El cuerpo vuelve a presentar 21/21 combinaciones profesor–arquitectura,
cinco checkpoints y 1.000 éxitos cada uno, con distinción entre doce paquetes
históricos pasivos y nueve CF posteriores. Los controles matched también
pasan esos nueve. No se afirma equivalencia ni ranking de arquitecturas.

La transferencia principal conserva 449.791/450.000 éxitos, 18 combinaciones,
cinco semillas y cinco condiciones (incluido control), dos celdas con fallos
y 16 combinaciones que pasan todas. Los agregados completos históricos son
808.413/810.000. Las 810.000 filas ya están entregadas y todas sus
coloraciones finales se reconstruyen contra 4.000 grafos etiquetados.
Coinciden conflictos, paletas, completitud y conteos de cada celda; esto
complementa los 15.000 replays, sin ejecutar todas las trayectorias restantes.
Los pilotos de grado 200 o tamaño 4.096
siguen siendo exploratorios.

El prospectivo BE mantiene su endpoint registrado: +5,77 puntos de BA,
con intervalo [−13,73,+25,26], muy heterogéneo. No es prueba de efecto cero.
La exactitud con igual peso por tipo mejora en diez pares (51,70→86,94%),
pero el baseline E·P obtiene mayor BA por ocurrencia y el éxito CF baja
9.204→7.405 de 10.000. Los 181 falsos negativos del banco del profesor
son de κ=0 sin vecinos participantes; no se equiparan a los conteos de
visitas del rollout, ni a ejecuciones distintas.

GM 50→100% BA se explica como aprendizaje de rechazos ausentes en el brazo
nativo, no como superioridad general de CF. Ambos brazos completan todos
sus casos. Se reporta solapamiento lógico de ambos brazos: BE nativo
2,125–2,9% frente a CF 99,475–100%; GM nativo 37,725–39,5% frente a CF100%.
Las claves finas sí están reservadas. Los presupuestos iguales no igualan
su contenido informacional.

## Qué se incorpora de Martín y cómo se reconcilia

Se incorporan selectivamente los resúmenes de su estudio de 100 modelos,
el follow-up de cinco modelos con perturbación de vecinos, clasificación de
fallos y bibliografía pertinente. Sus protocolos/namespaces se mantienen.
La reescritura canónica completa de Martín no estaba presente en el árbol
publicado; no se inventa una segunda versión textual terminada ni se hace
un merge automático.

BE CF-high tiene una ventaja estándar más consistente y 4.298/5.000 éxitos
funcionales frente a 5.000 matched-high. Solo dos de cinco semillas cumplen
su criterio de ≥99% BA sobre tipos no vistos. El conteo derivado correcto
de BE all-pass es 32/40, frente al 34/40 del protocolo conservado. Sus
presupuestos son próximos (BE 6.776/6.774), no exactamente iguales.
Sus snapshots CF ya son fuertes a 1.500: no se atribuye la discrepancia de
ramas a 4.000 versus 1.500 actualizaciones. El soporte, ponderación y
endpoints también cambian, sin control que aísle una causa.

El follow-up de vecinos es evidencia positiva: cinco modelos
completan 5.000/5.000 en la batería v2 fresca; en 95105 el control completa
301/1.000 frente a 1.000 del modelo perturbado. No subsana automáticamente
los errores de frontera silenciosa del otro estudio. Los datos de los
105 entrenamientos (100+5) ya están entregados y auditados. Los freezes
contienen 220 referencias (200+20), de las que 15 reutilizan controles:
205 rutas congeladas distintas. Los 315 `.pt` son 105 archivos de cada
nombre `u001500.pt`, `u004000.pt` y `last.pt`, no 315 modelos entrenados.

La campaña nueva de transferencia de 240.000 ejecuciones queda separada:
sus resúmenes reportan que todas completan; GM no ofrece propuestas inválidas,
de modo que su hipótesis direccional de rechazo no se puede discriminar allí.
Se corrigió un error de nuestra descripción: seis brazos a 4.000 updates,
no dos brazos y dos hitos. Sus 240 JSON crudos se entregaron en el cuarto
paquete y coinciden con `ROWS.sha256`; se recomputan todos sus resúmenes y
contadores. GM mantiene cero propuestas inválidas; CT registra3.743.920, todas
contenidas, y 287.110 rechazos de ofertas válidas, con 120.000/120.000 éxitos
por profesor. No se confunde esta autenticación con nueva inferencia ni
se suma su denominador a las 810.000 filas históricas.

## Matemática y especificaciones

- Proposición de subdeterminación: dos reglas de H coincidentes en soporte
  nativo y distintas en una consulta admisible; prueba explícita y separación
  entre soporte completo y banco observado. La pertenencia a H es una hipótesis.
- Proposición de relevancia: la escritura depende de permiso, participación,
  COMMIT y decodificación. P=0 enmascara el bit; una aceptación activa puede
  ser identidad. En KW una escritura obligatoria omitida rompe la nueva cota.
- GM: condición suficiente de reparaciones libres y condición de conflicto
  superviviente en el sufijo. Un contraejemplo ejecutado muestra que otro
  extremo posterior puede reparar una omisión.
- Progreso: cota por oferta vecina que incluye Johansson con participación
  uno, con prueba y U_v definido. No se certifican esas hipótesis para la red.
- Contratos: dominio K4 frente a extensión de transferencia; participación
  BE; veto adicional JP; primer punto Linial; índices KW; probabilidades y
  campos preservados de las consultas históricas.

El testigo KW tiene además un prefijo exacto completo construido con IDs
legales, separado del testigo histórico cuyos IDs iniciales no se recuperan
del registro parcial. Las pruebas pertinentes pasan: 330 regresiones más
una de este prefijo. Los experimentos verifican correspondencia con código;
las demostraciones sustentan las afirmaciones matemáticas.

## Novedad y mis respuestas a las cinco preguntas

1. La síntesis es científicamente correcta con contratos y dominios explícitos;
   transferencia es un eje de evaluación de las otras propiedades.
2. El prospectivo identifica principalmente adquisición de respuestas fuera
   del soporte nativo; no compara CF con otras formas de adquirir negativos.
3. La proposición COMMIT es útil y ahora está acompañada por replays completos
   y por el caso de decodificación identidad, además de una prueba.
4. Es esencial recuperar funcionalidad y transferencia positivas. Ya están
   en el cuerpo y abstract con sus diferencias de protocolo.
5. Hay una contribución empírica plausible para discutir en TMLR. La distinción
   éxito/fidelidad ya tiene precedentes: la aportación defendible son estos
   contratos, cobertura y mecanismos verificables. No se predice aceptación.

Se comparan CLRS, Hint-ReLIC, SALSA-CLRS, IIT/DIITO, Qharabagh et al.,
Discrete NAR, Wittig et al., Mind the Gap y el trabajo concurrente Premature
Floods mediante fuentes primarias. Se elimina un dato de presentación «oral»
no confirmado y se usa el preprint verificable. No se reclama prioridad.

## Pendientes reales y decisiones de los autores

1. Revisión independiente de teoría, estadísticas y posicionamiento de esta
   versión. El informe adversarial actual es del integrador; Claude revisó
   la versión anterior. No se atribuye aprobación a ninguno de los autores.
2. Autoría/orden, título final, extensión del cuerpo y formato/anónimo de envío.
3. Si se desea reclamar superioridad CF: nuevo control con negativos informativos
   y soporte/balance comparables. Si se desea generalización a intervenciones
   nuevas: familias completas reservadas a nivel de input efectivo. Ninguno
   es indispensable para las afirmaciones modestas de esta versión.

La entrega Git actualiza el estado anterior sin commit/push; no se realiza
merge a `main` ni publicación en revista.
Los experimentos negativos, variantes ineficaces, fuentes previas y cambios
ajenos se preservan. No hay un entrenamiento científico pendiente para cerrar
los resultados nuevos presentados aquí.

## Rutas, fuentes y hashes

Trabajo aislado: `/workspace/paper-integration-20261031`, rama de revisión
`codex/integrated-paper-20261031`, a partir del commit de Pedro. El contenido
integrado se identifica mediante el commit de entrega y el manifiesto de bytes.

| Objeto | Identificador verificable |
|---|---|
| Pedro, base textual | `5993a5e4782d44670926224263688943a1411ebf` |
| Martín, evidencia importada | `d155d3ed104d10a015fa1061e7a5fc7f5ebcdb7b` |
| Ancestro común | `c3affd16ac5654c6e2a700d2312710753adb9518` |
| PDF integrado | `22726ee0bd51cfe775d2eb8eef7ff40e2714b23957cc88774263a7b55018e45c` |
| Congelamiento funcional | `2bccf5822df6495f74c98722a4f4532cbbd9fff0e53bb331c21308d42c210944` |
| Congelamiento replay transferencia | `2f0780396dcea5eb287bbf7a56574087f61c9ada831a6c05f94b4da25a715b0f` |
| Congelamiento inferencia del release | `eaa72b5d40eb215f20c4fdb71dfb9539465dfa77bda78b96c336ea12cd814ca4` |
| Congelamiento reconstrucción terminal | `b20d7810f8caf537907396df94fe5930e9425ddbfb58e032e70d517faeb2d6f3` |
| Archivo de siete pesos originales | `26b152e9f1d317cb654f6d1368fb110373b36e785522edb2604a8789f6f58597` |
| PDF Pedro preservado | `196e2fccbd3401abebc42a995cf132b5c8a4a9406633172d97eb6623ddec5984` |

Lecturas por fase: [reconciliación](COAUTHOR_RECONCILIATION.md),
[teoría](CONTINUATION_PROGRESS_THEORY_AUDIT.md),
[experimentos](CONTINUATION_EXPERIMENTAL_RECONCILIATION.md),
[revisión crítica](INTEGRATED_MANUSCRIPT_REVIEW.md),
[bibliografía](INTEGRATED_LITERATURE_AUDIT.md) y
[reproducción](INTEGRATED_PAPER_REPRODUCE.md).

En `results/integrated-paper-20261031/`: `sources.json` autentica 261 imports;
`functional-freeze.json`, `functional-summary.json` y `functional/*.json.gz`
guardan protocolo/pesos y todas las ejecuciones; `continuation-reinference.json`
y `evidence-reconciliation.json` distinguen predicciones de aritmética;
`redundant-write-observer.json` verifica identidades; `transfer-replay-summary.json`
y `transfer-replay/*.json.gz` conservan los 15.000 casos; `paper-build.json`
identifica PDF/fuentes/figuras y compilador; `manifest.json` autentica los
archivos entregados. `verification.json` incluye SHA del manifiesto y el
resultado de sus comprobaciones, sin circularidad de hashes.

El archivo original `results/paper-revision-20261030/paper-revision-runtime.tar.gz`
y el nuevo `integration-replay-runtime.tar.gz` permiten restaurar los 307
archivos de datos/pesos usados por los replays disponibles. El comando
`python scripts/verify-coauthor-integration.py --restored-runtime` verifica
entrega y archivos restaurados. El cuerpo y PDF se reconstruyen con los
builders `build-coauthor-*`; los históricos se ejecutan en su propio checkout.

En `results/martin-artifacts-20261031/`, `download-receipt.json` fija los tres
paquetes y `member-index.json` todos los miembros restaurados;
`checkpoint-inventory-audit.json` reconcilia 220/205/315;
`raw-query-label-audit.json` y `recomputed-metrics-audit.json` verifican
etiquetas/análisis; `model-reinference-audit.json` y `reinference/` conservan
todos los logits nuevos; `functional-replay-audit.json` y `functional/`
guardan las 120.000 ejecuciones; `raw-terminal-audit.json` verifica las
810.000 coloraciones históricas. `separate-transfer-availability.json`
conserva el inventario del faltante inicial, ya cerrado, y `source-recovery/` conserva el
fuente histórico exacto de esa campaña. `prior-integration.tar.gz` preserva
la entrega previa completa. La receta de descarga/restauración y los wrappers
de ejecución efectiva están en el documento de reproducción.

## Verificación de entrega

El verificador final autentica todos los archivos del manifiesto vigente,
260 imports intactos (una corrección bibliográfica declarada), las 199
fuentes selladas del diagnóstico inicial y las 198 del replay nuevo (con
fuentes compartidas),307 miembros de los runtimes previos y2.233 miembros
de los cuatro paquetes del release, además de sus copias restauradas. Reconcilia los conteos de 21
paquetes y 449.791/450.000, comprueba las 50.173.920 decisiones y las 120.000
ejecuciones nuevas y confirma cero referencias pendientes y cero desbordes.
La revisión visual abarca todas las páginas mediante contactos y las páginas
12, 13, 15, 18, 23 y 31 completas. El recibo `verification.json`
contiene la cifra exacta de archivos, el resultado y el SHA del manifiesto
que efectivamente se comprobó. La entrega de240 archivos del contraste
adicional está cerrada; la revisión científica externa sigue pendiente.

En `results/martin-transfer-rows-20261101/`, el cuarto archivo descargado
(SHA256 `d650d45adfbaff8c992b1a24cbcf602abb7acc09b0995557e9f811ccb4d8e5a1`)
contiene240 resultados y cinco metadatos. `member-index.json` autentica los245
miembros; `audit-freeze.json` sella las entradas/fuentes;
`raw-transfer-audit.json` registra la comprobación completa; `prior-records/`
preserva los documentos anteriores. No se modificó el PDF de 34p SHA
`22726ee0bd51cfe775d2eb8eef7ff40e2714b23957cc88774263a7b55018e45c`.
