# Auditoría de la entrega experimental de Martín

Actualización del 1 de noviembre de 2026: el cuarto paquete ya está recibido
y verificado. Sus 240 JSON originales coinciden con `ROWS.sha256`; las
240.000 filas, los resúmenes y las identidades de 60 checkpoints/cuatro
condiciones coinciden. Se reconstruyeron las identidades de los casos sin
repetir inferencia neuronal. Este pendiente de datos queda cerrado. Recibos
en `results/martin-transfer-rows-20261101/`; el artículo y PDF no cambian.

Nota editorial (1 de noviembre de 2026): los estados de verificación y
entrega descritos aquí son documentación interna de trabajo. Se han retirado
del artículo, que conserva los protocolos, resultados y límites científicos.

Fecha: 31 de octubre de 2026. Continuación de la integración autorizada,
sin entrenamiento, selección de modelos, modificación de resultados originales,
push, merge compartido ni publicación. Los recibos de esta auditoría están en
`results/martin-artifacts-20261031/`. La revisión fue realizada por el integrador;
no constituye revisión científica independiente de otro investigador.

## Procedencia, integridad y preservación

Se descargaron los tres paquetes, sus manifiestos y README del
[release publicado](https://github.com/mrioswilson/distributed-coloring-learning/releases/tag/pedrorev-experimental-data-2026-10-31).
La referencia del tag consultada en GitHub apunta al commit
`d155d3ed104d10a015fa1061e7a5fc7f5ebcdb7b`, también base de los resultados
importados. La integridad del contenido se fija por SHA256, no por la
mutabilidad de una URL de release.

| Archivo | Bytes | SHA256 verificado |
|---|---:|---|
| pedrorev-cf.tar.gz | 115.398.845 | `8353b7135cefad7159832ad0b4df901a6e0cd6168e57b0872a49d041d84fe2cc` |
| pedrorev-cf-followup.tar.gz | 8.708.945 | `6a009a4abd5853b9e7259e82d3ee458bdbbff7a6720c99b1eada5559a67773bb` |
| confirmatory-rows.tar.gz | 145.792.662 | `ed1039ac91f572eeb4c0e5e5b06e630d08c6d407a24aceb28b14fd57233ac489` |

Coinciden todos los hashes internos: 1.988 miembros regulares, de los cuales
1.976 son payload científico y 12 metadatos. El primer paquete tiene 1.147
miembros (1.142 científicos), el segundo 28 (24) y el tercero 813 (810).
Los inventarios de procedencia coinciden byte a byte con los JSON originales
de las campañas. El CSV entrante coincide con el existente al normalizar
terminadores de línea. No se encontró un archivo faltante de esos inventarios.

Antes de cambiar la integración se preservó su entrega completa: 844 archivos
verificados contra el manifiesto previo, incluyendo manifiesto y verificación,
en `prior-integration.tar.gz`, SHA256
`bbb46242e78fc498df5321c6aa8fc1243f98509080cb2e053202dc8801b41764`.
El anterior manifiesto era
`610a46c89c185065e6b24d523e5bdbc356829159d8509c4d2705552c72268fc8`.
Los resultados históricos, freezes y fuentes científicas permanecen intactos.

## El conteo 220 / 205 / 315

Los tres números describen objetos distintos y son compatibles:

| Objeto | Conteo comprobado |
|---|---:|
| Referencias del freeze inicial: 100 entrenamientos × dos hitos | 200 |
| Referencias del seguimiento: cinco nuevos + 15 controles reutilizados | 20 |
| Referencias totales | 220 |
| Rutas congeladas distintas: 200 + cinco nuevas | 205 |
| Directorios de entrenamiento: 100 + cinco nuevos | 105 |
| Archivos `.pt`: 105 `u001500.pt`, 105 `u004000.pt`, 105 `last.pt` | 315 |

Por tanto, «205 rutas distintas» y «220 referencias» son correctos; «220
rutas distintas» sería incorrecto. Tampoco hay 315 réplicas de entrenamiento.
Los pesos nuevos del follow-up están dentro del primer paquete, bajo
`data/pedrorev-cf/runs/.../cf-neighbor/`; el segundo aporta su nuevo banco y
sus respuestas. Los tres paquetes en conjunto cubren lo congelado.

`checkpoint-inventory-audit.json` comprueba todas las hashes de pesos,
respuestas y bancos adquiridos, los 31 fuentes de los freezes originales y
la coincidencia de sus metadatos de inicialización emparejada. Esto último
autentica el registro; no recrea el entrenamiento desde su inicialización.

## Protocolo y niveles de comprobación

`MARTIN_ARTIFACTS_REPLAY_PROTOCOL.md` se congeló antes de los nuevos resultados.
No se consumió un nuevo test independiente: se repitieron baterías publicadas.
Los endpoints, namespaces, pesos, casos, semillas y controles no cambiaron.
El cómputo neuronal se ejecuta en CPU float32, un hilo y procesos en serie.
Los siguientes conteos son esfuerzo de verificación, no nuevas réplicas.

1. **Arrays y métricas.** Se recomputaron 985.992 etiquetas de consulta,
   elegibilidad, disponibilidad y claves BE desde los arrays: todo coincide.
   Se ejecutaron los analizadores originales sin cambiar sus fuentes y se
   reconstruyeron los JSON completos de análisis: 426 filas del estudio inicial
   y 40 del seguimiento, las 466 idénticas. No se trata ya solo de revisar
   aritmética de resúmenes.
2. **Inferencia de continuaciones.** Las 220 evaluaciones checkpoint–batería
   reproducen las 50.173.920 decisiones binarias almacenadas, sin discrepancias.
   Ningún vector completo de logits es idéntico bit a bit al original: la
   diferencia absoluta máxima es `3.0517578125e-05`, sin cambios de signo.
   Se conservan los logits nuevos comprimidos y recibos individuales, sin
   sustituir las respuestas originales. No se afirma reproducibilidad numérica
   universal en cualquier plataforma.
3. **Transferencia histórica, datos crudos.** Se autentican las 810 rutas y se
   reconstruyen 4.000 grafos etiquetados, sus fingerprints y las restricciones
   de paleta de cada coloración final. Conflictos, colores incompletos, paleta
   y clasificación terminal coinciden. Se recuperan 808.413/810.000 éxitos,
   1.559 colisiones y 28 cutoffs en 15 celdas. El primario conserva
   449.791/450.000. Se verifican también los 20.000 tapes y 25.000 pares
   caso–paleta compartidos. Esta auditoría no ejecuta 810.000 redes ni
   reconstruye todas las trayectorias intermedias: sus indicadores almacenados
   de invariantes se autentican. Los 15.000 replays neuronales previos de todas
   las celdas fallidas siguen siendo evidencia separada y más fuerte sobre
   sus transiciones.
4. **Ejecución autónoma de los estudios nuevos.** Se repitieron los 100
   checkpoints primarios y las 20 combinaciones modelo–batería del follow-up:
   120.000 ejecuciones, cero errores técnicos y coincidencia de todos los
   campos de resumen originales (éxito, clases/familias de fallo, pasos y
   límites de confianza cuando se guardaban). El estudio inicial tiene
   35.543/40.000 éxitos BE, con 32/40 checkpoints all-pass, y 30.000/30.000
   para cada uno de GM/ColorTrials. El seguimiento registra 19.301/20.000,
   incluidos los 5.000/5.000 de los cinco modelos nuevos; el control CF-high
   95105 reproduce sus 301/1.000. El total 114.844/120.000 es esfuerzo de
   reproducción con controles reutilizados, no otro estimando poblacional.
   Se conservan todas las filas nuevas. Los scripts originales guardaron
   resúmenes funcionales, no filas por caso: se puede comprobar paridad de
   resúmenes, no igualdad con vectores originales ausentes. Recibos:
   `functional-replay-audit.json`, `functional-crosscheck-summary.json` y
   `functional/<estudio>/*.json.gz`.

El congelamiento de inferencia es
`eaa72b5d40eb215f20c4fdb71dfb9539465dfa77bda78b96c336ea12cd814ca4`;
el de reconstrucción terminal,
`b20d7810f8caf537907396df94fe5930e9425ddbfb58e032e70d517faeb2d6f3`.
Ambos sellan el protocolo y sus entradas; no se regeneran para hacer pasar
un resultado. Los recibos principales son `model-reinference-audit.json`,
`raw-query-label-audit.json`, `recomputed-metrics-audit.json` y
`raw-terminal-audit.json`.

## Hallazgo inicial fuera de los tres paquetes y cierre posterior

La campaña nueva de transferencia tiene su propio inventario
`results/pedro-revision/transfer-experiment/ROWS.sha256`: enumera 240 JSON
con 1.000 filas cada uno bajo `results/pedro-revision/transfer-experiment/rows/`.
En la inspección inicial ninguno estaba en el checkout, los tres tar ni los
nueve assets del release. La cuarta entrega posterior cierra esa ausencia.
Sus resúmenes y diseño sí están presentes, y sus identidades de pesos
coinciden con los checkpoints entregados. Las 240 rutas y hashes del faltante inicial
quedan en `separate-transfer-availability.json`. Este hallazgo es compatible
con que no falte ningún miembro de los manifiestos internos de los tres tar:
esos manifiestos no incluían esa otra campaña. No debe interpretarse la
completitud de un paquete como completitud de todos los registros del estudio.

Se corrigió además un error de nuestra primera integración, detectado al
contrastar el diseño y el código: son seis brazos (pasivo/matched/CF × dos
presupuestos), cinco semillas por profesor GM/CT, un hito a 4.000 y cuatro
condiciones: 60 checkpoints × cuatro = 240 celdas. La descripción anterior
decía dos brazos y dos hitos. Los resúmenes reportan 240.000/240.000 éxitos y
cero propuestas inválidas GM; no se reejecuta esa campaña ni se presenta
como reconstrucción independiente. No se cambió ningún resultado original.

El diseño de esa campaña sella una versión anterior de
`src/dcl/pedrorev/transfer.py`: SHA256
`615dfdf67622cd22a3f4b49960279c10002cd739b7a328bacb21d1bab187a4cd`,
distinta de la versión importada actual. Se recuperó exactamente desde
`670f268f39846768865519a138a0326a783f1fda` y se conserva en `source-recovery/`,
sin tocar el fuente actual sellado. El diff y una comparación AST confirman
que solo se añadieron un observador opcional desactivado por defecto y el
campo nullable `last` de salida. No se identifica una diferencia en las
transiciones o los contadores cuando ese observador está desactivado.
Esto recuperó el fuente histórico. Los240 archivos crudos se recuperaron
después, byte a byte, mediante el cuarto paquete.

## Consecuencias científicas

Se cierra la ausencia de datos publicados que impedía re-inferir los 105
entrenamientos de Martín y reconstruir la transferencia completa. No cambia
la conclusión científica: las campañas tienen soportes y endpoints distintos,
la ventaja estándar BE no equivale a extrapolación lógica y GM sigue sin
comparación con otro método que adquiera negativos informativos.

Se conserva la corrección derivada de 32/40 BE funcionalmente all-pass,
frente al 34/40 del protocolo original, sin modificar ese protocolo. Se
conservan los fallos del criterio de 99% en tipos no vistos, los resultados
negativos y las variantes ineficaces. El seguimiento positivo de vecinos no
resuelve por sí mismo los falsos negativos de frontera κ=0 del prospectivo
de Pedro. Ninguna repetición de datos consumidos amplía la población evaluada.

## Restauración y reproducción

La receta vigente está en `INTEGRATED_PAPER_REPRODUCE.md`. Los helpers
`integrated-release-download.py` y `integrated-release-restore.py` verifican
hashes y rechazan diferencias sobre archivos existentes. Son idempotentes:
una segunda ejecución mantiene los recibos byte a byte, comprobado en
`recovery-idempotence.json`. La restauración separa metadatos por paquete y
no sobrescribe el README del proyecto ni las fuentes de los estudios.

Dos detalles de empaquetado requieren usar la receta corregida:

- Los manifiestos externos se llaman `pedrorev-cf.SHA256SUMS`, etc.; una
  instrucción del README publicado añade `.tar.gz` al nombre del manifiesto.
  Los helpers usan los nombres reales y los SHA fijados arriba.
- Cada tar contiene `README.md` y metadatos en su raíz. Extraerlos todos
  directamente sobre el repositorio pisaría esos nombres. El restaurador
  guarda esos miembros bajo `package-metadata/<paquete>/`.

Esto no supone falta de datos científicos. La evaluación independiente por
otro investigador/agente y los acuerdos editoriales de coautores permanecen
pendientes; no se les atribuye aprobación mediante esta auditoría.

## Cierre del cuarto paquete — 1 de noviembre de2026

Se descargó `pedrorev-transfer-rows.tar.gz` (3.352.199 bytes), SHA256
`d650d45adfbaff8c992b1a24cbcf602abb7acc09b0995557e9f811ccb4d8e5a1`, y sus
dos sidecars. Los245 miembros son240 JSON originales y cinco metadatos.
Cada JSON coincide con el hash del `ROWS.sha256` anterior a la entrega;
las copias de diseño, inventario y resumen son byte-idénticas a sus originales.
Se conservan todos los datos recibidos y los documentos anteriores.

La auditoría recalcula los 240 resúmenes desde240.000 filas y comprueba sus
contadores y60 identidades de checkpoint, las cuatro condiciones y el diseño
completo, sin duplicados. Todas las ejecuciones están registradas como
exitosas; GM no emitió propuestas inválidas. CT tiene3.743.920 propuestas
inválidas, todas contenidas, y 287.110 rechazos de propuestas válidas: no son
287.110fallos globales. Se regeneran3.000 identidades de casos compartidas,
concordantes con todas las filas y los ocho hashes de controles exactos.

Este chequeo autentica archivos originales y recomputa sus métricas; no
ejecuta nuevas redes. Esos JSON no guardan colores completos ni trazas
intermedias. El congelamiento de esta auditoría tiene SHA256
`06c45d581b845bde3d814e08805b8f52d265ff0221e9406e4e770ce4a139b126`.
Los recibos,245 hashes y comprobación de recuperación idempotente están en
`results/martin-transfer-rows-20261101/`. El helper reproducible es
`scripts/integrated-transfer-delivery.py`. Artículo/PDF y resultados previos
permanecen idénticos. Este pendiente de datos queda cerrado.
