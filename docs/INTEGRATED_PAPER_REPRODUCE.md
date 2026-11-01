# Reproducir la versión integrada

Todos los comandos parten de la raíz del repositorio integrado. Leer primero
las reglas de coordinación del workspace y reservar turno si se escriben
resultados. No se deben ejecutar builders históricos contra esta versión.

## Entorno

Python ≥3.11; el entorno usado está enumerado en
`requirements/integrated-paper-tested.txt`. PyTorch se usa en CPU float32
con un hilo. Instalar la rueda CPU apropiada al sistema. Para compilar hacen
falta pdfLaTeX, BibTeX, Latin Modern y los paquetes de `main.tex`; para revisar
las páginas se usa pypdfium2/Pillow. No se utiliza el binario histórico x86
de Tectonic ni se cambia su recibo.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements/integrated-paper-tested.txt
source .venv/bin/activate
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

En los comandos siguientes, `python` denota el intérprete de ese entorno.
Un sistema con otras versiones debe registrar las suyas y comprobar paridad;
el recibo describe el entorno usado, no garantiza neutralidad numérica universal.

## Reconstrucción desde resultados compactos

Los dos respaldos históricos se versionan en partes para respetar el límite
por archivo de GitHub. Antes de verificar un clon limpio, recomponerlos:

```bash
python scripts/restore-coauthor-snapshots.py
```

El helper comprueba cada parte y el SHA256 de los archivos completos,
conservando cualquier archivo ya existente con bytes idénticos. No reemplaza
archivos diferentes. No descarga datos ni modifica el artículo.

```bash
python scripts/build-coauthor-assets.py
python scripts/build-coauthor-paper.py
python scripts/verify-coauthor-integration.py
```

El builder de figuras lee `functional-summary.json` y el análisis de soporte.
El builder de PDF fija `SOURCE_DATE_EPOCH` según el congelamiento diagnóstico,
ejecuta LaTeX/BibTeX y guarda fuentes efectivamente leídas, versiones, hashes,
texto, fuentes tipográficas y renders. La inspección visual no se infiere
automáticamente de compilar.

Una compilación con otra versión de LaTeX puede cambiar bytes del PDF: el
verificador compara la entrega congelada. Para documentar una reconstrucción
distinta se prepara otro manifiesto con `package-coauthor-integration.py
--manifest-only`, preservando primero el recibo y manifiesto entregados.
No se reescribe ningún congelamiento científico para conseguir que pase.

## Restaurar los dos runtimes portátiles

Antes de extraer, comprobar hashes y miembros con el verificador anterior.
Los tar contienen exclusivamente archivos regulares bajo `data/` y sus
índices registran cada SHA256. Extraer desde la raíz en una copia de trabajo
destinada a reproducir, preservando archivos ya existentes:

```bash
tar -xzf results/paper-revision-20261030/paper-revision-runtime.tar.gz
tar -xzf results/integrated-paper-20261031/integration-replay-runtime.tar.gz
python scripts/verify-coauthor-integration.py --restored-runtime
```

El primero contiene los 300 archivos originales del ensayo de 40 modelos,
incluidos bancos/pesos. El segundo contiene siete pesos originales por SHA
para las quince celdas históricas con fallos; hay pesos reutilizados entre
identidades. Los datos de los 105 entrenamientos de la rama paralela se
restauran desde los tres paquetes externos descritos a continuación.

## Restaurar el release de Martín

Desde la raíz de esta integración, después de comprobar la entrega compacta:

```bash
python scripts/integrated-release-download.py
python scripts/integrated-release-restore.py
python scripts/verify-coauthor-integration.py --restored-runtime
```

Los hashes están fijados en el descargador y en
[MARTIN_ARTIFACT_DELIVERY_AUDIT.md](MARTIN_ARTIFACT_DELIVERY_AUDIT.md).
Se verifica cada uno de los 1.988 miembros contra su manifiesto interno.
Los archivos científicos se restauran en sus rutas originales; los README,
SHA256SUMS y provenance de cada paquete se conservan por separado bajo
`results/martin-artifacts-20261031/package-metadata/`. Los archivos existentes
solo se aceptan si sus bytes ya coinciden. Una diferencia detiene la operación.
Los helpers no regeneran recibos existentes con otra fecha. El verificador
con `--restored-runtime` comprueba también los tres archivos descargados y
todos los miembros restaurados, y compara de nuevo los signos de los logits
guardados por nuestra inferencia contra las respuestas originales.

No extraer estos tres tar directamente sobre la raíz: contienen nombres
README/metadatos compartidos. El manifiesto externo se llama, por ejemplo,
`pedrorev-cf.SHA256SUMS`, sin `.tar.gz` en su nombre. Los pesos nuevos del
follow-up están en el primer paquete y sus nuevos bancos/respuestas en el
segundo; se necesitan los tres paquetes para el alcance completo.

`--preserve-prior` es una opción del restaurador para archivar una integración
anterior antes de actualizarla; no hace falta en un clon limpio. Nuestra
entrega anterior ya está preservada, con 844 miembros autenticados, en
`prior-integration.tar.gz`, indexado por `prior-integration.json`.

## Reanalizar sin entrenar ni alterar resultados históricos

```bash
python scripts/integrated-reconcile-evidence.py
python scripts/integrated-reinfer-continuations.py
```

El primero recompone etiquetas y claves desde arrays y valida la aritmética
de los resúmenes adicionales. El segundo realiza de nuevo las 160.000
predicciones de continuación y compara todas las métricas. No se confunde
la validación inicial de resúmenes de Martín con repetir su inferencia:
esta última se ejecutó separadamente tras recibir el release.

## Ejecutar simulaciones nuevas en otro directorio

Los drivers originales son reanudables: omiten resultados ya guardados.
Para una repetición efectiva, el wrapper exige un destino inexistente, copia
el congelamiento original y vuelve a ejecutar los casos en serie. No borra
ni reemplaza resultados. Los dos comandos completos son:

```bash
python scripts/replay-coauthor-integration.py functional --destination /tmp/dcl-functional-replay-unique
python scripts/replay-coauthor-integration.py transfer --destination /tmp/dcl-transfer-replay-unique
```

El primero realiza 30.000 ejecuciones; compara todos los vectores/estados
y resúmenes científicos con la entrega. El segundo realiza las 15.000
ejecuciones históricas y compara sus clasificaciones con las publicadas.
Los tiempos se miden otra vez y no tienen por qué coincidir. Los casos son
los mismos consumidos; repetirlos no genera nuevos tests independientes.

Existe una verificación técnica reducida de 960 ejecuciones (primer cohorte
de 32 casos, seis modelos, cinco modos), que comprueba cada fila contra el
prefijo de los resultados completos:

```bash
python scripts/replay-coauthor-integration.py functional --limit 32 --destination /tmp/dcl-functional-smoke-unique
```

Los drivers científicos originales y su congelamiento permanecen intactos:
`integrated-functional-diagnostic.py` y `integrated-replay-transfer.py`.
No se repite el comando `freeze` sobre la entrega. La prueba de escrituras
redundantes se repite con `integrated-observe-redundant-writes.py` y produce
un recibo separado; es una observación posterior, sin intervención.

## Repetir los análisis del release con ejecución efectiva

`integrated-release-model-audit.py` y `integrated-release-raw-audit.py` son
los drivers sellados de esta auditoría; no se modifica su freeze. Sus modos
de inferencia/rollout son reanudables, por lo que ejecutar el driver sobre
la entrega existente puede omitir trabajo ya guardado. Para una repetición
efectiva usar el wrapper con un destino inexistente dentro de `results/`:

```bash
python scripts/replay-release-artifacts.py all --destination results/fresh-replay-release-unique
```

Ejecuta en serie inventarios, reconstrucción de etiquetas/análisis, 220
evaluaciones de continuación, 120 evaluaciones funcionales y reconstrucción
terminal histórica. Guarda nuevos logits y todas las filas funcionales.
`model-replay-freeze.json` se copia sin cambiarlo; el freeze de auditoría
terminal conserva las mismas entradas científicas con rutas del nuevo destino.
Los resultados originales permanecen intactos. Las filas funcionales nuevas
también se comparan con las 120.000 de nuestra entrega; los programas originales
de Martín solo habían guardado resúmenes funcionales, por lo que no se afirma
comparación por vector terminal con filas originales que no existen.

Se pueden repetir las tres partes por separado en destinos nuevos:

```bash
python scripts/replay-release-artifacts.py infer --destination results/fresh-replay-infer-unique
python scripts/replay-release-artifacts.py functional --destination results/fresh-replay-functional-unique
python scripts/replay-release-artifacts.py raw --destination results/fresh-replay-raw-unique
```

Un chequeo técnico de inferencia usa solo la primera referencia fija y sus
449.280 consultas; no sustituye la repetición completa ni elige un modelo:

```bash
python scripts/replay-release-artifacts.py infer --first-checkpoint --destination results/fresh-replay-release-smoke-unique
```

El wrapper exige que el destino esté dentro del árbol `results/` porque los
drivers sellados registran rutas relativas al checkout. Conviene ejecutar
una sola tarea neuronal a la vez en un host con memoria limitada. Los tiempos
de este host no se usan como comparación de eficiencia científica.

## Pruebas pertinentes y disponibilidad

```bash
python -m pytest -q -p no:cacheprovider tests/test_coauthorintegration.py tests/test_integration_support_witness.py tests/test_paperrevision.py tests/test_multicf.py tests/test_multiteacher.py tests/test_multiteacher_e1.py tests/test_degree.py tests/test_integration_teachers.py
```

La entrega registra 330 pruebas aprobadas en la selección original y una
adicional del prefijo completo con IDs legales. Esas pruebas verifican
correspondencia y regresiones; no sustituyen las demostraciones ni certifican
todas las políticas. Los resultados negativos y modos ineficaces permanecen.

No faltan miembros de los manifiestos internos de los tres paquetes. Se verifican ahora los
bancos y pesos de los 105 entrenamientos y todas las filas históricas.
Esto no equivale a regenerar todo el entrenamiento o las 810.000 trayectorias
neuronales completas. Las comparaciones siguen usando casos consumidos;
ningún conteo de reproducción se añade a las réplicas originales. La revisión
científica independiente y las decisiones editoriales siguen pendientes.

## Cuarto paquete: originales de las 240 celdas adicionales

La entrega se recupera y comprueba con:

```bash
python scripts/integrated-transfer-delivery.py
python scripts/verify-coauthor-integration.py --restored-runtime
```

El helper fija el SHA256 del cuarto tar, verifica los 240 hashes originales,
restaura solo las filas en sus rutas y guarda los cinco metadatos en su propio
namespace. Rechaza bytes diferentes preexistentes y conserva los recibos en
una repetición. También ofrece etapas `download`, `restore` y `audit`.
Los metadatos originales `ROWS.sha256`, `design.json` y `summary.json` quedan
intactos. Se recomputan todas las métricas/contadores y se reconstruyen las
identidades de casos con el generador original, sin ejecutar las redes.

El tar de3.352.199 bytes y sus sidecars se conservan en
`results/martin-transfer-rows-20261101/downloads/`, con SHA256
`d650d45adfbaff8c992b1a24cbcf602abb7acc09b0995557e9f811ccb4d8e5a1`.
Este paquete pequeño también está incluido en el manifiesto compacto.
No quedan rutas faltantes de los cuatro paquetes entregados. Los JSON de
esta campaña contienen indicadores y contadores por ejecución, sin vectores
completos de colores ni trayectorias intermedias. No se afirma reconstruirlos
ni repetir240.000 rollouts neuronales. El resultado no añade un test nuevo.

Para repetir esa otra campaña se debe usar además la versión de
`transfer.py` fijada por su `design.json`, recuperada por hash en
`results/martin-artifacts-20261031/source-recovery/`. La versión actual
añade un observador opcional y un campo nullable de salida; se conservan
diff/recibo y no se reemplaza el fuente actual. Los drivers de su época se
ejecutan en otra copia, preservando el diseño y sus salidas originales.
