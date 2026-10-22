# Aprender algoritmos distribuidos de coloración

Estudiamos si una red neuronal pequeña puede aprender una regla local de coloración y ejecutarla de forma autónoma en grafos mucho mayores que los usados para entrenar. **Entrenamos con 40 nodos y evaluamos hasta 40.960: un aumento de 1.024 veces.**

La pregunta central es si imitar decisiones locales alcanza para producir una coloración global correcta. Una precisión local muy alta puede esconder errores poco frecuentes que invalidan grafos grandes.

**Evaluación final completada — 17 de octubre de 2026, hora de Chile:** terminaron los 565 entrenamientos y las 931.700 evaluaciones de las 1.210 versiones de modelos. Los [resultados completos](results/m6study/2026-10-17-final/README.md) conservan todas las semillas, controles, presupuestos y fallos observados.

**Para compartir:** [informe ejecutivo en PDF, con gráficos y referencias](docs/informes/M6B_informe_ejecutivo.pdf) — cinco páginas, con un resumen autónomo en la primera.

## El problema y el algoritmo profesor

Queremos asignar un color a cada nodo de un grafo, de modo que **dos nodos unidos por una arista tengan colores distintos**. Si el máximo número de vecinos de un nodo es Δ, usamos una paleta de Δ+1 colores: por ejemplo, cuatro colores para Δ=3.

El profesor, también llamado *oracle*, implementa una regla aleatorizada de tipo Luby. Todos los nodos comienzan sin color. En cada ronda síncrona, cada nodo aún sin colorear:

1. Decide si participa, con probabilidad 0,5.
2. Descarta los colores ya fijados en sus vecinos.
3. Propone uno de los colores disponibles, elegido uniformemente.
4. Intercambia propuestas con sus vecinos. Si dos participantes vecinos proponen el mismo color, ambos esperan.
5. Confirma su propuesta (`COMMIT`) si es válida; en caso contrario espera (`WAIT`).

Un color confirmado es permanente. Cada nodo usa dos números aleatorios independientes por ronda: uno para participar y otro para elegir el candidato. Una *random tape* es la secuencia de esos números; distintas tapes producen distintas ejecuciones sobre un mismo grafo.

## Qué aprende la red

Usamos **M4B**, una red neuronal de grafos (*GNN*) con tres salidas o «cabezas»:

| Cabeza | Decisión que aprende |
|---|---|
| `PARTICIPATE` | ¿Participo en esta ronda? |
| `CANDIDATE` | ¿Qué color propongo? |
| `COMMIT` | ¿Confirmo esa propuesta o espero? |

Cada nodo aplica la misma red y los mismos pesos. `PARTICIPATE` usa su estado local y su sorteo de participación. Para proponer un color, recibe mensajes aprendidos de sus vecinos; después hay un segundo intercambio que comunica las propuestas para decidir `COMMIT`. En ambos intercambios, los mensajes se resumen mediante un máximo por componente (`MAX`).

La red recibe estados de color y números aleatorios. Debe aprender a interpretar los mensajes: durante la ejecución no recibe una lista de colores disponibles ni decisiones del profesor. Las decisiones de todos los nodos se aplican simultáneamente al terminar cada ronda.

**M4B nombra la arquitectura; M6B-Counterfactual nombra el método de entrenamiento que estudiamos.** Se entrena un modelo distinto por grado y réplica; no estamos transfiriendo un único modelo entre grados. Las dimensiones internas se mantienen fijas:

| Grado máximo Δ | Colores | Parámetros entrenables |
|---|---:|---:|
| 3 | 4 | 8.967 |
| 5 | 6 | 9.737 |
| 8 | 9 | 10.892 |
| 10 | 11 | 11.662 |

La regla usa solo información local y comunicación entre vecinos. La implementación simula esas rondas en un proceso con GPU; los nodos del grafo son los participantes del algoritmo simulado.

## Por qué usamos entrenamiento contrafactual

Entrenamos desde cero con estados extraídos de trayectorias del profesor. El aprendizaje es supervisado: el profesor proporciona ejemplos y etiquetas de decisiones correctas.

El problema delicado está en `COMMIT`: **su etiqueta debe corresponder a la propuesta que realmente está evaluando**. Por ejemplo, si un vecino ya tiene rojo, el profesor puede proponer azul y confirmar. Si la red propone rojo en ese mismo estado, su etiqueta correcta es esperar. Copiar la confirmación del profesor enseñaría una decisión peligrosa.

Durante las primeras diez épocas (*warmup*), `COMMIT` aprende con propuestas del profesor y sus etiquetas correspondientes. Después, M6B-Counterfactual combina dos clases de ejemplos:

- **Propuestas propias:** las que producen las cabezas de la red, con etiquetas recalculadas para esas propuestas.
- **Propuestas contrafactuales:** propuestas modificadas deliberadamente para exponer conflictos con colores permanentes o con propuestas simultáneas. Así se enseña a rechazar errores que aparecen poco en las trayectorias correctas del profesor.

La pérdida conjunta es `L_PARTICIPATE + L_CANDIDATE + 0,5 L_COMMIT_propio + 0,5 L_COMMIT_contrafactual`. Todos los módulos se entrenan desde cero; las decisiones discretas intermedias separan el gradiente de `COMMIT` del de las otras dos cabezas.

Comparamos cuatro métodos para separar los efectos de cada componente:

| Método | Qué supervisión recibe `COMMIT` |
|---|---|
| **COUNTERFACTUAL** | Propuestas propias y propuestas modificadas, siempre con etiquetas concordantes. Es el tratamiento principal. |
| **OWN_ONLY** | Solo propuestas propias después del warmup; repite su evaluación para mantener la ponderación y el presupuesto del tratamiento. Aísla el aporte de los contrafactuales. |
| **ORACLE_MATCHED** | Propuestas del profesor con etiquetas concordantes durante todo el entrenamiento. |
| **TRADITIONAL** | Reproduce la política histórica, incluida la aplicación de etiquetas del profesor a propuestas de la red después del warmup. Es una referencia histórica con esa discrepancia explícita. |

La comparación principal Counterfactual / Own-only comparte datos, inicialización, orden y número de actualizaciones. A la época 120 se verificó que los pesos de `PARTICIPATE`, `CANDIDATE` y del primer módulo de mensajes son exactamente iguales en los 80 pares. Las diferencias quedan en el segundo módulo de mensajes y `COMMIT`.

## Cómo está organizado el estudio

La configuración principal usa **800 grafos de 40 nodos, tres tapes por grafo y 120 épocas**, con 20 réplicas por grado. Una réplica usa una semilla que determina su inicialización, grafos y tapes; los controles emparejados reutilizan esos mismos datos y condiciones.

Las familias de grafos son:

- **REGULAR:** grafos conexos donde todos los nodos tienen grado Δ. Es el entrenamiento principal.
- **IRREGULAR:** grafos conexos obtenidos eliminando aristas, con grados variables y máximo Δ.
- **MIXED:** entrenamiento con mitad REGULAR y mitad IRREGULAR, para estudiar generalización estructural. En la prueba se informan ambas familias por separado.

| Fase | Pregunta | Entrenamientos únicos |
|---|---|---:|
| **A — Tratamiento principal** | ¿Se reproduce Counterfactual en 20 semillas y cuatro grados? | 80 |
| **B — Controles emparejados** | ¿Qué cambia al quitar las propuestas contrafactuales? | 80 |
| **Referencias** | ¿Cómo se compara con supervisión del profesor y entrenamiento histórico? | 15 |
| **C — Presupuestos** | ¿Importa más variar grafos, tapes o actualizaciones? Incluye la ampliación seleccionada con validación. | 310 adicionales |
| **D — Estructura** | ¿Qué aporta entrenar en MIXED? | 80 |
| **Total entrenado** | | **565** |
| **E — Prueba final** | ¿Funcionan los modelos congelados en grafos independientes y mucho mayores? | Sin nuevos entrenamientos |

C comienza con cinco semillas en Δ=3 y amplía tres presupuestos seleccionados mediante validación a 20 semillas y cuatro grados. Se reutilizan configuraciones ya entrenadas, por eso la tabla cuenta entrenamientos únicos. El detalle está en el [protocolo y la ampliación de presupuestos](docs/M6VALIDATION.md).

## Qué mide la evaluación final

Ejecutamos la red ronda tras ronda sobre sus propios estados, sin que el profesor corrija o filtre sus acciones. Las reglas exactas se consultan para registrar diagnósticos después de decidir; no intervienen en la evolución del grafo.

Distinguimos tres resultados:

- **Imitación local:** cuánto coinciden las decisiones con la regla del profesor.
- **Éxito global:** todos los nodos terminan coloreados y ninguna arista une nodos del mismo color, dentro del límite de 256 rondas.
- **Rondas:** cuánto tarda el proceso en terminar.

Una decisión distinta de la del profesor puede seguir produciendo una coloración válida; por eso estas métricas se informan por separado.

También contamos confirmaciones inseguras, conflictos y errores según **k**, la cantidad de colores aún disponibles para un nodo. Tener k=1 es un caso delicado, pero no suponemos que todos los fallos se concentren allí.

La prueba final usa grafos y tapes independientes del entrenamiento y la validación, compartidos entre los modelos de un mismo grado para permitir comparaciones emparejadas:

| Nodos | Grafos REGULAR | Grafos IRREGULAR |
|---|---:|---:|
| 40 | 100 | 100 |
| 160 | 100 | 100 |
| 640 | 100 | 100 |
| 2.560 | 50 | 50 |
| 10.240 | 25 | 25 |
| 40.960 | 10 | 10 |

Antes de abrir esa prueba se fijaron dos formas de tomar los pesos de cada modelo (*checkpoints*): **`best`**, la época elegida exclusivamente por validación, y **`epoch-120`**, la última época común. Esta última conserva el emparejamiento de las cabezas entre Counterfactual y Own-only; `best` puede escoger épocas distintas.

Se evaluaron ambas versiones de los 565 modelos y cuatro épocas intermedias de las 20 réplicas principales Counterfactual Δ=3: **1.210 versiones y 931.700 ejecuciones modelo–grafo**. Son observaciones correlacionadas por compartir grafos o pesos. Los resultados del test no se usan para cambiar el entrenamiento ni la selección.

## Resultados finales publicados

La prueba final terminó el **17 de octubre de 2026 a las 22:43, hora de Chile**. La entrega contiene las **931.700 ejecuciones modelo–grafo** de las 1.210 versiones previstas.

Para los modelos principales a **época 120**, entrenados en REGULAR con 40 nodos, los resultados en **40.960 nodos** son:

| Método | Δ | Réplicas evaluadas | Coloraciones correctas | Confirmaciones inseguras |
|---|---:|---:|---:|---:|
| Counterfactual | 3 | 20/20 | 400/400 | 0 |
| Counterfactual | 5 | 20/20 | 400/400 | 0 |
| Counterfactual | 8 | 20/20 | 400/400 | 0 |
| Counterfactual | 10 | 20/20 | 400/400 | 0 |
| Own-only | 3 | 20/20 | 390/400 | 10 |
| Own-only | 5 | 20/20 | 330/400 | 367 |
| Own-only | 8 | 20/20 | 172/400 | 1.016 |
| Own-only | 10 | 20/20 | 98/400 | 96.781 |

Cada fila combina **20 modelos × los mismos 20 grafos grandes**: diez REGULAR y diez IRREGULAR. Las 400 ejecuciones no son 400 grafos independientes. Las tablas completas incluyen también las referencias, MIXED y los distintos presupuestos.

En los seis tamaños de prueba, Counterfactual principal a época 120 obtiene **61.600/61.600 coloraciones correctas**. El presupuesto sí importa: al reducir a 200 grafos de entrenamiento con tres tapes, baja a 345/400 en Δ=8 y 164/400 en Δ=10 para los grafos de 40.960 nodos.

En Δ=3, ambos métodos usan en promedio aproximadamente 18,2 rondas en esos grafos grandes. La diferencia observada está en la corrección, sin aumento del promedio de rondas en esa comparación. Los intervalos y la variación entre semillas están en las tablas enlazadas; los resultados finitos no establecen corrección universal.

**La elección del checkpoint también importa.** Con la política `best`, Counterfactual Δ=3 presenta 108 fallos de 770 pruebas en la réplica de semilla 17, cuya época seleccionada fue la 10, durante el warmup. Counterfactual Δ=8 presenta un fallo de 770 pruebas en la semilla 18, seleccionada en la época 30. Estos denominadores reúnen todos los tamaños de esa réplica. Los fallos se conservan en los resultados; las dos políticas de selección estaban fijadas antes del test.

![Comparación final de Counterfactual y Own-only](results/m6study/2026-10-17-final/main-comparison.png)

Consultar el [análisis final](results/m6study/2026-10-17-final/ANALISIS.md), el [resumen por familia y tamaño](results/m6study/2026-10-17-final/success-summary.csv), las [comparaciones emparejadas](results/m6study/2026-10-17-final/paired-by-seed.csv) y los [fallos individuales](results/m6study/2026-10-17-final/errors.csv). El [manifiesto final](results/m6study/2026-10-17-final/snapshot.json) identifica el protocolo, denominadores y hashes de todos los modelos. Se conserva el [corte parcial del 16 de octubre](results/m6study/2026-10-16/README.md) como registro histórico.

## Dónde seguir leyendo y colaborar

| Archivo o carpeta | Para qué sirve |
|---|---|
| [Análisis final](results/m6study/2026-10-17-final/ANALISIS.md) | Resultados, incertidumbre, controles, presupuestos y conclusiones del estudio completo. |
| [REPORT.md](REPORT.md) | Preguntas científicas, auditoría histórica y decisiones del diseño. |
| [docs/M6STUDY.md](docs/M6STUDY.md) | Protocolo base, instalación CUDA, entrenamiento, reanudación y métricas. |
| [docs/M6VALIDATION.md](docs/M6VALIDATION.md) | Revisión de validación, ampliación de C y protocolo final de E. |
| [configs/m6study/](configs/m6study/) | Planes concretos de cada fase. |
| [src/dcl/oracle.py](src/dcl/oracle.py) | Algoritmo profesor. |
| [src/dcl/m4_models.py](src/dcl/m4_models.py) | Arquitectura M4B y sus tres cabezas. |
| [src/dcl/counterfactual.py](src/dcl/counterfactual.py) | Propuestas modificadas y etiquetas exactas para `COMMIT`. |
| [src/dcl/study/](src/dcl/study/) | Datos, entrenamiento, evaluación, persistencia e informes del estudio actual. |
| [tests/](tests/) | Comprobaciones de reglas, emparejamiento, reanudación y evaluación. |
| [docs/HISTORIA_M1_M4.md](docs/HISTORIA_M1_M4.md) | README histórico: cómo llegamos desde aprender componentes aislados hasta la red completa. |

Para una primera revisión, seguir este recorrido: **este README → análisis final → protocolo → código de entrenamiento y contrafactuales**. Al proponer nuevos experimentos, separar las hipótesis nuevas del protocolo congelado que se evaluó.

## Preparar una copia local

Requiere Python 3.11 o posterior:

```bash
git clone https://github.com/mrioswilson/distributed-coloring-learning.git
cd distributed-coloring-learning
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,study]'
```

Para reproducir el entorno GPU usado en el estudio, seguir las [instrucciones de instalación CUDA](docs/M6STUDY.md#instalación-y-comprobaciones) y el [registro de versiones](requirements-m6study.lock). En una copia de trabajo sin otra cola GPU activa, las comprobaciones se ejecutan con `python -m pytest -q`.

GitHub incluye código, planes, tablas, figuras y hashes. Los checkpoints y los resultados originales por grafo permanecen en el servidor, fuera de Git por su volumen. Reanudar la evaluación o regenerar el corte con [el exportador](experiments/export_m6study_results.py) requiere esos archivos originales; las tablas publicadas pueden consultarse directamente.

## Comparación de siete maestros con fases nativas

El [plan native-policy-v1](configs/multiteacher/comparison.json) añade los cinco
maestros solicitados y vuelve a entrenar Luby/Johansson como controles comunes.
Las reglas están separadas del aprendizaje en
[programs.py](src/dcl/multiteacher/programs.py). El alcance es grafos simples no
dirigidos de grado máximo tres,
cuatro colores globales y etiquetas de vértices contiguas, con identificadores
aleatorios distintos del universo 0..65535.

| Maestro | Regla y alcance implementados |
|---|---|
| Luby | Participación 0,5, color libre uniforme, esperar ante propuesta vecina igual. |
| Johansson | La misma regla de ensayo de color con participación 1. |
| Jones–Plassmann | Versión sincrónica por prioridades fijas: máximos locales de identificador entre pendientes; menor color libre. |
| ColorTrials, Schneider–Wattenhofer | Preferencia discreta por cada color libre; gana si supera estrictamente todas las preferencias vecinas de ese color. Conserva el menor color ganado y repite. |
| Barenboim–Elkin, caso base Δ≤3 | Dos reducciones polinomiales de Linial desde los IDs, después reducción de Kuhn–Wattenhofer hasta cuatro colores. |
| Especulativo, Gebremedhin–Manne | Cuatro bloques de orden aleatorio; asignación simultánea por posición, detección de choques entre vértices de la misma posición y reparación secuencial. |
| Greedy secuencial | Un vértice por paso en orden de identificador creciente; menor color libre. |

**Alcance de las atribuciones:** ColorTrials es la subrutina publicada repetida,
con preferencias 0..3 usando la cota global de grado. No se implementa
RankingTrials ni se atribuye a esta repetición la cota rápida del algoritmo
completo de Schneider–Wattenhofer. En Δ=3, el esquema Delta-Color de
Barenboim–Elkin entra en su caso base: se implementa esa rama con inicialización
de Linial y reducción KW, sin la recursión defectiva para grados mayores ni la
constante temporal de su inicialización más rápida. No son implementaciones
generales de esos dos artículos.

Fuentes primarias: [Jones–Plassmann](https://doi.org/10.1137/0914041),
[Linial, sección 4, construcción geométrica](https://courses.csail.mit.edu/6.852/08/papers/Linial.pdf),
[Schneider–Wattenhofer, ColorTrials](https://tik-old.ee.ethz.ch/file/22a700b74fc2f37acb56d690e62c86cf/podcfp107_schneider_188.pdf),
[Barenboim–Elkin, versión 2, Algorithm 2](https://arxiv.org/pdf/0812.1379v2),
[Kuhn–Wattenhofer, Theorem 5.1](https://ac.informatik.uni-freiburg.de/publications/publications/podc06.pdf)
y [Gebremedhin–Manne, Algorithm 1](https://eecs.wsu.edu/~assefaw/publications/Concurrency12-2000.pdf).
Las versiones consultadas se conservan en `references/` de la ejecución.

Se usa una GNN común de dos rondas de mensajes, 36.719 parámetros y cabezas
para color/WAIT, color auxiliar y marca de reparación. Aprende las elecciones
locales dentro de los calendarios y fases suministrados por cada programa. Se
proporcionan colores/IDs/sorteos, evaluaciones polinomiales y relaciones locales
de igualdad; no se afirma que aprenda aritmética de cuerpos finitos. El greedy
conserva su calendario centralizado. En ejecución aprendida se aplican las
acciones sin consultar al maestro ni filtrar conflictos. La ausencia de oracle
se verifica con una prueba que prohíbe invocar sus decisiones.
Luby/Johansson/JP/ColorTrials tienen un límite de 128 pasos; una coloración
incompleta cuenta como fallo. BE, GM y greedy recorren sus calendarios nativos.

Presupuesto fijado antes del entrenamiento final: **cinco inicializaciones por
maestro**, 800 grafos cúbicos N40, tres trayectorias por grafo, **17.760
actualizaciones de 1.680 decisiones locales**: 29.836.800 decisiones supervisadas
por modelo. Las cinco semillas comparten el mismo pool; miden variabilidad de
inicialización/muestreo, no cinco muestras independientes de datos. Los grafos,
tapes e inicializaciones se emparejan entre maestros. Las trayectorias nativas
difieren; cada decisión habilitada tiene la misma probabilidad de muestreo.
Hay 50 grafos nuevos de validación local y cuatro ejecuciones autónomas de
seguimiento N40/160, comunes a todas las redes.

Este presupuesto vuelve a la escala de exposición a nodos de M6B, pero la
arquitectura, representación y pérdida directa son nuevas para admitir estados
auxiliares y reparación. Los controles nuevos permiten comparar maestros bajo
ese protocolo; no replican exactamente M6B ni el piloto Johansson anterior.
Las curvas usan hitos 100/500/1000/4000/8000/17760 y muestran media y rango entre
semillas, sin seleccionar checkpoints. El runner general admite tres procesos
con dos hilos; la reanudación restaura optimizador y generadores.

En esta máquina, el mmap sobre el disco compartido y la presión de memoria
obligaron a reanudar en disco local. El driver operativo
[run_multiteacher_threads.py](experiments/run_multiteacher_threads.py) comparte
el intérprete entre hasta tres entrenamientos independientes, protegiendo inicialización
y restauración del RNG. Se verificó igualdad bit a bit de pesos y muestreo frente
a la ejecución serial, también al mezclar modelos nuevos y reanudados. Los
modelos no usan azar de torch después de inicializarse. El plan científico,
modelo, pérdida y presupuesto se conservan; los cambios de concurrencia y sus
drivers quedan archivados con hashes. El prefijo del disco compartido se
preserva en `data/multiteacher-virtiofs-prefix-20261019/`.
Al coexistir pools grandes, el calendario
[run_multiteacher_grouped.py](experiments/run_multiteacher_grouped.py) mantiene
activas sólo réplicas del mismo maestro, con una barrera entre maestros. La
prueba completa de14 modelos/168 evaluaciones conservó exactamente los pesos
de la semilla0 para los siete maestros frente al calendario serial.
Para greedy, [run_multiteacher_cached.py](experiments/run_multiteacher_cached.py)
usa en RAM los mismos contextos de dos saltos, empaquetados por
[rooted_multiteacher_cache.py](experiments/rooted_multiteacher_cache.py), sin
cambiar muestreo ni tensores. Se verificaron los96.000 ejemplos completos,
una actualización B1680 y la reanudación exacta; los14 modelos del control
técnico coinciden bit a bit con la ejecución anterior. El pool original y los
hashes de la caché se conservan para auditoría.

Los 35 modelos se congelan en el último paso antes de generar el test común:
396 casos por modelo, N40/160/640/2560/10240, REGULAR/IRREGULAR y controles
PATH/CYCLE/K4. Son 13.860 ejecuciones GNN y 2.772 referencias exactas. Se verifica
separación de grafos fuente/validación/test y se guardan las coloraciones completas.
Los controles de grado menor siguen usando cuatro colores. Los pasos nativos
secuenciales y paralelos no representan el mismo coste distribuido. Cinco
semillas y cero fallos observados no demostrarían corrección universal.

```bash
.venv/bin/python -m pytest -q tests/test_multiteacher.py
.venv/bin/python experiments/run_multiteacher.py --plan configs/multiteacher/comparison.json --root data/multiteacher-20261019
.venv/bin/python experiments/analyze_multiteacher.py --root data/multiteacher-20261019
```

La raíz de resultados conserva fuentes, pools con hashes, checkpoints,
congelación, evaluaciones y `summary.json`, `success-by-seed.csv`,
`learning-curves.csv` y las curvas PNG/PDF. El intento técnico de verificación
se guarda aparte en `data/multiteacher-technical-20261019/`. Revisión científica
independiente pendiente.

Una revisión previa al test detectó una entrada auxiliar no local: la posición
en bloques especulativos estaba presente también en los otros maestros. Se
restringió esa posición y su relación de igualdad a GM, con una prueba por
maestro. El prefijo anterior se conserva íntegro como intento técnico en
`data/multiteacher-global-slots-prefix-20261019/`; no había congelación ni test
final. La comparación corregida regenera los pools y entrena todos los modelos
desde cero con el mismo presupuesto y semillas.

Entradas supervisadas almacenadas en el pool común (800 grafos ×3 trayectorias):

| Maestro | Decisiones almacenadas | Pasos nativos medios por trayectoria N40 |
|---|---:|---:|
| Luby | 248.584 | 7,904 |
| Johansson | 189.256 | 4,657 |
| Jones–Plassmann | 256.080 | 5,939 |
| ColorTrials repetido | 162.455 | 3,290 |
| BE, caso base Δ≤3 | 1.440.000 | 15 |
| Especulativo GM | 197.348 | 22,228 |
| Greedy secuencial | 96.000 | 40 |

Cada modelo recibe 29.836.800 extracciones con reposición de su pool. Estas
entradas no son muestras estadísticamente independientes. Los pasos de esta
tabla describen programas con distintos modelos de cómputo; no constituyen una
comparación de velocidad distribuida. `paired-cost.csv` compara cada GNN con
su maestro sobre exactamente los casos en que esa GNN tuvo éxito.

La precisión local se desglosa por fase y se acompaña de la frecuencia de la
etiqueta mayoritaria en cada fase de validación (`teacher-task-baselines.csv`).
Eso permite distinguir progreso aprendido de una precisión alta obtenida
simplemente esperando o dejando casi todas las marcas de reparación en cero.
`learning-by-phase.csv` y sus figuras PNG/PDF muestran las fases auxiliares/KW
de BE y las fases de GM; para la detección se usa F1 de las marcas positivas.

**Resultado final, 20 de octubre de 2026.** Los 35 modelos completaron el
presupuesto y quedaron congelados antes del test. Pasaron las 2.772
referencias exactas; se evaluaron las 13.860 ejecuciones GNN restantes.
La tabla muestra porcentajes de coloraciones completas correctas; la
precisión local mide imitación en validación nueva REGULAR N40.

| Maestro | Precisión local (%) | REGULAR N40 (%) | REGULAR N10240 (%) | IRREGULAR N10240 (%) | Éxitos GNN / 1.980 |
|---|---:|---:|---:|---:|---:|
| Luby | 98,936 | 100,00 | 76,67 | 20,00 | 1.857 |
| Johansson | 98,272 | 100,00 | 93,33 | 53,33 | 1.919 |
| Jones–Plassmann | 99,989 | 99,60 | 33,33 | 26,67 | 1.635 |
| ColorTrials repetido | 100,000 | 100,00 | 100,00 | 100,00 | 1.980 |
| BE, caso base Δ≤3 | 99,564 | 93,00 | 0,00 | 0,00 | 872 |
| Especulativo GM | 99,995 | 100,00 | 100,00 | 100,00 | 1.975 |
| Greedy secuencial | 100,000 | 100,00 | 100,00 | 60,00 | 1.842 |

N40 contiene 50 grafos regulares por dos tapes y cinco inicializaciones
(500 ejecuciones GNN). Cada celda N10240 contiene **tres grafos** por dos tapes
y cinco inicializaciones (30 ejecuciones); no son 30 grafos independientes.
Las curvas muestran el rango entre inicializaciones, no un intervalo de confianza.

ColorTrials resolvió los cuatro casos de seguimiento en las cinco semillas
desde el hito 500 (840.000 decisiones supervisadas/modelo), cuando la imitación
local media todavía era 94,47 %. Al final alcanzó 100 % de imitación local y
1.980/1.980 coloraciones válidas, incluidos PATH/CYCLE/K4. Este resultado
finito favorece a ColorTrials repetido bajo este protocolo; no demuestra que
la regla aprendida sea correcta para todos los grafos.

Greedy llegó al 100 % de imitación local en las cinco semillas a 500 updates,
pero falla al cambiar de topología: 100 % en todos los tamaños REGULAR frente
a 60 % en IRREGULAR N10240 (rango entre semillas 0–100 %). Jones–Plassmann
terminó con 99,989 % de imitación local, pero 288 ejecuciones incompletas y
57 colisiones. Por tanto, una imitación casi perfecta sobre estados del
maestro en grafos cúbicos pequeños no garantiza ejecución autónoma al escalar
o cambiar de distribución.

En BE, la fase auxiliar Linial terminó con 96,73 % de precisión local media
y la reducción KW con 100 %. La precisión agregada 99,564 % oculta esa
diferencia: la etiqueta mayoritaria de KW ya obtiene 94,47 %. Los fallos
autónomos finales son 684 colisiones y 424 salidas de paleta; ninguno de los
casos N10240 termina correctamente. El desglose identifica la fase con
errores persistentes, sin establecer una causa única mediante una ablación.

GM terminó con 100 % en asignación y detección locales (F1 de marcas positivas
igual a 1) y 99,821 % en reparación. Sus cinco fallos finales pertenecen a la
semilla3: tres tapes del K4 y dos casos IRREGULAR N160. Las 123 fallas de Luby,
61 de Johansson y 138 de greedy son colisiones. No se aplicaron filtros de
validez ni decisiones del maestro durante las ejecuciones GNN.

Pasaron **75 pruebas dirigidas** de los módulos multiteacher/Johansson.
La auditoría final comprobó los 36 fuentes congelados, 35 pesos, presupuestos
e inicializaciones emparejadas, drivers operativos/caché y todos los archivos
de pools. Son 800 grafos de entrenamiento, 50 de validación local y 197 de
test, sin solapamiento; los cuatro grafos de seguimiento también se excluyen
del test. Todas las coloraciones completas se verificaron contra las aristas
y la paleta de cuatro colores. Revisión científica independiente pendiente.

Artefactos finales: [comparación CSV](results/multiteacher/2026-10-20/comparison.csv),
[informe JSON](results/multiteacher/2026-10-20/comparison.json),
[curvas de aprendizaje](results/multiteacher/2026-10-20/learning-curves.png),
[aprendizaje por fase](results/multiteacher/2026-10-20/learning-by-phase.png),
[transferencia por tamaño](results/multiteacher/2026-10-20/success-by-size.png)
y [resultados por semilla](results/multiteacher/2026-10-20/success-by-seed.csv).
Las tres figuras también se entregan en PDF y sus valores en CSV/JSON.

La [entrega compacta versionada](results/multiteacher/2026-10-20/manifest.json)
conserva planes, hashes de modelos, tablas, curvas y procedencia. El archivo
completo `data/multiteacher-20261019/` conserva datasets, pesos y coloraciones
por vértice fuera de Git; se necesita ese archivo para repetir su auditoría
completa. Los comandos anteriores permiten volver a ejecutar el protocolo.
Los CSV versionados usan saltos de línea LF según `.gitattributes`; el
manifiesto conserva tanto su hash versionado como el hash del archivo original.
Los hashes de `analysis-provenance.json` corresponden al archivo original.

## Cinco réplicas nuevas E1 frente a Luby

La [entrega E1 del 21 de octubre de 2026](results/e1-retrained/2026-10-21/README.md)
reúne código, configuración, procedencia, tablas y cuatro gráficos de cinco
réplicas nuevas UniversalColorGNN NORMALIZED/COUNTERFACTUAL, de 16.035 parámetros.
Son checkpoints nuevos a época 120, separados de los históricos no disponibles.
Cada réplica logró 20.480/20.480 coloraciones correctas en los grafos aleatorios
evaluados; ninguno de los 100 tests MMD detectó diferencias con Luby tras Holm.
Se observan pequeñas diferencias locales: la ausencia de rechazo no demuestra
equivalencia general. Revisión científica independiente pendiente.


## Repetición de siete maestros con E1 — 22 de octubre de 2026

Se repitió la comparación con el núcleo UniversalColorGNN NORMALIZED y
entrenamiento contrafactual:16035 parámetros para Luby/Johansson/greedy;
19396 con contexto nativo para JP/ColorTrials/BE/GM. Los35 modelos comparten
los pools y presupuestos anteriores:800 grafos N40/Δ3, tres tapes,
17760 actualizaciones de1680 decisiones y cinco inicializaciones por maestro.
Se evalúan396 casos comunes por actor, hasta N10240, con16632 ejecuciones.

Luby, Johansson, ColorTrials, BE y greedy logran1980/1980 éxitos cada uno;
JP obtiene1631 y GM1951. Luby mejora respecto de1857 y Johansson de1919.
JP empeora en los tamaños mayores aunque su tasa global sea parecida; GM
registra29 ejecuciones incompletas. La comparación incluye cambios de
representación y entrenamiento:en BE se aprende el punto y se suministra su
valor polinomial. Es un seguimiento exploratorio sobre casos ya publicados.

El [informe con curvas y límites](results/multiteacher-e1/2026-10-22/README.md),
la [tabla antes/después](results/multiteacher-e1/2026-10-22/architecture-comparison.csv)
y la [auditoría](results/multiteacher-e1/2026-10-22/audit.json) conservan los
resultados de las cinco semillas y todos los fallos. El
[protocolo](docs/MULTITEACHER_E1.md) y las
[definiciones de métricas](docs/MULTITEACHER_E1_METRICAS.md) explican las adaptaciones.
Los fuentes originales E1 y resultados anteriores se conservan íntegros.
