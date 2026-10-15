# Estudio sistemático del aprendizaje supervisado de coloración distribuida Δ+1

## Preguntas y alcance

Se estudia si una GNN pequeña aprende de manera reproducible un protocolo
aleatorio de coloración distribuida y si su ejecución autónoma generaliza desde
40 nodos hacia tamaños mayores. Se separan reproducibilidad entre inicializaciones,
efecto de Δ, diversidad estructural, grafos distintos, tapes y actualizaciones.
No se implementa aprendizaje reforzado ni se modifica el algoritmo profesor.

La alta accuracy local y la fiabilidad global de una trayectoria son resultados
distintos. Una sola decisión incorrecta puede invalidar una coloración grande.
Resultados experimentales finitos no permiten afirmar corrección universal.

## Auditoría histórica y preservación

El repositorio recibido estaba en `feat/m6-counterfactual`, HEAD `369f7f8`, con
once archivos del estudio M6 sin seguimiento. Se creó `feat/m6-systematic-study`
y se guardaron sin cambios en el commit local `8df8a58`. Los archivos y resultados
históricos conservan sus nombres. Los resultados nuevos usan otra carpeta.

Hallazgos reproducidos:

1. `collect_training_examples` de los entrenadores M6 históricos usa bases de
   semillas fijas 100.000/200.000; variar `--seed` cambia inicialización/orden,
   pero no los grafos ni tapes. El estudio nuevo separa flujos por réplica.
2. `evaluate` en `train_m4_seed.py` compara COMMIT sobre propuestas de la GNN con
   `y_commit` de propuestas del profesor. Cuando un candidato aprendido está
   prohibido, estas etiquetas pueden ser opuestas. Se reproduce con un vecino
   fijo de color 0, propuesta válida del profesor de color 1 y propuesta propia
   de color 0. La validación nueva usa etiquetas de las propuestas efectivas.
3. `_edge_index` histórico retorna forma `(0,)` para grafos sin aristas; ahora
   retorna `(2,0)`, comprobado con un nodo aislado. Esto preserva la interfaz
   para los grafos regulares anteriores y permite diagnósticos locales.

El warmup M6 histórico ya suministraba propuestas y etiquetas concordantes.
Se conserva ese mecanismo. La variante Δ=3 mantiene inicialización, nombres de
parámetros, pesos, salidas duras/blandas y generación contrafactual histórica.
El oracle original no requiere cambios para variar la paleta y permanece intacto.

## Diseño experimental preregistrado

El protocolo `m6study-v1` se detalla en [docs/M6STUDY.md](docs/M6STUDY.md). Fase A:
80 réplicas Counterfactual desde cero, 20 semillas por grado Δ=3,5,8,10, 800 grafos
regulares conexos, N=40, tres ejecuciones de Luby por grafo, 120 épocas, warmup 10.
Los pilotos completos Δ=3/semilla 0 y Δ=10/semilla 0 cuentan entre estas réplicas.
Las dimensiones internas M4B son constantes, con paleta Δ+1 y entradas/salidas
parametrizadas. Los parámetros son 8.967, 9.737, 10.892 y 11.662 respectivamente.

Own-only conserva datos, inicialización, orden y presupuesto; sustituye el
segundo conjunto contrafactual por las mismas propuestas propias y la misma
ponderación. ORACLE_MATCHED separa supervisión del profesor con etiquetas
concordantes de supervisión sobre propuestas propias. TRADITIONAL conserva
explícitamente la política histórica como referencia, con presupuesto comparable.

Se preparan estudios de grafos/tapes con cinco semillas Δ=3 y presupuesto fijo
de 2.400 trayectorias, además de un control de 30.720 actualizaciones. El estudio
MIXED compara regularidad contra variación de grados con semillas emparejadas.
Estas fases tienen planes independientes y no se lanzan por adelantado.

La selección utiliza exclusivamente validación independiente. Los checkpoints
de épocas 20,40,60,80,120 permiten medir épocas sin cinco entrenamientos nuevos.
La prueba final solo se ejecuta tras congelar checkpoints y configuración; usa
REGULAR/IRREGULAR, N=40,160,640,2560,10240,40960 y cohortes comunes por grado.

## Diagnósticos y análisis

Se registran finalización, corrección global, aristas monocromáticas, rondas,
nodos sin colorear, commits inseguros, conflictos con colores permanentes o
propuestas simultáneas, espera innecesaria y tiempos. Cada observación identifica
grafo, tape, réplica, distribución, grado declarado y grados reales.

El diagnóstico por k cuenta estados, accuracy de candidato, propuestas
prohibidas, commits inseguros y rechazos de propuestas válidas. El antecedente
de 25 divergencias históricas en k=1 es una hipótesis, no una conclusión para
estos nuevos datos. Las probabilidades contrafactuales 0,17/0,04 se mantienen
invariables. Una posible ablación dirigida a k=1 necesita un protocolo distinto.

Las tablas llevan denominadores e intervalos apropiados. Entre semillas se
reportan media, mediana, dispersión, extremos y bootstrap de semillas condicionado
en el test observado. Los métodos se comparan de forma emparejada. Los estados
locales correlacionados no se presentan como observaciones binomiales independientes.

## Estado y limitaciones

Los resultados nuevos deben consultarse en los archivos originales de
`data/m6study-release/` y las tablas/figuras producidas por `experiments/m6study aggregate`.
Este documento no copia resultados históricos como evidencia de reproducibilidad
del estudio nuevo. Los smoke tests comprueban ejecución, no fiabilidad científica.

La validación de selección es pequeña (16 grafos por candidato) y tiene incertidumbre.
El test de 40.960 nodos contiene diez grafos por distribución/modelo; sus intervalos
son necesariamente amplios. Los tamaños grandes exigen evaluar memoria y tiempo.
La aleatoriedad es reproducible en el mismo entorno; la reanudación rechaza cambios
de fuente, versiones o dispositivo. El servidor/contenedor debe seguir encendido;
un reinicio requiere reanudar la cola. Los resultados negativos e incompletos se
conservan y se distinguen de éxitos.

La conclusión científica quedará pendiente de las réplicas y del test congelado.
Deberá separar aproximación local del profesor, fiabilidad global observada,
estabilidad de rondas y ausencia de garantías universales.

## Incidente técnico durante el piloto

El primer intento Δ=3 completó la época 1 (148 actualizaciones), pero la escritura
del historial JSON falló al serializar un resultado exitoso `numpy.bool_`. Se
conserva en `data/m6study/`, incluido el checkpoint y traceback. Se corrigió
la conversión a booleano nativo y se añadió un test de persistencia de una
coloración exitosa. Las ejecuciones principales usan `data/m6study-release/`
y comienzan desde cero con el código corregido. No se cambian etiquetas ni
acciones de inferencia. Este intento técnico no cuenta como réplica científica.

## Validación técnica de la entrega

Se ejecutaron 93 tests con éxito, incluidos reanudación exacta en CPU/CUDA,
recuperación tras interrupción, gradientes de todos los parámetros, ausencia de
shield, separación de datos y evaluación/figuras reanudables. Los smoke tests
CUDA de Δ=3,5,8,10 completaron tres épocas cada uno. Con los checkpoints
históricos M4B, Own-only y Counterfactual de semilla 10001 se comprobaron
salidas duras/blandas exactas y una trayectoria autónoma idéntica de siete
rondas (grafo técnico de semilla 931, tape 815). Estas comprobaciones son
validación de implementación y no resultados del nuevo test final.

## Cierre de A y del bloque inicial B

Terminaron las 80 réplicas Counterfactual y los 20 controles Own-only de Δ=3,
todos con 120 épocas y checkpoints completos. Los checkpoints seleccionados
obtuvieron cada uno 16/16 éxitos y cero commits inseguros en validación. Esta
validación se empleó para seleccionar checkpoints, por lo que no se interpreta
como estimación independiente de fiabilidad final.

En los 20 pares de Δ=3 se verificó igualdad exacta, a época 120, de datos,
inicialización, actualizaciones, exposición y pesos de PARTICIPATE/CANDIDATE/
message1; los pesos de message2/COMMIT difieren en todos los pares. La igualdad
de resultados de validación no distingue aún los métodos. El registro por par
está en `data/m6study-release/reports/phase-b-initial-pairing-verification.json`.

La continuación de los planes de B, referencias, C y D está descrita en
[docs/M6CONTINUATION.md](docs/M6CONTINUATION.md). Antes del test final se revisarán
las ampliaciones de presupuesto y se fijarán las vistas de checkpoints, incluida
la comparación a igual época para preservar el emparejamiento de P/C/message1.
