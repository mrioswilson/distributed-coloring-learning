# Estudio supervisado M6B (protocolo m6study-v1)

La carpeta operativa de esta entrega es `data/m6study-release/`. El intento
piloto anterior se conserva en `data/m6study/` con su traceback y checkpoint.
El código está en `src/dcl/study/`. Los scripts históricos, sus nombres y sus
resultados se conservan. Los resultados nuevos van a `data/m6study-release/`; las
configuraciones versionadas están en `configs/m6study/`.

## Instalación y comprobaciones

Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Instalar primero la variante de PyTorch CUDA adecuada al servidor.
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[dev,study]'
.venv/bin/python -m pytest -q
experiments/m6study smoke --device cuda --root data/m6study-smoke-verification
```

`requirements-m6study.lock` registra las versiones del entorno validado. El
registro de cada entrenamiento contiene además versiones, hash Git, estado
Git y hashes SHA-256 de todos los módulos fuente. Los entrenamientos principales
requieren una copia con los cambios guardados en Git. La reanudación rechaza
cambios de fuente, versiones, dispositivo, configuración o datos.

## Lanzamiento, progreso, detención y reanudación

```bash
experiments/m6study plan
tmux new-session -d -s m6study 'experiments/m6study run --plan configs/m6study/phase-a.json --root data/m6study-release'
experiments/m6study status --root data/m6study-release
tmux capture-pane -pt m6study -S -30
tmux attach -t m6study
experiments/m6study stop --root data/m6study-release
# Después de que la sesión termine, reanudar en la misma sesión nominal:
tmux new-session -d -s m6study 'experiments/m6study run --plan configs/m6study/phase-a.json --root data/m6study-release --resume'
```

Para salir de tmux y dejar la cola funcionando: `Ctrl-b`, después `d`.
La cola sobrevive al cierre de SSH y del chat. El contenedor y el servidor deben
seguir encendidos. Después de reiniciar el servidor/contenedor se usa el comando
de reanudación. No se configura un reinicio automático del servidor.

`stop` crea `data/m6study-release/STOP`. El proceso termina después de persistir la época
en curso. También maneja SIGTERM/SIGINT de esta forma. Una interrupción abrupta
dentro de una época hace repetir esa época desde el último estado completo.
Una ejecución terminada se salta automáticamente. Un fallo genera un log y un
JSON con traceback; por defecto detiene la cola. Los fallos de los dos pilotos
siempre bloquean el resto de la Fase A.

La cola adquiere un bloqueo global en `~/.cache/dcl-m6-gpu.lock`, compartido
entre distintas carpetas del estudio y con la evaluación final. Solamente hay
un entrenamiento/evaluación de esta infraestructura en GPU a la vez. Los
scripts históricos no adquieren este bloqueo: no ejecutarlos en paralelo.

Rutas:

- Registro central: `data/m6study-release/registry.sqlite` (SQLite con WAL).
- Logs separados por ejecución e intento: `data/m6study-release/logs/`.
- Checkpoints: `data/m6study-release/runs/<id>/last.pt`, `best.pt`, `epoch-020.pt`,
  `epoch-040.pt`, `epoch-060.pt`, `epoch-080.pt`, `epoch-120.pt`.
- Configuración y datos: `config.json`, `data_manifest.json` en cada ejecución.
- Métricas completas: `history.json`, actualizado después de cada época.

`last.pt` es la autoridad de reanudación: contiene modelo, Adam, época, estados
aleatorios Python/NumPy/PyTorch/CUDA, generadores de orden y contrafactuales,
historial, criterio de selección, identificación de datos, parámetros y
procedencia. La escritura es atómica con fsync; los checkpoints por época son
inmutables. El archivo `best.pt` es el puntero actual de selección por validación.

Antes de cada época se comprueba espacio libre (mínimo 10 GiB), RAM disponible
(mínimo 2 GiB) y memoria libre GPU (mínimo 512 MiB). El historial guarda RAM,
memoria GPU asignada/reservada y espacio. No cambia drivers ni mata procesos
ajenos. La cola principal exige CUDA; no pasa silenciosamente a CPU.

## Arquitectura, etiquetas y semillas

Para `q=Δ+1`, el estado tiene `q+3` entradas: one-hot de estado no coloreado y
colores, `r_p`, `r_c`. `CANDIDATE` tiene `q+1` clases incluyendo ausencia de
candidato. La entrada de PARTICIPATE mide `q+2`, la representación de segunda
comunicación `2q+5`; las dimensiones internas siguen siendo 64 y 32. M4A y M4B
aceptan `num_colors`; el valor por defecto conserva la interfaz histórica.

| Δ | Paleta | Parámetros M4B |
|---|---|---|
| 3 | 4 | 8.967 |
| 5 | 6 | 9.737 |
| 8 | 9 | 10.892 |
| 10 | 11 | 11.662 |

El oracle original ya acepta `num_colors` y se utiliza sin alterar su regla,
su umbral 0,5 ni el orden de selección de colores. Las etiquetas de COMMIT del
tratamiento y de Own-only siempre se recalculan para las propuestas suministradas.
El warmup de diez épocas usa propuestas y etiquetas correctas del profesor.
Después se optimiza `L_P+L_C+0,5 L_own+0,5 L_cf`. Las cinco partes de M4B siguen
entrenables desde cero. Las propuestas duras desconectan el gradiente de COMMIT
de las cabezas P/C; ambas reciben su propia pérdida supervisada.

La inferencia genera solamente los dos vectores de números aleatorios; no llama
al oracle. Ejecuta las acciones aprendidas, incluso si son incorrectas. Las reglas
exactas se usan después de decidir las acciones para diagnósticos y nunca filtran
commits ni cambian estados. Los colores confirmados permanecen fijos.

Las semillas proceden de SHA-256 con nombres de flujo y coordenadas: réplica,
grado, finalidad, índice de grafo y tape. Inicialización, minibatches,
contrafactuales, grafos y tapes usan flujos separados. El método y la familia de
entrenamiento no cambian la inicialización. Los controles reutilizan exactamente
los datos, la inicialización y el orden; cada réplica utiliza grafos y tapes
independientes. Los presupuestos usan prefijos reproducibles para facilitar
comparaciones de diversidad. Los manifiestos guardan semillas y fingerprints
de cada grafo y hash de cada trayectoria. Se rechaza un solapamiento de grafos
entre entrenamiento y validación/prueba.

IRREGULAR elimina una cantidad aleatoria de aristas fuera de un árbol generador,
preservando conectividad y grado ≤Δ. MIXED alterna cantidades iguales de grafos
regulares e irregulares (salvo un grafo si la cantidad es impar). PATH/CYCLE están
disponibles para diagnósticos dispersos con la misma paleta declarada.

## Protocolos y controles

- Fase A: 80 entrenamientos, semillas 0–19, Δ=3,5,8,10, 800 grafos regulares,
  N=40, tres tapes, 120 épocas. Los primeros son pilotos completos Δ=3/semilla 0
  y Δ=10/semilla 0 y cuentan entre las 80 réplicas. Un piloto debe completar
  entrenamiento, validación e inferencia antes de avanzar.
- Fase B inicial: 20 Own-only emparejados para Δ=3. El plan separado
  `phase-b-all-degrees.json` permite extender a los otros grados.
- Referencias: cinco semillas Δ=3 con ORACLE_MATCHED (COMMIT aprende únicamente
  propuestas correctas, tres cabezas desde época 1) y TRADITIONAL (política de
  pérdida histórica, incluido warmup P/C y etiquetas históricas tras warmup).
  Este último es una referencia explícita, no se presenta como supervisión
  corregida. `references-original-budget.json` añade cinco controles tradicionales
  con un tape para reproducir también el presupuesto original.
- Fase C: cinco semillas Δ=3, 45 configuraciones por épocas y 45 por actualizaciones.
  Se varía grafos con tapes=3 y tapes con grafos=800; se incluyen 200×12, 400×6,
  800×3, 1200×2, 2400×1. Se deduplican las configuraciones compartidas.
  El segundo plan fija 256 actualizaciones por época (30.720 totales), usando
  permutaciones con repetición al agotar los ejemplos. Se registran rondas y
  nodos realmente mostrados. Las trayectorias iguales no implican rondas iguales.
- Fase D: 80 entrenamientos MIXED con las mismas semillas y arquitectura que A.

Los planes B/C/D están preparados y se lanzan explícitamente con `run --plan`.
La primera cola ejecuta únicamente A. No se lanza el producto cartesiano de
presupuestos para todos los grados/semillas. No se implementa aprendizaje reforzado.

## Validación, selección y prueba final

La validación local contiene 200 grafos MIXED independientes de entrenamiento,
un tape por grafo. Se evalúa en todas las épocas, con COMMIT comparado con la
regla exacta de las propuestas propias. La accuracy de acción compara la acción
final con la del profesor y se etiqueta por separado.

La selección usa inferencia autónoma en cuatro grafos por familia (REGULAR,
IRREGULAR) y tamaño (40,160): 16 grafos de validación por candidato. Se evalúan
época 1, múltiplos de 10 y última época. Criterio lexicográfico preregistrado:
tasa de éxito, menor frecuencia de commits inseguros por estado, menor promedio
de rondas, accuracy de acción respecto del profesor, menor pérdida local corregida.
El mejor checkpoint es el mejor entre estos candidatos; no se evalúa el test
final para seleccionar checkpoints. El límite es 256 rondas.

Solo después de completar la selección se congela el protocolo:

```bash
experiments/m6study freeze --plan configs/m6study/phase-a.json --root data/m6study-release
tmux new-session -d -s m6study 'experiments/m6study evaluate --freeze data/m6study-release/frozen-final-test.json --root data/m6study-release'
experiments/m6study aggregate --root data/m6study-release
```

Para incluir controles, preparar una unión deduplicada de los planes A/B antes
de congelar. El archivo congelado conserva hashes de checkpoints y criterio de
selección. No permite congelar ejecuciones incompletas ni sobrescribirse.
Para estudiar épocas sin reentrenar, incluir `--checkpoint-epochs 20 40 60 80 120`
en el comando `freeze`, antes de ejecutar el test. Añade estas vistas además del
checkpoint seleccionado y las identifica por separado en las tablas. Esta
evaluación adicional aumenta el coste; no forma parte de la primera cola.
La prueba final es independiente y no incluye los 700 grafos históricos.
Todos los modelos usan los mismos grafos y tapes para cada grado/familia/tamaño:

| N | Grafos por familia |
|---|---|
| 40 | 100 |
| 160 | 100 |
| 640 | 100 |
| 2.560 | 50 |
| 10.240 | 25 |
| 40.960 | 10 |

Cada evaluación persiste un JSON atómico por grafo, con IDs, semillas, distribución,
grados, resultado, rondas, conflictos, rechazos válidos, tiempos y diagnósticos
P/C/COMMIT por k. Se reanuda con el mismo comando y salta grafos ya persistidos.
El contador de conflictos simultáneos evalúa todas las propuestas participantes,
aunque el vecino termine esperando; las dos categorías de conflicto pueden
solaparse. El tiempo de inferencia excluye el diagnóstico; también se guarda el
tiempo completo de ejecución. El coste incluye la generación inicial de datos.

## Tablas y figuras

`aggregate` genera CSV desde datos originales y figuras PDF/PNG en
`data/m6study-release/reports/`, separando validación de prueba final. Incluye pérdidas,
accuracies, éxito por tamaño/grado/semilla/presupuesto, rondas, coste/calidad,
comparaciones emparejadas y desglose por k. Un archivo científico `REPORT.md`
en la raíz explica el diseño; el reporte de resultados se regenera en `reports/`.
Mientras no haya test final, sus tablas permanecen vacías y no se inventan valores.

Las tasas por semilla llevan denominadores e intervalos Wilson y Clopper–Pearson
del 95%. La dispersión entre semillas usa media, mediana, desviación estándar,
mínimo, máximo y bootstrap de semillas (condicionado en los grafos observados).
Las comparaciones de métodos usan McNemar exacto sobre pares de grafo/tape,
indicando denominadores. No se suman evaluaciones correlacionadas de diferentes
modelos para simular un test independiente más grande. Las bandas de figuras
son desviación estándar descriptiva; los valores p son exploratorios y no llevan
corrección por multiplicidad. Los contadores por k conservan denominadores;
los estados de una trayectoria están correlacionados. Revisar especialmente k=1,
sin presuponer que el mecanismo histórico se repita en otros grados.
