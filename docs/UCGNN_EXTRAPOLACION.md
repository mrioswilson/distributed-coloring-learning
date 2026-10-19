# UCGNN: auditoría independiente y extrapolación extrema v1

Protocolo fijado el 19 de octubre de 2026, antes de la evaluación principal CUDA.
Rama `feat/ucgnn-extreme-degree-study`; base original `708bb57`.
Worktree: `/workspaces/AlgoLearning/ucgnn-extreme-degree-study`.
No se publican cambios ni se escriben checkpoints históricos.

## Pregunta y propiedades distintas

¿Hasta dónde funciona una política de tamaño fijo entrenada sólo con Δ=3,
K=4 y N=40? Se distinguen: admitir paletas variables, producir coloraciones
propias, reproducir al profesor, extrapolar conjuntamente N y Δ, y coste.
Los éxitos finitos no prueban funcionamiento para grados arbitrarios.

## Qué implementó Claude

La implementación original permanece en `src/dcl/universal/`:

| Elemento | Referencia y comportamiento verificado |
|---|---|
| Arquitectura | `model.py`: 16.035 parámetros NORMALIZED; estado/mensaje 32, capas ocultas 64; sin constructor dependiente de K |
| Entradas | Estado local de color, dos números aleatorios, aristas y paleta pública; sin etiquetas ni disponibilidad exacta |
| Primera comunicación | Mensajes por nodo y color; máximo componente a componente de vecinos inmediatos |
| Selección | Disponibilidad aprendida α; suma/prefijo explícitos; intervalos normalizados; MLP compartida y argmax dentro de la paleta |
| Segunda comunicación | Mensajes de estado y propuestas; máximo de vecinos; COMMIT neuronal sobre el slot propuesto |
| Entrenamiento | `data.py`, `train.py`, `rules.py`; trayectorias del profesor a N=40; pérdidas de participación/candidato/COMMIT |
| Contrafactual | Inyecciones de propuestas conflictivas; etiquetas recalculadas para las propuestas recibidas; sólo entrenamiento |
| Evaluación | `evaluation.py`: las acciones se fijan antes de los observadores exactos; no hay shield ni reparación de candidatos |
| Semillas y datos | Streams de grafos/cintas compartidos con M6; inicialización/orden/contrafactual específicos de UCGNN |

No se encontró fuga de etiquetas en la ruta de inferencia revisada. Los tests
originales comprueban ejecución sin oracle y ausencia de shield. Las máscaras
de nodos ya coloreados y del rango público de paleta pertenecen al protocolo.
No hay pesos ni embeddings aprendidos específicos de nombres de color.

La arquitectura incorpora conocimiento algorítmico explícito: `cumsum` y
normalización de α. No corresponde describirla como una MLP que descubrió
por sí sola todas las operaciones de Luby. El denominador se trunca a 0,5;
α sólo está identificado hasta una escala y no debe interpretarse su suma
cruda como un conteo exacto. El prefijo impone el orden de los identificadores
de color. COMMIT comparte pesos y no usa ese orden. Relabelar colores y
mantener el mismo `r_c` no exige la misma selección de candidato, tampoco
para el profesor, que selecciona sobre colores ordenados.

El máximo sobre vecinos no depende de multiplicidades. El grafo global no
se reduce para tomar decisiones; `palette.max()` sólo fija la dimensión de
relleno en lotes disjuntos. Suma y prefijo son operaciones locales entre los
colores del mismo nodo. Hay dos comunicaciones inmediatas, sin difusión
global ni adyacencia densa N×N.

El protocolo original documenta una corrección previa del diagnóstico de α:
un umbral absoluto 0,5 clasificaba erróneamente disponibilidades cuyo nivel
aprendido era menor. Claude conservó esos pilotos bajo
`archive/ucgnn_pilot-diagnostic-v0` y pasó a un umbral relativo al máximo por
nodo. Esa corrección de métricas no modifica las acciones. Los resultados
4.880/4.880 auditados proceden de las carpetas vigentes, no del archivo v0.

## Resultado 4.880/4.880 y auditoría

Checkpoint: `e1-normalized-counterfactual-s00-90d67ee02dc0/epoch-120.pt`.
SHA256 del archivo:
`382448e45d600c9bb2e77ce217e3392185996e67b7a5b95d5d05e6c178cd2eed`.
Réplica/semilla 0, época 120; 800 grafos REGULAR de N=40, tres cintas, sólo Δ=3.

| Cohorte original | Grados | N | Grafos por familia y tamaño | Total |
|---|---|---|---|---|
| final_test | 3,5,8,10 | 40,160,640,2560,10240,40960 | 100,100,100,50,25,10 | 3.080 |
| ucgnn_extra | 4,6,7,12,15 | 40,160,640,2560,10240 | 50,50,50,20,10 | 1.800 |

Se contaron los JSON originales, no sólo las tablas: 4.880 ejecuciones, 4.880
fingerprints y semillas de grafo distintos, 4.880 éxitos, cero commits
inseguros. Una cinta por grafo y un solo modelo. No hay N=40.960 en la cohorte
de grados adicionales. Los grafos `IRREGULAR` originales cumplen máximo
**≤Δ**, y algunos no alcanzan el límite declarado. La nueva familia alcanza
**exactamente Δ** protegiendo todas las aristas de un hub.

Se reprodujeron 16 casos en CPU, de ambas familias: (Δ,N)=(3,40), (3,640),
(3,40960), (10,160), (10,640), (15,40), (15,640), (15,10240). Coincidieron
éxito, terminación, rondas, errores y recuentos locales discretos comparados.
Esto valida una muestra representativa, no vuelve a ejecutar los 4.880 casos.
La reproducción CPU es explícita y se separa de la batería principal CUDA.

Los primeros 50 JSON del piloto CPU conservan un campo `edge_chunk` heredado
del evaluador original; no describe el bloqueo real del adaptador. El
adaptador siempre usa ocho colores y 32 MiB por bloque de mensajes. No se
reescriben esos registros. La evaluación principal registra explícitamente
`edge_chunk_policy=per-color-block-32MiB` y `color_chunk=8`.

El mismo objeto de modelo procesó grafos Δ=3/10/20: hash de todos los pesos y
buffers idéntico, mismos objetos de parámetros, sin optimizador ni capas
nuevas. Se guardan los hashes completos en `audit/reproduction.json`.

El inventario inicial contiene 20 entrenamientos completos: E1/E2/E3 con
cinco réplicas cada uno, A1 y C1 con una, y P0/P2/P3 de 20 épocas. E2F/E3F
están en la cola original de Claude con presupuestos superiores; esa cola
sigue intacta. El inventario es una fotografía temporal, no convierte los
trabajos posteriores de Claude en pendientes de este coordinador.

## Evaluador y preservación

El nuevo paquete `src/dcl/extreme/` añade auditoría, generación, fragmentación,
trabajadores aislados, coordinación persistente, diagnósticos y reporte.
No modifica `src/dcl/universal/` ni `tests/test_universal.py`.

Cada resultado se escribe atómicamente con `fsync` después de un grafo y
contiene checkpoint/hashes antes y después, época, semilla de entrenamiento,
grados y tamaños vistos, semillas de grafo y cinta, fingerprint, versión Git
y hash de fuentes, dispositivo, métricas por k, seguridad, rondas, colores,
tiempo de inferencia y máximos de CUDA. Los observadores exactos no cambian
las acciones ni las propuestas.

Se excluyen semillas/fingerprints históricos de las evaluaciones UCGNN y de
los manifiestos e historiales guardados de M6/UCGNN: 585 manifiestos,
246.520 fingerprints en la fotografía inicial. Las semillas nuevas usan
`ucgnn-extreme-v1` con namespace 202610192218. El piloto CPU usa otro
namespace. Los completos son pruebas deterministas: sólo existe un conjunto
de aristas etiquetadas por tamaño; su reutilización CPU/CUDA se marca y no
se presenta como una nueva muestra independiente.

## Diseño predefinido y presupuesto

Configuración ejecutable: `configs/ucgnn/extreme-v1.json`.

1. **A funcional:** réplica E1-0, Δ=3/5/8/10/15/20/30/50/100/200; cinco
   familias; N=max(40,2(Δ+1)), salvo completos N=Δ+1. Todos satisfacen
   N>Δ y paridad NΔ. Primero todas las configuraciones pequeñas.
2. **Exploración Δ=500:** cinco familias si al menos el 80 % de los casos
   medidos del piloto son correctos, con límites de tiempo y memoria.
3. **B principal:** E1-0…4; REGULAR/IRREGULAR, 20 grafos por familia, grado y
   tamaño; N=160/640/2560 si es válido. Prioridad 15/20/30/50/100, después
   3/5/8/10/200. No se detiene por un fallo de coloración.
4. **Precisión:** ampliar a 60 grafos por familia en N=640 para los cinco
   grados prioritarios. Es una ampliación predefinida, no selección de casos
   favorables. Otras tasas bajas se pueden estudiar con una extensión de
   protocolo nueva; no se cambian los pesos.
5. **Estrés:** cinco grafos por familia HETEROGENEOUS/BIPARTITE en N=640 y
   grados 15/20/30/50/100/200; un completo por grado. Se mantienen separados
   del test principal. El bipartito usa un conjunto de desplazamientos
   circulantes y permutaciones, no muestreo uniforme de todos los bipartitos.
6. **C grande:** N=10.240/40.960 para grados prioritarios cuando el volumen
   lógico por ronda es ≤32 GiB y la memoria estimada ≤3 GiB. Cinco grafos por
   familia y modelo, exploratorios. La regla depende de recursos, no de éxito.
7. **D mecanismos:** barridos controlados y contrafactuales de los modelos
   existentes. No se utilizan para elegir ni modificar checkpoints.
8. **E comparación:** E2/E3 con sus cinco réplicas y A1/C1 con una; mismos
   grafos de los grados prioritarios a N=160/640, 20 por familia.

La cola contempla 13.429 ejecuciones de grafos y una acción de diagnóstico,
condicionadas a viabilidad y al presupuesto activo de 24 horas. No promete
terminarlas todas dentro de ese presupuesto. Límites: 300 s/caso ordinario,
600 s/caso grande, asignador CUDA ≤3 GiB y reserva externa ≥768 MiB.
Esperar a otra cola no consume el presupuesto activo. Los trabajadores se
ejecutan secuencialmente, en procesos propios; un timeout sólo termina su
propio proceso. OOM/timeout son fallos técnicos; nunca cuentan como
coloraciones incorrectas. Errores de programación bloquean la cola para no
acumular resultados inválidos. La CPU no sustituye automáticamente a CUDA.

E1/E2/E3 principales comparten 800 grafos totales, tres cintas y 120 épocas;
su distribución por grado cambia (800; 400/400; 200/200/200/200). Eso no
iguala exactamente rondas supervisadas, actualizaciones ni tiempo. Se
registran los presupuestos reales. E2F/E3F no se mezclan con ellos.

## Mecanismos e incertidumbre

Se guardan rondas, commits inseguros, conflictos fijos y simultáneos
(pueden solaparse), commits válidos rechazados, nodos pendientes, tiempo,
memoria, exactitud local por k y frecuencias/entropía de colores.

Los diagnósticos de selección construyen estrellas parcialmente coloreadas
con conjuntos libres bajos, altos y dispersos, k=1/2/3/4/5/10/20/50/100/200/500
cuando caben, K=4/11/21/31/51/101/201/501. Comparan la malla uniforme de 1.001
valores de `r_c` y vecinos float32 de cada frontera j/k con el profesor al
mismo valor representable. Se conservan todas las propuestas en NPZ.

Se separan identificación de ocupados, conteo relativo de masa disponible,
error de intervalos y error de selección de posición. El conteo relativo no
es la suma cruda de α. En la réplica 0, K=k=201 da 5/1.001 selecciones exactas
en la malla, pese a disponibilidad correcta e intervalos con error ~1e-7.
Es evidencia de desviación en el scorer, no de un error de conteo en ese
estado. El piloto CPU de 50 grafos sigue siendo completamente correcto.

COMMIT se prueba con nombres de color de extremos e interiores y conflictos
fijos, simultáneos y múltiples. Además, para un nodo central que participa
y propone un color válido, sus vecinos tienen seis tipos binarios posibles
en ese slot. Max depende del conjunto de tipos presentes: hay 2^6=64 casos.
Las cinco réplicas E1 aciertan 64/64 en la evaluación CPU; C1 acierta 48/64.
Esta reducción se refiere exclusivamente al camino de COMMIT; no prueba
terminación ni corrección universal del algoritmo completo.

La auditoría de épocas también detecta que `best.pt` de E1-0 es **época 1**:
acierta 48/64 estados y acepta 16 commits inseguros. Época 10 da 48/64;
épocas 20 y 120 dan 64/64. La selección por validación ordinaria no garantiza
seguridad contrafactual. Se mantiene la época 120 predefinida para este
estudio. Ver `audit/ablation-and-epochs.json`, reproducible mediante
`dcl.extreme.audit.ablation_and_epochs` y la ejecución completa de `audit`.

E1-0 y C1-0 comparten exactamente datos e inicialización. Sus pesos de
participación y candidato son idénticos bit a bit en época 120; sólo cambian
los componentes de COMMIT. La pérdida contrafactual no propaga gradiente a
participación/candidato porque usa propuestas duras y parámetros separados.
Por tanto, la coincidencia de sus curvas de selección es esperable; no es
una réplica independiente de ese aprendizaje.

Los intervalos de éxito son Clopper–Pearson bilaterales del 95 %, por modelo,
grado, N y familia. Los grafos se comparten entre modelos: cinco modelos
por 20 grafos dan 100 ejecuciones sobre **20 grafos**, no 100 grafos
independientes. `shared-graphs-replicates.csv` permite comprobar qué grafos
han sido evaluados con las cinco réplicas y el éxito conjunto. No se agrupan
familias/tamaños/modelos correlacionados en un intervalo binomial global.

## Recursos y equivalencia

Original: activaciones N×K×32 y mensajes temporales E_chunk×K×32; 32 floats
por color, por arista dirigida, en cada una de dos comunicaciones. El número
fijo de pesos no implica coste de comunicación constante.

La fragmentación procesa ocho colores y hasta 32 MiB de mensajes temporales
por bloque. Primero obtiene α, normaliza sobre K completo y vuelve a
calcular la primera comunicación para puntuar. Conserva el prefijo global
**local al nodo** y la agregación max original. Stage 2 se acumula por bloques.
No consulta disponibilidad exacta ni elimina propuestas inseguras.

La equivalencia CPU compara logits con tolerancia 2e-5 y exige igualdad de
decisiones duras y pesos, en paletas hasta 501. Los tests incluyen ambas
variantes, fragmentación de aristas y una trayectoria completa. CUDA debe
pasar su propia comprobación antes de la primera evaluación principal.
Las diferencias de flotantes y su máximo se guardan; no se afirma igualdad
bit a bit de todos los logits. Cualquier diferencia de decisiones bloquea
el comienzo de la cola.

El coste lógico por ronda registrado es 2·E_dir·K·32·4 bytes. La estimación
de trabajo del evaluador por fragmentos registra también 3·E_dir·K·32·4:
recalcula la primera agregación sobre el mismo estado para ahorrar memoria.
Esto añade trabajo y lecturas, no una tercera dependencia de comunicación
del protocolo ni información de vecinos más lejanos. La selección grande
usa el volumen lógico de las dos etapas, como se fijó en la configuración.
La estimación
de memoria es conservadora y se contrasta con máximos medidos, no se
presenta como medición. Se reportan tiempo por ronda, bytes por nodo y por
arista junto con N, K y E. No se requiere enviar por red física: se mide
la implementación centralizada y se calcula su volumen lógico de mensajes.

## Operación persistente

Desde el worktree:

```bash
experiments/ucgnn-extreme start
experiments/ucgnn-extreme status
tmux attach -t ucgnn-extrapolation
tail -f data/ucgnn-extreme/logs/graphs.log
experiments/ucgnn-extreme report
experiments/ucgnn-extreme stop
```

`start` crea la sesión o informa que ya existe. Tras una interrupción, vuelva
a ejecutar `start`; el coordinador omite resultados persistidos. `stop`
solicita parada entre grafos. No se borran resultados completados. El
presupuesto consumido se conserva en `budget.json` y no se reinicia al
reanudar. Si se agota, se registra `budget_exhausted`; no se amplía solo.

La cola espera a que termine **toda** la cola anterior de Claude, incluidos
intervalos entre comandos. Luego adquiere el mismo lock global de GPU.
No detiene Ollama, Docker, Claude ni otros servicios, ni modifica drivers.
Un segundo coordinador se rechaza mediante un lock propio. Los cambios
de fuente/configuración/checkpoints requieren un protocolo/root nuevos;
commits de documentación con fuentes idénticas permiten reanudar.

Estado y resultados persisten en `/workspaces`, pero tmux no sobrevive a la
destrucción del contenedor. En un contenedor recreado, hay que recuperar el
volumen, entorno CUDA/PyTorch e intérprete (variable opcional `UCGNN_PYTHON`)
y ejecutar `start`. No se afirma arranque automático tras destruir Docker.

Archivos principales:

- `data/ucgnn-extreme/audit/`: censo, reproducción, hashes, exclusiones y gates.
- `data/ucgnn-extreme/results/`: un JSON por evaluación principal CUDA.
- `data/ucgnn-extreme/cpu-pilot/`: comprobaciones CPU, separadas.
- `data/ucgnn-extreme/graphs/`: cohortes nuevas compartidas entre modelos.
- `data/ucgnn-extreme/diagnostics/`: CSV y curvas crudas por estado/modelo.
- `data/ucgnn-extreme/logs/`, `status.json`, `budget.json`: operación.
- `data/ucgnn-extreme/report/INFORME.md` y `UCGNN_extrapolacion_parcial.pdf`:
  informe generado, tablas y gráficos, con estado temporal explícito.

Los checkpoints siguen bajo
`/workspaces/AlgoLearning/distributed-coloring-learning/data/ucgnn/runs/`.

## Tests

```bash
experiments/ucgnn-test-isolated -q
```

Se ejecutan los tests originales sin modificarlos y los nuevos de invariantes
de grafos, equivalencia, cache, diagnósticos, reanudación, intervalos,
trabajador por subprocess y clasificación de OOM. Tres tests M6 intentaban
adquirir el lock global real aun ejecutándose en CPU y chocaron con la cola
activa de Claude. El wrapper de tests usa `Path.home()` temporal únicamente
en pytest, sin cambiar HOME ni el lock real. La batería principal nunca usa
esa sustitución. La comprobación CUDA adicional se ejecuta en el coordinador
cuando puede adquirir la GPU.
