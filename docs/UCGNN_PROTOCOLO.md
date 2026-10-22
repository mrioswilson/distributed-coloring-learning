# Protocolo experimental UCGNN (`ucgnn-v1`)

Escrito y versionado **antes** de evaluar cualquier grado no observado. La
arquitectura está en [UCGNN_ARQUITECTURA.md](UCGNN_ARQUITECTURA.md).

## Reglas de no modificación

1. La arquitectura y el entrenamiento se pueden ajustar únicamente con información
   de validación del grado de entrenamiento Δ=3 (piloto técnico).
2. La arquitectura queda **congelada** en el commit que precede a la primera
   evaluación sobre Δ≠3 (la evaluación corta del piloto en Δ=5). Después sólo se
   corrigen errores de implementación (fallos, formas, etiquetas), documentados
   en el informe con su efecto.
3. No se modifica un modelo ni su protocolo después de ver sus resultados en
   grados no observados. Una arquitectura nueva sería `ucgnn-v2`, con la
   contaminación declarada y resultados etiquetados como exploratorios.
4. La selección `best` usa sólo validación de los grados de entrenamiento. La
   vista principal preregistrada es la **época 120**.
5. Todos los tamaños planificados se evalúan, en orden creciente, aunque fracasen
   los pequeños. Sólo los recursos pueden interrumpir el orden; se informa.
6. Los resultados negativos se conservan en `data/ucgnn/` y en los informes.

## Datos y semillas

- Grafos y tapes de entrenamiento/validación: flujos nombrados de M6B
  (`m6study-v1`, réplica, grado). La réplica `s` de UCGNN usa los mismos grafos y
  tapes que el especialista M6B Δ de la réplica `s` (prefijos si usa menos grafos).
- Inicialización, orden de minibatches y contrafactuales: flujos `ucgnn-v1` que
  dependen sólo de la réplica; los experimentos E1–E3 con la misma réplica comparten
  inicialización.
- Prueba principal: la cohorte congelada de M6B (`final_test`, espacio 20261014),
  idéntica en grafos y tapes a la usada para los especialistas:
  N=40/160/640/2.560/10.240/40.960 con 100/100/100/50/25/10 grafos por familia
  (REGULAR, IRREGULAR) y grado. Se verifica que no se solapa con entrenamiento ni
  validación (huella SHA-256).

## Experimentos

Presupuesto común: N=40 REGULAR, tres tapes por grafo, **2.400 trayectorias**,
120 épocas, warmup 10, lotes de 128 rondas, Adam 1e-3, `hidden=64`, `message=32`.
Validación local: 200 grafos MIXED en total, repartidos entre grados entrenados.
Validación autónoma: 16 grafos por grado entrenado (REGULAR/IRREGULAR × N=40/160).

| Id | Grados de entrenamiento | Grafos por grado | Pregunta |
|---|---|---|---|
| P0 | 3 | 800 | piloto técnico, 20 épocas |
| E1 | 3 | 800 | extrapolación a Δ=5, 8, 10 |
| E2 | 3, 10 | 400, 400 | interpolación a Δ=5, 8 |
| E3 | 3, 5, 8, 10 | 200 × 4 | entrenamiento mixto |
| A1 | 3 | 800 | ablación `RAW_PREFIX` (prefijo sin normalizar) |

Secundarios, sólo tras los principales: E2-full (800+800) y E3-full (800×4), que dan
a cada grado el presupuesto completo de un especialista (presupuesto total mayor).

Replicación: tras establecer viabilidad con la réplica 0, réplicas 0–4 de E1, E2 y
E3. No se lanza una batería mayor antes de esa revisión.

## Evaluación

- Grados Δ=3, 5, 8, 10 en la cohorte congelada, todas las familias y tamaños.
- Grados adicionales no observados por ningún modelo: Δ=4, 6, 7, 12, 15, cohorte
  `ucgnn_extra` (espacio 20261019), N=40/160/640/2.560/10.240 con 50/50/50/20/10
  grafos por familia. Para E3, Δ=4, 6, 7 son interpolación y Δ=12, 15 extrapolación.
- Referencias: especialistas M6B Counterfactual 800×3 (20 réplicas por grado,
  época 120 y `best`) a partir de las tablas publicadas de la prueba final; rondas
  del profesor sobre el mismo grafo y tape (oracle vectorizado, probado igual a
  `run_oracle`).
- Se distingue siempre `seen_degree` (grado de entrenamiento) de no observado, e
  interpolación (entre grados de entrenamiento) de extrapolación.

Métricas por grafo: éxito, terminación, aristas monocromáticas, rondas, commits
inseguros, conflictos con vecinos fijados y simultáneos, commits válidos
rechazados, accuracy de participación, candidato, COMMIT y acción respecto del
profesor, propuestas prohibidas, errores de disponibilidad aprendida `α` por
ranura, trayectoria idéntica al profesor y ronda de primera divergencia, todo
separado por `k`; tiempo de inferencia y memoria GPU máxima.

Propiedades de la arquitectura: parámetros por Δ, dimensión de mensajes, memoria y
tiempo de un forward en función de N y Δ (incluidas paletas nunca vistas),
consistencia ante reetiquetado de nodos (10 grafos por familia, N=160, por grado).

## Estadística

Los éxitos se informan con denominadores e intervalos exactos de Clopper–Pearson
por modelo/familia/tamaño. Las ejecuciones de distintos modelos sobre los mismos
grafos no son ensayos independientes. Las comparaciones con especialistas son
sobre los mismos grafos/tapes. Con una réplica no se estima variabilidad entre
inicializaciones. Un número finito de éxitos no demuestra corrección universal.

## Bitácora de congelación

- `65641dc`: arquitectura, entrenamiento y protocolo. Durante los tests se eliminó
  el sesgo de salida del puntuador de colores (una constante común a todos los
  colores se anula en la softmax y recibía gradiente exactamente cero).
- Piloto P0 (Δ=3, 20 épocas): datos de entrenamiento y validación idénticos a los
  del especialista M6B Δ=3 réplica 0 (semillas, huellas y hashes de tapes).
  Evaluación corta Δ=3 en la cohorte piloto: 100/100 éxitos.
- Corrección de diagnóstico (sólo evaluación): `α` importa salvo escala por nodo,
  porque `lo/hi` se normalizan por `k̂`. El diagnóstico `α>0,5` era inadecuado
  (≈80% de "errores" en el piloto con éxito total); se sustituye por `α ≥ 0,5·max α`
  y por la desviación de los extremos `[lo,hi)` respecto de la partición exacta.
  **La arquitectura queda congelada en este punto, antes de evaluar Δ=5.**
- Control suplementario declarado antes de ejecutarlo (tras ver E1, A1 y los pilotos
  P2/P3, y antes de ver E2/E3): **C1 = E1 con supervisión Own-only de COMMIT**
  (mismos datos, inicialización y presupuesto; sin propuestas contrafactuales). Mide
  si el entrenamiento contrafactual es el que hace transferible la regla de COMMIT.
  Se evalúa igual que E1 (época 120, cohorte congelada y grados adicionales).
- Diagnósticos post hoc, sin uso para seleccionar: barrido de `r_c` sobre estados del
  profesor (umbrales aprendidos frente a `j/k`) y evaluación de las épocas intermedias
  de E1 en la cohorte congelada.
