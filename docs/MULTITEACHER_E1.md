# Repetición de siete maestros con E1

Protocolo `e1-native-v1`, fijado el 22 de octubre de 2026 antes de entrenar
las semillas oficiales. Este seguimiento corrige la elección de arquitectura
de `native-policy-v1`: aquella comparación usó una GNN distinta, de 36719
parámetros. Sus resultados originales se conservan completos.

## Qué significa E1 en esta comparación

E1 identifica `UniversalColorGNN`, variante NORMALIZED y pérdida COUNTERFACTUAL.
Se importan sus módulos canónicos sin modificar
`src/dcl/universal/model.py` ni los demás fuentes originales. Se mantienen
64 dimensiones internas, mensajes de 32, MAX, un estado por nodo/color,
disponibilidad aprendida, prefijos normalizados y tres decisiones separadas.

| Maestro | Modelo en este seguimiento | Adaptación explícita |
|---|---|---|
| Luby | E1, 16035 parámetros | Ninguna entrada adicional; participación 0,5 |
| Johansson | E1, 16035 parámetros | Etiquetas de participación 1 |
| JP | E1 + contexto nativo, 19396 parámetros | Diferencias de identificadores y decisión de participación después del primer mensaje |
| ColorTrials repetido | E1 + contexto nativo, 19396 parámetros | Preferencias por color y disponibilidad **aprendida** en el segundo mensaje; selección de candidato después de este intercambio |
| BE caso base Δ≤3 | E1 + contexto nativo, 19396 parámetros | Fases Linial/KW, comparaciones auxiliares y consultas a las opciones nativas |
| Especulativo GM | E1 + contexto nativo, 19396 parámetros | Fase, comparación de posición suministrada por el calendario, diferencia de ID y clasificación de marcas |
| Greedy secuencial | E1, 16035 parámetros | Calendario nativo determina qué vértice puede escribir |

Las extensiones añaden 3361 parámetros compartidos entre ranuras. Todos los
modelos incluyen exactamente el mismo núcleo E1 de 16035 parámetros. Los cuatro
maestros que requieren contexto comparten también la arquitectura de adaptadores;
los parámetros de cada réplica se inicializan iguales entre esos maestros.
Las entradas no utilizadas por un maestro se anulan. Luby/Johansson/greedy no
pueden leer identificadores, preferencias, fases auxiliares ni relaciones
precalculadas. Ningún algoritmo paralelo recibe un rango global de partición.

**Alcance de las adaptaciones.** E1 sin entradas adicionales no puede representar
reglas que dependen de identificadores o preferencias que nunca observa. Por eso
estos cuatro modelos extendidos se nombran explícitamente y no se presentan
como checkpoints E1 canónicos. En ColorTrials, COMMIT recibe disponibilidad
aprendida y su gradiente puede alcanzar ese canal; esto también difiere de E1.

En la fase Linial de BE se predice el punto `x` con una cabeza compartida de
tamaño variable (campo de 13 o 7 elementos), y el adaptador local suministrado
calcula el par `(x, p_v(x))` con las evaluaciones polinomiales ya dadas. El estudio
anterior aprendía una clase conjunta entre 169 pares. Esta restricción a pares
localmente válidos es un cambio de representación y un sesgo inductivo adicional;
una diferencia BE no puede atribuirse exclusivamente al núcleo neuronal.
En KW se aprenden participación, opción y confirmación; en GM-DETECT se aprende
la marca. No se sustituyen esas decisiones por sus etiquetas exactas.
En ambas comunicaciones de KW, el canal de ocupación de cada vecino indica
si su auxiliar coincide con la opción consultada del receptor. Esta comparación
de estado no es una lista de opciones libres: los módulos aprenden a agregarla
y decidir. Es necesaria también en COMMIT para distinguir consultas
contrafactuales bloqueadas de las libres.

## Datos y presupuesto fijados

- Siete maestros, cinco semillas de inicialización 0–4 por maestro: 35 modelos.
- Se reutilizan **los mismos pools inmutables** de `native-policy-v1`: 800 grafos
  REGULAR, N40/Δ3, tres tapes, paleta común de cuatro colores. Las cinco semillas
  ven el mismo pool; no son cinco réplicas de generación de datos.
- 17760 actualizaciones Adam, tasa 0,001, 1680 decisiones de raíz muestreadas por
  actualización: 29836800 decisiones por modelo. Muestreo con reemplazo.
- Warmup de 1480 actualizaciones: 1/12 del presupuesto, correspondiente a la
  fracción 10/120 del E1 original. No se presenta esto como repetir sus 120
  épocas de lotes de 128 rondas; se conserva el presupuesto de la comparación.
- Inicialización del núcleo con la función original `model_seed(seed,
  "initialization")`; minibatches/contrafactuales con generadores privados.
- Hitos 100,500,1000,4000,8000,17760. Validación local en los mismos 50 grafos y
  cuatro rollouts N40/N160 REGULAR/IRREGULAR de seguimiento del estudio anterior.
- Todos los modelos se evalúan en el último paso obligatorio. No se elige una
  semilla, checkpoint ni presupuesto usando el resultado de validación o test.

La implementación por vecindarios completos a distancia dos evita procesar
estados enteros por una sola decisión secuencial. Tests comparan logits y
gradientes del adaptador canónico con `UniversalColorGNN` sobre el grafo completo.
El pool antiguo se abre solo para lectura y se verifican sus hashes. La caché
greedy anterior conserva exactamente los cuatro tensores de entrada y etiquetas.

Para evitar repetir millones de MLP idénticos, los mensajes binarios de E1
se evalúan una vez por patrón posible (4 en etapa1,16 en etapa2). En el caso
canónico también se reutiliza `state1` por estado y subconjunto de mensajes.
Son evaluaciones de los mismos módulos, no reglas exactas de disponibilidad.
Se verifica equivalencia numérica de logits/gradientes; no se promete igualdad
bit a bit con el entrenador original ni de subgradientes ante empates MAX
entre categorías diferentes. La reanudación y el paralelismo de esta
implementación sí se comprueban bit a bit en controles técnicos.

## Pérdida y contrafactuales

Para decisiones de color: `L_P + L_C + 0,5 L_COMMIT_propio +
0,5 L_COMMIT_contrafactual`. Durante warmup, las dos pérdidas COMMIT reciben
propuestas del maestro. Después, las etiquetas siempre se calculan para las
propuestas efectivamente recibidas: no se copia una confirmación del maestro
cuando la red propuso otra cosa.

Cada raíz muestreada es un contexto local independiente. Las inyecciones de
conflictos fijos (0,17) y simultáneos (0,04) se hacen sobre las propuestas de
esa estrella local. Es la pérdida E1 con un muestreador local documentado,
**no** una reproducción de la distribución de inyecciones de su entrenador
original de grafos completos. ColorTrials/KW reciben además consultas de
candidato uniformes con probabilidad 0,5.

Las reglas de confirmación siguen la semántica nativa: Luby/Johansson/JP
rechazan conflictos fijos y propuestas simultáneas iguales; asignación GM y
greedy consultan los colores fijados; KW consulta sus opciones auxiliares;
ColorTrials confirma solo un color que gana la comparación nativa de
preferencias. Estos oráculos calculan **etiquetas de entrenamiento y métricas**;
no filtran la inferencia. BE-Linial usa entropía cruzada del punto y GM-DETECT
usa BCE de la marca, decisiones que no tienen el protocolo P/C/COMMIT original.

## Inferencia y evaluación

Se conserva el intérprete nativo original: calendarios secuenciales/fases
suministrados, toda predicción aplicada, cero consulta al maestro durante
inferencia aprendida. Un conflicto irreversible termina los cuatro algoritmos
paralelos, con límite de 128 pasos. Esta convención coincide con el estudio
anterior; difiere del evaluador E1 original que continúa tras colisiones y
admite 256 rondas. La comparación distribucional E1/Luby publicada aparte
conserva aquella semántica y no se mezcla con estos números.

Se congelan los 35 checkpoints y hashes antes de generar las evaluaciones de
este seguimiento. Se reutilizan los mismos 396 casos por modelo: N40/160/640/
2560/10240, REGULAR/IRREGULAR, PATH/CYCLE y K4. Son casos de test **ya publicados**,
por lo que este seguimiento es exploratorio; no es un nuevo test ciego.
Las nuevas decisiones de diseño no se reajustan a sus resultados.

Total previsto: 13860 rollouts aprendidos y 2772 referencias exactas, 16632
evaluaciones. Se verifican coloraciones completas, fallos y casos emparejados
con el estudio anterior. Resultados por maestro, semilla, familia/tamaño,
componentes y fases; media/rango descriptivos, sin tratar tapes y semillas
sobre un mismo grafo como grafos independientes. Pasos nativos de algoritmos
secuenciales y paralelos no representan el mismo coste LOCAL.

## Interpretación y reproducción

El estudio anterior usó una cabeza de acción directa; este usa las decisiones
E1 y entrenamiento contrafactual. La comparación antes/después mide el paquete
**arquitectura + representación + entrenamiento**. No es una ablación causal
que cambia exclusivamente la arquitectura. El control entre siete maestros
comparte datos, presupuesto y núcleo inicial, con las extensiones aquí descritas.
La evidencia finita no demuestra corrección universal ni equivalencia con los
algoritmos exactos. Revisión científica independiente pendiente.

La revisión de implementación anterior al test detectó que la primera versión
del adaptador KW omitía esa ocupación en su segundo mensaje. Se agregó un test
de dos estados auxiliares propios con la misma consulta y confirmaciones
opuestas; falla en la versión anterior y pasa con la corrección. El prefijo
anterior se conserva como técnico, sin entrenamientos BE ni evaluaciones
finales. La cohorte corregida comienza desde cero, sin reutilizar pesos ni
cambiar grafos, semillas o presupuesto a partir de resultados de test.

Desde la raíz del repositorio:

```bash
.venv/bin/python -m pytest -q tests/test_multiteacher_e1.py tests/test_universal.py tests/test_multiteacher.py
.venv/bin/python experiments/run_multiteacher_e1.py \
  --root /tmp/dcl-multiteacher-e1-20261022 \
  --source data/multiteacher-20261019 --trainers 1
```

No sobrescribe el estudio antiguo ni los cinco E1/Luby ya publicados. Al
reanudar se comprueban configuración, datos, código, versiones, inicialización,
optimizador y ambos generadores privados. Las operaciones de entrenamiento
no consumen RNG global después de inicializar; los entrenadores opcionales
comparten solo pools de lectura y agrupan las réplicas por maestro.
