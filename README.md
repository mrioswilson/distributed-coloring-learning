# Learning Distributed Graph-Coloring Algorithms

Este repositorio contiene experimentos para estudiar hasta qué punto una red neuronal puede **aprender, ejecutar y extrapolar algoritmos distribuidos locales**.

El problema utilizado como caso de estudio es una variante aleatorizada de \((\Delta+1)\)-coloring. En los experimentos actuales trabajamos principalmente con grafos cúbicos (\(\Delta=3\)) y cuatro colores.

La idea general es separar progresivamente los distintos componentes de un algoritmo distribuido y estudiar cuáles pueden ser aprendidos a partir de ejecuciones de un algoritmo maestro (*oracle*).

---

## 1. Problema

Cada nodo debe terminar con un color en

\[
\{0,1,2,3\}
\]

de forma que para toda arista \(\{u,v\}\),

\[
c_u \neq c_v.
\]

Los nodos inicialmente están sin colorear.

El algoritmo maestro opera sincrónicamente en rondas y utiliza una **random tape local** independiente para cada nodo.

En cada ronda, un nodo no coloreado:

1. decide aleatoriamente si participa;
2. calcula qué colores siguen disponibles según sus vecinos ya coloreados;
3. elige aleatoriamente un candidato entre esos colores;
4. intercambia candidatos con sus vecinos;
5. hace `COMMIT` si su candidato no entra en conflicto;
6. de lo contrario hace `WAIT` y vuelve a intentarlo en una ronda posterior.

Los colores aceptados mediante `COMMIT` son permanentes.

---

## 2. Objetivo del proyecto

Queremos distinguir entre varias preguntas:

- ¿Puede una red neuronal imitar una regla local distribuida?
- ¿Puede ejecutar esa regla de forma iterada sin acumular errores?
- ¿Puede extrapolar desde grafos pequeños hacia grafos mucho mayores?
- ¿Qué arquitecturas permiten representar correctamente las relaciones entre mensajes vecinos?
- ¿Qué tipos de errores locales son realmente importantes para la corrección global?
- Eventualmente: ¿puede una política aprendida descubrir una regla mejor que el algoritmo maestro?

Actualmente entrenamos en grafos cúbicos aleatorios de tamaño

\[
N=40
\]

y evaluamos extrapolación hasta

\[
N=10240.
\]

---

## 3. Oracle

El oracle implementa un algoritmo aleatorizado de \(4\)-coloring para grafos de grado máximo \(3\).

Para cada nodo no coloreado \(v\), en cada ronda se generan dos números aleatorios:

```text
activation_draw[v]
candidate_draw[v]
```

La participación se decide mediante:

```text
activation_draw[v] < 0.5
```

La paleta disponible es:

\[
A_v =
\{0,1,2,3\}
\setminus
\{\text{colores fijados en vecinos de }v\}.
\]

El candidato se escoge mediante:

\[
j =
\lfloor
r_v^{(c)} |A_v|
\rfloor.
\]

Dos participantes vecinos que proponen el mismo color hacen ambos `WAIT`.

---

## 4. Experimentos M1–M4

La progresión experimental elimina gradualmente información proporcionada directamente por el oracle.

### M1 — aprender `COMMIT`

La red recibe:

- estado de color;
- participación;
- candidato.

Debe aprender solamente:

```text
COMMIT / WAIT
```

Resultado principal:

- imitación exacta de la regla local;
- 0 errores observados hasta \(N=10240\);
- ejecución closed-loop idéntica al oracle.

---

### M2 — aprender `PARTICIPATE + COMMIT`

La red recibe:

- estado de color;
- `activation_draw`;
- `candidate_offer`.

Debe aprender:

```text
PARTICIPATE
COMMIT / WAIT
```

Se comparan dos arquitecturas.

#### M2A

Agregación directa:

```text
features de vecinos
        ↓
componentwise MAX
```

Esta arquitectura pierde ciertas asociaciones entre propiedades del mismo vecino.

Por ejemplo, puede saber que:

```text
algún vecino participa
```

y que:

```text
algún vecino propone color 2
```

sin poder representar adecuadamente que:

```text
el mismo vecino participa Y propone color 2
```

El rendimiento global se degrada rápidamente.

#### M2B

Antes del `MAX`, cada nodo transforma su información mediante un MLP:

```text
features
   ↓
  MLP
   ↓
mensaje aprendido
   ↓
  MAX
```

Esto permite construir representaciones del tipo:

```text
"soy un vecino activo y propongo color 2"
```

M2B mantiene 100% de éxito de coloring hasta \(N=10240\) en la evaluación realizada.

---

### M3 — aprender `CANDIDATE + COMMIT`

La red recibe:

- estado de color;
- participación;
- `candidate_draw`.

Debe aprender:

```text
CANDIDATE
COMMIT / WAIT
```

Aquí aparece una dificultad interesante.

La selección del candidato depende de fronteras exactas como:

\[
1/4,\ 1/3,\ 1/2,\ 2/3,\ 3/4.
\]

Las redes aprenden aproximaciones muy buenas de estas fronteras, pero pequeños desplazamientos producen errores cerca de los umbrales.

En los experimentos:

- prácticamente todos los errores de candidato ocurren cerca de estas fronteras;
- lejos de ellas, la selección aprendida es esencialmente exacta;
- una pequeña tasa de error local puede convertirse en muchos errores al ejecutar millones de decisiones.

---

### M4 — aprender el algoritmo completo

La red recibe solamente:

```text
estado de color
activation_draw
candidate_draw
```

y debe aprender internamente:

```text
PARTICIPATE
    ↓
CANDIDATE
    ↓
COMMIT / WAIT
```

Se comparan nuevamente dos arquitecturas.

#### M4A

Usa agregación `MAX` directa.

Su éxito cae rápidamente al aumentar \(N\).

#### M4B

Usa MLPs para aprender mensajes antes de agregarlos.

Entrenando únicamente en \(N=40\), M4B mantiene buena capacidad de extrapolación hasta \(N=10240\).

En 10 seeds de entrenamiento independientes, para \(N=10240\):

```text
mean success ≈ 0.768
std          ≈ 0.150
min          = 0.56
max          = 0.96
```

El número de rondas aprendido permanece muy cercano al del oracle.

---

## 5. Una observación importante sobre las métricas

Una alta accuracy local **no implica necesariamente un buen algoritmo distribuido**.

Por ejemplo, distintas seeds de M4B tienen accuracies locales muy similares, pero tasas de éxito global muy diferentes.

Por eso distinguimos entre:

```text
action_accuracy
```

e

```text
success_rate
```

La acción local se interpreta como una de:

```text
WAIT
COMMIT-0
COMMIT-1
COMMIT-2
COMMIT-3
```

La métrica principal de ejecución es:

```text
success = finished AND proper_coloring
```

---

## 6. Errores seguros versus errores peligrosos

Un diagnóstico posterior de M4B mostró algo especialmente importante.

La red puede diferir muchas veces del oracle sin que eso sea necesariamente malo.

En cambio, los fracasos observados ocurren cuando aparece un **unsafe commit**:

```text
un nodo hace COMMIT con un color
que ya tiene uno de sus vecinos
```

En una muestra de 250 ejecuciones en \(N=10240\):

```text
sin unsafe commit  -> 166 éxitos, 0 fracasos
con unsafe commit  ->   0 éxitos, 84 fracasos
```

Es decir, en esa muestra:

\[
\text{success}
\iff
\text{no unsafe commit}.
\]

Esto sugiere que para aprendizaje de algoritmos distribuidos puede ser más importante estudiar la preservación de **invariantes locales de seguridad** que la simple imitación exacta del oracle.

---

## 7. Open-loop y closed-loop

### Open-loop

El oracle genera toda la trayectoria.

La red solamente predice qué habría hecho.

Un error de la red no modifica el siguiente estado.

Esto mide principalmente:

```text
calidad de imitación local
```

### Closed-loop

Las decisiones de la red se aplican realmente.

Por tanto:

```text
estado siguiente = consecuencia de decisiones aprendidas
```

Esto mide:

```text
ejecución real del algoritmo aprendido
```

Closed-loop es la evaluación principal.

---

## 8. Instalación

Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd distributed-coloring-learning
```

Crear el entorno:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instalar:

```bash
pip install -e .
```

Ejecutar tests:

```bash
pytest -v
```

Actualmente deberían pasar:

```text
19 tests
```

---

## 9. Estructura del repositorio

```text
src/dcl/
    graphs.py
    oracle.py
    dataset.py
    model.py

    staged_data.py
    staged_models.py

    m4_data.py
    m4_models.py

tests/
    test_graphs.py
    test_oracle.py
    test_dataset.py
    test_model.py
    test_staged.py
    test_m4.py

experiments/
    benchmark_oracle.py

    train_m1.py
    evaluate_m1_extrapolation.py
    evaluate_m1_closed_loop.py

    train_staged.py
    evaluate_staged.py
    evaluate_action_accuracy.py
    diagnose_m3_thresholds.py

    train_m4.py
    evaluate_m4.py

    train_m4_seed.py
    evaluate_m4_seed.py
    summarize_m4b_seeds.py
    analyze_m4b_failure_modes.py
```

Los resultados generados y checkpoints no se incluyen en Git por defecto.

---

## 10. Ejemplos de ejecución

### M1

```bash
python experiments/train_m1.py
python experiments/evaluate_m1_extrapolation.py
python experiments/evaluate_m1_closed_loop.py
```

### M2

```bash
python experiments/train_staged.py --task m2 --arch A
python experiments/train_staged.py --task m2 --arch B
```

### M3

```bash
python experiments/train_staged.py --task m3 --arch A
python experiments/train_staged.py --task m3 --arch B
```

Diagnóstico de fronteras:

```bash
python experiments/diagnose_m3_thresholds.py --arch A
python experiments/diagnose_m3_thresholds.py --arch B
```

### M4

```bash
python experiments/train_m4.py --arch A
python experiments/train_m4.py --arch B
```

Evaluación:

```bash
python experiments/evaluate_m4.py --arch A
python experiments/evaluate_m4.py --arch B
```

---

## 11. Próximos experimentos

Algunas direcciones abiertas:

### Replicación entre seeds

Ya iniciada para M4B.

Objetivo:

- separar propiedades robustas de la arquitectura de efectos de inicialización.

### Topology-OOD

Hasta ahora la extrapolación principal es:

```text
random cubic N=40
        ↓
random cubic N >> 40
```

Esto es extrapolación en tamaño.

Topology-OOD significa mantener aproximadamente las mismas restricciones locales pero probar familias estructuralmente distintas de grafos.

La pregunta es:

> ¿La red aprendió una regla distribuida general o una regularidad particular de los random cubic graphs?

### Composición modular

Idea para una etapa posterior:

```text
ParticipationModule
        ↓
CandidateModule
        ↓
CommitModule
```

Comparar:

```text
módulos aprendidos separadamente y luego ensamblados
```

contra:

```text
entrenamiento conjunto end-to-end
```

### Reinforcement Learning

Una vez caracterizada la capacidad de imitar el algoritmo:

- eliminar progresivamente la supervisión del oracle;
- usar recompensa por coloring correcto;
- penalizar número de rondas;
- permitir que la red encuentre estrategias diferentes.

La pregunta final sería:

> ¿Puede una red descubrir un algoritmo distribuido correcto y eventualmente mejor que el algoritmo utilizado como maestro?

---

## 12. Criterio general

El objetivo del proyecto no es simplemente lograr alta accuracy sobre los datos del oracle.

Queremos distinguir entre:

\[
\text{imitation learning}
\]

y

\[
\text{algorithm learning}.
\]

Una política aprendida puede diferir del oracle y aun así ser perfectamente válida si:

1. siempre produce una coloración correcta;
2. respeta las restricciones de comunicación distribuida;
3. extrapola a tamaños y topologías no observados;
4. mantiene una buena complejidad en rondas.

En última instancia, nos interesa entender **qué hace que una regla neuronal pueda comportarse como un algoritmo distribuido**.
