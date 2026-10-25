# Extensibilidad matemática de los siete profesores

Auditoría de las **variantes implementadas**, base `291d4b82`, 25 de octubre
de 2026. Fuentes conservadas: `multiteacher/programs.py`,
`integration/TEACHER_CORRECTNESS_AUDIT.md`,
`TEACHER_MODEL_INFORMATION_COMPATIBILITY.md`, reconstrucciones de Pedro y
M6B. Las reglas históricas no se editan. La implementación nueva está en
`src/dcl/degree/teachers.py`; no se importa desde la inferencia neuronal.

## Clasificación y contratos

| Profesor | Clase | Extensión admitida | IDs y auxiliares | Comunicación matemática por paso |
|---|---|---|---|---|
| LUBY | B | Mismo ensayo con participación 1/2; aumentar paleta. Global Δ+1 y local d(v)+1 | Dos draws privados; sin IDs ni reloj numérico | Colores/pending; luego participación y propuestas: dos intercambios |
| JOHANSSON | B | Variante siempre activa, propuesta uniforme disponible y veto de ofertas iguales; ambas paletas | Draw privado; sin IDs | Dos intercambios como Luby |
| COLOR_TRIALS | B | Preferencias independientes en {0,…,Δ}; menor color que gana estrictamente; ambas paletas | Δ global suministrado, vector privado por color; sin IDs | Disponibilidad local en el primero; preferencias y disponibilidad del competidor en el segundo |
| JP | B | Máximo estricto de prioridad pendiente y first-fit en paleta ampliada; ambas paletas | IDs distintos y fijos, comparaciones coherentes por arista | Un intercambio basta matemáticamente; se conservan dos aprendidos |
| BE_BASE_D3 | C | **No se evalúa** fuera de Δ≤3/K4. Requiere otra variante, nuevos parámetros algebraicos y una interfaz distinta | IDs numéricos acotados, F13/F7, auxiliares, bound/target, opciones KW | Dos fases Linial y trece KW en el dominio nativo; no es la recursión BE general |
| SPECULATIVE_GM | B | Se conservan cuatro bloques, slots, detección y reparación serial; sólo paletas ampliadas; ambas paletas | IDs, slots, permiso, marks, barreras y cola suministrados | Colores en ASSIGN/REPAIR; color, slot y orden en DETECT |
| GREEDY | B | Mismo orden ascendente de IDs y first-fit; ambas paletas | Permiso de escritura del calendario externo | Un intercambio basta; dos aprendidos conservados |

Todos exigen grafos simples no dirigidos sin bucles, etiquetas contiguas,
1≤n≤65536. El límite n proviene del espacio de IDs del sistema conservado,
incluso en los experimentos anónimos; no se presenta como límite matemático
del algoritmo anónimo. El número de colores es un parámetro nuevo explícito,
por eso se usa B y no A para las seis extensiones del dominio K4.

## Proposiciones condicionales y pruebas

Las probabilidades condicionales siguientes fijan el estado persistente y
la historia **antes** de muestrear las entradas privadas de la ronda. El
objeto `Snapshot` del código incluye también esas entradas ya muestreadas;
condicionar sobre todo ese objeto haría determinista la decisión y no es
la condición usada en las cotas de progreso.

**Disponibilidad, ambas paletas.** En cualquier estado parcial propio,
si U es el conjunto pendiente y A_v la lista propia sin colores fijados por
vecinos, |A_v|≥|C_v|−(d(v)−d_U(v))≥d_U(v)+1.
El argumento sólo cuenta colores distintos prohibidos; no requiere listas
idénticas. Las listas locales aquí son {0,…,d(v)}, no listas arbitrarias
inferidas a partir de una ejecución global.

**Seguridad de Luby/Johansson.** La propuesta pertenece a A_v y COMMIT exige
ausencia de una oferta igual de todo vecino participante pendiente. No
colisiona con colores antiguos ni con otra fijación simultánea. Colores
fijados no se modifican. Esto demuestra seguridad por inducción, para el
profesor exacto y bajo las hipótesis indicadas.

**Progreso de Luby, modelo ideal.** Con propuestas propias uniformes y
participación independiente 1/2,
Pr(COMMIT_v|S)=(1/2)|A_v|⁻¹ Σ_c ∏_u(1−q_u(c)/2)≥1/4.
Se aplica log(1−x)≥−2log(2)x en [0,1/2], Jensen y
Σ_c∈A_v q_u(c)≤1. No se necesita acotar Δ por tres.

**Progreso de la variante Johansson, modelo ideal.** Si u es vecino pendiente
de v pendiente, |A_u|≥2 y q_u(c)≤1/2. La misma desigualdad y Jensen dan
|A_v|⁻¹ Σ_c ∏_u(1−q_u(c))≥2^(−2d_U(v)/|A_v|)≥1/4.
Esta prueba concierne a la variante siempre activa implementada, no importa
el teorema de otro algoritmo Johansson ni hipótesis de ausencia de triángulos.

**ColorTrials.** Un color c sólo compite si c∈A_u y u está pendiente.
Dos extremos de una arista no pueden ganar estrictamente el mismo color.
Para cualquier c∈A_v, el evento pref_v(c)=Δ y todas sus ≤Δ preferencias
competidoras <Δ tiene probabilidad al menos
ρ_CT=(Δ+1)⁻¹(Δ/(Δ+1))^Δ>0. En ese evento v fija algún color, aunque
seleccione otro ganador menor. El caso Δ=0 termina en un paso.
Las preferencias se generan con el mismo bound global en todos los nodos;
usar alfabetos locales diferentes requeriría otro análisis. Los colores
fuera de la lista de un competidor no compiten. Las dos comunicaciones
siguen siendo suficientes, con mensajes de longitud dependiente de la paleta.

**Terminación aleatoria.** Si cada pendiente progresa con probabilidad
condicional ≥ρ, Pr(T>t)≤n(1−ρ)^t y hay terminación casi segura.
`horizon` fija un cutoff con esta cota ≤10⁻⁹ en el modelo ideal, y un
mínimo práctico de 128 pasos. Se registran los cutoffs de estudiantes.
Los draws float32 y un PRNG finito no son uniformes continuos exactos:
las cotas ideales no constituyen una certificación formal del generador,
sus leyes numéricas o un checkpoint.

**JP.** Los máximos locales estrictos pendientes forman un conjunto
independiente; first-fit existe por la desigualdad de disponibilidad.
El máximo global pendiente progresa cada paso. Termina en ≤n pasos.
Se sustituye únicamente el cap histórico 128 por n. El orden relativo debe
ser transitivo y proceder de un único conjunto de IDs, no de signos
independientes por arista.

**GM.** Sólo asignaciones del mismo slot pueden crear conflictos: una
asignación posterior evita todos los colores anteriores. Marcar el extremo
de menor ID de cada conflicto del mismo slot cubre toda arista incorrecta;
el subgrafo no marcado es propio. La reparación serial visita una vez cada
marcado y evita todos los colores actuales de vecinos. El último extremo
reparado de una arista evita al otro. d(v)+1 colores propios bastan también
con listas distintas. No cambia el calendario: 2ceil(n/min(4,n))+m pasos,
m≤n. Los conflictos intermedios son legales; una predicción incorrecta de
marks no recibe reparación exacta. Las barreras/colas no se llaman rondas
LOCAL gratuitas.

**Greedy.** Un único permiso por paso conserva el orden; first-fit propio
existe por d(v)+1 y evita los colores anteriores. Termina en n escrituras.
El calendario global forma parte de la información suministrada al sistema.

## Por qué BE no se extrapola

La fase F13 histórica exige p>Δr con r=4; para Δ=4 ya no cumple
13>16. La segunda F7 y los bloques de cuatro opciones KW también están
especializados. Aumentar el grado manteniendo esas estructuras carece de la
prueba de existencia de un punto/opción libre.

Una construcción distinta Linial+KW podría elegir nuevos primos y bloques
K=Δ+1: q=ceil(KB/(K+1)), bloques disjuntos de K opciones por color viejo
≥q, K(B−q)≤q. Existencia y seguridad siguen por el conteo de Δ vecinos y
la disjunción de bloques; q<B para B>K garantiza descenso. Incluso KW desde
los IDs sería una alternativa más lenta. Esto no es ejecutar el caso base
BE_BASE_D3 original ni implementar la recursión, coloraciones defectivas y
preprocesamiento de BE general. Cambiaría fases, aritmética y entradas.
Se deja como variante futura; no se computan falsos fallos de estudiantes BE
en las tablas de transferencia.

## Evidencia y referencias

Para Luby se reutiliza como referencia diferencial la implementación
`universal/rules.py` empleada en el estudio de paletas locales hasta grado
200. Los tests nuevos comparan propuestas y COMMIT con las mismas cintas,
en ambas paletas y Δ3/10/50/200. También se contrasta el nuevo adaptador
disperso con `LocalPaletteAdapter` sobre los mismos núcleos y K variable.
Las cohortes canónicas de `ucgnn-paper-study` y `ucgnn-local-palette-study`
no son los pesos de las doce certificaciones sample-complexity: sus éxitos
de alto grado no se trasladan a éstos sin ejecutarlos de nuevo.

Las pruebas anteriores son argumentos matemáticos condicionales propios de
las reglas codificadas. Los tests diferenciales escalares, replays nativos,
validación exacta de cada cinta y resultados neuronales son evidencia finita
distinta. Ninguna ejecución finita prueba corrección universal de una red.

Se revisaron fuentes primarias el 25-10-2026:
[Jones–Plassmann](https://doi.org/10.1137/0914041),
[Schneider–Wattenhofer, ColorTrials](https://tik-old.ee.ethz.ch/file/22a700b74fc2f37acb56d690e62c86cf/podcfp107_schneider_188.pdf),
[Kuhn–Wattenhofer](https://ac.informatik.uni-freiburg.de/publications/publications/podc06.pdf),
[Barenboim–Elkin v2](https://arxiv.org/abs/0812.1379v2),
[Gebremedhin–Manne](https://eecs.wsu.edu/~assefaw/publications/Concurrency12-2000.pdf).
La fuente de Johansson y la de Luby en ScienceDirect devolvieron 403 en esta
revisión; se conserva la referencia histórica sin atribuirle pruebas nuevas
no verificadas. La fuente JP describe un heurístico asíncrono: nuestra
variante síncrona usa la prueba explícita anterior, sin heredar su complejidad.
