# Reconstrucción de M6B y transferencia de la supervisión contrafactual

Estado de la evidencia: código original verificado, 21 artefactos históricos
autenticados y repetición de 30 ejecuciones históricas. Los recibos están en
`results/multiteacher-counterfactual/audit/`. No se han reescrito los originales.

## Qué significan los números 249, 476 y 500

El archivo original es
`data/results/m6_overnight_three_way_seed10001.csv`, conservado en el checkout
histórico. Tiene 500 índices de grafo distintos. Sus columnas suman:

| Política | Éxitos | Checkpoint original SHA-256 |
|---|---:|---|
| M4B | 249/500 | `ed90d0325574545d8a1ad3c2c65b5d8c03136a7e35ab77ba48d4cf02b08a9af0` |
| Control de propuestas propias | 476/500 | `f283e3c8f7db3d08ad86f9189449b581cd9c3b8a54fff5da9f5256f515fcfe28` |
| M6B-Counterfactual | 500/500 | `00cc7bbe317ae70df7c3853e99a38f715bb5969678be8c0e2a2e2dd46b450711` |

Los tres modelos corresponden a **una semilla de entrenamiento, 10001**. Cada
caso es un grafo cúbico de N=10.240 y una cinta compartida por las tres
políticas. `experiments/overnight_m6_threeway.py` utiliza semillas de grafo
`50_000_000 + 10_240*1000 + i` y de cinta
`60_000_000 + 10_240*1000 + i`. El criterio original es terminación antes de
1.000 rondas y coloración propia. Se retienen aristas incorrectas y no hay
shield. Son 500 casos apareados; no son cinco entrenamientos independientes
ni la certificación nueva de 5×1.000.

La repetición de los primeros diez casos de las tres políticas coincide en
éxito, terminación, rondas y aristas incorrectas, usando los pesos originales.
Esto autentica una repetición parcial del benchmark histórico. Las otras 490
filas se verifican como evidencia original autenticada, no como nuevas
ejecuciones. El control 476 es `train_m6_ownonly.py`; no debe confundirse con
`train_m6_oracle_control.py`, que pertenece al experimento DAgger de semilla
1234 y adquiere otras trayectorias docentes.

## Arquitectura, generación de estados y entrenamiento verificados

M6B usa la clase **M4B**, de 8.967 parámetros con K=4, hidden=64 y message=32.
Sus dimensiones dependen de K. E1 NORMALIZED canónico tiene 16.035 parámetros,
comparte funciones por nodo/color y utiliza prefijos normalizados. M6B no es
un alias de E1. El nombre M6B distingue aquí un procedimiento de entrenamiento.

`experiments/train_m6_counterfactual.py` genera 800 grafos cúbicos N40 con
semillas `100_000+i`, y tres trayectorias por grafo, con cintas
`200_000+i+1_000_000*k`. `run_oracle(...,save_trace=True)` produce las rondas;
`encode_m4_round` conserva estados completos y máscaras de vértices pendientes.
La validación original usa 200 grafos nuevos y una cinta por grafo. El batch
contiene **128 estados de grafo**, no 128 decisiones de raíz. Se barajan las
rondas completas cada época. Adam usa lr=.001. Los valores por defecto son
120 épocas y diez de calentamiento. La inicialización y el generador
contrafactual se fijan con la semilla 10001 y 10018 respectivamente.

Los checkpoints originales de control y contrafactual indican época 53,
800 grafos, tres cintas y warmup=10. El script selecciona por una terna de
validación `(action_acc,joint_acc,-loss)`. Por tanto, 120 épocas es el horizonte
configurado, y no la época del checkpoint evaluado. La cantidad de rondas y
actualizaciones reconstruida se conserva en
`audit/m6-regenerated-training-size.json`; la historia completa de Adam de ese
entrenamiento primitivo no está en esos checkpoints. No se presenta una nueva
generación de datos como recuperación de esa historia ausente.

## Intervención exacta

Durante el calentamiento se calcula COMMIT con propuestas docentes y la
etiqueta correspondiente. Después:

1. La red produce participaciones y candidatos duros propios. Se detienen los
   gradientes a través de estas decisiones discretas.
2. `exact_commit_labels` vuelve a calcular confirmaciones para **esas**
   participaciones y candidatos.
3. `make_counterfactual_proposals` copia las propuestas docentes. Cada arista
   dirigida desde un vecino fijo hacia uno pendiente puede inyectar su color
   con probabilidad .17. Cada arista no dirigida entre pendientes puede
   imponer participación y un candidato común con probabilidad .04. Las
   aristas se recorren en orden aleatorio y varias intervenciones pueden
   sobrescribir una propuesta anterior.
4. Se recalcula COMMIT sobre el conjunto final de propuestas, después de
   **todas** las sobrescrituras.
5. Las propuestas alteradas se pasan por la segunda comunicación original de
   M4B y por su cabeza COMMIT. No cambia la arquitectura.

Permanecen fijos el grafo, los colores permanentes, la máscara de pendientes
y las dos entradas aleatorias de la ronda. Se recalculan la codificación de
ofertas, los mensajes de la segunda etapa, sus agregaciones y las etiquetas.
Las intervenciones pueden proponer colores ocupados: son mensajes admisibles
para consultar la continuación de seguridad, aunque no sean propuestas que
el profesor emitiría al elegir exclusivamente de su paleta libre. No se
aplican esas propuestas para inventar una historia de estados propios.

Si `c_v` es la clase 1..K y 0 significa ausencia de candidato, la regla es

\[
y_v=\mathbf1[U_v\land p_v\land c_v\ne0]
\mathbf1[\nexists u\sim v:\neg U_u\land C_u=c_v]
\mathbf1[\nexists u\sim v:U_u\land p_u\land c_u=c_v].
\]

Los colores permanentes y los candidatos usan aquí la misma codificación por
clases. La fórmula consulta los mensajes intervenidos, no los originales.

La pérdida posterior al calentamiento es

\[
L=L_P+L_C+\tfrac12L_{M,propio}+\tfrac12L_{M,cf}.
\]

El control propio repite el mismo contexto propio en las dos mitades COMMIT.
El control tradicional conservado aplica etiquetas de propuestas docentes
sobre propuestas aprendidas; es una referencia histórica deliberadamente
separada. El control docente concordante usa las propuestas docentes al
evaluar COMMIT. Estas diferencias impiden atribuir las tres cifras sólo a
«más datos», o sólo a una arquitectura.

## Qué se transfiere

Son transferibles la supervisión por fases, el desacoplamiento del gradiente
duro, la reconsulta de confirmaciones ante mensajes efectivamente presentes,
el contraste entre decisiones propias y perturbadas, el calentamiento y la
evaluación autónoma sin filtros. No se transfieren automáticamente la
codificación K-dependiente de M4B, las probabilidades .17/.04, la validez de
una historia especulativa ni la regla COMMIT de Luby a otro profesor.

La implementación nueva tiene adaptadores declarativos en
`src/dcl/multicf/oracles.py`, historias legales repetibles en `states.py`,
bancos finitos y contabilidad en `data.py`, y pérdidas comunes a las tres
arquitecturas en `training.py`. No modifica los profesores ni sus módulos
históricos de inferencia. Los detalles y límites de cada intervención están
en `MULTITEACHER_COUNTERFACTUAL_PROTOCOL.md`.

## Clasificación de evidencia

**Implementación verificada:** funciones, pérdidas, arquitectura, semillas y
generadores descritos arriba; hashes de los 21 originales; sumas del CSV;
repetición parcial de políticas originales. **Resultado histórico reportado
y autenticado:** los 500 casos completos, de los que se han vuelto a ejecutar
diez por política. **Reconstrucción:** tamaño regenerado del entrenamiento y
extrapolación del horizonte configurado; no sustituye los logs perdidos.
Las nuevas políticas se identifican siempre como entrenamientos nuevos.
