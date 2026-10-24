# Reconciliación metodológica con los experimentos de Pedro

Esta investigación conserva el informe fuente
`docs/integration/PEDRO_RESEARCH_RECONSTRUCTION.md` y añade recibos de una
nueva búsqueda de artefactos, en `results/multiteacher-counterfactual/audit/`.

## Fuentes y disponibilidad

Se verificaron los commits `fc756cef`, `80f486eb` y `6836d855`, y los 16 hashes
fuente registrados en `PEDRO_SOURCE_MANIFEST.json` contra el Git histórico.
Los 16 coinciden. La diferencia del profesor entre el commit original y la
base integrada es el rechazo explícito de grafos dirigidos, multigrafos y
bucles en el constructor; las reglas de decisión se conservan.

Se inventariaron 6.614 archivos locales de checkpoints bajo
`/workspaces/AlgoLearning`, y se inspeccionaron los activos y manifiestos
publicados de las tres releases disponibles. No se encontró ningún hash de
los cinco E1 reentrenados ni de los 35 E1 nativos esperados. El directorio
original `/workspace/distributed-coloring-learning/data` sigue ausente. Las
fuentes, planes, historias agregadas y tablas sí están disponibles. Este
resultado delimita la búsqueda realizada; no demuestra que los pesos no
existan en una máquina o archivo no accesible.

Los experimentos nuevos que reutilizan mecanismos de Pedro son
**reproducciones metodológicas nuevas**, no reevaluaciones de sus checkpoints.
No se inventan pesos ni se atribuyen a Pedro las ejecuciones nuevas.

## Denominadores originales

El archivo `results/multiteacher-e1/2026-10-22/architecture-comparison.csv`
produce la siguiente comparación:

| Profesor | GNN directa | E1 nativo/adaptado | Denominador por columna |
|---|---:|---:|---:|
| Luby | 1857 | 1980 | 1980 |
| Johansson | 1919 | 1980 | 1980 |
| JP | 1635 | 1631 | 1980 |
| ColorTrials | 1980 | 1980 | 1980 |
| BE | 872 | 1980 | 1980 |
| GM | 1975 | 1951 | 1980 |
| Greedy | 1842 | 1980 | 1980 |

1.980 son cinco inicializaciones por 396 casos publicados, no 1.980 grafos
independientes. El seguimiento E1 reutiliza los tests ya conocidos y se
interpreta como exploratorio. Los cinco entrenamientos comparten el mismo
pool docente; replican inicialización y muestreo, no cinco datasets nuevos.
JP y GM son excepciones explícitas. Estos resultados no equivalen al gate
nuevo de cinco políticas y mil ejecuciones independientes por política.

## Comparación de métodos

| Factor | M6B histórico | E1 NORMALIZED histórico | GNN directa Pedro | E1 nativo Pedro | Complejidad muestral | Nuevo banco contrafactual |
|---|---|---|---|---|---|---|
| Núcleo | M4B, 8967 a K4 | E1, 16035 | LocalGNN, 36719 | E1 16035; 19396 con contexto | E1/DeepSets/Transformer con adaptadores | Los mismos tres núcleos e interfaces |
| Entradas | Estado y dos draws; K fijo | Estado/draws por color | 42 locales + 19 relacionales | Dos vecindarios completos; disponibilidad aprendida | Mismas interfaces nativas; orden relativo JP/GM | Idénticas a complejidad muestral en el control principal |
| IDs | No | No para Luby | Numéricos donde corresponde | Diferencias numéricas JP/GM | Signos coherentes de orden | Mismo adaptador relativo; nuevas asignaciones mediante replay |
| Mensajes | Dos etapas MLP/MAX | Dos etapas por color | MAX y SUM en dos etapas | ColorTrials decide candidato tras segundo intercambio | Lo mismo | Lo mismo, sin mensajes privilegiados nuevos |
| Supervisión | Propia + contrafactual | Propia + contrafactual en cohorte canónica | Acción directa pasiva | Propia + contrafactual de estrella | P/C/COMMIT almacenados pasivamente | Concordante y/o contrafactual finito; propia con consultas contadas como tratamiento separado |
| Datos | 800 N40 × 3 cintas | 800 N40 × 3 en referencia canónica | 800 N40 × 3, pool común | Pool nativo común | Tamaños 8/12/16/24; cinco datasets | Mismos tamaños; cobertura prospectiva de aislados/desconexiones |
| Batch | 128 estados completos | 128 estados completos | 1680 raíces | 1680 raíces | 64 raíces | 64 raíces en controles principales |
| Actualizaciones | Hasta 120 épocas; mejor cp época 53 | 17760 en cohorte 120 épocas | 17760 | 17760 | 500; control 1500 | Curva 500/1500/4000 antes de ampliar por development |
| COMMIT | Etiqueta de ofertas efectivas | Ídem | Implícito en acción | .5 propia + .5 cf; warmup 1480 | Etiqueta docente aplicada a salida con propuestas aprendidas | Control docente concordante; .5 docente + .5 cf; warmup 100 |
| Selección | Métrica local de validación | Según cohorte histórica | Última actualización fijada | Última actualización fijada | Primer B común perfecto en dev | Primer hito común perfecto en dev, por tratamiento |
| Evaluación | 500 N10240, una semilla | Baterías propias de la cohorte | 396 casos × 5 | Misma batería publicada | 1000 × 5; batería congelada | Nueva distribución/batería de 1000 × 5 |
| Auxiliares exactos | Ninguna decisión de coloración | Codificación/prefijos | Fases, aritmética y calendario declarados | Decodificación algebraica y calendario | Los mismos; sin shield | Los mismos; oráculos sólo en adquisición/entrenamiento |

Los números de actualizaciones no miden exposición comparable: un batch de
128 rondas N40 puede contener miles de decisiones pendientes, y 1680 raíces
es 26,25 veces 64. Los 17.760 updates de Pedro representan 29.836.800
presentaciones de decisiones por modelo. La exposición a componentes y las
consultas nuevas de COMMIT se deben contar por separado.

## Mecanismos plausibles y controles necesarios

1. **Concordancia de COMMIT.** Pedro vuelve a etiquetar propuestas propias y
   perturbadas; el estudio pasivo no lo hace. El control `matched` corrige la
   asociación entrada/etiqueta usando propuestas docentes en COMMIT, sin
   adquirir una etiqueta nueva. `counterfactual` añade cuatro consultas
   almacenadas por raíz de fase de color. `owncf` añade consultas propias
   distintas con un caché explícito. Las diferencias son medibles y no se
   describen como etiquetas gratuitas.
2. **Optimización/exposición.** La comparación 500/1500/4000 usa idéntica
   inicialización, banco y secuencia de batches dentro de cada tratamiento.
   El replay de BE a 1500 reproduce los pesos originales de E1; también se
   estudian los otros dos núcleos. Esto identifica sensibilidad a optimización
   en ese punto, sin convertirlo en un mínimo muestral.
3. **Interfaz.** El adaptador KW de Pedro comunica ocupación relativa al
   receptor en ambas etapas; ya estaba corregido en su cohorte final. Se
   conserva, sin reclamar su reparación como descubrimiento nuevo. El signo
   de prioridades de la integración cambia la representación respecto de
   diferencias numéricas de Pedro; no debe atribuirse su efecto a pérdida o
   arquitectura si se cambian a la vez.
4. **Cobertura.** El fallo GM aislado puede persistir con accuracy local
   redondeada a 1. La nueva distribución de training y dev incluye aislados y
   componentes desconectadas por diseño. El caso histórico 83105 se usa como
   regresión y conserva sus resultados originales; no sustituye al test final.
5. **Arquitectura.** La GNN directa y el E1 adaptado difieren también en
   decodificación Linial, supervisión, representación y loss. La tabla
   histórica no identifica causalmente un efecto puro de arquitectura.

Las conclusiones causales nuevas se limitan a los factores realmente
controlados. Las contribuciones cuantitativas de contrafactuales, cobertura,
optimización y representación se consignan en
`MULTITEACHER_COUNTERFACTUAL_RESULTS.md`, incluyendo resultados negativos.
