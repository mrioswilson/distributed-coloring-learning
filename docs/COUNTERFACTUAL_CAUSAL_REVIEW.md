# Revisión causal final de entrenamiento contrafactual

## Afirmación defendible

Los procedimientos emparejados producen respuestas COMMIT muy distintas ante
las continuaciones tipadas evaluadas, aunque tengan fidelidad nativa perfecta
y existan políticas funcionalmente certificadas. CF reproduce mejor estas
consultas concretas para BE y GM y mejora descriptivamente CT; no se demuestra
necesidad, superioridad funcional universal ni eficiencia a igual total de
etiquetas. Las consultas no son un test final intocado de familias de
intervención nuevas. Se retira esa interpretación más amplia.

## Diseño real y factores que cambian

| Factor | Passive | Matched | Counterfactual |
|---|---|---|---|
| Arquitectura/inicialización por semilla | igual | igual | igual |
| Banco nativo, orden de minibatches y updates al hito | igual | igual | igual |
| Contexto de pérdida COMMIT | mensajes propios, etiqueta nativa | mensajes docentes coherentes | mitad docente, mitad consulta añadida tras warmup |
| BE etiquetas adquiridas | 3.000 | 3.000 | 6.776 |
| CT etiquetas adquiridas | 30.000 | 30.000 | 70.000 |
| GM etiquetas adquiridas | 3.000 | 3.000 | 6.104–6.120 |
| Estados de training | prefijos docentes, cobertura declarada | mismos | mismos |
| Evaluación de fidelidad principal | development común | mismo | mismo |

Passive no es etiquetado coherente con propuestas propias: esa variante
dinámica `owncf` es un control separado con consultas nuevas cacheadas y
cobradas. Matched utiliza los mensajes del profesor, no las propuestas propias
con etiquetas nuevas. De lo contrario se compararía otra intervención.

`objective` y `train` en `src/dcl/multicf/training.py` muestran el cambio
de pérdida, warmup y variantes. Los pesos retienen Adam, RNG y visitas.
La reconstrucción BE comprueba mismo hash inicial y de banco por brazo,
visitas y RNG final de minibatches iguales, y reproduce todos los numeradores.
Esto apoya emparejamiento; no crea independencia entre arquitecturas ni
garantiza aleatorización causal sobre una población externa de entrenamientos.

## Denominadores BE y variabilidad

Para E1/4.000 updates, cada semilla tiene 28.800 acciones nativas (todas
correctas en los tres brazos), 24.960 raíces de color y cuatro consultas por
raíz: 99.840 consultas. Por brazo hay 499.200 ocurrencias, **cinco**
entrenamientos emparejados, no 499.200 réplicas independientes.

| Método | Media descriptiva | Rango entre cinco semillas |
|---|---:|---:|
| Passive | 22,640425% | 9,160657–42,329728% |
| Matched | 9,268630% | 9,090545–9,397035% |
| CF | 99,993189% | 99,977965–100% |

Las medias son sobre semillas; no se da un p-valor binomial sobre decisiones
correlacionadas. Cinco diferencias con el mismo signo dan p=0,0625 en un
test de signos bilateral exacto antes de multiplicidad. Ello no invalida
el gran contraste descriptivo, pero limita la inferencia poblacional.

Los BE/GM CF comparten exactamente las cabezas COMMIT de las tres
arquitecturas por semilla/hito. Las pérdidas con propuestas fijas no acoplan
esos pesos al scorer de paleta. Quince etiquetas de arquitectura–semilla
no son quince entrenamientos COMMIT independientes. CT sí acopla el scorer
a la segunda comunicación. Los hashes de todos los tensores constan en
`audit/shared-confirmation-heads.json` y su verificación de cierre.

## Separación de bancos y novedad

Training usa n8/12/16/24; diagnóstico usa 60 trayectorias n28/36 con
namespaces distintos. No hay acceso del optimizador al banco de evaluación.
Sin embargo, es development usado repetidamente y la novedad de un grafo
completo no implica novedad del estado local recibido. Las cuatro opciones
de color y el generador de intervenciones se reutilizan.

Se reconstruyeron exactamente las consultas existentes con RNG 17001.
Sólo 91/499.200 ocurrencias coinciden con todo el input serializado de
training; entre 99,277% y 99,540% coinciden en un **tipo suficiente para
COMMIT**. La clave se deriva de los inputs reales del segundo estadio BE,
no de la etiqueta: fase, permiso, signo del offset KW, participación propia,
conjunto MAX de igualdad/opción ocupada, participación y oferta vecinal.
Descarta IDs y campos irrelevantes para esa cabeza, no para propuestas.

Los tamaños de subconjunto con tipos no vistos son 549,459,717,522,722;
los aciertos CF 549,437,717,510,722 frente a matched
158,342,229,182,246. Es análisis posterior de ocurrencias correlacionadas,
no un holdout prospectivo de mecanismos. La revisión adversarial exige además
revisar prevalencia y baselines en ese subconjunto antes de interpretarlo.
El informe de estratos y confusiones de cierre es la referencia para ello.
Los positivos de esos subconjuntos son 52,108,90,56,60; los negativos
497,351,627,466,662. Todos los brazos aceptan esos positivos sin error;
la ventaja reside en especificidad. CF supera a rechazar siempre en las
cinco semillas. La BA media por semilla es 99,115709% CF frente 66,529900%
matched y 50% rechazo constante. No demuestra mejora de respuestas positivas.

## Baselines y mecanismo

La revisión independiente detectó fuerte desequilibrio: sólo
3.070,2.895,2.891,2.773,3.115 positivos por 99.840. Rechazar siempre
obtiene aproximadamente 97% de accuracy total y cero progreso; aceptar sólo
raíces elegibles y activas es otro baseline semántico informativo. La brecha
bruta no basta para caracterizar aprendizaje de la regla.

`results/final-closure/review-ml-be-strata.json` conserva reconstrucción de
las mismas consultas, TP/TN/FP/FN, accuracy balanceada y estratos
no elegible/elegible silencioso/elegible activo. Todos los 34 errores CF
son falsos positivos en elegibles silenciados. En elegibles activos CF acierta
todas las consultas, frente a passive/matched que aceptan todas las opciones,
incluidas ocupadas. Eso demuestra aprendizaje de rechazo de opciones
prohibidas en esos estados, además de elegibilidad redundante. No demuestra
que CF mejore la probabilidad de terminar bajo la política autónoma.

## Evidencia causal más sólida

La accuracy/BA media por semilla en BE completo es respectivamente
22,640425/60,141255% passive, 9,268630/53,253641% matched,
99,993189/99,996494% CF; siempre-rechazar da 97,046474/50% y sólo
elegibilidad/participación 98,598958/99,278139%. Ambos son controles
semánticos no entrenados, no algoritmos funcionales competitivos.

Los controles E1 de optimización a banco fijo 500→1.500 updates, semillas
83101–83105, tienen nuevas baterías independientes y no cambian etiquetas ni
linaje inicial. Luby y JP pasan 1.000/1.000 por semilla en ambos puntos.
BE pasa de 72,90,84,92,42 éxitos/1.000 a 1.000 en las cinco semillas.
Éste aísla mejor el efecto de **optimización adicional en ese banco**, no
del componente contrafactual. Véase
`results/sample-complexity/optimization-confirmation/results.json`.

Matched frente a CF controla muchas condiciones, pero añade etiquetas y
cambia distribución/pérdida. No identifica el efecto exclusivo de la naturaleza
contrafactual. El control JP de cuatro representaciones a igual dimensión,
etiquetas, inicialización y updates apoya sensibilidad representacional en
development; tampoco es una certificación final ni una imposibilidad numérica.

## Decisión sobre un último experimento

No se entrena otro modelo. Las conclusiones del manuscrito se limitan al
contraste de procedimientos ya existente y a mecanismos observados. Aislar
una mejora CF a recursos iguales requeriría otro diseño (misma arquitectura,
total de etiquetas, updates y criterio, semillas emparejadas, bancos finales
nuevos y clases de intervención retenidas). Ese diseño se especifica como
trabajo futuro **sin ejecutar ni preregistrar retrospectivamente resultados**.
Si se quisiera recuperar la afirmación causal fuerte, esa evidencia sería
necesaria; no se la obtiene reinterpretando los datos presentes.
