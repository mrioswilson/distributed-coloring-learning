# Revisor 1 — Algoritmos distribuidos y demostraciones

Fecha: 26 de octubre de 2026. Revisión interna independiente del borrador de
cierre, desde una perspectiva de algoritmos distribuidos; no es una revisión
externa por una persona ni una réplica de laboratorio independiente.

Se revisaron `paper/ucgnn/main.tex`, los profesores nativos y extendidos,
`degree/adapter.py`, `multiteacher_e1/model.py`, `samplecomplexity/model.py`,
`paperstudy/certification.py`, `scripts/final-counterfactual-audit.py`,
`docs/UCGNN_MATHEMATICAL_ANALYSIS.md` y
`docs/integration/TEACHER_CORRECTNESS_AUDIT.md`. Las líneas de abajo se
refieren al borrador inicial de cierre; los anclajes por nombre siguen siendo
válidos cuando se incorporen correcciones. No se modificó el manuscrito ni
el código científico y no se entrenó ningún modelo.

## Dictamen

Las demostraciones centrales revisadas son correctas bajo las hipótesis
declaradas. Hay dos correcciones de especificación/caso frontera que deben
incorporarse: distinguir el bound de preferencias nativo de ColorTrials del
grado real, y exigir un clique de al menos dos vértices para el ejemplo de
estancamiento siempre activo. Son reparaciones locales; no requieren nuevos
experimentos. Una vez corregidas, no he encontrado un bloqueo matemático
sustantivo para una revisión externa de las afirmaciones acotadas.

El resultado fuerte «las redes aprendidas heredan la corrección y terminación
de los profesores a cualquier grado» seguiría siendo injustificado. El nuevo
borrador lo retira expresamente. Deben mantenerse las restricciones de
interfaz, aritmética y estados alcanzables del certificado MAX, y la exclusión
de BE de alto grado.

## Objeciones y resolución exigida

| ID | Clasificación | Localización y objeción | Corrección / estado |
|---|---|---|---|
| D1 | Error demostrado de especificación | `main.tex`, apéndice Exact Rules, párrafo ColorTrials (líneas 601–613): Δ está definido como grado máximo real, pero la explicación usa alfabeto `{0,…,Δ}` sin separar el nativo. `multiteacher/programs.py:153` usa siempre `{0,…,3}`, también en caminos y aislados; `degree/teachers.py:92–95` distingue `native4` de transferencia. | Introducir `D≥Δ` para el bound de preferencias: `D=3` nativo, `D=Δ` en transferencia. El evento extremo da `(D+1)^−1(D/(D+1))^Δ` para Δ competidores; puede usarse el exponente conservador D. Precisar el caso D=Δ=0 por separado. No cambian los resultados funcionales ni la seguridad. Una frase que diga que la fórmula con Δ sólo corresponde a transferencia también resuelve la discrepancia. |
| D2 | Error demostrado de caso frontera | Contraejemplo Safety without useful progress (`main.tex:702–708`): «Always-active vertices never progress» es falso en `K=1`, pues el único vértice no tiene competidor. | Añadir `K≥2` al contraejemplo o limitar esa oración a cliques no triviales. La fórmula con participación 1/2 sí es correcta también para K=1. |
| D3 | Problema editorial | Modelo y unidades (`main.tex:107–116`, apéndice BE/GM): el coste se separa de las rondas LOCAL, pero conviene explicitar que un paso nativo no equivale siempre a dos intercambios aprendidos ni a la complejidad completa del sistema. BE puede realizar una reducción matemática con un intercambio; el modelo común utiliza dos etapas. GM añade barreras, bloques y cola globales. | Añadir una definición corta de paso del intérprete frente a intercambio de mensajes. Conservar que los horizontes/costes de los profesores no son comparables como rondas anónimas gratuitas. No hace falta ejecutar otra medición. |
| D4 | Limitación reconocida | Certificado MAX (`main.tex:711–738` y `certification.py`): se certifica un canal canónico de confirmación con seis tipos, no disponibilidad/propuestas/progreso, ni contextos BE/GM/CT. El paso de coeficientes reales a kernels float32 depende del modelo de error. | Ya está resuelto en el texto: hipótesis de tipos, mensajes no negativos, gather exacto, aritmética IEEE y ausencia de TF32/overflow; se distinguen cinco pesos históricos certificados de los 45 núcleos seleccionados que no pasan la condición suficiente. No ampliar esa conclusión. |
| D5 | Limitación reconocida | Interfaz de alto grado (`main.tex:184–193`): la compresión de categorías es algebraicamente válida para MAX no negativo y las categorías/contextos que conserva. La paridad finita de logits/acciones no prueba identidad bit a bit para cualquier hardware o tamaño de batch. Además la adaptación de paletas locales cambia la entrada fuera del dominio histórico. | El borrador limita correctamente la paridad a controles finitos y declara la nueva representación. Mantener la distinción entre igualdad de función en aritmética real, paridad observada y semántica del profesor. |
| D6 | Limitación reconocida | Extensiones y causalidad de Δ (`main.tex:401–511`): el cambio de grado también cambia K y el alfabeto/escala CT. No identifica una causa puramente atribuible al número de vecinos. | Resuelto por la delimitación explícita y por no llamar a la sonda una tasa funcional aleatoria. La sonda de receptor con grado/paleta fija permite una conclusión local sobre escala, no una teoría universal de capacidad. |
| D7 | Limitación reconocida | Simetría y TV (`main.tex:885–906`): uniformidad en cliques exige exactamente n colores, equivarianza de vértices y éxito condicional de probabilidad positiva. Acoplamiento y no rechazo son argumentos diferentes. | Las hipótesis y advertencias están presentes. No extender a K>n, a IDs/calendarios no transformados o a igualdad de leyes por ausencia de rechazo. La cota por transición exige errores uniformes sobre estados comunes previos a la divergencia. |
| D8 | Nueva investigación fuera del alcance | MEAN dilution (`main.tex:740–755`) usa familias de grado no acotado en aritmética real; la existencia asintótica de un contraejemplo no garantiza que aparezca antes del límite práctico n≤65.536 ni en las trayectorias del modelo. | La restricción de alcance ya excluye inferir un fallo alcanzable. Puede añadirse «sobre la familia matemática no acotada» para evitar lectura como un testigo dentro de todos los límites del benchmark. Buscar un fallo concreto menor no es requisito de cierre. |

## Demostraciones comprobadas

1. **Disponibilidad.** El conteo `|A_v|≥d_U(v)+1` es válido para listas
   globales y locales de tamaño al menos `d(v)+1`. Cuenta colores prohibidos
   por vecinos ya fijados; repetir colores sólo mejora el margen. No supone
   listas idénticas.
2. **Seguridad Luby/Johansson.** La disponibilidad evita conflictos antiguos;
   el veto de ofertas iguales evita escrituras simultáneas incompatibles.
   Por inducción, la coloración parcial permanece propia. La prueba no se
   aplica a un COMMIT neuronal con errores.
3. **Progreso 1/4.** Condicionar antes de los nuevos draws es indispensable.
   El producto de probabilidades en la ecuación de progreso usa independencia
   entre nodos y entre participación/propuesta dentro de cada nodo. El chord
   de `log(1−x)` y Jensen están aplicados con el sentido correcto. Para
   Johansson siempre activo, todo vecino pendiente tiene al menos dos colores
   disponibles; por ello su masa por color es ≤1/2 y la cota 1/4 es válida
   incluso con listas locales diferentes.
4. **ColorTrials.** La victoria estricta de dos extremos para el mismo color
   es imposible. El evento de preferencia máxima y competidores por debajo
   prueba progreso positivo; debe parametrizarse con el bound D correcto
   según D1. No se requiere que el color fijado sea el color usado en ese
   evento: un ganador menor también produce progreso.
5. **Terminación ideal.** La cota condicional uniforme y la iteración de la
   esperanza dan una cola geométrica por vértice; la unión no requiere
   independencia entre sucesos de confirmación. No demuestra un resultado
   para todos los ciclos de un PRNG finito ni para el checkpoint.
6. **JP.** Los máximos locales de un orden único coherente son independientes
   y contienen el máximo global pendiente. First-fit existe. IDs sin orden
   global coherente no satisfacen la hipótesis; signos aleatorios por arista
   no serían una representación alternativa del mismo algoritmo.
7. **GM.** En el historial exacto sólo pueden persistir conflictos de slot
   igual. Marcar el extremo menor de cada uno cubre todas las aristas malas.
   El último extremo reparado evita al otro, y el subgrafo sin reparaciones
   ya era propio. El argumento vale con listas propias `d(v)+1`. Omisiones
   de escritura o marcas aprendidas incorrectas rompen sus premisas; no debe
   consultarse la referencia como si esas historias fueran prefijos exactos.
8. **BE.** Los grados polinómicos son 4 y 2 porque el espacio inicial tiene
   65.536 IDs y el primer bound es 169. `13>3·4` y `7>3·2` dejan puntos.
   El decoder usa una codificación interna `MAX_FIELD=13`, pero el intérprete
   recodifica con el campo actual; los pares siguen siendo inyectivos.
   Para KW, `q=ceil(4B/5)` y `4(B−q)≤q` dejan bloques disjuntos de cuatro
   opciones y un descenso hasta cuatro. Extender grado sin nuevos campos y
   bloques no está justificado; la exclusión de BE es correcta.
9. **Proposición de propuestas difusas y submasa.** Los signos de la
   exponencial y la normalización por η son correctos. El bound debe valer
   en todos los estados alcanzables para obtener una conclusión uniforme de
   terminación. Las histogramas finitas no prueban esa premisa.
10. **MAX/MEAN/SUM.** MAX elimina multiplicidad bajo las condiciones
    expuestas; la continuidad demuestra la obstrucción MEAN con aceptación
    estricta; la coordenada SUM `o+ReLU(p+a−1)` representa exactamente la
    presencia de los dos tipos conflictivos. Por tanto sería erróneo llamar
    a un fallo empírico SUM una imposibilidad de representación.

## Comprobación independiente del key BE

Se siguió el camino real `QueryPolicy.first → RootedE1.second → read_commit`.
En KW, el logit de COMMIT no lee el estado h1 ni alpha. Lee los bits de la
ranura consultada, el contexto de fase/permisión/offset signado y el MAX de
mensajes de vecinos, con residual de IDs/slot nulo para BE. El conjunto de
triples conservado por `be_commit_key` es por ello suficiente en aritmética
real, sin invocar el label del oráculo. Esta reducción no determina las
propuestas y no prueba novedad de familias de intervención.

Como comprobación pequeña de los datos ya existentes se cargó el checkpoint
CF BE/E1, semilla 94101, 4.000 updates y banco nativo 3.000; se evaluaron
las primeras 512 raíces KW del banco diagnóstico con cuatro consultas
originales por raíz. Las 2.048 consultas se agruparon en 32 keys, 22 con
más de una ocurrencia. La variación máxima de logits dentro de un key fue
**0.0**. No se adquirieron etiquetas nuevas ni se produjo un checkpoint.
Esta comprobación finita apoya la derivación estructural; no es una prueba
de equivalencia para kernels arbitrarios.

## Recomendación

Aceptar las afirmaciones matemáticas condicionales tras D1 y D2. D3 es una
mejora editorial recomendable. D4–D7 están adecuadamente delimitadas; D8 no
debe convertirse en una campaña adicional. No he encontrado motivo para
reabrir entrenamiento, ampliar BE o perseguir éxito perfecto. La condición
para revisión externa es mantener las afirmaciones presentes, no recuperar
interpretaciones fuertes retiradas.

## Addendum — comprobación de las correcciones

Se releyó el manuscrito corregido el 26 de octubre de 2026, sin ejecutar
experimentos adicionales. Versión comprobada de `paper/ucgnn/main.tex`:

```text
SHA-256 48db1d2dc46733e018489a8d18a4e00abba652c2fa113ec0b3d7671e6b35816c
```

| Objeción | Estado tras la relectura |
|---|---|
| D1 | **Resuelta.** El párrafo ColorTrials introduce `D≥Δ`, distingue `D=3` nativo de `D=Δ` en transferencia y usa la cota correcta `(D+1)^−1(D/(D+1))^Δ` para `D>0`. El caso de aislados se trata por separado. La especificación coincide ahora con ambos caminos del código. |
| D2 | **Resuelta.** El contraejemplo exige `K≥2`; la afirmación de estancamiento siempre activo ya no incluye el singleton. |
| D3 | **Mejora editorial opcional pendiente.** Se mantienen la distinción de regímenes/costes y la advertencia de que no son algoritmos LOCAL anónimos de igual coste. Sigue siendo útil explicitar una vez que paso del intérprete e intercambio aprendido no son unidades idénticas, especialmente para BE. No es un bloqueo de validez del resultado. |
| D4 | **Delimitada correctamente.** El certificado sigue restringido a tipos/interfaz/arimética, sin convertirlo en terminación ni extenderlo a residuals nativos. |
| D5 | **Delimitada y mejorada.** El texto declara ahora que la participación de profesores con contexto promedia sobre slots válidos en el adaptador estructural. Por tanto no presenta la máscara de pertenencia como única adaptación ni la paridad finita como identidad numérica universal. |
| D6–D7 | **Delimitadas correctamente.** Se conservan el confounding grado/paleta/preferencias y las hipótesis de simetría/acoplamiento. No se reintroducen garantías universales. |
| D8 | **Fuera del alcance; no bloqueante.** La proposición MEAN sigue siendo un resultado sobre la interfaz matemática y no una garantía de hallar un testigo alcanzable dentro del límite práctico de tamaños. No se requiere otro experimento. |

**Dictamen actualizado:** no subsiste un bloqueo matemático sustantivo
identificado en las afirmaciones revisadas. D1 y D2 quedaron reparadas.
La recomendación es permitir revisión externa del manuscrito acotado,
manteniendo sus hipótesis y límites actuales. La posible aclaración D3 no
requiere reapertura científica.
