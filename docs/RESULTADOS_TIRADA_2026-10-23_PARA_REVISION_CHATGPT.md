# Resultados de la tirada de complejidad muestral de algoritmos distribuidos

**Documento de revisión científica — resultados del 23–24 de octubre de 2026.**

Este documento reúne el contexto, los principales resultados y sus límites para poder revisarlos sin depender del historial de la conversación. Describe experimentos ejecutados y distingue sus resultados de las hipótesis y los experimentos futuros propuestos. No se han realizado nuevos entrenamientos para redactarlo.

**Instantánea de evidencia:** commit `08aa8c12597e360276be8e03d28943b7eabb87cc`, rama `research/sample-complexity-20261023`. Protocolo principal: `sample-complexity-native-v1`. Los resultados se refieren a las variantes implementadas de los profesores, no a todas las variantes bibliográficas con los mismos nombres.

## 1. Los resultados más importantes

La pregunta experimental fue: **¿cuántas etiquetas de un profesor necesita cada arquitectura para resolver sin un solo fallo un benchmark distribuido definido, en cinco entrenamientos independientes?**

El estudio principal completó **735 entrenamientos nuevos**, correspondientes a siete profesores, tres arquitecturas, cinco semillas y siete presupuestos de supervisión. Cada entrenamiento recibió exactamente **500 actualizaciones de Adam**.

El resultado principal fue **12 combinaciones profesor–arquitectura PASS y 9 NOT CERTIFIED**, sobre 21 combinaciones posibles. Un PASS exige que **cada una de las cinco políticas** complete sus **1.000 ejecuciones finales** sin ningún fallo. Un resultado cercano al 100% no basta.

Los hallazgos centrales son:

1. **Luby:** E1 NORMALIZED y LocalPaletteTransformer pasan con un presupuesto de rejilla de 1.000 consultas, que corresponde a **999 etiquetas adquiridas por semilla**. DeepSets pasa con **3.000 etiquetas** por semilla.
2. **Johansson, JP y Greedy:** las tres arquitecturas pasan en el primer presupuesto probado, B = 1.000. Esto no demuestra que 1.000 sea su mínimo real: no se probaron presupuestos inferiores.
3. **ColorTrials, BE y GM:** ninguna de las tres arquitecturas alcanza la certificación principal de cinco semillas para estos profesores. Las razones y la evidencia disponible son diferentes en cada caso.
4. **BE cambia radicalmente al aumentar la optimización:** con E1, las mismas 3.000 etiquetas y una evaluación adicional independiente, pasar de 500 a 1.500 actualizaciones cambia el resultado de **0/5 a 5/5 semillas plenamente exitosas**. Este punto no sustituye el resultado principal ni estima un nuevo mínimo de etiquetas.
5. **GM presenta un fallo localizado:** las tres arquitecturas fallan en la semilla 83105, dejando vértices aislados sin colorear. Development de esa semilla no contenía vértices aislados.
6. **Resolver el problema no equivale a copiar todas las decisiones del profesor.** Hay políticas certificadas cuya fidelidad de acciones y transiciones es inferior al 100% en estados comunes de development.
7. **El suplemento de distribuciones no detectó diferencias:** 0 rechazos tras Holm entre 210 hipótesis contrastables. Este resultado no demuestra igualdad de distribuciones.

## 2. Qué se comparó y qué significa una observación

Las arquitecturas principales fueron **E1 NORMALIZED**, **DeepSets** y **LocalPaletteTransformer**. Comparten la información correspondiente a cada profesor, el intérprete nativo, las pérdidas y el presupuesto de optimización. Cambian el procesamiento de la paleta. La atención del Transformer actúa sobre los colores dentro de un nodo; la comunicación entre nodos sigue utilizando mensajes aprendidos. DeepSets tampoco elimina esa comunicación.

Una consulta supervisada obtiene **una etiqueta para un nodo, estado y componente de acción concretos**. En una fase normal de coloración se adquieren tres componentes: participación, propuesta y confirmación. Por tanto, un ejemplo de ese tipo cuesta tres consultas. Elegir un punto en una fase Linial de BE o una marca de conflicto en GM cuesta una consulta para ese componente.

Esto explica por qué un presupuesto máximo B = 1.000 puede contener 999 etiquetas: los paquetes de componentes se conservan completos. Las cuentas exactas se registran por componente y por semilla. La unidad no es una trayectoria, una ronda ni una actualización de gradiente, y las etiquetas de componentes distintos no tienen necesariamente el mismo coste o dificultad.

Se adquirió una biblioteca máxima por profesor y semilla y se compartió entre arquitecturas. Los presupuestos menores acceden a prefijos anidados de esa biblioteca. La adquisición principal total fue de **34.999.967 etiquetas**, contando cada biblioteca una sola vez. No se multiplica artificialmente esa adquisición por el número de arquitecturas o de veces que se reutilizan los datos.

Cada presupuesto se entrena desde una construcción nueva de la inicialización de su semilla, sin continuar los pesos del presupuesto anterior. Dentro de una semilla se conserva la misma inicialización de partida para comparar presupuestos de forma emparejada. Los siete presupuestos de una semilla no son siete réplicas independientes del entrenamiento.

## 3. Diseño del benchmark y regla de certificación

| Elemento | Diseño ejecutado |
|---|---|
| Profesores | LUBY, JOHANSSON, COLOR_TRIALS, JP, BE_BASE_D3, SPECULATIVE_GM y GREEDY |
| Semillas principales | 83101, 83102, 83103, 83104 y 83105 |
| Presupuestos máximos B | 1.000; 3.000; 10.000; 30.000; 100.000; 300.000; 1.000.000 |
| Optimizador principal | Adam, 500 actualizaciones, learning rate 0,001, batch de 64 ejemplos raíz |
| Dominio | Grafos simples no dirigidos, grado máximo 3, paleta global de cuatro colores |
| Training | Tamaños n = 8, 12, 16 y 24 |
| Development | 28 ejecuciones por semilla, n = 28 y 36 |
| Final | 1.000 ejecuciones por política, n = 32, 48 y 64 |
| Familias finales | Caminos, ciclos, cúbicos, irregulares, bipartitos, escaleras circulares y uniones desconectadas de K4 |
| Asignación final por familia | 143 ejecuciones para cada una de las primeras seis familias y 142 para K4 |

Los tamaños de vértices son disjuntos entre training, development y final. Esto excluye la repetición del mismo grafo entre particiones, incluso considerando isomorfismos. Cada caso usa semillas de generación, cintas y asignaciones de IDs nuevas según la partición; no se exige que los valores numéricos de ID pertenezcan a conjuntos disjuntos. Algunas familias deterministas pueden repetirse como clases de isomorfismo dentro de una partición; no se afirma haber probado 1.000 topologías distintas por política.

Una política pasa únicamente si en **todos** sus casos finales consigue una coloración propia, respeta la paleta y los invariantes, termina dentro del horizonte fijado y no produce errores técnicos, cutoffs ni estados terminales incorrectos. En GM se permiten los conflictos especulativos temporales propios del protocolo; eso no permite una coloración final incorrecta. En BE una coloración auxiliar impropia ya es un fallo de invariantes.

El procedimiento de selección fue:

1. Entrenar y medir development en toda la rejilla fijada.
2. Nominar el primer presupuesto en el que las cinco semillas consiguen 28/28 en development.
3. Congelar los cinco checkpoints y evaluarlos una sola vez en final.
4. Declarar PASS solamente si las cinco políticas completan 1.000/1.000.

Un fallo final no autoriza a escoger después otro presupuesto con esa misma batería. Los resultados negativos se conservan. **B\*** significa el menor presupuesto validado bajo este procedimiento y esta rejilla, no el mínimo matemático de supervisión.

De las 21 combinaciones, **16 obtuvieron candidato final**: 12 pasaron y cuatro fallaron. Las otras cinco no obtuvieron candidato común en development y no intentaron la confirmación estudiantil. De ahí salen las **80.000 ejecuciones finales principales**: 16 combinaciones × cinco políticas × 1.000 casos.

Los siete profesores exactos completaron sus baterías antes de confirmar alumnos: **35.980 ejecuciones exactas principales**, sumando development y final. Los horizontes se fijaron por profesor: JP hasta n pasos, Greedy n escrituras, BE 15 pasos y GM una cota de planificación y reparación dependiente de n; los profesores aleatorizados usan plazos derivados de cotas conservadoras de cola. Se abandonó el antiguo cutoff común de 128. Los pasos nativos de profesores distintos no representan automáticamente la misma unidad de comunicación.

**Cronología:** hubo piloto y congelación prospectiva local de configuración, fuentes y baterías. No fue una preregistración externa con sello temporal independiente; el commit público de los artefactos congelados fue posterior al inicio del entrenamiento principal. Esta limitación de procedencia debe acompañar cualquier evaluación metodológica.

## 4. Resultados principales por profesor y arquitectura

### 4.1. Régimen A: interfaz anónima

| Profesor | E1 NORMALIZED | DeepSets | LocalPaletteTransformer |
|---|---|---|---|
| Luby | PASS; B\* = 1.000; 5/5 | PASS; B\* = 3.000; 5/5 | PASS; B\* = 1.000; 5/5 |
| Johansson | PASS; B\* = 1.000; 5/5 | PASS; B\* = 1.000; 5/5 | PASS; B\* = 1.000; 5/5 |
| ColorTrials | NOT CERTIFIED; candidato B = 300.000; 3/5 | NOT CERTIFIED; sin candidato hasta B = 1.000.000 | NOT CERTIFIED; sin candidato hasta B = 1.000.000 |

### 4.2. Régimen B: interfaz extendida

| Profesor | E1 NORMALIZED | DeepSets | LocalPaletteTransformer |
|---|---|---|---|
| JP | PASS; B\* = 1.000; 5/5 | PASS; B\* = 1.000; 5/5 | PASS; B\* = 1.000; 5/5 |
| BE_BASE_D3 | NOT CERTIFIED; sin candidato hasta B = 1.000.000 | NOT CERTIFIED; sin candidato hasta B = 1.000.000 | NOT CERTIFIED; sin candidato hasta B = 1.000.000 |
| SPECULATIVE_GM | NOT CERTIFIED; candidato B = 30.000; 4/5 | NOT CERTIFIED; candidato B = 30.000; 4/5 | NOT CERTIFIED; candidato B = 30.000; 4/5 |
| Greedy | PASS; B\* = 1.000; 5/5 | PASS; B\* = 1.000; 5/5 | PASS; B\* = 1.000; 5/5 |

**Cómo leer la tabla:** 0/5 confirmadas por falta de candidato no significa cinco fallos observados en final. Tampoco debe interpretarse NOT CERTIFIED como imposibilidad arquitectónica o como demostración de que cualquier política con menos de un millón de etiquetas fallará. No se confirmaron en final todos los presupuestos.

### 4.3. La comparación más clara de eficiencia muestral: Luby

| Arquitectura | B\* en la rejilla | Etiquetas adquiridas por semilla | Development a B = 1.000 | Confirmación en su candidato |
|---|---:|---:|---:|---:|
| E1 NORMALIZED | 1.000 | 999 | 5/5 semillas perfectas | 5/5; cada política 1.000/1.000 |
| DeepSets | 3.000 | 3.000 | 3/5 semillas perfectas | 5/5; cada política 1.000/1.000 |
| LocalPaletteTransformer | 1.000 | 999 | 5/5 semillas perfectas | 5/5; cada política 1.000/1.000 |

Bajo el procedimiento utilizado, E1 y Transformer se certifican con aproximadamente un tercio de las etiquetas de DeepSets. La afirmación se refiere a estos puntos discretos, cinco semillas y 500 actualizaciones. DeepSets a B = 1.000 no fue confirmado en final porque no superó el filtro de development; no se ha demostrado que ninguna política DeepSets pueda pasar con menos de 3.000 etiquetas.

En Johansson, JP y Greedy las tres arquitecturas empatan en el primer presupuesto probado. La rejilla no permite ordenar sus necesidades por debajo de 1.000. La tirada, por tanto, **no establece una arquitectura ganadora para todos los profesores**.

## 5. Qué falló y qué explicación permite la evidencia

### 5.1. ColorTrials: candidato de E1 rechazado en confirmación

E1 llegó a 5/5 semillas perfectas en development con B = 300.000. Los cinco checkpoints se congelaron y la confirmación produjo:

| Semilla | Ejecuciones correctas | Fallos de invariantes | PASS de la política |
|---|---:|---:|---|
| 83101 | 993/1.000 | 7 | No |
| 83102 | 982/1.000 | 18 | No |
| 83103 | 1.000/1.000 | 0 | Sí |
| 83104 | 1.000/1.000 | 0 | Sí |
| 83105 | 1.000/1.000 | 0 | Sí |

El resultado del paquete es **NOT CERTIFIED, 3/5**, con 25 fallos de invariantes. Un promedio elevado no cambia esa clasificación. DeepSets y Transformer no obtuvieron un presupuesto con las cinco semillas perfectas en development; sus políticas no pasaron a final en el protocolo principal.

Esta evidencia identifica falta de generalización perfecta del candidato de E1 y variabilidad entre entrenamientos. No aísla una causa única entre cobertura, optimización y capacidad. Tampoco autoriza a llamar intrínsecamente imposible a ColorTrials para estas arquitecturas. La interfaz auditada sí incluye los dos intercambios necesarios para transmitir disponibilidad y preferencias.

### 5.2. GM: el mismo caso raro afecta a las tres arquitecturas

Las tres arquitecturas nominaron B = 30.000. En cada una, las semillas 83101–83104 completaron 1.000/1.000. La semilla **83105** completó **921/1.000**, con **79 ejecuciones incompletas**. Cada paquete quedó en **4/5**, por lo que ninguno pasa.

El diagnóstico posterior examinó los pesos congelados y las trazas ya registradas. Todos esos fallos pertenecen a grafos irregulares y dejan pendientes solamente vértices de grado cero. En un grafo diagnóstico de un único vértice, la misma política de esa semilla predice participación pero rechaza la confirmación de la asignación. El profesor exacto sí termina.

Hay tres datos que delimitan la explicación:

- El prefijo de entrenamiento de la semilla afectada contenía **cinco ejemplos raíz de asignación de vértices aislados**, con tres etiquetas de componentes cada uno.
- La reconstrucción del orden de minibatches comprobó **nueve presentaciones** de esos ejemplos durante el entrenamiento.
- La batería de development de esa semilla contenía **cero vértices aislados**.

Por tanto, no fue ausencia absoluta de ese tipo de ejemplo: hubo supervisión, pero el ajuste de ese caso raro falló y development no lo expuso. Es evidencia de un defecto localizado de ajuste y cobertura de validación. No demuestra una limitación universal de representación, ni que la ausencia de IDs sea la causa.

El diagnóstico fue **post hoc**. No se modificaron pesos, etiquetas, presupuestos o reglas de decisión ni se repitió la confirmación. El ejemplo de un solo vértice sirve para explicar el fallo, no para certificar de nuevo la política. Que las tres arquitecturas fallen en la misma semilla tampoco constituye tres experimentos causales independientes: comparten elementos del núcleo, los datos y el procedimiento.

### 5.3. BE: dificultad principal en las reducciones KW

Con 500 actualizaciones, ninguna arquitectura logró un candidato común de cinco semillas perfectas en la rejilla hasta B = 1.000.000. No hubo confirmación principal estudiantil para BE.

En las ejecuciones de development registradas, los **1.563 primeros fallos** de BE se localizaron en la etapa **KW**, y no en las fases Linial. Esto localiza dónde aparece el primer incumplimiento observado; no demuestra que toda la computación previa sea exactamente imitada ni identifica por sí solo una causa causal de esos incumplimientos.

La observación que cambia su interpretación procede del control de optimización siguiente.

## 6. BE con las mismas etiquetas y más optimización

Se compararon dos brazos de E1 NORMALIZED con **B = 3.000**, las mismas etiquetas y la misma inicialización por semilla: 500 frente a 1.500 actualizaciones. Después se evaluaron los checkpoints congelados sobre **baterías finales nuevas, independientes del estudio principal y emparejadas entre los dos brazos**.

Se incluyeron todos los profesores de control previstos —Luby, JP y BE— y las cinco semillas. No se eligió únicamente BE después de ver qué brazo funcionaba mejor.

| Profesor | E1, 500 actualizaciones | E1, 1.500 actualizaciones |
|---|---|---|
| Luby | PASS, 5/5 | PASS, 5/5 |
| JP | PASS, 5/5 | PASS, 5/5 |
| BE | NOT CERTIFIED, 0/5 | PASS, 5/5 |

En BE, la diferencia por semilla fue:

| Semilla | Correctas con 500 actualizaciones | Correctas con 1.500 actualizaciones |
|---|---:|---:|
| 83101 | 72/1.000 | 1.000/1.000 |
| 83102 | 90/1.000 | 1.000/1.000 |
| 83103 | 84/1.000 | 1.000/1.000 |
| 83104 | 92/1.000 | 1.000/1.000 |
| 83105 | 42/1.000 | 1.000/1.000 |

Los fallos del brazo de 500 actualizaciones fueron de invariantes. El brazo de 1.500 logró **5.000/5.000** entre sus cinco políticas. El suplemento completo conservó **30.000 ejecuciones estudiantiles** y **15.000 exactas**, separadas de las 80.000 finales principales.

La conclusión apoyada es concreta: **con esos datos e inicializaciones, el límite de optimización de 500 actualizaciones era determinante para el resultado de BE**. Aumentar etiquetas manteniendo ese límite no equivale a permitir que el optimizador aproveche los datos disponibles. Por eso sería incorrecto convertir el fracaso principal hasta un millón de etiquetas en una afirmación de que BE necesita más de un millón para ser aprendido.

Este control no demuestra que 3.000 sea el mínimo para 1.500 actualizaciones, que cualquier inicialización vaya a funcionar, ni que toda dificultad de BE sea de optimización. Tampoco demuestra que se haya aprendido su aritmética algebraica: parte de ella sigue suministrada por la interfaz.

Es un **protocolo adicional exploratorio**, documentado después de los controles de development y con evaluación independiente. Su resultado positivo se informa junto al resultado principal negativo de 500 actualizaciones. No se usan sus casos para volver a seleccionar un presupuesto del estudio principal.

## 7. Información suministrada y alcance de las adaptaciones

Comparar profesores sin auditar sus observaciones confundiría información insuficiente con incapacidad de aprendizaje. Se mantuvieron dos regímenes:

| Profesor | Información adicional relevante y computación suministrada |
|---|---|
| Luby | Interfaz anónima: estado local, observaciones de vecinos y aleatoriedad privada del protocolo |
| Johansson | Interfaz anónima de la variante implementada, con propuestas aleatorias y participación siempre activa como objetivo |
| ColorTrials | Interfaz anónima con preferencias privadas y dos intercambios para propagar disponibilidad y competencia |
| JP | Comparaciones explícitas de orden entre IDs vecinos y propio, representadas mediante −1/0/+1 |
| BE | Colores auxiliares, fase, cotas y objetivo; evaluaciones polinómicas locales, relaciones de igualdad y decodificación suministradas |
| GM | Comparaciones de prioridades, relación de posición, permisos y planificación de bloques y reparación |
| Greedy | Permiso de escritura del nodo elegido por un planificador secuencial basado en IDs |

En JP y GM, representar directamente el orden evita depender exclusivamente de diferencias numéricas pequeñas entre IDs. En Greedy el planificador ya selecciona qué nodo puede escribir: el modelo aprende las decisiones correspondientes bajo ese permiso, no el ordenamiento global desde una red anónima.

En BE se suministran la aritmética de cuerpos finitos y el intérprete de fases. La red sigue aprendiendo qué punto o color elegir y cuándo recolorear. La decodificación algebraica no busca por el alumno una respuesta válida. Aun así, el resultado debe describirse como aprendizaje **con esa estructura suministrada**, no como descubrimiento desde cero de todo BE.

Dos argumentos ilustran por qué el tipo de información importa:

1. En una arista con dos extremos anónimamente indistinguibles, intercambiar sus IDs cambia cuál debe actuar en JP. Una interfaz que no distingue esas asignaciones no puede reproducir ambas decisiones.
2. En la primera fase F13 de BE, las asignaciones (1,2) y (1,14) tienen el mismo orden de IDs, pero cambian el primer punto polinómico admisible: 0 frente a 1. El orden por sí solo no sustituye la información algebraica.

Se prueban nuevas asignaciones válidas de identificadores; para JP y GM existen además regresiones de invariancia bajo transformaciones crecientes válidas. No se afirma esa invariancia para BE, donde transformar IDs puede cambiar los polinomios.

**Cautela histórica:** la cohorte integrada anterior de 70 modelos ya recibía diferencias numéricas de IDs y ayudas algebraicas para BE. El nuevo éxito de JP no permite afirmar que todo su fracaso histórico se debiera a no disponer de IDs. Cambiaron también representación, datos, pérdidas y entrenamiento. Falta una ablación emparejada que aísle el efecto del adaptador.

Los argumentos de suficiencia informacional tampoco prueban representabilidad universal por cada anchura concreta de red. Recibir la información necesaria, poder representar una regla y encontrarla mediante optimización son cuestiones distintas.

## 8. Modelos pequeños, comunicación e imitación interactiva

### 8.1. Controles a B = 3.000

La siguiente tabla cuenta **semillas perfectas en development**, no certificaciones finales:

| Profesor | E1 pasivo, 500 actualizaciones | E1 pasivo, 1.500 | Pequeño, 500 | Sin comunicación, 500 | Interactivo, 500 totales |
|---|---:|---:|---:|---:|---:|
| Luby | 5/5 | 5/5 | 1/5 | 0/5 | 4/5 |
| JP | 5/5 | 5/5 | 0/5 | 0/5 | 5/5 |
| BE | 0/5 | 5/5 | 0/5 | 0/5 | 1/5 |

El control sin comunicación borra la información recibida de vecinos, conservando el estado local y los permisos externos. Su resultado apoya la utilidad de la información de vecinos bajo este presupuesto. No prueba que esta GNN concreta sea necesaria ni que ninguna regla local alternativa pudiera resolver el dominio.

### 8.2. Curvas completas del modelo pequeño

Se ejecutó además la rejilla de siete presupuestos del modelo pequeño para Luby, JP y BE: **105 registros**, de los cuales 15 reutilizan por identidad los checkpoints del control B = 3.000 y 90 corresponden a los otros presupuestos.

| Profesor | Parámetros del modelo pequeño | Resultado hasta B = 1.000.000 | Confirmación estudiantil |
|---|---:|---|---|
| Luby | 551 | Sin candidato común en development | No intentada |
| JP | 1.560 | Sin candidato común en development | No intentada |
| BE | 1.560 | Sin candidato común en development | No intentada |

Se congelaron baterías finales independientes y se validaron sus profesores exactos, pero ningún alumno pequeño llegó al requisito previo para usarlas. Por eso 0/5 confirmadas no equivale a cinco fallos finales observados.

La conclusión queda condicionada a **estos tamaños, datos, pérdidas y 500 actualizaciones**. Más etiquetas no rescataron la certificación del pequeño bajo ese entrenamiento. No se estableció un tamaño mínimo de red ni se separó completamente insuficiencia de capacidad de dificultad de optimización.

### 8.3. Qué hizo el control interactivo

El procedimiento tipo DAgger fue una sola agregación: 1.500 etiquetas pasivas y 250 actualizaciones; después hasta 1.500 etiquetas sobre estados visitados por esa política; finalmente otras 250 actualizaciones con los datos agregados. El presupuesto máximo total siguió siendo B = 3.000.

Las consultas se limitaron a estados admisibles para el profesor; las trayectorias fallidas se conservaron. Se reinició Adam al agregar datos, por lo que el contraste no aísla solamente la distribución de estados. GM se excluyó porque no se demostró la admisibilidad de historias especulativas arbitrarias del alumno.

Los resultados de development son mixtos: Luby baja de 5/5 a 4/5, JP mantiene 5/5 y BE sube de 0/5 a 1/5. **No hay una ventaja general demostrada de supervisión interactiva.** Solo se evaluó este punto de presupuesto, con una iteración de agregación y sin confirmación final de estos controles.

## 9. Éxito funcional, fidelidad y tiempo son resultados distintos

### 9.1. Políticas certificadas que no copian todas las acciones

Una auditoría adicional cubrió todas las raíces y transiciones de las 28 trayectorias de development del profesor por semilla, para los **735 checkpoints principales**. No usó datos finales ni actualizó pesos. Sustituye para estas curvas el diagnóstico original limitado al prefijo de 256 decisiones, que se conserva como registro parcial.

En los checkpoints certificados se observaron los siguientes rangos entre las cinco semillas:

| Profesor y arquitectura | Fidelidad de acciones | Fidelidad de transiciones |
|---|---:|---:|
| Luby — E1 | 94,2–97,2% | 68,0–78,7% |
| Luby — DeepSets | 93,1–95,8% | 58,7–70,6% |
| Luby — Transformer | 91,7–96,2% | 56,7–75,7% |
| Johansson — E1 | 87,7–96,2% | 43,1–73,0% |
| Johansson — DeepSets | 82,0–91,6% | 31,9–52,9% |
| Johansson — Transformer | 88,2–95,4% | 34,8–63,8% |

Son mínimos y máximos descriptivos entre semillas, no intervalos de confianza. Las acciones se comparan sobre estados comunes generados por el profesor. La fidelidad de transiciones comprueba la igualdad del estado actualizado en un paso, de modo que una diferencia local puede cambiar toda la transición. También contempla que acciones redundantes distintas pueden dejar el mismo estado. El denominador de acciones nativas de esta auditoría no es el número de tres componentes supervisados utilizados para contabilizar consultas.

Todas las políticas de esta tabla pasan sus respectivas baterías finales. Así, el éxito funcional observado no exige coincidencia exacta de acciones o transiciones con el profesor. Esto no implica que las diferencias sean irrelevantes en todos los grafos: la fidelidad se midió en development y el éxito en otras baterías finitas.

JP y Greedy tienen fidelidades mucho más cercanas al 100%: E1 y DeepSets alcanzan 100% en ambas métricas en sus cinco candidatos. Transformer presenta alguna discrepancia, aunque conserva la certificación funcional. Tampoco estos resultados finitos demuestran identidad universal de políticas.

### 9.2. Pasos de ejecución

Las comparaciones de pasos se hacen dentro de cada profesor, con casos emparejados. Como ejemplo, para E1 los promedios por semilla del cociente de pasos alumno/profesor se sitúan aproximadamente entre **0,956 y 1,029 en Luby**, y entre **1,020 y 1,064 en Johansson**. En los candidatos de JP y Greedy el cociente registrado es 1 en las tres arquitecturas.

Igualar pasos nativos no equivale a igualar coste distribuido. El núcleo neuronal usa dos etapas de intercambio por paso; las reglas exactas de JP y Greedy requieren una lectura vecinal en la interpretación utilizada. Además, la planificación y la aritmética suministradas tienen coste. No se midieron bits transmitidos, latencia de red ni complejidad asíncrona. Estos datos no establecen una mejora asintótica de complejidad del algoritmo aprendido.

## 10. Distribución de coloraciones completas

El suplemento independiente de ley global comparó:

- Siete profesores, tres arquitecturas, cinco semillas y dos grafos fijos de ocho vértices: un camino y un cúbico.
- 128 trayectorias por banco.
- Un instrumento robusto auditado para la paleta nativa de cuatro colores, con separación de selección y evaluación.
- 9.999 permutaciones y corrección de Holm sobre las **210 hipótesis planificadas**.

Las **210 fueron contrastables y ninguna rechazó**. Se aplicó además una regla conservadora de inutilidad estadística que puede conservar p = 1; no puede crear rechazos y limita la lectura de los no rechazos.

La conclusión es: **en este suplemento no se detectaron diferencias distribucionales tras la corrección utilizada**. No se ha demostrado igualdad, equivalencia aproximada ni potencia suficiente frente a cualquier alternativa. Solo se probaron dos grafos pequeños.

Las leyes se comparan condicionadas a las ejecuciones exitosas. Por eso una política funcionalmente no certificada podría producir una distribución aparentemente compatible entre sus éxitos, sin que el test elimine sus fallos.

La tirada sí aporta ejemplos de éxito funcional con fidelidad de acciones inferior al 100%. En cambio, **no aporta una diferencia de ley global detectada que sirva de ejemplo positivo de éxito funcional con distinta distribución**. Esa distinción es importante al evaluar la hipótesis sobre separación funcional y distribucional.

## 11. Variabilidad, no monotonía e incertidumbre

Se conservaron las cinco curvas individuales, sin suavizado. Hubo **34 descensos de éxito en development entre presupuestos adyacentes de una misma semilla**: 21 en ColorTrials, 11 en BE y dos en GM.

La cuenta de semillas perfectas ilustra esa inestabilidad:

| B | ColorTrials — E1 | BE — Transformer | GM — DeepSets |
|---:|---:|---:|---:|
| 1.000 | 3/5 | 2/5 | 3/5 |
| 3.000 | 4/5 | 2/5 | 3/5 |
| 10.000 | 4/5 | 4/5 | 3/5 |
| 30.000 | 3/5 | 4/5 | 5/5 |
| 100.000 | 3/5 | 3/5 | 4/5 |
| 300.000 | 5/5 | 4/5 | 5/5 |
| 1.000.000 | 4/5 | 3/5 | 5/5 |

Esta tabla describe development; no son siete confirmaciones finales. También pueden existir regresiones individuales aunque la cuenta total de semillas perfectas no cambie. Las diferencias de acierto son múltiplos de 1/28 por política, de modo que la resolución de development es limitada.

El presupuesto de 500 actualizaciones es fijo. Ampliar la biblioteca no garantiza que cada etiqueta se presente al optimizador, ni que una etiqueta rara se visite lo suficiente. Las cuentas de reutilización y visitas se conservaron precisamente para distinguir etiquetas adquiridas de ejemplos efectivamente presentados.

Para una política con cero fallos en 1.000 casos, el límite inferior unilateral del 95% es **0,9970087504549047**, aproximadamente **99,7009%**, bajo los supuestos estadísticos especificados. Con estratos independientes fijos, el mismo resultado es conservador para la media ponderada por la asignación de casos; no exige fingir que todas las familias tienen idéntica probabilidad de éxito.

Ese límite se informa **por política**. No se mezclan los 5.000 casos de cinco políticas para fabricar un intervalo de una sola política, ni se interpreta como garantía simultánea para todas las celdas. Tampoco convierte el criterio de aprobación en un umbral del 95%: la regla observacional sigue exigiendo cero fallos.

Las cinco semillas son las réplicas de entrenamiento. Los casos finales miden cada política congelada, no son entrenamientos adicionales. Cinco semillas apoyan la robustez observada de esas políticas, pero ofrecen información limitada sobre la probabilidad de que una nueva inicialización pase.

## 12. Qué puede sostenerse y qué queda abierto

| Afirmación | Evaluación a partir de esta tirada |
|---|---|
| El presupuesto certificado depende del profesor y de la arquitectura | Apoyado dentro del procedimiento finito; Luby diferencia arquitecturas y los profesores muestran resultados distintos |
| E1 o Transformer siempre necesitan menos supervisión que DeepSets | No establecido: hay empates en el borde inferior de la rejilla y profesores sin certificación |
| BE es inaprendible para estas redes | No respaldado: E1 pasa el control adicional con más optimización y las mismas 3.000 etiquetas |
| Todo el fracaso principal de BE se explica por pocas etiquetas | No respaldado: el control identifica sensibilidad a las actualizaciones con las etiquetas fijas |
| El nuevo adaptador de IDs causó por sí solo el éxito de JP | No aislado: faltan ablaciones con datos, pérdidas y optimización emparejados |
| La información de vecinos aporta una ventaja en los controles ensayados | Apoyado al comparar E1 con el control sin comunicación en B = 3.000; alcance limitado a esos controles |
| Un pequeño concreto requiere más etiquetas bajo 500 actualizaciones | El umbral no se alcanzó hasta el máximo; no se ha separado completamente capacidad de optimización ni estimado su mínimo real |
| La imitación interactiva es más eficiente en general | No establecido: resultados mixtos y un solo punto de presupuesto |
| PASS funcional exige fidelidad exacta de acciones y transiciones | Refutado como exigencia empírica por los candidatos certificados con fidelidad inferior al 100% |
| Las leyes globales son iguales | No demostrado por 0/210 rechazos |
| PASS demuestra correctitud para cualquier grafo y cinta | No: la certificación pertenece a las baterías y al dominio evaluados |

Las principales limitaciones son el dominio de grado máximo tres y cuatro colores globales, los tamaños finales hasta 64, el pequeño development, cinco semillas y el presupuesto fijo de optimización. No se validaron grados 200, paletas locales arbitrarias o tamaños sin límite.

Los valores B\* son mínimos observados en una rejilla y condicionados a la selección por development. Los casos no certificados son umbrales no alcanzados por el procedimiento; no deben transformarse en cotas inferiores informacionales ni analizarse como umbrales monótonos sin justificar esa hipótesis.

Los controles de optimización, capacidad, modalidad y los diagnósticos son exploratorios. La identificación del fallo de aislados en GM se hizo después de observar final. Es una explicación útil y reproducible, pero no un factor elegido prospectivamente para una prueba causal.

Una garantía matemática exigiría argumentos adicionales sobre los estados alcanzables, la seguridad y el progreso de la política aprendida, el horizonte y el dominio, además de tratar correctamente precisión y aleatoriedad. Las pruebas de los profesores exactos no se transfieren automáticamente a sus imitadores.

## 13. Reproducibilidad, recursos y artefactos

El entrenamiento se ejecutó en **CPU**, con tres trabajadores y un hilo Torch por trabajador; se registraron **cero horas de entrenamiento GPU**. Las cuentas de parámetros, tiempos y operaciones se conservaron separadas de las consultas. La estimación mediante profiler cubre operaciones dominantes concretas y **no es una medición completa de FLOPs**.

Como referencia de capacidad, E1/DeepSets/Transformer tienen respectivamente **16.035 / 17.907 / 16.995 parámetros** en Luby; en JP son **19.396 / 21.268 / 20.356**. Por tanto, igualar el número de actualizaciones no iguala automáticamente tiempo o coste por actualización.

El cierre de la tirada incluyó:

- **601 pruebas automatizadas aprobadas**, incluidas las del recuperador de archivos.
- Verificación de conservación de las fuentes y los 70 modelos del estudio histórico.
- Manuscrito de **30 páginas** compilado y revisado visualmente.
- Archivo de recuperación de **1.145.842.922 bytes**, con **3.004 archivos y 921 checkpoints**. Ese número de checkpoints abarca artefactos de la entrega y no sustituye el conteo de 735 entrenamientos principales.
- Coincidencia de los hashes de los doce archivos publicados con sus versiones locales.
- Recuperación desde un clon Git independiente y desde el paquete descargado de la release, usando el entorno Python registrado.
- Regeneración idéntica byte a byte de nueve PNG, tablas, informes y texto del manuscrito. Los PDF regenerados pueden variar en metadatos temporales; el PDF publicado conserva su hash registrado.
- Repetición desde inicialización nueva de una celda Luby/E1 con etiquetas guardadas: los pesos finales coincidieron exactamente. Es una comprobación numérica de una celda, no una nueva réplica independiente de toda la cohorte. No generó consultas docentes ni nuevas confirmaciones finales.

Durante la recuperación se detectó una conexión HTTP bloqueada. Se corrigió el descargador para usar rangos acotados, timeout de 30 segundos y reanudación de la caché. La recuperación completada verificó el SHA-256 global y los hashes de cada archivo. La corrección afecta al transporte de artefactos, manteniendo los hashes científicos originales.

**Fuentes de evidencia y entrega:**

- [Release con checkpoints, PDF y documentación](https://github.com/mrioswilson/distributed-coloring-learning/releases/tag/sample-complexity-2026-10-23).
- [Tabla detallada principal, instantánea verificada](https://github.com/mrioswilson/distributed-coloring-learning/blob/08aa8c12597e360276be8e03d28943b7eabb87cc/results/sample-complexity/results-table.json).
- [Protocolo completo](https://github.com/mrioswilson/distributed-coloring-learning/blob/08aa8c12597e360276be8e03d28943b7eabb87cc/docs/SAMPLE_COMPLEXITY_PROTOCOL.md).
- [Auditoría de compatibilidad informacional](https://github.com/mrioswilson/distributed-coloring-learning/blob/08aa8c12597e360276be8e03d28943b7eabb87cc/docs/TEACHER_MODEL_INFORMATION_COMPATIBILITY.md).
- [Resultados de optimización independiente](https://github.com/mrioswilson/distributed-coloring-learning/blob/08aa8c12597e360276be8e03d28943b7eabb87cc/results/sample-complexity/optimization-confirmation/results.json).
- [Auditoría completa de fidelidad](https://github.com/mrioswilson/distributed-coloring-learning/blob/08aa8c12597e360276be8e03d28943b7eabb87cc/results/sample-complexity/full-fidelity.json).
- [Diagnóstico de GM y comprobación de cobertura](https://github.com/mrioswilson/distributed-coloring-learning/blob/08aa8c12597e360276be8e03d28943b7eabb87cc/results/sample-complexity/structural-failure-witnesses.json).
- [Comprobante de publicación y recuperación](https://github.com/mrioswilson/distributed-coloring-learning/blob/08aa8c12597e360276be8e03d28943b7eabb87cc/results/sample-complexity/publication-verification.json).

Los datos por semilla, curvas, pruebas de ley global y controles pequeños están bajo `results/sample-complexity/`. El texto del artículo y la auditoría extensa están en `paper/PAPER_DRAFT.md` y `docs/PAPER_SCIENTIFIC_AUDIT.md`. Este resumen contiene las cifras necesarias para una primera revisión sin abrir esos archivos.

## 14. Preguntas y experimentos futuros para discutir

Estas propuestas **no son resultados ejecutados de esta tirada**:

1. **Optimización de BE:** repetir una rejilla equilibrada de etiquetas y actualizaciones para las tres arquitecturas, con nuevas semillas y confirmación independiente. ¿Qué parte de la diferencia persiste al permitir convergencia comparable?
2. **Adaptador de IDs:** comparar magnitudes, signos y otras representaciones con el mismo conjunto de etiquetas, pérdidas, semillas y actualizaciones. ¿Cuánto explica la representación frente al resto del paquete?
3. **Cobertura de GM:** incluir prospectivamente aislados y otros casos raros en training y development. Una política corregida necesitaría otra batería final independiente; no reemplazaría el resultado negativo original.
4. **ColorTrials:** localizar los errores por componente, disponibilidad y tipo de vecindario, y contrastar causas en nuevos experimentos. Los 25 fallos actuales son evidencia para formular hipótesis, no para retocar y reconfirmar el mismo candidato.
5. **Umbrales inferiores:** explorar presupuestos menores de 1.000 para los profesores que pasan en el borde de la rejilla, con una nueva selección y confirmación.
6. **Interacción y capacidad:** ampliar presupuestos, número de agregaciones y controles de historia del optimizador; permitir también más optimización a los modelos pequeños.
7. **Ley global:** estudiar potencia frente a alternativas conocidas y ampliar grafos, tamaños y muestras antes de presentar no rechazo como compatibilidad fuerte o plantear un test de equivalencia.
8. **Garantías:** investigar invariantes verificables y cobertura formal de estados que permitan vincular una política concreta con seguridad y progreso más allá de una batería finita.

## 15. Texto sugerido para solicitar una revisión crítica

> Revisa científicamente el estudio descrito en este documento. Trabaja con los datos proporcionados y señala cuándo una pregunta no puede resolverse sin evidencia adicional. No supongas experimentos ni acceso al repositorio que no se hayan descrito.
>
> 1. Distingue resultados confirmatorios principales, controles exploratorios y diagnósticos posteriores a final.
> 2. Evalúa si B\* está interpretado correctamente como presupuesto validado de una rejilla bajo un algoritmo de entrenamiento y selección concretos.
> 3. Examina si el control de BE permite separar supervisión de optimización y qué conclusiones causales siguen sin identificar.
> 4. Revisa la suficiencia informacional, las ayudas codificadas y la atribución de las mejoras de JP respecto a experimentos históricos.
> 5. Comprueba que se distinguen cinco políticas de 5.000 ejecuciones de una sola política y que no se exagera la evidencia de cinco semillas.
> 6. Evalúa el diagnóstico de GM, las debilidades del pequeño development y el tratamiento de los casos sin candidato final.
> 7. Separa éxito funcional, fidelidad de acciones, tiempo y ley global. Valora especialmente lo que permiten concluir 0/210 rechazos.
> 8. Propón una narrativa para el artículo, identifica afirmaciones demasiado fuertes y prioriza los siguientes experimentos con mayor valor científico.
>
> Para cada objeción, indica si es un error demostrado, una limitación reconocida o una hipótesis que requeriría datos nuevos. Conserva los resultados negativos y el criterio estricto de cero fallos.
