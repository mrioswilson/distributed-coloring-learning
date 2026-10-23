# CODEX — Sample Complexity and Learnability of Distributed Algorithms

## Misión científica

Continúa nuestro proyecto conjunto de aprendizaje de algoritmos distribuidos de coloración.

Quiero introducir una nueva dimensión central para el artículo:

**¿Cuántas observaciones de un algoritmo profesor necesita una arquitectura neuronal para aprender a resolver correctamente el problema distribuido?**

Adoptaremos un criterio binario:

**Un modelo supera el benchmark únicamente si completa correctamente el 100% de las ejecuciones finales preregistradas.**

No utilizar un umbral de 95% ni llamar «aprendido» a un modelo que consigue una tasa elevada pero presenta fallos.

Este criterio es empírico y relativo al benchmark. No confundirlo con una demostración matemática de correctitud para todos los grafos y todas las cintas aleatorias.

Queremos comparar:

- E1 NORMALIZED.
- DeepSets.
- LocalPaletteTransformer.

Con múltiples algoritmos profesores y cinco semillas independientes de entrenamiento.

La variable principal será el presupuesto de supervisión necesario para alcanzar el criterio de éxito.

Este estudio debe quedar integrado en el repositorio, en el manuscrito y en el informe de auditoría científica.

---

# I. Primera obligación: auditar la información disponible

Nuestros modelos originales fueron diseñados para redes anónimas.

Algunos profesores de Pedro, en cambio, utilizan identificadores únicos, órdenes de ejecución, fases u otra información explícita.

Es incorrecto comparar sus capacidades de imitación sin examinar primero la información recibida por cada modelo.

## Profesores

Auditar:

1. LUBY.
2. JOHANSSON.
3. COLOR_TRIALS.
4. JP.
5. BE_BASE_D3.
6. SPECULATIVE_GM.
7. GREEDY.

Para cada profesor identificar formalmente:

- Información local requerida.
- Necesidad de identificadores únicos.
- Necesidad de ordenamientos.
- Aleatoriedad privada.
- Número y tipo de intercambios de mensajes.
- Relojes y fases.
- Información algebraica.
- Estado local persistente.
- Garantías de seguridad.
- Garantías de progreso.
- Algoritmos auxiliares codificados.

Construir una tabla:

| Profesor | Anónimo | IDs | Reloj/fases | Orden externo | Información auxiliar | Adaptación necesaria |
|---|---|---|---|---|---|---|

La tabla debe reflejar el código realmente implementado, no únicamente las referencias bibliográficas originales.

## Adaptación de las arquitecturas

Cuando un profesor requiere IDs, las arquitecturas deben recibir esa información mediante una representación adecuada y explícita.

Investigar representaciones relacionales de identificadores y prioridades, especialmente comparaciones de orden, en lugar de depender únicamente de sus magnitudes numéricas.

Cuando requiera fases, programas auxiliares o planificación, incorporar las interfaces correspondientes y documentar qué computación continúa siendo aprendida.

No convertir silenciosamente un profesor con IDs en otro profesor anónimo.

No eliminar características esenciales del profesor para forzarlo a una interfaz anónima.

No suministrar respuestas oraculares durante inferencia que permitan resolver la tarea sin aprender las decisiones correspondientes.

### Dos regímenes experimentales

**Régimen A: anónimo.**

Utilizar profesores que admitan esa interfaz sin información adicional.

**Régimen B: extendido.**

Proporcionar la información adicional estrictamente requerida por cada profesor.

Los resultados deben presentarse separadamente.

Para profesores basados en IDs, probar generalización ante nuevas asignaciones válidas de identificadores y sus transformaciones que preservan el orden, cuando corresponda.

La prueba debe respetar los supuestos formales del profesor.

### Correctitud de las adaptaciones

Verificar que cada arquitectura recibe información suficiente para representar las acciones requeridas.

No afirmar representabilidad universal solamente porque el entrenamiento converge.

Cuando sea posible, producir argumentos formales de suficiencia o contraejemplos de indistinguibilidad para interfaces insuficientes.

Entregable:

`docs/TEACHER_MODEL_INFORMATION_COMPATIBILITY.md`

---

# II. Definición de aprendizaje exitoso

Sea:

- T un profesor.
- M una arquitectura.
- B el presupuesto de consultas al profesor.
- s una semilla independiente de entrenamiento.
- E una batería final congelada.

Definir:

\[
\operatorname{PASS}(M,T,B,s;E)=1
\]

si y sólo si la política entrenada satisface todas las condiciones siguientes en todas las ejecuciones de E:

1. La coloración final es propia.
2. Cada color pertenece a la paleta correspondiente.
3. Se respetan los invariantes exigidos por el protocolo.
4. La ejecución termina dentro del horizonte establecido.
5. No se produce ningún error técnico.
6. No existen cutoffs.
7. No aparecen estados terminales incorrectos.

En cualquier otro caso:

\[
\operatorname{PASS}=0.
\]

No sustituir fallos por nuevas ejecuciones aleatorias.

No descartar grafos difíciles.

No contar un cutoff como éxito.

No aplicar reparaciones externas no incluidas en el modelo evaluado.

## Regla para las cinco semillas

La combinación arquitectura–profesor supera el benchmark en un presupuesto B solamente si:

\[
\boxed{
\bigwedge_{s=1}^{5}
\operatorname{PASS}(M,T,B,s;E_s)=1.
}
\]

Es decir, **los cinco entrenamientos deben completar correctamente sus respectivas baterías finales**.

Adicionalmente, reportar el número de semillas que superan la prueba: 0/5, 1/5, ..., 5/5.

No ocultar la variabilidad de entrenamiento.

La clasificación principal será binaria:

- PASS: cinco semillas superan íntegramente la batería.
- FAIL / NOT CERTIFIED: al menos una semilla no la supera.

Esta clasificación pertenece al benchmark y al presupuesto evaluado. No implica que una arquitectura sea matemáticamente incapaz de aprender el profesor.

---

# III. Horizonte de evaluación y profesores exactos

El antiguo cutoff común de 128 pasos no es adecuado para todos los profesores.

Por ejemplo, un algoritmo secuencial puede necesitar un número de pasos proporcional a n, mientras que uno aleatorizado puede tener una cola de terminación no acotada.

Antes de entrenar modelos, establecer para cada profesor:

- Unidad de paso.
- Horizonte de evaluación.
- Justificación matemática o experimental del horizonte.
- Relación con la ejecución exacta.
- Diferencia entre plazo de evaluación y garantía de terminación.

El horizonte debe fijarse prospectivamente.

No modificarlo después de observar fallos de un alumno.

Validar la batería con el profesor exacto.

Si el profesor exacto no supera el horizonte fijado, reconsiderar el diseño antes de congelar el benchmark. No seleccionar retrospectivamente sólo las cintas donde el profesor termina.

Conservar además una métrica de eficiencia temporal frente al profesor exacto.

No comparar directamente pasos nativos de profesores con semánticas temporales diferentes como si fueran unidades idénticas.

---

# IV. Qué significa una observación del profesor

Ésta es la variable experimental más importante.

No confundir:

- Una trayectoria completa.
- Una ronda distribuida.
- Una decisión de un nodo.
- Una consulta al profesor.
- Una época de entrenamiento.
- Una actualización del optimizador.

## Unidad primaria

Definir una consulta supervisada como la obtención de una decisión etiquetada del profesor para un nodo, estado y componente de acción especificados.

Cada acción requerida cuenta de forma trazable.

Por ejemplo, si un estado necesita etiquetas separadas de participación, propuesta y confirmación, contabilizar las etiquetas efectivamente obtenidas.

Para profesores con otras interfaces, registrar el tipo de consulta.

Este cómputo no implica que todas las etiquetas tengan idéntica dificultad o coste. Por eso registrar también las cuentas desglosadas por tipo.

## Contabilidad adicional obligatoria

Para cada entrenamiento registrar:

1. Número de consultas supervisadas.
2. Consultas por componente.
3. Número de trayectorias independientes.
4. Número de grafos de entrenamiento.
5. Número de estados locales distintos.
6. Número de estados globales distintos.
7. Cantidad de datos únicos almacenados.
8. Reutilizaciones de esos datos.
9. Épocas.
10. Actualizaciones de parámetros.
11. Tiempo CPU/GPU.
12. Parámetros entrenables.

Reutilizar una observación diez veces en entrenamiento no representa diez nuevas consultas al profesor.

Las observaciones provenientes de una misma trayectoria están correlacionadas: conservar esa estructura para el análisis estadístico.

El presupuesto primario será de supervisión, no de actualizaciones de gradiente.

---

# V. Presupuestos de supervisión

Utilizar inicialmente la escala:

\[
B\in
\{10^3,3\cdot10^3,10^4,3\cdot10^4,
10^5,3\cdot10^5,10^6\}.
\]

Estos valores son presupuestos máximos de consultas supervisadas acumuladas, según la unidad definida anteriormente.

Si un profesor produce ejemplos de acciones heterogéneas, estratificar el muestreo y registrar cuánto presupuesto se asignó a cada componente.

Evitar que un modelo fracase simplemente porque nunca recibió ejemplos suficientes de una fase excepcional del protocolo.

Pero tampoco utilizar información de evaluación final para elegir retrospectivamente esa distribución de ejemplos.

## Curvas de aprendizaje

Para cada profesor y arquitectura producir:

\[
\operatorname{Success}_{M,T}(B),
\]

\[
\operatorname{ActionFidelity}_{M,T}(B),
\]

\[
\operatorname{Rounds}_{M,T}(B).
\]

Mostrar cada una por semilla.

No promediar antes de examinar las cinco curvas.

La tasa de éxito es descriptiva. El criterio de aprendizaje exitoso sigue siendo estrictamente 100%.

---

# VI. Diseño de semillas y validación

Utilizar cinco semillas independientes de entrenamiento como protocolo principal.

Cada semilla debe controlar:

- Inicialización neuronal.
- Orden de ejemplos.
- Generación aleatoria de datos.
- Operaciones estocásticas de entrenamiento.

Conservar identificadores, estados RNG y hashes.

Los grafos y cintas de evaluación deben ser independientes de los utilizados para producir los ejemplos docentes.

## Tres conjuntos separados

### Training

Datos utilizados para actualizar parámetros.

### Development

Datos utilizados para controlar convergencia, seleccionar hiperparámetros y determinar candidatos de presupuesto.

### Final confirmation

Batería congelada e independiente para determinar PASS o FAIL.

El conjunto de confirmación final no puede utilizarse para ajustar el modelo, escoger el presupuesto o rediseñar el procedimiento después de consultar sus resultados.

## Evaluación final

Realizar al menos:

**1.000 ejecuciones completas independientes por semilla**, distribuidas de forma preregistrada entre grafos y familias.

Por tanto, una combinación profesor–arquitectura validada con cinco semillas habrá superado 5.000 ejecuciones finales sin errores.

Los resultados individuales deben quedar disponibles.

No sumar artificialmente las cinco semillas para presentar un intervalo correspondiente a una única política.

Con cero fallos en 1.000 ejecuciones de una política fija, reportar el límite inferior unilateral binomial del 95% para la probabilidad de éxito bajo la distribución de evaluación.

No interpretar ese intervalo como una garantía sobre todos los grafos.

---

# VII. Estimación de complejidad muestral

Definir el presupuesto mínimo observado que supera la validación:

\[
B^*_{M,T}
=
\min\{B:\text{se valida PASS para las cinco semillas}\}.
\]

Este mínimo sólo podrá afirmarse respecto de la rejilla de presupuestos y del protocolo de evaluación utilizado.

No asumir que la probabilidad de éxito o el comportamiento del optimizador mejoran monótonamente con B.

## Procedimiento para elegir presupuestos

1. Entrenar y evaluar en development según la rejilla fijada.
2. Identificar el primer presupuesto candidato donde las cinco semillas alcanzan 100% en development.
3. Congelar los cinco checkpoints y la configuración.
4. Evaluar una sola vez en final confirmation.
5. Declarar PASS únicamente si las cinco políticas obtienen cero fallos.

Si la confirmación falla, registrar NOT CERTIFIED para ese candidato.

No seguir ajustando el mismo modelo utilizando información de la batería final.

Cualquier intento posterior de certificación deberá formar parte de un protocolo adicional documentado y utilizar evaluación independiente.

Si no se obtiene PASS dentro de los presupuestos disponibles, reportar:

\[
B^*>\text{máximo presupuesto evaluado}
\]

en el sentido de umbral no alcanzado en el protocolo, sin afirmar imposibilidad matemática.

Para el análisis de curvas, también registrar el primer presupuesto que supera development y las posibles regresiones en presupuestos posteriores.

---

# VIII. Separar datos y optimización

Existe una confusión potencial que debemos evitar.

Un modelo puede necesitar más supervisión o simplemente más actualizaciones de gradiente.

Por eso no basta con aumentar simultáneamente ejemplos, épocas y tiempo de entrenamiento.

## Protocolo primario

Evaluar curvas de aprendizaje desde inicializaciones nuevas con:

- Presupuesto docente controlado.
- Presupuesto de optimización especificado.
- Hiperparámetros registrados.
- Criterios comunes de selección.
- Número de actualizaciones documentado.

La comparación principal debe utilizar un diseño de optimización tan equivalente como sea razonable entre arquitecturas.

Si los costes por actualización difieren, conservar medidas separadas de tiempo y FLOPs estimados.

## Experimento de control

Sobre combinaciones representativas, mantener fijo B y aumentar únicamente el número de actualizaciones.

Esto permitirá distinguir:

- Insuficiencia de datos.
- Optimización insuficiente.
- Inestabilidad del entrenamiento.
- Limitaciones de representación.

No atribuir automáticamente a complejidad muestral un fracaso provocado por un límite artificial de entrenamiento.

---

# IX. Dos modalidades de aprendizaje

## A. Imitación pasiva

El alumno observa trayectorias generadas por el profesor.

El entrenamiento utiliza acciones etiquetadas sobre estados del profesor.

Ésta será la modalidad principal para comparar complejidad muestral.

## B. Imitación interactiva

El alumno ejecuta su política y solicita etiquetas al profesor sobre estados que encuentra durante sus propias trayectorias.

Investigar una metodología tipo DAgger cuando el profesor esté bien definido para esos estados.

No asumir que todos los profesores pueden actuar correctamente sobre estados arbitrarios producidos por el alumno.

Establecer previamente las condiciones de admisibilidad de cada consulta.

Comparar ambas modalidades utilizando el mismo presupuesto total de consultas docentes.

El objetivo es responder:

**¿Se aprende mejor consultando más al profesor o consultándolo sobre estados más relevantes para la política aprendida?**

Conservar la modalidad interactiva como un experimento separado, no mezclar sus presupuestos con los de imitación pasiva.

---

# X. Diseño experimental y grafos

El estudio principal utilizará el dominio nativo apropiado de los profesores:

- Grafos simples no dirigidos.
- Grado máximo 3, salvo profesores con especificaciones diferentes verificadas.
- Cuatro colores globales cuando esa sea la interfaz original.
- Identificadores únicos cuando sean requeridos.
- Fases y planificación cuando estén formalmente especificadas.

No afirmar que estos resultados validan paletas locales arbitrarias ni grados 200.

## Entrenamiento

Generar familias variadas de grafos con tamaños pequeños e intermedios, manteniendo las restricciones del profesor.

## Development y confirmación

Reservar grafos no observados durante entrenamiento.

Incluir:

- Caminos.
- Ciclos.
- Regulares cúbicos.
- Irregulares de grado máximo 3.
- Bipartitos.
- Grafos con simetrías.
- Casos adversariales conocidos.

Incluir tamaños mayores que los utilizados durante entrenamiento, hasta donde el dominio nativo del profesor y los recursos permitan.

Congelar la composición de la batería antes del entrenamiento confirmatorio.

Si una familia de grafos presenta restricciones que impiden utilizar un profesor, documentarlo y no forzar una comparación inválida.

---

# XI. Arquitecturas y controles

Arquitecturas principales:

1. E1 NORMALIZED.
2. DeepSets.
3. LocalPaletteTransformer.

Conservar la distinción entre su procesamiento de colores y su mecanismo de comunicación entre nodos.

Incluir como controles donde corresponda:

- Profesor exacto.
- Modelo pequeño normalizado.
- Modelo sin comunicación.
- GNN de acción directa de Pedro, cuando exista una implementación comparable y reproducible.

No atribuir una diferencia únicamente a la arquitectura si también cambia la interfaz de información o el objetivo de entrenamiento.

Para cada modelo registrar:

- Parámetros.
- Tipo de agregación.
- Profundidad de comunicación.
- Representación de la paleta.
- Características adicionales.
- Reglas codificadas.
- Acciones aprendidas.
- Coste de inferencia.

Investigar si modelos más pequeños alcanzan 100% con más o menos supervisión que los modelos grandes.

No deducir minimalidad teórica de un experimento finito.

---

# XII. Resultado primario

El resultado principal será una tabla:

| Profesor | E1: B* | DeepSets: B* | Transformer: B* |
|---|---|---|---|
| Luby | | | |
| Johansson | | | |
| ColorTrials | | | |
| JP | | | |
| BE | | | |
| GM | | | |
| Greedy | | | |

Cada celda debe incluir:

- Presupuesto mínimo validado.
- Número de semillas PASS.
- Número de consultas docentes.
- Desglose de fallos.
- Coste computacional.
- Restricciones de información.

Cuando no se alcance el criterio, registrar NOT CERTIFIED y el máximo presupuesto evaluado.

No reemplazar esos resultados por porcentajes promedio cercanos a 100%.

## Visualización principal

Construir curvas de éxito funcional frente al presupuesto docente.

El eje horizontal será logarítmico.

Representar individualmente las cinco semillas.

Mostrar explícitamente el 100% como frontera de aprobación.

No suavizar curvas de manera que oculten caídas o fallos.

---

# XIII. Resultados secundarios

Además del criterio binario de aprendizaje, evaluar:

### Fidelidad de acciones

¿El alumno hace lo mismo que el profesor en estados equivalentes?

### Fidelidad de transiciones

¿Produce los mismos tipos de cambios de estado?

### Complejidad

¿Sus rondas son comparables con las del profesor?

### Generalización

¿Mantiene correctitud al aumentar n o cambiar la familia de grafos?

### Distribución global

¿Genera la misma ley de coloraciones completas que el profesor?

Utilizar el test estadístico robusto previamente validado, con sus adaptaciones auditadas para paletas e interfaces de profesores nuevos.

La igualdad de distribuciones no debe formar parte del criterio de 100% de éxito funcional.

Son dos objetivos distintos.

Para estudiar equivalencia distribucional, no utilizar la ausencia de rechazo como demostración de igualdad.

---

# XIV. Análisis estadístico de eficiencia muestral

La variable primaria es el primer presupuesto validado bajo la regla PASS.

Analizar:

1. Presupuestos alcanzados.
2. Presupuestos censurados.
3. Variabilidad entre semillas.
4. Variabilidad entre grafos.
5. Interacción profesor–arquitectura.
6. Influencia de información adicional.
7. Influencia del tamaño del modelo.
8. Influencia de modalidad pasiva o interactiva.

No utilizar cada ejecución de una misma política como réplica independiente del entrenamiento.

Las semillas son las unidades de replicación de entrenamiento.

Las ejecuciones finales estiman el desempeño de cada política congelada.

Presentar comparaciones con incertidumbre y reconocer que cinco semillas limitan las conclusiones sobre la distribución de entrenamientos posibles.

Si realizas análisis de tiempo hasta alcanzar el criterio, tratar correctamente los presupuestos no alcanzados como datos censurados.

No asumir monotonía cuando los datos no la respalden.

---

# XV. Hipótesis científicas

Investigar, sin imponer resultados:

**H1.** La complejidad muestral depende del profesor incluso cuando se mantiene fija la arquitectura.

**H2.** Arquitecturas que alcanzan el mismo 100% pueden necesitar presupuestos docentes muy diferentes.

**H3.** Algunas dificultades históricas de JP y BE se deben a la interfaz informacional, al entrenamiento o a ambos.

**H4.** La supervisión interactiva puede mejorar eficiencia frente a la imitación pasiva en determinados profesores.

**H5.** La comunicación entre nodos puede aportar una ventaja de eficiencia muestral independiente del número total de parámetros.

**H6.** Alcanzar 100% de éxito funcional no exige reproducir la distribución global del profesor.

Distinguir hipótesis confirmatorias de análisis exploratorios.

---

# XVI. Validación previa del protocolo

Antes de entrenar la cohorte completa, ejecutar un piloto.

El piloto debe verificar:

- Contabilidad correcta de consultas.
- Reproducibilidad de semillas.
- Independencia de particiones.
- Compatibilidad de profesores.
- Correctitud de adaptadores.
- Ausencia de información privilegiada indebida.
- Implementación del criterio PASS.
- Funcionamiento de los cutoffs.
- Validez de los profesores exactos.
- Disponibilidad de datos suficientes.
- Coste de entrenamientos.
- Coste de evaluaciones.

Realizar pruebas adicionales sobre problemas sintéticos cuyo comportamiento sea conocido para comprobar la contabilidad de muestras y la lógica del gate 100%.

El piloto puede utilizar modelos y datos reducidos, pero no debe contaminar la batería final.

Congelar el protocolo después del piloto.

Crear:

`docs/SAMPLE_COMPLEXITY_PROTOCOL.md`

---

# XVII. Gestión de recursos

Priorizar una ejecución eficiente.

Reutilizar implementaciones y herramientas existentes.

No reciclar checkpoints previamente entrenados como si fueran entrenamientos nuevos con presupuestos controlados.

Evitar generar repetidamente datos idénticos del profesor.

Utilizar almacenamiento en caché con hashes, conservando las cuentas reales de consultas únicas.

Antes del estudio completo estimar:

- Cantidad de entrenamientos.
- Cantidad de consultas.
- Tiempo GPU.
- Memoria.
- Espacio en disco.
- Coste de evaluación.

Realizar inicialmente un piloto limitado a 12 horas GPU-proceso acumuladas.

Con sus resultados, fijar un presupuesto principal computacional razonable y una estrategia de ejecución balanceada que no favorezca retrospectivamente determinadas arquitecturas.

No lanzar cientos de entrenamientos completos sin una estimación de recursos.

Si el presupuesto disponible impide completar todas las celdas, priorizar comparaciones equilibradas y presentar explícitamente los resultados faltantes.

No fabricar datos ni extrapolar curvas para rellenar tablas.

---

# XVIII. Integración en el paper

Actualizar el borrador:

`paper/PAPER_DRAFT.md`

Y el manuscrito LaTeX correspondiente.

Quiero introducir una sección central titulada provisionalmente:

**How Much Supervision Does It Take to Learn a Distributed Algorithm?**

Debe explicar:

1. Qué significa observar un profesor.
2. Compatibilidad informacional.
3. Definición de aprendizaje exitoso.
4. Complejidad muestral.
5. Variabilidad entre entrenamientos.
6. Diferencias entre arquitecturas.
7. Diferencias entre profesores.
8. Papel de las GNN.
9. Imitación pasiva versus interactiva.
10. Generalización funcional versus distribucional.

No convertir una validación empírica de 100% en una afirmación de correctitud universal.

Explicar con precisión el alcance estadístico del protocolo.

Relacionar los resultados con teoría de aprendizaje por imitación y complejidad muestral, utilizando referencias académicas verificables.

No inventar cotas teóricas de complejidad muestral a partir de curvas empíricas.

---

# XIX. Informe para auditoría independiente

Actualizar:

`docs/PAPER_SCIENTIFIC_AUDIT.md`

Agregar una sección extensa dedicada a esta nueva línea.

Debe permitir revisar:

- Definición exacta de cada consulta.
- Compatibilidad de cada profesor.
- Adaptaciones arquitectónicas.
- Presupuestos.
- Semillas.
- Métodos de entrenamiento.
- Datos de evaluación.
- Correctitud de PASS.
- Resultados y fallos.
- Variabilidad.
- Evidencia estadística.
- Amenazas a la validez.
- Limitaciones.
- Interpretación científica.

Incluir una sección crítica:

**Does 100% Empirical Success Mean the Algorithm Has Been Learned?**

Explicar qué demuestra exactamente el experimento y qué no demuestra.

Proponer las condiciones adicionales necesarias para obtener garantías matemáticas.

---

# XX. Reproducibilidad

Ordenar los scripts y configuraciones del experimento.

Generar:

- Manifiestos de entrenamiento.
- Datos docentes con hashes.
- Checkpoints.
- Logs.
- Tablas de consultas.
- Curvas.
- Resultados por semilla.
- Baterías de evaluación.
- Tests unitarios.
- Tests de regresión.
- Scripts para reproducir las figuras.

Actualizar el README con comandos concretos para reproducir el estudio de complejidad muestral.

No mezclar estos nuevos checkpoints con los 70 anteriores.

Preservar los experimentos históricos.

Al finalizar ejecutar las pruebas, crear commits descriptivos y realizar push al repositorio conforme al flujo Git establecido.

No hacer force-push.

---

# XXI. Preguntas obligatorias del informe final

Responder:

**P1.** ¿Cuántas observaciones de Luby necesita cada arquitectura para superar el benchmark al 100%?

**P2.** ¿Cómo cambia ese presupuesto con Johansson, ColorTrials, JP, BE, GM y Greedy?

**P3.** ¿Qué profesores requieren una adaptación informacional específica y qué información se les suministra?

**P4.** ¿Las adaptaciones resuelven los fracasos anteriores de JP y BE?

**P5.** ¿La diferencia de aprendizaje depende de la semilla de entrenamiento?

**P6.** ¿Qué arquitectura necesita menos supervisión para superar el mismo benchmark?

**P7.** ¿Los modelos pequeños son más o menos eficientes muestralmente?

**P8.** ¿Qué cambia cuando se mantiene fijo el número de etiquetas pero se permite más optimización?

**P9.** ¿La supervisión interactiva mejora la eficiencia frente a observar pasivamente al profesor?

**P10.** ¿Alcanzar 100% funcional implica fidelidad de acciones, complejidad o distribución global?

**P11.** ¿Qué conclusiones son robustas a las cinco semillas?

**P12.** ¿Qué resultados deben incorporarse al cuerpo principal del paper y cuáles al apéndice?

---

# XXII. Criterio de éxito del proyecto

No evaluar el éxito científico por obtener el mayor número posible de celdas PASS.

El objetivo es entender cuándo un algoritmo distribuido resulta aprendible bajo cierta información y arquitectura, y cuánta supervisión requiere.

Una celda que no supera el benchmark también puede ser muy informativa si podemos explicar su fracaso.

Distinguir:

- Información insuficiente.
- Representación insuficiente.
- Supervisión insuficiente.
- Optimización insuficiente.
- Inestabilidad entre semillas.
- Generalización insuficiente.

No atribuir a una causa lo que los experimentos todavía no permiten identificar.

---

# XXIII. Instrucción final

Comienza inspeccionando los siete profesores y sus interfaces.

Verifica primero si E1, DeepSets y Transformer reciben la información que cada profesor requiere.

Corrige y documenta las adaptaciones necesarias.

Implementa la contabilidad de consultas docentes.

Diseña y valida el benchmark con éxito obligatorio de 100%.

Realiza pilotos, congela el protocolo y ejecuta los entrenamientos con cinco semillas independientes.

Construye las curvas de complejidad muestral.

Separa correctamente los efectos de supervisión, optimización y arquitectura.

Investiga los profesores difíciles y conserva todos los resultados negativos.

Integra los hallazgos en el paper y en el informe científico Markdown.

Verifica la reproducibilidad y publica los resultados mediante commits y push.

**La pregunta científica central es: qué información y qué cantidad de supervisión necesita una arquitectura neuronal para aprender de manera funcionalmente perfecta, dentro de un benchmark definido, un algoritmo distribuido de coloración, y cómo depende esa necesidad del profesor y del modelo.**