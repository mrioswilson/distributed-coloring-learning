# CODEX — Factorial Study of Graph Size, Degree, and Distributional Generalization

## Objetivo científico

Continúa nuestro proyecto de investigación sobre aprendizaje de algoritmos distribuidos de coloración.

Queremos estudiar sistemáticamente cómo cambia la distribución de coloraciones completas producida por nuestros modelos neuronales cuando aumentamos, por separado:

1. El número de vértices \(n\).
2. El grado máximo \(\Delta\).

La pregunta central es:

**¿Qué determina que E1, DeepSets y Transformer dejen de producir distribuciones de coloraciones completas semejantes a las de su profesor de Luby: el tamaño del grafo, el grado, la cantidad de colores disponibles o una interacción entre esos factores?**

Debes implementar, ejecutar y analizar un experimento factorial controlado, aprovechando los modelos y el test estadístico robusto desarrollados anteriormente.

No te limites a diseñar el experimento. Debes realizarlo, verificar los resultados y actualizar el manuscrito.

No entrenes nuevos modelos.

---

# I. Contexto y resultados anteriores

Localiza e inspecciona los repositorios, worktrees, checkpoints y datos de:

- `distributed-coloring-learning`
- `ucgnn-paper-study`
- `ucgnn-local-palette-study`
- `ucgnn-global-distribution-study`
- `ucgnn-robust-distribution-tests`

Verifica sus ubicaciones efectivas antes de trabajar.

Los resultados más recientes muestran que un test omnibus robusto rechazó igualdad entre distribuciones de coloraciones completas con la siguiente frecuencia:

| Arquitectura | Rechazos Holm 5% |
|---|---:|
| E1 NORMALIZED | 52/265 |
| DeepSets | 107/265 |
| LocalPaletteTransformer | 181/265 |

Estos valores no son medidas de distancia entre distribuciones y dependen de la potencia del test.

El hallazgo más importante para este estudio es el siguiente.

Para un grafo regular de 512 vértices y grado 100, con el checkpoint `normalized-s41000`, se observó el evento:

\[
A=\{\text{se utilizan como máximo 100 colores}\}.
\]

Las probabilidades estimadas fueron:

\[
P_{\mathrm{Luby}}(A)=0.008,
\]

\[
P_{\mathrm{E1}}(A)=1.000.
\]

El análisis independiente proporcionó una cota inferior simultánea de 0.894 sobre la variación total de las distribuciones.

En cambio, para siete grafos pequeños, las cinco réplicas E1 satisficieron una tolerancia de equivalencia aproximada de TV ≤ 0.05, respaldada mediante cotas estadísticas.

Por tanto, tenemos evidencia de distribuciones muy próximas en algunos grafos pequeños y extraordinariamente diferentes en determinados grafos grandes.

**Todavía no sabemos qué variable explica la divergencia.**

El objetivo de esta continuación es investigarlo controladamente.

---

# II. Modelos

Utilizar exclusivamente:

1. Profesor Luby exacto con paletas locales.
2. E1 NORMALIZED.
3. DeepSets.
4. LocalPaletteTransformer.

Conservar las cinco réplicas independientes de entrenamiento de cada arquitectura.

Total:

- Un profesor.
- Cinco E1.
- Cinco DeepSets.
- Cinco Transformer.

Reutilizar exactamente los checkpoints históricos.

No reentrenar.

No mezclar réplicas.

No seleccionar pesos basándose en resultados favorables.

---

# III. Matriz factorial principal

Construir la siguiente matriz de experimentos:

| Vértices \(n\) | \(\Delta=3\) | \(\Delta=10\) | \(\Delta=50\) | \(\Delta=100\) |
|---|---|---|---|---|
| 128 | ✓ | ✓ | ✓ | ✓ |
| 256 | ✓ | ✓ | ✓ | ✓ |
| 512 | ✓ | ✓ | ✓ | ✓ |
| 1024 | ✓ | ✓ | ✓ | ✓ |

Son 16 combinaciones de tamaño y grado.

La matriz debe mantenerse equilibrada, en la medida en que el presupuesto computacional lo permita.

No eliminar celdas después de observar resultados estadísticos.

## Familia principal: grafos regulares

Para cada combinación \((n,\Delta)\), generar múltiples grafos simples, conectados y etiquetados, de grado exactamente \(\Delta\).

Utilizar un generador reproducible de grafos regulares aleatorios.

Comprobar:

- Número exacto de vértices.
- Grado de cada vértice.
- Simplicidad.
- Conectividad.
- Ausencia de duplicados.
- Semillas y hashes.

Cuando sea factible, utilizar al menos tres grafos independientes por celda.

La cantidad definitiva debe establecerse mediante un piloto de costes y congelarse antes de las evaluaciones confirmatorias.

No tratar esos tres grafos como una muestra representativa universal de todos los grafos regulares.

## Una distinción conceptual importante

En un grafo regular:

\[
d(v)=\Delta
\]

para todos los vértices.

Por tanto, las paletas locales tienen todas el mismo tamaño:

\[
K_v=\Delta+1.
\]

Aunque el algoritmo no recibe explícitamente el máximo global, la intervención sobre \(\Delta\) también modifica el tamaño de la paleta.

Esto significa que el experimento factorial separa los efectos observados de \(n\) y \(\Delta\), pero **no identifica causalmente por separado el efecto del grado y el del tamaño de la paleta**.

No atribuir exclusivamente al número de vecinos una diferencia que también puede depender del número de colores.

La interpretación debe reconocer esta limitación.

---

# IV. Experimento secundario: grafos irregulares

Repetir una selección equilibrada de la matriz con grafos irregulares.

Incluir distribuciones de grados heterogéneas, con:

- Muchos nodos de grado bajo.
- Nodos de grado intermedio.
- Algunos hubs.
- Grado máximo controlado.

En este régimen, cada nodo utiliza:

\[
C_v=\{0,\ldots,d(v)\}.
\]

Registrar no sólo el grado máximo, sino también:

- Grado medio.
- Varianza de grados.
- Distribución empírica de grados.
- Fracción de nodos con grado cercano a \(\Delta\).
- Tamaños efectivos de las paletas locales.

El objetivo es comprobar si las tendencias de los grafos regulares persisten cuando las paletas son heterogéneas.

No mezclar resultados de grafos regulares e irregulares sin estratificación.

---

# V. Qué distribución vamos a comparar

Para cada grafo fijo \(G\), el objeto estadístico será el vector completo de colores:

\[
X_G=(c_1,\ldots,c_n).
\]

Cada algoritmo induce una distribución:

\[
P_M^G
\]

sobre las coloraciones finales.

Compararemos:

\[
H_0:P_L^G=P_M^G
\]

frente a:

\[
H_1:P_L^G\ne P_M^G.
\]

Cada observación es una ejecución completa independiente.

Los colores de distintos vértices dentro de una misma ejecución no son observaciones independientes.

Utilizar exclusivamente las paletas locales.

Conservar las reglas originales de participación, propuesta y COMMIT.

No introducir un escudo de seguridad nuevo.

Registrar ejecuciones incompletas o inválidas como resultados separados.

Analizar tanto la tasa de éxito como la ley condicional de coloraciones completas cuando sea necesario.

---

# VI. Test estadístico principal

Reutilizar el procedimiento omnibus V2 validado en el estudio robusto anterior.

No modificar arbitrariamente sus componentes para aumentar el número de rechazos.

Auditar que se reproduce la versión congelada, con sus:

- Testigos locales.
- Testigos de pares y triples.
- Estadísticos estructurales.
- Kernels globales.
- Selección mediante una muestra separada.
- Evaluación sobre un banco independiente.
- Permutaciones conjuntas.
- Corrección por comparaciones múltiples.

Usar nuevamente el MMD antiguo como baseline metodológico secundario.

Verificar la resolución de los valores-p respecto del ajuste de Holm.

No utilizar ausencia de rechazo como evidencia de igualdad.

## Tamaño muestral

Tomar como punto de partida 500 coloraciones por generador y grafo, incluyendo la partición de selección y evaluación del omnibus.

Ejecutar un piloto de recursos y potencia antes de fijar definitivamente ese valor.

Si el coste resulta excesivo para completar toda la matriz, diseñar una batería confirmatoria reducida pero equilibrada, declarando explícitamente qué comparaciones conservan suficiente potencia.

Nunca elegir las celdas según si sus resultados preliminares fueron significativos.

---

# VII. No basta con comparar valores-p

Ésta es una instrucción central.

Queremos conocer **cómo cambia la diferencia entre las distribuciones**, no solamente si un test la detecta.

Un porcentaje mayor de rechazos puede deberse a:

- Mayor distancia entre distribuciones.
- Mayor potencia del test en ese grafo.
- Diferencias de dimensión.
- Diferencias de geometría del kernel.

Por tanto, debes acompañar los contrastes con tamaños de efecto interpretables.

## Medidas principales

### A. Test omnibus

Resultado del contraste global de dos muestras.

### B. Diferencias en eventos observables

Buscar y validar eventos sobre coloraciones completas cuyas probabilidades difieran.

### C. Cotas inferiores de variación total

Para cualquier evento A:

\[
\operatorname{TV}(P_L^G,P_M^G)
\ge
|P_L^G(A)-P_M^G(A)|.
\]

Construir cotas inferiores estadísticamente justificadas.

### D. Distancias estructurales

Comparar distribuciones de propiedades de coloraciones finales.

### E. MMD

Registrar el tamaño del efecto del kernel, indicando explícitamente que no es una distancia universalmente comparable entre grafos de tamaños distintos.

Si deseas comparar magnitudes entre tamaños, debes justificar una normalización que conserve un significado estadístico consistente.

No presentar las tasas de rechazo como estimaciones de TV.

---

# VIII. Estadísticas estructurales prioritarias

El resultado anterior sobre E1 indica que debemos estudiar especialmente el uso de colores.

Para cada coloración completa registrar:

\[
U(X)=|\{c_v:v\in V\}|,
\]

el número de colores distintos utilizados.

También registrar:

- Color máximo utilizado.
- Indicador de utilización del color \(\Delta\).
- Fracción de colores de la paleta que aparecen.
- Tamaño de la clase de color más grande.
- Entropía de las clases de color.
- Frecuencia de utilización de cada color.
- Color promedio.
- Reutilización de colores entre vértices no adyacentes.

Para los grafos regulares utilizar como evento especialmente importante:

\[
A_{n,\Delta}
=
\{U(X)\le\Delta\}.
\]

Este evento significa que el algoritmo consigue una coloración válida sin utilizar los \(\Delta+1\) colores disponibles.

En el caso histórico \(n=512,\Delta=100\), ese evento distinguió radicalmente E1 de Luby.

Quiero saber cómo cambia:

\[
P_M(A_{n,\Delta})
\]

cuando modificamos n manteniendo \(\Delta\) fijo, y cuando modificamos \(\Delta\) manteniendo n fijo.

Pero el análisis no debe limitarse a ese evento: usarlo como una medición preregistrada y conservar el contraste global completo.

---

# IX. Análisis principal: efecto de n

Para cada grado fijo:

\[
\Delta\in\{3,10,50,100\},
\]

comparar:

\[
n=128,256,512,1024.
\]

Preguntas:

1. ¿Aumenta la distinguibilidad de E1 frente a Luby con n?
2. ¿Cambia la probabilidad de utilizar el color máximo?
3. ¿Se amplifican sesgos pequeños de selección cuando aumenta el número de nodos?
4. ¿El comportamiento depende de la estructura del grafo?
5. ¿Las cinco réplicas de E1 presentan tendencias semejantes?

Realizar las mismas comparaciones para DeepSets y Transformer.

Tener presente que las distribuciones pertenecen a espacios diferentes cuando cambia n.

No comparar directamente dos distribuciones de coloraciones de grafos de tamaños distintos mediante TV como si pertenecieran al mismo espacio.

Comparar funciones, eventos y métricas que tengan una interpretación consistente entre tamaños.

---

# X. Análisis principal: efecto de Δ

Para cada tamaño fijo:

\[
n\in\{128,256,512,1024\},
\]

comparar:

\[
\Delta=3,10,50,100.
\]

Preguntas:

1. ¿Aumenta la discrepancia entre Luby y E1 al crecer \(\Delta\)?
2. ¿Cambia la utilización de colores altos?
3. ¿La diferencia aparece gradualmente o mediante cambios bruscos?
4. ¿DeepSets y Transformer siguen patrones semejantes?
5. ¿Qué ocurre con la variabilidad entre checkpoints?

No asumir monotonía.

Un gráfico con más rechazos a \(\Delta=100\) que a \(\Delta=3\) no basta para demostrar que la distancia distribucional aumenta con el grado.

---

# XI. Interacción entre n y Δ

Este es el objetivo central del diseño factorial.

Analizar conjuntamente ambos factores.

Una hipótesis exploratoria es que la discrepancia depende de la relación:

\[
\rho=\frac{n}{\Delta+1}.
\]

Esta razón compara la cantidad de nodos con el número de colores disponibles, pero no es por sí sola una medida universal de dificultad.

Investigar si algunas métricas se organizan mejor según:

- n.
- Δ.
- n/(Δ+1).
- Densidad del grafo.
- Grado medio.
- Alguna interacción entre estas variables.

No seleccionar retrospectivamente la variable que produce la curva más atractiva y presentarla como hipótesis confirmada.

Para el análisis factorial, construir modelos estadísticos adecuados a las variables de respuesta:

- Modelos binomiales para eventos observables.
- Efectos de grafo y checkpoint cuando corresponda.
- Contrastes de interacción.
- Intervalos de confianza.
- Análisis de sensibilidad.

Los cinco checkpoints de una arquitectura no deben tratarse como una muestra grande de entrenamientos.

Los grafos compartidos entre modelos requieren considerar el diseño emparejado.

Evitar pseudorreplicación.

---

# XII. Replicación del caso n=512, Δ=100

Este análisis tendrá prioridad especial.

Reexaminar el caso:

`regular-n512-d100 / normalized-s41000`

El resultado histórico mostró:

\[
P_L(U\le100)=0.008,
\]

\[
P_{\mathrm{E1}}(U\le100)=1.000.
\]

Queremos saber si es:

- Un fenómeno de ese checkpoint.
- Un fenómeno de ese grafo concreto.
- Un fenómeno frecuente de E1 a grado 100.
- Un comportamiento que aparece gradualmente al aumentar n o Δ.

Procedimiento:

1. Reproducir exactamente el caso histórico, sin modificar sus datos.
2. Generar grafos regulares nuevos de n=512 y grado 100.
3. Evaluar las cinco réplicas E1.
4. Comparar contra Luby.
5. Obtener nuevas estimaciones independientes del evento \(U\le100\).
6. Construir intervalos y cotas inferiores de TV.
7. Investigar cuáles colores son sistemáticamente utilizados o evitados.

Distinguir claramente reproducción histórica de replicación independiente.

No utilizar el resultado histórico como dato confirmatorio nuevo.

---

# XIII. Explicación mecanística

Queremos comprender por qué aparecen diferencias distribucionales, no sólo detectarlas.

Investigar, en estados congelados:

- Distribución de candidatos por posición del color.
- Probabilidad de proponer colores altos.
- Cambios de distribución conforme los vecinos fijan colores.
- Probabilidad de no utilizar los últimos colores disponibles.
- Relación entre propuestas y colores finalmente confirmados.
- Posibles sesgos producidos por prefijos normalizados en E1.
- Diferencias entre E1, DeepSets y Transformer.

Separar:

**Distribución de candidatos**

\[
q_M(c\mid S)
\]

de:

**Distribución de coloraciones completas**

\[
P_M^G(X).
\]

Una diferencia en propuestas no explica automáticamente una diferencia global.

Intentar establecer una cadena de evidencia entre:

1. Sesgo de propuestas.
2. Probabilidades de confirmación.
3. Evolución de clases de color.
4. Estadísticas finales.
5. Distinguibilidad global.

Si se hacen intervenciones sobre las políticas, conservarlas como análisis exploratorios separados de los contrastes confirmatorios originales.

No atribuir causalidad basándose únicamente en correlaciones.

---

# XIV. Controles de calidad estadística

Antes de ejecutar la batería confirmatoria:

- Verificar checkpoints y hashes.
- Verificar paletas locales.
- Verificar generación de grafos.
- Verificar independencia de semillas.
- Verificar ejecución sin estado residual.
- Verificar validez de coloraciones.
- Verificar la reproducción del omnibus V2.
- Verificar la calibración del test.
- Verificar resolución de valores-p y Holm.
- Fijar reglas de multiplicidad.
- Fijar tamaños muestrales.
- Fijar métricas y eventos principales.
- Fijar el tratamiento de cutoffs.
- Congelar el protocolo.

Crear:

`docs/FACTORIAL_SIZE_DEGREE_PROTOCOL.md`

Guardar el commit de preregistración antes de ejecutar las evaluaciones confirmatorias.

No ajustar hipótesis ni umbrales después de observar los resultados.

---

# XV. Recursos

Crear un worktree nuevo, por ejemplo:

`/workspaces/AlgoLearning/ucgnn-size-degree-study`

Rama:

`research/ucgnn-size-degree`

No modificar checkpoints históricos.

No hacer push.

No detener procesos ajenos.

Presupuesto inicial máximo:

**12 horas GPU-proceso acumuladas.**

Utilizar CPU para análisis estadísticos cuando sea apropiado.

Implementar una cola persistente, reanudable y auditable.

Ejecutar primero pilotos para estimar recursos.

Si el presupuesto no permite cubrir las 16 celdas con la precisión prevista, definir una reducción equilibrada y justificada antes de observar resultados confirmatorios.

Preservar todos los resultados y registrar explícitamente las celdas incompletas.

---

# XVI. Figuras obligatorias

Quiero que el informe incluya visualizaciones que permitan interpretar el efecto de ambas variables.

## Figura 1. Matriz factorial de significación

Una matriz por arquitectura:

- Filas: n.
- Columnas: Δ.
- Celdas: resultados del test global.

Identificar la cantidad de grafos y checkpoints por celda.

No confundir porcentaje de rechazos con tamaño del efecto.

## Figura 2. Matriz factorial de discrepancia estructural

Para cada arquitectura, representar:

\[
P_M(U\le\Delta)-P_L(U\le\Delta).
\]

Incluir incertidumbre.

Ésta debería ser una de las figuras centrales.

## Figura 3. Efecto de n con Δ fijo

Curvas para cada grado.

## Figura 4. Efecto de Δ con n fijo

Curvas para cada tamaño.

## Figura 5. Utilización del color máximo

Comparar las probabilidades de utilizar el color Δ.

## Figura 6. Número de colores utilizados

Distribuciones por modelo y grafo.

## Figura 7. Variabilidad entre checkpoints

Mostrar las cinco réplicas sin ocultarlas detrás de un promedio.

## Figura 8. Replicación n=512, Δ=100

Análisis detallado del caso más importante.

## Figura 9. Relación entre distribución de candidatos y resultados globales

Separar datos descriptivos de evidencia causal.

Generar todas las figuras automáticamente a partir de los datos archivados.

---

# XVII. Actualización del manuscrito

Actualizar:

`paper/ucgnn/main.tex`

Agregar una sección provisional:

**How Graph Size and Degree Affect Distributional Generalization**

Desarrollar:

1. Motivación.
2. Diseño factorial.
3. Definición de distribuciones.
4. Test estadístico.
5. Métricas estructurales.
6. Resultados por tamaño.
7. Resultados por grado.
8. Interacciones.
9. Estudio detallado de E1.
10. Interpretación mecanística.
11. Limitaciones.

No concluir que E1 se aleja de Luby por causa de Δ si no podemos distinguir el efecto de las paletas.

No concluir que n produce divergencia solamente porque el test tiene más potencia sobre grafos grandes.

Distinguir resultados observados, hipótesis, demostraciones y conclusiones causales.

Compilar el PDF final.

---

# XVIII. Informe final

El informe debe responder explícitamente:

**P1.** ¿La diferencia entre E1 y Luby aumenta cuando crece el número de vértices manteniendo fijo el grado?

**P2.** ¿Aumenta cuando crece el grado manteniendo fijo el tamaño?

**P3.** ¿Existe una interacción significativa entre n y Δ?

**P4.** ¿El fenómeno se reproduce en las cinco réplicas de E1?

**P5.** ¿Se reproduce en grafos regulares nuevos?

**P6.** ¿Qué ocurre en DeepSets y Transformer?

**P7.** ¿Qué estadística de las coloraciones finales explica mejor las diferencias?

**P8.** ¿El caso n=512, Δ=100 es excepcional o representativo de una tendencia más amplia?

**P9.** ¿Qué relación existe entre sesgos locales de candidatos y distribuciones globales?

**P10.** ¿Qué podemos afirmar rigurosamente sobre la pérdida de equivalencia distribucional fuera del dominio de entrenamiento?

Quiero también una sección final llamada:

**What We Can Prove, What We Observe, and What Remains Open**

Separar cuidadosamente las tres categorías.

---

# XIX. Instrucción final

Ejecuta esta investigación de forma autónoma.

Comienza inspeccionando el código y los resultados históricos.

Crea el worktree independiente.

Implementa los generadores de grafos y verifica la matriz factorial.

Reutiliza el test omnibus robusto congelado.

Ejecuta los pilotos, establece los tamaños muestrales y congela el protocolo.

Completa las evaluaciones confirmatorias dentro del presupuesto.

Analiza conjuntamente el efecto de n, Δ y la estructura del grafo.

Investiga especialmente la utilización de colores altos y la replicación del caso n=512, Δ=100.

Produce las tablas, figuras, teoría, documentación y análisis de incertidumbre.

Actualiza el manuscrito y compílalo.

No entrenes nuevos modelos.

No hagas push.

**La pregunta que queremos resolver es por qué una red neuronal que aprendió a colorear a partir de Luby puede seguir produciendo coloraciones correctas mientras su distribución de soluciones completas se aleja de la del profesor, y qué papel desempeñan el tamaño del grafo y su grado en esa divergencia.**