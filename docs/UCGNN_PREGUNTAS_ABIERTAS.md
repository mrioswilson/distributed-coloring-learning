# UCGNN: qué entendemos y qué falta explicar

Resumen ejecutivo de la auditoría del 20 de octubre de 2026.
[Documento principal](UCGNN_MODELOS_EXPLICADOS.md) ·
[Datos y figuras](../results/ucgnn/2026-10-20-mechanisms/selection-summary.csv).

## Lo que ya tiene una explicación concreta

**La misma red procesa otras paletas porque comparte los pesos por nodo y
color.** No tiene 201 cabezas ni embeddings nuevos al recibir 201 colores.
Se repiten las mismas MLPs en más ranuras. Sus 16.035 parámetros son constantes;
el trabajo, la memoria y el mensaje implementado crecen con la paleta.

**Una parte importante de la extrapolación está estructurada por el diseño.**
Las entradas binarias y el máximo convierten vecindarios grandes en conjuntos
de pocos tipos de mensaje. El prefijo y la normalización son operaciones
deterministas incorporadas. Los pesos aprenden disponibilidad, puntuación,
participación y seguridad sobre esa representación.

**NORMALIZED normaliza masa aprendida, no divide por el número público K.**
Divide prefijos de α por `max(sum(α),0.5)`. Con 201 colores libres, la suma de
E1-0 es aproximadamente 95,895, aunque el prefijo normalizado sea casi el
exacto. La disponibilidad se reconoce relativamente; α no es un conteo ni
necesariamente una probabilidad calibrada de estar libre.

**Colorear y reproducir Luby son propiedades diferentes.** En K=k=201,
E1-0 coincide sólo 5/1.001 veces con el candidato del profesor, pero propone
199 colores válidos y mantiene unos 196 colores efectivos. Su probabilidad
de colisión por pares en una malla densa es ≈1,025 veces la uniforme ideal.
El desfase de unas pocas posiciones destruye coincidencia exacta sin destruir
esa diversidad. El estado aparece realmente: k=201 representa el 39,35 % de
las decisiones sin colorear de E1-0 en los casos originales de Δ=200.

**A1 pierde progreso porque concentra candidatos.** En el estado inicial con
101 colores libres, propone sólo los colores 40–43, con aproximadamente una
cuarta parte de la masa en cada uno. En los cuatro negativos reconstruidos,
todos sus participantes chocan en la primera ronda. E1, sobre exactamente los
mismos estados y uniformes, produce 45–233 propuestas seguras. Los negativos
de A1 son falta de terminación al corte, sin commits inseguros; no demuestran
un bloqueo permanente con probabilidad uno.

**A1 no aísla causalmente sólo la normalización.** Añade también la masa total
como entrada explícita del scorer. La matriz inicial pasa de `[64,36]` a
`[64,37]`: ésos son los 64 parámetros adicionales. El distinto consumo del
generador aleatorio cambia también la inicialización de módulos posteriores.
El fallo funcional está localizado; la atribución causal necesita otro control.

**C1 tiene una regla local incompleta identificable.** E1-0 y C1 comparten
exactamente los pesos de candidato y participación. En los 64 conjuntos de
tipos vecinos del canal COMMIT, E1 reconoce ambos conflictos; C1 coincide con
rechazar colisiones simultáneas, pero omite el color ya fijado. Acepta los 16
conjuntos que contienen ese conflicto sin una colisión simultánea. Una política
que normalmente evita proponer colores fijados puede ocultar esta omisión en
rollouts correctos. La rama contrafactual fuerza esos ejemplos y sus etiquetas.

## Qué sigue siendo una hipótesis o una propiedad no demostrada

- El scorer de E1 parece seleccionar por un **cuantil monótono deformado**.
  Se extrajo una preferencia continua de sus ReLU sin ajustar pesos. La regla
  aproximada `floor(K*g(r_c))` reproduce 907/1.001 candidatos en K=201 y
  888/1.001 en K=501, con errores de como máximo una posición. Es útil como
  descripción restringida del estado libre; queda refutada como identidad
  exacta y no se ha establecido para todos los contextos de ocupación.
- No conocemos una condición general de progreso para la política aprendida.
  Ser seguro al confirmar no garantiza encontrar propuestas confirmables.
- La reducción de COMMIT a 64 conjuntos es una caracterización fuerte de ese
  canal, pero no certifica la política completa, su terminación ni todos los
  efectos de aritmética flotante.
- Los éxitos hasta Δ=500 son evidencia finita. En ese grado extremo sólo hay
  cinco sondas de familias distintas y una réplica. No justifican una afirmación
  de corrección para grado arbitrario ni uniformidad en cualquier estado.
- Los mensajes PyTorch contienen 32K floats por arista dirigida y etapa. Es
  posible describir una codificación discreta suficiente de O(log K) bits,
  reconstruyendo los mensajes en el receptor, pero no está implementada ni
  medida y no resuelve el coste de cálculo local.

## Cinco investigaciones propuestas, por importancia científica

Estas propuestas no se ejecutaron como nuevas baterías ni alteraron el protocolo
de resultados existente. Deben fijarse antes de observar los nuevos casos.

1. **Predecir la selección en paletas y contextos retenidos.** Describir el
   scorer usando sólo un subconjunto de paletas y tipos locales; retener, por
   ejemplo, K=17/63/251/997 y patrones libres bajos, altos y dispersos. Predecir
   candidatos, monotonía, masa prohibida y probabilidades por color sin ajustar
   después g. Esto distingue una deformación de cuantil transferible de una
   aproximación que cambia arbitrariamente con el contexto. Un error de rango
   que crezca sin control o una concentración imprevista falsaría la versión
   fuerte de la hipótesis.

2. **Medir y explicar progreso desde estados parciales difíciles.** Fijar
   estados equivalentes con k pequeño, solapamientos entre paletas libres y
   subgrafos de nodos pendientes diversos. Medir propuestas seguras por ronda,
   tiempo hasta el siguiente COMMIT y distribución de tiempos de terminación,
   además de seguridad. Contrastar la ley conjunta real con las predicciones
   derivadas de q; no sustituir correlaciones vecinales por independencia sin
   comprobarla. Es la prueba más directa del mecanismo que sostiene o rompe
   la terminación.

3. **Certificar los canales locales finitos y sus márgenes numéricos.** Extender
   la reducción de tipos de disponibilidad y COMMIT a todas las condiciones
   propias legales, márgenes y errores de kernels relevantes. Combinarla con
   una caracterización del umbral de participación. El objetivo es separar
   propiedades locales verificables de la condición de progreso pendiente,
   evitando inferir una prueba global de los éxitos de grafos.

4. **Aislar causalmente el papel de la normalización.** Diseñar un control con
   igual dimensión de entrada, módulos, inicialización por tensor, datos y
   número de actualizaciones; variar sólo los cocientes de prefijo. Hacer un
   piloto pequeño preregistrado y conservar negativos. Este control sería una
   investigación posterior: A1 actual no resuelve por sí sola esa causalidad.

5. **Verificar una implementación de mensajes comprimidos.** Conservar la
   función y los pesos; transmitir color, candidato y participación y
   reconstruir los mensajes neuronales en destino. Exigir igualdad de
   decisiones y trazados antes de medir bytes, latencia y cómputo local en
   una ejecución distribuida. Esto convertiría una posibilidad de codificación
   en evidencia de eficiencia de comunicación, sin confundirla con una cota
   todavía inexistente sobre las rondas totales.

La prioridad científica es caracterizar **selección y progreso** con garantías
limitadas pero precisas. Otra cifra global de éxito, por sí sola, aportaría
menos comprensión que distinguir estas propiedades y verificar sus mecanismos.
