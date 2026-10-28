# Decisión de cierre científico y editorial

**READY WITH MINOR REVISIONS.** El manuscrito de afirmaciones delimitadas está
preparado para revisión científica externa. No identificamos un obstáculo
científico sustantivo pendiente dentro de ese alcance. Aún no es el archivo
formateado/anónimo listo para envío. Las revisiones menores restantes son
autoría/atribución final, formato del venue y distribución verificable de los
artefactos al destinatario externo, sin publicar por iniciativa propia.

La decisión se basa en recálculos de datos, hashes, replays, pruebas y
revisiones, no en que el porcentaje agregado sea alto. El PDF compilado tiene
18 páginas, nueve de texto principal y el resto apéndices/referencias; el
recibo final establece el número y hash exactos. Compila sin referencias
indefinidas, glifos ausentes ni cajas desbordadas. Los recibos de revisión
visual y recuperación se enlazan en `results/final-closure/`.

## P1. ¿Qué se demostró experimentalmente con los siete profesores?

Cobertura funcional nativa de 21 combinaciones: siete variantes explícitas
por tres arquitecturas, cinco checkpoints cada una y 1.000/1.000 ejecuciones
por checkpoint. Son 105.000 observaciones de los pesos seleccionados.
Doce paquetes históricos pasivos y nueve nuevos CF usan protocolos distintos.
Los controles matched también cierran los nueve. No es corrección universal,
garantía de terminación ni probabilidad de éxito de un nuevo entrenamiento.
Fuente verificable: `native-certifications.json` y registro C01.

## P2. ¿Qué aprendieron E1, DeepSets y Transformer?

Módulos locales de decisión con comunicación: participación, propuestas,
representaciones de disponibilidad y COMMIT; en BE selección de punto/opción,
en GM marcas/reparación. DeepSets y Transformer cambian sobre todo el scorer
de paleta, manteniendo los intercambios aprendidos. No aprendieron aritmética
finita, construcción de slots/colas, orden global o decoder: se suministran.
La tarea aprendida depende de esa interfaz. Las mismas soluciones válidas no
demuestran mismas decisiones, transiciones, tiempos o leyes completas.

## P3. ¿Qué aporta y qué no aporta CF?

Mejor fidelidad al oráculo de continuación tipado declarado en estos bancos.
En BE/E1 a4.000updates: passive22,6404%, matched9,2686%, CF99,9932%.
Pero siempre-rechazar da97,0465% por el desequilibrio; BA por semilla es
60,1413%,53,2536%,99,9965% respectivamente. CF acierta100% entre elegibles
activas, frente67,8117% passive/matched. La mejora refleja rechazo de opciones
negativas, no una nueva capacidad de responder positivos (todosFN0).
No demuestra necesidad, mejora funcional universal ni eficiencia a igual
etiquetas: BEusa3.000/3.000/6.776. Fuentes C02–C04 y revisión causal.

## P4. ¿Cuál es la evidencia causal más sólida sobre entrenamiento?

El control independiente E1 de optimización a banco/prefijo/inicialización
fijos y nuevas baterías:500→1.500updates. BE pasa de72,90,84,92,42
éxitos/1.000 a1.000 en cinco semillas. Luby/JP ya pasan en ambos puntos.
Aísla optimización adicional en esos bancos; no identifica efecto CF.
Matched/CF comparte muchas condiciones pero cambia cantidad/contextos/pérdida
de etiquetas. Su diferencia es de procedimientos completos. FuenteC05.

## P5. ¿Qué papel tienen IDs e interfaces?

Una interfaz anónima no puede reproducir dos acciones ID-dependientes con
la misma observación. Ésa es una obstrucción elemental de información.
JP con comparaciones relativas logra60/60 en las cinco semillas a igual
dimensión/presupuesto del control; magnitudes numéricas fallan bajo ese
presupuesto. Es sensibilidad de representación, no imposibilidad de aprender
IDs numéricos. BE necesita valores/predicados algebraicos; GM/Greedy usan
calendarios suministrados. Ninguno debe venderse como LOCAL anónimo equivalente.

## P6. ¿Hasta dónde generalizan pesos congelados?

Confirmación de seis profesores (sinBE): globaln64/Δ3,4,10;
localn64/Δ4; globaln256/Δ4. Principales449.791/450.000; dieciséis de
dieciocho combinaciones pasan todas las condiciones en cinco semillas.
Incluyendo controles:808.413/810.000, quince celdas fallidas. Es contabilidad,
no tasa promedio de un algoritmo. Grado200 y tamaño4.096 son pilotos; este
último sólo regular y cuatro profesores. No se confirma el producto de máximos.
Δ cambia K y las preferencias CT. Fuentes C07–C10.

## P7. ¿Qué fallos son más informativos y qué los explica?

Johansson–Transformer pierde diversidad de propuesta en una clique y llega
al cutoff con COMMIT correcto. GM–Transformer CF omite reparación programada
tras proponer ocupado y recibir rechazo correcto; controles aceptan escritura
prohibida. CT matched tiene confirmaciones erróneas incluso aΔ3; la sonda de
alto grado revela sensibilidad de escala con grado local/paleta constantes.
Se reprodujeron quince primeros fallos, conservando batches completos.
No se identifican universalmente capacidad, optimización o causa numérica.

## P8. ¿Qué resultados matemáticos están demostrados?

Lema de disponibilidad; seguridad/progreso condicional de reglas exactas;
JP≤n, Greedyn y coste GM con scheduler; BE especializadoΔ≤3; obstrucción
por observación; progreso con propuestas difusas/submasa; MAX64tipos bajo
interfaz/arimética explícitas; limitación MEAN y construcción SUM; simetría
de cliqueK=n y desigualdad de couplingTV. No son certificados generales de
los21pesos:45núcleos canónicos actuales no pasan el certificado suficiente;
cinco E1 históricos distintos sí tienen certificados condicionales.
Las garantías de profesores no se transfieren automáticamente a redes.

## P9. ¿Qué es novedoso frente a la literatura?

La evidencia integrada sobre estas siete variantes/interfaces, los grandes
contrastes de continuación con contabilidad/desequilibrio auditados y los
mecanismos reproducibles de transferencia. No lo son GNNcoloring,MAX,NAR,
imitación de trazas, supervisión interactiva/intervenciones ni distinción
genérica tarea/traza. Se contrastaron Hint-ReLIC2023, no-hint2023 y trabajos
de coloración/extrapolación2026 además de fuentes clásicas. No se reclama
prioridad ni SOTA. Matriz y fuentes en `NOVELTY_AND_RELATED_WORK.md`.

## P10. ¿Qué objeciones importantes quedan para un revisor externo?

Protocolos heterogéneos; semántica off-policy elegida, no identificada sólo
por trazas; etiquetas CF adicionales; development reutilizado; más99% de
ocurrencias de tipos BE ya vistas; heads compartidas; cinco semillas;
confusión Δ/K; auxiliares suministrados; piloto escaso y numérica de batches;
novedad incremental y necesidad de juicio experto externo. Todas se declaran.
La revisión adversarial corrigió errores concretos, añadió baselines y
retiró inferencias fuertes. No se afirma que desaparezcan las limitaciones.

## P11. ¿Qué requiere experimentos y qué puede ser trabajo futuro?

No hace falta otro entrenamiento para las afirmaciones conservadas.
Recuperar causalidad CF a igual etiquetas o generalización a familias de
intervención retenidas sí exige un nuevo diseño prospectivo independiente.
ConfirmarΔ200/n4096, separar grado/paleta, BEgeneral, asynchrony y capacidad
uniforme pueden quedar fuera. No se inicia una campaña para recuperar una
afirmación que el manuscrito ha retirado. Diseño mínimo en `FUTURE_WORK.md`.

## P12. ¿Está científicamente preparado para revisión externa?

Sí, para el alcance escrito. Dos recuentos independientes reconcilian810k;
los105pesos nativos se autentican y sus105kcolores finales se verifican;
quince replays coinciden;480regresiones pasan; todas las diferencias BE y
confusiones se reconstruyen con consultas originales; revisiones de pruebas
y bibliografía se incorporaron; el PDF compila y fue inspeccionado.
Es revisión interna automatizada, no aprobación humana de coautores ni
arbitraje externo ya realizado.

## P13. ¿Un artículo integrado o dos?

Uno integrado centrado en información, continuación y transferencia de
reglas, con resultados históricos de distribuciones en suplemento. Dividir
ahora separaría evidencia complementaria y produciría dos relatos más débiles.
Un futuro artículo de leyes/instrumentos estadísticos debe justificar su
contribución propia y evitar duplicación. Es recomendación editorial.

## P14. ¿Qué venue encaja?

Primera opción TMLR: [criterios oficiales](https://jmlr.org/tmlr/acceptance-criteria.html)
centrados en evidencia/afirmaciones y claridad, compatibles con mecanismos
y resultados negativos. [Guía de autores](https://jmlr.org/tmlr/author-guide.html)
para estilo/anónimo. LoG es alternativa temática en próxima convocatoria;
[LoG2026](https://logconference.org/cfp/) ya cerró. No se predice aceptación.
PODC/DISC es encaje menos natural para esta contribución principalmente
empírica, sin nuevo teorema general de complejidad distribuida.

## P15. ¿Qué falta exactamente para enviar?

1. Acuerdo de coautores sobre nombres/orden, atribución histórica y versión
   científica; no lo puede suplir el editor automatizado.
2. Aplicar plantilla del venue, anonimizar la versión de envío y ordenar
   suplemento/enlaces según sus límites y política de uso de herramientas.
3. Entregar al destinatario el paquete local con checksums y permisos/enlace
   adecuado; autorizar por separado cualquier push o publicación/release.
4. Releer la prueba final tras cambios de formato y comprobar su nuevo hash.

Estos puntos no requieren nuevos modelos. Si se reintrodujera causalidad CF
igualada, corrección aprendida universal o certificación de máximos piloto,
la decisión científica cambiaría: esas afirmaciones siguen sin demostrarse.

## Criterios de cierre y evidencia

| Criterios solicitados | Estado verificable |
|---|---|
| 1–2:21certificaciones/profesores | recibo nativo105políticas, inventarios y auditoría matemática |
| 3–4:810k/quincefallos | auditoría primaria y revisor3, quince replays coincidentes |
| 5:contrafactual exacto | recursos, overlap, baselines y confusiones; afirmación fuerte retirada |
| 6:pruebas | revisión1, dos correcciones incorporadas, hipótesis explícitas |
| 7–9:novedad/historia/negativos | fuentes primarias, cohortes distintas, fallos preservados |
| 10–11:narrativa/reproducción | diez secciones, apéndices, registro de14afirmaciones/225hashes y paquete relativo |
| 12:PDF | compilación, diagnóstico y revisión visual con hash en recibos |
| 13:adversarial | cuatro informes y addenda; objeciones sustantivas respondidas mediante corrección/delimitación |
| 14:decisión editorial | este documento; revisiones menores de envío enumeradas, sin campaña pendiente |

La frontera del resultado está cerrada: fallos y límites permanecen en el
artículo. La decisión **READY WITH MINOR REVISIONS** no convierte desarrollo
en confirmación, un piloto en certificación ni revisión interna en arbitraje.
