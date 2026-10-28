# Revisión adversarial interna 4 — novedad y publicación

Fecha de búsqueda y revisión: 26 de octubre de 2026. Revisión realizada por
un agente distinto del editor y de los otros tres revisores. Es revisión
interna de ejecución independiente, no arbitraje humano externo ni una
replicación independiente de entrenamientos. No se entrenaron modelos ni se
modificó el manuscrito durante esta revisión.

## Dictamen

**READY WITH MINOR REVISIONS para revisión externa de las afirmaciones
delimitadas**, condicionado a corregir el antecedente directo omitido y la
documentación de reproducción. No considero listo el fichero actual para
enviarlo a una revista: aún requiere formato, autoría/anónimo, responsabilidad
de coautores y empaquetado verificable. Es una distinción editorial; no una
solicitud de nuevos entrenamientos.

El proyecto puede resultar interesante como estudio mecanístico y metodológico
de aprendizaje de reglas locales. La suma 21/21 o 810.000 no establece por sí
sola una contribución novedosa. La parte más defendible es la conjunción de
interfaces explícitas, desacoplamiento entre ejecución nativa y continuaciones
declaradas, y mecanismos reproducibles de fallo bajo cambios de paleta y
preferencias. La comparación a igual total de etiquetas y la generalización
amplia a familias de intervención retenidas siguen sin estar identificadas.
El borrador ya limita esas afirmaciones; no es necesario rehabilitarlas.

## Verificaciones

Se leyó `paper/ucgnn/main.tex`, las referencias nuevas, los tres informes
independientes, el recibo de compilación y las instrucciones de reproducción.
La versión de PDF revisada por metadatos compiló en 17 páginas, sin referencias
indefinidas, glifos ausentes, cajas desbordadas ni caracteres fuera de página.
Esto no sustituye la revisión visual que realiza el editor sobre el PDF final.

La búsqueda focalizada incluyó combinaciones de «learning distributed
algorithms», «graph coloring», «neural algorithmic reasoning», «counterfactual»,
«interventions» y «causal regularisation». Se usaron fuentes primarias para
los contrastes científicos siguientes; no se interpreta una búsqueda negativa
como prueba de prioridad mundial.

## Objeciones y remedios

### N-01 — Falta el antecedente directo Hint-ReLIC

**Categoría: evidencia insuficiente en el contraste bibliográfico. Prioridad
alta, corregible con lectura/citación, sin nuevos experimentos.**

El borrador inicial relaciona continuaciones con DAgger y causal confusion,
pero omite un trabajo específicamente sobre intervenciones en neural
algorithmic reasoning: [Bevilacqua et al., ICML 2023, *Neural Algorithmic
Reasoning with Causal Regularisation*](https://proceedings.mlr.press/v202/bevilacqua23a.html).
Hint-ReLIC construye augmentaciones que preservan el siguiente paso de una
traza y regulariza invariancia de su predicción, con análisis y evaluación
fuera de distribución. No es el mismo procedimiento, pero es un antecedente
más próximo que la imitación interactiva genérica.

**Corrección concreta:** añadirlo al párrafo de estado del arte y a la matriz.
Explicar la diferencia operacional: aquí se adquieren etiquetas explícitas
COMMIT para mensajes/consultas alternativos en estados admisibles; allí se
preserva el target de siguiente paso bajo augmentaciones y se exige
invariancia. Evitar reclamar como novedad intervenir datos algorítmicos,
usar una formulación causal o separar precisión final de seguimiento de
trazas. La contribución presente sigue siendo el resultado concreto y su
auditoría, no esas ideas generales.

### N-02 — Falta un contraste cercano sobre supervisión de trazas

**Categoría: problema editorial y delimitación de novedad.**

[Rodionov y Prokhorenkova, NeurIPS 2023, *Neural Algorithmic Reasoning Without
Intermediate Supervision*](https://proceedings.neurips.cc/paper_files/paper/2023/file/a2370db7c99791ad5d9f3ef48ad6d464-Paper-Conference.pdf)
compara aprendizaje desde pares entrada–salida con supervisión intermedia y
regularización de cómputos, demostrando que imitar una trayectoria concreta
no es requisito general para lograr resultados finales competitivos.

**Remedio:** incluir una comparación breve. Este antecedente refuerza la
decisión de no presentar «funcionalidad no implica fidelidad» como descubrimiento
conceptual general. No exige reproducir sus baselines CLRS porque el artículo
actual no afirma superar sus resultados en su benchmark.

### N-03 — Un oracle fuera de trayectoria no identifica un algoritmo único

**Categoría: limitación conceptual reconocida; aclaración necesaria.**

Una traza nativa puede dejar indeterminadas las respuestas a mensajes que
el profesor nunca generaría. El banco actual elige una extensión tipada y
específica. La superioridad CF mide fidelidad a esa extensión declarada,
no necesariamente identificación del único comportamiento contrafactual
implícito en «el algoritmo original». Los errores BE en raíces KW ineligibles
ilustran ese punto: una decisión COMMIT redundante durante ejecución nativa
se convierte en objeto explícito de consulta.

**Remedio:** preservar «declared continuation oracle» y los contratos de
admisibilidad, con la estratificación y baselines del revisor ML. La narrativa
de identificación completa del algoritmo requeriría nuevas hipótesis y queda
fuera del alcance. Esta limitación no invalida comparar la regla de continuación
bien especificada.

### N-04 — El tamaño de la campaña no es novedad ni superioridad

**Categoría: limitación reconocida y presentación.**

21/21 combina profesores y protocolos heterogéneos, suministros algebraicos y
calendarios distintos. 810.000 reutiliza grafos/cintas entre políticas.
Ninguno es una comparación homogénea de rendimiento ni prueba de superioridad
frente a solvers clásicos, modelos CLRS o arquitecturas modernas. Las
limitaciones del borrador son correctas y deben permanecer en abstract,
contribuciones y tablas, no sólo en notas de apéndice.

**Remedio:** mantener la cobertura como evidencia de ejecutabilidad del
sistema especificado, y centrar la aportación científica en los mecanismos:
COMMIT subdeterminado por estados nativos, pérdida de diversidad de propuestas,
reparación programada omitida y sensibilidad a escala/paleta. No añadir una
campaña SOTA para justificar una afirmación que el artículo no necesita.

### N-05 — Documentación de build obsoleta puede destruir el cierre

**Categoría: error demostrado de documentación/reproducibilidad. Prioridad
alta, resoluble ahora.**

El `paper/ucgnn/README.md` leído afirma que `main.tex` se genera desde Markdown,
que el PDF actual tiene 36 páginas y recomienda generadores históricos como
`build-multicf-paper.py` y `build-integrated-paper.py`. El nuevo `main.tex`
es autoritativo y el recibo leído tiene 17 páginas. Esos comandos pueden
sobrescribir el nuevo cierre con otra versión del manuscrito.

**Remedio:** actualizar README y entradas principales al nuevo flujo
`build-closure-assets.py` / `build-closure-paper.py`. Marcar los generadores
anteriores como reproducción histórica en una copia separada. Conservar
el archivo previo y hashes para identificar inequívocamente qué PDF se está
reconstruyendo. Un hash del PDF sólo autentica el fichero concreto, no corrige
instrucciones equivocadas para generarlo.

### N-06 — Preservación local no equivale a paquete reproducible de envío

**Categoría: limitación reconocida; tarea editorial de artefactos.**

El manuscrito admite correctamente que un checkout Git sin los grandes
artefactos no reproduce el estudio. La revisión externa necesita poder
resolver pesos/filas/manifiestos sin depender de rutas privadas del worktree
del editor. Los hashes y recibos identifican los artefactos, pero no prueban
por sí solos acceso desde un entorno limpio ni disponibilidad indefinida.

**Remedio:** aportar manifiesto con rutas relativas, comandos de recuperación,
SHA-256 de paquetes y una prueba acotada de recuperación/reproducción desde
una ubicación nueva. No es necesario repetir 810.000 inferencias ni entrenar.
La entrega puede quedar local para revisión de coautores, pues el usuario no
ha autorizado una publicación nueva. Distinguir «reproducible localmente» de
«archivo público estable disponible para revisores».

### N-07 — La novedad de las proposiciones debe seguir limitada

**Categoría: limitación reconocida.**

Indistinguibilidad de observaciones, seguridad frente a progreso, órbitas de
simetría, desigualdad de coupling y límites de permutaciones son argumentos
con antecedentes claros. MAX idempotente permite la reducción de tipos,
pero su mera propiedad algebraica no es un nuevo principio de GNNs. El
borrador lo presenta prudentemente como certificado específico y condicional.

**Remedio:** conservar ese lenguaje. El incremento defendible es comprobar
su aplicación a cabezas/pesos concretos y mostrar qué condiciones faltan en
otros, no un teorema general de extrapolación. El reciente
[Wittig et al., versión 2 de agosto de 2026](https://arxiv.org/abs/2602.13106v2)
es un contraste relevante para separar garantías de aprendizaje de evidencia
finita de ejecución; no hace falta demostrar sus hipótesis para los modelos
actuales cuando no se reclama su tipo de garantía.

### N-08 — El artículo debe decir qué se aprendió antes de enumerar reservas

**Categoría: problema editorial.**

La cautela es correcta, pero una introducción dominada por «no demostramos»
puede ocultar la enseñanza positiva para un lector. El bloque integrado ya
es mucho más compacto que el inventario anterior. Todavía conviene formular
una frase afirmativa por contribución con resultado y alcance: las trazas
nativas dejan gates locales sin identificar; ampliar paletas altera propuestas
incluso a grado local fijo; confirmar seguridad de COMMIT no certifica progreso.

**Remedio:** mantener esas tres ideas como eje y usar las limitaciones para
delimitarlas, sin añadir resultados al texto principal por haber sido medidos.
Las tablas completas y las distribuciones históricas pertenecen al suplemento.

## Matriz crítica abreviada

| Contribución candidata | Conocido antes | Incremento concreto permitido | Objeción especializada |
|---|---|---|---|
| Imitación local de siete reglas | NAR, trazas CLRS, modelos de coloreado | cobertura auditada de estas siete variantes e interfaces | calendario/aritmética suministrados simplifican el problema |
| Continuaciones CF | DAgger, causal confusion y Hint-ReLIC | contratos específicos y grandes diferencias de reglas bajo procedimientos emparejados | más etiquetas, plantilla compartida, desarrollo reutilizado |
| Función frente a trayectoria | NAR sin hints y trabajo de imitación | evidencia concreta con fidelidad nativa perfecta y diferente continuación | no es una separación conceptual nueva en general |
| Transferencia congelada | extrapolación/alineamiento y coloreado transferible previos | condiciones precisas, listas locales, variabilidad y replays mecanísticos | Δ cambia K y escala de preferencias; sólo dominios seleccionados |
| Certificado MAX | predicados de presencia e idempotencia conocidas | verificación finita de la cabeza canónica y límites de aplicación | sin contexto nativo y sin garantía de progreso |

## Artículo integrado frente a dos publicaciones

Recomiendo **un artículo integrado** con eje información–continuación–transferencia.
Los mismos mecanismos conectan las tres partes y sostienen una conclusión más
útil que tres listas de resultados. Dividir ahora separaría evidencia que
necesita contexto común y obligaría a repetir interfaz, profesores y cohortes.
No veo dos contribuciones causales independientes y maduras que justifiquen
partir el resultado principal por volumen de ejecuciones.

Las distribuciones completas y el desarrollo/validación de instrumentos
estadísticos podrían sostener un trabajo distinto si se delimita una pregunta
metodológica independiente y se evita doble uso editorial de tablas/resultados.
Eso es una opción futura, no condición del cierre ni autorización para una
segunda publicación. La matriz de antecedentes y las atribuciones históricas
deben acompañar cualquier reutilización.

## Venue y estado de envío

**TMLR es la primera opción recomendada**, como juicio de ajuste temático y
editorial, no como predicción de aceptación. Sus
[criterios oficiales](https://jmlr.org/tmlr/acceptance-criteria.html) priorizan
que afirmaciones y evidencia sean correctas y convincentes y que los hallazgos
interesen a su audiencia; permiten reducir afirmaciones cuando falta evidencia.
Esto encaja con un estudio mecanístico y negativo bien auditado.
La [guía de autores](https://jmlr.org/tmlr/author-guide.html) exige formato TMLR,
anonimización y material completo. El estilo `article` actual es un borrador
para revisión, no un envío preparado; el límite de suplemento del sistema
tampoco permite adjuntar sin planificación todo el runtime grande.

LoG ofrece un buen ajuste por algoritmo sobre grafos y análisis empírico, pero
el [CFP 2026](https://logconference.org/cfp/) cerró papers el 7 de agosto y
journal track el 30 de septiembre; no debe presentarse como convocatoria
abierta el 26 de octubre. La siguiente edición requiere comprobar su propio
CFP cuando exista. El CFP actual exige responsabilidad de los autores y
declaración de uso de GenAI para contenido científico central asistido; debe
prepararse honestamente para un eventual envío. La contribución actual no se
orienta a una mejora de complejidad distribuida que justifique priorizar una
conferencia teórica de algoritmos distribuidos.

## Condición de cierre de esta perspectiva

N-01 y N-05 deben corregirse antes de cerrar la revisión interna; N-02/N-08
requieren mejoras breves de contexto y redacción. N-03/N-04/N-07 se resuelven
manteniendo las restricciones ya aceptadas y las correcciones del revisor ML.
N-06 distingue lo recuperable local de lo efectivamente entregable; basta
evidencia de recuperación acotada para revisión local y un plan concreto de
entrega para el envío posterior. No queda una necesidad demostrada de nueva
campaña científica para sostener el artículo delimitado. El editor debe
registrar la resolución de estas observaciones sobre el hash del PDF final,
sin convertir este dictamen interno en aprobación de revisores externos.

## Addendum de verificación de correcciones — 26 de octubre de 2026

Se releyeron el texto corregido, `closure-references.bib`,
`docs/NOVELTY_AND_RELATED_WORK.md` y las instrucciones de build. El SHA-256
del **main.tex inspeccionado en este addendum** es
`48db1d2dc46733e018489a8d18a4e00abba652c2fa113ec0b3d7671e6b35816c`.
Una edición posterior requiere la asociación del editor con su nuevo hash;
este addendum no afirma haber visto un PDF recompilado de un TeX posterior.

| Objeción | Verificación y estado |
|---|---|
| N-01 | **Resuelta.** El texto cita Hint-ReLIC con título/autores/proceedings correctos y distingue augmentaciones que preservan siguiente paso de las consultas explícitas que pueden cambiar la respuesta. Aparece también en la matriz del apéndice y el documento de novedad. |
| N-02 | **Resuelta.** Se añadió Rodionov–Prokhorenkova, su comparación sin hints y la renuncia a prioridad conceptual de output frente a trayectoria. La referencia apunta al PDF primario de NeurIPS. |
| N-03 | **Resuelta para la afirmación delimitada.** Se mantiene el oráculo tipado/declarado, presupuesto desigual y desarrollo reutilizado. El nuevo bloque BE incluye prevalencia, BA y baselines de rechazo/elegibilidad, y muestra dónde se rechazan opciones ocupadas entre consultas elegibles activas. No se recupera la identificación completa del profesor. |
| N-04 | **Resuelta por alcance explícito.** El texto mantiene procedencia heterogénea del 21/21, principal frente a controles en 810.000, conteos de grafos/cintas y fallos por mecanismo. No se reclama SOTA ni superioridad arquitectónica general. |
| N-05 | **Resuelta.** README ahora declara main.tex autoritativo, consulta el recibo para páginas/hashes y recomienda sólo scripts de cierre para el PDF actual. Los tres generadores históricos tienen guardia antes de escribir. Su ejecución se comprobó: los tres salen con estado no cero y mensaje «Historical generator», sin generar un manuscrito antiguo. |
| N-06 | **Permanece tarea editorial de entrega, no nueva investigación.** README diferencia restauración runtime de clonación de fuente y publicación no autorizada. Debe asociarse al manifiesto/recibo de recuperación relocalizada y `delivery.json` efectivamente terminados por el editor; su mera mención no se certifica aquí como archivo entregado. |
| N-07 | **Resuelta manteniendo límites.** Las garantías siguen siendo condicionales y se separan certificados históricos de pesos actuales; no se reclama un teorema de aprendizaje/generalización universal. |
| N-08 | **Mejora suficiente para revisión externa.** El hilo sigue siendo interfaces, continuación y transferencia. Los mecanismos y el nuevo análisis BE hacen explícita una enseñanza científica más precisa que el contador de éxitos. La maquetación final y su revisión visual corresponden al recibo del editor. |

No aparece un nuevo bloqueo científico sustantivo desde esta perspectiva.
Se mantiene **READY WITH MINOR REVISIONS** para revisión externa del artículo
integrado. Los pendientes de autoría/atribución, estilo y anonimización del
venue, declaración de asistencia pertinente y entrega reproducible local
no se confunden con un arbitraje humano completado o una publicación pública.
No es necesario un nuevo entrenamiento para resolverlos.
