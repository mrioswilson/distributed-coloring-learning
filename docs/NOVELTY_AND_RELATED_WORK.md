# Novedad y antecedentes: revisión focalizada final

Fecha de búsqueda y consulta: **26 de octubre de 2026**. Se consultaron
artículos originales, proceedings, páginas de los autores y documentación
oficial de venues. Las fuentes del repositorio y sus auditorías bibliográficas
previas se conservaron. Esta búsqueda focalizada no demuestra ausencia de
todo antecedente; no se reclama prioridad con «first».

## Pregunta y contribución delimitadas

El trabajo estudia cuándo una política local reproduce reglas específicas
de coloración distribuida dentro de una interfaz suministrada, no si una GNN
puede colorear por primera vez. La contribución defendible combina cobertura
funcional de variantes heterogéneas, oráculos de continuación tipados,
contrastes de entrenamiento con límites auditados y transferencia congelada
con mecanismos de fallo reproducibles. La distinción conceptual entre
resolver y reproducir un algoritmo no es nueva.

## Matriz de contribuciones y objeciones

| Candidata | Qué se conocía / fuente primaria | Resultado concreto aquí | Evidencia y diferencia metodológica | Objeción especializada / decisión |
|---|---|---|---|---|
| Aprender coloración con GNN | [Lemos et al., 2019](https://arxiv.org/abs/1903.04598): clasificación de colorabilidad, embeddings y transferencia | siete reglas explícitas, tres núcleos, 21 paquetes nativos con cinco pesos cada uno | protocolos, IDs/schedules y todos los hashes conservados | No es novedoso usar GNN; conservar sólo la comparación concreta |
| Ejecución neuronal/MAX/trazas | [Veličković et al., ICLR2020](https://openreview.net/pdf?id=SkgKO0EtvS), [CLRS2022](https://proceedings.mlr.press/v162/velickovic22a.html) ya enseñan ejecución paso a paso y MAX | separación medida entre acciones nativas, continuación y funcionalidad en estos profesores | oráculos de fase y diagnósticos sobre pesos congelados | No reclamar invención de NAR/MAX ni comparabilidad con CLRS sin ejecutar ese benchmark |
| Interacción y shift | [DAgger, 2011](https://proceedings.mlr.press/v15/ross11a.html) agrega datos en distribuciones inducidas por aprendiz | bancos finitos de continuaciones docentes, con prefijos admisibles | estado base compartido, consultas COMMIT adicionales, contabilidad explícita | No es DAgger nuevo; distinguir estados del aprendiz de intervenciones en mensajes |
| Intervención causal en imitación | [de Haan et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html) estudia confusión causal | errores de continuación de profesores concretos y diferencia de procedimientos | no se equiparan presupuestos totales, no identificación del efecto exclusivo CF | La afirmación causal fuerte se retira |
| Intervenciones **en NAR** | [Hint-ReLIC, ICML2023](https://proceedings.mlr.press/v202/bevilacqua23a.html) usa augmentaciones que preservan siguiente paso e invariancia de predicción | etiquetas explícitas del oráculo para mensajes/colores alternativos que pueden cambiar la respuesta | contratos tipados, bancos de raíces, sensibilidad COMMIT medida | Antecedente directo omitido inicialmente; incorporado por revisión R4. No reclamar intervención algorítmica como idea nueva |
| Resultado final frente a traza | [Rodionov–Prokhorenkova, NeurIPS2023](https://papers.nips.cc/paper_files/paper/2023/file/a2370db7c99791ad5d9f3ef48ad6d464-Paper-Conference.pdf) compara aprendizaje sin supervisión intermedia | BE nativo perfecto y respuestas alternativas distintas; negativos y baselines auditados | mismo banco/semilla/updates, respuesta a continuación tipada | El contraste conceptual precede; novedad es la cuantificación/mecanismo específico, no la idea |
| Identificadores e información | [Linial, 1992](https://courses.csail.mit.edu/6.852/08/papers/Linial.pdf), [Loukas, 2020](https://arxiv.org/abs/1907.03199), [Sato et al., 2019](https://arxiv.org/abs/1905.10261) conectan información local, simetría y GNN | control JP cuatro representaciones y auditoría de helpers | 3.000 etiquetas,500updates,5semillas; comparaciones relativas60/60 | La obstrucción es elemental; fallar con magnitudes no demuestra imposibilidad de aprender IDs numéricos |
| Generalización algorítmica | [Xu et al., ICLR2021](https://arxiv.org/abs/2009.11848), [Flood and Echo Net](https://arxiv.org/abs/2310.06970) estudian alineación y ejecución | traslado sin updates a condiciones congeladas de grado/lista/tamaño, y 15 replays | 810.000 filas reconciliadas,450.000 principales; no promedio ficticio | No es primera extrapolación; grado y K se confunden; pilotos no son certificaciones |
| Garantías aprendidas de extrapolación | [Wittig et al., 2026v2](https://arxiv.org/abs/2602.13106v2) establece condiciones de aprendizaje/generalización con regularización | garantías condicionales de reglas/interfaz más comprobación finita de pesos | no se prueba que nuestras políticas satisfagan hipótesis uniformes | No competir en teoremas de generalización universal ni transferir su resultado a estas redes |
| GNN y coloración paralela | [Langedal–Manne, ALENEX2025](https://arxiv.org/abs/2408.05054) aprende ordenamiento para greedy paralelo | imitación de decisiones de varias reglas, incluido scheduler declarado | nuestro objetivo no es minimizar colores ni acelerar LF/SL | No reclamar SOTA de coloring ni más rapidez que heurísticas no ensayadas |
| Coloración NAR reciente | [Vanderbush–Weber, 2026](https://arxiv.org/abs/2601.05137) optimiza coloración aproximada con warm starts | evaluación de reglas locales con objetivo de propia completa, K determinado | no warmstart por instancia ni búsqueda competitiva de k | Mismo campo general; problemas/criterios distintos; no superioridad numérica cruzada |
| Transferencia de geometría de coloración | [Le–Zhao–Weber, 2026](https://arxiv.org/abs/2606.03923) aprende geometría contrastiva y transferencia | fidelidad a continuaciones y dependencia de paleta/preferencias | no se evalúan sus encoders ni su objetivo aproximado | No reivindicar primera generalización de coloración en 2026 |
| Ejecución asíncrona | [Graph Neural Networks Gone Hogwild, ICLR2025](https://arxiv.org/abs/2407.00494) da robustez bajo hipótesis específicas | estudio síncrono/scheduled con dos intercambios aprendidos | no hay experimento de asynchrony ni pérdida de mensajes | Mantener fuera de alcance, no presentarlo como deployment distribuido robusto |
| Invariancia de conjuntos y atención | [Deep Sets](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | sustitución del procesamiento local de paleta conservando comunicación | tres núcleos con mismos predicados por profesor, auxiliares declarados | No llamarlos modelos sin comunicación ni nueva familia de atención |
| Comparar leyes completas | [Gretton et al., 2012](https://www.jmlr.org/papers/v13/gretton12a.html), [Hemerik–Goeman](https://doi.org/10.1007/s11749-017-0571-1), [Holm](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) | reconciliación del objeto estadístico e instrumentos en bancos conservados | filas completas, bloques grafo/peso, potencia y no rechazo delimitados | No es nuevo test general ni igualdad por p>α; estudios históricos quedan en suplemento |

## Profesores frente a sus artículos fuente

Las fuentes originales de Luby, Johansson, Jones–Plassmann, Linial,
Kuhn–Wattenhofer, Barenboim–Elkin, Schneider–Wattenhofer y
Gebremedhin–Manne constan en `integration-references.bib` y los manifiestos
de verificación anteriores. Se contrastaron las definiciones del código y
las hipótesis de las pruebas. Para [GM](https://eecs.wsu.edu/~assefaw/publications/Concurrency12-2000.pdf)
la fuente describe especulación, detección y reparación serial en memoria
compartida; no autoriza llamarlo algoritmo LOCAL anónimo. El
[BE original](https://arxiv.org/pdf/0812.1379v2) contiene una construcción más
amplia que nuestro caso base. [KW](https://ac.informatik.uni-freiburg.de/publications/publications/podc06.pdf)
es antecedente de reducción iterativa, no de aprender disponibilidad.
La página editorial de Johansson devolvió 403; se conserva la verificación
bibliográfica previa y se prueba directamente la variante implementada,
sin importar un teorema no inspeccionado.

## Búsqueda y metadatos

Consultas focalizadas: neural algorithmic reasoning + distributed coloring;
learning distributed algorithms; counterfactual supervision + algorithmic
reasoning; graph coloring generalization 2025/2026; causal regularisation;
learning from traces; graph-local information; provable extrapolation. Se
abrieron fuentes primarias y se comprobaron título, autores, año y objeto.
Wittig v2 es de 23/08/2026; los trabajos de coloración citados de 2026 son
de enero y junio. No se presenta un preprint como artículo aceptado si no
se verificó su publicación. OpenReview bloqueó algunas páginas de foro;
se usaron PDFs oficiales/arXiv de autores. No se usa una revisión comercial
o agregador como fuente técnica. La bibliografía nueva está separada en
`closure-references.bib` y el catálogo reproducible en `literature-sources.json`.

## Recomendación de narrativa y venue

Mantener un artículo integrado: información → aprendizaje de reglas →
continuaciones → transferencia. La estadística de leyes globales es soporte
histórico, no un segundo núcleo que diluya el argumento. Un artículo futuro
sobre instrumentos/leyes necesitaría una contribución propia y delimitación
sin duplicar estas mismas contribuciones.

TMLR es la primera opción razonada: sus [criterios oficiales](https://jmlr.org/tmlr/acceptance-criteria.html)
priorizan correspondencia entre afirmaciones y evidencia y claridad para
una audiencia interesada. Es una recomendación de encaje, no pronóstico de
aceptación. Su [guía](https://jmlr.org/tmlr/author-guide.html) requiere estilo
y anonimización de envío; el `article` actual es borrador de revisión externa.
[LoG](https://logconference.org/cfp/) encaja temáticamente, pero los plazos
de 2026 ya cerraron; no se inventa una fecha de próxima convocatoria.
PODC/DISC serían una apuesta menos natural para esta versión: la contribución
central es empírica y no ofrece un nuevo resultado general de complejidad
distribuida. Esta última evaluación es juicio editorial, no política del venue.
