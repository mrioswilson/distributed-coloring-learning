# Revisión adversarial y resolución del cierre

Se encargaron cuatro revisiones a agentes distintos del editor y entre sí,
siguiendo las cuatro perspectivas solicitadas. Son revisiones internas de
ejecución independiente: **no** arbitraje humano externo, nueva réplica de
entrenamiento ni certificación por cuatro equipos científicos externos.
Los informes iniciales y sus addenda se preservan sin borrar críticas.

1. [Algoritmos distribuidos y matemáticas](reviews/REVIEW_1_DISTRIBUTED.md).
2. [Machine learning](reviews/REVIEW_2_ML.md), con auditoría reproducible
   `REVIEW_2_BE_STRATA.py` sobre las consultas originales.
3. [Estadística experimental](reviews/REVIEW_3_STATISTICS.md), con recuento
   independiente de las 810.000 filas y sus hashes.
4. [Novedad y publicación](reviews/REVIEW_4_NOVELTY_EDITORIAL.md), con fuentes
   primarias consultadas separadamente.

Los addenda identifican el SHA de la versión releída. El PDF/manuscrito final
y las correcciones menores posteriores de leyenda/unidad de paso quedan
identificados en el manifiesto final y su recibo de compilación. Ningún
dictamen de revisión se usa como sustituto de los datos o demostraciones.

## Registro de objeciones

| ID | Categoría | Objeción | Respuesta y estado |
|---|---|---|---|
| D1 | Error demostrado | CT nativo usa bound3, no grado real de cada grafo | **Corregido**: parámetro D≥Δ, D3 nativo/DΔ transferencia, cota correcta |
| D2 | Error demostrado | clique de estancamiento incluye K1 | **Corregido**: K≥2; aislado separado |
| D3 | Problema editorial | paso del intérprete frente a intercambios | **Corregido**: ejemplo BE puntos/KW y no llamar quince pasos quince rondas aprendidas |
| D4 | Limitación reconocida | certificado MAX sólo canal canónico/aritmética condicional | **Delimitado**: no extender a auxiliares/progreso; cinco históricos frente45actuales |
| D5 | Limitación reconocida | paridad finita no es equivalencia numérica universal | **Delimitado**: adapter declarado, pesos congelados, kernels/batches separados |
| D6 | Limitación reconocida | Δ,K y preferencias confunden extrapolación | **Delimitado** en generalización; sondas locales no tasas universales |
| D7 | Limitación reconocida | uniformidad/TV necesitan condiciones y no rechazo no basta | **Delimitado**, pruebas completas y K=n explícito |
| D8 | Nueva investigación fuera de alcance | testigo MEAN asintótico podría exceder n≤65536 | **No reclamado** como fallo dentro del benchmark; familia matemática y alcanzabilidad separadas |
| ML-01 | Evidencia insuficiente en interpretación inicial | BE≈97% negativos; baseline trivial omitido | **Resuelto con datos existentes**: confusiones, BA, always-reject/eligibility, elegibles activas |
| ML-02 | Evidencia insuficiente para generalización amplia | tipos repetidos, cuatro colores/generador conocidos | **Afirmación amplia retirada**; subconjunto posthoc con ambas clases y baselines |
| ML-03 | Limitación reconocida | CF añade etiquetas, presentaciones y pérdida | **Sin afirmación causal a igual recursos**; costes de banco y entrenamiento separados |
| ML-04 | Limitación reconocida | development reutilizado y heads idénticas | **Delimitado**: cinco semillas, no quince réplicas ni cientos de miles de trainings |
| ML-05 | Problema editorial | inferencia se adapta además de tensores | **Corregido**: media sobre slots válidos, normalización y listas explícitas |
| ML-06 | Problema editorial | se habían omitido small/no-communication | **Corregido**: tabla apéndice a B3000,500updates,28casos/semilla; sin certificación falsa |
| ML-07 | Limitación conceptual | oráculo off-policy no único a partir de trazas | **Delimitado**: oráculo de continuación elegido, no identificación completa del algoritmo |
| R3-01 | Evidencia insuficiente | accuracy sola no caracteriza mejora BE | **Resuelto**: mismas confusiones/estratos; todo brazo FN0, mejora de especificidad |
| R3-02 | Evidencia insuficiente/etiquetado | CP piloto no tiene cobertura automática | **Corregido**: CP sólo confirmatorio, interval_scope; pilotos vacíos |
| R3-03 | Limitación reconocida | pseudorreplicación de consultas/arquitecturas | **Delimitado**: semilla como unidad; sin p binomial de decisiones |
| R3-04 | Evidencia insuficiente para afirmación fuerte | development y subconjuntos posteriores | **Afirmación confirmatoria retirada**; etiquetas development/posthoc en texto/figura |
| R3-05 | Evidencia insuficiente | causalidad CF a etiquetas iguales no identificada | **Retirada**; contraste de procedimientos y control optimizador separados |
| R3-06 | Limitación numérica | CP presupone independencia/neutralidad de batch | **Añadido** supuesto; conteos exactos y replays no dependen de vender CP como garantía de kernels |
| R3-07 | Limitación reconocida | condiciones elegidas tras piloto y multiplicidad | **Delimitado**: namespaces nuevos, intervalos puntuales, sin certificación global |
| R3-08 | Fuera de alcance/limitación | no repetir todos los tests históricos de leyes | **No se reclama** nueva validación de cada contraste; resultados numéricos fuertes no reintroducidos |
| N-01 | Evidencia insuficiente bibliográfica | antecedente directo Hint-ReLIC omitido | **Corregido**: ICML2023, contraste explícito invariancia/etiquetas alternativas y matriz |
| N-02 | Problema editorial/novedad | no-hint2023 demasiado cercano para omitir | **Corregido** en introducción, matriz y bibliografía |
| N-03 | Limitación conceptual | semántica contrafactual no identificada sólo por trazas | **Delimitado**, coincideML07 |
| N-04 | Limitación reconocida | tamaño de campaña no es novedad/SOTA | **Sin reivindicación** de prioridad, superioridad global o competición de heurísticas |
| N-05 | Error demostrado de reproducción | README/generadores históricos sobrescriben cierre | **Corregido**: fuenteautoritat, flujo nuevo; tresgeneradores rechazan antes de escribir |
| N-06 | Limitación editorial | rutas locales no son entrega portable | **Resuelto para entrega local** mediante manifiesto, paquete relativo y smoke reubicado; publicación/anónimo externo pendiente sin autorización |
| N-07 | Limitación reconocida | proposiciones elementales no son nuevos teoremas generales | **Delimitado**, auditoría clasifica estatus/antecedentes |
| N-08 | Problema editorial | inventario/advertencias pueden ocultar contribución | **Reestructurado**: información→reglas→continuaciones→transferencia; detalle en apéndices |

## Qué cambió por las críticas

La revisión no fue una formalidad. Añadió un baseline BE de 97,0465% por
rechazo constante y otro de 98,5990% por elegibilidad, reveló que todos los
34 errores CF son falsos positivos y que los otros brazos aceptan opciones
ocupadas en raíces activas elegibles. Con ambas clases presentes también en
tipos nuevos, la ventaja CF persiste, pero el mecanismo es principalmente
rechazo de continuaciones negativas. Se corrigieron dos especificaciones
matemáticas, se retiraron inferencias CP exploratorias y se incorporó un
antecedente NAR causal directo. Se conserva esa historia de corrección.

## Dictamen tras las respuestas

Los revisores matemático, ML y estadístico comprobaron los cambios y no
identifican objeciones sustantivas pendientes para las afirmaciones acotadas.
El revisor editorial mantiene revisiones menores de envío y entrega, que
se distinguen de ampliar la investigación. Ningún resultado exige entrenar
una matriz nueva para sostener el manuscrito actual. Las afirmaciones causal
fuerte, extrapolación universal y capacidad identificada siguen **sin probar**
y permanecen fuera del artículo. La recomendación global fundamentada es
`PAPER_CLOSURE_DECISION.md`, con sus quince respuestas y pendientes exactos.
