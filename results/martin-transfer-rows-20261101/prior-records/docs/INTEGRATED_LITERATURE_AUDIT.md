# Posicionamiento bibliográfico de la integración

Fecha de verificación: 31 de octubre de 2026. Fuentes primarias y comparación
dirigida de los precedentes cercanos; no se afirma revisión sistemática
exhaustiva ni prioridad. La biblioteca adicional de Martín se conserva,
pero el artículo cita solo los trabajos pertinentes a sus afirmaciones.

| Precedente primario | Qué impide reclamar como nuevo | Diferencia concreta del artículo |
|---|---|---|
| [CLRS, Veličković et al., ICML 2022](https://proceedings.mlr.press/v162/velickovic22a.html) | Aprender ejecución de algoritmos a partir de trazas | Contratos tipados de continuaciones en profesores concretos de coloración y relación con ejecución |
| [Hint-ReLIC, Bevilacqua et al., ICML 2023](https://proceedings.mlr.press/v202/bevilacqua23a.html) | Regularización causal de razonamiento algorítmico, con aumentos que preservan el siguiente hint | Consultas de módulos después del intercambio e intervenciones diagnósticas de componentes; no una regularización causal general nueva |
| [SALSA-CLRS, Minder et al., 2023](https://arxiv.org/abs/2309.12253) | Benchmarks de algoritmos distribuidos y aleatorizados | Siete variantes heterogéneas con contabilidad de operaciones suministradas, continuaciones y mecanismos verificables |
| [Qharabagh et al., v2 2026](https://arxiv.org/abs/2601.23207) | Aprendizaje de instrucciones locales y resultados de ejecución exacta bajo hipótesis | Evaluación empírica de políticas concretas y fallos alcanzados; no teorema de learnability ni primer aprendizaje de reglas distribuidas |
| [Discrete NAR, Rodionov y Prokhorenkova, ICML 2025](https://proceedings.mlr.press/v267/rodionov25a.html) | Estados finitos predefinidos que permiten verificar transiciones y corrección | El certificado MAX es condicional y estrecho; no verifica ejecución general ni progreso |
| [Wittig et al., v2 2026](https://arxiv.org/abs/2602.13106v2) | Condiciones suficientes de aprendizaje/extrapolación e imposibilidades | Resultados finitos de transferencia con modelos congelados; sin inferir extrapolación ilimitada |
| [IIT, Geiger et al., ICML 2022](https://proceedings.mlr.press/v162/geiger22a.html) | Entrenamiento con interchanges contrafactuales y alineamiento causal | Aquí el alineamiento es una interfaz de mensajes declarada; no se descubre una abstracción causal interna |
| [DIITO, Wu et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.318/) | Destilación de dinámica causal mediante intervenciones | La fidelidad de continuación se refiere al oráculo explícito, sin identificar mecánica interna por mera exactitud |
| [Mind the Gap, Saldyt y Kambhampati, v2 2025](https://arxiv.org/abs/2505.18623) | Diferenciar algoritmos efectivos y fieles, estudiar brechas mecánicas | Acoplamiento al intérprete, soporte de COMMIT, reparación y rescates completos de componentes en coloración |
| [Premature Floods, Najafgholizadeh y Afsharchi, 10 octubre 2026](https://arxiv.org/abs/2610.14982) | Fallos pese a reproducir cada hint y explicación por mecanismos concretos | Trabajo concurrente muy cercano en espíritu; aquí son continuaciones, cotas KW, diversidad y colas GM, sin afirmar primera separación |

Las páginas de arXiv confirman autores, títulos, versiones y fechas de los
cuatro últimos preprints relevantes de 2025/2026. PMLR/ACL confirman las
entradas de conferencia y páginas de Discrete NAR, IIT y DIITO. Se comprobaron
además los registros primarios CLRS/Hint-ReLIC utilizados por la auditoría
de Martín. Las referencias históricas permanecen con su alcance original.

Correcciones integradas: CLRS es el benchmark; *Neural algorithmic reasoning*
de 2021 es el artículo de perspectiva. Se usa el registro ICML/PMLR correcto
para Hint-ReLIC. No se mantiene el calificativo «oral ICML 2026» importado
para Wittig: el acceso al registro oficial indicado no pudo confirmarlo y
el registro arXiv consultado no especifica esa presentación. Se cita su
v2 verificable; esto no niega que exista otra versión de conferencia.

La contribución defendible es la evaluación conjunta y trazable de módulos
locales de siete profesores, con éxito funcional y transferencia positivos,
contratos de continuación y mecanismos de progreso/reparación comprobados.
Las proposiciones de soporte y relevancia son elementales y se presentan
como herramientas de interpretación. No basta agregar referencias para
resolver una objeción de novedad: los rescates de batería completa y la
distinción entre desacuerdo de contrato y transición efectiva constituyen
evidencia específica adicional, con alcance finito y explícito.

Valoración editorial: hay una contribución empírica plausible para discusión
en TMLR; no es una predicción de aceptación. La revisión independiente debe
juzgar si la evidencia disponible y la precisión de los contratos compensan
la amplitud del artículo y sus mecanismos experimentales. La limitación
inicial de los 105 runtimes adicionales queda cerrada con el release y su
auditoría de inferencia; esto no modifica los precedentes ni demuestra
prioridad bibliográfica.
