# Auditoría de afirmaciones matemáticas

Se revisan las reglas implementadas y el nuevo manuscrito, no se conserva
un teorema por aparecer en una versión antigua. La lectura adversarial
independiente está en `docs/reviews/REVIEW_1_DISTRIBUTED.md`. Pruebas completas
en los apéndices del manuscrito y documentos matemáticos conservados.

| ID | Resultado y estatus | Hipótesis esenciales | Veredicto y límite |
|---|---|---|---|
| M01 | Lema de disponibilidad | estado parcial propio, listas tamaño ≥d(v)+1 | correcto; contar colores distintos sólo fortalece cota |
| M02 | Seguridad exacta Luby/Johansson | propuestas disponibles y veto igual-oferta exacto, colores permanentes | inducción correcta; no para COMMIT arbitrario |
| M03 | Fórmula de progreso | condicionar antes de draws, independencia entre nodos y P/C dentro del nodo | correcta; marginales uniformes solos no bastan |
| M04 | Progreso ≥1/4 Luby | p=1/2, propuesta uniforme disponible | chord/Jensen correctos también para listas locales |
| M05 | Progreso ≥1/4 variante Johansson | siempre activo, propuestas uniformes, vecino pendiente tiene ≥2 disponibles | correcto para esta variante; no importar otro teorema Johansson |
| M06 | Progreso CT positivo | alfabeto común bound D≥Δ, preferencias independientes, competición sólo en colores disponibles | cota (D+1)^−1(D/(D+1))^Δ; D=3 nativo, D=Δ extensión; aislado termina |
| M07 | Cola geométrica y terminación casi segura | cota uniforme condicional en todo estado alcanzable, seguridad | expectativa iterada+unión correctas; no prueba de PRNG finito/checkpoints |
| M08 | Correctitud/≤n JP | IDs distintos, orden coherente fijo, first-fit exacto | máximos independientes y máximo global progresa; cap protocolario separado |
| M09 | Correctitud Greedy/n escrituras | un permiso por paso, orden externo, first-fit | correcta; construir orden/calendario no es LOCAL anónimo gratuito |
| M10 | Correctitud GM y coste | cuatro bloques, escrituras disponibles, detección que cubre conflictos mismo-slot, reparación serial completa | correcta; omitir una reparación invalida progreso, conflictos temporales permitidos |
| M11 | BE especializado | IDs<2^16, polinomios grados4/2, campos13/7, Δ≤3, bloques KW disjuntos | conteos correctos; trece reducciones49→4; no es BE general |
| M12 | Obstrucción informacional | dos contextos admisibles indistinguibles con distintas acciones | correcta, elemental; no implica imposibilidad de colorear ni de aprender IDs numéricos |
| M13 | Progreso por anticoncentración | rechazo exacto, independencia, p_min>0,p_max<1, soporte disponible y maxq≤β/|A| | log-chord/Jensen y signo negativo correctos; condición sólo local para cota local |
| M14 | Submasa difusa | masa η>0 con cota en subconjunto, resto hipótesisM13 | normalizar y monotonicidad correctas; no certificado empírico uniforme |
| M15 | Seguridad≠progreso | clique K≥2, propuestas deterministas iguales | contraejemplo correcto; se corrigió caso frontera K=1 |
| M16 | Certificado MAX 64 conjuntos | seis tipos legales, mensajes no negativos, MAX, un único slot/gather, centro activo, sin residual nativo | correcto en aritmética real; no cubre CT/BE/GM por omitir contexto |
| M17 | Certificado numérico condicional | pesosfloat32 como racionales, intervalos outward64; gamma/error32 bajo IEEE sin overflow/TF32 | alcance explícito; no certificación universal CUDA/BLAS |
| M18 | Obstrucción MEAN | mensajes/clasificador continuos sin grado/K extra, aceptación estricta | dilución por vecinos ordinarios correcta; no para toda GNN MEAN ni necesariamente alcanzable |
| M19 | Construcción SUM | seis tipos binarios, mensaje b=o+ReLU(p+a−1) | presencia exacta representable; fallo SUM entrenado no imposibilidad |
| M20 | Simetría de scorer sin posición | inputs idénticos, equivarianza determinista y contexto invariante | scores empatan; controles principales tienen posición y quedan fuera |
| M21 | Uniformidad clique | K=n, equivarianza de vértices, inputs intercambiables, evento éxito invariante con masa>0 | acción transitiva en n! coloraciones; no basta anonimato, no K>n |
| M22 | TV por coupling | coupling válido de leyes completas, mismo grafo/objetos | TV≤Pr(X≠Y) correcto; uniformidad de error transición requiere hipótesis adicional |
| M23 | Cota CP cero desacuerdos | r corridas pareadas independientes, política/grafo/protocolo fijo | 1−α^(1/r) correcta; no varios grafos ni checkpoints simultáneos |
| M24 | No rechazo≠igualdad | test finito con potencia limitada | advertencia lógica correcta; no convertir p>α en equivalencia |
| M25 | Equivarianza nodal | observaciones, IDs y calendarios transformados coherentemente, aritmética real | parámetro compartido/MAX conservan simetría; no implica corrección, independencia de grado ni equivar. de colores |

## Revisión de código y casos frontera

Las implementaciones relevantes son `multiteacher/programs.py`,
`degree/teachers.py`, `degree/adapter.py`, `multiteacher_e1/model.py`,
`paperstudy/certification.py`. El oráculo escalar independiente y 480 tests
seleccionados comprueban dominios, profesores, arquitecturas y numérica; los
tests complementan las pruebas, no las sustituyen.

Casos revisados: aislados, listas singleton, Δ0 donde la interfaz lo admite,
grados exactos en generadores, suma par regular, listas distintas, máximos
globales de IDs, padding nativo, slots distintos, cola GM vacía, raíces KW
bajas que no recolorean, campos insuficientes fuera de BE nativo, conjunto
MAX vacío, precisión finita y tie-breaking. Los pilotos no requieren
extender BE; usar campos históricos a Δ4 retiraría la prueba de existencia.

La equivalencia matemática real del adaptador y sus pruebas de paridad
numérica son objetos distintos. El orden de reducción de un kernel puede
cambiar con batch, aunque el MAX de tipos en aritmética real sea idéntico.
Por ello los primeros fallos se reproducen en su batch original y no se
promete neutralidad numérica en todo hardware.

## Certificados que realmente se conservan

Los cinco E1 históricos en `results/ucgnn-paper-study/certificates/` tienen
certificados condicionales MAX; pertenecen a otra cohorte. Ninguno de los
45 núcleos canónicos Luby/Johansson/Greedy seleccionados en transferencia
pasa la condición suficiente all-types. No se afirma seguridad universal de
las 21 combinaciones ni se transfieren certificados entre pesos diferentes.
Un certificado suficiente fallido no demuestra por sí solo fallo en un
estado alcanzable; sus tipos pueden ser intervenciones legales no visitadas.

## Estatus epistemológico y antecedentes

M01–M11 son lemas/proposiciones condicionales para reglas exactas y variantes
documentadas. M12 es una obstrucción de observación elemental. M13–M20 son
proposiciones de representación/progreso con supuestos explícitos; no son
garantías aprendidas de datos. M21–M24 son argumentos estándar de simetría,
acoplamiento y estadística. El 21/21, las diferencias CF, paridad, tamaños y
grados son resultados experimentales. Capacidad versus optimización, cotas
uniformes sobre políticas y superioridad causal CF permanecen abiertos.

Los antecedentes primarios incluyen Linial, Kuhn–Wattenhofer, Barenboim–Elkin,
Luby, Johansson, Jones–Plassmann y Gebremedhin–Manne. No se reivindica novedad
de sus garantías, de indistinguibilidad, de la desigualdad de coupling ni de
la construcción genérica safety/progress. La novedad empírica y los mecanismos
concretos se delimitan en `NOVELTY_AND_RELATED_WORK.md`.
