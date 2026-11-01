# Revisión independiente de la revisión de Pedro — informe de Claude para ChatGPT

Fecha: 30 de octubre de 2026. Autor del informe: Claude (Opus 5.5), por encargo de Martín Ríos-Wilson.
Alcance: auditoría científica, matemática, experimental y editorial de la revisión del artículo hecha por Pedro Montealegre.
Este informe **no** modifica el manuscrito, no reentrena modelos y no ejecuta campañas nuevas.
Las verificaciones se hicieron por lectura, recomputación desde artefactos congelados, re-inferencia sobre pesos congelados y recompilación del LaTeX.

Convenciones de este documento:

- **B** = nuestra versión anterior (`c3affd16`).
- **P** = versión de Pedro (`5993a5e4`).
- "P:`archivo`:línea" = ruta relativa en el árbol de `5993a5e4`, bajo `paper/ucgnn/` salvo indicación.
- **VERIFICADO** = recomputado desde datos crudos, arrays o pesos.
- **RECIBO** = coincide con un JSON de Pedro pero no se pudo recomputar desde lo primitivo.
- **NO VERIFICABLE** = requiere datos ausentes.

---

## A. Resumen ejecutivo

**Qué cambió Pedro.** Pedro reescribió el artículo como una pregunta de identificación:

> "¿puede el acuerdo perfecto con las acciones observadas de un profesor ocultar una regla local incorrecta?"

El título pasa de *Information, Counterfactual Fidelity, and Frozen Transfer* a *Continuation Fidelity, Progress, and Failed Repair*.

Sus cambios principales son:

1. **Hizo explícitos los contratos de continuación.** Ecuación del oráculo $O_T=E_v\,p_v\,F_v(c_v)\,R_T$, tabla por profesor y ejemplo KW.
2. **Hizo explícito qué es aprendido y qué es suministrado.** Nueva figura de arquitectura.
3. **Reanalizó el diagnóstico BE histórico** con exactitud ponderada por tipo y baselines programados.
4. **Añadió un experimento prospectivo nuevo de 40 modelos.** BE y GM sobre E1, 10 semillas pareadas × {nativo, CF}, con igual número de objetivos (6.600 por modelo) y holdout del 20 % de tipos COMMIT suficientes.
5. **Añadió auditorías post-test de ese experimento**: regla programada, soporte, solapamiento y confusiones.
6. **Añadió intervenciones de un componente** sobre dos *witnesses* históricos de fallo, Johansson y GM, con una nueva Figura 3.
7. **Bajó al apéndice** la cobertura funcional 21/21 y el resultado de identificadores. Retiró la figura de transferencia.
8. **Hizo trabajo de procedencia**: recuperación por SHA256 de 225 checkpoints y re-verificación de sus 40 pesos históricos.

**Tesis del nuevo artículo.** El acuerdo nativo no identifica la respuesta del módulo local ante intervenciones tipificadas. La fidelidad de confirmación (COMMIT) no garantiza ni progreso ni reparación. Los fallos de transferencia congelada se explican por defectos de componentes distintos del veto: diversidad de propuestas y reparación programada.

**Evaluación preliminar.**

- **Mejoras reales y valiosas:**
  - precisión conceptual (contratos, separación aprendido/suministrado, estimandos explícitos);
  - honestidad estadística: reporta todos los resultados adversos;
  - mecanismos de fallo mucho más visibles y apoyados en intervenciones causales locales;
  - integridad técnica excelente: hashes, cronología interna, re-inferencia 40/40 sin discrepancias, tests en verde.
- **El experimento prospectivo nuevo es metodológicamente más débil de lo que su lugar en el abstract sugiere.** Las auditorías independientes de esta revisión encontraron:
  - el control nativo tiene etiquetas COMMIT que son copia de la participación, así que GM 50→100 % está garantizado por diseño;
  - la novedad del test es muy distinta entre brazos y el paper no lo reporta;
  - el endpoint BE está dominado (96,6 %) por consultas fuera de política;
  - la diferencia BE de +5,77 pp descansa en tres colapsos del brazo nativo.
- **Ninguna de estas observaciones invalida una afirmación central tal como está escrita.** Pedro ya rebaja mucho el alcance, pero varias requieren corrección antes de un envío.
- **La tesis de identificación se usa como titular pero no está formulada como resultado.** Lo mismo ocurre con los argumentos de progreso y reparación, que siguen siendo informales.
- **Hay pérdida de contribuciones** respecto de B:
  - la cobertura funcional amplia (dimensión A) y la generalización sin reentrenamiento (dimensión C) quedan relegadas;
  - la figura de transferencia desaparece;
  - el resultado de identificadores pasa al apéndice.

**Hallazgo científico transversal.** No lo formula ninguno de los dos manuscritos, y emerge de cruzar los datos de Pedro con nuestro trabajo local paralelo: **la fidelidad de continuación (B) y el éxito funcional (A) no sólo son distintos, sino que pueden entrar en conflicto bajo esta supervisión.**

- En BE, CF mejora la exactitud por tipo pero reduce los éxitos autónomos (9.204 → 7.405 de 10.000).
- Los errores fuera de política de tipo "aceptar" son funcionalmente inocuos. Los errores de tipo "rechazar" en el estrato elegible-activo-libre matan el progreso.
- Esto unifica los resultados de GM y BE y conecta directamente con la tesis de Pedro sobre seguridad frente a progreso. Ver J-M5.

---

## B. Procedencia y versiones

**Repositorio.** Local: `/srv/projects/AlgoLearning/distributed-coloring-learning`, visto desde contenedores como `/workspaces/AlgoLearning/distributed-coloring-learning`. Remoto: `github.com/mrioswilson/distributed-coloring-learning`.

**Versiones del manuscrito.**

| Rol | Rama | Commit (completo) | Autor y fecha | `main.tex` sha256 | PDF sha256 | Páginas |
|---|---|---|---|---|---|---|
| **B** — versión anterior | `research/scientific-closure-20261026` (también en origin) | `c3affd16ac5654c6e2a700d2312710753adb9518` | Martín, 2026-10-28 | `5f197f7d…538340` (1.051 líneas) | `3b05ee8d…8103f` | 18 |
| **P** — revisión de Pedro | `origin/codex/paper-revision-20261030` | `e8241184bcf81505838b74c7acd2834f22676d78` (artículo + evidencia) y `5993a5e4782d44670926224263688943a1411ebf` (handoff) | Pedro Montealegre `<pedromb@gmail.com>`, 2026-10-30 17:03 y 17:11 UTC | `bd303c9d…032bb3` (1.086 líneas + 13 `revision-*.tex`) | `196e2fcc…5ddec5984` | 27 |

**Cómo se localizó la rama de Pedro.** No estaba en las ramas locales. Se descubrió con `git ls-remote` y se trajo con `git fetch` a `refs/remotes/origin/codex/paper-revision-20261030`, sin tocar el árbol de trabajo. Base común: `git merge-base` de P con B es exactamente `c3affd16`.

**Otras comprobaciones de procedencia.**

- `paper/ucgnn/archive/pre-revision-c3affd16.{tex,pdf}` de P son idénticos byte a byte a B.
- **Contenido del commit de Pedro:**
  - 157 archivos; incluye 60 JSON crudos por semilla (≈224 MB) bajo `results/paper-revision-20261030/prospective/`;
  - un tarball de runtime de 26,3 MB con SHA256 `9a33d4bb…a8cec8a`, que coincide con lo declarado;
  - código nuevo: `src/dcl/paperrevision/{experiment,keys}.py`, 14 scripts y 2 tests;
  - ediciones menores a `src/dcl/artifacts.py`, `localpalettes/config.py`, `sizedegree/common.py`, `scripts/build-closure-paper.py` y `tests/test_globaldist_integration.py`. Son cambios de rutas y una guarda; no alteran cifras históricas (verificado).
- **Bibliografía:** P no modifica ningún `.bib`.
- **Commits previos de Pedro** en `origin/main` (`fc756cef`, `80f486eb`, `6836d855`, 21–22 oct.) son su estudio de siete profesores. Ya estaban integrados en nuestra historia vía `34a798d1` y no son parte de esta revisión.

**Trabajo nuestro que NO es de Pedro, y que no debe confundirse con el suyo.**

- **La rama local `research/pedro-review-manuscript-revision`:**
  - commits `9497d39d…6853dd7f7e338821e074c145bb98dafe171b6e87`, 2026-10-30 08:48–09:27 −03, autor Martín con co-autoría Claude;
  - es nuestra respuesta paralela a las objeciones de Pedro;
  - no está publicada y no es ancestro de P.
- **Tiene trabajo sin commit:**
  - `paper/ucgnn/main.tex` modificado, sha256 `5c142e42…c9a9219a` en el disco actual;
  - archivos no rastreados `paper/ucgnn/archive/pre-pedro-review-c3affd16.*`, `figures/pedrorev-*`, `tables/pedrorev-*.tex` y `scripts/build-pedrorev-assets.py`.
- **Ese trabajo no fue tocado.** Según la auditoría de esa línea, su `main.tex` está a medio editar y no compila: macros indefinidas, apéndices eliminados y referencias rotas. El `paper/ucgnn/main.pdf` del disco es el build de B.
- Ver §E.4 para los puntos de contacto con P.

**Consecuencia práctica.** Hay **dos reescrituras completas** de `main.tex` divergiendo desde `c3affd16`. Cualquier integración será manual.

---

## C. El handoff original de Pedro (resumen fiel)

**Documento:** `docs/PAPER_REVISION_HANDOFF_MARTIN.md` en `5993a5e4`, sha256 `3a9103f5…79bb8c`. Hay copia en el paquete.

**Documentos asociados** (copiados en `pedro-original-docs/`): `PAPER_REVISION_CLAIMS.md` (matriz afirmación/evidencia/límite), `PAPER_REVISION_AUDIT3.md`, `PAPER_REVISION_REPORT.md`, `PAPER_REVISION_PROTOCOL.md` y `PAPER_REVISION_REPRODUCE.md`.

Posición de Pedro, en sus términos:

- **Eje del argumento:** "fidelidad de continuaciones, progreso y reparación".
  - Se explicitan oráculos, estados e intervenciones admisibles, y lo que aprende la red frente a lo que suministra el intérprete.
  - La cobertura nativa 7×3 queda como "evidencia finita bajo protocolos heterogéneos, sin ranking general ni certificado universal".
- **Ensayo nuevo de 40 modelos.** BE y GM, E1 con contexto nativo (19.396 parámetros), 10 semillas pareadas, 2 brazos, 6.600 objetivos por modelo, 1.500 actualizaciones y congelación antes del test. Resultados:
  - BE: BA 84,64 → 90,41, Δ +5,77 [−13,73; +25,26], éxitos 9.204 / 7.405;
  - GM: 50 → 100, Δ +50 [50; 50], éxitos 10.000 / 10.000.
- **"Interpretación científica que hay que mantener":**
  - en BE, la regla de elegibilidad/participación obtiene 99,71 % BA, por encima de CF; CF mejora la exactitud por tipo (86,94 frente a 81,27 %);
  - en GM el control nativo no recibe negativos ASSIGN/REPAIR y no hay comparación con otra adquisición informativa;
  - el holdout fino es válido, pero las configuraciones lógicas cubren 99,475–100 % (BE) y 100 % (GM); no se afirma extrapolación;
  - BE 105001/105008 ajustan sus auxiliares pero fallan en contextos retenidos; 105010 es una omisión de participación localizada;
  - los dos witnesses históricos son intervenciones seleccionadas, no una solución poblacional;
  - los pasos 14,53/13,43 incluyen fallos; todos los éxitos toman 15 pasos.
- **Figuras:** quedan tres (arquitectura, contraste histórico, witnesses). Se retiran la gráfica prospectiva de pares y las barras redundantes.
- **Objeciones que considera resueltas:**
  - definición de oráculos;
  - módulos aprendidos frente a suministrados;
  - interpretación de la comparación CF, que ahora se ciñe a un contraste de adquisición a igual presupuesto;
  - protagonismo de los mecanismos de fallo;
  - enlace teoría–experimento, vía hipótesis de progreso faltantes.
- **Lo que deja abierto:**
  - revisión científica humana independiente;
  - disponibilidad de los 810.000 registros crudos históricos;
  - un control GM con negativos informativos a igual presupuesto;
  - evaluación con condiciones lógicas realmente nuevas;
  - autoría, orden, venue y formato.
- **Sus tres auto-auditorías:**
  - no encontraron "un nuevo error matemático central";
  - corrigieron redundancia de figuras, el denominador de pasos y precisiones de exposición;
  - Pedro declara explícitamente que no sustituyen una revisión independiente.

**Declarado frente a efectivamente presente.** Todo lo declarado en el handoff aparece en el manuscrito, con dos excepciones:

- `PAPER_REVISION_REPORT.md:97` dice "La Figura 4 compacta se conserva", pero el `main.tex` final no incluye ninguna figura de transferencia. La afirmación es obsoleta, de una iteración intermedia.
- La afirmación "los cuarenta pesos de Pedro están disponibles en este workspace" (P:`main.tex`:510) se refiere al workspace aislado de Pedro (`/workspace/paper-revision-20261030`). Los recibos apuntan a rutas que no existen en esta máquina, así que **no es verificable aquí**.

---

## D. Nueva contribución científica (según P)

**Pregunta central** (P:`main.tex`:32–34): "Can perfect agreement on a teacher's observed actions conceal an incorrect local rule?"

**Contribuciones declaradas** (P:`main.tex`:75–84):

1. Contratos de continuación para siete profesores, que sitúan la intervención entre dos comunicaciones aprendidas y contabilizan el intérprete suministrado.
2. Reconstrucción del diagnóstico pareado, exposición de su desbalance de clases y solapamiento de tipos, y comparación prospectiva a igual presupuesto con tipos COMMIT retenidos. Se declara que "no es un test de condiciones lógicas nuevas".
3. Conexión de los fallos congelados de propuesta y reparación con las hipótesis de progreso faltantes, mediante intervenciones de un componente sobre witnesses preservados.

**Afirmaciones y evidencia:**

| Afirmación | Evidencia en P | Estado tras esta auditoría |
|---|---|---|
| Tres procedimientos BE/E1 con 28.800/28.800 acuerdos nativos difieren en continuaciones | §5, Tabla `tab:be`, `tab:type-weighting` (P:`revision-type-analysis.tex`) | Histórico, números heredados de B; tabla de tipos coherente con `be-type-counts.json` (CF BA 99,9965) |
| CF histórico más exacto, pero con más etiquetas y tipos familiares | §5 (texto de B conservado) | Igual que en B |
| A igual presupuesto, GM 50→100 % BA; BE +5,77 pp incierto; BE con menos éxitos autónomos | `revision-prospective.tex`:39–68 | **VERIFICADO** numéricamente; ver J-M1 a M3 sobre interpretación |
| La regla programada $G=E\,p$ supera a CF en BA BE (99,71 frente a 90,41); CF la supera por tipo (86,94 frente a 81,27) | `revision-prospective-audit.tex`:9–34 | **VERIFICADO**; CF supera a la regla por tipo sólo en 7/10 semillas |
| El test retiene contextos pero reutiliza configuraciones lógicas vistas | `revision-prospective-audit.tex`:48–60 | **VERIFICADO** para CF; **omite** que el brazo nativo vio sólo el 2,1–2,9 % (BE) |
| Veto correcto con propuestas colapsadas o reparación omitida explica dos fallos | `revision-rescues.tex`, Fig. 3, Tabla `tab:rescues` | **RECIBO**; coherente con `component-rescues.json` y con el texto histórico |
| La confirmación correcta no garantiza progreso | `revision-theory.tex` | Argumento informal; ver H |

---

## E. Comparación con nuestra versión anterior

### E.1 Tabla de cambios sustantivos

| Aspecto | Versión anterior (B) | Versión de Pedro (P) | Consecuencia científica | Evaluación |
|---|---|---|---|---|
| Título | *Learning Distributed Coloring Rules: Information, Counterfactual Fidelity, and Frozen Transfer* | *Learning Local Coloring Rules: Continuation Fidelity, Progress, and Failed Repair* | Desplaza el foco de la amplitud (información, transferencia) a la identificación y los mecanismos | Editorial razonable; pierde "transfer" y "information" como señales |
| Abstract | Pregunta "¿cuándo una política ejecuta un algoritmo y no sólo resuelve la tarea?". Cifras: 21/21; 22,64/9,27/99,99 %; 810.000 ejecuciones; 15 celdas fallidas; pruebas condicionales | Pregunta de identificación; experimento de 40 modelos con GM y BE; regla programada 99,71 %; mecanismos; cobertura como contexto final | Más centrado, pero carga en el abstract los números del experimento más débil (GM por construcción, BE nulo) | Mejora de foco; **problema** de selección de titulares (J-M1, M2) |
| Pregunta científica | "¿Bajo qué interfaces pueden aprender, qué reproducen y dónde falla la transferencia?" | "¿Puede el acuerdo perfecto ocultar una regla incorrecta?" | Pregunta más aguda y publicable; pero la "regla correcta" fuera de política es una elección de contrato, no algo identificado (lo admite P:`revision-contracts.tex`:22–23) | Mejora, con tensión conceptual (J-M4) |
| Contribuciones | Cobertura nativa; separación agreement/continuation; transferencia y fallos; resultados condicionales | Contratos; diagnóstico y ensayo prospectivo; mecanismos ↔ hipótesis de progreso | La cobertura funcional y la transferencia dejan de ser contribuciones | Pérdida de A y C como contribuciones; ganancia en B y mecanismos |
| Modelo formal | §2 "Computational Model" | Mismo texto + contratos (ec. del oráculo, tabla, ejemplo KW) | Responde a la objeción 2 | Mejora clara; detalles menores en H |
| Definición de profesores | Tabla 7 profesores | Idéntica | — | Sin cambio |
| Arquitecturas e interfaces | Texto | Texto + Figura 1 + párrafo aprendido/suministrado | Responde a la objeción 4 | Mejora clara |
| Entrenamiento y contratos | §4 | Idéntico + admisibilidad | — | Sin cambio sustantivo |
| Resultados: funcional | §5 en el cuerpo (21/21, Clopper–Pearson, negativos preservados) | Apéndice B, verbatim | Dimensión A demotada | Discutible (K) |
| Resultados: continuación | §6 | §5 + reponderación por tipo + experimento prospectivo + auditoría post-test + diagnósticos de fallo BE | Mucha más profundidad; la nueva evidencia **matiza a la baja** la ventaja de CF | Mejora de rigor; problemas nuevos de diseño (J) |
| Resultados: identificadores | §7 en el cuerpo | Apéndice C, verbatim | Pierde visibilidad la obstrucción de información | Editorial |
| Teoremas | Apéndices; "Safety and progress" en la Discusión | Apéndices idénticos byte a byte; la discusión se reubica como subsección "What Confirmation Can and Cannot Guarantee" | Sin teoremas nuevos; mejor ubicación narrativa | Neutro o mejora; argumentos siguen informales (H) |
| Generalización | §8 con figura de grilla de transferencia; 810.000 | §6 con nota de procedencia, sin figura, con intervenciones de un componente y Figura 3 | C pasa de "amplitud" a "mecanismos" | Mejora en mecanismos; pérdida de visibilidad de la transferencia exitosa (448/450 celdas sin fallo) |
| Discusión | Safety/progress; output laws; limitaciones; "freeze scope" con trabajo futuro | Reescrita en torno al ensayo prospectivo; sin trabajo futuro; incluye frases de proceso ("this workspace", "pending review") | Más autocrítica; lenguaje no apto para envío | Mixta (J-M8) |
| Limitaciones | Dispersas | Muy explícitas | Honestidad alta | Mejora |
| Conclusión | Cobertura + separación + transferencia + teoría | Acuerdo ≠ continuación; ensayo estrecho; BE empeora la completitud; veto correcto ≠ progreso | Coherente con la nueva tesis | Mejora de coherencia |

### E.2 Cambios que mejoran la precisión o la claridad

- **Contratos de oráculo** (P:`revision-contracts.tex`). Por primera vez se define qué es una continuación correcta, por profesor.
- **Figura y párrafo de arquitectura** (P:`revision-architecture.tex`). Deja explícito qué es aprendido (transformaciones y mensajes) y qué es suministrado (agenda, decodificación, actualización, cola, reloj).
- **Tres estimandos distintos sobre las mismas consultas** (ocurrencias, BA, por tipo), con baselines programados (P:`revision-type-analysis.tex`).
- **Contabilidad de pasos** (14,53/13,43 incluyen fallos; todos los éxitos toman 15). Evita un falso "speedup".
- **Intervenciones de un componente** con control de COMMIT exacto. Separan defecto de propuesta de veto incorrecto en dos casos.
- **Procedencia de los totales históricos.** Se declara que no son regenerados.

### E.3 Cambios que pueden debilitar una contribución

- **La cobertura funcional 21/21 desaparece del cuerpo.** Es la evidencia más amplia y la más fácil de comunicar.
- **Desaparece la figura de transferencia y la generalización exitosa pierde visibilidad.** 448 de 450 celdas primarias sin fallos y hasta Δ=10 o n=256 sin reentrenamiento. El abstract ya no menciona la transferencia exitosa, sólo sus fallos.
- **La obstrucción de identificadores pasa al apéndice.** El título pierde "Information".
- **Se pone en el abstract un experimento cuyo resultado más limpio (GM) está garantizado por diseño.** El resultado BE es estadísticamente nulo y además negativo en lo funcional. Esto puede leerse como debilitamiento del mensaje "CF aprende continuaciones", sin ganar a cambio una afirmación positiva fuerte.
- **Se elimina el párrafo de trabajo futuro** de B.

### E.4 Nuestra línea local paralela (contexto, no auditada en profundidad aquí)

La rama local `research/pedro-review-manuscript-revision` aborda las mismas objeciones con otra estrategia experimental. Cifras **reportadas por esa rama**, no re-auditadas en esta revisión:

- **Experimento A (presupuesto de etiquetas; BE, GM, ColorTrials; 5 semillas; 4.000 actualizaciones):**
  - BE CF 93,81–100 % BA frente a matched ≈53 % a igual número de etiquetas, en 5/5 semillas;
  - GM 100 frente a 50;
  - CT 100 frente a 99,3.
- **Experimento B:** familias de intervención retenidas.
- **Experimento C:** baselines de reglas; P∧E da 99,26 %.
- **Experimento D:** transferencia nula.
- **Experimento E:** sonda de escala ColorTrials.
- **Experimento F:** seguimiento con actividad vecina; el brazo cf-neighbor alcanza TPR 1,0 y 1.000/1.000.
- **Revisión bibliográfica** de 49 entradas nuevas en `revision-references.bib`.

**Convergencias con P:**

- GM 50→100 con ambos brazos funcionalmente perfectos;
- los brazos nativos casi nunca rechazan opciones ocupadas;
- **CF puede dañar el éxito funcional en BE** (en nuestra línea, cf-high 4.298/5.000 y cf-low 3.270/5.000).

**Divergencias:**

- **Magnitud de la ventaja BE de CF:** robusta en la nuestra, nula en la de P. Las diferencias de diseño son plausibles: mezcla de consultas, 4.000 frente a 1.500 actualizaciones, holdout de familias frente a holdout de tipos.
- **Mecanismo del daño BE.** Nuestra hipótesis era que CF aprende a rechazar cuando hay vecinos activos. La auditoría numérica de esta revisión la **refuta para los datos de P**: los 181 falsos negativos caen todos en consultas con vecinos silenciosos y en dos tipos con $\kappa=0$ (frontera de elegibilidad $a_v=q$).
- **Discrepancia interna de nuestra rama:** su protocolo dice "34/40" ejecuciones BE perfectas y los datos dan 32/40.

---

## F. Resolución de las siete objeciones

| # | Objeción | Estado en P | Evidencia y ubicación en P | Problema nuevo o residual |
|---|---|---|---|---|
| 1 | Contribución central poco clara | **Resuelta parcialmente (mejora clara)** | Pregunta y tres contribuciones (`main.tex`:32–34, 75–84); conclusión coherente (`main.tex`:517–528) | La tesis de no-identificabilidad no se formula como resultado (H-M1); el abstract prioriza un experimento débil |
| 2 | Definición incompleta de los oráculos | **Resuelta parcialmente** | Ec. (1) y tabla por profesor (`revision-contracts.tex`:2–58); clave suficiente, hash y adquisición completos para BE/GM (`revision-prospective-methods.tex`) | La distribución de intervenciones del diagnóstico **histórico** no es reimplementable: perturbaciones vecinas 0,25/0,15, variante "todos ofrecen", tasa de silencio, participación = permiso (`main.tex`:236–239 frente a `src/dcl/multicf/oracles.py`:122–147). Detalles menores (H) |
| 3 | Limitaciones de la comparación contrafactual | **Reconocida como limitación + experimento nuevo que introduce problemas nuevos** | `revision-prospective*.tex`; Discusión `main.tex`:477–491 | Control nativo con etiquetas = participación (información no igualada); novedad desigual entre brazos no reportada; endpoint BE dominado por consultas fuera de política; resultado GM por construcción; familias de intervención no retenidas (J-M1 a M3) |
| 4 | Módulos aprendidos frente a operaciones suministradas | **Resuelta** | Fig. 1 y párrafo (`revision-architecture.tex`); `main.tex`:189–207 | Falta decir que las primeras etapas pareadas son bit a bit idénticas, así que la diferencia autónoma BE se debe sólo a pesos COMMIT (fortaleza no explotada) |
| 5 | Poco protagonismo de los mecanismos de fallo (Johansson, GM, ColorTrials) | **Resuelta para Johansson y GM; parcial para ColorTrials** | Fig. 3, `revision-rescues.tex`, `tab:rescues`; `main.tex`:417–466 | ColorTrials sigue en una frase (11 celdas) y un párrafo de sonda de escala; dos witnesses seleccionados, sin tasa poblacional (declarado) |
| 6 | Conexión teoría–experimentos | **Resuelta parcialmente** | `revision-theory.tex` (subsección "What Confirmation Can and Cannot Guarantee"); enlaces a witnesses | Argumentos informales ($U_v$ indefinido; condición de reparación GM mal enunciada; Diffuse requiere $p<1$); no hay proposición sobre qué estratos de consulta afectan una ejecución (H-M2, M3) |
| 7 | Novedad frente al estado del arte | **Abierta (sin cambios)** | Párrafo de trabajos relacionados y Apéndice J idénticos a B; ningún `.bib` modificado | Nuestra línea local sí hizo una revisión bibliográfica (49 entradas), pendiente de integrar |

---

## G. Auditoría de resultados

Métodos de esta auditoría:

1. Recomputación desde los 60 JSON crudos por semilla y desde los arrays del runtime (labels, claves y split).
2. Re-inferencia de los 40 modelos congelados: 0/40 matrices de confusión discrepantes.
3. Ejecución de los tests (7 pass) y de `verify-paper-revision.py` (`verified: true`).
4. `audit-revision-continuations.py` y `build-revision-assets.py` reproducen byte a byte los archivos comprometidos.
5. Manifiesto: 235/235 entradas correctas.
6. PDF: hash correcto, 27 páginas; las 394 cifras multi-dígito del `.tex` aparecen en el PDF.
7. Recompilación con tectonic del paquete LaTeX: texto equivalente; 95 de ~14.050 palabras difieren, todas por guionado o URLs; 28 páginas con este motor frente a 27 con pdfLaTeX.

### G.1 Experimento prospectivo (P §5.x, Apéndice A)

**Protocolo reconstruido:**

- BE_BASE_D3/E1 y SPECULATIVE_GM/E1 con adaptadores residuales nativos (19.396 parámetros); semillas 105001–105010; brazos nativo y CF.
- **Datos por modelo:** 3.000 objetivos base comunes más 3.600 auxiliares.
  - Los estados auxiliares son comunes al par.
  - El brazo nativo recibe la etiqueta COMMIT nativa y CF la etiqueta intervenida: offset de opción $k=i\bmod 4$ y silencio con probabilidad 1/4 cuando $k=3$; vecinos nativos.
  - GM con cuotas 1.800 ASSIGN / 1.800 REPAIR; BE 3.600 KW.
- **Split:** 20 % de los buckets hash SHA256 de la clave suficiente $K$, con la sal fija `revision-20261030-commit-types-v1`.
  - Training: un estado se retiene sólo si sus claves nativa y CF son de entrenamiento.
  - Test: se retiene una consulta si su clave CF está retenida.
  - **El mismo split para todas las semillas.**
- **Optimización:** Adam 1e-3, batch 64, 1.500 actualizaciones, 100 de warmup matched.
- **Test:** 4.000 consultas por semilla y profesor (tamaños 28 y 36), más 1.000 ejecuciones autónomas por modelo (tamaños 32, 48 y 64) con referencia exacta 1.000/1.000 previa.

**Cronología** (UTC, VERIFICADO desde recibos y fechas del tarball; autoatestada, sin sello externo):

| Hora | Evento |
|---|---|
| 09:54:44 | Protocolo congelado (hashes de 36 fuentes coinciden con el commit) |
| 09:54:45 | Primeros datos de entrenamiento |
| 10:11:58 | 40 pesos congelados |
| 10:12:12–10:25:19 | Tests |
| 11:04, 11:23, 16:07, 16:39 | Auditorías post-test |
| 17:03 | Commit |

**Resultados:**

| Cifra | Valor en P | Recomputado | Estado |
|---|---|---|---|
| BE BA nativo / CF (media de 10 semillas) | 84,64 / 90,41 | 84,6428 / 90,4145 | VERIFICADO |
| BE Δ pareado e IC bootstrap | +5,77 [−13,73; +25,26] | +5,7717; IC con su semilla RNG [−13,732; +25,258]; exacto [−13,69; +25,26]; t [−19,75; +31,29]; sign-flip p=0,38; mediana +0,52 | VERIFICADO |
| GM BA y Δ | 50 / 100; +50 [50; 50] | idéntico en 10/10 semillas | VERIFICADO |
| Éxitos autónomos BE | 9.204 / 7.405 | idem | VERIFICADO |
| Éxitos autónomos GM | 10.000 / 10.000 | idem; pasos medios 26,0116 en ambos | VERIFICADO |
| BE por tipo nativo → CF | 51,70 → 86,94 | idem; **pareado 10/10 semillas a favor de CF** (+15 a +55 pp, p=0,002) | VERIFICADO (P sólo reporta medias) |
| Regla $G=E\,p$, BE | 99,71 BA / 81,27 por tipo | idem | VERIFICADO; CF > regla por tipo sólo en 7/10 semillas (pierde en 105001 y 105008, empata en 105009) |
| Regla $G=E\,p$, GM | 53,03 / 62,34 | idem | VERIFICADO |
| Confusión BE elegible-activo | CF 228/228 ocupadas rechazadas, 814/995 libres aceptadas; nativo 995/995 y 0/228 | idem; **los 181 FN están en 105001 (98) y 105008 (83)**; las otras 8 semillas: 100 % | VERIFICADO |
| Confusión BE elegible-silencioso | Sólo en el apéndice: CF 81 FP / 133, nativo 42 FP / 133 | idem | VERIFICADO; **CF es peor que nativo aquí** y el texto principal no lo dice |
| Etiquetas GM | nativo 4.344–4.354 positivas y 0 negativas; CF 1.038–1.098 negativas | idem | VERIFICADO |
| Soporte | BE 17–23 tipos/semilla; masa máxima 95,05–96,30 %; GM 82–92 tipos | idem | VERIFICADO |
| Solapamiento lógico con el entrenamiento CF | BE 99,475–100 %, GM 100 % | idem | VERIFICADO |
| Solapamiento lógico con el entrenamiento **nativo** | **No reportado** | **BE 2,1–2,9 %, GM 37,7–39,5 %** (`continuation-audit.json`) | Omisión relevante |
| Fracción de consultas BE con raíz inelegible | No reportada como tal | **38.644 / 40.000 = 96,61 %** (96,15–97,05 % por semilla) | Hallazgo |
| BE 105001 / 105008: BA auxiliar frente a test | 99,510 / 100 frente a 52,756 / 52,094 | Recuentos coinciden con los arrays | RECIBO (requiere inferencia; el protocolo la hizo sobre 40 modelos sin discrepancias en el test) |
| BE 105010 | 204 / 1.000 en ambos brazos; participación omitida en 13/118; witness paso 7, raíz 31 | — | RECIBO |
| Pasos BE | 14,53 / 13,43; éxitos = 15; fallos 9,0804 / 8,9576 | idem | VERIFICADO |
| Presupuesto | 6.600 objetivos por modelo; 3.000 base en cada banco | idem; inicialización, minibatches y presentaciones idénticos; 3.600 auxiliares visitados | VERIFICADO |

**Interpretación** (ver J para la severidad):

- **(i) Bimodalidad BE.**
  - Por semilla, las diferencias se agrupan en {−47, +0,5, +49,6}.
  - Las semillas nativas 105004, 105005 y 105007 colapsan a ≈50 % BA, aceptando prácticamente todas las raíces inelegibles. Suman 11.523 de los 11.688 FP pooled, y aun así logran 1.000/1.000 éxitos autónomos.
  - **Sin esas tres semillas, la diferencia media es −13,02 pp.**
  - El "+5,77" es, por tanto, el saldo entre tres colapsos inocuos del nativo y dos colapsos dañinos de CF. No es un efecto medio interpretable.
- **(ii) Endpoint dominado por consultas fuera de política.**
  - La intervención fija la participación de la raíz igual a su *permiso*, no a su elegibilidad. Así crea consultas "inelegible pero activa" que el profesor nunca produce y que son el 96,6 % del banco BE.
  - En ejecución autónoma esas consultas sólo importan si la participación aprendida falla.
- **(iii) Información no igualada.**
  - En todo objetivo COMMIT nativo la etiqueta coincide con la participación de la raíz: BE 45.440/45.440 y GM 43.468/43.468.
  - "Igual presupuesto" iguala recuentos, no información. El resultado GM 50→100 está determinado por el diseño. Pedro lo declara para GM, pero lo pone en el abstract como resultado.
- **(iv) Novedad desigual.**
  - El test es casi enteramente familiar para CF a nivel lógico y casi enteramente nuevo para el nativo, en BE.
  - La frase del abstract "The held-out input contexts mostly reuse logical oracle configurations seen in training" (P:`main.tex`:46–47) es cierta **sólo para el brazo CF**.
- **(v) Mecanismo de los FN de CF.**
  - Todos caen en consultas con vecinos silenciosos y en los dos tipos con $\kappa=0$, es decir con color auxiliar exactamente en el umbral KW.
  - Esto es inferido: las predicciones son constantes dentro de un tipo y la descomposición es única dada la cuenta de TP.
  - CF trata $\kappa=0$ como $\kappa=-1$, pese a haber visto 96–111 consultas activas con $\kappa=0$ en entrenamiento.
  - Es un error de frontera de elegibilidad, más específico que la descripción de P ("generalization across contexts").
- **(vi) Unidad de análisis.** El split es el mismo en todas las semillas, de modo que las 10 semillas replican la aleatoriedad de entrenamiento, no el muestreo de tipos. El BE de test tiene en total 29 entradas distintas a través de las semillas.

### G.2 Resultados históricos (heredados de B)

- **21 combinaciones profesor–arquitectura** (5 checkpoints × 1.000/1.000 cada uno, 105 políticas).
  - Texto verbatim en el Apéndice B de P, y `tables/closure-native21.tex` idéntico byte a byte.
  - Las cohortes son heterogéneas: 12 históricas pasivas con semillas 83101–83105 y 9 nuevas CF con semillas 94101–94105.
  - P mantiene correctamente "sin ranking". NO VERIFICABLE aquí sin los runtimes de cierre.
- **Diagnóstico BE passive / matched / CF** (60,14 / 53,25 / 99,9965 BA; 22,64 / 9,27 / 99,99 % de exactitud por ocurrencia).
  - Heredado e idéntico.
  - La nueva tabla por tipo es coherente con `be-type-counts.json`.
  - Los subconjuntos de tipos no vistos (549/459/717/522/722) son los de B.
- **Transferencia congelada: totales heredados.**
  - 808.413 / 810.000 en total; 449.791 / 450.000 en primarios; 15 celdas fallidas.
  - P lo declara honestamente: "no regenerado aquí".
  - Los "225 checkpoints recuperados" no están definidos en el texto y no cuadran con las 180 políticas del estudio de transferencia. Es una aclaración pendiente.
- **Witnesses de fallo con intervenciones:**
  - Johansson/Transformer 83101 (Δ=10, cliques): cambiar una propuesta 2→0 completa en el paso 11, frente al cutoff 128.
  - GM/Transformer 94102 (listas locales Δ=4): cambiar la propuesta de reparación 0→1 permite reparar.
  - En ambos, el control de COMMIT exacto mantiene el fallo.
  - Estado RECIBO (`component-rescues.json`, `failure-mechanisms.json`), con hashes de witnesses verificados por el generador.
  - Son dos casos seleccionados post-hoc, lo que está declarado.
- **Identificadores:** la Proposición de indistinguibilidad y la tabla JP (60/60 con comparaciones relativas) son verbatim de B, ahora en el Apéndice C.
- **Certificados condicionales:**
  - Prop. "Canonical finite-type MAX COMMIT" sin cambios.
  - "Los 45 núcleos canónicos seleccionados fallan la condición suficiente; cinco pesos E1 históricos distintos pasan certificados condicionales": sin cambios.
  - P lo relaciona correctamente con los experimentos: BE y GM usan COMMIT residual, que la proposición excluye.

---

## H. Auditoría matemática

**Inventario:**

- **Sin cambios:** todos los resultados numerados son verbatim. Los apéndices B:608–800 = P:627–819 y B:958–1000 = P:982–1024 son idénticos byte a byte. Cubren:
  - Lemma Availability y la ecuación de progreso;
  - cotas 1/4, terminación, JP/Greedy/GM y la cadena BE;
  - Prop. Diffuse proposals y Cor. Diffuse submass;
  - Counterexample;
  - Prop. MAX, MEAN/SUM y Symmetry/coupling.
- **Movidos al apéndice:** Prop. Observation indistinguishability y las cotas Clopper–Pearson/AM–GM.
- **Nuevos, todos definicionales:**
  - el oráculo $O_T$ y la tabla de contratos;
  - el ejemplo KW;
  - la clave suficiente $K$, el split hash y la adquisición;
  - la regla $G=E\,p$;
  - la suficiencia gruesa $O_T=E_v p_v(1-o_v)$.
- **Prosa reubicada, sin resultados numerados:** la subsección "What Confirmation Can and Cannot Guarantee".

**Correspondencia definición–código:**

- El oráculo de P coincide con `src/dcl/multicf/oracles.py`:79–119 para los siete profesores.
- La clave $K$ es genuinamente suficiente para el cómputo MAX de la segunda etapa de E1: con IDs y offsets con signo y mensajes MAX no negativos, la dispersión de logits dentro de un tipo es ≤ 3,8e-6 en los modelos entrenados.
- Consecuencia no explotada por P: las primeras etapas pareadas son bit a bit idénticas, así que **la diferencia autónoma BE se debe exclusivamente a los pesos COMMIT**.

**Ejemplo KW (replay):**

- La etapa 2 de Linial y todos los pasos KW se reproducen.
- La etapa 1 no se puede comprobar porque no se guardan los IDs.
- Inconsistencias menores:
  - el texto usa $c_v=1$ con $p_v=1$ como negativo activo, pero la consulta almacenada con $c=1$ es la silenciosa; los negativos activos almacenados son $c=3$ y $c=0$;
  - la indexación de pasos es inconsistente: "native step 12" es índice base 0, mientras "KW step 7" en 105010 es conteo base 1 y corresponde a la reducción KW 5.

**Problemas matemáticos, por severidad:**

- **H-M1 (MAJOR): la no-identificabilidad es el titular pero no un resultado enunciado.**
  - El rastro nativo nunca muestra una propuesta ocupada: en BE, GM y Greedy todo elegible-activo es libre, y en BE la participación nativa siempre es igual a la elegibilidad.
  - Por tanto "aceptar sii $p$" y "$E\,p$" son ambos consistentes con el rastro. El brazo nativo BE efectivamente se reparte 3/7 entre ellos según la semilla.
  - Corrección: enunciar el soporte nativo por profesor y el principio de selección que define $O_T$. Suavizar "wrong continuation responses" (P:`main.tex`:33, 518), porque "incorrecta" es relativa a un contrato elegido.
- **H-M2 (MAJOR): falta una proposición sobre qué estratos de consulta pueden afectar una ejecución.**
  - Las escrituras ocurren sólo si participación y COMMIT se activan (`multiteacher_e1/model.py`:228), y KW y la reparación GM no reintentan.
  - Así, COMMIT sobre consultas silenciosas nunca afecta una ejecución, y las consultas inelegibles-activas sólo importan si la participación falla. La aceptación en elegible-activo-libre es lo que determina la completitud.
  - Esto explica a la vez GM (+50 BA sin cambio funcional) y BE (181 FN con menos éxitos).
  - Corrección: añadir la proposición y reportar ese estrato como endpoint co-primario.
- **H-M3 (MAJOR): los argumentos de progreso y reparación (contribución 3) son informales.** En P:`revision-theory.tex`:2–21:
  - $U_v$ no está definido;
  - la frase sobre GM trata la violación de una condición *suficiente* de reparación como fallo, cuando un vecino posterior en la cola aún podría resolver el conflicto;
  - falta el lema KW: una raíz elegible que rechaza o calla rompe el invariante;
  - Prop. Diffuse requiere $p_{\max}<1$, por lo que no cubre Johansson siempre-activo.
  - Corrección: una proposición numerada con cota difusa bilateral para $p=1$, citando el Counterexample por su etiqueta.
- **H-M4 (MAJOR): el oráculo del diagnóstico histórico no es reimplementable desde el paper.** Faltan:
  - el reparto nativo participación/candidato por profesor;
  - la participación de la raíz fijada al permiso;
  - el silencio 1/4;
  - las perturbaciones vecinas de Luby, Johansson y JP (0,25; 0,15; variante "todos ofrecen el color de la raíz").

  El apéndice prospectivo sí es completo para BE y GM.

**Problemas menores:**

- Los contratos admiten listas generales $c\in C_u$, pero el código es sólo K4 (`oracles.py`:96).
- $R_{\rm JP}$ es más estricto que el propio JP.
- La fila BE de la tabla de contratos omite la intervención de participación.
- LINIAL apunta al *primer* punto no bloqueado.
- El resumen de MAX dice "every required local type", cuando la proposición exige los 64 subconjuntos de tipos.
- Se pierden de B la frase sobre equivariancia y la advertencia de que las hipótesis de Diffuse "no están establecidas para estos checkpoints". Diffuse, Submass y el Counterexample ya no se citan desde el cuerpo.
- El bootstrap es frágil con datos bimodales: conviene reportar también tests de signo (BE p≈0,11, GM p≈0,002) y medianas.
- El holdout fino prueba invariancia ante entradas que el oráculo ignora. Para eso existe un certificado finito exhaustivo computable, más fuerte que un muestreo.

**Pérdidas matemáticas:** ninguna proposición numerada perdida. Sólo se pierden frases interpretativas (las listadas arriba).

---

## I. Evaluación de la narrativa frente a A/B/C

Nuestra visión anterior organizaba el trabajo en tres dimensiones: **A. Functional success**, **B. Continuation fidelity** y **C. Generalization**.

**¿P la conserva?** Parcialmente:

- **B** pasa a ser el eje.
- **A** pasa a ser contexto (Apéndice B) y criterio secundario del ensayo nuevo.
- **C** se reduce a "mecanismos de fallo de la transferencia congelada"; la amplitud de la generalización exitosa deja de ser un resultado destacado.

**¿La precisa?** Sí, en B:

- define el objeto (contratos);
- separa estimandos;
- separa lo aprendido de lo suministrado;
- muestra que B depende del contrato elegido fuera de política.

**¿Cambia la pregunta científica?** Sí: de "qué y dónde aprenden y fallan" a "el acuerdo no identifica la regla". Es una pregunta más aguda, pero **más estrecha**.

**¿Fortalece los resultados?**

- **Fortalece la honestidad y los mecanismos.**
- **Debilita la afirmación positiva** de que la supervisión contrafactual enseña continuaciones correctas de forma robusta: a igual presupuesto, el efecto BE es nulo y funcionalmente adverso, y el de GM es tautológico por diseño.

**¿Reduce o amplía las afirmaciones?** Las reduce casi en todas partes. Añade una sola afirmación nueva: el ensayo prospectivo, también acotada.

**¿Propone una interpretación distinta?** Sí, y es su aportación conceptual principal: **la fidelidad de confirmación no implica progreso**. P la apoya con witnesses (C), con datos prospectivos (BE: mejor por tipo, peor funcionalmente) y con argumentos condicionales.

**¿Introduce nuevas contribuciones?** Sí:

- contratos explícitos;
- estimandos por tipo con baselines programados;
- intervenciones de un componente sobre witnesses;
- el ensayo prospectivo, con su auditoría.

**¿Pierde contribuciones importantes?** Pierde visibilidad de:

- la cobertura funcional amplia (21/21);
- la generalización sin reentrenamiento (448/450 celdas, Δ hasta 10, n hasta 256, pilotos hasta 4.096);
- la obstrucción de información o identificadores.

Ninguna se elimina; todas se degradan a apéndice o se quedan sin figura.

**Tensión conceptual principal.** Nuestra visión trata A, B y C como **tres dimensiones de éxito** que una política debería satisfacer conjuntamente. La de P trata B como **diagnóstico de qué regla se aprendió**, y A como contexto. Los datos de P (y los nuestros) muestran que **B puede mejorar mientras A empeora**. Esto obliga a decidir:

- **Opción (a):** la fidelidad de continuación es un objetivo deseable, y entonces hay que explicar por qué entrenarla puede dañar el comportamiento funcional.
- **Opción (b):** es una herramienta de diagnóstico. Si la extensión fuera de política es una elección de contrato y no algo identificado, "incorrecta" no es la palabra adecuada.

La síntesis que proponemos discutir es:

> A, B y C son dimensiones distintas y no monótonamente alineadas. B es identificable sólo relativo a un contrato. Los estratos de B que importan para A son los que la ejecución autónoma puede visitar.

Esta síntesis (H-M2) conserva la estructura A/B/C y además incorpora la mejor idea de P.

---

## J. Problemas científicos pendientes (priorizados)

### CRITICAL

Ninguno encontrado. No hay errores de datos, de código ni de integridad que invaliden una afirmación central tal como P la formula: las cifras son exactas y los límites están mayormente declarados.

### MAJOR (corregir antes del envío)

- **M1. Información no igualada en el control nativo; resultado GM por construcción.**
  - Problema: las etiquetas COMMIT nativas son copia de la participación (BE 45.440/45.440; GM 43.468/43.468). GM nativo nunca ve un negativo. "Equal budget" iguala recuentos, no información.
  - Ubicación: el abstract (P:`main.tex`:40–43) y la Tabla `tab:prospective` presentan GM 50→100 como resultado.
  - Solución: presentar GM como una comprobación de que CF aporta la clase ausente, no como efecto; sacarlo del abstract o reformularlo; declarar la propiedad también para BE.
- **M2. Novedad desigual entre brazos no reportada.**
  - Problema: el solapamiento lógico con el entrenamiento es de 99,5–100 % para CF frente a 2,1–2,9 % para el nativo (BE) y 37,7–39,5 % (GM).
  - Ubicación: la frase del abstract (P:`main.tex`:46–47) es engañosa.
  - Solución: reportar ambas cifras y reformular la frase.
- **M3. Endpoint BE no interpretable tal como está resumido.**
  - Problema:
    - el 96,6 % son consultas inelegibles-activas creadas por la intervención;
    - hay unas 20 entradas efectivas por semilla y el mismo split en todas;
    - la distribución es bimodal: +5,77 descansa en tres colapsos nativos inocuos y, sin ellos, da −13,02;
    - los 181 FN de CF son un error de frontera $\kappa=0$ concentrado en dos semillas.
  - Solución: reportar el estrato elegible-activo como co-primario, junto con medianas, tests de signo y la descripción de la bimodalidad. Mantener explícitamente que +5,77 es un resumen poco informativo.
- **M4. La no-identificabilidad no está formalizada; "wrong/incorrect rule" es relativo al contrato** (H-M1).
- **M5. Falta la proposición sobre qué estratos afectan la ejecución autónoma** (H-M2). Es la pieza que convertiría la tesis de P en un resultado y unificaría GM y BE.
- **M6. Argumentos de progreso y reparación informales** (H-M3).
- **M7. Oráculo del diagnóstico histórico no reimplementable** (H-M4).
- **M8. Lenguaje de proceso y procedencia en el manuscrito.**
  - Ejemplos:
    - "The forty original Pedro weights are available in this workspace" (P:`main.tex`:510);
    - "Independent scientific review of this revision remains pending" (P:`main.tex`:515);
    - "native ARM LaTeX… old x86 Tectonic" (P:`main.tex`:1078–1079);
    - "receipts", "revision evidence" y "in the revision evidence" (P:`revision-type-analysis.tex`:29).
  - Son impropios de un artículo y algunos no son verificables fuera de la máquina de Pedro. Deben ir a un README de reproducibilidad.
- **M9 (proceso). Dos reescrituras completas divergentes del mismo `main.tex`.** P está publicada; la nuestra local está sin commit y sin compilar. Hay que decidir la base antes de seguir redactando.

### MINOR

- **Interpretación y reporte:**
  - CF es peor que nativo en raíces elegibles silenciosas (81 frente a 42 FP); sólo aparece en el apéndice.
  - CF supera a la regla programada por tipo sólo en 7/10 semillas; P sólo da medias.
  - La ventaja pareada por tipo (10/10 semillas, p=0,002) es el resultado BE más sólido y el texto no la explota.
- **Diagnóstico de fallos:** la caracterización de los fallos CF 105001/105008 puede ser más precisa (frontera $\kappa=0$, vecinos silenciosos).
- **Ejemplo KW y numeración:** el ejemplo usa $c_v=1$ y la indexación de pasos no es uniforme.
- **Documentación de Pedro:**
  - `PAPER_REVISION_REPORT.md`:97 contiene una afirmación obsoleta ("Figura 4 compacta se conserva");
  - el recibo `original-workspace-integrity.json` nombra HEAD `6836d855` y no `c3affd16`; es menor, porque el archivo base es idéntico a B.
- **Procedencia y preregistro:**
  - "225 checkpoints recuperados" no está definido y no cuadra con 180 políticas de transferencia;
  - el preregistro es autoatestado, sin sello externo; no hay registro de la sonda de tiempo que fijó 1.500 actualizaciones; el código de análisis y la semilla del bootstrap no están congelados (no cambia nada);
  - anomalía de CPU en BE 105003/cf (553,6 s de reloj y 65,3 s de CPU, frente a ≈11 s).
- **Contratos:** los detalles menores listados en H.
- **Discusión:** frases perdidas sobre equivariancia y sobre las hipótesis de Diffuse.
- **Mecanismos:** ColorTrials sigue sin un mecanismo de fallo presentado con igual detalle.
- **Código:** cambio de comportamiento: la búsqueda de checkpoints de `sizedegree` ahora exige que todos estén presentes.

### EDITORIAL (elección razonable entre autores)

- El título y el orden de las dimensiones.
- Bajar al apéndice la cobertura 21/21 y los identificadores.
- Retirar la figura de transferencia.
- El abstract es muy denso en cifras.
- El título de la subsección "What Confirmation Can and Cannot Guarantee" promete garantías que no enuncia.
- La eliminación del párrafo de trabajo futuro.

---

## K. Recomendaciones

**Conservar de Pedro:**

- contratos de oráculo y ejemplo KW, una vez corregidos;
- figura y párrafo aprendido/suministrado;
- tabla de estimandos por tipo y baselines programados, en el diagnóstico histórico y en el prospectivo;
- contabilidad de pasos;
- Figura 3 y las intervenciones de un componente, con su tabla en el apéndice;
- la subsección sobre lo que la confirmación no garantiza, una vez formalizada;
- la declaración honesta de que los totales históricos no se regeneran;
- el apéndice prospectivo completo, como modelo de especificación reimplementable.

**Modificar:**

1. Reformular el experimento prospectivo (M1–M3):
   - GM como sanity check de la clase ausente;
   - BE con el estrato elegible-activo como co-primario, junto con la bimodalidad y el resultado pareado por tipo;
   - reportar el solapamiento del brazo nativo.
   - Valorar si pertenece al cuerpo o a un apéndice con un párrafo de síntesis.
2. Añadir dos resultados formales cortos: soporte nativo y no-identificabilidad (M4), y qué estratos afectan una ejecución (M5).
3. Formalizar el progreso y la reparación (M6) y completar el oráculo histórico (M7).
4. Restituir en el cuerpo una versión breve de A (cobertura 21/21, una tabla o frase) y de C (una figura compacta de transferencia o una frase en el abstract sobre 448/450 celdas sin fallos).
5. Eliminar el lenguaje de proceso (M8).

**Discutir entre coautores:**

- La tesis: identificación (P) frente a tres dimensiones (B), o la síntesis propuesta en §I.
- Si el artículo afirma algo positivo sobre la supervisión contrafactual o se limita a diagnosticar.
- Cómo integrar nuestra línea local, cuyos datos y mecanismos aún no están auditados aquí:
  - Experimento A: presupuesto con ColorTrials;
  - Experimento B: familias de intervención retenidas;
  - Experimento F: actividad vecina;
  - revisión bibliográfica para la objeción 7.
- Qué hacer con la discrepancia BE entre ambas líneas: robusta en la nuestra y nula en la de P.
- Qué rama es la base del texto (M9).
- Si se diseña el control "negativos informativos a igual presupuesto" que el propio Pedro sugiere. Implica un protocolo y un test nuevos, fuera del alcance de esta etapa.

---

## L. Evidencia y referencias

Índice completo: `handoff/pedro-changes-independent-review/EVIDENCE_INDEX.md`. Fuentes principales:

- **Manuscritos:**
  - P: `git show 5993a5e4:paper/ucgnn/main.tex` y `paper/ucgnn/revision-*.tex`.
  - B: `git show c3affd16:paper/ucgnn/main.tex`.
  - Diff: `manuscript-pedro/main.tex.diff-vs-c3affd16.patch` en el paquete.
- **Resultados de Pedro** (`results/paper-revision-20261030/` en `5993a5e4`):
  - `prospective-summary.json`, `continuation-audit.json`, `be-failure-diagnostics.json`;
  - `component-rescues.json`, `failure-mechanisms.json`, `be-step-accounting.json`;
  - `protocol-frozen.json`, `weights-frozen.json`, `verification.json`, `third-audit-checks.json`, `manifest.json`;
  - crudos `prospective/{be_base_d3,speculative_gm}/s1050NN/{cf-test,native-test,teacher-reference}.json` (≈224 MB, no incluidos en el paquete; están en git).
- **Runtime:** `results/paper-revision-20261030/paper-revision-runtime.tar.gz`, sha256 `9a33d4bb8bd8b621fd35fd877e8c42f8afaa43f311c5ca8c4abda316551cec8a`, en git y no incluido en el paquete.
- **Código:** `src/dcl/paperrevision/{experiment,keys}.py`, `scripts/{audit-revision-continuations,probe-revision-be-failures,build-revision-assets,verify-paper-revision}.py`, y el oráculo histórico `src/dcl/multicf/oracles.py`.
- **Scripts y salidas de esta auditoría:** `verification/` del paquete.
  - `numbers/01–10`: tabla por semilla, bootstrap, sensibilidad, estratos, auditoría de arrays, rangos por fase, FN por tipo, tipos históricos y frontera κ.
  - `math/`: replay del ejemplo KW, Linial y números.
  - `protocol/`: tiempos del tarball, logs del audit y del builder.

**Afirmaciones que sólo pueden verificarse con datos no incluidos o ausentes:**

- **Totales de transferencia de B** (810.000 filas crudas): no publicadas, no disponibles en esta máquina.
- **21/21 de cobertura:** requiere los runtimes del cierre.
- **Pesos históricos de Pedro:** "40 disponibles" sólo en el workspace de Pedro.
- **Etapa 1 del ejemplo KW:** faltan IDs.
- **Diagnósticos post-test de 105001/105008/105010 y witnesses Johansson/GM:** verificables con el runtime y los checkpoints recuperados; aquí quedan como RECIBO.

---

## M. Preguntas para la segunda revisión (ChatGPT)

1. **Tesis.** ¿Es la pregunta "¿el acuerdo perfecto puede ocultar una regla incorrecta?" una contribución publicable por sí misma, dado que la regla fuera de política es una elección de contrato (P:`revision-contracts.tex`:22–23)? ¿O debe reformularse como "el rastro nativo no identifica la extensión; diferentes procedimientos eligen extensiones distintas"?
2. **Síntesis A/B/C.** ¿Es correcta la síntesis propuesta en §I? Es decir: dimensiones no monótonamente alineadas, B relativo a un contrato y estratos de B relevantes para A. ¿Mejora la tesis de P?
3. **GM 50→100 %.** Dado que las etiquetas COMMIT nativas son copia de la participación, ¿tiene este resultado algún valor evidencial más allá de una comprobación de diseño? ¿Debe estar en el abstract?
4. **BE +5,77 pp.**
   - ¿Es defendible reportar la media pareada como endpoint primario con una distribución bimodal {−47, +0,5, +49,6}?
   - ¿Qué resumen recomendarías: estrato elegible-activo, medianas, tests de signo o resultado por tipo pareado (10/10)?
5. **Novedad del test.** ¿Cómo cambia la interpretación del ensayo el hecho de que el test sea familiar lógicamente para CF (99,5–100 %) y casi nuevo para el nativo (2–3 %)? ¿Es el contraste, en el fondo, "entrenar en la distribución de test frente a no hacerlo"?
6. **Proposición de estratos (H-M2).** Verificar el argumento: escritura sólo si participación ∧ COMMIT, y sin reintento en KW ni en reparación GM. ¿Basta para afirmar que la aceptación en elegible-activo-libre determina la completitud? ¿Bajo qué supuestos?
7. **Argumentos de progreso.** Revisar P:`revision-theory.tex`:
   - definición de $U_v$;
   - la afirmación GM sobre la condición de reparación (¿suficiente o necesaria?);
   - aplicabilidad de Prop. Diffuse a Johansson con $p=1$.
8. **Witnesses.**
   - ¿Son las intervenciones de un componente (propuesta frente a COMMIT exacto) un diseño causal válido para distinguir "defecto de propuesta" de "veto incorrecto" en un estado?
   - ¿Qué falta para extraer algo poblacional?
9. **Fallos CF BE 105001/105008.** El error es la frontera $\kappa=0$ (color auxiliar en el umbral KW) con vecinos silenciosos, pese a haber visto ejemplos de $\kappa=0$ en entrenamiento. ¿Qué hipótesis (capacidad, optimización, desbalance de clases dentro de la frontera) son compatibles y cuáles se podrían discriminar con artefactos existentes?
10. **Pérdidas.** ¿Es aceptable bajar al apéndice la cobertura 21/21, la generalización sin reentrenamiento y la obstrucción de identificadores? ¿Qué mínima presencia en el cuerpo recomendarías?
11. **Divergencia entre líneas.** Nuestra línea local reporta una ventaja BE robusta de CF (5/5 semillas, ≈+40 pp a igual presupuesto) y P una nula. ¿Qué diferencias de diseño podrían explicarlo, y cuál de los dos diseños responde mejor a la objeción 3?
    - Diferencias candidatas: mezcla de consultas, 4.000 frente a 1.500 actualizaciones, holdout de familias frente a holdout de tipos, y supervisión de "inelegible forzada".
12. **Novedad frente al estado del arte (objeción 7).** P no la modifica. Con la tesis de P, ¿cuál es el precedente más cercano (Hint-ReLIC, causal confusion en imitación, DAgger, no-hint reasoning) y qué distingue realmente este trabajo?
13. **Editorial.** ¿Qué fragmentos del manuscrito de P (M8) deben salir del texto científico? ¿Es adecuado el nivel de autocrítica para una venue exigente o resulta defensivo?
