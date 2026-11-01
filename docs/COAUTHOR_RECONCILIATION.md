# Coauthor reconciliation

Date: 31 October 2026. This is an integration decision record, not a claim of
author agreement or independent human review. Textual base: Pedro
`5993a5e4782d44670926224263688943a1411ebf`; additional evidence: Martín
`d155d3ed104d10a015fa1061e7a5fc7f5ebcdb7b`; common ancestor:
`c3affd16ac5654c6e2a700d2312710753adb9518`.

**Actualización tras la entrega del release:** se verificaron los tres paquetes
y todos sus 1.988 miembros. Los datos antes ausentes están disponibles:
220 referencias de checkpoint, 205 rutas congeladas distintas y bancos de los
105 entrenamientos; también las 810.000 filas históricas. Se repitieron
50.173.920 decisiones de continuación sin discrepancias binarias y se
reconstruyeron todas las coloraciones históricas finales. Véase
[auditoría de entrega](MARTIN_ARTIFACT_DELIVERY_AUDIT.md). La matriz inicial
conserva su evaluación anterior a esa recepción, no describe faltantes vigentes.
También coinciden todos los resúmenes de 120.000 ejecuciones autónomas
repetidas. Las 240 rutas crudas del contraste adicional que se identificaron
como faltantes ya están autenticadas mediante la cuarta entrega:240.000 filas,
cobertura completa y todos los resúmenes/contadores coincidentes. Los recibos
iniciales se conservan como historia; no quedan datos pendientes de esas
entregas. No se ha realizado una nueva campaña neuronal de240.000 casos.

The user's 31 October instruction authorizes the integration and necessary
experiments. No shared branch is merged, no result is overwritten, and no
publication or push is authorized. `results/integrated-paper-20261031/sources.json`
records the hashes of 261 imported files. Pedro's source/PDF are preserved as
`paper/ucgnn/archive/pre-integration-5993a5e4.*`. Martín's published canonical
manuscript is still the common ancestor; his uncommitted rewrite is not present
and is not treated as a completed second manuscript.

## Phase 1: objection and decision matrix

This initial matrix was written before editing central manuscript claims and
before the newly supplied artifact release. Facts,
reported results, interpretations, and editorial choices are distinguished.
Machine-readable verification and subsequent decisions are added separately.

| Objection | Assessment | Evidence | Integration decision |
|---|---|---|---|
| Functional success, continuation fidelity and generalization should be visible | Agree with qualification | Pedro main.tex §§2,6, Appendix B; native-certifications.json; generalization-cells.csv | Define functional success and contract-relative fidelity; define transfer of either property rather than treating generalization as a logically independent output variable. Restore compact native coverage to the body. |
| GM 50→100 BA is not general CF superiority | Correct; already acknowledged by Pedro | prospective-summary.json; continuation-audit.json; oracle code | Keep the fixed primary result and its negative-label explanation; remove its role as the abstract's main positive result. Equal label counts do not equal informative support. |
| GM result is guaranteed/tautological by design | Overstatement in Claude's report | Native labels contain no negative ASSIGN/REPAIR confirmations | Lack of negative data explains the comparison; it does not mathematically force a trained network's extrapolation or CF's 100% score. |
| BE +5.77 pp is a null result | Incorrect wording; uncertainty is broad | Ten paired outcomes and bootstrap [-13.73,+25.26] pp | Preserve the mean and interval as the registered endpoint. Describe heterogeneity, not proof of zero effect. Do not remove adverse seeds to estimate a preferred effect. |
| BE type accuracy improves in every pair while autonomous completion falls | Verified in prior read-only reanalysis | prospective-summary.json; 98+83 false negatives in seeds105001/105008 | Report occurrence BA, equal-type accuracy, seed outcomes, eligible/free and occupied strata, silent errors, and functional results together. |
| Native logical overlap is omitted | Correct | continuation-audit.json already contains both arms | Report BE native2.125–2.9% vs CF99.475–100%; GM native37.725–39.5% vs CF100%. Interpret support acquisition, not novel-rule extrapolation. |
| 96.6% of BE queries are ineligible-active | Approximately correct but conflates two counts | Raw banks: 38,644 ineligible, of which38,526 active, out of40,000 | Give exact denominators when used. Ineligible-active errors can matter if PARTICIPATE errs; silent-root COMMIT has no direct write effect. |
| Eligible-active stratum should become coprimary | Scientifically useful, not retroactively coprimary | The primary endpoint was fixed before test | Mark the added strata as post-test. Prespecify endpoints only for a genuinely new study. |
| A formal native-support statement is needed | Agree | Contract definition and native oracle implementations | Add an elementary proposition with a hypothesis class and admissible witness; do not present finite observed agreement as agreement on every reachable input. |
| A transition-relevance proposition is needed | Agree | native_prediction: P∧COMMIT; ColoringProgram.apply; invariant_after | Restrict to color phases, account for permissions and decoding; include KW's immediate shrinking-bound consequence and GM's possible later repair. |
| GM skipped repair always causes failure | Too strong | Serial queue can later repair another endpoint | State a sufficient repair condition and an unresolved-edge suffix condition; give a legal three-node counterexample to necessity of every scheduled repair. |
| Diffuse progress misses always-active Johansson | Correct scope limitation | Existing p_max<1 hypothesis | Extend the argument using a bound on each neighbor's per-color offer probability; do not assert learned checkpoints meet that bound. Define U_v. |
| Contracts lack historical sampling details | Correct | multicf/oracles.py proposal_bank | Specify root offsets, silence, neighbor resampling/flipping, forced equal offers, coherent fields, K4 oracle domain, JP extension and first Linial point. |
| Positive transfer has disappeared | Partly editorial | Pedro body retains449,791/450,000 and18-combination table | Keep it in body/abstract, along with16 all-pass combinations and two failed checkpoint-condition cells; separate exploratory high-degree/large-size pilots. |
| New Martín BE advantage contradicts Pedro | Not established | Protocols and endpoints differ; Martín has high BA already at1500 updates, failed unseen-type criterion, and functional deterioration | Keep campaigns separate; do not attribute the discrepancy to update count or one mechanism without control. |
| Neighbor-message follow-up explains Pedro's181 FN | False for these models | All181 are κ=0 with no participating neighbor; independently queried frozen checkpoints | Include Martín's neighbor-context defect and Pedro's boundary defect as distinct mechanisms. |
| New follow-up F supplies positive functional evidence | Supported by published summaries; new-weight replay not yet available | cf-followup/analysis.json and20 functional summaries; fixed-label neighbor perturbation | Include as a separate, post-v1-motivated follow-up with fresh v2 battery and5 seeds; distinguish reported inference from independently checked arithmetic. |
| Historical mechanisms rely on only two witnesses | New evidence extends coverage | Martín failed-cell-classification.json:15 cells,1,587 failed observations | Incorporate complete classification as a retrospective replay, not a fresh test or population theorem. No rescue fraction is inferred from classification alone. |
| Generalization to held-out late KW occupied queries is new-rule generalization | Not established | BE queried COMMIT reads signed offset/phase and message types, not raw bound B | Explain the holdout at the generator level; effective-type/logical novelty requires separate evidence. |
| New protocol reports34/40 all-pass BE models | Incorrect count | The40 published functional summaries have8 nonperfect models | Correct derived reporting to32/40; retain the original protocol as historical evidence. |
| Bibliography needs a deeper comparison | Agree | LITERATURE_REVIEW.md and revision-references.bib | Verify selected close precedents from primary sources; cite LOCAL instruction learning, causal abstraction/intervention training, SALSA-CLRS, discrete transition verification and recent failure analysis. |
| Process language belongs outside the article | Agree | Workspace/engine/review-status sentences in Pedro's discussion and reproduction appendix | Move machine-specific availability/build history to reproducibility docs; retain a concise scientific artifact limitation. |

## Scope decisions

The article studies learned decision modules inside explicit supplied interpreters.
It does not claim to learn schedules, finite-field arithmetic, general distributed
complexity bounds, arbitrary lists, universal correctness, or a generally superior
CF acquisition method. Seven scoped teachers and three palette architectures
provide functional context under heterogeneous protocols, not a ranking.

No new informative-negative training campaign is needed for this contribution:
we do not make the claim it would identify. A new functional diagnostic instead
intervenes on frozen BE components over the existing final batteries. Its protocol
is frozen before its own replay outcomes are computed; it is explicitly post-test
and selected by historical failures. All negative results will be retained.

The integrated theory and experimental reports will state which items are
proved, independently recomputed, reported with hashes, or still unavailable.

## Cierre de la integración

Se completaron las pruebas explícitas de soporte y relevancia, la condición
de progreso que admite P=1, el contraejemplo GM y la especificación histórica
de consultas. El nuevo prefijo KW completo establece un testigo alcanzable
bajo IDs legales. Las 330 regresiones pertinentes y esa prueba adicional pasan.

Los 40 modelos reproducen sus 160.000 métricas de continuación. Se rehicieron
las claves/etiquetas de los 20 pares y se validaron 466 filas adicionales.
Los replays de componentes completaron 30.000 ejecuciones, con 1.799 rescates
CF solo por COMMIT y 796 solo por P; C sola no rescata. Un observador distingue
84 errores contractuales activos sin cambio de estado. Las quince celdas
históricas se reprodujeron completas (15.000 casos), no solo sus testigos.

El artículo integrado recupera cobertura y transferencia positivas y usa
una narrativa unificada. La bibliografía compara precedentes cercanos y el
PDF compila sin referencias pendientes ni cajas desbordadas. Los seis
documentos de la entrega separan hechos, inferencias y limitaciones.

No se consideran resueltas por una frase las dos cuestiones experimentales
fuertes: superioridad ante otros negativos informativos y generalización a
familias de intervención realmente nuevas. No se afirman. Tampoco está
resuelta una comparación alternativa de adquisición por el hecho de repetir
los modelos. La ausencia inicial de los 105 runtimes de Martín y de las filas
históricas sí se cierra mediante descarga, hashes e inferencia/reconstrucción
efectivas; no por añadir una frase de disponibilidad. La revisión externa y
acuerdo de autoría continúan pendientes, como consta en el handoff final.
