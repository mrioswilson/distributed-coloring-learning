# Informe final: paletas locales e identificabilidad

Estado: COHORTE COMPLETA.
18,848/18.848 evaluaciones de grafos; 73 estructuras
distintas en 76 especificaciones; 3,255 leyes condicionales estimadas;
1,260 celdas de clasificación. GPU-proceso acumulada: 3.0239 h / 12 h.
Estados de ejecución: {'success': 14694, 'progress_cutoff': 2892, 'unsafe_failure': 1262}. No entrenamiento nuevo; no push remoto.

1. **Paletas locales sin Δ.** Los resultados medidos por tratamiento son:

- normalized: LOCAL 760/760, GLOBAL 760/760; rondas medias 9.03 vs 8.86; autorizaciones inseguras LOCAL 0.
- raw_endpoints: LOCAL 637/760, GLOBAL 637/760; rondas medias 54.21 vs 54.18; autorizaciones inseguras LOCAL 0.
- sum: LOCAL 160/760, GLOBAL 160/760; rondas medias 203.58 vs 203.62; autorizaciones inseguras LOCAL 0.
- mean: LOCAL 431/760, GLOBAL 431/760; rondas medias 8.43 vs 8.31; autorizaciones inseguras LOCAL 18727.
- deepsets: LOCAL 760/760, GLOBAL 760/760; rondas medias 9.17 vs 9.02; autorizaciones inseguras LOCAL 0.
- transformer: LOCAL 760/760, GLOBAL 760/760; rondas medias 9.78 vs 9.82; autorizaciones inseguras LOCAL 0.
- teacher: LOCAL 152/152, GLOBAL 152/152; rondas medias 8.97 vs 8.97; autorizaciones inseguras LOCAL 0.

2. **Cambio frente a GLOBAL.** Las líneas anteriores comparan la misma matriz de
grafos/cintas. `regime-summary.csv` añade sobredimensionamiento 1.5 y 2;
`paired-comparisons.csv` contiene contrastes emparejados y bootstrap por estructura
real. Las rondas incluyen el corte 256. No se interpreta el cutoff como timeout.

3. **Mecanismos.** El código usa representaciones compartidas por color y consulta
el estado de cada vecino en los colores válidos del receptor. DeepSets y Transformer
reciben c/(K_v−1) e inversa de su propio K: sí tienen información de orden. MAX
conserva el conjunto finito de tipos de COMMIT cuando cambian las listas;
SUM/MEAN conservan multiplicidades y no heredan esa reducción. Estos hechos
arquitectónicos y los certificados de signos se distinguen de explicaciones
causales no identificadas por comparar sólo tasas de éxito.

4. **Distancia a Luby en los mismos estados.** Esta tabla resume primero por red
y estado, después sobre cinco réplicas. Las E1 originales son un grupo separado.

```text
  treatment   k  networks       tv  fidelity  effective_colors  max_mass  forbidden_mass
   deepsets   4         5 0.006293  0.995313          3.999034  0.254004             0.0
   deepsets  51         5 0.138417  0.191821         33.475094  0.108496             0.0
   deepsets 201         5 0.164151  0.021167         63.164472  0.106464             0.0
e1-original   4         5 0.003583  0.998535          3.999692  0.252271             0.0
e1-original  51         5 0.045875  0.353857         48.292984  0.036786             0.0
e1-original 201         5 0.063895  0.037598        173.183752  0.024146             0.0
 normalized   4         5 0.005402  0.997778          3.999332  0.253864             0.0
 normalized  51         5 0.031652  0.445605         49.970335  0.023383             0.0
 normalized 201         5 0.046439  0.052710        194.645756  0.008148             0.0
transformer   4         5 0.004993  0.994458          3.999418  0.254034             0.0
transformer  51         5 0.259850  0.092700         18.314257  0.181128             0.0
transformer 201         5 0.290935  0.020313         23.708436  0.180060             0.0
```

5. **Fidelidad frente a ley.** La tabla diferencia F con r compartido de TV sin
observar r. El control analítico de cuantil desplazado tiene F=0 y TV=0: ninguna
muestra de candidatos puede distinguir esas leyes idénticas. Una baja fidelidad
del modelo no determina por sí sola su distancia distribucional.

6. **Número de candidatos observados.** `sample-requirements.csv` da, por red y
estado, el primer m de {1,5,20,100,500,2000} con potencia observada ≥.8 y FPR
observado ≤.05. No alcanzarlo se registra como pendiente de potencia, no igualdad.
`distinguishability.csv` contiene exactitud balanceada, potencia, FPR e intervalos;
son pruebas plug-in con estimación/calibración/test disjuntos, no reglas Bayes
exactas de una ley desconocida. Los puntos de cada curva usan prefijos anidados.

7. **Paleta y disponibilidad.** `law-summary.csv` y los CSV por estado separan
ALL_FREE, SINGLE, bloques, alternados, dispersos y estados autónomos. No existe
un m universal para todos esos contextos. Los límites conservadores de TV se
ensanchan con K y pueden impedir declarar equivalencia a tolerancia .05 incluso
cuando la estimación puntual es pequeña. No se confunde esto con desigualdad
probada ni con falta de éxito de coloración.

8. **Distancia y velocidad.** No hay una ordenación necesaria: se demuestra un
contraejemplo de leyes deterministas con el mismo TV/entropía y probabilidades
de confirmar 2^(-K) o 1/2 según solapamientos. `clique-progress.csv` compara
uniforme, ley aprendida estimada y mezcla 50:50 en el mismo estado. Los experimentos
de estados heterogéneos comparan la fórmula con 256 rondas independientes y
COMMIT exacto/aprendido. Las correlaciones de rollouts no establecen causalidad.

9. **Qué está demostrado.** Disponibilidad k_v≥t_v+1; seguridad del profesor exacto;
confirmación ≥1/4 y cola n(3/4)^t; fidelidad distinta de igualdad de ley; equivalencia
de kernels bajo igualdad conjunta de acciones; relación Bayes–TV, cotas producto
y límites de incertidumbre; transferencia condicional de un certificado MAX a
listas heterogéneas. Para los 20 modelos MAX primarios, hashes y certificados
previos se verificaron, con 1.280 testigos locales adicionales. Las afirmaciones
float32 conservan sus hipótesis explícitas de redondeo y ausencia de TF32/overflow.
No hay demostración universal de progreso neuronal.

10. **Tesis del paper.** El experimento separa conocimiento global del grado,
política candidata y regla de seguridad. Una política útil puede diferir del
profesor punto a punto y en distribución. La evidencia nueva debe presentarse
con sus denominadores y dominios finitos; los negativos no se reoptimizan.

Limitaciones: cinco redes por tratamiento; 76 especificaciones con familias
deterministas, estructuras repetidas y cinco controles previamente examinados;
matriz de grados finita; sondas K=501 locales, no grandes rollouts de grado 500;
incertidumbre MC y LRT estimada; estados autónomos seleccionados por regla fija
pero no muestreo uniforme del espacio de estados. Las trazas autónomas guardan
resúmenes de diversidad/colisiones, y los estados controlados histogramas completos.
Los casos no ejecutados y fallos técnicos permanecen en execution-manifest.json.

Reproducción: [entrada y comandos](../../LOCAL_PALETTES.md),
[protocolo](../../docs/LOCAL_PALETTES_STATISTICAL_PROTOCOL.md),
[teoría y pruebas](../../docs/LOCAL_PALETTES_AND_DISTRIBUTIONS_THEORY.md),
[paper fuente](../../paper/ucgnn/main.tex). Los CSV, figuras y tablas se generan
con `python -m dcl.localpalettes.finalize`; no se copiaron cifras manualmente.
