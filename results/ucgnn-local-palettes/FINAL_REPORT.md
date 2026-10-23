# Informe final: paletas locales e identificabilidad

Estado: COHORTE COMPLETA.
18,848 evaluaciones de grafos de 18.848 previstas; 73
estructuras en 76 especificaciones; 3,255 leyes condicionales estimadas;
1,260 celdas empíricas de clasificación. GPU-proceso acumulada:
3.0353 h / 12 h, incluyendo la réplica exploratoria adicional.
Estados: {'success': 14694, 'progress_cutoff': 2892, 'unsafe_failure': 1262}. No hay OOM ni timeout técnico en la batería.
No se entrenaron modelos nuevos ni se hizo push remoto.

1. **¿Qué modelos colorean con paletas locales sin Δ global?**

```text
        model   LOCAL  GLOBAL rounds_LOCAL rounds_GLOBAL  unsafe_LOCAL
   normalized 760/760 760/760        9.025         8.859             0
raw_endpoints 637/760 637/760       54.213        54.182             0
          sum 160/760 160/760      203.579       203.618             0
         mean 431/760 431/760        8.426         8.308         18727
     deepsets 760/760 760/760        9.172         9.018             0
  transformer 760/760 760/760        9.782         9.821             0
      teacher 152/152 152/152        8.974         8.974             0
```

NORMALIZED, DeepSets y Transformer completaron todos los casos LOCAL de las cinco
réplicas. NORMALIZED son los pesos 41000–41004 del paper anterior; las cinco E1
originales 0–4 se estudian separadamente en los diagnósticos de candidatos.
Estos son resultados finitos, no corrección universal.

2. **¿Qué cambia respecto de GLOBAL?**

La tabla compara los mismos grafos y cintas. Las tasas agregadas coinciden,
pero MEAN tiene 23 éxitos exclusivos GLOBAL y
23 exclusivos LOCAL. Hay 33 ejecuciones MEAN LOCAL con
coloración final propia y autorizaciones contrarias a la regla exacta:
un rollout correcto no certifica COMMIT. Los cuatro regímenes y todos los
intervalos/contrastes están en `regime-summary.csv`, `paired-comparisons.csv`
y `cluster-uncertainty.csv`. Rondas incluyen cutoff 256; no es timeout técnico.
También incluyen terminaciones impropias: el tiempo de MEAN no debe interpretarse
como velocidad de una coloración correcta.

3. **¿Qué mecanismos explican las diferencias?**

Las representaciones compartidas permiten consultar cada color válido del
receptor con los mismos pesos. DeepSets y Transformer reciben c/(K_v−1) y 1/K_v:
sí tienen orden funcional. Sus mensajes no contienen posición ni K del vecino.
MAX reduce COMMIT a 64 conjuntos legales también con listas heterogéneas.
Los certificados muestran que las cinco SUM rechazan un testigo seguro de grado
8 y las cinco MEAN aceptan conflictos en testigos de grados 8 y 16. Son errores
de estos pesos: SUM puede representar la regla exacta con otros parámetros.
En LOCAL, SUM perdió 6,271,641 oportunidades observadas que COMMIT exacto habría aceptado
en el mismo estado y con las mismas propuestas; no es una simulación de una
trayectoria alternativa completa con shield.

DeepSets y Transformer también proponen colores ocupados, en
13/465 y
13/465 pares red–estado, respectivamente.
Las masas máximas estimadas son 0.152 y
0.559. Sus COMMIT MAX rechazan esos conflictos.
Los ejemplos completos están en `candidate-counterexamples/`. No se observaron
candidatos prohibidos de NORMALIZED ni E1 original en los bancos auditados;
esa observación no es una prueba universal de seguridad de propuestas.

4. **¿Cuánto se parecen las leyes a Luby?**

Medias descriptivas de cinco modelos, tras calcular cada métrica por red y estado:

```text
  treatment   k     tv  fidelity  effective_colors  entropy_effective_colors  max_mass
   deepsets   4 0.0063    0.9953            3.9990                    3.9995    0.2540
   deepsets  51 0.1384    0.1918           33.4751                   43.6044    0.1085
   deepsets 201 0.1642    0.0212           63.1645                  143.0618    0.1065
e1-original   4 0.0036    0.9985            3.9997                    3.9998    0.2523
e1-original  51 0.0459    0.3539           48.2930                   49.4982    0.0368
e1-original 201 0.0639    0.0376          173.1838                  189.4573    0.0241
 normalized   4 0.0054    0.9978            3.9993                    3.9997    0.2539
 normalized  51 0.0317    0.4456           49.9703                   50.1475    0.0234
 normalized 201 0.0464    0.0527          194.6458                  195.7433    0.0081
transformer   4 0.0050    0.9945            3.9994                    3.9997    0.2540
transformer  51 0.2598    0.0927           18.3143                   33.4871    0.1811
transformer 201 0.2909    0.0203           23.7084                   93.3039    0.1801
```

TV y fidelidad son diferentes. `effective_colors` es 1/Σq²;
`entropy_effective_colors` es exp(H). No se comparan las dos definiciones como
si fueran la misma métrica. Los intervalos de TV y métricas completas por estado
están en `candidate-laws.csv`; sus límites son puntuales salvo indicación contraria.

5. **¿Qué cambia al esconder r?**

Con r compartido se mide imitación de una función acoplada, no igualdad de ley.
El cuantil desplazado del control analítico tiene fidelidad cero y ley uniforme
idéntica: ningún número de candidatos independientes lo distingue del profesor.
Las redes pueden tener baja fidelidad y una ley bastante cercana al uniforme.
Igualdad de candidatos marginales tampoco implica igualdad de trayectorias si
cambia la participación, la dependencia entre nodos o COMMIT.

6. **¿Cuántos candidatos hay que observar?**

Ejemplo all-free K=201; medias sobre cinco redes por grupo:

```text
  treatment   m  balanced_accuracy  power    fpr
   deepsets   5             0.7566 0.5688 0.0461
   deepsets  20             0.9238 0.9000 0.0508
   deepsets 100             0.9992 1.0000 0.0523
   deepsets 500             1.0000 1.0000 0.0398
e1-original   5             0.5828 0.1695 0.0461
e1-original  20             0.7273 0.3250 0.0500
e1-original 100             0.9352 0.8156 0.0578
e1-original 500             0.9973 1.0000 0.0609
 normalized   5             0.5363 0.0586 0.0453
 normalized  20             0.6359 0.1023 0.0430
 normalized 100             0.8727 0.6258 0.0492
 normalized 500             0.9723 0.9711 0.0461
transformer   5             0.8848 0.8109 0.0422
transformer  20             0.9848 0.9938 0.0477
transformer 100             1.0000 1.0000 0.0461
transformer 500             1.0000 1.0000 0.0461
```

Son resultados del LRT plug-in con estimación, calibración y test independientes,
sin observar r, logits o activaciones. La tasa FPR de 256 ensayos fluctúa; el
nivel nominal .05 está garantizado marginalmente sobre calibración independiente,
no para toda realización individual. No se recalibró después de ver el test.
`sample-requirements.csv` conserva el primer m con potencia observada ≥.8 Y FPR
observado ≤.05. No cumplir ese criterio conjunto puede deberse al FPR observado,
aunque la potencia sea alta; no implica indistinguibilidad ni siempre falta de potencia.
Los intervalos por red están en `distinguishability.csv`. No son certificados
Bayes exactos de una ley neuronal desconocida.

7. **¿Cómo depende de K y del patrón disponible?**

Estos contrastes a m=2000 ilustran la dependencia del estado:

```text
  treatment             state  balanced_accuracy  power    fpr
   deepsets  k201-ALTERNATING             1.0000 1.0000 0.0570
   deepsets       k4-ALL_FREE             0.5387 0.0836 0.0492
   deepsets k51-SINGLE-middle             0.5000 0.0000 0.0000
   deepsets  k51-THREE-spread             0.9477 0.9086 0.0445
e1-original  k201-ALTERNATING             1.0000 1.0000 0.0398
e1-original       k4-ALL_FREE             0.5145 0.0562 0.0500
e1-original k51-SINGLE-middle             0.5000 0.0000 0.0000
e1-original  k51-THREE-spread             0.5008 0.0641 0.0406
 normalized  k201-ALTERNATING             1.0000 1.0000 0.0414
 normalized       k4-ALL_FREE             0.4949 0.0438 0.0516
 normalized k51-SINGLE-middle             0.5000 0.0000 0.0000
 normalized  k51-THREE-spread             0.5000 0.0539 0.0383
transformer  k201-ALTERNATING             1.0000 1.0000 0.0602
transformer       k4-ALL_FREE             0.5344 0.0727 0.0539
transformer k51-SINGLE-middle             0.5000 0.0000 0.0000
transformer  k51-THREE-spread             0.9762 0.9648 0.0500
```

En SINGLE-middle K=51, los 20 pares de los tres tratamientos seguros y
E1 original tienen TV estimada 0.0000 y cota superior puntual máxima
0.0242, menor que la tolerancia .05: equivalencia aproximada bajo
ese criterio, no igualdad exacta. En algunos SINGLE-last del Transformer hay
muchas propuestas prohibidas. `singleton-product-laws.csv` usa la fórmula exacta
TV(P^m,Q^m)=1−(1−ε)^m en la masa verdadera desconocida ε; sus valores estimados
incluyen intervalos binomiales simultáneos. Es un análisis analítico exploratorio,
no nuevos ensayos empíricos B3. Cero observaciones no prueba probabilidad cero.

8. **¿Más parecido a Luby significa más rápido?**

No es una implicación matemática. El contraejemplo de leyes deterministas con
idéntico TV/entropía tiene probabilidades de confirmar 2^(-K) o 1/2 según los
solapamientos; la segunda familia es una construcción matemática por nodo, no
una implementación determinista anónima de nodos inicialmente idénticos.
En la clique all-free K=201, p=1/2 y COMMIT exacto, los valores plug-in son:

```text
    treatment  exact_progress  collision      tv
     deepsets         0.27094    0.01858 0.16415
  e1-original         0.29503    0.00598 0.06390
         mean         0.29991    0.00514 0.04616
   normalized         0.29938    0.00514 0.04644
raw_endpoints         0.00029    0.24971 0.97942
          sum         0.22728    0.18264 0.28860
  transformer         0.23799    0.04751 0.29093
```

El uniforme exacto da 0.30383. RAW conserva unos cuatro colores efectivos y
casi pierde progreso; Transformer es fácilmente distinguible, pero conserva
progreso comparable aquí y colorea la cohorte. La mezcla 50:50 con uniforme
está en `clique-progress.csv`. `heterogeneous-distribution-progress.csv` compara
marginales y solapamientos en el MISMO estado completo. Las correlaciones de
rollouts no establecen causalidad.

La mayor discrepancia original de la fórmula motivó una réplica independiente,
explícitamente post hoc (`progress-replication.json`). Se conservaron los 256
ensayos originales y se añadieron 1.024; el SE mostrado cubre observaciones,
con incertidumbre adicional en los marginales predictivos. No se presenta como
una prueba de igualdad de las probabilidades empíricas y teóricas.

9. **¿Qué está demostrado y qué sigue siendo empírico?**

Demostrados bajo hipótesis explícitas: |A_v|≥t_v+1; seguridad del profesor exacto;
confirmación ≥1/4 y cola n(3/4)^t; separación entre fidelidad y ley; equivalencia
de kernels bajo igualdad conjunta de acciones; relación Bayes–TV y cotas producto;
fórmula producto singleton; condiciones suficientes de progreso; transferencia
de certificados MAX a listas heterogéneas. Se verificaron los hashes de los 20
MAX principales y 1.280 testigos locales. Los contraejemplos concretos SUM/MEAN
tienen intervalos en aritmética real y condicionalmente float32. Las hipótesis
numéricas incluyen cotas de error afín, IEEE y ausencia de TF32/overflow.
No hay demostración universal de progreso o terminación de las redes.

10. **¿Cómo cambia la tesis del paper?**

La generalización observada permite eliminar conocimiento global de Δ sin
reentrenar tres de los tratamientos estudiados. La fidelidad al profesor y
la uniformidad exacta de candidatos no son requisitos empíricos de ese éxito.
Una política segura puede tolerar errores de propuestas gracias a COMMIT
aprendido; eso no garantiza progreso en todos los estados. Las tasas finitas
no identifican necesidad universal de MAX o de una arquitectura particular.

Limitaciones: cinco redes por tratamiento; 76 especificaciones con 73 estructuras,
controles ya examinados y familias deterministas; matriz finita de grados;
K=501 es una sonda local, no un rollout de grado 500; estimación MC y LRT plug-in;
estados autónomos seleccionados por regla fija, no muestreo uniforme del espacio
de estados. El bootstrap de diferencias puede degenerar en cohortes sin fallos;
se añaden límites conservadores por clusters y no se interpreta [0,0] como
equivalencia poblacional. Los resultados exactos, estadísticos y exploratorios
están separados. Se conservaron todos los negativos, sin optimizar pesos para ocultarlos.

Reproducción y acceso: [entrada](../../LOCAL_PALETTES.md),
[protocolo](../../docs/LOCAL_PALETTES_STATISTICAL_PROTOCOL.md),
[teoría](../../docs/LOCAL_PALETTES_AND_DISTRIBUTIONS_THEORY.md),
[manuscrito PDF](../../paper/ucgnn/main.pdf). Código fuente genera tablas y figuras
desde los CSV y arrays archivados; los bancos completos de clasificación tienen
hashes en `candidate-array-manifest.json`. Todos los cambios son locales.
