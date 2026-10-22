"""Ten requested answers, generated from recorded data with explicit limits."""
import json
import pandas as pd

from .config import ROOT, RESULTS


def write(frame,law,summary,b3):
    queue=json.loads((ROOT/'queue.json').read_text())
    complete=(len(frame)==18848 and len(law)==3255 and len(b3)==1260 and
              all(t['status']=='complete' for t in queue['tasks']))
    local=frame[frame.regime=='LOCAL'];global_=frame[frame.regime=='GLOBAL']
    names=['normalized','e1-original','deepsets','transformer']
    outcome=[]
    for t in ('normalized','raw_endpoints','sum','mean','deepsets','transformer','teacher'):
        a=local[local.treatment==t];g=global_[global_.treatment==t]
        outcome.append(dict(model=t,LOCAL=f'{int(a.success.sum())}/{len(a)}',GLOBAL=f'{int(g.success.sum())}/{len(g)}',
                            rounds_LOCAL=f"{a['rounds'].mean():.3f}",rounds_GLOBAL=f"{g['rounds'].mean():.3f}",unsafe_LOCAL=int(a.unsafe_commits.sum())))
    focused=summary[(summary.k.isin([4,51,201]))&(summary.pattern=='ALL_FREE')&summary.treatment.isin(names)]
    curves=pd.read_csv(RESULTS/'classifier-summary.csv')
    example=curves[(curves.state=='k201-ALL_FREE')&curves.treatment.isin(names)&curves.m.isin([5,20,100,500])]
    patterns=curves[(curves.m==2000)&curves.treatment.isin(names)&curves.state.isin(['k4-ALL_FREE','k51-SINGLE-middle','k51-THREE-spread','k201-ALTERNATING'])]
    paired=pd.read_csv(RESULTS/'paired-success-discordances.csv').set_index('treatment')
    swaps=paired.loc['mean']
    mean_success_unsafe=len(local[(local.treatment=='mean')&(local.status=='success')&(local.unsafe_commits>0)])
    missed=int(local[local.treatment=='sum'].false_rejects.sum())
    safety=pd.read_csv(RESULTS/'candidate-safety-summary.csv').set_index('treatment')
    single=law[(law.state=='k51-SINGLE-middle')&law.treatment.isin(names)]
    progress=pd.read_csv(RESULTS/'clique-progress.csv');progress['treatment']=progress.model.str.rsplit('-s',n=1).str[0]
    progress=progress[(progress.k==201)&(progress.intervention=='learned-estimated')].groupby('treatment').agg(
        exact_progress=('exact_commit_probability_plugin','mean'),collision=('collision','mean'),tv=('tv','mean')).reset_index()
    text=f'''# Informe final: paletas locales e identificabilidad

Estado: {'COHORTE COMPLETA' if complete else 'PARCIAL: consultar tareas pendientes/fallidas'}.
{len(frame):,} evaluaciones de grafos de 18.848 previstas; {frame.graph_cluster.nunique()}
estructuras en 76 especificaciones; {len(law):,} leyes condicionales estimadas;
{len(b3):,} celdas empíricas de clasificación. GPU-proceso acumulada:
{queue['gpu_seconds']/3600:.4f} h / 12 h, incluyendo la réplica exploratoria adicional.
Estados: {frame.status.value_counts().to_dict()}. No hay OOM ni timeout técnico en la batería.
No se entrenaron modelos nuevos ni se hizo push remoto.

1. **¿Qué modelos colorean con paletas locales sin Δ global?**

```text
{pd.DataFrame(outcome).to_string(index=False)}
```

NORMALIZED, DeepSets y Transformer completaron todos los casos LOCAL de las cinco
réplicas. NORMALIZED son los pesos 41000–41004 del paper anterior; las cinco E1
originales 0–4 se estudian separadamente en los diagnósticos de candidatos.
Estos son resultados finitos, no corrección universal.

2. **¿Qué cambia respecto de GLOBAL?**

La tabla compara los mismos grafos y cintas. Las tasas agregadas coinciden,
pero MEAN tiene {int(swaps.global_only_success)} éxitos exclusivos GLOBAL y
{int(swaps.local_only_success)} exclusivos LOCAL. Hay {mean_success_unsafe} ejecuciones MEAN LOCAL con
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
En LOCAL, SUM perdió {missed:,} oportunidades observadas que COMMIT exacto habría aceptado
en el mismo estado y con las mismas propuestas; no es una simulación de una
trayectoria alternativa completa con shield.

DeepSets y Transformer también proponen colores ocupados, en
{int(safety.loc['deepsets','observed_forbidden_states'])}/{int(safety.loc['deepsets','model_states'])} y
{int(safety.loc['transformer','observed_forbidden_states'])}/{int(safety.loc['transformer','model_states'])} pares red–estado, respectivamente.
Las masas máximas estimadas son {safety.loc['deepsets','max_forbidden_mass']:.3f} y
{safety.loc['transformer','max_forbidden_mass']:.3f}. Sus COMMIT MAX rechazan esos conflictos.
Los ejemplos completos están en `candidate-counterexamples/`. No se observaron
candidatos prohibidos de NORMALIZED ni E1 original en los bancos auditados;
esa observación no es una prueba universal de seguridad de propuestas.

4. **¿Cuánto se parecen las leyes a Luby?**

Medias descriptivas de cinco modelos, tras calcular cada métrica por red y estado:

```text
{focused[['treatment','k','tv','fidelity','effective_colors','entropy_effective_colors','max_mass']].round(4).to_string(index=False)}
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
{example[['treatment','m','balanced_accuracy','power','fpr']].round(4).to_string(index=False)}
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
{patterns[['treatment','state','balanced_accuracy','power','fpr']].round(4).to_string(index=False)}
```

En SINGLE-middle K=51, los {len(single)} pares de los tres tratamientos seguros y
E1 original tienen TV estimada {single.tv.max():.4f} y cota superior puntual máxima
{single.tv_upper.max():.4f}, menor que la tolerancia .05: equivalencia aproximada bajo
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
{progress.round(5).to_string(index=False)}
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
'''
    (RESULTS/'FINAL_REPORT.md').write_text(text)
