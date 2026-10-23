"""Data-derived report; primary scientific inferences remain frozen."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *

ARCHES=('normalized','deepsets','transformer')
NAMES=dict(normalized='E1 NORMALIZED',deepsets='DeepSets',transformer='LocalPaletteTransformer')
def ci(r):return f"{r['estimate']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"

def witness_label(e):
    if e['kind']=='threshold':return f"color del vértice {e['ids'][0]} ≤ {e['threshold']:.6g}"
    if e['kind'] in ('structural','structural_threshold'):
        threshold=e.get('threshold',e.get('event_threshold'))
        return f"{e['name']} ≤ {threshold:.8g}"
    if e['kind']=='categorical':
        count=sum(value>0 for value in e['scores'])
        return f"patrón en vértices {e['ids']} entre {count} tuplas seleccionadas"
    return e['kind']

def intervention_summary():
    path=OUT/'candidate-intervention/results.json'
    if not path.exists():return None
    r=read(path);rows=r.get('rows',[])
    return dict(status=r['status'],cells=len(rows),rows=rows,
                equivalent=sum(x['approximate_equivalence'] for x in rows),
                effects=sum(x['intervention_tv_effect_lower']>0 for x in rows),
                unsafe=sum(x['unsafe_commits'] for x in rows))

def facts():
    f=verify_frozen();summary=read(OUT/'summary.json');audit=read(OUT/'audit.json')
    assert audit['passed']
    fc=read(OUT/'factorial-contrasts.json');events=read(OUT/'event-differences.json')
    tests={stage:read(OUT/f'{stage}.json') for stage in f['hypothesis_families']}
    target=[r for r in events if r['stratum']=='main' and r['n']==512 and r['delta']==100 and r['architecture']=='normalized' and r['event']=='used_le_delta']
    return f,summary,audit,fc,events,tests,target

def main():
    f,s,a,fc,events,tests,target=facts();primary=tests['main'];w=read(OUT/'witnesses.json')
    def cell(arch,n,d,event='used_le_delta'):
        return next(r for r in fc if r['event']==event and r['group']==arch and r['kind']=='cell' and r['contrast']==f'n{n}-d{d}')
    def count(arch,kind,event='used_le_delta'):
        rows=[r for r in fc if r['group']==arch and r['event']==event and r['kind']==kind]
        return sum(r['ci'][0]>0 for r in rows),sum(r['ci'][1]<0 for r in rows)
    ca=cell('normalized',128,50,'uses_top_color')['ci'];cb=cell('normalized',1024,50,'uses_top_color')['ci']
    top_reduction_lower=max(0.,max(0.,ca[0],-ca[1])-max(abs(cb[0]),abs(cb[1])))
    method=['| Arquitectura | V2 | MMD antiguo | Sólo V2 | Sólo MMD |','|---|---:|---:|---:|---:|']
    for arch in ARCHES:
        r=next(r for r in s['counts'] if r['architecture']==arch and r['stratum']=='main')
        method.append(f"| {NAMES[arch]} | {r['robust']}/{r['comparisons']} | {r['old']}/{r['comparisons']} | {r['robust_only']} | {r['old_only']} |")
    strata_tables={}
    for stage in ('secondary','historical-replication'):
        tt=['| Arquitectura | V2 | MMD antiguo |','|---|---:|---:|']
        for arch in ARCHES:
            r=next(r for r in s['counts'] if r['architecture']==arch and r['stratum']==stage)
            tt.append(f"| {NAMES[arch]} | {r['robust']}/{r['comparisons']} | {r['old']}/{r['comparisons']} |")
        strata_tables[stage]=tt
    secondary_top=['| n | Δ | P_L(T) | D_T E1 | D_T DeepSets | D_T Transformer |',
                   '|---:|---:|---:|---:|---:|---:|'];secondary_rows=[]
    for g in sorted([g for g in graphs() if g['stage']=='secondary'],key=lambda g:(g['n'],g['delta'])):
        cells=[];teacher=None
        for arch in ARCHES:
            rr=[r for r in events if r['graph']==g['id'] and r['architecture']==arch and r['event']=='uses_top_color']
            assert len(rr)==5
            if any(r['difference'] is None for r in rr):
                cells.append('No estimable');continue
            estimate=float(np.mean([r['difference'] for r in rr]));lo=float(np.mean([r['ci'][0] for r in rr]));hi=float(np.mean([r['ci'][1] for r in rr]));teacher=rr[0]['probability_Luby']
            secondary_rows.append(dict(graph=g['id'],n=g['n'],delta=g['delta'],architecture=arch,
                                       degree_mean=float(np.mean(g['degrees'])),degree_variance=g['degree_variance'],
                                       probability_Luby=teacher,probability_model=float(np.mean([r['probability_model'] for r in rr])),
                                       difference=estimate,ci=[lo,hi],confidence_at_least=.975,
                                       interval_method='Linear image of the original simultaneous elementary CP family; shared teacher enters once'))
            cells.append(f'{estimate:+.3f} [{lo:+.3f},{hi:+.3f}]')
        teacher_text=f'{teacher:.3f}' if teacher is not None else 'No estimable'
        secondary_top.append(f"| {g['n']} | {g['delta']} | {teacher_text} | "+' | '.join(cells)+' |')
    atomic_json(OUT/'secondary-top-color-summary.json',secondary_rows)
    pd.DataFrame(secondary_rows).to_csv(OUT/'secondary-top-color-summary.csv',index=False)
    table=['| n | Δ=3 | Δ=10 | Δ=50 | Δ=100 |','|---:|---:|---:|---:|---:|']
    for n in NS:table.append('| '+str(n)+' | '+' | '.join(ci(cell('normalized',n,d)) for d in DS)+' |')
    target_table=['| Grafo | P_L(A) | Rango P_E1(A) | Menor cota TV de A entre cinco E1 | V2 |','|---|---:|---:|---:|---:|']
    target_both=[r for r in events if r['stratum']=='main' and r['n']==512 and r['delta']==100 and r['architecture']=='normalized']
    best_target_lower=min(max(q['tv_lower'] for q in target_both if q['graph']==r['graph'] and q['model']==r['model']) for r in target)
    for gid in sorted(set(r['graph'] for r in target)):
        rows=[r for r in target if r['graph']==gid];p=[r['probability_model'] for r in rows]
        reject=sum(r['reject'] for r in primary if r['graph']==gid and r['architecture']=='normalized')
        target_table.append(f"| {gid} | {rows[0]['probability_Luby']:.3f} | {min(p):.3f}–{max(p):.3f} | {min(r['tv_lower'] for r in rows):.3f} | {reject}/5 |")
    interactions=[]
    for arch in ARCHES:
        for event in ('used_le_delta','uses_top_color'):
            rows=[r for r in fc if r['group']==arch and r['event']==event and r['kind']=='interaction']
            corner=next(r for r in rows if r['contrast']=='full-corners')
            interactions.append(f"- {NAMES[arch]}, {'A' if event=='used_le_delta' else 'T'}: {sum(r['significant'] for r in rows)}/10 interacciones; cuatro esquinas {ci(corner)}.")
    witnesses=[]
    examples=[]
    for arch in ARCHES:
        rows=[r for r in w if r['stratum']=='main' and r['architecture']==arch and r['reject']]
        kinds={name:sum(r['witness_name']==name for r in rows) for name in set(r['witness_name'] for r in rows)}
        witnesses.append(f"- {NAMES[arch]}: {len(rows)} testigos; "+', '.join(f'{k} ({v})' for k,v in sorted(kinds.items(),key=lambda x:-x[1])[:8])+'.')
        candidates=[r for r in w if r['stratum']=='main' and r['architecture']==arch]
        if candidates:examples.append(max(candidates,key=lambda r:r['tv_lower']))
    atomic_json(OUT/'representative-heldout-witnesses.json',examples)
    witness_examples=[]
    for r in examples:
        witness_examples.append(f"- **{NAMES[r['architecture']]}**, `{r['graph']} / {r['model']}`: evento «{witness_label(r['event'])}». Holdout Luby={r['count_Luby']}/{r['n_Luby']}={r['probability_Luby']:.3f}; red={r['count_model']}/{r['n_model']}={r['probability_model']:.3f}; diferencia {r['difference']:+.3f}, intervalo simultáneo [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}], TV≥{r['tv_lower']:.3f}.")
    color=pd.read_csv(OUT/'color-usage.csv');lookup={(r.graph,r.model,int(r.color)):r for r in color.itertuples()};cr=[]
    for r in target:
        candidates=[]
        for c in range(101):
            x=lookup[(r['graph'],'Luby',c)];y=lookup[(r['graph'],r['model'],c)]
            cx=ast.literal_eval(x.appearance_ci);cy=ast.literal_eval(y.appearance_ci)
            lo,hi=cy[0]-cx[1],cy[1]-cx[0]
            candidates.append(dict(graph=r['graph'],model=r['model'],color=c,p_Luby=x.appearance_probability,p_model=y.appearance_probability,
                                   difference=y.appearance_probability-x.appearance_probability,ci=[lo,hi],tv_lower=max(0.,lo,-hi)))
        cr.append(max(candidates,key=lambda x:x['tv_lower']))
    pd.DataFrame(cr).to_csv(OUT/'target-selected-color-events.csv',index=False)
    best=max(cr,key=lambda r:r['tv_lower'])
    probes=pd.read_csv(OUT/'candidate-probe-summary.csv');initial=probes[(probes['round']==0)&probes['graph'].str.startswith('main-')&(probes['n']==512)]
    pt=['| Checkpoint E1 | q_grid(top), K=51 | q_grid(top), K=101 |','|---|---:|---:|']
    for spec in [r for r in specs() if r['treatment']=='normalized']:
        rows=initial[initial['model']==spec['label']]
        pt.append(f"| {spec['label']} | {rows[rows['delta']==50].iloc[0]['q_top']:.6f} | {rows[rows['delta']==100].iloc[0]['q_top']:.6f} |")
    initial_s4=initial[(initial['model']=='normalized-s41004')&(initial['delta']==100)].iloc[0]['q_top']
    final_s4=np.mean([r['probability_model'] for r in events if r['stratum']=='main' and r['n']==512 and r['delta']==100 and r['model']=='normalized-s41004' and r['event']=='uses_top_color'])
    th=[r for r in events if r['stratum']=='historical-replication' and r['architecture']=='normalized' and r['event']=='used_le_delta']
    replicas=['| Checkpoint E1 | Media P_E1(A), 3 grafos | Rango entre grafos | Menor cota TV de A |',
              '|---|---:|---:|---:|']
    for spec in [r for r in specs() if r['treatment']=='normalized']:
        rr=[r for r in target if r['model']==spec['label']]
        pp=[r['probability_model'] for r in rr]
        replicas.append(f"| {spec['label']} | {np.mean(pp):.3f} | {min(pp):.3f}–{max(pp):.3f} | {min(r['tv_lower'] for r in rr):.3f} |")
    architecture_effects=['| Arquitectura | D_A(256,50) | D_A(512,100) | D_T(128,100) |',
                          '|---|---:|---:|---:|']
    for arch in ARCHES:
        architecture_effects.append(f"| {NAMES[arch]} | {ci(cell(arch,256,50))} | {ci(cell(arch,512,100))} | {ci(cell(arch,128,100,'uses_top_color'))} |")
    iv=intervention_summary();iv_text=('La intervención exploratoria no produjo bancos evaluables.' if not iv else f"Estado de la intervención exploratoria: `{iv['status']}`; no produjo celdas evaluables.")
    iv_short='Los registros de propuestas y COMMIT son descriptivos.'
    if iv and iv['cells']:
        iv_table=['| Arquitectura | Celdas | TV(I,L)≤0,05 respaldada | TV(original,I)>0 respaldada | Rango cotas superiores TV(I,L) |',
                  '|---|---:|---:|---:|---:|']
        for arch in ARCHES:
            rr=[r for r in iv['rows'] if r['architecture']==arch]
            if rr:
                bounds=[r['conditional_tv_upper'] for r in rr]
                iv_table.append(f"| {NAMES[arch]} | {len(rr)}/20 | {sum(r['approximate_equivalence'] for r in rr)} | {sum(r['intervention_tv_effect_lower']>0 for r in rr)} | {min(bounds):.3f}–{max(bounds):.3f} |")
        iv_text=f'''### Intervención exploratoria separada

Se reemplazó únicamente el canal de candidatos por la propuesta exacta local
del profesor, sobre el estado de la política intervenida I. Se conservaron
sus pesos, participación y COMMIT originales, sin escudo. El plan se versionó
antes de obtener datos de esta intervención; su tamaño se eligió sólo por
coste. Estado: `{iv['status']}`, {iv['cells']}/60 celdas, tres grafos nuevos
n512-d100 y el histórico, con las quince redes.

{chr(10).join(iv_table)}

Cada pareja I–Luby comparte números aleatorios; parejas distintas son
independientes. No se aplicó un test de dos muestras independientes a esas
parejas. La desigualdad de acoplamiento y cotas binomiales simultáneas dan
las cotas superiores de TV condicional. Cotas de eventos originales y nuevos
dan los efectos inferiores TV(original,I), con cobertura conjunta al menos
95%. Se observaron {iv['unsafe']} COMMIT inseguros, sin corregirlos.

Un límite superior ≤0,05 respalda equivalencia aproximada sólo de I con Luby.
Un límite superior grande puede reflejar un acoplamiento poco ajustado; no
demuestra discrepancia residual. El efecto controlado del canal de propuestas
es distinto de identificar mediación completa o separar grado de paleta.
Estos resultados no cambian ninguna de las 720 hipótesis originales.
'''
        iv_short=f"La intervención separada en candidatos respaldó un efecto sobre la ley completa en {iv['effects']}/{iv['cells']} celdas y TV(I,Luby)≤0,05 en {iv['equivalent']}/{iv['cells']}."
    supplement=OUT/'candidate-intervention/witness-effects.json'
    if supplement.exists() and read(supplement)['rows']:
        sw=read(supplement);st=['| Arquitectura | Celdas | Efecto TV(original,I)>0 | Mayor cota inferior del efecto |',
                              '|---|---:|---:|---:|']
        for arch in ARCHES:
            rr=[r for r in sw['rows'] if r['architecture']==arch]
            if rr:st.append(f"| {NAMES[arch]} | {len(rr)}/20 | {sum(r['intervention_tv_effect_lower']>0 for r in rr)} | {max(r['intervention_tv_effect_lower'] for r in rr):.3f} |")
        iv_text+=f'''
### Suplemento exploratorio con los testigos originales

El addendum `{sw['addendum_commit']}` se registró después de inspeccionar
ejemplos originales y antes de cualquier muestra intervenida. Conserva el
único evento elegido por el procedimiento original en cada una de las 60
celdas; no lo cambia usando la intervención. Reestima sus probabilidades en
el holdout original y combina la cota TV(original,Luby) con la cota de
acoplamiento TV(I,Luby), mediante la desigualdad triangular. Evalúa además
el mismo evento bajo I y obtiene una cota directa TV(original,I). Ambas
cotas pertenecen a esta misma familia simultánea, por lo que se toma su máximo.

{chr(10).join(st)}

Las 360 probabilidades (seis por celda) tienen cobertura simultánea al menos
95% mediante CP y unión de eventos. Ésta es una familia distinta de la tabla
anterior: no se toma el máximo entre sus cotas ni se afirma cobertura conjunta
95% sobre ambas familias. La figura 12 utiliza este suplemento. No se añadieron
simulaciones ni se modificaron los 720 contrastes principales. Las celdas con
cota cero se conservan y no prueban ausencia de efecto.
'''
        iv_short+=f" La familia suplementaria de testigos respaldó un efecto en {sw['effect_positive']}/{sw['cells_completed']} celdas, con su propia cobertura simultánea del 95%."
    models=read(OUT/'exploratory-binomial-models.json');predict=[]
    for arch in ARCHES:
        rows=sorted([r for r in models if r['architecture']==arch and r['event']=='used_le_delta'],key=lambda r:r['mean_heldout_cross_entropy'])
        predict.append(f"- {NAMES[arch]}: `{rows[0]['form']}`, pérdida por ejecución {rows[0]['mean_heldout_cross_entropy']:.5f}; se archivan las seis formas y los tres folds.")
    prediction_table=['| Forma predictiva para A | E1 | DeepSets | Transformer |',
                      '|---|---:|---:|---:|']
    for form in ('size','degree','ratio','density','additive','interaction'):
        vals=[next(r for r in models if r['architecture']==arch and r['event']=='used_le_delta' and r['form']==form) for arch in ARCHES]
        prediction_table.append('| '+form+' | '+' | '.join(f"{r['mean_heldout_cross_entropy']:.5f}"+(' (no convergió)' if not r['converged'] else '') for r in vals)+' |')
    np_,nm=count('normalized','size');dp,dm=count('normalized','degree');join=lambda x:'\n'.join(x)
    text=f'''# Tamaño, grado y generalización distribucional: estudio factorial

Se completó la matriz principal de **16 celdas y 48 grafos regulares nuevos**,
con tres grafos por celda y los quince checkpoints históricos. Se realizaron
500 intentos nuevos por generador para el contraste y 125 adicionales e
independientes para testigos. El estrato secundario conserva ocho grafos
irregulares; el histórico replica el mismo grafo n512-d100 con nuevas semillas.
No hubo entrenamiento nuevo ni push.

## Diseño e interpretación

Cada observación es una coloración completa etiquetada. Las listas son siempre
locales, C_v={{0,...,d(v)}}. El estimando principal condiciona a completar
correctamente antes de 128 rondas; los cuatro estados operacionales se
conservan por separado. No se mezclaron checkpoints ni se modificó COMMIT.

El omnibus V2 es idéntico al validado: selección/evaluación separadas,
permutaciones conjuntas, B=19.999, empates inclusivos y p=(1+excedencias)/20.000.
Su resolución 0,00005 permite Holm para las 720 hipótesis principales. Los
120 contrastes irregulares y 15 históricos tienen familias separadas, al igual
que el MMD antiguo, los eventos y los testigos: no hay una garantía conjunta
del 5% sobre su unión. Los atajos conservadores se refinan cuando pueden
afectar el conjunto de rechazos de Holm; un p=1 abreviado no se interpreta
como una estimación precisa de la cola.

Los dos eventos preregistrados son **A={{U≤Δ}}** y **T={{aparece el color Δ}}**.
A significa que falta algún color de la paleta, no necesariamente el máximo.
Se estiman sus probabilidades y diferencias D=P_M−P_L. El modelo binomial
saturado conserva una probabilidad por grafo/generador, condicionado al número
de completaciones válidas. Los promedios de arquitectura dan pesos iguales a
los cinco checkpoints y tres grafos fijos; el profesor compartido entra una vez.

Se preregistraron 1.800 contrastes de celda, tamaño, grado e interacción.
CP simultáneo recibe presupuesto de fallo 2,5%, y Hoeffding ponderado otro
2,5%; la intersección proporciona cobertura simultánea al menos 95%. Son
intervalos de la cohorte fija, no de una población de infinitos grafos o
entrenamientos. Un evento en un grafo aporta una cota TV inferior, nunca una
cota superior. Comparar cotas inferiores entre tamaños no ordena sus TV reales.

## Contrastes globales sobre los mismos bancos nuevos

{join(method)}

Cada arquitectura tiene 240 comparaciones principales. Se conservan ganancias
y pérdidas frente al MMD antiguo. No se interpreta esta tabla como distancia
ni se compara directamente con los 265 casos por arquitectura históricos.

## P1. ¿Aumenta la diferencia E1–Luby con n a grado fijo?

No se establece un aumento general de la TV. Los efectos observables
dependen del grado y del evento.

Para A, los doce contrastes adyacentes de tamaño dieron **{np_} incrementos y
{nm} decrementos** con intervalo simultáneo que excluye cero. Los efectos
completos E1 se muestran abajo; las figuras incluyen además T, cada grafo y
las cinco réplicas. Son cambios en una probabilidad observable, no simplemente
en la potencia del test. La evidencia no establece una ley monótona de TV.

{join(table)}

El evento T aporta información distinta. A grado 50, su diferencia pasó de
{ci(cell('normalized',128,50,'uses_top_color'))} en n=128 a
{ci(cell('normalized',1024,50,'uses_top_color'))} en n=1024: el sesgo observable
se atenúa en ese régimen. A grado 100, el cambio de A entre n256 y n512 es
{ci(next(r for r in fc if r['group']=='normalized' and r['event']=='used_le_delta' and r['contrast']=='n512-n256|d100'))}.
Estos comportamientos dependen del observable y no ordenan las TV completas.
De las bandas simultáneas de las dos celdas se deduce conservadoramente una
reducción de al menos {top_reduction_lower:.3f} en |D_T| entre n128 y n1024
a grado 50, para esta cohorte fija. Es una implicación de las bandas ya
registradas; no es una comparación de TV entre espacios de distinta dimensión.

## P2. ¿Aumenta con el grado a tamaño fijo?

Se observan aumentos en algunos riesgos, sin identificar una ley monótona de
TV ni separar causalmente grado y paleta.

Para A, los doce contrastes adyacentes de grado dieron **{dp} incrementos y
{dm} decrementos** respaldados simultáneamente. Todos los signos y no
detecciones aparecen en `factorial-contrasts.csv` y en las figuras 2 y 4.
En regulares, cambiar Δ cambia también K=Δ+1: sus efectos causales separados
no están identificados. La respuesta debe leerse por evento y régimen.
En n128-d100, A tiene diferencia {ci(cell('normalized',128,100))}, mientras
T tiene diferencia {ci(cell('normalized',128,100,'uses_top_color'))}. Que falte
algún color bajo ambas leyes no implica que falten los mismos colores.

## P3. ¿Existe interacción entre n y Δ?

Sí, para contrastes de probabilidades en esta cohorte. Por ejemplo, la
interacción E1 de A entre n256/n512 y Δ50/Δ100 es
{ci(next(r for r in fc if r['group']=='normalized' and r['event']=='used_le_delta' and r['contrast']=='n256:512-d50:100'))}.

{join(interactions)}

Son diferencias de diferencias de probabilidades, definidas antes de los datos.
Un intervalo que excluye cero respalda esa interacción en esta cohorte y
escala; no demuestra una interacción causal invariante a reparametrizaciones.

## P4. ¿Se reproduce en las cinco E1?

En los tres grafos nuevos n512-d100, **{sum(r['reject'] for r in primary if r['architecture']=='normalized' and r['n']==512 and r['delta']==100)}/15**
comparaciones rechazaron. La menor cota TV simultánea de A entre las quince
fue **{min(r['tv_lower'] for r in target):.3f}**. Sus probabilidades E1 variaron
entre {min(r['probability_model'] for r in target):.3f} y {max(r['probability_model'] for r in target):.3f}.
La figura 7 conserva las diferencias entre checkpoints; cinco entrenamientos
no constituyen una muestra grande de todas las redes que podrían entrenarse.
Al utilizar ambos eventos preregistrados A y T, la menor cota inferior de TV
entre las quince parejas grafo/checkpoint es {best_target_lower:.3f}. Elegir
el mayor límite de los dos eventos conserva la cobertura simultánea de su
familia. La intensidad varía entre checkpoints, aunque la diferencia no se
limita a s41000.

{join(replicas)}

## P5. ¿Se reproduce en grafos nuevos?

{join(target_table)}

Tres realizaciones independientes respaldan replicabilidad dentro de este
generador, sin demostrar universalidad. Los rangos leave-one-graph y
leave-one-checkpoint son análisis de sensibilidad, no intervalos de población.

## P6. ¿Qué ocurre en DeepSets y Transformer?

Ambos se evaluaron en las mismas 16 celdas, tres grafos y cinco checkpoints.
La tabla global y las figuras 2–5 permiten comparar probabilidades, efectos y
bandas en la misma escala. Las diferencias entre arquitecturas no se resumen
como una clasificación por número de rechazos. Los irregulares aparecen en
un estrato separado, con sus histogramas de grados y paletas; una realización
por celda limita su generalización.

Tres cortes en probabilidades comparables, con intervalos simultáneos:

{join(architecture_effects)}

Además, en cinco de las ocho celdas irregulares las paletas locales fuerzan
A≡1 bajo cualquier algoritmo: U≤min_t[t+#{{v:d(v)≥t}}]≤Δ. Por ejemplo, en
n1024-d100 la cota determinista es U≤44. Esas diferencias nulas de A no
constituyen evidencia de semejanza global. Se anotan expresamente y se
conservan el omnibus, T y los demás perfiles; no se usa A para extrapolar una
tendencia de regulares a esas celdas.

### Estrato irregular separado

{join(strata_tables['secondary'])}

Holm se aplica a sus 120 hipótesis, separado de las 720 regulares. El evento
T sigue siendo observable aun cuando A esté forzado. Las diferencias medias
sobre los cinco checkpoints del mismo grafo son:

{join(secondary_top)}

Los intervalos proceden de las cotas CP elementales ya simultáneas: el profesor
compartido se resta una vez. No son intervalos de una población de grafos o
entrenamientos. La figura 11 muestra T y marca dónde las listas fuerzan A.
El grado medio, varianza, histogramas y fracción cercana al máximo están en
los metadatos archivados. Cambian también las paletas elegibles por vértice;
estos ocho casos no identifican una tendencia universal del grado máximo.

## P7. ¿Qué propiedades finales distinguen las leyes?

Los testigos adaptativos se eligieron sólo en las primeras 500 muestras y se
validaron en las 125 reservadas. En la cohorte principal:

{join(witnesses)}

Ejemplos con las mayores cotas inferiores dentro de cada arquitectura:

{join(witness_examples)}

`used_colors` es U; `mean_color` es el promedio de las etiquetas numéricas;
`class_entropy` es −Σp_c log(p_c), sin normalizar. Las definiciones completas,
incluidas las tuplas categóricas si se seleccionan, están en
`representative-heldout-witnesses.json`. Primero se eligió un evento por
contraste usando las 500 muestras principales; después se estimó en el
holdout. Elegir estos ejemplos entre los 855 testigos no invalida sus cotas
CP simultáneas sobre esa familia. No prueban que una sola propiedad explique
todas las diferencias de una arquitectura ni identifican una causa.

Los eventos por color se midieron todos prospectivamente con una familia CP
simultánea separada. El mayor testigo de color seleccionado en el objetivo
fue `{best['graph']} / {best['model']}`, «aparece el color {best['color']}»:
Luby={best['p_Luby']:.3f}, modelo={best['p_model']:.3f}, diferencia
{best['difference']:+.3f}, intervalo [{best['ci'][0]:+.3f},{best['ci'][1]:+.3f}],
TV≥{best['tv_lower']:.3f}. La selección del máximo no invalida esa cobertura
simultánea; no se usa un intervalo ingenuo del evento elegido retrospectivamente.

También se archivan U, color máximo, U/K, mayor clase, entropía, color medio y
reutilización entre no vecinos. Wasserstein sobre proyecciones normalizadas
tiene bandas DKW puntuales descriptivas. Una media estructural próxima puede
coexistir con una diferencia enorme en un evento de cola; esas proyecciones
no identifican siempre la ley completa.

## P8. ¿El caso n512-d100 es excepcional?

La reproducción histórica recuperó exactamente 2/250 frente a 250/250 para A
en s41000, manteniendo su cota TV≥0,894. En la replicación independiente del
mismo grafo, las cinco E1 dieron probabilidades entre
{min(r['probability_model'] for r in th):.3f} y {max(r['probability_model'] for r in th):.3f},
frente a {th[0]['probability_Luby']:.3f} de Luby. Los tres grafos nuevos se
presentan arriba, y la matriz completa evita reducir la explicación a ese caso.
No se mezcló ninguna observación histórica con las nuevas.

Los quince contrastes del grafo histórico con datos nuevos, bajo su familia
Holm separada, dieron:

{join(strata_tables['historical-replication'])}

## P9. ¿Cómo se relacionan candidatos y leyes completas?

La cuadrícula fina de 4.096 uniformes, sobre estados iniciales comunes, dio:

{join(pt)}

Las referencias uniformes son 1/51≈0,019608 y 1/101≈0,009901. Un cero de
cuadrícula no demuestra masa operacional exactamente cero. Se conservaron
cuadrículas de 1.024, sus discrepancias y estados de rondas 2,4,8, con el
mismo análisis para las tres arquitecturas. En el estado inicial no aislado,
la reducción MAX de tipos de mensaje hace que, a K fijo, la función de
candidatos no dependa de n ni del grafo regular. Las comprobaciones numéricas
de esa invariancia fueron {s['initial_invariance_all']}.

Un contraejemplo al uso aislado del estado inicial es `normalized-s41004`:
q_grid(top)={initial_s4:.6f} con K=101, pero el color máximo aparece con
probabilidad media {final_s4:.3f} en sus coloraciones finales de los tres
grafos nuevos n512-d100. La disponibilidad cambia durante las rondas; una
masa inicial de cuadrícula nula no prueba ausencia en toda trayectoria.

Una referencia de primera ronda, sin condicionar al éxito futuro, con
participación p, propuestas independientes
q_c y COMMIT exacto da P(v confirma c)=p q_c(1−p q_c)^Δ. Los vecinos deben
evitar participar y proponer el mismo color. El profesor ideal tiene p=1/2 y
q_c=1/(Δ+1). Es una relación no lineal entre propuestas y confirmaciones;
las confirmaciones de vértices vecinos no son observaciones independientes.
La fórmula no presupone que el COMMIT aprendido coincida en todos los estados
con la regla exacta, ni se utilizó para modificar ninguna acción.

Las trayectorias registran propuestas, participación, disponibilidad de
referencia, COMMIT y clases finales sin modificar decisiones. Se observaron
{a['mechanism_unsafe_commits']} COMMIT inseguros en esos diagnósticos. La cadena
temporal es descriptiva: correlación y contabilidad de autorizaciones no
identifican por sí solas el efecto causal de intervenir en una propuesta.

{iv_text}

## P10. ¿Qué podemos afirmar sobre pérdida de equivalencia?

Las cotas positivas de eventos respaldan diferencias entre leyes completas
condicionales en los grafos y checkpoints indicados. Un no rechazo no prueba
igualdad. La equivalencia TV≤0,05 en los 35 casos E1 pequeños anteriores
conserva su alcance previo y no se extrapola a n≥128. No obtenemos una cota
superior útil de TV a partir de MMD pequeña o medias similares.

Las seis formas binomiales exploratorias se compararon con interceptos de
checkpoint y tres folds que dejan fuera grafos completos. Para A, la menor
pérdida predictiva fue:

{join(predict)}

{join(prediction_table)}

Se presentan todas las formas, sin convertir la selección predictiva en una
hipótesis causal confirmada ni en una escala universal de discrepancia.
La razón n/K y la densidad están ligadas por el propio diseño; una pequeña
diferencia de predicción no identifica una variable causal única. Los
coeficientes, convergencia y pérdidas de cada fold están archivados.

## Auditoría y recursos

Hubo {s['planned_confirmation_runs']:,} intentos confirmatorios incluyendo
holdouts, con estados {s['statuses_confirmation']} en orden [válida, inválida,
cutoff, error]. Los controles de traslado pasaron 29 pruebas, reprodujeron V2
y registraron 12/200, 10/200, 10/200 rechazos nulos y 0/20 en el puntual,
con intervalos exactos archivados. Son controles de leyes empíricas iguales;
la validación extensa anterior sigue siendo la referencia de calibración.
Estrella m=500 al umbral 0,05/720: V2 30/30, antiguo 0/30; límite CP puntual
inferior de potencia 0,884. Difuso: ambos 30/30. No se garantiza esa potencia
para toda alternativa: persisten las limitaciones de orden alto y masa rara.

La tabla `event-precision-and-power-guarantees.csv` aporta garantías
conservadoras adicionales para los contrastes Bernoulli preregistrados. Con
500 completaciones válidas por banco, efectos absolutos de aproximadamente
0,094 (celda de arquitectura), 0,133 (cambio adyacente) o 0,189 (interacción)
son suficientes para potencia al menos 80% por el componente Hoeffding,
incluyendo su presupuesto simultáneo. No son garantías de potencia del
omnibus, de TV arbitraria ni de obtener siempre 500 éxitos antes del cutoff.
Para diferencias menores las cotas pueden ser poco informativas.

Se verificaron {a['new_algorithm_seeds']:,} semillas nuevas distintas, sin
solapamiento con {a['distinct_old_seeds']:,} históricas, y
{a['distinct_transfer_test_seeds']:,} auxiliares. Se auditaron hashes, paletas,
validez, estados, independencia de particiones, un banco completo Luby, dos
replays V2 y Holm. Coste: **{a['gpu_process_hours']:.3f}/12 horas GPU-proceso**;
la estadística usa CPU. El commit de protocolo es
`{read(OUT/'protocol-commit.json')['commit']}`, anterior a los nuevos bancos.

## What We Can Prove, What We Observe, and What Remains Open

**Demostrable:** simetría de colores del profesor ideal en regulares con
paletas comunes; invariancia inicial de MAX a K fijo; validez condicional del
test por intercambio y cobertura simultánea bajo ejecuciones independientes;
un evento con probabilidades distintas implica leyes distintas. Las identidades
ideales no son certificados bit a bit de toda la aritmética numérica.

**Observado:** efectos, interacciones, replicaciones, variación entre grafos y
checkpoints y perfiles de propuestas/COMMIT de esta cohorte. Los intervalos
delimitan cuáles diferencias respaldan las muestras.

**Abierto:** ordenar las TV completas, separar grado de paleta, establecer
mediación completa y extender estas conclusiones a todos los grafos o a todos
los entrenamientos. Los no rechazos grandes no establecen equivalencia.

Las nueve figuras requeridas, los suplementos y el PDF de 16 páginas con
todas las distribuciones de U se regeneran de los datos archivados. Se
conservan los resultados y secciones anteriores del manuscrito.
'''
    (OUT/'FINAL_REPORT.md').write_text(text)
    short=f'''# Resumen factorial

Matriz 4×4 completada: tres grafos regulares nuevos por celda, quince
checkpoints históricos, 500 intentos por generador más 125 de holdout.

{join(method)}

En los tres grafos nuevos n512-d100, la menor cota TV del evento preregistrado
U≤100 entre las quince comparaciones E1 fue {min(r['tv_lower'] for r in target):.3f}.
Conservando A y T, las quince parejas tienen TV≥{best_target_lower:.3f}, con
la cobertura simultánea de los eventos preregistrados.
La respuesta al tamaño depende del grado y del evento. A grado 50, las bandas
simultáneas respaldan que |D_T| cae al menos {top_reduction_lower:.3f} entre
n128 y n1024. A grado 100, D_A cambia {ci(next(r for r in fc if r['group']=='normalized' and r['event']=='used_le_delta' and r['contrast']=='n512-n256|d100'))}
entre n256 y n512. Estos cambios en probabilidades no ordenan las TV
completas. Las cinco E1 difieren considerablemente y se mantienen separadas.
Grado y paleta siguen confundidos.

Los testigos independientes muestran también diferencias en propiedades que
el uso total de colores puede ocultar. Las definiciones, probabilidades e
intervalos de los ejemplos E1, DeepSets y Transformer aparecen en el informe.
{iv_short}

Auditoría: {a['new_algorithm_seeds']:,} semillas nuevas sin solapamiento;
{a['gpu_process_hours']:.3f}/12 horas GPU-proceso; sin entrenamiento ni push.
'''
    (OUT/'RESUMEN_EJECUTIVO.md').write_text(short)
    from .manuscript import make_paper
    make_paper()
    print('Report P1-P10 and manuscript source generated',flush=True)

if __name__=='__main__':main()
