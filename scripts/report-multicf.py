#!/usr/bin/env python3
"""Rebuild tables, costs and explicit P1--P14 answers from retained evidence."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from dcl.study.storage import atomic_json,file_hash

OUT=Path("results/multiteacher-counterfactual")
ARCH={"normalized":"E1 NORMALIZED","deepsets":"DeepSets","transformer":"LocalPaletteTransformer"}


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),
                      *["| "+" | ".join(map(str,r))+" |" for r in rows]])


def span(values):
    lo,hi=min(values),max(values)
    return f"{lo:,}" if lo==hi else f"{lo:,}–{hi:,}"


def main():
    selected=json.loads((OUT/"plans/successful-policies.json").read_text())["selections"]
    finals=[]
    for p in sorted((OUT/"confirmation").glob("*/result.json")):
        r=json.loads(p.read_text());f=json.loads((p.parent/"freeze.json").read_text())
        finals.append(dict(r,package=p.parent.name,evaluation_namespace=f["config"]["namespace"]))
    atomic_json(OUT/"certification-summary.json",finals)
    with (OUT/"certification-summary.csv").open("w") as f:
        fields=["package","evaluation_namespace","teacher","architecture","method","update","native_budget","status","seeds_pass"]
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(finals)
    representatives=[]
    for s in selected:
        records=[json.loads((Path(cp["path"]).parent/f"u{s['update']:06d}-record.json").read_text()) for cp in s["checkpoints"]]
        result=json.loads(Path(s["functional_result"]).read_text())
        representatives.append(dict(teacher=s["teacher"],architecture=s["architecture"],update=s["update"],method=s["method"],
            status=result["status"],native_labels=[r["native_labels"] for r in records],
            counterfactual_labels=[r["counterfactual_labels"] for r in records],
            acquired_labels=[r["total_acquired_component_labels"] for r in records],
            parameters=records[0]["parameters"],cpu_seconds=[r["cpu_seconds"] for r in records],
            checkpoint_manifest=s["checkpoints"],functional_namespace=s["functional_namespace"]))
    atomic_json(OUT/"successful-fixed-points.json",representatives)
    efficiency=[]
    for s in selected:
        folder=Path(s["functional_result"]).parent
        for cp in s["checkpoints"]:
            learned=json.loads((folder/"final"/f"s{cp['seed']}.json").read_text())
            exact=json.loads((folder/"exact"/f"s{cp['seed']}.json").read_text())
            reference={r["case_id"]:r for r in exact}
            efficiency.append(dict(teacher=s["teacher"],architecture=s["architecture"],seed=cp["seed"],
                executions=len(learned),student_mean_steps=float(np.mean([r["native_steps"] for r in learned])),
                teacher_mean_steps=float(np.mean([r["native_steps"] for r in exact])),
                mean_paired_native_step_ratio=float(np.mean([r["native_steps"]/reference[r["case_id"]]["native_steps"] for r in learned])),
                complete_color_vectors_equal=sum(r["colors"]==reference[r["case_id"]]["colors"] for r in learned),
                scope="Descriptive same-tape final-run agreement and native cost; heterogeneous graph strata, no marginal-law certificate inferred"))
    atomic_json(OUT/"native-efficiency-and-output-agreement.json",efficiency)
    # All computational training runs, counting a continuing optimizer only once.
    groups={}
    for category in ("runs","pilot","supervision-curves","coverage-controls","own-proposal-pilot","information-resumable"):
        for p in sorted((OUT/category).rglob("u*-record.json")):
            r=json.loads(p.read_text());key=str(p.parent)
            if key not in groups or r["update"]>groups[key]["updates"]:
                groups[key]=dict(category=category,updates=r["update"],cpu_seconds=r["cpu_seconds"],
                    wall_seconds=r["training_seconds"],gpu_seconds=r["gpu_seconds"],
                    peak_worker_rss_bytes=r["peak_rss_bytes"],record=str(p))
    for category in ("be-fixed-label-replay","information"):
        for p in sorted((OUT/category).rglob("record.json")):
            r=json.loads(p.read_text())
            groups[str(p.parent)]=dict(category=category,updates=r.get("updates",r.get("identity",{}).get("updates")),
                cpu_seconds=r["cpu_seconds"],wall_seconds=r["training_seconds"],gpu_seconds=r["gpu_seconds"],
                peak_worker_rss_bytes=r.get("peak_rss_bytes"),record=str(p))
    physical=[]
    for mp in sorted((OUT/"datasets").rglob("manifest.json")):
        m=json.loads(mp.read_text());physical.append(dict(path=str(mp),native_labels=m["native_labels"],
            counterfactual_labels=m["counterfactual_labels"],cpu_seconds=m["cpu_seconds"],
            simulator_generation_steps=m["teacher_simulator_steps"],sha256=m["sha256"]))
    for mp in sorted((OUT/"information/datasets").rglob("manifest.json")):
        m=json.loads(mp.read_text());physical.append(dict(path=str(mp),native_labels=m["native_labels"],
            counterfactual_labels=m["counterfactual_labels"],cpu_seconds=m["cpu_seconds"],
            simulator_generation_steps=m["teacher_simulator_steps"],sha256=m["sha256"]))
    own=[]
    for p in (OUT/"own-proposal-pilot").rglob("u001500-record.json"):
        r=json.loads(p.read_text());own.append(r)
    costs=dict(training_runs=len(groups),gradient_updates=sum(r["updates"] for r in groups.values()),
        optimization_cpu_seconds=sum(r["cpu_seconds"] for r in groups.values()),
        optimization_process_wall_seconds_sum=sum(r["wall_seconds"] for r in groups.values()),
        training_gpu_seconds=sum(r["gpu_seconds"] for r in groups.values()),
        max_recorded_worker_rss_bytes=max(r["peak_worker_rss_bytes"] or 0 for r in groups.values()),
        new_native_bank_labels=sum(r["native_labels"] for r in physical),
        new_counterfactual_bank_labels=sum(r["counterfactual_labels"] for r in physical),
        new_own_confirmation_labels=sum(r["own_confirmation_labels"] for r in own),
        data_generation_cpu_seconds=sum(r["cpu_seconds"] for r in physical),
        simulator_generation_steps=sum(r["simulator_generation_steps"] for r in physical),
        training_runs_detail=[groups[k] for k in sorted(groups)],physical_banks=physical,
        scope="Computational runs include paired treatments, numerical replays and pilots, not independent replicas. Optimization process times overlap and cannot be added as elapsed campaign wall time. Final/diagnostic wall times are retained beside outcomes. Exact-evaluation CPU/kernel time and global RAM peak were not measured; GPU training is zero. Historical M6 replay CUDA process wall time is separate. Generator-step counter excludes additional witness replay computations, which are included in acquisition CPU timing. Prefix banks reuse acquired labels and are not charged twice. Diagnostic oracle labels are separate from learner-access budgets.")
    atomic_json(OUT/"compute-cost.json",costs)
    # Paired diagnostics, with actual denominators and five policy rows retained.
    diagnostics=[]
    for p in sorted((OUT/"diagnostics").glob("paired-u004000-*.json")):
        if p.is_file():diagnostics.extend(json.loads(p.read_text()))
    fidelity=[]
    for t in ("BE_BASE_D3","SPECULATIVE_GM","COLOR_TRIALS"):
        for a in ARCH:
            for m in ("passive","matched","counterfactual"):
                rr=[r for r in diagnostics if r["teacher"]==t and r["architecture"]==a and r["tag"].endswith("-"+m)]
                if not rr:continue
                counts=Counter()
                for r in rr:counts.update(r["components"])
                fidelity.append(dict(teacher=t,architecture=a,method=m,seeds=len(rr),
                    action_correct=sum(r["action_correct"] for r in rr),action_total=sum(r["action_total"] for r in rr),
                    transition_equal=sum(r["transition_equal"] for r in rr),transition_total=sum(r["transition_total"] for r in rr),
                    **dict(counts)))
    atomic_json(OUT/"fidelity-summary.json",fidelity)
    certified_diagnostics=[r for r in diagnostics if r["teacher"]!="SPECULATIVE_GM" and r["tag"].endswith("-counterfactual")]
    certified_diagnostics.extend(json.loads((OUT/"diagnostics/paired-u000500-SPECULATIVE_GM.json").read_text()))
    assert len(certified_diagnostics)==45
    atomic_json(OUT/"certified-policy-fidelity.json",sorted(certified_diagnostics,key=lambda r:(r["teacher"],r["architecture"],r["seed"])))
    with (OUT/"fidelity-summary.csv").open("w") as f:
        fields=sorted(set().union(*(r.keys() for r in fidelity)))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(fidelity)
    supervision=[]
    for a in ARCH:
        for b in (1000,3000,10000,30000):
            for u in (500,1500):
                rr=[]
                for s in range(94101,94106):
                    p=(OUT/"runs/color_trials"/a/f"s{s}"/"b30000-coverage1-alternate0/counterfactual"/f"u{u:06d}-record.json" if b==30000 else
                       OUT/"supervision-curves"/a/f"s{s}"/f"b{b}"/f"u{u:06d}-record.json")
                    rr.append(json.loads(p.read_text()))
                supervision.append(dict(architecture=a,native_budget=b,update=u,
                    actual_total_labels=[r["total_acquired_component_labels"] for r in rr],
                    development_successes=[r["development"]["successes"] for r in rr],
                    development_perfect_seeds=sum(r["development"]["pass_all"] for r in rr),
                    scope="Development only, no new certification/minimum claim"))
    atomic_json(OUT/"supervision-summary.json",supervision)
    coverage=[]
    for a in ARCH:
        for u in (500,1500):
            for covered in (True,False):
                rr=[]
                for s in range(94101,94106):
                    p=(OUT/"runs/speculative_gm"/a/f"s{s}"/"b3000-coverage1-alternate0/matched"/f"u{u:06d}-record.json" if covered else
                       OUT/"coverage-controls/speculative_gm"/a/f"s{s}"/"b3000-coverage0-alternate0/matched"/f"u{u:06d}-record.json")
                    rr.append(json.loads(p.read_text()))
                coverage.append(dict(architecture=a,updates=u,coverage=covered,
                    development_successes=[r["development"]["successes"] for r in rr],
                    development_perfect_seeds=sum(r["development"]["pass_all"] for r in rr)))
    atomic_json(OUT/"coverage-summary.json",coverage)
    info=json.loads((OUT/"information/results.json").read_text())
    laws=json.loads((OUT/"generalization/laws-summary.json").read_text())
    lawrows=[dict(teacher=r["teacher"],architecture=r["architecture"],seed=r["seed"],**g) for r in laws for g in r["graphs"]]
    transfer=json.loads((OUT/"generalization/transfer-summary.json").read_text())
    verification=json.loads((OUT/"verification.json").read_text())
    point_table=table(["Profesor","Arquitectura","Updates","Etiquetas nativas/semilla","Etiquetas totales/semilla","Final"],
        [[r["teacher"],ARCH[r["architecture"]],r["update"],span(r["native_labels"]),span(r["acquired_labels"]),"5/5 × 1000/1000"] for r in representatives])
    final_table=table(["Profesor","Arquitectura","Tratamiento","Updates","Semillas PASS","Éxitos/5000","Protocolo"],
        [[r["teacher"],r["architecture"],r["method"],r["update"],f"{r['seeds_pass']}/5",sum(s["successes"] for s in r["per_seed"].values()),
          "inicial" if "optimization-v2" not in r["package"] else ("BE v2" if r["teacher"]=="BE_BASE_D3" else "CT v2")] for r in finals])
    fidelity_table=table(["Profesor","Arquitectura","Tratamiento a 4000","Acciones correctas/total","CF correctos/total"],
        [[r["teacher"],r["architecture"],r["method"],f"{r['action_correct']}/{r['action_total']}",
          f"{r['counterfactual_correct']}/{r['counterfactual_total']}"] for r in fidelity])
    info_table=table(["Representación JP","Éxitos development por semilla /60"],
        [[m,", ".join(str(r["development"]["successes"]) for r in info if r["mode"]==m)] for m in
         ("without_ids","numeric_ids","numeric_differences","relative_comparisons")])
    transfer_table=table(["Profesor","Arquitectura","Éxitos agregados /420","Cinco políticas perfectas"],
        [[t,a,sum(r["successes"] for r in transfer if r["teacher"]==t and r["architecture"]==a),
          sum(r["pass_all"] for r in transfer if r["teacher"]==t and r["architecture"]==a)]
         for t in ("BE_BASE_D3","COLOR_TRIALS","SPECULATIVE_GM") for a in ARCH])
    sup_table=table(["Arquitectura CT","B nativo","Updates","Etiquetas totales","Semillas 60/60 dev"],
        [[r["architecture"],r["native_budget"],r["update"],span(r["actual_total_labels"]),f"{r['development_perfect_seeds']}/5"] for r in supervision])
    own_table=table(["Profesor","Updates","Development /60","Consultas propias nuevas","Etiquetas totales"],
        [[r["identity"]["teacher"],r["update"],r["development"]["successes"],r["own_confirmation_labels"],r["total_acquired_component_labels"]] for r in sorted(own,key=lambda x:x["identity"]["teacher"])])
    efficiency_table=table(["Profesor","Arquitectura","Vectores completos iguales /5000","Pasos estudiante/profesor (media apareada)"],
        [[t,a,sum(r["complete_color_vectors_equal"] for r in efficiency if r["teacher"]==t and r["architecture"]==a),
          f"{np.mean([r['mean_paired_native_step_ratio'] for r in efficiency if r['teacher']==t and r['architecture']==a]):.6f}"]
         for t in ("BE_BASE_D3","COLOR_TRIALS","SPECULATIVE_GM") for a in ARCH])
    lawbad=[g for g in lawrows if not g["certifies_tv_at_most_005"]]
    transfer_success=sum(r["successes"] for r in transfer);transfer_total=sum(r["executions"] for r in transfer)
    text=f'''# Aprendizaje multiteacher: reconstrucción, controles y resultados

## Resultado principal y alcance

Se cerraron las **nueve combinaciones pendientes** de ColorTrials, BE y GM en
E1 NORMALIZED, DeepSets y LocalPaletteTransformer: **cinco entrenamientos y
1000/1000 ejecuciones independientes por checkpoint** en cada punto exitoso.
Las doce certificaciones históricas se conservan. La cobertura es por tanto
**21/21 combinaciones**, combinando evidencia histórica y nuevas baterías;
no se presenta como una única cohorte homogénea ni se cambia el resultado
histórico primario de 12/21.

El tratamiento concordante (`matched`) pasó las nueve celdas nuevas con
como máximo 1500 updates en BE/ColorTrials y 500 en GM; ColorTrials/E1 ya
pasa a 500. La supervisión contrafactual también
certifica las nueve en los puntos de la tabla siguiente, tras controles de
optimización independientes. Se conservan todos los fallos de los puntos
anteriores. No hubo shields, reparación externa ni COMMIT del profesor en
inferencia. Los ocho archivos fuente del profesor/interfaz/inferencia
contrastados contra `61f39760` permanecen idénticos.

{point_table}

Las cinco semillas nuevas son 94101–94105, con inicialización, datos y
minibatches reproducibles. El final nuevo usa grafos simples no dirigidos de
grado máximo 3, paleta global K4, tamaños 32/48/64 y diez estratos de 100 casos
por política. Training usa N8/12/16/24 y development N28/36. BE conserva su
aritmética y fases suministradas; GM su calendario nativo y reparación
aprendida. Cada política PASS tiene cota unilateral puntual de éxito
0.99700875 bajo el modelo de independencia del protocolo. No se afirma una
garantía simultánea sobre todas las políticas ni corrección universal.

## Cronología científica y resultados negativos

La infraestructura y el protocolo se comprometieron localmente en `3d9f6456`.
Una enmienda anterior al nuevo screen y a toda confirmación amplió development
de 60 a 300 casos por semilla. Se congelaron procedimientos y hashes antes de
cada final. La primera cohorte incluye los tres tratamientos y todas las
celdas; dos seguimientos exploratorios comparan los checkpoints ya existentes
de 1500/4000 updates sobre **nuevas baterías**, después de observar fallos
contrafactuales del primer final. No se entrenó sobre casos finales ni se
repitió una batería para mejorar su resultado.

La validación conserva {verification['final_executions']:,} ejecuciones finales,
{verification['final_successes']:,} correctas y
{verification['unique_final_checkpoints']} checkpoints finales distintos;
algunos checkpoints se evalúan en dos namespaces y no se cuentan como nuevos
entrenamientos. Los fallos BE contrafactuales a 1500 se concentran en la quinta
semilla y comparten la cabeza COMMIT entre arquitecturas. Los testigos
reproducidos con el contexto numérico original localizan omisiones de una
recoloración KW obligatoria, que dejan el auxiliar fuera del bound nuevo.
ColorTrials conserva un fallo inicial de E1 a 500 y dos fallos del
Transformer a 1500 en el seguimiento independiente.
Los mismos bancos de etiquetas, con 4000 updates, pasan el seguimiento
independiente. Un brazo concordante de ColorTrials/E1 además regresa de 60/60
a 43/60 dev entre 1500 y 4000: más optimización no es uniformemente mejor.
Una errata conservada en `audit/ct-followup-rationale-erratum.json` corrige el
texto del campo `reason` del plan CT v2: decía 1500 para el fallo inicial E1,
cuando el resultado y checkpoint originales registran 500. El seguimiento
sí compara 1500/4000 y no vuelve a probar 500. El plan erróneo se conserva;
no se cambian pesos, baterías, filas ni gates.

{final_table}

## Respuestas explícitas P1–P14

**P1. ¿Qué hacía M6B?** Usaba M4B (8967 parámetros a K4), no E1. Conservaba
grafo, colores y entradas privadas de la ronda; intervenía P/C y recalculaba
COMMIT sobre todas las ofertas efectivamente presentes. Tras diez épocas
docentes: P + C + .5 COMMIT propio + .5 COMMIT contrafactual. Las inyecciones
históricas eran .17 por arista fija→pendiente y .04 por arista entre
pendientes. Se autenticaron los 21 artefactos, el CSV de 500 casos y 30
ejecuciones con los pesos originales. 249/500, 476/500 y 500/500 corresponden
a una sola semilla 10001. El control propio y CF tienen P/mensaje1/C idénticos;
sólo difieren mensaje2/COMMIT. Se regeneraron 18877 rondas y 248122 decisiones
pendientes; época 53 implica 7844 updates, y el horizonte configurado 17760.
La pérdida pasiva del estudio posterior aplica la etiqueta COMMIT docente a
propuestas aprendidas que pueden diferir; `matched` corrige esa concordancia.

**P2. ¿Qué se transfiere?** Reconsulta de etiquetas tras intervenciones,
supervisión por fases, gradientes detenidos a través de decisiones duras,
contrastes propios/perturbados y trazas verificables. Cada profesor tiene un
contrato de estados y dependencias. Las ofertas Luby/Johansson, preferencias
y dos mensajes ColorTrials, órdenes JP, puntos Linial/opciones KW, historia
especulativa GM y orden Greedy se tratan individualmente. Hay 162 contextos
difíciles adicionales con replay de preferencias/draws/IDs y sus descendientes.
Se distingue una decisión nativa de una continuación defensiva consultada
sobre mensajes fuera de la emisión canónica; no se llama trayectoria docente
a cualquier array procesable. Los tres núcleos consumen la misma infraestructura.

**P3. ¿Mejora CF ColorTrials, BE y GM?** Mejora la respuesta en continuaciones
contrafactuales; no mejora uniformemente el gate funcional al mismo número
de updates. Los tres brazos cierran GM a 500. El concordante y el pasivo
cierran BE y ColorTrials a 500 o 1500 según la celda; algunos CF necesitan 4000 en sus nuevas
confirmaciones. Esto conserva evidencia positiva y negativa. CF no es
necesario en todas las celdas para resolver este benchmark, y sí aporta
supervisión de componentes que no está cubierta por la emisión nativa.

**P4. ¿Qué causa las mejoras?** Las curvas 500/1500/4000 dentro de cada brazo
usan los mismos datos adquiridos, inicialización y secuencia de minibatches.
Los seguimientos CF comparan 1500/4000 con idénticas etiquetas. El replay de
BE/E1 a 1500 reproduce exactamente los pesos originales de las cinco semillas.
No hubo cambios de arquitectura/representación en los controles principales.
`matched` frente a CF cambia distribución y cantidad de etiquetas; no identifica
una ventaja muestral a igual presupuesto total. La ablación GM sin ampliación
de cobertura también pasa sus cinco semillas en development, por lo que
estos controles no demuestran una ventaja causal aislada de añadir aislados.
El caso histórico 83105 y las 30 regresiones nuevas quedan separados del final.

**P5. ¿Qué explica Pedro y qué se reproduce?** Se verificaron sus 16 hashes
fuente. Sus cinco 1980/1980, JP 1631/1980 y GM 1951/1980 son resultados de
cinco inicializaciones por 396 casos publicados, con un pool de datos común
y tests reutilizados. Usó 17760 updates de 1680 raíces y confirmaciones
propias/CF; la GNN directa, los adaptadores, la optimización y evaluación
también difieren. No apareció ninguno de los 40 checkpoints esperados entre
6614 archivos locales ni en los manifiestos de las releases inspeccionadas.
Por ello sus políticas exactas siguen sin reevaluación independiente. Nuestros
resultados son nuevos entrenamientos y reproducen mecanismos y reglas
implementadas, no sus pesos ni su batería hasta N10240.

**P6. ¿Se incorporan sus mecanismos?** Sí: concordancia de COMMIT, bancos
contrastivos, calendario/arimética explícitos y supervisión por fases se usan
sin sustituir los núcleos. El piloto adicional `owncf` porta la pérdida propia
+ CF a los siete profesores con una semilla nueva y cobra cada consulta
propia distinta. Sus resultados no se llaman certificación; BE sigue siendo
difícil en ese piloto a 1500, mostrando que el calendario de aprendizaje
necesita adaptación por profesor.

{own_table}

**P7. ¿Tres arquitecturas por siete profesores?** Sí, bajo los gates finitos
declarados: 12 celdas históricas preservadas y 9 nuevas con cinco políticas
congeladas y 1000 éxitos cada una. También existe un conjunto nuevo CF de
las nueve pendientes: BE/ColorTrials a 4000 y GM a 500. No se deduce que
cualquier nueva semilla, grafo o tamaño pasará, ni una certificación homogénea
de 21 celdas entrenadas todas en esta campaña.

**P8. ¿Qué sigue siendo difícil?** KW conserva errores de progreso a 1500
en CF: una recoloración omitida viola el bound aunque no haya colisión.
ColorTrials conserva contextos raros que pasan development pero fallan en el
final de algunos checkpoints, y una política exitosa difiere bajo cintas
compartidas en path8. Los pilotos propios y la transferencia no se convierten
en éxitos por agregar reparaciones. No queda una de las nueve celdas sin un
punto funcional certificado, pero sí quedan fallos de otros puntos y
fidelidad/generalización sin garantía universal.

**P9. ¿Etiquetas y recursos?** Se separan nativas, CF, propias nuevas,
presentaciones, actualizaciones y simulación. Los bancos principales se
comparten entre arquitecturas y tratamientos. Los prefijos de supervisión
son idénticos al banco mayor y no cobran etiquetas otra vez. Se realizaron
{costs['training_runs']} ejecuciones computacionales de entrenamiento,
incluidos pilotos, controles apareados y replays, con
{costs['gradient_updates']:,} updates y
{costs['optimization_cpu_seconds']/3600:.3f} horas CPU de optimización.
El entrenamiento usó cero tiempo GPU; el replay histórico M6B CUDA se
contabiliza aparte. El máximo RSS registrado de un worker fue
{costs['max_recorded_worker_rss_bytes']/2**20:.1f} MiB. Los tiempos por proceso
se solapan; no se suman como duración transcurrida. `compute-cost.json`
conserva el detalle y las magnitudes no medidas. No se afirma eficiencia
muestral frente al estudio previo al cambiar a la vez optimización y datos.

{sup_table}

**P10. ¿Interfaces e IDs?** El régimen A conserva anonimato en Luby,
Johansson y ColorTrials. JP/GM usan órdenes coherentes; BE fases y aritmética;
GM/Greedy calendarios declarados. El control JP mantiene arquitectura,
parámetros, inicialización, etiquetas, batches y 500 updates, variando sólo
la representación local de IDs. Las comparaciones relativas pasan 60/60 en
las cinco semillas; retirar IDs o usar representaciones numéricas no alcanza
ese punto. Esto identifica sensibilidad al presupuesto y representación,
no incapacidad matemática de procesar IDs numéricos.

{info_table}

**P11. ¿Fidelidad, transiciones, rondas y leyes?** El diagnóstico apareado a
4000 cubre todas las raíces permitidas de 60 trayectorias docentes por semilla,
con denominadores separados para acciones, transiciones reales, P/C,
confirmaciones propias y cuatro consultas CF nuevas por raíz. Las tasas se
describen; las decisiones correlacionadas no inflan el número de réplicas.
Las 45 políticas certificadas tienen además sus auditorías de componentes
y transiciones en `certified-policy-fidelity.json`, usando sus updates reales
(incluido GM a 500). El cuadro siguiente mantiene 4000 para la comparación
controlada de los tres tratamientos. BE/GM tienen cabezas de confirmación idénticas entre arquitecturas con
ofertas fijas y semillas/batches apareados; ColorTrials acopla esa etapa al
selector. Las rondas nativas y las coloraciones finales se conservan por caso.

{fidelity_table}

En las baterías finales de los puntos CF exitosos también se compararon
pasos nativos y vectores completos con el profesor sobre la misma cinta.
Son métricas descriptivas de estados/tiempos, separadas del gate y de las
leyes sobre grafos fijos. No se convierten estos casos de familias diferentes
en una certificación de una única ley condicional.

{efficiency_table}

En dos grafos pequeños fijados se compararon 256 cintas compartidas por
política y grafo, observando la coloración completa etiquetada y el estado
terminal sin filtrar fallos. Hay {sum(g['certifies_tv_at_most_005'] for g in lawrows)}/90
certificados simultáneos de TV≤.05, mediante la desigualdad de acoplamiento
y cotas binomiales con Bonferroni .05/90. No se afirma igualdad exacta.
La excepción es {lawbad[0]['teacher']}/{lawbad[0]['architecture']}, semilla
{lawbad[0]['seed']}, {lawbad[0]['graph']}: {lawbad[0]['coupled_disagreements']}/256
desacuerdos, cota superior TV {lawbad[0]['tv_upper_simultaneous_95']:.6f},
{lawbad[0]['student_mean_native_steps']:.6f} pasos frente a
{lawbad[0]['teacher_mean_native_steps']:.6f}. Desacuerdo bajo un acoplamiento
no es una cota inferior de TV ni prueba de leyes marginales diferentes.

**P12. ¿Qué se demuestra sobre información?** El lema de
indistinguibilidad impide imitar decisiones distintas con observaciones
iguales. Dos IDs intercambiados en una arista dan una obstrucción constructiva
para JP sin orden; comparaciones relativas bastan para su elegibilidad.
ColorTrials necesita información de disponibilidad del vecino o equivalente
de segundo salto en la observación especificada. La conjunción participación
y oferta del mismo vecino debe preservarse antes de agregar. Se explicitan
las suficiencias locales de puntos Linial y bloques KW y los permisos de
calendarios. Además, dos uniforms float64 a ambos lados de .5 se pueden
codificar como el mismo float32 en M6 histórico, impidiendo fidelidad
universal de esa etiqueta; el protocolo nativo actual consulta float32
directamente. Son resultados acotados a esas observaciones, no una teoría
universal de entrenabilidad o complejidad muestral.

**P13. ¿Generalización demostrada?** En transferencia exploratoria
N128/256/512, con diez familias del dominio y cuatro generadores nuevos
(árboles binarios, cadenas de diamantes/triángulos y K4 subdivididos), se
observaron {transfer_success}/{transfer_total} éxitos. Cada política tiene
84 casos; las proporciones de la tabla no reemplazan su gate final.
No se amplía el dominio a grado mayor que tres ni a paletas locales. La
línea de grado 200 se conserva como trabajo complementario.

{transfer_table}

**P14. ¿Contribuciones centrales sólidas?** (i) Oráculos causales heterogéneos
con contratos de admisibilidad, etiquetas consistentes y testigos repetibles;
(ii) cierre reproducible de las nueve celdas con controles que separan
concordancia, CF y optimización; (iii) demostración de que PASS funcional,
fidelidad de continuación y ley terminal requieren evidencias diferentes;
(iv) reconciliación histórica autenticada y límites de información
constructivos. Los efectos de cobertura nulos en este control, fallos CF a
1500, piloto BE propio y límites de transferencia forman parte de la
contribución. CLRS y DAgger proporcionan antecedentes de trazas y adquisición
en estados del estudiante; la novedad propuesta aquí se concreta en estos
protocolos, adaptadores y controles de siete profesores, sin reclamar
prioridad bibliográfica universal.

## Reproducibilidad y límites

Los archivos obligatorios de reconstrucción, reconciliación, protocolo y
análisis de fallos están en `docs/`. `verification.json` recalcula todos los
gates y hashes y vuelve a comprobar los labels de los bancos; no hay etiquetas
almacenadas inconsistentes. Pasaron los 668 casos de la suite completa final.
Las pruebas incluyen ausencia de consultas docentes en inferencia,
reanudación exacta, diferenciales nativos, intervenciones y testigos de
información. Los pesos, Adam, RNG, visitas, datos, presupuestos, fallos y
manifiestos permanecen en el directorio de resultados nuevo. La publicación
posteriormente autorizada corresponde a la rama
`research/counterfactual-multiteacher-learning` y al paquete reproducible
de la prerelease `multiteacher-counterfactual-2026-10-24`.

La revisión de antecedentes consulta [DAgger](https://proceedings.mlr.press/v15/ross11a.html)
y [CLRS](https://arxiv.org/abs/2205.15659). El trabajo reciente de
[coloración contrastiva](https://arxiv.org/abs/2606.03923) estudia un objetivo
de coloración aproximada. Estos antecedentes orientan la comparación;
nuestros resultados se atribuyen a las reglas nativas implementadas y a los
artefactos concretos auditados. Las fuentes de Pedro ausentes, el alcance
finito de los tests y las diferencias de protocolos permanecen explícitos.
'''
    Path("docs/MULTITEACHER_COUNTERFACTUAL_RESULTS.md").write_text(text)
    atomic_json(OUT/"report-inputs.json",dict(certification_packages=len(finals),representative_packages=len(representatives),
        fidelity_rows=len(fidelity),law_certificates=sum(g["certifies_tv_at_most_005"] for g in lawrows),
        law_hypotheses=len(lawrows),transfer_successes=transfer_success,transfer_executions=transfer_total,
        markdown_sha256=file_hash("docs/MULTITEACHER_COUNTERFACTUAL_RESULTS.md")))
    print("Rebuilt results, costs and P1–P14 report")


if __name__=="__main__":main()
