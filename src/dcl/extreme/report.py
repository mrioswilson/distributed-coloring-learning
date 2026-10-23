"""Rebuild tables, uncertainty and standalone scientific figures from records."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from .common import atomic_json, now


def exact_interval(successes,n):
    if not n: return None,None
    return (0.0 if not successes else float(beta.ppf(.025,successes,n-successes+1)),
            1.0 if successes==n else float(beta.ppf(.975,successes+1,n-successes)))


def records(root):
    all_rows=[]
    for directory,cohort in ((root/"results","cuda-main"),(root/"cpu-pilot","cpu-pilot")):
        for p in sorted(directory.glob("*.json")):
            r=json.loads(p.read_text())
            g=r.get("graph",{})
            model=r.get("model",{})
            config=r.get("training_config",model.get("config",{}))
            flat={k:v for k,v in r.items() if isinstance(v,(str,int,float,bool)) or v is None}
            flat.update(cohort=cohort,degree=g.get("degree"),palette=g.get("palette"),n=g.get("n"),
                        family=g.get("family"),graph_id=g.get("graph_id"),graph_seed=g.get("graph_seed"),
                        tape_seed=g.get("tape_seed"),fingerprint=g.get("fingerprint"),
                        training_seed=r.get("training_seed",config.get("seed")),
                        experiment=r.get("experiment",config.get("name")),
                        training_degrees=json.dumps(r.get("training_degrees",model.get("train_degrees",[]))),
                        local_metrics_file=str(p.resolve()))
            all_rows.append((r,flat))
    return all_rows


def summary_rows(rows):
    groups=defaultdict(list)
    for r in rows:
        key=tuple(r.get(k) for k in ("cohort","experiment","training_seed","degree","n","family"))
        groups[key].append(r)
    result=[]
    for key,group in groups.items():
        measured=[r for r in group if r.get("success") is not None]
        successes=sum(bool(r["success"]) for r in measured)
        lo,hi=exact_interval(successes,len(measured))
        row=dict(zip(("cohort","experiment","training_seed","degree","n","family"),key))
        row.update(evaluations=len(group),measured=len(measured),technical_or_design_skips=len(group)-len(measured),
                   independent_graphs=len({r["fingerprint"] for r in measured}),successes=successes,
                   success_rate=successes/len(measured) if measured else None,
                   ci95_lower=None if key[-1]=="CLIQUE" else lo,ci95_upper=None if key[-1]=="CLIQUE" else hi)
        for k in ("unsafe_commits","fixed_neighbor_conflicts","simultaneous_conflicts","valid_commits_rejected",
                  "uncolored_nodes","states","candidate_correct","participation_correct","forbidden_offers"):
            row[k]=sum(r.get(k,0) for r in measured)
        for k in ("rounds","inference_seconds","gpu_peak_bytes","colors_used","color_entropy_nats"):
            values=[r[k] for r in measured if k in r]
            row[k+"_mean"]=float(np.mean(values)) if values else None
            row[k+"_max"]=float(np.max(values)) if values else None
        result.append(row)
    return result


def markdown_table(frame):
    # No optional tabulate dependency needed.
    columns=list(frame.columns)
    def fmt(x):
        if pd.isna(x): return "—"
        if isinstance(x,(float,np.floating)): return f"{x:.4g}"
        return str(x)
    return "| "+" | ".join(columns)+" |\n|"+"|".join("---" for _ in columns)+"|\n"+"\n".join(
        "| "+" | ".join(fmt(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None))


def build(root):
    root=Path(root).resolve()
    out=root/"report"
    out.mkdir(parents=True,exist_ok=True)
    pairs=records(root)
    flat=[r for _,r in pairs]
    frame=pd.DataFrame(flat)
    frame.to_csv(out/"cases.csv",index=False)
    summary=pd.DataFrame(summary_rows(flat))
    summary.to_csv(out/"success-by-model-degree-size-family.csv",index=False)
    inventory=json.loads((root/"audit/inventory.json").read_text())
    pd.DataFrame(inventory["cohorts"]).to_csv(out/"original-cohorts.csv",index=False)
    pd.DataFrame([{**{k:v for k,v in r.items() if isinstance(v,(str,int,float,bool))},
                   "experiment":r["config"]["name"],"seed":r["config"]["seed"],
                   "training_degrees":json.dumps(r["config"]["degrees"]),
                   "graphs_per_degree":json.dumps(r["config"]["graphs_per_degree"]),
                   "tapes":r["config"]["tapes"],"epochs":r["epochs_completed"]} for r in inventory["runs"]]).to_csv(out/"models.csv",index=False)
    by_k=[]
    color_rows=[]
    joint=defaultdict(list)
    for r,f in pairs:
        if f.get("success") is None: continue
        for color,count in enumerate(r.get("color_counts",[])):
            color_rows.append({"cohort":f["cohort"],"experiment":f["experiment"],"training_seed":f["training_seed"],
                               "degree":f["degree"],"n":f["n"],"family":f["family"],"graph_id":f["graph_id"],
                               "color":color,"nodes":count,"fraction":count/f["n"]})
        for k,v in r.get("by_k",{}).items():
            by_k.append({"cohort":f["cohort"],"experiment":f["experiment"],"training_seed":f["training_seed"],
                         "degree":f["degree"],"n":f["n"],"family":f["family"],"graph_id":f["graph_id"],"k":int(k),**v})
        if f["cohort"]=="cuda-main" and f["experiment"]=="E1":
            joint[(f["degree"],f["n"],f["family"],f["graph_id"])].append(f)
    pd.DataFrame(by_k).to_csv(out/"by-k-by-graph-model.csv",index=False)
    pd.DataFrame(color_rows).to_csv(out/"color-frequencies-by-graph-model.csv",index=False)
    joint_rows=[]
    for key,rs in joint.items():
        row=dict(zip(("degree","n","family","graph_id"),key))
        row.update(models_evaluated=len({r["training_seed"] for r in rs}),
                   all_observed_models_success=all(r["success"] for r in rs),
                   all_five_models_evaluated={r["training_seed"] for r in rs}==set(range(5)))
        joint_rows.append(row)
    pd.DataFrame(joint_rows).to_csv(out/"shared-graphs-replicates.csv",index=False)
    diagnostic_rows=[]
    diag_reports=[]
    for p in sorted((root/"diagnostics").glob("*/*/selection.csv")):
        df=pd.read_csv(p)
        df.insert(0,"device",p.parents[1].name)
        df.insert(1,"run",p.parent.name)
        diagnostic_rows.append(df)
        done=p.parent/"done.json"
        if done.exists():
            d=json.loads(done.read_text())
            diag_reports.append({"run":p.parent.name,"device":d["device"],"selection_states":d["selection_states"],
                                 "unsafe_counterfactual_commits":d["unsafe_counterfactual_commits"],
                                 "commit_presence_cases":d.get("commit_presence_cases"),
                                 "commit_presence_correct":d.get("commit_presence_correct"),
                                 "participation_accuracy":d["participation"]["correct"]/d["participation"]["draws"]})
    diagnostics=pd.concat(diagnostic_rows,ignore_index=True) if diagnostic_rows else pd.DataFrame()
    diagnostics.to_csv(out/"controlled-selection.csv",index=False)
    pd.DataFrame(diag_reports).to_csv(out/"controlled-commit-and-participation.csv",index=False)
    statuses=Counter(r.get("status") for r in flat if r["cohort"]=="cuda-main")
    cpu=[r for r in flat if r["cohort"]=="cpu-pilot" and r.get("success") is not None]
    gpu=[r for r in flat if r["cohort"]=="cuda-main" and r.get("success") is not None]
    state=json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {"status":"not_started"}
    info={"generated_utc":now(),"original":inventory["preliminary"],"cuda_statuses":dict(statuses),
          "cpu_pilot":{"cases":len(cpu),"successes":sum(r["success"] for r in cpu),"max_degree":max((r["degree"] for r in cpu),default=None)},
          "cuda":{"cases":len(gpu),"successes":sum(r["success"] for r in gpu),"max_degree":max((r["degree"] for r in gpu),default=None)},
          "coordinator":state,"diagnostic_models_completed":len(diag_reports)}
    atomic_json(out/"summary.json",info)
    lines=["# UCGNN: auditoría y extrapolación extrema",f"\nGenerado: {info['generated_utc']}.",
           "\nInforme parcial reproducible. Los resultados CPU del piloto y la evaluación principal CUDA se mantienen separados.",
           f"\nEl censo histórico contiene **{inventory['preliminary']['successes']}/{inventory['preliminary']['cases']}** éxitos, "
           "en 4.880 grafos distintos, réplica 0, época 120. Δ=3/5/8/10: N=40/160/640/2.560/10.240/40.960, "
           "100/100/100/50/25/10 grafos por familia (3.080). Δ=4/6/7/12/15: N=40/160/640/2.560/10.240, "
           "50/50/50/20/10 por familia (1.800). Los grados adicionales no tienen N=40.960.",
           "\nSe reprodujeron 16 casos históricos en CPU sin discrepancias discretas; pesos idénticos antes/después de Δ=3/10/20. "
           "Ver `../audit/reproduction.json`, `../audit/equivalence-cpu.json` y `original-cohorts.csv`.",
           "\nLa red NORMALIZED tiene 16.035 parámetros, dimensión de estado/mensaje 32 y capas intermedias de 64. "
           "No tiene parámetros por color. Sí incorpora prefijos acumulados y normalización por masa de disponibilidad aprendida. "
           "Estos operadores son un sesgo algorítmico explícito. Los pesos aprenden disponibilidad, puntuación, participación y COMMIT. "
           "Las reglas exactas observan las acciones después de su ejecución; no hay corrección externa en la ruta evaluada.",
           f"\nCPU: {sum(r['success'] for r in cpu)}/{len(cpu)}. CUDA: {sum(r['success'] for r in gpu)}/{len(gpu)} casos medidos. "
           f"Estado del coordinador: `{state['status']}`. Los OOM, tiempos agotados y exclusiones de recursos no son fallos de coloración.",
           "\n## Resultados por modelo, grado, tamaño y familia",
           "\nLos intervalos son Clopper–Pearson bilaterales del 95 %, condicionados al modelo, tamaño, grado y familia. "
           "No se agrupan ejecuciones de distintos modelos sobre el mismo grafo como muestras independientes. "
           "CLIQUE es un caso determinista de estrés y no lleva intervalo binomial. No se presenta un intervalo global mezclando cohortes."]
    if not summary.empty:
        selected=summary[(summary["cohort"]=="cuda-main") & (summary["experiment"]=="E1")]
        if selected.empty: selected=summary[summary["cohort"]=="cpu-pilot"]
        lines.append("\n"+markdown_table(selected[["cohort","training_seed","degree","n","family","measured","successes","ci95_lower","rounds_mean","unsafe_commits"]]))
    lines += ["\n## Fidelidad local y mecanismos",
              "\nLos barridos controlados usan estados de estrella con conjuntos libres bajos, altos o dispersos; k=1/2/3/4/5/10/20/50/100/200/500 cuando caben. "
              "La malla uniforme de 1.001 puntos y los valores float32 adyacentes a j/k se reportan por separado. "
              "Los porcentajes sobre esa malla no son estimaciones basadas en grafos aleatorios independientes."]
    if not diagnostics.empty:
        d=diagnostics[(diagnostics["run"].str.startswith("e1-")) & (diagnostics["run"].str.contains("s00")) & (diagnostics["pattern"]=="SCATTERED") & (diagnostics["k"]==diagnostics["palette"])]
        lines.append("\n"+markdown_table(d[["device","palette","k","uniform_grid_accuracy","availability_slot_errors","max_interval_endpoint_error","distinct_proposed_colors"]]))
    lines += ["\nUna coloración válida no implica fidelidad al profesor. Una disponibilidad bien identificada y prefijos precisos pueden coexistir con errores de selección al estrecharse los intervalos. "
              "Ver `controlled-selection.csv` y los NPZ originales con cada r_c, propuesta y selección exacta.",
              "\nCOMMIT se diagnostica con conflictos fijos, simultáneos y múltiples, usando varios nombres de color. "
              "Además se evalúan los 64 conjuntos de los seis tipos de mensajes vecinos posibles en el slot propuesto. "
              "Max ignora multiplicidades; esta reducción caracteriza sólo el camino de COMMIT para un nodo no coloreado que participa. "
              "No demuestra corrección universal de la política completa.",
              "\nLa auditoría adicional de épocas encuentra que `best.pt` de E1-0 corresponde a la época 1: acierta 48/64 estados de COMMIT y acepta 16 inseguros. "
              "Época 10 también da 48/64; épocas 20 y 120 dan 64/64. La batería usa la época 120 predefinida, no `best.pt`. "
              "E1 y C1 tienen exactamente los mismos pesos de participación y candidato, y distintos pesos de COMMIT; comparten datos e inicialización. "
              "Esto concuerda con la separación de parámetros y las decisiones duras sin gradiente entre etapas. Ver `../audit/ablation-and-epochs.json`.",
              "\n## Recursos y diseño pendiente",
              "\nLa implementación original materializa N×K×32 y temporalmente E_chunk×K×32. "
              "El evaluador por fragmentos conserva los parámetros y las operaciones, recalcula la primera etapa y mantiene bloques N×min(K,8)×32. "
              "No crea adyacencia densa N×N. El volumen lógico de comunicación por ronda de la implementación es 2·E_dir·K·32·4 bytes. "
              "La ejecución por fragmentos recalcula la primera agregación y procesa 3·E_dir·K·32·4 bytes de mensajes en total; "
              "reutiliza el mismo estado y no añade otra dependencia entre rondas o vecinos. "
              "La fragmentación reduce memoria local; no elimina esa dependencia de K en los mensajes. "
              "`cases.csv` incluye tiempos, máximos de CUDA y localización de los registros completos; `cost` en cada JSON conserva las estimaciones previas.",
              "\nLa cola prioriza 20 grafos nuevos por familia principal, grado, tamaño y modelo; después amplía a 60 en N=640 y Δ=15/20/30/50/100. "
              "Los cinco modelos comparten los grafos. Las pruebas grandes tienen cinco grafos por familia y carácter exploratorio: "
              "se admiten sólo si el volumen lógico por ronda es ≤32 GiB y la memoria estimada ≤3 GiB. "
              "Δ=500 es exploratorio, condicionado al umbral de éxito predeclarado del piloto, sin reajustar pesos. "
              "La cola tiene un presupuesto activo de 24 h, 300 s por caso ordinario y 600 s por caso grande.",
              "\nE1/E2/E3 originales tienen 800 grafos totales y tres cintas a 120 épocas; su reparto por grado y número de rondas supervisadas difiere. "
              "La comparación registra tiempo y presupuesto real. E2F/E3F de la cola de Claude tienen más grafos totales y no se mezclan con esos controles.",
              "\n## Límites de las conclusiones",
              "\nEstos resultados son evidencia finita. Ni el número fijo de parámetros, ni aceptar paletas variables, ni los éxitos observados prueban coloración correcta para grados arbitrariamente grandes. "
              "La investigación distingue capacidad arquitectónica, corrección empírica, fidelidad local, escalamiento de tamaño y coste de recursos.",
              "\n## Reproducción y referencias",
              "\nCódigo original: `src/dcl/universal/{model,data,rules,train,evaluation}.py`. Código del estudio: `src/dcl/extreme/`. "
              "Protocolo previo: `configs/ucgnn/extreme-v1.json` y `docs/UCGNN_EXTRAPOLACION.md`. "
              "Inventario/checkpoints: `../audit/inventory.json`. Cada resultado JSON conserva hashes, semillas, configuración, versión Git y métricas por k. "
              "Los archivos M6 y checkpoints UCGNN originales se leen sin escribirlos.",
              "\nComandos desde el worktree: `experiments/ucgnn-extreme status`; `experiments/ucgnn-extreme report`; "
              "`experiments/ucgnn-extreme start` para crear o recuperar tmux; `tmux attach -t ucgnn-extrapolation`. "
              "El estado persistente permite reanudar tras perder el contenedor, aunque tmux no sobreviva a su destrucción."]
    (out/"INFORME.md").write_text("\n".join(lines)+"\n")
    figures(out,frame,diagnostics,info)
    return info


def figures(out,frame,diagnostics,info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out/"UCGNN_extrapolacion_parcial.pdf") as pdf:
        fig=plt.figure(figsize=(8.27,11.7))
        title="UCGNN · Auditoría y extrapolación extrema"
        paragraphs=[title,info["generated_utc"],
                    "Resultado histórico verificado: 4.880/4.880; una réplica, época 120.",
                    "Δ=3/5/8/10 hasta N=40.960. Δ=4/6/7/12/15 hasta N=10.240.",
                    "Reproducción independiente CPU: 16 casos sin discrepancias discretas.",
                    "16.035 parámetros. Mismos pesos para todas las paletas.",
                    "Arquitectura con prefijos normalizados explícitos; sin shield en inferencia.",
                    f"Piloto CPU: {info['cpu_pilot']['successes']}/{info['cpu_pilot']['cases']} hasta Δ={info['cpu_pilot']['max_degree']}.",
                    f"Principal CUDA: {info['cuda']['successes']}/{info['cuda']['cases']} hasta Δ={info['cuda']['max_degree']}.",
                    f"Coordinador: {info['coordinator']['status']}.",
                    "Éxito de coloración y fidelidad exacta a Luby son propiedades distintas.",
                    "Los resultados de CPU y CUDA se presentan separados.",
                    "Intervalos por modelo/cohorte; grafos compartidos no multiplican la muestra.",
                    "Consultar INFORME.md y CSV/JSON asociados para denominadores y límites.",
                    "Evidencia finita: no es una demostración para grados arbitrarios."]
        for i,text in enumerate(paragraphs):
            fig.text(.08,.94-i*.052,text,fontsize=14 if i==0 else 10,weight="bold" if i==0 else "normal")
        pdf.savefig(fig);plt.close(fig)
        if not diagnostics.empty:
            fig,ax=plt.subplots(figsize=(9,5))
            subset=diagnostics[(diagnostics["pattern"]=="SCATTERED") & (diagnostics["k"]==diagnostics["palette"])]
            for (run,device),group in subset.groupby(["run","device"]):
                if "s00" not in run: continue
                group=group.sort_values("palette")
                ax.plot(group["palette"],100*group["uniform_grid_accuracy"],"o-",label=run.split("-")[0]+" "+device)
            ax.set(xscale="log",xlabel="Paleta K (k=K colores disponibles)",ylabel="Selección exacta del profesor (%)",ylim=(-2,102),
                   title="Fidelidad local: barrido controlado de r_c (1.001 valores)")
            ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
            fig.savefig(out/"candidate-fidelity.png",dpi=160);pdf.savefig(fig);plt.close(fig)
        if not frame.empty and "inference_seconds" in frame:
            measured=frame[frame["success"].notna()]
            fig,axes=plt.subplots(1,2,figsize=(11,4.5))
            for cohort,group in measured.groupby("cohort"):
                group=group[group["family"]=="REGULAR"].sort_values("degree")
                axes[0].scatter(group["degree"],group["inference_seconds"],label=cohort,s=22)
            axes[0].set(xscale="log",yscale="log",xlabel="Grado máximo",ylabel="Segundos de inferencia por grafo",title="Tiempo observado; ver N en CSV")
            axes[0].legend(fontsize=8)
            gpu=measured[measured["device"]=="cuda"]
            if "gpu_peak_bytes" in gpu and len(gpu):
                axes[1].scatter(gpu["degree"],gpu["gpu_peak_bytes"]/2**20,s=22)
            axes[1].set(xscale="log",xlabel="Grado máximo",ylabel="Máximo asignado CUDA (MiB)",title="Memoria medida; principal CUDA")
            for ax in axes: ax.grid(alpha=.25)
            fig.tight_layout();fig.savefig(out/"resources.png",dpi=160);pdf.savefig(fig);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    args=p.parse_args()
    print(json.dumps(build(args.root)),flush=True)


if __name__=="__main__": main()
