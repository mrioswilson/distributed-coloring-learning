"""Freeze a local, immutable scientific handoff; no network or checkpoint writes.

Run from the worktree with PYTHONPATH=src. The snapshot stores the raw-record
references and summaries, not the historical checkpoints or graph datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from dcl.extreme.common import atomic_json,file_hash,now,provenance
from dcl.extreme.report import build,markdown_table


def text_page(pdf,title,paragraphs):
    fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.08,.95,title,fontsize=17,weight="bold")
    y=.90
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph,width=102):
            fig.text(.08,y,line,fontsize=10)
            y-=.021
        y-=.018
    pdf.savefig(fig);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("data/ucgnn-extreme"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root,out=args.root.resolve(),args.output.resolve()
    if out.exists(): raise SystemExit("Snapshot already exists; choose a new name")
    summary=build(root)
    cuda_cases=summary["cuda"]["cases"]
    waiting=summary["coordinator"]["status"]=="waiting_for_existing_jobs"
    cuda_note=("CUDA principal aún no ha comenzado: la cola persistente espera a que termine la cola previa de Claude, sin interrumpirla. "
               "Se ejecutará sola, empezando por la equivalencia CUDA y el piloto." if waiting and not cuda_cases else
               f"Estado del coordinador: {summary['coordinator']['status']}. CUDA principal: {summary['cuda']['successes']}/{cuda_cases} casos correctos.")
    out.mkdir(parents=True)
    audit=out/"audit"
    audit.mkdir()
    for name in ("gate.json","inventory.json","reproduction.json","equivalence-cpu.json",
                 "ablation-and-epochs.json","m6-preservation.json","claude-evaluation-status.json"):
        if (root/"audit"/name).exists(): shutil.copy2(root/"audit"/name,audit/name)
    for path in (root/"report").iterdir():
        if path.suffix in (".csv",".png",".md",".json"):
            shutil.copy2(path,out/path.name)
    for name in ("status.json","queue-manifest.json"):
        if (root/name).exists(): shutil.copy2(root/name,out/name)
    shutil.copy2(root/"logs/tests-final.log",audit/"tests-final.log")
    live=json.loads((root/"audit/claude-evaluation-status.json").read_text())
    e1=[{"view":key,**value} for key,value in live["views"].items()
        if key.startswith("e1-") and key.endswith("epoch-120")]
    e1=pd.DataFrame(e1).sort_values("view")
    e1.to_csv(out/"original-e1-all-replicates.csv",index=False)
    aggregate=int(e1["cases"].sum())
    successes=int(e1["successes"].sum())
    completed=int((e1["cases"]==4880).sum())
    checkpoint_hashes={r["run"]:{"checkpoint":r.get("checkpoint"),"checkpoint_sha256":r.get("checkpoint_sha256"),
                                 "weights_sha256":r.get("weights_sha256")}
                       for r in json.loads((root/"audit/inventory.json").read_text())["runs"] if r.get("checkpoint")}
    atomic_json(audit/"frozen-checkpoints.json",checkpoint_hashes)
    notes=("\n\n## Actualización del inventario original de Claude\n\n"
           f"Fotografía: {live['utc']}. Las {completed} réplicas E1-0…4 tienen {successes}/{aggregate} "
           "evaluaciones correctas, sobre los mismos 4.880 grafos: son 24.400 ejecuciones, no 24.400 grafos independientes. "
           "Esto amplía la reproducibilidad del resultado histórico hasta Δ=15; no añade pruebas históricas de Δ>15.\n\n"
           +markdown_table(e1)+"\n\n"+cuda_note+" "
           "Los 50 experimentos CPU reales ya están persistidos. El presupuesto activo de 24 h se cuenta después de la espera.\n")
    # Rewrite copied references to the snapshot's accompanying audit directory.
    report=(out/"INFORME.md").read_text().replace("../audit/","audit/")
    (out/"INFORME.md").write_text(report+notes)
    with PdfPages(out/"UCGNN_informe_auditoria.pdf") as pdf:
        text_page(pdf,"UCGNN · Auditoría y extrapolación",[
            "Informe parcial. Fecha UTC: "+now(),
            "Hallazgo principal: una red entrenada sólo con Δ=3 y cuatro colores resolvió 50/50 pruebas funcionales CPU hasta Δ=200 y 201 colores. "
            "Una réplica, diez grados y cinco familias; un caso por combinación. No es aún una batería estadística de grados extremos.",
            "La selección aprendida se aparta fuertemente de Luby: con k=K=201 coincide en 5/1.001 valores de la malla uniforme. "
            "En ese estado identifica disponibilidad e intervalos con gran precisión. Coloración correcta y copia exacta del profesor son propiedades distintas.",
            f"Verificación histórica: 4.880/4.880 para la réplica inicial. El inventario posterior contiene {completed} réplicas E1 completas: "
            f"{successes}/{aggregate} ejecuciones sobre 4.880 grafos compartidos. Los grados extra hasta Δ=15 llegan a N=10.240, no a N=40.960.",
            "Se reprodujeron 16 casos históricos sin discrepancias discretas. Mismos pesos y mismos objetos de parámetros tras Δ=3/10/20. "
            "No se detectó shield, corrección externa de propuestas ni oracle en la ruta neuronal auditada.",
            "Advertencia sobre checkpoints: best.pt de E1-0 es época 1 y falla 16 de 64 estados controlados de COMMIT. "
            "Época 120 acierta 64/64. Se evalúa el checkpoint predefinido de época 120.",
            cuda_note+" Verificación CUDA obligatoria antes del piloto. No hay sustitución automática por CPU.",
            "Conclusión: evidencia favorable de extrapolación funcional con estrategia de selección distinta; sin demostración para grados arbitrarios. "
            "La reproducibilidad de grados >15 con las cinco réplicas queda pendiente de la cola principal."])
        text_page(pdf,"Qué se verificó y qué significa",[
            "Arquitectura NORMALIZED: 16.035 parámetros; dimensión de estado/mensaje 32, capas ocultas 64. "
            "No se crean capas al cambiar K ni hay embeddings específicos de nombres de colores.",
            "Hay conocimiento algorítmico explícito: suma, prefijo acumulado y normalización de disponibilidad aprendida. "
            "El scorer debe aprender a convertir r_c en selección. El denominador se trunca a 0,5; la suma de α no es por sí sola un conteo exacto.",
            "Dos etapas de comunicación inmediata. El máximo de mensajes no depende de multiplicidades de vecinos. "
            "La fragmentación recalcula la primera etapa sobre el mismo estado; no incorpora otra dependencia de vecinos lejanos.",
            "Resultado histórico: Δ=3/5/8/10, seis tamaños de 40 a 40.960, 100/100/100/50/25/10 grafos por familia y tamaño, total 3.080. "
            "Δ=4/6/7/12/15, cinco tamaños hasta 10.240, 50/50/50/20/10 por familia, total 1.800. REGULAR e IRREGULAR.",
            "Las cinco réplicas E1 aciertan 64/64 conjuntos de tipos vecinos en COMMIT; C1 Own-only acierta 48/64 y acepta conflictos con colores fijos. "
            "Es un diagnóstico del camino local de COMMIT, no una prueba de terminación o de la política completa.",
            "E1-0/C1-0 tienen datos e inicialización idénticos y pesos de participación/candidato idénticos bit a bit. "
            "El contrafactual cambia COMMIT; la separación de parámetros y las decisiones duras impiden su gradiente hacia candidato/participación.",
            "Tests: 209 aprobados, uno omitido en CPU. Se aislaron los locks de tests, sin tocar el lock real de Claude ni las pruebas originales. "
            "Incluyen equivalencia por fragmentos hasta 501 colores, reanudación, clasificación de OOM y muerte abrupta del coordinador.",
            "Preservación: 21 archivos publicados de M6 mantienen su SHA256. Checkpoints originales sólo leídos. "
            "246.520 fingerprints históricos excluidos en la fotografía inicial. No se hizo push ni publicación remota."])
        fig=plt.figure(figsize=(11.69,8.27))
        ax=fig.add_axes([.05,.14,.9,.78]);ax.imshow(plt.imread(root/"report/candidate-fidelity.png"));ax.axis("off")
        fig.text(.08,.07,"Malla uniforme controlada; no son grafos aleatorios independientes. C1 coincide con E1 en candidato.",fontsize=11)
        pdf.savefig(fig);plt.close(fig)
        text_page(pdf,"Batería pendiente y recursos",[
            "Etapa A: Δ=3/5/8/10/15/20/30/50/100/200; REGULAR, IRREGULAR, HETEROGENEOUS, BIPARTITE y CLIQUE. "
            "N=max(40,2(Δ+1)), salvo completos N=Δ+1. Etapa exploratoria Δ=500 condicionada al umbral de éxito predeclarado del piloto.",
            "Etapa B: cinco réplicas E1, 20 grafos por familia principal a N=160/640/2.560 cuando sea válido. "
            "Prioridad Δ=15/20/30/50/100; ampliación a 60 grafos en N=640. Cinco modelos por 20 grafos son 100 ejecuciones sobre 20 grafos.",
            "Grandes: N=10.240/40.960 para grados seleccionados por un límite lógico de mensajes de 32 GiB/ronda y memoria estimada ≤3 GiB. "
            "Cinco grafos por familia y modelo, exploratorios. No se lanza Δ=200,N=40.960.",
            "Memoria original: activaciones N·K·32 y mensajes temporales E_chunk·K·32. El evaluador usa bloques de ocho colores y 32 MiB de mensajes. "
            "No hay adyacencia densa N×N. Mantiene los pesos y exige equivalencia de decisiones en los casos de validación.",
            "Comunicación lógica original: 2·E_dir·K·32·4 bytes/ronda. Trabajo de mensajes del evaluador por fragmentos: 3·E_dir·K·32·4 por recalcular la primera etapa. "
            "Los parámetros constantes no implican memoria ni comunicación constantes.",
            "Tiempos de inferencia CPU REGULAR del piloto: Δ=3,N=40: 0,010 s; Δ=20,N=42: 0,028 s; "
            "Δ=50,N=102: 0,271 s; Δ=100,N=202: 2,820 s; Δ=200,N=402: 31,077 s. Cambian N y Δ a la vez; no aíslan el efecto del grado. "
            +("Aún no hay memoria CUDA medida para la batería nueva." if not cuda_cases else "Las mediciones CUDA se conservan en cases.csv."),
            "Límites: 24 h de trabajo activo, 300 s por caso ordinario, 600 s por caso grande, asignador CUDA ≤3 GiB, reserva externa ≥768 MiB. "
            "OOM/timeout se registran como técnicos; un fallo de coloración no detiene el estudio. No se detienen servicios ajenos.",
            "Comparación posterior: E2/E3 existentes y controles A1/C1 sobre grafos compartidos. E1/E2/E3 tienen 800 grafos totales, tres cintas y 120 épocas, "
            "pero rondas supervisadas y cómputo distintos. E2F/E3F de Claude tienen presupuesto mayor y se mantienen separados."])
        text_page(pdf,"Cómo revisar y reproducir",[
            "Worktree: /workspaces/AlgoLearning/ucgnn-extreme-degree-study. Rama: feat/ucgnn-extreme-degree-study. "
            "El snapshot incluye hashes y la revisión del código en snapshot.json.",
            "Comandos: experiments/ucgnn-extreme status ; experiments/ucgnn-extreme report ; tmux attach -t ucgnn-extrapolation . "
            "Para recuperar la sesión tras una interrupción: experiments/ucgnn-extreme start .",
            "Estado persistente: data/ucgnn-extreme/status.json y budget.json. Resultados: results/ y cpu-pilot/. "
            "Logs: logs/. Curvas controladas y NPZ: diagnostics/. Checkpoints originales: /workspaces/AlgoLearning/distributed-coloring-learning/data/ucgnn/runs/.",
            "Referencias de esta entrega: INFORME.md, original-cohorts.csv, original-e1-all-replicates.csv, controlled-selection.csv, "
            "controlled-commit-and-participation.csv, success-by-model-degree-size-family.csv, cases.csv y audit/reproduction.json.",
            "La cola conserva resultados por grafo, semillas, hashes, versión Git, métricas locales, tiempos y recursos; "
            "no repite los completados. La sesión tmux resiste cerrar SSH. Tras destruir el contenedor hay que restaurar el entorno y ejecutar start; "
            "el volumen contiene el estado necesario.",
            "Los intervalos de confianza son exactos Clopper–Pearson del 95 % por modelo, grado, tamaño y familia. "
            "No se obtiene un intervalo binomial global mezclando modelos sobre los mismos grafos. Los completos son estrés determinista.",
            "Este documento es una fotografía parcial. Los datos en ejecución pueden avanzar después de su fecha. "
            "El informe vivo se reconstruye desde los JSON sin cambiar pesos ni corregir decisiones."])
    (out/"README.md").write_text(
        "# UCGNN: entrega parcial de auditoría\n\n"
        "[Informe PDF](UCGNN_informe_auditoria.pdf) · [Informe detallado](INFORME.md)\n\n"
        "Snapshot local, sin publicación remota. Auditoría histórica, 50 pruebas CPU hasta Δ=200 y diagnósticos de nueve modelos. "
        +cuda_note+" Su estado temporal se conserva en status.json.\n\n"
        "El resultado histórico está replicado en cinco E1: 24.400/24.400 ejecuciones sobre 4.880 grafos compartidos hasta Δ=15. "
        "El nuevo piloto extremo usa una réplica: 50/50, sin una afirmación de corrección universal.\n\n"
        "[Protocolo y operación](../../../docs/UCGNN_EXTRAPOLACION.md). "
        "Para regenerar una entrega nueva: PYTHONPATH=src python experiments/ucgnn_extreme_snapshot.py --output RUTA_NUEVA.\n")
    files={str(p.relative_to(out)):file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    atomic_json(out/"snapshot.json",{"utc":now(),"provenance":provenance(),"summary":summary,
                                    "original_e1_models_complete":completed,"original_e1_model_evaluations":aggregate,
                                    "original_e1_unique_shared_graphs":4880,"files_sha256":files})
    print(out/"UCGNN_informe_auditoria.pdf")


if __name__=="__main__":main()
