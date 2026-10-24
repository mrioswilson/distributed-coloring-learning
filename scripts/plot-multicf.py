#!/usr/bin/env python3
"""Publication figures; show five unsmoothed seed curves and exact denominators."""
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcl.study.storage import atomic_json,file_hash

OUT=Path("results/multiteacher-counterfactual")
FIG=OUT/"figures"


def save(fig,name):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf","png"):fig.savefig(FIG/f"{name}.{ext}",dpi=170,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    spec=importlib.util.spec_from_file_location("multicf_analysis",Path(__file__).with_name("analyze-multicf.py"))
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.figures()
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    rows=json.loads((OUT/"supervision-summary.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.2),sharey=True)
    for ax,arch in zip(axes,("normalized","deepsets","transformer")):
        for u,color in ((500,"#c07000"),(1500,"#007f86")):
            rr=sorted([r for r in rows if r["architecture"]==arch and r["update"]==u],key=lambda r:r["native_budget"])
            for seed in range(5):
                ax.plot([r["actual_total_labels"][seed] for r in rr],
                    [r["development_successes"][seed]/60 for r in rr],color=color,alpha=.28,lw=.9)
            ax.plot([r["actual_total_labels"][0] for r in rr],
                [np.mean(r["development_successes"])/60 for r in rr],color=color,marker="o",label=f"{u} updates")
        ax.set_xscale("log");ax.set_ylim(-.02,1.04);ax.axhline(1,color="black",ls=":",lw=.8)
        ax.set_title(arch);ax.set_xlabel("Acquired P/C/COMMIT + CF labels")
        ax.grid(alpha=.16)
    axes[0].set_ylabel("Development success (60 per policy)");axes[-1].legend(loc="lower right")
    fig.suptitle("ColorTrials: fixed optimizer effort, nested acquired-label banks",y=1.02)
    fig.tight_layout();save(fig,"supervision-curves")
    fidelity=json.loads((OUT/"fidelity-summary.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    colors={"passive":"#718096","matched":"#c07000","counterfactual":"#007f86"}
    for ax,t in zip(axes,("BE_BASE_D3","SPECULATIVE_GM","COLOR_TRIALS")):
        for j,method in enumerate(colors):
            rr=[next(r for r in fidelity if r["teacher"]==t and r["architecture"]==a and r["method"]==method)
                for a in ("normalized","deepsets","transformer")]
            rates=[100*(1-r["counterfactual_correct"]/r["counterfactual_total"]) for r in rr]
            bars=ax.bar(np.arange(3)+(j-1)*.24,rates,width=.23,label=method,color=colors[method])
            for bar,value in zip(bars,rates):
                label="0" if value==0 else (f"{value:.3f}" if value<.1 else f"{value:.1f}")
                ax.text(bar.get_x()+bar.get_width()/2,value+.5,label,ha="center",fontsize=7)
        ax.set_xticks(range(3),["E1","DeepSets","Transformer"]);ax.set_title(t)
        ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel("Counterfactual confirmation errors (%)")
    axes[-1].legend(loc="upper right",fontsize=8)
    fig.suptitle("4000 updates; five paired seeds; fresh development interventions",y=1.01)
    fig.tight_layout();save(fig,"counterfactual-fidelity")
    info=json.loads((OUT/"information/results.json").read_text())
    modes=("without_ids","numeric_ids","numeric_differences","relative_comparisons")
    fig,ax=plt.subplots(figsize=(7,3.4))
    for i,mode in enumerate(modes):
        values=[r["development"]["successes"]/60 for r in info if r["mode"]==mode]
        ax.bar(i,np.mean(values),color="#007f86" if i==3 else "#b8c4cc",width=.65)
        ax.scatter(i+np.linspace(-.14,.14,5),values,c="#29364a",s=22,zorder=3)
    ax.set_xticks(range(4),["No IDs","Numeric IDs","Numeric differences","Relative comparisons"])
    ax.set_ylim(0,1.08);ax.set_ylabel("Development success (60 per policy)")
    ax.axhline(1,color="black",ls=":",lw=.8)
    ax.set_title("JP / E1: same 3000 native-label budget and 500 Adam updates")
    ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True);fig.tight_layout();save(fig,"jp-information")
    atomic_json(FIG/"manifest.json",dict(source_sha256=file_hash(__file__),
        inputs={p:file_hash(OUT/p) for p in ("training-curves.json","supervision-summary.json","fidelity-summary.json","information/results.json")},
        outputs={str(p):file_hash(p) for p in sorted(FIG.glob("*")) if p.suffix in ("png","pdf")}))
