"""Descriptive analyses of the completed, fixed-protocol measurements."""
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY=("normalized","raw_endpoints","sum","mean","deepsets","transformer")


def analyze():
    from .report import OUTPUT,PAPER,csv,latex_table
    from .config import ROOT
    source=OUTPUT/"cases.csv"
    if not source.exists() or source.stat().st_size<10:return ""
    cases=pd.read_csv(source)
    if cases.empty:return ""
    primary=cases[(cases.kind=="paper")&(cases.selection=="epoch120")&cases.treatment.isin(PRIMARY)&
                  cases.status.isin(["ok","coloring_failure"])].copy()
    if primary.empty:return ""
    seeds=primary.groupby(["treatment","seed"],as_index=False).agg(
        executions=("success","size"),successes=("success","sum"),unsafe=("unsafe_commits","sum"),
        monochromatic_edges=("monochromatic_edges","sum"),pending=("uncolored_nodes","sum"),
        unfinished=("finished",lambda x:int((~x).sum())),rounds_mean=("rounds","mean"),rounds_max=("rounds","max"))
    seeds["success_rate"]=seeds.successes/seeds.executions
    csv(OUTPUT/"overall-by-seed.csv",seeds)
    summary=seeds.groupby("treatment").agg(networks=("seed","nunique"),executions=("executions","sum"),
        successes=("successes","sum"),seed_rate_min=("success_rate","min"),seed_rate_max=("success_rate","max"),
        unsafe=("unsafe","sum"),monochromatic_edges=("monochromatic_edges","sum"),unfinished=("unfinished","sum"),
        pending=("pending","sum"),rounds_max=("rounds_max","max")).reindex(PRIMARY).reset_index()
    csv(OUTPUT/"overall-results.csv",summary)
    table=pd.DataFrame({"Model":summary.treatment,"Networks":summary.networks,
        "Proper / executions":[f"{s}/{n}" for s,n in zip(summary.successes,summary.executions)],
        "Seed rate range":[f"{a:.3f}--{b:.3f}" for a,b in zip(summary.seed_rate_min,summary.seed_rate_max)],
        "Unfinished":summary.unfinished,"Unsafe COMMIT":summary.unsafe})
    latex_table(table,PAPER/"tables/overall.tex",
        "Primary epoch-120 algorithmic outcomes across all strata. Totals are descriptive: networks share 288 graph identities and 304 graph--tape cases. Technical failures are reported separately. Unsafe denotes violation of the exact rejection rule; final improper coloring is assessed separately.","tab:overall")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    candidates=pd.read_csv(OUTPUT/"candidate-diagnostics.csv")
    free=candidates[(~candidates.heldout_prediction)&(candidates.pattern=="LOW")&
                    (candidates.available_count==candidates.palette)&candidates.treatment.isin(PRIMARY)].copy()
    csv(OUTPUT/"all-free-candidates.csv",free)
    fig,axes=plt.subplots(1,2,figsize=(8,3.1))
    for name,g in free.groupby("treatment"):
        for ax,y in zip(axes,("effective_colors","max_mass")):
            grouped=g.groupby("palette")[y]
            line=ax.plot(grouped.mean().index,grouped.mean().values,marker="o",label=name)[0]
            ax.fill_between(grouped.mean().index,grouped.min(),grouped.max(),alpha=.12,color=line.get_color())
    for ax in axes:ax.set(xscale="log",xlabel="Palette size K");ax.grid(alpha=.2)
    axes[0].set(ylabel="Effective candidate colors",yscale="log")
    axes[1].set(ylabel="Largest grid probability")
    axes[0].legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(PAPER/"figures/candidate-diversity.pdf");plt.close(fig)

    progress=pd.read_csv(OUTPUT/"progress-diagnostics.csv")
    traces=pd.read_csv(OUTPUT/"progress-round-traces.csv")
    first=traces[traces["round"]==1].groupby(["label","state_id"],as_index=False).agg(
        empirical_safe_first=("safe_proposals","mean"),empirical_commits_first=("confirmations","mean"))
    ps=progress.groupby(["label","treatment","state_id"],as_index=False).agg(
        tapes=("tape","size"),successes=("success","sum"),rounds_mean=("rounds","mean"),rounds_max=("rounds","max"),
        first_confirmation_censored=("first_confirmation_right_censored","sum"),
        max_wait=("max_interconfirmation_wait","max"),unsafe=("unsafe_commits","sum"),
        predicted_safe_first=("predicted_safe_proposals_first_round","first"))
    ps=ps.merge(first,on=["label","state_id"],validate="one_to_one")
    csv(OUTPUT/"progress-state-summary.csv",ps)
    fig,axes=plt.subplots(1,2,figsize=(8,3.1))
    for name,g in ps[ps.treatment.isin(PRIMARY)].groupby("treatment"):
        axes[0].scatter(g.predicted_safe_first,g.empirical_safe_first,s=12,label=name)
        axes[1].scatter(g.predicted_safe_first,g.empirical_commits_first,s=12,label=name)
    for ax in axes:
        ax.plot([0,17],[0,17],"--",color="gray",lw=.8)
        ax.set(xlabel="Predicted exact-rule safe proposals",xlim=(-.3,17),ylim=(-.3,17));ax.grid(alpha=.2)
    axes[0].set(ylabel="Observed safe proposals, first round")
    axes[1].set(ylabel="Observed confirmations, first round")
    axes[0].legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(PAPER/"figures/progress-predictions.pdf");plt.close(fig)

    resource=pd.read_csv(OUTPUT/"resources.csv")
    rr=resource[resource.treatment.isin(PRIMARY)&(resource.degree==200)].copy()
    rr["allocated_mib"]=rr.peak_gpu_bytes/2**20;rr["forward_ms"]=rr.forward_seconds*1000
    rr["dense_gflops"]=rr.approx_dense_flops/1e9
    csv(OUTPUT/"resource-summary.csv",rr)
    latex_table(rr[["treatment","parameters","dense_gflops","allocated_mib","forward_ms"]].rename(columns={
        "treatment":"Model","parameters":"Params","dense_gflops":"Dense GFLOPs","allocated_mib":"Allocated MiB","forward_ms":"Round ms"}),
        PAPER/"tables/resources.tex","One-round resource probe at N=400, degree 200, K=201 (one prespecified seed per model). FLOPs are approximate; neighbor reductions, nonlinearities and overhead are itemized separately in CSV. CUDA allocated memory excludes driver/reserved memory.","tab:resources")

    certificates=pd.read_csv(OUTPUT/"commit-certificates.csv")
    latex_table(certificates.groupby("treatment",as_index=False).agg(
        Networks=("label","size"),Applicable=("finite_max_applicable","sum"),
        Certified=("real_exact_rule_certified",lambda x:int(x.eq(True).sum())),
        Unsafe_masks=("unsafe_masks_certified","sum"),Margin_min=("real_margin_lower","min")),
        PAPER/"tables/certificates.tex","Finite-type real-arithmetic certificates, with explicit arithmetic assumptions. Unsafe mask totals across networks are descriptive. SUM/MEAN are outside this certificate's scope.","tab:certificates")

    # A separate target is essential: the existing teacher_fidelity column
    # always refers to the original uniform-available teacher.
    from .teacher import cyclic_law
    tail=[]
    for path in (ROOT/"mechanisms/selection").glob("*/K201-k201-LOW.npz"):
        with np.load(path,allow_pickle=False) as f:
            q=f["q"]
            for beta in (2,4,8):
                eta=float(q[q<=beta/201].sum())
                tail.append({"label":path.parent.name,"palette":201,"beta":beta,"diffuse_mass_eta":eta,
                    "conditional_lower_bound_p_half":.5*eta*.5**(beta/eta) if eta else 0.,
                    "assumptions":"grid q approximation; participation probability assumed exactly one half; exact COMMIT and private independence"})
    csv(OUTPUT/"diffuse-tail-mass.csv",tail)
    cyclic=[]
    for path in sorted((ROOT/"mechanisms/selection/cyclic_teacher-s41005-epoch120").glob("*.npz")):
        if "-prediction" in path.stem:continue
        with np.load(path,allow_pickle=False) as f:
            k=len(f["q"]);available=f["available"]
            mask=np.zeros(k,bool);mask[available]=True
            start=np.floor(f["rc"].astype(np.float64)*k).astype(int)
            target=available[np.searchsorted(available,start)%len(available)]
            meta=next(candidates[(candidates.label=="cyclic_teacher-s41005-epoch120")&
                                (candidates.palette==k)&(candidates.available_count==len(available))&
                                (candidates.pattern==path.stem.rsplit("-",1)[1])].itertuples())
            cyclic.append({"palette":k,"available_count":len(available),"pattern":meta.pattern,
                "cyclic_teacher_fidelity":float((f["choices"]==target).mean()),
                "tv_cyclic_teacher":float(.5*np.abs(f["q"]-cyclic_law(mask)).sum()),
                "uniform_teacher_fidelity":meta.teacher_fidelity,"forbidden_rate":meta.forbidden_rate})
    csv(OUTPUT/"cyclic-candidate-diagnostics.csv",cyclic)
    alternative=cases[(cases.treatment=="cyclic_teacher")&(cases.selection=="epoch120")].groupby("degree",as_index=False).agg(
        executions=("success","size"),successes=("success","sum"),states=("states","sum"),
        cyclic_correct=("cyclic_candidate_correct","sum"),uniform_correct=("candidate_correct","sum"),
        forbidden=("forbidden_offers","sum"),rounds_mean=("rounds","mean"),rounds_max=("rounds","max"))
    alternative["cyclic_fidelity"]=alternative.cyclic_correct/alternative.states
    alternative["uniform_fidelity"]=alternative.uniform_correct/alternative.states
    csv(OUTPUT/"cyclic-degree-summary.csv",alternative)
    return "```csv\n"+summary.to_csv(index=False).rstrip()+"\n```"
