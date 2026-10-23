"""Rebuild unsmoothed per-seed curves and tables from frozen observations."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from dcl.study.storage import atomic_json, file_hash
from .protocol import REGIME

OUT=Path("results/sample-complexity")
NAMES={"normalized":"E1 NORMALIZED","deepsets":"DeepSets","transformer":"LocalPaletteTransformer"}


def read(path): return json.loads(Path(path).read_text())


def markdown_table(headers,rows):
    return "| "+" | ".join(headers)+" |\n|"+"|".join("---" for _ in headers)+"|\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in rows)+"\n"


def figures(records,config):
    outputs={};folder=OUT/"figures";folder.mkdir(parents=True,exist_ok=True)
    colors=plt.get_cmap("tab10").colors
    full_path=OUT/"full-fidelity.json"
    full_rows=read(full_path)["rows"] if full_path.exists() else []
    full_lookup={(r["teacher"],r["architecture"],r["seed"],r["budget"]):r for r in full_rows}
    for regime in ("A","B"):
        teachers=[t for t in config["teachers"] if REGIME[t]==regime]
        for metric in ("success","failures","fidelity","rounds"):
            fig,axes=plt.subplots(len(teachers),3,figsize=(10,2.8*len(teachers)),squeeze=False,sharex=True)
            for i,t in enumerate(teachers):
                for j,a in enumerate(config["architectures"]):
                    ax=axes[i,j]
                    for si,s in enumerate(config["seeds"]):
                        rs=sorted([r for r in records if (r["identity"]["teacher"],r["identity"]["architecture"],r["identity"]["seed"])==(t,a,s)],key=lambda r:r["identity"]["budget"])
                        x=[r["identity"]["budget"] for r in rs]
                        if metric=="success": y=[r["development"]["success_rate"] for r in rs]
                        elif metric=="failures": y=[r["development"]["executions"]-r["development"]["successes"] for r in rs]
                        elif metric=="fidelity": y=[full_lookup.get((t,a,s,r["identity"]["budget"]),r["fidelity"])["action_fidelity"] for r in rs]
                        else: y=[r["development"]["mean_native_steps"] for r in rs]
                        ax.plot(x,y,marker=("o","s","^","x","+")[si],ls="-",lw=1,ms=4,color=colors[si],label=str(s))
                    ax.set_xscale("log");ax.grid(alpha=.2)
                    if metric=="success":
                        ax.axhline(1,color="black",ls="--",lw=.8);ax.set_ylim(-.025,1.045)
                    elif metric=="fidelity": ax.set_ylim(-.025,1.045)
                    elif metric=="failures":
                        ax.set_yscale("symlog",linthresh=1);ax.axhline(0,color="black",ls="--",lw=.8)
                        ax.set_yticks([0,1,3,10,28],labels=["0","1","3","10","28"])
                    if metric in ("success","fidelity"): ax.yaxis.set_major_formatter(PercentFormatter(1))
                    ax.set_xticks(config["grid"],labels=("1k","3k","10k","30k","100k","300k","1m"),fontsize=7)
                    if i==0: ax.set_title(NAMES[a],fontsize=11)
                    if j==0: ax.set_ylabel(t+"\n"+{"success":"Development success","failures":"Failures (zero required)","fidelity":"Action fidelity","rounds":"Native steps attempted"}[metric],fontsize=10)
                    if i==len(teachers)-1: ax.set_xlabel("Acquired label budget B (log)",fontsize=10)
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc="lower center",ncol=5,title="Independent training seed",fontsize=9)
            fig.suptitle(f"Regime {regime}: {metric}; five separate unsmoothed development curves",fontsize=11)
            fig.tight_layout(rect=(0,.085,1,.97))
            for ext in ("png","pdf"):
                path=folder/f"{metric}-regime-{regime.lower()}.{ext}"
                fig.savefig(path,dpi=160,bbox_inches="tight");outputs[str(path)]=file_hash(path)
                if metric=="success":
                    shutil.copyfile(path,Path("paper/ucgnn/figures")/f"sample-{path.name}")
            plt.close(fig)
    atomic_json(OUT/"figure-manifest.json",dict(files=outputs,training_sha256=file_hash(OUT/"training.json"),
                                              full_fidelity_sha256=file_hash(full_path) if full_path.exists() else None))
    return outputs


def capacity_figure(records,config):
    path=OUT/"capacity/training.json"
    if not path.exists():return
    small=read(path);teachers=list(dict.fromkeys(r["identity"]["teacher"] for r in small))
    fig,axes=plt.subplots(len(teachers),2,figsize=(8,7.5),sharex=True,squeeze=False)
    colors=plt.get_cmap("tab10").colors
    for i,t in enumerate(teachers):
        for j,(label,source,arch) in enumerate((("E1 NORMALIZED",records,"normalized"),("Small E1",small,"small"))):
            ax=axes[i,j]
            for k,seed in enumerate(config["seeds"]):
                curve=sorted([r for r in source if (r["identity"]["teacher"],r["identity"]["architecture"],r["identity"]["seed"])==(t,arch,seed)],key=lambda r:r["identity"]["budget"])
                ax.plot([r["identity"]["budget"] for r in curve],[r["development"]["success_rate"] for r in curve],
                        marker=("o","s","^","x","+")[k],lw=1,ms=4,color=colors[k],label=str(seed))
            ax.set_xscale("log");ax.set_ylim(-.025,1.045);ax.axhline(1,color="black",ls="--",lw=.8)
            ax.set_xticks(config["grid"],labels=("1k","3k","10k","30k","100k","300k","1m"),fontsize=7)
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
            if i==0:ax.set_title(label)
            if j==0:ax.set_ylabel(t+"\nDevelopment success")
            if i==len(teachers)-1:ax.set_xlabel("Acquired label budget B (log)")
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=5,title="Training seed")
    fig.tight_layout(rect=(0,.09,1,1));manifest=read(OUT/"figure-manifest.json")
    for ext in ("png","pdf"):
        target=OUT/"figures"/f"capacity-comparison.{ext}";fig.savefig(target,dpi=160,bbox_inches="tight")
        manifest["files"][str(target)]=file_hash(target)
    manifest["small_training_sha256"]=file_hash(path);atomic_json(OUT/"figure-manifest.json",manifest);plt.close(fig)


def model_inventory(records,controls):
    entries=[]
    combined=records+[r["record"] for r in controls]
    for t,arch in sorted({(r["identity"]["teacher"],r["identity"]["architecture"]) for r in combined}):
        r=next(r for r in combined if (r["identity"]["teacher"],r["identity"]["architecture"])==(t,arch))
        entries.append(dict(teacher=t,architecture=arch,parameters=r["parameters"],regime=REGIME[t],
                    hidden_message_dimensions=[8,4] if arch=="small" else [64,32],
                    graph_aggregation="MAX",graph_exchange_stages=2,
                    received_information_erased=arch=="no_communication",
                    palette_representation={"deepsets":"Shared phi, mean palette context, ordered color position",
                                            "transformer":"Two-head within-palette attention, ordered color position"}.get(arch,"Normalized ordered availability prefixes"),
                    coded_rules="Pending/write permissions, phase/scheduler/bound transitions, palette alphabet, termination checks; BE polynomial/option decoder and auxiliary-target comparison",
                    learned_actions="Participation, proposal and confirmation; Linial point or GM conflict mark in corresponding phases",
                    inference_cost_source="Training/control development_seconds; native-efficiency.json for main candidate policies; partial profiler estimates for main architectures only"))
    atomic_json(OUT/"model-inventory.json",entries)


def analyze(records,certs,config):
    result=[]
    flop_rows=read(OUT/"flops-estimates.json")["training"] if (OUT/"flops-estimates.json").exists() else []
    for c in certs:
        rs=[r for r in records if r["identity"]["teacher"]==c["teacher"] and r["identity"]["architecture"]==c["architecture"]]
        grid=sorted({r["identity"]["budget"] for r in rs})
        by_budget={b:[r for r in rs if r["identity"]["budget"]==b] for b in grid}
        regressions=[]
        for seed in config["seeds"]:
            curve=sorted([r for r in rs if r["identity"]["seed"]==seed],key=lambda r:r["identity"]["budget"])
            for left,right in zip(curve,curve[1:]):
                if right["development"]["success_rate"]<left["development"]["success_rate"]:
                    regressions.append(dict(seed=seed,from_budget=left["identity"]["budget"],to_budget=right["identity"]["budget"],
                                            from_success=left["development"]["success_rate"],to_success=right["development"]["success_rate"]))
        final_failures=Counter()
        for s in c["per_seed"].values(): final_failures.update(s["failures"])
        row=dict(**c,development_perfect_seeds={str(b):sum(r["development"]["pass_all"] for r in by_budget[b]) for b in grid},
                 development_failures_at_max={str(r["identity"]["seed"]):r["development"]["failures"] for r in by_budget[grid[-1]]},
                 actual_queries_by_budget={str(b):{str(r["identity"]["seed"]):r["accounting"]["queries"] for r in by_budget[b]} for b in grid},
                 native_parameters=rs[0]["parameters"],total_training_seconds=sum(r["training_seconds"] for r in rs),
                 total_cpu_seconds=sum(r["cpu_seconds"] for r in rs),gpu_seconds=0.,final_failures=dict(final_failures),regressions=regressions,
                 first_individual_development_pass={str(s):next((r["identity"]["budget"] for r in sorted(rs,key=lambda r:r["identity"]["budget"]) if r["identity"]["seed"]==s and r["development"]["pass_all"]),None) for s in config["seeds"]})
        row["dominant_profiled_flops_estimate"]=sum(f["dominant_profiled_flops_estimate"] for f in flop_rows if
              f["teacher"]==c["teacher"] and f["architecture"]==c["architecture"]) if flop_rows else None
        result.append(row)
    return result


def generalization(records,certifications,root=Path("data/sample-complexity-v1")):
    """Export family/size/seed outcomes and matched exact-teacher time ratios."""
    output=[]
    for record in records:
        i=record["identity"]
        source=root/"runs"/i["teacher"].lower()/i["architecture"]/f"s{i['seed']}"/f"b{i['budget']}"/"development.json"
        if not source.exists():
            if (OUT/"generalization.json").exists(): return read(OUT/"generalization.json")
            continue
        rows=read(source)
        exact={r["case_id"]:r for r in read(OUT/"exact"/i["teacher"].lower()/f"development-s{i['seed']}.json")}
        for family,n in sorted({(r["family"],r["n"]) for r in rows}):
            group=[r for r in rows if (r["family"],r["n"])==(family,n)]
            good=[r for r in group if all(r[k] for k in ("proper","palette","invariants","terminated","no_technical_error","no_cutoff","terminal_correct"))]
            output.append(dict(split="development",**i,family=family,n=n,executions=len(group),successes=len(good),
                          failures=dict(Counter(r["failure"] for r in group if r not in good)),
                          mean_attempted_steps=float(np.mean([r["native_steps"] for r in group])),
                          successful_mean_step_ratio_to_matched_exact=float(np.mean([r["native_steps"]/exact[r["case_id"]]["native_steps"] for r in good])) if good else None,
                          source_sha256=file_hash(source)))
    for c in certifications:
        for seed in c["per_seed"]:
            source=OUT/"final"/c["teacher"].lower()/c["architecture"]/f"s{seed}.json"
            rows=read(source);exact={r["case_id"]:r for r in read(OUT/"exact"/c["teacher"].lower()/f"final-s{seed}.json")}
            for family,n in sorted({(r["family"],r["n"]) for r in rows}):
                group=[r for r in rows if (r["family"],r["n"])==(family,n)]
                good=[r for r in group if all(r[k] for k in ("proper","palette","invariants","terminated","no_technical_error","no_cutoff","terminal_correct"))]
                output.append(dict(split="final",teacher=c["teacher"],architecture=c["architecture"],seed=int(seed),budget=c["candidate_budget"],
                          family=family,n=n,executions=len(group),successes=len(good),
                          failures=dict(Counter(r["failure"] for r in group if r not in good)),
                          mean_attempted_steps=float(np.mean([r["native_steps"] for r in group])),
                          successful_mean_step_ratio_to_matched_exact=float(np.mean([r["native_steps"]/exact[r["case_id"]]["native_steps"] for r in good])) if good else None,
                          source_sha256=file_hash(source)))
    atomic_json(OUT/"generalization.json",output)
    return output


def efficiency(certifications):
    rows=[]
    for c in certifications:
        for seed in c["per_seed"]:
            actual=read(OUT/"final"/c["teacher"].lower()/c["architecture"]/f"s{seed}.json")
            exact={r["case_id"]:r for r in read(OUT/"exact"/c["teacher"].lower()/f"final-s{seed}.json")}
            good=[r for r in actual if r["terminal_correct"] and r["invariants"] and r["no_cutoff"] and r["no_technical_error"]]
            exact_stages=2 if c["teacher"] in ("LUBY","JOHANSSON","COLOR_TRIALS") else 1
            ratio=float(np.mean([r["native_steps"]/exact[r["case_id"]]["native_steps"] for r in good])) if good else None
            rows.append(dict(teacher=c["teacher"],architecture=c["architecture"],seed=int(seed),
                    budget=c["candidate_budget"],executions=len(actual),successful=len(good),
                    mean_attempted_native_steps=float(np.mean([r["native_steps"] for r in actual])),
                    exact_mean_native_steps=float(np.mean([exact[r["case_id"]]["native_steps"] for r in actual])),
                    success_conditional_mean_paired_step_ratio=ratio,
                    neural_graph_exchange_stages_per_step=2,
                    exact_nominal_exchange_stages_per_step=exact_stages,
                    nominal_exchange_stage_ratio=2/exact_stages*ratio if ratio is not None else None,
                    measured_cohort_wall_seconds=c["per_seed"][seed]["seconds"],
                    scope="Within-teacher native steps; nominal stages assume reset-state neighbor reads, not actual network measurements. Scheduler/arithmetic setup remains supplied. Failed attempted cost retained separately; wall time includes feature encoder/interpreter"))
    atomic_json(OUT/"native-efficiency.json",rows)
    return rows


def primary_table(rows,regime=None):
    ts=list(dict.fromkeys(r["teacher"] for r in rows if regime is None or r["regime"]==regime))
    body=[]
    for t in ts:
        line=[t]
        for a in NAMES:
            r=next(r for r in rows if r["teacher"]==t and r["architecture"]==a)
            if r["status"]=="PASS": value=f"{r['validated_budget']:,}; {r['seeds_pass']}/5 PASS"
            elif r["final_attempted"]: value=f"NOT CERTIFIED; {r['seeds_pass']}/5 (candidate {r['candidate_budget']:,})"
            else: value=f"NOT CERTIFIED; 0/5 confirmed (no candidate; max {r['max_budget_evaluated']:,})"
            line.append(value)
        body.append(line)
    return markdown_table(["Teacher","E1: B*","DeepSets: B*","Transformer: B*"],body)


def control_table(controls,records,config):
    rows=[]
    for t in config["optimization_control_teachers"]:
        baseline=[r for r in records if r["identity"]["teacher"]==t and r["identity"]["architecture"]=="normalized" and r["identity"]["budget"]==config["control_budget"]]
        values=[f"{sum(r['development']['pass_all'] for r in baseline)}/5"]
        for kind,arch in (("fixed_labels_more_optimization","normalized"),("capacity_or_communication","small"),
                          ("capacity_or_communication","no_communication"),("interactive","normalized")):
            group=[r["record"] for r in controls if r["kind"]==kind and r["record"]["identity"]["teacher"]==t and r["record"]["identity"]["architecture"]==arch]
            values.append(f"{sum(r['development']['pass_all'] for r in group)}/5" if len(group)==5 else "missing")
        rows.append(["BE (D3)" if t=="BE_BASE_D3" else t,*values])
    return markdown_table(["Teacher","Passive E1 500 updates","Passive E1 1,500 updates","Small 500","No messages 500","Interactive 500 total"],rows)


def optimization_table():
    path=OUT/"optimization-confirmation/results.json"
    if not path.exists():return "Independent optimizer-control confirmation is not yet complete.\n"
    results=read(path)["arms"];ts=list(dict.fromkeys(r["teacher"] for r in results))
    return markdown_table(["Teacher","500 updates, B=3,000","1,500 updates, B=3,000"],
             [[t,*[f"{r['status']} ({r['seeds_pass']}/5)" for u in (500,1500) for r in results if r["teacher"]==t and r["updates"]==u]] for t in ts])


def structural_witness_text():
    path=OUT/"structural-failure-witnesses.json"
    if not path.exists():return ""
    witness=read(path);groups=witness["final_failure_summary"]
    failed_seeds={f["seed"] for g in groups for f in g["failures"]}
    if not groups or not all(g["failed_cases"] and g["all_pending_isolated"] for g in groups):return ""
    bad=[w for w in witness["singleton_witnesses"] if w["seed"] in failed_seeds and not w["singleton_success"]]
    if not bad:return ""
    acquired=sorted({w["acquired_isolated_assign_bundles"] for w in bad})
    presented=sorted({w["presented_isolated_assign_bundles"] for w in bad})
    missing=sorted({w["isolated_vertices_in_development_suite"] for w in bad})
    describe=lambda values:str(values[0]) if len(values)==1 else f"{values[0]}–{values[-1]}"
    decision=("an incorrect confirmation rejecting assignment despite positive participation" if all(
              w["trace"][0]["participation"][0] and w["trace"][0]["commit_logit"][0]<0 for w in bad) else "an erroneous WAIT in the assignment phase")
    counts=", ".join(f"{g['architecture']}: {g['failed_cases']}" for g in groups)
    return (f"The GM confirmation failures ({counts}) leave only isolated vertices uncolored and occur in training seed(s) {', '.join(map(str,sorted(failed_seeds)))}. "
            f"A post hoc singleton witness using the frozen candidate weights reproduces {decision}, while the exact teacher succeeds. "
            f"The cached prefix contains {describe(acquired)} isolated-assignment bundles (three component labels per bundle), "
            f"and the exact recorded minibatch order presents them {describe(presented)} times; "
            f"the affected seed's development suite contains {describe(missing)} isolated vertices. This identifies a rare-state fit defect missed by development coverage. "
            "It does not show an absence of ID information or prove a universal capacity limitation. The diagnostic changes no model, budget, primary classification or final tape. "
            "Full traces and presentation-count checks are in structural-failure-witnesses.json.\n\n")


def paper_section(rows,records,controls,config):
    passed=[r for r in rows if r["status"]=="PASS"]
    attempted=sum(r["final_attempted"] for r in rows)
    final_executions=sum(s["executions"] for r in rows for s in r["per_seed"].values())
    queries=sum(read(p)["queries"] for p in (OUT/"data").glob("*/*.json"))
    text="""# How Much Supervision Does It Take to Learn a Distributed Algorithm?

A teacher observation is one acquired decision label for a specified node,
state and action component. In ordinary color phases we count participation,
proposal and confirmation separately; Linial point selection and speculative
conflict marking each count as one. Replaying a stored label adds optimizer
work and no new supervision. Query counts, component types, trajectory clusters,
unique observations and data hashes are recorded independently of updates.
This distinction matters because sequential imitation changes the states a
policy visits [@ross2010efficient; @ross2011dagger]. Existing imitation
sample-complexity theory depends on its own state-space and horizon assumptions
[@rajaraman2020limits]; our finite curves imply no analogous theoretical bound.

Information compatibility is established before comparing budgets. Anonymous
regime A contains LUBY, JOHANSSON and repeated COLOR_TRIALS. Extended regime B
contains JP with explicit relative-ID comparisons, BE with auxiliary colors,
phase clock and supplied finite-field arithmetic, GM with priority comparisons
and its native block/repair schedule, and sequential GREEDY with writable-node
permissions. The three palette architectures have identical teacher-specific
observations and learned graph messages. Transformer attention acts within a
node's palette; it is not a replacement for graph communication. Interface
sufficiency arguments and an anonymous-JP indistinguishability example do not
prove universal representability by these fixed network widths.

For a policy to PASS, every frozen execution must end with a proper in-palette
coloring, all protocol invariants respected, termination by the teacher-specific
deadline, and no technical error, cutoff or incorrect terminal state. We use
no 95% functional threshold, repair filter or failed-tape replacement. A
teacher/architecture/budget combination passes only when all five independently
trained policies pass their respective 1,000-case final batteries. These are
five separate policies, not one policy tested 5,000 times.

All seven exact teachers passed every frozen final case before student
confirmation. Native deadlines distinguish n-step sequential greedy and JP's
at-most-n priority steps from BE's 15 reductions and the randomized teachers'
prospective tail deadlines. Native steps with different semantics are never
directly interpreted as identical distributed communication units.

"""
    text+=f"The balanced cohort comprises {len(records):,} fresh trainings, five seeds per cell, seven query budgets from 1,000 to 1,000,000, and 500 Adam updates per training. Its shared teacher libraries acquired {queries:,} component labels in total; smaller-budget policies only access their hashed prefixes. Training sizes are 8–24, development sizes 28/36, and confirmation sizes 32/48/64, in seven prescribed graph strata. The first budget where all five development policies are perfect nominates one frozen candidate; final results cannot choose another budget. Of 21 teacher/architecture packages, {attempted} nominated a candidate, {len(passed)} achieved five-seed PASS, and {final_executions:,} final student executions were retained.\n\n"
    for regime in ("A","B"):
        text+=f"**Regime {regime}. Smallest validated candidate budget, or explicit lack of certification.**\n\n"+primary_table(rows,regime)+"\n"
    text+="""A NOT CERTIFIED cell reports an unachieved protocol threshold B*>1,000,000,
not an impossibility theorem. Its earlier development budgets were evaluated;
policies failing development were not exhaustively tested in confirmation.
Per-seed queries, failures, compute and the distinction between an unattempted
candidate and a failed confirmation remain in the machine-readable tables.

![Anonymous-interface supervision curves. Each line is one training seed; these are development screens, not repeated final confirmations. The dashed line is the required 100% frontier.](ucgnn/figures/sample-success-regime-a.png)

![Extended-interface supervision curves with the same five-seed and 100% rules. Declines are retained without smoothing.](ucgnn/figures/sample-success-regime-b.png)

The grid does not assume monotonic improvement. Fixed optimization also means
larger libraries can leave more acquired observations unused; visit counts
are exported. Thus these are supervision curves conditional on the specified
optimizer budget, not pure information-theoretic sample requirements.

For zero failures in 1,000 independent runs of one frozen policy, the one-sided
95% binomial lower success bound is 0.9970087504549047. Under fixed independent
strata the same zero-failure bound conservatively controls their
allocation-weighted mean by the arithmetic–geometric mean inequality. It does
not cover every graph or simultaneously all trained policies. The five seeds
remain the units of training replication.

## Optimization, capacity and interactive controls

The following entries count seeds completing all development cases, at the
same B=3,000. They are exploratory development outcomes, never final PASS:

"""+control_table(controls,records,config)+"\n"
    text+="""The fixed-label control changes only E1's optimizer updates. The small and
no-message models share the same label prefix and update budget. The
one-aggregation interactive control allocates half the labels to passive
warmup and half to admissible states visited by the warm policy, with 250+250
updates and an Adam reset at aggregation. Failed learner trajectories remain
recorded. It excludes GM's unproved arbitrary-history interface. These limited
controls cannot identify every failure as a representation or data limitation.

Functional PASS does not logically imply action or transition fidelity,
teacher-like execution time, or equality of complete coloring laws. Separate
teacher-state fidelity and native-step curves retain all five seeds. Historical
law tests are not imported as evidence for the new checkpoints. The full
protocol, audit, negative results, dataset/checkpoint hashes and reproducible
figure sources accompany this finite empirical statement.

"""
    return text


def questions(rows,records,controls,config):
    def status(t,a):
        r=next(r for r in rows if r["teacher"]==t and r["architecture"]==a)
        if r["status"]=="PASS":
            q=list(r["actual_queries_by_budget"][str(r["validated_budget"])].values())
            actual=str(min(q)) if min(q)==max(q) else f"{min(q)}–{max(q)}"
            return f"{NAMES[a]}: B*={r['validated_budget']:,}, actual acquired labels {actual} per seed, 5/5 PASS"
        return f"{NAMES[a]}: NOT CERTIFIED, {r['seeds_pass']}/5 confirmed; protocol threshold unachieved through {r['max_budget_evaluated']:,}"
    regressions=sum(len(r["regressions"]) for r in rows)
    text="# Sample complexity results and required scientific questions\n\n"
    text+="**P1. How much Luby supervision?**\n\n"+"; ".join(status("LUBY",a) for a in NAMES)+". These are acquired component-label budgets under the frozen optimization/selection procedure, not trajectories or gradient updates.\n\n"
    text+="**P2. How does the budget change across teachers?**\n\n"+primary_table(rows)+"\nRegimes A/B must be interpreted separately; exact counts, per-seed component queries, failures and CPU costs are in `results-table.json`/`.csv`.\n\n"
    text+="**P3. Which information adaptations are required?**\n\nJP/GM receive edge order signs; BE receives auxiliary colors, phase/bound/target, local polynomial evaluation and raw equality relations; GM/greedy retain external scheduling. Luby/Johansson/ColorTrials use the anonymous regime. The full sufficiency/indistinguishability analysis is in `docs/TEACHER_MODEL_INFORMATION_COMPATIBILITY.md`.\n\n"
    text+="**P4. Do adaptations resolve earlier JP/BE failures?**\n\n"
    for t in ("JP","BE_BASE_D3"):text+=t+": "+"; ".join(status(t,a) for a in NAMES)+".\n\n"
    text+="These statements concern the new complete training/interface package. The previous integrated 70-model cohort already had numeric IDs and BE helpers. There is no matched raw-ID ablation at identical data/loss/updates, so a change cannot be attributed uniquely to the adapter. BE's arithmetic remains supplied. The additional fixed-label optimized BE point is reported separately in P8.\n\n"
    text+=structural_witness_text()
    if (OUT/"difficult-teacher-diagnostics.json").exists():
        diagnostics=read(OUT/"difficult-teacher-diagnostics.json");stages=Counter()
        for r in diagnostics["autonomous_failures"]:
            if r["teacher"]=="BE_BASE_D3":
                for f in r["first_failure_phase"]: stages[f["phase"]]+=f["count"]
        text+=f"Across all recorded BE development runs, first-failure counts by stage are {dict(stages)}. The separate common-state component audit at B=1,000,000 localizes agreement by phase and auxiliary-target sign; it does not identify a single causal failure mechanism.\n\n"
    text+=f"**P5. Does training seed matter?**\n\nAll five curves are retained. There are {regressions} observed adjacent-budget success declines across individual seed curves; no monotonicity assumption is imposed. Per-cell `first_individual_development_pass`, `development_perfect_seeds` and final summaries expose variability, including when only a subset of seeds passes. Five seeds cannot characterize all possible training runs.\n\n"
    winners=[]
    for t in config["teachers"]:
        good=[r for r in rows if r["teacher"]==t and r["status"]=="PASS"]
        if good:
            best=min(r["validated_budget"] for r in good)
            winners.append(f"{t}: {', '.join(NAMES[r['architecture']] for r in good if r['validated_budget']==best)} at {best:,}")
        else: winners.append(f"{t}: no certified ordering")
    text+="**P6. Which architecture needs least supervision?**\n\n"+"; ".join(winners)+". Ties and unachieved thresholds prevent a universal ranking. Earlier budgets that failed development were not all final-confirmed.\n\n"
    text+="**P7. Are smaller models more sample efficient?**\n\nAt the same B=3,000 and 500 updates, these development-only controls compare matched label prefixes:\n\n"+control_table(controls,records,config)+"\n"
    if (OUT/"capacity/certification.json").exists():
        small=read(OUT/"capacity/certification.json")
        text+="The separate small-model seven-budget grid has independently frozen confirmation batteries:\n\n"+markdown_table(["Teacher","Parameters","Small validated B*","Seeds confirmed","Final attempted"],
                     [[r["teacher"],r["parameters"],f"{r['validated_budget']:,}" if r["validated_budget"] is not None else "NOT CERTIFIED through 1,000,000",f"{r['seeds_pass']}/5",r["final_attempted"]] for r in small])+"\nA missing development candidate means no student final was attempted; 0/5 confirmed does not mean five observed final failures.\n\n"
    text+="Small-model conclusions apply to these matched training budgets and independent finite final batteries. Only one no-message budget is tested. Neither experiment establishes theoretical minimality.\n\n"
    text+="The unsmoothed per-seed capacity comparison is exported as `results/sample-complexity/figures/capacity-comparison.png` and `.pdf`.\n\n"
    text+="**P8. What changes with fixed labels and more optimization?**\n\nCompare the 500/1,500-update columns above, with identical labels and initialization per seed. The complete control records retain per-seed development success and time. Improvements identify a finite optimization-budget sensitivity, not a universal explanation of every failure. No extra teacher query is charged for replay.\n\n"
    text+="An additional exploratory protocol compares the same two frozen arms on independent, paired final batteries, after checking identical label-prefix and initialization hashes:\n\n"+optimization_table()+"\nThese are fixed-budget validations, not new B* minima; they do not replace primary statuses.\n\n"
    if (OUT/"optimization-confirmation/results.json").exists():
        arms=read(OUT/"optimization-confirmation/results.json")["arms"]
        be=[r for r in arms if r["teacher"]=="BE_BASE_D3"]
        text+="For BE, the paired fixed-label control changes the number of fully successful seeds from "+str(next(r["seeds_pass"] for r in be if r["updates"]==500))+"/5 to "+str(next(r["seeds_pass"] for r in be if r["updates"]==1500))+"/5 with more updates. This demonstrates optimization-budget sensitivity at those fixed labels, without estimating a minimum label budget for 1,500 updates.\n\n"
    text+="**P9. Does interactive supervision help?**\n\nThe last column uses the same maximum B=3,000 and 500 total updates, with one 1,500-label passive / 1,500-label admissible on-policy aggregation. All seeds and failed acquisition trajectories are retained. Adam resets at aggregation; this control does not isolate distribution choice from every optimizer-history effect. It is development-only and supports no final-certification or general DAgger efficiency claim.\n\n"
    comparisons=[]
    for t in config["optimization_control_teachers"]:
        passive=sum(r["development"]["pass_all"] for r in records if (r["identity"]["teacher"],r["identity"]["architecture"],r["identity"]["budget"])==(t,"normalized",3000))
        interactive=sum(r["record"]["development"]["pass_all"] for r in controls if r["kind"]=="interactive" and r["record"]["identity"]["teacher"]==t)
        comparisons.append(f"{t}: interactive {interactive}/5 versus passive {passive}/5")
    text+="Observed development-perfect seed counts are "+"; ".join(comparisons)+". These fixed-point observations do not establish a universal modality ranking.\n\n"
    law=read(OUT/"law/results.json") if (OUT/"law/results.json").exists() else None
    text+="**P10. Does functional 100% imply action, time or global-law fidelity?**\n\nNo. The functional gate constrains validity, invariants and completion; it does not require matching teacher choices, native step counts or probabilities of each complete coloring. These are separate measured outcomes. "
    if law:
        passed={(r["teacher"],r["architecture"],r["validated_budget"]) for r in rows if r["status"]=="PASS"}
        conditional=sum(r["reject"] and (r["teacher"],r["architecture"],r["budget"]) in passed for r in law["rows"])
        text+=f"The independent exploratory law supplement has {law['testable']}/210 testable hypotheses and {law['rejections']} Holm rejections; {conditional} rejections concern policies in functionally certified packages. Comparisons condition on successful rows; nonrejection never establishes equality."
    else: text+="The independent law supplement is not complete; no new-cohort law conclusion is asserted."
    text+="\n\n**P11. What is robust across all five seeds?**\n\n"+", ".join(f"{r['teacher']}/{r['architecture']} at {r['validated_budget']:,}" for r in rows if r["status"]=="PASS")+". This is robustness across these five trainings and batteries, not a guarantee for unseen initializations or all graphs. Every policy retains its own 1,000-run uncertainty statement.\n\n"
    if (OUT/"full-fidelity.json").exists():
        text+="The separate full-fidelity audit covers every root and every transition of all 28 development teacher trajectories per seed, including late KW stages. Its actual-state transition equality allows redundant actions that leave state unchanged. Original 256-decision prefix diagnostics remain in training logs and are not treated as a complete phase/family audit.\n\n"
    text+="**P12. What belongs in the main paper and appendix?**\n\nMain text: information regimes, query unit, strict five-seed gate, teacher-specific horizons, primary B* table, unsmoothed success curves, and concise fixed-label/interactive controls. Appendix and repository: all per-seed/per-budget counts, phase fidelity, native-step curves, checkpoint/data manifests, negative trajectories, source/RNG hashes, law-test details and resource/replay audits. No censored or missing cell is filled by extrapolation.\n"
    return text


def audit_section(rows,records,controls,config):
    certified=sum(r["status"]=="PASS" for r in rows)
    attempts=sum(r["final_attempted"] for r in rows)
    text=f"""## Sample complexity and learnability: independent audit extension

The new study contains {len(records)} controlled-budget fresh main trainings,
21 teacher/architecture packages and five training replicas per package.
{attempts} packages nominated one development-selected frozen candidate;
{certified} achieved the strict five-seed PASS. This section supersedes no
historical result and makes no previous checkpoint a query-controlled training.

### Observation, information, data and optimization audit

An observation is an acquired node/state/action-component label. Normal phases
count participation, proposal and teacher confirmation separately; BE Linial
point and GM conflict mark count separately in their phases. Root identities,
trajectory clusters, native step, global state hashes, component counts,
cache/prefix hashes, distinct observations, trajectories, graphs, bytes,
presentations and unused labels are exported. Simulator computations are a
separate cost and are not silently called supervised labels. A max-budget
library is cached once per teacher/seed; policies access only their prefix.
Optimization replays stored labels and never acquires counterfactual targets.

The information audit is `docs/TEACHER_MODEL_INFORMATION_COMPATIBILITY.md`.
Regime A is anonymous; regime B retains each native program's IDs/auxiliary
colors/phases/scheduler. All three architectures receive the same observations.
JP/GM use explicit relative order; BE's finite-field arithmetic is supplied,
with learned point/recolor decisions. No exact availability, eligibility,
winning color, student conflict mark or safety shield enters inference.
The anonymous two-node JP counterexample proves an insufficiency of a particular
interface, not a failure of neural learning generally. Interface sufficiency
does not establish fixed-width universal representability.

All seven query budgets through 1,000,000 are evaluated from fresh seed-matched
initialization at 500 Adam updates, batch 64 and lr .001. Phase-stratified
sampling is fixed before results; sampling with replacement has effective
passes and no literal epoch count. Five seeds control initialization, data
generation and batch order; saved Python/NumPy/Torch states permit audit.
Fixed-label 1,500-update, small/no-message and one-aggregation interactive
controls are exploratory and use development only. Interactive queries require
proper native states and retain failed learner acquisition trajectories.

### Evaluation, results, failures and statistical evidence

Training n=8–24, development n=28/36 and final n=32/48/64 have disjoint graph
sizes and independent namespaces/tapes. Seven native-valid families include
symmetry and four-color-necessary K4 unions. Every graph/ID/tape and deadline
was fixed and exact-teacher-validated before student confirmation. Sequential,
priority, randomized and algebraic native steps have distinct semantics;
within-teacher time ratios are meaningful, cross-teacher raw-round rankings
are not. The previous common 128-step cutoff is not reused as a universal rule.

Each final policy must complete all 1,000 preregistered executions with proper
palette-valid terminal coloring, all native invariants, no technical error,
no cutoff and no incorrect terminal state. Missing/duplicate case IDs fail.
GM's temporary conflicts are legal; BE auxiliary improperness is a failure.
Unsafe predictions are applied and recorded, without repair or tape replacement.
PASS requires conjunction across all five independent trainings. The consumed
attempt marker and frozen five checkpoint hashes prevent final-guided retries.
No candidate and failed confirmation remain distinct NOT CERTIFIED states.

Primary tables and full per-seed query/failure/compute fields are
`results/sample-complexity/results-table.json` and `.csv`. Raw final/exact
executions, batteries, training/control records, regressions and figure hashes
are versioned; full teacher caches/checkpoints/development rows are in the
separate verified runtime archive. `docs/SAMPLE_COMPLEXITY_RESULTS.md` answers
P1–P12 using those actual observations. The study's NOT CERTIFIED thresholds
are censored under its finite optimizer/selection procedure, not proof of an
intrinsic monotone sample requirement or mathematical impossibility.
The separate `results/sample-complexity/evidence-ledger.json` maps the new claim
groups to exact source/result hashes; the original integration ledger retains
its historical twelve claim groups.

With zero failures in one policy's 1,000 independent executions the one-sided
95% lower bound is 0.9970087504549047. Under fixed heterogeneous strata the
zero-failure arithmetic–geometric-mean argument bounds their allocation-weighted
mean. No policy receives an n=5,000 interval by pooling training replicas.
These are not simultaneous confidence guarantees over all tested cells.
Five seeds limit inference about the distribution of possible trainings.
Correlated trajectory roots are never counted as independent training replicas.

### Does 100% Empirical Success Mean the Algorithm Has Been Learned?

It means that these five frozen policies completed these finite independent
benchmark batteries without a recorded functional failure. Conditional on the
declared distribution/independence assumptions it also supports the separate
per-policy statistical statements. It does not prove identity to the teacher,
all-input correctness, termination on every random tape, matching complexity,
exact solution-law equality or architectural minimality. A different proper
policy can choose different actions and a different output law while passing.
Nonrejection of a robust distribution test is not evidence of exact equality.

Mathematical guarantees would require an explicit domain and computation model,
formal equivalence or verified safety/progress predicates on every reachable
local state, sound composition across graph nodes and rounds, treatment of
real/finite precision boundaries and supplied arithmetic/schedulers, and a
termination argument valid under the specified randomness. A finite discrete
domain could support exhaustive verified state enumeration; unbounded domains
need symbolic invariants and a transfer theorem. Learning-theoretic guarantees
also need justified coverage/complexity/stability assumptions and a proof for
the actual algorithm. Convergence or the finite curves supply none of these
automatically.
For randomized anonymous teachers, a valid finite tape can keep symmetric
vertices waiting or colliding in proposals for the entire deadline; universal
finite-deadline termination on every tape is generally false even for the exact
teacher. A mathematical theorem must specify almost-sure termination, an
explicit tail probability, or additional deterministic progress assumptions.

### Threats, causal limits and interpretation

Teacher-specific extra information and coded schedulers mean regimes are
different learning problems. Changes from historical JP/BE also change the
loss and controlled data/optimizer budget; an improved package does not isolate
the causal effect of relational IDs. BE arithmetic and greedy/GM scheduling
remain supplied. The no-message ablation changes available information and
must not be labeled a pure parameter-count intervention. Greater B at fixed
updates can increase unused data and cause regressions. The control changes
only updates at one B and cannot exclude every representational/generalization
failure. One interactive aggregation has limited coverage and an Adam reset.
Development contains 28 cases per seed, so candidate screening can be weak;
the independent final gate explicitly tests that uncertainty. Graph sizes,
families, finite ID space, finite tapes and only five training replicas bound
all conclusions. Native K4/degree-three results do not certify arbitrary local
palettes, degree 200 or high-degree BE recursion. CPU/GPU time and parameter
counts remain separate. Full FLOPs are unmeasured; the post hoc profiler
estimate in `flops-estimates.json` counts supported dense forward/backward
arithmetic by phase and excludes encoder, optimizer and uncounted
elementwise/reduction operations. It is not total hardware FLOPs.

The robust coloring-law supplement has its own fixed plan, independent banks,
teacher-specific reference and native K4 allocator. It conditions on successful
rows, retains functional failure counts, fixes the selection/evaluation split
before permutations and adjusts over all 210 hypotheses, with untestable cells
set to p=1. Conservative futility cannot create a rejection. It is exploratory
and never enters the functional gate or checkpoint selection. Historical law
results are not substituted for new policy evidence.

The fixed-label optimization confirmation is an additional exploratory protocol
documented after development controls. It freezes both 500/1,500-update arms
for all three preselected teachers, checks equal label-prefix/initialization
hashes and uses a new independent final namespace with matched cases across
arms. Its one-attempt markers and every individual outcome are retained.
It validates fixed optimizer/label points and never changes the primary
certification, nominates a new primary budget or retrofits an earlier final
battery. Its six arm-level results are reported separately.

A separate complete common-state fidelity audit reuses all 28 frozen
development teacher trajectories per seed for every primary checkpoint. It
uses no final data or training updates. The original 256-decision prefix
metrics remain in logs; their action-identical transition proxy is not an
if-and-only-if state equality measure. The full audit compares the actual
updated state channel, accounting for field-specific Linial decoding and
redundant KW/GM repair writes; three interpreter-based regression tests check
these cases. Reported fidelity curves use the full audit when available.

"""
    return text


def run():
    records=read(OUT/"training.json");config=read(OUT/"design.json")["config"]
    certs=read(OUT/"certification.json")
    controls=read(OUT/"controls.json") if (OUT/"controls.json").exists() else []
    rows=analyze(records,certs,config)
    generalization(records,certs)
    efficiency(certs)
    atomic_json(OUT/"results-table.json",rows)
    with (OUT/"results-table.csv").open("w",newline="") as f:
        keys=["teacher","architecture","regime","status","validated_budget","candidate_budget","seeds_pass","final_attempted",
              "max_budget_evaluated","native_parameters","total_training_seconds","total_cpu_seconds","gpu_seconds","dominant_profiled_flops_estimate","final_failures",
              "development_perfect_seeds","actual_queries_by_budget"]
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");writer.writeheader()
        for r in rows: writer.writerow({k:json.dumps(r[k],sort_keys=True) if isinstance(r[k],dict) else r[k] for k in keys})
    figures(records,config)
    capacity_figure(records,config)
    model_inventory(records,controls)
    section=paper_section(rows,records,controls,config)
    section+=structural_witness_text()
    law=read(OUT/"law/results.json") if (OUT/"law/results.json").exists() else None
    if law:
        section+=f"The independent exploratory coloring-law supplement tests 210 fixed teacher/checkpoint/graph hypotheses using the audited native K4 robust instrument, split selection/evaluation rows, 9,999 permutations and Holm over the complete family. {law['testable']} comparisons are testable and {law['rejections']} reject. Untestable comparisons retain p=1; conservative futility also cannot create rejection. These are conditional-on-success laws on two fixed small graphs, not equality guarantees or an additional functional PASS condition.\n\n"
    if (OUT/"capacity/certification.json").exists():
        small=read(OUT/"capacity/certification.json")
        section+="The small-model supplement repeats all seven budgets for three representative teachers and freezes independent final tapes under the same five-seed gate:\n\n"+markdown_table(["Teacher","Small parameters","Small validated B*","Seeds confirmed","Final attempted"],
                   [[r["teacher"],r["parameters"],f"{r['validated_budget']:,}" if r["validated_budget"] else "NOT CERTIFIED",f"{r['seeds_pass']}/5",r["final_attempted"]] for r in small])+"\nWithout a common development candidate no student final is attempted. These outcomes establish neither universal minimality nor the behavior of all possible small networks.\n\n"
    if (OUT/"optimization-confirmation/results.json").exists():
        section+="The additional fixed-label optimization protocol freezes both existing optimizer arms for all three control teachers and evaluates them on new paired final batteries. It checks identical label-prefix and initialization hashes and changes no primary outcome:\n\n"+optimization_table()+"\nA PASS here validates the fixed B=3,000/optimizer point only; it is not a new grid minimum or a retry of the primary confirmation.\n\n"
    (OUT/"paper-section.md").write_text(section.rstrip()+"\n")
    Path("docs/SAMPLE_COMPLEXITY_RESULTS.md").write_text(questions(rows,records,controls,config))
    audit=Path("docs/PAPER_SCIENTIFIC_AUDIT.md")
    original=audit.read_text().split("<!-- BEGIN_SAMPLE_COMPLEXITY_AUDIT -->")[0].rstrip()
    audit.write_text(original+"\n\n<!-- BEGIN_SAMPLE_COMPLEXITY_AUDIT -->\n"+audit_section(rows,records,controls,config)+"<!-- END_SAMPLE_COMPLEXITY_AUDIT -->\n")
    status_counts=Counter(r["status"] for r in rows)
    atomic_json(OUT/"summary.json",dict(trainings=len(records),status_counts=dict(status_counts),
                final_executions=sum(s["executions"] for r in rows for s in r["per_seed"].values()),
                confirmed_packages=[dict(teacher=r["teacher"],architecture=r["architecture"],budget=r["validated_budget"]) for r in rows if r["status"]=="PASS"],
                regression_count=sum(len(r["regressions"]) for r in rows),controls=len(controls),
                full_fidelity_checkpoints=len(read(OUT/"full-fidelity.json")["rows"]) if (OUT/"full-fidelity.json").exists() else None,
                scope="Strict finite native benchmark; no universal correctness, minimality, monotonicity, or distributional equivalence claim."))
    print(json.dumps(read(OUT/"summary.json")),flush=True)
    claims=[
      ("SC01","Acquisition and replay are distinct costs","Stored root-component labels, not epochs or independent root replicas",["training.json","design.json"]),
      ("SC02","Teacher-specific information is explicit","Interface sufficiency is not fixed-width representability",["preregistration.json","model-inventory.json"]),
      ("SC03","Strict five-seed functional classification","Finite frozen native benchmark, no universal guarantee",["certification.json","frozen-candidates.json"]),
      ("SC04","Fixed-label optimization confirmation","Additional independent fixed point, not a new primary B*",["optimization-confirmation/results.json","optimization-confirmation/frozen-plan.json"]),
      ("SC05","GM rare-state witness","Post hoc frozen-model diagnosis, no repair or budget retry",["structural-failure-witnesses.json"]),
      ("SC06","Individual supervision curves can regress","No monotonic threshold or optimizer-independent sample bound",["results-table.json","figure-manifest.json"]),
      ("SC07","Small-model grid remains separately scoped","No candidate is not five observed final failures; no minimality theorem",["capacity/training.json","capacity/certification.json"]),
      ("SC08","New coloring-law results are separate","Conditional on success; nonrejection is not equality",["law/results.json","law/frozen-plan.json"]),
      ("SC09","Complete action and actual-transition audit","Every development teacher trajectory, no final data or parameter updates",["full-fidelity.json"])]
    ledger=[dict(id=k,claim=claim,boundary=boundary,sources=[dict(path=str(OUT/p),sha256=file_hash(OUT/p)) for p in paths if (OUT/p).exists()])
            for k,claim,boundary,paths in claims]
    atomic_json(OUT/"evidence-ledger.json",ledger)


if __name__=="__main__": run()
