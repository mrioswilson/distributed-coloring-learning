"""Rebuild unsmoothed per-seed curves and tables from frozen observations."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    for regime in ("A","B"):
        teachers=[t for t in config["teachers"] if REGIME[t]==regime]
        for metric in ("success","fidelity","rounds"):
            fig,axes=plt.subplots(len(teachers),3,figsize=(13,2.6*len(teachers)),squeeze=False,sharex=True)
            for i,t in enumerate(teachers):
                for j,a in enumerate(config["architectures"]):
                    ax=axes[i,j]
                    for si,s in enumerate(config["seeds"]):
                        rs=sorted([r for r in records if (r["identity"]["teacher"],r["identity"]["architecture"],r["identity"]["seed"])==(t,a,s)],key=lambda r:r["identity"]["budget"])
                        x=[r["identity"]["budget"] for r in rs]
                        if metric=="success": y=[r["development"]["success_rate"] for r in rs]
                        elif metric=="fidelity": y=[r["fidelity"]["action_fidelity"] for r in rs]
                        else: y=[r["development"]["mean_native_steps"] for r in rs]
                        ax.plot(x,y,"o-",lw=1,ms=3,color=colors[si],label=str(s))
                    ax.set_xscale("log");ax.grid(alpha=.2)
                    if metric=="success":
                        ax.axhline(1,color="black",ls="--",lw=.8);ax.set_ylim(-.025,1.045)
                    elif metric=="fidelity": ax.set_ylim(-.025,1.045)
                    if i==0: ax.set_title(NAMES[a],fontsize=10)
                    if j==0: ax.set_ylabel(t+"\n"+{"success":"Development success","fidelity":"Teacher-state action fidelity","rounds":"Native steps attempted"}[metric],fontsize=8)
                    if i==len(teachers)-1: ax.set_xlabel("Maximum acquired component labels B (log)",fontsize=8)
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc="lower center",ncol=5,title="Independent training seed",fontsize=8)
            fig.suptitle(f"Regime {regime}: {metric}; five separate unsmoothed development curves",fontsize=11)
            fig.tight_layout(rect=(0,.085,1,.97))
            for ext in ("png","pdf"):
                path=folder/f"{metric}-regime-{regime.lower()}.{ext}"
                fig.savefig(path,dpi=160,bbox_inches="tight");outputs[str(path)]=file_hash(path)
                if metric=="success":
                    shutil.copyfile(path,Path("paper/ucgnn/figures")/f"sample-{path.name}")
            plt.close(fig)
    atomic_json(OUT/"figure-manifest.json",dict(files=outputs,training_sha256=file_hash(OUT/"training.json")))
    return outputs


def analyze(records,certs,config):
    result=[]
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
        result.append(row)
    return result


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
        rows.append([t,*values])
    return markdown_table(["Teacher","Passive E1 500 updates","Passive E1 1,500 updates","Small 500","No messages 500","Interactive 500 total"],rows)


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


def run():
    records=read(OUT/"training.json");config=read(OUT/"design.json")["config"]
    certs=read(OUT/"certification.json")
    controls=read(OUT/"controls.json") if (OUT/"controls.json").exists() else []
    rows=analyze(records,certs,config)
    atomic_json(OUT/"results-table.json",rows)
    with (OUT/"results-table.csv").open("w",newline="") as f:
        keys=["teacher","architecture","regime","status","validated_budget","candidate_budget","seeds_pass","final_attempted",
              "max_budget_evaluated","native_parameters","total_training_seconds","total_cpu_seconds","gpu_seconds","final_failures",
              "development_perfect_seeds","actual_queries_by_budget"]
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in rows: writer.writerow({k:json.dumps(r[k],sort_keys=True) if isinstance(r[k],dict) else r[k] for k in keys})
    figures(records,config)
    section=paper_section(rows,records,controls,config)
    (OUT/"paper-section.md").write_text(section)
    status_counts=Counter(r["status"] for r in rows)
    atomic_json(OUT/"summary.json",dict(trainings=len(records),status_counts=dict(status_counts),
                final_executions=sum(s["executions"] for r in rows for s in r["per_seed"].values()),
                confirmed_packages=[dict(teacher=r["teacher"],architecture=r["architecture"],budget=r["validated_budget"]) for r in rows if r["status"]=="PASS"],
                regression_count=sum(len(r["regressions"]) for r in rows),controls=len(controls),
                scope="Strict finite native benchmark; no universal correctness, minimality, monotonicity, or distributional equivalence claim."))
    print(json.dumps(read(OUT/"summary.json")),flush=True)


if __name__=="__main__": run()
