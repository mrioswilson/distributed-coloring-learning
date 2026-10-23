"""Rebuild integrated figures, complete Markdown draft and evidence ledger."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pypandoc
from dcl.behavior_statistics import write_csv

from dcl.integration.models import ARCHITECTURES
from dcl.multiteacher.programs import TEACHERS
from dcl.study.storage import atomic_json, file_hash

OUT = Path("results/integration")
FIG = Path("paper/ucgnn/figures")
NAMES = dict(normalized="E1", deepsets="DeepSets", transformer="Transformer", small="Small", no_communication="No messages")
TEACHER_NAMES = dict(zip(TEACHERS, ("Luby", "Johansson", "JP", "ColorTrials", "BE (base d3)", "Speculative GM", "Greedy")))
COLORS = dict(zip(ARCHITECTURES, ("#225ea8", "#238b45", "#88419d", "#dd8d29", "#737373")))


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def save_figure(fig, name, manifests):
    for extension in ("pdf", "png"):
        p = FIG / f"integration-{name}.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else {"Software": "AlgoLearning integration report"}
        fig.savefig(p, dpi=220, bbox_inches="tight", metadata=metadata)
        manifests[str(p)] = file_hash(p)
    plt.close(fig)


def figures(functional, training, manifests):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    grouped = defaultdict(lambda: [0, 0])
    for r in functional:
        if r["architecture"] == "exact": continue
        a = grouped[r["teacher"], r["architecture"]]
        a[0] += r["successes"]; a[1] += r["samples"]
    matrix = np.array([[100*grouped[t, a][0]/grouped[t, a][1] for a in ARCHITECTURES] for t in TEACHERS])
    fig, ax = plt.subplots(figsize=(7.7, 4.8))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(5), [NAMES[a] for a in ARCHITECTURES], rotation=16, ha="right")
    ax.set_yticks(range(7), TEACHERS)
    for i in range(7):
        for j in range(5): ax.text(j, i, f"{matrix[i,j]:.1f}%", ha="center", va="center", color="white" if matrix[i,j]>60 else "black")
    ax.set_title("Proper completion: 500 updates, two seeds, five fixed graphs")
    fig.colorbar(im, ax=ax, label="Valid complete runs (%)", fraction=.045, pad=.03)
    fig.tight_layout(); save_figure(fig, "functional", manifests)

    points = defaultdict(lambda: [0, 0])
    for r in functional:
        a = points[r["teacher"], r["architecture"], r["seed"]]
        a[0] += r["successes"]; a[1] += r["samples"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    markers = dict(zip(TEACHERS, ("o", "s", "^", "D", "P", "X", "v")))
    for r in training:
        k = r["teacher"], r["architecture"], r["seed"]
        successes, n = points[k]
        ax.scatter(r["local"]["accuracy"], successes/n, color=COLORS[r["architecture"]],
                   marker=markers[r["teacher"]], s=48, alpha=.85, edgecolor="white", linewidth=.4)
    from matplotlib.lines import Line2D
    arch_legend = [Line2D([], [], color=COLORS[a], marker="o", linestyle="", label=NAMES[a]) for a in ARCHITECTURES]
    teacher_legend = [Line2D([], [], color="#333333", marker=markers[t], linestyle="", label=t) for t in TEACHERS]
    fig.legend(handles=arch_legend, loc="lower center", bbox_to_anchor=(.43,1.0), ncol=5, frameon=False, fontsize=8)
    ax.legend(handles=teacher_legend, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    ax.set(xlabel="Native action agreement on common validation states", ylabel="Proper completion on fresh fixed-cohort runs",
           xlim=(0, 1.025), ylim=(-.025, 1.025))
    ax.grid(alpha=.18); fig.tight_layout(); save_figure(fig, "fidelity", manifests)

    power = read(OUT / "statistics/independent-power.json")["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, case, title in zip(axes, ("star_null", "localized"), ("Same-law null", "Known alternative: TV = 99/199")):
        for offset, method, color, label in [(-.12, "mixture", "#737373", "Mixture MMD"), (.12, "v2", "#225ea8", "V2")]:
            rows = [next(r for r in power if r["case"]==case and r["m"]==m and r["method"]==method) for m in (256,500)]
            values = np.array([r["rate"] for r in rows]); ci = np.array([r["pointwise_cp95"] for r in rows])
            ax.errorbar(np.arange(2)+offset, values, yerr=np.array([values-ci[:,0],ci[:,1]-values]),
                        fmt="o", capsize=4, color=color, label=label)
        ax.axhline(.05, color="black", lw=.8, linestyle=":")
        ax.set_xticks(range(2), ["256", "500"]); ax.set(xlabel="Independent runs per generator", title=title, ylim=(-.04,1.06))
        ax.grid(alpha=.15); ax.legend(fontsize=9)
    axes[0].set_ylabel("Rejection rate (95% CI)")
    fig.tight_layout(); save_figure(fig, "power", manifests)

    contrasts = read("results/ucgnn-size-degree/factorial-contrasts.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    for ax, event, title in zip(axes, ("used_le_delta", "uses_top_color"), ("A: at least one color absent", "T: top color appears")):
        for a in ARCHITECTURES[:3]:
            rows = [next(r for r in contrasts if r["event"]==event and r["group"]==a and r["kind"]=="cell" and r["contrast"]==f"n{n}-d100")
                    for n in (128,256,512,1024)]
            y=np.array([r["estimate"] for r in rows]); ci=np.array([r["ci"] for r in rows])
            ax.errorbar(np.arange(4),y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt="o-",capsize=3,color=COLORS[a],label=NAMES[a])
        ax.axhline(0,color="black",lw=.8); ax.set_xticks(range(4),["128","256","512","1024"])
        ax.set(xlabel="Graph size, fixed degree 100",title=title); ax.grid(alpha=.15)
    axes[0].set_ylabel("Model minus teacher event probability"); axes[1].legend(fontsize=8)
    fig.tight_layout(); save_figure(fig,"factorial",manifests)


def evidence_ledger():
    rows = [
      ("C01","Full-budget functional extrapolation","paper-study", "results/ucgnn-paper-study/FINAL_REPORT.md", "docs/UCGNN_PAPER_PROTOCOL.md and src/dcl/paperstudy/config.py", "Five seeds per primary architecture; final model specifications and 85-checkpoint archive", "Reported archive plus immutable artifact manifests; finite cohort only"),
      ("C02","Native teacher comparison","Pedro native", "results/multiteacher-e1/2026-10-22/architecture-comparison.csv", "configs/multiteacher_e1/comparison.json", "35 unavailable original weight digests in PEDRO_SOURCE_MANIFEST.json", "Published aggregate evidence; original raw/weight reproduction incomplete"),
      ("C03","Pedro E1 output-law findings","Pedro behavior", "results/e1-retrained/2026-10-21/distribution-tests.csv", "configs/e1_behavior/retrained.json", "Five missing retrained E1 digests; manifest.json", "0/100 reported rejections; coupling certificates reported, raw verification pending"),
      ("C04","Teacher correctness","Exact native rules", "docs/integration/TEACHER_CORRECTNESS_AUDIT.md", "src/dcl/multiteacher/programs.py", "No trained checkpoint", "Conditional proofs plus 1,442 finite-domain runs; no universal learned-policy inference"),
      ("C05","Matched full-law comparison","Six archived banks", "results/integration/statistics/matched-archives.json", "docs/integration/STATISTICAL_CROSSCHECK_PROTOCOL.md", "Our s41000 frozen models; every raw bank digest recorded", "Original V2 statistics replayed; exploratory historical reanalysis"),
      ("C06","Known-law sensitivity gap","Fresh independent control", "results/integration/statistics/independent-power.json", "docs/integration/STATISTICAL_CROSSCHECK_PROTOCOL.md", "No model; independent synthetic whole-vector laws", "20 trials per law/size; pointwise CP; no universal power conclusion"),
      ("C07","Balanced multiple-teacher learning","New 70-model cohort", "results/integration/multiteacher/functional.json", "configs/integration-balanced.json", "results/integration/multiteacher/frozen-checkpoints.json", "Fresh banks, two initializations, short budget; fixed graphs, not population rankings"),
      ("C08","New conditional global laws","New native K4 cohort", "results/integration/statistics/new-study-global.json", "docs/integration/DECLARED_ALPHABET_ADDENDUM.md", "Each checkpoint uses its own teacher reference", "350 planned Holm comparisons; untestable cells distinct from equality"),
      ("C09","Size/degree observable effects","Historical factorial", "results/ucgnn-size-degree/factorial-contrasts.json", "docs/FACTORIAL_SIZE_DEGREE_PROTOCOL.md", "15 epoch120 models, seeds41000–41004", "1,800 simultaneous fixed-cohort contrasts; degree and palette confounded"),
      ("C10","Input guard repair","New regression", "results/integration/teacher-guard-before.xml", "tests/test_integration_teachers.py", "No model", "21 failures before, pass after; no supported historical result invalidated"),
      ("C11","Declared alphabet repair","New integration compatibility", "results/integration/statistics/global-alphabet-before.txt", "src/dcl/integration/native_statistics.py", "All new weights/raw banks unchanged", "Failed before path p-value; exact matching-alphabet replay and null regression"),
      ("C12","Architecture necessity limits","Specified observations", "paper/PAPER_DRAFT.md", "docs/integration/TEACHER_CORRECTNESS_AUDIT.md", "No checkpoint for mathematical examples; small/no-message empirical controls separately", "Scoped information arguments and finite sufficiency, no global minimality theorem")]
    result=[]
    for key,claim,experiment,path,configuration,checkpoint,scope in rows:
        result.append(dict(id=key,claim=claim,experiment=experiment,source_file=path,configuration=configuration,
                           checkpoint=checkpoint,evidence_and_confidence=scope,limitations=scope,
                           source_sha256=file_hash(path) if Path(path).exists() and path!="paper/PAPER_DRAFT.md" else None))
    return result


def readable_prose(text):
    # Preserve code, citation keys, math and URLs while spacing prose numbers.
    protected = re.compile(r"(`[^`]*`|\[[^\]]*@[^\]]*\]|\$\$[\s\S]*?\$\$|\$[^$\n]*\$|https?://[^\s)]+)")
    parts=protected.split(text)
    for i in range(0,len(parts),2):
        parts[i]=re.sub(r"(?<=[a-z])(?=\d)"," ",parts[i])
        parts[i]=parts[i].replace("nominalalpha.", "nominal alpha = .")
        parts[i]=re.sub(r"(?<=[a-z])(?=\.\d)"," ",parts[i])
    for i in range(1,len(parts),2):
        if parts[i].startswith("$"):
            if parts[i-1] and parts[i-1][-1].isalnum(): parts[i-1]+=" "
            if i+1<len(parts) and parts[i+1] and parts[i+1][0].isalnum(): parts[i+1]=" "+parts[i+1]
    return "".join(parts).replace("traces,\\+pre/post", "traces, pre/post")


def references(template):
    keys=sorted(set(re.findall(r"@([a-z][a-z0-9]+)",template)))
    refs=[str(Path("paper/ucgnn/references.bib").resolve()),str(Path("paper/ucgnn/integration-references.bib").resolve())]
    source="---\nbibliography:\n"+"".join(" - "+p+"\n" for p in refs)+"nocite: |\n "+", ".join("@"+k for k in keys)+"\n---\n\n"
    return pypandoc.convert_text(source,"gfm",format="markdown",extra_args=["--citeproc","--wrap=none"]).strip()


def run(figures_only=False):
    FIG.mkdir(parents=True,exist_ok=True)
    functional=read(OUT/"multiteacher/functional.json"); training=read(OUT/"multiteacher/training.json")
    assert len(training)==70 and len(functional)==385
    grouped=defaultdict(lambda:[0,0])
    for r in functional:
        a=grouped[r["teacher"],r["architecture"]];a[0]+=r["successes"];a[1]+=r["samples"]
    manifests={};figures(functional,training,manifests)
    atomic_json(OUT/"figure-manifest.json",dict(files=manifests,inputs={str(p):file_hash(p) for p in [OUT/"multiteacher/functional.json",OUT/"multiteacher/training.json",OUT/"statistics/independent-power.json",Path("results/ucgnn-size-degree/factorial-contrasts.json")]}))
    if figures_only:return
    global_result=read(OUT/"statistics/new-study-global.json"); law=global_result["comparisons"]
    assert len(law)==350
    law_group=defaultdict(lambda:[0,0,0])
    for r in law:
        a=law_group[r["teacher"],r["architecture"]];a[0]+=r["reject"];a[1]+=r["testable"];a[2]+=1
    counts=table(["Teacher",*[NAMES[a] for a in ARCHITECTURES]],[[TEACHER_NAMES[t],*[f"{grouped[t,a][0]:,}" for a in ARCHITECTURES]] for t in TEACHERS])
    full=[t for t in TEACHERS if all(grouped[t,a][0]==1280 for a in ARCHITECTURES[:3])]
    result_text="All three large palette architectures completed 1,280/1,280 attempts for each of "+", ".join(full)+". At the same short budget, JP and BE retained substantial failures, while speculative GM was intermediate. These differences compare specified training/representation packages, not intrinsic teacher complexity. Per-teacher architecture differences and both replicas remain in the table; there is no universal winner."
    small=next(r for r in training if r["teacher"]=="LUBY" and r["architecture"]=="small")
    nomsg=sum(grouped[t,"no_communication"][0] for t in TEACHERS)
    baseline=f"The small LUBY model has {small['parameters']:,} trainable parameters and completes all 1,280 LUBY evaluations; the small greedy control also completes all 1,280 under its supplied schedule. Thus the larger core is not empirically necessary for those finite cells. In contrast, the no-communication control completes {nomsg:,}/8,960 total attempts across the seven teachers. This supports the value of received information under this protocol, without proving that the particular large GNN or a learned message MLP is necessary. Small-model failures for other teachers are retained and do not establish incapacity beyond this budget."
    tested=sum(r["testable"] for r in law);rejected=sum(r["reject"] for r in law)
    global_text=f"Of 350 planned teacher/checkpoint/graph hypotheses, {tested} are testable and {350-tested} have fewer than four successful model/reference rows. After Holm over the complete planned family, {rejected} reject. Each reference bank belongs to the corresponding teacher. These counts concern conditional output laws on a small fixed cohort; functional failures and limited power remain separate."
    global_table="**Table7. New Holm rejections / testable comparisons (ten planned per teacher and architecture).**\n\n"+table(["Teacher",*[NAMES[a] for a in ARCHITECTURES]],[[TEACHER_NAMES[t],*[f"{law_group[t,a][0]}/{law_group[t,a][1]}" for a in ARCHITECTURES]] for t in TEACHERS])
    ledger=evidence_ledger();atomic_json(OUT/"evidence-ledger.json",ledger)
    evidence=table(["ID","Claim","Experiment","Evidence boundary"],[[r["id"],r["claim"],r["experiment"],r["evidence_and_confidence"]] for r in ledger])
    template=Path("paper/PAPER_DRAFT.md.in").read_text()
    values=dict(NEW_FUNCTIONAL_TABLE=counts,NEW_RESULTS_PARAGRAPH=result_text,BASELINE_INTERPRETATION=baseline,
                NEW_GLOBAL_PARAGRAPH=global_text,NEW_GLOBAL_TABLE=global_table,EVIDENCE_TABLE=evidence,REFERENCES=references(template))
    supplement=Path("results/sample-complexity/paper-section.md")
    values["SAMPLE_COMPLEXITY_SECTION"]=(supplement.read_text().rstrip()+"\n\n") if supplement.exists() else "The prospective supervision study is specified in docs/SAMPLE_COMPLEXITY_PROTOCOL.md; confirmed results are not yet available."
    if supplement.exists(): values["REFERENCES"]=references(template+supplement.read_text())
    sc_summary=Path("results/sample-complexity/summary.json")
    if sc_summary.exists():
        sc=read(sc_summary)
        values["SAMPLE_COMPLEXITY_ABSTRACT"]=(f"A separate controlled-supervision study trains {sc['trainings']} fresh policies across seven teachers, three architectures, seven label budgets and five training seeds. Its strict benchmark certifies {sc['status_counts'].get('PASS',0)} of 21 teacher/architecture packages only when all five policies complete 1,000 final executions each without failure. Explicit information regimes, nonmonotone per-seed curves and fixed-label optimization controls separate this finite empirical certification from universal correctness and theoretical sample-complexity claims.")
    else: values["SAMPLE_COMPLEXITY_ABSTRACT"]="The prospective query-budgeted study is specified separately; no empirical certification is asserted before confirmation."
    text=template
    for key,value in values.items():text=text.replace("{{"+key+"}}",value)
    assert not re.search(r"\{\{[A-Z_]+\}\}",text)
    Path("paper/PAPER_DRAFT.md").write_text(readable_prose(text))
    write_csv(OUT/"functional-summary.csv",[dict(teacher=t,architecture=a,successes=grouped[t,a][0],attempts=grouped[t,a][1]) for t in TEACHERS for a in ARCHITECTURES])
    atomic_json(OUT/"summary.json",dict(models=70,banks=385,attempts=sum(r["samples"] for r in functional),
        complete_teacher_architectures=full,small_luby_parameters=small["parameters"],no_communication_successes=nomsg,
        new_law_planned=350,new_law_testable=tested,new_law_rejections=rejected,figures=manifests,
        draft_sha256=file_hash("paper/PAPER_DRAFT.md")))
    print(json.dumps(dict(models=70,attempts=sum(r["samples"] for r in functional),law_testable=tested,law_rejections=rejected,draft="paper/PAPER_DRAFT.md")),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--figures-only",action="store_true")
    run(p.parse_args().figures_only)
