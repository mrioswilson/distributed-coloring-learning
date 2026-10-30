#!/usr/bin/env python3
"""Vector figures and manuscript text derived from compact revision evidence."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=Path("results/paper-revision-20261030")
PAPER=Path("paper/ucgnn")


def write(name,text):
    for env in ("table", "figure", "longtable"):
        text=text.replace("\n\\begin{"+env+"}","\n\n\\begin{"+env+"}")
        text=text.replace("\\end{"+env+"}\n","\\end{"+env+"}\n\n")
    (PAPER/name).write_text(text.strip()+"\n")


def architecture():
    fig,ax=plt.subplots(figsize=(11.5,3.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.01",facecolor=color,edgecolor="#344054",linewidth=1))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=11)
    def arrow(a,b,**kw):ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",color="#344054",lw=1.4,**kw))
    box(.015,.61,.20,.29,"Raw local inputs\nphase, private draws,\npermissions, predicates","#eef0f3")
    box(.29,.61,.25,.29,"Exchange 1\nlearned messages\n+ MAX aggregation","#dce9ff")
    box(.64,.61,.33,.29,"Learned participation\nand palette proposal\nP / C messages","#dff3eb")
    box(.71,.16,.26,.28,"Exchange 2\nlearned messages\n+ MAX aggregation","#dce9ff")
    box(.36,.16,.25,.28,"Learned COMMIT\nqueried-slot decision","#dff3eb")
    box(.015,.16,.25,.28,"Supplied interpreter\nschedule, decoding,\nstate update","#eef0f3")
    arrow((.22,.75),(.28,.75));arrow((.55,.75),(.63,.75));arrow((.83,.60),(.83,.45))
    arrow((.70,.30),(.62,.30));arrow((.35,.30),(.275,.30))
    ax.text(.49,.495,"Typed continuation query enters here",ha="center",va="center",fontsize=10,color="#a34e06")
    arrow((.64,.49),(.72,.45),linestyle="dashed")
    ax.text(.50,.035,"Training only: native P/C targets and continuation COMMIT labels",ha="center",fontsize=10,color="#a34e06")
    arrow((.57,.065),(.53,.15),linestyle="dashed")
    fig.tight_layout(pad=.2);fig.savefig(PAPER/"figures/revision-architecture.pdf");plt.close(fig)


def historical():
    data=json.loads((OUT/"be-reanalysis.json").read_text())["rows"]
    methods=("passive","matched","counterfactual","always_reject","eligibility_only")
    lines=[]
    for method in methods:
        rows=[r for r in data if r["method"]==method and r["group"]=="all"]
        values=[100*np.mean([r[k] for r in rows]) for k in ("accuracy","balanced_accuracy","equal_type_accuracy")]
        lines.append(method.replace("_"," ")+" & "+" & ".join(f"{v:.3f}" for v in values)+r" \\")
    write("revision-type-analysis.tex",r'''
\paragraph{Equal weight per sufficient local type.}
Occurrence-weighted accuracy gives frequent local configurations more weight.
We additionally average accuracy within each sufficient COMMIT input type and
then over distinct types, separately within each training seed. This measures
the tested type set, not a natural input distribution or a new training replica.
Reconstruction with the original RNG and 256-root chunks reproduces every
historical numerator before this reweighting.
\begin{table}[htbp]\centering\small
\begin{tabular}{lrrr}
\toprule Procedure & Occurrence acc. (\%) & BA (\%) & Equal-type acc. (\%) \\
\midrule
'''+"\n".join(lines)+r'''
\bottomrule
\end{tabular}
\caption{Same historical BE queries, three different estimands. Entries
average five paired seeds. Equal-type accuracy weights the 115--132 distinct
tested types equally within seed, rather than their 99,840 occurrences.}
\label{tab:type-weighting}
\end{table}
These estimates retain both the strong rejection difference and the limits
of the familiar-type diagnostic. Per-seed seen/unseen, eligibility, silence
and occupied/free strata, including all confusion counts, accompany the
type-level occurrence counts in the revision evidence. The split uses the
union of native and stored CF training types in all historical arms, making
its novelty criterion common rather than treatment-dependent.
''')


def be_step_accounting(summary):
    """Keep run-length denominators explicit without changing primary outputs."""
    result = {"scope": "Descriptive accounting of original BE final rows; no new execution", "inputs": {}, "arms": {}}
    required = ("proper", "palette", "invariants", "terminated", "no_technical_error", "no_cutoff", "terminal_correct")
    for arm in ("native", "cf"):
        rows = []
        for seed in range(105001, 105011):
            path = OUT / "prospective/be_base_d3" / f"s{seed}" / (arm + "-test.json")
            result["inputs"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            stored = json.loads(path.read_text())
            rows.extend(stored["rows"])
            assert len(stored["rows"]) == 1000
        good = [r for r in rows if all(r.get(k) is True for k in required)]
        bad = [r for r in rows if not all(r.get(k) is True for k in required)]
        assert all(r["native_steps"] == 15 for r in good)
        original = next(r for r in summary["summary"] if r["teacher"] == "BE_BASE_D3")
        assert len(good) == original[arm + "_successes"]
        mean_all = float(np.mean([r["native_steps"] for r in rows]))
        assert np.isclose(mean_all, original[arm + "_mean_steps"])
        result["arms"][arm] = dict(runs=len(rows), successes=len(good), failures=len(bad),
                                  mean_all=mean_all, mean_success=15.,
                                  mean_failure=float(np.mean([r["native_steps"] for r in bad])))
    (OUT / "be-step-accounting.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def prospective():
    data=json.loads((OUT/"prospective-summary.json").read_text());table=[];detail=[]
    be_step_accounting(data)
    for r in data["summary"]:
        name="BE" if r["teacher"]=="BE_BASE_D3" else "GM"
        lo,hi=[100*v for v in r["paired_seed_bootstrap_95"]]
        table.append(f"{name} & {100*r['native_mean_ba']:.2f} & {100*r['cf_mean_ba']:.2f} & {100*r['mean_paired_ba_difference']:+.2f} [{lo:+.2f},{hi:+.2f}] & {r['native_successes']:,} / {r['cf_successes']:,}"+r" \\")
        if name=="GM":
            detail.append(f"{name} equal-type accuracy is {100*r['native_mean_equal_type_accuracy']:.2f}\\% native and {100*r['cf_mean_equal_type_accuracy']:.2f}\\% CF; both arms average {r['native_mean_steps']:.2f} native steps.")
    write("revision-prospective.tex",r'''
\section{Prospective Equal-Budget Continuation Experiment}\label{sec:prospective}
We test a separate acquisition-strategy contrast with BE/E1 and GM/E1
(19,396 parameters). Ten fresh paired seeds per teacher give forty models.
Before training, a fixed SHA256 partition holds out 20\% of sufficient
queried-slot COMMIT type buckets, common across seeds. Phase, root/context
features and distinct neighbor message types determine this key; MAX removes
multiplicity. Neither training arm supervises a held-out continuation type.
LINIAL and DETECT remain separate native targets. This is a split of input
contexts within the same intervention templates. It does not withhold all
coarser oracle-relevant configurations (quantified below).

Each pair shares 3,000 native component targets and 3,600 fresh auxiliary
rooted states. On each auxiliary state the native arm acquires one native
COMMIT target and CF one prespecified intervened COMMIT target. Both queries
must pass the training-type filter to retain that state. GM auxiliary quotas
are 1,800 ASSIGN and 1,800 REPAIR; BE uses KW only. Each model therefore
accesses 6,600 optimization targets. Native P/C conditioning is supplied to
both arms equally and charged separately: 7,200 root values plus retained
neighbor values. Simulator computation, rejected states and duplicate queries
are also recorded; this does not equate all teacher-computation costs.
The root option cycles through native-option offsets 0,1,2,3 modulo four.
Participation is the root permission, except that offset three independently
silences it with probability 1/4; neighbor messages remain native. The cursor
advances before filtering, including discarded roots. Test queries use the
same intervention and retain held-out CF types; their phase quotas are
4,000 KW or 2,000 each ASSIGN/REPAIR. This single-query design differs from
the historical four-query diagnostic. Appendix~\ref{app:prospective-method}
specifies the key, serialization, hash and acquisition order.

Adam, batch 64 and 1,500 updates are fixed before test, with 100 matched warmup
updates and the same subsequent native/auxiliary loss weighting. Paired data,
initialization, index sequences and first-stage final tensors agree. All forty
weights are frozen before generating 4,000 held-out queries per seed/teacher
on sizes 28/36 and evaluating 1,000 autonomous runs per model on sizes 32/48/64.
All exact references pass. Every final continuation has zero intersection
with its training sufficient-type set; both truth classes occur in every seed.

\begin{table}[htbp]\centering\small
\begin{tabular}{lrrlc}
\toprule Teacher & Native BA & CF BA & Paired difference [95\% interval] & Successes N / CF \\
\midrule
'''+"\n".join(table)+r'''
\bottomrule
\end{tabular}
\caption{New final experiment. BA is a percentage; differences and intervals are percentage points,
averaged over ten paired training seeds. Intervals bootstrap paired seeds
(10,000 resamples), conditional on this fixed split/protocol; individual
queries are not independent replications. Success totals are out of 10,000
autonomous runs per teacher/arm. No checkpoint selection uses these tests.}
\label{tab:prospective}
\end{table}
'''+"\n".join(detail)+r'''

GM gains 50 percentage points in every paired seed without changing any
autonomous outcome or mean step count. Its degenerate bootstrap interval
reflects those identical observed seed differences, not certainty over all
future types or training replicas. BE's mean difference is +5.77 points, but
the interval spans zero (\(-13.73,+25.26\)). BE equal-type accuracy increases
from 51.70\% to 86.94\%, while the two endpoints weight errors differently.
CF adds false negatives in two BE seeds and reduces autonomous successes from
9,204 to 7,405; one further seed has identical failures in both arms.
Mean step counts over \emph{all} final runs (14.53 native versus 13.43 CF)
include early exits on failure. Every successful BE run takes 15 native
steps in both arms; the lower overall mean is not a successful-run speedup.
These are substantive negative results for a uniform-benefit claim.

\input{revision-prospective-audit.tex}
\input{revision-prospective-mechanisms.tex}

The primary endpoint concerns held-out continuation types; autonomous success
is secondary. This finite comparison concerns the specified acquisitions at
equal target/context access. It supports neither extrapolation to new logical
conditions nor an advantage over alternative informative-negative acquisition
strategies. BE's occurrence-level endpoint must be read with its diagnostic
baseline and type weighting; GM's control lacks negative color confirmations.
No universal CF superiority, minimum sample complexity or all-source teacher
efficiency follows.
The protocol, forty checkpoints, all seed outcomes, per-component costs and
freeze receipts accompany the new results (Appendix~\ref{app:prospective}).
''')
    seeds=[]
    for teacher in ("BE_BASE_D3","SPECULATIVE_GM"):
        for seed in range(105001,105011):
            pair={r["arm"]:r for r in data["rows"] if r["teacher"]==teacher and r["seed"]==seed}
            n,c=pair["native"],pair["cf"]
            name="BE" if teacher=="BE_BASE_D3" else "GM"
            seeds.append(f"{name} & {seed} & {100*n['balanced_accuracy']:.3f} & {100*c['balanced_accuracy']:.3f} & {100*(c['balanced_accuracy']-n['balanced_accuracy']):+.3f} & {n['successes']} & {c['successes']}"+r" \\")
    costlines=[]
    for teacher in ("BE_BASE_D3","SPECULATIVE_GM"):
        subset=[r for r in data["costs"] if r["teacher"]==teacher]
        for key in sorted({k for r in subset for k in r["accounting"] if k.startswith("base_target_")}):
            values=[r["accounting"].get(key,0) for r in subset]
            name=("BE" if teacher=="BE_BASE_D3" else "GM")
            costlines.append(f"{name} & {key.replace('base_target_','').replace('_',' ')} & {min(values)}--{max(values)}"+r" \\")
    write("revision-prospective-details.tex",r'''
\section{New Prospective Replicas and Costs}\label{app:prospective}
\begin{longtable}{llrrrrr}
\caption{All new final seed outcomes. BA is a percentage; differences are
percentage points. Successes are out of 1,000 per model. N expands native confirmations.}\\
\toprule Teacher & Seed & N BA & CF BA & Difference & N success & CF success \\
\midrule\endfirsthead
\toprule Teacher & Seed & N BA & CF BA & Difference & N success & CF success \\
\midrule\endhead
'''+"\n".join(seeds)+r'''
\bottomrule
\end{longtable}
\begin{table}[htbp]\centering\small
\begin{tabular}{llr}
\toprule Teacher & Base target component & Count range across ten banks \\
\midrule
'''+"\n".join(costlines)+r'''
\bottomrule
\end{tabular}
\caption{Base supervision sums to exactly 3,000 per seed. Both arms add
3,600 auxiliary COMMIT targets. Context values and simulator costs are
reported separately, not absorbed into a free-label assumption.}
\end{table}
Each pair physically acquires 10,200 optimization targets: one shared
3,000-target base and two 3,600-target auxiliary banks. Each model accesses
6,600 targets, so shared physical acquisition and summed model access differ.
The shared auxiliary P/C context contains 7,200 root scalar occurrences and
additional real-neighbor occurrences. Base root P/C context overlaps base
target labels; its neighbor conditioning values are also recorded separately.
These are scalar occurrences, not claims about distinct simulator calls.

\code{results/paper-revision-20261030/prospective-summary.json} records the
per-seed component counts, complete-query duplicates, phase quotas, contexts
considered/rejected, observed simulator states, acquisition CPU/wall time and
paired bootstrap outputs. Training records separately retain presentations,
visited targets, source/data hashes and optimizer CPU/wall time. One run's
wall time includes a shared-host memory stall; wall time is not a FLOP or
teacher-sample-efficiency estimate. All training remains at update 1,500.
The original source/config freeze predates scientific training; the complete
forty-weight freeze is dated 30 October 2026, 10:11:58 UTC, before any new
final diagnostic generation. No final outcome selects a milestone, protocol
change, seed exclusion or replacement model.

The recorded source freeze covers the enumerated files, not a complete
transitive dependency or environment lock. In particular, graph generation,
storage and sample-complexity data helpers outside that list are not sealed
by this receipt. Their stability is not independently timestamped here.
The original freeze is preserved; this limitation is not repaired by
retrospectively adding hashes to it.

\input{revision-prospective-methods.tex}
\input{revision-prospective-audit-details.tex}
\input{revision-prospective-mechanism-details.tex}
''')


def continuation_audit():
    audit = json.loads((OUT / "continuation-audit.json").read_text())
    summary, rows = audit["summary"], audit["rows"]
    original = json.loads((OUT / "prospective-summary.json").read_text())
    controls, coverage, confusion = [], [], []

    def interval(values, decimals=0, scale=1):
        text = [f"{scale*v:,.{decimals}f}" for v in values]
        return text[0] if text[0] == text[1] else "--".join(text)

    for teacher in ("BE_BASE_D3", "SPECULATIVE_GM"):
        s = summary[teacher]
        name = "BE" if teacher == "BE_BASE_D3" else "GM"
        old = next(r for r in original["summary"] if r["teacher"] == teacher)
        for arm, label in (("native", "Native"), ("cf", "CF"), ("gate", "Eligibility/participation")):
            for metric in ("mean_ba", "mean_equal_type_accuracy"):
                if arm != "gate":
                    assert np.isclose(s[arm + "_" + metric], old[arm + "_" + metric])
            controls.append(f"{name} & {label} & {100*s[arm+'_mean_ba']:.2f} & "
                            f"{100*s[arm+'_mean_equal_type_accuracy']:.2f}" + r" \\")
        for phase, arms in s["phase_label_ranges"].items():
            for arm, counts in arms.items():
                coverage.append(f"{name} & {phase} & {arm} & {interval(counts['positive'])} & "
                                f"{interval(counts['negative'])}" + r" \\")
        for stratum, arms in s["strata_confusion_sums"].items():
            for arm, counts in arms.items():
                confusion.append(f"{name} & {stratum.replace('_', ' ')} & {arm} & " +
                                 " & ".join(f"{counts[k]:,}" for k in ("tp", "tn", "fp", "fn")) + r" \\")
    b, g = summary["BE_BASE_D3"], summary["SPECULATIVE_GM"]
    active = b["strata_confusion_sums"]["eligible_active"]
    values = {
        "BE_GATE_BA": f"{100*b['gate_mean_ba']:.2f}",
        "BE_CF_BA": f"{100*b['cf_mean_ba']:.2f}",
        "BE_GATE_TYPE": f"{100*b['gate_mean_equal_type_accuracy']:.2f}",
        "BE_CF_TYPE": f"{100*b['cf_mean_equal_type_accuracy']:.2f}",
        "BE_TYPES": interval(b["test_type_range"]),
        "GM_TYPES": interval(g["test_type_range"]),
        "BE_MASS": interval(b["dominant_type_mass_range"], 2, 100),
        "BE_LOGICAL": interval(b["cf_logical_occurrence_overlap_range"], 3, 100),
        "GM_TRAIN_POS": interval(g["native_training_labels"]["positive"]),
        "GM_CF_NEG": interval(g["cf_training_labels"]["negative"]),
        "GM_TEST_NEG": interval(g["test_negative_range"]),
        "BE_FREE": f"{active['native']['tp'] + active['native']['fn']:,}",
        "BE_CF_TP": f"{active['cf']['tp']:,}",
        "BE_OCCUPIED": f"{active['cf']['tn'] + active['cf']['fp']:,}",
    }

    def substitute(text):
        for key, value in values.items():
            text = text.replace("@" + key + "@", value)
        return text

    write("revision-prospective-audit.tex", substitute(r'''
\paragraph{Post-test diagnostic controls.}
The primary endpoint and all original outcomes above are retained.
The following controls and support audit were added after inspecting the test;
they are descriptive checks, not a revised primary analysis. The programmed
rule $G_v=E_vp_v$ uses supplied eligibility and root participation and ignores
query occupancy. It requires no training and is a diagnostic, not a learned
label-efficiency competitor.
\begin{table}[htbp]\centering\small
\begin{tabular}{llrr}
\toprule Teacher & Procedure & BA (\%) & Equal-type acc. (\%) \\
\midrule
''') + "\n".join(controls) + substitute(r'''
\bottomrule
\end{tabular}
\caption{Same final queries and sufficient-type key; means over ten seeds.
The programmed diagnostic ignores occupancy. BA averages sensitivity and
specificity; equal-type accuracy first averages within each distinct
type, then across types within seed. The diagnostic is post-test.}
\label{tab:prospective-controls}
\end{table}
For BE, the programmed rule scores @BE_GATE_BA@\% BA, above CF's @BE_CF_BA@\%,
whereas CF scores @BE_CF_TYPE@\% equal-type accuracy versus @BE_GATE_TYPE@\%.
A single negative type accounts for @BE_MASS@\% of each 4,000-query bank.
Class balancing therefore leaves the negative subtypes highly imbalanced;
high occurrence-level BA alone does not identify the occupancy condition.
Among eligible active BE queries, native and the programmed rule accept
@BE_FREE@/@BE_FREE@ free options but reject 0/@BE_OCCUPIED@ occupied options.
CF rejects @BE_OCCUPIED@/@BE_OCCUPIED@ occupied options but accepts only
@BE_CF_TP@/@BE_FREE@ free options. These are pooled descriptive counts over
ten banks, not independent training replications. Appendix~\ref{app:prospective-audit}
reports the full confusion strata.

GM's native control receives @GM_TRAIN_POS@ positive COMMIT labels and zero
negative COMMIT labels in ASSIGN/REPAIR per seed, while CF receives
@GM_CF_NEG@ negatives and the test has @GM_TEST_NEG@ negatives per seed.
The native model accepts all test queries, giving BA 50\%; the contrast
therefore demonstrates learning rejections absent from native color supervision.
DETECT has separate positive/negative conflict supervision and is excluded
from these color-label counts. This result does not compare CF with another
acquisition strategy supplying informative negative confirmations.

\paragraph{Support of the held-out queries.}
There are @BE_TYPES@ distinct sufficient test types per BE seed and @GM_TYPES@
per GM seed, despite 4,000 occurrences. For these color oracles
$O_T=E_vp_v(1-o_v)$, with $o_v$ the queried-option occupancy. A coarser
configuration $(\text{phase},E_v,p_v,o_v)$ is therefore oracle-sufficient.
CF training has already seen configurations covering @BE_LOGICAL@\% of BE
test occurrences and 100\% of GM test occurrences. The original fine-key
holdout remains disjoint: novelty is defined by contextual combinations
rather than by this coarser configuration. There is no separate evaluation
restricted to novel logical conditions. These tests do not establish extrapolation to new logical
conditions. Per-seed support counts and logical overlap are in
Appendix~\ref{app:prospective-audit}; no split, label or model was changed.
'''))
    seed_rows = []
    for r in rows:
        name = "BE" if r["teacher"] == "BE_BASE_D3" else "GM"
        seed_rows.append(f"{name} & {r['seed']} & {r['sufficient_test_types']} & "
                         f"{100*r['max_fine_type_fraction']:.2f} & "
                         f"{100*r['logical_test_occurrence_overlap']['cf']:.3f} & "
                         f"{100*r['gate']['balanced_accuracy']:.3f} & "
                         f"{100*r['gate']['equal_type_accuracy']:.3f}" + r" \\")
    write("revision-prospective-audit-details.tex", r'''
\subsection{Post-test coverage and confusion audit}\label{app:prospective-audit}
All tables here are descriptive additions after the final test. Labels are
recomputed from raw encoded state arrays and agree with stored base, auxiliary
and final color confirmations. Fine keys are reconstructed from those arrays;
their training/test unions equal the delivered key files and obey the original
hash filter. Model predictions are summarized from frozen result records;
this audit does not rerun neural inference or train another model.
{\small
\begin{longtable}{lllrr}
\caption{Ranges over ten seeds of base plus auxiliary color-COMMIT targets.
LINIAL and DETECT have different targets and are excluded. Minima and maxima
in separate columns need not occur at the same seed.}\\
\toprule Teacher & Phase & Arm & Positive labels & Negative labels \\
\midrule\endfirsthead
\toprule Teacher & Phase & Arm & Positive labels & Negative labels \\
\midrule\endhead
''' + "\n".join(coverage) + r'''
\bottomrule
\end{longtable}
}

\begin{longtable}{llrrrrr}
\caption{Final support and programmed diagnostic by seed. Types are distinct
fine sufficient keys; mass is the largest key's fraction of 4,000 queries.
Overlap is the fraction of occurrences whose coarser oracle configuration was
seen by CF training. Mass, overlap, BA and type accuracy are percentages.}\\
\toprule Rule & Seed & Types & Max mass & Overlap & Gate BA & Gate type acc. \\
\midrule\endfirsthead
\toprule Rule & Seed & Types & Max mass & Overlap & Gate BA & Gate type acc. \\
\midrule\endhead
''' + "\n".join(seed_rows) + r'''
\bottomrule
\end{longtable}

{\small
\begin{longtable}{lllrrrr}
\caption{Confusion counts summed over ten final banks, 40,000 queries per
teacher/procedure. Within eligible active queries, positives are free options
and negatives occupied options. Silent and ineligible queries are negative.
No GM query is ineligible. These are accounting totals, not IID replications.}\\
\toprule Rule & Stratum & Procedure & TP & TN & FP & FN \\
\midrule\endfirsthead
\toprule Rule & Stratum & Procedure & TP & TN & FP & FN \\
\midrule\endhead
''' + "\n".join(confusion) + r'''
\bottomrule
\end{longtable}
}
The reproducible analysis and input hashes are
\code{scripts/audit-revision-continuations.py} and
\code{results/paper-revision-20261030/continuation-audit.json}.
''')


def be_failure_diagnostics():
    data = json.loads((OUT / "be-failure-diagnostics.json").read_text())
    rows = {(r["seed"], r["arm"]): r for r in data["rows"] if r["teacher"] == "BE_BASE_D3"}
    original = json.loads((OUT / "prospective-summary.json").read_text())
    assert data["inputs_unchanged_after_probe"]
    for r in data["rows"]:
        assert r["stored_final_confusion_reproduced"]
        old = next(o for o in original["rows"]
                   if (o["teacher"], o["seed"], o["arm"]) == (r["teacher"], r["seed"], r["arm"]))
        assert np.isclose(r["final"]["ba"], old["balanced_accuracy"])
    selected = [rows[seed, "cf"] for seed in (105001, 105008)]
    table = []
    for r in selected:
        a, f = r["auxiliary_train"], r["final"]
        assert a["n"] == r["auxiliary_visited"] == 3600 and f["n"] == 4000
        table.append(f"{r['seed']} & {100*a['ba']:.3f} & {100*f['ba']:.3f} & "
                     f"{a['fn']} & {f['fn']} & {f['fp']}" + r" \\")
    native, cf = rows[105010, "native"], rows[105010, "cf"]
    assert native["final"]["first_stage"] == cf["final"]["first_stage"]
    assert native["auxiliary_train"]["first_stage"] == cf["auxiliary_train"]["first_stage"]
    ptest = native["final"]["first_stage"]["participation"]
    paux = native["auxiliary_train"]["first_stage"]["participation"]
    success = [o["successes"] for o in original["rows"]
               if o["teacher"] == "BE_BASE_D3" and o["seed"] == 105010]
    assert success == [204, 204]
    replay = data["first_failure"]
    assert replay["original_terminal_rows_reproduced"] and replay["replayed_rows"] == 32
    frame, = replay["witness"]
    v, = frame["offending_vertices"]
    assert v["only_one_participation_bit_changed"] and v["first_stage_and_candidates_unchanged"]
    values = {
        "TRAIN1": f"{100*selected[0]['auxiliary_train']['ba']:.3f}",
        "TRAIN8": f"{100*selected[1]['auxiliary_train']['ba']:.3f}",
        "FINAL1": f"{100*selected[0]['final']['ba']:.3f}",
        "FINAL8": f"{100*selected[1]['final']['ba']:.3f}",
        "N_BA": f"{100*native['final']['ba']:.3f}", "CF_BA": f"{100*cf['final']['ba']:.3f}",
        "PTEST_FN": str(ptest["fn"]), "PTEST_POS": str(ptest["tp"] + ptest["fn"]),
        "PAUX_FN": str(paux["fn"]), "PAUX_POS": str(paux["tp"] + paux["fn"]),
        "PAUX_FP": str(paux["fp"]), "STEP": str(frame["step"]), "NODE": str(v["node"]),
        "BOUND": str(frame["bound_before"]), "TARGET": str(frame["bound_after"]),
        "OLD": str(v["aux_before"]), "OPTION": str(v["candidate"]),
        "DECODED": str(v["own_proposal_decodes_to"]),
        "P_LOGIT": f"{v['participation_logit']:.6f}",
        "CM_LOGIT": f"{v['actual_commit_logit']:.6f}",
        "FORCED_LOGIT": f"{v['forced_active_commit_logit']:+.6f}",
        "PREDICTIONS": f"{data['commit_predictions']:,}",
        "FINAL_PREDICTIONS": f"{data['final_commit_predictions']:,}",
    }

    def fill(text):
        for key, value in values.items():
            text = text.replace("@" + key + "@", value)
        return text

    write("revision-prospective-mechanisms.tex", fill(r'''
\paragraph{Two distinct BE failure diagnoses (post-test).}
In the two deteriorated CF seeds, 105001 and 105008, COMMIT BA on the 3,600
acquired auxiliary queries is @TRAIN1@\% and @TRAIN8@\%, versus @FINAL1@\%
and @FINAL8@\% on the held-out contexts. Both models visited all auxiliary
queries. Failure to fit acquired labels alone does not explain these cases:
the observed COMMIT gap concerns generalization across contexts, without
identifying its optimization cause or an architectural impossibility.

Seed 105010 has a different defect: both arms complete only 204/1,000 runs,
despite final COMMIT BA @N_BA@\% native and @CF_BA@\% CF. Their shared
first stage omits participation on @PTEST_FN@/@PTEST_POS@ eligible test roots.
A replay of the original 32-case cohort localizes the first invariant
violation in one selected run: at KW step @STEP@, vertex @NODE@ must
recolor as the bound drops from @BOUND@ to @TARGET@, but remains at @OLD@
because participation is false. Its proposed value @DECODED@ is free;
changing only root participation makes COMMIT accept it. Thus this state
exposes an upstream omission, rather than an incorrect veto of the actual
silent message. These are selected post-test diagnostics, without retuning,
a continued rescue rollout or an estimate of causes across all failed runs
(Appendix~\ref{app:be-failure-diagnostics}).
'''))
    write("revision-prospective-mechanism-details.tex", fill(r'''
\subsection{Selected BE failure diagnoses after the test}\label{app:be-failure-diagnostics}
Frozen-model inference covers both arms of BE seeds 105001, 105002, 105004,
105008 and 105010, plus both GM arms in 105001. This deliberately selected
set includes good and bad outcomes; it does not estimate their prevalence.
There are @PREDICTIONS@ COMMIT predictions on existing native-color,
auxiliary-training and final banks. All @FINAL_PREDICTIONS@ final predictions
reproduce the original confusion counts; no weights, split or primary
results are changed. This diagnosis uses neural inference, separately from
the preceding array-only coverage audit.

\begin{table}[htbp]\centering\small
\begin{tabular}{lrrrrr}
\toprule CF seed & Aux. BA & Final BA & Aux. FN & Final FN & Final FP \\
\midrule
''') + "\n".join(table) + fill(r'''
\bottomrule
\end{tabular}
\caption{Selected BE COMMIT generalization failures. BA is a percentage.
Each row evaluates the arm's 3,600 acquired auxiliary queries and the 4,000
held-out final queries. All auxiliary targets were visited during training;
neither selected model has an auxiliary false positive. The contexts differ,
so this is a descriptive fit/test contrast, not a new randomized comparison.}
\label{tab:be-fit-test}
\end{table}
The two CF models also classify all 944 native-color base queries correctly.
Their held-out-context errors coexist with near-perfect acquired-label fit;
this does not identify a unique shortcut, optimization cause or limitation
of representational capacity.

For seed 105010, native and CF first-stage tensors are identical. On 4,000
teacher-prefix test states their participation head omits @PTEST_FN@ of
@PTEST_POS@ eligible roots, with no false activation. On 3,600 auxiliary
states it omits @PAUX_FN@ of @PAUX_POS@ eligible roots and falsely activates
@PAUX_FP@ ineligible root. Auxiliary P/C values condition the COMMIT loss;
they are not direct participation-loss targets. These first-stage scores on
auxiliary states are evaluations, not evidence of failure to fit a
participation training objective there. Candidate options equal the native
ones throughout both banks, with no occupied proposals at eligible roots.

The witness is \code{final-s105010-0000}, a 32-node path in the native arm.
Replay preserves the first 32 original cases in their original order,
cohort size 32 and root batch size 1,024. All 32 terminal color vectors,
step counts, failures, invariant flags and correctness flags match the
original records. In the selected case the state before step @STEP@ is a
proper auxiliary coloring bounded by @BOUND@. Vertex @NODE@ has value @OLD@
and must recolor when the interpreter lowers the bound to @TARGET@.
Its participation logit is @P_LOGIT@, so it waits, leaving @OLD@ outside
$[0,@TARGET@)$. Its option @OPTION@ decodes to the free value @DECODED@,
also chosen by the exact native rule. The actual COMMIT logit @CM_LOGIT@
correctly rejects silence. Setting only this root's participation bit to
true, with identical first-stage tensors, candidates and all other messages,
gives logit @FORCED_LOGIT@ and the correct positive oracle label. The
observer's re-evaluated actions match the original actions at this frame.
The intervention is only a local query; unchanged actions drive the replay.
It locates the first violation in this case, without showing that all 796
failed runs share this mechanism or that the intervened run would complete.

\path{scripts/probe-revision-be-failures.py} reproduces the diagnosis;
\path{results/paper-revision-20261030/be-failure-diagnostics.json} records
input hashes, counts, graph/tape seeds and the local intervention. This is
post-test explanatory evidence, not independent scientific review.
'''))


def rescues():
    rows=json.loads((OUT/"component-rescues.json").read_text())["rows"]
    lines=[]
    for r in rows:
        name="Johansson" if r["checkpoint"].startswith("JOHANSSON") else "GM"
        lines.append(f"{name} & {r['mode'].replace('_',' ')} & {r['steps']} & "+("complete/proper" if r["success"] else r["failure"])+r" \\")
    write("revision-rescues.tex",r'''
\paragraph{One-component interventions on copied witnesses.}
We reconstruct each original full inference cohort from its case/tape seeds
and recovered checkpoint SHA. The unchanged execution matches its stored
terminal colors, steps and status, and all actions match through the selected
intervention frame. At that frame only one candidate is replaced by a free
alternative; all other messages and first-stage tensors stay fixed and the
learned second exchange/COMMIT is recomputed. A separate control substitutes
exact confirmation on the unchanged proposals once. Neither intervention
changes a training run or any reported autonomous test.
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/revision-failure-mechanisms.pdf}
\caption{Two selected copied witness executions, showing local fragments
after one queried application. A: Johansson steps 10 to 11, with free colors
$\{0,2\}$ at both vertices. B: GM steps 32 to 33, with list $\{0,1\}$ at
the scheduled vertex 11; the dashed edge is an existing conflict. Changing
one proposal and recomputing learned COMMIT completes each copied run.
Exact-COMMIT substitution on unchanged proposals preserves both failures.
All other messages and first-stage tensors stay fixed at the intervention;
these cases do not estimate a population rescue rate. Full outcomes are
in Table~\ref{tab:rescues}.}
\label{fig:failure-mechanisms}
\end{figure}

These controls separate a proposal defect from an incorrect veto in the
selected states. Exact-COMMIT substitution is an identity at these already
correct vetoes, not an all-state oracle replacement or an audit of COMMIT
everywhere. The controls do not establish that changing the proposal module
would fix every failed execution, or that the first latent calibration
discrepancy caused the observed terminal failure.
''')
    write("revision-rescue-details.tex", r'''
\subsection{Complete outcomes of the selected component interventions}
\begin{table}[htbp]\centering\small
\begin{tabular}{llrl}
\toprule Witness & Intervention & Steps & Terminal outcome \\
\midrule
'''+"\n".join(lines)+r'''
\bottomrule
\end{tabular}
\caption{Posthoc component interventions in two selected first-failure
witnesses, not new population success rates. Baselines preserve the original
16-case inference cohort. Exact confirmation is evaluated on the actual
learned messages.}
\label{tab:rescues}
\end{table}
The source is \path{results/paper-revision-20261030/component-rescues.json}.
Figure~\ref{fig:failure-mechanisms} extracts actual vertices, colors and
proposals from those records and the original witness frames. The full
64-node graphs, case/tape identities and unchanged 16-case cohort remain
part of the diagnosis; the two-node drawing omits the rest of each graph.
''')


if __name__=="__main__":
    architecture();historical();prospective();continuation_audit();be_failure_diagnostics();rescues()
    subprocess.run([sys.executable, str(Path(__file__).with_name("build-revision-mechanism-figure.py"))], check=True)
