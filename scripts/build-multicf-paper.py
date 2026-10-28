#!/usr/bin/env python3
"""Add the evidence-derived extension while preserving the original manuscript."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import pypandoc
import pypdfium2 as pdfium

from dcl.study.storage import atomic_json,file_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/multiteacher-counterfactual"
PAPER=ROOT/"paper/ucgnn"


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),
                      *["| "+" | ".join(map(str,r))+" |" for r in rows]])


def short(t):return {"BE_BASE_D3":"BE","COLOR_TRIALS":"ColorTrials","SPECULATIVE_GM":"GM"}[t]


if __name__=="__main__":
    if (PAPER/"main.tex").read_text().startswith("% Authoritative closure source"):
        raise SystemExit("Historical generator: use scripts/build-closure-paper.py here, or a separate historical checkout.")
    p=argparse.ArgumentParser();p.add_argument("--compiler",default="/tmp/algolearning-verified-tectonic/tectonic");a=p.parse_args()
    compiler=Path(a.compiler)
    assert file_hash(compiler)=="a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da"
    reps=json.loads((OUT/"successful-fixed-points.json").read_text())
    fidelity=json.loads((OUT/"fidelity-summary.json").read_text())
    costs=json.loads((OUT/"compute-cost.json").read_text())
    info=json.loads((OUT/"information/results.json").read_text())
    report=json.loads((OUT/"report-inputs.json").read_text())
    points=table(["Teacher","Core","Updates","Native labels","All labels","Final"],
        [[short(r["teacher"]),r["architecture"],r["update"],min(r["native_labels"]),
          str(min(r["acquired_labels"])) if min(r["acquired_labels"])==max(r["acquired_labels"]) else f"{min(r['acquired_labels'])}--{max(r['acquired_labels'])}","5 x 1000/1000"] for r in reps])
    frows=[]
    for t in ("BE_BASE_D3","SPECULATIVE_GM","COLOR_TRIALS"):
        for arch in ("normalized","deepsets","transformer"):
            m=next(r for r in fidelity if r["teacher"]==t and r["architecture"]==arch and r["method"]=="matched")
            c=next(r for r in fidelity if r["teacher"]==t and r["architecture"]==arch and r["method"]=="counterfactual")
            frows.append([short(t),arch,f"{100*m['counterfactual_correct']/m['counterfactual_total']:.3f}",
                f"{100*c['counterfactual_correct']/c['counterfactual_total']:.3f}",f"{100*c['action_correct']/c['action_total']:.3f}"])
    ftable=table(["Teacher","Core","Matched CF (%)","CF-trained CF (%)","CF native actions (%)"],frows)
    itable=table(["JP representation","Development successes, five seeds (/60)"],
        [[mode,", ".join(str(r["development"]["successes"]) for r in info if r["mode"]==mode)]
         for mode in ("without_ids","numeric_ids","numeric_differences","relative_comparisons")])
    new=rf'''# Counterfactual supervision across native teachers

## Reconstruction and causal domains

We reconstruct M6B from authenticated original source and 21 historical
artifacts. The 249/500 traditional, 476/500 own-proposal and 500/500
counterfactual outcomes concern one training seed, 10001, on paired cubic
N10240 cases. Thirty original-policy executions are replayed exactly. M6B
uses the 8967-parameter M4B architecture, whereas canonical E1 has 16035
parameters. The own/CF checkpoints share every participation, first-message
and candidate tensor; only the second-message and COMMIT tensors differ.
The configured 120-epoch horizon is 17760 updates; the selected checkpoints
are epoch53, or 7844 updates. The traditional comparison also changes the
number of tapes per graph, so its improvement is not a pure CF effect.

Pedro's 16 source hashes are verified against commits fc756cef, 80f486eb and
6836d855. None of the 40 expected original E1 weight hashes was recovered
from 6614 local checkpoint files or the inspected public release manifests.
His five 1980/1980 results, JP 1631/1980 and GM 1951/1980 remain authenticated
reported evidence with five initializations over 396 published cases and a
common training pool. His 17760 updates of 1680 roots and own/CF training
also differ from the primary 500-update, batch64 passive study. New runs
below reproduce mechanisms, not those unavailable original policies.

We define a phase-specific label operation

$$\mathcal O_T(S,a,\xi)=D_T^\phi\bigl(R_T^\phi(S,a,\xi)\bigr),$$

where $S$ is an admissible native snapshot, $a$ a typed intervention,
$\xi$ the private input, and $R$ recomputes every causal descendant before
the phase decision $D$ is queried. We distinguish native decisions from
defensive continuation queries on offers that the native policy might not
emit. The latter are additional supervised tasks, not a claim that arbitrary
simulator arrays are reachable teacher states. For BE and JP, forced
participation cannot bypass phase eligibility or priority; ColorTrials
confirmation requires strict winning preferences under actual availability.
Linial points and GM detection marks have separate native oracles.

Admissible contexts are reconstructed from legal prefixes. Changing IDs
requires a new complete replay, including comparisons, blocks, permissions
and auxiliary initial colors. Private preference/draw interventions preserve
the persistent state and regenerate messages. KW arithmetic and Linial
evaluations are recomputed; speculative GM histories retain their supplied
schedule, actual assignment colors, marks and repair queue. We export 162
additional hard contexts with replay witnesses. The three palette cores
share these training-only adapters. All teacher and autonomous inference
source files remain unchanged from the baseline; no exact decision replaces
the neural confirmation head.

## Paired controls and strict new certifications

We train 135 primary runs: three pending teachers, three cores, five new
seeds and three treatments. Each preserves 500/1500/4000 checkpoints, the
same native bank, initialization and minibatch sequence within a seed/core.
Passive is the historical loss. Matched evaluates COMMIT with its stored
teacher proposals and corresponding label. Fixed-bank CF uses 100 matched
warmup updates, then P + C + .5 matched confirmation + .5 CF confirmation.
Each normal root acquires three native component labels and four additional
stored confirmation queries; Linial and DETECT each acquire one native label.
Own-proposal queries in a separate seven-teacher pilot are charged individually.

Training uses N8/12/16/24. An amendment recorded before the additional screen
and all new finals expands development from 60 to 300 cases at N28/36.
The first common perfect checkpoint is frozen per treatment. Final has five
independent 1000-case batteries at N32/48/64, ten fixed family strata,
native horizons, and the strict original invariant/terminal gate.

Matched passes all nine pending packages, using 1500 updates for BE,
500 for ColorTrials/E1, 1500 for its other cores and 500 for GM.
CF initially fails BE at 1500 in seed94105 for all cores, and
ColorTrials/E1 at 500 in seed94103. Independently frozen follow-ups compare
existing 1500/4000 CF checkpoints on new namespaces, retaining both outcomes.
BE1500 again fails in seed94105; ColorTrials/Transformer1500 has two failures
in its new battery. All six 4000-update packages pass. The following CF fixed
points therefore close all nine cells, while the historical twelve remain
separate: coverage is 21/21, not one homogeneous newly retrained cohort.

{points}

The complete extension retains 195000 final execution rows, 194979 correct,
and 170 distinct frozen checkpoints. Repeated checkpoints on a second
battery are not new training replicas. All eight representative failed
checkpoint/battery traces reproduce their original outcomes under the same
32-case cohort and 1024-root numerical batching. BE failures are omitted
required KW recolorings that violate the new auxiliary bound. No failure is
repaired or removed. An explicit erratum preserves and corrects the CT
follow-up plan's reason-text typo (the original E1 failure was at500,
not1500); the actual checkpoint hashes, batteries and rows are unchanged.

![All five unsmoothed development curves and descriptive means at fixed native banks. Perfect development is a selection screen, not a functional certificate.](figures/multicf-optimization-curves.pdf)

## Functional validity and continuation fidelity

At4000, all permitted roots in 60 common native development trajectories
per seed are audited, with four fresh continuation queries per color root.
The table reports descriptive aggregate percentages across five policies;
root decisions are not independent training replicas. Per core, the CF
denominators are 499200 for BE, 41036 for GM and 57576 for ColorTrials.
Full per-seed action, actual-transition, component and phase counts remain
machine-readable. The certified GM500 policies have separate full audits
at their actual checkpoint; the table fixes4000 for the paired comparison.
CF improves the broader continuation task, while the
matched/passive procedures already satisfy the functional gate. Thus CF
is not uniformly necessary, or better at the same update count, for native
functional success. A matched ColorTrials/E1 run regresses from60/60 to43/60
development successes between1500 and4000 updates.

{ftable}

For BE/GM, every second-stage and COMMIT tensor is identical across the
three palette cores in each paired fixed-proposal seed/update comparison.
Those losses do not use the candidate scorer, whereas ColorTrials reads its
candidate after the second exchange and couples those gradients. Shared
heads explain correlated architecture outcomes; they do not provide three
independent confirmations of a head-level effect.

![Errors on fresh continuation queries, separated from native action fidelity. Small nonzero errors retain explicit labels.](figures/multicf-counterfactual-fidelity.pdf)

Nested ColorTrials banks at native budgets1000/3000/10000/30000 and fixed
500/1500 updates provide development-only supervision curves. Every CF
label is charged; these do not establish a new certified minimum. A matched
GM control without the added isolate families also passes all five new
development seeds, so the tested ablation does not isolate a positive
coverage effect. The authenticated old seed83105 failures and all30 new
regressions remain a separate observation. The own-plus-CF portability pilot
passes60/60 development in six teachers for one new E1 seed, but BE obtains
22/60 at1500; copying the Luby curriculum is insufficient evidence of a
uniformly successful multiteacher procedure.

![Supervision and optimization are distinct axes; thin lines show all five seeds. These lower-budget points have no new final certification.](figures/multicf-supervision-curves.pdf)

## Information, complete outputs and transfer

JP's information control fixes the architecture, labels, batches,500 updates
and initialization, changing only two local context channels. Relative order
passes all five60-case development suites; raw endpoint IDs, differences
and removed IDs do not. Twenty numerical replays reproduce every original
ablation weight tensor and retain intermediate Adam/RNG checkpoints.

{itable}

An indistinguishability witness swaps the IDs on a pending edge: anonymous
observations are equal but native eligibility reverses. Edge order suffices
for the implemented priority decision. This is an observation-level
obstruction, not a claim that numeric IDs are unlearnable. ColorTrials has
a second-hop availability witness. Historical M6 also aliases the float64
draws $1/2-2^{{-28}}$ and $1/2+2^{{-28}}$ to the same float32 observation,
despite opposite activation labels. The current native teacher directly
queries its supplied float32 inputs, avoiding that particular obstruction.

![Information representation at a fixed optimizer and label point. Each dot is one training seed; development performance is not final certification.](figures/multicf-jp-information.pdf)

After functional selection, all45 successful CF policies are frozen before
fresh law and transfer tapes. On path8 and K4 plus four isolates,256
shared-input runs per graph/policy compare the full labelled terminal vector
and status, without success filtering. The coupling inequality and exact
one-sided binomial bounds with Bonferroni .05/90 yield89/90 simultaneous
TV-at-most-.05 certificates. These are approximate fixed-graph marginal-law
bounds, not equality. ColorTrials/E1 seed94104 on path8 has36/256 coupled
disagreements, upper bound .223630 and mean2.191406 native steps versus
2.097656. Coupled disagreement is not a TV lower bound or proof that the
marginal laws differ. Empirical full-vector histograms and native times are
retained separately.

Exploratory transfer obtains {report['transfer_successes']}/{report['transfer_executions']}
correct executions,84 per policy at N128/256/512. Its ten domain families
are supplemented by binary trees, diamond chains, triangle chains and
subdivided K4 components, all of maximum degree3. These finite observations
do not extend the teacher to higher degree, local palettes or arbitrary
graphs/tapes. Functional validity does not establish universal decision or
law identity.

The complete campaign contains {costs['training_runs']} computational training
runs, including paired arms, pilots and numerical replays, with
{costs['gradient_updates']} gradient updates and
{costs['optimization_cpu_seconds']/3600:.3f} CPU optimization hours.
Training GPU time is zero; the original M6 CUDA replay is separate.
Process times overlap. Acquired components, reused labels, presentations,
simulator work and optimization remain separate; no sample-efficiency claim
compares unequal update budgets. All sources, models, datasets, witnesses,
negative results and manifests accompany this extension. The verification
recomputes every final gate and all stored CF labels with zero label
inconsistencies. The full suite's639 cases and29 later added cases pass.

These contributions complement trace-oriented CLRS @velickovic2022clrs and
learner-state acquisition in DAgger @ross2011dagger. Recent contrastive
graph-coloring work @le2026contrastive studies approximate-coloring
objectives. Our contribution is the specified admissible-oracle framework,
controlled separation of training factors, and finite native execution
evidence; no comprehensive bibliographic priority or universal learnability
claim follows from this study.
'''
    new=new.replace("The full suite's639 cases and29 later\nadded cases pass.","All 668 cases in the final full test suite pass.")
    new=re.sub(r"\b(epoch|seed|batch|at|of|from|to|all|budgets|passes|obtains|mean|versus|between|yield|five)(?=\d)",r"\1 ",new,flags=re.I)
    new=re.sub(r"(?<=\d)(?=(?:updates|cases|case|epochs|steps|pages|roots|native|final|independent|gradient|CPU)\b)"," ",new)
    new=re.sub(r",(?=\d)",", ",new)
    new=new.replace("BE1500","BE 1500")
    # Preserve the source/PDF of the previous integrated draft byte-for-byte.
    for ext in ("tex","pdf"):
        original=subprocess.check_output(["git","show",f"61f39760:paper/ucgnn/main.{ext}"],cwd=ROOT)
        archive=PAPER/f"pre-counterfactual-main.{ext}"
        if archive.exists():assert archive.read_bytes()==original
        else:archive.write_bytes(original)
    for name in ("optimization-curves","counterfactual-fidelity","supervision-curves","jp-information"):
        for ext in ("pdf","png"):
            shutil.copyfile(OUT/"figures"/f"{name}.{ext}",PAPER/"figures"/f"multicf-{name}.{ext}")
    bib='''@misc{le2026contrastive,
 title={Contrastive Neural Algorithmic Reasoning for Graph Coloring},
 author={Le, Thien and Zhao, Tianyu and Weber, Melanie},
 year={2026}, eprint={2606.03923}, archivePrefix={arXiv},
 url={https://arxiv.org/abs/2606.03923}}
'''
    (PAPER/"multicf-references.bib").write_text(bib)
    (OUT/"paper-extension.md").write_text(new)
    fragment=pypandoc.convert_text(new,"latex",format="markdown",extra_args=["--natbib"])
    (PAPER/"multicf-details.tex").write_text(fragment)
    abstract="A counterfactual extension adds admissible phase-specific oracles and fixed-label optimization controls. It closes all nine previously pending teacher/core packages on fresh five-policy, 1000-execution batteries while preserving the twelve historical certifications. Concordant controls also close those cells; counterfactual continuation fidelity and required optimizer effort differ. Complete-output coupling, information ablations and larger-graph transfer delimit what functional success establishes."
    conclusion="The counterfactual extension supplies independently validated functional points for all nine pending packages, preserving the prior twelve as historical reference. Its controls show that label concordance and optimization can close a functional gap without demonstrating universal CF superiority. Counterfactual continuations, shared-head dependencies, information representation and complete-output laws remain separate scientific objects, with every failed point retained."
    md=subprocess.check_output(["git","show","61f39760:paper/PAPER_DRAFT.md"],cwd=ROOT,text=True)
    md=md.replace("\n# Introduction\n","\n"+abstract+"\n\n# Introduction\n",1)
    md=md.replace("\n# Conclusions\n","\n"+new+"\n\n# Conclusions\n",1)
    md=md.replace("These conclusions are conditional on the stated protocols and evidence.",conclusion+"\n\nThese conclusions are conditional on the stated protocols and evidence.",1)
    (ROOT/"paper/PAPER_DRAFT.md").write_text(md)
    tex=(PAPER/"pre-counterfactual-main.tex").read_text()
    tex=tex.replace(r"\section{Introduction}",pypandoc.convert_text(abstract,"latex",format="markdown")+"\n"+r"\section{Introduction}",1)
    tex=tex.replace(r"\section{Conclusions}","\\input{multicf-details.tex}\n\n"+r"\section{Conclusions}",1)
    tex=tex.replace("These conclusions are conditional on the stated protocols and evidence.",pypandoc.convert_text(conclusion,"latex",format="markdown")+"\n\nThese conclusions are conditional on the stated protocols and evidence.",1)
    tex=tex.replace(r"\bibliography{references.bib,integration-references.bib}",r"\bibliography{references.bib,integration-references.bib,multicf-references.bib}")
    (PAPER/"main.tex").write_text(tex)
    result=subprocess.run([str(compiler),"-X","compile","--keep-logs","main.tex"],cwd=PAPER,text=True,capture_output=True)
    (OUT/"paper-build.log").write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError("Paper compilation failed")
    doc=pdfium.PdfDocument(str(PAPER/"main.pdf"));pages=[p.get_textpage().get_text_range() for p in doc]
    log=(PAPER/"main.log").read_text(errors="replace")
    undefined=[line for line in log.splitlines() if "undefined" in line.lower() and any(k in line.lower() for k in ("citation","reference"))]
    if undefined:raise RuntimeError(f"Unresolved references: {undefined}")
    review=OUT/"paper-review";review.mkdir(exist_ok=True)
    selected_pages={0}
    for i,text in enumerate(pages):
        if any(s.lower() in text.lower() for s in ("Counterfactual supervision across native teachers","Paired controls and strict new","Functional validity and continuation","Information, complete outputs","195000","TV-at-most")):
            selected_pages.update(range(max(0,i-1),min(len(pages),i+3)))
    rendered=[]
    for i in sorted(selected_pages):
        path=review/f"page-{i+1:02d}.png";doc[i].render(scale=1.1).to_pil().save(path)
        rendered.append(dict(page=i+1,path=str(path.relative_to(ROOT)),sha256=file_hash(path)))
    atomic_json(OUT/"paper-build.json",dict(compiled=True,pages=len(doc),undefined_references=undefined,
        overfull_boxes=[line for line in log.splitlines() if "Overfull" in line],
        missing_characters=[line for line in log.splitlines() if "Missing character" in line],
        pdf_sha256=file_hash(PAPER/"main.pdf"),tex_sha256=file_hash(PAPER/"main.tex"),
        extension_sha256=file_hash(PAPER/"multicf-details.tex"),compiler_sha256=file_hash(compiler),
        renders=rendered,visual_review_passed=False))
    print({"compiled":True,"pages":len(doc),"rendered_pages":len(rendered)})
