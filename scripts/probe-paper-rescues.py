#!/usr/bin/env python3
"""Posthoc one-component interventions on copied first-failure executions.

Preserve the original full inference cohort and unmodified kernels up to the
intervention. Interventions are diagnostic only; no trial/model is repaired.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from dcl.degree.adapter import FrozenAdapter
from dcl.degree.evaluation import combine, split_decision, component_diagnostics
from dcl.degree.run import suite_for
from dcl.degree.teachers import DegreeProgram, availability
from dcl.multiteacher.programs import REPAIR, DETECT
from dcl.samplecomplexity.model import QueryPolicy
from dcl.study.storage import atomic_json, file_hash

BASE=Path("results/final-closure")
DEGREE=Path("results/multiteacher-degree-generalization")
OUT=Path("results/paper-revision-20261030")


@torch.inference_mode()
def run(witness_path, mode):
    witness=json.loads(witness_path.read_text())
    cp=next(r for r in json.loads((BASE/"failure-replay-audit.json").read_text())["rows"]
            if r["checkpoint"]==witness["checkpoint"] and r["condition"]==witness["condition"]["id"])
    index=json.loads((OUT/"recovered-checkpoints.json").read_text())
    checkpoint=Path(index[cp["sha256"]][0]); assert file_hash(checkpoint)==cp["sha256"]
    model=QueryPolicy(cp["teacher"],cp["architecture"]).eval()
    model.load_state_dict(torch.load(checkpoint,weights_only=False)["model"])
    # Observe the complete first-stage argmax, including non-writable centers,
    # so a one-entry override preserves every other supplied message exactly.
    original_scores=model._scores; captured={}
    def capture(*args,**kwargs):
        scores,alpha=original_scores(*args,**kwargs)
        captured["scores"]=scores.detach().numpy().copy()
        return scores,alpha
    model._scores=capture
    adapter=FrozenAdapter(model)
    plan=json.loads((DEGREE/"confirmatory/design.json").read_text())
    condition=witness["condition"]
    suite=suite_for(plan,condition,cp["seed"]%100-1)
    part=suite[witness["batch_start"]:witness["batch_start"]+witness["batch_size"]]
    target=next(i for i,(_,m) in enumerate(part) if m["case_id"]==witness["case"]["case_id"])
    programs=[DegreeProgram(g,cp["teacher"],m["tape_seed"],condition["palette"]) for g,m in part]
    if cp["teacher"]=="SPECULATIVE_GM":
        trigger=next(f for f in witness["frames"] if f["phase"]==REPAIR and f["scheduled_write_omissions"])
        vertex=trigger["roots"][0]
    else:
        trigger=next(f for f in witness["frames"] if sum(c<0 for c in f["colors"])==2
                     and len(set(f["candidates"]))==1)
        vertex=trigger["roots"][0]
    intervention=None; preintervention_matches=True; frames=[]
    while any(not p.finished for p in programs):
        groups=defaultdict(list)
        for i,p in enumerate(programs):
            if not p.finished:groups[p.phase,p.k].append((i,p,p.snapshot()))
        for group in groups.values():
            joined,state,offsets=combine([(p,s) for _,p,s in group])
            decision=adapter(joined,state)
            pieces=split_decision(decision,offsets)
            for local,(i,p,s) in enumerate(group):
                if i!=target:continue
                d=pieces[local]
                if p.steps<=trigger["step"]:
                    old=witness["frames"][p.steps]
                    preintervention_matches &= (s.colors.tolist()==old["colors"] and d.candidates.tolist()==old["candidates"]
                        and d.commit.tolist()==old["commit"] and d.choices.color.tolist()==old["predicted_color"])
                if p.steps==trigger["step"]:
                    free=availability(p,s)
                    intervention=dict(step=p.steps,phase=s.phase,vertex=vertex,available=np.flatnonzero(free[vertex]).tolist(),
                        baseline_candidates=d.candidates.tolist(),baseline_commit=d.commit.tolist(),baseline_color=d.choices.color.tolist())
                    if mode=="candidate_once":
                        replacement=int(np.flatnonzero(free[vertex])[0])
                        if replacement==int(d.candidates[np.flatnonzero(d.roots==vertex)[0]]):
                            replacement=int(next(c for c in np.flatnonzero(free[vertex]) if c!=replacement))
                        roots=np.flatnonzero(state.permission); adjacent=joined.neighbors[roots].ravel()
                        centers=np.unique(np.r_[roots,adjacent[adjacent<joined.n]])
                        full=np.zeros(joined.n,np.int64); full[centers]=captured["scores"].argmax(-1)
                        full[int(offsets[local])+vertex]=replacement
                        altered=adapter(joined,state,{"candidates":full})
                        assert np.sum(altered.candidates!=decision.candidates)==1
                        assert np.array_equal(altered.participation,decision.participation)
                        assert np.array_equal(altered.availability,decision.availability)
                        assert np.array_equal(altered.scores,decision.scores)
                        changed=split_decision(altered,offsets)
                        d=changed[local]; pieces[local]=d
                        intervention["replacement_candidate"]=replacement
                        intervention["only_one_candidate_changed_with_identical_first_stage"]=True
                    elif mode=="exact_commit_once":
                        confirm=component_diagnostics(p,s,d)["commit_matched"][1]
                        d.choices.color[d.roots]=np.where(d.participation & confirm,d.candidates+1,0)
                        intervention["oracle_confirm"]=confirm.tolist()
                    intervention.update(result_candidates=d.candidates.tolist(),result_commit=d.commit.tolist(),result_color=d.choices.color.tolist())
            for (i,p,s),d in zip(group,pieces):
                before=p.colors.copy(); p.apply(s,d.choices)
                if i==target:
                    frames.append(dict(step=p.steps,phase=s.phase,colors=p.colors.tolist(),bad_edges=p.bad_edges()))
                valid=np.all((p.colors>=-1)&(p.colors<p.palette)) and np.array_equal(before[~s.permission],p.colors[~s.permission])
                if cp["teacher"]!="SPECULATIVE_GM":
                    valid &= not p.bad_edges() and np.array_equal(before[before>=0],p.colors[before>=0])
                if not valid:p.finished,p.failure=True,p.failure or "invariant"
    p=programs[target]
    baseline_match=(p.steps==witness["case"]["steps"] and p.failure==witness["case"]["failure"] and p.colors.tolist()==witness["case"]["colors"])
    assert preintervention_matches
    if mode=="baseline":assert baseline_match
    return dict(checkpoint=cp["checkpoint"],checkpoint_sha256=cp["sha256"],condition=condition,
        case_id=witness["case"]["case_id"],mode=mode,preserved_batch_size=len(part),
        original_witness_sha256=file_hash(witness_path),preintervention_matches=preintervention_matches,
        baseline_terminal_matches=baseline_match,intervention=intervention,steps=p.steps,failure=p.failure,
        success=p.failure is None and p.bad_edges()==0 and bool(np.all(p.colors>=0)),
        terminal_bad_edges=p.bad_edges(),colors=p.colors.tolist(),frames=frames)


def main():
    torch.set_num_threads(1)
    paths=[BASE/"failure-replays"/c/f for c,f in (
        ("JOHANSSON-transformer-historical-primary-passive-s83101","global-n64-d10-mixture1000.json"),
        ("SPECULATIVE_GM-transformer-counterfactual-primary-counterfactual-s94102","local-n64-d4-mixture1000.json"))]
    rows=[]
    for path in paths:
        for mode in ("baseline","candidate_once","exact_commit_once"):
            result=run(path,mode);rows.append(result)
            print(result["checkpoint"],mode,result["steps"],result["success"],result["failure"],flush=True)
    atomic_json(OUT/"component-rescues.json",dict(rows=rows,script_sha256=file_hash(__file__),
        interpretation="Posthoc diagnostics on two selected witnesses. One candidate is replaced and the learned COMMIT recomputed on the same full cohort. Exact-COMMIT control acts on unchanged messages. This is not a repaired deployed policy, a new success-rate estimate, or population-level causal evidence."))


if __name__=="__main__":main()
