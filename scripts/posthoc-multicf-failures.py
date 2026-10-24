#!/usr/bin/env python3
"""Observe original-batch failed rollouts; diagnostics never veto an action."""
import json
from pathlib import Path

import numpy as np
import torch

import dcl.samplecomplexity.evaluation as evaluator
from dcl.multiteacher.programs import KW,LINIAL,DETECT,teacher_choices
from dcl.samplecomplexity.interactive import admissible
from dcl.samplecomplexity.training import initialize
from dcl.study.storage import atomic_json,file_hash
from dcl.multicf.protocol import cases
from dcl.multicf.states import validate_choice

OUT=Path("results/multiteacher-counterfactual")


def examine(directory):
    result=json.loads((directory/"result.json").read_text())
    frozen=json.loads((directory/"freeze.json").read_text());selected=frozen["selected"];config=frozen["config"]
    records=[]
    for cp in selected["checkpoints"]:
        raw=json.loads((directory/"final"/f"s{cp['seed']}.json").read_text())
        failures=[r for r in raw if r["failure"]]
        if not failures:continue
        # Every failure is enumerated; one witness per checkpoint limits the
        # diagnostic computation without selecting a favorable evaluation run.
        target=failures[0];suite=cases(config,"final",cp["seed"])
        index=next(i for i,(_,m) in enumerate(suite) if m["case_id"]==target["case_id"])
        offset=index//32*32;cohort=suite[offset:offset+32];events=[]
        original=evaluator.ColoringProgram
        class ObservedProgram(original):
            def __init__(self,g,teacher,tape_seed,max_rounds):
                super().__init__(g,teacher,tape_seed,max_rounds);self.diagnostic_target=tape_seed==target["tape_seed"]
                self.verified_prefix=True
            def apply(self,s,chosen):
                if not self.diagnostic_target:return super().apply(s,chosen)
                safe_query=admissible(self,s) if self.teacher!="SPECULATIVE_GM" else self.verified_prefix
                exact=teacher_choices(s,self.neighbors,self.teacher) if safe_query else None
                event=dict(native_step=self.steps,phase=s.phase,bound=s.bound,target=s.target,field=s.field,
                    old_aux=s.aux.tolist(),old_colors=s.colors.tolist(),permission=s.permission.tolist(),
                    predicted_color=chosen.color.tolist(),predicted_auxiliary=chosen.auxiliary.tolist(),predicted_mark=chosen.mark.tolist(),
                    oracle_defined=safe_query,
                    reference_color=exact.color.tolist() if exact else None,
                    reference_auxiliary=exact.auxiliary.tolist() if exact else None,
                    reference_mark=exact.mark.tolist() if exact else None)
                if self.verified_prefix:
                    try:validate_choice(self,s,chosen)
                    except ValueError:self.verified_prefix=False
                before,aux=self.colors.copy(),self.aux.copy();super().apply(s,chosen)
                event.update(new_aux=self.aux.tolist(),new_colors=self.colors.tolist(),
                             invariant=evaluator.invariant_after(self,s,before,aux))
                if s.phase==KW:
                    high=s.aux>=s.target;write=chosen.color>0
                    event["omitted_required_kw_roots"]=np.flatnonzero(high&~write).tolist()
                    event["unexpected_low_kw_writes"]=np.flatnonzero(~high&write).tolist()
                    event["out_of_bound_roots"]=np.flatnonzero((self.aux<0)|(self.aux>=s.target)).tolist()
                    event["equal_auxiliary_edges"]=[[int(u),int(v)] for u,v in self.graph.edges() if self.aux[u]==self.aux[v]]
                events.append(event)
        try:
            evaluator.ColoringProgram=ObservedProgram
            model=initialize(selected["teacher"],selected["architecture"],cp["seed"])
            model.load_state_dict(torch.load(cp["path"],weights_only=False)["model"])
            replayed,_=evaluator.evaluate(model,selected["teacher"],cohort)
        finally:evaluator.ColoringProgram=original
        new=next(r for r in replayed if r["case_id"]==target["case_id"])
        match=all(new[k]==target[k] for k in ("failure","native_steps","colors","invariants","terminal_correct"))
        records.append(dict(package=directory.name,seed=cp["seed"],checkpoint_sha256=cp["sha256"],
            all_failures=[{k:r[k] for k in ("case_id","family","n","failure","native_steps")} for r in failures],
            witness_case=target["case_id"],exact_original_outcome_replayed=match,events=events,
            geometry="Original cohort of 32 cases and 1024-root prediction chunks, preserving numerical batch context",
            scope="Post hoc diagnosis only. Every applied action is unchanged; no labels enter a training cache. Oracle queries stop when their state/prefix domain is no longer verified."))
    return records


if __name__=="__main__":
    torch.set_num_threads(1);records=[]
    for p in sorted((OUT/"confirmation").glob("*/result.json")):
        if json.loads(p.read_text())["status"]!="PASS":records.extend(examine(p.parent))
    atomic_json(OUT/"posthoc-failure-witnesses.json",dict(rows=records,script_sha256=file_hash(__file__)))
    print([(r["package"],r["witness_case"],r["exact_original_outcome_replayed"]) for r in records])
