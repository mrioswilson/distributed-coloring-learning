"""Audit completed task coverage against the frozen protocol and raw artifacts."""
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from dcl.study.storage import atomic_json,file_hash,now
from .config import ROOT,SEEDS,cohort_plan,protocol_hash
from .final_analysis import PRIMARY


def fingerprint_set(value):
    found=set()
    if isinstance(value,dict):
        if value.get("fingerprint"):found.add(value["fingerprint"])
        for child in value.values():found.update(fingerprint_set(child))
    elif isinstance(value,list):
        for child in value:found.update(fingerprint_set(child))
    return found


def audit():
    checks=[]
    def require(name,condition,evidence):
        checks.append({"requirement":name,"passed":bool(condition),"evidence":evidence})
        if not condition:raise RuntimeError(f"Audit failed: {name}: {evidence}")
    manifest=json.loads((ROOT/"manifest.json").read_text())
    require("all scheduled tasks completed",manifest["status"]=="complete" and
        all(t["status"]=="complete" for t in manifest["tasks"]),dict(Counter(t["action"] for t in manifest["tasks"])))
    require("GPU budget",manifest["active_seconds"]<=36*3600,manifest["active_seconds"]/3600)
    require("original preregistration unchanged",protocol_hash()==
        "6bb9af85893a7c48c41362a875cbbc0cd1f0d38b989785c0d931f68510f2087a",protocol_hash())
    snapshot=json.loads(Path("docs/historical/source-snapshot.json").read_text())
    require("historical sources unchanged",all(file_hash(r["source"])==r["sha256"] for r in snapshot["files"]),len(snapshot["files"]))
    training=pd.read_csv("results/ucgnn-paper-study/training.csv")
    expected={(t,s) for t in PRIMARY for s in SEEDS}|{("own_only",41000),("cyclic_teacher",41005)}
    require("32 full-budget training runs",set(zip(training.treatment,training.seed))==expected and
        training.complete.all() and training.epoch.eq(120).all(),len(training))
    paired=pd.read_csv("results/ucgnn-paper-study/causal-isolation-audit.csv")
    columns=["initial_tensors_exact_equal","data_hash_equal","orders_and_cf_tapes_equal","complete_both","updates_equal_at_completion"]
    require("five tensor/data/tape/update matched normalization pairs",set(paired.seed)==set(SEEDS) and
        paired[columns].all().all() and paired.paired_epochs_verified.eq(120).all(),paired.to_dict("records"))
    cohort=json.loads((ROOT/"confirmatory-cohort.json").read_text())
    cases=pd.read_csv("results/ucgnn-paper-study/cases.csv")
    plan=cohort_plan();reference={c["case_id"]:c for c in cohort["cases"]}
    require("frozen cohort case identities",set(reference)=={c["case_id"] for c in plan} and
        len(reference)==304 and cohort["graphs"]==288,{"cases":len(reference),"graphs":cohort["graphs"]})
    exclusion=json.loads((ROOT/"historical-exclusions.json").read_text())
    forbidden=set(exclusion["fingerprints"])
    # The secondary teacher was trained after the exclusion cache was frozen.
    # Check its train/validation graphs explicitly as well as every main run.
    for p in (ROOT/"runs").glob("*/data_manifest.json"):
        forbidden.update(fingerprint_set(json.loads(p.read_text())))
    require("fresh non-clique graphs",all(c["family"]=="CLIQUE" or c["fingerprint"] not in forbidden
        for c in reference.values()),"historical exclusion inventory plus all 32 current train/validation manifests")
    specs={}
    for p in (ROOT/"models").glob("*.json"):
        spec=json.loads(p.read_text());specs[spec["label"]]=spec
        require("checkpoint hash "+spec["label"],file_hash(spec["checkpoint"])==spec["checkpoint_sha256"],spec["checkpoint_sha256"])
    require("no duplicate evaluations",not cases.duplicated(["label","case_id"]).any(),len(cases))
    for label,spec in specs.items():
        subset=cases[cases.label==label]
        expected_cases={c["case_id"] for c in plan if (not spec["core_only"] or c["tier"]=="core") and
                        (spec["degree"] is None or c["degree"]==spec["degree"])}
        complete=json.loads((ROOT/"evaluations"/label/"complete.json").read_text())
        require("evaluation coverage "+label,set(subset.case_id)==expected_cases and complete["complete"] and
            complete["planned_cases"]==len(expected_cases) and subset.checkpoint_sha256.eq(spec["checkpoint_sha256"]).all(),len(subset))
        for row in subset.itertuples():
            c=reference[row.case_id]
            require("paired metadata "+label+" "+row.case_id,
                row.fingerprint==c["fingerprint"] and row.graph_id==c["graph_id"] and
                int(row.tape_seed)==c["tape_seed"] and row.degree==c["degree"] and row.n==c["n"],"cohort matched")
    cells=pd.read_csv("results/ucgnn-paper-study/cells.csv")
    require("no unexecuted/technical cells",cells.unexecuted.eq(0).all() and cells.technical_failures.eq(0).all(),len(cells))
    summaries=list((ROOT/"mechanisms/completed").glob("*.json"))
    require("mechanisms all epoch-120 universal checkpoints",len(summaries)==39,len(summaries))
    certs=[]
    for path in (ROOT/"mechanisms/certificates").glob("*.json"):
        c=json.loads(path.read_text());certs.append(c)
        require("realizable/reduced signs "+c["label"],len(c["actual_realizable_stars"])==64 and
                all(r["same_decision"] for r in c["actual_realizable_stars"]),64)
        if c["float32_model"]["applicable"]:
            require("float32 enclosure "+c["label"],all(r["inside_float32_error_enclosure"] for r in c["actual_realizable_stars"]),64)
        if c["real"].get("certified_exact_rule"):
            for dtype in ("numeric_float32","numeric_float64"):
                require("perturbed signs "+dtype+" "+c["label"],
                    all(r["perturbed_both_correct"] for r in c[dtype]["rows"]),"pooled messages +/-1e-6")
    require("39 complete certificates",len(certs)==39,len(certs))
    progress=pd.read_csv("results/ucgnn-paper-study/progress-diagnostics.csv")
    require("controlled progress coverage",len(progress)==3520 and
        progress.groupby(["label","state_id"]).tape.nunique().eq(16).all(),{"executions":len(progress),"models":progress.label.nunique()})
    held=pd.read_csv("results/ucgnn-paper-study/candidate-diagnostics.csv")
    held=held[held.heldout_prediction]
    require("frozen E1 held-out prediction coverage",len(held)==120 and set(held.palette)=={17,63,251,997},len(held))
    registration=json.loads((ROOT/"mechanisms/cpu-prediction-registration.json").read_text())
    for key,name in (("calibration_sha256","frozen-quantile-calibration.json"),("analytic_sha256","frozen-analytic-quantiles.json")):
        require("frozen predictor "+name,file_hash(ROOT/"mechanisms"/name)==registration[key],registration[key])
    result={"audit_utc":now(),"protocol_sha256":protocol_hash(),"queue_sha256":file_hash(ROOT/"manifest.json"),
        "tasks":len(manifest["tasks"]),"training_runs":len(training),"model_views":len(specs),"evaluation_cases":len(cases),
        "gpu_process_hours":manifest["active_seconds"]/3600,"checks_passed":len(checks),
        "checks":checks,"scope":"Coverage/provenance audit, not universal learned-algorithm correctness or independent code verification."}
    atomic_json("results/ucgnn-paper-study/completion-audit.json",result)
    print({k:v for k,v in result.items() if k!="checks"})


if __name__=="__main__":audit()
