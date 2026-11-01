#!/usr/bin/env python3
"""Independent frozen-model verification of the newly delivered two studies.

Stages preserve original data/results. Models are never trained or selected.
"""
import argparse
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from dcl.multicf.oracles import confirmation
from dcl.multicf.protocol import cases
from dcl.multiteacher.programs import LINIAL,DETECT
from dcl.pedrorev.evaluation import annotate,responses
from dcl.samplecomplexity.evaluation import evaluate
from dcl.samplecomplexity.protocol import passed_run,summarize_runs
from dcl.samplecomplexity.training import initialize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/martin-artifacts-20261031"
STUDIES=("cf-experiment","cf-followup")


def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def read(p):return json.loads(Path(p).read_text())


def write(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,indent=2)+"\n")


def bankspec(study,teacher):
    data=ROOT/"data"/("pedrorev-cf" if study=="cf-experiment" else "pedrorev-cf-followup")
    return data/"evaluation-banks"/teacher.lower()


def load_npz(p):
    with np.load(p,allow_pickle=False) as raw:return {k:raw[k] for k in raw.files}


def jobs():
    return [(study,row) for study in STUDIES for row in read(ROOT/"results/pedro-revision"/study/"freeze.json")["checkpoints"]]


def freeze():
    inputs={}
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"src/dcl").rglob("*.py"))}
    sources[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    protocol=ROOT/"docs/MARTIN_ARTIFACTS_REPLAY_PROTOCOL.md"
    sources[str(protocol.relative_to(ROOT))]=sha(protocol)
    references=[]
    for study in STUDIES:
        directory=ROOT/"results/pedro-revision"/study
        d=read(directory/"freeze.json")
        for p,digest in d["sources"].items():assert sha(ROOT/p)==digest,p
        config=ROOT/"configs"/("pedrorev-cf.json" if study=="cf-experiment" else "pedrorev-cf-followup.json")
        assert sha(config)==d["config_sha256"]
        for path in [config,*directory.glob("*.json"),*directory.glob("functional/**/*.json")]:inputs[str(path.relative_to(ROOT))]=sha(path)
        for row in d["checkpoints"]:
            assert sha(ROOT/row["checkpoint"])==row["sha256"]
            references.append(dict(study=study,**{k:row[k] for k in ("teacher","seed","arm","update","checkpoint","sha256")}))
    member=OUT/"member-index.json";inputs[str(member.relative_to(ROOT))]=sha(member)
    target=OUT/"model-replay-freeze.json"
    record=dict(scope="Post-publication replication of220 original evaluations;105 trainings,205 unique frozen checkpoints;120 primary functional model/battery evaluations",
        inputs=inputs,sources=sources,references=references,environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        batching=dict(query_chunk=256,functional_batch=1024,functional_cohort=32,torch_threads=1))
    if target.exists():assert read(target)==record
    else:write(target,record)
    print(json.dumps(dict(references=len(references),unique_checkpoints=len({r["checkpoint"] for r in references}),sources=len(sources))),flush=True)


def verify():
    p=OUT/"model-replay-freeze.json";d=read(p)
    for name,digest in {**d["sources"],**d["inputs"]}.items():assert sha(ROOT/name)==digest,name
    return sha(p)


def inventory():
    digest=verify();alljobs=jobs();sets=[]
    for study in STUDIES:
        sets.append({r["checkpoint"] for s,r in alljobs if s==study})
        for response in read(ROOT/"results/pedro-revision"/study/"responses-index.json"):
            assert sha(ROOT/response["path"])==response["sha256"],response["path"]
    grouped={}
    for _,r in alljobs:
        key=r["teacher"],r["seed"]
        if key in grouped:assert grouped[key]==r["initial_hash"]
        grouped[key]=r["initial_hash"]
    bankrefs=read(ROOT/"results/pedro-revision/cf-experiment/banks.json")
    for b in bankrefs:assert sha(ROOT/b["bank"]/"dataset.npz")==b["sha256"]
    checkpoints=sorted((ROOT/"data/pedrorev-cf").rglob("*.pt"))+sorted((ROOT/"data/pedrorev-cf-followup").rglob("*.pt"))
    assert len(checkpoints)==315
    epochs=Counter(p.name for p in checkpoints)
    record=dict(freeze_sha256=digest,references=len(alljobs),unique_frozen_paths=len(sets[0]|sets[1]),
        reused_control_paths=len(sets[0]&sets[1]),checkpoint_files=len(checkpoints),
        training_directories=len({p.parent for p in checkpoints}),snapshots_by_filename=dict(epochs),
        all_frozen_weight_hashes_match=True,all_saved_response_hashes_match=True,all_acquired_bank_hashes_match=True,
        original_source_freezes_match=True,shared_initial_hash_metadata_agrees=True)
    assert record["references"]==220 and record["unique_frozen_paths"]==205 and record["reused_control_paths"]==15 and record["training_directories"]==105
    write(OUT/"checkpoint-inventory-audit.json",record);print(json.dumps(record),flush=True)


def validate_arrays():
    digest=verify();records=[]
    for study in STUDIES:
        teachers=sorted({r["teacher"] for s,r in jobs() if s==study})
        for teacher in teachers:
            directory=bankspec(study,teacher)
            bank=load_npz(directory/"states.npz");q=load_npz(directory/"queries.npz")
            assert sha(directory/"states.npz")==read(directory/"manifest.json")["sha256"]
            assert sha(directory/"queries.npz")==read(directory/"queries-manifest.json")["sha256"]
            for start in range(0,len(q["y"]),256):
                sl=slice(start,start+256);use=q["row"][sl]
                arrays=tuple(bank[f"a{i}"][use] for i in range(4))
                truth=confirmation(arrays,teacher,q["p"][sl],q["c"][sl])
                assert np.array_equal(truth,q["y"][sl]),(study,teacher,start)
                flags,bound,key=annotate(arrays,teacher,q["p"][sl],q["c"][sl])
                assert np.array_equal(flags,q["flags"][sl]) and np.array_equal(bound,q["bound"][sl]) and np.array_equal(key,q["be_key"][sl])
            records.append(dict(study=study,teacher=teacher,queries=len(q["y"]),raw_oracle_labels_and_annotations_match=True))
            print(json.dumps(records[-1]),flush=True)
            del bank,q
    write(OUT/"raw-query-label-audit.json",dict(freeze_sha256=digest,records=records))


def reanalysis():
    digest=verify();records=[]
    for study,filename in (("cf-experiment","pedrorev-cf-analysis.py"),("cf-followup","pedrorev-cf-followup-analysis.py")):
        spec=importlib.util.spec_from_file_location("immutable_analysis",ROOT/"scripts"/filename)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        if hasattr(m,"training_keys"):m.training_keys=lru_cache(None)(m.training_keys)
        def save(ignored,value):
            assert value==read(ROOT/"results/pedro-revision"/study/"analysis.json"),(study,"recomputed metric mismatch")
            write(OUT/(study+"-recomputed-analysis.json"),value)
            records.append(dict(study=study,rows=len(value["rows"]),all_metrics_equal=True))
        m.atomic_json=save;m.main()
    write(OUT/"recomputed-metrics-audit.json",dict(freeze_sha256=digest,records=records))


def response_path(study,r):
    if study=="cf-experiment":return ROOT/f"data/pedrorev-cf/responses/{r['teacher'].lower()}/s{r['seed']}-{r['arm']}-u{r['update']:06d}.npy"
    return ROOT/f"data/pedrorev-cf-followup/responses/s{r['seed']}-{r['arm']}.npy"


def job_name(r):return f"{r['teacher'].lower()}-s{r['seed']}-{r['arm']}-u{r['update']:06d}"


def infer():
    digest=verify();records=[];current=None;bank=q=None
    for study,r in jobs():
        target=OUT/"reinference"/study/(job_name(r)+".json")
        if target.exists():record=read(target);assert record["freeze_sha256"]==digest
        else:
            key=study,r["teacher"]
            if key!=current:
                bank=q=None
                directory=bankspec(*key);bank=load_npz(directory/"states.npz");q=load_npz(directory/"queries.npz");current=key
            start=time.monotonic()
            actual=responses(r["teacher"],ROOT/r["checkpoint"],r["seed"],bank,q)
            original=np.load(response_path(study,r),allow_pickle=False)
            assert actual.shape==original.shape and np.isfinite(actual).all() and np.isfinite(original).all()
            buffer=io.BytesIO();np.save(buffer,actual,allow_pickle=False)
            rawpath=target.with_suffix(".npy.gz");rawpath.parent.mkdir(parents=True,exist_ok=True)
            rawpath.write_bytes(gzip.compress(buffer.getvalue(),mtime=0))
            record=dict(study=study,teacher=r["teacher"],seed=r["seed"],arm=r["arm"],update=r["update"],
                queries=len(actual),sign_mismatches=int(((actual>0)!=(original>0)).sum()),
                bitwise_equal=bool(np.array_equal(actual,original)),different_logits=int((actual!=original).sum()),
                max_abs_error=float(np.max(np.abs(actual-original))),freeze_sha256=digest,checkpoint_sha256=r["sha256"],
                original_response_sha256=sha(response_path(study,r)),raw_recomputed_logits=str(rawpath.relative_to(ROOT)),
                raw_recomputed_sha256=sha(rawpath),seconds=time.monotonic()-start)
            write(target,record)
        records.append(record);print(json.dumps({k:record[k] for k in ("study","teacher","seed","arm","update","queries","sign_mismatches","max_abs_error")}),flush=True)
        if record["sign_mismatches"]:raise RuntimeError("Prediction mismatch preserved; no models/splits were changed")
    write(OUT/"model-reinference-audit.json",dict(freeze_sha256=digest,models_and_batteries=len(records),
        decisions=sum(r["queries"] for r in records),all_decisions_match=all(r["sign_mismatches"]==0 for r in records),
        bitwise_equal_models=sum(r["bitwise_equal"] for r in records),max_abs_error=max(r["max_abs_error"] for r in records),records=records))


def functional():
    digest=verify();records=[];cache={}
    selected=[(s,r) for s,r in jobs() if r["update"]==4000]
    selected.sort(key=lambda x:(x[0]!="cf-followup",x[1]["arm"]!="cf-neighbor",x[1]["teacher"],x[1]["seed"],x[1]["arm"]))
    for study,r in selected:
        target=OUT/"functional"/study/(job_name(r)+".json.gz")
        if target.exists():
            with gzip.open(target,"rt") as f:d=json.load(f)
            assert d["freeze_sha256"]==digest
        else:
            frozen=read(ROOT/"results/pedro-revision"/study/"freeze.json");cfg=frozen["config"]
            cfg=dict(cfg,namespace=cfg["functional_namespace"],final_runs=cfg["functional_runs"],final_sizes=cfg["functional_sizes"])
            key=study,r["teacher"]
            if key not in cache:cache[key]=cases(cfg,"final",0)
            suite=cache[key]
            model=initialize(r["teacher"],"normalized",r["seed"]).eval()
            model.load_state_dict(torch.load(ROOT/r["checkpoint"],map_location="cpu",weights_only=False)["model"])
            rows,seconds=evaluate(model,r["teacher"],suite,batch_size=1024,cohort=32)
            result=summarize_runs(rows,[m["case_id"] for _,m in suite])
            family={m["case_id"]:m["family"] for _,m in suite}
            result["failures_by_family"]=dict(Counter(family[x["case_id"]] for x in rows if not passed_run(x)))
            source=ROOT/"results/pedro-revision"/study/"functional"
            if study=="cf-experiment":source=source/r["teacher"].lower()
            source=source/f"s{r['seed']}-{r['arm']}.json";old=read(source)
            matches={k:result[k]==old[k] for k in result}
            d=dict(study=study,teacher=r["teacher"],seed=r["seed"],arm=r["arm"],checkpoint_sha256=r["sha256"],
                freeze_sha256=digest,source_summary_sha256=sha(source),summary=result,matches=matches,
                technical_errors=sum(x["technical_error"] is not None for x in rows),rows=rows,seconds=seconds)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(gzip.compress(json.dumps(d,separators=(",",":"),sort_keys=True).encode(),mtime=0))
        record={k:d[k] for k in ("study","teacher","seed","arm","checkpoint_sha256","summary","matches","technical_errors","seconds")}
        record.update(path=str(target.relative_to(ROOT)),sha256=sha(target));records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ("matches","summary")}|dict(successes=d["summary"]["successes"],all_summary_fields_match=all(d["matches"].values()))),flush=True)
        if not all(d["matches"].values()) or d["technical_errors"]:raise RuntimeError("Functional difference preserved for diagnosis; no retuning/exclusion")
    assert len(records)==120
    write(OUT/"functional-replay-audit.json",dict(freeze_sha256=digest,evaluations=len(records),executions=sum(r["summary"]["executions"] for r in records),
        all_original_summary_fields_match=True,technical_errors=0,records=records,
        scope="100 original primary models and20 follow-up model/battery combinations; original per-case functional rows were not stored by those scripts. Fresh rows retained here."))


def main():
    os.chdir(ROOT);torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("freeze","inventory","arrays","metrics","infer","functional"))
    stage=p.parse_args().stage
    {"freeze":freeze,"inventory":inventory,"arrays":validate_arrays,"metrics":reanalysis,"infer":infer,"functional":functional}[stage]()


if __name__=="__main__":main()
