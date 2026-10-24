#!/usr/bin/env python3
"""Audit complete final gates, hashes, all stored intervention labels and sources."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from dcl.multiteacher.programs import LINIAL,DETECT,MAX_FIELD
from dcl.multicf.data import load
from dcl.multicf.oracles import confirmation,stage_targets
from dcl.multicf.protocol import summarize_runs,five_seed_gate
from dcl.study.storage import atomic_json,file_hash

OUT=Path("results/multiteacher-counterfactual")


if __name__=="__main__":
    torch.set_num_threads(1);cohorts=[];unique_checkpoints={};errors=[];labels=Counter()
    for path in sorted((OUT/"confirmation").glob("*/result.json")):
        result=json.loads(path.read_text());frozen=json.loads((path.parent/"freeze.json").read_text())
        selected=frozen["selected"];config=frozen["config"];summaries={}
        assert len(selected["checkpoints"])==5 and len({c["seed"] for c in selected["checkpoints"]})==5
        for cp in selected["checkpoints"]:
            assert file_hash(cp["path"])==cp["sha256"],cp["path"]
            unique_checkpoints[cp["path"]]=cp
            battery=json.loads((path.parent/"batteries"/f"s{cp['seed']}.json").read_text())
            rawpath=path.parent/"final"/f"s{cp['seed']}.json"
            raw=json.loads(rawpath.read_text())
            assert len(battery)==len(raw)==1000
            assert file_hash(rawpath)==result["per_seed"][str(cp["seed"])]["rows_sha256"]
            summaries[cp["seed"]]=summarize_runs(raw,[m["case_id"] for m in battery])
            assert summaries[cp["seed"]]["successes"]==result["per_seed"][str(cp["seed"])]["successes"]
            exact=json.loads((path.parent/"exact"/f"s{cp['seed']}.json").read_text())
            assert summarize_runs(exact,[m["case_id"] for m in battery])["pass_all"]
            expected={m["case_id"]:m for m in battery}
            for row in raw:
                m=expected[row["case_id"]]
                assert row["tape_seed"]==m["tape_seed"] and row["edges"]==m["edges"]
        gate=five_seed_gate(summaries,config["seeds"])
        assert gate["status"]==result["status"] and gate["seeds_pass"]==result["seeds_pass"]
        assert json.loads((path.parent/"consumed.json").read_text())["complete"] is True
        cohorts.append(dict(package=path.parent.name,namespace=config["namespace"],status=result["status"],
            seeds_pass=gate["seeds_pass"],executions=5000,successes=sum(s["successes"] for s in summaries.values())))
    # Exact teacher and autonomous-model source files have not been changed.
    inference_paths=["src/dcl/multiteacher/programs.py","src/dcl/multiteacher/model.py",
        "src/dcl/multiteacher_e1/model.py","src/dcl/samplecomplexity/model.py",
        "src/dcl/samplecomplexity/evaluation.py","src/dcl/integration/models.py",
        "src/dcl/paperstudy/models.py","src/dcl/universal/model.py"]
    unchanged=[]
    for p in inference_paths:
        base=subprocess.check_output(["git","show","61f39760:"+p])
        assert hashlib.sha256(base).hexdigest()==file_hash(p),p
        unchanged.append(dict(path=p,sha256=file_hash(p)))
    # Verify checkpoint training source receipts; compare each recorded file,
    # including modules that were actually present at training time.
    for cp in unique_checkpoints.values():
        payload=torch.load(cp["path"],map_location="cpu",weights_only=False)
        for p,h in payload["sources"].items():assert file_hash(p)==h,p
        assert payload["identity"]["seed"]==cp["seed"]
        assert "optimizer" in payload and "batch_rng" in payload and "torch_rng" in payload
    banks=[]
    manifests=list((OUT/"datasets").rglob("manifest.json"))+list((OUT/"supervision-curves/datasets").rglob("manifest.json"))+list((OUT/"information/datasets").rglob("manifest.json"))
    for mp in sorted(manifests):
        packed,m=load(mp.parent);teacher=m["identity"]["teacher"]
        if "witness_sha256" in m:assert file_hash(mp.parent/"witnesses.json")==m["witness_sha256"]
        native=packed["native"]
        for phase in sorted(np.unique(native[:,3])):
            use=np.flatnonzero(native[:,3]==phase)
            arrays=tuple(packed[f"a{i}"][use] for i in range(4))
            if phase==LINIAL:
                eq=arrays[2][:,0,:,:MAX_FIELD+2][:,:,2:]
                q=(arrays[0][:,0,15]*MAX_FIELD).round().astype(int)
                occupied=eq>.5
                valid=arrays[3][:,0,:,None]
                free=~(occupied&valid).any(1)
                free &= np.arange(MAX_FIELD)[None,:]<q[:,None]
                point=free.argmax(1)
                assert np.array_equal(point,native[use,1]//MAX_FIELD)
                labels["native_linial_point"]+=len(use)
            elif phase==DETECT:
                x,y,r,mask=arrays
                truth=(mask[:,0]&(r[:,0,:,0]>.5)&(r[:,0,:,1]>.5)&(y[:,0,:,5]>x[:,0,5,None])).any(1)
                assert np.array_equal(truth,native[use,2]>0)
                labels["native_detection_mark"]+=len(use)
            else:
                target=stage_targets(arrays,native[use],teacher)
                assert np.array_equal(target.participate,packed["p"][use])
                assert np.array_equal(target.candidate,packed["c"][use])
                assert np.array_equal(confirmation(arrays,teacher,packed["p"][use],packed["c"][use])>0,native[use,0]>0)
                for v in range(4):
                    assert np.array_equal(confirmation(arrays,teacher,packed["cf_p"][use,v],packed["cf_c"][use,v]),packed["cf_y"][use,v])
                labels["native_color_components"]+=3*len(use);labels["counterfactual_confirmation"]+=4*len(use)
        banks.append(dict(manifest=str(mp),sha256=m["sha256"],examples=len(native)))
    own_caches=[]
    for cp in sorted((OUT/"own-proposal-pilot").rglob("u001500.pt")):
        payload=torch.load(cp,map_location="cpu",weights_only=False);identity=payload["identity"]
        teacher,seed=identity["teacher"],identity["seed"]
        budget=30000 if teacher=="COLOR_TRIALS" else 3000
        packed,manifest=load(OUT/"datasets"/teacher.lower()/f"s{seed}"/f"b{budget}-coverage1-alternate0")
        cache=payload["query_cache"]
        for phase in sorted(np.unique(packed["native"][:,3])):
            if phase in (LINIAL,DETECT):continue
            items=[(k,v) for k,v in cache.items() if packed["native"][k[0],3]==phase]
            if not items:continue
            indices=np.array([k[0] for k,_ in items])
            p=np.array([k[1] for k,_ in items],bool);c=np.array([k[2] for k,_ in items],int)
            truth=confirmation(tuple(packed[f"a{i}"][indices] for i in range(4)),teacher,p,c)
            assert np.array_equal(truth,np.array([v for _,v in items]))
        assert len(cache)==payload["record"]["own_confirmation_labels"]
        own_caches.append(dict(teacher=teacher,queries=len(cache),checkpoint_sha256=file_hash(cp)))
    historical=json.loads((OUT/"audit/m6-original-evidence.json").read_text())
    for r in historical["artifacts"]:
        original=Path(r["path"])
        archived=OUT/"historical-recovery/m6-originals/data/results"/original.name
        assert r["match"] and file_hash(original if original.exists() else archived)==r["expected"]
    # Historical result directories remain byte-for-byte tracked originals.
    historical_diff=subprocess.check_output(["git","diff","61f39760","--","results/sample-complexity","results/m6study","results/multiteacher","results/multiteacher-e1"],text=True)
    assert not historical_diff
    receipt=dict(passed=True,cohorts=cohorts,final_executions=sum(r["executions"] for r in cohorts),
        final_successes=sum(r["successes"] for r in cohorts),unique_final_checkpoints=len(unique_checkpoints),
        verified_banks=banks,verified_stored_labels=dict(labels),verified_own_query_caches=own_caches,label_inconsistencies=0,
        unchanged_teacher_and_inference_sources=unchanged,historical_artifacts_authenticated=21,
        historical_result_diff_empty=True,script_sha256=file_hash(__file__),
        note="Final rows are policy/battery observations; repeated checkpoints and shared architecture heads are not independent training replicas. Prefix-bank verification rechecks stored labels; native-state replay witnesses and differential tests provide separate causal consistency checks.")
    atomic_json(OUT/"verification.json",receipt)
    print({k:receipt[k] for k in ("passed","final_executions","final_successes","unique_final_checkpoints","label_inconsistencies")})
