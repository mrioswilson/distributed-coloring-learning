#!/usr/bin/env python3
"""Verify new evidence, pairing, disjoint types, label contracts and PDF receipt."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch
from dcl.paperrevision.experiment import sources, load_npz
from dcl.paperrevision.keys import held_out
from dcl.multicf.oracles import confirmation
from dcl.study.storage import file_hash, atomic_json

OUT=Path("results/paper-revision-20261030")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--runtime",type=Path,default=Path("data/paper-revision-20261030"));args=parser.parse_args()
    config=json.loads(Path("configs/paper-revision-20261030.json").read_text())
    freeze=json.loads((OUT/"weights-frozen.json").read_text())
    protocol=json.loads((OUT/"protocol-frozen.json").read_text())
    assert freeze["sources"]==sources()==protocol["sources"]
    assert config==protocol["config"] and len(freeze["weights"])==40
    assert file_hash("docs/PAPER_REVISION_PROTOCOL.md")==protocol["protocol_sha256"]
    assert datetime.fromisoformat(protocol["created_utc"])<datetime.fromisoformat(freeze["created_utc"])
    pairs=[];total_targets=0;total_diagnostics=0;total_rollouts=0
    for teacher in config["teachers"]:
        for seed in config["seeds"]:
            root=args.runtime;data=root/"datasets"/teacher.lower()/f"s{seed}"
            test=root/"test"/teacher.lower()/f"s{seed}"
            manifests=[json.loads((p/"manifest.json").read_text()) for p in (data,test)]
            for p,m in zip((data,test),manifests):
                for name,digest in m["files"].items():assert file_hash(p/name)==digest
            assert manifests[0]["optimization_targets_per_model"]==6600
            assert sum(v for k,v in manifests[0]["accounting"].items() if k.startswith("base_target_"))==3000
            assert sum(manifests[0]["phase_counts"].values())==3600
            assert datetime.fromisoformat(manifests[1]["created_utc"])>datetime.fromisoformat(freeze["created_utc"])
            trainkeys=set(json.loads((data/"type-keys.json").read_text()));testkeys=set(json.loads((test/"type-keys.json").read_text()))
            assert not (trainkeys&testkeys)
            assert all(not held_out(json.loads(k)) for k in trainkeys)
            assert all(held_out(json.loads(k)) for k in testkeys)
            for directory in (data,test):
                aux=load_npz(directory/"aux.npz")
                for phase in np.unique(aux["native"][:,3]):
                    use=np.flatnonzero(aux["native"][:,3]==phase);arrays=tuple(aux[f"a{j}"][use] for j in range(4))
                    for prefix,label in (("","native_y"),("cf_","cf_y")):
                        assert np.array_equal(confirmation(arrays,teacher,aux[prefix+"p"][use],aux[prefix+"c"][use]),aux[label][use])
            records=[];states=[]
            for arm in ("native","cf"):
                directory=root/"models"/teacher.lower()/f"s{seed}"/arm
                rec=json.loads((directory/"record.json").read_text());records.append(rec)
                original=next(r for r in freeze["weights"] if (r["teacher"],r["seed"],r["arm"])==(teacher,seed,arm))
                assert file_hash(directory/"final.pt")==original["sha256"]==rec["checkpoint_sha256"]
                assert rec["updates"]==1500 and rec["parameters"]==19396 and rec["sources"]==freeze["sources"]
                states.append(torch.load(directory/"final.pt",weights_only=False)["model"])
                row=json.loads((OUT/"prospective"/teacher.lower()/f"s{seed}"/(arm+"-test.json")).read_text())
                assert len(row["rows"])==1000 and row["rollout_summary"]["complete"]
                assert row["diagnostic"]["train_test_type_intersection"]==0
                assert row["diagnostic"]["tp"]+row["diagnostic"]["fn"]>0
                assert row["diagnostic"]["tn"]+row["diagnostic"]["fp"]>0
                assert all(r["technical_error"] is None for r in row["rows"])
                total_targets+=6600;total_diagnostics+=row["diagnostic"]["n"];total_rollouts+=len(row["rows"])
            assert records[0]["initial_hash"]==records[1]["initial_hash"]
            assert records[0]["visits_sha256"]==records[1]["visits_sha256"]
            second=("core.message2.","core.state2.","core.commit_head.","context_message2.","context_state2.")
            assert all(torch.equal(states[0][k],states[1][k]) for k in states[0] if not k.startswith(second))
            pairs.append(dict(teacher=teacher,seed=seed,same_initialization_visits_first_stage=True,train_test_type_intersection=0))
    hist=json.loads((OUT/"be-reanalysis.json").read_text());assert hist["count_reproduction_all_match"]
    rescues=json.loads((OUT/"component-rescues.json").read_text())["rows"]
    assert len(rescues)==6 and all(r["preintervention_matches"] for r in rescues)
    assert all(r["baseline_terminal_matches"] for r in rescues if r["mode"]=="baseline")
    for r in rescues:
        if r["mode"]=="candidate_once":assert r["intervention"]["only_one_candidate_changed_with_identical_first_stage"]
    build=json.loads((OUT/"paper-build.json").read_text())
    assert file_hash("paper/ucgnn/main.pdf")==build["pdf_sha256"] and not build["unresolved_references"]
    assert Path("paper/ucgnn/main.tex").read_text().startswith("% Authoritative closure source")
    for path,digest in build["source_sha256"].items():assert file_hash(path)==digest
    result=dict(verified=True,models=40,pairs=pairs,model_access_target_count=total_targets,
                continuation_predictions=total_diagnostics,unique_paired_query_occurrences=total_diagnostics//2,
                autonomous_runs=total_rollouts,source_freeze_unchanged=True,all_auxiliary_labels_rechecked=True,
                historical_numerators_reproduced=True,pdf_sha256=build["pdf_sha256"],
                independent_scientific_review="Pending; these checks are not an independent review.")
    atomic_json(OUT/"verification.json",result);print(json.dumps({k:v for k,v in result.items() if k!="pairs"},indent=2))


if __name__=="__main__":main()
