#!/usr/bin/env python3
"""Independent evidence checks; optionally verify all runtime cache/checkpoints."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dcl.samplecomplexity.protocol import cases, five_seed_gate, summarize_runs
from dcl.samplecomplexity.runner import verify
from dcl.study.storage import file_hash
from dcl.behavior_statistics import holm_adjust


def read(path): return json.loads(Path(path).read_text())


def check_metadata(rows,suite):
    expected={m["case_id"]:m for _,m in suite}
    for r in rows:
        m=expected[r["case_id"]]
        assert all(r[k]==m[k] for k in ("tape_seed","graph_seed","fingerprint","family","n","edges"))


def main():
    p=argparse.ArgumentParser();p.add_argument("--runtime",action="store_true")
    args=p.parse_args();out=Path("results/sample-complexity");root=Path("data/sample-complexity-v1")
    config=read("configs/sample-complexity.json");design=read(out/"design.json")
    verify(config)
    records=read(out/"training.json");grid=design["grid"]
    expected={(t,a,s,b) for t in config["teachers"] for a in config["architectures"] for s in config["seeds"] for b in grid}
    keys=[tuple(r["identity"][k] for k in ("teacher","architecture","seed","budget")) for r in records]
    assert len(keys)==len(set(keys)) and set(keys)==expected
    prefixes=defaultdict(set);initials=defaultdict(set)
    for r in records:
        i=r["identity"];a=r["accounting"]
        assert 0<a["queries"]<=i["budget"] and a["queries"]==sum(a["by_component"].values())
        assert i["budget"]-a["queries"]<=2
        assert r["parameter_updates"]==config["updates"]==500
        assert a["distinct_local_states"]<=a["examples"]
        prefixes[i["teacher"],i["seed"],i["budget"]].add(a["prefix_sha256"])
        initials[i["teacher"],i["architecture"],i["seed"]].add(r["initial_hash"])
        if args.runtime:
            checkpoint=root/"runs"/i["teacher"].lower()/i["architecture"]/f"s{i['seed']}"/f"b{i['budget']}"/"final.pt"
            assert file_hash(checkpoint)==r["checkpoint_sha256"]
    assert all(len(s)==1 for s in prefixes.values())
    assert all(len(s)==1 for s in initials.values())
    for t in config["teachers"]:
        for arch in config["architectures"]:
            assert len({next(iter(initials[t,arch,s])) for s in config["seeds"]})==5
    acquired=0
    for t in config["teachers"]:
        for s in config["seeds"]:
            manifest=read(out/"data"/t.lower()/f"s{s}.json")
            acquired+=manifest["queries"]
            assert manifest["queries"]==sum(manifest["by_component"].values())
            assert {g["n"] for g in manifest["graphs"].values()}<=set(config["training_sizes"])
            if args.runtime: assert file_hash(root/"datasets"/t.lower()/f"s{s}"/"dataset.npz")==manifest["sha256"]
    exact_executions=0
    for split in ("development","final"):
        for s in config["seeds"]:
            suite=cases(config,split,s);ids=[m["case_id"] for _,m in suite]
            for t in config["teachers"]:
                rows=read(out/"exact"/t.lower()/f"{split}-s{s}.json")
                check_metadata(rows,suite)
                assert summarize_runs(rows,ids)["pass_all"];exact_executions+=len(rows)
    certifications=read(out/"certification.json")
    assert len(certifications)==21
    candidates=read(out/"frozen-candidates.json")["candidates"]
    final_executions=0
    for c in certifications:
        found=next(k for k in candidates if (k["teacher"],k["architecture"])==(c["teacher"],c["architecture"]))
        teacher,arch=c["teacher"],c["architecture"]
        first=next((b for b in grid if sum(r["development"]["pass_all"] for r in records if
                        (r["identity"]["teacher"],r["identity"]["architecture"],r["identity"]["budget"])==(teacher,arch,b))==5),None)
        assert c["candidate_budget"]==found["budget"]==first
        recomputed={}
        if c["final_attempted"]:
            assert read(out/"consumption"/f"{teacher.lower()}-{arch}.json")["complete"]
            for cp in found["checkpoints"]:
                s=cp["seed"];path=out/"final"/teacher.lower()/arch/f"s{s}.json"
                rows=read(path);suite=cases(config,"final",s)
                check_metadata(rows,suite)
                summary=summarize_runs(rows,[m["case_id"] for _,m in suite])
                recomputed[s]=summary;final_executions+=len(rows)
                assert len(rows)==1000 and summary["successes"]==c["per_seed"][str(s)]["successes"]
                assert file_hash(path)==c["per_seed"][str(s)]["rows_sha256"]
                if args.runtime: assert file_hash(cp["path"])==cp["sha256"]
        gate=five_seed_gate(recomputed,config["seeds"])
        assert gate["status"]==c["status"] and gate["seeds_pass"]==c["seeds_pass"]
    controls=read(out/"controls.json")
    assert len(controls)==60
    for r in controls:
        a=r["record"]["accounting"]
        assert 0<a["queries"]<=config["control_budget"]
        if r["kind"]=="interactive":
            assert r["record"]["interactive"]["total_updates"]==500
            assert r["record"]["interactive"]["total_queries"]<=3000
    figs=read(out/"figure-manifest.json")
    for path,sha in figs["files"].items(): assert file_hash(path)==sha
    law=read(out/"law/results.json") if (out/"law/results.json").exists() else None
    if law:
        assert law["planned"]==len(law["rows"])==210
        assert sum(r["reject"] for r in law["rows"])==law["rejections"]
        assert all(r["p"]==1. for r in law["rows"] if not r["testable"])
        assert np.allclose(holm_adjust([r["p"] for r in law["rows"]]),[r["p_holm"] for r in law["rows"]],rtol=0,atol=1e-15)
        for r in law["rows"]:
            for label,field in (("exact","teacher"),(f"{r['architecture']}-s{r['seed']}","model")):
                path=out/"law/banks"/r["teacher"].lower()/f"{r['graph']}-{label}.npz"
                assert file_hash(path)==r[f"{field}_bank_sha256"]
                with np.load(path,allow_pickle=False) as bank:
                    assert len(bank["colors"])==128 and bank["success"].sum()==r[f"valid_{field}"]
    small_results=[]
    if (out/"capacity/certification.json").exists():
        cp=read(out/"capacity/plan.json")
        assert file_hash("src/dcl/samplecomplexity/capacity.py")==cp["source_sha256"]
        small_config=dict(config,namespace=cp["plan"]["namespace"])
        small_records=read(out/"capacity/training.json");assert len(small_records)==105
        small_results=read(out/"capacity/certification.json");assert len(small_results)==3
        for c in small_results:
            first=next((b for b in grid if sum(r["development"]["pass_all"] for r in small_records if
                     (r["identity"]["teacher"],r["identity"]["budget"])==(c["teacher"],b))==5),None)
            assert c["candidate_budget"]==first
            summaries={}
            for s in config["seeds"]:
                suite=cases(small_config,"final",s)
                exact=read(out/"capacity/exact"/c["teacher"].lower()/f"s{s}.json")
                check_metadata(exact,suite);assert summarize_runs(exact,[m["case_id"] for _,m in suite])["pass_all"]
                if c["final_attempted"]:
                    rows=read(out/"capacity/final"/c["teacher"].lower()/f"s{s}.json")
                    check_metadata(rows,suite);summaries[s]=summarize_runs(rows,[m["case_id"] for _,m in suite])
            gate=five_seed_gate(summaries,config["seeds"])
            assert gate["status"]==c["status"] and gate["seeds_pass"]==c["seeds_pass"]
        if args.runtime:
            for r in small_records:assert file_hash(r["checkpoint"])==r["checkpoint_sha256"]
    optimized=[]
    if (out/"optimization-confirmation/results.json").exists():
        op=read(out/"optimization-confirmation/frozen-plan.json")
        assert file_hash("src/dcl/samplecomplexity/optimization.py")==op["source_sha256"]
        op_config=dict(config,namespace=op["plan"]["namespace"])
        optimized=read(out/"optimization-confirmation/results.json")["arms"];assert len(optimized)==6
        for c in optimized:
            summaries={}
            for s in config["seeds"]:
                suite=cases(op_config,"final",s)
                exact=read(out/"optimization-confirmation/exact"/c["teacher"].lower()/f"s{s}.json")
                check_metadata(exact,suite);assert summarize_runs(exact,[m["case_id"] for _,m in suite])["pass_all"]
                rows=read(out/"optimization-confirmation/final"/c["teacher"].lower()/f"u{c['updates']}"/f"s{s}.json")
                check_metadata(rows,suite);summaries[s]=summarize_runs(rows,[m["case_id"] for _,m in suite])
            gate=five_seed_gate(summaries,config["seeds"])
            assert gate["status"]==c["status"] and gate["seeds_pass"]==c["seeds_pass"]
        if args.runtime:
            for cp in op["checkpoints"]: assert file_hash(cp["path"])==cp["sha256"]
    if args.runtime and (out/"runtime-manifest.json").exists():
        for entry in read(out/"runtime-manifest.json")["files"]:
            assert Path(entry["path"]).stat().st_size==entry["bytes"] and file_hash(entry["path"])==entry["sha256"]
    if (out/"full-fidelity.json").exists():
        full=read(out/"full-fidelity.json")
        assert file_hash("src/dcl/samplecomplexity/full_fidelity.py")==full["source_sha256"]
        assert len(full["rows"])==len(records)
        lookup={tuple(r["identity"][k] for k in ("teacher","architecture","seed","budget")):r for r in records}
        for r in full["rows"]:
            key=tuple(r[k] for k in ("teacher","architecture","seed","budget"))
            assert r["checkpoint_sha256"]==lookup[key]["checkpoint_sha256"]
            assert 0<=r["action_correct"]<=r["action_total"] and 0<=r["identical_transitions"]<=r["transitions"]
            assert r["action_fidelity"]==r["action_correct"]/r["action_total"]
            assert r["transition_fidelity"]==r["identical_transitions"]/r["transitions"]
        if args.runtime:
            for b in full["banks"]: assert file_hash(root/"full-fidelity"/b["teacher"].lower()/f"s{b['seed']}"/"teacher-states.npz")==b["sha256"]
    if (out/"structural-failure-witnesses.json").exists():
        witness=read(out/"structural-failure-witnesses.json")
        assert file_hash("src/dcl/samplecomplexity/failure_witnesses.py")==witness["source_sha256"]
    if (out/"evidence-ledger.json").exists():
        for claim in read(out/"evidence-ledger.json"):
            for s in claim["sources"]:assert file_hash(s["path"])==s["sha256"]
    result=dict(passed=True,trainings=len(records),acquired_queries=acquired,
                exact_executions=exact_executions,final_student_executions=final_executions,
                status_counts=dict(Counter(c["status"] for c in certifications)),controls=len(controls),
                law_hypotheses=law["planned"] if law else None,small_grid_trainings=105 if small_results else 0,
                optimization_arms=len(optimized),runtime_verified=args.runtime)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
