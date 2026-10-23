#!/usr/bin/env python3
"""Independent evidence checks; optionally verify all runtime cache/checkpoints."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dcl.samplecomplexity.protocol import cases, five_seed_gate, summarize_runs
from dcl.samplecomplexity.runner import verify
from dcl.study.storage import file_hash


def read(path): return json.loads(Path(path).read_text())


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
    result=dict(passed=True,trainings=len(records),acquired_queries=acquired,
                exact_executions=exact_executions,final_student_executions=final_executions,
                status_counts=dict(Counter(c["status"] for c in certifications)),controls=len(controls),
                law_hypotheses=law["planned"] if law else None,runtime_verified=args.runtime)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
