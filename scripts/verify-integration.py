#!/usr/bin/env python3
"""Verify exported evidence, or additionally all local/released runtime bytes."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import re
import sys

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
from dcl.behavior_statistics import holm_adjust
from dcl.integration.study import sources
from dcl.multiteacher.programs import named_seed
from dcl.study.storage import file_hash
from dcl.sizedegree.common import verify_frozen

p=argparse.ArgumentParser();p.add_argument("--runtime",action="store_true")
p.add_argument("--skip-visual-check",action="store_true")
args=p.parse_args();torch.set_num_threads(1)
read=lambda path:json.loads(Path(path).read_text())
out=Path("results/integration");config=read("configs/integration-balanced.json")
training=read(out/"multiteacher/training.json");functional=read(out/"multiteacher/functional.json")
law=read(out/"statistics/new-study-global.json")["comparisons"]
assert len(training)==70 and len(functional)==385 and len(law)==350
expected={(t,a,s) for t in config["teachers"] for a in config["architectures"] for s in config["seeds"]}
assert {(r["teacher"],r["architecture"],r["seed"]) for r in training}==expected
assert len({(r["teacher"],r["label"],r["graph"]) for r in functional})==385
assert np.array_equal(holm_adjust([r["p"] for r in law]),[r["p_holm"] for r in law])
assert all(r["reject"]==(r["p_holm"]<=.05) for r in law)
assert all(not r["reject"] and r["p"]==1 for r in law if not r["testable"])
assert sum(r["testable"] for r in law)==227 and sum(r["reject"] for r in law)==6
assert sum(r["samples"] for r in functional)==49280
assert sum(r["successes"] for r in functional if r["architecture"]=="exact")==4480
by_identity={(r["teacher"],r["architecture"],r["seed"],r["graph"]):r for r in functional}
for r in law:
    model=by_identity[r["teacher"],r["architecture"],r["seed"],r["graph"]]
    teacher=by_identity[r["teacher"],"exact",None,r["graph"]]
    assert r["valid_model"]==model["successes"] and r["valid_teacher"]==teacher["successes"]
    assert r["model_bank_sha256"]==model["sha256"] and r["teacher_bank_sha256"]==teacher["sha256"]
    assert r["testable"]==(min(r["valid_model"],r["valid_teacher"])>=4)
design=read(out/"multiteacher/design.json")
assert design["source_hashes"]==sources() and design["updates"]==500
verify_frozen()  # All historical scientific source guards remain intact.
sets={};pool_rows=read(out/"multiteacher/pool-manifests.json")
for r in pool_rows:
    manifest=r["manifest"];fps=[x["fingerprint"] for x in manifest["records"]]
    assert len(fps)==len(set(fps))
    sets[r["teacher"],r["split"]]=set(fps)
for teacher in config["teachers"]:
    assert not sets[teacher,"train"]&sets[teacher,"validation"]
    for split in ("train","validation"):
        assert sets[teacher,split]==sets[config["teachers"][0],split]
graphs=read(out/"multiteacher/graphs.json");graphmap={g["id"]:g for g in graphs}
assert len(graphmap)==5 and len({g["fingerprint"] for g in graphs})==5
assert not {g["fingerprint"] for g in graphs}&set.union(*sets.values())
for g in graphs:
    graph=nx.Graph();graph.add_nodes_from(range(g["n"]));graph.add_edges_from(g["edges"])
    assert [graph.degree(v) for v in range(len(graph))]==g["degrees"]
    assert max(g["degrees"])<=3 and nx.number_of_selfloops(graph)==0
for group in ("files","inputs"):
    for name,digest in read(out/"figure-manifest.json")[group].items():assert file_hash(name)==digest
for claim in read(out/"evidence-ledger.json"):
    if claim["source_sha256"] is not None:assert file_hash(claim["source_file"])==claim["source_sha256"]
build=read(out/"paper-build.json")
assert build["compile_passed"] and not build["undefined_references"] and not build["missing_characters"] and not build["overfull_boxes"]
assert file_hash("paper/ucgnn/main.pdf")==build["pdf_sha256"]
assert file_hash("paper/ucgnn/main.tex")==build["latex_sha256"]
assert file_hash("paper/PAPER_DRAFT.md")==build["markdown_sha256"]
if not args.skip_visual_check:assert build["visual_review_passed"]
assert "{{" not in Path("paper/PAPER_DRAFT.md").read_text()
assert not re.search(r"\\\+",Path("paper/PAPER_DRAFT.md").read_text())
for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)",Path("README.md").read_text()):
    if ":" in target or target.startswith("#"):continue
    assert Path(target.split("#")[0]).exists(),target
runtime={"checked":False}
if args.runtime:
    all_seeds=[]
    for r in functional:
        assert file_hash(r["bank"])==r["sha256"]
        with np.load(r["bank"],allow_pickle=False) as a:
            assert a["colors"].shape==(128,r["n"]) and np.isin(a["status"],[0,1,2]).all()
            wanted=np.array([named_seed(config["protocol"],config["namespace"],"new-independent-bank",r["teacher"],r["label"],r["graph"],j) for j in range(128)],np.uint64)
            assert np.array_equal(a["seeds"],wanted);all_seeds.extend(map(int,a["seeds"]))
            valid=a["status"]==0;assert valid.sum()==r["successes"]
            x=a["colors"][valid];assert ((x>=0)&(x<4)).all()
            assert all((x[:,u]!=x[:,v]).all() for u,v in graphmap[r["graph"]]["edges"])
            assert (a["colors"][a["status"]==2]<0).any(1).all()
            assert (a["failure"][a["status"]==0]=="").all()
    assert len(all_seeds)==len(set(all_seeds))==49280
    frozen=read(out/"multiteacher/frozen-checkpoints.json")["models"]
    for r in frozen:assert file_hash(r["checkpoint"])==r["sha256"]
    for r in training:assert file_hash(r["checkpoint"])==r["checkpoint_sha256"]
    for r in pool_rows:
        m=r["manifest"];base=Path("data/integration-balanced/pools")/r["teacher"].lower()/r["split"]/m["attempt"]
        for name,digest in m["hashes"].items():assert file_hash(base/name)==digest
    runtime=dict(checked=True,raw_banks=385,unique_execution_seeds=49280,final_checkpoints=70,teacher_pools=14)
record=dict(passed=True,historical_source_guards=True,models=70,attempts=49280,new_law_testable=227,
            new_law_rejections=6,train_validation_test_overlap=False,paper_pages=build["pages"],runtime=runtime)
(out/("runtime-verification.json" if args.runtime else "quick-verification.json")).write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record),flush=True)
