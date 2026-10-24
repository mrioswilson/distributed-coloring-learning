#!/usr/bin/env python3
"""Read-only historical/source/release inventory; preserve fresh receipts."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from dcl.study.storage import atomic_json, file_hash

OUT=Path("results/multiteacher-counterfactual/audit")
HIST=Path("/workspaces/AlgoLearning/distributed-coloring-learning")


def inventory():
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads(Path("docs/integration/PEDRO_SOURCE_MANIFEST.json").read_text())
    needed=source["availability"]["needed_checkpoints"]
    wanted={r["sha256"] for r in needed}
    seen={}; files=[]; tick=time.perf_counter()
    roots=[Path("/workspaces/AlgoLearning")]
    if Path("/workspace/distributed-coloring-learning").exists():
        roots.append(Path("/workspace/distributed-coloring-learning"))
    for root in roots:
        for parent,dirs,names in os.walk(root):
            dirs[:]=[d for d in dirs if d not in (".git",".venv","node_modules","__pycache__")]
            for name in names:
                if not name.endswith((".pt",".pth",".ckpt")):
                    continue
                path=Path(parent)/name
                st=path.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
                if key not in seen:
                    seen[key]=file_hash(path)
                files.append(dict(path=str(path),sha256=seen[key],bytes=st.st_size))
    atomic_json(OUT/"local-checkpoints.json",dict(roots=list(map(str,roots)),files=files,
        unique_inodes=len(seen),seconds=time.perf_counter()-tick))
    matches=[r for r in files if r["sha256"] in wanted]
    releases=json.loads(subprocess.check_output(["gh","api","repos/mrioswilson/distributed-coloring-learning/releases"]))
    remote=[]
    for rel in releases:
        assets=[]
        for asset in rel["assets"]:
            entry={k:asset.get(k) for k in ("name","size","digest","browser_download_url")}
            assets.append(entry)
            if "manifest" in asset["name"].lower() and asset["size"]<3_000_000:
                directory=OUT/"release-manifests"/rel["tag_name"]
                directory.mkdir(parents=True,exist_ok=True)
                target=directory/asset["name"]
                if not target.exists():
                    subprocess.run(["gh","release","download",rel["tag_name"],"--repo",
                        "mrioswilson/distributed-coloring-learning","--pattern",asset["name"],"--dir",str(directory)],check=True)
                entry["manifest_sha256"]=file_hash(target)
                entry["wanted_hash_occurrences"]=[h for h in wanted if h in target.read_text()]
        remote.append(dict(tag=rel["tag_name"],published_at=rel["published_at"],assets=assets))
    atomic_json(OUT/"release-inventory.json",remote)
    verified=[]
    for path,expected in source["source_hashes"].items():
        historical=subprocess.check_output(["git","show",source["upstream_commit"]+":"+path])
        verified.append(dict(path=path,expected=expected,historical_sha256=hashlib.sha256(historical).hexdigest(),
            current_sha256=file_hash(path),historical_match=hashlib.sha256(historical).hexdigest()==expected))
    atomic_json(OUT/"pedro-availability.json",dict(needed_checkpoints=len(needed),
        local_files_inspected=len(files),local_matches=matches,
        missing_original_directory=not Path("/workspace/distributed-coloring-learning/data").exists(),
        source_verification=verified,
        status="Original weights unavailable in inspected local roots and published release manifests" if not matches else "Some exact original weights recovered",
        limitation="A manifest search is not a proof of absence from every remote computer or unpublished archive. No unavailable policy is attributed a new reproduction."))
    original=[]
    for line in Path("docs/m6-historical-artifacts.sha256").read_text().splitlines():
        expected,path=line.split(maxsplit=1);p=HIST/path.strip()
        original.append(dict(path=str(p),expected=expected,exists=p.exists(),
            actual=file_hash(p) if p.exists() else None,match=p.exists() and file_hash(p)==expected))
    rows=list(csv.DictReader((HIST/"data/results/m6_overnight_three_way_seed10001.csv").open()))
    totals={name:sum(int(r[f"{name}_success"]) for r in rows) for name in ("M4B","CONTROL","COUNTERFACTUAL")}
    cps={}
    for name,filename in [("M4B","m4b_seed10001_n40.pt"),("CONTROL","m6b_ownonly_seed10001_n40.pt"),
                          ("COUNTERFACTUAL","m6b_counterfactual_seed10001_n40.pt")]:
        p=HIST/"data/results"/filename
        cp=torch.load(p,map_location="cpu",weights_only=False)
        metadata={k:v for k,v in cp.items() if k not in ("model_state_dict","optimizer_state_dict")}
        cps[name]=dict(path=str(p),sha256=file_hash(p),metadata=metadata)
    atomic_json(OUT/"m6-original-evidence.json",dict(artifacts=original,rows=len(rows),
        unique_graph_indices=len({r["graph_i"] for r in rows}),successes=totals,checkpoints=cps,
        interpretation="One training seed (10001), 500 paired N10240 cubic graph/tape cases. CONTROL is own-proposal supervision, not extra-oracle data or a five-seed cohort."))
    print(json.dumps(dict(m6_rows=len(rows),m6_successes=totals,pedro_expected=len(needed),
        local_checkpoint_files=len(files),pedro_matches=len(matches),m6_historical_hash_matches=sum(r["match"] for r in original))),flush=True)


def replay_m6(count=10):
    sys.path.insert(0,str(Path("experiments").resolve()))
    from compare_m4_m6 import load_model
    from evaluate_m4_seed import run_m4
    from dcl.graphs import connected_random_regular_graph
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    originals=list(csv.DictReader((HIST/"data/results/m6_overnight_three_way_seed10001.csv").open()))
    paths={"M4B":"m4b_seed10001_n40.pt","CONTROL":"m6b_ownonly_seed10001_n40.pt",
           "COUNTERFACTUAL":"m6b_counterfactual_seed10001_n40.pt"}
    models={name:load_model(HIST/"data/results"/path) for name,path in paths.items()}
    rows=[];tick=time.perf_counter()
    for i in range(count):
        n=10240;graph_seed=50_000_000+n*1000+i;tape_seed=60_000_000+n*1000+i
        graph=connected_random_regular_graph(n=n,degree=3,seed=graph_seed)
        for name,model in models.items():
            r=run_m4(model,graph,tape_seed)
            bad=sum(r["colors"][u]>=0 and r["colors"][u]==r["colors"][v] for u,v in graph.edges())
            actual=dict(success=int(r["success"]),finished=int(r["finished"]),rounds=int(r["rounds"]),bad_edges=int(bad))
            expected={k:int(originals[i][f"{name}_{k}"]) for k in actual}
            rows.append(dict(graph_i=i,policy=name,actual=actual,expected=expected,match=actual==expected))
    atomic_json(OUT/"m6-original-replay.json",dict(count=count,rows=rows,all_match=all(r["match"] for r in rows),
        seconds=time.perf_counter()-tick,device="cuda" if torch.cuda.is_available() else "cpu",
        scope="Replay of first specified historical cases using authenticated original policies; not a fresh independent certification"))
    print(json.dumps(dict(replayed=len(rows),all_match=all(r["match"] for r in rows))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["inventory","replay-m6"])
    parser.add_argument("--count",type=int,default=10);args=parser.parse_args()
    inventory() if args.stage=="inventory" else replay_m6(args.count)
