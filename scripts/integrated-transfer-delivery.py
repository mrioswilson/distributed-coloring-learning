#!/usr/bin/env python3
"""Restore and audit the240 original transfer cells; no model inference.

Authenticate original bytes, reconstruct case identities, recompute saved
summaries and check counter identities. Original rows do not contain full
color vectors or intermediate trajectories, which are not reconstructed here.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/martin-transfer-rows-20261101"
EXPERIMENT=ROOT/"results/pedro-revision/transfer-experiment"
PACKAGE="pedrorev-transfer-rows"
PIN="d650d45adfbaff8c992b1a24cbcf602abb7acc09b0995557e9f811ccb4d8e5a1"
PREFIX="results/pedro-revision/transfer-experiment/"
COUNTERS=("bad_proposals","unsafe_accepts","contained","missed","participations","decisions")


def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def read(p):return json.loads(Path(p).read_text())


def immutable(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert read(p)==obj,(str(p),"Existing different record preserved")
    else:p.write_text(json.dumps(obj,indent=2)+"\n")


def manifest():
    rows={}
    for line in (EXPERIMENT/"ROWS.sha256").read_text().splitlines():
        h,name=line.split(maxsplit=1);name=name.lstrip("*")
        assert len(h)==64 and name not in rows
        rows[name]=h
    assert len(rows)==240
    return rows


def download():
    # Reuse the original downloader's safe temporary-file recovery, without
    # changing its three-package pins or its historical receipt.
    spec=importlib.util.spec_from_file_location("original_release_downloader",ROOT/"scripts/integrated-release-download.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT
    rows=[]
    for suffix in (".README.md",".SHA256SUMS",".tar.gz"):
        p=m.fetch(PACKAGE+suffix);h=sha(p)
        if suffix==".tar.gz":assert h==PIN,(p,h)
        rows.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=h,url=m.BASE+p.name))
    for line in (OUT/"downloads"/(PACKAGE+".SHA256SUMS")).read_text().splitlines():
        h,name=line.split(maxsplit=1);name=name.lstrip("*")
        assert Path(name).name==name and sha(OUT/"downloads"/name)==h,name
    immutable(OUT/"download-receipt.json",dict(scope="Fourth archive verified against the user supplied SHA256",files=rows))


def restore():
    archive=OUT/"downloads"/(PACKAGE+".tar.gz");assert sha(archive)==PIN
    expected=manifest();seen=set();records=[];meta={}
    with tarfile.open(archive,mode="r|gz") as tf:
        for member in tf:
            name=member.name;p=Path(name)
            assert member.isfile() and not p.is_absolute() and ".." not in p.parts and name not in seen,name
            seen.add(name);data=tf.extractfile(member).read();h=hashlib.sha256(data).hexdigest()
            scientific=name.startswith(PREFIX+"rows/")
            if scientific:
                relative=name[len(PREFIX):];assert h==expected[relative],name
                dest=ROOT/name
            else:
                assert name.startswith(PREFIX+"pedrorev-transfer-metadata/"),name
                dest=OUT/"package-metadata"/name;meta[p.name]=h
            if dest.exists():assert sha(dest)==h,(str(dest),"Existing different file preserved")
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            records.append(dict(member=name,path=str(dest.relative_to(ROOT)),sha256=h,bytes=len(data),scientific_payload=scientific))
    assert {r["member"][len(PREFIX):] for r in records if r["scientific_payload"]}==set(expected)
    assert len(records)==245 and sum(r["scientific_payload"] for r in records)==240
    for name in ("ROWS.sha256","design.json","summary.json"):assert meta[name]==sha(EXPERIMENT/name),name
    assert meta[PACKAGE+".README.md"]==sha(OUT/"downloads"/(PACKAGE+".README.md"))
    immutable(OUT/"member-index.json",dict(archive_sha256=PIN,files=records,
        original_240_hashes_match=True,original_metadata_copies_identical=True,existing_files_preserved=True))
    print(json.dumps(dict(archive_members=len(records),original_raw_files=240,all_original_hashes_match=True)),flush=True)


def audit():
    import numpy as np
    from dcl.degree.graphs import make_case
    from dcl.multiteacher.programs import named_seed
    index=read(OUT/"member-index.json");expected=manifest()
    for row in index["files"]:assert sha(ROOT/row["path"])==row["sha256"],row["path"]
    design=read(EXPERIMENT/"design.json");published=read(EXPERIMENT/"summary.json")["rows"]
    frozen=read(ROOT/"results/pedro-revision/cf-experiment/freeze.json")
    checkpoints={(r["teacher"],r["seed"],r["arm"]):r for r in frozen["checkpoints"] if r["update"]==4000}
    conditions={c["id"]:c for c in design["conditions"]}
    bykey={(r["teacher"],r["seed"],r["arm"],r["condition"]):r for r in published}
    assert len(bykey)==len(published)==240 and len(design["checkpoints"])==60 and len(conditions)==4
    cross={(t,s,a,c) for t,s,a,_ in design["checkpoints"] for c in conditions}
    assert set(bykey)==cross
    source=ROOT/"scripts/pedrorev-transfer-experiment.py";assert sha(source)==design["script_sha256"]
    recovered=read(ROOT/"results/martin-artifacts-20261031/source-recovery/receipt.json")
    assert sha(ROOT/recovered["path"])==design["transfer_source_sha256"]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (EXPERIMENT/"ROWS.sha256",EXPERIMENT/"design.json",EXPERIMENT/"summary.json",
        ROOT/"results/pedro-revision/cf-experiment/freeze.json",OUT/"member-index.json",source,ROOT/recovered["path"])}
    for t,s,a,h in design["checkpoints"]:
        cp=checkpoints[t,s,a];assert cp["sha256"]==h and sha(ROOT/cp["checkpoint"])==h
        inputs[cp["checkpoint"]]=h
    for name,h in expected.items():inputs[str((EXPERIMENT/name).relative_to(ROOT))]=h
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"src/dcl").rglob("*.py"))}
    sources[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    immutable(OUT/"audit-freeze.json",dict(scope=__doc__,inputs=inputs,sources=sources,
        new_training_runs=0,new_neural_rollouts=0))
    digest=sha(OUT/"audit-freeze.json")
    # Regenerate the original case protocol, without executing a teacher or NN.
    cases={};casecache={}
    for cid,c in conditions.items():
        key=c["n"],c["delta"],tuple(c["families"]),c["runs"]
        if key not in casecache:
            rows=[]
            for i in range(c["runs"]):
                pick=int(np.random.default_rng(named_seed(design["namespace"],"family",c["n"],c["delta"],i)).integers(len(c["families"])))
                _,m=make_case(design["namespace"],c["n"],c["delta"],c["families"][pick],i,0)
                rows.append({k:m[k] for k in ("case_id","family","n","delta")})
            casecache[key]=rows
        cases[cid]=casecache[key]
        h=hashlib.sha256("".join(m["case_id"] for m in cases[cid]).encode()).hexdigest()
        for teacher in {t for t,_,_,_ in design["checkpoints"]}:
            exact=read(EXPERIMENT/"exact"/(teacher.lower()+"-"+cid+".json"))
            assert exact["case_ids_sha"]==h and exact["executions"]==len(cases[cid])
    totals=Counter();by_teacher={};records=[]
    for key,old in sorted(bykey.items()):
        teacher,seed,arm,cid=key
        name=f"rows/{teacher.lower()}/s{seed}-{arm}-{cid}.json";path=EXPERIMENT/name
        assert sha(path)==expected[name]
        data=read(path);rows=data["rows"];summary=data["summary"]
        assert summary==old and len(rows)==len({r["case_id"] for r in rows})==conditions[cid]["runs"]==1000
        for row,m in zip(rows,cases[cid]):
            assert all(row[k]==m[k] for k in m),(name,row["case_id"])
            assert all(isinstance(row[k],int) and row[k]>=0 for k in (*COUNTERS,"conflicts","incomplete","steps"))
            assert row["bad_proposals"]==row["contained"]+row["unsafe_accepts"]
            assert row["bad_proposals"]<=row["participations"]<=row["decisions"]
            assert row["missed"]<=row["participations"]-row["bad_proposals"]
            assert (row["first_unsafe"] is None)==(row["unsafe_accepts"]==0)
            if row["success"]:assert row["failure"] is None and row["failure_kind"] is None and row["conflicts"]==row["incomplete"]==0
            else:assert row["failure_kind"]==("safety" if row["unsafe_accepts"] else "progress")
        recomputed=dict(executions=len(rows),successes=sum(r["success"] for r in rows),
            safety_failures=sum(r["failure_kind"]=="safety" for r in rows),progress_failures=sum(r["failure_kind"]=="progress" for r in rows),
            cutoffs=sum(r["failure"]=="cutoff" for r in rows),**{k:sum(r[k] for r in rows) for k in COUNTERS})
        assert all(summary[k]==v for k,v in recomputed.items()),name
        assert summary["checkpoint_sha256"]==checkpoints[teacher,seed,arm]["sha256"]
        totals.update(recomputed);by_teacher.setdefault(teacher,Counter()).update(recomputed)
        records.append(dict(path=str(path.relative_to(ROOT)),sha256=expected[name],teacher=teacher,seed=seed,arm=arm,condition=cid,**recomputed))
    assert len(records)==240 and totals["executions"]==240000
    assert all(sha(ROOT/p)==h for p,h in {**inputs,**sources}.items())
    result=dict(scope=__doc__,freeze_sha256=digest,archive_sha256=PIN,cells=len(records),checkpoints=60,conditions=4,
        totals=dict(totals),by_teacher={t:dict(v) for t,v in by_teacher.items()},records=records,
        all_original_hashes_match=True,all_summaries_match=True,complete_design_without_duplicates=True,
        regenerated_case_instances=sum(len(v) for v in casecache.values()),case_protocol_metadata_matches=True,
        exact_control_case_id_hashes_match=True,original_metadata_unchanged=True,
        new_training_runs=0,new_neural_rollouts=0,full_color_vectors_in_original_rows=False,full_trajectories_reconstructed=False)
    immutable(OUT/"raw-transfer-audit.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("records","scope")}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("download","restore","audit","all"),nargs="?",default="all")
    stage=p.parse_args().stage
    if stage in ("download","all"):download()
    if stage in ("restore","all"):restore()
    if stage in ("audit","all"):audit()


if __name__=="__main__":main()
