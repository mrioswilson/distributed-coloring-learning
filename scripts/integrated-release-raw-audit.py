#!/usr/bin/env python3
"""Reconstruct terminal correctness from all810000 restored historical rows.

No model inference. Stored intermediate invariant flags are authenticated,
not independently reconstructed from absent full execution traces.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np

from dcl.degree.graphs import graph_for
from dcl.study.graphs import fingerprint

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/martin-artifacts-20261031"


def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    start=time.monotonic()
    original=ROOT/"results/final-closure/generalization-cells.csv"
    incoming=OUT/"package-metadata/confirmatory-rows/results/final-closure/generalization-cells.csv"
    assert incoming.read_bytes().replace(b"\r\n",b"\n")==original.read_bytes().replace(b"\r\n",b"\n")
    cells=[r for r in csv.DictReader(incoming.open()) if r["stage"]=="confirmatory"]
    assert len(cells)==810
    freeze_path=OUT/"raw-audit-freeze.json"
    inputs={str(p.relative_to(ROOT)):sha(p) for p in (original,incoming,Path(__file__),OUT/"member-index.json",ROOT/"docs/MARTIN_ARTIFACTS_REPLAY_PROTOCOL.md")}
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"src/dcl").rglob("*.py"))}
    frozen=dict(inputs=inputs,sources=sources,scope=__doc__)
    if freeze_path.exists():assert json.loads(freeze_path.read_text())==frozen
    else:freeze_path.write_text(json.dumps(frozen,indent=2)+"\n")
    index=json.loads((ROOT/"results/multiteacher-degree-generalization/confirmatory/index.json").read_text())
    by_key={(c["checkpoint_id"],c["condition"]["id"]):c for c in index}
    graphs={};tapes=set();cases=set();unique_graphs=set()
    totals=Counter();primary=Counter();failures=Counter();summaries=[]
    for number,cell in enumerate(cells):
        path=ROOT/cell["rows_path"]
        assert sha(path)==cell["rows_sha256"],str(path)
        indexed=by_key[cell["checkpoint"],cell["condition"]]
        assert indexed["rows_sha256"]==cell["rows_sha256"] and indexed["sha256"]==cell["sha256"]
        rows=json.loads(path.read_text())
        assert len(rows)==int(cell["executions"])==1000
        assert len({r["case_id"] for r in rows})==len(rows)
        counts=Counter()
        for row in rows:
            assert row["n"]==int(cell["n"]) and row["delta"]==row["actual_delta"]==int(cell["delta"])
            assert row["palette"]==cell["palette"] and row["teacher"]==cell["teacher"]
            key=row["family"],row["n"],row["delta"],row["graph_seed"]
            if key not in graphs:
                g=graph_for(*key)
                degrees=np.array([g.degree(v) for v in range(len(g))])
                edges=np.array(list(g.edges()),dtype=int).reshape(-1,2).T
                assert len(g)==row["n"] and max(degrees)==row["delta"] and not nx.number_of_selfloops(g)
                graphs[key]=(degrees,edges,fingerprint(g));graph_for.cache_clear()
            degrees,(a,b),fp=graphs[key]
            assert fp==row["fingerprint"]
            color=np.array(row["colors"]);assert len(color)==row["n"]
            bound=degrees+1 if row["palette"]=="local" else row["delta"]+1
            conflicts=int(((color[a]>=0)&(color[a]==color[b])).sum())
            violation=int(((color < -1)|(color>=bound)).sum());incomplete=int((color<0).sum())
            assert (conflicts,violation,incomplete)==(row["conflicts"],row["palette_violations"],row["incomplete"])
            good=not conflicts and not violation and not incomplete and row["invariants"] and row["failure"] is None
            assert bool(good)==row["success"]
            counts.update(executions=1,successes=int(good),failures=int(not good))
            if not good:failures[str(row["failure"])]+=1
            unique_graphs.add((row["n"],row["delta"],fp));tapes.add(row["tape_seed"]);cases.add((row["case_id"],row["palette"]))
        assert counts["successes"]==int(cell["successes"])==indexed["successes"]
        assert counts["failures"]==int(cell["failures"])
        totals.update(counts)
        if cell["primary"]=="True":primary.update(counts)
        summaries.append(dict(checkpoint=cell["checkpoint"],condition=cell["condition"],primary=cell["primary"]=="True",
            **dict(counts),raw_sha256=cell["rows_sha256"]))
        if (number+1)%50==0:print(json.dumps(dict(cells=number+1,**dict(totals))),flush=True)
    assert totals["executions"]==810000 and totals["successes"]==808413
    assert primary["executions"]==450000 and primary["successes"]==449791
    assert sum(r["failures"]>0 for r in summaries)==15
    assert all(sha(ROOT/p)==s for p,s in {**inputs,**sources}.items())
    result=dict(scope=__doc__,totals=dict(totals),primary=dict(primary),failure_kinds=dict(failures),
        cells=810,failed_cells=15,summaries=summaries,unique_labeled_graphs=len(unique_graphs),
        unique_tapes=len(tapes),unique_case_palette_pairs=len(cases),reconstructed_graphs=len(graphs),
        all_stored_terminal_fields_agree=True,all_compact_cell_counts_agree=True,
        frozen_input_sha256=sha(freeze_path),seconds=time.monotonic()-start,
        full_trajectories_reconstructed=False,new_inference_executions=0)
    (OUT/"raw-terminal-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="summaries"}),flush=True)


if __name__=="__main__":main()
