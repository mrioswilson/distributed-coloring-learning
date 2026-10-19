"""Independent invariants for graph cohorts and exact inference streaming."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from dcl.extreme.common import weights_hash
from dcl.extreme.graphs import generate, prepare, seeds
from dcl.extreme.runtime import ChunkedPolicy, estimate_cost
from dcl.universal.evaluation import run_rollout
from dcl.universal.model import UniversalColorGNN


@pytest.mark.parametrize("family",["REGULAR","IRREGULAR","HETEROGENEOUS","BIPARTITE","CLIQUE"])
@pytest.mark.parametrize("degree",[3,20,50])
def test_new_graph_families_attain_declared_maximum(family,degree):
    n = degree+1 if family=="CLIQUE" else max(40,2*(degree+1))
    g = generate(n,degree,family,901)
    assert nx.is_connected(g) and max(dict(g.degree()).values())==degree
    assert not nx.number_of_selfloops(g)
    if family=="BIPARTITE": assert nx.is_bipartite(g)
    if family=="IRREGULAR": assert len(set(dict(g.degree()).values()))>1
    if family=="HETEROGENEOUS": assert sum(d in (2,3) for _,d in g.degree()) > .5*n


@pytest.mark.parametrize("variant",["NORMALIZED","RAW_PREFIX"])
@pytest.mark.parametrize("palette",[4,11,21,51,101,201])
def test_streaming_preserves_logits_actions_and_weights(variant,palette):
    torch.manual_seed(13)
    torch.set_num_threads(1)
    model = UniversalColorGNN(variant=variant).eval()
    before = weights_hash(model)
    n=36
    colors = torch.randint(-1,palette,(n,))
    colors[::2]=-1
    edges = torch.tensor([(i,(i+1)%n) for i in range(n)]+[((i+1)%n,i) for i in range(n)]).T
    args = colors,torch.rand(n),torch.rand(n),edges,torch.full((n,),palette)
    with torch.inference_mode():
        a = model(*args,edge_chunk=7)
        b = ChunkedPolicy(model,color_chunk=3,edge_budget_bytes=2048)(*args)
    for key in ("participation_logit","candidate_logits","commit_logit","availability","k_hat","lo","hi"):
        torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=2e-5,atol=2e-5)
    assert torch.equal(a.participate,b.participate)
    assert torch.equal(a.candidate,b.candidate)
    assert torch.equal(a.commit_logit>0,b.commit_logit>0)
    assert weights_hash(model)==before


def test_streamed_closed_loop_is_unchanged():
    from dcl.m4_data import _edge_index
    torch.manual_seed(4)
    model=UniversalColorGNN().eval()
    edges=_edge_index(generate(40,10,"REGULAR",82))
    original,ca=run_rollout(model,40,edges,11,814,max_rounds=12)
    streamed,cb=run_rollout(ChunkedPolicy(model,3),40,edges,11,814,max_rounds=12)
    assert np.array_equal(ca,cb)
    for k in ("success","rounds","unsafe_commits","states","by_k"):
        if k=="by_k":
            for v in original[k]:
                for metric in original[k][v]:
                    if "interval_deviation_sum" not in metric:
                        assert original[k][v][metric]==streamed[k][v][metric]
        else: assert original[k]==streamed[k]


def test_graph_cache_and_exclusion(tmp_path):
    meta=dict(namespace=901,degree=3,palette=4,n=40,family="REGULAR",graph_index=0)
    edges,info=prepare(tmp_path,meta,set(),set())
    cached,again=prepare(tmp_path,meta,set(),set())
    assert info==again and np.array_equal(edges,cached)
    other=tmp_path/"other"
    _,fresh=prepare(other,meta,{info["fingerprint"]},set())
    assert fresh["fingerprint"] != info["fingerprint"] and fresh["generation_attempt"]>0


def test_clique_cross_device_reuse_is_marked_as_deterministic(tmp_path):
    meta=dict(namespace=1,degree=3,palette=4,n=4,family="CLIQUE",graph_index=0)
    _,first=prepare(tmp_path,meta,set(),set())
    _,second=prepare(tmp_path,{**meta,"namespace":2},set(),set())
    assert first["fingerprint"]==second["fingerprint"]
    assert second["reused_deterministic_stress_graph"]
    assert second["deterministic_stress_graph"]


def test_cost_is_sparse_and_accounts_for_color_messages():
    a=estimate_cost(640,100,101)
    b=estimate_cost(2560,100,101)
    assert b["logical_message_bytes_per_round"]==4*a["logical_message_bytes_per_round"]
    assert a["message_floats_per_directed_edge_per_stage"]==101*32
    assert b["estimated_gpu_bytes"]<3*2**30


@pytest.mark.parametrize("palette",[4,21,101])
def test_cached_center_diagnostic_equals_original_stage1(palette):
    from dcl.extreme.diagnostics import learned_center,choices_from_center
    model=UniversalColorGNN().eval()
    colors=np.full(palette,-1,np.int64)
    colors[1:palette//2]=np.arange(palette//2-1)
    edges=torch.tensor([(0,j) for j in range(1,palette)]+[(j,0) for j in range(1,palette)]).T
    grid=np.array([0,.1,.333,.499,.501,.77,.999],np.float32)
    selected=choices_from_center(model,learned_center(model,colors,palette),grid)
    for i,rc in enumerate(grid):
        with torch.inference_mode():
            output=model.stage1(torch.tensor(colors),torch.zeros(palette),torch.full((palette,),float(rc)),
                                edges,torch.full((palette,),palette))
        assert selected[i]==int(output[1][0].argmax())


def test_exhaustive_commit_types_match_real_star():
    from dcl.extreme.diagnostics import commit_presence_table
    model=UniversalColorGNN().eval()
    table=commit_presence_table(model)
    # Representative multiplicities, including both conflict mechanisms.
    types=[(1,False,-1),(0,False,-1),(-1,False,1),(-1,False,0),(-1,True,1),(-1,True,0)]
    for mask in (0,1,2,8,16,32,63):
        indices=[j for j in range(6) if mask&(1<<j)]
        neighbors=[types[j] for j in indices for _ in range(2)]
        n=1+len(neighbors)
        colors=torch.tensor([-1]+[t[0] for t in neighbors])
        part=torch.tensor([True]+[t[1] for t in neighbors])
        cand=torch.tensor([0]+[t[2] for t in neighbors])
        edges=torch.tensor([(0,j) for j in range(1,n)]+[(j,0) for j in range(1,n)],dtype=torch.long).reshape(-1,2).T
        with torch.inference_mode():
            logit=float(model.commit_logits(colors,edges,torch.full((n,),4),part,cand)[0])
        assert logit==pytest.approx(table[mask]["logit"],abs=1e-5)


def test_coordinator_detects_entire_foreign_queue(monkeypatch):
    from dcl.extreme.coordinator import foreign_jobs
    process_list="""1997 bash experiments/ucgnn-queue-3.sh
2134 /path/.venv/bin/python -m dcl.universal.cli evaluate --checkpoint foo.pt
1000 /path/python -m dcl.extreme.worker --task own.json
1001 /path/claude --verbose
"""
    monkeypatch.setattr("subprocess.check_output",lambda *a,**k:process_list)
    jobs=foreign_jobs()
    assert len(jobs)==2 and any("queue-3" in j for j in jobs)


def test_resume_distinguishes_technical_outcomes_and_pending_resources(tmp_path):
    from dcl.extreme.common import atomic_json
    from dcl.extreme.coordinator import completed
    path=tmp_path/"case.json"
    task={"task_id":"abc"}
    assert not completed(path,task)
    for status in ("ok","coloring_failure","oom","timeout"):
        atomic_json(path,{"task_id":"abc","status":status})
        assert completed(path,task)
    for status in ("resource_unavailable","cuda_unavailable"):
        atomic_json(path,{"task_id":"abc","status":status})
        assert not completed(path,task)


def test_intervals_use_graphs_not_model_executions():
    from dcl.extreme.report import summary_rows
    rows=[]
    for seed in (0,1):
        for graph in ("g0","g1"):
            rows.append(dict(cohort="cuda-main",experiment="E1",training_seed=seed,degree=20,n=640,
                             family="REGULAR",fingerprint=graph,success=True))
    summary=summary_rows(rows)
    assert len(summary)==2
    assert all(r["independent_graphs"]==2 and r["measured"]==2 for r in summary)
    assert summary[0]["ci95_lower"]==pytest.approx(.025**.5)


def test_one_graph_worker_cli_persists_complete_provenance(tmp_path):
    import os
    import subprocess
    import sys
    from dcl.extreme.common import atomic_json,file_hash
    from dcl.universal.config import UConfig
    model=UniversalColorGNN()
    cp=tmp_path/"model.pt"
    config=UConfig().to_dict()
    torch.save({"config":config,"model_state_dict":model.state_dict(),"epoch":120},cp)
    info=dict(checkpoint=str(cp),checkpoint_sha256=file_hash(cp),weights_sha256=weights_hash(model),
              config=config,train_degrees=[3],train_sizes=[40],epoch=120,parameters=16035)
    graph=dict(namespace=172898,degree=3,palette=4,n=40,family="REGULAR",graph_index=0)
    task=dict(task_id="worker-smoke",stage="test",model=info,graph=graph,cost=estimate_cost(40,3,4))
    atomic_json(tmp_path/"task.json",task)
    atomic_json(tmp_path/"config.json",dict(color_chunk=8,max_rounds=3))
    atomic_json(tmp_path/"audit/historical-exclusions.json",dict(fingerprints=[],graph_seeds=[]))
    subprocess.run([sys.executable,"-m","dcl.extreme.worker","--task",str(tmp_path/"task.json"),
                    "--config",str(tmp_path/"config.json"),"--root",str(tmp_path),"--device","cpu",
                    "--output",str(tmp_path/"result.json")],check=True,capture_output=True,timeout=30,
                   env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1]/"src")})
    r=json.loads((tmp_path/"result.json").read_text())
    assert r["status"] in ("ok","coloring_failure")
    assert r["weights_sha256_before"]==r["weights_sha256_after"]==info["weights_sha256"]
    assert r["graph"]["graph_seed"] and r["graph"]["tape_seed"] and r["provenance"]["source_hash"]


def test_oom_is_technical_and_has_no_coloring_outcome(tmp_path,monkeypatch):
    from dcl.extreme import worker
    from dcl.extreme.common import atomic_json
    import sys
    task=dict(task_id="oom-test",stage="test",graph=dict(namespace=80,degree=3,palette=4,n=40,family="REGULAR",graph_index=0),model={})
    atomic_json(tmp_path/"task.json",task)
    atomic_json(tmp_path/"config.json",{})
    def fail(*args): raise torch.cuda.OutOfMemoryError("synthetic test; no GPU allocation")
    monkeypatch.setattr(worker,"evaluate",fail)
    monkeypatch.setattr(sys,"argv",["worker","--task",str(tmp_path/"task.json"),"--config",str(tmp_path/"config.json"),
                                    "--root",str(tmp_path),"--device","cpu","--output",str(tmp_path/"result.json")])
    worker.main()
    r=json.loads((tmp_path/"result.json").read_text())
    assert r["status"]=="oom" and r["success"] is None
    assert r["graph"]["graph_seed"] and r["graph"]["tape_seed"]
    assert not r["graph"]["graph_generated"]


def test_gpu_lock_survives_abrupt_coordinator_exit(tmp_path):
    import fcntl
    import os
    import signal
    import subprocess
    import sys
    import time
    marker=tmp_path/"child.pid"
    lock_path=tmp_path/"gpu.lock"
    child_code=f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
    supervisor_code=f"""from pathlib import Path
import fcntl,sys
from dcl.extreme.coordinator import run_bounded
with open({str(lock_path)!r}, 'a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    run_bounded([sys.executable, '-c', {child_code!r}],Path({str(tmp_path/'worker.log')!r}),60,(lock.fileno(),))
"""
    process=subprocess.Popen([sys.executable,"-c",supervisor_code],
                             env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1]/"src")})
    child_pid=None
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if marker.exists() and marker.read_text().strip():
                child_pid=int(marker.read_text());break
            if process.poll() is not None: break
            time.sleep(.02)
        assert child_pid is not None
        process.kill();process.wait(timeout=5)
        with lock_path.open("a+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:
        if process.poll() is None: process.kill();process.wait()
        if child_pid is not None:
            try: os.kill(child_pid,signal.SIGTERM)
            except ProcessLookupError: pass
