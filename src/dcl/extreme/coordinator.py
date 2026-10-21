"""Sequential, persistent, checkpoint-preserving CUDA experiment coordinator."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .common import atomic_json, digest, file_hash, now, provenance
from .plan import tasks
from .graphs import realized_or_planned_metadata


def foreign_jobs():
    lines=subprocess.check_output(["ps","-eo","pid=,args="],text=True).splitlines()
    found=[]
    for line in lines:
        if re.search(r"^\s*\d+\s+(?:\S*/)?(?:bash|sh)\s+(?:\S*/)?ucgnn-queue[^ ]*\.sh(?:\s|$)",line):
            found.append(line.strip()[:400])
        elif re.search(r"^\s*\d+\s+\S*python\S*\s+-[mu]\s+dcl\.(?:universal|study)\.cli\s+(?:evaluate|run|benchmark|permutation)(?:\s|$)",line):
            found.append(line.strip()[:400])
    return found


def gpu_free():
    try:
        value=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True).splitlines()[0]
        free,util=map(int,value.split(","))
        return free*2**20,util
    except (subprocess.CalledProcessError,FileNotFoundError,ValueError):
        return 0,0


def completed(path, task):
    if not path.exists(): return False
    r=json.loads(path.read_text())
    if r.get("task_id")!=task["task_id"]: raise RuntimeError("Wrong persisted task identity")
    # No resource-availability skip is treated as a completed experiment.
    return r.get("status") not in ("resource_unavailable","cuda_unavailable")


def run_bounded(command,log_path,timeout,inherited_fds=()):
    with log_path.open("a") as log:
        # Keep the GPU lock alive in the worker even if its coordinator is
        # killed abruptly; a replacement coordinator must wait for it.
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=inherited_fds)
        try:
            return process.wait(timeout=timeout),False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            return process.returncode,True


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--max-new",type=int,help="Bounded smoke run; still CUDA only")
    p.add_argument("--poll-seconds",type=float,default=30)
    args=p.parse_args()
    root=Path(args.root).resolve()
    config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text())
    root.mkdir(parents=True,exist_ok=True)
    for name in ("results","tasks","logs","attempts"):
        (root/name).mkdir(exist_ok=True)
    with (root/"coordinator.lock").open("a+") as own_lock:
        try: fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("Another extreme-degree coordinator is running")
        if args.resume: (root/"STOP").unlink(missing_ok=True)
        gate=json.loads((root/"audit/gate.json").read_text())
        if not gate["passed"]: raise SystemExit("Unresolved original-result audit: queue blocked")
        for name,key in (("inventory.json","inventory_sha256"),("reproduction.json","reproduction_sha256")):
            if file_hash(root/"audit"/name)!=gate[key]: raise SystemExit("Audit artifact changed since passing gate")
        equivalence=json.loads((root/"audit/equivalence-cpu.json").read_text())
        if not equivalence["passed"]: raise SystemExit("Streaming equivalence gate failed")
        inventory=json.loads((root/"audit/inventory.json").read_text())
        meta=provenance()
        if meta["git_status"]: raise SystemExit("Commit verified source before starting the CUDA study")
        manifest={"config":config,"config_sha256":digest(config),"git_head":meta["git_head"],
                  "source_hash":meta["source_hash"],"model_hashes":{r["run"]:r.get("weights_sha256") for r in inventory["runs"]}}
        manifest_path=root/"queue-manifest.json"
        if manifest_path.exists():
            old=json.loads(manifest_path.read_text())
            if {k:v for k,v in old.items() if k!="git_head"}!={k:v for k,v in manifest.items() if k!="git_head"}:
                raise SystemExit("Code, config or checkpoints changed: create a new root, never mix protocols")
            # Documentation-only commits can advance HEAD without changing
            # the frozen protocol. Every worker records its actual revision.
        else: atomic_json(manifest_path,manifest)
        budget_path=root/"budget.json"
        budget=json.loads(budget_path.read_text()) if budget_path.exists() else {"active_seconds":0.0}
        new=0
        stop={"requested":False}
        def request_stop(signum,frame): stop["requested"]=True
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        def state(status,**kwargs):
            atomic_json(root/"status.json",{"status":status,"utc":now(),"pid":os.getpid(),"new_this_session":new,
                                         "active_seconds":budget["active_seconds"],**kwargs})
        def stopping(): return stop["requested"] or (root/"STOP").exists()
        state("starting")
        gpu_lock_path=Path.home()/".cache/dcl-m6-gpu.lock"
        gpu_lock_path.parent.mkdir(parents=True,exist_ok=True)
        with gpu_lock_path.open("a+") as gpu_lock:
            # Wait for the WHOLE existing queue, not a lock gap between its
            # commands; otherwise its nonblocking lock would fail next time.
            while not stopping():
                jobs=foreign_jobs()
                free,util=gpu_free()
                if not jobs and free>=768*2**20 and util<20:
                    try:
                        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        break
                    except BlockingIOError: pass
                state("waiting_for_existing_jobs",foreign_jobs=jobs,free_gpu_bytes=free,gpu_utilization=util)
                print(f"{now()} waiting existing_jobs={len(jobs)} free_MiB={free//2**20}",flush=True)
                time.sleep(args.poll_seconds)
            if stopping():
                state("stopped")
                return
            # Never silently use CPU if CUDA is unavailable.
            eq_path=root/"audit/equivalence-cuda.json"
            if not eq_path.exists():
                cmd=[sys.executable,"-u","-m","dcl.extreme.validation","--root",str(root),"--config",str(config_path),"--device","cuda"]
                state("validating_cuda_streaming")
                code,timeout=run_bounded(cmd,root/"logs/equivalence-cuda.log",300,(gpu_lock.fileno(),))
                if code or timeout or not eq_path.exists():
                    state("blocked_cuda_equivalence",exit_code=code,timeout=timeout)
                    return
            if not json.loads(eq_path.read_text())["passed"]:
                state("blocked_cuda_equivalence")
                return
            last_report=0.0
            for task in tasks(config,inventory):
                if stopping(): state("stopped"); return
                if args.max_new is not None and new>=args.max_new:
                    state("bounded_run_complete"); return
                if budget["active_seconds"]>=config["wall_budget_hours"]*3600:
                    state("budget_exhausted"); return
                if task.get("kind")=="diagnostics":
                    marker=root/"diagnostics/cuda/queue-done.json"
                    if marker.exists(): continue
                    state("mechanism_diagnostics")
                    cmd=[sys.executable,"-u","-m","dcl.extreme.diagnostics","--root",str(root),"--device","cuda"]
                    tick=time.monotonic()
                    code,timeout=run_bounded(cmd,root/"logs/diagnostics-cuda.log",3600,(gpu_lock.fileno(),))
                    budget["active_seconds"]+=time.monotonic()-tick
                    atomic_json(budget_path,budget)
                    if code or timeout:
                        state("diagnostic_technical_failure",exit_code=code,timeout=timeout)
                    else: atomic_json(marker,{"completed_utc":now()})
                    continue
                result_path=root/"results"/f"{task['task_id']}.json"
                if completed(result_path,task): continue
                if task["stage"]=="A-exploratory":
                    pilot=[json.loads(p.read_text()) for p in (root/"results").glob("*.json")]
                    pilot=[r for r in pilot if r.get("stage")=="A-functional" and r.get("success") is not None]
                    if not pilot or sum(r["success"] for r in pilot)/len(pilot)<config["exploratory_min_pilot_success_fraction"]:
                        atomic_json(result_path,{"task_id":task["task_id"],"stage":task["stage"],
                                                 "graph":realized_or_planned_metadata(root,task["graph"]),
                                                 "model":task["model"],"provenance":meta,"cost":task["cost"],
                                                 "status":"exploratory_criterion_not_met","success":None,"device":"cuda"})
                        continue
                if task["cost"]["estimated_gpu_bytes"]>config["gpu_budget_bytes"]:
                    atomic_json(result_path,{"task_id":task["task_id"],"stage":task["stage"],"graph":realized_or_planned_metadata(root,task["graph"]),
                                             "model":task["model"],"provenance":meta,
                                             "cost":task["cost"],"status":"estimated_memory_limit","success":None,"device":"cuda"})
                    continue
                while not stopping():
                    free,util=gpu_free()
                    if free>=task["cost"]["estimated_gpu_bytes"]+config["gpu_reserve_bytes"]: break
                    state("waiting_for_vram",task_id=task["task_id"],free_gpu_bytes=free)
                    time.sleep(args.poll_seconds)
                if stopping(): state("stopped"); return
                task_path=root/"tasks"/f"{task['task_id']}.json"
                atomic_json(task_path,task)
                state("evaluating",task_id=task["task_id"],stage=task["stage"],graph=task["graph"])
                cmd=[sys.executable,"-u","-m","dcl.extreme.worker","--root",str(root),"--config",str(config_path),
                     "--task",str(task_path),"--output",str(result_path),"--device","cuda"]
                tick=time.monotonic()
                code,timeout=run_bounded(cmd,root/"logs/graphs.log",task["timeout_seconds"],(gpu_lock.fileno(),))
                budget["active_seconds"]+=time.monotonic()-tick
                atomic_json(budget_path,budget)
                if timeout or not result_path.exists():
                    atomic_json(result_path,{"task_id":task["task_id"],"stage":task["stage"],
                                             "graph":realized_or_planned_metadata(root,task["graph"]),
                                             "model":task["model"],"status":"timeout" if timeout else "worker_crash",
                                             "provenance":meta,"cost":task["cost"],
                                             "success":None,"device":"cuda","exit_code":code,"completed_utc":now()})
                record=json.loads(result_path.read_text())
                if record["status"] in ("resource_unavailable","cuda_unavailable"):
                    # Keep the attempt; next task/pass may have sufficient VRAM.
                    atomic_json(root/"attempts"/f"{task['task_id']}-{time.time_ns()}.json",record)
                else: new+=1
                print(f"{now()} {task['stage']} d{task['graph']['degree']} n{task['graph']['n']} "
                      f"{task['graph']['family']} {record['status']}",flush=True)
                if record["status"]=="technical_error":
                    # A programming error is not evidence about the network.
                    state("blocked_technical_error",task_id=task["task_id"])
                    return
                if time.monotonic()-last_report>300:
                    subprocess.run([sys.executable,"-m","dcl.extreme.report","--root",str(root)],check=True)
                    last_report=time.monotonic()
            pending=[p for p in (root/"results").glob("*.json")
                     if json.loads(p.read_text()).get("status") in ("resource_unavailable","cuda_unavailable")]
            diag_incomplete=not (root/"diagnostics/cuda/queue-done.json").exists()
            final_status="resource_limited_incomplete" if pending else ("diagnostic_incomplete" if diag_incomplete else "complete")
            state(final_status,pending_resource_cases=len(pending),diagnostics_incomplete=diag_incomplete)
            subprocess.run([sys.executable,"-m","dcl.extreme.report","--root",str(root)],check=True)


if __name__=="__main__": main()
