"""One bounded, isolated graph evaluation; called by the persistent coordinator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .common import atomic_json, file_hash, now, provenance, weights_hash
from .graphs import prepare
from .runtime import ChunkedPolicy
from dcl.universal.evaluation import run_rollout, teacher_rollout
from dcl.universal.train import load_model


def evaluate(task, config, root, device):
    root = Path(root)
    began = time.perf_counter()
    base = {"task_id":task["task_id"],"stage":task["stage"],"device":device,"started_utc":now(),
            "checkpoint":task["model"]["checkpoint"],"checkpoint_sha256":task["model"]["checkpoint_sha256"],
            "expected_weights_sha256":task["model"]["weights_sha256"],
            "training_seed":task["model"]["config"]["seed"],"training_degrees":task["model"]["train_degrees"],
            "training_n":task["model"]["train_sizes"],"checkpoint_epoch":task["model"]["epoch"],
            "experiment":task["model"]["config"]["name"],"training_config":task["model"]["config"],
            "parameters":task["model"]["parameters"],"cost":task["cost"],"graph":task["graph"],
            "provenance":provenance(),"runtime":"original-weights-color-streaming-v1"}
    if file_hash(base["checkpoint"]) != base["checkpoint_sha256"]:
        raise RuntimeError("Historical checkpoint changed; refusing evaluation")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        if not torch.cuda.is_available():
            return {**base,"status":"cuda_unavailable","success":None}
        free,total = torch.cuda.mem_get_info()
        base["cuda_free_before_bytes"] = free
        if free < task["cost"]["estimated_gpu_bytes"]+config["gpu_reserve_bytes"]:
            return {**base,"status":"resource_unavailable","success":None}
        torch.cuda.set_per_process_memory_fraction(min(.9,config["gpu_budget_bytes"]/total))
        base["gpu_name"] = torch.cuda.get_device_name()
    exclusions = json.loads((root / "audit/historical-exclusions.json").read_text())
    edges, info = prepare(root,task["graph"],set(exclusions["fingerprints"]),set(exclusions["graph_seeds"]))
    base.update(graph=info)
    model,saved = load_model(base["checkpoint"],device)
    before = weights_hash(model)
    if before != base["expected_weights_sha256"]:
        raise RuntimeError("Unexpected loaded weights")
    ids_before = [id(p) for p in model.parameters()]
    policy = ChunkedPolicy(model,config["color_chunk"])
    result,colors = run_rollout(policy,info["n"],edges,info["palette"],info["tape_seed"],config["max_rounds"],edge_chunk=None)
    # The adapter sets edge chunk length separately for each color block;
    # do not report the original evaluator's unrelated 192-MiB auto estimate.
    result.pop("edge_chunk",None)
    result["edge_chunk_policy"]="per-color-block-32MiB"
    result["color_chunk"]=config["color_chunk"]
    after = weights_hash(model)
    if after != before or ids_before != [id(p) for p in model.parameters()]:
        raise RuntimeError("Weights or parameter objects changed during inference")
    if bool(((colors < -1)|(colors >= info["palette"])).any()):
        raise RuntimeError("Out-of-palette color")
    # The teacher runs only AFTER autonomous execution, on separate arrays.
    tick = time.perf_counter()
    tc,tr = teacher_rollout(info["n"],edges,info["palette"],info["tape_seed"])
    counts = np.bincount(colors[colors>=0],minlength=info["palette"])
    probabilities = counts[counts>0]/max(1,counts.sum())
    return {**base,**result,"status":"ok" if result["success"] else "coloring_failure",
            "weights_sha256_before":before,"weights_sha256_after":after,"same_parameter_objects":True,
            "teacher_rounds":tr,"identical_final_coloring":bool(np.array_equal(tc,colors)),
            "teacher_observer_seconds":time.perf_counter()-tick,
            "inference_seconds_per_round":result["inference_seconds"]/result["rounds"],
            "gpu_peak_bytes_per_node":result.get("gpu_peak_bytes",0)/info["n"] if device=="cuda" else None,
            "gpu_peak_bytes_per_directed_edge":result.get("gpu_peak_bytes",0)/max(1,edges.shape[1]) if device=="cuda" else None,
            "directed_edges":int(edges.shape[1]),
            "colors_used":int((counts>0).sum()),"color_counts":counts.tolist(),
            "color_entropy_nats":float(-(probabilities*np.log(probabilities)).sum()),
            "max_color_fraction":float(counts.max()/max(1,counts.sum())),
            "final_colors_sha256":__import__('hashlib').sha256(colors.tobytes()).hexdigest(),
            "total_seconds":time.perf_counter()-began,"completed_utc":now()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--root",required=True)
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    task = json.loads(Path(args.task).read_text())
    config = json.loads(Path(args.config).read_text())
    try:
        record = evaluate(task,config,args.root,args.device)
    except torch.cuda.OutOfMemoryError:
        record = {"task_id":task["task_id"],"stage":task["stage"],"graph":task["graph"],"model":task["model"],
                  "device":args.device,"status":"oom","success":None,"completed_utc":now(),"traceback":traceback.format_exc(),
                  "provenance":provenance()}
    except Exception:
        record = {"task_id":task["task_id"],"stage":task["stage"],"graph":task["graph"],"model":task["model"],
                  "device":args.device,"status":"technical_error","success":None,"completed_utc":now(),
                  "traceback":traceback.format_exc(),"provenance":provenance()}
    atomic_json(args.output,record)
    print(json.dumps({"task_id":task["task_id"],"status":record["status"],"degree":task["graph"]["degree"],
                      "n":task["graph"]["n"],"family":task["graph"]["family"],
                      "success":record.get("success"),"rounds":record.get("rounds"),"seconds":record.get("total_seconds")}),flush=True)


if __name__ == "__main__":
    main()
