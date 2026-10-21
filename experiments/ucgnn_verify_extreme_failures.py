"""Replay one failure per model/degree/size/family with the original evaluator.

This checks negative outcomes of the color-streaming study without training,
changing checkpoints, or counting repeated cases as new graph samples.
Run with the validated study Python; --extreme-root points to its runtime data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extreme-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.extreme_root.resolve()
    # Use the study's unchanged Universal implementation and its graph cache.
    sys.path.insert(0, str(root.parents[1] / "src"))
    import numpy as np
    import torch
    from dcl.extreme.common import atomic_json, file_hash, now, provenance, weights_hash
    from dcl.study.storage import training_lock
    from dcl.universal.evaluation import run_rollout
    from dcl.universal.rules import exact_commit_labels
    from dcl.universal.train import load_model

    config = json.loads((root.parents[1] / "configs/ucgnn/extreme-v1.json").read_text())
    representatives = {}
    failures = 0
    for path in sorted((root / "results").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "coloring_failure":
            continue
        failures += 1
        graph = record["graph"]
        key = (record["experiment"], record["training_seed"], graph["degree"], graph["n"], graph["family"])
        representatives.setdefault(key, (path, record))
    if not representatives:
        raise RuntimeError("No persisted coloring failures to verify")

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rows = []
    with training_lock(root):
        free, total = torch.cuda.mem_get_info()
        if free < 2 * 2**30:
            raise RuntimeError("Original-evaluator replays require 2 GiB free CUDA memory")
        torch.cuda.set_per_process_memory_fraction(min(0.9, config["gpu_budget_bytes"] / total))
        for key, (path, saved) in sorted(representatives.items()):
            checkpoint = Path(saved["checkpoint"])
            if file_hash(checkpoint) != saved["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint changed: {checkpoint}")
            model, _ = load_model(checkpoint, "cuda")
            before = weights_hash(model)
            if before != saved["weights_sha256_before"]:
                raise RuntimeError(f"Weights differ from the saved case: {path}")
            graph = saved["graph"]
            with np.load(root / "graphs" / f"{graph['graph_id']}.npz", allow_pickle=False) as cache:
                edges = cache["edge_index"]
            # Observe the first autonomous proposals, then compute exact labels
            # on copies for diagnosis. These labels never enter the rollout.
            n, palette = graph["n"], graph["palette"]
            initial = np.full(n, -1, dtype=np.int64)
            tape = np.random.default_rng(graph["tape_seed"])
            rp, rc = tape.random(n), tape.random(n)
            with torch.inference_mode():
                proposal = model(torch.from_numpy(initial).cuda(), torch.tensor(rp, dtype=torch.float32, device="cuda"),
                                 torch.tensor(rc, dtype=torch.float32, device="cuda"), torch.from_numpy(edges).cuda(),
                                 torch.full((n,), palette, dtype=torch.long, device="cuda"))
                part, cand = proposal.participate.cpu().numpy(), proposal.candidate.cpu().numpy()
                histogram = np.bincount(cand[part], minlength=palette)
                safe = exact_commit_labels(initial, edges, part, cand)
                first_round = {"participants": int(part.sum()), "distinct_proposed_colors": int((histogram > 0).sum()),
                               "proposal_histogram": histogram.tolist(), "exact_safe_proposals": int(safe.sum()),
                               "executed_commits": int((part & (proposal.commit_logit > 0).cpu().numpy()).sum())}
                del proposal
            result, colors = run_rollout(model, graph["n"], edges, graph["palette"], graph["tape_seed"],
                                        config["max_rounds"])
            color_hash = hashlib.sha256(colors.tobytes()).hexdigest()
            metrics = ("success", "finished", "rounds", "uncolored_nodes", "monochromatic_edges",
                       "unsafe_commits", "forbidden_offers", "valid_commits_rejected")
            differences = {k: {"saved": saved[k], "replayed": result[k]}
                           for k in metrics if result[k] != saved[k]}
            after = weights_hash(model)
            row = {"source_record": str(path), "source_record_sha256": file_hash(path),
                   "graph": graph, "experiment": saved["experiment"], "training_seed": saved["training_seed"],
                   "checkpoint_sha256": saved["checkpoint_sha256"], "original_evaluator": True,
                   "metrics": {k: result[k] for k in metrics}, "first_round": first_round,
                   "metric_differences": differences,
                   "identical_final_colors": color_hash == saved["final_colors_sha256"],
                   "final_colors_sha256": color_hash, "weights_sha256_before": before,
                   "weights_sha256_after": after, "inference_seconds": result["inference_seconds"],
                   "gpu_peak_bytes": result["gpu_peak_bytes"]}
            row["passed"] = not differences and row["identical_final_colors"] and before == after
            rows.append(row)
            print(f"{key}: passed={row['passed']} pending={result['uncolored_nodes']}", flush=True)
            del model
            torch.cuda.empty_cache()
    report = {"generated_utc": now(), "saved_failures": failures, "representatives": len(rows),
              "passed": all(r["passed"] for r in rows), "provenance": provenance(), "rows": rows,
              "script_sha256": file_hash(__file__),
              "interpretation": "Repeated saved cases; no new independent graphs or fitted weights."}
    atomic_json(args.output, report)
    if not report["passed"]:
        raise SystemExit("A replay differs; inspect the persisted evidence before drawing conclusions")


if __name__ == "__main__":
    main()
