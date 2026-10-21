"""Independent inventory, original result census, and CPU reproduction.

Never writes inside the historical root. CPU reproduction is explicitly a
cross-device audit, not a replacement for the queued GPU study.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from .common import atomic_json, file_hash, now, provenance, weights_hash
from dcl.universal.evaluation import build_case, run_rollout
from dcl.universal.train import load_model


def ablation_and_epochs(historical,output):
    from .diagnostics import commit_presence_table
    base=Path(historical)/"runs"
    e1=list(base.glob("e1-*-s00-*"))
    c1=list(base.glob("c1-*-s00-*"))
    if len(e1)!=1 or len(c1)!=1: return
    model,a=load_model(e1[0]/"epoch-120.pt","cpu")
    own,b=load_model(c1[0]/"epoch-120.pt","cpu")
    equal={name:bool(torch.equal(value,own.state_dict()[name])) for name,value in model.state_dict().items()}
    checks=[]
    for name in ("best.pt","epoch-010.pt","epoch-020.pt","epoch-120.pt"):
        model,saved=load_model(e1[0]/name,"cpu")
        rows=commit_presence_table(model)
        checks.append({"checkpoint":name,"epoch":saved["epoch"],"cases":len(rows),
                       "correct":sum(r["correct"] for r in rows),"unsafe":sum(r["unsafe_commit"] for r in rows),
                       "weights_sha256":weights_hash(model)})
    value={"E1_C1_equal_parameters":equal,"E1_C1_same_initial_hash":a["initial_state_hash"]==b["initial_state_hash"],
           "E1_C1_same_training_data":a["data_hash"]==b["data_hash"],"E1_commit_checkpoints":checks}
    atomic_json(Path(output)/"ablation-and-epochs.json",value)
    return value


def extend_exclusions(historical, output):
    """Include every saved M6/UCGNN training and validation graph, read-only."""
    historical,output=Path(historical),Path(output)
    path=output/"historical-exclusions.json"
    value=json.loads(path.read_text())
    fingerprints=set(value["fingerprints"])
    graph_seeds=set(value["graph_seeds"])
    manifests=histories=0
    for base in (historical,historical.parent/"m6study-release"):
        for p in sorted((base/"runs").glob("*/data_manifest.json")):
            manifest=json.loads(p.read_text())
            manifests+=1
            for split in ("train","validation"):
                for r in manifest.get(split,[]):
                    fingerprints.add(r["fingerprint"])
                    graph_seeds.add(r["graph_seed"])
        for p in sorted((base/"runs").glob("*/history.json")):
            histories+=1
            for row in json.loads(p.read_text()):
                validation=row.get("validation_rollouts") or {}
                for r in validation.get("per_graph",[]):
                    fingerprints.add(r["fingerprint"])
                    graph_seeds.add(r["graph_seed"])
    atomic_json(path,{"fingerprints":sorted(fingerprints),"graph_seeds":sorted(graph_seeds),
                      "utc":now(),"training_manifests_scanned":manifests,"validation_histories_scanned":histories})
    # Frozen M6 final graphs are also present in the independently censused
    # original UCGNN final_test cohort (the original equality test verifies it).
    repo=historical.parents[1]
    publication=repo/"results/m6study/2026-10-17-final/publication-manifest.json"
    if publication.exists():
        original=json.loads(publication.read_text())
        checks={name:file_hash(publication.parent/name)==sha for name,sha in original["files"].items()}
        atomic_json(output/"m6-preservation.json",{"files":checks,"all_match":all(checks.values()),
                                                 "freeze_sha256":original["freeze_sha256"],"utc":now()})
        if not all(checks.values()): raise RuntimeError("M6 publication integrity mismatch")
    print(f"exclusions graphs={len(fingerprints)} seeds={len(graph_seeds)} manifests={manifests}",flush=True)


def inventory(historical, output):
    historical, output = Path(historical), Path(output)
    torch.set_num_threads(1)
    runs = []
    forbidden = set()
    for directory in sorted((historical / "runs").iterdir()):
        if not (directory / "config.json").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        history = json.loads((directory / "history.json").read_text()) if (directory / "history.json").exists() else []
        manifest = json.loads((directory / "data_manifest.json").read_text()) if (directory / "data_manifest.json").exists() else {}
        for split in ("train", "validation"):
            forbidden.update(r["fingerprint"] for r in manifest.get(split, []))
        path = directory / "epoch-120.pt"
        row = {"run": directory.name, "config": config, "epochs_completed": len(history),
               "training_graphs": len(manifest.get("train", [])), "validation_graphs": len(manifest.get("validation", [])),
               "checkpoints": [p.name for p in sorted(directory.glob("*.pt"))],
               "training_seconds": history[-1].get("elapsed_seconds") if history else None,
               "failures": [p.name for p in directory.glob("failure*.json")]}
        if path.exists():
            model, saved = load_model(path, "cpu")
            row.update(checkpoint=str(path.resolve()), checkpoint_sha256=file_hash(path), weights_sha256=weights_hash(model),
                       parameters=sum(p.numel() for p in model.parameters()), epoch=saved["epoch"],
                       train_degrees=sorted({r["degree"] for r in manifest["train"]}),
                       train_sizes=sorted({r["n"] for r in manifest["train"]}))
        runs.append(row)
    counts = defaultdict(lambda: Counter())
    fingerprints = defaultdict(set)
    graph_seeds = set()
    malformed = []
    seed0 = []
    for path in sorted((historical / "evaluations").glob("*/*/*.json")):
        if path.name == "view.json":
            continue
        try:
            r = json.loads(path.read_text())
            key = (path.parents[1].name, path.parent.name, r["degree"], r["n"], r["distribution"])
            counts[key].update(cases=1, successes=int(r["success"]), unsafe=r["unsafe_commits"],
                               unfinished=int(not r["finished"]), states=r["states"],
                               candidate_correct=r["candidate_correct"], forbidden=r["forbidden_offers"],
                               rounds=r["rounds"])
            fingerprints[key[:2]].add(r["fingerprint"])
            forbidden.add(r["fingerprint"])
            graph_seeds.add(r["graph_seed"])
            if path.parent.name == "e1-s00-epoch-120" and path.parents[1].name in ("final_test", "ucgnn_extra"):
                seed0.append((path, r))
        except Exception as e:
            malformed.append({"path": str(path), "error": repr(e)})
    cohorts = [{"split": k[0], "view": k[1], "degree": k[2], "n": k[3], "family": k[4], **dict(v)}
               for k, v in counts.items()]
    cases = [r for _, r in seed0]
    census = {"utc": now(), "provenance": provenance(), "historical_root": str(historical.resolve()),
              "runs": runs, "cohorts": cohorts, "malformed": malformed,
              "preliminary": {"cases": len(cases), "successes": sum(r["success"] for r in cases),
                              "unique_graphs": len({r["fingerprint"] for r in cases}),
                              "unique_graph_seeds": len({r["graph_seed"] for r in cases}),
                              "unsafe": sum(r["unsafe_commits"] for r in cases),
                              "epochs": sorted({r["checkpoint_epoch"] for r in cases}),
                              "seeds": sorted({r["seed"] for r in cases}),
                              "checkpoint_hashes": sorted({r["checkpoint_sha256"] for r in cases})},
              "evaluation_unique_graphs": [{"split": k[0], "view": k[1], "graphs": len(v)} for k,v in fingerprints.items()]}
    atomic_json(output / "inventory.json", census)
    atomic_json(output / "historical-exclusions.json", {"fingerprints": sorted(forbidden), "graph_seeds": sorted(graph_seeds)})
    print(json.dumps(census["preliminary"]), flush=True)
    ablation_and_epochs(historical,output)
    return census, seed0


def reproduce(historical, output, cases):
    torch.set_num_threads(1)
    cp = next((Path(historical) / "runs").glob("e1-*-s00-*/epoch-120.pt"))
    model, saved = load_model(cp, "cpu")
    initial = weights_hash(model)
    parameter_ids = [id(p) for p in model.parameters()]
    selected = []
    # Both families, seen/unseen degrees, three ordinary sizes; largest endpoints separately.
    for degree, n in ((3,40),(3,640),(3,40960),(10,160),(10,640),(15,40),(15,640),(15,10240)):
        for family in ("REGULAR", "IRREGULAR"):
            matches = [(p,r) for p,r in cases if r["degree"] == degree and r["n"] == n and r["distribution"] == family]
            if matches:
                selected.append(sorted(matches, key=lambda x:x[1]["graph_id"])[0])
    comparisons = []
    keys = ("success", "finished", "rounds", "unsafe_commits", "monochromatic_edges", "uncolored_nodes",
            "states", "candidate_correct", "participation_correct", "forbidden_offers", "valid_commits_rejected")
    for p, original in selected:
        target = Path(output) / "reproduction" / p.name
        if target.exists():
            comparisons.append(json.loads(target.read_text()))
            continue
        edges, info = build_case(original)
        if info["fingerprint"] != original["fingerprint"]:
            raise RuntimeError("Historical graph cannot be regenerated")
        result, colors = run_rollout(model, original["n"], edges, original["degree"]+1, original["tape_seed"])
        differences = {k: {"original": original.get(k), "cpu": result.get(k)} for k in keys if original.get(k) != result.get(k)}
        row = {"original_file": str(p.resolve()), "graph_id": original["graph_id"], "device": "cpu",
               "checkpoint": str(cp.resolve()), "weights_sha256": initial, "differences": differences,
               "result": result, "graph_fingerprint": info["fingerprint"]}
        atomic_json(target, row)
        comparisons.append(row)
        print(f"reproduce {original['graph_id']} differences={differences} success={result['success']}", flush=True)
    hashes = []
    from dcl.m4_data import _edge_index
    from dcl.study.graphs import generate_graph
    for degree in (3,10,20):
        edges = _edge_index(generate_graph(40, degree, "REGULAR", 812732+degree))
        result,_ = run_rollout(model, 40, edges, degree+1, 89204+degree)
        hashes.append({"degree": degree, "palette": degree+1, "hash": weights_hash(model),
                       "parameters": sum(p.numel() for p in model.parameters()), "success": result["success"],
                       "same_parameter_objects": parameter_ids == [id(p) for p in model.parameters()]})
    report = {"checkpoint": str(cp.resolve()), "checkpoint_sha256": file_hash(cp), "epoch": saved["epoch"],
              "config": saved["config"], "device": "cpu", "initial_weights_sha256": initial,
              "final_weights_sha256": weights_hash(model), "palettes": hashes,
              "reproduced": len(comparisons), "discrepant_cases": sum(bool(r["differences"]) for r in comparisons),
              "comparisons": comparisons, "provenance": provenance()}
    atomic_json(Path(output) / "reproduction.json", report)
    # A resource/runtime discrepancy is not silently accepted as algorithmic equivalence.
    passed = (len(cases) == 4880 and all(r["success"] for _,r in cases)
              and not any(r["differences"] for r in comparisons)
              and all(r["hash"] == initial and r["same_parameter_objects"] for r in hashes))
    atomic_json(Path(output) / "gate.json", {"passed": passed, "utc": now(), "cases": len(cases),
                                           "reproduced": len(comparisons), "reproduction_sha256": file_hash(Path(output)/"reproduction.json"),
                                           "inventory_sha256": file_hash(Path(output)/"inventory.json")})
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extend-exclusions",action="store_true")
    args = parser.parse_args()
    if args.extend_exclusions:
        extend_exclusions(args.historical,args.output)
        return
    _, cases = inventory(args.historical, args.output)
    if not reproduce(args.historical, args.output, cases):
        raise SystemExit("Audit reproduction gate failed; extensive experiments blocked")


if __name__ == "__main__":
    main()
