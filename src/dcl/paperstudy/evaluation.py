"""Independent paired graph cohort and unshielded, instrumented rollouts."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from dcl.extreme.graphs import generate
from dcl.m4_data import _edge_index
from dcl.study.config import digest
from dcl.study.graphs import degree_stats, fingerprint
from dcl.study.storage import atomic_json, file_hash, now
from dcl.universal.evaluation import observe
from dcl.universal.model import UOutput
from dcl.universal.rules import available_matrix, conflict_flags, exact_commit_labels
from .config import EXTREME, HISTORICAL, ROOT, SEEDS, cohort_plan, named_seed, protocol_hash
from .teacher import cyclic_offer
from .training import load_model, weights_hash


class CaseTimeout(Exception):
    pass


@contextmanager
def deadline(seconds):
    previous = signal.getsignal(signal.SIGALRM)
    def expired(*_):
        raise CaseTimeout(f"Case exceeded {seconds} seconds")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def exclusion_inventory(root=ROOT):
    """Read historical metadata, never change it. Hash all input manifests."""
    root = Path(root)
    cache = root / "historical-exclusions.json"
    if cache.exists():
        return json.loads(cache.read_text())
    fps, seeds, sources = set(), set(), {}
    def visit(value):
        if isinstance(value, dict):
            if value.get("fingerprint"):
                fps.add(value["fingerprint"])
            if value.get("graph_seed") is not None:
                seeds.add(int(value["graph_seed"]))
            for child in value.values():
                if isinstance(child, (list,dict)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    paths = [*(HISTORICAL / "data/ucgnn/runs").glob("*/data_manifest.json"),
             *(HISTORICAL / "data/ucgnn/runs").glob("*/history.json"),
             *(HISTORICAL / "data/m6study-release/runs").glob("*/data_manifest.json"),
             *(EXTREME / "data/ucgnn-extreme/graphs").glob("*.json"),
             *(root / "runs").glob("*/data_manifest.json"),*(root / "runs").glob("*/history.json")]
    for p in paths:
        visit(json.loads(p.read_text()))
        sources[str(p)] = file_hash(p)
    # Historical complete per-case exports contain the graph fingerprints and
    # seeds of the original/extra cohorts as well as the extreme study.
    for p in [EXTREME / "data/ucgnn-extreme/report/cases.csv",
              EXTREME / "data/ucgnn-extreme/report/original-cohorts.csv"]:
        if p.exists():
            with p.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("fingerprint"):
                        fps.add(row["fingerprint"])
                    if row.get("graph_seed"):
                        seeds.add(int(row["graph_seed"]))
            sources[str(p)] = file_hash(p)
    # Include legacy cached graphs, including validation rollouts, without
    # needing to deserialize every checkpoint or duplicate evaluation JSON.
    for p in (HISTORICAL / "data/ucgnn/cohort-cache").rglob("*.npz"):
        with np.load(p,allow_pickle=False) as f:
            if "fingerprint" in f:
                fps.add(str(f["fingerprint"].item()))
    result = {"fingerprints": sorted(fps), "graph_seeds": sorted(seeds), "sources":sources,
              "protocol_sha256":protocol_hash(), "created_utc":now()}
    atomic_json(cache,result)
    return result


def prepare_cohort(root=ROOT):
    root = Path(root)
    plan = cohort_plan()
    exclusion = exclusion_inventory(root)
    fps, seeds = set(exclusion["fingerprints"]), set(exclusion["graph_seeds"])
    graph_dir = root / "confirmatory-graphs"
    graph_dir.mkdir(parents=True,exist_ok=True)
    generated, used = {}, set()
    for row in plan:
        gid = row["graph_id"]
        if gid in generated:
            continue
        info_path = graph_dir / f"{gid}.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            if info["fingerprint"] in fps and row["family"] != "CLIQUE":
                raise RuntimeError("Cached test overlaps exclusions")
        else:
            if row["graph_seed"] in seeds:
                raise RuntimeError("Test graph seed overlaps historical/train/validation seed")
            graph = generate(row["n"],row["degree"],row["family"],row["graph_seed"])
            fp = fingerprint(graph)
            deterministic = row["family"] == "CLIQUE"
            if not deterministic and (fp in fps or fp in used):
                # No adaptive resampling after inspecting a test outcome.
                raise RuntimeError("Prespecified random graph fingerprint overlaps; protocol amendment required")
            info = {k:v for k,v in row.items() if k not in ("case_id","tape_seed","tape")}
            info.update(fingerprint=fp,deterministic_stress_graph=deterministic,
                        historical_fingerprint_overlap=fp in fps,**degree_stats(graph))
            edges = _edge_index(graph)
            temporary = info_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,edge_index=edges)
            temporary.replace(info_path.with_suffix(".npz"))
            atomic_json(info_path,info)
        used.add(info["fingerprint"])
        generated[gid] = info
    cases = [{**r,**generated[r["graph_id"]],"case_id":r["case_id"],"tape_seed":r["tape_seed"],"tape":r["tape"]} for r in plan]
    output = {"protocol_sha256":protocol_hash(),"cases":cases,"graphs":len(generated),
              "graph_tape_cases":len(cases),"created_utc":now()}
    p = root / "confirmatory-cohort.json"
    if p.exists() and digest(json.loads(p.read_text())["cases"]) != digest(cases):
        raise RuntimeError("Confirmatory cohort changed")
    if not p.exists():
        atomic_json(p,output)
    return cases


class SpecialistAdapter(torch.nn.Module):
    """Exact M4B hard-message inference exposed through the common interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, colors, rp, rc, edge_index, palette, edge_chunk=None):
        k = self.model.num_colors
        if not bool((palette == k).all()):
            raise ValueError("Specialist cannot process an unseen palette")
        x = torch.cat((torch.nn.functional.one_hot(colors+1,num_classes=k+1).float(),rp[:,None],rc[:,None]),1)
        p, c, commit = self.model(x,edge_index,hard_messages=True)
        choices = c.argmax(1)-1
        uncolored = colors < 0
        choices = torch.where(uncolored,choices,torch.full_like(choices,-1))
        scores = c[:,1:]
        # These fields are explicitly unavailable, not a learned availability.
        placeholder = torch.zeros_like(scores)
        return UOutput(p,scores,commit,(p>0)&uncolored,choices,placeholder,placeholder.sum(1),placeholder,placeholder)


def load_spec(spec, device="cuda"):
    if spec["kind"] == "paper":
        return load_model(spec["checkpoint"],device)
    if spec["kind"] == "historical":
        from dcl.universal.train import load_model as historical_load
        return historical_load(spec["checkpoint"],device)
    from dcl.m4_models import M4B
    saved = torch.load(spec["checkpoint"],map_location=device,weights_only=False)
    model = M4B(num_colors=spec["degree"]+1).to(device)
    model.load_state_dict(saved["model_state_dict"])
    return SpecialistAdapter(model).eval(),saved


@torch.inference_mode()
def rollout(model,n,edges,palette,tape_seed,max_rounds=256,initial_colors=None,cyclic=False):
    model.eval()
    device = next(model.parameters()).device
    colors = np.full(n,-1,np.int64) if initial_colors is None else np.asarray(initial_colors,np.int64).copy()
    rng = np.random.default_rng(tape_seed)
    edges_t = torch.as_tensor(edges,dtype=torch.long,device=device)
    palette_t = torch.full((n,),palette,dtype=torch.long,device=device)
    totals, by_k, traces = defaultdict(int),defaultdict(lambda:defaultdict(int)),[]
    first_unsafe, first_commit, previous_commit, max_wait = 0,0,0,0
    tick = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for t in range(1,max_rounds+1):
        rp,rc = rng.random(n),rng.random(n)
        out = model(torch.as_tensor(colors,device=device),torch.as_tensor(rp.astype(np.float32),device=device),
                    torch.as_tensor(rc.astype(np.float32),device=device),edges_t,palette_t,8192)
        p,c,commit = out.participate.cpu().numpy(),out.candidate.cpu().numpy(),(out.commit_logit>0).cpu().numpy()
        executed = (colors<0)&p&(c>=0)&commit
        # Decisions above are frozen before any exact observer runs.
        alpha,lo,hi = [getattr(out,key).cpu().numpy() for key in ("availability","lo","hi")]
        observed,observed_k = observe(colors,edges,palette,rp,rc,p,c,commit,alpha,lo,hi)
        if cyclic:
            offer,_ = cyclic_offer(colors,rc,edges,palette)
            observed["cyclic_candidate_correct"] = int(((colors<0)&(c==offer)).sum())
        for key,value in observed.items():
            totals[key] += value
        for key,counts in observed_k.items():
            for metric,value in counts.items():
                by_k[key][metric] += value
        fixed,simultaneous = conflict_flags(colors,edges,p,c)
        active = (colors<0)&p&(c>=0)
        histogram = np.bincount(c[active],minlength=palette)
        proposal_mass = histogram / max(1,int(active.sum()))
        if observed["unsafe_commits"] and not first_unsafe:
            first_unsafe = t
        if executed.any():
            first_commit = first_commit or t
            max_wait = max(max_wait,t-previous_commit)
            previous_commit = t
        traces.append({"round":t,"pending_before":int((colors<0).sum()),"participants":int(active.sum()),
                       "safe_proposals":int(observed_k_sum(observed_k,"valid")),"confirmations":int(executed.sum()),
                       "unsafe_commits":int(observed["unsafe_commits"]),"forbidden_proposals":int(observed["forbidden_offers"]),
                       "simultaneous_collisions":int((active&simultaneous).sum()),"fixed_conflicts":int((active&fixed).sum()),
                       "distinct_proposed_colors":int((histogram>0).sum()),"max_proposal_mass":float(proposal_mass.max()),
                       "candidate_pair_collision":float((proposal_mass**2).sum())})
        colors[executed] = c[executed]
        if (colors>=0).all():
            break
    src,dst = edges
    bad = int(((src<dst)&(colors[src]>=0)&(colors[src]==colors[dst])).sum())
    finished = bool((colors>=0).all())
    result = {"status":"ok" if finished and bad==0 else "coloring_failure", "finished":finished,
              "success":bool(finished and bad==0),"monochromatic_edges":bad,"rounds":t,
              "uncolored_nodes":int((colors<0).sum()),"first_unsafe_round":first_unsafe,
              "first_confirmation_round":first_commit,"max_interconfirmation_wait":max(max_wait,t-previous_commit),
              "trailing_wait_rounds":t-previous_commit,"trailing_wait_right_censored":not finished,
              "wall_seconds":time.perf_counter()-tick,"gpu_peak_bytes":int(torch.cuda.max_memory_allocated()) if device.type=="cuda" else 0,
              "by_k":{k:dict(v) for k,v in by_k.items()},"round_trace":traces,**dict(totals)}
    if not first_commit:
        result["first_confirmation_right_censored"] = True
        result["first_confirmation_observation_rounds"] = t
    else:
        result["first_confirmation_right_censored"] = False
        result["first_confirmation_observation_rounds"] = first_commit
    return result,colors


def observed_k_sum(rows,key):
    return sum(r[key] for r in rows.values())


def schedule_evaluations(cyclic_only=False):
    # Called only after all primary pilots and promotions have completed.
    specs = []
    for directory in sorted((ROOT/"runs").glob("*")):
        if not (directory/"epoch-120.pt").exists():
            continue
        config = json.loads((directory/"config.json").read_text())
        cyclic = config["treatment"] == "cyclic_teacher"
        if cyclic != cyclic_only:
            continue
        for policy in (("epoch120",) if cyclic else ("epoch120","best")):
            checkpoint = directory/("epoch-120.pt" if policy=="epoch120" else "best.pt")
            specs.append({"label":f"{directory.name}-{policy}","kind":"paper","treatment":config["treatment"],
                          "seed":config["seed"],"checkpoint":str(checkpoint),"selection":policy,
                          "core_only":policy=="best","degree":None})
    if not cyclic_only:
        for experiment in ("e1","a1","c1"):
            for p in sorted((HISTORICAL/"data/ucgnn/runs").glob(f"{experiment}-*/epoch-120.pt")):
                c = json.loads(p.with_name("config.json").read_text())
                specs.append({"label":f"historical-{experiment}-s{c['seed']}-epoch120","kind":"historical", "treatment":experiment.upper(),
                              "seed":c["seed"],"checkpoint":str(p),"selection":"epoch120","core_only":False,"degree":None})
        # Five independently trained degree specialists per compatible degree.
        for d in (3,5,10):
            for seed in range(5):
                matches = []
                for p in (HISTORICAL/"data/m6study-release/runs").glob(f"d{d}-regular-counterfactual-s{seed:02d}-*/epoch-120.pt"):
                    c = json.loads(p.with_name("config.json").read_text())
                    if c["graphs"]==800 and c["tapes"]==3 and c["epochs"]==120 and not c["updates_per_epoch"]:
                        matches.append(p)
                if len(matches)!=1:
                    raise RuntimeError(f"Expected one compatible M6B checkpoint d={d},s={seed}, found {matches}")
                specs.append({"label":f"m6b-d{d}-s{seed}-epoch120","kind":"specialist","treatment":"M6B", "seed":seed,
                              "checkpoint":str(matches[0]),"selection":"epoch120","core_only":False,"degree":d})
    # Epoch 120 is primary. Evaluate one seed across all primary treatments
    # before the next seed; validation-best is the final, optional tier.
    order={name:i for i,name in enumerate(("normalized","raw_endpoints","sum","mean","deepsets","transformer","own_only","cyclic_teacher"))}
    specs.sort(key=lambda s:(s["selection"]=="best",s["kind"]!="paper",s["seed"],order.get(s["treatment"],99),s["label"]))
    tasks = []
    for spec in specs:
        spec["checkpoint_sha256"] = file_hash(spec["checkpoint"])
        spec["protocol_sha256"] = protocol_hash()
        path = ROOT/"models"/f"{spec['label']}.json"
        if path.exists() and json.loads(path.read_text())!=spec:
            raise RuntimeError("Model specification changed")
        atomic_json(path,spec)
        tasks.append({"id":f"eval-{spec['label']}","action":"evaluate","status":"pending","attempts":[],"timeout":21600,
                      "model_spec":str(path),"arguments":["evaluate","--model-spec",str(path)]})
    return tasks


def evaluate_spec(spec_path,stop_requested=lambda:False):
    spec = json.loads(Path(spec_path).read_text())
    if spec["checkpoint_sha256"] != file_hash(spec["checkpoint"]):
        raise RuntimeError("Checkpoint hash mismatch")
    cases = prepare_cohort()
    model,saved = load_spec(spec)
    initial_hash = weights_hash(model.state_dict())
    directory = ROOT/"evaluations"/spec["label"]
    directory.mkdir(parents=True,exist_ok=True)
    planned,counts = 0,defaultdict(int)
    for case in cases:
        if spec["degree"] is not None and case["degree"]!=spec["degree"]:
            continue
        if spec["core_only"] and case["tier"]!="core":
            continue
        planned += 1
        output = directory/f"{case['case_id']}.json"
        if output.exists():
            row = json.loads(output.read_text())
            if row["checkpoint_sha256"] != spec["checkpoint_sha256"]:
                raise RuntimeError("Existing result uses different checkpoint")
            counts[row["status"]] += 1
            continue
        if stop_requested():
            return
        edges = np.load(ROOT/"confirmatory-graphs"/f"{case['graph_id']}.npz",allow_pickle=False)["edge_index"]
        tick = time.perf_counter()
        try:
            with deadline(180):
                result,colors = rollout(model,case["n"],edges,case["degree"]+1,case["tape_seed"],
                                        cyclic=spec["treatment"]=="cyclic_teacher")
            result["final_colors_sha256"] = digest(colors.tolist())
            if not result["success"]:
                np.savez_compressed(output.with_suffix(".npz"),colors=colors,edge_index=edges)
        except torch.OutOfMemoryError as error:
            result = {"status":"oom","success":None,"error":str(error),"wall_seconds":time.perf_counter()-tick}
            torch.cuda.empty_cache()
        except CaseTimeout as error:
            result = {"status":"timeout","success":None,"error":str(error),"wall_seconds":time.perf_counter()-tick}
        # spec.degree is an eligibility restriction, not the evaluated degree.
        row = {**spec,**case,**result,"completed_utc":now(),"epoch":saved["epoch"],"weights_sha256":initial_hash}
        atomic_json(output,row)
        counts[result["status"]] += 1
        atomic_json(directory/"progress.json",{"planned_cases":sum((spec["degree"] is None or c["degree"]==spec["degree"]) and
                    (not spec["core_only"] or c["tier"]=="core") for c in cases),"completed_cases":sum(counts.values()),
                    "statuses":dict(counts),"latest_case":case["case_id"]})
        if sum(counts.values())%20==0:
            print(f"{spec['label']} {sum(counts.values())} {dict(counts)}",flush=True)
    if weights_hash(model.state_dict()) != initial_hash:
        raise RuntimeError("Inference altered parameters")
    atomic_json(directory/"complete.json",{"planned_cases":planned,"statuses":dict(counts),"weights_sha256":initial_hash,
                                          "checkpoint_sha256":spec["checkpoint_sha256"],"complete":True,"utc":now()})
