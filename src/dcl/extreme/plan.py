from __future__ import annotations

from .common import digest
from .runtime import estimate_cost


def available_models(inventory):
    return [r for r in inventory["runs"] if r.get("epoch") == 120 and r["config"]["name"] in ("E1","E2","E3","A1","C1")]


def tasks(config, inventory):
    models = available_models(inventory)
    e1 = sorted((r for r in models if r["config"]["name"] == "E1"),key=lambda r:r["config"]["seed"])
    if not e1 or e1[0]["config"]["seed"] != 0:
        raise ValueError("The reported E1 replica 0 must be available")
    for r in e1:
        if r["train_degrees"] != [3] or r["train_sizes"] != [40]:
            raise ValueError("Pure-extrapolation checkpoints must be trained only on Delta=3, N=40")
    def item(stage,model,d,n,family,i):
        meta = {"namespace":config["namespace"],"degree":d,"palette":d+1,"n":n,"family":family,"graph_index":i}
        identity = digest({"checkpoint_sha256":model["checkpoint_sha256"],"graph":meta,
                           "protocol":config["protocol"],"max_rounds":config["max_rounds"],
                           "color_chunk":config["color_chunk"]})[:28]
        return {"task_id":identity,"stage":stage,"model":model,"graph":meta,
                "cost":estimate_cost(n,d,d+1,config["color_chunk"]),
                "timeout_seconds":config["large_timeout_seconds"] if n>=10240 else config["per_graph_timeout_seconds"]}
    seen = set()
    def unique(t):
        if t["task_id"] in seen:
            return False
        seen.add(t["task_id"])
        return True
    degrees = config["degrees"]
    # All degrees/families before any extensive replication.
    for d in degrees:
        n = max(40,2*(d+1))
        for family in ("REGULAR","IRREGULAR","HETEROGENEOUS","BIPARTITE","CLIQUE"):
            t = item("A-functional",e1[0],d,d+1 if family=="CLIQUE" else n,family,0)
            if unique(t): yield t
    # Degree 500 is an explicitly labelled, conditional stress probe.
    d = config["exploratory_degree"]
    for family in ("REGULAR","IRREGULAR","HETEROGENEOUS","BIPARTITE","CLIQUE"):
        t = item("A-exploratory",e1[0],d,d+1 if family=="CLIQUE" else 2*(d+1),family,0)
        if unique(t): yield t
    ordered = config["priority_degrees"] + [d for d in degrees if d not in config["priority_degrees"]]
    for d in ordered:
        for n in config["systematic_sizes"]:
            if n<=d or n*d%2: continue
            for model in e1:
                for family in ("REGULAR","IRREGULAR"):
                    for i in range(config["main_graphs_per_family"]):
                        t = item("B-main",model,d,n,family,i)
                        if unique(t): yield t
    # Increase precision on the same shared graph pool, never count model
    # evaluations on a shared graph as independent graph samples.
    for d in config["priority_degrees"]:
        for model in e1:
            for family in ("REGULAR","IRREGULAR"):
                for i in range(config["main_graphs_per_family"],config["precision_graphs_per_family"]):
                    t = item("B-precision",model,d,config["precision_size"],family,i)
                    if unique(t): yield t
    for d in config["priority_degrees"]+[200]:
        for model in e1:
            for family in ("HETEROGENEOUS","BIPARTITE","CLIQUE"):
                for i in range(1 if family=="CLIQUE" else config["stress_graphs_per_family"]):
                    t = item("B-stress",model,d,d+1 if family=="CLIQUE" else 640,family,i)
                    if unique(t): yield t
    for n in config["large_sizes"]:
        for d in config["priority_degrees"]:
            cost = estimate_cost(n,d,d+1,config["color_chunk"])
            # Predeclared rule, depends only on resource cost, never success.
            if cost["logical_message_bytes_per_round"] > config["large_message_bytes_limit"]: continue
            for model in e1:
                for family in ("REGULAR","IRREGULAR"):
                    for i in range(config["large_graphs_per_family"]):
                        t = item("C-large-exploratory",model,d,n,family,i)
                        if unique(t): yield t
    # Diagnostics are a separate queue action, after pure extrapolation.
    yield {"task_id":"mechanism-diagnostics","stage":"D-mechanisms","kind":"diagnostics","models":models}
    comparisons = sorted((r for r in models if r["config"]["name"] in ("E2","E3","A1","C1")),
                         key=lambda r:(r["config"]["name"],r["config"]["seed"]))
    for model in comparisons:
        for d in config["priority_degrees"]:
            for n in config["comparison_sizes"]:
                for family in ("REGULAR","IRREGULAR"):
                    for i in range(config["main_graphs_per_family"]):
                        t = item("E-existing-controls",model,d,n,family,i)
                        if unique(t): yield t
