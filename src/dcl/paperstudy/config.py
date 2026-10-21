from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

from dcl.study.config import digest
from dcl.universal.config import UConfig

PROTOCOL = "ucgnn-paper-v1"
PROTOCOL_COMMIT = "023deaf"
NAMESPACE = 202610210211
SEEDS = (41000, 41001, 41002, 41003, 41004)
DEGREES = (3, 5, 10, 15, 20, 50, 100, 200)
ROOT = Path("data/ucgnn-paper-study")
PYTHON = "/workspaces/AlgoLearning/distributed-coloring-learning/.venv/bin/python"
HISTORICAL = Path("/workspaces/AlgoLearning/distributed-coloring-learning")
EXTREME = Path("/workspaces/AlgoLearning/ucgnn-extreme-degree-study")


def named_seed(*values):
    return int(digest([PROTOCOL, *values])[:16], 16)


def protocol_hash():
    return hashlib.sha256(Path("docs/UCGNN_PAPER_PROTOCOL.md").read_bytes()).hexdigest()


@dataclass(frozen=True)
class Run:
    treatment: str
    seed: int

    @property
    def id(self):
        return f"{self.treatment}-s{self.seed}"

    @property
    def method(self):
        return "OWN_ONLY" if self.treatment == "own_only" else "COUNTERFACTUAL"

    def training_config(self):
        return UConfig(name="PAPER", seed=self.seed, method=self.method,
                       checkpoint_epochs=(1, 10, 20, 40, 60, 80, 100, 120))

    def to_dict(self):
        return {**asdict(self), "training": self.training_config().to_dict(),
                "teacher": "CYCLIC" if self.treatment == "cyclic_teacher" else "UNIFORM_AVAILABLE",
                "protocol": PROTOCOL, "protocol_commit": PROTOCOL_COMMIT, "protocol_sha256": protocol_hash()}


def cohort_plan():
    """Pure metadata generation; does not inspect or generate a graph."""
    rows = []
    for d in DEGREES:
        tiers = [("core", 400 if d == 200 else 160, 8)]
        if d in (3, 10, 20, 50, 100, 200):
            tiers.append(("size", 640, 4))
        for tier, n, count in tiers:
            for family in ("REGULAR", "IRREGULAR", "HETEROGENEOUS"):
                for i in range(count):
                    rows.append(dict(tier=tier, degree=d, n=n, family=family, index=i, tape=0))
        rows.extend(dict(tier="structured", degree=d, n=d+1, family="CLIQUE", index=0, tape=j) for j in range(3))
        rows.extend(dict(tier="structured", degree=d, n=max(160, 2*d), family="BIPARTITE", index=i, tape=0) for i in range(2))
    for row in rows:
        base = (NAMESPACE, row["degree"], row["n"], row["family"], row["index"])
        row.update(namespace=NAMESPACE, split="paper_confirm_v1",
                   graph_seed=named_seed(*base, "graph"), tape_seed=named_seed(*base, row["tape"], "tape"))
        row["graph_id"] = digest(base)[:24]
        row["case_id"] = digest([base, row["tape"]])[:24]
    return rows
