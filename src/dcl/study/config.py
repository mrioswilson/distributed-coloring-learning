from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_for(replica: int, degree: int, stream: str, *coordinates) -> int:
    # Named streams do not depend on call order, method, or experiment budget.
    return int(digest(["m6study-v1", replica, degree, stream, *coordinates])[:16], 16)


@dataclass(frozen=True)
class Config:
    degree: int = 3
    seed: int = 0
    method: str = "COUNTERFACTUAL"
    family: str = "REGULAR"
    n: int = 40
    graphs: int = 800
    tapes: int = 3
    epochs: int = 120
    warmup: int = 10
    batch_size: int = 128
    lr: float = 0.001
    hidden_dim: int = 64
    message_dim: int = 32
    val_graphs: int = 200
    val_rollouts: int = 4
    val_sizes: tuple[int, ...] = (40, 160)
    val_every: int = 10
    max_rounds: int = 256
    checkpoint_epochs: tuple[int, ...] = (20, 40, 60, 80, 120)
    # If set, each epoch has precisely this many optimizer updates, sampling
    # reproducible permutations with wraparound. Records real round exposure.
    updates_per_epoch: int | None = None
    fixed_neighbor_probability: float = 0.17
    simultaneous_probability: float = 0.04
    protocol: str = "m6study-v1"

    def __post_init__(self):
        if self.degree not in (3, 5, 8, 10):
            raise ValueError("Supported maximum degrees: 3, 5, 8, 10")
        if self.method not in ("COUNTERFACTUAL", "OWN_ONLY", "ORACLE_MATCHED", "TRADITIONAL"):
            raise ValueError("Unknown method")
        if self.family not in ("REGULAR", "IRREGULAR", "MIXED"):
            raise ValueError("Unknown training family")
        if self.n <= self.degree or self.n * self.degree % 2:
            raise ValueError("Invalid regular graph size/parity")
        for key in ("graphs", "tapes", "epochs", "batch_size", "val_graphs", "val_rollouts", "val_every", "max_rounds"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.seed < 0 or self.warmup < 0 or self.lr <= 0:
            raise ValueError("Invalid seed, warmup, or learning rate")
        if any(n <= self.degree or n * self.degree % 2 for n in self.val_sizes):
            raise ValueError("Invalid validation graph size/parity")
        if self.hidden_dim < 1 or self.message_dim < 1 or self.protocol != "m6study-v1":
            raise ValueError("Invalid dimensions or protocol")
        if self.updates_per_epoch is not None and self.updates_per_epoch < 1:
            raise ValueError("updates_per_epoch must be positive")
        if (self.fixed_neighbor_probability, self.simultaneous_probability) != (0.17, 0.04):
            raise ValueError("Reference counterfactual probabilities are frozen; create a separate protocol for an ablation")

    @property
    def num_colors(self):
        return self.degree + 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("val_sizes", "checkpoint_epochs"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)

    @property
    def id(self):
        return f"d{self.degree}-{self.family.lower()}-{self.method.lower()}-s{self.seed:02d}-{digest(self.to_dict())[:12]}"

    @property
    def pair_id(self):
        value = self.to_dict()
        value.pop("method")
        return digest(value)

    @property
    def dataset_id(self):
        return digest({k: self.to_dict()[k] for k in ("protocol", "degree", "seed", "family", "n", "graphs", "tapes")})
