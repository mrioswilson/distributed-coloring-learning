"""Configuration and named random streams for the Universal Color GNN study.

Training graphs and tapes reuse the M6B streams (``dcl.study.config.seed_for``,
namespace ``m6study-v1``), so a UCGNN run with replica ``s`` sees exactly the
graphs/tapes of the M6B specialist with replica ``s`` for each training degree
(prefixes when fewer graphs are used). Initialization, minibatch order and
counterfactual draws use separate UCGNN streams that depend only on the
replica, so different training mixtures share their initialization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from dcl.study.config import digest

PROTOCOL = "ucgnn-v1"
VARIANTS = ("NORMALIZED", "RAW_PREFIX")
METHODS = ("COUNTERFACTUAL", "OWN_ONLY")


def model_seed(replica: int, stream: str) -> int:
    return int(digest([PROTOCOL, replica, stream])[:16], 16)


@dataclass(frozen=True)
class UConfig:
    name: str = "U3"
    degrees: tuple[int, ...] = (3,)
    graphs_per_degree: tuple[int, ...] = (800,)
    seed: int = 0
    variant: str = "NORMALIZED"
    method: str = "COUNTERFACTUAL"
    family: str = "REGULAR"
    n: int = 40
    tapes: int = 3
    epochs: int = 120
    warmup: int = 10
    batch_size: int = 128
    lr: float = 0.001
    hidden_dim: int = 64
    message_dim: int = 32
    val_graphs_per_degree: int = 200
    val_rollouts: int = 4
    val_sizes: tuple[int, ...] = (40, 160)
    val_every: int = 10
    max_rounds: int = 256
    checkpoint_epochs: tuple[int, ...] = (10, 20, 40, 60, 80, 100, 120)
    fixed_neighbor_probability: float = 0.17
    simultaneous_probability: float = 0.04
    protocol: str = PROTOCOL

    def __post_init__(self):
        if len(self.degrees) != len(self.graphs_per_degree) or not self.degrees:
            raise ValueError("One graph count per training degree is required")
        if len(set(self.degrees)) != len(self.degrees) or any(d < 1 for d in self.degrees):
            raise ValueError("Training degrees must be distinct and positive")
        if self.variant not in VARIANTS or self.method not in METHODS:
            raise ValueError("Unknown variant or method")
        if self.family not in ("REGULAR", "IRREGULAR", "MIXED"):
            raise ValueError("Unknown training family")
        for d in self.degrees:
            if self.n <= d or self.n * d % 2:
                raise ValueError("Invalid regular graph size/parity")
            if any(n <= d or n * d % 2 for n in self.val_sizes):
                raise ValueError("Invalid validation graph size/parity")
        for key in ("tapes", "epochs", "batch_size", "val_graphs_per_degree", "val_rollouts", "val_every", "max_rounds"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if any(g < 1 for g in self.graphs_per_degree):
            raise ValueError("Graph counts must be positive")
        if self.seed < 0 or self.warmup < 0 or self.lr <= 0 or self.hidden_dim < 1 or self.message_dim < 1:
            raise ValueError("Invalid seed, warmup, learning rate or dimensions")
        if (self.fixed_neighbor_probability, self.simultaneous_probability) != (0.17, 0.04):
            raise ValueError("Counterfactual probabilities are frozen at the M6B values")
        if self.protocol != PROTOCOL:
            raise ValueError("Unknown protocol")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("degrees", "graphs_per_degree", "val_sizes", "checkpoint_epochs"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)

    @property
    def id(self):
        return f"{self.name.lower()}-{self.variant.lower()}-{self.method.lower()}-s{self.seed:02d}-{digest(self.to_dict())[:12]}"

    @property
    def max_palette(self):
        return max(self.degrees) + 1
