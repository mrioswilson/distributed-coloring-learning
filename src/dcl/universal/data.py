"""Teacher trajectories and batching for arbitrary palettes.

An example is one synchronous round of the unchanged oracle on one graph:
model inputs (colors, r_p, r_c, edges, palette size) and teacher labels
(participation, offer, commit). Labels are kept in separate fields and are
never part of the model input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dcl.m4_data import _edge_index
from dcl.oracle import run_oracle
from dcl.study.config import digest, seed_for
from dcl.study.graphs import degree_stats, fingerprint, generate_graph


@dataclass
class UExample:
    colors: np.ndarray        # int64 [N], -1 uncolored
    rp: np.ndarray            # float32 [N]
    rc: np.ndarray            # float32 [N]
    edge_index: np.ndarray    # int64 [2, E]
    palette: int              # declared palette size K = degree + 1
    degree: int               # declared maximum degree
    y_participate: np.ndarray  # float32 [N]
    y_offer: np.ndarray       # int64 [N], -1 for colored nodes
    y_commit: np.ndarray      # float32 [N]

    @property
    def mask(self):
        return self.colors < 0


def examples_from_trace(graph, trace, degree, edges=None):
    edges = _edge_index(graph) if edges is None else edges
    out = []
    for t in trace:
        out.append(UExample(
            colors=t.colors_before.astype(np.int64), rp=t.activation_draw.astype(np.float32),
            rc=t.candidate_draw.astype(np.float32), edge_index=edges, palette=degree + 1, degree=degree,
            y_participate=t.participate.astype(np.float32), y_offer=t.candidate_offer.astype(np.int64),
            y_commit=t.commit.astype(np.float32)))
    return out


def collect_examples(config, split):
    """Training/validation rounds; the final test is never generated here."""
    if split not in ("train", "validation"):
        raise ValueError("Final test cohorts cannot be used for training or selection")
    examples, records = [], []
    for degree, count in zip(config.degrees, config.graphs_per_degree):
        count = count if split == "train" else config.val_graphs_per_degree
        tapes = config.tapes if split == "train" else 1
        family = config.family if split == "train" else "MIXED"
        for i in range(count):
            # Same named streams as M6B (protocol m6study-v1) for this replica/degree.
            graph_seed = seed_for(config.seed, degree, f"{split}/graph", i)
            actual = family
            if family == "MIXED":
                actual = "REGULAR" if i % 2 == 0 else "IRREGULAR"
            graph = generate_graph(config.n, degree, actual, graph_seed)
            edges = _edge_index(graph)
            record = {"degree": degree, "graph_i": i, "split": split, "graph_seed": graph_seed,
                      "fingerprint": fingerprint(graph), "family": actual, "n": config.n,
                      **degree_stats(graph), "trajectories": []}
            for j in range(tapes):
                tape_seed = seed_for(config.seed, degree, f"{split}/tape", i, j)
                result = run_oracle(graph, tape_seed, num_colors=degree + 1, max_rounds=1000)
                if not result.solved:
                    raise RuntimeError("The unchanged oracle failed to color a generated graph")
                examples.extend(examples_from_trace(graph, result.trace, degree, edges))
                tape_hash = digest([tape_seed, [t.activation_draw.tolist() + t.candidate_draw.tolist() for t in result.trace]])
                record["trajectories"].append({"tape_i": j, "tape_seed": tape_seed, "rounds": result.rounds, "tape_hash": tape_hash})
            records.append(record)
    return examples, records


@dataclass
class UBatch:
    colors: torch.Tensor
    rp: torch.Tensor
    rc: torch.Tensor
    edge_index: torch.Tensor
    palette: torch.Tensor
    y_participate: torch.Tensor
    y_offer: torch.Tensor
    y_commit: torch.Tensor
    mask: torch.Tensor

    def inputs(self):
        """The only tensors a model forward pass receives."""
        return self.colors, self.rp, self.rc, self.edge_index, self.palette


def make_batch(examples, device):
    """Disjoint union of rounds; palettes may differ between examples."""
    offset, edges, palette = 0, [], []
    for e in examples:
        edges.append(e.edge_index + offset)
        palette.append(np.full(len(e.colors), e.palette, dtype=np.int64))
        offset += len(e.colors)
    cat = lambda key: np.concatenate([getattr(e, key) for e in examples])
    colors = cat("colors")
    arrays = dict(colors=colors, rp=cat("rp"), rc=cat("rc"), edge_index=np.concatenate(edges, axis=1),
                  palette=np.concatenate(palette), y_participate=cat("y_participate"), y_offer=cat("y_offer"),
                  y_commit=cat("y_commit"), mask=colors < 0)
    return UBatch(**{k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arrays.items()})
