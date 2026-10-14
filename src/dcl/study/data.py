import numpy as np
import torch

from dcl.m4_data import encode_m4_round
from dcl.oracle import run_oracle
from .config import seed_for, digest
from .graphs import generate_graph, fingerprint, degree_stats


def collect_examples(config, split):
    if split not in ("train", "validation"):
        raise ValueError("Final test cannot be used for training or selection")
    count = config.graphs if split == "train" else config.val_graphs
    tapes = config.tapes if split == "train" else 1
    family = config.family if split == "train" else "MIXED"
    examples, records = [], []
    for i in range(count):
        graph_seed = seed_for(config.seed, config.degree, f"{split}/graph", i)
        # Exactly balanced mixed datasets (up to one graph) rather than a
        # potentially unbalanced small Bernoulli sample.
        actual_family = family
        if family == "MIXED":
            actual_family = "REGULAR" if i % 2 == 0 else "IRREGULAR"
        graph = generate_graph(config.n, config.degree, actual_family, graph_seed)
        record = {"graph_i": i, "split": split, "graph_seed": graph_seed,
                  "fingerprint": fingerprint(graph), "family": actual_family,
                  "n": config.n, **degree_stats(graph), "trajectories": []}
        for j in range(tapes):
            tape_seed = seed_for(config.seed, config.degree, f"{split}/tape", i, j)
            result = run_oracle(graph, tape_seed, num_colors=config.num_colors, max_rounds=1000)
            if not result.solved:
                raise RuntimeError("The unchanged oracle failed to color a generated graph")
            encoded = [encode_m4_round(graph, trace, config.num_colors) for trace in result.trace]
            examples.extend(encoded)
            tape_hash = digest([tape_seed, [t.activation_draw.tolist() + t.candidate_draw.tolist() for t in result.trace]])
            record["trajectories"].append({"tape_i": j, "tape_seed": tape_seed, "rounds": result.rounds, "tape_hash": tape_hash})
        records.append(record)
    return examples, records


def make_batch(examples, device):
    offset, edges = 0, []
    for example in examples:
        edges.append(example.edge_index + offset)
        offset += len(example.x)
    arrays = [np.concatenate([e.x for e in examples]), np.concatenate(edges, axis=1)]
    for key in ("y_participate", "y_candidate_offer", "y_commit", "decision_mask"):
        arrays.append(np.concatenate([getattr(e, key) for e in examples]))
    return tuple(torch.from_numpy(array).to(device) for array in arrays)
