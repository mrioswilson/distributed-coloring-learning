import numpy as np

from dcl.dataset import encode_round
from dcl.graphs import connected_random_regular_graph
from dcl.oracle import run_oracle


G = connected_random_regular_graph(
    n=40,
    degree=3,
    seed=123,
)

result = run_oracle(
    G=G,
    seed=456,
    num_colors=4,
    save_trace=True,
)

trace = result.trace[0]
example = encode_round(G, trace)

print("x shape:", example.x.shape)
print("edge_index shape:", example.edge_index.shape)
print("y shape:", example.y.shape)
print()

print("participating:", example.decision_mask.sum())
print(
    "commit:",
    int(example.y[example.decision_mask].sum()),
)
print(
    "wait:",
    int(
        example.decision_mask.sum()
        - example.y[example.decision_mask].sum()
    ),
)

print()
print("Primeros nodos participantes:")
print()

shown = 0

for v in range(G.number_of_nodes()):
    if not example.decision_mask[v]:
        continue

    neighbors = list(G.neighbors(v))

    print(
        f"node={v:2d} "
        f"candidate={trace.candidate[v]} "
        f"neighbors={neighbors} "
        f"neighbor_candidates="
        f"{[int(trace.candidate[u]) for u in neighbors]} "
        f"commit={bool(trace.commit[v])}"
    )

    shown += 1

    if shown == 10:
        break
