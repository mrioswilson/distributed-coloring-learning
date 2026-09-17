import numpy as np

from dcl.graphs import connected_random_regular_graph
from dcl.oracle import UNCOLORED, run_oracle


G = connected_random_regular_graph(
    n=320,
    degree=3,
    seed=123,
)

result = run_oracle(
    G=G,
    seed=456,
    num_colors=4,
    save_trace=True,
)

print("solved:", result.solved)
print("rounds:", result.rounds)
print()
print(
    "round | uncolored_before | participating | committed | uncolored_after"
)
print("-" * 74)

for r in result.trace:
    before = int(np.sum(r.colors_before == UNCOLORED))
    participating = int(np.sum(r.participate))
    committed = int(np.sum(r.commit))
    after = int(np.sum(r.colors_after == UNCOLORED))

    print(
        f"{r.round_idx:5d} | "
        f"{before:16d} | "
        f"{participating:13d} | "
        f"{committed:9d} | "
        f"{after:15d}"
    )
