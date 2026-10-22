# Global complete-coloring study

Protocol frozen at 80195c5; base local-palette extension 25cbeb5.
Branch research/ucgnn-global-distribution, no remote push, no new training.
The resumable coordinator writes data/ucgnn-global-distribution/queue.json
and coordinator.log. Primary statistics are precomputed from the fixed
banks in a separate CPU process. No adaptive design changes.

53 graph entries (51 unique labeled graphs), 15 recent checkpoints,
250 complete runs per generator for each primary entry, 795 planned tests.
See GLOBAL_DISTRIBUTIONS.md for reproduction and scope.
