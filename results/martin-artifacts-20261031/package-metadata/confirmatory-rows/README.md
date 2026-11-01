# Historical confirmatory execution rows

This archive contains the original raw row files for all confirmatory cells referenced by `results/final-closure/generalization-cells.csv`, plus that inventory file. It preserves checkpoint, condition, seed, and case correspondence through the inventory paths and SHA256 values. It contains the 810 confirmatory cell files (810,000 executions according to the inventory); original failed and negative outcomes are retained.

## Restore from a clean clone

Download this archive and its `.SHA256SUMS` sidecar from the GitHub release, then from the clone root run:

```sh
sha256sum -c confirmatory-rows.tar.gz.SHA256SUMS
tar -xzf confirmatory-rows.tar.gz
sha256sum -c SHA256SUMS
```

The first manifest checks the archive. The included `SHA256SUMS` checks every raw cell file and the inventory. Each raw file was checked against the `rows_sha256` value in the inventory before packaging. Extracting at the clone root restores each file at its original repository-relative path.
