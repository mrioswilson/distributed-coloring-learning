# Pedro revision counterfactual study (100 models)

This archive contains the complete preserved `data/pedrorev-cf/` tree: checkpoints, acquired training banks, evaluation banks, queries, and individual prediction responses. Negative and failed outcomes are preserved. Its provenance copies are the published freeze and indexes.

## Restore from a clean clone

Download this archive and its `.SHA256SUMS` sidecar from the GitHub release, then from the clone root run:

```sh
sha256sum -c pedrorev-cf.tar.gz.SHA256SUMS
tar -xzf pedrorev-cf.tar.gz
sha256sum -c SHA256SUMS
```

The first manifest checks the downloaded archive. The included `SHA256SUMS` checks every extracted payload file. The original inventory is `provenance/freeze.json`; published bank and response inventories are included alongside it. SHA256 values of preserved data were checked against those inventories before packaging.
