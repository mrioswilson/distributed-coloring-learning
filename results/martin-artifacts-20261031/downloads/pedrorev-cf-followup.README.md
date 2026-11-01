# Pedro revision counterfactual follow-up

This archive contains the complete preserved `data/pedrorev-cf-followup/` tree, including the five follow-up models, banks, responses, and checkpoint snapshots present in the published freeze. Negative and failed outcomes are preserved. Provenance copies of the freeze and response index are included.

## Restore from a clean clone

Download this archive and its `.SHA256SUMS` sidecar from the GitHub release, then from the clone root run:

```sh
sha256sum -c pedrorev-cf-followup.tar.gz.SHA256SUMS
tar -xzf pedrorev-cf-followup.tar.gz
sha256sum -c SHA256SUMS
```

The first manifest checks the downloaded archive. The included `SHA256SUMS` checks every extracted payload file. SHA256 values of preserved checkpoints and responses were checked against the published freeze and index before packaging.
