# PedroRev transfer experiment rows

This release contains the original 240 JSON result files from the additional transfer experiment (240,000 rows), generated from the audited repository revision `d155d3ed104d10a015fa1061e7a5fc7f5ebcdb7b`. Each result file retains its `summary` and `rows`. The archive also includes copies of `ROWS.sha256`, `design.json`, and `summary.json` under `pedrorev-transfer-metadata/`; the source files and manifests were not modified.

## Download and verify from a clean clone

From the root of a clean clone of `mrioswilson/distributed-coloring-learning`, download the assets from the [experimental release](https://github.com/mrioswilson/distributed-coloring-learning/releases/tag/pedrorev-experimental-data-2026-10-31):

```sh
curl -fL -O https://github.com/mrioswilson/distributed-coloring-learning/releases/download/pedrorev-experimental-data-2026-10-31/pedrorev-transfer-rows.tar.gz
curl -fL -O https://github.com/mrioswilson/distributed-coloring-learning/releases/download/pedrorev-experimental-data-2026-10-31/pedrorev-transfer-rows.SHA256SUMS
curl -fL -O https://github.com/mrioswilson/distributed-coloring-learning/releases/download/pedrorev-experimental-data-2026-10-31/pedrorev-transfer-rows.README.md
sha256sum -c pedrorev-transfer-rows.SHA256SUMS
```

Extract the archive from the clone root to restore all 240 JSON files to their original paths and place the verification metadata alongside them in its own directory:

```sh
tar -xzf pedrorev-transfer-rows.tar.gz
cd results/pedro-revision/transfer-experiment
sha256sum -c pedrorev-transfer-metadata/ROWS.sha256
```

The archived `design.json` describes the 60 frozen teacher/seed/arm checkpoints and four conditions. `summary.json` contains the 240 per-file summaries. The source manifest covers the JSON paths relative to this experiment directory.
