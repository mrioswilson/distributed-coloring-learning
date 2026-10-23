# Preservation before scientific integration

Created: 2026-10-23T14:50:48.329883+00:00

Seven worktrees are preserved separately. No scientific edits were pending. Local publication configuration and host backups were excluded using Git info/exclude.

| Branch | Original commit | State |
|---|---|---|
| `feat/universal-color-gnn` | `1d8857b25e8b654e87351fce33813692c70a6250` | clean |
| `feat/ucgnn-extreme-degree-study` | `29251976fce39f1bce0e54e02901c08890a92e7e` | clean |
| `research/ucgnn-global-distribution` | `1ba4546c64d143c44480644ebb9d61f9917c8982` | clean |
| `research/ucgnn-local-palettes` | `25cbeb59775a3a8195a2484f42bfcf2c847a60e9` | clean |
| `research/ucgnn-paper-study` | `b5427faf16a1278bb04a2e7bac7f78d2bc0b73ab` | clean |
| `research/ucgnn-robust-distribution-tests` | `74134d865db4f142bfeea49b091200588d1f2803` | clean |
| `research/ucgnn-size-degree` | `9f64bb600cdef71c75f7eb7f29a03f768e50b587` | clean |

The size-degree branch descends from paper, local-palette, global-law and robust-test branches. The extreme-degree and universal-color histories retain separate refs. No blind merge is performed.

`results/integration/preserved-runtime.json.gz` records 247,247 local scientific runtime files (8,641,834,027 bytes). Scope: Recent ucgnn runtime and older model-run checkpoints; legacy M6 per-case evaluations, smoke and logs are excluded. Historical tracked manifests remain unchanged.

The adjacent JSON records exact commits, tree IDs, result directories, archive digests and checks. Only release-uploaded assets are remote recovery artifacts; the runtime inventory also includes files preserved locally. The existing robust and factorial archives include fifteen frozen checkpoints. Remote verification follows in `REMOTE_RECOVERY.json`.

Validation: latest full pytest (343 passed), extreme-degree full pytest (210 passed), Git fsck, diff whitespace check, explicit tracked secret-pattern scan, and archive SHA256 verification. No force push or update of main.
